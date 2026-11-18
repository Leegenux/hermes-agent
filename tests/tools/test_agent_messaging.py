"""Stage-2 same-tree agent messaging: public-path acceptance tests.

The only fake is the model/agent run function; the delegation registry, child build,
SessionDB, continuation engine and inbox hooks are the production ones.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from tests.waiting import wait_until
from tools.process_registry import process_registry


def _parent(session_id="parent-sess", db=None, profile_key=None):
    parent = MagicMock()
    parent.base_url, parent.api_key, parent.provider = "https://openrouter.ai/api/v1", "parent-key", "openrouter"
    parent.api_mode, parent.model, parent.platform = "chat_completions", "parent-model", "cli"
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent.session_id, parent._delegate_depth, parent._interrupt_requested = session_id, 0, False
    parent.status, parent._current_turn_id = "running", "turn-1"
    parent._subagent_id = None
    parent._delegate_parent_ref = None
    parent._delegate_root_session_id = None
    parent.enabled_toolsets = ["terminal", "file", "web", "agent_messaging"]
    parent.disabled_toolsets = []
    parent.valid_tool_names = {"delegate_task", "agent_message", "list_agents"}
    parent._session_db = db if db is not None else SessionDB()
    parent._session_db.create_session(session_id, "cli")
    parent._delegate_profile_key = profile_key or str(parent._session_db.db_path)
    parent._active_children, parent._active_children_lock = [], threading.Lock()
    parent._print_fn = parent.tool_progress_callback = parent.thinking_callback = None
    return parent


def _completion(delegation_id):
    def find():
        while not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("type") == "async_delegation" and evt.get("delegation_id") == delegation_id:
                return evt
        return None
    return wait_until(find, what=f"completion of {delegation_id}")


def _completions(*delegation_ids):
    """Collect several completion events regardless of arrival order."""
    remaining = set(delegation_ids)
    def find():
        while not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("type") == "async_delegation" and evt.get("delegation_id") in remaining:
                remaining.discard(evt["delegation_id"])
        return not remaining
    return wait_until(find, what=f"completions of {sorted(remaining)}")


def _drain_completions():
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


@pytest.fixture(autouse=True)
def _clean_state(tmp_path, monkeypatch):
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _recent_subagents
    from tools.delegate_tool_messages import reset_message_state

    # Every test (and every background dispatch it leaves behind) must use its
    # own database file. The session-wide conftest points the default at a
    # shared per-test home; a late worker from the previous test can then
    # create/lock the next test's file at the wrong moment on Windows.
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    _drain_completions()
    with _active_subagents_lock:
        _active_subagents.clear()
        _recent_subagents.clear()
    reset_message_state()
    yield
    with _active_subagents_lock:
        _active_subagents.clear()
        _recent_subagents.clear()
    reset_message_state()
    _drain_completions()


class _ScriptedChild:
    """A child stand-in that runs a caller-supplied function inside the real dispatch path."""

    def __init__(self, script):
        self.script = script
        self.builds = []
        self.turns = []
        self.before_persist = None
        self.finished = threading.Event()

    def factory(self, **kw):
        child = MagicMock()
        child.session_id = kw.get("session_id") or f"child-sess-{len(self.builds)}"
        child.model, child.provider = kw.get("model"), kw.get("provider")
        child._parent_session_id = kw.get("parent_session_id")
        child.status = "running"
        child._interrupt_requested = False
        child._delegate_inbox = []
        child._delegate_inbox_lock = threading.Lock()
        child._delegate_inbox_reopens = 0
        child._delegate_root_session_id = None
        child._delegate_profile_key = None
        child.get_activity_summary.return_value = {"api_call_count": 1, "current_tool": None, "last_activity_ts": 0}
        db = kw["session_db"]
        child._script_db = db
        child._session_db = db

        def run_conversation(user_message, conversation_history=None, **_):
            self.turns.append({"session_id": child.session_id, "message": user_message,
                               "history": conversation_history})
            try:
                if self.before_persist is not None:
                    self.before_persist(child, user_message)
                db.create_session(child.session_id, "subagent", parent_session_id=kw.get("parent_session_id"))
                db.append_message(child.session_id, "user", user_message)
                reply = self.script(child, user_message) or "done"
                db.append_message(child.session_id, "assistant", reply)
                return {"final_response": reply, "completed": True, "api_calls": 1}
            finally:
                self.finished.set()

        child.run_conversation.side_effect = run_conversation
        self.builds.append(dict(kw, child=child))
        return child


def _spawn(parent, goal, script):
    from tools.delegate_tool import delegate_task
    model = _ScriptedChild(script)
    with patch("run_agent.AIAgent", side_effect=model.factory):
        out = json.loads(delegate_task(tasks=[{"goal": goal}], background=True, parent_agent=parent))
        assert out["status"] == "dispatched", out
    return out["subagent_ids"][0], out["delegation_id"], model


def _live_child(root, sid, *, profile_key=None, accepting=True, interrupted=False, agent=None):
    from tools.delegate_tool_registry import _register_subagent
    if agent is None:
        agent = SimpleNamespace(
            session_id=f"{sid}-sess", _subagent_id=sid,
            _delegate_root_session_id=root.session_id,
            _delegate_profile_key=profile_key or str(root._session_db.db_path),
            status="running", _interrupt_requested=interrupted,
            _delegate_inbox=[], _delegate_inbox_lock=threading.Lock(),
        )
    else:
        agent._subagent_id = sid
        agent._delegate_root_session_id = root.session_id
        agent._delegate_profile_key = profile_key or str(root._session_db.db_path)
        agent.status = "running"
        agent._interrupt_requested = interrupted
    if getattr(agent, "_delegate_parent_ref", None) is None or not callable(getattr(agent, "_delegate_parent_ref", None)):
        agent._delegate_parent_ref = lambda: root
    _register_subagent({
        "subagent_id": sid, "agent": agent, "goal": "g", "status": "running",
        "owner_agent_session_id": root.session_id, "root_session_id": root.session_id,
        "profile_key": agent._delegate_profile_key, "accepting_messages": accepting,
    })
    return agent


def test_two_siblings_exchange_a_message_without_parent_relay():
    root = _parent()
    seen = {}

    def wait_script(child, user_message):
        # B waits until A's real send has arrived, then uses the production step drain.
        from tools.delegate_tool_messages import inject_inbox_messages

        wait_until(lambda: getattr(child, "_delegate_inbox", []), what="B inbox")
        messages = [{"role": "user", "content": user_message},
                    {"role": "assistant", "tool_calls": [{"id": "t1", "type": "function",
                                                          "function": {"name": "terminal", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "t1", "content": "tool output"}]
        inject_inbox_messages(child, messages)
        for m in messages:
            if m.get("display_kind") == "agent_message":
                child._script_db.append_message(child.session_id, "user", m["content"])
        seen["messages"] = messages
        return "B got it"

    b_sid, b_deleg, b_model = _spawn(root, "wait for a peer message", wait_script)
    b_sess = b_model.builds[0]["child"].session_id

    def send_script(child, user_message):
        from tools.agent_messaging_tool import agent_message
        result = json.loads(agent_message(target=b_sid, message="hello from A", parent_agent=child))
        seen["send"] = result
        return "A sent"

    a_sid, a_deleg, _ = _spawn(root, f"send a greeting to {b_sid}", send_script)
    _completions(a_deleg, b_deleg)

    assert seen["send"]["state"] == "accepted", seen["send"]
    assert seen["send"]["message_id"].startswith("msg-")
    assert seen["send"]["sender"]["id"] == a_sid
    assert seen["send"]["recipient"]["id"] == b_sid
    user_rows = [m for m in seen["messages"] if m.get("role") == "user" and "agent message" in str(m.get("content", ""))]
    assert len(user_rows) == 1
    assert a_sid in user_rows[0]["content"] and "hello from A" in user_rows[0]["content"]
    stored = root._session_db.get_messages_as_conversation(b_sess)
    assert any(a_sid in str(m.get("content")) and "hello from A" in str(m.get("content")) for m in stored)
    # Parent never relayed the text: its transcript has no message from A.
    assert not any("hello from A" in str(m.get("content"))
                   for m in root._session_db.get_messages_as_conversation(root.session_id))


def _real_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("agent: {}\n", encoding="utf-8")
    from run_agent import AIAgent
    from model_tools import _clear_tool_defs_cache
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    return AIAgent(session_db=SessionDB(db_path=tmp_path / "proof.db"),
                   model="test-model", provider="openai-compat", api_key="test",
                   base_url="http://127.0.0.1:1/v1", max_iterations=4,
                   quiet_mode=True, skip_context_files=True, skip_memory=True)


def test_siblings_can_exchange_a_two_way_message_without_parent_relay():
    root = _parent()
    a_sid_holder, b_ready, a_reply = {}, threading.Event(), {}

    def b_script(child, user_message):
        from tools.delegate_tool_messages import inject_inbox_messages
        from tools.agent_messaging_tool import agent_message

        wait_until(lambda: getattr(child, "_delegate_inbox", []), what="B inbox")
        messages = [{"role": "tool", "tool_call_id": "t1", "content": "tool output"}]
        inject_inbox_messages(child, messages)
        for m in messages:
            if m.get("display_kind") == "agent_message":
                child._script_db.append_message(child.session_id, "user", m["content"])
        reply = json.loads(agent_message(target=a_sid_holder["sid"], message="reply from B",
                                         message_id="msg-b-reply", parent_agent=child))
        a_reply["send"] = reply
        return "B replied"

    b_sid, b_deleg, _ = _spawn(root, "wait then reply", b_script)
    b_ready.set()

    def a_script(child, user_message):
        from tools.delegate_tool_messages import inject_inbox_messages
        from tools.agent_messaging_tool import agent_message

        a_sid_holder["sid"] = child._subagent_id
        b_ready.wait(5)
        first = json.loads(agent_message(target=b_sid, message="hello B", message_id="msg-a-hello",
                                         parent_agent=child))
        a_reply["first"] = first
        wait_until(lambda: getattr(child, "_delegate_inbox", []), what="A reply inbox")
        messages = [{"role": "tool", "tool_call_id": "t2", "content": "tool output"}]
        inject_inbox_messages(child, messages)
        for m in messages:
            if m.get("display_kind") == "agent_message":
                child._script_db.append_message(child.session_id, "user", m["content"])
        a_reply["messages"] = messages
        return "A got the reply"

    a_sid, a_deleg, _ = _spawn(root, "send then wait", a_script)
    a_sid_holder["sid"] = a_sid
    _completions(a_deleg, b_deleg)

    assert a_reply["first"]["state"] == "accepted", a_reply["first"]
    assert a_reply["send"]["state"] == "accepted", a_reply["send"]
    assert any(b_sid in str(m.get("content")) and "reply from B" in str(m.get("content"))
               for m in a_reply["messages"])


def test_busy_message_is_a_new_user_row_after_the_tool_result(tmp_path, monkeypatch):
    from agent.turn_iteration_prep import prepare_iteration
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_messages import message_status

    root = _parent()
    sender = _live_child(root, "sa-0-a")
    child = _real_agent(tmp_path, monkeypatch)
    try:
        _live_child(root, "sa-0-b", agent=child)
        sent = json.loads(agent_message(target="sa-0-b", message="please check the parser",
                                        message_id="msg-step-1", parent_agent=sender))
        assert sent["state"] == "accepted", sent
        messages = [
            {"role": "user", "content": "work"},
            {"role": "assistant", "tool_calls": [{"id": "t1", "type": "function",
                                                  "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "TOOL_RESULT_SENTINEL"},
        ]
        prepare_iteration(child, messages=messages, api_call_count=1)
        assert messages[-1]["role"] == "user" and messages[-1].get("display_kind") == "agent_message"
        assert "sa-0-a" in messages[-1]["content"] and "please check the parser" in messages[-1]["content"]
        assert messages[-2]["content"] == "TOOL_RESULT_SENTINEL"  # old tool result untouched
        assert child._delegate_inbox == []
        assert message_status("msg-step-1")["state"] == "delivered"
    finally:
        child._session_db.close()


def test_turn_end_hook_delivers_an_inbox_message_that_lost_the_step_race():
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_messages import message_status, reopen_finished_turn_for_inbox

    root = _parent()
    sender = _live_child(root, "sa-0-a")
    child = _live_child(root, "sa-0-b")
    sent = json.loads(agent_message(target="sa-0-b", message="late but real", message_id="msg-race-1",
                                    parent_agent=sender))
    assert sent["state"] == "accepted", sent
    messages = [{"role": "user", "content": "work"}, {"role": "assistant", "content": "final answer"}]
    assert reopen_finished_turn_for_inbox(child, messages) is True
    assert messages[-1]["role"] == "user" and "late but real" in messages[-1]["content"]
    assert message_status("msg-race-1")["state"] == "delivered"
    assert reopen_finished_turn_for_inbox(child, messages) is False  # empty inbox


def test_turn_end_after_the_reopen_budget_rejects_the_pending_message():
    """The fourth reopen is the bound; a fifth pending message is not silently
    left ``accepted`` while the turn is over."""
    from agent.conversation_loop import _verdict_action
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_messages import message_status

    root = _parent("root-a")
    sender = _live_child(root, "sa-0-a")
    child = _live_child(root, "sa-0-b")
    sent = json.loads(agent_message(target="sa-0-b", message="over budget", message_id="msg-reopen-5",
                                    parent_agent=sender))
    assert sent["state"] == "accepted", sent
    child._delegate_inbox_reopens = 4
    verdict = SimpleNamespace(action="break")
    loop_state = SimpleNamespace(messages=[{"role": "assistant", "content": "final"}])
    assert _verdict_action(child, loop_state, verdict) == "break"
    assert message_status("msg-reopen-5")["state"] == "rejected"
    assert message_status("msg-reopen-5")["reason"] == "target_finished_before_delivery"


def test_self_unknown_unrelated_tree_and_cross_profile_are_rejected():
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_registry import _register_subagent

    root = _parent("root-a")
    sender = _live_child(root, "sa-0-a")
    other_root = _parent("root-b")
    _live_child(other_root, "sa-0-x")
    sender._delegate_profile_key = "profile-a"
    _live_child(root, "sa-0-cross", profile_key="profile-b")

    def send(target):
        return json.loads(agent_message(target=target, message="hello", parent_agent=sender))

    assert send("sa-0-a")["state"] == "rejected" and send("sa-0-a")["reason"] == "self"
    assert send("sa-0-missing")["state"] == "rejected" and send("sa-0-missing")["reason"] == "unknown_target"
    assert send("sa-0-x")["state"] == "rejected" and send("sa-0-x")["reason"] == "unauthorized"
    assert send("sa-0-cross")["state"] == "rejected" and send("sa-0-cross")["reason"] == "unauthorized"


def test_duplicate_message_id_is_idempotent_but_a_conflict_is_rejected():
    from tools.agent_messaging_tool import agent_message

    root = _parent("root-a")
    sender = _live_child(root, "sa-0-a")
    target = _live_child(root, "sa-0-b")
    first = json.loads(agent_message(target="sa-0-b", message="one", message_id="msg-dup-1", parent_agent=sender))
    again = json.loads(agent_message(target="sa-0-b", message="one", message_id="msg-dup-1", parent_agent=sender))
    conflict = json.loads(agent_message(target="sa-0-b", message="two", message_id="msg-dup-1", parent_agent=sender))
    assert first["state"] == again["state"] == "accepted"
    assert first["message_id"] == again["message_id"] == "msg-dup-1"
    assert len(target._delegate_inbox) == 1
    assert conflict["state"] == "rejected" and conflict["reason"] == "message_id_conflict"


def test_duplicate_id_replays_recorded_state_after_the_target_leaves_the_registry():
    """Idempotency is a ledger property, not a target-resolution property."""
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _recent_subagents

    root = _parent("root-a")
    sender = _live_child(root, "sa-0-a")
    _live_child(root, "sa-0-b")
    first = json.loads(agent_message(target="sa-0-b", message="one", message_id="msg-dup-gone-1",
                                     parent_agent=sender))
    assert first["state"] == "accepted", first
    with _active_subagents_lock:
        _active_subagents.pop("sa-0-b", None)
        _recent_subagents.pop("sa-0-b", None)
    again = json.loads(agent_message(target="sa-0-b", message="one", message_id="msg-dup-gone-1",
                                     parent_agent=sender))
    assert again["state"] == "accepted", again
    assert again["message_id"] == "msg-dup-gone-1"


def test_accepted_is_not_delivered_until_the_target_consumes_it():
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_messages import message_status

    root = _parent("root-a")
    sender = _live_child(root, "sa-0-a")
    target = _live_child(root, "sa-0-b")
    out = json.loads(agent_message(target="sa-0-b", message="queued", message_id="msg-accept-1",
                                   parent_agent=sender))
    assert out["state"] == "accepted" and out["accepted"] is True
    assert message_status("msg-accept-1")["state"] == "accepted"
    assert target._delegate_inbox and target._delegate_inbox[0]["message_id"] == "msg-accept-1"


def test_a_message_still_queued_when_the_run_finishes_is_not_reported_delivered():
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_messages import message_status
    from tools.delegate_tool_registry import _unregister_subagent

    root = _parent("root-a")
    sender = _live_child(root, "sa-0-a")
    target = _live_child(root, "sa-0-b")
    out = json.loads(agent_message(target="sa-0-b", message="too late", message_id="msg-late-1",
                                   parent_agent=sender))
    assert out["state"] == "accepted", out
    _unregister_subagent("sa-0-b", agent=target)
    assert message_status("msg-late-1")["state"] == "rejected"
    assert message_status("msg-late-1")["reason"] == "target_finished_before_delivery"


def test_cancelled_and_closed_targets_are_rejected_explicitly():
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_registry import _recent_subagents, _active_subagents_lock

    root = _parent("root-a")
    sender = _live_child(root, "sa-0-a")
    _live_child(root, "sa-0-cancelled", accepting=False, interrupted=True)
    with _active_subagents_lock:
        _recent_subagents["sa-0-stopping"] = {
            "goal": "g", "state": "stopping", "owner_agent_session_id": root.session_id,
            "root_session_id": root.session_id, "subagent_id": "sa-0-stopping",
            "profile_key": str(root._session_db.db_path),
        }
        _recent_subagents["sa-0-noresume"] = {
            "goal": "g", "state": "resumable", "owner_agent_session_id": root.session_id,
            "root_session_id": root.session_id, "subagent_id": "sa-0-noresume",
            "profile_key": str(root._session_db.db_path),
        }

    cancelled = json.loads(agent_message(target="sa-0-cancelled", message="x", parent_agent=sender))
    stopping = json.loads(agent_message(target="sa-0-stopping", message="x", parent_agent=sender))
    noresume = json.loads(agent_message(target="sa-0-noresume", message="x", parent_agent=sender))
    assert cancelled["state"] == stopping["state"] == noresume["state"] == "rejected"
    assert cancelled["reason"] == "target_cancelled"
    assert stopping["reason"] == "target_stopping"
    assert noresume["reason"] == "not_continuable"


def test_stopped_sibling_receives_the_message_through_true_continuation():
    from tools.agent_messaging_tool import agent_message

    root = _parent("root-a")
    sent = {}

    def b_script(child, user_message):
        return "B first round"

    b_sid, b_deleg, b_model = _spawn(root, "first task", b_script)
    _completion(b_deleg)
    b_sess = b_model.builds[0]["child"].session_id

    a_model = _ScriptedChild(lambda child, user_message: "A base")

    def a_script(child, user_message):
        sent.update(json.loads(agent_message(target=b_sid, message="wake for round two",
                                             message_id="msg-cold-stop-1", parent_agent=child)))
        return "A woke B"

    a_model.script = a_script

    def factory(**kw):
        # A new child has no session_id yet; B's continuation carries B's id.
        if kw.get("session_id") == b_sess:
            return b_model.factory(**kw)
        return a_model.factory(**kw)

    from tools.delegate_tool import delegate_task
    with patch("run_agent.AIAgent", side_effect=factory):
        a_out = json.loads(delegate_task(tasks=[{"goal": f"wake {b_sid}"}], background=True, parent_agent=root))
        a_deleg = a_out["delegation_id"]
        a_sid = a_out["subagent_ids"][0]
        _completion(a_deleg)
        wait_until(lambda: sent.get("state") == "accepted", what="cold send accepted")
        wait_until(lambda: len(b_model.builds) >= 2, what="continued B build")
        from tools.delegate_tool_messages import message_status
        # A completed continuation is promoted delivered -> processed in its run-end
        # finally. The process-local mirror is updated just before the durable write,
        # so wait for the durable ``delivered_at`` itself; no row, no delivery.
        from tools.delegate_tool_collab import get_message
        wait_until(
            lambda: (get_message(str(root._session_db.db_path), "msg-cold-stop-1") or {}).get("delivered_at"),
            what="continued B delivery")
    assert sent["state"] == "accepted", sent
    delivered_row = get_message(str(root._session_db.db_path), "msg-cold-stop-1")
    assert delivered_row["delivered_at"] is not None, delivered_row
    assert message_status("msg-cold-stop-1")["state"] in ("delivered", "processed")
    stored = root._session_db.get_messages_as_conversation(b_sess)
    assert any("wake for round two" in str(m.get("content")) and a_sid in str(m.get("content"))
               for m in stored)
    assert b_model.builds[1]["session_id"] == b_sess
    continued_history = b_model.turns[1]["history"] or []
    assert any("first task" in str(m.get("content")) for m in continued_history)
    assert any("B first round" in str(m.get("content")) for m in continued_history)


def test_cold_message_stays_accepted_until_the_child_turn_persists_it(tmp_path):
    """The ledger must not claim delivery before the user row exists.

    The continuation is gated inside ``run_conversation`` before its first
    persistent write, so this is deterministic: old code stamped ``delivered``
    at the call boundary and fails here even though nothing was materialized.
    """
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_messages import message_status

    root = _parent("root-a", db=SessionDB(db_path=tmp_path / "state.db"))
    entered, release = threading.Event(), threading.Event()

    def b_first(child, user_message):
        return "B first round"

    b_sid, b_deleg, b_model = _spawn(root, "first task", b_first)
    _completion(b_deleg)
    b_sess = b_model.builds[0]["child"].session_id

    def hold_before_persist(child, user_message):
        entered.set()
        if not release.wait(10):
            raise AssertionError("test never released the gated child")

    b_model.before_persist = hold_before_persist
    sent = {}

    def a_script(child, user_message):
        sent.update(json.loads(agent_message(target=b_sid, message="materialize me",
                                             message_id="msg-gate-1", parent_agent=child)))
        return "A sent"

    a_model = _ScriptedChild(lambda child, user_message: "A base")
    a_model.script = a_script

    def factory(**kw):
        if kw.get("session_id") == b_sess:
            return b_model.factory(**kw)
        return a_model.factory(**kw)

    from tools.delegate_tool import delegate_task
    a_sid = None
    try:
        with patch("run_agent.AIAgent", side_effect=factory):
            a_out = json.loads(delegate_task(tasks=[{"goal": f"wake {b_sid}"}], background=True,
                                             parent_agent=root))
            a_sid = a_out["subagent_ids"][0]
            _completion(a_out["delegation_id"])
            wait_until(lambda: entered.is_set(), what="gated continuation entered")
            assert sent.get("state") == "accepted", sent
            assert message_status("msg-gate-1")["state"] == "accepted"
            assert not any("materialize me" in str(m.get("content"))
                           for m in root._session_db.get_messages_as_conversation(b_sess))
    finally:
        release.set()

    wait_until(lambda: message_status("msg-gate-1")["state"] == "delivered",
               what="gated message delivered")
    stored = root._session_db.get_messages_as_conversation(b_sess)
    assert any("materialize me" in str(m.get("content")) and a_sid in str(m.get("content"))
               for m in stored)


def test_cold_message_is_not_delivered_when_the_child_fails_before_persisting(tmp_path):
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_messages import message_status

    root = _parent("root-a", db=SessionDB(db_path=tmp_path / "state.db"))
    entered = threading.Event()

    def b_first(child, user_message):
        return "B first round"

    b_sid, b_deleg, b_model = _spawn(root, "first task", b_first)
    _completion(b_deleg)
    b_sess = b_model.builds[0]["child"].session_id
    b_model.finished.clear()

    def fail_before_persist(child, user_message):
        entered.set()
        raise RuntimeError("provider unavailable before persistence")

    b_model.before_persist = fail_before_persist
    sent = {}

    def a_script(child, user_message):
        sent.update(json.loads(agent_message(target=b_sid, message="do not claim me",
                                             message_id="msg-fail-1", parent_agent=child)))
        return "A sent"

    a_model = _ScriptedChild(lambda child, user_message: "A base")
    a_model.script = a_script

    def factory(**kw):
        if kw.get("session_id") == b_sess:
            return b_model.factory(**kw)
        return a_model.factory(**kw)

    from tools.delegate_tool import delegate_task
    with patch("run_agent.AIAgent", side_effect=factory):
        a_out = json.loads(delegate_task(tasks=[{"goal": f"wake {b_sid}"}], background=True,
                                         parent_agent=root))
        _completion(a_out["delegation_id"])
        wait_until(lambda: entered.is_set(), what="failing continuation entered")
        wait_until(lambda: b_model.finished.is_set(), what="failing continuation finished")

    assert sent.get("state") == "accepted", sent
    # Stage 4 honesty: the run ended before any user row existed. This is a provable
    # no-effect outcome, so the ledger rejects it instead of leaving a stale accepted.
    wait_until(lambda: message_status("msg-fail-1")["state"] == "rejected",
               what="terminal rejection after the pre-persistence failure")
    assert message_status("msg-fail-1")["reason"] == "continuation_failed_before_delivery"
    assert not any("do not claim me" in str(m.get("content"))
                   for m in root._session_db.get_messages_as_conversation(b_sess))


def test_two_busy_sends_are_both_accepted_and_two_cold_sends_claim_one_continuation():
    from tools.agent_messaging_tool import agent_message

    root = _parent("root-a")
    sender = _live_child(root, "sa-0-a")
    target = _live_child(root, "sa-0-b")

    def busy(n):
        return json.loads(agent_message(target="sa-0-b", message=f"m{n}", message_id=f"msg-busy-{n}",
                                        parent_agent=sender))
    with ThreadPoolExecutor(max_workers=2) as pool:
        out = [f.result(timeout=10) for f in (pool.submit(busy, 1), pool.submit(busy, 2))]
    assert all(r["state"] == "accepted" for r in out), out
    assert {e["message_id"] for e in target._delegate_inbox} == {"msg-busy-1", "msg-busy-2"}

    # Cold target: build a finished child, then race two sends through the public handler.
    def script(child, user_message):
        return "first"
    sid, deleg, model = _spawn(root, "cold target", script)
    _completion(deleg)

    def cold(n):
        return json.loads(agent_message(target=sid, message=f"cold {n}", message_id=f"msg-cold-{n}",
                                        parent_agent=sender))
    with patch("run_agent.AIAgent", side_effect=model.factory):
        with ThreadPoolExecutor(max_workers=2) as pool:
            cold_out = [f.result(timeout=20) for f in (pool.submit(cold, 1), pool.submit(cold, 2))]
        assert sum(r["state"] == "accepted" for r in cold_out) == 1, cold_out
        assert sum(r["state"] == "rejected" for r in cold_out) == 1, cold_out
        wait_until(lambda: len(model.builds) >= 2, what="cold continuation build")


def test_child_inherits_messaging_only_when_the_parent_has_it():
    from tools.delegate_tool_toolsets import _resolve_child_toolsets

    parent = SimpleNamespace(enabled_toolsets=["terminal", "file", "web"],
                             valid_tool_names={"agent_message", "delegate_task"},
                             disabled_toolsets=[])
    enabled, _ = _resolve_child_toolsets(parent, None, "leaf")
    assert "agent_messaging" in enabled

    without = SimpleNamespace(enabled_toolsets=["terminal", "file", "web"],
                              valid_tool_names={"delegate_task"}, disabled_toolsets=[])
    enabled2, _ = _resolve_child_toolsets(without, None, "leaf")
    assert "agent_messaging" not in enabled2


def test_list_agents_shows_the_same_tree_with_stable_ids():
    from tools.agent_messaging_tool import list_agents

    root = _parent("root-a")
    sender = _live_child(root, "sa-0-a")
    _live_child(root, "sa-0-b")
    other = _parent("root-b")
    _live_child(other, "sa-0-x")
    rows = json.loads(list_agents(parent_agent=sender))
    ids = {r["id"] for r in rows}
    assert {"lead", "sa-0-a", "sa-0-b"}.issubset(ids)
    assert "sa-0-x" not in ids
    assert next(r for r in rows if r["id"] == "sa-0-a")["self"] is True
