"""Stage-3/4 acceptance: bounded event-driven collaboration and durable recovery.

The only fake is the model/agent run function. The delegation registry, child build,
SessionDB schema/migration, durable collaboration store, continuation engine, inbox
hooks and completion queue are the production paths. No real model/API/network call.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from tests.waiting import wait_until
from tools.process_registry import process_registry

_ROOT = Path(__file__).resolve().parents[2]
_EVENTS: list[dict] = []


# ── harness ────────────────────────────────────────────────────────────────

def _parent(session_id="root-sess", db=None, profile_key=None):
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


def _drain_completions():
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _collect_events():
    while not process_registry.completion_queue.empty():
        evt = process_registry.completion_queue.get_nowait()
        if evt.get("type") == "async_delegation":
            _EVENTS.append(evt)


def _event_summary(evt):
    parts = [str(evt.get("summary") or "")]
    for row in evt.get("results") or ():
        if isinstance(row, dict):
            parts.append(str(row.get("summary") or ""))
    return "\n".join(p for p in parts if p)


def _wait_event(predicate, what, timeout=40.0):
    def find():
        _collect_events()
        return next((e for e in _EVENTS if predicate(e)), None)
    return wait_until(find, timeout=timeout, what=what)


@pytest.fixture(autouse=True)
def _clean_state(tmp_path, monkeypatch):
    import hermes_state
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _recent_subagents
    from tools.delegate_tool_messages import reset_message_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    try:
        from tools.delegate_tool_collab import reset_collab_state
        reset_collab_state()
    except ImportError:
        pass
    _EVENTS.clear()
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
    try:
        from tools.delegate_tool_collab import reset_collab_state
        reset_collab_state()
    except ImportError:
        pass
    _drain_completions()
    _EVENTS.clear()


class _ScriptedChild:
    """A controlled provider/event script running inside the real dispatch path."""

    def __init__(self, script):
        self.script = script
        self.builds = []
        self.turns = []
        self.before_persist = None
        self.after_persist = None
        self.finished = threading.Event()
        self._prefix = f"child-{id(self):x}"

    def factory(self, **kw):
        child = MagicMock()
        child.session_id = kw.get("session_id") or f"{self._prefix}-{len(self.builds)}"
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
                from tools.delegate_tool_messages import refresh_cold_message_delivery
                refresh_cold_message_delivery(child)
                if self.after_persist is not None:
                    self.after_persist(child, user_message)
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
    if not callable(getattr(agent, "_delegate_parent_ref", None)):
        agent._delegate_parent_ref = lambda: root
    _register_subagent({
        "subagent_id": sid, "agent": agent, "goal": "g", "status": "running",
        "owner_agent_session_id": root.session_id, "root_session_id": root.session_id,
        "profile_key": agent._delegate_profile_key, "accepting_messages": accepting,
    })
    return agent


def _drain_inbox(child):
    """Production step-boundary drain, with the fake provider persisting the new row."""
    from tools.delegate_tool_messages import inject_inbox_messages
    messages = [{"role": "tool", "tool_call_id": "t1", "content": "tool output"}]
    count = inject_inbox_messages(child, messages)
    for row in messages:
        if row.get("display_kind") == "agent_message":
            child._script_db.append_message(child.session_id, "user", row["content"])
    return count


def _is_resumable(root, sid):
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _recent_subagents
    with _active_subagents_lock:
        return sid not in _active_subagents and (_recent_subagents.get(sid) or {}).get("state") == "resumable"


def _db_path(root):
    return str(root._session_db.db_path)


# ── stage 3: event-driven ask/reply/continue with root evidence ─────────────

def test_ask_reply_continues_the_asker_and_root_receives_real_evidence(tmp_path):
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_collab import get_message

    root = _parent("root-collab", db=SessionDB(db_path=tmp_path / "state.db"))
    evidence_file = tmp_path / "b-evidence.txt"
    evidence = "parser-lines=17;sha=deterministic"
    evidence_file.write_text(evidence, encoding="utf-8")
    ids, sent, b_ready = {}, {}, threading.Event()

    def b_script(child, user_message):
        b_ready.set()
        wait_until(lambda: child._delegate_inbox, what="B inbox")
        _drain_inbox(child)
        from tools.delegate_tool_messages import finish_collaboration_run
        finish_collaboration_run(child, {"completed": True, "final_response": "B drained"})
        wait_until(lambda: _is_resumable(root, ids["a"]), what="A finished")
        sent["reply"] = json.loads(agent_message(
            target=ids["a"], message="B's real result is in b-evidence.txt; continue the task",
            message_id="msg-b-reply", parent_agent=child))
        return "B wrote evidence and replied"

    b_sid, _, b_model = _spawn(root, "receive and answer", b_script)
    ids["b"], b_sess = b_sid, b_model.builds[0]["child"].session_id

    a_model = _ScriptedChild(lambda c, m: "unused")

    def a_script(child, user_message):
        if len(a_model.turns) == 1:
            assert b_ready.wait(15), "B never became live"
            sent["ask"] = json.loads(agent_message(
                target=ids["b"], message="A asks B for the parser count",
                message_id="msg-a-ask", parent_agent=child))
            return "A asked; waiting for the reply"
        # The continued run must see its own prior conversation and B's file evidence.
        body = evidence_file.read_text(encoding="utf-8")
        return f"EVIDENCE={body}"

    a_model.script = a_script

    def factory(**kw):
        sid = kw.get("session_id")
        if sid == b_sess:
            return b_model.factory(**kw)
        return a_model.factory(**kw)

    from tools.delegate_tool import delegate_task
    with patch("run_agent.AIAgent", side_effect=factory):
        a_out = json.loads(delegate_task(tasks=[{"goal": f"ask {b_sid} then continue"}],
                                         background=True, parent_agent=root))
        ids["a"] = a_out["subagent_ids"][0]
        a_sess = a_model.builds[0]["child"].session_id
        wait_until(lambda: sent.get("reply", {}).get("state") == "accepted", timeout=30,
                   what="B's reply accepted as a continuation")
        try:
            evt = _wait_event(lambda e: "EVIDENCE=" in _event_summary(e),
                              "A's continued evidence")
        except AssertionError:
            print("DBG EVENTS", json.dumps(_EVENTS, default=str)[:4000], flush=True)
            raise
        wait_until(lambda: len(a_model.builds) >= 2, what="A's true continuation build")
    assert sent["ask"]["state"] == "accepted", sent["ask"]
    assert sent["reply"]["state"] == "accepted", sent["reply"]
    assert sent["reply"]["delivery"] == "continuation"
    # True continuation: same child id and session, prior history appended.
    assert a_model.builds[1]["child"]._subagent_id == ids["a"]
    assert a_model.builds[1]["session_id"] == a_sess
    history = a_model.turns[1]["history"] or []
    assert any("A asked; waiting" in str(m.get("content")) for m in history)
    assert any("A asks B" in str(m.get("content")) for m in root._session_db.get_messages_as_conversation(b_sess))
    # Root gets the evidence through the NATIVE completion channel, not a relay turn.
    assert evt["parent_session_id"] == root.session_id
    assert evidence in _event_summary(evt)
    # No parent relay: the root transcript never contains the peer text.
    root_rows = root._session_db.get_messages_as_conversation(root.session_id)
    assert not any("A asks B" in str(m.get("content")) or "b-evidence" in str(m.get("content")) for m in root_rows)
    # Durable states: A's reply was materialized and its consuming run ended.
    reply_row = get_message(_db_path(root), "msg-b-reply")
    assert reply_row["state"] == "processed", reply_row
    assert reply_row["outcome"] == "completed"
    ask_row = get_message(_db_path(root), "msg-a-ask")
    assert ask_row["state"] == "processed", ask_row


def test_delivered_precedes_processed_and_neither_is_inferred(tmp_path):
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_collab import get_message

    root = _parent("root-order", db=SessionDB(db_path=tmp_path / "state.db"))
    gate_entered, gate_release = threading.Event(), threading.Event()

    def b_first(child, user_message):
        return "B first round"

    b_sid, b_deleg, b_model = _spawn(root, "first", b_first)
    _wait_event(lambda e: e.get("delegation_id") == b_deleg, "B first completion")
    b_sess = b_model.builds[0]["child"].session_id

    def hold_after_persist(child, user_message):
        gate_entered.set()
        assert gate_release.wait(15), "test never released the gated continuation"

    b_model.after_persist = hold_after_persist
    sent = {}

    def a_script(child, user_message):
        sent.update(json.loads(agent_message(target=b_sid, message="consume me after the row",
                                             message_id="msg-order-1", parent_agent=child)))
        return "A sent"

    a_model = _ScriptedChild(lambda c, m: "A base")
    a_model.script = a_script

    def factory(**kw):
        return b_model.factory(**kw) if kw.get("session_id") == b_sess else a_model.factory(**kw)

    from tools.delegate_tool import delegate_task
    try:
        with patch("run_agent.AIAgent", side_effect=factory):
            a_out = json.loads(delegate_task(tasks=[{"goal": f"wake {b_sid}"}], background=True,
                                             parent_agent=root))
            _wait_event(lambda e: e.get("delegation_id") == a_out["delegation_id"], "A completion")
            wait_until(gate_entered.is_set, timeout=20, what="gated continuation persisted its row")
            wait_until(lambda: get_message(_db_path(root), "msg-order-1")["state"] == "delivered",
                       what="delivered state")
            row = get_message(_db_path(root), "msg-order-1")
            assert row["delivered_at"] and not row["processed_at"], row
    finally:
        gate_release.set()

    wait_until(lambda: get_message(_db_path(root), "msg-order-1")["state"] == "processed",
               what="processed after the consuming run ended")
    row = get_message(_db_path(root), "msg-order-1")
    assert row["processed_at"] and row["outcome"] == "completed"


def test_round_ceiling_stops_a_real_ping_pong(tmp_path, monkeypatch):
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_collab import get_message, tree
    from tools import delegate_tool_collab

    monkeypatch.setattr(delegate_tool_collab, "_limits", lambda: (2, 3))
    root = _parent("root-round", db=SessionDB(db_path=tmp_path / "state.db"))
    ids, sent, b_ready, spawns = {}, {}, threading.Event(), []

    b_model = _ScriptedChild(lambda c, m: "unused")

    def b_script(child, user_message):
        b_ready.set()
        if len(b_model.turns) == 1:
            wait_until(lambda: child._delegate_inbox, what="B first inbox")
            _drain_inbox(child)
            wait_until(lambda: _is_resumable(root, ids["a"]), what="A first run finished")
            sent["b1"] = json.loads(agent_message(target=ids["a"], message="B replies round one",
                                                  message_id="msg-b-1", parent_agent=child))
            return "B replied"
        sent["b2"] = json.loads(agent_message(target=ids["a"], message="B tries round three",
                                              message_id="msg-b-2", parent_agent=child))
        return "B blocked by the round ceiling"

    a_model = _ScriptedChild(lambda c, m: "unused")

    def a_script(child, user_message):
        if len(a_model.turns) == 1:
            assert b_ready.wait(15), "B never became live"
            sent["a1"] = json.loads(agent_message(target=ids["b"], message="A asks round one",
                                                  message_id="msg-a-1", parent_agent=child))
            return "A asked"
        wait_until(lambda: _is_resumable(root, ids["b"]), what="B first run finished")
        sent["a2"] = json.loads(agent_message(target=ids["b"], message="A continues round two",
                                              message_id="msg-a-2", parent_agent=child))
        return "A continued once"

    a_model.script = a_script
    b_model.script = b_script

    ids["b_sess"] = str(tmp_path / "b-session-id")

    def factory(**kw):
        sid = kw.get("session_id")
        if sid and sid == ids.get("b_sess"):
            return b_model.factory(**kw)
        if sid:
            return a_model.factory(**kw)
        # Fresh spawns: the test spawns B first, then A.
        spawns.append(sid)
        if len(spawns) == 1:
            child = b_model.factory(**kw)
            ids["b_sess"] = child.session_id
            return child
        return a_model.factory(**kw)

    from tools.delegate_tool import delegate_task
    with patch("run_agent.AIAgent", side_effect=factory):
        b_out = json.loads(delegate_task(tasks=[{"goal": "B round one"}], background=True, parent_agent=root))
        ids["b"] = b_out["subagent_ids"][0]
        a_out = json.loads(delegate_task(tasks=[{"goal": "A round one"}], background=True, parent_agent=root))
        ids["a"] = a_out["subagent_ids"][0]
        wait_until(lambda: sent.get("b2"), timeout=40, what="B's third-hop attempt")

    assert sent["a1"]["state"] == "accepted", sent["a1"]
    assert sent["b1"]["state"] == "accepted" and sent["b1"]["delivery"] == "continuation", sent["b1"]
    assert sent["a2"]["state"] == "accepted" and sent["a2"]["delivery"] == "continuation", sent["a2"]
    assert sent["b2"]["state"] == "rejected" and sent["b2"]["reason"] == "rounds_exhausted", sent["b2"]
    assert len(a_model.builds) == 2 and len(b_model.builds) == 2
    assert get_message(_db_path(root), "msg-b-2") is None  # refused before any durable admission
    t = tree(_db_path(root), root.session_id)
    assert t["rounds_used"] == 2 and t["state"] == "active", t


def test_tree_concurrency_ceiling_rejects_the_excess(tmp_path, monkeypatch):
    from tools.agent_messaging_tool import agent_message
    from tools import delegate_tool_collab

    monkeypatch.setattr(delegate_tool_collab, "_limits", lambda: (12, 1))
    root = _parent("root-conc", db=SessionDB(db_path=tmp_path / "state.db"))
    gate_c, release_c = threading.Event(), threading.Event()
    gate_d, release_d = threading.Event(), threading.Event()

    def c_script(child, user_message):
        return "C first"

    def d_script(child, user_message):
        return "D first"

    c_sid, c_deleg, c_model = _spawn(root, "C", c_script)
    _wait_event(lambda e: e.get("delegation_id") == c_deleg, "C first completion")
    d_sid, d_deleg, d_model = _spawn(root, "D", d_script)
    _wait_event(lambda e: e.get("delegation_id") == d_deleg, "D first completion")

    c_model.after_persist = lambda child, msg: (gate_c.set(), release_c.wait(20))
    d_model.after_persist = lambda child, msg: (gate_d.set(), release_d.wait(20))

    def factory(**kw):
        sid = kw.get("session_id")
        if sid == c_model.builds[0]["child"].session_id:
            return c_model.factory(**kw)
        if sid == d_model.builds[0]["child"].session_id:
            return d_model.factory(**kw)
        raise AssertionError(f"unexpected build {sid!r}")

    def send(target, mid):
        return json.loads(agent_message(target=target, message=f"work {mid}", message_id=mid,
                                        parent_agent=root))

    from tools.delegate_tool import delegate_task
    try:
        with patch("run_agent.AIAgent", side_effect=factory):
            with ThreadPoolExecutor(max_workers=2) as pool:
                outs = [f.result(timeout=30) for f in (pool.submit(send, c_sid, "msg-c-1"),
                                                       pool.submit(send, d_sid, "msg-d-1"))]
            assert sum(o["state"] == "accepted" for o in outs) == 1, outs
            rejected = next(o for o in outs if o["state"] == "rejected")
            assert rejected["reason"] == "concurrency_exhausted", rejected
            wait_until(lambda: gate_c.is_set() or gate_d.is_set(), timeout=20,
                       what="one continuation running")
            # Capacity is checked at the durable store, not only in memory.
            assert tree_rounds_used(root) >= 1
    finally:
        release_c.set()
        release_d.set()

    wait_until(lambda: _is_resumable(root, c_sid) and _is_resumable(root, d_sid), timeout=30,
               what="both continuations settled")
    from tools.delegate_tool_collab import tree
    assert tree(_db_path(root), root.session_id)["rounds_used"] == 1


def tree_rounds_used(root):
    from tools.delegate_tool_collab import tree
    return tree(_db_path(root), root.session_id)["rounds_used"]


def test_root_interrupt_stops_the_tree_and_messages_cannot_resurrect_it(tmp_path):
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_collab import get_message, tree
    from tools.delegate_tool_messages import finish_collaboration_run

    root = _parent("root-stop", db=SessionDB(db_path=tmp_path / "state.db"))
    sender = _live_child(root, "sa-stop-a")
    target = _live_child(root, "sa-stop-b")
    first = json.loads(agent_message(target="sa-stop-b", message="queued work",
                                     message_id="msg-stop-1", parent_agent=sender))
    assert first["state"] == "accepted", first
    assert target._delegate_inbox

    root._interrupt_requested = True
    finish_collaboration_run(root, {"completed": False})
    assert tree(_db_path(root), root.session_id)["state"] == "stopped"
    row = get_message(_db_path(root), "msg-stop-1")
    assert row["state"] == "rejected" and row["reason"] == "tree_stopped", row
    assert target._delegate_inbox == []
    assert target._interrupt_requested is True

    blocked = json.loads(agent_message(target="sa-stop-b", message="after the stop",
                                       message_id="msg-stop-2", parent_agent=sender))
    assert blocked["state"] == "rejected" and blocked["reason"] == "tree_stopped", blocked
    assert get_message(_db_path(root), "msg-stop-2") is None

    # A new explicit spawn is the only reactivation path; it starts a fresh round budget.
    fresh_sid, fresh_deleg, _ = _spawn(root, "fresh work after the stop", lambda c, m: "fresh")
    _wait_event(lambda e: e.get("delegation_id") == fresh_deleg, "fresh spawn completion")
    assert tree(_db_path(root), root.session_id)["state"] == "active"
    assert tree(_db_path(root), root.session_id)["rounds_used"] == 0
    allowed = json.loads(agent_message(target=fresh_sid, message="new work",
                                       message_id="msg-stop-3", parent_agent=sender))
    assert allowed["state"] == "accepted", allowed
    assert get_message(_db_path(root), "msg-stop-1")["state"] == "rejected"


def test_idle_root_is_an_explicit_refusal_and_is_not_durably_accepted(tmp_path):
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_collab import get_message

    root = _parent("root-idle", db=SessionDB(db_path=tmp_path / "state.db"))
    root.status, root._current_turn_id = "idle", None
    sender = _live_child(root, "sa-idle-a")
    out = json.loads(agent_message(target="lead", message="wake up", message_id="msg-idle-1",
                                   parent_agent=sender))
    assert out["state"] == "rejected" and out["reason"] == "target_idle", out
    # No gateway owns this test process: the durable row records the refused attempt explicitly,
    # never an ``accepted`` message and never a queued continuation.
    row = get_message(_db_path(root), "msg-idle-1")
    assert row is not None and row["state"] == "rejected" and row["reason"] == "target_idle", row


def test_a_running_root_consumes_a_child_message_without_a_new_turn(tmp_path):
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_collab import get_message
    from tools.delegate_tool_messages import finish_collaboration_run

    root = _parent("root-busy", db=SessionDB(db_path=tmp_path / "state.db"))
    sender = _live_child(root, "sa-root-a")
    out = json.loads(agent_message(target="lead", message="root should see this",
                                   message_id="msg-root-1", parent_agent=sender))
    assert out["state"] == "accepted", out
    messages = [{"role": "tool", "tool_call_id": "t1", "content": "tool output"}]
    from tools.delegate_tool_messages import inject_inbox_messages
    assert inject_inbox_messages(root, messages) == 1
    assert messages[-1]["role"] == "user" and "root should see this" in messages[-1]["content"]
    finish_collaboration_run(root, {"completed": True})
    row = get_message(_db_path(root), "msg-root-1")
    assert row["state"] == "processed" and row["outcome"] == "completed", row


def test_concurrent_same_id_delivers_exactly_one_inbox_row(tmp_path):
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_collab import get_message

    root = _parent("root-race", db=SessionDB(db_path=tmp_path / "state.db"))
    sender = _live_child(root, "sa-race-a")
    target = _live_child(root, "sa-race-b")

    def send():
        return json.loads(agent_message(target="sa-race-b", message="same body",
                                        message_id="msg-race-dup", parent_agent=sender))

    with ThreadPoolExecutor(max_workers=2) as pool:
        outs = [f.result(timeout=20) for f in (pool.submit(send), pool.submit(send))]
    assert [o["state"] for o in outs] == ["accepted", "accepted"], outs
    assert len(target._delegate_inbox) == 1
    row = get_message(_db_path(root), "msg-race-dup")
    assert row["state"] == "accepted", row
    with sqlite3.connect(_db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM delegate_collab_inbox "
                            "WHERE message_id='msg-race-dup'").fetchone()[0] == 1


# ── stage 4: durable store, restart, recovery ──────────────────────────────

def test_duplicate_id_is_idempotent_across_an_in_process_restart(tmp_path):
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_collab import get_message
    from tools.delegate_tool_messages import reset_message_state

    db_path = tmp_path / "state.db"
    root = _parent("root-dup", db=SessionDB(db_path=db_path))
    _live_child(root, "sa-dup-b")

    first = json.loads(agent_message(target="sa-dup-b", message="same body", message_id="msg-dup-9",
                                     parent_agent=root))
    assert first["state"] == "accepted", first

    # Simulate a fresh process: no in-memory mirror or live registry, same state.db.
    reset_message_state()
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _recent_subagents
    with _active_subagents_lock:
        _active_subagents.clear()
        _recent_subagents.clear()
    restarted = _parent("root-dup", db=SessionDB(db_path=db_path))
    again = json.loads(agent_message(target="sa-dup-b", message="same body", message_id="msg-dup-9",
                                     parent_agent=restarted))
    assert again["state"] == "accepted", again
    conflict = json.loads(agent_message(target="sa-dup-b", message="different body", message_id="msg-dup-9",
                                        parent_agent=restarted))
    assert conflict["state"] == "rejected" and conflict["reason"] == "message_id_conflict", conflict
    row = get_message(str(db_path), "msg-dup-9")
    assert row["state"] == "accepted"
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM delegate_collab_inbox WHERE message_id='msg-dup-9'").fetchone()[0] == 1


def test_durable_only_member_can_be_continued_after_the_registry_is_gone(tmp_path):
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_collab import get_message
    from tools.delegate_tool_messages import reset_message_state

    db_path = tmp_path / "state.db"
    root = _parent("root-durable", db=SessionDB(db_path=db_path))
    c_sid, c_deleg, c_model = _spawn(root, "durable child", lambda child, msg: "C first answer")
    _wait_event(lambda e: e.get("delegation_id") == c_deleg, "C completion")
    c_sess = c_model.builds[0]["child"].session_id

    # Fresh process: registry + process ledger gone, only state.db remains.
    reset_message_state()
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _recent_subagents
    with _active_subagents_lock:
        _active_subagents.clear()
        _recent_subagents.clear()
    restarted = _parent("root-durable", db=SessionDB(db_path=db_path))

    with patch("run_agent.AIAgent", side_effect=c_model.factory):
        out = json.loads(agent_message(target=c_sid, message="continue from the durable row",
                                       message_id="msg-durable-1", parent_agent=restarted))
        assert out["state"] == "accepted", out
        wait_until(lambda: get_message(str(db_path), "msg-durable-1")["state"] == "processed",
                   timeout=30, what="durable continuation processed")
    assert c_model.builds[1]["session_id"] == c_sess
    assert c_model.builds[1]["child"]._subagent_id == c_sid
    stored = restarted._session_db.get_messages_as_conversation(c_sess)
    assert any("continue from the durable row" in str(m.get("content")) for m in stored)


def test_durable_member_denied_across_tree_and_profile(tmp_path):
    from tools.agent_messaging_tool import agent_message
    from tools.delegate_tool_messages import reset_message_state

    db_path = tmp_path / "state.db"
    root = _parent("root-scope", db=SessionDB(db_path=db_path))
    c_sid, c_deleg, _ = _spawn(root, "scoped child", lambda child, msg: "C")
    _wait_event(lambda e: e.get("delegation_id") == c_deleg, "C completion")

    reset_message_state()
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _recent_subagents
    with _active_subagents_lock:
        _active_subagents.clear()
        _recent_subagents.clear()
    restarted = _parent("root-scope", db=SessionDB(db_path=db_path))

    # Same root, different persisted profile: the row is not resolvable (fail closed,
    # so this cannot become a cross-profile existence oracle).
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE resumable_subagents SET record_json=json_set(record_json, '$.profile_key', ?) "
                     "WHERE subagent_id=?", ("other-profile", c_sid))
    wrong_profile = json.loads(agent_message(target=c_sid, message="x", message_id="msg-scope-1",
                                             parent_agent=restarted))
    assert wrong_profile["state"] == "rejected" and wrong_profile["reason"] == "unknown_target", wrong_profile

    # Different persisted tree: not even resolvable from this root.
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE resumable_subagents SET record_json=json_set(record_json, "
                     "'$.root_session_id', 'other-root') WHERE subagent_id=?", (c_sid,))
    wrong_tree = json.loads(agent_message(target=c_sid, message="x", message_id="msg-scope-2",
                                          parent_agent=restarted))
    assert wrong_tree["state"] == "rejected" and wrong_tree["reason"] == "unknown_target", wrong_tree


def test_durable_members_are_listed_after_restart(tmp_path):
    from tools.agent_messaging_tool import list_agents
    from tools.delegate_tool_messages import reset_message_state

    db_path = tmp_path / "state.db"
    root = _parent("root-list", db=SessionDB(db_path=db_path))
    c_sid, c_deleg, _ = _spawn(root, "listed child", lambda child, msg: "C")
    _wait_event(lambda e: e.get("delegation_id") == c_deleg, "C completion")
    reset_message_state()
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _recent_subagents
    with _active_subagents_lock:
        _active_subagents.clear()
        _recent_subagents.clear()
    restarted = _parent("root-list", db=SessionDB(db_path=db_path))
    rows = json.loads(list_agents(parent_agent=restarted))
    row = next(r for r in rows if r["id"] == c_sid)
    assert row["status"] == "durable" and row["session_id"]


_REC_PREP = r'''
import json, os, sys
from pathlib import Path
home = os.path.abspath(os.environ["HERMES_HOME"])
assert home == os.path.abspath(sys.argv[1]), (home, sys.argv[1])
dbp = os.path.join(home, "state.db")
assert os.path.dirname(os.path.abspath(dbp)) == home
from hermes_state import SessionDB
from tools.delegate_tool_collab import (
    ensure_tree, begin_dispatch_message, begin_inbox_message, mark_delivered, mark_processed, stop_tree,
)
db = SessionDB(db_path=Path(dbp))
db.create_session("root-r", "cli")
for sid in ("c-norow", "c-marker", "c-inbox", "c-deliv", "c-proc"):
    db.create_session(sid, "subagent", parent_session_id="root-r")
db.create_session("root-s", "cli")
db.create_session("c-stop", "subagent", parent_session_id="root-s")
db.append_message("c-marker", "user", "[agent message m-cont-marker]\nFrom agent sa-x: hi")
ensure_tree(dbp, "root-r", "profile-x", max_rounds=5, max_concurrent=10)
ensure_tree(dbp, "root-s", "profile-x", max_rounds=5, max_concurrent=10)
for mid, target in (("m-cont-norow", "c-norow"), ("m-cont-marker", "c-marker"),
                    ("m-deliv", "c-deliv"), ("m-proc", "c-proc")):
    row, why = begin_dispatch_message(dbp, "root-r", mid, sender_id="lead", sender_session_id="root-r",
                                     recipient_id=target, recipient_session_id=target,
                                     target_input=target, fingerprint="fp-" + mid, round_no=1)
    assert row is not None, (mid, why)
mark_delivered(dbp, "m-cont-marker", target_session_id="c-marker")
mark_delivered(dbp, "m-deliv", target_session_id="c-deliv")
mark_delivered(dbp, "m-proc", target_session_id="c-proc")
mark_processed(dbp, "m-proc", "completed")
begin_inbox_message(dbp, "root-r", "m-inbox", sender_id="lead", sender_session_id="root-r",
                    recipient_id="c-inbox", recipient_session_id="c-inbox",
                    target_input="c-inbox", fingerprint="fp-inbox", round_no=1)
row, why = begin_dispatch_message(dbp, "root-s", "m-stop", sender_id="lead", sender_session_id="root-s",
                                 recipient_id="c-stop", recipient_session_id="c-stop",
                                 target_input="c-stop", fingerprint="fp-stop", round_no=1)
assert row is not None, why
stop_tree(dbp, "root-s", "root_stopped")
print(json.dumps({"pid": os.getpid(), "db": dbp}))
'''

_REC_RECOVER = r'''
import json, os, sys
home = os.path.abspath(os.environ["HERMES_HOME"])
assert home == os.path.abspath(sys.argv[1]), (home, sys.argv[1])
dbp = os.path.join(home, "state.db")
assert os.path.dirname(os.path.abspath(dbp)) == home
from tools.delegate_tool_collab import recover_tree, get_message
result = recover_tree(dbp)
ids = ("m-cont-norow", "m-cont-marker", "m-deliv", "m-proc", "m-inbox", "m-stop")
print(json.dumps({"recovery": result, "rows": {i: get_message(dbp, i) for i in ids}}))
'''


def _run_probe(script, home):
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = os.pathsep.join([str(_ROOT), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    proc = subprocess.run([sys.executable, "-c", script, str(home)], env=env, cwd=str(_ROOT),
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_fresh_process_recovery_is_conservative_and_never_redispatches(tmp_path):
    home = tmp_path / "recovery-home"
    home.mkdir()
    prep = _run_probe(_REC_PREP, home)
    assert prep["pid"] != os.getpid()
    out = _run_probe(_REC_RECOVER, home)

    assert out["recovery"]["dispatched"] == 0, out["recovery"]
    rows = out["rows"]
    assert rows["m-cont-norow"]["state"] == "needs_review"
    assert rows["m-cont-norow"]["reason"] == "recovery_unconfirmed_continuation"
    assert rows["m-cont-marker"]["state"] == "needs_review"
    assert rows["m-cont-marker"]["reason"] == "recovery_run_outcome_unknown"
    assert rows["m-cont-marker"]["delivered_at"] is not None
    assert rows["m-deliv"]["state"] == "needs_review"
    assert rows["m-deliv"]["reason"] == "recovery_run_outcome_unknown"
    assert rows["m-proc"]["state"] == "processed"
    assert rows["m-inbox"]["state"] == "rejected"
    assert rows["m-inbox"]["reason"] == "recovery_undelivered_no_effect"
    assert rows["m-stop"]["state"] == "rejected"
    assert rows["m-stop"]["reason"] == "tree_stopped"


def test_recovery_marks_an_undelivered_inbox_as_no_effect_not_a_retry(tmp_path):
    from tools.delegate_tool_collab import (
        begin_inbox_message, ensure_tree, get_message, recover_tree,
    )

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("root-u", "cli")
    db.create_session("c-u", "subagent", parent_session_id="root-u")
    ensure_tree(db_path, "root-u", "profile-u")
    begin_inbox_message(db_path, "root-u", "m-u", sender_id="lead", sender_session_id="root-u",
                        recipient_id="c-u", recipient_session_id="c-u", target_input="c-u",
                        fingerprint="fp-u", round_no=1)
    # Mark the owner dead so recovery is allowed to reconcile this row.
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE delegate_collab_messages SET owner_pid=999999, owner_started_at=0 "
                     "WHERE message_id='m-u'")
    result = recover_tree(db_path, "root-u")
    assert result["dispatched"] == 0
    row = get_message(db_path, "m-u")
    assert row["state"] == "rejected" and row["reason"] == "recovery_undelivered_no_effect", row


def test_schema_migration_adds_tables_without_touching_existing_rows(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("legacy", "cli")
    db.append_message("legacy", "user", "untouched-user-row")
    db.append_message("legacy", "assistant", "untouched-assistant-row")
    db.close()

    with sqlite3.connect(db_path) as conn:
        for table in ("delegate_collab_trees", "delegate_collab_messages", "delegate_collab_inbox"):
            conn.execute(f"DROP TABLE {table}")
        conn.commit()

    reopened = SessionDB(db_path=db_path)
    try:
        with sqlite3.connect(db_path) as conn:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'delegate_collab_%'")}
        assert names == {"delegate_collab_trees", "delegate_collab_messages", "delegate_collab_inbox"}
        rows = reopened.get_messages_as_conversation("legacy")
        assert [r["content"] for r in rows if r.get("role") == "user"] == ["untouched-user-row"]
        assert any(r.get("content") == "untouched-assistant-row" for r in rows)
    finally:
        reopened.close()


def test_terminal_cap_prunes_only_final_states(tmp_path):
    from tools.delegate_tool_collab import (
        begin_dispatch_message, ensure_tree, get_message, prune_messages,
    )

    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).create_session("root-cap", "cli")
    ensure_tree(db_path, "root-cap", "profile-cap")
    for i in range(6):
        mid = f"m-done-{i}"
        begin_dispatch_message(db_path, "root-cap", mid, sender_id="lead", sender_session_id="root-cap",
                               recipient_id=f"c-{i}", recipient_session_id=f"c-{i}",
                               target_input=f"c-{i}", fingerprint="fp" + mid, round_no=1)
        from tools.delegate_tool_collab import mark_delivered, mark_processed
        mark_delivered(db_path, mid, target_session_id=f"c-{i}")
        mark_processed(db_path, mid, "completed")
    begin_dispatch_message(db_path, "root-cap", "m-live", sender_id="lead", sender_session_id="root-cap",
                           recipient_id="c-live", recipient_session_id="c-live",
                           target_input="c-live", fingerprint="fplive", round_no=1)
    from tools.delegate_tool_collab import mark_needs_review
    begin_dispatch_message(db_path, "root-cap", "m-review", sender_id="lead", sender_session_id="root-cap",
                           recipient_id="c-r", recipient_session_id="c-r",
                           target_input="c-r", fingerprint="fpreview", round_no=1)
    mark_needs_review(db_path, "m-review", "ambiguous")
    pruned = prune_messages(db_path, "root-cap", cap=3)
    assert pruned >= 4
    assert get_message(db_path, "m-live")["state"] == "accepted"
    assert get_message(db_path, "m-review")["state"] == "needs_review"
    with sqlite3.connect(db_path) as conn:
        terminal = conn.execute("SELECT COUNT(*) FROM delegate_collab_messages "
                                "WHERE tree_id='root-cap' AND state IN ('processed','rejected')").fetchone()[0]
    assert terminal <= 3
