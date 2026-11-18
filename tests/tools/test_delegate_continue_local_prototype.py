"""Local acceptance for adopted upstream continuation (PR #154163) on the v0.23.1 base.

The scripted model is the only fake: registry, child build, SessionDB, dispatch, durable
records and completion delivery are the real ones. Round one deliberately writes a tool
result into the child transcript so a "new child + summary" implementation cannot pass.

These tests are the local prototype's contract; upstream's own tests (test_delegate_continue.py,
test_delegate_resumable*.py) remain the broader oracle.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from tests.waiting import wait_until
from tools.process_registry import process_registry

TOOL_SENTINEL = "TOOL_SENTINEL_9f3_local"


def _parent(session_id="parent-sess", db=None):
    parent = MagicMock()
    parent.base_url, parent.api_key, parent.provider = "https://openrouter.ai/api/v1", "parent-key", "openrouter"
    parent.api_mode, parent.model, parent.platform = "chat_completions", "parent-model", "cli"
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent.session_id, parent._delegate_depth, parent._interrupt_requested = session_id, 0, False
    parent.enabled_toolsets, parent.disabled_toolsets = ["terminal", "file", "web"], []
    parent._session_db = db if db is not None else SessionDB()
    parent._session_db.create_session(session_id, "cli")
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


def _spawn(parent, goal):
    from tools.delegate_tool import delegate_task
    out = json.loads(delegate_task(tasks=[{"goal": goal, "context": "repo at /src"}], background=True,
                                   parent_agent=parent))
    assert out["status"] == "dispatched", out
    _completion(out["delegation_id"])
    return out["subagent_ids"][0], out["delegation_id"]


@pytest.fixture(autouse=True)
def _empty_completion_queue():
    def drain():
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()
    drain()
    yield
    drain()


class _ToolWritingModel:
    """AIAgent stand-in that persists round one WITH a tool result, then records every turn."""

    def __init__(self):
        self.builds, self.turns = [], []

    def __call__(self, **kw):
        child = MagicMock()
        child.session_id = kw.get("session_id") or f"child-sess-{len(self.builds)}"
        child.model, child.provider = kw.get("model"), kw.get("provider")
        child._parent_session_id = kw.get("parent_session_id")
        child.get_activity_summary.return_value = {"api_call_count": 1, "current_tool": None, "last_activity_ts": 0}
        db = kw["session_db"]

        def run_conversation(user_message, conversation_history=None, **_):
            self.turns.append({"session_id": child.session_id, "message": user_message,
                               "history": conversation_history})
            db.create_session(child.session_id, "subagent", parent_session_id=kw.get("parent_session_id"))
            db.append_message(child.session_id, "user", user_message)
            db.append_message(child.session_id, "assistant", "I will run the parser",
                              tool_calls=[{"id": "call_1", "type": "function",
                                           "function": {"name": "terminal", "arguments": "{}"}}])
            db.append_message(child.session_id, "tool", TOOL_SENTINEL, tool_name="terminal", tool_call_id="call_1")
            reply = f"done: {user_message}"
            db.append_message(child.session_id, "assistant", reply)
            return {"final_response": reply, "completed": True, "api_calls": 1}

        child.run_conversation.side_effect = run_conversation
        self.builds.append(dict(kw, home=str(__import__("hermes_constants").get_hermes_home())))
        return child


def _forget_process_state():
    from tools.delegate_tool_registry import _active_subagents_lock, _recent_subagents
    with _active_subagents_lock:
        _recent_subagents.clear()


def _continue(parent, sid, message="round two", background=True):
    from tools.delegate_tool import delegate_task
    return json.loads(delegate_task(action="continue", subagent_id=sid, message=message,
                                    background=background, parent_agent=parent))


def test_continue_keeps_child_identity_and_the_real_tool_history():
    model, parent = _ToolWritingModel(), _parent()
    with patch("run_agent.AIAgent", side_effect=model):
        sid, first_delegation = _spawn(parent, "write the parser")
        out = _continue(parent, sid, "now add tests for it")
        assert out.get("status") == "dispatched", out
        assert out["subagent_ids"] == [sid] and out["delegation_id"] != first_delegation
        _completion(out["delegation_id"])

    first, again = model.builds
    assert again["session_id"] == model.turns[0]["session_id"]  # same conversation
    assert again["model"] == first["model"] and again["provider"] == first["provider"]
    history = model.turns[1]["history"]
    blob = json.dumps(history, default=str)
    assert TOOL_SENTINEL in blob, blob  # actual tool result, not a summary
    assert {"role": "tool", "content": TOOL_SENTINEL}.items() <= next(
        m for m in history if m.get("role") == "tool").items()
    assert history[0]["content"] == "write the parser"
    assert model.turns[1]["message"] == "now add tests for it"
    stored = parent._session_db.get_messages_as_conversation(model.turns[1]["session_id"])
    tail = [m["content"] for m in stored][-2:]
    assert [m["content"] for m in stored][-4:] == [
        "now add tests for it", "I will run the parser", TOOL_SENTINEL, "done: now add tests for it"]
    assert tail[-1] == "done: now add tests for it"


def test_two_simultaneous_continues_dispatch_exactly_one(monkeypatch):
    """Two public-path continues meet between the in-process claim and the durable claim."""
    try:
        import tools.delegate_tool_continue as cont
    except ImportError:  # base v0.23.1: no continuation module yet; both calls fail observably below
        cont = None
    from tools.delegate_tool import delegate_task

    model, parent = _ToolWritingModel(), _parent()
    with patch("run_agent.AIAgent", side_effect=model):
        sid, _ = _spawn(parent, "write the parser")

    if cont is not None:
        barrier = threading.Barrier(2, timeout=20)
        real_claim = cont._claim_resume

        def racing_claim(subagent_id, caller):
            got = real_claim(subagent_id, caller)
            barrier.wait()
            return got

        monkeypatch.setattr(cont, "_claim_resume", racing_claim)
    with patch("run_agent.AIAgent", side_effect=model):
        def call():
            return json.loads(delegate_task(action="continue", subagent_id=sid, message="race",
                                            background=True, parent_agent=parent))
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [f.result(timeout=40) for f in (pool.submit(call), pool.submit(call))]
        winners = [r for r in results if r.get("status") == "dispatched"]
        assert len(winners) == 1, results
        losers = [r for r in results if "error" in r]
        assert len(losers) == 1 and ("No finished subagent" in losers[0]["error"]
                                     or "being continued" in losers[0]["error"]), losers
        _completion(winners[0]["delegation_id"])
    # spawn + one continuation only: the loser never built a child
    assert len(model.builds) == 2


def test_a_cancelled_child_does_not_replay_on_its_own_but_an_explicit_continue_keeps_the_session():
    """Cancellation = stop the current turn, no automatic replay; a later explicit continue is a new
    user decision and still lands on the same child session (the durable row is ``stopped``, unpinned)."""
    from tools.delegate_tool import delegate_task
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, interrupt_subagent

    started, release = threading.Event(), threading.Event()
    model = _ToolWritingModel()

    def build(**kw):
        child = model(**kw)
        ok = child.run_conversation.side_effect

        def run_conversation(user_message, **rest):
            started.set()
            release.wait(10)
            ok(user_message, **rest)
            return {"final_response": "Operation interrupted.", "interrupted": True, "completed": False}
        child.run_conversation.side_effect = run_conversation
        return child

    parent = _parent()
    with patch("run_agent.AIAgent", side_effect=build):
        out = json.loads(delegate_task(tasks=[{"goal": "long job"}], background=True, parent_agent=parent))
        sid = out["subagent_ids"][0]
        assert started.wait(10)
        with _active_subagents_lock:
            assert sid in _active_subagents
        interrupt_subagent(sid)
        release.set()
        _completion(out["delegation_id"])

        import sqlite3
        conn = sqlite3.connect(parent._session_db.db_path)
        try:
            row = conn.execute("SELECT state, pinned FROM resumable_subagents WHERE subagent_id=?",
                               (sid,)).fetchone()
        except sqlite3.OperationalError:  # base has no durable table; continue below is the observable red
            row = None
        finally:
            conn.close()
        if row is not None:
            assert row == ("stopped", 0), row
        assert len(model.builds) == 1  # nothing auto-replayed

        resumed = _continue(parent, sid, "explicit restart")
        assert resumed.get("status") == "dispatched", resumed
        _completion(resumed["delegation_id"])

    assert len(model.builds) == 2
    assert model.turns[1]["session_id"] == model.turns[0]["session_id"]
    assert TOOL_SENTINEL in json.dumps(model.turns[1]["history"], default=str)


def test_non_owner_cannot_continue_but_the_owner_can():
    model, root = _ToolWritingModel(), _parent("root-sess")
    with patch("run_agent.AIAgent", side_effect=model):
        sid, _ = _spawn(root, "write the parser")
        foreign = _parent("unrelated-sess", db=root._session_db)
        refused = _continue(foreign, sid, "foreign work")
        assert "not yours" in refused["error"], refused
        assert len(model.builds) == 1
        own = _continue(root, sid, "owner follow-up")
        assert own.get("status") == "dispatched", own
        _completion(own["delegation_id"])
    assert model.turns[1]["session_id"] == model.turns[0]["session_id"]


def test_a_managed_worker_session_refuses_and_keeps_the_finished_child(monkeypatch):
    """The Telegram managed-worker path is fail-closed: no claim is consumed."""
    import agent.runtime_session_store as store
    from tools.delegate_tool_registry import _recent_subagents

    model, parent = _ToolWritingModel(), _parent()
    with patch("run_agent.AIAgent", side_effect=model):
        sid, _ = _spawn(parent, "write the parser")

    monkeypatch.setattr(store, "_worker_process", True)  # what a managed worker registers at startup
    out = _continue(parent, sid)
    assert "managed worker" in out.get("error", ""), out
    assert len(model.builds) == 1
    assert "resume" in _recent_subagents[sid]  # refused before the claim


def test_restart_recovery_continues_the_same_saved_session():
    """After the in-process registry is gone, the durable row rebuilds the same child."""
    model, parent = _ToolWritingModel(), _parent()
    with patch("run_agent.AIAgent", side_effect=model):
        sid, _ = _spawn(parent, "write the parser")
        _forget_process_state()  # a fresh process has no _recent_subagents entry
        out = _continue(parent, sid, "after restart")
        assert out.get("status") == "dispatched", out
        _completion(out["delegation_id"])

    assert out["subagent_ids"] == [sid]
    assert model.turns[1]["session_id"] == model.turns[0]["session_id"]
    blob = json.dumps(model.turns[1]["history"], default=str)
    assert TOOL_SENTINEL in blob
    assert model.turns[1]["message"] == "after restart"


def test_gateway_wrapped_session_db_parent_can_continue(tmp_path):
    """The Telegram/gateway parent holds an AsyncSessionDB wrapper, not a RuntimeSessionStore:
    continuation must read its db_path (not call its coroutine methods) and stay allowed."""
    from hermes_state import SessionDB

    class _GatewayDB:  # the shape gateway/session_execution.py hands an in-process agent
        def __init__(self, db):
            self._db, self.db_path = db, db.db_path

        def __getattr__(self, name):
            return getattr(self._db, name)

    model = _ToolWritingModel()
    root = _parent()
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent-sess", "telegram")
    root._session_db = _GatewayDB(db)
    with patch("run_agent.AIAgent", side_effect=model):
        sid, _ = _spawn(root, "write the parser")
        out = _continue(root, sid, "gateway follow-up")
        assert out.get("status") == "dispatched", out
        _completion(out["delegation_id"])
    assert model.turns[1]["session_id"] == model.turns[0]["session_id"]
    stored = db.get_messages_as_conversation(model.turns[1]["session_id"])
    assert [m["content"] for m in stored][-1] == "done: gateway follow-up"
    db.close()


def test_gateway_policy_only_marks_telegram_turns_managed_when_the_session_is_a_bypass(monkeypatch):
    """Preservation control: ordinary Telegram turns are in-process (continue allowed); safe/config-only
    sessions are managed workers, where the guard above refuses continuation."""
    from gateway import session_managed_worker as mw

    class _Policy:
        def __init__(self, ignore_user_config=False):
            self.ignore_user_config = ignore_user_config
            self.kanban_json = None
            self.platform = "telegram"
            self.request_json = "{}"

        def config(self):
            return {}

    ref = SimpleNamespace(session_id="sess-1")
    authority = SimpleNamespace(runner=object(), sessions={"sess-1": SimpleNamespace(source=object())})

    normal = _Policy(False)
    monkeypatch.setattr("gateway.session_policy.policy_for_source", lambda runner, source: normal)
    assert mw.managed_policy(authority, ref) is None

    bypass = _Policy(True)
    monkeypatch.setattr("gateway.session_policy.policy_for_source", lambda runner, source: bypass)
    assert mw.managed_policy(authority, ref) is bypass
