"""Shutdown vs explicit-stop gate at the native turn boundary (stage 3/4 review).

The production order is ``run_conversation`` -> ``finalize_turn`` (records the exit reason,
then ``clear_interrupt``) -> ``finish_collaboration_run``. A gate that reads only
``_interrupt_requested`` therefore never sees a stopped root, while a gate that treats every
interrupt as a user stop falsely closes a tree on a graceful, resumable gateway shutdown.

These tests drive the REAL interrupt API on a real ``AIAgent`` (built without ``__init__``,
as the native interrupt tests do), the real ``SessionDB`` schema, the real message store and
the real ``finish_collaboration_run`` hook. Only the model run and child agents are controlled
fixtures, exactly like the stage-3/4 acceptance file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from tools.process_registry import process_registry

_ROOT = Path(__file__).resolve().parents[2]


# ── harness ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_state(tmp_path, monkeypatch):
    import hermes_state
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _recent_subagents
    from tools.delegate_tool_messages import reset_message_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    assert str(hermes_state.DEFAULT_DB_PATH) == str(tmp_path / "state.db")
    from tools.delegate_tool_collab import reset_collab_state
    reset_collab_state()
    with _active_subagents_lock:
        _active_subagents.clear()
        _recent_subagents.clear()
    reset_message_state()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    with _active_subagents_lock:
        _active_subagents.clear()
        _recent_subagents.clear()
    reset_message_state()
    reset_collab_state()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _native_root(session_id: str, db: SessionDB):
    """A real AIAgent with the real interrupt mixin, no ``__init__`` side effects."""
    from run_agent import AIAgent

    root = AIAgent.__new__(AIAgent)
    root.session_id = session_id
    root._session_db = db
    root._delegate_profile_key = str(db.db_path)
    root._subagent_id = None
    root._delegate_parent_ref = None
    root._delegate_root_session_id = None
    root._interrupt_requested = False
    root._interrupt_message = None
    root._tool_interrupt_reason = None
    root._hard_interrupt_requested = threading.Event()
    root._execution_thread_id = None
    root._interrupt_thread_signal_pending = False
    root._active_children = []
    root._active_children_lock = threading.Lock()
    root._tool_worker_threads = None
    root._tool_worker_threads_lock = threading.Lock()
    root.quiet_mode = True
    root.status = "running"
    root._current_turn_id = "turn-native"
    root._delegate_inbox = []
    root._delegate_inbox_lock = threading.Lock()
    db.create_session(session_id, "cli")
    return root


def _live_child(root, sid: str):
    from tools.delegate_tool_registry import _register_subagent

    agent = SimpleNamespace(
        session_id=f"{sid}-sess", _subagent_id=sid,
        _delegate_root_session_id=root.session_id,
        _delegate_profile_key=str(root._session_db.db_path),
        status="running", _interrupt_requested=False,
        _delegate_inbox=[], _delegate_inbox_lock=threading.Lock(),
    )
    agent._delegate_parent_ref = lambda: root
    _register_subagent({
        "subagent_id": sid, "agent": agent, "goal": "g", "status": "running",
        "owner_agent_session_id": root.session_id, "root_session_id": root.session_id,
        "profile_key": agent._delegate_profile_key, "accepting_messages": True,
    })
    return agent


def _db_path(root) -> str:
    return str(root._session_db.db_path)


def _interrupt_reason(agent) -> str:
    from agent.interrupt_control import interrupted_during_api_call_reason
    return interrupted_during_api_call_reason(agent)


def _finish_after_finalizer(root, reason: str) -> dict:
    """The real envelope contract: finalize_turn clears the bit before the hook runs."""
    root.clear_interrupt()
    assert root._interrupt_requested is False
    from tools.delegate_tool_messages import finish_collaboration_run
    result = {"completed": False, "interrupted": True, "turn_exit_reason": reason}
    finish_collaboration_run(root, result)
    return result


def _send(root, sender, target: str, message_id: str, body: str = "queued") -> dict:
    from tools.agent_messaging_tool import agent_message
    return json.loads(agent_message(target=target, message=body, message_id=message_id,
                                    parent_agent=sender))


# ── gate: explicit stop ────────────────────────────────────────────────────

def test_explicit_user_stop_still_stops_the_tree_after_the_finalizer_clears_the_flag(tmp_path):
    """Reading only ``_interrupt_requested`` at the hook never stops a user-stopped tree."""
    from agent.interrupt_compat import request_hard_interrupt
    from tools.delegate_tool_collab import get_message, tree

    root = _native_root("root-finalizer-stop", SessionDB(db_path=tmp_path / "state.db"))
    sender = _live_child(root, "sa-fin-a")
    target = _live_child(root, "sa-fin-b")
    sent = _send(root, sender, "sa-fin-b", "msg-fin-1")
    assert sent["state"] == "accepted", sent

    assert request_hard_interrupt(root, "Stop requested") is True
    reason = _interrupt_reason(root)
    assert reason == "interrupted_during_api_call", reason
    _finish_after_finalizer(root, reason)

    assert tree(_db_path(root), root.session_id)["state"] == "stopped"
    row = get_message(_db_path(root), "msg-fin-1")
    assert row["state"] == "rejected" and row["reason"] == "tree_stopped", row
    assert target._interrupt_requested is True
    assert target._delegate_inbox == []


# ── gate: graceful resumable shutdown ──────────────────────────────────────

def test_gateway_shutdown_keeps_the_tree_recoverable_when_the_flag_is_still_set(tmp_path):
    """A shutdown is not a user stop: no durable tree stop, no child teardown."""
    from agent.interrupt_compat import request_hard_interrupt
    from tools.delegate_tool_collab import get_message, queued_inbox, tree
    from tools.delegate_tool_messages import finish_collaboration_run

    root = _native_root("root-shutdown-keep", SessionDB(db_path=tmp_path / "state.db"))
    sender = _live_child(root, "sa-shut-a")
    target = _live_child(root, "sa-shut-b")
    assert _send(root, sender, "sa-shut-b", "msg-shut-1", "survive the restart")["state"] == "accepted"

    assert request_hard_interrupt(root, "Gateway shutting down",
                                  tool_reason="gateway shutdown") is True
    reason = _interrupt_reason(root)
    assert reason == "interrupted_during_api_call(gateway_shutdown)", reason
    assert root._interrupt_requested is True  # pre-clear envelope shape
    finish_collaboration_run(root, {"completed": False, "interrupted": True,
                                    "turn_exit_reason": reason})

    assert tree(_db_path(root), root.session_id)["state"] == "active"
    row = get_message(_db_path(root), "msg-shut-1")
    assert row["state"] == "accepted" and row["reason"] is None, row
    assert queued_inbox(_db_path(root), root.session_id, "sa-shut-b")
    assert target._interrupt_requested is False
    assert target._delegate_inbox


def test_an_explicit_stop_is_not_masked_by_a_concurrent_shutdown_in_either_order(tmp_path):
    """Native attribution: shutdown never relabels a pending user Stop, and a later user
    Stop wins. Either order is a user stop and closes the tree."""
    from agent.interrupt_compat import request_hard_interrupt
    from tools.delegate_tool_collab import tree

    for order in ("user-first", "shutdown-first"):
        root = _native_root(f"root-order-{order}", SessionDB(db_path=tmp_path / f"{order}.db"))
        sender = _live_child(root, f"sa-{order}")
        _live_child(root, f"sa-{order}-b")
        assert _send(root, sender, f"sa-{order}-b", f"msg-{order}")["state"] == "accepted"
        if order == "user-first":
            request_hard_interrupt(root, "Stop requested")
            request_hard_interrupt(root, "Gateway shutting down", tool_reason="gateway shutdown")
        else:
            request_hard_interrupt(root, "Gateway shutting down", tool_reason="gateway shutdown")
            request_hard_interrupt(root, "Stop requested")
        reason = _interrupt_reason(root)
        assert reason == "interrupted_during_api_call", (order, reason)
        _finish_after_finalizer(root, reason)
        assert tree(_db_path(root), root.session_id)["state"] == "stopped", order


def test_a_stopped_root_cannot_consume_a_message_queued_before_the_stop(tmp_path):
    """The root's own inbox is closed at stop; a later step boundary must not resurrect a
    message the durable tree already rejected."""
    from agent.interrupt_compat import request_hard_interrupt
    from tools.delegate_tool_collab import get_message
    from tools.delegate_tool_messages import inject_inbox_messages

    root = _native_root("root-own-inbox", SessionDB(db_path=tmp_path / "state.db"))
    sender = _live_child(root, "sa-own-inbox")
    assert _send(root, sender, "lead", "msg-own-1", "do more work")["state"] == "accepted"
    assert root._delegate_inbox

    request_hard_interrupt(root, "Stop requested")
    _finish_after_finalizer(root, _interrupt_reason(root))

    row = get_message(_db_path(root), "msg-own-1")
    assert row["state"] == "rejected" and row["reason"] == "tree_stopped", row
    assert root._delegate_inbox == []
    messages = [{"role": "tool", "tool_call_id": "t1", "content": "tool output"}]
    assert inject_inbox_messages(root, messages) == 0
    assert len(messages) == 1
    assert get_message(_db_path(root), "msg-own-1")["state"] == "rejected"


def test_processed_outcome_records_the_real_interrupted_envelope(tmp_path):
    """``processed`` records the consuming run's outcome; the cleared flag must not rewrite
    an interrupted run as ``incomplete``."""
    from agent.interrupt_compat import request_hard_interrupt
    from tools.delegate_tool_collab import get_message
    from tools.delegate_tool_messages import inject_inbox_messages

    root = _native_root("root-outcome", SessionDB(db_path=tmp_path / "state.db"))
    sender = _live_child(root, "sa-outcome")
    assert _send(root, sender, "lead", "msg-outcome-1", "consume me")["state"] == "accepted"
    messages = [{"role": "tool", "tool_call_id": "t1", "content": "tool output"}]
    assert inject_inbox_messages(root, messages) == 1
    root._session_db.append_message(root.session_id, "user", messages[1]["content"])

    request_hard_interrupt(root, "Stop requested")
    _finish_after_finalizer(root, _interrupt_reason(root))

    row = get_message(_db_path(root), "msg-outcome-1")
    assert row["state"] == "processed" and row["outcome"] == "interrupted", row


def test_a_message_rejected_by_a_concurrent_stop_is_never_dispatched(tmp_path):
    """A cold continuation must re-check its own durable admission after the reserve
    transaction: a stop that rejected the row in between must not start a child run."""
    import time as _time
    from unittest.mock import patch

    from tools import delegate_tool_collab as collab
    from tools.delegate_tool_messages import send_agent_message

    db_path = tmp_path / "state.db"
    root = _native_root("root-dispatch-race", SessionDB(db_path=db_path))
    record = {
        "goal": "g", "context": None, "toolsets": None, "endpoint": [], "model": "m",
        "session_id": "c-race-sess", "route_request": None,
        "root_session_id": root.session_id, "profile_key": str(db_path),
    }
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO resumable_subagents (subagent_id, state, pinned, stop_cause, owner_pid,"
            " owner_started_at, record_json, created_at, updated_at, expires_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("c-race", "resumable", 0, None, None, None, json.dumps(record),
             _time.time(), _time.time(), None))
        conn.commit()

    real_begin = collab.begin_dispatch_message
    calls = []

    def begin_then_stop(*args, **kwargs):
        row, why = real_begin(*args, **kwargs)
        if row is not None:
            collab.stop_tree(str(db_path), root.session_id, "root_stopped")
        return row, why

    def fake_continue(*args, **kwargs):
        calls.append((args, kwargs))
        return json.dumps({"status": "dispatched"})

    with patch.object(collab, "begin_dispatch_message", side_effect=begin_then_stop),             patch("tools.delegate_tool_continue.continue_subagent", side_effect=fake_continue):
        out = send_agent_message(root, "c-race", "should not run", message_id="msg-dispatch-race")

    assert calls == [], calls
    assert out["state"] == "rejected" and out["reason"] == "tree_stopped", out
    row = collab.get_message(str(db_path), "msg-dispatch-race")
    assert row["state"] == "rejected" and row["reason"] == "tree_stopped", row
