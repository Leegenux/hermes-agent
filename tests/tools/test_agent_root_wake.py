"""Root-wake acceptance: an idle root is woken through the native gateway route admission.

The only fakes are the root/child agent run functions and the scripted root turn. The gateway
authority, route store, runtime admission ledger, collaboration store, stop admission and turn
boundaries are production code. No real model/API/network call.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from tests.waiting import async_wait_until, wait_until
from tools.process_registry import process_registry


def _drain_completions():
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _recent_subagents
    from tools.delegate_tool_messages import reset_message_state
    try:
        from tools.delegate_tool_collab import reset_collab_state
        reset_collab_state()
    except ImportError:
        pass
    _drain_completions()
    with _active_subagents_lock:
        _active_subagents.clear()
        _recent_subagents.clear()
    reset_message_state()
    yield
    _drain_completions()
    with _active_subagents_lock:
        _active_subagents.clear()
        _recent_subagents.clear()
    reset_message_state()
    try:
        from tools.delegate_tool_collab import reset_collab_state
        reset_collab_state()
    except ImportError:
        pass


async def _gateway(tmp_path, monkeypatch, *, schedule: bool = False):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from hermes_state import AsyncSessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_constants import get_hermes_home
    assert Path(get_hermes_home()).resolve() == tmp_path.resolve()
    if not schedule:
        monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    db = SessionDB(tmp_path / "state.db")
    assert Path(db.db_path).resolve() == (tmp_path / "state.db").resolve()
    runner.session_store._db = db
    runner._session_db = AsyncSessionDB(db)
    authority = await initialize_session_authority(
        runner, profile_id="default", instance_id="rootwake", db=db)
    actor = SessionActor(
        "test-owner", "default",
        frozenset({"session:create", "session:submit", "session:read", "session:control"}),
        "fixture")
    ref = create_local_session(
        authority, actor, dict(request_id="rootwake", source="cli", model="fixture", toolsets=[]))
    return runner, authority, ref, actor, db


def _root(ref, db, *, status="idle"):
    root = MagicMock()
    root.session_id = ref.session_id
    root._session_db = db
    root._delegate_profile_key = str(db.db_path)
    root._delegate_root_session_id = None
    root._subagent_id = None
    root._delegate_parent_ref = None
    root.status = status
    root._current_turn_id = None if status == "idle" else "turn-root"
    root._interrupt_requested = False
    root._delegate_inbox = []
    root._delegate_inbox_lock = threading.Lock()
    return root


def _child(root, sid="sa-wake-a", *, profile_key=None):
    from tools.delegate_tool_registry import _register_subagent
    child = SimpleNamespace(
        session_id=f"{sid}-sess", _subagent_id=sid,
        _delegate_root_session_id=root.session_id,
        _delegate_profile_key=profile_key or str(root._session_db.db_path),
        status="running", _interrupt_requested=False,
        _delegate_inbox=[], _delegate_inbox_lock=threading.Lock(),
    )
    child._delegate_parent_ref = lambda: root
    _register_subagent({
        "subagent_id": sid, "agent": child, "goal": "g", "status": "running",
        "owner_agent_session_id": root.session_id, "root_session_id": root.session_id,
        "profile_key": child._delegate_profile_key, "accepting_messages": True,
    })
    return child


def _send(child, mid, text="hello root"):
    return _send_to(child, "lead", mid, text)


def _send_to(child, target, mid, text):
    from tools.agent_messaging_tool import agent_message
    return json.loads(agent_message(target=target, message=text, message_id=mid,
                                    parent_agent=child))


def _send_thread(child, mid, text="hello root"):
    """Deprecated helper kept out of the tests: blocking a caller must run off the loop."""
    raise RuntimeError("use await asyncio.to_thread(_send, ...)")


def _root_wake_rows(authority, ref):
    from hermes_state_runtime import list_session_admissions
    rows = list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
    return [r for r in rows if str(r.get("request_id") or "").startswith('["agent-collab"')]


def _run_root_turn(db, root, text, answer="ROOT ANSWER"):
    from tools.delegate_tool_messages import finish_collaboration_run, refresh_root_wakes
    db.append_message(root.session_id, "user", text)
    refresh_root_wakes(root)
    db.append_message(root.session_id, "assistant", answer)
    finish_collaboration_run(root, {"completed": True, "final_response": answer})
    return answer


@pytest.mark.asyncio
async def test_idle_root_is_woken_on_its_own_route_and_its_output_is_stored(tmp_path, monkeypatch):
    from tools.delegate_tool_collab import get_message

    _runner, authority, ref, _actor, db = await _gateway(tmp_path, monkeypatch)
    root = _root(ref, db, status="idle")
    child = _child(root)
    try:
        out = await asyncio.to_thread(
            _send, child, "msg-idle-wake-1", "the build finished; read evidence.txt")
        assert out["state"] == "accepted", out
        assert out["delivery"] == "root_wake" and out["native_receipt"], out

        admissions = _root_wake_rows(authority, ref)
        assert len(admissions) == 1, admissions
        body = str(admissions[0]["payload"].get("text") or "")
        assert body.startswith("[agent message msg-idle-wake-1]"), body
        assert "the build finished" in body
        assert admissions[0]["status"] == "queued", admissions[0]

        answer = _run_root_turn(db, root, body)
        row = get_message(str(db.db_path), "msg-idle-wake-1")
        assert row["state"] == "processed" and row["outcome"] == "completed", row
        transcript = [m for m in db.get_messages(ref.session_id) if m.get("content")]
        assert any("msg-idle-wake-1" in str(m["content"]) for m in transcript)
        assert any(answer == str(m["content"]) for m in transcript)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_retried_idle_wake_replays_the_native_admission(tmp_path, monkeypatch):
    from tools.delegate_tool_wake import request_root_wake, wake_header

    _runner, authority, ref, _actor, db = await _gateway(tmp_path, monkeypatch)
    root = _root(ref, db, status="idle")
    child = _child(root)
    try:
        out = await asyncio.to_thread(_send, child, "msg-wake-retry", "replay me")
        assert out["state"] == "accepted", out
        first = _root_wake_rows(authority, ref)
        assert len(first) == 1, first

        # An exact producer retry carries the same deterministic identity and payload:
        # the native ledger must hand back the same committed admission, not a new turn.
        again = await asyncio.to_thread(
            request_root_wake,
            root_session_id=ref.session_id, profile_key=str(db.db_path),
            text=wake_header("msg-wake-retry", "sa-wake-a", "sa-wake-a", "replay me"),
            message_id="msg-wake-retry", tree_id=ref.session_id)
        assert again["ok"] is True, again
        assert again["admission_id"] == out["native_receipt"], (again, out)
        second = _root_wake_rows(authority, ref)
        assert len(second) == 1 and second[0]["admission_id"] == first[0]["admission_id"], second
    finally:
        db.close()


@pytest.mark.asyncio
async def test_busy_root_step_boundary_then_idle_wake_closes_the_completion(tmp_path, monkeypatch):
    from tools.delegate_tool_collab import get_message
    from tools.delegate_tool_messages import inject_inbox_messages

    _runner, authority, ref, _actor, db = await _gateway(tmp_path, monkeypatch)
    root = _root(ref, db, status="running")
    child = _child(root)
    try:
        busy = _send(child, "msg-busy-1", "seen at the next step")
        assert busy["state"] == "accepted" and busy["delivery"] == "next_step_inbox", busy
        messages = [{"role": "tool", "tool_call_id": "t1", "content": "tool output"}]
        assert inject_inbox_messages(root, messages) == 1
        db.append_message(ref.session_id, "user", messages[-1]["content"])
        from tools.delegate_tool_messages import finish_collaboration_run
        finish_collaboration_run(root, {"completed": True})
        assert get_message(str(db.db_path), "msg-busy-1")["state"] == "processed"

        root.status, root._current_turn_id = "idle", None
        idle = await asyncio.to_thread(_send, child, "msg-busy-2", "root is idle now")
        assert idle["state"] == "accepted" and idle["delivery"] == "root_wake", (
            idle.get("state"), idle.get("reason"), idle.get("error"))
        admissions = _root_wake_rows(authority, ref)
        assert len(admissions) == 1, admissions
        _run_root_turn(db, root, str(admissions[0]["payload"]["text"]))
        assert get_message(str(db.db_path), "msg-busy-2")["state"] == "processed"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_cross_profile_and_wrong_route_wake_are_denied(tmp_path, monkeypatch):
    from tools.delegate_tool_collab import get_message
    from tools.delegate_tool_wake import request_root_wake

    _runner, authority, ref, _actor, db = await _gateway(tmp_path, monkeypatch)
    root = _root(ref, db, status="idle")
    other = tmp_path / "other" / "state.db"
    other.parent.mkdir(parents=True, exist_ok=True)
    child = _child(root, profile_key=str(other))
    try:
        out = _send(child, "msg-cross-profile", "should not cross")
        assert out["state"] == "rejected" and out["reason"] == "unauthorized", out
        assert get_message(str(db.db_path), "msg-cross-profile") is None
        assert _root_wake_rows(authority, ref) == []

        denied = await asyncio.to_thread(
            request_root_wake,
            root_session_id=ref.session_id, profile_key=str(other),
            text="[agent message wrong]\nFrom agent x (x):\nhi",
            message_id="wrong", tree_id=ref.session_id)
        assert denied["ok"] is False and denied["reason"] in {"target_idle", "wrong_profile"}, denied
        assert _root_wake_rows(authority, ref) == []

        unknown = await asyncio.to_thread(
            request_root_wake,
            root_session_id="not-this-route", profile_key=str(db.db_path),
            text="[agent message wrong2]\nFrom agent x (x):\nhi",
            message_id="wrong2", tree_id=ref.session_id)
        assert unknown["ok"] is False and unknown["reason"] in {"target_idle", "unknown_target"}, unknown
    finally:
        db.close()


@pytest.mark.asyncio
async def test_idle_stop_cancels_the_queued_wake_and_cannot_resurrect(tmp_path, monkeypatch):
    from tools.delegate_tool_collab import get_message, tree

    _runner, authority, ref, _actor, db = await _gateway(tmp_path, monkeypatch)
    root = _root(ref, db, status="idle")
    child = _child(root)
    try:
        out = await asyncio.to_thread(_send, child, "msg-stop-wake-1", "wake, then stop")
        assert out["state"] == "accepted", (out.get("state"), out.get("reason"), out.get("error"))
        assert len(_root_wake_rows(authority, ref)) == 1

        assert await authority.record_collab_stop(ref.session_id) is True
        assert tree(str(db.db_path), ref.session_id)["state"] == "stopped"
        row = get_message(str(db.db_path), "msg-stop-wake-1")
        assert row["state"] == "rejected" and row["reason"] == "tree_stopped", row
        admissions = _root_wake_rows(authority, ref)
        assert len(admissions) == 1 and admissions[0]["status"] == "terminal", admissions
        assert admissions[0]["outcome"] == "cancelled", admissions[0]

        again = _send(child, "msg-stop-wake-2", "must not resurrect")
        assert again["state"] == "rejected" and again["reason"] == "tree_stopped", again
        assert len(_root_wake_rows(authority, ref)) == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_native_stop_is_durable_before_the_turn_exits(tmp_path, monkeypatch):
    from tools.delegate_tool_collab import ensure_tree, tree

    runner, authority, ref, actor, db = await _gateway(tmp_path, monkeypatch, schedule=True)
    ensure_tree(str(db.db_path), ref.session_id, str(db.db_path), activate=True)
    started, release = asyncio.Event(), asyncio.Event()

    async def handle(_event):
        started.set()
        await release.wait()
        return "done"

    monkeypatch.setattr(runner, "_handle_message", handle)
    try:
        from gateway.session_contract import Submission
        await authority.submit(actor, Submission("stop-before-exit", ref, {"text": "hold"}, "queue"))
        await async_wait_until(started.is_set, timeout=10, what="the root turn started")
        generation = authority._handle(ref).execution_generation
        await authority.interrupt(actor, ref, generation)
        # The turn has not returned; the stop is already durable.
        assert release.is_set() is False
        assert tree(str(db.db_path), ref.session_id)["state"] == "stopped"
        release.set()
        await async_wait_until(lambda: authority._handle(ref).execution_state == "idle",
                               timeout=10, what="the stopped turn settled")
    finally:
        release.set()
        db.close()


@pytest.mark.asyncio
async def test_stop_between_dispatch_and_registration_never_starts_a_run(tmp_path, monkeypatch):
    from tools.delegate_tool_collab import begin_dispatch_message, ensure_tree, get_message, stop_tree
    from tools.delegate_tool_messages import register_continuation_or_refuse

    _runner, _authority, ref, _actor, db = await _gateway(tmp_path, monkeypatch)
    root = _root(ref, db, status="idle")
    try:
        ensure_tree(str(db.db_path), ref.session_id, str(db.db_path), activate=True)
        row, why = begin_dispatch_message(
            str(db.db_path), ref.session_id, "msg-race-1",
            sender_id="sa-wake-a", sender_session_id="sa-wake-a-sess", recipient_id="sa-wake-b",
            recipient_session_id="sa-wake-b-sess", target_input="sa-wake-b",
            fingerprint="fp", round_no=1)
        assert row is not None and why is None, why
        stop_tree(str(db.db_path), ref.session_id, "root_stopped")

        child = SimpleNamespace(
            _delegate_message_meta={"message_id": "msg-race-1", "round": 1},
            _delegate_root_session_id=ref.session_id, session_id="sa-wake-b-sess",
            _delegate_profile_key=str(db.db_path))
        registered = []
        assert register_continuation_or_refuse(child, root, lambda: registered.append(True)) is False
        assert registered == []
        assert child._delegate_collab_refused == "tree_stopped"
        assert get_message(str(db.db_path), "msg-race-1")["state"] == "rejected"

        # The production registration entry point refuses before any live member exists.
        from tools.delegate_tool_child_run import _register_child
        from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
        child._subagent_id = "sa-race-register"
        assert _register_child(
            child, root, "goal", owner_session_id=None, owner_transport=None,
            owner_session_record=None) is None
        with _active_subagents_lock:
            assert "sa-race-register" not in _active_subagents

        # The other order: registration holds the tree lock first, then the stop interrupts the
        # registered member (the production stop fan-out).
        ensure_tree(str(db.db_path), ref.session_id, str(db.db_path), activate=True)
        row2, why2 = begin_dispatch_message(
            str(db.db_path), ref.session_id, "msg-race-2",
            sender_id="sa-wake-a", sender_session_id="sa-wake-a-sess", recipient_id="sa-wake-c",
            recipient_session_id="sa-wake-c-sess", target_input="sa-wake-c",
            fingerprint="fp2", round_no=2)
        assert row2 is not None, why2
        child2 = SimpleNamespace(
            _delegate_message_meta={"message_id": "msg-race-2", "round": 2},
            _delegate_root_session_id=ref.session_id, session_id="sa-wake-c-sess",
            _delegate_profile_key=str(db.db_path))
        registered2 = []
        assert register_continuation_or_refuse(child2, root, lambda: registered2.append(True)) is True
        from tools.delegate_tool_messages import note_user_stop
        assert note_user_stop(str(db.db_path), ref.session_id, "root_stopped") is True
        assert registered2 == [True]
        assert get_message(str(db.db_path), "msg-race-2")["state"] == "rejected"
        assert get_message(str(db.db_path), "msg-race-2")["reason"] == "tree_stopped"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_stop_racing_an_admitted_wake_rejects_and_cancels_it(tmp_path, monkeypatch):
    from tools.delegate_tool_collab import get_message, tree
    from tools.delegate_tool_messages import note_user_stop
    import tools.delegate_tool_wake as wake_mod

    _runner, authority, ref, _actor, db = await _gateway(tmp_path, monkeypatch)
    root = _root(ref, db, status="idle")
    child = _child(root)
    real = wake_mod.request_root_wake

    def racing(**kwargs):
        note_user_stop(str(db.db_path), ref.session_id, "root_stopped")
        return real(**kwargs)

    monkeypatch.setattr(wake_mod, "request_root_wake", racing)
    try:
        out = await asyncio.to_thread(_send, child, "msg-race-wake", "should not run")
        assert out["state"] == "rejected" and out["reason"] == "tree_stopped", out
        assert tree(str(db.db_path), ref.session_id)["state"] == "stopped"
        row = get_message(str(db.db_path), "msg-race-wake")
        assert row["state"] == "rejected" and row["reason"] == "tree_stopped", row
        admissions = _root_wake_rows(authority, ref)
        assert all(a["status"] == "terminal" for a in admissions), admissions
    finally:
        db.close()


@pytest.mark.asyncio
async def test_idle_root_closure_through_a_finished_sibling(tmp_path, monkeypatch):
    """B revives A by continuation; A reports to an idle root; the root owns the output."""
    from tests.tools import test_agent_collaboration as H
    from tools.delegate_tool_collab import get_message

    _runner, authority, ref, _actor, db = await _gateway(tmp_path, monkeypatch)
    root = H._parent("root-alias", db=db)
    root.session_id = ref.session_id
    root._delegate_profile_key = str(db.db_path)
    H._EVENTS.clear()
    report_sent = threading.Event()
    evidence = tmp_path / "closure.txt"
    evidence.write_text("A read the closure evidence", encoding="utf-8")
    runs = {"n": 0}
    report = {}

    def a_script(child, user_message):
        runs["n"] += 1
        if runs["n"] == 1:
            return "A finished; waiting for B"
        report.update(_send(child, "msg-a-report",
                            f"A read: {evidence.read_text(encoding='utf-8')}"))
        report_sent.set()
        return "A reported to lead"

    try:
        a_sid, a_deleg, a_model = H._spawn(root, "A waits for B", a_script)
        H._wait_event(lambda e: e.get("delegation_id") == a_deleg,
                      "A completion", timeout=40)
        root.status, root._current_turn_id = "idle", None
        b = H._live_child(root, "sa-closure-b")
        with patch("run_agent.AIAgent", side_effect=a_model.factory):
            out = await asyncio.to_thread(
                _send_to, b, a_sid, "msg-b-to-a", "A, report your evidence to lead")
            assert out["state"] == "accepted", out
            await async_wait_until(report_sent.is_set, timeout=40,
                                   what="A's continuation reported")
        assert report.get("state") == "accepted", report
        admissions = _root_wake_rows(authority, ref)
        assert len(admissions) == 1, admissions
        body = str(admissions[0]["payload"]["text"])
        assert "A read the closure evidence" in body
        _run_root_turn(db, root, body, answer="ROOT RECEIVED A")
        row = get_message(str(db.db_path), "msg-a-report")
        assert row["state"] == "processed" and row["outcome"] == "completed", row
        transcript = "\n".join(str(m.get("content")) for m in db.get_messages(ref.session_id))
        assert "A read the closure evidence" in transcript
        assert "ROOT RECEIVED A" in transcript
        assert "A, report your evidence to lead" not in transcript
        assert len(a_model.turns) == 2
        assert a_model.turns[0]["session_id"] == a_model.turns[1]["session_id"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_recovery_rejects_a_root_wake_under_a_stopped_tree(tmp_path, monkeypatch):
    from tools.delegate_tool_collab import begin_wake_message, ensure_tree, get_message, recover_tree, stop_tree

    _runner, _authority, ref, _actor, db = await _gateway(tmp_path, monkeypatch)
    try:
        ensure_tree(str(db.db_path), ref.session_id, str(db.db_path), activate=True)
        row, why = begin_wake_message(
            str(db.db_path), ref.session_id, "msg-recover-wake",
            sender_id="sa-wake-a", sender_session_id="sa-wake-a-sess", recipient_id="lead",
            recipient_session_id=ref.session_id, target_input="lead", fingerprint="fp", round_no=1)
        assert row is not None, why
        stop_tree(str(db.db_path), ref.session_id, "root_stopped")
        # Simulate a dead owner: recovery may touch this row.
        import sqlite3
        with sqlite3.connect(str(db.db_path)) as conn:
            conn.execute("UPDATE delegate_collab_messages SET owner_pid=999999 WHERE message_id=?",
                         ("msg-recover-wake",))
        result = recover_tree(str(db.db_path), ref.session_id)
        assert result["dispatched"] == 0
        row = get_message(str(db.db_path), "msg-recover-wake")
        assert row["state"] == "rejected" and row["reason"] == "tree_stopped", row
    finally:
        db.close()


@pytest.mark.asyncio
async def test_idle_stop_closes_the_tree_after_root_compression(tmp_path, monkeypatch):
    """A compressed root keeps the same admission identity; /stop must close that tree.

    Compression advances the live agent to a physical tip while the authority's FIFO and
    ``/stop`` resolve the lineage root. A tree registered under the tip (as a message from a
    still-live child is) must still be closed by the person's Stop, otherwise a later message
    resurrects work the user stopped.
    """
    import sqlite3

    from tools.delegate_tool_collab import ensure_tree, tree

    _runner, authority, ref, _actor, db = await _gateway(tmp_path, monkeypatch)
    try:
        tip = f"{ref.session_id}-c1"
        db.create_session(tip, source="cli", parent_session_id=ref.session_id)
        with sqlite3.connect(str(db.db_path)) as conn:
            conn.execute("UPDATE sessions SET end_reason='compression' WHERE id=?",
                         (ref.session_id,))
        assert authority.logical_owner(tip) == ref.session_id

        # The live root now carries the physical tip; the message path registers the tree there.
        ensure_tree(str(db.db_path), tip, str(db.db_path), activate=True)
        root = _root(ref, db, status="idle")
        root.session_id = tip

        # run_inbound /stop resolves the route's logical owner, exactly as production does.
        assert await authority.record_collab_stop(ref.session_id) is True
        assert tree(str(db.db_path), tip)["state"] == "stopped"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_idle_stop_records_the_tree_before_a_restarted_route_is_mounted(tmp_path, monkeypatch):
    """A /stop as the first message after a restart must still durably stop the tree.

    The route is not mounted in ``authority.sessions`` until a message arrives; the idle stop
    handler used to return after fencing the route, leaving a pre-restart tree active so a later
    child message could wake work the person stopped.
    """
    from gateway.run_inbound import GatewayInboundMixin
    from tools.delegate_tool_collab import ensure_tree, tree

    _runner, authority, _ref, _actor, db = await _gateway(tmp_path, monkeypatch)
    try:
        sid = "unmounted-root-after-restart"
        db.create_session(sid, source="cli")
        ensure_tree(str(db.db_path), sid, str(db.db_path), activate=True)
        assert sid not in authority.sessions

        stub = SimpleNamespace(
            session_store=SimpleNamespace(peek_session_id=lambda _key: sid),
            session_authority=authority,
        )
        await GatewayInboundMixin._hm_stop_owner_turn(stub, "telegram:chat-1")
        assert tree(str(db.db_path), sid)["state"] == "stopped"
    finally:
        db.close()
