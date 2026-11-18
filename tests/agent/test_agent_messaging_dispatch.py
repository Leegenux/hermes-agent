"""``agent_message`` / ``list_agents`` must receive the calling agent through the production
tool-dispatch paths, not only when a test hands ``parent_agent=`` to the function directly.

Registry dispatch (``model_tools.handle_function_call``) never injects ``parent_agent``, so a
registry-only registration made every live ``list_agents`` return ``[]`` and every
``agent_message`` reject with ``missing_sender`` (report 017). These tests drive the two real
executors — ``execute_tool_calls_sequential`` and ``invoke_tool`` — against a real ``AIAgent``,
a real ``SessionDB``, the real live-subagent registry and the real same-tree send engine.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

from agent.agent_runtime_helpers import invoke_tool
from agent.tool_executor import execute_tool_calls_sequential


def _tool_call(name: str, args: dict, call_id: str = "call-1") -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall.model_validate({
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    })


def _sequential(agent, name: str, args: dict) -> str:
    """Drive the real sequential executor and return the tool row the model would see."""
    messages: list[dict] = []
    assistant = ChatCompletionMessage(role="assistant", content=None, tool_calls=[_tool_call(name, args)])
    execute_tool_calls_sequential(agent, assistant, messages, "task-1", finalize=False)
    rows = [m for m in messages if m.get("role") == "tool"]
    assert len(rows) == 1, messages
    return rows[0]["content"]


def _concurrent(agent, name: str, args: dict) -> str:
    """Drive the real concurrent executor (``invoke_tool``), not just its resolver."""
    return invoke_tool(agent, name, args, "task-1", tool_call_id="call-1")


DISPATCH_PATHS = pytest.mark.parametrize("run", [_sequential, _concurrent], ids=["sequential", "concurrent"])


@pytest.fixture(autouse=True)
def _clean_registry():
    from tools.delegate_tool_messages import reset_message_state
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _recent_subagents

    with _active_subagents_lock:
        _active_subagents.clear()
        _recent_subagents.clear()
    reset_message_state()
    yield
    with _active_subagents_lock:
        _active_subagents.clear()
        _recent_subagents.clear()
    reset_message_state()


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A real root AIAgent, one live sibling, and one live child of a different tree."""
    from hermes_state import SessionDB
    from model_tools import _clear_tool_defs_cache
    from run_agent import AIAgent
    from tools.delegate_tool_registry import _register_subagent
    from tools.registry import invalidate_check_fn_cache

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("agent: {}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()

    db = SessionDB(db_path=tmp_path / "state.db")
    root = AIAgent(
        session_db=db, model="test-model", provider="openai-compat", api_key="test",
        base_url="http://127.0.0.1:1/v1", max_iterations=4,
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False,
    )
    root_session_id = root.session_id
    assert root_session_id

    def _child(sid: str, root_sid: str, session_id: str):
        agent = SimpleNamespace(
            session_id=session_id,
            _subagent_id=sid,
            status="running",
            _interrupt_requested=False,
            _delegate_root_session_id=root_sid,
            _delegate_profile_key=str(db.db_path),
            _delegate_inbox=[],
            _delegate_inbox_lock=threading.Lock(),
            _session_db=db,
            _memory_manager=None,
            valid_tool_names=None,
            enabled_toolsets=None,
            disabled_toolsets=None,
            quiet_mode=True,
            _current_turn_id=None,
        )
        # Every child of this root walks back to the real root agent.
        agent._delegate_parent_ref = lambda: root if root_sid == root_session_id else None
        _register_subagent({
            "subagent_id": sid, "agent": agent, "goal": "g", "status": "running",
            "owner_agent_session_id": root_sid, "root_session_id": root_sid,
            "profile_key": agent._delegate_profile_key, "accepting_messages": True,
        })
        return agent

    sibling = _child("sa-0-sibling", root_session_id, "sibling-session")
    foreign = _child("sa-0-foreign", "foreign-root-session", "foreign-session")
    return SimpleNamespace(root=root, sibling=sibling, foreign=foreign, db=db)


@DISPATCH_PATHS
def test_list_agents_reaches_dispatch_with_the_real_tree(tree, run):
    rows = json.loads(run(tree.root, "list_agents", {}))

    by_id = {row["id"]: row for row in rows}
    assert "lead" in by_id, rows
    assert by_id["lead"]["self"] is True
    assert by_id["lead"]["session_id"] == tree.root.session_id
    assert "sa-0-sibling" in by_id, rows
    assert by_id["sa-0-sibling"]["self"] is False
    # A live child of a different root is not part of this tree.
    assert "sa-0-foreign" not in by_id, rows


@DISPATCH_PATHS
def test_send_is_accepted_and_readable_in_the_sibling_inbox(tree, run):
    from tools.delegate_tool_collab import get_message

    result = json.loads(run(tree.root, "agent_message", {
        "target": "sa-0-sibling", "message": "hello sibling", "message_id": "msg-dispatch-1",
    }))

    assert result["state"] == "accepted", result
    assert result["accepted"] is True
    assert result["sender"]["id"] == "lead"
    assert result["recipient"]["id"] == "sa-0-sibling"

    inbox = tree.sibling._delegate_inbox
    assert len(inbox) == 1, inbox
    assert inbox[0]["message_id"] == "msg-dispatch-1"
    assert inbox[0]["sender_id"] == "lead"
    assert inbox[0]["content"] == "hello sibling"

    durable = get_message(str(tree.db.db_path), "msg-dispatch-1")
    assert durable is not None and durable["state"] == "accepted", durable
    assert durable["sender_id"] == "lead" and durable["recipient_id"] == "sa-0-sibling"


@DISPATCH_PATHS
def test_self_and_cross_tree_targets_are_still_rejected(tree, run):
    self_send = json.loads(run(tree.root, "agent_message", {"target": "lead", "message": "hi"}))
    assert self_send["state"] == "rejected" and self_send["reason"] == "self", self_send

    foreign_send = json.loads(run(tree.root, "agent_message", {
        "target": "sa-0-foreign", "message": "hi",
    }))
    assert foreign_send["state"] == "rejected", foreign_send
    assert foreign_send["reason"] == "unauthorized", foreign_send


@DISPATCH_PATHS
def test_unknown_target_reaches_resolution_instead_of_missing_sender(tree, run):
    result = json.loads(run(tree.root, "agent_message", {"target": "sa-0-nobody", "message": "hi"}))
    assert result["reason"] != "missing_sender", result
    assert result["state"] == "rejected" and result["reason"] == "unknown_target", result


def test_a_subagent_sees_itself_and_the_root_through_dispatch(tree):
    rows = json.loads(_concurrent(tree.sibling, "list_agents", {}))
    by_id = {row["id"]: row for row in rows}
    assert by_id["lead"]["self"] is False
    assert by_id["lead"]["session_id"] == tree.root.session_id
    assert by_id["sa-0-sibling"]["self"] is True
    assert "sa-0-foreign" not in by_id, rows
