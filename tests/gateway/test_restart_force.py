"""Force must reach both restart waits, including escalation of an already draining restart."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.i18n import t
from gateway.platforms.event import MessageEvent, MessageType
from gateway.restart import effective_stop_drain_timeout
from gateway.run import GatewayRunner
from gateway.run_shutdown import GatewayShutdownMixin
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def event(text):
    return MessageEvent(text=text, message_type=MessageType.TEXT,
                        source=make_restart_source(), message_id="force-test")


@pytest.fixture(autouse=True)
def isolated_supervisor(monkeypatch):
    monkeypatch.setattr("gateway.restart.is_gateway_supervisor_process", lambda: True)
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)


@pytest.mark.asyncio
@pytest.mark.parametrize("already_draining", [False, True])
async def test_slash_force_reaches_restart_and_does_not_claim_to_wait(already_draining):
    runner, adapter = make_restart_runner()
    runner._draining = already_draining
    runner._restart_requested = already_draining
    runner._running_agents["busy"] = MagicMock()
    runner.request_restart = MagicMock(return_value=True)
    reply = await runner._handle_message(event("/restart force"))
    assert runner.request_restart.call_args is not None
    assert runner.request_restart.call_args.kwargs.get("force") is True
    assert reply is None
    assert adapter.sent == [t("gateway.restart.restarting")]


@pytest.mark.asyncio
@pytest.mark.parametrize("already_draining", [False, True])
async def test_busy_force_ack_finishes_before_restart(already_draining):
    runner, adapter = make_restart_runner()
    runner._draining = already_draining
    runner._restart_requested = already_draining
    runner._running_agents["busy"] = MagicMock()
    sending, release = asyncio.Event(), asyncio.Event()
    trace = []

    async def send(*_args, **_kwargs):
        trace.append("send-start")
        sending.set()
        await release.wait()
        trace.append("send-complete")
        return SimpleNamespace(success=True, error=None, retryable=False)

    adapter.send = send
    runner.request_restart = MagicMock(side_effect=lambda **_kw: trace.append("restart"))
    request = asyncio.create_task(runner._handle_message(event("/restart force")))
    try:
        await asyncio.wait_for(sending.wait(), timeout=2)
        runner.request_restart.assert_not_called()
        release.set()
        assert await asyncio.wait_for(request, timeout=2) is None
        assert trace == ["send-start", "send-complete", "restart"]
        assert runner.request_restart.call_args.kwargs["force"] is True
    finally:
        release.set()
        if not request.done():
            request.cancel()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
async def test_force_skips_initial_after_turn_wait():
    runner, _ = make_restart_runner()
    runner.stop = AsyncMock()
    runner._await_active_work_before_restart = AsyncMock()
    try:
        assert runner.request_restart(via_service=True, force=True)
        await asyncio.wait_for(runner._restart_task, timeout=2)
        runner._await_active_work_before_restart.assert_not_awaited()
        runner.stop.assert_awaited_once_with(restart=True, detached_restart=False, service_restart=True)
    finally:
        task = getattr(runner, "_restart_task", None)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_force_escalates_existing_after_turn_wait_without_second_restart_task():
    runner, _ = make_restart_runner()
    runner.stop = AsyncMock()
    runner._restart_after_turn_timeout = 1800
    waiting = asyncio.Event()

    def live_work():
        waiting.set()
        return 1

    runner._active_work_count = live_work
    runner._awaitable_work_count = live_work
    runner._describe_active_work = lambda: "busy turn"
    runner._wedged_agent_count = lambda: 0
    runner._restart_safe_cron_count = lambda: 0
    assert runner.request_restart(via_service=True)
    task = runner._restart_task
    try:
        await asyncio.wait_for(waiting.wait(), timeout=2)
        assert not task.done()
        assert runner.request_restart(via_service=True, force=True)
        assert runner._restart_task is task
        await asyncio.wait_for(task, timeout=2)
        runner.stop.assert_awaited_once()
        assert not runner.request_restart(via_service=True, force=True)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("force", [False, True])
async def test_stop_uses_zero_chat_and_cron_budgets_only_for_force(force, monkeypatch):
    runner, _ = make_restart_runner()
    runner._restart_force = force
    runner._restart_drain_timeout = 42
    runner._cron_drain_timeout = 60
    runner._active_cron_job_count = lambda: 1
    runner._active_api_run_count = lambda: 1
    runner._drain_active_agents = AsyncMock(return_value=({}, True))
    monkeypatch.setattr(GatewayRunner, "_mark_running_sessions_resume_pending", AsyncMock(return_value=[]))
    ctx = GatewayShutdownMixin._StopContext(deferred_count=lambda: 1, started_at=time.monotonic())
    timeout = effective_stop_drain_timeout(runner)
    await runner._stop_drain_active_work(timeout, ctx)
    chat, cron = runner._drain_active_agents.call_args.args
    assert chat == (0 if force else 42)
    assert cron == (0 if force else 60)
    assert ctx.timed_out


@pytest.mark.asyncio
async def test_force_escalation_breaks_an_active_stop_drain():
    runner, _ = make_restart_runner()
    runner._drain_work_counts = lambda: (1, 1, 1, 1)
    waiting = asyncio.Event()
    runner._update_runtime_status = lambda *_args: waiting.set()
    drain = asyncio.create_task(runner._drain_active_agents(1800, 1800))
    try:
        await asyncio.wait_for(waiting.wait(), timeout=2)
        runner._restart_force = True
        _, timed_out = await asyncio.wait_for(drain, timeout=2)
        assert timed_out
    finally:
        if not drain.done():
            drain.cancel()
        await asyncio.gather(drain, return_exceptions=True)


@pytest.mark.asyncio
async def test_force_stop_keeps_resume_marks_interrupts_and_cleanup_in_order(monkeypatch):
    runner, _ = make_restart_runner()
    runner._restart_force = True
    runner._restart_requested = True
    runner._restart_drain_timeout = 1800
    runner._cron_drain_timeout = 1800
    runner._restart_command_source = make_restart_source()
    agent = MagicMock()
    runner._running_agents["busy"] = agent
    trace = []
    live = {"api": 1, "deferred": 1, "managed": 1}

    async def mark(*_args):
        trace.append("mark_resume")

    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, mark_resume_pending=mark,
    )
    runner._active_cron_job_count = lambda: 1
    runner._active_api_run_count = lambda: live["api"]
    runner._active_deferred_agent_worker_count = lambda: live["deferred"]

    def interrupt(kind):
        trace.append("interrupt_" + kind)
        if kind == "agent":
            runner._running_agents.clear()
        elif kind in live:
            live[kind] = 0
        return 1

    monkeypatch.setattr("gateway.run.request_hard_interrupt", lambda *_a, **_k: interrupt("agent"))
    runner._interrupt_api_server_runs = lambda _reason: interrupt("api")
    runner._interrupt_deferred_agent_workers = lambda _reason: interrupt("deferred")
    monkeypatch.setattr("gateway.run_shutdown.stop_managed_turns", lambda *_a: interrupt("managed"))
    monkeypatch.setattr("gateway.run_shutdown.interrupt_side_agents", lambda *_a: interrupt("side"))
    monkeypatch.setattr("gateway.run_shutdown.managed_turn_count", lambda _r: live["managed"])

    async def begin(_runner, ctx):
        ctx.started_at = time.monotonic()

    async def kill(_phase):
        trace.append("kill_tools")
        return []

    async def finalize(_runner, ctx):
        assert ctx.active_agents == {"busy": agent}
        trace.append("finalize")

    async def persist(_runner, _ctx):
        trace.append("persist")

    monkeypatch.setattr(GatewayRunner, "_stop_begin_teardown", begin)
    monkeypatch.setattr(GatewayRunner, "_stop_kill_tool_subprocesses_off_loop", staticmethod(kill))
    monkeypatch.setattr(GatewayRunner, "_stop_finalize_agents_and_adapters", finalize)
    monkeypatch.setattr(GatewayRunner, "_stop_release_runtime_state", AsyncMock())
    monkeypatch.setattr(GatewayRunner, "_stop_quiesce_and_close_session_dbs", lambda *_args: None)
    monkeypatch.setattr(GatewayRunner, "_stop_persist_exit_state", persist)
    settle = AsyncMock()
    monkeypatch.setattr("gateway.run_runtime.settle_gateway_runtime", settle)
    runner._notify_interrupted_cron_jobs = AsyncMock()
    await asyncio.wait_for(runner.stop(restart=True, service_restart=True), timeout=2)
    assert trace.index("mark_resume") < trace.index("interrupt_agent")
    for kind in ("agent", "api", "deferred", "managed", "side"):
        assert "interrupt_" + kind in trace
    assert trace.index("kill_tools") < trace.index("finalize") < trace.index("persist")
    assert settle.call_args.args[1] == 0
