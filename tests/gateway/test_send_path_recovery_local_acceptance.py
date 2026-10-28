"""Local acceptance tests for the unmodified upstream PR #123929 runtime.

Uses fake transports and an isolated SQLite file; never sends real messages.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from tests.gateway import test_send_path_recovery as recovery_tests

recovery_runner = recovery_tests.recovery_runner


@pytest.mark.asyncio
async def test_original_final_failure_lands_after_empty_recovery_sweep(
    recovery_runner, tmp_path, monkeypatch
):
    runner, clock = recovery_runner
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    home = tmp_path / "reviewer"
    home.mkdir()
    monkeypatch.setattr(runner, "_routed_profile_home", lambda profile: home)
    adapter, default = recovery_tests._adapter("reviewer"), recovery_tests._adapter()
    runner.adapters[Platform.TELEGRAM] = default
    runner._profile_adapters["reviewer"] = {Platform.TELEGRAM: adapter}
    dl.record_obligation(
        obligation_id="final-lands-late", session_key="agent:reviewer:telegram:dm:123",
        platform="telegram", chat_id="123", thread_id="57", content="late reply",
        adapter_profile="reviewer",
    )
    dl.record_obligation(
        obligation_id="default-unrelated", session_key="agent:default:telegram:dm:999",
        platform="telegram", chat_id="999", thread_id=None, content="must stay with default",
        adapter_profile="default",
    )
    dl.mark_failed("default-unrelated", "send_path_degraded")
    # A successful recovery scan happens before the original failure commits.
    runner._schedule_send_path_recovery(adapter, reason="recovered")
    job = runner._send_path_recovery_jobs[("telegram", "reviewer")]["task"]
    (await clock.next_window()).set_result(None)
    await asyncio.wait_for(job, timeout=5)
    adapter.send.assert_not_awaited()
    assert runner._send_path_recovery_jobs == {}

    source = SimpleNamespace(platform=Platform.TELEGRAM)
    owner = SimpleNamespace(
        gateway_runner=runner, _final_delivery_adapter=lambda _: adapter
    )
    await BasePlatformAdapter._finalize_delivery_obligation(
        owner, "final-lands-late",
        SendResult(success=False, error="send_path_degraded", retryable=True),
        SimpleNamespace(source=source), adapter,
    )
    job = runner._send_path_recovery_jobs[("telegram", "reviewer")]["task"]
    (await clock.next_window()).set_result(None)
    await asyncio.wait_for(job, timeout=5)
    adapter.send.assert_awaited_once()
    default.send.assert_not_awaited()
    assert adapter.send.await_args.kwargs["metadata"] == {"thread_id": "57"}
    with dl._connect() as conn:
        assert conn.execute(
            "SELECT state,attempts FROM delivery_obligations WHERE obligation_id='final-lands-late'"
        ).fetchone() == ("delivered", 1)
        assert conn.execute(
            "SELECT state,attempts FROM delivery_obligations WHERE obligation_id='default-unrelated'"
        ).fetchone() == ("failed", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ["stopped", "draining", "disconnected", "fatal", "degraded"])
async def test_unhealthy_runner_or_adapter_never_schedules(recovery_runner, guard):
    runner, _ = recovery_runner
    adapter = recovery_tests._adapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    target, attribute, value = {
        "stopped": (runner, "_running", False),
        "draining": (runner, "_draining", True),
        "disconnected": (adapter, "is_connected", False),
        "fatal": (adapter, "has_fatal_error", True),
        "degraded": (adapter, "send_path_degraded", True),
    }[guard]
    setattr(target, attribute, value)
    runner._schedule_send_path_recovery(adapter)
    assert not getattr(runner, "_send_path_recovery_jobs", {})


@pytest.mark.asyncio
async def test_cancel_before_first_coroutine_turn_retires_entry(recovery_runner):
    runner, _ = recovery_runner
    adapter = recovery_tests._adapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._schedule_send_path_recovery(adapter)
    job = runner._send_path_recovery_jobs[("telegram", "default")]["task"]
    job.cancel()
    await asyncio.gather(job, return_exceptions=True)
    await asyncio.sleep(0)
    assert runner._send_path_recovery_jobs == {}


@pytest.mark.asyncio
async def test_shutdown_after_scheduling_prevents_scan(recovery_runner):
    runner, clock = recovery_runner
    adapter = recovery_tests._adapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    sweep = runner._redeliver_failed_obligations_for_platform = AsyncMock()
    runner._schedule_send_path_recovery(adapter)
    job = runner._send_path_recovery_jobs[("telegram", "default")]["task"]
    quiet = await clock.next_window()
    runner._draining = True
    quiet.set_result(None)
    await asyncio.wait_for(job, timeout=5)
    sweep.assert_not_awaited()
    assert runner._send_path_recovery_jobs == {}
