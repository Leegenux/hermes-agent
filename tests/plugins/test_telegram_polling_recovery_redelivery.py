"""Regression: in-place polling recovery must wake the ledger for replies rejected while degraded."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Runner:
    def __init__(self):
        self.calls = []
        self.woken = asyncio.Event()

    def _schedule_send_path_recovery(self, adapter, *, reason="recovered"):
        self.calls.append((adapter.platform, {"reason": reason}))
        self.woken.set()


def _adapter(runner):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123456:test-token"))
    adapter._running = True
    adapter.gateway_runner = runner
    adapter._polling_network_error_count = 1
    generation, _ = adapter._begin_polling_generation()
    return adapter, generation


@pytest.mark.asyncio
async def test_in_place_recovery_wakes_failed_delivery_exactly_once():
    runner = _Runner()
    adapter, generation = _adapter(runner)
    adapter._send_path_degraded = True

    assert adapter._record_polling_progress(generation) is True

    await asyncio.wait_for(runner.woken.wait(), timeout=5)
    assert adapter._send_path_degraded is False
    assert [call[0] for call in runner.calls] == [Platform.TELEGRAM]

    assert adapter._record_polling_progress(generation) is True
    await asyncio.sleep(0)
    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_healthy_progress_does_not_wake_failed_delivery():
    runner = _Runner()
    adapter, generation = _adapter(runner)
    adapter._send_path_degraded = False

    assert adapter._record_polling_progress(generation) is True
    await asyncio.sleep(0)

    assert runner.calls == []


@pytest.mark.asyncio
async def test_recovery_without_runner_is_not_an_error():
    adapter, generation = _adapter(SimpleNamespace())
    adapter._send_path_degraded = True

    assert adapter._record_polling_progress(generation) is True
    assert adapter._send_path_degraded is False
