"""A stored ConnectError reply must reach the existing timer, not require a restart."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl
from gateway import run_startup
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_existing_timer_delivers_after_two_network_failures(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (12345, 67890))
    monkeypatch.setattr(dl, "ledger_enabled", lambda: True)
    clock = SimpleNamespace(time=lambda: now[0])
    monkeypatch.setattr(dl, "time", clock)
    monkeypatch.setattr(run_startup, "time", clock)
    local_asyncio = SimpleNamespace(**{name: getattr(asyncio, name) for name in dir(asyncio)})

    async def expire_deadline(awaitable, *, timeout):
        awaitable.close()
        now[0] += timeout
        raise TimeoutError

    local_asyncio.wait_for = expire_deadline
    monkeypatch.setattr(run_startup, "asyncio", local_asyncio)
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._draining = False
    runner._primary_profile_name = "default"
    runner._background_tasks = set()
    runner._profile_adapters = {}
    runner.session_store = None
    runner._async_session_store = SimpleNamespace(clear_resume_pending=AsyncMock(), _store=None)
    adapter = SimpleNamespace(
        platform=Platform.TELEGRAM, _owner_profile="default", is_connected=True,
        has_fatal_error=False, send_path_degraded=False,
        send=AsyncMock(return_value=SendResult(success=True, message_id="test-ack")),
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    dl.record_obligation(
        obligation_id="stalled", session_key="agent:default:telegram:dm:123",
        platform="telegram", chat_id="123", thread_id="57", content="original stored answer",
        adapter_profile="default",
    )
    dl.mark_failed("stalled", "httpx.ConnectError: ")
    with dl._connect() as conn:
        conn.execute("UPDATE delivery_obligations SET attempts=2 WHERE obligation_id='stalled'")

    runner._schedule_flood_redelivery(Platform.TELEGRAM, profile="default")
    worker = runner._flood_redelivery_tasks[("telegram", "default")]
    try:
        await asyncio.wait_for(worker, timeout=5)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.kwargs["content"].endswith("original stored answer")
    assert adapter.send.await_args.kwargs["metadata"] == {"thread_id": "57"}
    with dl._connect() as conn:
        assert conn.execute(
            "SELECT state,attempts FROM delivery_obligations WHERE obligation_id='stalled'"
        ).fetchone() == ("delivered", 3)
    assert runner._flood_redelivery_tasks == {}
