"""Controlled lifecycle probe: cancellation during error cleanup must not disappear.

The attempt client is injected; no SDK request or network IO occurs. The real
adapter, guard and finally block execute. This is not independent code review.
"""
import asyncio
from types import SimpleNamespace

import pytest

from agent.auxiliary_async_codex import AsyncCodexCompletionsAdapter


def test_external_cancellation_during_error_cleanup_is_not_swallowed(monkeypatch):
    async def run():
        cleanup_started = asyncio.Event()
        never = asyncio.Event()
        closed_client = []

        class BrokenStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise ValueError('primary iteration failure')

            async def close(self):
                cleanup_started.set()
                await never.wait()

        async def create(**kwargs):
            return BrokenStream()

        async def close():
            closed_client.append(True)

        attempt = SimpleNamespace(responses=SimpleNamespace(create=create), close=close)
        async def make_attempt(template):
            return attempt
        monkeypatch.setattr('agent.auxiliary_async_codex._attempt_client', make_attempt)
        sync = SimpleNamespace(_client=SimpleNamespace(base_url='http://127.0.0.1/v1'),
            _build_responses_kwargs=lambda kwargs: ({'model': 'test'}, 'test', 20))
        task = asyncio.create_task(AsyncCodexCompletionsAdapter(sync).create())
        try:
            await asyncio.wait_for(cleanup_started.wait(), 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert closed_client == [True]
        finally:
            never.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())
