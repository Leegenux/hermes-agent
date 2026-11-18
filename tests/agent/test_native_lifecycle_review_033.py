"""Controlled cancellation probe; no network requests."""
import asyncio
from types import SimpleNamespace

import pytest

from agent.auxiliary_async_codex import AsyncCodexCompletionsAdapter


def test_cancellation_during_client_close_does_not_interrupt_close(monkeypatch):
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        closed = []

        class Stream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise ValueError('primary')

            async def close(self):
                pass

        async def create(**kwargs):
            return Stream()

        async def close():
            started.set()
            await release.wait()
            closed.append(True)

        attempt = SimpleNamespace(responses=SimpleNamespace(create=create), close=close)
        async def make_attempt(template):
            return attempt
        monkeypatch.setattr('agent.auxiliary_async_codex._attempt_client', make_attempt)
        sync = SimpleNamespace(_client=SimpleNamespace(base_url='http://127.0.0.1/v1'),
                               _build_responses_kwargs=lambda kwargs: ({'model': 'test'}, 'test', 20))
        task = asyncio.create_task(AsyncCodexCompletionsAdapter(sync).create())
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed == [True]
    asyncio.run(run())


def test_cooperative_cleanup_timeout_leaves_no_pending_owned_tasks(monkeypatch):
    async def run():
        started, exited, never = asyncio.Event(), asyncio.Event(), asyncio.Event()
        before_tasks = set(asyncio.all_tasks())

        class Stream:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise ValueError('primary timeout probe')
            async def close(self):
                pass

        async def make_response(**kwargs):
            return Stream()
        async def close():
            started.set()
            try:
                await never.wait()
            finally:
                exited.set()
        attempt = SimpleNamespace(responses=SimpleNamespace(create=make_response), close=close)
        async def make_attempt(template):
            return attempt
        monkeypatch.setattr('agent.auxiliary_async_codex._attempt_client', make_attempt)
        sync = SimpleNamespace(_client=SimpleNamespace(base_url='http://127.0.0.1/v1'),
                               _build_responses_kwargs=lambda kwargs: ({'model': 'test'}, 'test', 20))
        task = asyncio.create_task(AsyncCodexCompletionsAdapter(sync).create())
        try:
            await asyncio.wait_for(started.wait(), 2)
            with pytest.raises(ValueError, match='primary timeout probe'):
                await asyncio.wait_for(asyncio.shield(task), 2)
            assert task.done(), 'test timeout must not stand in for adapter completion'
            await asyncio.wait_for(exited.wait(), 0.5)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            pending = [item for item in set(asyncio.all_tasks()) - before_tasks if not item.done()]
            assert not pending, pending
        finally:
            never.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())
