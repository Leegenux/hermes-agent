"""The foreground refresher must survive a queued successor after stream finalization.

Topic scope and continuation behavior adapted from PR #136696, head
21e6dc3c2ded95aa5620889d90083feb0eea1a7a; transport is fake, runner/base methods are real.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
async def test_stream_finalize_leaves_sibling_topic_refreshing(monkeypatch):
    obj = TelegramAdapter(PlatformConfig(enabled=True, token='test-no-network'))
    obj._bot = SimpleNamespace(send_chat_action=AsyncMock())
    runner = object.__new__(GatewayRunner)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='123', chat_type='group', thread_id='10')
    scfg = SimpleNamespace(cursor='', edit_interval=0.01, buffer_threshold=1,
                           fresh_final_after_seconds=0, transport='edit')
    _, finalize = runner._build_stream_consumer_config(source, scfg, obj, on_missing_cursor='raise')
    finalize()
    stop = asyncio.Event()
    async def action(**kwargs):
        assert kwargs['message_thread_id'] == 20
        stop.set()
    obj._bot.send_chat_action.side_effect = action
    await asyncio.wait_for(obj._keep_typing('123', interval=0.01,
        metadata={'thread_id': '20'}, stop_event=stop), timeout=2.0)
    assert obj._bot.send_chat_action.await_count == 1


@pytest.mark.asyncio
async def test_queued_work_rearms_refresh_after_stream_finalization(monkeypatch):
    # No rate limiting in this test: the independent fairness suite pins that contract.
    monkeypatch.setattr('plugins.platforms.telegram.telegram_typing_gate.TYPING_CHAT_MIN_INTERVAL_SECS', 0)
    obj = TelegramAdapter(PlatformConfig(enabled=True, token='test-no-network'))
    obj._bot = SimpleNamespace(send_chat_action=AsyncMock())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='123', chat_type='group', thread_id='10')
    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: obj
    runner._refresh_agent_cache_message_count = AsyncMock()
    ctx = TurnContext(source=source, session_key='s', session_id='session')
    ctx._status_thread_metadata = {'thread_id': '10'}
    ctx.history = []
    ctx.context_prompt = ''
    ctx._interrupt_depth = 0
    stop = asyncio.Event()
    obj._active_sessions['s'] = stop
    repeated = asyncio.Event()
    calls = []
    async def action(**kwargs):
        calls.append(kwargs)
        if len(calls) >= 3:
            repeated.set()
    obj._bot.send_chat_action.side_effect = action
    scfg = SimpleNamespace(cursor='', edit_interval=0.01, buffer_threshold=1,
                           fresh_final_after_seconds=0, transport='edit')
    _, before_finalize = runner._build_stream_consumer_config(source, scfg, obj, on_missing_cursor='raise')
    async def deliver(*args):
        before_finalize()
    runner._run_agent_deliver_first_response = deliver
    async def followup(**kwargs):
        await asyncio.wait_for(repeated.wait(), 2.0)
        return {'final_response': 'done', 'messages': []}
    runner._run_agent = followup
    task = asyncio.create_task(obj._keep_typing('123', interval=0.01,
        metadata=ctx._status_thread_metadata, stop_event=stop))
    try:
        await runner._run_agent_queued_followup(ctx, obj, 'continue', None, 'first', {'messages': []}, None)
        assert len(calls) >= 3
        assert all(c['message_thread_id'] == 10 for c in calls)
    finally:
        stop.set()
        await asyncio.wait_for(task, 2.0)
