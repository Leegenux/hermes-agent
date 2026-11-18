"""Existing live work cannot be starved by repeated new-topic/turn arrivals."""
from itertools import pairwise
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize('same_topic', [False, True])
async def test_continuous_work_gets_typing_while_short_topics_keep_arriving(monkeypatch, same_topic):
    now = [100.0]
    monkeypatch.setattr('asyncio.get_running_loop', lambda: SimpleNamespace(time=lambda: now[0]))
    obj = TelegramAdapter(PlatformConfig(enabled=True, token='offline-no-network'))
    sent = []
    async def action(**kw):
        sent.append((now[0], kw['message_thread_id']))
    obj._bot = SimpleNamespace(send_chat_action=AsyncMock(side_effect=action))
    await obj.send_typing('4242', {'thread_id': '11'})
    for tick in range(1, 41):
        now[0] = 100.0 + tick * 0.5
        if tick % 4 == 0:
            topic = '22' if same_topic else str(1000+tick)
            await obj.send_typing('4242', {'thread_id': topic})
            await obj.stop_typing('4242', {'thread_id': topic})
        await obj.send_typing('4242', {'thread_id': '11'})
    live = [t for t, topic in sent if topic == 11]
    assert len(live) >= 5, ('continuous work lost every slot to new arrivals', sent)
    assert max(b-a for a,b in pairwise(live)) < 5.0
    assert all(b[0]-a[0] >= 2.0 for a,b in pairwise(sent)), 'chat budget still applies'


@pytest.mark.asyncio
async def test_existing_two_field_waiters_survive_gate_update(monkeypatch):
    now = [100.0]
    monkeypatch.setattr('asyncio.get_running_loop', lambda: SimpleNamespace(time=lambda: now[0]))
    obj = TelegramAdapter(PlatformConfig(enabled=True, token='offline-no-network'))
    obj._bot = SimpleNamespace(send_chat_action=AsyncMock())
    await obj.send_typing('4242', {'thread_id': '11'})
    # Exact in-memory format emitted by the preceding production gate.
    obj._telegram_typing_waiters['4242']['11'] = (100.0, 100.0)
    now[0] = 100.5
    await obj.send_typing('4242', {'thread_id': '22'})
    now[0] = 102.0
    await obj.send_typing('4242', {'thread_id': '22'})
    assert [c.kwargs['message_thread_id'] for c in obj._bot.send_chat_action.await_args_list] == [11, 22]

