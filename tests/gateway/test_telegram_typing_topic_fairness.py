"""Regression contracts: topic pauses and shared typing budget must not hide sibling work."""
import asyncio
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def adapter():
    obj = TelegramAdapter(PlatformConfig(enabled=True, token='test-no-network'))
    obj._bot = SimpleNamespace(send_chat_action=AsyncMock())
    return obj


def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr('asyncio.get_running_loop', lambda: SimpleNamespace(time=lambda: now[0]))
    return now


@pytest.mark.asyncio
async def test_paused_topic_does_not_suppress_sibling_or_allow_detached_rearm():
    obj = adapter()
    obj.pause_typing_for_chat('4242', metadata={'thread_id': '11'})
    await obj.send_typing('4242', metadata={'thread_id': '11'})
    await obj.send_typing('4242', metadata={'thread_id': '22'})
    assert [c.kwargs['message_thread_id'] for c in obj._bot.send_chat_action.await_args_list] == [22]


@pytest.mark.asyncio
async def test_pause_scope_is_used_by_the_actual_refresh_loop():
    obj = adapter()
    stop = asyncio.Event()
    async def sent(**kwargs):
        assert kwargs['message_thread_id'] == 22
        stop.set()
    obj._bot.send_chat_action.side_effect = sent
    obj.pause_typing_for_chat('4242', metadata={'thread_id': '11'})
    await asyncio.wait_for(obj._keep_typing('4242', interval=0.01,
        metadata={'thread_id': '22'}, stop_event=stop), timeout=2.0)
    assert obj._bot.send_chat_action.await_count == 1
    # Finishing B must not resume A's prompt.
    await obj.send_typing('4242', metadata={'thread_id': '11'})
    assert obj._bot.send_chat_action.await_count == 1


@pytest.mark.asyncio
async def test_stable_topic_order_shares_budget_fairly(monkeypatch):
    obj = adapter()
    now = clock(monkeypatch)
    for tick in range(6):
        now[0] = 100.0 + tick * 2.1
        await obj.send_typing('4242', metadata={'thread_id': '11'})
        now[0] += 0.01
        await obj.send_typing('4242', metadata={'thread_id': '22'})
    topics = [c.kwargs['message_thread_id'] for c in obj._bot.send_chat_action.await_args_list]
    assert topics == [11, 22, 11, 22, 11, 22]


@pytest.mark.asyncio
async def test_pausing_a_waiting_topic_releases_its_queue_position(monkeypatch):
    obj = adapter()
    now = clock(monkeypatch)
    await obj.send_typing('4242', metadata={'thread_id': '11'})
    await obj.send_typing('4242', metadata={'thread_id': '22'})
    obj.pause_typing_for_chat('4242', metadata={'thread_id': '22'})
    now[0] += 2.1
    await obj.send_typing('4242', metadata={'thread_id': '11'})
    assert obj._bot.send_chat_action.await_count == 2


@pytest.mark.asyncio
async def test_abandoned_waiter_does_not_block_live_topic_forever(monkeypatch):
    obj = adapter()
    now = clock(monkeypatch)
    await obj.send_typing('4242', metadata={'thread_id': '11'})
    await obj.send_typing('4242', metadata={'thread_id': '22'})
    now[0] += 10.0
    await obj.send_typing('4242', metadata={'thread_id': '11'})
    assert obj._bot.send_chat_action.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('topics', [('11', '22', '33'), ('11', '22', '33', '44')])
async def test_concurrent_topics_refresh_before_platform_expiry(monkeypatch, topics):
    obj = adapter()
    now = clock(monkeypatch)
    sends = []
    async def action(**kwargs):
        sends.append((now[0], kwargs['message_thread_id']))
    obj._bot.send_chat_action.side_effect = action
    # The production refresher must offer admission often enough to use the shared budget.
    for tick in range(40):
        for index, topic in enumerate(topics):
            now[0] = 100.0 + tick * 0.5 + index * 0.001
            await obj.send_typing('4242', metadata={'thread_id': topic})
    assert all(b[0] - a[0] >= 1.0 for a, b in pairwise(sends))
    for topic in topics:
        times = [time for time, actual in sends if actual == int(topic)]
        assert len(times) >= 3
        assert max(b - a for a, b in pairwise(times)) < 5.0


@pytest.mark.asyncio
async def test_abandoned_waiter_cannot_hide_work_for_a_full_typing_lifetime(monkeypatch):
    obj = adapter()
    now = clock(monkeypatch)
    await obj.send_typing('4242', {'thread_id': '11'})
    await obj.send_typing('4242', {'thread_id': '22'})  # one-shot waiter never retries
    for time in (101.0, 102.0, 103.0):
        now[0] = time
        await obj.send_typing('4242', {'thread_id': '11'})
    assert obj._bot.send_chat_action.await_count >= 2


@pytest.mark.asyncio
async def test_message_thread_alias_cannot_bypass_a_topic_pause():
    obj = adapter()
    obj.pause_typing_for_chat('4242', {'thread_id': '11'})
    await obj.send_typing('4242', {'message_thread_id': '11'})
    assert obj._bot.send_chat_action.await_count == 0


@pytest.mark.asyncio
async def test_default_refreshers_keep_three_live_topics_visible():
    obj = adapter()
    stop, done = asyncio.Event(), asyncio.Event()
    sends = {11: [], 22: [], 33: []}
    async def action(**kwargs):
        sends[kwargs['message_thread_id']].append(asyncio.get_running_loop().time())
        if all(len(times) >= 2 for times in sends.values()):
            done.set()
    obj._bot.send_chat_action.side_effect = action
    tasks = [asyncio.create_task(obj._keep_typing('4242', metadata={'thread_id': str(topic)},
             stop_event=stop)) for topic in sends]
    try:
        await asyncio.wait_for(done.wait(), timeout=10.0)
        assert all(max(b-a for a,b in pairwise(times)) < 5.0 for times in sends.values())
    finally:
        stop.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
