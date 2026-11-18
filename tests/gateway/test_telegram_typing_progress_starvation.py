"""Typing must not starve behind repeated progress edits; real gate and adapter, fake Bot API."""
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def setup_clock(monkeypatch):
    obj = TelegramAdapter(PlatformConfig(enabled=True, token='offline-no-network'))
    now = [100.0]
    obj._bot = SimpleNamespace(send_chat_action=AsyncMock())
    # No I/O coroutine scheduling in these gate tests: the external Bot API is immediate.
    monkeypatch.setattr('asyncio.get_running_loop', lambda: SimpleNamespace(time=lambda: now[0]))
    return obj, now


@pytest.mark.asyncio
async def test_repeated_progress_slot_holds_cannot_starve_typing(monkeypatch):
    obj, now = setup_clock(monkeypatch)
    sent = []
    async def action(**kwargs):
        sent.append(now[0])
    obj._bot.send_chat_action.side_effect = action
    for tick in range(40):
        now[0] = 100.0 + tick * 0.5
        if tick >= 3 and tick % 2:
            obj._hold_chat_outbound_slot('4242')
        await obj.send_typing('4242', {'thread_id': '11'})
    assert len(sent) >= 5, sent
    assert max(b-a for a, b in pairwise(sent)) < 5.0
    assert all(b-a >= 2.0 for a, b in pairwise(sent)), 'chat typing rate limit still applies'


@pytest.mark.asyncio
async def test_first_pulse_is_not_hidden_by_an_existing_progress_slot(monkeypatch):
    obj, _now = setup_clock(monkeypatch)
    obj._hold_chat_outbound_slot('4242')
    await obj.send_typing('4242', {'thread_id': '11'})
    assert obj._bot.send_chat_action.await_count == 1
    assert obj._chat_outbound_slot_remaining('4242') == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_recent_pulse_yields_but_overdue_pulse_can_refresh(monkeypatch):
    obj, now = setup_clock(monkeypatch)
    await obj.send_typing('4242', {'thread_id': '11'})
    now[0] = 102.0
    obj._hold_chat_outbound_slot('4242')
    await obj.send_typing('4242', {'thread_id': '11'})
    assert obj._bot.send_chat_action.await_count == 1
    now[0] = 103.0
    obj._hold_chat_outbound_slot('4242')
    await obj.send_typing('4242', {'thread_id': '11'})
    assert obj._bot.send_chat_action.await_count == 2


@pytest.mark.asyncio
async def test_soft_slot_priority_does_not_bypass_hard_flood_or_pause(monkeypatch):
    obj, _now = setup_clock(monkeypatch)
    obj._hold_chat_outbound_slot('4242')
    obj._record_send_flood_cooldown('4242', 60.0)
    await obj.send_typing('4242', {'thread_id': '11'})
    assert obj._bot.send_chat_action.await_count == 0
    obj.pause_typing_for_chat('777', {'thread_id': '22'})
    await obj.send_typing('777', {'thread_id': '22'})
    assert obj._bot.send_chat_action.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('topics', [('11', '22'), ('11', '22', '33'), ('11', '22', '33', '44')])
async def test_stable_topics_with_progress_keep_fair_bounded_admission(monkeypatch, topics):
    obj, now = setup_clock(monkeypatch)
    sent = []
    async def action(**kwargs):
        sent.append((now[0], kwargs['message_thread_id']))
    obj._bot.send_chat_action.side_effect = action
    for tick in range(60):
        now[0] = 100.0 + tick * 0.5
        if tick % 2:
            obj._hold_chat_outbound_slot('4242')
        for topic in topics:
            await obj.send_typing('4242', {'thread_id': topic})
    assert all(b[0]-a[0] >= 1.0 for a, b in pairwise(sent))
    for topic in topics:
        times = [t for t, actual in sent if actual == int(topic)]
        assert len(times) >= 4, (topic, times)
        assert max(b-a for a, b in pairwise(times)) < 5.0, (topic, times)
