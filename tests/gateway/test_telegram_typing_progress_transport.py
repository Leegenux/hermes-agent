"""Real Telegram edit/gate paths with controlled clock and fake external Bot API only."""
import asyncio
from itertools import pairwise
from types import SimpleNamespace

import pytest
import plugins.platforms.telegram.adapter as adapter_module
import plugins.platforms.telegram.telegram_typing_gate as gate_module
from gateway.config import PlatformConfig


class ClockAsyncio:
    def __init__(self, now):
        self.now = now
    def get_running_loop(self):
        return SimpleNamespace(time=lambda: self.now[0])
    def __getattr__(self, name):
        return getattr(asyncio, name)


class BotTransport:
    def __init__(self, now):
        self.now, self.actions, self.edits = now, [], []
        self.entered = asyncio.Event()
        self.hold_action = False
        self.fail_action = False
    async def send_chat_action(self, **kw):
        self.actions.append((self.now[0], kw))
        self.entered.set()
        if self.hold_action:
            await asyncio.Event().wait()
        if self.fail_action:
            raise OSError('controlled offline transport failure')
        return True
    async def edit_message_text(self, **kw):
        self.edits.append(kw)
        return SimpleNamespace(message_id=900)


def setup(monkeypatch):
    now = [100.0]
    facade = ClockAsyncio(now)
    monkeypatch.setattr(adapter_module, 'asyncio', facade)
    monkeypatch.setattr(gate_module, 'asyncio', facade)
    obj = adapter_module.TelegramAdapter(PlatformConfig(enabled=True, token='offline-no-network'))
    bot = BotTransport(now)
    obj._bot = bot
    obj._rich_send_disabled = True
    return obj, bot, now


@pytest.mark.asyncio
async def test_real_progress_edit_path_cannot_hide_overdue_typing(monkeypatch):
    obj, bot, now = setup(monkeypatch)
    sent = []
    for tick in range(15):
        now[0] = 100.0 + tick
        result = await obj.edit_message('4242', '900', 'progress '+str(tick))
        assert result.success
        await obj.send_typing('4242', {'thread_id': '11'})
    sent = [t for t, _ in bot.actions]
    assert len(sent) >= 4, sent
    assert max(b-a for a, b in pairwise(sent)) < 5.0
    assert len(bot.edits) == 15, 'typing must not freeze actual interim preview edits'


@pytest.mark.asyncio
async def test_typing_does_not_extend_slot_or_block_final_edit(monkeypatch):
    obj, bot, now = setup(monkeypatch)
    assert (await obj.edit_message('4242', '900', 'progress')).success
    now[0] = 100.2
    await obj.send_typing('4242', {'thread_id': '11'})
    assert len(bot.actions) == 1
    assert obj._chat_outbound_slot_remaining('4242') == pytest.approx(0.8)
    assert (await obj.edit_message('4242', '900', 'done', finalize=True)).success
    assert len(bot.edits) == 2


@pytest.mark.asyncio
async def test_cancelled_action_consumes_budget_but_does_not_block_next_eligible_tick(monkeypatch):
    obj, bot, now = setup(monkeypatch)
    bot.hold_action = True
    task = asyncio.create_task(obj.send_typing('4242', {'thread_id': '11'}))
    await asyncio.wait_for(bot.entered.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    bot.hold_action = False
    now[0] = 100.5
    await obj.send_typing('4242', {'thread_id': '11'})
    assert len(bot.actions) == 1, 'unknown API outcome must not trigger an immediate duplicate'
    now[0] = 103.0
    obj._hold_chat_outbound_slot('4242')
    await obj.send_typing('4242', {'thread_id': '11'})
    assert len(bot.actions) == 2


@pytest.mark.asyncio
async def test_transient_failure_keeps_hard_cooldown_even_when_typing_is_overdue(monkeypatch):
    obj, bot, now = setup(monkeypatch)
    bot.fail_action = True
    await obj.send_typing('4242', {'thread_id': '11'})
    now[0] = 110.0
    obj._hold_chat_outbound_slot('4242')
    await obj.send_typing('4242', {'thread_id': '11'})
    assert len(bot.actions) == 1
    bot.fail_action = False
    now[0] = 131.0
    obj._hold_chat_outbound_slot('4242')
    await obj.send_typing('4242', {'thread_id': '11'})
    assert len(bot.actions) == 2


@pytest.mark.asyncio
async def test_non_head_topic_cannot_use_soft_slot_priority_to_jump_queue(monkeypatch):
    obj, bot, now = setup(monkeypatch)
    await obj.send_typing('4242', {'thread_id': '11'})
    await obj.send_typing('4242', {'thread_id': '22'})
    now[0] = 102.0
    obj._hold_chat_outbound_slot('4242')
    await obj.send_typing('4242', {'thread_id': '11'})
    assert len(bot.actions) == 1
    await obj.send_typing('4242', {'thread_id': '22'})
    assert [kw['message_thread_id'] for _, kw in bot.actions] == [11, 22]
