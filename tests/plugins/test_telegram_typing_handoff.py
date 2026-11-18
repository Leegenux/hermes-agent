"""Typing indicators must survive the reconnect that replaces the adapter mid-turn."""

import asyncio
from unittest.mock import AsyncMock, Mock


def _adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    return object.__new__(TelegramAdapter)


def _live_adapter():
    live = _adapter()
    live._bot = Mock()
    live._telegram_typing_cooldown_until = {}
    live._bot.send_chat_action = AsyncMock()
    live._typing_in_cooldown = lambda chat_id: False
    live._metadata_thread_id = lambda metadata: None
    live._message_thread_id_for_typing = lambda thread_id: thread_id
    live.config = Mock(typing_indicator=True)
    return live


def test_send_typing_forwards_to_replacement_when_bot_is_gone():
    """The refresh loop keeps calling the old instance; it must reach the live adapter."""
    old, live = _adapter(), _live_adapter()
    old._bot = None
    old._replacement_telegram_adapter = lambda: live

    asyncio.run(old.send_typing("42", {"thread_id": None}))

    live._bot.send_chat_action.assert_awaited_once()


def test_send_typing_stays_silent_without_bot_or_replacement():
    old = _adapter()
    old._bot = None
    old._replacement_telegram_adapter = lambda: None

    asyncio.run(old.send_typing("42", None))  # must not raise
