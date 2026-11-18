"""Per-chat gate for Telegram typing keep-alives.

``_keep_typing`` runs once per session, but Telegram meters per chat: every DM topic shares one
chat id, so N active topics sent N ``sendChatAction`` calls per tick into one chat. Typing honors
hard flood windows and briefly yields to send/edit pacing, but overdue work must not stay hidden.
Typing never takes that slot itself: claiming it froze a group's streamed preview on its first chunk.
"""

from __future__ import annotations

import asyncio
from typing import Any

# A single/two-topic chat retains its normal budget; >=3 topics share a faster bounded budget.
# Refreshers check admission every 0.5s without doing I/O when the gate is closed.
TYPING_CHAT_MIN_INTERVAL_SECS = 2.0
TYPING_BUSY_CHAT_MIN_INTERVAL_SECS = 1.0
TYPING_TOPIC_IDLE_SECS = 2.5
TYPING_SOFT_SLOT_YIELD_SECS = 3.0


def _typing_in_cooldown(adapter: Any, chat_id: Any) -> bool:
    """A transient typing failure (``_record_typing_cooldown``) pauses this chat's typing calls."""
    until_by_chat: dict[str, float] = adapter.__dict__.setdefault("_telegram_typing_cooldown_until", {})
    until = until_by_chat.get(str(chat_id))
    if until is None:
        return False
    if asyncio.get_running_loop().time() < until:
        return True
    until_by_chat.pop(str(chat_id), None)
    return False


def _waiting_turn(adapter: Any, key: str, metadata, now: float) -> tuple[dict, str | None]:
    """Least-recently-served admission, driven by each topic's existing refresher.

    Waiting topics keep their position when they retry. Paused or vanished refreshers cannot
    block the head forever; all queue mutation happens on the adapter's event loop.
    """
    queues: dict = adapter.__dict__.setdefault('_telegram_typing_waiters', {})
    for chat, waiting in list(queues.items()):
        for topic, stamp in list(waiting.items()):
            seen = stamp[0]
            if len(stamp) == 2:  # Adapters may retain the previous gate's in-memory queue.
                waiting[topic] = (*stamp, seen)
            topic_metadata = {'thread_id': topic} if topic is not None else None
            if now - seen > TYPING_TOPIC_IDLE_SECS or adapter.typing_is_paused(chat, topic_metadata):
                waiting.pop(topic)
        if not waiting:
            queues.pop(chat)
    topic = adapter._metadata_thread_id(metadata)
    waiting = queues.setdefault(key, {})
    previous = waiting.get(topic, (now, float('-inf'), now))
    first_seen = previous[2] if len(previous) > 2 else previous[0]
    waiting[topic] = (now, previous[1], first_seen)
    return waiting, topic


def forget_typing_waiter(adapter: Any, chat_id: Any, metadata=None) -> None:
    """Remove just a finished topic, without resetting the chat's rate-limit timestamp."""
    from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id
    key = str(normalize_telegram_chat_id(chat_id))
    queues = adapter.__dict__.get('_telegram_typing_waiters', {})
    waiting = queues.get(key)
    if waiting is None:
        return
    topic = adapter._metadata_thread_id(metadata)
    waiting.pop(topic, None)
    if not waiting:
        queues.pop(key, None)


def typing_may_fire(adapter: Any, chat_id: Any, metadata=None) -> bool:
    """Reserve one shared chat tick, fairly among active topic refreshers."""
    from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id
    key = str(normalize_telegram_chat_id(chat_id))
    if adapter.typing_is_paused(chat_id, metadata):
        forget_typing_waiter(adapter, chat_id, metadata)
        return False
    now = asyncio.get_running_loop().time()
    waiting, topic = _waiting_turn(adapter, key, metadata, now)
    if _typing_in_cooldown(adapter, chat_id) or adapter._send_flood_cooldown_remaining(chat_id) is not None:
        return False
    last: dict[str, float] = adapter.__dict__.setdefault('_telegram_typing_last_sent', {})
    # A fixed 2s/chat budget makes three active topics lapse (>6s each). Keep the usual
    # budget for one/two topics, but admit up to one action/s while three or more compete.
    interval = TYPING_BUSY_CHAT_MIN_INTERVAL_SECS if len(waiting) >= 3 else TYPING_CHAT_MIN_INTERVAL_SECS
    # New work gets finite initial priority, not -inf forever: repeatedly opening
    # short topics must not take every slot from a continuously working sibling.
    first = min(waiting, key=lambda candidate: max(
        waiting[candidate][1], waiting[candidate][2] - TYPING_SOFT_SLOT_YIELD_SECS))
    if now - last.get(key, float('-inf')) < interval or first != topic:
        return False
    # Send/edit pacing is a soft courtesy, not a reason to hide work indefinitely.
    # An initial pulse is prompt; an already-served topic yields for at most 3s.
    # Fairness, chat typing rate and both hard cooldowns above still apply. Never
    # reserve/reset the send/edit slot: that froze previews in upstream #152305.
    if (adapter._chat_outbound_slot_remaining(chat_id) > 0
            and now - waiting[topic][1] < TYPING_SOFT_SLOT_YIELD_SECS):
        return False
    waiting[topic] = (now, now, waiting[topic][2])
    for stale in [k for k, t in last.items() if now - t >= TYPING_CHAT_MIN_INTERVAL_SECS]:
        del last[stale]
    last[key] = now
    return True
