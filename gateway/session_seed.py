"""Outbound delivery → session seeding: give a delivery path the session a reply will look in.

An outbound message reaches the transcript through ``gateway.mirror.mirror_to_session``, which only
APPENDS; a session is created by the first INBOUND message. So anything an agent sent into a
destination nobody has spoken in yet — a greeting posted into a fresh Telegram topic, a cron brief
into a new thread, a webhook delivery into a quiet chat — was dropped from history, and the reply
that followed started from an empty transcript.

A hand-built ``SessionSource`` is the second half of the same bug: the source must be keyed EXACTLY
as the adapter's own inbound path keys that destination, or the seed lands in an orphan session the
reply never opens. Coordinate normalization differs per platform (Discord uses the thread id AS the
chat id; Telegram/Slack/Matrix key on the parent chat) — so it lives here once, next to the adapters,
instead of in a table duplicated by each delivery path (hermes#53414, #66172, #141429, #67384).

Callers: ``tools/send_message_tool._mirror_sent_message``, ``cron/scheduler_delivery._seed_cron_session``,
``gateway/platforms/webhook._mirror_delivery``.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Platforms whose in-thread inbound messages key with the PARENT chat's own chat_type instead of
# "thread". Fallback only: a session that already exists for the chat answers this exactly.
_THREAD_REPLY_CHAT_TYPE = {"slack": "group", "matrix": "group", "telegram": "group"}

# Platforms whose inbound path keys an in-thread message with the THREAD id as chat_id.
_THREAD_KEYS_CHAT_ID = {"discord"}


def stored_chat_type(platform_name: str, chat_id: str) -> Optional[str]:
    """chat_type of any session already recorded for this chat, ignoring the thread.

    The chat type belongs to the CHAT, not the thread, so an existing session answers the
    platform question authoritatively — that is what keeps a Matrix DM room or a Slack DM thread
    ("dm") from being seeded under a table's "group"."""
    try:
        from hermes_state_registry import acquire, release_or_close
        db = acquire()
        try:
            session_id = db.find_session_by_origin(platform=str(platform_name), chat_id=str(chat_id))
            row = db.get_session(session_id) if session_id else None
        finally:
            release_or_close(db)
        return str((row or {}).get("chat_type") or "").strip() or None
    except Exception as e:
        logger.debug("seed: chat_type lookup failed for %s:%s: %s", platform_name, chat_id, e)
        return None


def delivery_chat_type(platform_name: str, chat_id: str, thread_id: Optional[str] = None,
                       hint: Optional[str] = None) -> Optional[str]:
    """chat_type the adapter's inbound path stamps on a message in this destination.

    ``hint`` wins: a caller holding the origin conversation (cron) knows it authoritatively.
    Then the chat's own existing session. Then, for a thread, the per-platform table — a thread
    implies its parent exists, so the table's default only risks an orphan row, never a wrong write
    into someone else's session. ``None`` for a chat we know nothing about, where dm-versus-group
    decides the key: never guess that one."""
    if hint:
        return str(hint)
    stored = stored_chat_type(platform_name, chat_id)
    if stored:
        return stored
    if thread_id:
        return _THREAD_REPLY_CHAT_TYPE.get(str(platform_name).lower(), "thread")
    return None


def delivery_source(platform_name: str, chat_id: str, *, thread_id: Optional[str] = None,
                    user_id: Optional[str] = None, user_name: Optional[str] = None,
                    chat_name: Optional[str] = None, scope_id: Optional[str] = None,
                    chat_type: Optional[str] = None, adapter=None):
    """``SessionSource`` an inbound message in this exact destination resolves to, or None.

    ``adapter`` (the live platform adapter) also stamps ``source.profile`` through the configured
    routes and keeps the platform's own chat-name/normalization behaviour; without one the source is
    built bare (CLI, tests)."""
    from gateway.config import Platform
    from gateway.session import SessionSource
    try:
        platform = Platform(str(platform_name).lower())
    except (ValueError, KeyError):
        logger.debug("seed: unknown platform %s", platform_name)
        return None
    # Discord threads are channels of their own: its inbound path uses the thread id as chat_id,
    # while a send_message target names the parent channel.
    keys_thread_as_chat = str(platform_name).lower() in _THREAD_KEYS_CHAT_ID
    seed_chat_id = str(thread_id) if (thread_id and keys_thread_as_chat) else str(chat_id)
    resolved = delivery_chat_type(platform_name, seed_chat_id, thread_id=thread_id, hint=chat_type)
    if resolved is None:
        logger.debug("seed: chat_type unknown for %s:%s (no session yet) — not guessing dm/group",
                     platform_name, chat_id)
        return None
    fields = dict(
        chat_id=seed_chat_id, chat_name=chat_name, chat_type=resolved,
        user_id=None if user_id in (None, "") else str(user_id), user_name=user_name,
        thread_id=str(thread_id) if thread_id else None,
        scope_id=str(scope_id) if scope_id else None,
        parent_chat_id=str(chat_id) if seed_chat_id != str(chat_id) else None)
    if adapter is not None:
        build = getattr(adapter, "build_source", None)
        if build is not None:
            return build(**fields)
    return SessionSource(platform=platform, **fields)


def ensure_destination_session(platform_name: str, chat_id: str, *, thread_id: Optional[str] = None,
                              user_id: Optional[str] = None, user_name: Optional[str] = None,
                              chat_name: Optional[str] = None, scope_id: Optional[str] = None,
                              chat_type: Optional[str] = None, allow_group_key: bool = False,
                              adapter=None) -> Optional[str]:
    """Session id for a delivery destination, CREATED if the first reply has not happened yet.

    Returns None (caller keeps its previous best-effort/no-op behaviour) when there is no live
    adapter store, when the destination's key cannot be derived (see ``delivery_chat_type``), or
    when the key embeds the PARTICIPANT — a group/channel OUTSIDE a thread is keyed per user, so
    only the inbound message can say whose session the reply will open. Pass ``allow_group_key``
    only when the caller holds that participant (cron carries the origin's real user_id)."""
    store = getattr(adapter, "_session_store", None) if adapter is not None else None
    if store is None:
        return None
    source = delivery_source(
        platform_name, chat_id, thread_id=thread_id, user_id=user_id, user_name=user_name,
        chat_name=chat_name, scope_id=scope_id, chat_type=chat_type, adapter=adapter)
    if source is None:
        return None
    if source.thread_id is None and str(source.chat_type) in ("group", "channel") and not allow_group_key:
        logger.debug("seed: %s:%s is a %s outside a thread — participant-keyed, not seeding",
                     platform_name, chat_id, source.chat_type)
        return None
    try:
        return getattr(store.get_or_create_session(source, touch_activity=False), "session_id", None)
    except Exception as e:
        logger.warning("seed: creating session for %s:%s thread=%s failed: %s",
                       platform_name, chat_id, thread_id, e)
        return None
