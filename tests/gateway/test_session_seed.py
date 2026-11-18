"""Behaviour contracts for ``gateway/session_seed.py``.

A delivery path writes into a session that only an INBOUND message would otherwise create, so the
seed's key must equal the key that inbound reply resolves to. Getting it wrong is the
"continuation amnesia" class: hermes#53414 (Discord send_message), #66172 / #141429 (cron seed key),
#67384 (cron explicit-thread target never seeded at all).
"""

import os
from pathlib import Path

import pytest

from gateway import session_seed
from gateway.config import Platform, load_gateway_config
from gateway.mirror import mirror_to_session
from gateway.session import SessionSource, SessionStore, build_session_key
from gateway.session_seed import delivery_source, ensure_destination_session

CHAT, THREAD, USER = "-1003982677050", "306", "515050268"


class _Entry:
    session_id = "sess-seeded"


class _Store:
    """SessionStore stand-in: records the source it was asked to create."""

    def __init__(self):
        self.calls = []

    def get_or_create_session(self, source, touch_activity=True):
        self.calls.append(source)
        return _Entry()


class _Adapter:
    """Live-adapter stand-in; ``build_source`` mirrors the base class contract (stamps platform)."""

    def __init__(self, platform, store=None):
        self.platform = platform
        self._session_store = store or _Store()

    def build_source(self, **kw):
        return SessionSource(platform=self.platform, **kw)


@pytest.fixture
def no_stored_chat_type(monkeypatch):
    """Isolate the per-platform table from state.db (its own contract is tested below)."""
    monkeypatch.setattr(session_seed, "stored_chat_type", lambda *a, **k: None)


def _inbound(platform, chat_id, chat_type, thread_id=None, user_id=USER, scope_id=None):
    return SessionSource(platform=platform, chat_id=chat_id, chat_name="c", chat_type=chat_type,
                         user_id=user_id, thread_id=thread_id, scope_id=scope_id)


@pytest.mark.parametrize(
    "platform,chat_id,thread_id,hint,scope_id,inbound",
    [
        # Telegram forum topic: parent chat + the supergroup's own "group" slot.
        ("telegram", CHAT, THREAD, None, None,
         dict(platform=Platform.TELEGRAM, chat_id=CHAT, chat_type="group", thread_id=THREAD)),
        # Discord thread: the thread id IS the chat id on the inbound path.
        ("discord", "123", "900", None, None,
         dict(platform=Platform.DISCORD, chat_id="900", chat_type="thread", thread_id="900")),
        # Slack thread: parent channel, team scope_id in the key.
        ("slack", "C123", "1700.1", None, "T1",
         dict(platform=Platform.SLACK, chat_id="C123", chat_type="group", thread_id="1700.1",
              scope_id="T1")),
        # Matrix DM thread: only the caller's hint carries "dm" (the table says "group").
        ("matrix", "!room", "$evt", "dm", None,
         dict(platform=Platform.MATRIX, chat_id="!room", chat_type="dm", thread_id="$evt")),
    ],
)
def test_seed_key_equals_the_inbound_key(platform, chat_id, thread_id, hint, scope_id, inbound,
                                         no_stored_chat_type):
    source = delivery_source(platform, chat_id, thread_id=thread_id, chat_type=hint,
                             scope_id=scope_id)
    assert build_session_key(source) == build_session_key(SessionSource(chat_name="c", **inbound))


def test_chat_type_of_an_existing_session_wins_over_the_table(monkeypatch):
    """Telegram DM topics: the chat's recorded "dm" (not the table's "group") is authoritative."""
    monkeypatch.setattr(session_seed, "stored_chat_type", lambda *a, **k: "dm")
    source = delivery_source("telegram", USER, thread_id="9")
    assert source.chat_type == "dm"
    assert build_session_key(source) == build_session_key(
        _inbound(Platform.TELEGRAM, USER, "dm", "9"))


def test_group_outside_a_thread_is_never_seeded():
    """A group/channel outside a thread keys per PARTICIPANT — coordinates cannot decide it."""
    adapter = _Adapter(Platform.TELEGRAM)
    assert ensure_destination_session("telegram", CHAT, chat_type="group", adapter=adapter) is None
    assert adapter._session_store.calls == []


def test_group_outside_a_thread_seeds_when_the_caller_carries_the_participant():
    """Cron holds the origin's real user_id, so its flat in_channel seed matches the reply."""
    adapter = _Adapter(Platform.TELEGRAM)
    assert ensure_destination_session(
        "telegram", CHAT, user_id=USER, chat_type="group", allow_group_key=True,
        adapter=adapter) == "sess-seeded"
    assert build_session_key(adapter._session_store.calls[-1]) == build_session_key(
        _inbound(Platform.TELEGRAM, CHAT, "group"))


def test_mirror_without_a_session_is_dropped():
    """Baseline the bug: the mirror only appends, so an unseeded destination loses the message."""
    assert mirror_to_session("telegram", CHAT, "hello", thread_id=THREAD) is False


@pytest.fixture
def live_store(monkeypatch, tmp_path):
    """The gateway's own store over a TEMP home.

    Pins ``HERMES_HOME`` and ``hermes_state.DEFAULT_DB_PATH`` so this test never reads or
    writes the developer's real ``~/.hermes`` state store.
    """
    home = tmp_path / "home"
    (home / "sessions").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db", raising=False)
    return SessionStore(home / "sessions", load_gateway_config())


def _transcript(session_id):
    from hermes_state_registry import acquire, release_or_close
    db = acquire()
    try:
        return [(m["role"], m["content"]) for m in db.get_messages(session_id)]
    finally:
        release_or_close(db)


def test_a_seeded_greeting_is_what_the_reply_will_read(live_store):
    """End to end: seed -> mirror -> the reply's inbound source resumes THAT session."""
    adapter = _Adapter(Platform.TELEGRAM, live_store)
    seeded = ensure_destination_session("telegram", CHAT, thread_id=THREAD, user_id=USER,
                                        adapter=adapter)
    assert seeded
    assert mirror_to_session("telegram", CHAT, "welcome to this topic", thread_id=THREAD,
                             user_id=USER, session_id=seeded) is True

    reply = live_store.get_or_create_session(
        _inbound(Platform.TELEGRAM, CHAT, "group", THREAD))
    assert reply.session_id == seeded
    assert [text for _role, text in _transcript(seeded)] == ["welcome to this topic"]


def test_send_message_mirror_seeds_through_the_live_adapter(live_store, monkeypatch):
    """The production entry point: send_message's mirror seeds the topic then records itself."""
    from tools import send_message_tool

    adapter = _Adapter(Platform.TELEGRAM, live_store)
    monkeypatch.setattr(send_message_tool, "_live_adapter", lambda platform: (None, adapter))
    monkeypatch.setenv("HERMES_SESSION_USER_ID", USER)

    assert send_message_tool._mirror_sent_message("telegram", CHAT, "greeting", THREAD) is True
    reply = live_store.get_or_create_session(_inbound(Platform.TELEGRAM, CHAT, "group", THREAD))
    assert [text for _role, text in _transcript(reply.session_id)] == ["greeting"]


def test_cli_send_without_a_gateway_seeds_through_the_offline_store(live_store, monkeypatch):
    """`hermes send` has no live adapter — and `_live_adapter` answers `(None, None)` there, it does
    not raise. A store-only shim must still seed the greeting."""
    from tools import send_message_tool

    monkeypatch.setattr(send_message_tool, "_live_adapter", lambda _platform: (None, None))
    monkeypatch.setattr(send_message_tool, "_offline_seed_store",
                        lambda: _Adapter(Platform.TELEGRAM, live_store))
    monkeypatch.setenv("HERMES_SESSION_USER_ID", USER)

    assert send_message_tool._mirror_sent_message("telegram", CHAT, "greeting", THREAD) is True
    reply = live_store.get_or_create_session(_inbound(Platform.TELEGRAM, CHAT, "group", THREAD))
    assert [text for _role, text in _transcript(reply.session_id)] == ["greeting"]


def test_cron_seed_lands_in_the_thread_session(live_store):
    """Cron's thread seed keeps its USER-role brief but rides the shared resolver."""
    from cron.scheduler_delivery import _seed_cron_session

    adapter = _Adapter(Platform.TELEGRAM, live_store)
    job = {"id": "job1", "name": "brief"}
    assert _seed_cron_session(
        job, adapter, "telegram", CHAT, "the brief", thread_id=THREAD, chat_type=None,
        user_id="system:cron", user_name="Cron", chat_name="chan", scope_id=None) is True
    reply = live_store.get_or_create_session(_inbound(Platform.TELEGRAM, CHAT, "group", THREAD))
    rows = _transcript(reply.session_id)
    assert [(role, text.endswith("the brief")) for role, text in rows] == [("user", True)]
