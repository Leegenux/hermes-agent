"""Display names may be absent on restored sources; routing and connector checks still apply."""
from dataclasses import replace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_ingress_context import native_callback, register_transport_home
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from plugins.platforms.discord.adapter import DiscordAdapter


async def _fixture(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setattr(GatewayRunner, '_VOICE_MODE_PATH', tmp_path / 'gateway_voice_mode.json')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    db = SessionDB(tmp_path / 'state.db')
    runner.session_store._db = db
    runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='name-fixture', db=db)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42',
                                  chat_name='Original chat', user_name='Original user')
    human = MessageEvent(text='human', source=source, message_id='human')
    with native_callback(runner, human, tmp_path):
        await authority.admit_native(human)
    route = runner.session_store._generate_session_key(source)
    event = MessageEvent(text='completed result', source=source, internal=True,
                         metadata={'gateway_session_key': route,
                                   'gateway_session_id': runner.session_store.peek_session_id(route)})
    return authority, adapter, event


@pytest.mark.asyncio
@pytest.mark.parametrize('names', [dict(chat_name=None, user_name=None),
                                  dict(chat_name='Renamed chat', user_name='Renamed user')])
async def test_completion_with_missing_or_changed_display_names_is_admitted(tmp_path, monkeypatch, names):
    authority, adapter, event = await _fixture(tmp_path, monkeypatch)
    try:
        event.source = replace(event.source, **names)
        first = await authority.admit_automation(adapter, event, 'completion-1')
        retry = await authority.admit_automation(adapter, event, 'completion-1')
        assert retry.admission_id == first.admission_id
        row = get_session_admission(authority.db, admission_id=first.admission_id)
        assert row['payload']['text'] == 'completed result'
        assert row['status'] == 'queued'
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [dict(chat_id='other'), dict(thread_id='other')])
async def test_completion_with_wrong_route_is_refused(tmp_path, monkeypatch, changes):
    authority, adapter, event = await _fixture(tmp_path, monkeypatch)
    try:
        event.source = replace(event.source, **changes)
        with pytest.raises(RuntimeStoreError):
            await authority.admit_automation(adapter, event, 'completion-1')
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [dict(user_id='other'), dict(scope_id='other'), dict(profile='other')])
async def test_completion_source_drift_cannot_replace_the_committed_source(tmp_path, monkeypatch, changes):
    """The producer's source only names the route; the wake still runs as the committed human."""
    authority, adapter, event = await _fixture(tmp_path, monkeypatch)
    try:
        committed = event.source
        event.source = replace(event.source, **changes)
        admission = await authority.admit_automation(adapter, event, 'completion-1')
        stored = get_session_admission(authority.db, admission_id=admission.admission_id)['payload']['native_text_v1']
        assert stored['source']['user_id'] == committed.user_id
        for key, value in changes.items():
            assert stored['source'].get(key) != value
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_completion_cannot_borrow_another_adapter(tmp_path, monkeypatch):
    authority, _adapter, event = await _fixture(tmp_path, monkeypatch)
    try:
        other = DiscordAdapter(PlatformConfig(enabled=True, token='other-fixture', typing_indicator=False))
        with pytest.raises(RuntimeStoreError):
            await authority.admit_automation(other, event, 'completion-1')
    finally:
        authority.db.close()
