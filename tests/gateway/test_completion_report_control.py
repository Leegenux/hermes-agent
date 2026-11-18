"""Durable report fencing must be reached from the real authority interrupt."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import SessionActor, SessionRef
from hermes_state import SessionDB
from hermes_state_report_obligations import create_report, get_report
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input


@pytest.mark.asyncio
async def test_authority_interrupt_cancels_report_durably(tmp_path, monkeypatch):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='owner')
        identity = '["async_delegation","task-1",""]'
        admit_session_input(db, epoch=epoch, principal_id='automation:route', session_id='s',
            request_id=identity, payload={'text': 'done', 'local_automation_v1': {
                'identity': identity, 'owner': 's', 'route': 'route'}})
        row = claim_session_input(db, epoch=epoch, session_id='s')
        report = create_report(db, epoch=epoch, profile_id='default', admission_id=row['admission_id'],
                               identity=identity, generation=row['generation'], revision=0)
        runner = SimpleNamespace(_draining=False)
        authority = SessionAuthority(runner, profile_id='default', instance_id='owner', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(None, 'route')
        # Collaboration is an independent optional local extension. Fence behavior
        # must also be exercised on a clean Git base where this seam is absent.
        monkeypatch.setattr(authority, 'record_collab_stop', AsyncMock(return_value=False), raising=False)
        monkeypatch.setattr(authority, 'agent', lambda ref: None)
        actor = SessionActor('operator', 'default', frozenset({'session:control'}), 'test')
        await authority.interrupt(actor, SessionRef('default', 's'), row['generation'])
        assert get_report(db, report['report_id'])['state'] == 'cancelled'
