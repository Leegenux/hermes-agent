"""Completion report obligations must be fenced durably, on the same SessionDB."""
import json

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


@pytest.fixture
def owner(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('parent', source='test')
    db.create_session('other', source='test')
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    yield db, epoch
    db.close()


def completion(db, epoch, sid='parent', identity='["async_delegation","task-1",""]'):
    route = 'route:' + sid
    rt.admit_session_input(db, epoch=epoch, principal_id='automation:' + route,
        session_id=sid, request_id=identity,
        payload={'text': 'finished', 'local_automation_v1': {'identity': identity, 'owner': sid, 'route': route}})
    return rt.claim_session_input(db, epoch=epoch, session_id=sid)


def test_report_identity_replay_and_cancel_revision(owner):
    from hermes_state_report_obligations import create_report, cancel_reports, transition_report
    db, epoch = owner
    row = completion(db, epoch)
    kwargs = dict(epoch=epoch, profile_id='default', admission_id=row['admission_id'],
                  identity=row['request_id'], generation=row['generation'], revision=0)
    first = create_report(db, **kwargs)
    assert create_report(db, **kwargs) == first
    assert first['state'] == 'open' and first['continuation_request_id']
    cancel_reports(db, epoch=epoch, profile_id='default', session_id='parent')
    with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
        transition_report(db, epoch=epoch, report_id=first['report_id'], revision=0,
                          expected='open', state='queued')
    with pytest.raises(rt.RuntimeStoreError, match='stale_generation'):
        create_report(db, **kwargs)


def test_nonautomation_source_cannot_create_report(owner):
    from hermes_state_report_obligations import create_report
    db, epoch = owner
    rt.admit_session_input(db, epoch=epoch, principal_id='messaging:user', session_id='parent',
                           request_id='fake', payload={'text': 'fake'})
    row = rt.claim_session_input(db, epoch=epoch, session_id='parent')
    with pytest.raises(rt.RuntimeStoreError, match='permission_denied'):
        create_report(db, epoch=epoch, profile_id='default', admission_id=row['admission_id'],
                      identity='fake', generation=row['generation'], revision=0)
