"""Atomic continuation enrollment when an original completion settles."""
import pytest
from hermes_state import SessionDB
import hermes_state_runtime as rt
from hermes_state_report_obligations import create_report, _load, _save, _REPORT
from gateway.session_results import finish_result


def test_settlement_enqueues_explicitly_unreported_redirect(tmp_path):
    db = SessionDB(db_path=tmp_path/'state.db')
    db.create_session('parent', source='test')
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    identity = '["async_delegation","task-1",""]'
    rt.admit_session_input(db, epoch=epoch, principal_id='automation:route', session_id='parent',
        request_id=identity, payload={'text':'finished', 'local_automation_v1':
        {'identity':identity, 'owner':'parent', 'route':'route'}})
    row = rt.claim_session_input(db, epoch=epoch, session_id='parent')
    report = create_report(db, epoch=epoch, profile_id='default', admission_id=row['admission_id'],
        identity=identity, generation=row['generation'], revision=0)
    def stage(conn):
        value = _load(conn, _REPORT + report['report_id'])
        value.update(redirected=True, explicitly_unreported=True, receipt_admission_id=row['admission_id'])
        _save(conn, _REPORT + report['report_id'], value)
    db._execute_write(stage)
    finish_result(db, epoch=epoch, row=row, response='side', outcome='completed',
                  _continuation_authorize=lambda conn: None)
    with db._read_ctx() as conn:
        value = _load(conn, _REPORT + report['report_id'])
    assert value['state'] == 'queued'
    following = rt.list_session_admissions(db, session_id='parent')
    assert len(following) == 1
    assert following[0]['admission_id'] == value['continuation_admission_id']
    assert following[0]['request_id'] == report['continuation_request_id']
    db.close()
