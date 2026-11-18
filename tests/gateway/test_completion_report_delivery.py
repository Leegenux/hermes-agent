"""Prepared reports become confirmed only on an exact ledgered response ACK."""
import hashlib
import pytest
from hermes_state import SessionDB
import hermes_state_runtime as rt
import hermes_state_report_obligations as ro
from gateway import delivery_ledger as ledger


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    db=SessionDB(db_path=tmp_path/'state.db')
    db.create_session('s',source='test');epoch=rt.begin_runtime_epoch(db,instance_id='owner')
    identity='["async_delegation","delivery-test",""]'
    rt.admit_session_input(db,epoch=epoch,principal_id='automation:route',session_id='s',request_id=identity,
        payload={'text':'saved','local_automation_v1':{'identity':identity,'owner':'s','route':'route'}})
    row=rt.claim_session_input(db,epoch=epoch,session_id='s')
    report=ro.create_report(db,epoch=epoch,profile_id='default',admission_id=row['admission_id'],identity=identity,generation=row['generation'],revision=0)
    def stage(conn):
        report.update(state='preparing',receipt_admission_id=row['admission_id'],
                      prepared_output={'text':'report','sha256':hashlib.sha256(b'report').hexdigest()},
                      prepared_response_sha256=hashlib.sha256(b'report').hexdigest())
        ro._save(conn,ro._REPORT+report['report_id'],report)
    db._execute_write(stage)
    rt.settle_session_input(db,epoch=epoch,admission_id=row['admission_id'],
                            generation=row['generation'],outcome='completed')
    monkeypatch.setattr(ledger,'_db_path',lambda:tmp_path/'state.db')
    ledger.record_obligation(obligation_id='oid',session_key='route',platform='discord',chat_id='42',thread_id=None,content='report')
    yield db,epoch,row,report
    db.close()


def test_only_exact_ledger_ack_confirms(prepared):
    from gateway.completion_report_delivery import bind_report_delivery
    db,epoch,row,report=prepared
    bind_report_delivery(db,epoch=epoch,admission_id=row['admission_id'],report_ids=[report['report_id']],obligation_id='oid',response='report')
    assert ro.get_report(db,report['report_id'])['state']=='preparing'
    ledger.mark_delivered('oid')
    assert ro.get_report(db,report['report_id'])['state']=='confirmed'


def test_failed_then_cancelled_ack_never_confirms(prepared):
    from gateway.completion_report_delivery import bind_report_delivery
    db,epoch,row,report=prepared
    bind_report_delivery(db,epoch=epoch,admission_id=row['admission_id'],report_ids=[report['report_id']],obligation_id='oid',response='report')
    ledger.mark_failed('oid','outage')
    assert ro.get_report(db,report['report_id'])['state']=='preparing'
    ro.cancel_reports(db,epoch=epoch,profile_id='default',session_id='s',control_id='stop')
    ledger.mark_delivered('oid')
    assert ro.get_report(db,report['report_id'])['state']=='unknown'


def test_wrong_response_cannot_bind(prepared):
    from gateway.completion_report_delivery import bind_report_delivery
    db,epoch,row,report=prepared
    with pytest.raises(rt.RuntimeStoreError):
        bind_report_delivery(db,epoch=epoch,admission_id=row['admission_id'],report_ids=[report['report_id']],obligation_id='oid',response='different')
    assert ro.get_report(db,report['report_id']).get('delivery_obligation_id') is None
