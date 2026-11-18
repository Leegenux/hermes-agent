import json
from types import SimpleNamespace
import pytest
from hermes_state import SessionDB
import hermes_state_runtime as rt
import hermes_state_report_obligations as ro
from gateway.session_results import finish_result
from gateway.completion_report_scope import report_turn, completion_scope, prepare_response

@pytest.fixture
def setup(tmp_path):
    db=SessionDB(db_path=tmp_path/'state.db'); db.create_session('s',source='test')
    epoch=rt.begin_runtime_epoch(db,instance_id='owner'); identity='["async_delegation","task-029",""]'
    rt.admit_session_input(db,epoch=epoch,principal_id='automation:route',session_id='s',request_id=identity,
       payload={'text':'child-result','local_automation_v1':{'identity':identity,'owner':'s','route':'route'}})
    row=rt.claim_session_input(db,epoch=epoch,session_id='s')
    report=ro.create_report(db,epoch=epoch,profile_id='default',admission_id=row['admission_id'],identity=identity,generation=row['generation'],revision=0)
    def stage(conn):
        r=ro._load(conn,ro._REPORT+report['report_id']); r.update(redirected=True,explicitly_unreported=True,receipt_admission_id=row['admission_id']); ro._save(conn,ro._REPORT+report['report_id'],r)
    db._execute_write(stage)
    yield db,epoch,row,report,identity
    db.close()

def settle(db,epoch,row,guard=lambda c: None):
    return finish_result(db,epoch=epoch,row=row,response='side',outcome='completed',_continuation_authorize=guard)

def test_guard_reject_rolls_back(setup):
    db,epoch,row,report,_=setup
    def refuse(conn): raise rt.RuntimeStoreError('runtime_draining')
    with pytest.raises(rt.RuntimeStoreError,match='runtime_draining'): settle(db,epoch,row,refuse)
    assert rt.get_session_admission(db,admission_id=row['admission_id'])['status']=='started'
    assert ro.get_report(db,report['report_id'])['state']=='open'
    assert len(rt.list_session_admissions(db,session_id='s',pending_only=False))==1

def test_queue_cap_rolls_back(setup,monkeypatch):
    db,epoch,row,report,_=setup; monkeypatch.setattr(rt,'ADMISSION_QUEUE_MAX',0)
    with pytest.raises(rt.RuntimeStoreError,match='queue_full'): settle(db,epoch,row)
    assert rt.get_session_admission(db,admission_id=row['admission_id'])['status']=='started'
    assert ro.get_report(db,report['report_id'])['state']=='open'

def test_stop_first_prevents_enqueue(setup):
    db,epoch,row,report,_=setup
    ro.cancel_reports(db,epoch=epoch,profile_id='default',session_id='s',control_id='stop')
    settle(db,epoch,row)
    assert ro.get_report(db,report['report_id'])['state']=='cancelled'
    assert len(rt.list_session_admissions(db,session_id='s',pending_only=False))==1

def test_settle_replay_no_duplicate(setup):
    db,epoch,row,_report,_=setup; settle(db,epoch,row); settle(db,epoch,row)
    assert len(rt.list_session_admissions(db,session_id='s',pending_only=False))==2

@pytest.mark.asyncio
async def test_restored_continuation_scope_unreported_bounded(setup):
    db,epoch,row,report,identity=setup; settle(db,epoch,row)
    follow=rt.claim_session_input(db,epoch=epoch,session_id='s')
    a=SimpleNamespace(db=db,epoch=epoch,profile_id='default',runner=SimpleNamespace())
    event=SimpleNamespace(text=follow['payload']['text'])
    async with report_turn(a,follow,event):
        scope=completion_scope.get(); assert scope['identities']==[identity] and scope['continuation']
        scope['redirected']=True
        raw=json.dumps({'v':1,'side_response':'side','reports':[],'unreported':[identity]})
        captured={'result':{'final_response':raw}}
        assert await prepare_response(a,follow,raw,captured)=='side'
    assert ro.get_report(db,report['report_id'])['state']=='unknown'
    settle(db,epoch,follow)
    assert len(rt.list_session_admissions(db,session_id='s',pending_only=False))==2

@pytest.mark.asyncio
async def test_scope_reaches_real_gateway_worker_and_disables_stream(setup):
    import threading
    from gateway.run import GatewayRunner
    from gateway.run_turn_runner import TurnRunner
    db,epoch,row,_report,identity=setup
    runner=GatewayRunner.__new__(GatewayRunner)
    runner._get_executor=lambda: None
    a=SimpleNamespace(db=db,epoch=epoch,profile_id='default',runner=runner)
    runner._completion_report_scopes={}
    turn=TurnRunner.__new__(TurnRunner)
    main=threading.get_ident()
    async with report_turn(a,row,SimpleNamespace(text='child')):
        def worker():
            return threading.get_ident(), completion_scope.get(), turn._setup_stream_consumer('discord')
        worker_id,scope,disabled=await runner._run_in_executor_with_context(worker)
        assert worker_id != main and scope['identities']==[identity]
        assert disabled==(None,None,None,False)

@pytest.mark.asyncio
async def test_failed_turn_does_not_expose_raw_protocol(setup):
    db,epoch,row,_report,identity=setup
    a=SimpleNamespace(db=db,epoch=epoch,profile_id='default',runner=SimpleNamespace())
    # Use existing original report scope (the fixture pre-created it at the same revision).
    event=SimpleNamespace(text='child-result')
    raw=json.dumps({'v':1,'side_response':'','reports':[{'identity':identity,'text':'part'}],'unreported':[]})
    async with report_turn(a,row,event):
        result=await prepare_response(a,row,raw,{'result':{'failed':True,'final_response':raw}})
    assert result != raw

@pytest.mark.asyncio
@pytest.mark.parametrize('fault',['profile','epoch','identity','revision'])
async def test_continuation_reference_mismatch_refused(setup,fault):
    db,epoch,row,_report,_identity=setup; settle(db,epoch,row)
    follow=rt.claim_session_input(db,epoch=epoch,session_id='s')
    a=SimpleNamespace(db=db,epoch=epoch,profile_id='default',runner=SimpleNamespace())
    if fault=='profile': a.profile_id='other'
    if fault=='epoch': a.epoch=epoch+1
    if fault=='identity': follow['payload']['local_automation_v1']['identities']=['["async_delegation","wrong",""]']
    if fault=='revision': follow['payload']['completion_report_continuation_v1']['revision']+=1
    with pytest.raises(rt.RuntimeStoreError):
        async with report_turn(a,follow,SimpleNamespace(text='')): pass

@pytest.mark.asyncio
async def test_tampered_reference_rejected(setup):
    db,epoch,row,_report,_identity=setup; settle(db,epoch,row)
    follow=rt.claim_session_input(db,epoch=epoch,session_id='s')
    follow['payload']['completion_report_continuation_v1']['original_admission_id']='wrong'
    a=SimpleNamespace(db=db,epoch=epoch,profile_id='default',runner=SimpleNamespace())
    with pytest.raises(rt.RuntimeStoreError,match='permission_denied'):
        async with report_turn(a,follow,SimpleNamespace(text='')): pass
