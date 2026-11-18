import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest

from gateway.session_authority import SessionAuthority, LiveSession
from gateway.session_contract import SessionActor, SessionRef
from gateway.session_report_controls import fence_reports, command_control_id
from hermes_state import SessionDB
import hermes_state_runtime as rt
import hermes_state_report_obligations as ro


def make_report(db, epoch, sid='s'):
    identity='["async_delegation","task-019",""]'
    route='route-'+sid
    rt.admit_session_input(db,epoch=epoch,principal_id='automation:'+route,session_id=sid,request_id=identity,
        payload={'text':'done','local_automation_v1':{'identity':identity,'owner':sid,'route':route}})
    row=rt.claim_session_input(db,epoch=epoch,session_id=sid)
    report=ro.create_report(db,epoch=epoch,profile_id='default',admission_id=row['admission_id'],identity=identity,generation=row['generation'],revision=0)
    queued=rt.admit_session_input(db,epoch=epoch,principal_id='automation:'+route,session_id=sid,
        request_id=report['continuation_request_id'],payload={'text':'continuation'})
    ro.transition_report(db,epoch=epoch,report_id=report['report_id'],revision=0,expected='open',state='queued',continuation_admission_id=queued['admission_id'])
    return row,report,queued

@pytest.mark.asyncio
async def test_observer_retry_after_schedule_failure_publishes_once(tmp_path, monkeypatch):
    with SessionDB(db_path=tmp_path/'db') as db:
        db.create_session('s',source='test'); epoch=rt.begin_runtime_epoch(db,instance_id='owner')
        _, _report, queued=make_report(db,epoch)
        authority=SessionAuthority(SimpleNamespace(_draining=False),profile_id='default',instance_id='owner',db=db,epoch=epoch)
        live=authority.sessions['s']=LiveSession(None,'route-s')
        published=[]; original=live.event_stream.publish
        def publish(*args,**kwargs):
            if len(args)>1 and args[1].get('admission_id')==queued['admission_id'] and args[1].get('outcome')=='cancelled': published.append(args[1])
            return original(*args,**kwargs)
        monkeypatch.setattr(live.event_stream,'publish',publish)
        calls=[]
        def schedule(ref):
            calls.append(ref)
            if len(calls)==1: raise RuntimeError('injected scheduler failure after observer publication')
        monkeypatch.setattr(authority,'_schedule',schedule)
        waiter=asyncio.get_running_loop().create_future(); authority.waiters[queued['admission_id']]=waiter
        authority.native_waiters.add(queued['admission_id'])
        with pytest.raises(RuntimeError,match='scheduler failure'):
            await fence_reports(authority,'s','control-1')
        assert waiter.done() and waiter.result() is None
        assert queued['admission_id'] not in authority.native_waiters
        await fence_reports(authority,'s','control-1')
        assert len(published)==1, f'duplicate completion events: {len(published)}'

@pytest.mark.asyncio
async def test_control_storage_failure_rolls_back_all(tmp_path, monkeypatch):
    with SessionDB(db_path=tmp_path/'db') as db:
        db.create_session('s',source='test'); epoch=rt.begin_runtime_epoch(db,instance_id='owner')
        _, report, queued=make_report(db,epoch)
        authority=SessionAuthority(SimpleNamespace(_draining=False),profile_id='default',instance_id='owner',db=db,epoch=epoch)
        authority.sessions['s']=LiveSession(None,'route-s')
        old=ro._save
        def fail(conn,key,value):
            old(conn,key,value)
            if key.startswith(ro._REPORT): raise RuntimeError('injected before commit')
        monkeypatch.setattr(ro,'_save',fail)
        with pytest.raises(RuntimeError,match='injected before commit'): await fence_reports(authority,'s','stop-1')
        assert ro.get_report(db,report['report_id'])['state']=='queued'
        assert rt.get_session_admission(db,admission_id=queued['admission_id'])['status']=='queued'
        with db._read_ctx() as conn: assert ro.control_revision_on_conn(conn,profile_id='default',session_id='s')==0


@pytest.mark.asyncio
async def test_control_replay_reconstructed_authority_and_other_session(tmp_path, monkeypatch):
    with SessionDB(db_path=tmp_path/'db') as db:
        db.create_session('s',source='test'); db.create_session('other',source='test')
        epoch=rt.begin_runtime_epoch(db,instance_id='owner')
        _, _report, _queued=make_report(db,epoch)
        _, other, _=make_report(db,epoch,'other')
        a=SessionAuthority(SimpleNamespace(_draining=False),profile_id='default',instance_id='owner',db=db,epoch=epoch)
        a.sessions['s']=LiveSession(None,'route-s')
        monkeypatch.setattr(a,'_schedule',lambda ref: None)
        d=await fence_reports(a,'s','one')
        b=SessionAuthority(SimpleNamespace(_draining=False),profile_id='default',instance_id='owner',db=db,epoch=epoch)
        b.sessions['s']=LiveSession(None,'route-s')
        monkeypatch.setattr(b,'_schedule',lambda ref: None)
        replay=await fence_reports(b,'s','one')
        assert d['revision']==replay['revision']==1
        assert ro.get_report(db,other['report_id'])['state']=='queued'
        with db._read_ctx() as conn: assert ro.control_revision_on_conn(conn,profile_id='default',session_id='s')==1

@pytest.mark.asyncio
async def test_interrupt_unauthorized_and_stale_do_not_fence(tmp_path):
    with SessionDB(db_path=tmp_path/'db') as db:
        db.create_session('s',source='test'); epoch=rt.begin_runtime_epoch(db,instance_id='owner')
        row,report,_=make_report(db,epoch)
        a=SessionAuthority(SimpleNamespace(_draining=False),profile_id='default',instance_id='owner',db=db,epoch=epoch)
        a.sessions['s']=LiveSession(None,'route-s')
        ref=SessionRef('default','s')
        actor=SessionActor('operator','default',frozenset(), 'test')
        with pytest.raises(rt.RuntimeStoreError): await a.interrupt(actor,ref,row['generation'])
        actor=SessionActor('operator','default',frozenset({'session:control'}), 'test')
        with pytest.raises(rt.RuntimeStoreError,match='stale_generation'): await a.interrupt(actor,ref,row['generation']+1)
        assert ro.get_report(db,report['report_id'])['state']=='queued'
        with db._read_ctx() as conn: assert ro.control_revision_on_conn(conn,profile_id='default',session_id='s')==0


def test_claimed_continuation_is_not_misclassified_as_cancelled(tmp_path):
    with SessionDB(db_path=tmp_path/'db') as db:
        db.create_session('s',source='test'); epoch=rt.begin_runtime_epoch(db,instance_id='owner')
        original,report,queued=make_report(db,epoch)
        rt.settle_session_input(db,epoch=epoch,admission_id=original['admission_id'],generation=original['generation'],outcome='completed')
        started=rt.claim_session_input(db,epoch=epoch,session_id='s')
        assert started['admission_id']==queued['admission_id'] and started['status']=='started'
        ro.cancel_reports(db,epoch=epoch,profile_id='default',session_id='s',control_id='stop')
        assert rt.get_session_admission(db,admission_id=queued['admission_id'])['status']=='started'
        assert ro.get_report(db,report['report_id'])['state']!='cancelled'


def test_event_identity_stable_same_message():
    def ev(mid): return SimpleNamespace(source=SimpleNamespace(platform=SimpleNamespace(value='discord'),chat_id='42',thread_id='1'),message_id=mid,get_command=lambda:'stop')
    a,b=ev('123'),ev('123')
    assert command_control_id(a)==command_control_id(a)==command_control_id(b)
    assert command_control_id(a)!=command_control_id(ev('124'))
