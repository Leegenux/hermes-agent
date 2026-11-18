import asyncio
from types import SimpleNamespace
import pytest
from gateway.session_authority import SessionAuthority, LiveSession
from gateway.session_report_controls import fence_reports
from hermes_state import SessionDB
import hermes_state_runtime as rt
from tests.gateway.test_019_independent_review import make_report

@pytest.mark.asyncio
async def test_cleanup_failure_after_published_retry_pays_waiter_once(tmp_path, monkeypatch):
    import gateway.session_authority as mod
    with SessionDB(db_path=tmp_path/'db') as db:
        db.create_session('s', source='test')
        epoch=rt.begin_runtime_epoch(db, instance_id='owner')
        _,_report,queued=make_report(db,epoch)
        a=SessionAuthority(SimpleNamespace(_draining=False),profile_id='default',instance_id='owner',db=db,epoch=epoch)
        live=a.sessions['s']=LiveSession(None,'route-s')
        events=[]; original_publish=live.event_stream.publish
        def publish(*args,**kwargs):
            if len(args)>1 and args[1].get('admission_id')==queued['admission_id'] and args[1].get('outcome')=='cancelled': events.append(args[1])
            return original_publish(*args,**kwargs)
        monkeypatch.setattr(live.event_stream,'publish',publish)
        orig_drop=mod.drop_producer_turn
        calls=[]
        def drop(*args):
            calls.append(1)
            if len(calls)==1: raise RuntimeError('injected cleanup failure')
            return orig_drop(*args)
        monkeypatch.setattr(mod,'drop_producer_turn',drop)
        scheduled=[]
        monkeypatch.setattr(a,'_schedule',lambda ref: scheduled.append(ref))
        waiter=asyncio.get_running_loop().create_future()
        a.waiters[queued['admission_id']]=waiter
        a.native_waiters.add(queued['admission_id'])
        with pytest.raises(RuntimeError,match='injected cleanup failure'):
            await fence_reports(a,'s','control-021')
        assert len(events)==1 and not waiter.done()
        await fence_reports(a,'s','control-021')
        assert len(events)==1 and waiter.done() and waiter.result() is None
        assert queued['admission_id'] not in a.waiters
        assert queued['admission_id'] not in a.native_waiters
        assert len(calls)==2 and len(scheduled)==1
