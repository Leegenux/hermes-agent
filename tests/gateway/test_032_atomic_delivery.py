import hashlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from tests.gateway.test_completion_report_delivery import prepared
from tests.gateway.test_032_safety_guards import adapter_and_event
from gateway import delivery_ledger as ledger
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
import hermes_state_report_obligations as ro
import gateway.completion_report_delivery as delivery


@pytest.mark.asyncio
async def test_atomic_post_bind_fault_leaves_no_recoverable_row(prepared,monkeypatch):
    db,epoch,row,report=prepared
    adapter,event=adapter_and_event(monkeypatch,db,epoch,row,report)
    network=AsyncMock(return_value=SendResult(success=True));monkeypatch.setattr(adapter,'send',network)
    save=delivery._save
    def fail(conn,key,value):
        save(conn,key,value)
        raise RuntimeError('post-bind fault')
    monkeypatch.setattr(delivery,'_save',fail)
    with pytest.raises(RuntimeError,match='post-bind fault'):
        await adapter.send_final_ledgered(event,'route','report',{},reply_to=None)
    oid=ledger.compute_obligation_id('route','m','report')
    with db._read_ctx() as conn:
        assert conn.execute('select 1 from delivery_obligations where obligation_id=?',(oid,)).fetchone() is None
    assert ro.get_report(db,report['report_id']).get('delivery_obligation_id') is None
    assert network.await_count==0


@pytest.mark.asyncio
async def test_format_failure_recovers_full_tail_on_native_path(prepared,monkeypatch):
    db,epoch,row,report=prepared
    response='A'*3500+'END-MUST-ARRIVE'
    def stage(conn):
        saved=ro._load(conn,ro._REPORT+report['report_id'])
        saved.update(prepared_response_sha256=hashlib.sha256(response.encode()).hexdigest(),
                     prepared_output={'text':response,'sha256':hashlib.sha256(response.encode()).hexdigest()})
        ro._save(conn,ro._REPORT+report['report_id'],saved)
    db._execute_write(stage)
    adapter,event=adapter_and_event(monkeypatch,db,epoch,row,report)
    sent=[]
    async def send(chat_id,content,reply_to=None,metadata=None):
        sent.append(content)
        return SendResult(success=False,error='formatting error') if len(sent)==1 else SendResult(success=True,message_id='full')
    monkeypatch.setattr(adapter,'send',send)
    result,_=await adapter.send_final_ledgered(event,'route',response,{},reply_to=None)
    assert not result.success and len(sent)==1
    assert ro.get_report(db,report['report_id'])['state']=='preparing'
    claims=ledger.sweep_failed_for_runtime('discord',now=time.time()+60,profile='default')
    assert len(claims)==1
    runner=SimpleNamespace(_obligation_adapter=AsyncMock(return_value=adapter),
                           _arm_flood_timers_for_waiting_rows=AsyncMock())
    assert await GatewayRunner._redeliver_claimed_obligations(runner,claims)==1
    assert len(sent)==2 and sent[1].endswith(response)
    assert ro.get_report(db,report['report_id'])['state']=='confirmed'


@pytest.mark.asyncio
async def test_stop_blocks_already_claimed_native_retry(prepared,monkeypatch):
    db,epoch,row,report=prepared
    adapter,_event=adapter_and_event(monkeypatch,db,epoch,row,report)
    delivery.bind_report_delivery(db,epoch=epoch,admission_id=row['admission_id'],report_ids=[report['report_id']],obligation_id='oid',response='report')
    ledger.mark_failed('oid','network outage')
    claims=ledger.sweep_failed_for_runtime('discord',now=time.time()+60,profile='default')
    assert len(claims)==1
    ro.cancel_reports(db,epoch=epoch,profile_id='default',session_id='s',control_id='stop')
    network=AsyncMock(return_value=SendResult(success=True));monkeypatch.setattr(adapter,'send',network)
    runner=SimpleNamespace(_obligation_adapter=AsyncMock(return_value=adapter),
                           _arm_flood_timers_for_waiting_rows=AsyncMock())
    assert await GatewayRunner._redeliver_claimed_obligations(runner,claims)==0
    assert network.await_count==0
    with db._read_ctx() as conn:
        assert conn.execute("select state from delivery_obligations where obligation_id='oid'").fetchone()[0]=='abandoned'
    assert ro.get_report(db,report['report_id'])['state']=='unknown'
