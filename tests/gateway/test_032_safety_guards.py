import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from tests.gateway.test_completion_report_delivery import prepared
from gateway import delivery_ledger as ledger
from gateway.platforms.event import MessageEvent
from gateway.platforms.base import SendResult
from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
import hermes_state_report_obligations as ro


def adapter_and_event(monkeypatch,db,epoch,row,report):
    adapter=DiscordAdapter(PlatformConfig(enabled=True,token='fixture',typing_indicator=False))
    monkeypatch.setattr(adapter,'_final_delivery_adapter',lambda source:adapter)
    monkeypatch.setattr(ledger,'ledger_enabled',lambda:True)
    source=adapter.build_source(chat_id='42',chat_type='dm',user_id='42')
    event=MessageEvent(text='internal notice',source=source,message_id='m')
    event._completion_report_delivery={'authority':SimpleNamespace(db=db,epoch=epoch),'admission_id':row['admission_id'],'report_ids':[report['report_id']]}
    return adapter,event

@pytest.mark.asyncio
async def test_fallback_partial_text_must_not_confirm(prepared,monkeypatch):
    db,epoch,row,report=prepared
    response='A'*3500+'END-MUST-ARRIVE'
    def stage(conn):
        x=ro._load(conn,ro._REPORT+report['report_id'])
        x['prepared_response_sha256']=hashlib.sha256(response.encode()).hexdigest()
        x['prepared_output']={'text':response,'sha256':hashlib.sha256(response.encode()).hexdigest()}
        ro._save(conn,ro._REPORT+report['report_id'],x)
    db._execute_write(stage)
    adapter,event=adapter_and_event(monkeypatch,db,epoch,row,report)
    sent=[]
    async def send(chat_id,content,reply_to=None,metadata=None):
        sent.append(content)
        return SendResult(success=False,error='formatting error') if len(sent)==1 else SendResult(success=True,message_id='fallback')
    monkeypatch.setattr(adapter,'send',send)
    result,_=await adapter.send_final_ledgered(event,'route',response,{},reply_to=None)
    assert not result.success and len(sent)==1 and 'END-MUST-ARRIVE' in sent[0]
    assert ro.get_report(db,report['report_id'])['state']!='confirmed', 'partial fallback falsely confirmed'

@pytest.mark.asyncio
async def test_failed_binding_must_not_leave_boot_recoverable_row(prepared,monkeypatch):
    import gateway.completion_report_delivery as delivery
    db,epoch,row,report=prepared
    adapter,event=adapter_and_event(monkeypatch,db,epoch,row,report)
    async_send=AsyncMock(return_value=SendResult(success=True,message_id='m'))
    monkeypatch.setattr(adapter,'send',async_send)
    def reject(*args,**kwargs): raise RuntimeError('injected binding failure')
    monkeypatch.setattr(delivery,'bind_report_delivery',reject)
    with pytest.raises(RuntimeError,match='binding failure'):
        await adapter.send_final_ledgered(event,'route','report',{},reply_to=None)
    assert async_send.await_count==0
    monkeypatch.setattr(ledger,'_owner_alive',lambda *args:False)
    claimed=ledger.sweep_recoverable(deliverable_platforms={'discord'})
    oid=ledger.compute_obligation_id('route','m','report')
    assert not [x for x in claimed if x['obligation_id']==oid], 'unbound unsent text is eligible for boot recovery'


@pytest.mark.asyncio
async def test_stop_during_send_ack_does_not_confirm(prepared,monkeypatch):
    import asyncio
    db,epoch,row,report=prepared
    adapter,event=adapter_and_event(monkeypatch,db,epoch,row,report)
    entered=asyncio.Event(); release=asyncio.Event()
    async def send(chat_id,content,reply_to=None,metadata=None):
        entered.set(); await release.wait()
        return SendResult(success=True,message_id='m')
    monkeypatch.setattr(adapter,'send',send)
    task=asyncio.create_task(adapter.send_final_ledgered(event,'route','report',{},reply_to=None))
    await asyncio.wait_for(entered.wait(),5)
    ro.cancel_reports(db,epoch=epoch,profile_id='default',session_id='s',control_id='stop')
    release.set(); await task
    assert ro.get_report(db,report['report_id'])['state']=='unknown'


@pytest.mark.asyncio
@pytest.mark.parametrize('flag',['failed','interrupted'])
async def test_failed_real_execute_admission_filters_raw(tmp_path,monkeypatch,flag):
    from gateway.session_authority import SessionAuthority,LiveSession
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import execute_admission
    from gateway.session_results import execution_result
    import gateway.session_completion_notes as notes
    import gateway.session_automation as automation
    import gateway.session_policy as policy
    import gateway.session_managed_worker as managed
    import hermes_state_runtime as rt
    from hermes_state import SessionDB
    adapter=DiscordAdapter(PlatformConfig(enabled=True,token='fixture',typing_indicator=False))
    source=adapter.build_source(chat_id='42',chat_type='dm',user_id='42')
    ident='["async_delegation","task-031",""]'; raw='RAW_UNTRUSTED MEDIA:payload'
    with SessionDB(db_path=tmp_path/'real.db') as db:
        db.create_session('s',source='test'); epoch=rt.begin_runtime_epoch(db,instance_id='owner')
        rt.admit_session_input(db,epoch=epoch,principal_id='automation:route',session_id='s',request_id=ident,payload={'text':'done','local_automation_v1':{'identity':ident,'owner':'s','route':'route'}})
        row=rt.claim_session_input(db,epoch=epoch,session_id='s')
        async def handle(event):
            execution_result.get()['result']={'final_response':raw,'messages':[],flag:True}
            return raw
        runner=SimpleNamespace(_draining=False,config=SimpleNamespace(multiplex_profiles=False),_handle_message=handle,_adapter_for_source=lambda s: adapter)
        authority=SessionAuthority(runner,profile_id='default',instance_id='owner',db=db,epoch=epoch)
        authority.sessions['s']=LiveSession(source,'route')
        monkeypatch.setattr(notes,'note_if_unviewed',AsyncMock(return_value=None))
        monkeypatch.setattr(policy,'policy_for_source',lambda *a:None)
        monkeypatch.setattr(managed,'managed_policy',lambda *a:None)
        monkeypatch.setattr(automation,'restore_local_automation',lambda a,r,row:MessageEvent(text=row['payload']['text'],source=source,internal=True,message_id=row['admission_id']))
        response=await execute_admission(authority,SessionRef('default','s'),row)
        assert not response or 'RAW_UNTRUSTED' not in response
        assert not response or 'MEDIA:' not in response
        assert not authority.pending_deliveries or all('RAW_UNTRUSTED' not in d[3] for d in authority.pending_deliveries.values())


def test_ack_transaction_rolls_back_both(prepared,monkeypatch):
    import gateway.completion_report_delivery as delivery
    db,epoch,row,report=prepared
    delivery.bind_report_delivery(db,epoch=epoch,admission_id=row['admission_id'],report_ids=[report['report_id']],obligation_id='oid',response='report')
    original=delivery._save
    def fail(conn,key,value):
        original(conn,key,value)
        raise RuntimeError('injected ACK transaction failure')
    monkeypatch.setattr(delivery,'_save',fail)
    with pytest.raises(RuntimeError,match='ACK transaction failure'): ledger.mark_delivered('oid')
    with db._read_ctx() as conn:
        state=conn.execute('select state from delivery_obligations where obligation_id=?',('oid',)).fetchone()[0]
    assert state!='delivered'
    assert ro.get_report(db,report['report_id'])['state']=='preparing'
