"""Atomic admission inside settlement: real SessionDB, no nested transactions."""
import json
import pytest
from hermes_state import SessionDB
import hermes_state_runtime as rt

@pytest.fixture
def owned(tmp_path):
    db=SessionDB(db_path=tmp_path/'state.db'); db.create_session('s',source='test')
    epoch=rt.begin_runtime_epoch(db,instance_id='owner')
    yield db,epoch
    db.close()

def args(epoch,request='next',payload=None,session='s'):
    return dict(epoch=epoch,principal_id='producer',session_id=session,request_id=request,payload=payload or {'text':'value'})

def started(db,epoch):
    rt.admit_session_input(db,**args(epoch,'head'))
    return rt.claim_session_input(db,epoch=epoch,session_id='s')

def settle(db,epoch,row,callback):
    return rt.settle_session_input(db,epoch=epoch,admission_id=row['admission_id'],generation=row['generation'],outcome='completed',_terminal_write=callback)

def test_settlement_commits_same_connection_admission_once(owned):
    db,epoch=owned; row=started(db,epoch); receipts=[]
    def effect(conn,_row):
        receipts.append(rt._admit_session_input_on_conn(conn,**args(epoch)))
        receipts.append(rt._admit_session_input_on_conn(conn,**args(epoch)))
    settle(db,epoch,row,effect)
    assert receipts[0]==receipts[1]
    assert rt.get_session_admission(db,admission_id=row['admission_id'])['outcome']=='completed'
    rows=rt.list_session_admissions(db,session_id='s',pending_only=False)
    assert [r['request_id'] for r in rows]==['head','next']

@pytest.mark.parametrize('mode',['permission','queue_full','injected','conflict'])
def test_refusal_rolls_back_settlement_and_admission(owned,monkeypatch,mode):
    db,epoch=owned; row=started(db,epoch)
    if mode=='queue_full':monkeypatch.setattr(rt,'ADMISSION_QUEUE_MAX',0)
    if mode=='conflict':rt.admit_session_input(db,**args(epoch,'next',{'text':'original'}))
    def effect(conn,_row):
        def guard(_): raise rt.RuntimeStoreError('permission_denied')
        kw={'_authorize_write':guard} if mode=='permission' else {}
        rt._admit_session_input_on_conn(conn,**args(epoch),**kw)
        if mode=='injected':raise RuntimeError('injected rollback')
    error = RuntimeError if mode=='injected' else rt.RuntimeStoreError
    reason = {'permission':'permission_denied','queue_full':'queue_full',
              'injected':'injected rollback','conflict':'admission_conflict'}[mode]
    with pytest.raises(error,match=reason):
        settle(db,epoch,row,effect)
    assert rt.get_session_admission(db,admission_id=row['admission_id'])['status']=='started'
    found=rt.find_session_admission(db,principal_id='producer',session_id='s',request_id='next')
    assert (found is not None)==(mode=='conflict')

def test_live_replay_guards_and_conflict(owned):
    db,epoch=owned; calls=[]
    def run(conn):
        first=rt._admit_session_input_on_conn(conn,**args(epoch),_authorize_write=lambda _:calls.append('authorize'))
        replay=rt._admit_session_input_on_conn(conn,**args(epoch),_authorize_write=lambda _:calls.append('bad'),_replay_guard=lambda _:calls.append('replay'))
        assert first==replay
        with pytest.raises(rt.RuntimeStoreError,match='admission_conflict'):
            rt._admit_session_input_on_conn(conn,**args(epoch,payload={'text':'different'}))
    db._execute_write(run); assert calls==['authorize','replay']

def test_retired_tombstone_exact_retry_and_miss(owned):
    db,epoch=owned; original=rt.admit_session_input(db,**args(epoch,'old'))
    from hermes_state_terminal import ADMISSION_PREFIX,identity_key
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    def retire(conn):
        raw=dict(conn.execute('SELECT * FROM session_admissions WHERE admission_id=?',(original['admission_id'],)).fetchone())
        raw['status']='terminal';raw['outcome']='completed'
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',(ADMISSION_PREFIX+original['admission_id'],json.dumps(raw)))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',(identity_key('producer','s','old'),json.dumps(original['admission_id'])))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',(RETIRED_PREFIX+'s','true'))
        conn.execute('DELETE FROM session_admissions WHERE admission_id=?',(original['admission_id'],))
    db._execute_write(retire)
    result=db._execute_write(lambda c:rt._admit_session_input_on_conn(c,**args(epoch,'old'),
        _authorize_write=lambda _:pytest.fail('retired replay authorized as a new write'),
        _replay_guard=lambda _:pytest.fail('retired replay is not a live replay')))
    assert result['admission_id']==original['admission_id'] and result['status']=='terminal'
    assert rt.admit_session_input(db,**args(epoch,'old'))==result
    for kw,reason in [(args(epoch,'old',{'text':'changed'}),'admission_conflict'),(args(epoch,'missing'),'not_found')]:
        with pytest.raises(rt.RuntimeStoreError,match=reason):db._execute_write(lambda c:rt._admit_session_input_on_conn(c,**kw))

def test_stale_epoch_and_generation_never_run_effect(owned):
    db,epoch=owned; row=started(db,epoch); called=[]
    def effect(conn,_):called.append(True);rt._admit_session_input_on_conn(conn,**args(epoch))
    with pytest.raises(rt.RuntimeStoreError,match='stale_generation'):
        rt.settle_session_input(db,epoch=epoch,admission_id=row['admission_id'],generation=row['generation']+1,outcome='completed',_terminal_write=effect)
    rt.begin_runtime_epoch(db,instance_id='next')
    with pytest.raises(rt.RuntimeStoreError,match='stale_epoch'):settle(db,epoch,row,effect)
    with pytest.raises(rt.RuntimeStoreError,match='stale_epoch'):
        db._execute_write(lambda conn:rt._admit_session_input_on_conn(conn,**args(epoch)))
    assert not called and rt.find_session_admission(db,principal_id='producer',session_id='s',request_id='next') is None
