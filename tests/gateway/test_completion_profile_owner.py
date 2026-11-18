import pytest
from hermes_state import SessionDB
import hermes_state_runtime as rt
from hermes_state_report_obligations import create_report


@pytest.mark.parametrize('owner_matches',[True,False])
def test_native_absolute_owner_must_match_database_home(tmp_path,owner_matches):
    with SessionDB(db_path=tmp_path/'state.db') as db:
        db.create_session('s',source='test');epoch=rt.begin_runtime_epoch(db,instance_id='owner')
        identity='["async_delegation","owner-test",""]'
        rt.admit_session_input(db,epoch=epoch,principal_id='automation:route',session_id='s',request_id=identity,
            payload={'text':'saved','native_text_v1':{'route':'route','source':{'profile':None},
                'automation':{'owner':'s','identity':identity}}})
        row=rt.claim_session_input(db,epoch=epoch,session_id='s')
        owner=str(tmp_path if owner_matches else tmp_path/'other')
        kwargs=dict(epoch=epoch,profile_id=owner,admission_id=row['admission_id'],identity=identity,generation=row['generation'],revision=0)
        if owner_matches:
            assert create_report(db,**kwargs)['profile_id']==owner
        else:
            with pytest.raises(rt.RuntimeStoreError,match='permission_denied'): create_report(db,**kwargs)
