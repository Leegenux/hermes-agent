import hashlib, importlib.util, json, os, pathlib, subprocess
import pytest


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    # Replay's read-only git apply --numstat inherits cwd; keep every Git
    # operation outside the checkout as well as the explicitly created repo.
    monkeypatch.chdir(tmp_path)


def load():
    p=pathlib.Path(os.environ.get('REPLAY_DRIVER', pathlib.Path(__file__).resolve().parents[2] / 'local_patch_kit' / 'replay.py'))
    spec=importlib.util.spec_from_file_location('stage4_driver',p)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def setup(tmp_path, content):
    root=tmp_path/'repo'; root.mkdir(); subprocess.run(['git','init','-q',str(root)],check=True)
    patch=b'diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n'+b''.join(b'+'+line for line in content.splitlines(keepends=True))
    kit=tmp_path/'kit'; kit.mkdir(); (kit/'one.patch').write_bytes(patch)
    (kit/'manifest.json').write_text(json.dumps({'version':1,'entries':[{'id':'one','patch':'one.patch','sha256':hashlib.sha256(patch).hexdigest(),'paths':['new.py']}]}))
    return root,kit/'manifest.json',tmp_path/'work'


@pytest.mark.parametrize('newline',[b'\n',b'\r\n'])
def test_registered_new_file_exact_bytes_and_idempotence(tmp_path,newline):
    drv=load(); raw=b'x=1'+newline+b'y=2'+newline; root,manifest,work=setup(tmp_path,raw)
    assert drv.replay(root,manifest,apply=True,workspace=work)['changed']==['new.py']
    assert (root/'new.py').read_bytes()==raw
    assert drv.replay(root,manifest,apply=True,workspace=work)['changed']==[]
    assert (root/'new.py').read_bytes()==raw


def test_unknown_existing_and_modified_file_refused_without_write(tmp_path):
    drv=load(); raw=b'x=1\r\ny=2\r\n'; root,manifest,work=setup(tmp_path,raw)
    (root/'new.py').write_bytes(b'x=8\r\ny=2\r\n')
    before=(root/'new.py').read_bytes()
    with pytest.raises(drv.ReplayConflictError): drv.replay(root,manifest,apply=True,workspace=work)
    assert (root/'new.py').read_bytes()==before


def test_concurrent_target_change_aborts_before_write(tmp_path,monkeypatch):
    drv=load(); raw=b'x=1\r\ny=2\r\n'; root,manifest,work=setup(tmp_path,raw)
    original=drv.read_regular; count=0
    def injected(r,p):
        nonlocal count
        count+=1
        if count==2: (root/'new.py').write_bytes(b'other=1\n')
        return original(r,p)
    monkeypatch.setattr(drv,'read_regular',injected)
    with pytest.raises(RuntimeError,match='changed concurrently'): drv.replay(root,manifest,apply=True,workspace=work)
    assert (root/'new.py').read_bytes()==b'other=1\n'


@pytest.mark.parametrize('newline',[b'\n',b'\r\n'])
def test_existing_file_change_and_true_conflict(tmp_path,newline):
    drv=load();root,manifest,work=setup(tmp_path,b'x=1\ny=2\n')
    patch=(manifest.parent/'one.patch')
    data=b'diff --git a/new.py b/new.py\n--- a/new.py\n+++ b/new.py\n@@ -1,2 +1,2 @@\n x=1\n-y=2\n+y=3\n'
    patch.write_bytes(data)
    payload=json.loads(manifest.read_text());payload['entries'][0]['sha256']=hashlib.sha256(data).hexdigest();manifest.write_text(json.dumps(payload))
    target=root/'new.py'; target.write_bytes(b'x=1'+newline+b'y=2'+newline)
    assert drv.replay(root,manifest,apply=True,workspace=work)['changed']==['new.py']
    assert target.read_bytes()==b'x=1'+newline+b'y=3'+newline
    assert drv.replay(root,manifest,apply=True,workspace=work)['changed']==[]
    target.write_bytes(b'x=1'+newline+b'y=4'+newline)
    with pytest.raises(drv.ReplayConflictError):drv.replay(root,manifest,apply=True,workspace=work)
    assert target.read_bytes()==b'x=1'+newline+b'y=4'+newline
