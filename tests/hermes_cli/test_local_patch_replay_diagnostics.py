"""Replay conflict diagnostics: structured, non-destructive, and never a partial install."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / 'local_patch_kit' / 'replay.py'


def load_driver():
    spec = importlib.util.spec_from_file_location('replay_diagnostics_under_test', DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def driver():
    return load_driver()


class Kit:
    def __init__(self, home):
        self.home = home
        self.root = home / 'target'
        self.kit = home / 'kit'
        self.root.mkdir(parents=True)
        self.kit.mkdir(parents=True)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.write_bytes('a.py', b'value = 1\n')
        self.write_bytes('b.py', b'other = 1\n')
        self.write_bytes('c.py', b'c = 1\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'base')

    def git(self, *args):
        return subprocess.check_output(['git', '-c', 'core.autocrlf=false', '-C', str(self.root), *args])

    def write_bytes(self, name, data):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data.encode() if isinstance(data, str) else data)
        return target

    def read_bytes(self, name):
        target = self.root / name
        return target.read_bytes() if target.is_file() else None

    def index_bytes(self):
        return (self.root / '.git' / 'index').read_bytes()

    def modification_patch(self, filename, path, text, topic=None):
        old = self.read_bytes(path)
        assert old is not None
        self.write_bytes(path, text)
        try:
            data = self.git('diff', '--binary', '--', path)
        finally:
            self.write_bytes(path, old)
        assert data.startswith(b'diff --git '), data
        (self.kit / filename).write_bytes(data)
        return {'id': topic or filename.split('.')[0], 'patch': filename,
                'sha256': hashlib.sha256(data).hexdigest(), 'paths': [path]}

    def new_file_patch(self, filename, path, text, topic=None):
        lines = text.splitlines()
        data = (f'diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n'
                f'+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n'
                + ''.join('+' + line + '\n' for line in lines)).encode()
        (self.kit / filename).write_bytes(data)
        return {'id': topic or filename.split('.')[0], 'patch': filename,
                'sha256': hashlib.sha256(data).hexdigest(), 'paths': [path]}

    def write_manifest(self, entries, name='manifest.json'):
        path = self.kit / name
        path.write_text(json.dumps(dict(version=1, entries=entries)), encoding='utf-8')
        return path

    def replay(self, driver, manifest, apply=True, all_conflicts=None):
        kwargs = {} if all_conflicts is None else {'all_conflicts': all_conflicts}
        return driver.replay(self.root, manifest, apply=apply, workspace=self.home / 'scratch', **kwargs)


@pytest.fixture(autouse=True)
def _sandbox_cwd(tmp_path, monkeypatch):
    """Run from the disposable sandbox: the repo guard refuses bare `git apply` from the checkout."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def kit(tmp_path):
    return Kit(tmp_path)


def conflicts_of(error):
    conflicts = getattr(error, 'conflicts', None)
    assert conflicts, 'conflict failures must expose structured diagnostics'
    return conflicts


def assert_safe_recovery(diagnostic):
    recovery = diagnostic['recovery']
    assert recovery['safe'] is True
    assert recovery['destructive'] is False
    assert recovery['inspect'], 'diagnostics must include read-only inspection commands'
    assert all(isinstance(command, str) and diagnostic['path'] in command for command in recovery['inspect'])
    assert recovery['steps']
    rendered = json.dumps(recovery)
    for forbidden in ('rm -rf', 'git reset --hard', 'git checkout --', 'git clean'):
        assert forbidden not in rendered


def assert_untouched(kit, index, files):
    assert kit.index_bytes() == index
    for name, data in files.items():
        assert kit.read_bytes(name) == data
    backups = kit.kit / 'backups'
    assert not backups.exists() or not list(backups.iterdir())


def test_confirmed_stale_untracked_new_file_reports_validated_history(kit, driver):
    old = kit.new_file_patch('old-new.patch', 'new.py', 'answer = 42\n', topic='legacy')
    current = kit.new_file_patch('new.patch', 'new.py', 'answer = 43\n', topic='current')
    kit.write_manifest([old], name='manifest-2026old.json')
    manifest = kit.write_manifest([current])
    kit.write_bytes('new.py', b'answer = 42\n')
    index = kit.index_bytes()

    with pytest.raises(RuntimeError) as caught:
        kit.replay(driver, manifest, apply=True, all_conflicts=True)

    error = caught.value
    assert error.diagnostics_complete is True
    conflicts = conflicts_of(error)
    assert len(conflicts) == 1
    diagnostic = conflicts[0]
    assert diagnostic['topic'] == 'current'
    assert diagnostic['path'] == 'new.py'
    assert diagnostic['index_tracked'] is False
    assert diagnostic['blocked'] is True
    assert diagnostic['classification'] == 'confirmed_stale_new_file'
    assert diagnostic['stale_confirmed'] is True
    assert diagnostic['current'] == {'exists': True, 'sha256': hashlib.sha256(b'answer = 42\n').hexdigest(), 'size': 12}
    assert diagnostic['target']['available'] is True
    assert diagnostic['target']['sha256'] == hashlib.sha256(b'answer = 43\n').hexdigest()
    assert diagnostic['target']['size'] == 12
    history = diagnostic['history']
    assert history['consulted'] is True
    assert 'manifest-2026old.json' in history['scanned']
    assert 'manifest-2026old.json' in history['usable']
    assert history['unusable'] == []
    assert history['match']['manifest'] == 'manifest-2026old.json'
    assert history['match']['patch'] == 'old-new.patch'
    assert history['match']['sha256'] == diagnostic['current']['sha256']
    assert_safe_recovery(diagnostic)
    assert_untouched(kit, index, {'new.py': b'answer = 42\n'})


def test_diff_summary_is_bounded_and_does_not_leak_file_content(kit, driver):
    current = kit.new_file_patch('new.patch', 'new.py', 'secret = "target body"\n')
    manifest = kit.write_manifest([current])
    kit.write_bytes('new.py', b'answer = 42\n')

    with pytest.raises(RuntimeError) as caught:
        kit.replay(driver, manifest, apply=True, all_conflicts=True)

    summary = conflicts_of(caught.value)[0]['diff_summary']
    assert summary['content_included'] is False
    assert summary['added_lines'] == 1
    assert summary['removed_lines'] == 0
    assert summary['hunks'] == 1
    rendered = json.dumps(summary)
    assert 'secret =' not in rendered
    assert 'answer = 42' not in rendered
    assert len(rendered) < 1200


def test_untracked_user_edit_is_suspected_not_stale(kit, driver):
    old = kit.new_file_patch('old-new.patch', 'new.py', 'answer = 42\n')
    current = kit.new_file_patch('new.patch', 'new.py', 'answer = 43\n')
    kit.write_manifest([old], name='manifest-2026old.json')
    manifest = kit.write_manifest([current])
    kit.write_bytes('new.py', b'answer = 7\n')
    index = kit.index_bytes()

    with pytest.raises(RuntimeError) as caught:
        kit.replay(driver, manifest, apply=True, all_conflicts=True)

    diagnostic = conflicts_of(caught.value)[0]
    assert diagnostic['index_tracked'] is False
    assert diagnostic['stale_confirmed'] is False
    assert diagnostic['classification'] == 'suspected_diverged_new_file'
    assert diagnostic['history']['match'] is None
    assert diagnostic['history']['matches'][0]['manifest'] == 'manifest-2026old.json'
    assert diagnostic['current']['sha256'] == hashlib.sha256(b'answer = 7\n').hexdigest()
    assert_safe_recovery(diagnostic)
    assert_untouched(kit, index, {'new.py': b'answer = 7\n'})


def test_tracked_same_path_is_never_confirmed_stale(kit, driver):
    old = kit.new_file_patch('old-new.patch', 'new.py', 'answer = 42\n')
    current = kit.new_file_patch('new.patch', 'new.py', 'answer = 43\n')
    kit.write_manifest([old], name='manifest-2026old.json')
    manifest = kit.write_manifest([current])
    kit.write_bytes('new.py', b'answer = 42\n')
    kit.git('add', 'new.py')
    kit.git('commit', '-qm', 'user tracked new.py')
    index = kit.index_bytes()

    with pytest.raises(RuntimeError) as caught:
        kit.replay(driver, manifest, apply=True, all_conflicts=True)

    diagnostic = conflicts_of(caught.value)[0]
    assert diagnostic['index_tracked'] is True
    assert diagnostic['classification'] == 'tracked_path_conflict'
    assert diagnostic['stale_confirmed'] is False
    assert_safe_recovery(diagnostic)
    assert_untouched(kit, index, {'new.py': b'answer = 42\n'})


def test_unregistered_patch_file_cannot_confer_stale_trust(kit, driver):
    rogue = kit.new_file_patch('rogue.patch', 'new.py', 'answer = 42\n')
    assert (kit.kit / rogue['patch']).is_file()
    current = kit.new_file_patch('new.patch', 'new.py', 'answer = 43\n')
    manifest = kit.write_manifest([current])
    kit.write_bytes('new.py', b'answer = 42\n')

    with pytest.raises(RuntimeError) as caught:
        kit.replay(driver, manifest, apply=True, all_conflicts=True)

    diagnostic = conflicts_of(caught.value)[0]
    assert diagnostic['classification'] == 'unregistered_existing_file'
    assert diagnostic['stale_confirmed'] is False
    assert diagnostic['history']['matches'] == []
    assert diagnostic['history']['usable'] == []
    assert_safe_recovery(diagnostic)


def test_tampered_history_is_unusable_and_never_confirmed(kit, driver):
    old = kit.new_file_patch('old-new.patch', 'new.py', 'answer = 42\n')
    history = kit.write_manifest([old], name='manifest-2026old.json')
    history_before = history.read_bytes()
    (kit.kit / 'old-new.patch').write_bytes(b'diff --git a/new.py b/new.py\nnew file mode 100644\n')
    current = kit.new_file_patch('new.patch', 'new.py', 'answer = 43\n')
    manifest = kit.write_manifest([current])
    kit.write_bytes('new.py', b'answer = 42\n')

    with pytest.raises(RuntimeError) as caught:
        kit.replay(driver, manifest, apply=True, all_conflicts=True)

    diagnostic = conflicts_of(caught.value)[0]
    assert diagnostic['stale_confirmed'] is False
    assert diagnostic['classification'] == 'unregistered_existing_file'
    assert diagnostic['history']['usable'] == []
    assert [item['manifest'] for item in diagnostic['history']['unusable']] == ['manifest-2026old.json']
    assert 'hash' in diagnostic['history']['unusable'][0]['error']
    assert history.read_bytes() == history_before


def test_all_conflicts_collects_every_independent_path_without_writing(kit, driver):
    applies = kit.modification_patch('c.patch', 'c.py', 'c = 2\n', topic='applies')
    conflicts_a = kit.modification_patch('a.patch', 'a.py', 'value = 2\n', topic='edited')
    conflicts_new = kit.new_file_patch('n.patch', 'new.py', 'answer = 43\n', topic='newfile')
    manifest = kit.write_manifest([applies, conflicts_a, conflicts_new])
    kit.write_bytes('a.py', b'value = 999\n')
    kit.write_bytes('new.py', b'answer = 42\n')
    index = kit.index_bytes()

    with pytest.raises(RuntimeError) as caught:
        kit.replay(driver, manifest, apply=True, all_conflicts=True)

    error = caught.value
    assert error.diagnostics_complete is True
    conflicts = conflicts_of(error)
    assert {(item['topic'], item['path']) for item in conflicts} == {('edited', 'a.py'), ('newfile', 'new.py')}
    assert all(item['blocked'] is True for item in conflicts)
    assert_untouched(kit, index, {'c.py': b'c = 1\n', 'a.py': b'value = 999\n', 'new.py': b'answer = 42\n'})


def test_default_conflict_refuses_concurrent_target_edits(kit, driver, monkeypatch):
    entry = kit.modification_patch('a.patch', 'a.py', 'value = 2\n', topic='edited')
    manifest = kit.write_manifest([entry])
    kit.write_bytes('a.py', b'value = 999\n')
    index = kit.index_bytes()
    diagnose = driver._safe_diagnose

    def diagnose_then_edit(*args, **kwargs):
        result = diagnose(*args, **kwargs)
        kit.write_bytes('a.py', b'value = 1000\n')
        return result

    monkeypatch.setattr(driver, '_safe_diagnose', diagnose_then_edit)
    with pytest.raises(RuntimeError, match='target changed concurrently'):
        kit.replay(driver, manifest, apply=True)
    assert_untouched(kit, index, {'a.py': b'value = 1000\n'})


def test_default_mode_still_reports_only_the_first_conflict(kit, driver):
    applies = kit.modification_patch('c.patch', 'c.py', 'c = 2\n', topic='applies')
    conflicts_a = kit.modification_patch('a.patch', 'a.py', 'value = 2\n', topic='edited')
    manifest = kit.write_manifest([applies, conflicts_a])
    kit.write_bytes('a.py', b'value = 999\n')
    index = kit.index_bytes()

    with pytest.raises(RuntimeError) as caught:
        kit.replay(driver, manifest, apply=True)

    error = caught.value
    assert str(error).startswith('conflict: edited / a.py')
    assert error.diagnostics_complete is False
    assert len(conflicts_of(error)) == 1
    assert conflicts_of(error)[0]['topic'] == 'edited'
    assert_untouched(kit, index, {'c.py': b'c = 1\n', 'a.py': b'value = 999\n'})


def test_success_shape_and_idempotent_repeat_are_unchanged(kit, driver):
    entry = kit.modification_patch('a.patch', 'a.py', 'value = 2\n', topic='first')
    manifest = kit.write_manifest([entry])

    ready = driver.replay(kit.root, manifest, apply=False, workspace=kit.home / 'scratch')
    assert ready['status'] == 'ready'
    assert set(ready) == {'status', 'root', 'topics', 'files', 'changed', 'items', 'backup', 'validation', 'restart_required'}
    assert ready['items'] == [{'id': 'first', 'state': 'restored', 'files': 1}]
    assert ready['changed'] == ['a.py']
    assert kit.read_bytes('a.py') == b'value = 1\n'

    applied = driver.replay(kit.root, manifest, apply=True, workspace=kit.home / 'scratch')
    assert applied['status'] == 'applied'
    assert applied['changed'] == ['a.py']
    assert kit.read_bytes('a.py') == b'value = 2\n'

    repeat = driver.replay(kit.root, manifest, apply=True, workspace=kit.home / 'scratch')
    assert repeat['changed'] == []
    assert repeat['items'] == [{'id': 'first', 'state': 'present', 'files': 1}]


def test_active_manifest_hash_mismatch_is_still_rejected(kit, driver):
    entry = kit.modification_patch('a.patch', 'a.py', 'value = 2\n')
    manifest = kit.write_manifest([entry])
    (kit.kit / 'a.patch').write_bytes(b'diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n')
    index = kit.index_bytes()

    with pytest.raises(ValueError, match='hash'):
        kit.replay(driver, manifest, apply=True, all_conflicts=True)
    assert_untouched(kit, index, {'a.py': b'value = 1\n'})


def test_cli_all_conflicts_emits_structured_json_and_exit_1(kit, driver):
    conflicts_a = kit.modification_patch('a.patch', 'a.py', 'value = 2\n', topic='edited')
    conflicts_new = kit.new_file_patch('n.patch', 'new.py', 'answer = 43\n', topic='newfile')
    manifest = kit.write_manifest([conflicts_a, conflicts_new])
    kit.write_bytes('a.py', b'value = 999\n')
    kit.write_bytes('new.py', b'answer = 42\n')
    index = kit.index_bytes()

    completed = subprocess.run(
        [sys.executable, str(DRIVER), '--root', str(kit.root), '--manifest', str(manifest), '--all-conflicts'],
        capture_output=True, text=True, check=False,
    )

    assert completed.returncode == 1, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload['status'] == 'failed'
    assert payload['diagnostics_complete'] is True
    assert len(payload['conflicts']) == 2
    for diagnostic in payload['conflicts']:
        assert set(diagnostic) >= {'topic', 'path', 'index_tracked', 'classification', 'current', 'target', 'history', 'recovery', 'diff_summary'}
        assert diagnostic['blocked'] is True
    assert 'Traceback' not in completed.stdout + completed.stderr
    assert_untouched(kit, index, {'a.py': b'value = 999\n', 'new.py': b'answer = 42\n'})


def test_concurrent_change_is_refused_while_collecting_conflicts(kit, driver, monkeypatch):
    entry = kit.modification_patch('a.patch', 'a.py', 'value = 2\n', topic='edited')
    manifest = kit.write_manifest([entry])
    kit.write_bytes('a.py', b'value = 999\n')
    index = kit.index_bytes()
    original_reader = driver.read_regular
    reads = {'a.py': 0}

    def racing_reader(root, name):
        data = original_reader(root, name)
        if name == 'a.py':
            reads['a.py'] += 1
            if reads['a.py'] > 1:
                return data + b'# concurrent edit\n'
        return data

    monkeypatch.setattr(driver, 'read_regular', racing_reader)

    with pytest.raises(RuntimeError, match='concurrent'):
        kit.replay(driver, manifest, apply=True, all_conflicts=True)
    assert_untouched(kit, index, {'a.py': b'value = 999\n'})
