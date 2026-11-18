"""Manual recovery entry: staged manifest and interpreter selection are explicit.

These tests run the real installed shell asset from a disposable ``home/bin`` layout.
The guard is the real ``native_guard.py``; most cases use a recording driver so the
entry's selection is observable, while one case runs the real replay driver with
``--check`` to prove the staged manifest is applied read-only.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_SOURCE = REPO_ROOT / 'local_patch_kit'
ENTRY_SOURCE = KIT_SOURCE / 'reapply-local-patches.sh'
GUARD_SOURCE = KIT_SOURCE / 'native_guard.py'
DRIVER_SOURCE = KIT_SOURCE / 'replay.py'

RECORDING_DRIVER = '''\
"""Recording stand-in for the replay driver: exposes only what the guard needs."""
import json
import os
from pathlib import Path
import subprocess
import sys


def git(root, *args, data=None):
    return subprocess.run([os.environ.get('ENTRY_GIT', 'git'), '-c', 'core.autocrlf=false',
                           '-C', str(root), *args], input=data, capture_output=True, check=False)


def main():
    record = {'argv': sys.argv[1:], 'manifest': None, 'patch': None, 'root': None,
              'workspace': None, 'report': None, 'check': False, 'all_conflicts': False,
              'interpreter': os.environ.get('FAKE_PYTHON_MARKER')}
    arguments = sys.argv[1:]
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value in ('--root', '--manifest', '--patch', '--workspace', '--report'):
            record[value[2:]] = arguments[index + 1]
            index += 2
            continue
        if value == '--check':
            record['check'] = True
        elif value == '--all-conflicts':
            record['all_conflicts'] = True
        elif value.startswith('--') and '=' in value:
            key, _, item = value[2:].partition('=')
            record[key] = item
        index += 1
    Path(os.environ['ENTRY_RECORD']).write_text(json.dumps(record), encoding='utf-8')
    print('{"status": "recorded"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
'''


def _posix(path):
    return Path(path).as_posix()


def _shell_path(path):
    """``C:/x`` -> ``/c/x`` so colon-separated PATH entries survive MSYS startup."""
    text = Path(path).as_posix()
    if len(text) > 1 and text[1] == ':':
        return f'/{text[0].lower()}{text[2:]}'
    return text


def _path_env(*directories):
    return ':'.join(_shell_path(directory) for directory in directories)


def _tool_source(name):
    found = shutil.which(name)
    assert found, f'test host needs {name}'
    return _posix(found)


@pytest.fixture(scope='session')
def tool_dir(tmp_path_factory):
    """A minimal PATH that keeps shell plumbing but hides every real Python."""
    tools = tmp_path_factory.mktemp('shell-tools')
    for name in ('dirname', 'cygpath', 'timeout', 'git'):
        target = tools / name
        target.write_bytes(f'#!/bin/sh\nexec "{_tool_source(name)}" "$@"\n'.encode())
        target.chmod(0o755)
    return tools


class Home:
    """Disposable ``home/bin`` + ``home/hermes-agent`` + ``home/patches/replay-kit``."""

    def __init__(self, base, bash, *, real_driver=False, git=True):
        self.base = base
        self.bash = bash
        self.home = base / 'hermes home 测试'
        self.bin = self.home / 'bin'
        self.bin.mkdir(parents=True)
        self.root = self.home / 'hermes-agent'
        self.root.mkdir()
        self.kit = self.home / 'patches/replay-kit'
        self.kit.mkdir(parents=True)
        self.record = base / 'record.json'

        shutil.copy2(ENTRY_SOURCE, self.bin / 'reapply-local-patches.sh')
        shutil.copy2(GUARD_SOURCE, self.bin / 'native_patch_guard.py')
        assert (self.bin / 'reapply-local-patches.sh').read_bytes() == ENTRY_SOURCE.read_bytes()
        if real_driver:
            shutil.copy2(DRIVER_SOURCE, self.bin / 'reapply_local_patches.py')
        else:
            (self.bin / 'reapply_local_patches.py').write_text(RECORDING_DRIVER, encoding='utf-8')

        self.git_enabled = git
        if git:
            self.git('init', '-q')
            self.git('config', 'user.name', 'Fixture')
            self.git('config', 'user.email', 'fixture@example.invalid')
            self.write('example.py', 'answer = 1\n')
            self.git('add', '.')
            self.git('commit', '-qm', 'base')
            self.patch = self.patch_file('example.patch', 'answer = 1\n', 'answer = 2\n')
        else:
            self.write('example.py', 'answer = 1\n')
            (self.kit / 'example.patch').write_bytes(
                b'diff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n'
                b'@@ -1 +1 @@\n-answer = 1\n+answer = 2\n')
        self.manifest = self.write_manifest([self.entry('example', 'example.patch', 'example.py')],
                                            name='manifest.json')

    # -- repository helpers -------------------------------------------------
    def git(self, *args):
        return subprocess.check_output(
            ['git', '-c', 'core.autocrlf=false', '-C', _posix(self.root), *args])

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode() if isinstance(text, str) else text)
        return path

    def read(self, name):
        return (self.root / name).read_bytes()

    def patch_file(self, name, old, new):
        path = self.root / 'example.py'
        assert path.read_text(encoding='utf-8') == old, f'{name} does not match the checked-out base'
        path.write_text(new, encoding='utf-8')
        diff = self.git('diff', '--binary', '--', 'example.py')
        self.git('checkout', '-q', '--', 'example.py')
        (self.kit / name).write_bytes(diff)
        return diff

    def entry(self, topic, patch, path):
        data = (self.kit / patch).read_bytes()
        return {'id': topic, 'patch': patch, 'sha256': hashlib.sha256(data).hexdigest(), 'paths': [path]}

    def write_manifest(self, entries, name):
        path = self.kit / name
        path.write_text(json.dumps({'version': 1, 'entries': entries}), encoding='utf-8')
        return path

    def move_head(self, *, old='answer = 1\n', new='answer = 10\n', ported='answer = 20\n'):
        """Upstream moved and the registered set must be ported to the new base."""
        self.write('example.py', new)
        self.git('commit', '-qam', 'upstream')
        head = self.git('rev-parse', 'HEAD').decode().strip()
        self.patch_file('example-ported.patch', new, ported)
        staged = self.write_manifest(
            [self.entry('example', 'example-ported.patch', 'example.py')],
            name=f'manifest-{head}.json')
        return head, staged

    # -- interpreter stubs --------------------------------------------------
    def interpreter(self, path, marker, sink=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ['#!/bin/sh', f'export FAKE_PYTHON_MARKER={marker}']
        if sink is not None:
            lines.append(f'printf "%s\\n" "$FAKE_PYTHON_MARKER" >> "{_posix(sink)}"')
        lines.append('exec "$FAKE_REAL_PYTHON" "$@"')
        path.write_bytes(('\n'.join(lines) + '\n').encode())
        path.chmod(0o755)
        return path

    def failing_interpreter(self, path, code=7):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f'#!/bin/sh\nexit {code}\n'.encode())
        path.chmod(0o755)
        return path

    # -- execution -----------------------------------------------------------
    def run(self, *args, path_dir=None, env=None, timeout=180):
        environment = {key: value for key, value in os.environ.items()
                       if key not in {'HERMES_PATCH_PYTHON', 'HERMES_AGENT_ROOT', 'VIRTUAL_ENV',
                                      'HERMES_ENV_ACTIVE', 'HERMES_PM_ENV_ACTIVE', 'PYTHONPATH',
                                      'PYTHONHOME'}}
        environment['PATH'] = path_dir if path_dir is not None else environment['PATH']
        environment['ENTRY_RECORD'] = str(self.record)
        environment['ENTRY_GIT'] = _posix(shutil.which('git'))
        environment['FAKE_REAL_PYTHON'] = _posix(sys.executable)
        environment['HERMES_HOME'] = _posix(self.home / 'hermes-home')
        environment.update(env or {})
        return subprocess.run(
            [self.bash, _posix(self.bin / 'reapply-local-patches.sh'), *args],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=timeout, env=environment, cwd=self.base)

    def recorded(self):
        assert self.record.is_file(), 'the driver was expected to run'
        return json.loads(self.record.read_text(encoding='utf-8'))


@pytest.fixture
def home(tmp_path, real_bash):
    return Home(tmp_path, real_bash)


# --------------------------------------------------------------------------
# Manifest selection
# --------------------------------------------------------------------------

def test_real_entry_check_selects_staged_manifest_for_current_head(tmp_path, real_bash):
    home = Home(tmp_path, real_bash, real_driver=True)
    old_manifest = home.manifest.read_bytes()
    head, staged = home.move_head()

    completed = home.run('--check')

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload['status'] == 'ready'
    assert payload['changed'] == ['example.py']
    assert home.read('example.py') == b'answer = 10\n'
    report = json.loads((home.kit / 'latest-report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'ready'
    assert home.manifest.read_bytes() == old_manifest
    assert staged.is_file()
    assert '\r' not in completed.stdout


def test_falls_back_to_main_manifest_when_head_has_no_staged_set(home):
    completed = home.run()

    assert completed.returncode == 0, completed.stderr
    record = home.recorded()
    assert record['manifest'] == str(home.manifest)
    assert record['check'] is False


def test_staged_manifest_for_another_commit_does_not_hijack_the_entry(home):
    (home.kit / f"manifest-{'0' * 40}.json").write_text('{"version": 1, "entries": []}', encoding='utf-8')

    completed = home.run()

    assert completed.returncode == 0, completed.stderr
    assert home.recorded()['manifest'] == str(home.manifest)


def test_default_entry_without_git_head_fails_before_the_driver(tmp_path, real_bash):
    home = Home(tmp_path, real_bash, git=False)

    completed = home.run()

    assert completed.returncode != 0
    assert not home.record.exists()
    assert 'cannot select' in completed.stderr
    assert 'Traceback' not in completed.stderr


# --------------------------------------------------------------------------
# Interpreter selection
# --------------------------------------------------------------------------

def test_windows_venv_python_wins_over_env_and_path(home, tool_dir, tmp_path):
    venv = home.interpreter(home.root / '.venv/Scripts/python.exe', 'venv')
    env_python = home.interpreter(tmp_path / 'env python', 'env')
    path_python = home.interpreter(tmp_path / 'path bin/python', 'path')
    path_dir = _path_env(tmp_path / 'path bin', tool_dir)

    completed = home.run(path_dir=path_dir, env={'HERMES_PATCH_PYTHON': _posix(env_python)})

    assert completed.returncode == 0, completed.stderr
    assert venv.is_file()
    assert home.recorded()['interpreter'] == 'venv'


def test_posix_venv_python_is_used_when_windows_location_is_absent(home, tool_dir, tmp_path):
    venv = home.interpreter(home.root / '.venv/bin/python', 'posix-venv')
    env_python = home.interpreter(tmp_path / 'env python', 'env')

    completed = home.run(path_dir=_path_env(tool_dir), env={'HERMES_PATCH_PYTHON': _posix(env_python)})

    assert completed.returncode == 0, completed.stderr
    assert venv.is_file()
    assert home.recorded()['interpreter'] == 'posix-venv'


def test_invalid_venv_falls_through_to_hermes_patch_python(home, tool_dir, tmp_path):
    home.failing_interpreter(home.root / '.venv/Scripts/python.exe', code=7)
    env_python = home.interpreter(tmp_path / 'env python', 'env')

    completed = home.run(path_dir=_path_env(tool_dir), env={'HERMES_PATCH_PYTHON': _posix(env_python)})

    assert completed.returncode == 0, completed.stderr
    assert home.recorded()['interpreter'] == 'env'
    assert 'probe' in completed.stderr
    assert 'python.exe' in completed.stderr


def test_hermes_patch_python_is_used_before_path_python(home, tool_dir, tmp_path):
    env_python = home.interpreter(tmp_path / 'env python', 'env')
    home.interpreter(tmp_path / 'path bin/python', 'path')
    path_dir = _path_env(tmp_path / 'path bin', tool_dir)

    completed = home.run(path_dir=path_dir, env={'HERMES_PATCH_PYTHON': _posix(env_python)})

    assert completed.returncode == 0, completed.stderr
    assert home.recorded()['interpreter'] == 'env'


def test_path_python_is_the_last_resort(home, tool_dir, tmp_path):
    home.interpreter(tmp_path / 'path bin/python', 'path')
    path_dir = _path_env(tmp_path / 'path bin', tool_dir)

    completed = home.run(path_dir=path_dir)

    assert completed.returncode == 0, completed.stderr
    assert home.recorded()['interpreter'] == 'path'


def test_hermes_patch_python_path_with_spaces_and_unicode_runs(home, tool_dir, tmp_path):
    env_python = home.interpreter(tmp_path / 'python dir 测试/python tool', 'env-space')

    completed = home.run(path_dir=_path_env(tool_dir), env={'HERMES_PATCH_PYTHON': _posix(env_python)})

    assert completed.returncode == 0, completed.stderr
    assert home.recorded()['interpreter'] == 'env-space'


def test_hermes_patch_python_accepts_msys_and_backslash_paths(home, tool_dir, tmp_path):
    env_python = home.interpreter(tmp_path / 'env python', 'env')
    path_dir = _path_env(tool_dir)

    for spelling in (_shell_path(env_python), str(env_python)):
        completed = home.run(path_dir=path_dir, env={'HERMES_PATCH_PYTHON': spelling})
        assert completed.returncode == 0, completed.stderr
        assert home.recorded()['interpreter'] == 'env'


def test_windows_store_alias_is_rejected_without_execution(home, tool_dir, tmp_path):
    sentinel = tmp_path / 'alias-ran.txt'
    alias = tmp_path / 'Microsoft/WindowsApps/python.exe'
    alias.parent.mkdir(parents=True)
    alias.write_bytes(f'#!/bin/sh\nprintf "ran\\n" > "{_posix(sentinel)}"\nexit 0\n'.encode())
    alias.chmod(0o755)

    completed = home.run(path_dir=_path_env(tool_dir), env={'HERMES_PATCH_PYTHON': _posix(alias)})

    assert completed.returncode != 0
    assert not sentinel.exists()
    assert not home.record.exists()
    assert 'WindowsApps' in completed.stderr


def test_store_alias_on_path_is_skipped_for_a_later_real_python(home, tool_dir, tmp_path):
    sentinel = tmp_path / 'path-alias-ran.txt'
    alias_dir = tmp_path / 'Microsoft/WindowsApps'
    alias_dir.mkdir(parents=True)
    alias = alias_dir / 'python'
    alias.write_bytes(f'#!/bin/sh\nprintf "ran\\n" > "{_posix(sentinel)}"\nexit 0\n'.encode())
    alias.chmod(0o755)
    home.interpreter(tmp_path / 'real bin/python', 'path')
    path_dir = _path_env(alias_dir, tmp_path / 'real bin', tool_dir)

    completed = home.run(path_dir=path_dir)

    assert completed.returncode == 0, completed.stderr
    assert not sentinel.exists()
    assert home.recorded()['interpreter'] == 'path'
    assert 'WindowsApps' in completed.stderr


def test_no_usable_interpreter_exits_nonzero_without_running_the_driver(home, tool_dir, tmp_path):
    completed = home.run(path_dir=_path_env(tool_dir),
                         env={'HERMES_PATCH_PYTHON': _posix(tmp_path / 'missing python')})

    assert completed.returncode != 0
    assert not home.record.exists()
    assert 'no usable Python' in completed.stderr


def test_hermes_patch_python_is_one_path_not_a_command_line(home, tool_dir, tmp_path):
    env_python = home.interpreter(tmp_path / 'env python', 'env')

    completed = home.run(path_dir=_path_env(tool_dir),
                         env={'HERMES_PATCH_PYTHON': f'{_posix(env_python)} -X dev'})

    assert completed.returncode != 0
    assert not home.record.exists()
    assert 'not found' in completed.stderr


# --------------------------------------------------------------------------
# Legacy and explicit inputs, flag forwarding
# --------------------------------------------------------------------------

def test_explicit_manifest_does_not_require_a_git_head(tmp_path, real_bash):
    home = Home(tmp_path, real_bash, git=False)
    stub = home.manifest

    completed = home.run('--manifest', _posix(stub), path_dir=None,
                         env={'HERMES_PATCH_PYTHON': _posix(
                             home.interpreter(tmp_path / 'env python', 'env'))})

    assert completed.returncode == 0, completed.stderr
    record = home.recorded()
    assert record['manifest'] == _posix(stub)
    assert record['patch'] is None


def test_legacy_positional_patch_is_forwarded_without_head_discovery(tmp_path, real_bash):
    home = Home(tmp_path, real_bash, git=False)
    patch = home.kit / 'example.patch'

    completed = home.run(_posix(patch), env={'HERMES_PATCH_PYTHON': _posix(
        home.interpreter(tmp_path / 'env python', 'env'))})

    assert completed.returncode == 0, completed.stderr
    record = home.recorded()
    assert record['patch'] == _posix(patch)
    assert record['manifest'] is None


def test_check_and_all_conflicts_forward_unchanged_and_late_overrides_win(home, tool_dir, tmp_path):
    home.interpreter(tmp_path / 'path bin/python', 'path')
    path_dir = _path_env(tmp_path / 'path bin', tool_dir)
    workspace = tmp_path / 'user workspace'
    report = tmp_path / 'user report.json'

    completed = home.run('--check', '--all-conflicts', '--workspace', _posix(workspace),
                         '--report', _posix(report), path_dir=path_dir)

    assert completed.returncode == 0, completed.stderr
    record = home.recorded()
    assert record['check'] is True
    assert record['all_conflicts'] is True
    assert record['workspace'] == _posix(workspace)
    assert record['report'] == _posix(report)
    argv = record['argv']
    assert argv.count('--manifest') == 1
    assert argv.index('--report') < argv.index(_posix(report))
    assert record['manifest'] == str(home.manifest)
