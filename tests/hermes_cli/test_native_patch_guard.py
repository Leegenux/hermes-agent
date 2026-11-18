"""Required patches gate native completion and launch without touching a real installation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from local_patch_kit import native_guard as guard


@pytest.fixture
def kit(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    root = home / 'source'
    root.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    (root / 'example.py').write_text('answer = 1\n', encoding='utf-8')
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-qm', 'base'], check=True)
    patch = b'diff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-answer = 1\n+answer = 2\n'
    (root / 'example.py').write_text('answer = 2\n', encoding='utf-8')
    registry = home / 'patches/replay-kit'
    registry.mkdir(parents=True)
    (registry / 'example.patch').write_bytes(patch)
    (registry / 'manifest.json').write_text(json.dumps(dict(version=1, entries=[dict(
        id='example', patch='example.patch', sha256=hashlib.sha256(patch).hexdigest(), paths=['example.py'])])), encoding='utf-8')
    (home / 'bin').mkdir()
    source = Path(__file__).resolve().parents[2] / 'local_patch_kit'
    shutil.copy2(source / 'replay.py', home / 'bin/reapply_local_patches.py')
    shutil.copy2(source / 'native_guard.py', home / 'bin/native_patch_guard.py')
    (registry / 'native-binding.json').write_text(json.dumps(dict(root=str(root.resolve()))), encoding='utf-8')
    monkeypatch.chdir(root)
    return root, registry


def pending(registry):
    return registry / 'native-update-pending.json'


def orphan(registry):
    path = pending(registry)
    data = json.loads(path.read_text())
    data['owner'] = dict(pid=2147483647, created='fixture-owner-gone')
    path.write_text(json.dumps(data), encoding='utf-8')


def test_restore_is_verified_before_intent_is_removed(kit):
    root, registry = kit
    guard.prepare(root, os.getpid())
    (root / 'example.py').write_text('answer = 1\n')
    guard.complete(root)
    assert (root / 'example.py').read_text() == 'answer = 2\n'
    assert not pending(registry).exists()
    guard.complete(root)  # repeated native completion is harmless


def test_conflict_keeps_pending_and_source_bytes(kit):
    root, registry = kit
    guard.prepare(root, os.getpid())
    (root / 'example.py').write_text('answer = 3\n')
    with pytest.raises(RuntimeError, match='conflict'):
        guard.complete(root)
    assert pending(registry).is_file()
    assert (root / 'example.py').read_text() == 'answer = 3\n'


def test_prepare_refuses_unregistered_edits(kit):
    root, registry = kit
    (root / 'unregistered.py').write_text('x = 1\n')
    with pytest.raises(RuntimeError, match='unregistered'):
        guard.prepare(root, os.getpid())
    assert not pending(registry).exists()


def _move_head_with_ported_set(root, registry):
    """Upgrade shape: the old edit is gone, HEAD moved, and the ported set applies only to the new base."""
    subprocess.run(['git', '-C', str(root), 'checkout', '-q', '--', 'example.py'], check=True)
    (root / 'example.py').write_text('answer = 10\n', encoding='utf-8')
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-qam', 'upstream'], check=True)
    patch = (b'diff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n'
             b'@@ -1 +1 @@\n-answer = 10\n+answer = 20\n')
    (registry / 'example-ported.patch').write_bytes(patch)
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True,
                          capture_output=True, text=True).stdout.strip()
    return head, json.dumps(dict(version=1, entries=[dict(
        id='example', patch='example-ported.patch', sha256=hashlib.sha256(patch).hexdigest(), paths=['example.py'])]))


def test_manifest_for_new_head_replays_ported_set(kit):
    root, registry = kit
    original = (registry / 'manifest.json').read_bytes()
    assert guard.rehearse(root)['changed'] == []
    head, content = _move_head_with_ported_set(root, registry)
    (registry / f'manifest-{head}.json').write_text(content, encoding='utf-8')
    assert guard.rehearse(root, apply=True)['changed'] == ['example.py']
    assert (root / 'example.py').read_text() == 'answer = 20\n'
    assert guard.rehearse(root)['changed'] == []
    assert (registry / 'manifest.json').read_bytes() == original


def test_manifest_for_other_head_falls_back_and_fails_closed(kit):
    root, registry = kit
    _head, content = _move_head_with_ported_set(root, registry)
    (registry / f"manifest-{'0' * 40}.json").write_text(content, encoding='utf-8')
    with pytest.raises(RuntimeError, match='conflict'):
        guard.rehearse(root, apply=True)
    assert (root / 'example.py').read_text() == 'answer = 10\n'


def test_staged_manifest_change_is_not_silently_accepted(kit):
    root, registry = kit
    (registry / f"manifest-{'0' * 40}.json").write_text('{}', encoding='utf-8')
    guard.prepare(root, os.getpid())
    (registry / f"manifest-{'0' * 40}.json").write_text('{ }', encoding='utf-8')
    with pytest.raises(RuntimeError, match='changed'):
        guard.complete(root)
    assert pending(registry).is_file()


def test_manifest_change_is_not_silently_accepted(kit):
    root, registry = kit
    guard.prepare(root, os.getpid())
    with (registry / 'manifest.json').open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(RuntimeError, match='changed'):
        guard.complete(root)
    assert pending(registry).is_file()


def test_driver_change_is_not_silently_accepted(kit):
    root, registry = kit
    guard.prepare(root, os.getpid())
    with (root.parent / 'bin/reapply_local_patches.py').open('ab') as stream:
        stream.write(b'\n')
    with pytest.raises(RuntimeError, match='changed'):
        guard.complete(root)
    assert pending(registry).is_file()


def test_launch_refuses_a_live_update_even_if_patches_present(kit):
    root, registry = kit
    guard.prepare(root, os.getpid())
    with pytest.raises(RuntimeError, match='still running'):
        guard.check_launch(root)
    assert pending(registry).is_file()


def test_killed_update_cannot_boot_unpatched_source(kit):
    root, registry = kit
    guard.prepare(root, os.getpid())
    (root / 'example.py').write_text('answer = 1\n')
    orphan(registry)
    with pytest.raises(RuntimeError, match='manual replay'):
        guard.check_launch(root)
    assert pending(registry).is_file()
    assert (root / 'example.py').read_text() == 'answer = 1\n'


def test_manual_recovery_allows_next_launch(kit):
    root, registry = kit
    guard.prepare(root, os.getpid())
    orphan(registry)
    guard.check_launch(root)
    assert not pending(registry).exists()


def test_required_files_missing_never_clear_intent(kit):
    root, registry = kit
    guard.prepare(root, os.getpid())
    (root.parent / 'bin/reapply_local_patches.py').unlink()
    with pytest.raises(OSError):
        guard.complete(root)
    assert pending(registry).is_file()


def test_existing_intent_is_not_replaced_by_next_update(kit):
    root, registry = kit
    guard.prepare(root, os.getpid())
    before = pending(registry).read_bytes()
    with pytest.raises(RuntimeError, match='pending'):
        guard.prepare(root, os.getpid())
    assert pending(registry).read_bytes() == before


def test_foreign_root_binding_is_refused(kit):
    root, registry = kit
    (registry / 'native-binding.json').write_text(json.dumps(dict(root=str(root.parent / 'other'))))
    with pytest.raises(ValueError, match='binding'):
        guard.prepare(root, os.getpid())
    assert not pending(registry).exists()


def test_generated_launcher_blocks_before_checkout_import(kit, tmp_path):
    from hermes_cli._launchers import _launcher_script
    root, _ = kit
    guard.prepare(root, os.getpid())
    (root / 'hermes_bootstrap.py').write_text("raise AssertionError('application imported before patch gate')\n")
    launcher = tmp_path / 'native-launcher.py'
    launcher.write_text(_launcher_script('hermes', root, None), encoding='utf-8')
    command = [sys.executable, '-I', str(launcher)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert 'still running' in result.stderr
    assert 'application imported before patch gate' not in result.stderr


@pytest.fixture
def native_completion(kit, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import update_cmd
    root, registry = kit
    guard.prepare(root, os.getpid())
    (root / 'example.py').write_text('answer = 1\n')
    events = []
    monkeypatch.setattr(update_cmd, '_m', lambda: SimpleNamespace(PROJECT_ROOT=root))
    monkeypatch.setattr(update_cmd, '_unrestored_autostash_notice', lambda: None)
    monkeypatch.setattr(update_cmd, '_write_fleet_restart_pending_marker', lambda **kw: None)
    monkeypatch.setattr(update_cmd, '_accept_completion_pm_receipt', lambda *args: None)
    monkeypatch.setattr(update_cmd, 'adopt_retired_channel', lambda request: False)
    monkeypatch.setattr(update_cmd._completion_receipt, 'record_stage', lambda *args, **kw: None)
    monkeypatch.setattr(update_cmd._completion_receipt, 'record_followup', lambda *args, **kw: None)
    monkeypatch.setattr(update_cmd._completion_receipt, 'finalize_update_receipt', lambda *args, **kw: None)
    def child(request):
        events.append((root / 'example.py').read_text())
        return dict(exit_code=0, receipt=None, windows_resume=None)
    monkeypatch.setattr(update_cmd, 'run_completion', child)
    request = dict(source=str(root), expected_sha='fixture', windows_resume=None, receipt=dict(update_id='fixture'))
    return update_cmd, request, events, root, registry


def test_native_completion_runs_only_after_required_replay(native_completion):
    update, request, events, _, registry = native_completion
    update._complete_source_update(request)
    assert events == ['answer = 2\n']
    assert not pending(registry).exists()


def test_native_completion_failure_never_runs_new_child(native_completion):
    update, request, events, root, registry = native_completion
    (root / 'example.py').write_text('answer = 3\n')
    with pytest.raises(SystemExit) as stopped:
        update._complete_source_update(request)
    assert stopped.value.code == 1
    assert events == []
    assert pending(registry).exists()


def test_native_emergency_resume_cannot_bypass_missing_patches(kit, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import update_cmd_windows as windows, update_cmd, update_pause_record
    root, registry = kit
    guard.prepare(root, os.getpid())
    (root / 'example.py').write_text('answer = 1\n')
    resumed, synced = [], []
    monkeypatch.setattr(update_cmd, '_m', lambda: SimpleNamespace(PROJECT_ROOT=root))
    monkeypatch.setattr(update_pause_record, 'tree_is_whole', lambda token: (True, ''))
    monkeypatch.setattr(update_pause_record, 'sync', lambda token: synced.append(dict(token)))
    monkeypatch.setattr(windows, '_resume_paused_set', lambda token: resumed.append(True))
    with pytest.raises(RuntimeError, match='required local patch'):
        windows._resume_unless_torn(dict(pause_id='fixture', resume_needed=True))
    assert resumed == []
    assert synced
    assert pending(registry).exists()


def test_native_zip_fallback_refused_before_download(kit, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import update_cmd, update_cmd_zip
    root, _ = kit
    downloaded = []
    monkeypatch.setattr(update_cmd, '_m', lambda: SimpleNamespace(PROJECT_ROOT=root))
    monkeypatch.setattr(update_cmd_zip, '_download_and_swap_zip', lambda *a: downloaded.append(True))
    with pytest.raises(RuntimeError, match='ZIP'):
        update_cmd_zip._update_via_zip(SimpleNamespace(), completion_request={})
    assert downloaded == []


def test_interrupted_replay_preserves_required_intent(kit, monkeypatch):
    root, registry = kit
    guard.prepare(root, os.getpid())
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(guard, 'rehearse', interrupted)
    with pytest.raises(KeyboardInterrupt):
        guard.complete(root)
    assert pending(registry).exists()


def test_native_interruption_maps_to_130_without_new_child(native_completion, monkeypatch):
    from hermes_cli import update_local_patches
    update, request, events, _, registry = native_completion
    monkeypatch.setattr(update_local_patches.subprocess, 'run',
                        lambda command, **kwargs: subprocess.CompletedProcess(command, 130))
    with pytest.raises(SystemExit) as stopped:
        update._complete_source_update(request)
    assert stopped.value.code == 130
    assert events == []
    assert pending(registry).exists()


def test_native_prepare_precedes_gateway_pause(kit, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import update_cmd
    root, registry = kit
    (root / 'unregistered.py').write_text('x = 1\n')
    paused = []
    monkeypatch.setattr(update_cmd, '_m', lambda: SimpleNamespace(
        PROJECT_ROOT=root, _run_pre_update_backup=lambda args: None))
    monkeypatch.setattr(update_cmd, 'git_operation_in_progress', lambda root: None)
    monkeypatch.setattr(update_cmd, '_resolve_update_options', lambda *args: SimpleNamespace(
        gw_input_fn=input, assume_yes=True))
    monkeypatch.setattr(update_cmd, '_begin_update_receipt_and_plan', lambda args: None)
    monkeypatch.setattr(update_cmd, '_record_pre_update_backup_outcome', lambda *args: None)
    monkeypatch.setattr(update_cmd, '_record_snapshot_stage', lambda *args: None)
    monkeypatch.setattr(update_cmd, '_pause_gateways_for_update', lambda opts: paused.append(True))
    with pytest.raises(RuntimeError, match='required local patch prepare'):
        update_cmd._cmd_update_impl(SimpleNamespace(), False)
    assert paused == []
    assert not pending(registry).exists()


def test_replay_failure_writes_a_real_partial_receipt(kit, monkeypatch):
    from hermes_cli import update_local_patches, update_receipt
    root, registry = kit
    guard.prepare(root, os.getpid())
    (root / 'example.py').write_text('answer = 3\n')
    monkeypatch.setattr(update_receipt, '_receipt_dir', lambda: registry / 'receipts')
    update_receipt.begin_update_receipt()
    with pytest.raises(SystemExit) as stopped:
        update_local_patches.complete_update(root)
    assert stopped.value.code == 1
    receipt = json.loads((registry / 'receipts/latest.json').read_text())
    assert receipt['outcome'] == 'partial'
    assert any(row['name'] == 'local_patches' and row['outcome'] == 'failed' for row in receipt['stages'])
    assert pending(registry).exists()


def test_real_killed_owner_cannot_boot_missing_patches(kit):
    root, registry = kit
    proc = subprocess.Popen([sys.executable, '-I', '-c', 'import sys; sys.stdin.buffer.read(1)'],
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        guard.prepare(root, proc.pid)
        (root / 'example.py').write_text('answer = 1\n')
        proc.kill()
        proc.wait(timeout=10)
        with pytest.raises(RuntimeError, match='manual replay'):
            guard.check_launch(root)
        assert pending(registry).exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        proc.stdin.close()


def test_preswap_adapter_restores_its_replaced_source_file(kit):
    from hermes_cli import update_local_patches as captured_adapter
    root, registry = kit
    expected = Path(captured_adapter.__file__).read_bytes().replace(b'\r\n', b'\n')
    target = root / 'hermes_cli/update_local_patches.py'
    target.parent.mkdir()
    target.write_bytes(expected)
    name = b'hermes_cli/update_local_patches.py'
    lines = expected.splitlines(keepends=True)
    patch = (b'diff --git a/' + name + b' b/' + name + b'\nnew file mode 100644\n--- /dev/null\n+++ b/'
             + name + b'\n@@ -0,0 +1,' + str(len(lines)).encode() + b' @@\n' + b''.join(b'+' + line for line in lines))
    (registry / 'hook.patch').write_bytes(patch)
    spec = json.loads((registry / 'manifest.json').read_text())
    spec['entries'].append(dict(id='native-hook', patch='hook.patch', sha256=hashlib.sha256(patch).hexdigest(),
                                paths=[name.decode()]))
    (registry / 'manifest.json').write_text(json.dumps(spec), encoding='utf-8')
    guard.prepare(root, os.getpid())
    target.unlink()  # model the source swap removing this initially untracked module
    (root / 'example.py').write_text('answer = 1\n')
    captured_adapter.complete_update(root)
    assert target.read_bytes() == expected
    assert (root / 'example.py').read_text() == 'answer = 2\n'
    assert not pending(registry).exists()
