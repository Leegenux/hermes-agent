"""Required-patch gate installed outside the checkout; reuse the existing replay driver."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def paths(root):
    root = Path(root).resolve()
    kit = root.parent / 'patches/replay-kit'
    return root, kit, root.parent / 'bin/reapply_local_patches.py', kit / 'native-update-pending.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(root):
    root, kit, driver, _ = paths(root)
    binding = json.loads((kit / 'native-binding.json').read_text(encoding='utf-8'))
    if Path(binding['root']).resolve() != root:
        raise ValueError('native patch binding names another source checkout')
    # Every staged target counts, so the record reads the same before and after HEAD moves.
    targets = {path.name: digest(path) for path in sorted(kit.glob('manifest-*.json'))}
    return dict(root=str(root), manifest=digest(kit / 'manifest.json'), targets=targets, driver=digest(driver),
                guard=digest(root.parent / 'bin/native_patch_guard.py'),
                binding=digest(kit / 'native-binding.json'))


def process_identity(pid):
    """Creation identity; None for a gone PID, unknown conservatively stays live."""
    if pid <= 0:
        return None
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None if ctypes.get_last_error() == 87 else 'unknown'
        times = [wintypes.FILETIME() for _ in range(4)]
        try:
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return 'unknown'
            if exit_code.value != 259:  # STILL_ACTIVE; an exited Windows PID can remain queryable
                return None
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                return 'unknown'
            return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart='], capture_output=True,
                            text=True, timeout=5, check=False)
    return result.stdout.strip() or (None if result.returncode else 'unknown')


def owner_alive(owner):
    current = process_identity(owner['pid'])
    return current is not None and (current == 'unknown' or current == owner['created'])


def driver_module(root):
    _, _, driver, _ = paths(root)
    spec = importlib.util.spec_from_file_location('_required_patch_replay', driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_for(root, driver):
    """``manifest-<HEAD>.json`` when that commit has a ported set staged, else ``manifest.json``.
    A newer commit with nothing staged falls back to the old set, which then fails closed."""
    root, kit, _, _ = paths(root)
    head = driver.git(root, 'rev-parse', 'HEAD')
    if head.returncode:
        raise RuntimeError('git cannot name the checked-out commit')
    staged = kit / f"manifest-{head.stdout.decode('ascii').strip()}.json"
    return staged if staged.is_file() else kit / 'manifest.json'


def rehearse(root, *, apply=False):
    root, kit, _, _ = paths(root)
    driver = driver_module(root)
    return driver.replay(root, manifest_for(root, driver), apply=apply, workspace=kit / 'native-scratch')


def unregistered(root):
    root = paths(root)[0]
    driver = driver_module(root)
    managed = {p for _, chunks in driver.load_manifest(manifest_for(root, driver)) for p, _ in chunks}
    changed = set()
    for args in [('diff', 'HEAD', '--name-only', '-z'), ('ls-files', '--others', '--exclude-standard', '-z')]:
        result = driver.git(root, *args)
        if result.returncode:
            raise RuntimeError('git cannot inventory local source edits')
        changed.update(x.decode('utf-8') for x in result.stdout.split(b'\0') if x)
    return sorted(changed - managed)


def prepare(root, owner_pid):
    root, _, _, pending = paths(root)
    if pending.exists():
        raise RuntimeError('required local patch replay is already pending; recover it before another update')
    before = fingerprint(root)
    extra = unregistered(root)
    if extra:
        raise RuntimeError('unregistered local source edits: ' + ', '.join(extra))
    result = rehearse(root)
    if result['changed']:
        raise RuntimeError('required patches are missing before update; run manual replay first')
    created = process_identity(owner_pid)
    if created in (None, 'unknown'):
        raise RuntimeError('cannot bind native update to a live process creation identity')
    if before != fingerprint(root):
        raise RuntimeError('patch inputs changed during preparation')
    body = dict(schema=1, inputs=before, owner=dict(pid=owner_pid, created=created))
    # O_EXCL never replaces an existing obligation. A torn create fails closed.
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(body, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def read_intent(root):
    root, _, _, pending = paths(root)
    data = pending.read_bytes()
    intent = json.loads(data)
    if intent.get('schema') != 1 or intent['inputs'] != fingerprint(root):
        raise RuntimeError('required patch inputs changed while native update was pending')
    return pending, data, intent


def retire(pending, before):
    if pending.read_bytes() != before:
        raise RuntimeError('pending update record changed concurrently')
    pending.unlink()


def complete(root):
    _, _, _, pending = paths(root)
    if not pending.exists():
        return None
    pending, before, _ = read_intent(root)
    result = rehearse(root, apply=True)
    if rehearse(root)['changed']:
        raise RuntimeError('required patches remain missing after replay')
    read_intent(root)
    retire(pending, before)
    return result


def check_launch(root):
    _, _, _, pending = paths(root)
    if not pending.exists():
        return None
    pending, before, intent = read_intent(root)
    if owner_alive(intent['owner']):
        raise RuntimeError('native update is still running; required patch replay has not finished')
    result = rehearse(root)
    if result['changed']:
        raise RuntimeError('required local patches are missing after an interrupted update; run manual replay first')
    read_intent(root)
    retire(pending, before)
    return result


def verify_resume(root):
    _, _, _, pending = paths(root)
    if not pending.exists():
        return None
    pending, before, _ = read_intent(root)
    result = rehearse(root)
    if result['changed']:
        raise RuntimeError('required local patches remain missing; paused gateways stay stopped; run manual replay first')
    read_intent(root)
    retire(pending, before)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'complete', 'launch', 'resume'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--owner', type=int)
    args = parser.parse_args()
    try:
        if args.phase == 'prepare':
            if args.owner is None:
                raise ValueError('prepare requires updater process owner')
            result = prepare(args.root, args.owner)
        elif args.phase == 'complete':
            result = complete(args.root)
        elif args.phase == 'resume':
            result = verify_resume(args.root)
        else:
            result = check_launch(args.root)
        if result is not None:
            print(f"Local patches: {args.phase}; {result['topics']} topics, {result['files']} files, {len(result['changed'])} changed.")
        return 0
    except KeyboardInterrupt:
        print('Required local patch replay interrupted; update remains incomplete.', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, SyntaxError, subprocess.SubprocessError) as exc:
        print(f'Required local patch replay incomplete: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
