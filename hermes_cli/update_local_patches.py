"""Pre-swap adapter for required source patches; coordination lives outside the checkout."""
import os
from pathlib import Path
import subprocess
import sys

from hermes_cli import update_receipt as _receipt


def enabled(root: Path) -> bool:
    kit = Path(root).resolve().parent / 'patches/replay-kit'
    return any((kit / name).exists() for name in (
        'native-binding.json', 'native-update-pending.json'))


def _run(root: Path, phase: str) -> None:
    root = Path(root).resolve()
    if not enabled(root):
        return
    guard = root.parent / 'bin/native_patch_guard.py'
    command = [sys.executable, '-I', '-S', '-B', '-u', str(guard), phase, '--root', str(root)]
    if phase == 'prepare':
        command.extend(['--owner', str(os.getpid())])
    try:
        result = subprocess.run(command, cwd=root, stdin=subprocess.DEVNULL, timeout=300, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('required local patch replay timed out; update remains incomplete') from exc
    if result.returncode == 130:
        raise KeyboardInterrupt
    if result.returncode:
        raise RuntimeError(f'required local patch {phase} did not finish (exit {result.returncode}); update incomplete')


def prepare_update(root: Path) -> None:
    _run(root, 'prepare')


def complete_update(root: Path) -> None:
    try:
        _run(root, 'complete')
    except (RuntimeError, KeyboardInterrupt) as exc:
        _receipt.record_stage('local_patches', 'failed', reason=str(exc))
        _receipt.record_followup('local_patches', str(exc) or 'interrupted',
                                 retry='run the registered manual replay, then relaunch Hermes')
        _receipt.finalize_update_receipt('partial')
        raise SystemExit(130 if isinstance(exc, KeyboardInterrupt) else 1) from exc


def verify_resume(root: Path) -> None:
    _run(root, 'resume')


def refuse_zip(root: Path) -> None:
    if enabled(root):
        raise RuntimeError('required local patches need usable Git; refusing ZIP fallback before source swap')



