"""Native read stability with Windows stat/fstat timestamp semantics.

A path API may expose birth time as ctime while descriptor API exposes change
 time. Full unredacted stable reads must work; mutation/partial/redaction must
 still refuse write baselines and background skill-read authority.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

from tools import file_tools as files, file_tools_read_tracking as tracker
from tools.skill_manager_guards import _background_review_has_read, _reset_background_review_read_marks
from tools.skill_provenance import set_current_write_origin, reset_current_write_origin


@contextmanager
def differing_windows_handle_ctime(monkeypatch):
    """Deterministic API mismatch, not simulated file instability."""
    native = os.fstat
    def fstat(fd):
        result = native(fd)
        if not stat.S_ISREG(result.st_mode):
            return result
        attributes = {key: getattr(result, key) for key in dir(result) if key.startswith('st_')}
        attributes['st_ctime_ns'] = attributes['st_ctime_ns'] + 1000000000
        return SimpleNamespace(**attributes)
    with monkeypatch.context() as context:
        context.setattr(os, 'fstat', fstat)
        yield


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows-only cross-API timestamp mismatch')
def test_windows_stable_file_retains_full_read_baseline_and_native_skill_read_mark(tmp_path, monkeypatch):
    path = tmp_path / 'SKILL.md'
    path.write_bytes(b'---\nname: native-probe\n---\nbody\n')
    token = set_current_write_origin('background_review')
    try:
        _reset_background_review_read_marks()
        with differing_windows_handle_ctime(monkeypatch):
            assert tracker._file_version(str(path)) is not None
            output = files.read_file_tool(str(path), task_id='win-stable-ctime')
            assert 'body' in output
            assert tracker._has_full_write_baseline(str(path), 'win-stable-ctime')
            assert _background_review_has_read(path)
    finally:
        reset_current_write_origin(token)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows-only cross-API timestamp mismatch')
@pytest.mark.parametrize('mode', ['partial', 'redacted'])
def test_windows_timestamp_compatibility_does_not_bless_incomplete_or_redacted_reads(tmp_path, monkeypatch, mode):
    path = tmp_path / 'SKILL.md'
    path.write_bytes(b'---\nname: native-probe\n---\nbody\n')
    token = set_current_write_origin('background_review')
    try:
        _reset_background_review_read_marks()
        if mode == 'redacted':
            monkeypatch.setattr(files, 'redact_sensitive_text',
                                lambda text, **kwargs: text.replace('body', '<synthetic-redacted>'))
        task_id = 'win-ctime-' + mode
        with differing_windows_handle_ctime(monkeypatch):
            output = files.read_file_tool(str(path), offset=2 if mode == 'partial' else 1, task_id=task_id)
            assert output
            assert not tracker._has_full_write_baseline(str(path), task_id)
            assert not _background_review_has_read(path)
    finally:
        reset_current_write_origin(token)


@pytest.mark.parametrize('mutation', ['same-size-restored-mtime', 'replacement'])
def test_file_version_refuses_change_during_hash_even_if_size_and_mtime_match(tmp_path, monkeypatch, mutation):
    path = tmp_path / 'synthetic.txt'
    path.write_bytes(b'original bytes')
    before = path.stat()
    native = tracker.hashlib.file_digest
    outcome = {}
    def digest_and_mutate(stream, algorithm):
        digest = native(stream, algorithm)
        try:
            if mutation == 'replacement':
                other = tmp_path / 'replacement.txt'
                other.write_bytes(b'changed! bytes')
                os.utime(other, ns=(before.st_atime_ns, before.st_mtime_ns))
                os.replace(other, path)
            else:
                path.write_bytes(b'changed! bytes')
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            outcome['mutated'] = True
        except OSError as exc:
            # Windows may deny mutation while the snapshot descriptor is open.
            # That is also fail-closed, not evidence that mutation succeeded.
            outcome['blocked'] = type(exc).__name__
            raise
        return digest
    monkeypatch.setattr(tracker.hashlib, 'file_digest', digest_and_mutate)
    assert tracker._file_version(str(path)) is None
    assert outcome
    assert path.read_bytes() == (b'changed! bytes' if outcome.get('mutated') else b'original bytes')


def test_snapshot_digest_changes_for_rewritten_bytes_with_preserved_mtime(tmp_path):
    path = tmp_path / 'synthetic.txt'
    path.write_bytes(b'original bytes')
    before = path.stat()
    old = tracker._file_version(str(path))
    path.write_bytes(b'changed! bytes')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    new = tracker._file_version(str(path))
    assert old is not None and new is not None
    assert old[-1] != new[-1]
