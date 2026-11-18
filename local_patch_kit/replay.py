#!/usr/bin/env python
"""Replay the registered local patch set, rehearsing everything before touching the target."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import time


def git(root, *args, data=None):
    return subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(root), *args], input=data, capture_output=True, check=False)


def safe_path(value):
    p = PurePosixPath(value)
    if (not value or '\\' in value or ':' in value or p.is_absolute()
            or any(x in ('..', '.git') for x in p.parts)):
        raise ValueError('unsafe path: ' + value)
    return p.as_posix()


def read_regular(root, name):
    p = root / name
    for parent in (p, *p.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('symlink path: ' + name)
    return p.read_bytes() if p.exists() else None


def load_manifest(manifest):
    spec = json.loads(manifest.read_text(encoding='utf-8'))
    if spec.get('version') != 1:
        raise ValueError('unsupported manifest version')
    entries, seen, owned = [], set(), set()
    for entry in spec['entries']:
        name = entry['id']
        if name in seen or not set(entry.get('depends_on', [])).issubset(seen):
            raise ValueError('duplicate id or unmet dependency: ' + name)
        seen.add(name)
        paths = [safe_path(p) for p in entry['paths']]
        if len(paths) != len(set(paths)) or owned.intersection(paths):
            raise ValueError('overlapping patch paths: ' + name + '; consolidate their final state first')
        owned.update(paths)
        patch_path = manifest.parent / safe_path(entry['patch'])
        data = patch_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('patch hash mismatch: ' + name)
        chunks = re.split(rb'(?=^diff --git )', data, flags=re.MULTILINE)
        chunks = [x for x in chunks if x.strip()]
        found = []
        for chunk in chunks:
            result = subprocess.run(['git', 'apply', '--numstat', '-z', '-'], input=chunk, capture_output=True, check=False)
            records = result.stdout.split(b'\0')
            if result.returncode or len(records) != 2:
                raise ValueError('invalid single-file patch: ' + name)
            path = safe_path(records[0].split(b'\t', 2)[2].decode('utf-8'))
            # No mode changes, binary payloads, renames or deletions: those require explicit support.
            if (not chunk.startswith(b'diff --git a/' + path.encode() + b' b/' + path.encode() + b'\n')
                    or re.search(rb'^(?:GIT binary patch|deleted file mode [0-7]+|new mode [0-7]+|old mode [0-7]+|new file mode 120000)$',
                                 chunk, flags=re.MULTILINE)):
                raise ValueError('unsupported patch path/mode: ' + name)
            found.append((path, chunk))
        if sorted(p for p, _ in found) != sorted(paths):
            raise ValueError('patch paths differ from manifest: ' + name)
        entries.append((entry, found))
    return entries


class ReplayConflictError(RuntimeError):
    """A rehearsal conflict; optionally carries one diagnostic per conflicting path."""

    def __init__(self, message, *, conflicts=None, diagnostics_complete=True):
        super().__init__(message)
        self.conflicts = list(conflicts or [])
        self.diagnostics_complete = bool(diagnostics_complete)


def _index_tracked(root, path):
    result = git(root, 'ls-files', '--error-unmatch', '--', path)
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return 'unknown'


def _chunk_kind(chunk):
    return 'new_file' if re.search(rb'^new file mode [0-7]+\r?$', chunk, flags=re.MULTILINE) else 'modification'


def _numstat(chunk):
    result = subprocess.run(['git', 'apply', '--numstat', '-z', '-'], input=chunk, capture_output=True, check=False)
    records = [record for record in result.stdout.split(b'\0') if record]
    if result.returncode or len(records) != 1:
        return None
    parts = records[0].split(b'\t', 2)
    if len(parts) != 3:
        return None
    try:
        return dict(added=int(parts[0]), removed=int(parts[1]))
    except ValueError:
        return None


def _diff_summary(chunk, check, current, target):
    stats = _numstat(chunk) or dict(added=0, removed=0)
    # git's own diagnostics quote paths and line numbers, never patch bodies. Keep only
    # those bounded lines: file content must not leak through a conflict report.
    messages = []
    for raw in check.stderr.decode('utf-8', errors='replace').splitlines():
        line = raw.strip()
        if line.startswith(('error:', 'warning:', 'hint:')):
            messages.append(line[:200])
        if len(messages) == 4:
            break
    if current is None or target is None:
        relation = 'unknown'
    else:
        relation = 'identical' if current == target else 'differs'
    return dict(added_lines=stats['added'], removed_lines=stats['removed'],
                hunks=len(re.findall(rb'^@@ ', chunk, flags=re.MULTILINE)),
                git_apply_messages=messages, current_vs_target=relation, content_included=False)


def _new_file_target(chunk, path, workspace):
    """Reconstruct a new-file chunk's post-image with git apply in a disposable repository."""
    with tempfile.TemporaryDirectory(prefix='target-', dir=workspace) as tmp:
        stage = Path(tmp)
        if git(stage, 'init', '-q').returncode:
            return None
        if git(stage, 'apply', '-', data=chunk).returncode:
            return None
        target = stage / path
        if not target.is_file() or target.is_symlink():
            return None
        return target.read_bytes()


def _history_candidates(manifest):
    candidates, seen = [], set()
    for candidate in [manifest.parent / 'manifest.json', *sorted(manifest.parent.glob('manifest-*.json'))]:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved == manifest or resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(candidate)
    return candidates


def _history_scan(manifest, path, workspace):
    """Validate sibling manifests and reconstruct their new-file targets for one path."""
    history = dict(consulted=True, scanned=[], usable=[], unusable=[], unreconstructed=[], matches=[], match=None)
    for candidate in _history_candidates(manifest):
        history['scanned'].append(candidate.name)
        try:
            entries = load_manifest(candidate)
        except Exception as exc:
            # A malformed or hash-tampered manifest cannot confer trust; it is reported instead.
            history['unusable'].append(dict(manifest=candidate.name, error=str(exc)[:300]))
            continue
        history['usable'].append(candidate.name)
        for entry, chunks in entries:
            for candidate_path, chunk in chunks:
                if candidate_path != path or _chunk_kind(chunk) != 'new_file':
                    continue
                data = _new_file_target(chunk, candidate_path, workspace)
                if data is None:
                    history['unreconstructed'].append(dict(manifest=candidate.name, patch=entry['patch'],
                                                          error='historical new-file target could not be reconstructed'))
                    continue
                history['matches'].append(dict(manifest=candidate.name, patch=entry['patch'], topic=entry['id'],
                                               patch_sha256=entry['sha256'],
                                               sha256=hashlib.sha256(data).hexdigest(), size=len(data)))
    return history


_RECOVERY_STEPS = {
    'confirmed_stale_new_file': 'The existing untracked file is byte-identical to the new-file target of a validated historical manifest, so it is a leftover from an earlier replay. Review it, move it aside yourself into a backup directory outside the checkout, then re-run with --check. Nothing is moved, overwritten or deleted automatically.',
    'suspected_diverged_new_file': 'The file sits on the path of a registered historical new-file patch but its bytes differ from that target, so it may hold local work. Compare it with the patch, reconcile manually, then re-run with --check. Nothing is moved, overwritten or deleted automatically.',
    'tracked_path_conflict': 'The path is tracked by Git, so leftover-new-file recovery does not apply. Reconcile the tracked file with the patch intent and update the patch or manifest yourself; the replay stays blocked and wrote nothing.',
    'unregistered_existing_file': 'An untracked file already occupies a new-file patch path and no validated historical manifest explains it. Inspect it before touching anything; the replay stays blocked and wrote nothing.',
    'suspected_local_modification': 'The current bytes no longer satisfy the patch context. Compare your local edits with the patch, update the patch or manifest yourself (or revert your edit), then re-run with --check; nothing was written.',
    'missing_preimage_conflict': 'The patch expects a file that is missing from the checkout. Restore the file yourself (or update the patch) and re-run with --check; nothing was written.',
    'unknown_index_state': 'Git could not report whether the path is tracked, so no classification is trusted. Inspect the path with the commands below and reconcile manually; nothing was written.',
}


def _recovery(root, path, classification):
    quoted = str(root)
    inspect = [
        'git -C "' + quoted + '" status --short -- "' + path + '"',
        'git -C "' + quoted + '" ls-files -- "' + path + '"',
        'git -C "' + quoted + '" diff --stat -- "' + path + '"',
    ]
    step = _RECOVERY_STEPS.get(classification, 'Inspect the path with the commands below and reconcile manually; nothing was written.')
    return dict(safe=True, destructive=False, automatic='none', inspect=inspect, steps=[step])


def _diagnose(root, manifest, entry, path, chunk, check, original, workspace):
    current = original.get(path)
    tracked = _index_tracked(root, path)
    kind = _chunk_kind(chunk)
    target = _new_file_target(chunk, path, workspace) if kind == 'new_file' else None
    history = dict(consulted=False, scanned=[], usable=[], unusable=[], unreconstructed=[], matches=[], match=None)
    if kind == 'new_file' and current is not None:
        # History is only consulted for the leftover-new-file case it can actually explain.
        history = _history_scan(manifest, path, workspace)
        current_sha = hashlib.sha256(current).hexdigest()
        exact = [item for item in history['matches'] if item['sha256'] == current_sha]
        history['match'] = exact[0] if exact else None
    if tracked == 'unknown':
        classification = 'unknown_index_state'
    elif tracked:
        classification = 'tracked_path_conflict'
    elif kind == 'new_file' and current is not None:
        if history['match'] is not None:
            classification = 'confirmed_stale_new_file'
        elif history['matches']:
            classification = 'suspected_diverged_new_file'
        else:
            classification = 'unregistered_existing_file'
    elif current is None:
        classification = 'missing_preimage_conflict'
    else:
        classification = 'suspected_local_modification'
    return dict(
        topic=entry['id'], path=path, kind=kind, index_tracked=tracked,
        classification=classification, blocked=True,
        stale_confirmed=classification == 'confirmed_stale_new_file',
        diff_summary=_diff_summary(chunk, check, current, target),
        current=dict(exists=current is not None,
                     sha256=hashlib.sha256(current).hexdigest() if current is not None else None,
                     size=len(current) if current is not None else None),
        target=dict(available=target is not None,
                    sha256=hashlib.sha256(target).hexdigest() if target is not None else None,
                    size=len(target) if target is not None else None,
                    note=('reconstructed with git apply in a disposable scratch repository' if target is not None
                          else 'chunk post-image depends on its pre-image; the patch alone does not determine it')),
        history=history,
        recovery=_recovery(root, path, classification),
    )


def _safe_diagnose(root, manifest, entry, path, chunk, check, original, workspace):
    try:
        return _diagnose(root, manifest, entry, path, chunk, check, original, workspace)
    except Exception as exc:
        return dict(
            topic=entry['id'], path=path, kind=_chunk_kind(chunk), index_tracked='unknown',
            classification='diagnostics_unavailable', blocked=True, stale_confirmed=False,
            diagnostics_complete=False, error=str(exc)[:300],
            diff_summary=dict(added_lines=None, removed_lines=None, hunks=None,
                              git_apply_messages=[], current_vs_target='unknown', content_included=False),
            current=dict(exists=original.get(path) is not None, sha256=None, size=None),
            target=dict(available=False, sha256=None, size=None, note='diagnostics could not be computed'),
            history=dict(consulted=False, scanned=[], usable=[], unusable=[], unreconstructed=[], matches=[], match=None),
            recovery=dict(safe=True, destructive=False, automatic='none', inspect=[],
                          steps=['Diagnostics could not be computed for this conflict; inspect the path manually. Nothing was written or moved.']),
        )


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.replay-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def replay(root, manifest, *, apply=False, workspace=None, all_conflicts=False):
    root, manifest = Path(root).resolve(), Path(manifest).resolve()
    if git(root, 'rev-parse', '--show-toplevel').returncode:
        raise ValueError('target is not a git checkout')
    workspace = Path(workspace or manifest.parent / 'scratch').resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    lock_path = workspace / ('apply-' + hashlib.sha256(str(root).encode()).hexdigest()[:16] + '.lock')
    lock = None
    try:
        if apply:
            try:
                lock = lock_path.open('x')
                lock.write(str(os.getpid()))
                lock.flush()
            except FileExistsError as exc:
                raise RuntimeError('another replay holds ' + str(lock_path)) from exc
        entries = load_manifest(manifest)
        paths = sorted(p for _, chunks in entries for p, _ in chunks)
        original = {p: read_regular(root, p) for p in paths}
        items = []
        conflicts = []
        with tempfile.TemporaryDirectory(prefix='rehearsal-', dir=workspace) as tmp:
            stage = Path(tmp)
            initialized = git(stage, 'init', '-q')
            if initialized.returncode:
                raise RuntimeError(initialized.stderr.decode(errors='replace'))
            for p, data in original.items():
                if data is not None:
                    q = stage / p
                    q.parent.mkdir(parents=True, exist_ok=True)
                    q.write_bytes(data.replace(b'\r\n', b'\n'))
            for entry, chunks in entries:
                states = []
                for p, chunk in chunks:
                    if git(stage, 'apply', '--reverse', '--check', '-', data=chunk).returncode == 0:
                        states.append('present')
                        continue
                    # New-file chunks can carry literal CRLF while the rehearsal tree is
                    # normalized. Accept only the exact reconstructed registered bytes.
                    if (_chunk_kind(chunk) == 'new_file' and original[p] is not None
                            and original[p] == _new_file_target(chunk, p, workspace)):
                        states.append('present')
                        continue
                    check = git(stage, 'apply', '--check', '-', data=chunk)
                    if check.returncode:
                        diagnostic = _safe_diagnose(root, manifest, entry, p, chunk, check, original, workspace)
                        conflicts.append(diagnostic)
                        if not all_conflicts:
                            if any(read_regular(root, name) != original[name] for name in paths):
                                raise RuntimeError('target changed concurrently; rehearsal discarded')
                            raise ReplayConflictError(
                                'conflict: ' + entry['id'] + ' / ' + p + '\n' + check.stderr.decode(errors='replace'),
                                conflicts=[diagnostic], diagnostics_complete=False)
                        states.append('conflict')
                        continue
                    result = git(stage, 'apply', '-', data=chunk)
                    if result.returncode:
                        raise RuntimeError('apply failed: ' + result.stderr.decode(errors='replace'))
                    if git(stage, 'apply', '--reverse', '--check', '-', data=chunk).returncode:
                        raise RuntimeError('post-apply verification failed: ' + p)
                    states.append('restored')
                if 'conflict' in states:
                    state = 'conflict'
                elif all(s == 'present' for s in states):
                    state = 'present'
                else:
                    state = 'restored'
                items.append(dict(id=entry['id'], state=state, files=len(chunks)))
            if conflicts:
                # Reporting conflicts must not hide an edit made while the rehearsal ran.
                if any(read_regular(root, p) != original[p] for p in paths):
                    raise RuntimeError('target changed concurrently; rehearsal discarded')
                raise ReplayConflictError(
                    'conflicts: ' + ', '.join(sorted(c['topic'] + ' / ' + c['path'] for c in conflicts)) + '; no files were written',
                    conflicts=conflicts,
                    diagnostics_complete=all(c.get('diagnostics_complete', True) for c in conflicts))
            final = {p: read_regular(stage, p) for p in paths}
            # Keep each existing file's CRLF convention; global Git settings must not rewrite it.
            for p, data in final.items():
                old = original[p]
                if old and old.count(b'\n') == old.count(b'\r\n') and b'\r\n' in old:
                    final[p] = data.replace(b'\n', b'\r\n')
            for p, data in final.items():
                if data is not None and p.endswith('.py'):
                    ast.parse(data, filename=p)
            changed = [p for p in paths if final[p] != original[p]]
            # Any edit made by another session invalidates the entire rehearsal.
            if any(read_regular(root, p) != original[p] for p in paths):
                raise RuntimeError('target changed concurrently; rehearsal discarded')
            backup = None
            if apply and changed:
                backup = manifest.parent / 'backups' / (time.strftime('%Y%m%d-%H%M%S') + '-' + str(time.time_ns()))
                backup.mkdir(parents=True)
                for p in changed:
                    if original[p] is not None:
                        q = backup / p
                        q.parent.mkdir(parents=True, exist_ok=True)
                        q.write_bytes(original[p])
                (backup / 'recovery.json').write_text(json.dumps(dict(root=str(root), missing=[p for p in changed if original[p] is None], changed=changed), indent=2), encoding='utf-8')
                written = []
                try:
                    for p in changed:
                        if read_regular(root, p) != original[p]:
                            raise RuntimeError('target changed concurrently: ' + p)
                        atomic_write(root / p, final[p])
                        written.append(p)
                    if any(read_regular(root, p) != final[p] for p in paths):
                        raise RuntimeError('read-back mismatch after installation')
                except BaseException:
                    for p in reversed(written):
                        # Never overwrite a concurrent editor's newer data during rollback.
                        if read_regular(root, p) != final[p]:
                            continue
                        if original[p] is None:
                            (root / p).unlink()
                        else:
                            atomic_write(root / p, original[p])
                    raise
        return dict(status='applied' if apply else 'ready', root=str(root), topics=len(items), files=len(paths), changed=changed, items=items, backup=str(backup) if backup else None, validation='git apply/reverse + Python AST; not a functional regression suite', restart_required=bool(apply and changed))
    finally:
        if lock is not None:
            lock.close()
            lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest')
    source.add_argument('--patch', help='legacy explicit patch; gets the same rehearsal and safety checks')
    parser.add_argument('--check', action='store_true', help='rehearse without changing target')
    parser.add_argument('--all-conflicts', action='store_true', help='report every conflict instead of stopping at the first; nothing is written while any conflict remains')
    parser.add_argument('--workspace')
    parser.add_argument('--report')
    args = parser.parse_args()
    try:
        manifest = args.manifest
        if args.patch:
            data = Path(args.patch).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            kit = Path(__file__).resolve().parent.parent / 'patches/replay-kit/legacy' / digest
            kit.mkdir(parents=True, exist_ok=True)
            stats = subprocess.run(['git', 'apply', '--numstat', '-z', '-'], input=data, capture_output=True, check=True)
            paths = [r.split(b'\t', 2)[2].decode('utf-8') for r in stats.stdout.split(b'\0') if r]
            atomic_write(kit / 'explicit.patch', data)
            manifest = kit / 'manifest.json'
            atomic_write(manifest, json.dumps(dict(version=1, entries=[dict(id='explicit', patch='explicit.patch', sha256=digest, paths=paths)])).encode('utf-8'))
        result = replay(args.root, manifest, apply=not args.check, workspace=args.workspace, all_conflicts=args.all_conflicts)
        code = 0
    except Exception as exc:
        result, code = dict(status='failed', error=str(exc)), 1
        conflicts = getattr(exc, 'conflicts', None)
        if conflicts is not None:
            result['conflicts'] = conflicts
            result['diagnostics_complete'] = bool(getattr(exc, 'diagnostics_complete', True))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        atomic_write(Path(args.report), (text + '\n').encode('utf-8'))
    print(text)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
