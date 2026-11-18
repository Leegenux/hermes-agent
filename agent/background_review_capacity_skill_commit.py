"""Native skill capacity transactions; dormant, ordinary approvals remain binding.

Per-file atomic visibility, whole-package fencing/rollback under native lock.
Not directory-wide visibility atomicity or hard CAS versus uncooperative writers.
No source deletion: new instruction files are quarantined on failed installation.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import stat
import uuid

from agent.background_review_capacity import validate_candidate
from agent.background_review_capacity_commit import _cancelled, _review_binding
from agent.background_review_capacity_drafts import _SCOPE, _digest
from agent.background_review_capacity_package import direct_path, package_state
from hermes_constants import get_hermes_home

_AUTHORITY = ContextVar('native_skill_capacity_commit_authority', default=None)


@contextmanager
def skill_capacity_commit_authority(profile_home):
    """Trusted integration/test scope only; NOT an approval bypass or activation default."""
    home = Path(profile_home)
    direct_path(home)
    if home.resolve() != get_hermes_home().resolve():
        raise ValueError('Skill capacity authority belongs to another profile')
    token = _AUTHORITY.set(home.resolve())
    try:
        yield
    finally:
        _AUTHORITY.reset(token)


def _event(action, payload):
    from tools import write_approval as wa
    record = wa.stage_write(wa.SKILLS, {'action': action, '_native_capacity_draft': True, **payload},
        origin='background_review', summary='Skill capacity transaction evidence, not approval',
        metadata={'capacity_transaction': {'version': 1, 'state': action,
                                           'installation_authorized': False}})
    if wa.get_pending(wa.SKILLS, record['id']) != record:
        raise ValueError('Skill capacity journal not durable')
    return record


def _gate(target):
    from tools import skill_manager_tool as smt, write_approval as wa
    # Re-evaluate ordinary gates; no _skill_gate_bypass or manual-origin spoofing.
    if wa.evaluate_gate(wa.SKILLS).allow is not True:
        raise ValueError('Ordinary skill approval has not allowed this write')
    root = target.documents[0].path.parent
    direct_path(root)
    home = get_hermes_home().resolve()
    from agent.background_review_capacity_sources import require_active_skill
    require_active_skill(target)
    existing = smt._find_skill(target.name)
    if (_AUTHORITY.get() != home or not existing or
            Path(existing['path']).resolve() != root.resolve() or
            smt._background_review_preflight('patch', target.name)):
        raise ValueError('Skill identity/profile/lock gate refused')


def _load(scope, target):
    from tools import write_approval as wa
    from agent.background_review_capacity_skill_drafts import _preview_operations
    review, packet = _review_binding(scope)
    items = [t for t in packet['packet']['targets'] if t['kind'] == 'skill' and t['target'] == target.name]
    if len(items) != 1:
        raise ValueError('No unique independently reviewed skill candidate')
    item = items[0]
    draft = wa.get_pending(wa.SKILLS, item['pending_id'])
    if draft is None or _digest(draft) != item['record_sha256']:
        raise ValueError('Skill candidate record changed')
    cap, payload = draft['metadata']['capacity'], draft['payload']
    candidates = {name: base64.b64decode(blob, validate=True) for name, blob in cap['candidates'].items()}
    if validate_candidate(target, candidates):
        raise ValueError('Skill structural/source checks refused')
    ops = payload['operations'] if payload['action'] == 'batch' else [payload]
    # Reuse native shape/path/frontmatter/size/description and exact read-before-write checks.
    if _preview_operations(target, ops) != candidates:
        raise ValueError('Native operation preview no longer equals reviewed bytes')
    from tools import skill_manager_tool as smt
    root = target.documents[0].path.parent
    before = (root / 'SKILL.md').read_text(encoding='utf-8-sig')
    if smt._description_budget_error(candidates['SKILL.md'].decode('utf-8-sig'), before=before):
        raise ValueError('Native description budget refused')
    return review, draft, candidates



def _capture_native_history(target, candidates):
    """Native optional history, but never ledger a partial before-state as creation."""
    from hashlib import sha256
    from tools import skill_ledger as ledger
    if not ledger.ledger_enabled():
        return None
    root = target.documents[0].path.parent
    before = ledger.capture_before(root, skill=target.name)
    if before is None:
        raise ValueError('Enabled native history could not capture originals')
    expected = {name: digest for name, digest, _ in target.package_files
                if not any(part in ledger.TRANSIENT_DIRS for part in Path(name).parts[:-1])}
    originals = {doc.path.relative_to(root).as_posix(): doc.raw for doc in target.documents}
    if any(name not in expected and originals.get(name) != raw for name, raw in candidates.items()
           if name in originals or any(part in ledger.TRANSIENT_DIRS for part in Path(name).parts[:-1])):
        raise ValueError('Native history policy excludes a modified instruction')
    actual = {}
    for item in before:
        name = Path(item['path']).relative_to(root).as_posix()
        raw = ledger.read_blob(item['sha256'])
        if name in actual or raw is None or sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Native original history blob missing, duplicated or corrupt')
        actual[name] = item['sha256']
    if actual != expected:
        raise ValueError('Native history original manifest is incomplete or changed')
    return before


def _state(files, dirs):
    return tuple(sorted((name, *values) for name, values in files.items())), tuple(sorted(dirs))


def _check(scope, target, root, expected, journal, run):
    from tools import write_approval as wa
    _cancelled(run)
    _gate(target)
    _review_binding(scope)
    if wa.get_pending(wa.SKILLS, journal['id']) != journal:
        raise ValueError('Skill prepare journal changed')
    if package_state(root) != expected:
        raise ValueError('Skill package changed; reject stale transaction')


def _rollback(root, originals, modes, created, touched, allowed):
    import utils
    from hashlib import sha256
    current = package_state(root)
    if current not in allowed:
        return 'in_doubt'
    files = {name: (digest, size) for name, digest, size in current[0]}
    dirs = set(current[1])
    for name in reversed(touched):
        if package_state(root) != _state(files, dirs):
            return 'in_doubt'
        path = root / name
        if name in originals:
            try:
                utils.atomic_write_bytes(path, originals[name], mode=modes[name], fsync_dir=True)
            except Exception:
                if path.read_bytes() != originals[name]:
                    return 'in_doubt'
            files[name] = (sha256(originals[name]).hexdigest(), len(originals[name]))
        elif path.exists():
            quarantine = _AUTHORITY.get() / 'pending/skills/.capacity-failed' / uuid.uuid4().hex
            direct_path(quarantine)
            quarantine.mkdir(parents=True, exist_ok=False, mode=0o700)
            path.rename(quarantine / path.name)  # Retain new bytes, never erase evidence.
            files.pop(name, None)
        if package_state(root) != _state(files, dirs):
            return 'in_doubt'
    for directory in reversed(created):
        if package_state(root) != _state(files, dirs):
            return 'in_doubt'
        directory.rmdir()  # Only transaction-created empty directories; foreign contents refuse.
        dirs.remove(directory.relative_to(root).as_posix())
    return 'rolled_back'

def _commit_one(scope, target, run):
    from tools import skill_manager_tool as smt
    import utils
    root, key = target.documents[0].path.parent, 'skill:' + target.name
    original_state = target.package_files, target.package_directories
    journal, touched, created, allowed = None, [], [], {original_state}
    originals, modes = {}, {}
    reason = 'validation_failed'
    with smt._skill_mutation_lock(target.name):
        try:
            _cancelled(run)
            _gate(target)
            review, draft, candidates = _load(scope, target)
            reason = 'native_history_capture_refused'
            ledger_before = _capture_native_history(target, candidates)
            originals = {doc.path.relative_to(root).as_posix(): doc.raw for doc in target.documents}
            modes = {name: stat.S_IMODE((root / name).stat().st_mode) for name in originals}
            files = {name: (digest, size) for name, digest, size in target.package_files}
            dirs = set(target.package_directories)
            payload = {'target': target.name, 'draft_id': draft['id'], 'draft_sha256': _digest(draft),
                'review_id': review['id'], 'review_sha256': _digest(review),
                'source_manifest': draft['metadata']['capacity']['source_manifest'],
                'candidate_manifest': draft['metadata']['capacity']['candidate_manifest'],
                'backup_reference': draft['metadata']['capacity']['backup_reference']}
            reason = 'prepare_not_durable'
            journal = _event('skill_capacity_prepare', payload)
            _check(scope, target, root, _state(files, dirs), journal, run)
            reason = 'write_or_readback_failed'
            for name, raw in sorted(candidates.items()):
                if originals.get(name) == raw:
                    continue
                _check(scope, target, root, _state(files, dirs), journal, run)
                path = root / name
                missing = [p for p in path.parent.parents if p != root and p.is_relative_to(root)]
                missing = [*reversed(missing), path.parent] if path.parent != root else []
                for parent in missing:
                    if parent.exists():
                        continue
                    parent.mkdir(mode=0o700)
                    created.append(parent)
                    dirs.add(parent.relative_to(root).as_posix())
                    allowed.add(_state(files, dirs))
                    _check(scope, target, root, _state(files, dirs), journal, run)
                touched.append(name)
                from hashlib import sha256
                files[name] = (sha256(raw).hexdigest(), len(raw))
                allowed.add(_state(files, dirs))
                utils.atomic_write_bytes(path, raw, mode=modes.get(name, 0o644), fsync_dir=True)
                if path.read_bytes() != raw:
                    raise ValueError('Skill candidate did not read back')
                _check(scope, target, root, _state(files, dirs), journal, run)
            reason = 'security_scan_refused'
            if smt._security_scan_skill(root, fail_closed=True):
                raise ValueError('Native skill security scan refused')
            _check(scope, target, root, _state(files, dirs), journal, run)
            reason = 'receipt_not_durable'
            observed = _event('skill_capacity_observed', {**payload, 'prepare_id': journal['id']})
            _check(scope, target, root, _state(files, dirs), journal, run)
            scope.commit_results[key] = {'state': 'committed', 'prepare_id': journal['id'],
                                         'observed_id': observed['id']}
            smt._record_success('patch', target.name, {'success': True}, file_path=None,
                absorbed_into=None, task_id=None, session_id=None, ledger_before=ledger_before)
        except BaseException as exc:
            state = 'refused'
            if touched or created:
                state = 'in_doubt'
                try:
                    state = _rollback(root, originals, modes, created, touched, allowed)
                    if state == 'rolled_back' and package_state(root) != original_state:
                        state = 'in_doubt'
                except Exception:
                    state = 'in_doubt'
            scope.commit_errors[key] = {'state': state, 'reason': reason}
            scope.rejected_targets.add(('skill', target.name))
            if not isinstance(exc, Exception):
                raise


def commit_reviewed_skills(review_run=None):
    scope = _SCOPE.get()
    if scope is None or _AUTHORITY.get() is None:
        return
    from tools.skill_provenance import set_current_write_origin, reset_current_write_origin
    token = set_current_write_origin('background_review')
    try:
        for target in scope.plan.targets:
            key = 'skill:' + target.name
            if target.kind != 'skill' or key in scope.commit_attempted:
                continue
            scope.commit_attempted.add(key)
            try:
                _commit_one(scope, target, review_run)
            except Exception:
                scope.commit_errors[key] = {'state': 'refused', 'reason': 'source_or_lock_unavailable'}
                scope.rejected_targets.add(('skill', target.name))
    finally:
        reset_current_write_origin(token)
