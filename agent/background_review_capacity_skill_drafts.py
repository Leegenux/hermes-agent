"""Native skill capacity drafts: full package backup, never a live commit."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import shutil
import uuid

from agent.background_review_capacity import (
    INSTRUCTION_SUFFIXES, _safe_candidate_name, normalized_text, validate_candidate)
from agent.background_review_capacity_drafts import _SCOPE, _digest, _manifest
from agent.background_review_capacity_package import direct_path, file_manifest, package_state
from hermes_constants import get_hermes_home


def _admitted_target(scope, operations):
    from tools import skill_manager_tool as smt

    found = []
    for op in operations:
        name = op.get('name')
        if not isinstance(name, str) or not name:
            raise ValueError('Capacity operation has no native skill identity')
        existing = smt._find_skill(name)
        target = next((t for t in scope.plan.targets if t.kind == 'skill' and
                       ((existing and t.documents[0].path.parent.resolve() ==
                         Path(existing['path']).resolve()) or t.name == name)), None)
        found.append((target, existing))
    if not any(target for target, _ in found):
        return None
    target = found[0][0]
    if target is None or any(t != target or not item for t, item in found):
        raise ValueError('A capacity skill draft must contain exactly one admitted package')
    root = target.documents[0].path.parent
    direct_path(root)
    from agent.background_review_capacity_sources import require_active_skill
    require_active_skill(target)
    if any(Path(item['path']).resolve() != root.resolve() for _, item in found):
        raise ValueError('Capacity source is not an identified active-profile skill')
    if not target.package_files:
        raise ValueError('Capacity package was not fully frozen')
    return target


def _preview_operations(target, operations):
    from tools import skill_manager_tool as smt, write_approval as wa
    from tools.skill_manager_guards import _background_review_read_before_write_guard

    root = target.documents[0].path.parent
    candidates = {doc.path.relative_to(root).as_posix(): doc.raw for doc in target.documents}
    for op in operations:
        action = op['action']
        if action not in {'patch', 'edit', 'write_file'}:
            raise ValueError('Capacity drafts cannot remove, delete or create skills')
        if shape := smt._op_shape_error(action, op):
            raise ValueError(shape)
        rewrite = action == 'edit' or (action == 'patch' and op.get('content'))
        name = 'SKILL.md' if rewrite else op.get('file_path') or 'SKILL.md'
        if (not _safe_candidate_name(name) or smt._validate_file_path(name)
                or Path(name).suffix.lower() not in INSTRUCTION_SUFFIXES):
            raise ValueError('Capacity may edit only native, non-escaping instruction paths')
        if (root / name).exists():
            if _background_review_read_before_write_guard(op['name'], root / name, action, name):
                raise ValueError('Native read-before-write guard refused capacity draft')
        if rewrite:
            text = op['content']
        elif action == 'write_file':
            text = op['file_content']
        else:
            if name not in candidates:
                raise ValueError('Capacity patch targets a missing instruction')
            text, error = wa._fold_patch(normalized_text(candidates[name]), op['old_string'],
                                         op['new_string'], op.get('replace_all', False))
            if error:
                raise ValueError('Native patch preview rejected capacity operation')
        if not isinstance(text, str) or len(text.encode('utf-8')) > smt.MAX_SKILL_FILE_BYTES:
            raise ValueError('Native instruction byte limit rejected capacity candidate')
        if smt._validate_content_size(text, label=name):
            raise ValueError('Native instruction size validator rejected capacity candidate')
        if name == 'SKILL.md' and smt._validate_frontmatter(text):
            raise ValueError('Native frontmatter validator rejected capacity candidate')
        candidates[name] = text.encode('utf-8')
    if errors := validate_candidate(target, candidates):
        raise ValueError('; '.join(errors))
    return candidates


def _snapshot_package(target):
    from tools import skill_manager_batch as batch

    root = target.documents[0].path.parent
    base = get_hermes_home() / 'pending' / 'skills' / '.capacity-backups' / uuid.uuid4().hex
    direct_path(base)
    base.mkdir(parents=True, exist_ok=False, mode=0o700)
    snapshots, error = batch._snapshot_skills([root.name], base,
                                              lambda name: {'path': root}, {})
    if error or not snapshots or root.name not in snapshots:
        raise ValueError('Complete native skill snapshot failed; state retained')
    original, snapshot, existed = snapshots[root.name]
    expected = target.package_files, target.package_directories
    if original != root or snapshot is None or package_state(snapshot) != expected:
        raise ValueError('Complete native skill snapshot failed verification; state retained')
    if package_state(root) != expected:
        raise ValueError('Source package changed during snapshot; reject stale proposal')
    return snapshot


def verify_skill_capacity_record(record, *, expected_metadata_sha):
    """Offline backup/candidate verification against the trusted stager's digest."""
    try:
        cap = record['metadata']['capacity']
        if (_digest(cap) != expected_metadata_sha or cap['kind'] != 'skill'
                or cap['version'] != 1 or record['subsystem'] != 'skills'
                or _digest(record['payload']) != cap['payload_sha256']):
            return ('Capacity skill metadata or payload changed',)
        from agent.background_review_capacity_backups import verify_backing_reference
        if errors := verify_backing_reference(cap):
            return errors
        relative = cap['snapshot_relative']
        parts = relative.split('/')
        if (not _safe_candidate_name(relative) or len(parts) != 5
                or parts[:3] != ['pending', 'skills', '.capacity-backups']
                or not re.fullmatch('[0-9a-f]{32}', parts[3])):
            return ('Invalid capacity snapshot location',)
        files, directories = package_state(get_hermes_home() / relative)
        if file_manifest(files) != cap['source_manifest'] or list(directories) != cap['directory_manifest']:
            return ('Incomplete or changed capacity package backup',)
        blobs = cap['candidates']
        if set(blobs) != set(cap['candidate_manifest']):
            return ('Incomplete capacity candidate manifest',)
        originals = {name for name, _, _ in files if Path(name).suffix.lower() in INSTRUCTION_SUFFIXES}
        if not originals <= blobs.keys():
            return ('Missing original instruction in capacity candidate',)
        for name, blob in blobs.items():
            if not _safe_candidate_name(name) or Path(name).suffix.lower() not in INSTRUCTION_SUFFIXES:
                return ('Invalid capacity candidate path',)
            raw = base64.b64decode(blob, validate=True)
            if (_manifest(raw) != cap['candidate_manifest'][name]
                    or not normalized_text(raw).strip() or len(normalized_text(raw)) >= 6000):
                return ('Capacity candidate bytes or strict target rejected',)
    except (OSError, KeyError, TypeError, ValueError, UnicodeError) as exc:
        return (f'Invalid capacity package backup: {exc}',)
    return ()


def _stage(target, payload, operations, scope):
    from tools import skill_manager_tool as smt, write_approval as wa

    names = [op['name'] for op in operations]
    with smt._skill_mutation_locks(names):
        package_key = str(target.documents[0].path.parent.resolve())
        if package_key in scope.skill_packages:
            raise ValueError('This review already staged the admitted capacity package')
        for op in operations:
            if error := smt._background_review_preflight(op['action'], op['name']):
                raise ValueError(error.get('error', 'Native skill guard refused capacity draft'))
        if wa.evaluate_gate(wa.SKILLS).blocked:
            raise ValueError('Native skill approval gate denied capacity draft')
        from agent.background_review_capacity_backups import require_prebackup, _snapshot_path
        reference, original_backup = require_prebackup(scope, target)
        candidates = _preview_operations(target, operations)
        snapshot = _snapshot_path(original_backup)
        payload = {**payload, '_native_capacity_draft': True}
        cap = {'version': 1, 'kind': 'skill', 'target': target.name,
               'state': 'draft', 'independent_review_passed': False,
               'backup_reference': reference,
               'target_chars': 6000, 'strictly_below': True,
               'source_root': str(target.documents[0].path.parent),
               'snapshot_relative': snapshot.relative_to(get_hermes_home()).as_posix(),
               'source_manifest': file_manifest(target.package_files),
               'directory_manifest': list(target.package_directories),
               'candidate_manifest': {name: _manifest(raw) for name, raw in candidates.items()},
               'candidates': {name: base64.b64encode(raw).decode('ascii') for name, raw in candidates.items()},
               'payload_sha256': _digest(payload)}
        record = wa.stage_write(wa.SKILLS, payload, summary=f'Capacity draft: {target.name}',
                                origin=wa.current_origin(), metadata={'capacity': cap})
        expected = _digest(cap)
        durable = wa.get_pending(wa.SKILLS, record['id'])
        if (durable != record or verify_skill_capacity_record(durable, expected_metadata_sha=expected)
                or package_state(target.documents[0].path.parent) !=
                   (target.package_files, target.package_directories)):
            raise ValueError('Capacity skill pending/source did not read back intact')
        if record['id'] in scope.records:
            raise ValueError('Capacity pending identity collision; snapshots retained')
        scope.records[record['id']] = expected
        scope.skill_packages.add(package_key)
        return json.dumps({'success': True, 'staged': True, 'pending_id': record['id'],
                           'capacity_state': 'draft', 'message':
                           'Full package backed up; draft only, independent review and commit not performed.'},
                          ensure_ascii=False)


def stage_skill_capacity(payload):
    """None keeps the ordinary path unchanged; admitted packages never fall through on failure."""
    from tools.skill_provenance import is_unattended_review

    scope = _SCOPE.get()
    if scope is None or not is_unattended_review():
        return None
    target = None
    try:
        operations = payload['operations'] if payload.get('action') == 'batch' else [payload]
        if (target := _admitted_target(scope, operations)) is None:
            return None
        return _stage(target, payload, operations, scope)
    except (OSError, KeyError, TypeError, ValueError, UnicodeError) as exc:
        if target is not None:
            from agent.background_review_capacity_drafts import record_capacity_rejection
            record_capacity_rejection('skill', target.name)
        return json.dumps({'success': False, 'error': f'Capacity skill draft rejected: {exc}'},
                          ensure_ascii=False)


def restore_skill_capacity_backup(record, destination: Path, *, expected_metadata_sha):
    """Recover a verified full package to a NEW offline directory; never live rollback."""
    if errors := verify_skill_capacity_record(record, expected_metadata_sha=expected_metadata_sha):
        raise ValueError('; '.join(errors))
    direct_path(destination)
    if destination.resolve().is_relative_to(get_hermes_home().resolve()):
        raise ValueError('Offline recovery must be outside the active profile')
    cap = record['metadata']['capacity']
    snapshot = get_hermes_home() / cap['snapshot_relative']
    shutil.copytree(snapshot, destination, dirs_exist_ok=False)
    files, directories = package_state(destination)
    if file_manifest(files) != cap['source_manifest'] or list(directories) != cap['directory_manifest']:
        raise ValueError('Offline package recovery readback failed; state retained')
