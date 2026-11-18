"""Capacity drafts on the native pending store; never a commit authorization.

Only a trusted native review scope can attach these snapshots. Raw originals,
preview and exact payload are tied to an in-memory digest before later review.
"""
from __future__ import annotations

import base64
import contextvars
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from agent.background_review_capacity import normalized_text, validate_candidate


@dataclass
class CapacityDraftScope:
    plan: object
    records: dict[str, str] = field(default_factory=dict)
    skill_packages: set[str] = field(default_factory=set)
    backups: dict[tuple[str, str], dict] = field(default_factory=dict)
    backup_errors: dict[tuple[str, str], str] = field(default_factory=dict)
    rejected_targets: set[tuple[str, str]] = field(default_factory=set)
    review_results: dict[str, str] = field(default_factory=dict)
    review_errors: list[str] = field(default_factory=list)
    commit_results: dict[str, dict] = field(default_factory=dict)
    commit_errors: dict[str, dict] = field(default_factory=dict)
    commit_attempted: set[str] = field(default_factory=set)


_SCOPE = contextvars.ContextVar('native_capacity_draft_scope', default=None)


@contextmanager
def capacity_review_scope(plan, *, prepare=False):
    scope = CapacityDraftScope(plan) if plan else None
    token = _SCOPE.set(scope)
    try:
        if scope is not None and prepare:
            from agent.background_review_capacity_backups import prepare_capacity_backups
            prepare_capacity_backups(scope)
        yield scope
    finally:
        _SCOPE.reset(token)


def _digest(value) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':')).encode('utf-8')).hexdigest()


def record_capacity_rejection(kind, name):
    scope = _SCOPE.get()
    if scope is not None and any(t.kind == kind and t.name == name for t in scope.plan.targets):
        scope.rejected_targets.add((kind, name))
        return True
    return False


def _manifest(data: bytes) -> dict:
    return {'sha256': sha256(data).hexdigest(), 'bytes': len(data),
            'chars': len(normalized_text(data))}


def capacity_memory_metadata(store, payload):
    """None outside the trusted scope; validate a capacity preview before staging."""
    from tools.skill_provenance import is_unattended_review
    from tools.memory_tool_store import ENTRY_DELIMITER

    scope = _SCOPE.get()
    if scope is None or not is_unattended_review():
        return None
    target = next((t for t in scope.plan.targets if t.kind == 'memory'
                   and t.name == payload.get('target')), None)
    if target is None:
        return None
    from agent.background_review_capacity_backups import require_prebackup
    reference, _ = require_prebackup(scope, target)
    if len(target.documents) != 1 or store._path_for(target.name) != target.documents[0].path:
        raise ValueError('Capacity source identity mismatch')
    operations = payload['operations'] if payload.get('action') == 'batch' else [payload]
    preview = store.resolve_batch_entries(target.name, operations)
    if not preview.get('success'):
        raise ValueError('Native memory preview rejected')
    candidate = ENTRY_DELIMITER.join(preview['candidate_entries']).encode('utf-8')
    name = target.documents[0].path.name
    if validate_candidate(target, {name: candidate}):
        raise ValueError('Capacity candidate size, source or structure rejected')
    # Ordinary approval/replay must not apply a draft and then discard its only
    # raw backup. Only the later verified capacity transaction may consume it.
    payload['_native_capacity_draft'] = True
    raw = target.documents[0].raw
    return {'capacity': {
        'version': 1, 'kind': 'memory', 'target': target.name,
        'target_chars': target.target_chars, 'state': 'draft',
        'independent_review_passed': False, 'payload_sha256': _digest(payload),
        'backup_reference': reference,
        'source_manifest': {name: _manifest(raw)},
        'candidate_manifest': {name: _manifest(candidate)},
        'originals': {name: base64.b64encode(raw).decode('ascii')},
        'candidates': {name: base64.b64encode(candidate).decode('ascii')},
    }}


def capacity_metadata_sha(record) -> str:
    return _digest(record['metadata']['capacity'])


def _decode_manifest(cap, data_key, manifest_key, expected_name):
    blobs, manifest = cap[data_key], cap[manifest_key]
    if set(blobs) != {expected_name} or set(manifest) != {expected_name}:
        raise ValueError('Incomplete capacity snapshot manifest')
    raw = base64.b64decode(blobs[expected_name], validate=True)
    if _manifest(raw) != manifest[expected_name]:
        raise ValueError('Capacity snapshot bytes do not match manifest')
    return raw


def verify_capacity_record(record, *, expected_metadata_sha: str) -> tuple[str, ...]:
    """Check against a digest captured by the trusted stager, not a model assertion."""
    try:
        cap = record['metadata']['capacity']
        if capacity_metadata_sha(record) != expected_metadata_sha:
            return ('Capacity metadata digest changed',)
        if cap['version'] != 1 or cap['kind'] != 'memory' or record['subsystem'] != 'memory':
            return ('Unexpected capacity record kind',)
        from agent.background_review_capacity_backups import verify_backing_reference
        if errors := verify_backing_reference(cap):
            return errors
        name = {'memory': 'MEMORY.md', 'user': 'USER.md'}[cap['target']]
        if record['payload'].get('target') != cap['target'] or _digest(record['payload']) != cap['payload_sha256']:
            return ('Capacity payload changed',)
        _decode_manifest(cap, 'originals', 'source_manifest', name)
        raw = _decode_manifest(cap, 'candidates', 'candidate_manifest', name)
        if not normalized_text(raw).strip() or len(normalized_text(raw)) > cap['target_chars']:
            return ('Capacity candidate target rejected',)
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        return (f'Invalid capacity snapshot: {exc}',)
    return ()


def finish_capacity_stage(record, metadata):
    """Native stage_write is best-effort; a capacity draft requires exact readback."""
    if metadata is None:
        return
    from tools import write_approval as wa

    expected = _digest(metadata['capacity'])
    durable = wa.get_pending(wa.MEMORY, record['id'])
    if durable is None or durable != record or verify_capacity_record(durable, expected_metadata_sha=expected):
        raise ValueError('Capacity pending snapshot did not read back intact')
    scope = _SCOPE.get()
    if scope is None:
        raise ValueError('Capacity stage lost its trusted review scope')
    scope.records[record['id']] = expected


def restore_capacity_backup(record, destination: Path, *, expected_metadata_sha: str):
    """Offline recovery into a NEW directory only. No overwrite, live rollback or deletion."""
    if errors := verify_capacity_record(record, expected_metadata_sha=expected_metadata_sha):
        raise ValueError('; '.join(errors))
    cap = record['metadata']['capacity']
    name = {'memory': 'MEMORY.md', 'user': 'USER.md'}[cap['target']]
    raw = _decode_manifest(cap, 'originals', 'source_manifest', name)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    with (destination / name).open('xb') as stream:
        stream.write(raw)
    if (destination / name).read_bytes() != raw:
        raise ValueError('Offline restore readback failed; recovery state retained')
