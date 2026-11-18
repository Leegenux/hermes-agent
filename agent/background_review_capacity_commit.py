"""Profile-bound memory capacity transactions on native locks/pending/atomic I/O.

Dormant without a trusted caller's authority context (no tool or config default).
Each file is atomic; this is not cross-file visibility atomicity or hard CAS
against external processes which ignore the existing native memory lock.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import stat

from agent.background_review_capacity import normalized_text, validate_candidate
from agent.background_review_capacity_drafts import _SCOPE, _digest
from agent.background_review_capacity_package import direct_path
from hermes_constants import get_hermes_home

_AUTHORITY = ContextVar('native_memory_capacity_commit_authority', default=None)


@contextmanager
def memory_capacity_commit_authority(profile_home):
    """Trusted integration/test caller only; ordinary tools cannot mint this grant.

    Production activation must separately obtain the profile's explicit approval.
    Do not infer a grant from a model payload, environment variable or draft.
    """
    home = Path(profile_home)
    direct_path(home)
    if home.resolve() != get_hermes_home().resolve():
        raise ValueError('Capacity authority belongs to another profile')
    token = _AUTHORITY.set(home.resolve())
    try:
        yield
    finally:
        _AUTHORITY.reset(token)


def _review_binding(scope):
    from tools import write_approval as wa
    from agent.background_review_capacity_critic import review_packet, validate_verdict

    if scope.review_errors or len(scope.review_results) != 1:
        raise ValueError('No unique accepted independent review')
    review_id, expected = next(iter(scope.review_results.items()))
    found = [r for sub in (wa.MEMORY, wa.SKILLS) if (r := wa.get_pending(sub, review_id)) is not None]
    if len(found) != 1:
        raise ValueError('Independent review record missing or ambiguous')
    record = found[0]
    if (_digest(record) != expected or record['id'] != review_id or
            record['origin'] != 'background_review' or record['action'] != 'capacity_review'):
        raise ValueError('Independent review identity or digest changed')
    # Reconstruct the immutable packet, not live source checks: a previous
    # independently committed store may now legitimately have candidate bytes.
    # The current target's live source is always checked under its own lock.
    packet = review_packet(scope, check_sources=False)
    payload = record['payload']
    references = [{key: t[key] for key in ('kind', 'target', 'pending_id', 'record_sha256')}
                  for t in packet['packet']['targets']]
    if (payload['packet_sha256'] != packet['packet_sha256'] or payload['targets'] != references
            or payload.get('_native_capacity_draft') is not True):
        raise ValueError('Independent review does not cover this exact frozen packet')
    validate_verdict(json.dumps(payload['verdict']), packet['packet_sha256'])
    return record, packet


def _event(action, payload):
    from tools import write_approval as wa
    record = wa.stage_write(wa.MEMORY,
        {'action': action, '_native_capacity_draft': True, **payload},
        origin='background_review', summary='Memory capacity transaction evidence; not an approval',
        metadata={'capacity_transaction': {'version': 1, 'state': action,
                                          'installation_authorized': False}})
    if wa.get_pending(wa.MEMORY, record['id']) != record:
        raise ValueError('Capacity transaction evidence did not read back intact')
    return record


def _cancelled(review_run):
    from agent.background_review_capacity_policy import require_current_capacity_policy
    require_current_capacity_policy()
    if review_run is not None and review_run.cancel_requested.is_set():
        raise ValueError('Capacity transaction cancelled')


def _source_guard(path, original):
    direct_path(path)
    if path.read_bytes() != original:
        raise ValueError('Capacity source changed; reject stale proposal')


def _commit_one(scope, store, target, review_run):
    from tools import write_approval as wa
    from tools.memory_tool_store import ENTRY_DELIMITER, _scan_memory_content
    from tools import memory_tool
    import utils

    name, path = target.name, target.documents[0].path
    reason, written, journal, candidate = 'validation_failed', False, None, None
    original = target.documents[0].raw
    mode = None
    with store._file_lock(path):
        try:
            _cancelled(review_run)
            home = _AUTHORITY.get()
            if (home is None or home != get_hermes_home().resolve() or not path.resolve().is_relative_to(home)
                    or path != store._path_for(name) or
                    (memory_tool.fcntl is None and memory_tool.msvcrt is None)):
                raise ValueError('No current-profile locked memory commit authority')
            limit = store._char_limit(name)
            if (type(limit) is not int or limit <= 0 or
                    target.target_chars != limit * 6 // 10 or
                    len(normalized_text(original)) * 10 < limit * 9):
                raise ValueError('Native memory budget or trigger eligibility changed')
            review, packet = _review_binding(scope)
            items = [t for t in packet['packet']['targets'] if t['kind'] == 'memory' and t['target'] == name]
            if len(items) != 1:
                raise ValueError('Memory capacity candidate missing or duplicated')
            item = items[0]
            draft = wa.get_pending(wa.MEMORY, item['pending_id'])
            if draft is None or _digest(draft) != item['record_sha256']:
                raise ValueError('Memory draft changed between review binding and candidate use')
            cap = draft['metadata']['capacity']
            candidate = base64.b64decode(cap['candidates'][path.name], validate=True)
            if errors := validate_candidate(target, {path.name: candidate}):
                raise ValueError('; '.join(errors))
            text = normalized_text(candidate)
            entries = store._parse_entries(text)
            if (ENTRY_DELIMITER.join(entries).encode('utf-8') != candidate or
                    any(_scan_memory_content(entry) for entry in entries)):
                raise ValueError('Native canonical serialization or memory threat scan rejected candidate')
            _source_guard(path, original)
            mode = stat.S_IMODE(path.stat().st_mode)
            payload = {'target': name, 'draft_id': draft['id'], 'draft_sha256': _digest(draft),
                       'review_id': review['id'], 'review_sha256': _digest(review),
                       'packet_sha256': packet['packet_sha256'],
                       'source_manifest': cap['source_manifest'],
                       'candidate_manifest': cap['candidate_manifest'],
                       'backup_reference': cap['backup_reference']}
            reason = 'prepare_not_durable'
            journal = _event('capacity_transaction_prepare', payload)
            _review_binding(scope)
            if wa.get_pending(wa.MEMORY, journal['id']) != journal:
                raise ValueError('Capacity prepare journal changed before replacement')
            _cancelled(review_run)
            _source_guard(path, original)
            reason = 'write_failed'
            written = True
            utils.atomic_write_bytes(path, candidate, tmp_prefix='.mem_', mode=mode, fsync_dir=True)
            reason = 'readback_failed'
            if path.read_bytes() != candidate:
                raise ValueError('Capacity write readback did not match reviewed bytes')
            _cancelled(review_run)
            reason = 'receipt_not_durable'
            observed = _event('capacity_transaction_observed',
                              {**payload, 'prepare_id': journal['id'], 'prepare_sha256': _digest(journal)})
            _review_binding(scope)
            if wa.get_pending(wa.MEMORY, journal['id']) != journal:
                raise ValueError('Capacity prepare journal changed before acknowledgement')
            _cancelled(review_run)
            if path.read_bytes() != candidate:
                raise ValueError('Memory source changed before commit acknowledgement')
            store._set_entries(name, entries)
            scope.commit_results[name] = {'state': 'committed', 'prepare_id': journal['id'],
                'observed_id': observed['id'], 'observed_sha256': _digest(observed),
                'candidate_sha256': cap['candidate_manifest'][path.name]['sha256']}
        except BaseException as exc:
            state = 'refused'
            if written:
                state = 'in_doubt'
                try:
                    actual = path.read_bytes()
                    if actual == original:
                        state = 'unchanged'
                    elif actual == candidate:
                        # Never restore over unknown/newer external bytes. A native
                        # cooperative writer cannot enter until this lock exits.
                        try:
                            utils.atomic_write_bytes(path, original, tmp_prefix='.mem_', mode=mode, fsync_dir=True)
                        except Exception:
                            pass
                        if path.read_bytes() == original:
                            state = 'rolled_back'
                    if state in ('unchanged', 'rolled_back'):
                        store._set_entries(name, store._parse_entries(normalized_text(original)))
                except Exception:
                    pass
            scope.commit_errors[name] = {'state': state, 'reason': reason}
            scope.rejected_targets.add(('memory', name))
            if journal is not None:
                try:
                    _event('capacity_transaction_outcome', {'target': name, 'prepare_id': journal['id'],
                                                           'state': state, 'reason': reason})
                except Exception:
                    pass  # Original + prepare remain recoverable; do not claim a durable outcome.
            if not isinstance(exc, Exception):
                raise


def commit_reviewed_memory(store, review_run=None):
    scope = _SCOPE.get()
    if scope is None or _AUTHORITY.get() is None:
        return
    for target in scope.plan.targets:
        if target.kind != 'memory' or target.name in scope.commit_attempted:
            continue
        scope.commit_attempted.add(target.name)
        if store is None:
            scope.commit_errors[target.name] = {'state': 'refused', 'reason': 'store_unavailable'}
            scope.rejected_targets.add(('memory', target.name))
            continue
        try:
            path, home = target.documents[0].path, _AUTHORITY.get()
            direct_path(path)
            if (home != get_hermes_home().resolve() or not path.resolve().is_relative_to(home)
                    or path != store._path_for(target.name)):
                raise ValueError('Memory source/profile changed before lock')
            _commit_one(scope, store, target, review_run)
        except Exception:
            scope.commit_errors[target.name] = {'state': 'refused', 'reason': 'source_or_lock_unavailable'}
            scope.rejected_targets.add(('memory', target.name))
