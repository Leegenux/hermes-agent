"""Ordering and fail-closed prebackup contracts on the real native worker/tools."""
import json
import copy
from pathlib import Path

import pytest

from agent import background_review as native
from hermes_constants import get_hermes_home
from tools import write_approval as wa
from tests.agent.test_background_review_capacity_drafts import stage_fixture
from tests.agent.test_background_review_capacity_skill_drafts import stage_skill_fixture


def observe_fork(monkeypatch, observer, *, callback=None):
    worker = native._run_review_in_thread
    def instrumented_worker(agent, *args, **kwargs):
        fork = native._run_review_fork
        def inspected_fork(*a, **kw):
            observer()
            return fork(*a, **kw)
        monkeypatch.setattr(native, '_run_review_fork', inspected_fork)
        if callback is not None:
            agent.background_review_callback = callback
        return worker(agent, *args, **kwargs)
    monkeypatch.setattr(native, '_run_review_in_thread', instrumented_worker)


@pytest.mark.parametrize('kind', ['memory', 'skills'])
def test_full_original_is_durable_before_native_fork_can_generate_candidate(monkeypatch, kind):
    seen = []
    def before_model():
        backups = [r for r in wa.list_pending(kind) if r['action'] == 'capacity_backup']
        assert len(backups) == 1, 'No durable original existed before the model fork'
        cap = backups[0]['metadata']['capacity']
        assert cap['state'] == 'backup' and cap['source_manifest']
        if kind == 'skills':
            root = get_hermes_home() / cap['snapshot_relative']
            assert (root / 'assets/raw.bin').read_bytes() == bytes(range(256))
            assert (root / 'scripts/empty').is_dir()
        seen.append(backups[0])
    observe_fork(monkeypatch, before_model)
    if kind == 'memory':
        reply, _ = stage_fixture(monkeypatch)
    else:
        reply, _, before, after, _ = stage_skill_fixture(monkeypatch)
        assert before == after
    assert reply.get('staged'), reply
    candidate = wa.get_pending(kind, reply['pending_id'])
    ref = candidate['metadata']['capacity']['backup_reference']
    assert ref['id'] == seen[0]['id']
    assert ref['subsystem'] == kind
    assert seen[0]['id'] != reply['pending_id']


def test_skill_candidate_reuses_pre_model_snapshot_instead_of_copying_twice(monkeypatch):
    from tools import skill_manager_batch as batch
    copier = batch._snapshot_skills
    copies = []
    def count_copy(*args, **kwargs):
        copies.append(args[0])
        return copier(*args, **kwargs)
    monkeypatch.setattr(batch, '_snapshot_skills', count_copy)
    reply, _, before, after, _ = stage_skill_fixture(monkeypatch)
    assert reply.get('staged'), reply
    assert len(copies) == 1
    assert before == after


@pytest.mark.parametrize('kind', ['memory', 'skills'])
def test_disappearing_backing_record_during_model_work_refuses_candidate(monkeypatch, kind):
    def hide_backing_record():
        records = wa.list_pending(kind)
        assert any(r['action'] == 'capacity_backup' for r in records)
        getter = wa.get_pending
        backup_ids = {r['id'] for r in records if r['action'] == 'capacity_backup'}
        monkeypatch.setattr(wa, 'get_pending',
                            lambda sub, pid: None if pid in backup_ids else getter(sub, pid))
    observe_fork(monkeypatch, hide_backing_record)
    if kind == 'memory':
        reply, _ = stage_fixture(monkeypatch)
    else:
        reply, _, before, after, _ = stage_skill_fixture(monkeypatch)
        assert before == after
    assert reply.get('success') is False and not reply.get('staged'), reply


def test_model_time_source_drift_retains_pre_model_original_and_never_overwrites(monkeypatch):
    captured = []
    def change_after_backup():
        backup = next(r for r in wa.list_pending('skills') if r['action'] == 'capacity_backup')
        cap = backup['metadata']['capacity']
        snapshot = get_hermes_home() / cap['snapshot_relative']
        original = (snapshot / 'assets/raw.bin').read_bytes()
        live = Path(cap['source_root']) / 'assets/raw.bin'
        live.write_bytes(b'new user bytes during model')
        captured.append((snapshot, live, original))
    observe_fork(monkeypatch, change_after_backup)
    reply, _, _, after, _ = stage_skill_fixture(monkeypatch)
    assert reply.get('success') is False and not reply.get('staged'), reply
    snapshot, live, original = captured[0]
    assert (snapshot / 'assets/raw.bin').read_bytes() == original == bytes(range(256))
    assert live.read_bytes() == after['assets/raw.bin'] == b'new user bytes during model'


def test_native_callback_reports_draft_without_claiming_capacity_installed(monkeypatch):
    delivered = []
    observe_fork(monkeypatch, lambda: None, callback=delivered.append)
    reply, _ = stage_fixture(monkeypatch)
    assert reply.get('staged'), reply
    assert len(delivered) == 1
    assert 'capacity' in delivered[0].lower() and 'draft' in delivered[0].lower()
    assert 'not installed' in delivered[0].lower()


def test_native_callback_reports_failed_capacity_backup_even_without_draft(monkeypatch):
    delivered = []
    observe_fork(monkeypatch, lambda: None, callback=delivered.append)
    reply, _, before, after, _ = stage_skill_fixture(monkeypatch, alteration='backup-failure')
    assert reply.get('success') is False and before == after
    assert len(delivered) == 1
    assert 'blocked' in delivered[0].lower() and 'not installed' in delivered[0].lower()


@pytest.mark.parametrize('flags', [{'include_plan': False}, {'explicit': True}])
def test_ordinary_or_explicit_worker_does_not_create_capacity_prebackup(monkeypatch, flags):
    reply, _ = stage_fixture(monkeypatch, **flags)
    assert reply.get('staged')
    assert not any(r['action'] == 'capacity_backup' for r in wa.list_pending('memory'))


def test_failed_prebackup_cannot_fall_through_to_add_only_live_memory_write(monkeypatch):
    from tools.memory_tool import MemoryStore, memory_tool, get_memory_dir
    from tools.skill_provenance import set_current_write_origin, reset_current_write_origin
    attempted = []
    def attempt_add_during_model():
        path = get_memory_dir() / 'MEMORY.md'
        before = path.read_bytes()
        store = MemoryStore(memory_char_limit=10000, user_char_limit=5000)
        store.load_from_disk()
        token = set_current_write_origin('background_review')
        try:
            reply = json.loads(memory_tool(action='add', target='memory',
                                          content='SYNTHETIC new rule', store=store))
        finally:
            reset_current_write_origin(token)
        assert reply.get('success') is False, reply
        assert path.read_bytes() == before
        attempted.append(reply)
    observe_fork(monkeypatch, attempt_add_during_model)
    reply, _ = stage_fixture(monkeypatch, lose_pending=True)
    assert reply.get('success') is False and len(attempted) == 1


@pytest.mark.parametrize('kind', ['memory', 'skills'])
def test_prebackup_without_any_candidate_is_independently_recoverable(monkeypatch, tmp_path, kind):
    saved = []
    def before_model():
        from agent.background_review_capacity_drafts import _SCOPE
        scope = _SCOPE.get()
        saved.extend((wa.get_pending(r['subsystem'], r['id']), r['metadata_sha256'])
                     for r in scope.backups.values())
    observe_fork(monkeypatch, before_model)
    if kind == 'memory':
        _, original = stage_fixture(monkeypatch)
    else:
        _, _, original, _, _ = stage_skill_fixture(monkeypatch)
    from agent.background_review_capacity_backups import restore_prebackup
    record, expected = saved[0]
    destination = tmp_path / 'prebackup-only-recovery'
    restore_prebackup(record, destination, expected_metadata_sha=expected)
    if kind == 'memory':
        assert (destination / 'MEMORY.md').read_bytes() == original
    else:
        assert {p.relative_to(destination).as_posix(): p.read_bytes()
                for p in destination.rglob('*') if p.is_file()} == original
        assert (destination / 'scripts/empty').is_dir()


def test_valid_native_dotted_skill_name_is_preserved_in_prebackup(monkeypatch):
    from agent.background_review_capacity import inspect_skill_capacity
    from agent.background_review_capacity_drafts import capacity_review_scope
    from tests.agent.test_background_review_capacity_skill_drafts import HEADER
    root = get_hermes_home() / 'skills' / 'capacity-example.v1'
    root.mkdir(parents=True)
    (root / 'SKILL.md').write_bytes((HEADER + 'x' * 10000).encode())
    plan = inspect_skill_capacity((('capacity-example.v1', root / 'SKILL.md'),))
    with capacity_review_scope(plan, prepare=True) as scope:
        assert not scope.backup_errors, scope.backup_errors
        reference = scope.backups[('skill', 'capacity-example.v1')]
        record = wa.get_pending('skills', reference['id'])
        assert record['metadata']['capacity']['snapshot_relative'].endswith('/capacity-example.v1')


def test_native_callback_does_not_label_rejected_candidate_as_zero_blocked(monkeypatch):
    delivered = []
    observe_fork(monkeypatch, lambda: None, callback=delivered.append)
    reply, _ = stage_fixture(monkeypatch, chars=6002)
    assert reply.get('success') is False and not reply.get('staged')
    assert len(delivered) == 1 and '1 blocked' in delivered[0].lower(), delivered


@pytest.mark.parametrize('field,value', [('origin', 'foreground'), ('id', 'altered-id')])
def test_backing_record_identity_cannot_change_while_preserving_capacity_metadata(monkeypatch, field, value):
    def alter_backing_record():
        ids = {r['id'] for r in wa.list_pending('memory') if r['action'] == 'capacity_backup'}
        getter = wa.get_pending
        def changed(subsystem, pending_id):
            result = getter(subsystem, pending_id)
            if pending_id in ids:
                result = copy.deepcopy(result)
                result[field] = value
            return result
        monkeypatch.setattr(wa, 'get_pending', changed)
    observe_fork(monkeypatch, alter_backing_record)
    reply, _ = stage_fixture(monkeypatch)
    assert reply.get('success') is False and not reply.get('staged'), reply
