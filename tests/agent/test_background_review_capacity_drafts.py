"""Real native memory-tool staging with a synthetic fork in an isolated profile."""
import copy
import importlib
import json
from types import SimpleNamespace

import pytest

from agent import background_review as native
from agent.background_review_capacity import inspect_memory_capacity
from tools.memory_tool import MemoryStore, get_memory_dir, memory_tool
from tools.skill_provenance import set_current_write_origin, reset_current_write_origin
from tools import write_approval as wa


def stage_fixture(monkeypatch, *, chars=40, change_source=False, lose_pending=False,
                  explicit=False, include_plan=True, raw_layout=False):
    store = MemoryStore(memory_char_limit=10000, user_char_limit=5000)
    store.load_from_disk()
    original = 'SYNTHETIC preference: blue.\n' + 'x' * 9000
    assert store.add('memory', original)['success']
    path = get_memory_dir() / 'MEMORY.md'
    if raw_layout:
        path.write_bytes(b'\xef\xbb\xbf' + original.encode('utf-8').replace(b'\n', b'\r\n'))
    original_bytes = path.read_bytes()
    if raw_layout:
        assert b'\r\r\n' not in original_bytes, 'fixture doubled existing CRLF'
    plan = inspect_memory_capacity(SimpleNamespace(_memory_store=store,
        _memory_enabled=True, _user_profile_enabled=False), review_memory=True,
        paths={'memory': path, 'user': get_memory_dir() / 'USER.md'})
    assert plan.targets
    if change_source:
        path.write_bytes(original_bytes + b' concurrent edit')
    before = path.read_bytes()
    results, failures = [], []
    agent = SimpleNamespace(provider='openai', api_mode='chat_completions',
        memory_notifications='on', background_review_callback=None,
        _safe_print=lambda value: None,
        _emit_auxiliary_failure=lambda *a: failures.append(a))
    if lose_pending:
        monkeypatch.setattr(wa, 'get_pending', lambda *a: None)
    def synthetic_fork(parent, snapshot, prompt, cfg, run, state, *flags):
        token = set_current_write_origin('background_review')
        try:
            result = json.loads(memory_tool(target='memory', operations=[{
                'action':'patch', 'old_text':original,
                'content':'SYNTHETIC preference: blue. ' + 'y' * max(0, chars - 27)
            }], store=store))
            results.append(result)
        finally:
            reset_current_write_origin(token)
    monkeypatch.setattr(native, '_run_review_fork', synthetic_fork)
    native._run_review_in_thread(agent, [], 'Synthetic review only', task_cfg={},
        review_memory=True, explicit=explicit, capacity_plan=plan if include_plan else None)
    assert not failures
    assert path.read_bytes() == before
    return results[0], original_bytes


def test_native_capacity_draft_contains_full_original_and_preview(monkeypatch):
    result, original = stage_fixture(monkeypatch)
    assert result.get('staged')
    record = wa.get_pending('memory', result['pending_id'])
    assert record and record.get('metadata', {}).get('capacity')
    cap = record['metadata']['capacity']
    assert cap['state'] == 'draft'
    assert cap['target'] == 'memory'
    assert cap['target_chars'] == 6000
    assert cap['source_manifest']['MEMORY.md']['bytes'] == len(original)
    assert cap['candidate_manifest']['MEMORY.md']['chars'] <= 6000
    assert cap['independent_review_passed'] is False


@pytest.mark.parametrize('options', [{'chars':6002}, {'change_source':True}, {'lose_pending':True}],
                         ids=['over-target', 'source-changed', 'pending-not-durable'])
def test_failed_capacity_staging_cannot_report_staged_or_modify_live(monkeypatch, options):
    result, _ = stage_fixture(monkeypatch, **options)
    assert not result.get('staged')
    assert result.get('success') is False


def test_ordinary_background_patch_keeps_original_gate_without_capacity_metadata(monkeypatch):
    result, _ = stage_fixture(monkeypatch, include_plan=False)
    assert result.get('staged')
    record = wa.get_pending('memory', result['pending_id'])
    assert 'capacity' not in record.get('metadata', {})


def test_explicit_review_does_not_gain_automatic_capacity_authority(monkeypatch):
    # Synthetic fork sets unattended origin deliberately; explicit worker must not
    # attach the trusted capacity scope. Ordinary gate still controls this call.
    result, _ = stage_fixture(monkeypatch, explicit=True)
    record = wa.get_pending('memory', result['pending_id'])
    assert record and 'capacity' not in record.get('metadata', {})


def test_verified_backup_restores_exact_bytes_to_new_offline_directory(monkeypatch, tmp_path):
    result, original = stage_fixture(monkeypatch)
    record = wa.get_pending('memory', result['pending_id'])
    drafts = importlib.import_module('agent.background_review_capacity_drafts')
    expected = drafts.capacity_metadata_sha(record)
    destination = tmp_path / 'offline-recovery'
    drafts.restore_capacity_backup(record, destination, expected_metadata_sha=expected)
    assert (destination / 'MEMORY.md').read_bytes() == original
    assert drafts.verify_capacity_record(record, expected_metadata_sha=expected) == ()


@pytest.mark.parametrize('damage', ['missing', 'bytes', 'manifest', 'candidate', 'payload'])
def test_incomplete_or_tampered_backup_is_rejected_without_offline_partial_restore(
        monkeypatch, tmp_path, damage):
    result, _ = stage_fixture(monkeypatch)
    record = wa.get_pending('memory', result['pending_id'])
    drafts = importlib.import_module('agent.background_review_capacity_drafts')
    expected = drafts.capacity_metadata_sha(record)
    damaged = copy.deepcopy(record)
    cap = damaged['metadata']['capacity']
    if damage == 'missing':
        cap['originals'].pop('MEMORY.md')
    elif damage == 'bytes':
        cap['originals']['MEMORY.md'] = 'bm90IG9yaWdpbmFs'
    elif damage == 'manifest':
        cap['source_manifest']['MEMORY.md']['sha256'] = '0' * 64
    elif damage == 'candidate':
        cap['candidates']['MEMORY.md'] = 'bm90IGNhbmRpZGF0ZQ=='
    else:
        damaged['payload']['operations'][0]['content'] = 'not reviewed'
    destination = tmp_path / 'must-not-be-created'
    assert drafts.verify_capacity_record(damaged, expected_metadata_sha=expected)
    with pytest.raises(ValueError):
        drafts.restore_capacity_backup(damaged, destination, expected_metadata_sha=expected)
    assert not destination.exists()


def test_offline_recovery_preserves_original_bom_crlf(monkeypatch, tmp_path):
    result, original = stage_fixture(monkeypatch, raw_layout=True)
    assert result.get('staged'), result
    record = wa.get_pending('memory', result['pending_id'])
    drafts = importlib.import_module('agent.background_review_capacity_drafts')
    destination = tmp_path / 'bom-recovery'
    drafts.restore_capacity_backup(record, destination,
                                  expected_metadata_sha=drafts.capacity_metadata_sha(record))
    assert original.startswith(b'\xef\xbb\xbf') and b'\r\n' in original
    assert (destination / 'MEMORY.md').read_bytes() == original


def test_capacity_scope_does_not_leak_into_next_ordinary_review(monkeypatch):
    first, _ = stage_fixture(monkeypatch)
    second, _ = stage_fixture(monkeypatch, include_plan=False)
    assert wa.get_pending('memory', first['pending_id'])['metadata']['capacity']
    assert 'metadata' not in wa.get_pending('memory', second['pending_id'])


def test_offline_restore_never_overwrites_existing_destination(monkeypatch, tmp_path):
    result, _ = stage_fixture(monkeypatch)
    record = wa.get_pending('memory', result['pending_id'])
    drafts = importlib.import_module('agent.background_review_capacity_drafts')
    destination = tmp_path / 'existing'
    destination.mkdir()
    (destination / 'MEMORY.md').write_bytes(b'UNRELATED ORIGINAL')
    with pytest.raises(FileExistsError):
        drafts.restore_capacity_backup(record, destination,
                                      expected_metadata_sha=drafts.capacity_metadata_sha(record))
    assert (destination / 'MEMORY.md').read_bytes() == b'UNRELATED ORIGINAL'


def test_draft_cannot_be_replayed_by_normal_pending_approval(monkeypatch):
    result, original = stage_fixture(monkeypatch)
    record = wa.get_pending('memory', result['pending_id'])
    from tools.memory_tool import apply_memory_pending

    store = MemoryStore(memory_char_limit=10000, user_char_limit=5000)
    store.load_from_disk()
    reply = apply_memory_pending(record['payload'], store)
    assert reply['success'] is False
    assert (get_memory_dir() / 'MEMORY.md').read_bytes() == original
    assert wa.get_pending('memory', result['pending_id']) is not None
