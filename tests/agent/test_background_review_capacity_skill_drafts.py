"""Real native skill_manage drafts; synthetic fork, zero model calls/business execution."""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import background_review as native
from agent.background_review_capacity import inspect_skill_capacity
from hermes_constants import get_hermes_home
from tools import skill_manager_tool as smt, write_approval as wa
from tools.skill_provenance import set_current_write_origin, reset_current_write_origin

HEADER = '---\nname: capacity-example\ndescription: Use when synthetic tests. Probe.\n---\n'
CHILD = 'references/nested/rules.md'


def stage_skill_fixture(monkeypatch, *, alteration=None, read_source=True,
                        include_plan=True, explicit=False, flat=False, raw_layout=False):
    root = get_hermes_home() / 'skills' / 'category' / 'capacity-example'
    (root / 'references/nested').mkdir(parents=True)
    (root / 'scripts/empty').mkdir(parents=True)
    (root / 'assets').mkdir()
    main = HEADER + '# Rules\nKeep safety and owner.\n' + 'x' * 10000
    raws = {'SKILL.md': main.encode(), CHILD: b'Keep the threshold = 7.\n',
            'scripts/must-not-run.py': b'raise RuntimeError("BUSINESS MUST NEVER RUN")\n',
            'assets/raw.bin': bytes(range(256))}
    if raw_layout:
        raws['SKILL.md'] = b'\xef\xbb\xbf' + raws['SKILL.md'].replace(b'\n', b'\r\n')
    for name, raw in raws.items():
        (root / name).write_bytes(raw)
    plan = inspect_skill_capacity((('category/capacity-example', root / 'SKILL.md'),))
    assert len(plan.targets) == 1 and not plan.errors
    candidate = HEADER + '# Rules\nKeep safety and owner.\n'
    ops = [{'action': 'patch', 'name': 'category/capacity-example', 'content': candidate}]
    if alteration == 'child-over-target':
        # Update disk AND freeze the correct source before requesting the partial rewrite.
        (root / CHILD).write_bytes(b'z' * 6000)
        plan = inspect_skill_capacity((('category/capacity-example', root / 'SKILL.md'),))
    elif alteration == 'source-script':
        (root / 'scripts/must-not-run.py').write_bytes(b'concurrent script revision')
    elif alteration == 'source-file-added':
        (root / 'assets/new.bin').write_bytes(b'concurrent asset')
    elif alteration == 'source-directory-added':
        (root / 'assets/new-empty').mkdir()
    elif alteration == 'source-main':
        (root / 'SKILL.md').write_bytes(raws['SKILL.md'] + b' concurrent instruction')
    elif alteration == 'candidate-6000':
        ops[0]['content'] = HEADER + 'y' * (6000 - len(HEADER))
    elif alteration == 'candidate-5999':
        ops[0]['content'] = HEADER + 'y' * (5999 - len(HEADER))
    elif alteration == 'new-link':
        ops[0]['content'] += '[Part](references/part.md)\n'
        ops.append({'action': 'write_file', 'name': 'capacity-example',
                    'file_path': 'references/part.md', 'file_content': 'Kept rule.\n'})
    elif alteration == 'broken-link':
        ops[0]['content'] += '[Missing](references/missing.md)\n'
    elif alteration == 'script-edit':
        ops.append({'action': 'write_file', 'name': 'capacity-example',
                    'file_path': 'scripts/must-not-run.py', 'file_content': 'not allowed'})
    elif alteration == 'remove':
        ops.append({'action': 'remove_file', 'name': 'capacity-example', 'file_path': CHILD})
    elif alteration == 'traversal':
        ops.append({'action': 'write_file', 'name': 'capacity-example',
                    'file_path': 'references/../outside.md', 'file_content': 'not allowed'})
    elif alteration == 'file-parent-collision':
        ops.append({'action': 'write_file', 'name': 'capacity-example',
                    'file_path': CHILD + '/child.md', 'file_content': 'impossible tree'})
    elif alteration == 'targeted':
        ops = [{'action': 'patch', 'name': 'capacity-example',
                'old_string': 'x' * 10000, 'new_string': ''}]
    elif alteration == 'targeted-miss':
        ops = [{'action': 'patch', 'name': 'capacity-example',
                'old_string': 'NEVER EXISTING MATCH 7c23403b', 'new_string': 'no'}]
    elif alteration == 'add-only':
        ops = [{'action': 'write_file', 'name': 'capacity-example',
                'file_path': 'references/new.md', 'file_content': 'not consolidation'}]
    elif alteration == 'denied':
        monkeypatch.setattr(wa, 'evaluate_gate', lambda *a: wa.GateDecision(blocked=True))
    elif alteration == 'locked':
        from tools import skill_usage
        skill_usage.set_locked('capacity-example', True)
    elif alteration == 'pending-lost':
        monkeypatch.setattr(wa, 'get_pending', lambda *a: None)
    elif alteration == 'backup-failure':
        from tools import skill_manager_batch
        monkeypatch.setattr(skill_manager_batch, '_snapshot_skills',
                            lambda *a: (None, 'synthetic complete snapshot failure'))
    elif alteration == 'backup-incomplete':
        from tools import skill_manager_batch
        copier = skill_manager_batch.shutil.copytree
        def incomplete_copy(source, destination, *args, **kwargs):
            kwargs['ignore'] = skill_manager_batch.shutil.ignore_patterns('raw.bin')
            return copier(source, destination, *args, **kwargs)
        monkeypatch.setattr(skill_manager_batch.shutil, 'copytree', incomplete_copy)
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    results, trusted, failures = [], [], []
    agent = SimpleNamespace(provider='openai', api_mode='chat_completions',
        memory_notifications='on', background_review_callback=None,
        _safe_print=lambda value: None, _emit_auxiliary_failure=lambda *a: failures.append(a))
    def synthetic_fork(parent, snapshot, prompt, cfg, run, state, *flags):
        from tools import skill_manager_guards as guards, skills_tool
        from agent.background_review_capacity_drafts import _SCOPE
        token = set_current_write_origin('background_review')
        try:
            guards._reset_background_review_read_marks()
            if read_source:
                viewed = json.loads(skills_tool.skill_view('category/capacity-example'))
                assert viewed.get('success'), viewed
                viewed = json.loads(skills_tool.skill_view('category/capacity-example', file_path=CHILD))
                assert viewed.get('success'), viewed
            raw = (smt.skill_manage(**ops[0]) if flat else
                   smt.skill_manage(action='', name='', operations=ops))
            if alteration == 'repeated':
                first = json.loads(raw)
                assert first.get('staged'), first
                raw = smt.skill_manage(action='', name='', operations=ops)
            results.append(json.loads(raw))
            scope = _SCOPE.get()
            trusted.append(dict(scope.records) if scope else {})
        finally:
            reset_current_write_origin(token)
    monkeypatch.setattr(native, '_run_review_fork', synthetic_fork)
    native._run_review_in_thread(agent, [], 'Synthetic skill review only', task_cfg={},
        explicit=explicit, capacity_plan=plan if include_plan else None)
    assert not failures, failures
    after = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    return results[0], root, before, after, trusted[0]


def test_native_skill_capacity_batch_stages_complete_package_without_live_write(monkeypatch):
    reply, root, before, after, trusted = stage_skill_fixture(monkeypatch)
    assert reply.get('staged'), reply
    assert before == after
    record = wa.get_pending('skills', reply['pending_id'])
    cap = record['metadata']['capacity']
    assert cap['kind'] == 'skill' and cap['state'] == 'draft'
    assert cap['independent_review_passed'] is False
    assert set(cap['source_manifest']) == set(before)
    assert set(cap['candidate_manifest']) == {'SKILL.md', CHILD}
    assert reply['pending_id'] in trusted
    assert all(v['chars'] < 6000 for v in cap['candidate_manifest'].values())


@pytest.mark.parametrize('alteration', ['child-over-target', 'source-script', 'source-file-added',
    'source-directory-added', 'source-main', 'candidate-6000', 'broken-link', 'script-edit',
    'remove', 'traversal', 'pending-lost', 'backup-failure', 'file-parent-collision',
    'targeted-miss', 'add-only', 'denied', 'backup-incomplete'])
def test_skill_capacity_failure_keeps_live_and_never_claims_staged(monkeypatch, alteration):
    reply, root, before, after, trusted = stage_skill_fixture(monkeypatch, alteration=alteration)
    assert reply.get('success') is False, reply
    assert not reply.get('staged') and not trusted
    assert before == after


@pytest.mark.parametrize('alteration', ['candidate-5999', 'new-link', 'targeted'])
def test_complete_small_candidate_and_new_linked_instruction_can_stage(monkeypatch, alteration):
    reply, _, before, after, _ = stage_skill_fixture(monkeypatch, alteration=alteration)
    assert reply.get('staged'), reply
    assert before == after
    cap = wa.get_pending('skills', reply['pending_id'])['metadata']['capacity']
    assert all(v['chars'] < 6000 for v in cap['candidate_manifest'].values())


def test_native_read_before_write_still_required_for_capacity_draft(monkeypatch):
    reply, _, before, after, trusted = stage_skill_fixture(monkeypatch, read_source=False)
    assert reply.get('success') is False
    assert before == after and not trusted


def test_native_locked_skill_still_refuses_capacity_draft(monkeypatch):
    reply, _, before, after, trusted = stage_skill_fixture(monkeypatch, alteration='locked')
    assert reply.get('success') is False and reply.get('_locked')
    assert before == after and not trusted


def test_flat_native_shape_cannot_escape_capacity_staging(monkeypatch):
    reply, _, before, after, _ = stage_skill_fixture(monkeypatch, flat=True)
    assert reply.get('staged'), reply
    assert before == after


@pytest.mark.parametrize('flags', [{'include_plan': False}, {'explicit': True}])
def test_unscoped_and_explicit_skill_writes_keep_native_behavior(monkeypatch, flags):
    reply, _, before, after, trusted = stage_skill_fixture(monkeypatch, **flags)
    assert reply.get('success') and not reply.get('staged'), reply
    assert before['SKILL.md'] != after['SKILL.md'] and not trusted


def test_whole_package_offline_restore_preserves_bytes_empty_dirs_and_raw_layout(monkeypatch, tmp_path):
    reply, _, before, after, trusted = stage_skill_fixture(monkeypatch, raw_layout=True)
    assert reply.get('staged'), reply
    drafts = importlib.import_module('agent.background_review_capacity_skill_drafts')
    record = wa.get_pending('skills', reply['pending_id'])
    destination = tmp_path / 'offline-package'
    drafts.restore_skill_capacity_backup(record, destination,
        expected_metadata_sha=trusted[reply['pending_id']])
    assert {p.relative_to(destination).as_posix(): p.read_bytes()
            for p in destination.rglob('*') if p.is_file()} == before == after
    assert (destination / 'scripts/empty').is_dir()
    assert (destination / 'SKILL.md').read_bytes().startswith(b'\xef\xbb\xbf')
    assert b'\r\n' in (destination / 'SKILL.md').read_bytes()


def test_normal_pending_approval_cannot_install_unreviewed_skill_capacity_draft(monkeypatch):
    reply, root, before, _, trusted = stage_skill_fixture(monkeypatch)
    assert reply.get('staged'), reply
    record = wa.get_pending('skills', reply['pending_id'])
    replay = json.loads(smt.apply_skill_pending(record['payload']))
    assert replay.get('success') is False
    assert (root / 'SKILL.md').read_bytes() == before['SKILL.md']
    assert wa.get_pending('skills', reply['pending_id']) is not None


@pytest.mark.parametrize('damage', ['metadata', 'payload', 'backup-bytes'])
def test_tampered_skill_draft_never_creates_offline_restore(monkeypatch, tmp_path, damage):
    reply, _, _, _, trusted = stage_skill_fixture(monkeypatch)
    assert reply.get('staged'), reply
    drafts = importlib.import_module('agent.background_review_capacity_skill_drafts')
    record = copy.deepcopy(wa.get_pending('skills', reply['pending_id']))
    if damage == 'metadata':
        record['metadata']['capacity']['source_manifest'].pop('assets/raw.bin')
    elif damage == 'payload':
        record['payload']['operations'][0]['content'] = 'not reviewed'
    else:
        cap = record['metadata']['capacity']
        (get_hermes_home() / cap['snapshot_relative'] / 'assets/raw.bin').write_bytes(b'damaged')
    destination = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError):
        drafts.restore_skill_capacity_backup(record, destination,
            expected_metadata_sha=trusted[reply['pending_id']])
    assert not destination.exists()


def test_same_native_review_cannot_create_duplicate_capacity_skill_drafts(monkeypatch):
    reply, _, before, after, trusted = stage_skill_fixture(monkeypatch, alteration='repeated')
    assert reply.get('success') is False and not reply.get('staged'), reply
    assert before == after
    assert len(trusted) == 1
    records = wa.list_pending('skills')
    assert len([r for r in records if r['action'] == 'capacity_backup']) == 1
    assert len([r for r in records if r['metadata']['capacity']['state'] == 'draft']) == 1


def test_offline_skill_restore_never_overwrites_existing_directory(monkeypatch, tmp_path):
    reply, _, _, _, trusted = stage_skill_fixture(monkeypatch)
    assert reply.get('staged'), reply
    drafts = importlib.import_module('agent.background_review_capacity_skill_drafts')
    destination = tmp_path / 'existing-directory'
    destination.mkdir()
    (destination / 'keep.bin').write_bytes(b'UNRELATED ORIGINAL')
    with pytest.raises(FileExistsError):
        drafts.restore_skill_capacity_backup(wa.get_pending('skills', reply['pending_id']), destination,
            expected_metadata_sha=trusted[reply['pending_id']])
    assert list(destination.iterdir()) == [destination / 'keep.bin']
    assert (destination / 'keep.bin').read_bytes() == b'UNRELATED ORIGINAL'


def test_capacity_draft_rejects_other_profile_source(monkeypatch, tmp_path):
    from agent.background_review_capacity_drafts import capacity_review_scope
    from agent.background_review_capacity_skill_drafts import stage_skill_capacity
    external = tmp_path / 'other-profile' / 'skills' / 'capacity-example'
    external.mkdir(parents=True)
    original = (HEADER + 'x' * 10000).encode()
    (external / 'SKILL.md').write_bytes(original)
    plan = inspect_skill_capacity((('capacity-example', external / 'SKILL.md'),))
    assert plan.targets and not plan.errors
    # Isolate native resolver ownership; no real other profile is opened or altered.
    monkeypatch.setattr(smt, '_find_skill', lambda name: {'path': external})
    token = set_current_write_origin('background_review')
    try:
        with capacity_review_scope(plan):
            reply = json.loads(stage_skill_capacity({'action': 'patch', 'name': 'capacity-example',
                'content': HEADER + '# Preserved rule\n'}))
    finally:
        reset_current_write_origin(token)
    assert reply.get('success') is False and not reply.get('staged')
    assert (external / 'SKILL.md').read_bytes() == original
    assert wa.list_pending('skills') == []


def test_offline_recovery_cannot_create_a_new_live_skill(monkeypatch):
    reply, _, _, _, trusted = stage_skill_fixture(monkeypatch)
    assert reply.get('staged'), reply
    drafts = importlib.import_module('agent.background_review_capacity_skill_drafts')
    destination = get_hermes_home() / 'skills' / 'unapproved-new-live-skill'
    with pytest.raises(ValueError):
        drafts.restore_skill_capacity_backup(wa.get_pending('skills', reply['pending_id']), destination,
            expected_metadata_sha=trusted[reply['pending_id']])
    assert not destination.exists()
