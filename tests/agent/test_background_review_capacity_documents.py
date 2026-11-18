"""Capacity transactions must validate complete instruction sets, never just the hub."""
from types import SimpleNamespace

import pytest

from agent import background_review as native
from agent import background_review_capacity as capacity
from hermes_constants import get_hermes_home


HEADER = b'---\nname: capacity-example\ndescription: synthetic test\n---\n'


@pytest.fixture(autouse=True)
def capacity_optin():
    # Keep every existing document/path assertion; enable only this synthetic profile.
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault('auxiliary', {}).setdefault('background_review', {})['capacity_consolidation'] = True
    save_config(cfg)


def padded_main(chars):
    return HEADER + b'x' * (chars - len(HEADER))


def skill_fixture(monkeypatch, child_raw, *, main_chars=100):
    root = get_hermes_home() / 'skills' / 'category' / 'capacity-example'
    child = root / 'references' / 'nested' / 'rules.md'
    child.parent.mkdir(parents=True, exist_ok=True)
    (root / 'SKILL.md').write_bytes(padded_main(main_chars))
    child.write_bytes(child_raw)
    agent = SimpleNamespace(_memory_store=None, _memory_enabled=False, _user_profile_enabled=False)
    captured = []
    monkeypatch.setattr(native, '_run_review_in_thread', lambda *a, **kw: captured.append(kw))
    target, _ = native.spawn_background_review_thread(agent, [], review_skills=True, task_cfg={})
    target()
    return root, captured[0].get('capacity_plan')


@pytest.mark.parametrize('raw,trigger', [
    (b'x' * 9999, False), (b'x' * 10000, True),
    (b'\xef\xbb\xbf' + b'x' * 9999, False),
    (b'x' * 9998 + b'\r\n', False),
    (b'\xef\xbb\xbf' + b'x' * 9999 + b'\r\n', True),
], ids=['9999', '10000', 'bom-9999', 'crlf-9999', 'bom-crlf-10000'])
def test_native_skill_scope_sees_deep_documents_and_normalizes(monkeypatch, raw, trigger):
    root, plan = skill_fixture(monkeypatch, raw)
    assert bool(plan and plan.targets) is trigger
    if trigger:
        assert len(plan.targets) == 1
        target = plan.targets[0]
        assert target.kind == 'skill'
        assert target.strictly_below and target.target_chars == 6000
        assert {d.path.relative_to(root).as_posix() for d in target.documents} == {
            'SKILL.md', 'references/nested/rules.md'}


def test_invalid_utf8_is_an_explicit_inventory_failure_not_zero(monkeypatch):
    _, plan = skill_fixture(monkeypatch, b'\xff' * 10000)
    assert plan and plan.errors
    assert not plan.targets


@pytest.mark.parametrize('length,valid', [(5999, True), (6000, False)])
def test_every_skill_candidate_document_is_strictly_below_target(monkeypatch, length, valid):
    _, plan = skill_fixture(monkeypatch, b'x' * 10000)
    assert plan and plan.targets
    errors = capacity.validate_candidate(plan.targets[0], {
        'SKILL.md': padded_main(200), 'references/nested/rules.md': b'x' * length})
    assert (not errors) is valid


def test_missing_unchanged_document_cannot_be_omitted_from_candidate(monkeypatch):
    _, plan = skill_fixture(monkeypatch, b'x' * 10000)
    assert plan and plan.targets
    errors = capacity.validate_candidate(plan.targets[0], {'SKILL.md': padded_main(200)})
    assert errors


def test_validation_refuses_concurrent_source_change(monkeypatch):
    root, plan = skill_fixture(monkeypatch, b'x' * 10000)
    assert plan and plan.targets
    (root / 'references/nested/rules.md').write_bytes(b'NEW USER EDIT')
    assert capacity.validate_candidate(plan.targets[0], {
        'SKILL.md': padded_main(200), 'references/nested/rules.md': b'x' * 100})


@pytest.mark.parametrize('path', ['../outside.md', '/outside.md', 'C:/outside.md',
                                  'references/../../outside.md', 'references\\outside.md'])
def test_candidate_paths_cannot_escape_skill_transaction(monkeypatch, path):
    _, plan = skill_fixture(monkeypatch, b'x' * 10000)
    assert plan and plan.targets
    assert capacity.validate_candidate(plan.targets[0], {
        'SKILL.md': padded_main(200), 'references/nested/rules.md': b'x' * 100,
        path: b'cannot write outside'})


def test_new_instruction_document_is_validated_too(monkeypatch):
    _, plan = skill_fixture(monkeypatch, b'x' * 10000)
    assert plan and plan.targets
    assert capacity.validate_candidate(plan.targets[0], {
        'SKILL.md': padded_main(200), 'references/nested/rules.md': b'x' * 100,
        'references/other.md': b'x' * 6000})


def test_frontmatter_identity_is_not_removed_to_meet_budget(monkeypatch):
    _, plan = skill_fixture(monkeypatch, b'x' * 10000)
    assert plan and plan.targets
    assert capacity.validate_candidate(plan.targets[0], {
        'SKILL.md': b'# emptied identity', 'references/nested/rules.md': b'x' * 100})


def test_broken_local_instruction_link_is_rejected(monkeypatch):
    _, plan = skill_fixture(monkeypatch, b'x' * 10000)
    assert plan and plan.targets
    assert capacity.validate_candidate(plan.targets[0], {
        'SKILL.md': padded_main(200) + b'\n[required](references/missing.md)',
        'references/nested/rules.md': b'x' * 100})


@pytest.mark.parametrize('length,valid', [(6000, True), (6001, False)])
def test_memory_candidate_target_is_inclusive(tmp_path, length, valid):
    path = tmp_path / 'MEMORY.md'
    path.write_bytes(b'x' * 9000)
    doc = capacity.read_capacity_document(path)
    target = capacity.CapacityTarget('memory', 'memory', (doc,), 6000)
    assert (not capacity.validate_candidate(target, {'MEMORY.md': b'x' * length})) is valid


def test_new_live_document_invalidates_frozen_instruction_manifest(monkeypatch):
    root, plan = skill_fixture(monkeypatch, b'x' * 10000)
    assert plan and plan.targets
    (root / 'references/new-user-rule.md').write_bytes(b'New rule from another writer')
    assert capacity.validate_candidate(plan.targets[0], {
        'SKILL.md': padded_main(200), 'references/nested/rules.md': b'x' * 100})


def test_whitespace_is_not_a_nonempty_memory_candidate(tmp_path):
    path = tmp_path / 'MEMORY.md'
    path.write_bytes(b'x' * 9000)
    target = capacity.CapacityTarget('memory', 'memory',
                                    (capacity.read_capacity_document(path),), 6000)
    assert capacity.validate_candidate(target, {'MEMORY.md': b' \n\t '})
