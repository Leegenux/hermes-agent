"""Native worker/store/locks/files; provider double only, never business execution.

Removing the commit hook loses real file changes; relaxing the gate or manifest
fence installs stale content; false rollback claims fail exact package checks.
"""
import base64
from contextlib import nullcontext
import importlib
import json
from pathlib import Path

import pytest

from agent import background_review_capacity_critic as critic
from hermes_constants import get_hermes_home
from tests.agent.test_background_review_capacity_critic import exercise
from tools import skill_manager_tool as smt, write_approval as wa


def run_skill_commit(monkeypatch, *, damage=None, authority=True, new_link=False, layout=False, deep=False):
    root = get_hermes_home() / 'skills/category/capacity-example'
    real_review = critic.review_capacity_drafts
    native_atomic = importlib.import_module('utils').atomic_write_bytes
    getter = wa.get_pending
    stages, captured = [], {}
    first = True
    if deep:
        import copy
        native_manage = smt.skill_manage
        def deep_input(*args, **kwargs):
            if kwargs.get('operations'):
                kwargs = copy.deepcopy(kwargs)
                for op in kwargs['operations']:
                    for slot in ('content', 'file_path'):
                        if isinstance(op.get(slot), str):
                            op[slot] = op[slot].replace('references/part.md', 'references/added/deep/part.md')
            return native_manage(*args, **kwargs)
        monkeypatch.setattr(smt, 'skill_manage', deep_input)

    def atomic(path, data, **kwargs):
        nonlocal first
        if not Path(path).is_relative_to(root):
            return native_atomic(path, data, **kwargs)
        if first:
            first = False
            native_atomic(path, data, **kwargs)
            if damage == 'foreign':
                Path(path).write_bytes(b'EXTERNAL SOURCE MUST SURVIVE')
            if damage in {'postwrite', 'foreign', 'rollback'}:
                raise OSError('synthetic write failure after replacement')
        elif damage == 'rollback':
            raise OSError('synthetic rollback failure')
        else:
            native_atomic(path, data, **kwargs)

    def after_review(*args, **kwargs):
        real_review(*args, **kwargs)
        from agent.background_review_capacity_drafts import _SCOPE
        scope = _SCOPE.get()
        captured['scope'] = scope
        captured['before'] = {p.relative_to(root).as_posix(): p.read_bytes()
                              for p in root.rglob('*') if p.is_file()}
        if damage in {'stage', 'denied'}:
            monkeypatch.setattr(wa, 'evaluate_gate', lambda *a, **k: wa.GateDecision(
                stage=damage == 'stage', blocked=damage == 'denied'))
        elif damage == 'locked':
            from tools import skill_usage
            skill_usage.set_locked('capacity-example', True)
        elif damage == 'source':
            (root / 'assets/raw.bin').write_bytes(b'CONCURRENT ASSET REVISION')
        elif damage == 'draft':
            draft_id = next(iter(scope.records))
            def changed(sub, pid):
                record = getter(sub, pid)
                if pid == draft_id and record:
                    record['summary'] += ' changed'
                return record
            monkeypatch.setattr(wa, 'get_pending', changed)
        elif damage == 'journal':
            def lost(sub, pid):
                record = getter(sub, pid)
                return None if record and record['action'] == 'skill_capacity_prepare' else record
            monkeypatch.setattr(wa, 'get_pending', lost)
        elif damage == 'scan':
            monkeypatch.setattr(smt, '_security_scan_skill', lambda path, **kw: 'synthetic blocked scan')
        elif damage == 'scan-unavailable':
            monkeypatch.setattr(smt, '_guard_agent_created_enabled', lambda: True)
            def unavailable(*a, **kw):
                raise OSError('synthetic configured scanner unavailable')
            monkeypatch.setattr(smt, 'scan_skill', unavailable)
        elif damage == 'readmark-lost':
            from tools.skill_manager_guards import _reset_background_review_read_marks
            _reset_background_review_read_marks()
        elif damage == 'lock-unavailable':
            def lock_failure(*a):
                raise OSError('synthetic native lock unavailable')
            monkeypatch.setattr(smt, '_skill_mutation_lock', lock_failure)
        elif damage == 'rollback-race':
            native_rename = Path.rename
            def race_after_quarantine(path, destination):
                value = native_rename(path, destination)
                if path == root / 'references/part.md':
                    (root / 'SKILL.md').write_bytes(b'EXTERNAL REVISION DURING ROLLBACK')
                return value
            monkeypatch.setattr(Path, 'rename', race_after_quarantine)
            monkeypatch.setattr(smt, '_security_scan_skill', lambda path, **kw: 'force rollback')
        elif damage == 'duplicate':
            cm = importlib.import_module('agent.background_review_capacity_skill_commit')
            real_commit = cm.commit_reviewed_skills
            def twice(run=None):
                real_commit(run)
                real_commit(run)
            monkeypatch.setattr(cm, 'commit_reviewed_skills', twice)
        elif damage == 'cancelled':
            args[3].cancel()
        monkeypatch.setattr(importlib.import_module('utils'), 'atomic_write_bytes', atomic)

    monkeypatch.setattr(critic, 'review_capacity_drafts', after_review)
    try:
        cm = importlib.import_module('agent.background_review_capacity_skill_commit')
    except ModuleNotFoundError as exc:
        if exc.name != 'agent.background_review_capacity_skill_commit':
            raise
        cm = None  # RED: exercise existing worker; assert absent commit behavior, not import failure.
    flags = {'raw_layout': layout}
    if new_link:
        flags['alteration'] = 'new-link'
    context = cm.skill_capacity_commit_authority(get_hermes_home()) if cm and authority else nullcontext()
    with context:
        result = exercise(monkeypatch, kind='skill', flags=flags)
    scope = captured['scope']
    stages.extend(r for r in wa.list_pending('skills') if r['action'].startswith('skill_capacity_'))
    after = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    return root, captured['before'], after, scope, stages, result


@pytest.mark.parametrize('new_link', [False, True])
def test_native_worker_installs_exact_reviewed_instruction_package(monkeypatch, new_link):
    root, before, after, scope, stages, result = run_skill_commit(monkeypatch, new_link=new_link)
    assert after['SKILL.md'] != before['SKILL.md']
    assert len(after['SKILL.md'].decode()) < 6000
    assert after['assets/raw.bin'] == before['assets/raw.bin']
    assert after['scripts/must-not-run.py'] == before['scripts/must-not-run.py']
    assert (root / 'scripts/empty').is_dir()
    if new_link:
        assert after['references/part.md'] == b'Kept rule.\n'
    assert scope.commit_results['skill:category/capacity-example']['state'] == 'committed'
    assert not scope.commit_errors
    assert {r['action'] for r in stages} >= {'skill_capacity_prepare', 'skill_capacity_observed'}
    assert result[3], 'native callback must surface the actual outcome'


def test_without_trusted_profile_activation_skill_commit_is_dormant(monkeypatch):
    _, before, after, scope, stages, _ = run_skill_commit(monkeypatch, authority=False)
    assert before == after and not scope.commit_results and not stages


@pytest.mark.parametrize('damage', ['stage', 'denied', 'locked', 'draft', 'journal', 'cancelled',
                                   'readmark-lost', 'lock-unavailable'])
def test_changed_native_gate_or_evidence_refuses_before_any_live_write(monkeypatch, damage):
    _, before, after, scope, _, _ = run_skill_commit(monkeypatch, damage=damage)
    assert before == after and not scope.commit_results
    assert scope.commit_errors['skill:category/capacity-example']['state'] == 'refused'


def test_full_package_asset_change_rejects_stale_proposal_without_overwrite(monkeypatch):
    _, before, after, scope, _, _ = run_skill_commit(monkeypatch, damage='source')
    assert after['assets/raw.bin'] == b'CONCURRENT ASSET REVISION'
    assert before['SKILL.md'] == after['SKILL.md']
    assert not scope.commit_results and scope.commit_errors


@pytest.mark.parametrize('damage', ['postwrite', 'scan', 'scan-unavailable'])
@pytest.mark.parametrize('new_link', [False, True])
def test_failed_skill_transaction_restores_full_original_bytes(monkeypatch, damage, new_link):
    _, before, after, scope, _, _ = run_skill_commit(
        monkeypatch, damage=damage, new_link=new_link, layout=True)
    assert after == before
    assert before['SKILL.md'].startswith(b'\xef\xbb\xbf') and b'\r\n' in before['SKILL.md']
    assert not scope.commit_results
    assert scope.commit_errors['skill:category/capacity-example']['state'] == 'rolled_back'


@pytest.mark.parametrize('damage', ['foreign', 'rollback'])
def test_unknown_external_bytes_or_failed_restore_reports_in_doubt(monkeypatch, damage):
    _, before, after, scope, _, _ = run_skill_commit(monkeypatch, damage=damage)
    assert not scope.commit_results
    assert scope.commit_errors['skill:category/capacity-example']['state'] == 'in_doubt'
    if damage == 'foreign':
        assert after['SKILL.md'] == b'EXTERNAL SOURCE MUST SURVIVE'
    else:
        assert after['SKILL.md'] != before['SKILL.md']


@pytest.mark.parametrize('damage', [None, 'scan'])
def test_new_deep_instruction_directories_commit_or_restore_without_losing_empty_dirs(monkeypatch, damage):
    root, before, after, scope, _, _ = run_skill_commit(
        monkeypatch, new_link=True, deep=True, damage=damage)
    if damage is None:
        assert after['references/added/deep/part.md'] == b'Kept rule.\n'
        assert scope.commit_results
    else:
        assert before == after and not (root / 'references/added').exists()
        assert scope.commit_errors['skill:category/capacity-example']['state'] == 'rolled_back'
    assert (root / 'scripts/empty').is_dir()


def test_duplicate_native_commit_does_not_repeat_writes_or_journal(monkeypatch):
    _, _, after, scope, stages, _ = run_skill_commit(monkeypatch, damage='duplicate')
    assert len(after['SKILL.md'].decode()) < 6000
    assert len([r for r in stages if r['action'] == 'skill_capacity_prepare']) == 1
    assert len(scope.commit_results) == 1


def test_skill_commit_callback_does_not_call_skill_transaction_a_memory_commit(monkeypatch):
    *_, result = run_skill_commit(monkeypatch)
    assert any('1 skill commit(s) verified' in str(a) for a in result[3])


def test_mixed_result_callback_keeps_in_doubt_visible():
    from types import SimpleNamespace
    from agent.background_review_capacity_drafts import CapacityDraftScope
    from agent.background_review_capacity_backups import capacity_outcome_actions
    scope = CapacityDraftScope(SimpleNamespace(errors=[]))
    scope.backups[('memory', 'memory')] = {'synthetic': True}
    scope.commit_results['memory'] = {'state': 'committed'}
    scope.commit_errors['skill:test'] = {'state': 'in_doubt', 'reason': 'synthetic'}
    text = ' '.join(capacity_outcome_actions(scope, 'on'))
    assert '1 memory commit(s) verified' in text and 'recovery required' in text


def test_skill_authority_cannot_be_minted_for_another_profile(tmp_path):
    from agent.background_review_capacity_skill_commit import skill_capacity_commit_authority
    with pytest.raises(ValueError):
        with skill_capacity_commit_authority(tmp_path / 'not-current-profile'):
            pytest.fail('wrong-profile grant was accepted')



def test_external_revision_between_rollback_steps_is_not_overwritten(monkeypatch):
    _, _, after, scope, _, _ = run_skill_commit(monkeypatch, damage='rollback-race', new_link=True)
    assert after['SKILL.md'] == b'EXTERNAL REVISION DURING ROLLBACK'
    assert scope.commit_errors['skill:category/capacity-example']['state'] == 'in_doubt'
