"""Native capacity worker -> native ledger blob/delta -> native isolated undo.

Missing pre-capture must not make an existing package look newly created in the
native history. Actual files/stores/rollback are used, not a fake ledger result.
"""
from pathlib import Path

import pytest

from tests.agent.test_background_review_capacity_skill_commit import run_skill_commit
from tools import skill_ledger as ledger


@pytest.mark.parametrize('new_link', [False, True])
def test_native_capacity_history_keeps_exact_before_bytes_and_real_undo(monkeypatch, new_link):
    root, before, after, scope, _, _ = run_skill_commit(
        monkeypatch, new_link=new_link, layout=True)
    assert scope.commit_results and before != after
    rows = [r for r in ledger.list_entries(skill='category/capacity-example') if r['action'] == 'patch']
    assert len(rows) == 1
    row = rows[0]
    assert row['actor'] == 'curator'
    b = {Path(item['path']).relative_to(root).as_posix(): ledger.read_blob(item['sha256'])
         for item in row['before']}
    a = {Path(item['path']).relative_to(root).as_posix(): ledger.read_blob(item['sha256'])
         for item in row['after']}
    assert b == {'SKILL.md': before['SKILL.md']}
    expected_after = {'SKILL.md': after['SKILL.md']}
    if new_link:
        expected_after['references/part.md'] = b'Kept rule.\n'
    assert a == expected_after, 'unchanged binary/scripts must not become after-only creations'
    ok, message = ledger.rollback_entry(row['id'])
    assert ok, message
    restored = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert restored == before
    assert (root / 'scripts/empty').is_dir()
    history = ledger.list_entries(skill='category/capacity-example')
    assert any(r['action'] == 'pre-rollback' for r in history)
    assert any(r['action'] == 'rollback' for r in history)


@pytest.mark.parametrize('failure', ['none', 'throw', 'incomplete', 'incomplete-asset', 'blob-corrupt'])
def test_enabled_native_history_capture_failure_refuses_capacity_install(monkeypatch, failure):
    real_capture = ledger.capture_before
    def broken(*args, **kwargs):
        if failure == 'none':
            return None
        if failure == 'throw':
            raise OSError('synthetic native history capture failure')
        items = real_capture(*args, **kwargs)
        assert items
        if failure == 'incomplete':
            return [i for i in items if Path(i['path']).name != 'SKILL.md']
        if failure == 'incomplete-asset':
            return [i for i in items if Path(i['path']).name != 'raw.bin']
        for item in items:
            if Path(item['path']).name == 'SKILL.md':
                (ledger.blobs_dir() / item['sha256']).write_bytes(b'CORRUPTED ORIGINAL BLOB')
        return items
    monkeypatch.setattr(ledger, 'capture_before', broken)
    _, before, after, scope, _, _ = run_skill_commit(monkeypatch)
    assert before == after and not scope.commit_results
    assert scope.commit_errors['skill:category/capacity-example']['state'] == 'refused'
    assert not ledger.list_entries(skill='category/capacity-example')


def test_disabled_native_ledger_not_silently_enabled_or_required(monkeypatch):
    monkeypatch.setattr(ledger, 'ledger_enabled', lambda: False)
    _, before, after, scope, _, _ = run_skill_commit(monkeypatch)
    assert before != after and scope.commit_results
    assert not ledger.list_entries(skill='category/capacity-example')


def test_failed_capacity_transaction_does_not_enter_native_history_as_success(monkeypatch):
    _, before, after, scope, _, _ = run_skill_commit(monkeypatch, damage='scan')
    assert before == after and not scope.commit_results
    assert not [r for r in ledger.list_entries(skill='category/capacity-example') if r['action'] == 'patch']


def test_native_history_exclusion_policy_cannot_silently_drop_modified_instruction():
    from agent.background_review_capacity import inspect_skill_capacity
    from agent.background_review_capacity_skill_commit import _capture_native_history
    from hermes_constants import get_hermes_home
    root = get_hermes_home() / 'skills/native-history-policy'
    child = root / 'references/env/rules.md'
    child.parent.mkdir(parents=True)
    header = b'---\nname: native-history-policy\ndescription: Use when synthetic tests. Probe.\n---\n'
    (root / 'SKILL.md').write_bytes(header + b'x' * 10000)
    child.write_bytes(b'original excluded instruction')
    plan = inspect_skill_capacity((('native-history-policy', root / 'SKILL.md'),))
    assert len(plan.targets) == 1 and not plan.errors
    candidate = {'SKILL.md': header + b'kept root rule',
                 'references/env/rules.md': b'original excluded instruction'}
    before = _capture_native_history(plan.targets[0], candidate)
    assert before and all('env' not in Path(i['path']).relative_to(root).parts for i in before)
    candidate['references/env/rules.md'] = b'changed excluded instruction'
    with pytest.raises(ValueError, match='excludes a modified instruction'):
        _capture_native_history(plan.targets[0], candidate)
    assert child.read_bytes() == b'original excluded instruction'
