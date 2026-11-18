"""Runtime draft protocol reaches the real worker only after verified originals.
No real provider or model calls; native pending/tools/files remain exercised.
"""
import copy
import json

import pytest

from agent import background_review as native
from agent import background_review_capacity_backups as backups
from agent.background_review_capacity_drafts import _SCOPE
from tests.agent.test_background_review_capacity_drafts import stage_fixture
from tests.agent.test_background_review_capacity_skill_drafts import stage_skill_fixture
from tools import write_approval as wa

MARKER = 'CAPACITY_DRAFT_SUBMISSION_JSON\n'


def capture_native_fork_prompt(monkeypatch):
    worker = native._run_review_in_thread
    seen = []

    def observed_worker(*args, **kwargs):
        original_fork = native._run_review_fork

        def inspect(parent, messages, prompt, *rest):
            scope = _SCOPE.get()
            seen.append((prompt, scope))
            return original_fork(parent, messages, prompt, *rest)

        monkeypatch.setattr(native, '_run_review_fork', inspect)
        return worker(*args, **kwargs)

    monkeypatch.setattr(native, '_run_review_in_thread', observed_worker)
    return seen


@pytest.mark.parametrize('kind', ['memory', 'skills'])
def test_worker_explains_native_pending_submission_only_after_intact_prebackup(monkeypatch, kind):
    seen = capture_native_fork_prompt(monkeypatch)
    if kind == 'memory':
        reply, _ = stage_fixture(monkeypatch)
    else:
        reply, _, before, after, _ = stage_skill_fixture(monkeypatch)
        assert before == after
    assert reply.get('staged')
    prompt, scope = seen[0]
    assert MARKER in prompt, 'Native fork still receives no usable draft-submission protocol'
    payload = json.loads(prompt.split(MARKER, 1)[1].splitlines()[0])
    assert payload['proposal_destination'] == 'native_pending'
    assert payload['installation_authorized'] is False
    assert payload['ordinary_approval_unchanged'] is True
    expected = {(t.kind, t.name, t.target_chars, t.strictly_below) for t in scope.plan.targets}
    actual = {(t['kind'], t['target'], t['target_chars'], t['strictly_below'])
              for t in payload['targets']}
    assert actual == expected
    assert all(t['backup_verified'] is True for t in payload['targets'])
    assert 'Do not stop at a text-only proposal' in prompt
    assert 'even when no new conversation fact' in prompt
    record = wa.get_pending(kind, reply['pending_id'])
    assert record['metadata']['capacity']['state'] == 'draft'
    assert record['metadata']['capacity']['independent_review_passed'] is False
    assert not scope.commit_results


@pytest.mark.parametrize('damage', ['missing_reference', 'record_missing', 'record_changed', 'backup_error'])
def test_worker_never_advertises_staging_ready_for_unverifiable_backup(monkeypatch, damage):
    real_prepare = backups.prepare_capacity_backups

    def altered_prepare(scope):
        real_prepare(scope)
        key = ('memory', 'memory')
        if damage == 'missing_reference':
            scope.backups.clear()
        elif damage == 'backup_error':
            scope.backup_errors[key] = 'Synthetic original-readback failure'
        else:
            getter = wa.get_pending
            backup_id = scope.backups[key]['id']

            def altered_get(subsystem, pending_id):
                result = getter(subsystem, pending_id)
                if pending_id == backup_id:
                    if damage == 'record_missing':
                        return None
                    result = copy.deepcopy(result)
                    result['origin'] = 'foreground'
                return result

            monkeypatch.setattr(wa, 'get_pending', altered_get)

    monkeypatch.setattr(backups, 'prepare_capacity_backups', altered_prepare)
    seen = capture_native_fork_prompt(monkeypatch)
    reply, _ = stage_fixture(monkeypatch)
    assert not reply.get('staged') and reply.get('success') is False
    assert len(seen) == 1 and MARKER not in seen[0][0]


@pytest.mark.parametrize('options', [{'include_plan': False}, {'explicit': True}])
def test_ordinary_or_explicit_review_gets_no_runtime_capacity_submission_notice(monkeypatch, options):
    seen = capture_native_fork_prompt(monkeypatch)
    reply, _ = stage_fixture(monkeypatch, **options)
    assert reply.get('staged')
    assert len(seen) == 1 and MARKER not in seen[0][0]
    assert 'capacity' not in wa.get_pending('memory', reply['pending_id']).get('metadata', {})
