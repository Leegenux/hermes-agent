"""Capacity staging must not request ordinary human approval after automatic commit.

Regression: verified native capacity metadata was still returned with the ordinary
proposal_staged=True/manual-approve envelope; the native summary faithfully echoed it.
Real tool/store/worker/locks, deterministic provider only. No gate relaxation.
"""
import json

import pytest

from agent import background_review as native
from agent.background_review_capacity_backups import capacity_outcome_actions
from tests.agent.test_background_review_capacity_drafts import stage_fixture
from tests.agent.test_background_review_capacity_commit import run_commit
from tools import write_approval as wa


def messages(reply):
    return [{'role': 'assistant', 'tool_calls': [{
        'id': 'native-memory-proposal', 'type': 'function',
        'function': {'name': 'memory', 'arguments': json.dumps({
            'target': 'memory', 'operations': [{'action': 'patch',
                                               'old_text': 'original', 'content': 'candidate'}]})}}]},
        {'role': 'tool', 'tool_call_id': 'native-memory-proposal',
         'content': json.dumps(reply)}]


def test_capacity_memory_stage_is_not_an_ordinary_human_approval_request(monkeypatch):
    reply, _ = stage_fixture(monkeypatch)
    assert reply['success'] and reply['staged']
    record = wa.get_pending('memory', reply['pending_id'])
    assert record['metadata']['capacity']['state'] == 'draft'
    assert reply['proposal_staged'] is False
    assert 'capacity draft' in reply['message'].lower()
    assert '/memory pending' not in reply['message']
    assert native.summarize_background_review_actions(messages(reply), [], 'on') == []


@pytest.mark.parametrize('options', [{'include_plan': False}, {'explicit': True}])
def test_ordinary_and_explicit_background_proposals_keep_human_approval(monkeypatch, options):
    reply, _ = stage_fixture(monkeypatch, **options)
    record = wa.get_pending('memory', reply['pending_id'])
    assert 'capacity' not in record.get('metadata', {})
    assert reply['staged'] and reply['proposal_staged'] is True
    assert 'staged for your approval' in reply['message']
    assert f'{wa.pending_count(wa.MEMORY)} pending write(s) now await review' in reply['message']
    assert '/memory pending' in reply['message']
    assert native.summarize_background_review_actions(messages(reply), [], 'on') == [reply['message']]


@pytest.mark.parametrize('targets', [('memory',), ('user',)])
@pytest.mark.parametrize('authority', [True, False])
def test_native_memory_and_user_outcomes_never_mix_capacity_commit_with_human_approval(
        monkeypatch, targets, authority):
    _, before, after, delivered, scope, stages = run_commit(
        monkeypatch, targets=targets, authority=authority)
    assert stages and all(reply['proposal_staged'] is False for reply in stages)
    actions = [line for reply in stages for line in
               native.summarize_background_review_actions(messages(reply), [], 'on')]
    actions.extend(capacity_outcome_actions(scope, 'on'))
    assert not any('staged for your approval' in line or '/memory pending' in line for line in actions)
    assert actions and delivered
    if authority:
        assert len(scope.commit_results) == 1
        assert after[targets[0]] != before[targets[0]]
        assert any('1 memory commit(s) verified' in line for line in actions)
    else:
        assert before == after and not scope.commit_results
        assert any('not installed' in line for line in actions)
        assert not any('commit(s) verified' in line for line in actions)
