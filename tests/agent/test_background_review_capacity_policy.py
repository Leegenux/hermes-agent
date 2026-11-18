"""Native opt-in must authorize only the current profile's actual capacity run.

Real config backend + worker/tools/files; provider double only. Remove the native
policy hook -> enabled installs fail. Trust task_cfg/loose truthiness -> disabled
factory cases fail. Stop checking revocation -> original-byte assertions fail.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib

import pytest

from agent import background_review as native
from hermes_constants import get_hermes_home
from hermes_cli.config import load_config, save_config
from tests.agent.test_background_review_capacity_commit import run_commit
from tests.agent.test_background_review_capacity_skill_commit import run_skill_commit
from tests.agent.test_background_review_capacity_drafts import stage_fixture
from tools.memory_tool import MemoryStore
from tools import write_approval as wa


def policy(value):
    cfg = load_config()
    cfg.setdefault('auxiliary', {}).setdefault('background_review', {})['capacity_consolidation'] = value
    save_config(cfg)


@pytest.mark.parametrize('target', ['memory', 'user'])
@pytest.mark.parametrize('enabled', [True, False, None, 'true', 1])
def test_actual_profile_config_enables_native_memory_commit_without_test_authority(
        monkeypatch, target, enabled):
    policy(enabled)
    _, before, after, _, scope, _ = run_commit(monkeypatch, targets=(target,), authority=False)
    assert (after[target] != before[target]) is (enabled is True)
    assert bool(scope.commit_results) is (enabled is True)
    assert importlib.import_module('agent.background_review_capacity_commit')._AUTHORITY.get() is None


@pytest.mark.parametrize('damage', [None, 'stage', 'denied'])
def test_enabled_native_skill_commit_still_requires_ordinary_allow(monkeypatch, damage):
    policy(True)
    _, before, after, scope, _, _ = run_skill_commit(monkeypatch, authority=False, damage=damage)
    assert (before != after) is (damage is None)
    assert bool(scope.commit_results) is (damage is None)
    assert importlib.import_module('agent.background_review_capacity_skill_commit')._AUTHORITY.get() is None


@pytest.mark.parametrize('enabled', [True, False, None, 'true', 1])
def test_factory_uses_real_profile_optin_not_supplied_task_config(monkeypatch, enabled):
    policy(enabled)
    store = MemoryStore(memory_char_limit=10000, user_char_limit=5000)
    store.load_from_disk()
    assert store.add('memory', 'Public fixture. ' + 'x' * 9000)['success']
    agent = SimpleNamespace(_memory_store=store, _memory_enabled=True, _user_profile_enabled=False)
    calls = []
    monkeypatch.setattr(native, '_run_review_in_thread', lambda *a, **kw: calls.append(kw))
    target, _ = native.spawn_background_review_thread(agent, [], review_memory=True,
        task_cfg={'enabled': True, 'capacity_consolidation': True})
    target()
    assert len(calls) == 1
    plan = calls[0].get('capacity_plan')
    assert bool(plan) is (enabled is True)
    if enabled is True:
        assert [item.name for item in plan.targets] == ['memory']


def test_enabled_profile_does_not_enable_another_profile(monkeypatch, tmp_path):
    policy(True)
    other = tmp_path / 'other-profile'
    other.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(other))
    _, before, after, _, scope, _ = run_commit(monkeypatch, authority=False)
    assert before == after and not scope.commit_results
    assert not load_config().get('auxiliary', {}).get('background_review', {}).get('capacity_consolidation')


def test_factory_aborts_if_owning_profile_changes_before_worker_entry(monkeypatch, tmp_path):
    policy(True)
    owner = get_hermes_home()
    store = MemoryStore(memory_char_limit=10000, user_char_limit=5000)
    store.load_from_disk()
    assert store.add('memory', 'Public fixture. ' + 'x' * 9000)['success']
    agent = SimpleNamespace(_memory_store=store, _memory_enabled=True, _user_profile_enabled=False)
    calls = []
    monkeypatch.setattr(native, '_run_review_in_thread', lambda *a, **kw: calls.append(kw))
    target, _ = native.spawn_background_review_thread(agent, [], review_memory=True, task_cfg={})
    other = tmp_path / 'other-profile'
    other.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(other))
    policy(True)
    target()
    assert not calls, 'Captured owner must not be rebound to another profile that also opted in'
    assert (owner / 'memories/MEMORY.md').read_text().startswith('Public fixture.')


@pytest.mark.parametrize('kind', ['memory', 'skills'])
@pytest.mark.parametrize('when', ['prepare', 'observed'])
def test_revoked_profile_optin_refuses_or_rolls_back_without_clobbering_original(
        monkeypatch, kind, when):
    policy(True)
    genuine = wa.stage_write
    action = ('capacity_transaction_' if kind == 'memory' else 'skill_capacity_') + when
    seen = []
    def revoke(subsystem, payload, *args, **kwargs):
        result = genuine(subsystem, payload, *args, **kwargs)
        if payload.get('action') == action:
            seen.append(action)
            policy(False)
        return result
    monkeypatch.setattr(wa, 'stage_write', revoke)
    if kind == 'memory':
        _, before, after, _, scope, _ = run_commit(monkeypatch, authority=False, layout=True)
    else:
        _, before, after, scope, _, _ = run_skill_commit(monkeypatch, authority=False, layout=True)
    assert seen, 'Native opt-in never reached its real transaction boundary'
    assert before == after and not scope.commit_results
    assert scope.commit_errors
    assert {item['state'] for item in scope.commit_errors.values()} <= {'refused', 'rolled_back'}


def test_profile_optin_does_not_turn_explicit_review_into_capacity_exception(monkeypatch):
    policy(True)
    result, _ = stage_fixture(monkeypatch, explicit=True)
    assert result['staged'] and result['proposal_staged'] is True
    assert 'capacity' not in wa.get_pending('memory', result['pending_id']).get('metadata', {})
