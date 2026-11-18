"""Native review capacity admission. All data is synthetic and profile-isolated."""
from types import SimpleNamespace

import pytest

from agent import background_review as native
from tools.memory_tool import MemoryStore, get_memory_dir


@pytest.fixture(autouse=True)
def capacity_optin():
    # Admission contracts describe the explicitly enabled profile, not a new default.
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault('auxiliary', {}).setdefault('background_review', {})['capacity_consolidation'] = True
    save_config(cfg)


def run_preflight(monkeypatch, *, memory=0, user=0, review_memory=True,
                  enabled=True, user_enabled=True, limits=(10000, 5000)):
    directory = get_memory_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'MEMORY.md').write_text('m' * memory, encoding='utf-8')
    (directory / 'USER.md').write_text('u' * user, encoding='utf-8')
    store = MemoryStore(memory_char_limit=limits[0], user_char_limit=limits[1])
    store.load_from_disk()
    frozen = store.format_for_system_prompt('memory')
    agent = SimpleNamespace(_memory_store=store, _memory_enabled=enabled,
                            _user_profile_enabled=user_enabled)
    calls = []
    monkeypatch.setattr(native, '_run_review_in_thread', lambda *args, **kw: calls.append((args, kw)))
    target, prompt = native.spawn_background_review_thread(
        agent, [], review_memory=review_memory, task_cfg={})
    target()
    assert len(calls) == 1
    assert store.format_for_system_prompt('memory') == frozen
    assert (directory / 'MEMORY.md').read_text(encoding='utf-8') == 'm' * memory
    return calls[0][1].get('capacity_plan'), prompt


@pytest.mark.parametrize('memory,user,expected', [
    (8999, 4499, ()), (9000, 4499, ('memory',)),
    (9001, 4499, ('memory',)), (8999, 4500, ('user',)),
    (9000, 4500, ('memory', 'user')),
])
def test_only_high_water_memory_stores_are_admitted(monkeypatch, memory, user, expected):
    plan, _ = run_preflight(monkeypatch, memory=memory, user=user)
    actual = tuple(t.name for t in plan.targets) if plan else ()
    assert actual == expected
    if plan:
        assert {t.name: t.target_chars for t in plan.targets} == {
            name: (6000 if name == 'memory' else 3000) for name in expected}


def test_user_watermarks_follow_its_own_configured_limit(monkeypatch):
    plan, _ = run_preflight(monkeypatch, user=2250, limits=(10000, 2500))
    assert plan is not None
    assert [(t.name, t.target_chars) for t in plan.targets] == [('user', 1500)]


@pytest.mark.parametrize('scope,enabled,user_enabled', [
    (False, True, True), (True, False, False),
])
def test_disabled_memory_and_skill_only_review_do_not_admit_memory(
        monkeypatch, scope, enabled, user_enabled):
    plan, _ = run_preflight(monkeypatch, memory=9000, user=4500,
                            review_memory=scope, enabled=enabled, user_enabled=user_enabled)
    assert plan is None


def test_memory_flag_does_not_authorize_disabled_user_store(monkeypatch):
    plan, _ = run_preflight(monkeypatch, memory=9000, user=4500, user_enabled=False)
    assert plan is not None
    assert [t.name for t in plan.targets] == ['memory']


def test_preflight_is_fresh_when_worker_starts_not_cached_at_spawn(monkeypatch):
    directory = get_memory_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'MEMORY.md'
    path.write_text('m' * 8999, encoding='utf-8')
    store = MemoryStore(memory_char_limit=10000, user_char_limit=5000)
    store.load_from_disk()
    agent = SimpleNamespace(_memory_store=store, _memory_enabled=True,
                            _user_profile_enabled=False)
    calls = []
    monkeypatch.setattr(native, '_run_review_in_thread', lambda *a, **kw: calls.append(kw))
    target, _ = native.spawn_background_review_thread(agent, [], review_memory=True, task_cfg={})
    path.write_text('m' * 9000, encoding='utf-8')
    target()
    assert [t.name for t in calls[0]['capacity_plan'].targets] == ['memory']
    assert store._char_count('memory') == 8999  # parent cache was not refreshed
