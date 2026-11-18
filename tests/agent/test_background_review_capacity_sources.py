"""All-source capacity behavior with native catalog, registry, tools, store and files.

Only the model provider is deterministic; no model calls or production writes.
Removing external identity admission loses commits; weakening registry/profile or
approval/lock checks would install into disabled, foreign or stale packages.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import background_review as native
from agent.background_review_capacity import capture_skill_sources, inspect_skill_capacity
from agent.background_review_capacity_drafts import _SCOPE
from hermes_constants import get_hermes_home
from tools import skill_manager_tool as smt, skills_tool, write_approval as wa

HEADER = '---\nname: capacity-probe\ndescription: Use when testing capacity.\n---\n'
CHILD = 'references/nested/rules.md'
CANDIDATE = 'Owner remains upstream. Retry limit = 7. No business execution.\n'


def seed(monkeypatch, tmp_path, kind, *, disabled=False, backup=False, foreign=False):
    home = get_hermes_home()
    base = home / 'skills' if kind == 'local' else tmp_path / 'installed' / kind
    if backup:
        base = base / 'backups'
    if foreign:
        default = tmp_path / 'default-home'
        monkeypatch.setattr('hermes_constants.get_default_hermes_root', lambda: default)
        base = default / 'profiles/other/skills'
    root = base / 'category/capacity-probe'
    (root / 'references/nested').mkdir(parents=True)
    (root / 'assets').mkdir()
    (root / 'scripts/empty').mkdir(parents=True)
    (root / 'SKILL.md').write_bytes(HEADER.encode() + b'Keep upstream attribution.\n')
    (root / CHILD).write_bytes(CANDIDATE.encode() * 200)
    (root / 'assets/raw.bin').write_bytes(bytes(range(256)))
    (root / 'scripts/do-not-run.py').write_bytes(b'raise RuntimeError("NEVER EXECUTE")\n')
    cfg = {'auxiliary': {'background_review': {'enabled': True, 'capacity_consolidation': True}},
           'skills': {}}
    name = 'capacity-probe'
    if kind == 'external':
        cfg['skills']['external_dirs'] = [str(base)]
    elif kind == 'create':
        cfg['skills']['create_dir'] = str(base)
    if kind == 'plugin':
        from hermes_cli import plugins
        from hermes_cli.plugins_manifest import PluginManifest
        pm = plugins.PluginManager(scope_key=str(home))
        ctx = plugins.PluginContext(PluginManifest(name='capacity-bundle', path=str(base)), pm)
        ctx.register_skill('capacity-probe', root / 'SKILL.md', frontmatter={
            'name': 'capacity-probe', 'description': 'Use when testing capacity.'})
        pm._discovered = True
        monkeypatch.setattr(plugins, 'get_plugin_manager', lambda: pm)
        name = 'capacity-bundle:capacity-probe'
    else:
        pm = None
    if disabled:
        cfg['skills']['disabled'] = [name]
    (home / 'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
    return home, root, name, pm


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def transaction(monkeypatch, root, name, *, damage=None):
    captured, delivered, failures = {}, [], []
    parent = SimpleNamespace(provider='synthetic', api_mode='chat_completions',
        memory_notifications='on', background_review_callback=delivered.append,
        _safe_print=lambda value: None, _emit_auxiliary_failure=lambda *a: failures.append(a),
        _active_children=[], _background_review_agent=None)
    builds = []

    class ProviderDouble:
        provider, model, base_url = 'synthetic', 'offline-fixture', 'offline-only'
        _memory_enabled = _user_profile_enabled = False
        _session_messages = []
        session_input_tokens, session_output_tokens, session_api_calls = 13, 2, 1
        _review_input_token_budget = 100000

        def __init__(self, is_critic):
            self.is_critic = is_critic

        def release_clients(self):
            pass

        def run_conversation(self, *, user_message, conversation_history):
            captured['scope'] = _SCOPE.get()
            if self.is_critic:
                packet = json.loads(user_message.partition('CAPACITY_REVIEW_PACKET_JSON\n')[2])
                checks = ('rules', 'parameters', 'conditions', 'permissions', 'ownership',
                          'business_conclusions', 'topics', 'code_units')
                if damage:
                    damage()
                return {'final_response': json.dumps({'decision': 'PASS',
                    'packet_sha256': packet['packet_sha256'],
                    'checks': dict.fromkeys(checks, True), 'issues': []})}
            for fp in (None, CHILD):
                viewed = json.loads(skills_tool.skill_view(name, file_path=fp, preprocess=False))
                assert viewed.get('success'), viewed
            reply = smt.skill_manage(action='', name='', operations=[{
                'action': 'patch', 'name': name, 'file_path': CHILD,
                'old_string': CANDIDATE * 200, 'new_string': CANDIDATE}])
            captured['reply'] = json.loads(reply)
            return {'final_response': ''}

    def builder(*a, **kw):
        builds.append(True)
        return ProviderDouble(len(builds) > 1), {}, False

    monkeypatch.setattr(native, 'build_cache_parity_fork', builder)
    target, _ = native.spawn_background_review_thread(parent, [], review_skills=True, task_cfg={})
    # Real AIAgent tools carry the review origin; the provider double invokes the
    # same tool directly, so bind its origin as the native tool executor does.
    from tools.skill_provenance import set_current_write_origin, reset_current_write_origin
    token = set_current_write_origin('background_review')
    try:
        target()
    finally:
        reset_current_write_origin(token)
    assert not failures, failures
    return captured, delivered


@pytest.mark.parametrize('kind', ['local', 'external', 'create', 'plugin'])
def test_enabled_installed_source_is_discovered_and_committed_with_complete_history(monkeypatch, tmp_path, kind):
    home, root, name, _ = seed(monkeypatch, tmp_path, kind)
    before = snapshot(root)
    captured, delivered = transaction(monkeypatch, root, name)
    assert captured['reply'].get('staged'), captured['reply']
    scope = captured['scope']
    assert any(v['state'] == 'committed' for v in scope.commit_results.values()), scope.commit_errors
    assert (root / CHILD).read_bytes() == CANDIDATE.encode()
    assert (root / 'SKILL.md').read_bytes() == before['SKILL.md']
    assert snapshot(root)['assets/raw.bin'] == before['assets/raw.bin']
    assert snapshot(root)['scripts/do-not-run.py'] == before['scripts/do-not-run.py']
    assert (root / 'scripts/empty').is_dir()
    assert any('1 skill commit(s) verified' in str(x) for x in delivered), delivered
    from tools import skill_ledger as ledger
    entries = ledger.list_entries(skill=name)
    entries = [x for x in entries if x['action'] == 'patch']
    assert entries and entries[0]['before'] and entries[0]['after'], entries
    # Native ledger stores a delta, while the capacity backup stores the package.
    # A missing after-root would instead record every original as a deletion.
    assert {x['path'] for x in entries[0]['before']} == {str(root / CHILD)}
    assert {x['path'] for x in entries[0]['after']} == {str(root / CHILD)}
    assert ledger.read_blob(entries[0]['before'][0]['sha256']) == before[CHILD]
    assert ledger.read_blob(entries[0]['after'][0]['sha256']) == CANDIDATE.encode()
    from agent.background_review_capacity_skill_drafts import restore_skill_capacity_backup
    record = wa.get_pending('skills', next(iter(scope.records)))
    destination = tmp_path / 'offline-restored'
    restore_skill_capacity_backup(record, destination, expected_metadata_sha=next(iter(scope.records.values())))
    assert snapshot(destination) == before


@pytest.mark.parametrize('kind', ['local', 'external', 'create', 'plugin'])
@pytest.mark.parametrize('excluded', ['disabled', 'backup', 'foreign'])
def test_disabled_backup_and_other_profile_sources_never_enter_capacity_plan(monkeypatch, tmp_path, kind, excluded):
    _, root, _, _ = seed(monkeypatch, tmp_path, kind, **{excluded: True})
    plan = inspect_skill_capacity(capture_skill_sources())
    assert not any(t.documents[0].path.parent == root for t in plan.targets), plan


@pytest.mark.parametrize('kind', ['external', 'plugin'])
@pytest.mark.parametrize('damage', ['lock', 'approval', 'disabled', 'unregister', 'source'])
def test_external_commit_rechecks_current_identity_and_ordinary_guards(monkeypatch, tmp_path, kind, damage):
    home, root, name, pm = seed(monkeypatch, tmp_path, kind)
    before = snapshot(root)

    def mutate():
        if damage == 'lock':
            from tools import skill_usage
            skill_usage.set_locked(root.name, True)
        elif damage == 'approval':
            monkeypatch.setattr(wa, 'evaluate_gate', lambda *a, **kw: wa.GateDecision(stage=True))
        elif damage == 'disabled':
            cfg = json.loads((home / 'config.yaml').read_text())
            cfg['skills']['disabled'] = [name]
            (home / 'config.yaml').write_text(json.dumps(cfg))
        elif damage == 'unregister':
            if pm:
                pm.remove_plugin_skill(name)
            else:
                cfg = json.loads((home / 'config.yaml').read_text())
                cfg['skills']['external_dirs'] = []
                (home / 'config.yaml').write_text(json.dumps(cfg))
        else:
            (root / 'assets/raw.bin').write_bytes(b'UPSTREAM PACKAGE UPDATE')

    captured, _ = transaction(monkeypatch, root, name, damage=mutate)
    assert captured['reply'].get('staged'), captured['reply']
    assert not captured['scope'].commit_results
    assert captured['scope'].commit_errors
    after = snapshot(root)
    if damage == 'source':
        assert after['assets/raw.bin'] == b'UPSTREAM PACKAGE UPDATE'
        assert after[CHILD] == before[CHILD]
    else:
        assert after == before


@pytest.mark.parametrize('kind', ['external', 'create', 'plugin'])
def test_external_postwrite_failure_restores_exact_original_package(monkeypatch, tmp_path, kind):
    _, root, name, _ = seed(monkeypatch, tmp_path, kind)
    import utils
    real_atomic = utils.atomic_write_bytes
    before = snapshot(root)
    failed = False

    def injected(path, raw, **kw):
        nonlocal failed
        result = real_atomic(path, raw, **kw)
        if Path(path) == root / CHILD and not failed:
            failed = True
            raise OSError('Injected post-replacement failure')
        return result

    monkeypatch.setattr(utils, 'atomic_write_bytes', injected)
    captured, _ = transaction(monkeypatch, root, name)
    assert captured['reply'].get('staged')
    assert failed and snapshot(root) == before
    assert not captured['scope'].commit_results
    assert any(x['state'] == 'rolled_back' for x in captured['scope'].commit_errors.values())
    assert (root / 'scripts/empty').is_dir()


@pytest.mark.parametrize('kind', ['external', 'plugin'])
def test_default_home_never_admits_nested_other_profile_even_if_registered(monkeypatch, tmp_path, kind):
    home, _, name, pm = seed(monkeypatch, tmp_path, kind)
    monkeypatch.setattr('hermes_constants.get_default_hermes_root', lambda: home)
    foreign = home / 'profiles/other/skills/capacity-probe'
    foreign.mkdir(parents=True)
    (foreign / 'SKILL.md').write_bytes(HEADER.encode() + b'x' * 10000)
    if kind == 'external':
        cfg = json.loads((home / 'config.yaml').read_text())
        cfg['skills']['external_dirs'] = [str(foreign.parent)]
        (home / 'config.yaml').write_text(json.dumps(cfg))
    else:
        from hermes_cli import plugins
        from hermes_cli.plugins_manifest import PluginManifest
        ctx = plugins.PluginContext(PluginManifest(name='capacity-bundle', path=str(foreign)), pm)
        ctx.register_skill('capacity-probe', foreign / 'SKILL.md')
    assert not any(p == foreign / 'SKILL.md' for _, p in capture_skill_sources())


@pytest.mark.parametrize('container', ['', 'skills', 'plugins'])
def test_plugin_registration_cannot_admit_profile_infrastructure_container(monkeypatch, tmp_path, container):
    home, _, _, pm = seed(monkeypatch, tmp_path, 'plugin')
    root = home / container if container else home
    root.mkdir(parents=True, exist_ok=True)
    main = root / 'SKILL.md'
    main.write_bytes(HEADER.encode() + b'x' * 10000)
    from hermes_cli import plugins
    from hermes_cli.plugins_manifest import PluginManifest
    ctx = plugins.PluginContext(PluginManifest(name='capacity-bundle', path=str(root)), pm)
    ctx.register_skill('capacity-probe', main)
    assert not any(p == main for _, p in capture_skill_sources())


def test_installed_explicitly_loadable_skill_is_not_excluded_by_prompt_relevance(monkeypatch, tmp_path):
    _, root, name, _ = seed(monkeypatch, tmp_path, 'local')
    # Environment/app relevance hides a skill from the prompt offer, but native
    # explicit skill_view still loads it; relevance is not installation/disable.
    monkeypatch.setattr(skills_tool, 'skill_matches_environment', lambda fm: False)
    monkeypatch.setattr(skills_tool, 'skill_matches_apps', lambda fm: False)
    assert json.loads(skills_tool.skill_view(name, preprocess=False))['success']
    assert any(p == root / 'SKILL.md' for _, p in capture_skill_sources())


def test_plugin_owner_disabled_excludes_registered_skill(monkeypatch, tmp_path):
    home, root, _, _ = seed(monkeypatch, tmp_path, 'plugin')
    cfg = json.loads((home / 'config.yaml').read_text())
    cfg['plugins'] = {'disabled': ['capacity-bundle']}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    assert not any(p == root / 'SKILL.md' for _, p in capture_skill_sources())


def test_plugin_resolver_does_not_expand_ordinary_mutation_permissions(monkeypatch, tmp_path):
    _, root, name, _ = seed(monkeypatch, tmp_path, 'plugin')
    before = snapshot(root)
    assert smt._find_skill(name) is None
    result = json.loads(smt.skill_manage(action='patch', name=name,
        file_path=CHILD, old_string=CANDIDATE * 200, new_string=CANDIDATE))
    assert not result.get('success')
    assert snapshot(root) == before


def test_plugin_registry_retarget_after_review_cannot_change_destination(monkeypatch, tmp_path):
    _, root, name, pm = seed(monkeypatch, tmp_path, 'plugin')
    other = tmp_path / 'new-package/capacity-probe'
    other.mkdir(parents=True)
    (other / 'SKILL.md').write_bytes(HEADER.encode() + b'Other installed version.\n')
    before = snapshot(root)

    def retarget():
        from hermes_cli import plugins
        from hermes_cli.plugins_manifest import PluginManifest
        ctx = plugins.PluginContext(PluginManifest(name='capacity-bundle', path=str(other)), pm)
        ctx.register_skill('capacity-probe', other / 'SKILL.md')

    captured, _ = transaction(monkeypatch, root, name, damage=retarget)
    assert captured['reply'].get('staged')
    assert not captured['scope'].commit_results and captured['scope'].commit_errors
    assert snapshot(root) == before
    assert snapshot(other) == {'SKILL.md': HEADER.encode() + b'Other installed version.\n'}


def test_plugin_skill_main_view_marks_exact_read_for_capacity_write_guard(monkeypatch, tmp_path):
    _, root, name, _ = seed(monkeypatch, tmp_path, 'plugin')
    from tools.skill_provenance import set_current_write_origin, reset_current_write_origin
    from tools.skill_manager_guards import _reset_background_review_read_marks, _background_review_has_read
    token = set_current_write_origin('background_review')
    try:
        _reset_background_review_read_marks()
        assert json.loads(skills_tool.skill_view(name, preprocess=False))['success']
        assert _background_review_has_read(root / 'SKILL.md')
    finally:
        reset_current_write_origin(token)
