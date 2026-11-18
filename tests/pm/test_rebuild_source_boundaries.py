"""Controlled direct-runtime and publication race regression tests."""
import pytest
from pm.plugin_inputs import Members, StagedUpdate


def test_direct_runtime_refuses_incompatible_modes(monkeypatch):
    from pm import client
    monkeypatch.setattr(client, 'is_runtime', lambda: True)
    for options in ({}, {'explicit': True, 'repair': True},
                    {'explicit': True, 'evict_incompatible_plugins': True},
                    {'explicit': True, 'plugins': StagedUpdate({})}):
        with pytest.raises(ValueError, match='source rebuild requires'):
            client.sync_venv(rebuild_source=True, **options)


def test_direct_runtime_rebuild_and_drift_does_not_publish(tmp_path, monkeypatch):
    from pm import client, install, paths
    from pm.environments import install_state_dir, runtime_facts_path, selected_venv
    from pm.lock import Facts
    repo = tmp_path / 'repo'
    repo.mkdir()
    store = tmp_path / 'tools'
    store.mkdir()
    monkeypatch.setattr(paths, 'repo_root', lambda: repo)
    monkeypatch.setattr(paths, 'store_root', lambda: store)
    monkeypatch.setattr(client, 'is_runtime', lambda: True)
    venv = type(install.get_package('venv'))
    state = {'stamp': 'first', 'calls': 0, 'drift': False}
    monkeypatch.setattr(venv, 'expected_stamp', lambda self, extras, **kwargs: state['stamp'])
    def apply(self, *args, **kwargs):
        state['calls'] += 1
        target = install_state_dir(repo) / 'environments' / str(state['calls']) / 'venv'
        target.mkdir(parents=True)
        (target / 'pyvenv.cfg').write_text('home = fixture\n', encoding='utf-8')
        if state['drift']:
            state['stamp'] = 'changed'
        return {'environment': target}
    monkeypatch.setattr(venv, 'apply', apply)
    client.sync_venv([], explicit=True, plugins=Members([]))
    first = selected_venv(repo)
    client.sync_venv(explicit=True, plugins=Members([]))
    assert state['calls'] == 1 and selected_venv(repo) == first
    client.sync_venv(explicit=True, plugins=Members([]), rebuild_source=True)
    second = selected_venv(repo)
    assert state['calls'] == 2 and second != first
    fact = Facts(runtime_facts_path(repo)).get('venv')
    state['drift'] = True
    with pytest.raises(ValueError, match='Dependency inputs changed'):
        client.sync_venv(explicit=True, plugins=Members([]), rebuild_source=True)
    assert state['calls'] == 3
    assert selected_venv(repo) == second
    assert Facts(runtime_facts_path(repo)).get('venv') == fact
