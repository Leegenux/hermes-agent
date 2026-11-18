"""Offline real worker publication contract for explicit source rebuild."""
import pytest
from tests.pm.test_environment_build import locked_project, build_worker, client, isolated_python
from pm.plugin_inputs import Members


def test_source_rebuild_over_worker(locked_project, build_worker, tmp_path, monkeypatch):
    import pm
    from pm import paths
    from pm.environments import selected_venv, runtime_facts_path
    from pm.lock import Facts

    source, _, _ = locked_project
    manifest = source / 'pyproject.toml'
    manifest.write_text(manifest.read_text(encoding='utf-8-sig').replace('[tool.uv.workspace]\nmembers=["member"]\n', ''), encoding='utf-8')
    monkeypatch.setattr(paths, 'repo_root', lambda: source)
    pm.lock_project(source, offline=True, explicit=True)
    marker = source / 'rebuild_marker.py'
    marker.write_text('VALUE = 1\n', encoding='utf-8')
    pm.sync_venv(['chosen'], explicit=True, plugins=Members([]))
    first = selected_venv(source)
    old = Facts(runtime_facts_path(source)).get('venv')
    assert (first.parent / 'workspace' / 'rebuild_marker.py').read_text() == 'VALUE = 1\n'
    marker.write_text('VALUE = 2\n', encoding='utf-8')
    pm.sync_venv(explicit=True, plugins=Members([]))
    assert selected_venv(source) == first
    with pytest.raises(ValueError):
        pm.sync_venv(rebuild_source=True, plugins=Members([]))
    with pytest.raises(ValueError):
        pm.sync_venv(explicit=True, repair=True, rebuild_source=True)
    pm.sync_venv(explicit=True, plugins=Members([]), rebuild_source=True)
    second = selected_venv(source)
    new = Facts(runtime_facts_path(source)).get('venv')
    assert second != first
    assert new['stamp'] == old['stamp']
    assert new['extras'] == old['extras']
    assert (second.parent / 'workspace' / 'rebuild_marker.py').read_text() == 'VALUE = 2\n'
    marker.write_text('VALUE = 3\n', encoding='utf-8')
    pm.sync_venv(explicit=True, repair=True)
    third = selected_venv(source)
    assert third != second
    assert (third.parent / 'workspace' / 'rebuild_marker.py').read_text() == 'VALUE = 2\n'
    assert Facts(runtime_facts_path(source)).get('venv')['stamp'] == old['stamp']
