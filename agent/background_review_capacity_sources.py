"""Current-profile installed/loadable identities for native capacity transactions.

Reuse the native catalog and plugin registry, not a writable-root allowlist.
Registry membership is re-read at every transaction boundary; source bytes are
independently fenced by the existing full-package SHA checks. No write authority.
"""
from pathlib import Path

from hermes_constants import get_hermes_home
from agent.background_review_capacity_package import direct_path


_EXCLUDED_PARTS = frozenset({
    'backup', 'backups', '.backup', '.backups', '.archive', '.curator_backups',
    '.capacity-backups', '.capacity-failed', '__pycache__', '.git', 'node_modules'})


def _current_profile_path(path):
    """Configured roots must not turn backups or another profile into live skills."""
    from hermes_constants import get_default_hermes_root
    from agent.skill_utils import is_excluded_skill_path

    path = Path(path)
    if path.name != 'SKILL.md' or not path.is_file():
        return False
    if any(p.casefold() in _EXCLUDED_PARTS for p in path.parts):
        return False
    if is_excluded_skill_path(path):
        return False
    direct_path(path)
    resolved, home = path.resolve(), get_hermes_home().resolve()
    default = get_default_hermes_root().resolve()
    profiles = default / 'profiles'
    containers = {home, default, profiles, home / 'skills', home / 'plugins'}
    if resolved.parent in containers:
        return False
    if resolved.is_relative_to(profiles) and (home == default or not resolved.is_relative_to(home)):
        return False
    if home != default and any(resolved.is_relative_to(default / d) for d in ('skills', 'plugins')):
        return False
    if any(resolved.is_relative_to(home / d) for d in ('pending', 'memories', 'cron', 'cache')):
        return False
    return True


def active_skill_sources():
    """Exact enabled native load names, paths, precedence and plugin owner identity.

    Discovery belongs to the existing runtime; do not load/execute new plugin code
    here. Registered plugin sources may live outside home, e.g. entrypoint packages.
    """
    from tools import skills_tool as st
    from hermes_cli.config import load_config_readonly
    from hermes_cli.plugins import get_plugin_manager, _get_disabled_plugins
    from hermes_cli.plugins_content import plugin_skill_owners

    load_config_readonly()  # unreadable configuration must not imply enabled
    sources = {}
    # Native prompt relevance (apps/environment) is not a disable switch:
    # include installed skills that explicit skill_view can still load.
    for entry in st._skill_catalog(include_hidden=True):
        name, path = entry.get('load_name'), entry['path']
        if (name and _current_profile_path(path)
                and st.skill_matches_platform(st._safe_frontmatter(path))):
            sources[name] = Path(path)
    manager = get_plugin_manager()
    if Path(manager.home_path).resolve() != get_hermes_home().resolve():
        raise ValueError('Plugin registry belongs to another profile')
    disabled_plugins = _get_disabled_plugins()
    disabled_skills = st._get_disabled_skill_names()
    for item in manager.list_plugin_skill_metadata():
        name = item['name']
        if name in disabled_skills or plugin_skill_owners(manager, name) & disabled_plugins:
            continue
        path = manager.find_plugin_skill(name)
        if path is None or not _current_profile_path(path):
            continue
        if not st.skill_matches_platform(st._safe_frontmatter(path)):
            continue
        if name in sources and sources[name].resolve() != Path(path).resolve():
            raise ValueError('Conflicting active skill identity')
        sources[name] = Path(path)
    return tuple(sources.items())


def resolve_active_skill(name):
    """Resolve an alias exactly as native skill_view, then require active inventory membership."""
    from tools import skills_tool as st

    if st.skill_lookup_path_error(name):
        return None
    inventory = dict(active_skill_sources())
    if name in inventory:
        return inventory[name].parent
    # Preserve local category/basename/frontmatter aliases without granting plugin
    # namespace aliases to unrelated local packages or shadowed copies.
    if ':' in name:
        return None
    roots, _ = st.skill_search_dirs()
    error, directory, main = st.locate_skill(name, None, roots)
    if error is not None or directory is None:
        return None
    if any(main.resolve() == p.resolve() for p in inventory.values()):
        return directory
    return None


def require_active_skill(target):
    root = target.documents[0].path.parent
    direct_path(root)
    active = resolve_active_skill(target.name)
    if active is None or active.resolve() != root.resolve():
        raise ValueError('Capacity skill is no longer an enabled installed current-profile identity')
    return root


def find_scoped_capacity_skill(name):
    """Native manager bridge ONLY for a frozen package in the active capacity scope.

    Ordinary skill_manage keeps its existing resolver and permissions. This bridge
    lets the same lock/read/ledger hooks resolve namespaced package skills during
    the reviewed capacity transaction without pretending they are locally created.
    """
    from agent.background_review_capacity_drafts import _SCOPE

    scope = _SCOPE.get()
    if scope is None:
        return None
    root = resolve_active_skill(name)
    if root is None:
        return None
    matches = [t for t in scope.plan.targets if t.kind == 'skill'
               and t.documents[0].path.parent.resolve() == root.resolve()]
    return {'path': root} if len(matches) == 1 else None
