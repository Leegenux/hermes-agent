"""Profile-local opt-in for the existing native capacity transaction contexts.

No payload/env/task_cfg grant. Disabled profiles keep ordinary review unchanged.
Revocation fences each existing native transaction boundary; rollback remains
permitted only by the underlying transaction's exact owned-byte checks.
"""
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from pathlib import Path

from hermes_constants import get_hermes_home

_POLICY_HOME = ContextVar('native_capacity_policy_home', default=None)


def profile_capacity_enabled(expected_home=None):
    from hermes_cli.config import load_config_readonly
    from hermes_cli.config_backend import ConfigBackendUnavailable
    from agent.background_review_capacity_package import direct_path

    try:
        home = get_hermes_home().resolve()
        if expected_home is not None and home != Path(expected_home).resolve():
            return False
        direct_path(get_hermes_home())
        direct_path(get_hermes_home() / 'config.yaml')
        config = load_config_readonly()
        auxiliary = config.get('auxiliary')
        task = auxiliary.get('background_review') if isinstance(auxiliary, dict) else None
        return isinstance(task, dict) and task.get('capacity_consolidation') is True
    except (Exception, ConfigBackendUnavailable):
        return False


def require_current_capacity_policy():
    """No extra policy for a separately granted trusted/manual host context."""
    home = _POLICY_HOME.get()
    if home is not None and not profile_capacity_enabled(home):
        raise ValueError('Current-profile capacity opt-in revoked or unavailable')


@contextmanager
def capacity_profile_authority(scope, *, explicit=False):
    if scope is None or explicit or not profile_capacity_enabled():
        yield
        return
    from agent.background_review_capacity_commit import memory_capacity_commit_authority
    from agent.background_review_capacity_skill_commit import skill_capacity_commit_authority

    home = get_hermes_home().resolve()
    token = _POLICY_HOME.set(home)
    try:
        with ExitStack() as stack:
            stack.enter_context(memory_capacity_commit_authority(home))
            stack.enter_context(skill_capacity_commit_authority(home))
            yield
    finally:
        _POLICY_HOME.reset(token)
