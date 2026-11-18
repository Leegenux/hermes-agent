"""Classic CLI without a gateway: ``HERMES_CLI_LOCAL=1`` runs the agent and its tools in this process.

The classic CLI is a gateway client: ``GatewayAgent`` and ``GatewaySessionStore`` send every turn to the
gateway, so a ``hermes update`` the agent runs is a gateway child and a gateway restart takes it (and the
conversation) down. This mode restores the in-process pair ``hermes_cli.oneshot`` still builds
(``run_agent.AIAgent`` plus the registry's ``SessionDB``). The rest of the CLI needs no change: its
``GatewaySessionStore`` and ``process_view`` checks already fall through to in-process code when those facades
are absent, and the updater's holder scan already spares its own non-gateway ancestry
(``_self_and_non_gateway_ancestor_pids``).

Covers the classic CLI (interactive, ``-q``/``-Q``) only: ``-z``, stream-json and the TUI stay gateway clients.

The flag is read once and removed from the environment: tools, the updater and any gateway that the updater
restarts must not inherit it.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

logger = logging.getLogger(__name__)

ENV_FLAG = "HERMES_CLI_LOCAL"

# ``_init_agent`` arguments only ``GatewayAgent`` takes; each drives a gateway protocol this mode lacks.
GATEWAY_ONLY_AGENT_KWARGS = frozenset({
    "push_reasoning", "connection_update_callback", "model_confirm_callback", "modal_withdraw_callback",
    "one_turn", "one_turn_reasoning", "one_turn_alias",
})

_enabled: bool | None = None
_announced = False


def enabled() -> bool:
    global _enabled
    if _enabled is None:
        raw = os.environ.pop(ENV_FLAG, "")
        _enabled = raw.strip().lower() in {"1", "true", "yes", "on"}
    return _enabled


def session_store():
    """The registry's one in-process handle. Announced on stderr, so ``-q`` stdout stays clean."""
    global _announced
    from hermes_state_registry import acquire
    store = acquire()
    if not _announced:
        _announced = True
        print("[local mode] No gateway: the agent and its tools run in this process.", file=sys.stderr, flush=True)
    return store


def create_titled_session(title):
    """``chat -c <title> --create-if-missing``: the row and user-provenance title, written here."""
    from hermes_state_ids import new_session_id
    store = session_store()
    session_id = store.create_session(new_session_id(datetime.now()), "cli")
    store.set_session_title(session_id, title)
    return session_id


def _default_toolsets():
    """The profile's ``cli`` toolsets. A gateway resolves these itself, so the CLI leaves the list None,
    which ``AIAgent`` would read as "every tool": that would ignore the user's platform tool config."""
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools
    return sorted(_get_platform_tools(load_config(), "cli"))


def build_agent(**kwargs):
    """Build the in-process agent from ``_init_agent``'s arguments, doing what the gateway does at session
    open: load plugins and MCP tools, which the classic CLI otherwise leaves to it."""
    from hermes_cli.mcp_startup import ensure_mcp_discovery_before_agent_build
    from hermes_cli.plugins import discover_plugins
    from run_agent import AIAgent

    discover_plugins()
    ensure_mcp_discovery_before_agent_build(logger=logger)
    if kwargs.get("enabled_toolsets") is None:
        kwargs["enabled_toolsets"] = _default_toolsets()
    return AIAgent(**{key: value for key, value in kwargs.items() if key not in GATEWAY_ONLY_AGENT_KWARGS})
