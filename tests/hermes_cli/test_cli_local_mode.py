"""HERMES_CLI_LOCAL: the classic CLI's agent and store run in this process, never dialing a gateway."""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import cli_local_mode, gateway_session_store


@pytest.fixture
def local_flag(monkeypatch):
    monkeypatch.setattr(cli_local_mode, "_enabled", None)
    monkeypatch.setattr(cli_local_mode, "_announced", False)
    monkeypatch.delenv(cli_local_mode.ENV_FLAG, raising=False)
    return monkeypatch


@pytest.mark.parametrize("raw, expected", [("1", True), ("true", True), (" ON ", True), ("0", False), ("", False)])
def test_flag_values(local_flag, raw, expected):
    local_flag.setenv(cli_local_mode.ENV_FLAG, raw)
    assert cli_local_mode.enabled() is expected


def test_flag_is_not_inherited_by_children(local_flag):
    """A gateway the updater restarts from this CLI must not come up in local mode."""
    import os
    local_flag.setenv(cli_local_mode.ENV_FLAG, "1")
    assert cli_local_mode.enabled() is True
    assert cli_local_mode.ENV_FLAG not in os.environ
    assert cli_local_mode.enabled() is True  # still on for this process


def test_store_is_a_session_db_and_never_dials(local_flag, tmp_path, capsys):
    from hermes_state import SessionDB
    from hermes_state_registry import release
    local_flag.setenv(cli_local_mode.ENV_FLAG, "1")
    local_flag.setenv("HERMES_HOME", str(tmp_path))
    local_flag.setattr(gateway_session_store, "GatewaySessionStore",
                       lambda *a, **k: pytest.fail("local mode built the gateway store"))
    store = gateway_session_store.session_store()
    try:
        assert isinstance(store, SessionDB)
        captured = capsys.readouterr()
        assert "[local mode]" in captured.err and captured.out == ""
    finally:
        release(store)


def test_gateway_mode_keeps_the_gateway_store(local_flag):
    sentinel = object()
    local_flag.setattr(gateway_session_store, "GatewaySessionStore", lambda: sentinel)
    assert gateway_session_store.session_store() is sentinel


def test_titled_session_is_written_locally(local_flag, tmp_path):
    from hermes_state_registry import acquire, release
    local_flag.setenv(cli_local_mode.ENV_FLAG, "1")
    local_flag.setenv("HERMES_HOME", str(tmp_path))
    session_id = gateway_session_store.create_titled_session("my thread")
    store = acquire()
    try:
        assert store.get_session_title(session_id) == "my thread"
        assert store.get_session(session_id)["source"] == "cli"
    finally:
        release(store)
        release(store)  # create_titled_session's own handle


def _stub_build(monkeypatch, toolsets=("web", "file")):
    import hermes_cli.config
    import hermes_cli.mcp_startup
    import hermes_cli.plugins
    import hermes_cli.tools_config
    seen = {}
    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=lambda **kwargs: seen.update(kwargs) or "agent"))
    monkeypatch.setattr(hermes_cli.plugins, "discover_plugins", lambda: seen.setdefault("plugins", True))
    monkeypatch.setattr(hermes_cli.mcp_startup, "ensure_mcp_discovery_before_agent_build",
                        lambda **k: seen.setdefault("mcp", True))
    monkeypatch.setattr(hermes_cli.config, "load_config", dict)
    monkeypatch.setattr(hermes_cli.tools_config, "_get_platform_tools", lambda cfg, platform: set(toolsets))
    return seen


def test_build_drops_gateway_only_kwargs_and_loads_what_the_gateway_would(monkeypatch):
    seen = _stub_build(monkeypatch)
    kwargs = {key: "x" for key in cli_local_mode.GATEWAY_ONLY_AGENT_KWARGS}
    assert cli_local_mode.build_agent(model="m", enabled_toolsets=["picked"], **kwargs) == "agent"
    assert not cli_local_mode.GATEWAY_ONLY_AGENT_KWARGS & set(seen)
    assert seen["model"] == "m" and seen["enabled_toolsets"] == ["picked"]
    assert seen["plugins"] and seen["mcp"]


def test_unpicked_toolsets_resolve_from_config_not_every_tool(monkeypatch):
    """None means "the gateway resolves it" to the CLI, and "all tools" to AIAgent."""
    seen = _stub_build(monkeypatch, toolsets=("web", "file"))
    cli_local_mode.build_agent(enabled_toolsets=None)
    assert seen["enabled_toolsets"] == ["file", "web"]


def test_every_init_agent_kwarg_is_accepted_by_the_real_agent():
    """A new ``_init_agent`` argument only GatewayAgent knows would crash local mode at the first turn."""
    root = Path(__file__).resolve().parents[2]

    def init_params():
        tree = ast.parse((root / "run_agent.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AIAgent")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        return {a.arg for a in init.args.args + init.args.kwonlyargs}

    tree = ast.parse((root / "hermes_cli" / "cli_agent_setup_mixin.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "AIAgent"]
    assert calls, "_init_agent no longer builds through the AIAgent name"
    passed = {kw.arg for call in calls for kw in call.keywords}
    assert passed - init_params() <= cli_local_mode.GATEWAY_ONLY_AGENT_KWARGS
