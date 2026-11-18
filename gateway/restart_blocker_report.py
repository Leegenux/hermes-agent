"""Readable lines for the ``/restart`` refusal: what each blocking unit is, where, and what it is doing.

The refusal used to say only ``1 agent``; a user then had to ask which chat and why. The units come
from ``_describe_active_work`` (the same list ``hermes update`` prints while draining). This module
adds what a user recognises — session title, group name, topic, the agent's latest activity — and
renders each unit with ``describe_active_work_unit`` so both surfaces word a unit the same way.
"""

from __future__ import annotations

from contextlib import suppress

_MAX_LINES = 8


def _text(value):
    return value if isinstance(value, str) and value else None


def _enrich_chat_unit(runner, unit: dict) -> None:
    """Add origin (platform / group / topic) and live activity for one chat unit; best-effort."""
    key = unit.get("session")
    with suppress(Exception):
        origin = getattr(runner.session_store._entries.get(key), "origin", None)
        if origin is not None:
            unit.update({"platform": _text(getattr(origin.platform, "value", None)),
                         "chat_type": _text(origin.chat_type), "chat_name": _text(origin.chat_name),
                         "thread_id": _text(origin.thread_id)})
    with suppress(Exception):
        agent = runner._sessions_map()[key].turn.agent
        unit["session_id"] = _text(getattr(agent, "session_id", None))
        summary = agent.get_activity_summary()
        unit["activity"] = _text(summary.get("last_activity_desc"))
        calls = summary.get("api_call_count")
        unit["api_calls"] = calls if isinstance(calls, int) else None


async def restart_blocker_lines(runner, current_session_key: str | None) -> list[str]:
    """``• <unit>`` per blocking unit (capped), the asking chat marked; [] when nothing is readable."""
    from hermes_cli.update_cmd_drain_report import describe_active_work_unit

    units: list = []
    with suppress(Exception):
        units = list(runner._describe_active_work())
    db = getattr(runner, "_session_db", None)
    lines: list[str] = []
    for unit in units[:_MAX_LINES]:
        if unit.get("kind") == "chat":
            _enrich_chat_unit(runner, unit)
            if unit.get("session_id") and db is not None:
                with suppress(Exception):
                    unit["title"] = await db.get_session_title(unit["session_id"])
        with suppress(Exception):
            text = describe_active_work_unit(unit)
            if unit.get("kind") == "chat" and unit.get("session") == current_session_key:
                text += " ← this chat"
            lines.append(f"• {text}")
    if len(units) > _MAX_LINES:
        lines.append(f"• … and {len(units) - _MAX_LINES} more")
    return lines
