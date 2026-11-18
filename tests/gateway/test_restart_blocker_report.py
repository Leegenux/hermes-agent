"""The /restart refusal names each blocking turn in words a user can act on.

Before: "⚠️ Not restarting: work is still running … \\n1 agent" — the user had to ask which chat it
was and what it was doing.
"""

import asyncio
import time
from types import SimpleNamespace

from gateway.config import Platform
from gateway.restart_blocker_report import restart_blocker_lines
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_state import SessionState


class _Agent:
    model = "gpt-test"
    session_id = "sess-1"

    def get_activity_summary(self):
        return {"current_tool": "write_file", "seconds_since_activity": 2.0,
                "last_activity_desc": "tool write_file", "api_call_count": 14}


class _DB:
    async def get_session_title(self, session_id):
        return {"sess-1": "能否无线连接管理设备"}.get(session_id)


def _runner(key, origin):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.session_store = SimpleNamespace(_entries={key: SimpleNamespace(origin=origin)})
    runner._session_db = _DB()
    state = SessionState()
    state.turn.agent = _Agent()
    state.turn.started_ts = time.time() - 192
    runner._sessions_map()[key] = state
    return runner


def test_refusal_line_names_title_group_topic_and_current_tool():
    key = "agent:main:telegram:group:-100398:20472"
    origin = SessionSource(platform=Platform.TELEGRAM, chat_id="-100398", chat_name="Claude Code",
                           chat_type="group", thread_id="20472")
    lines = asyncio.run(restart_blocker_lines(_runner(key, origin), current_session_key="agent:main:telegram:dm:1"))
    assert len(lines) == 1
    line = lines[0]
    for part in ("能否无线连接管理设备", "telegram group \"Claude Code\" › topic 20472", "running 3m1",
                 "now: tool write_file", "14 model calls"):
        assert part in line, (part, line)
    assert "this chat" not in line


def test_refusal_marks_the_asking_chat_and_degrades_without_origin_or_title():
    key = "agent:main:telegram:dm:1"
    runner = _runner(key, None)
    runner._session_db = None
    lines = asyncio.run(restart_blocker_lines(runner, current_session_key=key))
    assert lines[0].startswith("• ") and lines[0].endswith("← this chat")
    assert "now: tool write_file" in lines[0]
