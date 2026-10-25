"""In-chat /restart names the running work before it starts the drain.

Parity with the CLI's live-work guard (``hermes_cli/gateway_lifecycle_guards``): the CLI refuses a
restart while turns/cron/delegations are live and offers ``--yes``; the gateway slash path used to
start draining unannounced, so a busy gateway looked hung for up to ``restart_after_turn_timeout``.
"""

import pytest
from unittest.mock import MagicMock

from agent.i18n import t
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m1",
    )


@pytest.fixture(autouse=True)
def _hermetic_restart_context(monkeypatch):
    for var in ("INVOCATION_ID", "XPC_SERVICE_NAME", "HERMES_S6_SUPERVISED_CHILD",
                "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)


@pytest.mark.asyncio
async def test_restart_with_live_work_names_it_and_does_not_restart():
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    runner._drain_work_counts = MagicMock(return_value=(2, 1, 0, 0))

    result = await runner._handle_message(_event("/restart"))

    assert result == t("gateway.restart.blocked", count=3, breakdown="2 agent, 1 cron")
    assert result != "gateway.restart.blocked"  # catalog resolved, not the bare key
    runner.request_restart.assert_not_called()


@pytest.mark.asyncio
async def test_restart_force_bypasses_the_guard():
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    runner._drain_work_counts = MagicMock(return_value=(1, 0, 0, 0))

    await runner._handle_message(_event("/restart force"))

    runner.request_restart.assert_called_once()


@pytest.mark.asyncio
async def test_restart_with_no_live_work_still_restarts():
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    runner._drain_work_counts = MagicMock(return_value=(0, 0, 0, 0))

    await runner._handle_message(_event("/restart"))

    runner.request_restart.assert_called_once()
