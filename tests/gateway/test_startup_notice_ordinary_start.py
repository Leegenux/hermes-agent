"""An ordinary start announces itself on the home channels (#27870).

The planned-restart marker only exists when the previous process went down *for a restart*, so a
cold boot, a supervisor/crash recovery and a CLI ``stop`` + ``start`` (all of Windows) reached the
boot pass with nothing owed and left the home channel silent — on a messaging-only install that is
indistinguishable from a dead gateway. These tests pin the branch in ``_await_startup_boot_sends``,
the notice itself and the throttle that keeps a crash loop from paging (#60438).
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from agent.i18n import t
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner

ONLINE_NOTICE = t("gateway.startup.online")


def _adapter():
    return SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True, message_id="m1")))


def _make_runner(tmp_path, monkeypatch, *, adapters):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # gateway.run snapshots the home at import; the marker writer reads it live.
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = object.__new__(GatewayRunner)
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                gateway_restart_notification=True,
                home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id="home-1", name="home-1"),
            )
        }
    )
    config.sessions_dir = tmp_path / "sessions"
    runner.config = config
    runner.adapters = adapters
    runner._profile_configs = {}
    runner._profile_adapters = {}
    runner._planned_restart_notice_lock = None
    for name in (
        "_home_channel_transports",
        "_served_home_channel_transports",
        "_send_home_channel_message",
        "_send_home_channel_startup_notifications",
        "_send_ordinary_start_notice",
        "_await_startup_boot_sends",
    ):
        method = getattr(GatewayRunner, name, None)
        if method is not None:  # absent on the pre-fix base, where the red run must still set up
            setattr(runner, name, method.__get__(runner, GatewayRunner))
    # Not under test here: the thread-target resolver reads live session state.
    runner._thread_metadata_for_target = lambda *a, **k: None
    return runner


def test_ordinary_start_notifies_the_home_channel(tmp_path, monkeypatch):
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})

    __import__("asyncio").run(runner._send_ordinary_start_notice())

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[:2] == ("home-1", ONLINE_NOTICE)
    assert not (tmp_path / ".restart_pending.json").exists(), "the debt must be discharged too"
    stamped = json.loads((tmp_path / ".startup_notice.json").read_text(encoding="utf-8"))
    assert stamped["sent_at"] > 0


def test_second_start_inside_the_window_is_throttled(tmp_path, monkeypatch):
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})

    __import__("asyncio").run(runner._send_ordinary_start_notice())
    __import__("asyncio").run(runner._send_ordinary_start_notice())

    adapter.send.assert_awaited_once(), "a crash loop must not page the home channel every attempt"


def test_an_unreachable_home_stays_owed_and_burns_no_throttle(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch, adapters={})
    # `adapters={}` alone is not enough: the delivery transport falls back to a process-wide
    # router that other suites populate, so pin "nothing is live" explicitly.
    monkeypatch.setattr(runner, "_served_home_channel_transports", lambda: iter(()))

    __import__("asyncio").run(runner._send_ordinary_start_notice())

    assert (tmp_path / ".restart_pending.json").exists(), (
        "a start that reached nobody must keep the notice owed for the reconnect replay"
    )
    assert not (tmp_path / ".startup_notice.json").exists(), (
        "a start that reached nobody must not burn the throttle window"
    )


@pytest.mark.asyncio
async def test_a_platform_that_connects_late_still_gets_the_notice(tmp_path, monkeypatch):
    """The point of owing the marker: offline at boot, announced when it comes back (#112109)."""
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={})
    live = runner._served_home_channel_transports
    monkeypatch.setattr(runner, "_served_home_channel_transports", lambda: iter(()))

    await runner._send_ordinary_start_notice()
    adapter.send.assert_not_awaited()

    runner.adapters = {Platform.TELEGRAM: adapter}  # the adapter reconnects
    monkeypatch.setattr(runner, "_served_home_channel_transports", live)
    await runner._replay_pending_planned_restart_notification()

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[:2] == ("home-1", ONLINE_NOTICE)
    assert not (tmp_path / ".restart_pending.json").exists()


@pytest.mark.asyncio
async def test_boot_pass_picks_the_notice_for_the_start_it_actually_had(tmp_path, monkeypatch):
    """Marker present -> planned-restart replay; marker absent -> ordinary notice. Never both."""
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})
    runner._claim_pending_obligations = AsyncMock(return_value=set())
    runner._send_restart_notification = AsyncMock(return_value=None)
    runner._replay_pending_planned_restart_notification = AsyncMock(return_value=None)
    runner._redeliver_claimed_obligations = AsyncMock(return_value=None)
    runner._send_ordinary_start_notice = AsyncMock(return_value=set())

    await runner._await_startup_boot_sends(planned_restart_notification_pending=True)
    runner._replay_pending_planned_restart_notification.assert_awaited_once()
    runner._send_ordinary_start_notice.assert_not_awaited()

    runner._replay_pending_planned_restart_notification.reset_mock()
    await runner._await_startup_boot_sends(planned_restart_notification_pending=False)
    runner._send_ordinary_start_notice.assert_awaited_once()
    runner._replay_pending_planned_restart_notification.assert_not_awaited()


@pytest.mark.asyncio
async def test_ordinary_start_reaches_the_home_channel_through_the_boot_pass(tmp_path, monkeypatch):
    """End to end through the real boot pass: no marker, the home channel still hears about it.

    Fails on the pre-fix base on an assertion (the channel got nothing), not on a missing symbol,
    because the only stub is ``_send_restart_notification``.
    """
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    adapter = _adapter()
    runner = _make_runner(tmp_path, monkeypatch, adapters={Platform.TELEGRAM: adapter})
    runner._claim_pending_obligations = AsyncMock(return_value=set())
    runner._send_restart_notification = AsyncMock(return_value=None)
    runner._redeliver_claimed_obligations = AsyncMock(return_value=None)

    await runner._await_startup_boot_sends(planned_restart_notification_pending=False)

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[:2] == ("home-1", ONLINE_NOTICE)
