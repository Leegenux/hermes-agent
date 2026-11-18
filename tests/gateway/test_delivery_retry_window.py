"""Bounded runtime retries must outlast two network failures without requiring reboot."""
import pytest

from gateway import delivery_ledger as dl


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (12345, 67890))
    return dl


def _failed(ledger, oid, *, attempts=2, created=1000.0, updated=1000.0,
            error="httpx.ConnectError: ", profile="default"):
    ledger.record_obligation(
        obligation_id=oid, session_key=f"agent:{profile}:telegram:dm:123",
        platform="telegram", chat_id="123", thread_id="57", content="stored answer",
        adapter_profile=profile,
    )
    ledger.mark_failed(oid, error)
    with ledger._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET attempts=?,created_at=?,updated_at=? WHERE obligation_id=?",
            (attempts, created, updated, oid),
        )


@pytest.mark.parametrize("error", ["httpx.ConnectError: ", "httpx.ReadError: ", "HTTP 503"])
def test_network_failure_after_two_attempts_is_scheduled_and_delivered(ledger, error):
    _failed(ledger, "reply", error=error)
    pending = ledger.pending_retries(now=1119.0)
    assert pending == [{"platform": "telegram", "profile": "default", "not_before": 1120.0}]
    assert ledger.sweep_failed_for_runtime("telegram", now=1119.0) == []
    claimed = ledger.sweep_failed_for_runtime("telegram", now=1120.0)
    assert len(claimed) == 1
    assert claimed[0]["content"] == "stored answer"
    assert claimed[0]["attempts"] == 3
    assert claimed[0]["needs_marker"] is True
    ledger.mark_delivered("reply")
    assert ledger.sweep_failed_for_runtime("telegram", now=1121.0) == []


def test_twentieth_attempt_is_reserved_for_restart(ledger, monkeypatch):
    _failed(ledger, "last", attempts=19)
    assert ledger.pending_retries(now=1120.0) == []
    assert ledger.sweep_failed_for_runtime("telegram", now=1120.0) == []
    with ledger._connect() as conn:
        assert conn.execute(
            "SELECT state,attempts FROM delivery_obligations WHERE obligation_id='last'"
        ).fetchone() == ("failed", 19)
        conn.execute("UPDATE delivery_obligations SET owner_pid=999999999 WHERE obligation_id='last'")
    monkeypatch.setattr(ledger, "_owner_alive", lambda *_: False)
    claimed = ledger.sweep_recoverable(now=1120.0)
    assert len(claimed) == 1
    assert claimed[0]["attempts"] == 20
    assert claimed[0]["content"] == "stored answer"
    assert claimed[0]["needs_marker"] is True


def test_runtime_window_expires_without_bypassing_dead_targets_profiles_or_flood_wait(ledger):
    _failed(ledger, "old", attempts=0, created=-801.0)
    _failed(ledger, "blocked", attempts=0, error="Forbidden: bot was blocked by the user")
    _failed(ledger, "foreign", attempts=0, profile="other")
    _failed(ledger, "flood", attempts=0, created=-801.0, error="flood_control:3600")
    assert sorted(ledger.pending_retries(now=1000.0), key=lambda row: row["profile"]) == [
        {"platform": "telegram", "profile": "default", "not_before": 4600.0},
        {"platform": "telegram", "profile": "other", "not_before": 1030.0},
    ]
    assert ledger.sweep_failed_for_runtime("telegram", now=1000.0) == []
    with ledger._connect() as conn:
        states = dict(conn.execute("SELECT obligation_id,state FROM delivery_obligations"))
    assert states == {"old": "failed", "blocked": "failed", "foreign": "failed", "flood": "failed"}
    assert ledger.sweep_failed_for_runtime("telegram", now=4600.0)[0]["obligation_id"] == "flood"


def test_runtime_age_limit_preserves_boot_recovery(ledger, monkeypatch):
    _failed(ledger, "paused", created=1000.0)
    now = 1000.0 + 30 * 60 + 1
    assert ledger.pending_retries(now=now) == []
    assert ledger.sweep_failed_for_runtime("telegram", now=now) == []
    with ledger._connect() as conn:
        assert conn.execute(
            "SELECT state FROM delivery_obligations WHERE obligation_id='paused'"
        ).fetchone() == ("failed",)
        conn.execute("UPDATE delivery_obligations SET owner_pid=999999999 WHERE obligation_id='paused'")
    monkeypatch.setattr(ledger, "_owner_alive", lambda *_: False)
    claimed = ledger.sweep_recoverable(now=now)
    assert len(claimed) == 1
    assert claimed[0]["obligation_id"] == "paused"
