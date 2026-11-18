"""Durable same-tree collaboration state (stage 3/4).

The store is deliberately small and additive: one tree policy row, one message ledger with
explicit states, and the busy-target inbox. It stores no message body and no credentials —
only a fingerprint over the body plus routing/state metadata.

States are not interchangeable:

``accepted``   admitted, delivery not proven
``delivered``  the exact message became a ``role:user`` row in the target transcript
``processed``  the run that consumed that row ended (its outcome is recorded separately)
``needs_review`` recovery found an ambiguous delivery/processing outcome; never auto-redispatch

All writes go through ``state.db``'s canonical schema (see ``hermes_state_common.SCHEMA_SQL``),
so an old database gains the tables on its next open without touching existing rows.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("tools.delegate_tool")

DEFAULT_MAX_ROUNDS = 12
DEFAULT_MAX_CONCURRENT = 3
DEFAULT_MESSAGE_CAP = 4096

MESSAGE_STATES = ("pending", "accepted", "delivered", "processed", "rejected", "needs_review")

_MESSAGE_COLS = (
    "message_id", "tree_id", "sender_id", "sender_session_id", "recipient_id", "recipient_session_id",
    "target_input", "fingerprint", "state", "delivery", "round", "reason", "error", "outcome",
    "created_at", "updated_at", "accepted_at", "delivered_at", "processed_at", "owner_pid",
    "owner_started_at",
)
_TREE_COLS = (
    "tree_id", "root_session_id", "profile_key", "state", "max_rounds", "max_concurrent",
    "rounds_used", "stop_reason", "created_at", "updated_at",
)
_INBOX_COLS = (
    "seq", "message_id", "tree_id", "target_id", "target_session_id", "state", "created_at", "updated_at",
)

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_RECOVERED: set[str] = set()


class CollabStoreError(RuntimeError):
    """A durable collaboration write could not be committed; callers must fail closed."""


def reset_collab_state() -> None:
    """Test-only: forget per-process locks and the one-shot recovery cache."""
    with _LOCKS_GUARD:
        _LOCKS.clear()
        _RECOVERED.clear()


def _tree_lock(tree_id: str) -> threading.RLock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(tree_id), threading.RLock())


def _now() -> float:
    return time.time()


def _limits() -> tuple[int, int]:
    """``(max_rounds, max_concurrent)`` from ``delegation.*`` or safe defaults."""
    try:
        from tools.delegate_tool_config import _load_config
        cfg = _load_config() or {}
    except Exception:  # health: allow BLE001 -- an unreadable config keeps the safe ceilings
        cfg = {}
    try:
        rounds = cfg.get("max_collab_rounds")
        rounds = int(rounds) if rounds not in (None, "") else DEFAULT_MAX_ROUNDS
    except (TypeError, ValueError):
        rounds = DEFAULT_MAX_ROUNDS
    try:
        concurrent = cfg.get("max_collab_concurrency")
        concurrent = int(concurrent) if concurrent not in (None, "") else DEFAULT_MAX_CONCURRENT
    except (TypeError, ValueError):
        concurrent = DEFAULT_MAX_CONCURRENT
    return max(1, rounds), max(1, concurrent)


# ── identity / connection ──────────────────────────────────────────────────

def _root_agent(agent: Any) -> Optional[Any]:
    cur, seen = agent, set()
    for _ in range(16):
        if cur is None:
            return None
        sid = getattr(cur, "session_id", None)
        if isinstance(sid, str) and sid:
            if sid in seen:
                return cur
            seen.add(sid)
        ref = getattr(cur, "_delegate_parent_ref", None)
        parent = ref() if callable(ref) else None
        if parent is None:
            return cur
        cur = parent
    return cur


def db_path_for(agent: Any) -> Optional[str]:
    """The profile ``state.db`` this agent's tree writes to; None in a managed worker."""
    from agent.runtime_session_store import is_worker_process
    if is_worker_process():
        return None
    for candidate in (_root_agent(agent), agent):
        path = getattr(getattr(candidate, "_session_db", None), "db_path", None)
        if isinstance(path, (str, Path)) and str(path):
            return os.path.abspath(str(path))
    from hermes_constants import get_hermes_home
    return str(get_hermes_home() / "state.db")


def profile_key_for(agent: Any) -> Optional[str]:
    for candidate in (_root_agent(agent), agent):
        value = getattr(candidate, "_delegate_profile_key", None)
        if isinstance(value, (str, Path)) and str(value):
            return os.path.normcase(os.path.abspath(str(value)))
        path = getattr(getattr(candidate, "_session_db", None), "db_path", None)
        if isinstance(path, (str, Path)) and str(path):
            return os.path.normcase(os.path.abspath(str(path)))
    return None


def _owner_identity() -> dict[str, Any]:
    from tools.delegate_tool_resumable import _owner_identity as identity
    return identity()


def _write(db_path: Optional[str], fn, *, immediate: bool = True):
    """Run ``fn(conn)`` in one transaction. Raises CollabStoreError: callers must not
    treat an uncommitted collaboration admission as accepted."""
    if not db_path:
        raise CollabStoreError("no durable store for this agent (managed worker)")
    from hermes_cli.sqlite_util import transaction
    from tools.async_delegation import open_state_db
    try:
        with transaction(open_state_db(Path(db_path), "state.db (delegate collaboration)"),
                         immediate=immediate) as conn:
            return fn(conn)
    except CollabStoreError:
        raise
    except Exception as exc:
        raise CollabStoreError(str(exc)) from exc


def _read(db_path: Optional[str], fn):
    if not db_path or not Path(db_path).is_file():
        return None
    try:
        return _write(db_path, fn, immediate=False)
    except CollabStoreError:
        raise
    except Exception:
        logger.debug("collaboration store read failed (%s)", db_path, exc_info=True)
        return None


def _msg(row) -> Optional[dict[str, Any]]:
    return dict(zip(_MESSAGE_COLS, row)) if row is not None else None


def _tree(row) -> Optional[dict[str, Any]]:
    return dict(zip(_TREE_COLS, row)) if row is not None else None


def _inbox(row) -> Optional[dict[str, Any]]:
    return dict(zip(_INBOX_COLS, row)) if row is not None else None


def _select(table_cols: tuple[str, ...]) -> str:
    return ", ".join(table_cols)


# ── tree policy ────────────────────────────────────────────────────────────

def ensure_tree(db_path: Optional[str], root_session_id: str, profile_key: str, *,
                max_rounds: Optional[int] = None, max_concurrent: Optional[int] = None,
                activate: bool = False) -> Optional[dict[str, Any]]:
    """Create the tree row if absent. ``activate`` is only for a NEW explicit spawn:
    it may reopen a stopped tree and resets the round budget; messages never call it."""
    root = str(root_session_id or "").strip()
    if not root:
        return None
    default_rounds, default_concurrent = _limits()
    rounds = max(1, int(max_rounds or default_rounds))
    concurrent = max(1, int(max_concurrent or default_concurrent))
    now = _now()

    def _do(conn):
        conn.execute(
            """INSERT OR IGNORE INTO delegate_collab_trees
                   (tree_id, root_session_id, profile_key, state, max_rounds, max_concurrent,
                    rounds_used, stop_reason, created_at, updated_at)
               VALUES (?,?,?,?,?,?,0,NULL,?,?)""",
            (root, root, str(profile_key or ""), "active", rounds, concurrent, now, now))
        if activate:
            conn.execute(
                """UPDATE delegate_collab_trees
                      SET state='active', rounds_used=0, stop_reason=NULL,
                          max_rounds=?, max_concurrent=?, updated_at=?
                    WHERE tree_id=? AND state='stopped'""",
                (rounds, concurrent, now, root))
        row = conn.execute(f"SELECT {_select(_TREE_COLS)} FROM delegate_collab_trees WHERE tree_id=?",
                           (root,)).fetchone()
        return _tree(row)

    return _write(db_path, _do)


def tree(db_path: Optional[str], tree_id: str) -> Optional[dict[str, Any]]:
    def _do(conn):
        row = conn.execute(f"SELECT {_select(_TREE_COLS)} FROM delegate_collab_trees WHERE tree_id=?",
                           (str(tree_id),)).fetchone()
        return _tree(row)
    return _read(db_path, _do)


def stop_tree(db_path: Optional[str], tree_id: str, reason: str = "tree_stopped") -> int:
    """Stop the tree and cancel admitted-but-undelivered work. Delivered rows stay
    delivered (their run is ended by the native interrupt path)."""
    now = _now()

    def _do(conn):
        cur = conn.execute(
            "UPDATE delegate_collab_trees SET state='stopped', stop_reason=?, updated_at=? WHERE tree_id=?",
            (str(reason), now, str(tree_id)))
        if cur.rowcount != 1:
            return 0
        pending = conn.execute(
            f"SELECT {_select(_MESSAGE_COLS)} FROM delegate_collab_messages "
            "WHERE tree_id=? AND state='accepted'", (str(tree_id),)).fetchall()
        for row in pending:
            entry = _msg(row)
            conn.execute(
                "UPDATE delegate_collab_messages SET state='rejected', reason='tree_stopped', updated_at=? "
                "WHERE message_id=? AND state='accepted'",
                (now, entry["message_id"]))
        conn.execute(
            "UPDATE delegate_collab_inbox SET state='rejected', updated_at=? "
            "WHERE tree_id=? AND state='queued'", (now, str(tree_id)))
        return len(pending)

    with _tree_lock(tree_id):
        return _write(db_path, _do) or 0


# ── message ledger ─────────────────────────────────────────────────────────

def get_message(db_path: Optional[str], message_id: str) -> Optional[dict[str, Any]]:
    def _do(conn):
        row = conn.execute(f"SELECT {_select(_MESSAGE_COLS)} FROM delegate_collab_messages "
                           "WHERE message_id=?", (str(message_id),)).fetchone()
        return _msg(row)
    return _read(db_path, _do)


def _insert_message(conn, *, tree_id: str, message_id: str, sender_id: str, sender_session_id: Optional[str],
                    recipient_id: str, recipient_session_id: Optional[str], target_input: str,
                    fingerprint: str, delivery: str, round_no: int, state: str = "accepted") -> None:
    now, owner = _now(), _owner_identity()
    conn.execute(
        """INSERT INTO delegate_collab_messages
               (message_id, tree_id, sender_id, sender_session_id, recipient_id, recipient_session_id,
                target_input, fingerprint, state, delivery, round, reason, error, outcome,
                created_at, updated_at, accepted_at, delivered_at, processed_at, owner_pid, owner_started_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,?,?,?,NULL,NULL,?,?)""",
        (message_id, tree_id, sender_id, sender_session_id, recipient_id, recipient_session_id,
         target_input, fingerprint, state, delivery, int(round_no), now, now, now,
         owner.get("owner_pid"), owner.get("owner_started_at")))


def _fetch_message(conn, message_id: str) -> Optional[dict[str, Any]]:
    return _msg(conn.execute(f"SELECT {_select(_MESSAGE_COLS)} FROM delegate_collab_messages "
                             "WHERE message_id=?", (message_id,)).fetchone())


def _prune_conn(conn, tree_id: str, cap: int = DEFAULT_MESSAGE_CAP) -> int:
    """Bounded ledger: remove oldest TERMINAL rows beyond *cap* only."""
    cap = max(1, int(cap))
    total = conn.execute("SELECT COUNT(*) FROM delegate_collab_messages WHERE tree_id=?",
                         (tree_id,)).fetchone()[0]
    excess = max(0, int(total) - cap)
    if not excess:
        return 0
    ids = [r[0] for r in conn.execute(
        "SELECT message_id FROM delegate_collab_messages WHERE tree_id=? "
        "AND state IN ('processed','rejected') ORDER BY created_at ASC LIMIT ?",
        (tree_id, excess)).fetchall()]
    for mid in ids:
        conn.execute("DELETE FROM delegate_collab_inbox WHERE message_id=?", (mid,))
        conn.execute("DELETE FROM delegate_collab_messages WHERE message_id=?", (mid,))
    return len(ids)


def _admit(conn, tree_id: str, round_no: int) -> Optional[str]:
    row = _tree(conn.execute(f"SELECT {_select(_TREE_COLS)} FROM delegate_collab_trees WHERE tree_id=?",
                             (tree_id,)).fetchone())
    if row is None:
        return "unknown_tree"
    if row["state"] != "active":
        return "tree_stopped"
    if int(round_no) > int(row["max_rounds"]):
        return "rounds_exhausted"
    return None


def begin_inbox_message(db_path: Optional[str], tree_id: str, message_id: str, *,
                        sender_id: str, sender_session_id: Optional[str], recipient_id: str,
                        recipient_session_id: Optional[str], target_input: str, fingerprint: str,
                        round_no: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Durably admit a busy-target message and queue its inbox row atomically."""
    def _do(conn):
        why = _admit(conn, tree_id, round_no)
        if why:
            return None, why
        try:
            _insert_message(
                conn, tree_id=tree_id, message_id=message_id, sender_id=sender_id,
                sender_session_id=sender_session_id, recipient_id=recipient_id,
                recipient_session_id=recipient_session_id, target_input=target_input,
                fingerprint=fingerprint, delivery="next_step_inbox", round_no=round_no)
            now = _now()
            conn.execute(
                """INSERT INTO delegate_collab_inbox
                       (message_id, tree_id, target_id, target_session_id, state, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (message_id, tree_id, recipient_id, recipient_session_id, "queued", now, now))
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                return None, "duplicate"
            raise
        _prune_conn(conn, tree_id)
        return _fetch_message(conn, message_id), None

    with _tree_lock(tree_id):
        return _write(db_path, _do)


def begin_wake_message(db_path: Optional[str], tree_id: str, message_id: str, *,
                       sender_id: str, sender_session_id: Optional[str], recipient_id: str,
                       recipient_session_id: Optional[str], target_input: str, fingerprint: str,
                       round_no: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Durably admit an addressed message for an IDLE root, before its native route wake.

    The row is the durable receipt that a wake was attempted; the gateway's own admission
    ledger is the durable receipt that a turn was queued. No body is stored here.
    """
    def _do(conn):
        why = _admit(conn, tree_id, round_no)
        if why:
            return None, why
        try:
            _insert_message(
                conn, tree_id=tree_id, message_id=message_id, sender_id=sender_id,
                sender_session_id=sender_session_id, recipient_id=recipient_id,
                recipient_session_id=recipient_session_id, target_input=target_input,
                fingerprint=fingerprint, delivery="root_wake", round_no=round_no)
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                return None, "duplicate"
            raise
        _prune_conn(conn, tree_id)
        return _fetch_message(conn, message_id), None

    with _tree_lock(tree_id):
        return _write(db_path, _do)


def pending_root_wakes(db_path: Optional[str], root_session_id: str,
                       states: tuple[str, ...] = ("accepted", "delivered")) -> list[dict[str, Any]]:
    """Root-bound wake rows that still need a delivery/processing transition."""
    marks = ",".join("?" for _ in states)
    def _do(conn):
        rows = conn.execute(
            f"SELECT {_select(_MESSAGE_COLS)} FROM delegate_collab_messages "
            f"WHERE recipient_id='lead' AND delivery='root_wake' AND state IN ({marks}) "
            "AND (recipient_session_id=? OR recipient_session_id IS NULL) ORDER BY created_at",
            (*states, str(root_session_id))).fetchall()
        return [m for m in (_msg(r) for r in rows) if m]
    return _read(db_path, _do) or []


def message_materialized(db_path: Optional[str], message: dict[str, Any]) -> bool:
    """Exact marker lookup in the target transcript (same rule as recovery)."""
    def _do(conn):
        return _marker_recorded(conn, message)
    return bool(_read(db_path, _do))


def tree_lock(tree_id: str) -> threading.RLock:
    """The per-tree admission lock shared by stop, dispatch and registration."""
    return _tree_lock(tree_id)


def begin_dispatch_message(db_path: Optional[str], tree_id: str, message_id: str, *,
                           sender_id: str, sender_session_id: Optional[str], recipient_id: str,
                           recipient_session_id: Optional[str], target_input: str, fingerprint: str,
                           round_no: int) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Reserve one tree round + one concurrency slot and record an accepted continuation.
    The state flip and the counters are one IMMEDIATE transaction, so two senders cannot
    both pass the cap."""
    def _do(conn):
        why = _admit(conn, tree_id, round_no)
        if why:
            return None, why
        row = _tree(conn.execute(f"SELECT {_select(_TREE_COLS)} FROM delegate_collab_trees WHERE tree_id=?",
                                 (tree_id,)).fetchone())
        active = conn.execute(
            "SELECT COUNT(*) FROM delegate_collab_messages WHERE tree_id=? AND delivery='continuation' "
            "AND processed_at IS NULL AND state IN ('accepted','delivered')", (tree_id,)).fetchone()[0]
        if int(active) >= int(row["max_concurrent"]):
            return None, "concurrency_exhausted"
        if int(row["rounds_used"]) >= int(row["max_rounds"]):
            return None, "rounds_exhausted"
        try:
            _insert_message(
                conn, tree_id=tree_id, message_id=message_id, sender_id=sender_id,
                sender_session_id=sender_session_id, recipient_id=recipient_id,
                recipient_session_id=recipient_session_id, target_input=target_input,
                fingerprint=fingerprint, delivery="continuation", round_no=round_no)
        except Exception as exc:
            if "UNIQUE" in str(exc).upper():
                return None, "duplicate"
            raise
        conn.execute("UPDATE delegate_collab_trees SET rounds_used=rounds_used+1, updated_at=? "
                     "WHERE tree_id=?", (_now(), tree_id))
        _prune_conn(conn, tree_id)
        return _fetch_message(conn, message_id), None

    with _tree_lock(tree_id):
        return _write(db_path, _do)


def release_dispatch(db_path: Optional[str], tree_id: str, message_id: str, reason: str,
                     error: Optional[str] = None) -> bool:
    """A reserved dispatch was refused before it ran: reject it and give the round back."""
    def _do(conn):
        cur = conn.execute(
            "UPDATE delegate_collab_messages SET state='rejected', reason=?, error=?, updated_at=? "
            "WHERE message_id=? AND state='accepted' AND delivery='continuation'",
            (str(reason), error, _now(), message_id))
        if cur.rowcount == 1:
            conn.execute("UPDATE delegate_collab_trees SET rounds_used=MAX(0, rounds_used-1), updated_at=? "
                         "WHERE tree_id=?", (_now(), tree_id))
        return cur.rowcount == 1
    with _tree_lock(tree_id):
        return bool(_write(db_path, _do))


def release_round(db_path: Optional[str], tree_id: str) -> None:
    """Hand one reserved round back without changing the message state (needs_review)."""
    def _do(conn):
        conn.execute("UPDATE delegate_collab_trees SET rounds_used=MAX(0, rounds_used-1), updated_at=? "
                     "WHERE tree_id=?", (_now(), str(tree_id)))
    with _tree_lock(tree_id):
        _write(db_path, _do)


def mark_delivered(db_path: Optional[str], message_id: str,
                   target_session_id: Optional[str] = None) -> bool:
    now = _now()

    def _do(conn):
        cur = conn.execute(
            "UPDATE delegate_collab_messages SET state='delivered', delivered_at=?, updated_at=?, "
            "recipient_session_id=COALESCE(?, recipient_session_id) "
            "WHERE message_id=? AND (state='accepted' OR (state='rejected' AND reason='tree_stopped'))",
            (now, now, target_session_id, message_id))
        if cur.rowcount == 1:
            conn.execute("UPDATE delegate_collab_inbox SET state='delivered', updated_at=? "
                         "WHERE message_id=?", (now, message_id))
        return cur.rowcount == 1

    row = get_message(db_path, message_id)
    if row is None:
        return False
    with _tree_lock(row["tree_id"]):
        return bool(_write(db_path, _do))


def mark_processed(db_path: Optional[str], message_id: str, outcome: str = "completed") -> bool:
    now = _now()

    def _do(conn):
        cur = conn.execute(
            "UPDATE delegate_collab_messages SET state='processed', processed_at=?, outcome=?, updated_at=? "
            "WHERE message_id=? AND state='delivered'",
            (now, str(outcome), now, message_id))
        return cur.rowcount == 1

    row = get_message(db_path, message_id)
    if row is None:
        return False
    with _tree_lock(row["tree_id"]):
        return bool(_write(db_path, _do))


def mark_rejected(db_path: Optional[str], message_id: str, reason: str,
                  error: Optional[str] = None) -> bool:
    now = _now()

    def _do(conn):
        cur = conn.execute(
            "UPDATE delegate_collab_messages SET state='rejected', reason=?, error=?, updated_at=? "
            "WHERE message_id=? AND state IN ('pending','accepted')",
            (str(reason), error, now, message_id))
        if cur.rowcount == 1:
            conn.execute("UPDATE delegate_collab_inbox SET state='rejected', updated_at=? "
                         "WHERE message_id=? AND state='queued'", (now, message_id))
        return cur.rowcount == 1

    row = get_message(db_path, message_id)
    if row is None:
        return False
    with _tree_lock(row["tree_id"]):
        return bool(_write(db_path, _do))


def mark_needs_review(db_path: Optional[str], message_id: str, reason: str) -> bool:
    """Ambiguous outcome. Terminal for automation: no future dispatch from this row."""
    now = _now()

    def _do(conn):
        cur = conn.execute(
            "UPDATE delegate_collab_messages SET state='needs_review', reason=?, updated_at=? "
            "WHERE message_id=? AND state IN ('accepted','delivered')",
            (str(reason), now, message_id))
        if cur.rowcount == 1:
            conn.execute("UPDATE delegate_collab_inbox SET state='needs_review', updated_at=? "
                         "WHERE message_id=? AND state='queued'", (now, message_id))
        return cur.rowcount == 1

    row = get_message(db_path, message_id)
    if row is None:
        return False
    with _tree_lock(row["tree_id"]):
        return bool(_write(db_path, _do))


def queued_inbox(db_path: Optional[str], tree_id: str, target_id: str) -> list[dict[str, Any]]:
    def _do(conn):
        rows = conn.execute(
            f"SELECT {_select(_INBOX_COLS)} FROM delegate_collab_inbox "
            "WHERE tree_id=? AND target_id=? AND state='queued' ORDER BY seq",
            (str(tree_id), str(target_id))).fetchall()
        return [d for d in (_inbox(r) for r in rows) if d]
    return _read(db_path, _do) or []


def prune_messages(db_path: Optional[str], tree_id: str, cap: int = DEFAULT_MESSAGE_CAP) -> int:
    """Drop the oldest TERMINAL rows beyond *cap*. Non-terminal/needs-review rows are
    never pruned, so idempotency is bounded only for very old finished ids."""
    with _tree_lock(tree_id):
        return int(_write(db_path, lambda conn: _prune_conn(conn, str(tree_id), cap)) or 0)


# ── recovery ───────────────────────────────────────────────────────────────

def _marker_recorded(conn, message: dict[str, Any]) -> bool:
    marker = f"[agent message {message['message_id']}]"
    sids = [message.get("recipient_session_id"), message.get("recipient_id")]
    for sid in dict.fromkeys(s for s in sids if isinstance(s, str) and s):
        try:
            hit = conn.execute(
                "SELECT 1 FROM messages WHERE session_id=? AND role='user' AND instr(content, ?) > 0 LIMIT 1",
                (sid, marker)).fetchone()
        except Exception:  # health: allow BLE001 -- one unreadable segment cannot prove absence
            continue
        if hit:
            return True
    return False


def recover_tree(db_path: Optional[str], tree_id: Optional[str] = None) -> dict[str, Any]:
    """Conservative restart reconciliation. Never dispatches or retries a run.

    Only rows owned by another, provably dead process are touched. The caller's own
    accepted work is left alone. Ambiguous continuation intent becomes ``needs_review``;
    an inbox row that never materialized (no user row, no run) is rejected as no-effect.
    """
    from tools.async_delegation import _owner_liveness
    alive = _owner_liveness()
    if alive is None:
        return {"scanned": 0, "dispatched": 0, "skipped": "liveness_unavailable", "counts": {}}

    def _do(conn):
        where, args = "state IN ('accepted','delivered')", []
        if tree_id:
            where += " AND tree_id=?"
            args.append(str(tree_id))
        rows = [_msg(r) for r in conn.execute(
            f"SELECT {_select(_MESSAGE_COLS)} FROM delegate_collab_messages WHERE {where} "
            "ORDER BY created_at", args).fetchall()]
        counts: dict[str, int] = {}
        for row in rows:
            if row is None or alive(row.get("owner_pid"), row.get("owner_started_at")):
                continue
            t = _tree(conn.execute(f"SELECT {_select(_TREE_COLS)} FROM delegate_collab_trees "
                                   "WHERE tree_id=?", (row["tree_id"],)).fetchone())
            now = _now()
            materialized = row["state"] == "delivered" or _marker_recorded(conn, row)
            if materialized:
                # Delivery is proven; processing is not. A stopped tree interrupts the
                # run, it does not turn an existing user row into "never happened".
                if row["state"] != "delivered":
                    conn.execute(
                        "UPDATE delegate_collab_messages SET state='delivered', delivered_at=?, updated_at=? "
                        "WHERE message_id=? AND (state='accepted' OR "
                        "(state='rejected' AND reason='tree_stopped'))", (now, now, row["message_id"]))
                    conn.execute("UPDATE delegate_collab_inbox SET state='delivered', updated_at=? "
                                 "WHERE message_id=?", (now, row["message_id"]))
                state, reason = "needs_review", "recovery_run_outcome_unknown"
            elif t is not None and t["state"] == "stopped":
                state, reason = "rejected", "tree_stopped"
            elif row["delivery"] == "next_step_inbox":
                state, reason = "rejected", "recovery_undelivered_no_effect"
            else:
                state, reason = "needs_review", "recovery_unconfirmed_continuation"
            conn.execute(
                "UPDATE delegate_collab_messages SET state=?, reason=?, updated_at=? WHERE message_id=?",
                (state, reason, now, row["message_id"]))
            if state == "rejected":
                conn.execute("UPDATE delegate_collab_inbox SET state='rejected', updated_at=? "
                             "WHERE message_id=? AND state='queued'", (now, row["message_id"]))
            counts[f"{state}:{reason}"] = counts.get(f"{state}:{reason}", 0) + 1
        return {"scanned": len(rows), "dispatched": 0, "counts": counts}

    return _write(db_path, _do) or {"scanned": 0, "dispatched": 0, "counts": {}}


def maybe_recover(db_path: Optional[str]) -> None:
    """One reconciliation pass per db per process, before a tree is used."""
    if not db_path:
        return
    key = os.path.normcase(os.path.abspath(str(db_path)))
    with _LOCKS_GUARD:
        if key in _RECOVERED:
            return
        _RECOVERED.add(key)
    try:
        recover_tree(db_path)
    except Exception:
        logger.warning("collaboration recovery failed for %s", db_path, exc_info=True)


# ── engine helpers ─────────────────────────────────────────────────────────

def fingerprint_for(sender_id: str, target_input: str, body: str) -> str:
    import hashlib
    body_hash = hashlib.sha256(str(body).encode()).hexdigest()
    return hashlib.sha256(f"{sender_id}\0{target_input}\0{body_hash}".encode()).hexdigest()


def ensure_tree_for_child(child: Any, parent_agent: Any) -> Optional[dict[str, Any]]:
    """Register a child's tree. Only a fresh spawn (no durable continuation claim) may
    reactivate a stopped tree; a message-driven continuation never resurrects it."""
    root = getattr(child, "_delegate_root_session_id", None) or getattr(parent_agent, "session_id", None)
    profile = getattr(child, "_delegate_profile_key", None) or profile_key_for(parent_agent)
    if not isinstance(root, str) or not root or not profile:
        return None
    claim = getattr(child, "_delegate_claim", None)
    parent_is_resumed_run = (isinstance(getattr(parent_agent, "_delegate_message_meta", None), dict)
                             or isinstance(getattr(parent_agent, "_delegate_continue_history", None), list))
    activate = not isinstance(claim, tuple) and not parent_is_resumed_run
    try:
        return ensure_tree(db_path_for(parent_agent), root, str(profile), activate=activate)
    except CollabStoreError:
        logger.warning("could not register collaboration tree %s", root, exc_info=True)
        return None
