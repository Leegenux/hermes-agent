"""Same-tree addressed messages between a root agent and its delegation tree.

This is a thin control plane over the existing delegation lifecycle:

* a live target gets an append-only next-step inbox entry, drained by the normal
  conversation loop as a standalone ``role:user`` row;
* a finished target is rebuilt by ``delegate_task(action='continue')`` on its own
  ``subagent_id``/``session_id``.

There is no resident agent, scheduler, second registry, or durable inbox here.
An ``accepted`` result is admission only; ``delivered`` is recorded only when the
message is materialized into the target's conversation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from tools import delegate_tool_collab as collab

logger = logging.getLogger(__name__)

_MESSAGE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_LEDGER_CAP = 1024
_INBOX_ATTR = "_delegate_inbox"
_INBOX_LOCK_ATTR = "_delegate_inbox_lock"
_REOPEN_ATTR = "_delegate_inbox_reopens"
_DELIVERED_IDS_ATTR = "_delegate_delivered_ids"
_COLLAB_ROUND_ATTR = "_delegate_collab_round"
_MAX_REOPENS = 4

_LEDGER_LOCK = threading.RLock()
_LEDGER: OrderedDict[str, dict[str, Any]] = OrderedDict()


def reset_message_state() -> None:
    """Test-only reset of the process-local message ledger."""
    with _LEDGER_LOCK:
        _LEDGER.clear()


def _now() -> float:
    return time.time()


def _profile_key(value: Any) -> Optional[str]:
    """Stable profile identity from an agent, record, or explicit path."""
    if isinstance(value, dict):
        for key in ("profile_key", "profile_db_path", "db_path"):
            candidate = value.get(key)
            if isinstance(candidate, (str, Path)) and str(candidate):
                return _normalize_path(candidate)
        resume = value.get("resume")
        if isinstance(resume, dict):
            return _profile_key(resume)
        route = value.get("route")
        if isinstance(route, dict):
            return _profile_key(route)
        return None
    if value is None:
        return None
    candidate = getattr(value, "_delegate_profile_key", None)
    if not isinstance(candidate, (str, Path)) or not str(candidate):
        candidate = getattr(getattr(value, "_session_db", None), "db_path", None)
    if isinstance(candidate, (str, Path)) and str(candidate):
        return _normalize_path(candidate)
    return None


def _normalize_path(path: Any) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def _root_session_id(agent: Any) -> Optional[str]:
    sid = getattr(agent, "_delegate_root_session_id", None)
    if isinstance(sid, str) and sid:
        return sid
    sid = getattr(agent, "session_id", None)
    return str(sid) if isinstance(sid, str) and sid else None


def _root_agent(agent: Any) -> Optional[Any]:
    cur, seen = agent, set()
    for _ in range(16):
        sid = getattr(cur, "session_id", None)
        if isinstance(sid, str) and sid in seen:
            break
        if isinstance(sid, str) and sid:
            seen.add(sid)
        ref = getattr(cur, "_delegate_parent_ref", None)
        parent = ref() if callable(ref) else None
        if parent is None:
            # A delegated child without a live parent chain has no tree root; do
            # not mistake the child itself for the root (its session id is not
            # the root id, even though it carries the root's stamp).
            return None if getattr(cur, "_subagent_id", None) else cur
        cur = parent
    return None


def _lineage_tip(sid: Optional[str], agent: Any) -> Optional[str]:
    if not sid:
        return None
    db = getattr(agent, "_session_db", None)
    resolver = getattr(db, "resolve_resume_session_id", None)
    if callable(resolver):
        try:
            resolved = resolver(sid)
        except (AttributeError, TypeError, ValueError, RuntimeError, OSError):
            resolved = None
        if isinstance(resolved, str) and resolved:
            return resolved
    return sid


def _same_session_id(left: Optional[str], right: Optional[str], agent: Any) -> bool:
    if not left or not right:
        return False
    return left == right or _lineage_tip(left, agent) == _lineage_tip(right, agent)


def sender_identity(agent: Any) -> dict[str, Optional[str]]:
    """Stable model-facing identity for the authoring agent."""
    subagent_id = getattr(agent, "_subagent_id", None)
    session_id = getattr(agent, "session_id", None)
    if isinstance(subagent_id, str) and subagent_id:
        return {"id": subagent_id, "session_id": session_id, "name": subagent_id, "kind": "subagent"}
    if isinstance(session_id, str) and session_id:
        return {"id": "lead", "session_id": session_id, "name": "lead", "kind": "root"}
    return {"id": None, "session_id": None, "name": None, "kind": None}


def _entry_public(entry: dict[str, Any]) -> dict[str, Any]:
    state = entry.get("state")
    return {
        "message_id": entry.get("message_id"),
        "state": state,
        "accepted": state in ("accepted", "delivered", "processed"),
        "delivered": state in ("delivered", "processed") or bool(entry.get("delivered_at")),
        "processed": state == "processed",
        "needs_review": state == "needs_review",
        "delivery": entry.get("delivery"),
        "reason": entry.get("reason"),
        "sender": entry.get("sender"),
        "recipient": entry.get("recipient"),
        "error": entry.get("error"),
        "round": entry.get("round"),
        "tree_id": entry.get("tree_id"),
    }


def _mirror(entry: dict[str, Any]) -> dict[str, Any]:
    """Keep a bounded process-local copy for the synchronous ``message_status`` API.
    The durable row remains authoritative for admission and every transition."""
    with _LEDGER_LOCK:
        _LEDGER[str(entry["message_id"])] = dict(entry)
        while len(_LEDGER) > _LEDGER_CAP:
            _LEDGER.popitem(last=False)
    return entry


def _mirror_row(row: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not row:
        return None
    entry = {
        "message_id": row.get("message_id"),
        "state": row.get("state"),
        "delivery": row.get("delivery"),
        "reason": row.get("reason"),
        "error": row.get("error"),
        "round": row.get("round"),
        "tree_id": row.get("tree_id"),
        "delivered_at": row.get("delivered_at"),
        "processed_at": row.get("processed_at"),
        "sender": {"id": row.get("sender_id"), "session_id": row.get("sender_session_id"),
                   "name": row.get("sender_id"), "kind": "subagent" if row.get("sender_id") != "lead" else "root"},
        "recipient": {"id": row.get("recipient_id"), "session_id": row.get("recipient_session_id"),
                      "kind": "subagent"},
        "content": "",
        "target_input": row.get("target_input"),
    }
    return _mirror(entry)


def _store_path(agent: Any) -> Optional[str]:
    return collab.db_path_for(agent)


def message_status(message_id: str) -> dict[str, Any]:
    with _LEDGER_LOCK:
        entry = _LEDGER.get(str(message_id or ""))
        if entry is None:
            return {"message_id": message_id, "state": "unknown", "accepted": False}
        return _entry_public(entry)


def _mark_entry(message_id: str, *, state: str, delivery: Optional[str] = None,
                reason: Optional[str] = None, error: Optional[str] = None) -> None:
    with _LEDGER_LOCK:
        entry = _LEDGER.get(message_id)
        if entry is None:
            return
        if entry.get("state") == "processed" and state != "processed":
            return
        entry["state"] = state
        entry["delivery"] = delivery if delivery is not None else entry.get("delivery")
        entry["reason"] = reason
        entry["error"] = error
        entry["updated_at"] = _now()
        done = entry.get("done")
        if state != "pending" and done is not None:
            done.set()


def mark_message_delivered(message_id: str, db_path: Optional[str] = None) -> None:
    """The message became a user row in the target transcript."""
    mid = str(message_id or "")
    if not mid:
        return
    _mark_entry(mid, state="delivered", delivery="conversation")
    if db_path:
        try:
            collab.mark_delivered(db_path, mid)
        except collab.CollabStoreError:
            logger.warning("collaboration delivery write failed for %s", mid, exc_info=True)


def mark_message_delivered_if_recorded(message_id: str, db: Any, session_ids: Any,
                                       db_path: Optional[str] = None) -> bool:
    """Promote to ``delivered`` only after the exact message is a stored user row.

    A cold continuation runs ``run_conversation`` in a worker; the user row is
    materialized inside that turn. Stamping at the call boundary would claim
    delivery even when the turn fails before writing anything. This check is the
    honest boundary: no row, no ``delivered``. Database errors are swallowed and
    leave the entry at ``accepted`` (admission already happened).
    """
    mid = str(message_id or "")
    if not mid or db is None:
        return False
    marker = f"[agent message {mid}]"
    if isinstance(session_ids, str):
        session_ids = [session_ids]
    seen = set()
    for sid in session_ids or ():
        if not isinstance(sid, str) or not sid or sid in seen:
            continue
        seen.add(sid)
        try:
            rows = db.get_messages_as_conversation(sid)
        except Exception:
            logger.debug("message delivery transcript lookup failed for %s", sid, exc_info=True)
            continue
        for row in rows or ():
            if (isinstance(row, dict) and row.get("role") == "user"
                    and marker in str(row.get("content") or "")):
                mark_message_delivered(mid, db_path or getattr(db, "db_path", None))
                return True
    return False


# ── Inbox (busy target) ─────────────────────────────────────────────────────

def _inbox(agent: Any) -> list:
    inbox = getattr(agent, _INBOX_ATTR, None)
    if not isinstance(inbox, list):
        inbox = []
        setattr(agent, _INBOX_ATTR, inbox)
    return inbox


def _inbox_lock(agent: Any) -> Any:
    lock = getattr(agent, _INBOX_LOCK_ATTR, None)
    if lock is None or not hasattr(lock, "__enter__"):
        lock = threading.Lock()
        setattr(agent, _INBOX_LOCK_ATTR, lock)
    return lock


def enqueue_agent_inbox(agent: Any, entry: dict[str, Any]) -> bool:
    """Append one immutable entry. Returns False for a cancelled/closing agent."""
    if getattr(agent, "_interrupt_requested", False):
        return False
    with _inbox_lock(agent):
        _inbox(agent).append(dict(entry))
    return True


def _pop_inbox(agent: Any) -> list[dict[str, Any]]:
    with _inbox_lock(agent):
        entries = list(_inbox(agent))
        _inbox(agent).clear()
    return entries


def _message_row(entries: list[dict[str, Any]]) -> dict[str, Any]:
    blocks = []
    for entry in entries:
        blocks.append(
            f"[agent message {entry['message_id']}]\n"
            f"From agent {entry.get('sender_name') or entry.get('sender_id')} "
            f"({entry.get('sender_id')}):\n{entry.get('content', '')}"
        )
    return {"role": "user", "content": "\n\n".join(blocks), "display_kind": "agent_message"}


def _mark_entries_delivered(entries: list[dict[str, Any]], agent: Any) -> None:
    db_path = _store_path(agent)
    delivered = getattr(agent, _DELIVERED_IDS_ATTR, None)
    if not isinstance(delivered, list):
        delivered = []
        setattr(agent, _DELIVERED_IDS_ATTR, delivered)
    for entry in entries:
        mid = str(entry.get("message_id") or "")
        mark_message_delivered(mid, db_path)
        if mid and mid not in delivered:
            delivered.append(mid)


def inject_inbox_messages(agent: Any, messages: list) -> int:
    """Drain the inbox into ONE new user row after the newest tool result.

    Called at the step boundary. If there is no tool result yet, entries stay in
    the inbox for the next boundary or the turn-end hook.
    """
    if not isinstance(messages, list) or not messages:
        return 0
    tool_idx = next((i for i in range(len(messages) - 1, -1, -1)
                     if isinstance(messages[i], dict) and messages[i].get("role") == "tool"), None)
    if tool_idx is None:
        return 0
    entries = _pop_inbox(agent)
    if not entries:
        return 0
    messages.insert(tool_idx + 1, _message_row(entries))
    _mark_entries_delivered(entries, agent)
    return len(entries)


def reopen_finished_turn_for_inbox(agent: Any, messages: list) -> bool:
    """Turn-end hook: an accepted message that lost the final step race gets one
    more iteration in the same conversation. Bounded per turn."""
    if not isinstance(messages, list) or getattr(agent, "_interrupt_requested", False):
        return False
    reopens = getattr(agent, _REOPEN_ATTR, 0)
    if not isinstance(reopens, int) or reopens >= _MAX_REOPENS:
        return False
    entries = _pop_inbox(agent)
    if not entries:
        return False
    messages.append(_message_row(entries))
    setattr(agent, _REOPEN_ATTR, reopens + 1)
    _mark_entries_delivered(entries, agent)
    return True


def reject_undelivered_inbox(agent: Any, reason: str) -> int:
    """Close-time honesty: entries still queued when the run ends were accepted
    but never processed. Mark them rejected rather than leaving a false
    ``accepted`` state behind."""
    entries = _pop_inbox(agent)
    db_path = _store_path(agent)
    for entry in entries:
        mid = str(entry.get("message_id") or "")
        _mark_entry(mid, state="rejected", delivery=None, reason=reason)
        if db_path and mid:
            try:
                collab.mark_rejected(db_path, mid, reason)
            except collab.CollabStoreError:
                logger.warning("collaboration rejection write failed for %s", mid, exc_info=True)
    return len(entries)


# ── Target resolution ──────────────────────────────────────────────────────

def _live_records() -> dict[str, dict[str, Any]]:
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
    with _active_subagents_lock:
        return dict(_active_subagents)


def _recent_records() -> dict[str, dict[str, Any]]:
    from tools.delegate_tool_registry import _active_subagents_lock, _recent_subagents
    with _active_subagents_lock:
        return dict(_recent_subagents)


def _resolve_target(sender: Any, target: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    target = str(target or "").strip()
    if not target:
        return None, "invalid_target"
    sender_info = sender_identity(sender)
    root_sid = _root_session_id(sender)
    if target in ("lead", "root"):
        if sender_info["kind"] == "root":
            return None, "self"
        root = _root_agent(sender)
        if root is None:
            return None, "root_unavailable"
        return {
            "id": "lead", "session_id": _root_session_id(root) or root_sid,
            "kind": "root", "agent": root, "record": None, "state": "running"
            if getattr(root, "status", None) == "running" or getattr(root, "_current_turn_id", None) else "inactive",
            "root_session_id": _root_session_id(root) or root_sid, "profile_key": _profile_key(root),
        }, None

    if target in (sender_info.get("id"), sender_info.get("session_id")):
        return None, "self"

    live = _live_records()
    recent = _recent_records()

    record = live.get(target)
    if record is None:
        record = next((r for r in live.values()
                       if str(getattr(r.get("agent"), "session_id", "") or "") == target), None)
    if record is not None:
        agent = record.get("agent")
        sid = record.get("subagent_id") or getattr(agent, "_subagent_id", None) or target
        if sid in (sender_info.get("id"), getattr(sender, "_subagent_id", None)):
            return None, "self"
        return {
            "id": sid, "session_id": getattr(agent, "session_id", None), "kind": "subagent",
            "agent": agent, "record": record, "state": "running",
            "root_session_id": record.get("root_session_id") or _root_session_id(agent),
            "profile_key": record.get("profile_key") or _profile_key(agent),
            "accepting_messages": record.get("accepting_messages", True),
        }, None

    record = recent.get(target)
    if record is None:
        record = next((r for r in recent.values()
                       if str(r.get("session_id") or (r.get("resume") or {}).get("session_id") or "") == target), None)
    if record is not None:
        sid = record.get("subagent_id") or target
        if sid == sender_info.get("id"):
            return None, "self"
        return {
            "id": sid, "session_id": record.get("session_id") or (record.get("resume") or {}).get("session_id"),
            "kind": "subagent", "agent": None, "record": record, "state": record.get("state") or "resumable",
            "root_session_id": record.get("root_session_id") or (record.get("resume") or {}).get("root_session_id"),
            "profile_key": record.get("profile_key") or _profile_key(record),
        }, None

    # Stage 4: a durable child survives the live/recent registry. Only a row whose
    # persisted root + profile match this sender is even considered.
    store_agent = _root_agent(sender) or sender
    if root_sid and _profile_key(sender):
        from tools.delegate_tool_resumable import load_tree_member
        durable = load_tree_member(store_agent, root_sid, _profile_key(sender), target)
        if durable is not None:
            sid = durable.get("subagent_id") or target
            if sid == sender_info.get("id"):
                return None, "self"
            state = durable.get("state")
            state = "resumable" if state in ("resumable", "completed") else state
            record = {
                **durable, "subagent_id": sid, "state": state,
                "owner_agent_session_id": root_sid,
                "root_session_id": durable.get("root_session_id") or root_sid,
                "profile_key": durable.get("profile_key") or _profile_key(sender),
            }
            return {
                "id": sid, "session_id": durable.get("session_id"), "kind": "subagent",
                "agent": None, "record": record, "state": state,
                "root_session_id": record["root_session_id"], "profile_key": record["profile_key"],
            }, None

    return None, "unknown_target"


def _authorized(sender: Any, target: dict[str, Any]) -> tuple[bool, Optional[str]]:
    sender_root = _root_session_id(sender)
    target_root = target.get("root_session_id")
    if not sender_root or not target_root:
        return False, "unauthorized"
    if not _same_session_id(sender_root, target_root, sender):
        return False, "unauthorized"
    sender_profile, target_profile = _profile_key(sender), target.get("profile_key")
    if isinstance(target_profile, (str, Path)) and str(target_profile):
        target_profile = _normalize_path(target_profile)
    if not sender_profile or not target_profile or sender_profile != target_profile:
        return False, "unauthorized"
    return True, None


# ── Send engine ────────────────────────────────────────────────────────────

def _fingerprint(sender_id: str, target_input: str, content: str) -> str:
    return collab.fingerprint_for(sender_id, target_input, content)


def _reject(message_id: Optional[str], reason: str, error: Optional[str] = None) -> dict[str, Any]:
    return {"message_id": message_id, "state": "rejected", "accepted": False,
            "reason": reason, "error": error}


def _accepted_row(row: Optional[dict[str, Any]], reason: str = "internal_error") -> dict[str, Any]:
    if row is None:
        return _reject(None, reason)
    entry = _mirror_row(row)
    return _entry_public(entry) if entry else _reject(row.get("message_id"), reason)


def _owner_agent_for(target: dict[str, Any], sender: Any) -> tuple[Optional[Any], Optional[str]]:
    """The live agent that should own a cold rebuild (preserves the target's parent)."""
    record = target.get("record") or {}
    owner_sid = str(record.get("owner_agent_session_id") or "")
    sender_sid = getattr(sender, "session_id", None)
    if owner_sid and owner_sid == sender_sid:
        return sender, None
    root = _root_agent(sender)
    root_sid = _root_session_id(root) if root is not None else None
    if owner_sid and root_sid and _same_session_id(owner_sid, root_sid, sender):
        return root, None
    for rec in _live_records().values():
        agent = rec.get("agent")
        if agent is not None and str(getattr(agent, "session_id", "") or "") == owner_sid:
            return agent, None
    if not owner_sid:
        return root, None
    return None, "owner_unavailable"


def _enqueue(agent: Any, payload: dict[str, Any]) -> bool:
    return enqueue_agent_inbox(agent, {
        "message_id": payload["message_id"],
        "sender_id": payload["sender"]["id"],
        "sender_name": payload["sender"]["name"],
        "content": payload["content"],
    })


def _send_to_live(payload: dict[str, Any], target: dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Queue into the live target's inbox under the registry lock."""
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock

    agent = target.get("agent")
    target_id = target.get("id")
    with _active_subagents_lock:
        record = _active_subagents.get(target_id)
        if record is None or record.get("agent") is not agent:
            return False, "target_finished_before_delivery"
        if record.get("accepting_messages", True) is False or getattr(agent, "_interrupt_requested", False):
            return False, "target_cancelled"
        if not _enqueue(agent, payload):
            return False, "target_cancelled"
    return True, None


def _dispatch_cold(payload: dict[str, Any], target: dict[str, Any], sender: Any,
                   db_path: str, tree_id: str) -> dict[str, Any]:
    mid = payload["message_id"]
    # The reserve transaction and the rebuild are not one atomic step: a root stop can
    # reject this row in between. Re-read the durable admission before starting any run,
    # so a row a stop already rejected is never dispatched (no in-flight resurrection).
    current = collab.get_message(db_path, mid)
    if current is None or current.get("state") != "accepted":
        return _rejected_from_store(db_path, mid)
    record = target.get("record") or {}
    state = record.get("state") or "resumable"
    if state != "resumable":
        collab.release_dispatch(db_path, tree_id, payload["message_id"],
                                "target_stopping" if state == "stopping" else "target_closed")
        return _rejected_from_store(db_path, payload["message_id"])
    if not record.get("resume") and not record.get("session_id") and not record.get("route_request"):
        collab.release_dispatch(db_path, tree_id, payload["message_id"], "not_continuable")
        return _rejected_from_store(db_path, payload["message_id"])
    owner, err = _owner_agent_for(target, sender)
    if owner is None:
        collab.release_dispatch(db_path, tree_id, payload["message_id"], err or "owner_unavailable")
        return _rejected_from_store(db_path, payload["message_id"])

    from tools.delegate_tool_continue import continue_subagent

    text = (
        f"[agent message {payload['message_id']}]\n"
        f"From agent {payload['sender']['name']} ({payload['sender']['id']}):\n{payload['content']}"
    )
    try:
        raw = continue_subagent(target.get("id"), text, owner, background=True,
                                message_meta={"message_id": payload["message_id"],
                                              "round": payload.get("round")})
        result = json.loads(raw) if isinstance(raw, str) else {}
    except Exception:  # health: allow BLE001 -- ambiguity becomes needs_review, never a retry
        logger.warning("cold message dispatch outcome unknown for %s", payload["message_id"], exc_info=True)
        collab.mark_needs_review(db_path, payload["message_id"], "dispatch_outcome_unknown")
        collab.release_round(db_path, tree_id)
        return _accepted_row(collab.get_message(db_path, payload["message_id"]), "needs_review")
    if result.get("error"):
        collab.release_dispatch(db_path, tree_id, payload["message_id"],
                                "continuation_refused", str(result["error"]))
        return _rejected_from_store(db_path, payload["message_id"])
    return _accepted_row(collab.get_message(db_path, payload["message_id"]))


def _rejected_from_store(db_path: str, message_id: str) -> dict[str, Any]:
    row = collab.get_message(db_path, message_id)
    entry = _mirror_row(row)
    return _entry_public(entry) if entry else _reject(message_id, "continuation_refused")


def _duplicate_response(db_path: str, mid: str, sender_id: str, target_text: str,
                        text: str) -> dict[str, Any]:
    """An id collision during admission: the winner may be this same sender retrying
    concurrently (idempotent) or another sender (conflict)."""
    row = collab.get_message(db_path, mid)
    if (row is not None and str(row.get("sender_id")) == sender_id
            and row.get("fingerprint") == _fingerprint(sender_id, target_text, text)):
        entry = _mirror_row(row)
        return _entry_public(entry) if entry else _reject(mid, "unknown_target")
    return _reject(mid, "message_id_conflict",
                   "message_id already belongs to a different sender, recipient, or body")


def _deliver_root(root: Any, target_info: dict[str, Any], payload: dict[str, Any], db_path: str,
                  tree_id: str, sender: dict[str, Any], target_text: str, text: str,
                  round_no: int) -> dict[str, Any]:
    mid = payload["message_id"]
    root_sid = target_info.get("session_id") or (_root_session_id(root) if root is not None else None)
    if not root_sid:
        return _reject(mid, "root_unavailable")
    busy = root is not None and (getattr(root, "status", None) == "running"
                                 or getattr(root, "_current_turn_id", None))
    if busy:
        # A root in a turn keeps the existing next-step inbox: one new user row at the next step
        # boundary (or the bounded turn-end reopen), never an interrupting turn.
        row, why = collab.begin_inbox_message(
            db_path, tree_id, mid, sender_id=str(sender["id"]),
            sender_session_id=sender.get("session_id"), recipient_id="lead",
            recipient_session_id=root_sid, target_input=target_text,
            fingerprint=_fingerprint(str(sender["id"]), target_text, text), round_no=round_no)
        if row is None:
            return (_duplicate_response(db_path, mid, str(sender["id"]), target_text, text)
                    if why == "duplicate" else _reject(mid, _admission_reason(why)))
        if not _enqueue(root, payload):
            collab.mark_rejected(db_path, mid, "target_cancelled")
            return _rejected_from_store(db_path, mid)
        return _accepted_row(row)

    # An idle root has no turn to drain an in-memory inbox. Admit through the native gateway route
    # (the owning authority's FIFO), with the durable message row written first.
    row, why = collab.begin_wake_message(
        db_path, tree_id, mid, sender_id=str(sender["id"]),
        sender_session_id=sender.get("session_id"), recipient_id="lead",
        recipient_session_id=root_sid, target_input=target_text,
        fingerprint=_fingerprint(str(sender["id"]), target_text, text), round_no=round_no)
    if row is None:
        return (_duplicate_response(db_path, mid, str(sender["id"]), target_text, text)
                if why == "duplicate" else _reject(mid, _admission_reason(why)))
    from tools.delegate_tool_wake import cancel_root_wakes, request_root_wake, wake_header
    profile = target_info.get("profile_key") or _profile_key(sender)
    wake = request_root_wake(
        root_session_id=str(root_sid), profile_key=profile,
        text=wake_header(mid, str(sender["id"]), sender.get("name"), text),
        message_id=mid, tree_id=tree_id)
    if not wake.get("ok"):
        reason = str(wake.get("reason") or "target_idle")
        if wake.get("unknown"):
            # The native admission may have committed after the producer stopped waiting. A
            # rejected row would then be a run the user was told never happened.
            collab.mark_needs_review(db_path, mid, reason)
            return _accepted_row(collab.get_message(db_path, mid), reason)
        collab.mark_rejected(db_path, mid, reason, wake.get("error"))
        return _rejected_from_store(db_path, mid)
    # A stop can land between the durable admission and the native wake commit. The stop path
    # cancels queued agent-collab admissions; this re-check closes the newly-committed case.
    current_tree = collab.tree(db_path, tree_id)
    if current_tree is None or current_tree.get("state") != "active":
        cancel_root_wakes(profile, str(root_sid), wake.get("admission_id"))
        collab.mark_rejected(db_path, mid, "tree_stopped")
        return _rejected_from_store(db_path, mid)
    out = _accepted_row(row)
    out["native_receipt"] = wake.get("admission_id")
    out["delivery"] = "root_wake"
    return out


def _deliver_live(target_info: dict[str, Any], payload: dict[str, Any], db_path: str, tree_id: str,
                  sender: dict[str, Any], target_text: str, text: str,
                  round_no: int) -> dict[str, Any]:
    mid = payload["message_id"]
    row, why = collab.begin_inbox_message(
        db_path, tree_id, mid, sender_id=str(sender["id"]),
        sender_session_id=sender.get("session_id"), recipient_id=str(target_info["id"]),
        recipient_session_id=target_info.get("session_id"), target_input=target_text,
        fingerprint=_fingerprint(str(sender["id"]), target_text, text), round_no=round_no)
    if row is None:
        return (_duplicate_response(db_path, mid, str(sender["id"]), target_text, text)
                if why == "duplicate" else _reject(mid, _admission_reason(why)))
    ok, live_why = _send_to_live(payload, target_info)
    if not ok:
        collab.mark_rejected(db_path, mid, live_why or "target_cancelled")
        return _rejected_from_store(db_path, mid)
    return _accepted_row(row)


def _deliver_cold(target_info: dict[str, Any], payload: dict[str, Any], sender_agent: Any, db_path: str,
                  tree_id: str, sender: dict[str, Any], target_text: str, text: str,
                  round_no: int) -> dict[str, Any]:
    mid = payload["message_id"]
    row, why = collab.begin_dispatch_message(
        db_path, tree_id, mid, sender_id=str(sender["id"]),
        sender_session_id=sender.get("session_id"), recipient_id=str(target_info["id"]),
        recipient_session_id=target_info.get("session_id"), target_input=target_text,
        fingerprint=_fingerprint(str(sender["id"]), target_text, text), round_no=round_no)
    if row is None:
        return (_duplicate_response(db_path, mid, str(sender["id"]), target_text, text)
                if why == "duplicate" else _reject(mid, _admission_reason(why)))
    return _dispatch_cold({**payload, "round": round_no}, target_info, sender_agent, db_path, tree_id)


def send_agent_message(sender_agent: Any, target: Optional[str], message: Optional[str], *,
                       message_id: Optional[str] = None) -> dict[str, Any]:
    """Public engine behind ``agent_message``. Always returns an explicit state."""
    if sender_agent is None:
        return _reject(None, "missing_sender")
    text = str(message or "").strip()
    if not text:
        return _reject(None, "empty_message")
    mid = str(message_id).strip() if message_id is not None else f"msg-{uuid.uuid4().hex}"
    if not _MESSAGE_ID_RE.match(mid):
        return _reject(None, "invalid_message_id")

    sender = sender_identity(sender_agent)
    if not sender.get("id"):
        return _reject(None, "missing_sender")
    target_text = str(target or "").strip()
    db_path = _store_path(sender_agent)
    if not db_path:
        return _reject(mid, "store_unavailable")

    collab.maybe_recover(db_path)

    # Idempotency is a durable property: replay before target resolution so a retry
    # keeps working after the target left the live/recent registry.
    existing = collab.get_message(db_path, mid)
    if existing is not None and str(existing.get("sender_id")) == str(sender["id"]):
        if existing.get("fingerprint") != _fingerprint(str(sender["id"]), target_text, text):
            return _reject(mid, "message_id_conflict",
                           "message_id already belongs to a different sender, recipient, or body")
        entry = _mirror_row(existing)
        return _entry_public(entry) if entry else _reject(mid, "unknown_target")

    target_info, why = _resolve_target(sender_agent, target_text)
    if target_info is None:
        return _reject(mid, why or "invalid_target")
    allowed, why = _authorized(sender_agent, target_info)
    if not allowed:
        return _reject(mid, why or "unauthorized")

    tree_id = target_info.get("root_session_id") or _root_session_id(sender_agent)
    profile = _profile_key(sender_agent)
    try:
        t = collab.ensure_tree(db_path, tree_id, profile)
    except collab.CollabStoreError as exc:
        return _reject(mid, "store_unavailable", str(exc))
    if t is None:
        return _reject(mid, "store_unavailable")
    if t.get("state") != "active":
        return _reject(mid, "tree_stopped")
    sender_round = getattr(sender_agent, _COLLAB_ROUND_ATTR, 0)
    round_no = (int(sender_round) if isinstance(sender_round, int) else 0) + 1
    if round_no > int(t.get("max_rounds") or 0):
        return _reject(mid, "rounds_exhausted")

    payload = {"message_id": mid, "sender": sender, "content": text,
               "target_input": target_text, "round": round_no}

    try:
        if target_info["kind"] == "root":
            return _deliver_root(target_info.get("agent"), target_info, payload, db_path, tree_id,
                                 sender, target_text, text, round_no)
        record = target_info.get("record") or {}
        if target_info.get("agent") is not None and record.get("agent") is not None:
            return _deliver_live(target_info, payload, db_path, tree_id, sender, target_text, text, round_no)
        return _deliver_cold(target_info, payload, sender_agent, db_path, tree_id,
                             sender, target_text, text, round_no)
    except collab.CollabStoreError as exc:
        return _reject(mid, "store_unavailable", str(exc))
    except Exception as exc:  # health: allow BLE001 -- every failure becomes an explicit state
        logger.warning("agent message %s failed after admission", mid, exc_info=True)
        try:
            if collab.mark_needs_review(db_path, mid, "dispatch_outcome_unknown"):
                collab.release_round(db_path, tree_id)
        except collab.CollabStoreError:
            pass
        return _reject(mid, "internal_error", str(exc))


def _admission_reason(why: Optional[str]) -> str:
    return {
        "duplicate": "message_id_conflict",
        "unknown_tree": "store_unavailable",
        "rounds_exhausted": "rounds_exhausted",
        "concurrency_exhausted": "concurrency_exhausted",
        "tree_stopped": "tree_stopped",
    }.get(str(why), str(why or "rejected"))


_ROOT_WAKE_IDS_ATTR = "_delegate_root_wake_ids"


def refresh_root_wakes(agent: Any) -> int:
    """Promote root-bound idle wakes to ``delivered`` only after the exact stored user row.

    Called at the turn's step boundary. A row that never materialized keeps ``accepted`` (its
    native wake may still be queued behind another turn); it is never inferred delivered.
    """
    if getattr(agent, "_subagent_id", None):
        return 0
    root_sid = getattr(agent, "session_id", None)
    db_path = _store_path(agent)
    if not isinstance(root_sid, str) or not root_sid or not db_path:
        return 0
    ids = getattr(agent, _ROOT_WAKE_IDS_ATTR, None)
    if not isinstance(ids, list):
        ids = []
        setattr(agent, _ROOT_WAKE_IDS_ATTR, ids)
    promoted = 0
    for row in collab.pending_root_wakes(db_path, root_sid, states=("accepted",)):
        mid = str(row.get("message_id") or "")
        if not mid or not collab.message_materialized(db_path, row):
            continue
        if collab.mark_delivered(db_path, mid, root_sid):
            _mark_entry(mid, state="delivered", delivery="root_wake")
            promoted += 1
        if mid not in ids:
            ids.append(mid)
    return promoted


def _mark_root_wakes_processed(agent: Any, outcome: str) -> None:
    ids = getattr(agent, _ROOT_WAKE_IDS_ATTR, None)
    if not isinstance(ids, list) or not ids:
        return
    db_path = _store_path(agent)
    if db_path:
        for mid in list(ids):
            try:
                if collab.mark_processed(db_path, str(mid), outcome):
                    _mark_entry(str(mid), state="processed", delivery="root_wake")
            except collab.CollabStoreError:
                logger.warning("root wake processed write failed for %s", mid, exc_info=True)
    ids.clear()


def note_user_stop(db_path: Optional[str], root_session_id: Optional[str],
                   reason: str = "root_stopped", *, root_agent: Any = None) -> bool:
    """Durable stop at the native stop admission, before the turn exits.

    Safe to call for a session with no collaboration tree (no-op). It closes queued root work and
    interrupts live members, so a crash after the Stop signal cannot leave an active tree behind.
    """
    root_sid = str(root_session_id or "")
    if not db_path or not root_sid:
        return False
    try:
        if collab.tree(db_path, root_sid) is None:
            return False
        collab.stop_tree(db_path, root_sid, reason)
    except collab.CollabStoreError:
        logger.warning("durable collaboration stop failed for %s", root_sid, exc_info=True)
        return False
    if root_agent is not None:
        reject_undelivered_inbox(root_agent, "tree_stopped")
    for record in _live_records().values():
        if record.get("root_session_id") != root_sid:
            continue
        record["accepting_messages"] = False
        member = record.get("agent")
        if member is not None:
            member._interrupt_requested = True
            reject_undelivered_inbox(member, "tree_stopped")
    return True


def register_continuation_or_refuse(child: Any, parent_agent: Any, register: Any) -> bool:
    """Register a cold continuation under the same tree lock stop takes.

    A stop that committed first leaves the tree stopped; the continuation is rejected and never
    registered (so no run can start). A stop that lands after registration sees the live member
    and interrupts it. There is no window where a stopped tree gains an unregistered run.
    """
    meta = getattr(child, "_delegate_message_meta", None)
    if not isinstance(meta, dict) or not meta.get("message_id"):
        register()
        return True
    root_sid = (getattr(child, "_delegate_root_session_id", None)
                or getattr(parent_agent, "session_id", None))
    db_path = _store_path(parent_agent)
    if not isinstance(root_sid, str) or not root_sid or not db_path:
        register()
        return True
    mid = str(meta["message_id"])
    with collab.tree_lock(root_sid):
        try:
            t = collab.tree(db_path, root_sid)
        except collab.CollabStoreError:
            t = None
        if t is not None and t.get("state") != "active":
            try:
                collab.mark_rejected(db_path, mid, "tree_stopped")
            except collab.CollabStoreError:
                logger.warning("rejecting stopped continuation %s failed", mid, exc_info=True)
            child._delegate_collab_refused = "tree_stopped"
            return False
        register()
    return True


# ── run boundaries ─────────────────────────────────────────────────────────

def run_outcome(result: Any, agent: Any) -> str:
    """The consuming run's real outcome: ``completed|failed|interrupted|incomplete``.

    The turn finalizer clears ``_interrupt_requested`` before this hook runs, so the
    envelope's own ``interrupted`` bit is authoritative; the agent flag is only a
    fallback for callers that never built a finalizer result.
    """
    if isinstance(result, dict):
        if result.get("interrupted"):
            return "interrupted"
        if result.get("error"):
            return "failed"
        if result.get("completed") is False:
            return "incomplete"
    if getattr(agent, "_interrupt_requested", False):
        return "interrupted"
    return "completed"


def gateway_shutdown_ended_turn(result: Any, agent: Any) -> bool:
    """Whether this turn's recorded interrupt was the gateway's resumable shutdown.

    ``interrupt_issuer`` slugs the trusted ``tool_reason`` the gateway passes
    (``gateway.run._INTERRUPT_TOOL_REASON_GATEWAY_SHUTDOWN`` == ``interrupt_control._REASON_GATEWAY_SHUTDOWN``)
    into ``interrupted_by_system(gateway_shutdown)`` / ``interrupted_during_api_call(gateway_shutdown)``.
    A plain user stop has no issuer (``interrupted_by_user`` / ``interrupted_during_api_call``).

    Attribution is per turn, never a process-global "shutdown seen once" flag: a user Stop that
    lands before or after a concurrent shutdown keeps its own reason (native first-cause rules),
    and a later turn cannot inherit it.
    """
    reason = ""
    if isinstance(result, dict):
        reason = str(result.get("turn_exit_reason") or "")
    if not reason:
        reason = str(getattr(agent, "_tool_interrupt_reason", None) or "")
    if not reason:
        return False
    if reason == "gateway shutdown":
        return True
    if reason.startswith(("interrupted_by_system(", "interrupted_during_api_call(")):
        return "gateway_shutdown" in reason
    return False


def _turn_was_interrupted(result: Any, agent: Any) -> bool:
    if isinstance(result, dict) and "interrupted" in result:
        return bool(result.get("interrupted"))
    return bool(getattr(agent, "_interrupt_requested", False))


def _mark_delivered_processed(agent: Any, outcome: str) -> None:
    ids = getattr(agent, _DELIVERED_IDS_ATTR, None)
    if not isinstance(ids, list) or not ids:
        return
    db_path = _store_path(agent)
    if db_path:
        for mid in list(ids):
            try:
                if collab.mark_processed(db_path, str(mid), outcome):
                    _mark_entry(str(mid), state="processed", delivery="conversation")
            except collab.CollabStoreError:
                logger.warning("collaboration processed write failed for %s", mid, exc_info=True)
    ids.clear()


def _tree_scope_ids(sid: Any, agent: Any) -> list[str]:
    """The tree ids a turn's root id can own: itself, its compression lineage and its tip.

    Compression moves the live agent to a physical tip while durable tree/member rows keep the
    id they were registered under. A Stop must close all of them, or a later message registered
    after the rotation survives the person's Stop.
    """
    ids = [str(sid)]
    db = getattr(agent, "_session_db", None)
    for attr in ("get_compression_lineage", "get_compression_tip"):
        try:
            value = getattr(db, attr)(str(sid))
        except Exception:  # health: allow BLE001 -- the id itself is always tried
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for candidate in values:
            if isinstance(candidate, str) and candidate and candidate not in ids:
                ids.append(candidate)
    return ids


def finish_collaboration_run(agent: Any, result: Any) -> None:
    """Single exit hook for every turn envelope.

    A root whose turn was a USER interruption closes its tree: queued work is rejected and
    later sends/recovery cannot resurrect it. A gateway shutdown is the opposite case — the
    native lifecycle marks its children resumable, so the tree must stay recoverable rather
    than be durably stopped by a stop that no person asked for. Messages this run delivered
    are ``processed`` (the run consumed them); an outcome is recorded, not a success claim.
    """
    db_path = _store_path(agent)
    if not db_path:
        return
    if not getattr(agent, "_subagent_id", None) and _turn_was_interrupted(result, agent):
        root_sid = getattr(agent, "session_id", None)
        if isinstance(root_sid, str) and root_sid:
            if gateway_shutdown_ended_turn(result, agent):
                # Resumable shutdown: the child records and this tree stay recoverable. A
                # message row admitted before the shutdown stays accepted; it is NOT a
                # replayable body, so restart recovery still rejects it as undelivered.
                logger.info("collaboration tree %s kept active across a resumable gateway shutdown",
                            root_sid)
            else:
                tree_ids = _tree_scope_ids(root_sid, agent)
                for tree_id in tree_ids:
                    collab.stop_tree(db_path, tree_id, "root_stopped")
                # The root's own next-step queue is closed too: a later step boundary in
                # this process must not inject a message the durable tree already rejected.
                reject_undelivered_inbox(agent, "tree_stopped")
                # A wake admitted before this Stop must not run after it.
                try:
                    from tools.delegate_tool_wake import cancel_root_wakes
                    for tree_id in tree_ids:
                        cancel_root_wakes(_profile_key(agent), tree_id)
                except Exception:  # health: allow BLE001 -- the durable stop already stands
                    logger.debug("cancelling queued collaboration wakes failed", exc_info=True)
                for record in _live_records().values():
                    if record.get("root_session_id") not in tree_ids:
                        continue
                    record["accepting_messages"] = False
                    member = record.get("agent")
                    if member is not None:
                        # Stop work this tree already dispatched; a message-driven
                        # continuation is owned by the root but not in its turn's child list.
                        member._interrupt_requested = True
                        reject_undelivered_inbox(member, "tree_stopped")
    _mark_delivered_processed(agent, run_outcome(result, agent))
    _mark_root_wakes_processed(agent, run_outcome(result, agent))


def finish_message_run(db_path: Optional[str], message_id: Optional[str], delivered: bool,
                       outcome: str) -> None:
    """Cold continuation boundary: ``processed`` only after the exact row was
    materialized; a run that failed before persistence is rejected, never claimed."""
    if not db_path or not message_id:
        return
    try:
        if delivered:
            if collab.mark_processed(db_path, str(message_id), outcome):
                _mark_entry(str(message_id), state="processed", delivery="conversation")
        else:
            reason = "continuation_failed_before_delivery"
            if collab.mark_rejected(db_path, str(message_id), reason):
                _mark_entry(str(message_id), state="rejected", delivery=None, reason=reason)
    except collab.CollabStoreError:
        logger.warning("cold collaboration terminal write failed for %s", message_id, exc_info=True)


def refresh_cold_message_delivery(agent: Any) -> None:
    """Turn-start verification for a message-driven continuation.

    ``run_conversation`` persists the user turn before the first step boundary, so
    this promotes ``accepted`` to ``delivered`` while the run is still going (not only
    at run end). A missing row changes nothing; the terminal boundary uses the same
    exact-marker check, never a call-boundary guess.
    """
    meta = getattr(agent, "_delegate_message_meta", None)
    if not isinstance(meta, dict) or getattr(agent, "_delegate_message_delivered", False) is True:
        return
    mid = meta.get("message_id")
    db = getattr(agent, "_session_db", None)
    if not mid or db is None:
        return
    try:
        db_path = getattr(db, "db_path", None)
        if mark_message_delivered_if_recorded(
                str(mid), db, [getattr(agent, "session_id", None), meta.get("session_id")],
                db_path=str(db_path) if db_path else None):
            agent._delegate_message_delivered = True
    except Exception:
        logger.debug("cold delivery refresh failed for %s", mid, exc_info=True)


# ── Discovery ──────────────────────────────────────────────────────────────

def list_tree_agents(caller_agent: Any) -> list[dict[str, Any]]:
    """Model-facing tree rows for the caller's own root/profile only."""
    if caller_agent is None:
        return []
    root = _root_agent(caller_agent)
    if root is None:
        return []
    root_sid = _root_session_id(root)
    profile = _profile_key(caller_agent)
    caller_info = sender_identity(caller_agent)
    rows: dict[str, dict[str, Any]] = {
        "lead": {
            "id": "lead", "session_id": root_sid, "name": "lead", "kind": "root",
            "status": "running" if getattr(root, "status", None) == "running" else "inactive",
            "parent_id": None, "goal": None, "self": caller_info.get("kind") == "root",
        }
    }

    def same_scope(record: dict[str, Any]) -> bool:
        record_root = record.get("root_session_id")
        if not _same_session_id(root_sid, record_root, caller_agent):
            return False
        record_profile = record.get("profile_key") or _profile_key(record)
        if isinstance(record_profile, (str, Path)) and str(record_profile):
            record_profile = _normalize_path(record_profile)
        return not (profile and record_profile and profile != record_profile)

    for sid, record in _live_records().items():
        if not same_scope(record):
            continue
        agent = record.get("agent")
        rows[sid] = {
            "id": sid, "session_id": getattr(agent, "session_id", None), "name": sid,
            "kind": "subagent", "status": "running" if getattr(agent, "status", None) == "running" else "inactive",
            "parent_id": record.get("parent_id"), "goal": record.get("goal"),
            "self": sid == caller_info.get("id"),
        }
    for sid, record in _recent_records().items():
        if sid in rows or not same_scope(record):
            continue
        state = record.get("state") or "resumable"
        rows[sid] = {
            "id": sid, "session_id": record.get("session_id") or (record.get("resume") or {}).get("session_id"),
            "name": sid, "kind": "subagent",
            "status": "stopping" if state == "stopping" else "inactive",
            "parent_id": record.get("parent_id"), "goal": record.get("goal"),
            "self": sid == caller_info.get("id"),
        }

    # Stage 4: durable members are discoverable after the live/recent registry is gone.
    if root_sid and profile:
        store_agent = root or caller_agent
        collab.maybe_recover(collab.db_path_for(store_agent))
        try:
            from tools.delegate_tool_resumable import list_tree_members
            durable_rows = list_tree_members(store_agent, root_sid, profile)
        except Exception:  # health: allow BLE001 -- discovery degrades to live/recent only
            durable_rows = []
        for record in durable_rows:
            sid = record.get("subagent_id")
            if not sid or sid in rows:
                continue
            rows[sid] = {
                "id": sid, "session_id": record.get("session_id"), "name": sid, "kind": "subagent",
                "status": "durable", "parent_id": record.get("parent_id"),
                "goal": record.get("goal"), "self": sid == caller_info.get("id"),
            }
    return [rows[key] for key in sorted(rows)]
