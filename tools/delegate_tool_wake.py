"""Native idle-root wake for same-tree collaboration.

A tool handler runs in a worker thread inside the gateway process. It has no adapter, but the
gateway already publishes exactly the handles a producer needs: ``initialize_session_authority``
binds ``(pid, authority, running_loop)`` in ``gateway.session_cron`` for every served profile, and
``gateway.session_automation.admit_route_wake`` commits a producer event through the real runtime
store guards. This module uses those two native facts; it adds no scheduler, runner, adapter
registry, poller or second delivery path.

A wake is admitted only for the root's OWN live route in the OWNING profile home:
same state.db, same session, same committed route, adapter resolved by the runner. The text is the
normal addressed-message header. The admission identity is deterministic
(``["agent-collab", route, tree:message]``), so a retried wake replays its durable native ledger
receipt instead of starting a second turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("tools.delegate_tool")

WAKE_TIMEOUT_SECONDS = 30.0
WAKE_PRODUCER = "agent-collab"


def _norm(path: Any) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def profile_home(profile_key: Any) -> Optional[Path]:
    """The profile home a tree's ``profile_key`` names, else the process home."""
    if isinstance(profile_key, (str, Path)) and str(profile_key):
        path = Path(str(profile_key))
        if path.name == "state.db":
            return path.parent
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:  # health: allow BLE001 -- no home means no native owner, fail closed
        return None


def _refusal(reason: str, error: Optional[str] = None, *, unknown: bool = False) -> dict[str, Any]:
    return {"ok": False, "reason": reason, "error": error, "unknown": unknown}


def _receipt_id(receipt: Any) -> Optional[str]:
    for attr in ("admission_id", "request_id"):
        value = getattr(receipt, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(receipt, dict):
        for key in ("admission_id", "request_id"):
            value = receipt.get(key)
            if isinstance(value, str) and value:
                return value
    return None


async def _admit(profile_home_path: Path, root_session_id: str, text: str,
                 message_id: str, tree_id: str) -> dict[str, Any]:
    """Admit on the owner loop; runs inside the gateway process and its profile scope."""
    from gateway.session_cron import owner_for_home
    bound = owner_for_home(profile_home_path)
    if bound is None:
        return _refusal("target_idle", "no live gateway owns this profile")
    authority, _loop = bound
    if _norm(Path(authority.db.db_path).parent) != _norm(profile_home_path):
        return _refusal("wrong_profile", "the owner loop does not serve this tree's state.db")

    root_sid = str(root_session_id or "")
    if not root_sid:
        return _refusal("unknown_target")
    logical = getattr(authority, "logical_owner", lambda sid: sid)
    owner_sid = logical(root_sid)
    if not isinstance(owner_sid, str) or not owner_sid:
        return _refusal("unknown_target")

    row = authority.db.get_session(owner_sid)
    if row is None:
        return _refusal("unknown_target", "the addressed root has no stored session here")
    expected_profile = str(getattr(authority, "profile_id", "") or "default")
    row_profile = str(row.get("profile_name") or "")
    if row_profile and row_profile != expected_profile:
        return _refusal("wrong_profile", "the stored root belongs to another profile")

    live = authority.sessions.get(owner_sid)
    if live is None or live.source is None:
        return _refusal("target_idle", "the root route is not mounted in this gateway")
    route = str(getattr(live, "route", "") or "")
    if not route:
        return _refusal("target_idle")
    entry = authority.runner.session_store.lookup_by_session_key(route)
    if entry is None:
        return _refusal("no_committed_route")
    if logical(str(entry.session_id)) != owner_sid:
        return _refusal("route_identity_mismatch", "the route now points at another session")

    source = live.source
    source_profile = str(getattr(source, "profile", "") or "")
    if source_profile and source_profile != expected_profile:
        return _refusal("wrong_profile", "the live route belongs to another profile")
    adapter = None
    for attr in ("_delivery_adapter_for", "_adapter_for_source"):
        resolve = getattr(authority.runner, attr, None)
        if callable(resolve):
            adapter = resolve(source)
            if adapter is not None:
                break
    if adapter is None:
        return _refusal("no_route", "no adapter serves the root route")

    from gateway.session_automation import admit_route_wake
    from hermes_state_runtime import RuntimeStoreError
    token = f"{tree_id}:{message_id}"
    try:
        receipt = await admit_route_wake(
            authority.runner, adapter, source, text, WAKE_PRODUCER,
            token=token, return_receipt=True,
        )
    except RuntimeStoreError as exc:
        return _refusal("route_refused", str(getattr(exc, "reason", None) or exc))
    except Exception as exc:  # health: allow BLE001 -- an admission that may have committed is unknown
        return _refusal("wake_outcome_unknown", str(exc), unknown=True)
    if not receipt:
        return _refusal("no_committed_route")
    return {"ok": True, "admission_id": _receipt_id(receipt), "session_id": owner_sid,
            "route": route, "profile": expected_profile}


def request_root_wake(*, root_session_id: str, profile_key: Any, text: str,
                      message_id: str, tree_id: str) -> dict[str, Any]:
    """Blocking producer entry used by the tool layer. Never runs a turn itself."""
    home = profile_home(profile_key)
    if home is None:
        return _refusal("target_idle")
    from gateway.session_cron import owner_for_home
    bound = owner_for_home(home)
    if bound is None:
        return _refusal("target_idle")
    _authority, loop = bound
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        # The tool layer is synchronous; waiting on its own loop would deadlock. Real tool handlers
        # run in the gateway's executor, so this only guards a misuse.
        return _refusal("wake_on_loop", "request_root_wake must run off the gateway loop")
    if loop.is_closed():
        return _refusal("target_idle", "the gateway loop is closed")
    try:
        future = asyncio.run_coroutine_threadsafe(
            _admit(home, root_session_id, text, message_id, tree_id), loop)
        return future.result(timeout=WAKE_TIMEOUT_SECONDS)
    except TimeoutError:
        # The admission may still commit after the timeout: never report a definite refusal that
        # allows the caller to mark the message rejected and then see it run.
        return _refusal("wake_timeout", "admission did not answer in time", unknown=True)
    except Exception as exc:  # health: allow BLE001 -- unknown outcome, never a silent retry
        logger.warning("idle-root wake failed for %s", root_session_id, exc_info=True)
        return _refusal("wake_outcome_unknown", str(exc), unknown=True)


async def _cancel(profile_home_path: Path, root_session_id: str,
                  admission_id: Optional[str]) -> int:
    from gateway.session_cron import owner_for_home
    bound = owner_for_home(profile_home_path)
    if bound is None:
        return 0
    authority, _loop = bound
    from gateway.session_contract import SessionActor, SessionRef
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions
    ref = SessionRef(authority.profile_id, str(root_session_id))
    actor = SessionActor("agent-collab-stop", authority.profile_id,
                         frozenset({"session:submit", "session:read", "session:control", "session:operator"}),
                         "collab")
    rows = await asyncio.to_thread(list_session_admissions, authority.db,
                                   session_id=str(root_session_id), pending_only=False)
    cancelled = 0
    for row in rows:
        if admission_id and row.get("admission_id") != admission_id:
            continue
        if row.get("status") != "queued":
            continue
        if not str(row.get("request_id") or "").startswith('["agent-collab"'):
            continue
        try:
            await authority.cancel_queued(actor, ref, row["admission_id"])
            cancelled += 1
        except RuntimeStoreError:
            continue  # claimed meanwhile; the native interrupt path owns it
    return cancelled


def cancel_root_wakes(profile_key: Any, root_session_id: str,
                      admission_id: Optional[str] = None) -> int:
    """Best-effort cancellation of queued native wakes for a stopped tree."""
    home = profile_home(profile_key)
    if home is None:
        return 0
    from gateway.session_cron import owner_for_home
    bound = owner_for_home(home)
    if bound is None:
        return 0
    _authority, loop = bound
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop or loop.is_closed():
        return 0
    try:
        return asyncio.run_coroutine_threadsafe(
            _cancel(home, str(root_session_id), admission_id), loop).result(timeout=10.0)
    except Exception:  # health: allow BLE001 -- cancellation is compensating; the durable stop stands
        logger.warning("cancelling collaboration wakes failed for %s", root_session_id, exc_info=True)
        return 0


def wake_header(message_id: str, sender_id: str, sender_name: Optional[str], content: str) -> str:
    return (f"[agent message {message_id}]\n"
            f"From agent {sender_name or sender_id} ({sender_id}):\n{content}")
