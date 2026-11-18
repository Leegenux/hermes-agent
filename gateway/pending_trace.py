"""Bounded, content-free identifiers for gateway handoff diagnostics."""
import hashlib
import logging
import re

logger = logging.getLogger('gateway.pending_trace')
_SAFE_ID = re.compile(r'[0-9A-Za-z_-]{1,80}\Z')


def _identifier(value):
    if isinstance(value, (str, int)) and _SAFE_ID.fullmatch(str(value)):
        return str(value)
    return 'unknown'


def _fingerprint(value):
    if isinstance(value, str) and value:
        return hashlib.sha256(value.encode('utf-8', errors='replace')).hexdigest()[:16]
    return 'unknown'


def trace(reason, session_key=None, event=None, admission_id=None, generation=None, agent=None, producer=None, runner=None):
    try:
        if admission_id is None and event is not None:
            # The metrics tag carries the committed ID of this exact busy arrival;
            # do not infer an admission from the currently executing turn.
            from hermes_cli.observability.shared_metrics_busy import _Arrival
            arrival = getattr(event, '_busy_arrival', None)
            if isinstance(arrival, _Arrival):
                admission_id = arrival.admission_id
        if generation is None and runner is not None and session_key:
            try:
                generation = runner._current_session_run_generation(session_key)
            except Exception:
                generation = None
        if not logger.isEnabledFor(logging.INFO):
            return
        logger.info('handoff reason=%s session_key_sha256=%s message_id_sha256=%s admission_id=%s generation=%s agent=%s producer_sha256=%s',
                    reason, _fingerprint(session_key), _fingerprint(str(event.message_id)) if event is not None and isinstance(getattr(event, 'message_id', None), (str, int)) else 'unknown',
                    _identifier(admission_id), _identifier(generation),
                    _identifier(id(agent) if agent is not None else None), _fingerprint(producer))
    except Exception:
        # Diagnostic failure must never change a queue or interrupt result.
        return
