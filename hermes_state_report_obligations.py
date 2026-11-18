"""Private completion-report state on the owning SessionDB transaction.

Callers authorize the owning profile; identities are revalidated against persisted
producer admissions. This module neither schedules work nor performs transport I/O.
"""
import hashlib
import json

from hermes_state_runtime import RuntimeStoreError, _admission, _epoch, _session, _text

_REPORT = 'completion-report:v1:'
_FENCE = 'completion-report-fence:v1:'
_CONTROL = 'completion-report-control:v1:'
_EDGES = {
    'open': {'queued', 'unknown', 'cancelled'},
    'queued': {'preparing', 'unknown', 'cancelled'},
    'preparing': {'confirmed', 'unknown', 'cancelled'},
    'unknown': {'cancelled'},
    'confirmed': set(),
    'cancelled': set(),
}


def _digest(parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def _load(conn, key, default=None):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    return default if row is None else json.loads(row[0])


def _save(conn, key, value):
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                 'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                 (key, json.dumps(value, ensure_ascii=False, sort_keys=True)))


def control_revision_on_conn(conn, *, profile_id, session_id):
    for value in (profile_id, session_id):
        _text(value)
    return _load(conn, _FENCE + _digest([profile_id, session_id]), 0)


def _check_revision(conn, report, revision):
    if type(revision) is not int or revision != report['revision'] or revision != control_revision_on_conn(
            conn, profile_id=report['profile_id'], session_id=report['session_id']):
        raise RuntimeStoreError('stale_generation')


def _completion_owner(conn, *, epoch, admission_id, identity, generation, profile_id):
    _epoch(conn, epoch)
    row = _admission(conn, admission_id)
    session = _session(conn, row['target_session_id'])
    if (row['status'] != 'started' or row['owner_epoch'] != epoch
            or type(generation) is not int or row['generation'] != generation
            or session['runtime_generation'] != generation):
        raise RuntimeStoreError('stale_generation')
    payload = json.loads(row['payload_json'])
    native = payload.get('native_text_v1', {})
    descriptor = native.get('automation') or payload.get('local_automation_v1', {})
    route = native.get('route') or descriptor.get('route')
    identities = descriptor.get('identities') or [descriptor.get('identity')]
    if (not route or row['principal_id'] != 'automation:' + route
            or descriptor.get('owner') != row['target_session_id'] or identity not in identities):
        raise RuntimeStoreError('permission_denied')
    if native and (native.get('source', {}).get('profile') or 'default') != profile_id:
        # The native envelope carries a display alias; authority ownership may be
        # the canonical HOME path. Only this connection's exact database parent
        # can stand in for that alias, never an arbitrary other profile path.
        import os
        database = next((entry[2] for entry in conn.execute('PRAGMA database_list') if entry[1] == 'main'), '')
        if (not os.path.isabs(profile_id) or not database
                or os.path.normcase(os.path.normpath(profile_id)) != os.path.normcase(os.path.dirname(database))):
            raise RuntimeStoreError('permission_denied')
    try:
        parsed = json.loads(identity)
    except (TypeError, ValueError) as exc:
        raise RuntimeStoreError('permission_denied') from exc
    if not isinstance(parsed, list) or len(parsed) != 3 or parsed[0] != 'async_delegation' or not parsed[1]:
        raise RuntimeStoreError('permission_denied')
    return row, route


def create_report_on_conn(conn, *, epoch, profile_id, admission_id, identity, generation, revision):
    for value in (profile_id, admission_id, identity):
        _text(value)
    row, route = _completion_owner(conn, epoch=epoch, admission_id=admission_id,
                                  identity=identity, generation=generation, profile_id=profile_id)
    report_id = _digest([profile_id, row['target_session_id'], identity])
    report = dict(report_id=report_id, profile_id=profile_id, session_id=row['target_session_id'],
                  admission_id=admission_id, identity=identity, generation=generation, route=route,
                  revision=revision, state='open', continuation_admission_id=None,
                  continuation_request_id='completion-report:' + report_id)
    _check_revision(conn, report, revision)
    old = _load(conn, _REPORT + report_id)
    if old is not None:
        if any(old[key] != report[key] for key in ('profile_id', 'session_id', 'admission_id', 'identity', 'generation', 'route', 'revision')):
            raise RuntimeStoreError('admission_conflict')
        return old
    _save(conn, _REPORT + report_id, report)
    return report


def create_report(db, **kwargs):
    return db._execute_write(lambda conn: create_report_on_conn(conn, **kwargs))


def transition_report_on_conn(conn, *, epoch, report_id, revision, expected, state,
                              continuation_admission_id=None):
    _epoch(conn, epoch)
    report = _load(conn, _REPORT + _text(report_id))
    if report is None:
        raise RuntimeStoreError('not_found')
    _check_revision(conn, report, revision)
    if report['state'] != expected or state not in _EDGES.get(expected, set()):
        raise RuntimeStoreError('admission_conflict')
    if continuation_admission_id is not None:
        continuation = _admission(conn, continuation_admission_id)
        if (continuation['target_session_id'] != report['session_id']
                or continuation['request_id'] != report['continuation_request_id']
                or continuation['principal_id'] != 'automation:' + report['route']):
            raise RuntimeStoreError('permission_denied')
        report['continuation_admission_id'] = continuation_admission_id
    report['state'] = state
    _save(conn, _REPORT + report_id, report)
    return report


def transition_report(db, **kwargs):
    return db._execute_write(lambda conn: transition_report_on_conn(conn, **kwargs))


def cancel_reports_on_conn(conn, *, epoch, profile_id, session_id, control_id=None, _return_details=False):
    _epoch(conn, epoch)
    _session(conn, session_id)
    revision = control_revision_on_conn(conn, profile_id=profile_id, session_id=session_id)
    if control_id is not None:
        token = _CONTROL + _digest([profile_id, session_id, _text(control_id)])
        saved = _load(conn, token)
        if saved is not None:
            details = saved if isinstance(saved, dict) else {'revision': saved, 'cancelled_ids': []}
            return details if _return_details else revision
    revision += 1
    cancelled_ids = []
    _save(conn, _FENCE + _digest([profile_id, session_id]), revision)
    # Only report-owned queued admissions are cancelled; ordinary/unknown FIFO inputs are untouched.
    rows = conn.execute('SELECT key,value FROM state_meta WHERE key LIKE ?', (_REPORT + '%',)).fetchall()
    for key, value in rows:
        report = json.loads(value)
        if (report['profile_id'], report['session_id']) != (profile_id, session_id) or report['state'] in ('confirmed', 'cancelled'):
            continue
        continuation_id = report.get('continuation_admission_id')
        in_flight = report['state'] in ('preparing', 'unknown')
        if continuation_id:
            changed = conn.execute("UPDATE session_admissions SET status='terminal',outcome='cancelled' "
                         "WHERE admission_id=? AND target_session_id=? AND request_id=? "
                         "AND principal_id=? AND status='queued'",
                         (continuation_id, session_id, report['continuation_request_id'], 'automation:' + report['route']))
            if changed.rowcount == 1:
                cancelled_ids.append(continuation_id)
            else:
                linked = _admission(conn, continuation_id)
                in_flight = in_flight or (linked['status'], linked['outcome']) != ('terminal', 'cancelled')
        # A started/settled-but-unconfirmed execution may have sent output; the stop
        # fence prevents future confirmation but cannot establish non-delivery.
        report['state'] = 'unknown' if in_flight else 'cancelled'
        _save(conn, key, report)
    details = {'revision': revision, 'cancelled_ids': cancelled_ids}
    if control_id is not None:
        _save(conn, token, details)
    return details if _return_details else revision


def cancel_reports(db, **kwargs):
    return db._execute_write(lambda conn: cancel_reports_on_conn(conn, **kwargs))


def get_report(db, report_id):
    with db._read_ctx() as conn:
        return _load(conn, _REPORT + _text(report_id))
