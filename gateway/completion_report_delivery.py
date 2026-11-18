"""Bind prepared report text to the existing final-response ledger.

Confirmation means the existing transport returned success for this exact final
response. It does not prove semantic report quality, user reading, or exactly-once
network delivery after an ambiguous acknowledgement.
"""
import hashlib
import json

from hermes_state_runtime import RuntimeStoreError, _epoch
from hermes_state_report_obligations import _REPORT, _load, _save, _check_revision


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def bind_report_delivery(db, *, epoch, admission_id, report_ids, obligation_id, response, _conn=None):
    if not obligation_id or not isinstance(response, str) or not report_ids or len(set(report_ids)) != len(report_ids):
        raise RuntimeStoreError('invalid_params')
    def write(conn):
        _epoch(conn, epoch)
        # Ledger connections use tuple rows; SessionDB uses mapping rows.
        # Read only the required admission columns without changing either
        # connection's row_factory or opening a second transaction.
        row = conn.execute('SELECT status,outcome,target_session_id,principal_id '
                           'FROM session_admissions WHERE admission_id=?', (admission_id,)).fetchone()
        if row is None or (row[0], row[1]) != ('terminal', 'completed'):
            raise RuntimeStoreError('stale_generation')
        delivered = conn.execute('SELECT session_key,content FROM delivery_obligations WHERE obligation_id=?',
                                 (obligation_id,)).fetchone()
        if delivered is None or delivered[1] != response:
            raise RuntimeStoreError('admission_conflict')
        for report_id in report_ids:
            report = _load(conn, _REPORT + report_id)
            if (report is None or report['state'] != 'preparing'
                    or report.get('receipt_admission_id') != admission_id
                    or report['session_id'] != row[2]
                    or report['route'] != delivered[0]
                    or report.get('prepared_response_sha256') != _sha(response)
                    or row[3] != 'automation:' + report['route']):
                raise RuntimeStoreError('permission_denied')
            _check_revision(conn, report, report['revision'])
            report['delivery_obligation_id'] = obligation_id
            _save(conn, _REPORT + report_id, report)
    # The ledger invokes this inside its INSERT transaction. Never nest another
    # db._execute_write: an unbound recoverable row must not survive a crash.
    return write(_conn) if _conn is not None else db._execute_write(write)


def confirm_reports_for_delivery_on_conn(conn, obligation_id):
    """Inside the ledger's successful ACK transaction, including native retry recovery."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_meta'").fetchone():
        return
    delivered = conn.execute('SELECT state,content,session_key,updated_at FROM delivery_obligations '
                             'WHERE obligation_id=?', (obligation_id,)).fetchone()
    if delivered is None or delivered[0] != 'delivered':
        return
    rows = conn.execute('SELECT key,value FROM state_meta WHERE key LIKE ?', (_REPORT + '%',)).fetchall()
    for key, value in rows:
        report = json.loads(value)
        if (report['state'] != 'preparing' or report.get('delivery_obligation_id') != obligation_id
                or report.get('prepared_response_sha256') != _sha(delivered[1])
                or report['route'] != delivered[2]):
            continue
        try:
            _check_revision(conn, report, report['revision'])
        except RuntimeStoreError as exc:
            if exc.reason != 'stale_generation':
                raise
            report['state'] = 'unknown'
        else:
            report['state'] = 'confirmed'
            report['confirmed_at'] = delivered[3]
        _save(conn, key, report)


def report_dispatch_allowed_on_conn(conn, obligation_id):
    """Recheck persistent stop/new at the native recovery dispatch boundary."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='state_meta'").fetchone():
        return True
    rows = conn.execute('SELECT value FROM state_meta WHERE key LIKE ?', (_REPORT + '%',)).fetchall()
    for (value,) in rows:
        report = json.loads(value)
        if report.get('delivery_obligation_id') != obligation_id:
            continue
        if report['state'] != 'preparing':
            return False
        try:
            _check_revision(conn, report, report['revision'])
        except RuntimeStoreError as exc:
            if exc.reason != 'stale_generation':
                raise
            return False
    return True
