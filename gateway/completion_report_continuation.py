"""One bounded report continuation on the ORIGINAL settlement's transaction.

Authorization is supplied by the existing owner authority. This code never opens
another transaction or schedules work: its row is consumed by the native FIFO.
"""
from copy import deepcopy
import json

from hermes_state_runtime import _admit_session_input_on_conn, RuntimeStoreError
from hermes_state_report_obligations import _REPORT, _check_revision, _save


def queue_unreported_on_conn(conn, *, epoch, row, authorize):
    if authorize is None or row['payload'].get('completion_report_continuation_v1') is not None:
        return []
    if not row['principal_id'].startswith('automation:'):
        return []
    results = []
    saved = conn.execute('SELECT key,value FROM state_meta WHERE key LIKE ?', (_REPORT + '%',)).fetchall()
    for key, text in saved:
        report = json.loads(text)
        if (report['admission_id'] != row['admission_id'] or report['session_id'] != row['target_session_id']
                or report['state'] != 'open' or not report.get('redirected')
                or report.get('explicitly_unreported') is not True
                or report.get('receipt_admission_id') != row['admission_id']):
            continue
        _check_revision(conn, report, report['revision'])
        if report['route'] != row['principal_id'][len('automation:'):]:
            raise RuntimeStoreError('permission_denied')
        payload = deepcopy(row['payload'])
        envelope = payload.get('native_text_v1')
        descriptor = envelope.get('automation') if envelope is not None else payload.get('local_automation_v1')
        if not isinstance(descriptor, dict) or descriptor.get('owner') != report['session_id']:
            raise RuntimeStoreError('permission_denied')
        descriptor['identity'] = report['continuation_request_id']
        descriptor['identities'] = [report['identity']]
        if envelope is not None:
            envelope['event']['message_id'] = report['continuation_request_id']
        payload['text'] = ('[COMPLETION REPORT CONTINUATION]\n'
            'Complete the remaining report for the saved child result below. Do not rerun '
            'the child or its business task. This is the single bounded continuation.\n\n'
            + row['payload']['text'])
        payload['completion_report_continuation_v1'] = {
            'report_id': report['report_id'], 'revision': report['revision'],
            'original_admission_id': row['admission_id']}
        queued = _admit_session_input_on_conn(conn, epoch=epoch, principal_id=row['principal_id'],
            session_id=row['target_session_id'], request_id=report['continuation_request_id'],
            payload=payload, _authorize_write=authorize)
        if queued['status'] != 'queued':
            raise RuntimeStoreError('admission_conflict')
        report['state'] = 'queued'
        report['continuation_admission_id'] = queued['admission_id']
        _save(conn, key, report)
        results.append(queued['admission_id'])
    return results
