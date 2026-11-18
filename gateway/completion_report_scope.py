"""Trusted completion-turn scope. Prepared content is not a delivery receipt."""
import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import hashlib
import json

from gateway.completion_report_protocol import format_instruction, parse_receipt, ReceiptError
from hermes_state_runtime import RuntimeStoreError

completion_scope = ContextVar('completion_report_scope', default=None)


def _descriptor(row):
    if not row.get('principal_id', '').startswith('automation:'):
        return None
    payload = row.get('payload', {})
    envelope = payload.get('native_text_v1', {})
    descriptor = envelope.get('automation') or payload.get('local_automation_v1')
    if not isinstance(descriptor, dict):
        return None
    identities = descriptor.get('identities') or [descriptor.get('identity')]
    if not isinstance(identities, list) or not identities or len(identities) > 32:
        return None
    try:
        parsed = [json.loads(value) for value in identities]
    except (ValueError, TypeError):
        return None
    if any(not isinstance(value, list) or len(value) != 3 or value[0] != 'async_delegation' for value in parsed):
        return None
    return descriptor, sorted(set(identities)), envelope.get('route') or descriptor.get('route')


@asynccontextmanager
async def report_turn(authority, row, event):
    found = _descriptor(row)
    if found is None:
        yield
        return
    _, identities, route = found
    from hermes_state_report_obligations import create_report_on_conn, control_revision_on_conn
    profile_id = str(authority.profile_id)
    continuation = row['payload'].get('completion_report_continuation_v1')
    def create(conn):
        if continuation is not None:
            from hermes_state_runtime import _epoch, _admission, _session
            from hermes_state_report_obligations import _load, _REPORT, _check_revision
            _epoch(conn, authority.epoch)
            current = _admission(conn, row['admission_id'])
            session = _session(conn, row['target_session_id'])
            report = _load(conn, _REPORT + continuation['report_id'])
            if (report is None or report['state'] != 'queued'
                    or report['profile_id'] != profile_id or report['session_id'] != row['target_session_id']
                    or report['route'] != route or row['principal_id'] != 'automation:' + route
                    or report['admission_id'] != continuation['original_admission_id']
                    or report['continuation_admission_id'] != row['admission_id']
                    or report['continuation_request_id'] != row['request_id']
                    or identities != [report['identity']] or current['status'] != 'started'
                    or current['owner_epoch'] != authority.epoch or current['generation'] != row['generation']
                    or session['runtime_generation'] != row['generation']):
                raise RuntimeStoreError('permission_denied')
            _check_revision(conn, report, continuation['revision'])
            return [report]
        revision = control_revision_on_conn(conn, profile_id=profile_id, session_id=row['target_session_id'])
        return [create_report_on_conn(conn, epoch=authority.epoch, profile_id=profile_id,
            admission_id=row['admission_id'], identity=identity, generation=row['generation'], revision=revision)
            for identity in identities]
    reports = await asyncio.to_thread(authority.db._execute_write, create)
    scope = {'identities': identities, 'reports': reports, 'route': route,
             'admission_id': row['admission_id'], 'redirected': False, 'continuation': continuation is not None}
    scopes = getattr(authority.runner, '_completion_report_scopes', None)
    if scopes is None:
        scopes = authority.runner._completion_report_scopes = {}
    if route in scopes:
        raise RuntimeStoreError('admission_conflict')
    scopes[route] = scope
    event.text += format_instruction(identities)
    token = completion_scope.set(scope)
    try:
        yield
    finally:
        completion_scope.reset(token)
        if scopes.get(route) is scope:
            scopes.pop(route)


def redirect_instruction(runner, session_key, running_agent, text):
    scope = getattr(runner, '_completion_report_scopes', {}).get(session_key)
    if scope is None:
        return text, None
    state = runner._peek_session_state(session_key)
    if state is None or state.turn.agent is not running_agent:
        return text, None
    return text + format_instruction(scope['identities']), scope


async def prepare_response(authority, row, response, captured):
    scope = completion_scope.get()
    if scope is None:
        return response
    result = captured.get('terminal_result', captured.get('result', {}))
    if result.get('interrupted') or result.get('failed') or result.get('error') or response is None:
        from hermes_state_report_obligations import _load, _save, _REPORT, _check_revision
        def fail_closed(conn):
            from hermes_state_runtime import _epoch
            _epoch(conn, authority.epoch)
            for original in scope['reports']:
                report = _load(conn, _REPORT + original['report_id'])
                if report is None or report['state'] in ('cancelled', 'confirmed'):
                    continue
                try:
                    _check_revision(conn, report, original['revision'])
                except RuntimeStoreError as exc:
                    if exc.reason == 'stale_generation':
                        continue  # a real stop/new already owns the disposition
                    raise
                report['state'] = 'unknown'
                _save(conn, _REPORT + original['report_id'], report)
        await asyncio.to_thread(authority.db._execute_write, fail_closed)
        safe = None if result.get('interrupted') or response is None else '⚠️ 完成汇报回合失败，结果已保留，尚未确认送达。'
        result['final_response'] = safe or ''
        return safe
    from gateway.session_results import _redacted
    from hermes_state_report_obligations import _load, _save, _REPORT, _check_revision
    raw = result.get('final_response') or response or ''
    try:
        receipt = _redacted(parse_receipt(raw, scope['identities']))
    except ReceiptError:
        receipt = None
    def stage(conn):
        from hermes_state_runtime import _epoch
        _epoch(conn, authority.epoch)
        for original in scope['reports']:
            report = _load(conn, _REPORT + original['report_id'])
            if report is None:
                raise RuntimeStoreError('not_found')
            _check_revision(conn, report, original['revision'])
            expected_state = 'queued' if scope.get('continuation') else 'open'
            if report['state'] != expected_state:
                raise RuntimeStoreError('admission_conflict')
            if receipt is None:
                report['state'] = 'unknown'
            else:
                report['receipt_admission_id'] = row['admission_id']
                report['redirected'] = scope['redirected']
                part = next((part for part in receipt['reports'] if part['identity'] == report['identity']), None)
                if part is not None:
                    report['state'] = 'preparing'
                    report['prepared_response_sha256'] = hashlib.sha256(receipt['text'].encode()).hexdigest()
                    report['prepared_output'] = {'text': part['text'],
                        'sha256': hashlib.sha256(part['text'].encode()).hexdigest()}
                else:
                    report['explicitly_unreported'] = True
                    if scope.get('continuation'):
                        report['state'] = 'unknown'  # the bounded continuation never schedules itself
            _save(conn, _REPORT + original['report_id'], report)
    try:
        await asyncio.to_thread(authority.db._execute_write, stage)
    except RuntimeStoreError as exc:
        if exc.reason == 'stale_generation':
            return ''  # a real /stop or /new superseded this turn: never resurrect a reply
        raise
    text = receipt['text'] if receipt is not None else '⚠️ 完成结果已保留，但汇报格式未通过校验；尚不能确认已向你汇报。'
    if 'result' not in captured:
        captured['result'] = {'messages': []}
    captured['result']['final_response'] = text
    captured['result']['completion_report_receipt'] = receipt
    captured['result']['completion_report_redirected'] = scope['redirected']
    return text
