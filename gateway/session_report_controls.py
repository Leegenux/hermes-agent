"""Completion-report stop effects; reuse authority mutation tracking and observers."""
import asyncio
import json
import uuid


def command_control_id(event):
    """Same authorized platform event through busy and slash dispatch has one identity."""
    saved = getattr(event, '_completion_report_control_id', None)
    if saved is None:
        saved = json.dumps(['messaging-control', event.source.platform.value,
                            str(event.source.chat_id), event.source.thread_id,
                            str(event.message_id) if event.message_id else uuid.uuid4().hex,
                            event.get_command()], separators=(',', ':'))
        event._completion_report_control_id = saved
    return saved


async def fence_reports(authority, session_id, control_id=None):
    """Commit the fence and queued cancellation before interrupting, then pay observers.

    The caller has authorized session control. No scheduling or transport side effect
    occurs within the database transaction; mutation tracking shields postcommit payment.
    """
    from gateway.session_runtime_workers import track_mutation
    from gateway.session_contract import SessionRef
    from hermes_state_report_obligations import cancel_reports_on_conn
    control_id = control_id or 'operator-control:' + uuid.uuid4().hex

    async def commit_and_settle():
        def write(conn):
            return cancel_reports_on_conn(conn, epoch=authority.epoch,
                profile_id=str(authority.profile_id), session_id=session_id,
                control_id=control_id, _return_details=True)
        details = await asyncio.to_thread(authority.db._execute_write, write)
        paid = getattr(authority, '_completion_report_cancelled', None)
        if paid is None:
            paid = authority._completion_report_cancelled = set()
        if session_id in authority.sessions:
            ref = SessionRef(authority.profile_id, session_id)
            for admission_id in details['cancelled_ids']:
                if admission_id not in paid:
                    authority.cancel_obligations.add(admission_id)
                    authority._settle_cancelled(ref, admission_id)
                    paid.add(admission_id)
        details['control_id'] = control_id
        return details
    return await asyncio.shield(track_mutation(authority, commit_and_settle()))
