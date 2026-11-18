"""Goal receipts edit the actual progress bubble, not a separate assistant explanation."""
import asyncio
import json
from contextlib import suppress
from queue import Queue
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run_turn_progress import GatewayTurnProgressMixin
from gateway.session import SessionSource


class CaptureAdapter:
    MAX_MESSAGE_LENGTH = 4000
    name = 'receipt-test'

    def __init__(self):
        self.sent, self.edits = [], []
        self.sent_event, self.edit_event = asyncio.Event(), asyncio.Event()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        message_id = str(len(self.sent) + 1)
        self.sent.append({'message_id': message_id, 'content': content, 'metadata': metadata})
        self.sent_event.set()
        return SendResult(success=True, message_id=message_id)

    async def edit_message(self, chat_id, message_id, content, metadata=None):
        self.edits.append({'message_id': message_id, 'content': content, 'metadata': metadata})
        self.edit_event.set()
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, *args, **kwargs):
        pass


class Driver(GatewayTurnProgressMixin):
    _approval_owner = None

    def __init__(self, adapter):
        self._runner = SimpleNamespace(_delivery_adapter_for=lambda source: adapter)
        self._ctx = SimpleNamespace(
            source=SessionSource(platform=Platform.TELEGRAM, chat_id='receipt-chat', thread_id='receipt-topic'),
            progress_queue=Queue(), tool_progress_enabled=True, progress_mode='all', progress_grouping='accumulate',
            _run_still_current=lambda: True, agent_holder=[SimpleNamespace(is_interrupted=False)],
            stream_consumer_holder=[None], _native_slack_task_cards=False, _voice_ack_guild=[None],
            _progress_reply_to='request', _progress_metadata={'thread_id': 'receipt-topic'},
            _cleanup_progress=False, _cleanup_msg_ids=[], log_queue=None, _live_status_adapter=None,
            last_tool=[None], last_progress_msg=[None], repeat_count=[0], last_was_terminal_block=[False],
            long_tool_hint_fired=[False])

    def _publish_execution(self, *args, **kwargs):
        pass

    def _publish_api_tool(self, *args, **kwargs):
        pass


def result_for(action='update', **kwargs):
    data = {'success': True, 'action': action,
            'previous': {'status': 'done', 'turns_used': 8, 'max_turns': 20},
            'changes': {'objective': {'before': 'Initial survey', 'after': 'Add controlled experiments'}},
            'goal': {'goal': 'Add controlled experiments', 'status': 'done', 'turns_used': 8, 'max_turns': 20}}
    data.update(kwargs)
    return json.dumps(data)


def test_start_is_id_correlated_and_never_claims_success():
    driver = Driver(CaptureAdapter())
    driver.combined_tool_start_callback('call-1', 'goal_manage', {'action': 'update'})
    assert not driver._ctx.progress_queue.empty(), 'Goal start must enter the native progress lane'
    event = driver._ctx.progress_queue.get_nowait()
    assert event['tool_call_id'] == 'call-1'
    assert '成功' not in event['text'] and '执行中' in event['text']


@pytest.mark.asyncio
async def test_completion_edits_same_topic_message_with_applied_diff():
    adapter = CaptureAdapter()
    driver = Driver(adapter)
    task = asyncio.create_task(driver.send_progress_messages())
    args = {'action': 'update', 'objective': 'Add controlled experiments'}
    try:
        driver.progress_callback('tool.started', 'goal_manage', None, args)
        driver.combined_tool_start_callback('call-1', 'goal_manage', args)
        await asyncio.wait_for(adapter.sent_event.wait(), timeout=5)
        driver.combined_tool_complete_callback('call-1', 'goal_manage', args, result_for())
        await asyncio.wait_for(adapter.edit_event.wait(), timeout=5)
        assert len(adapter.sent) == 1
        assert adapter.edits[-1]['message_id'] == adapter.sent[0]['message_id']
        text = adapter.edits[-1]['content']
        for value in ('Initial survey', 'Add controlled experiments', '成功', '8/20', 'done', '未修改'):
            assert value in text
        assert adapter.edits[-1]['metadata'] == {'thread_id': 'receipt-topic'}
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.parametrize('error', ['budget_exhausted', 'persistence_unverified', 'goal_changed'])
def test_failure_does_not_show_requested_changes_as_applied(error):
    driver = Driver(CaptureAdapter())
    driver.combined_tool_complete_callback('call-1', 'goal_manage',
        {'action': 'update', 'objective': 'Unapplied request'}, json.dumps({'success': False, 'error': error}))
    assert not driver._ctx.progress_queue.empty(), 'Goal failure must be visible'
    text = driver._ctx.progress_queue.get_nowait()['text']
    assert error in text and '失败' in text
    assert 'Unapplied request' not in text and '成功' not in text


def test_other_tools_and_progress_off_do_not_get_receipt_events():
    driver = Driver(CaptureAdapter())
    driver.combined_tool_start_callback('other', 'web_search', {'query': 'unchanged'})
    driver.combined_tool_complete_callback('other', 'web_search', {}, '{"success":true}')
    assert driver._ctx.progress_queue.empty()
    driver._ctx.tool_progress_enabled = False
    driver.combined_tool_start_callback('goal', 'goal_manage', {'action': 'status'})
    driver.combined_tool_complete_callback('goal', 'goal_manage', {'action': 'status'}, result_for())
    assert driver._ctx.progress_queue.empty()


@pytest.mark.asyncio
async def test_real_turn_runner_wires_goal_callbacks_and_edits_the_bubble(monkeypatch, tmp_path):
    from tests.gateway.test_run_progress_topics import _run_with_agent, _ACTIVE_ADAPTER
    from tests.waiting import wait_until

    class ReceiptAgent:
        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            args = {'action': 'update', 'objective': 'Add controlled experiments'}
            assert callable(self.tool_start_callback) and callable(self.tool_complete_callback)
            self.tool_progress_callback('tool.started', 'goal_manage', None, args)
            self.tool_start_callback('native-call', 'goal_manage', args)
            wait_until(lambda: _ACTIVE_ADAPTER['adapter'].sent, what='Goal start bubble sent')
            self.tool_complete_callback('native-call', 'goal_manage', args, result_for())
            return {'final_response': 'done', 'messages': [], 'api_calls': 1}

    adapter, result = await _run_with_agent(monkeypatch, tmp_path, ReceiptAgent,
        session_id='goal-receipt-wiring', config_data={'display': {'tool_progress': 'all'}})
    assert result['final_response'] == 'done'
    assert len(adapter.sent) == 1 and adapter.edits
    assert adapter.edits[-1]['message_id'] == 'progress-1'
    assert 'Initial survey' in adapter.edits[-1]['content']
    assert 'Add controlled experiments' in adapter.edits[-1]['content']
    assert adapter.sent[0]['metadata']['thread_id'] == '17585'
