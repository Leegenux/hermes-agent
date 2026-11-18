"""Goal receipt isolation, cancellation, redaction and transport failures."""
import asyncio
import json
from contextlib import suppress

import pytest

from gateway.goal_progress import GoalProgressLane, receipt_text
from gateway.platforms.base import SendResult
from tests.gateway.test_goal_progress_receipts import CaptureAdapter, Driver, result_for


def event(driver):
    return driver._ctx.progress_queue.get_nowait()


@pytest.mark.asyncio
async def test_out_of_order_completions_edit_their_own_message_ids():
    adapter = CaptureAdapter()
    driver = Driver(adapter)
    lane = GoalProgressLane(driver, adapter)
    for call in ('one', 'two'):
        driver.combined_tool_start_callback(call, 'goal_manage', {'action': 'update'})
        await lane.apply(event(driver))
        await lane.flush(force=True)
    for call in ('two', 'one'):
        driver.combined_tool_complete_callback(call, 'goal_manage', {'action': 'update'}, result_for())
        await lane.apply(event(driver))
        await lane.flush(force=True)
    assert len(adapter.sent) == 2
    assert [edit['message_id'] for edit in adapter.edits] == ['2', '1']


@pytest.mark.asyncio
async def test_cancel_drains_fast_completion_even_during_empty_queue_sleep():
    adapter = CaptureAdapter()
    driver = Driver(adapter)
    task = asyncio.create_task(driver.send_progress_messages())
    driver.combined_tool_start_callback('one', 'goal_manage', {'action': 'resume'})
    await asyncio.wait_for(adapter.sent_event.wait(), timeout=5)
    driver.combined_tool_complete_callback('one', 'goal_manage', {'action': 'resume'}, result_for('resume',
        changes={}, goal={'goal': 'Same goal', 'status': 'active', 'turns_used': 8, 'max_turns': 20}))
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    assert len(adapter.sent) == 1
    assert adapter.edits and adapter.edits[-1]['message_id'] == '1'
    assert 'done → active' in adapter.edits[-1]['content']
    assert '未重置' in adapter.edits[-1]['content']


def test_receipt_text_redacts_credentials_paths_and_multiline_fence_injection():
    fake_secret = 'sk-ant-' + 'testing' * 15
    data = json.loads(result_for())
    data['changes']['objective']['after'] = 'C:/Users/private-user/project/report.md\n```\n' + fake_secret
    text = receipt_text({'action': 'update'}, data, completed=True)
    assert fake_secret not in text and 'private-user' not in text and '```' not in text
    assert '[本机路径]' in text


def test_old_plugin_result_does_not_claim_unchanged_fields_without_diff():
    data = json.loads(result_for())
    del data['changes']
    text = receipt_text({'action': 'update'}, data, completed=True)
    assert '工具未返回变更明细' in text and '未修改' not in text


@pytest.mark.parametrize('result', ['not JSON', [], None])
def test_unparseable_result_does_not_claim_goal_succeeded(result):
    text = receipt_text({'action': 'update'}, result, completed=True)
    assert '结果未确认' in text and '成功' not in text


def test_single_deferred_goal_call_is_recognized_without_widening_other_tools():
    driver = Driver(CaptureAdapter())
    driver.combined_tool_start_callback('bridge', 'tool_call', {
        'calls': [{'name': 'goal_manage', 'arguments': {'action': 'status'}}]})
    assert event(driver)['tool_call_id'] == 'bridge'


@pytest.mark.asyncio
async def test_transient_edit_failure_retries_without_a_duplicate_message():
    class Transient(CaptureAdapter):
        async def edit_message(self, *args, **kwargs):
            result = await super().edit_message(*args, **kwargs)
            return SendResult(success=False, retryable=True, error='test-only transient') if len(self.edits) == 1 else result
    adapter = Transient()
    driver = Driver(adapter)
    lane = GoalProgressLane(driver, adapter)
    driver.combined_tool_start_callback('one', 'goal_manage', {'action': 'status'})
    await lane.apply(event(driver))
    driver.combined_tool_complete_callback('one', 'goal_manage', {'action': 'status'}, result_for('status', changes={}))
    await lane.apply(event(driver))
    await lane.flush(force=True)
    await lane.flush(force=True)
    assert len(adapter.sent) == 1 and len(adapter.edits) == 2
    assert all(edit['message_id'] == '1' for edit in adapter.edits)


@pytest.mark.asyncio
async def test_separate_grouping_still_edits_each_goal_bubble_and_bounds_text():
    class Small(CaptureAdapter):
        MAX_MESSAGE_LENGTH = 240
    adapter = Small()
    driver = Driver(adapter)
    driver._ctx.progress_grouping = 'separate'
    lane = GoalProgressLane(driver, adapter)
    driver.combined_tool_start_callback('one', 'goal_manage', {'action': 'update'})
    await lane.apply(event(driver))
    data = json.loads(result_for())
    data['changes']['objective'] = {'before': 'old' * 1000, 'after': 'new' * 1000}
    driver.combined_tool_complete_callback('one', 'goal_manage', {'action': 'update'}, data)
    await lane.apply(event(driver))
    await lane.flush(force=True)
    assert len(adapter.sent) == 1 and adapter.edits[-1]['message_id'] == '1'
    assert len(adapter.edits[-1]['content']) <= 240 - 64
