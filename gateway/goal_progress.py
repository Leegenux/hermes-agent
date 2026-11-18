"""Goal-only, ID-correlated receipts on the gateway's existing topic-aware progress lane."""
from __future__ import annotations

import json
import re
import time

from agent.redact import redact_sensitive_text


EVENT = 'goal.receipt'
_FIELDS = {'objective': '目标', 'verification': '验收条件', 'stop_when': '停止条件'}
_PATH = re.compile(r'(?:[A-Za-z]:[/\\]Users[/\\]|/Users/)[^\s,;，；。]+')


def goal_arguments(name, args):
    if not isinstance(args, dict):
        return None
    if name == 'goal_manage':
        return args
    calls = args.get('calls')
    if name == 'tool_call' and isinstance(calls, list) and len(calls) == 1:
        call = calls[0]
        if isinstance(call, dict) and call.get('name') == 'goal_manage' and isinstance(call.get('arguments'), dict):
            return call['arguments']
    return None


def _text(value, limit=180):
    if not isinstance(value, (str, int, float, bool)):
        return '未提供'
    text = ' '.join(str(value).split()).replace('```', "'''")
    text = _PATH.sub('[本机路径]', redact_sensitive_text(text, force=True))
    return text if len(text) <= limit else text[:limit - 1] + '…'


def _budget(snapshot):
    used, maximum = snapshot.get('turns_used'), snapshot.get('max_turns')
    if type(used) is int and type(maximum) is int and used >= 0 and maximum > 0:
        return f'{used}/{maximum}'
    return '未返回'


def receipt_text(args, result=None, *, completed=False):
    action = _text(args.get('action', '未知'), 20)
    if not completed:
        return f'🎯 goal_manage · {action} · 执行中\n等待工具执行及持久化结果'
    try:
        data = json.loads(result) if isinstance(result, str) else result
    except (TypeError, ValueError):
        data = None
    if not isinstance(data, dict):
        return f'🎯 goal_manage · {action} · 结果未确认\n结果无法解析，请查询状态'
    if data.get('success') is not True:
        error = _text(data.get('error', 'unknown_error'), 80)
        message = _text(data.get('message', '请求未确认应用；请查询当前状态'), 240)
        return f'🎯 goal_manage · {action} · 失败\n错误：{error}\n原因：{message}'
    lines = [f'🎯 goal_manage · {action} · 成功']
    goal = data.get('goal')
    if not isinstance(goal, dict):
        return '\n'.join(lines + ['当前没有目标' if goal is None else '状态未返回'])
    changes = data.get('changes')
    if action == 'update':
        for key, label in _FIELDS.items():
            delta = changes.get(key) if isinstance(changes, dict) else None
            if isinstance(delta, dict) and 'before' in delta and 'after' in delta:
                lines += [f'{label}已修改：', '原：' + _text(delta['before']), '新：' + _text(delta['after'])]
            else:
                lines.append(f'{label}：' + ('未修改' if isinstance(changes, dict) else '工具未返回变更明细'))
    else:
        lines.append('目标：' + _text(goal.get('goal')))
        if action == 'create':
            contract = goal.get('contract') or {}
            if isinstance(contract, dict):
                lines += [label + '：' + _text(contract.get(key)) for key, label in _FIELDS.items() if key != 'objective']
    previous = data.get('previous')
    previous = previous if isinstance(previous, dict) else {}
    status = _text(goal.get('status'), 24)
    old_status = _text(previous.get('status'), 24)
    lines.append('状态：' + (f'{old_status} → {status}' if previous and old_status != status else status)
                 + ('（未恢复执行）' if action == 'update' and status in {'done', 'paused'} else ''))
    budget = _budget(goal)
    old_budget = _budget(previous)
    if previous:
        budget = f'{old_budget} → {budget}' + ('（未重置）' if old_budget == budget and budget != '未返回' else '（发生变化或未确认）')
    lines.append('轮数：' + budget)
    if action == 'status':
        lines.append('本次修改：无，只读查询')
    return '\n'.join(lines)


def queue_goal_receipt(driver, call_id, name, args, result=None, *, completed=False):
    ctx = driver._ctx
    from gateway.config import Platform
    values = goal_arguments(name, args)
    if (values is None or not call_id or not ctx.tool_progress_enabled
            or ctx._native_slack_task_cards or ctx.source.platform != Platform.TELEGRAM
            or not driver._native_card_gate()):
        return
    ctx.progress_queue.put({'type': EVENT, 'tool_call_id': str(call_id),
                            'text': receipt_text(values, result, completed=completed)})


class GoalProgressLane:
    """One editable bubble per real call ID; unrelated accumulated tool rows are untouched."""

    def __init__(self, driver, adapter):
        self.driver, self.adapter = driver, adapter
        self.calls = {}
        self.next_flush = 0.0

    @staticmethod
    def accepts(event):
        return isinstance(event, dict) and event.get('type') == EVENT

    async def apply(self, event):
        call_id = event.get('tool_call_id')
        if not isinstance(call_id, str) or not call_id or not isinstance(event.get('text'), str):
            return
        record = self.calls.get(call_id)
        if record is None:
            state = self.driver._progress_edit_state(self.adapter)
            state.can_edit = True
            record = self.calls[call_id] = {'state': state, 'text': None, 'pending': False}
        state = record['state']
        text = event['text']
        # Keep each receipt within one platform message without exposing raw result JSON.
        while state._progress_len_fn(text) > state._PROGRESS_TEXT_LIMIT and text:
            text = text[:-1]
        if text != record['text']:
            record.update(text=text, pending=True)
            state.progress_lines = [text]
        await self.flush()

    async def flush(self, *, force=False):
        if (not self.driver._ctx._run_still_current() or self.driver._agent_interrupted()
                or (not force and time.monotonic() < self.next_flush)):
            return
        for record in self.calls.values():
            if record['pending']:
                self.next_flush = time.monotonic() + 1.5
                state = record['state']
                ok = await self.driver._progress_send_or_edit(state, record['text'])
                record['pending'] = not (ok and state.progress_msg_id)
                if not force:
                    break
