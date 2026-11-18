"""Native memory commits, real files/locks/journals; deterministic provider only.

Breaks caught: installing unreviewed/stale drafts, ignoring durable-journal errors,
clobbering concurrent writers, claiming rollback without exact byte readback.
"""
import base64
from contextlib import nullcontext
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import background_review as native
from agent.background_review_capacity import inspect_memory_capacity
from agent.background_review_capacity_drafts import _SCOPE
from hermes_constants import get_hermes_home
from tools import write_approval as wa
from tools.memory_tool import MemoryStore, get_memory_dir, memory_tool
from tools.skill_provenance import set_current_write_origin, reset_current_write_origin


def run_commit(monkeypatch, *, targets=('memory',), authority=True, damage=None, layout=False):
    store = MemoryStore(memory_char_limit=10000, user_char_limit=5000)
    store.load_from_disk()
    originals = {}
    for target in ('memory', 'user'):
        content = 'Keep blue and threshold 7.\n' + ('x' * (9000 if target == 'memory' else 4500)
                                                       if target in targets else 'untouched')
        if damage == 'quota-shift' and target == 'memory':
            header = 'Keep blue and threshold 7.\n'
            content = header + 'x' * (9000 - len(header))
        assert store.add(target, content)['success']
        path = store._path_for(target)
        if layout:
            path.write_bytes(b'\xef\xbb\xbf' + content.encode().replace(b'\n', b'\r\n'))
        originals[target] = path.read_bytes()
    frozen = dict(store._system_prompt_snapshot)
    plan = inspect_memory_capacity(SimpleNamespace(_memory_store=store,
        _memory_enabled=True, _user_profile_enabled=True), review_memory=True,
        paths={t: store._path_for(t) for t in ('memory', 'user')})
    assert {t.name for t in plan.targets} == set(targets)
    seen, delivered, stages, captured = [], [], [], {}
    race_pool = None
    race_future = None
    parent = SimpleNamespace(provider='synthetic', api_mode='chat_completions',
        memory_notifications='on', _safe_print=lambda _: None,
        background_review_callback=delivered.append,
        _emit_auxiliary_failure=lambda *a: captured.setdefault('worker_errors', []).append(a),
        _active_children=[], _background_review_agent=None)
    run = native.prepare_background_review_run(parent)
    native_atomic = importlib.import_module('utils').atomic_write_bytes
    native_read = Path.read_bytes
    source_path = store._path_for(targets[0])
    first_write = True
    bad_read = False

    def fault_write(path, data, **kwargs):
        nonlocal first_write, bad_read
        if Path(path) != source_path:
            return native_atomic(path, data, **kwargs)
        if first_write:
            first_write = False
            if damage == 'prewrite':
                raise OSError('Synthetic pre-replace failure')
            native_atomic(path, data, **kwargs)
            if damage == 'readback':
                bad_read = True
            if damage == 'foreign':
                Path(path).write_bytes(b'EXTERNAL WRITER REVISION')
            if damage in ('postwrite', 'foreign', 'rollback'):
                raise OSError('Synthetic post-replace failure')
        elif damage == 'rollback':
            raise OSError('Synthetic rollback unavailable')
        else:
            return native_atomic(path, data, **kwargs)

    def fault_read(path):
        nonlocal bad_read
        if path == source_path and bad_read:
            bad_read = False
            return b'SYNTHETIC READBACK MISMATCH'
        return native_read(path)

    class Provider:
        _memory_enabled = True
        _user_profile_enabled = True
        _memory_store = store
        _session_messages = []
        session_input_tokens = 13
        session_api_calls = 1
        _review_input_token_budget = 100000
        provider = 'synthetic'
        model = 'fixture'
        base_url = 'offline-only'

        def __init__(self, critic):
            self.critic = critic

        def release_clients(self):
            pass

        def run_conversation(self, *, user_message, conversation_history):
            seen.append(self.critic)
            if not self.critic:
                token = set_current_write_origin('background_review')
                try:
                    for target in targets:
                        raw = originals[target].decode('utf-8-sig').replace('\r\n', '\n')
                        reply = json.loads(memory_tool(target=target, operations=[
                            {'action': 'patch', 'old_text': raw,
                             'content': 'Keep blue and threshold 7.'}], store=store))
                        assert reply.get('staged'), reply
                        stages.append(reply)
                        if damage == 'duplicate':
                            duplicate = json.loads(memory_tool(target=target, operations=[
                                {'action': 'patch', 'old_text': raw,
                                 'content': 'Keep blue and threshold 7.'}], store=store))
                            assert duplicate.get('staged'), duplicate
                            stages.append(duplicate)
                finally:
                    reset_current_write_origin(token)
                return {'final_response': 'Generated staged drafts'}
            packet = json.loads(user_message.partition('CAPACITY_REVIEW_PACKET_JSON\n')[2])
            captured['scope'] = _SCOPE.get()
            verdict = {'packet_sha256': packet['packet_sha256'], 'decision': 'PASS',
                'checks': {k: True for k in ('rules', 'parameters', 'conditions', 'permissions',
                    'ownership', 'business_conclusions', 'topics', 'code_units')}, 'issues': []}
            return {'final_response': json.dumps(verdict)}

    def builder(*args, **kwargs):
        return Provider(bool(seen)), {}, False

    monkeypatch.setattr(native, 'build_cache_parity_fork', builder)
    # Delay faults until critic has produced a trusted record; preserve all native
    # generation, critic, tool, snapshot, gate, worker and callback code.
    from agent import background_review_capacity_critic as critic
    real_review = critic.review_capacity_drafts
    def reviewed(*args, **kwargs):
        nonlocal race_pool, race_future
        result = real_review(*args, **kwargs)
        scope = _SCOPE.get()
        captured['scope'] = scope
        if damage in ('prewrite', 'postwrite', 'foreign', 'rollback', 'readback'):
            monkeypatch.setattr('utils.atomic_write_bytes', fault_write)
        if damage == 'readback':
            monkeypatch.setattr(Path, 'read_bytes', fault_read)
        if damage == 'cancelled':
            run.cancel()
        elif damage == 'unreviewed':
            scope.review_results.clear()
        elif damage == 'quota-shift':
            # 60% floor remains 6000 but the 90% line moves above 9000.
            # Comparing only target_chars would wrongly retain write authority.
            store.memory_char_limit = 10001
        elif damage == 'scanner':
            monkeypatch.setattr('tools.memory_tool_store._scan_memory_content', lambda _: 'Synthetic scan block')
        elif damage == 'native-race':
            from concurrent.futures import ThreadPoolExecutor
            from contextlib import contextmanager
            from contextvars import copy_context
            import threading
            start, done = threading.Event(), threading.Event()
            original_lock = store._file_lock
            def other_writer():
                assert start.wait(10), 'Commit never reached its native lock'
                other = MemoryStore(memory_char_limit=10000, user_char_limit=5000)
                try:
                    assert other.add('memory', 'concurrent native entry')['success']
                finally:
                    done.set()
            @contextmanager
            def contended_lock(path):
                start.set()
                assert done.wait(10), 'Concurrent native writer did not complete'
                with original_lock(path):
                    yield
            race_pool = ThreadPoolExecutor(max_workers=1)
            race_future = race_pool.submit(copy_context().run, other_writer)
            monkeypatch.setattr(store, '_file_lock', contended_lock)
        elif damage == 'source':
            source_path.write_bytes(originals[targets[0]] + b' concurrent edit')
        elif damage in ('review', 'backup', 'journal', 'journal-drift', 'draft-transient', 'receipt'):
            getter = wa.get_pending
            hits = {'prepare': 0, 'draft': 0}
            def altered(subsystem, pid):
                record = getter(subsystem, pid)
                if record:
                    action = record['action']
                    if damage == 'review' and action == 'capacity_review':
                        record['origin'] = 'foreground'
                    elif damage == 'backup' and action == 'capacity_backup':
                        record['metadata']['capacity']['source_manifest'] = {}
                    elif damage == 'journal' and action == 'capacity_transaction_prepare':
                        return None
                    elif damage == 'journal-drift' and action == 'capacity_transaction_prepare':
                        hits['prepare'] += 1
                        if hits['prepare'] > 1:
                            record['origin'] = 'foreground'
                    elif damage == 'draft-transient' and record.get('metadata', {}).get('capacity', {}).get('state') == 'draft':
                        hits['draft'] += 1
                        if hits['draft'] == 2:
                            record['metadata']['capacity']['candidates']['MEMORY.md'] = base64.b64encode(b'UNREVIEWED BYTE SUBSTITUTION').decode()
                    elif damage == 'receipt' and action == 'capacity_transaction_observed':
                        return None
                return record
            monkeypatch.setattr(wa, 'get_pending', altered)
        return result
    monkeypatch.setattr(critic, 'review_capacity_drafts', reviewed)
    # During RED this module does not exist yet; the native flow must still run
    # and fail the observable installed-byte assertion, not collection/import.
    try:
        transaction = importlib.import_module('agent.background_review_capacity_commit')
    except ModuleNotFoundError:
        context = nullcontext()
    else:
        context = transaction.memory_capacity_commit_authority(get_hermes_home()) if authority else nullcontext()
    try:
        with context:
            native._run_review_in_thread(parent, [], 'Synthetic capacity review', task_cfg={},
                                        review_run=run, review_memory=True, capacity_plan=plan)
    finally:
        if race_pool is not None:
            race_pool.shutdown(wait=True)
            race_future.result(timeout=15)
    assert not captured.get('worker_errors'), captured
    assert store._system_prompt_snapshot == frozen, 'Main prompt cache must stay frozen'
    assert run.request_done.is_set() and not parent._active_children
    after = {t: store._path_for(t).read_bytes() for t in ('memory', 'user')}
    return store, originals, after, delivered, captured['scope'], stages


@pytest.mark.parametrize('targets', [('memory',), ('user',), ('memory', 'user')])
@pytest.mark.parametrize('layout', [False, True])
def test_reviewed_memory_transaction_installs_only_triggered_stores_and_retains_recovery(
        monkeypatch, targets, layout):
    store, before, after, delivered, scope, stages = run_commit(monkeypatch, targets=targets, layout=layout)
    for target in ('memory', 'user'):
        assert after[target] == (b'Keep blue and threshold 7.' if target in targets else before[target])
        if layout:
            assert before[target].startswith(b'\xef\xbb\xbf') and b'\r\n' in before[target]
        if target in targets:
            assert store._entries_for(target) == ['Keep blue and threshold 7.']
    assert len(scope.commit_results) == len(targets)
    assert not scope.commit_errors
    assert f'{len(targets)} memory commit(s) verified' in delivered[0]
    for reply in stages:
        record = wa.get_pending('memory', reply['pending_id'])
        cap = record['metadata']['capacity']
        assert base64.b64decode(cap['originals'][next(iter(cap['originals']))]) == before[cap['target']]
        assert record['payload']['_native_capacity_draft'] is True
    records = wa.list_pending('memory')
    assert len([r for r in records if r['action'] == 'capacity_transaction_prepare']) == len(targets)
    assert len([r for r in records if r['action'] == 'capacity_transaction_observed']) == len(targets)


def test_absent_authority_keeps_native_drafts_uninstalled(monkeypatch):
    _, before, after, delivered, scope, _ = run_commit(monkeypatch, authority=False)
    assert before == after
    assert not getattr(scope, 'commit_results', {})
    assert 'not installed' in delivered[0]


@pytest.mark.parametrize('damage', ['review', 'unreviewed', 'quota-shift', 'backup', 'journal', 'journal-drift', 'draft-transient', 'duplicate', 'cancelled', 'source', 'scanner', 'prewrite'])
def test_commit_refuses_untrusted_drifted_cancelled_or_unprepared_transaction(monkeypatch, damage):
    _, before, after, delivered, scope, _ = run_commit(monkeypatch, damage=damage)
    expected = before['memory'] + b' concurrent edit' if damage == 'source' else before['memory']
    assert after['memory'] == expected and after['user'] == before['user']
    assert not scope.commit_results and scope.commit_errors
    assert 'blocked' in delivered[0] and 'not installed' in delivered[0]


@pytest.mark.parametrize('damage', ['postwrite', 'readback', 'receipt'])
def test_postwrite_failure_restores_exact_original_bom_crlf_and_retains_journal(monkeypatch, damage):
    _, before, after, delivered, scope, _ = run_commit(monkeypatch, damage=damage, layout=True)
    assert before == after
    assert not scope.commit_results and scope.commit_errors
    assert 'rolled_back' in delivered[0]
    assert any(r['action'] == 'capacity_transaction_prepare' for r in wa.list_pending('memory'))


@pytest.mark.parametrize('damage', ['foreign', 'rollback'])
def test_uncertain_rollback_never_overwrites_foreign_edit_or_claims_success(monkeypatch, damage):
    _, before, after, delivered, scope, _ = run_commit(monkeypatch, damage=damage, layout=True)
    assert after['memory'] == (b'EXTERNAL WRITER REVISION' if damage == 'foreign' else b'Keep blue and threshold 7.')
    assert not scope.commit_results and scope.commit_errors
    assert 'in_doubt' in delivered[0]
    assert 'not installed' not in delivered[0] and 'commit(s) verified' not in delivered[0]
    backups = [r for r in wa.list_pending('memory') if r['action'] == 'capacity_backup']
    assert len(backups) == 1
    assert base64.b64decode(backups[0]['metadata']['capacity']['originals']['MEMORY.md']) == before['memory']


def test_committed_scope_cannot_replay_same_transaction_or_ordinary_approval(monkeypatch):
    from agent.background_review_capacity_commit import memory_capacity_commit_authority, commit_reviewed_memory
    from tools.memory_tool import apply_memory_pending
    store, _, after, _, scope, _ = run_commit(monkeypatch)
    records_before = wa.list_pending('memory')
    token = _SCOPE.set(scope)
    try:
        with memory_capacity_commit_authority(get_hermes_home()):
            commit_reviewed_memory(store)
    finally:
        _SCOPE.reset(token)
    assert wa.list_pending('memory') == records_before
    for record in records_before:
        if record['action'].startswith('capacity_transaction_'):
            assert apply_memory_pending(record['payload'], store)['success'] is False
    assert store._path_for('memory').read_bytes() == after['memory']


def test_authority_is_not_a_cross_profile_or_persistent_switch(tmp_path):
    from agent.background_review_capacity_commit import memory_capacity_commit_authority, _AUTHORITY
    foreign = tmp_path / 'unapproved-profile'
    with pytest.raises(ValueError, match='another profile'):
        with memory_capacity_commit_authority(foreign):
            pytest.fail('Unapproved profile was admitted')
    assert not foreign.exists()
    assert _AUTHORITY.get() is None
    with memory_capacity_commit_authority(get_hermes_home()):
        assert _AUTHORITY.get() == get_hermes_home().resolve()
    assert _AUTHORITY.get() is None


def test_native_writer_winning_lock_after_review_preserves_its_revision(monkeypatch):
    from agent.background_review_capacity import normalized_text
    _, before, after, delivered, scope, _ = run_commit(monkeypatch, damage='native-race')
    assert normalized_text(after['memory']) == normalized_text(before['memory']) + '\n§\nconcurrent native entry'
    assert after['user'] == before['user']
    assert not scope.commit_results and scope.commit_errors
    assert 'not installed' in delivered[0] and 'blocked' in delivered[0]
