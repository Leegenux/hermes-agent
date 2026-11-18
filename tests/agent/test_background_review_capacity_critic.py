"""Native creator/critic worker, real stores and dispatch guards; fake provider only.

Removing the critic hook must lose the durable review. Relaxing its parser or
binding must let the adversarial verdict/source cases pass, which is forbidden.
"""
import json
from types import SimpleNamespace

import pytest

from agent import background_review as native
from tools import write_approval as wa
from tests.agent.test_background_review_capacity_drafts import stage_fixture
from tests.agent.test_background_review_capacity_skill_drafts import stage_skill_fixture

CHECKS = ('rules', 'parameters', 'conditions', 'permissions', 'ownership',
          'business_conclusions', 'topics', 'code_units')


def exercise(monkeypatch, *, kind='memory', damage=None, flags=None):
    real_worker, real_fork = native._run_review_in_thread, native._run_review_fork
    seen, delivered, released, receipts, builds = [], [], [], [], []
    captured = {}

    def worker(parent, snapshot, prompt, task_cfg=None, **kwargs):
        synthetic_generator = native._run_review_fork
        run = native.prepare_background_review_run(parent)
        captured['run'] = run
        parent.background_review_callback = delivered.append
        parent._active_children = []
        parent._background_review_agent = None
        cfg = {'extra_tools': ['terminal', 'write_file']}

        class ProviderDouble:
            _memory_enabled = True
            _user_profile_enabled = False
            _session_messages = []
            session_input_tokens = 13
            session_output_tokens = 2
            session_api_calls = 1
            _review_input_token_budget = 80
            provider = 'synthetic'
            model = 'fixture'
            base_url = 'offline-only'

            def __init__(self, critic):
                self.critic = critic

            def release_clients(self):
                released.append(self.critic)

            def run_conversation(self, *, user_message, conversation_history):
                if not self.critic:
                    synthetic_generator(parent, snapshot, prompt, task_cfg, run,
                                        SimpleNamespace(), True, False)
                    if damage == 'cancelled':
                        run.cancel()
                    if damage == 'budget':
                        self.session_input_tokens = 80
                    if damage == 'iterations':
                        self.session_api_calls = 16
                    return {'final_response': 'generator output MUST NOT reach critic'}
                from hermes_cli.plugins import _get_pre_tool_call_directive_details
                from agent.background_review_capacity_drafts import _SCOPE
                captured['scope'] = _SCOPE.get()
                seen.append((user_message, conversation_history))
                assert self._review_input_token_budget == 67
                assert not run.request_done.is_set()
                assert native.prepare_background_review_run(parent) is None
                for tool in ('memory', 'skill_manage', 'terminal', 'write_file',
                             'read_file', 'execute_code', 'goal_manage'):
                    assert _get_pre_tool_call_directive_details(tool, {}).action == 'block'
                packet = json.loads(user_message.partition('CAPACITY_REVIEW_PACKET_JSON\n')[2])
                captured['packet'] = packet
                verdict = {'packet_sha256': packet['packet_sha256'], 'decision': 'PASS',
                           'checks': {name: True for name in CHECKS}, 'issues': []}
                if damage == 'false':
                    verdict['checks']['permissions'] = False
                elif damage == 'missing':
                    verdict['checks'].pop('ownership')
                elif damage == 'integer':
                    verdict['checks']['rules'] = 1
                elif damage == 'issues':
                    verdict['issues'] = ['Threshold 7 was dropped']
                elif damage == 'hash':
                    verdict['packet_sha256'] = '0' * 64
                elif damage == 'FAIL':
                    verdict['decision'] = 'FAIL'
                elif damage == 'source':
                    from hermes_constants import get_hermes_home
                    path = get_hermes_home() / 'memories' / 'MEMORY.md'
                    path.write_bytes(path.read_bytes() + b' concurrent change')
                elif damage == 'pending':
                    getter = wa.get_pending
                    draft_id = packet['packet']['targets'][0]['pending_id']
                    def altered(subsystem, pending_id):
                        record = getter(subsystem, pending_id)
                        if pending_id == draft_id:
                            record['origin'] = 'foreground'
                        return record
                    monkeypatch.setattr(wa, 'get_pending', altered)
                elif damage == 'throw':
                    raise RuntimeError('Synthetic critic unavailable')
                text = json.dumps(verdict)
                if damage == 'malformed':
                    text = '```json\n' + text + '\n```'
                elif damage == 'duplicate':
                    text = text[:-1] + ', "decision": "PASS"}'
                if damage == 'persist':
                    getter = wa.get_pending
                    monkeypatch.setattr(wa, 'get_pending', lambda sub, pid: None
                        if (getter(sub, pid) or {}).get('action') == 'capacity_review'
                        else getter(sub, pid))
                return {'final_response': text}

        def builder(agent, config, *, max_iterations, **other):
            assert agent is parent and config is cfg
            critic = bool(builds)
            if critic:
                assert max_iterations == 1
            builds.append(critic)
            return ProviderDouble(critic), {}, False

        monkeypatch.setattr(native, 'build_cache_parity_fork', builder)
        monkeypatch.setattr(native, '_run_review_fork', real_fork)
        kwargs['review_run'] = run
        real_worker(parent, [{'role': 'user', 'content': 'PRIVATE WINDOW MUST NOT REPLAY'}],
                    prompt, task_cfg=cfg, **kwargs)
        captured['children'] = list(parent._active_children)
        for sub in ('memory', 'skills'):
            receipts.extend(r for r in wa.list_pending(sub) if r['action'] == 'capacity_review')

    monkeypatch.setattr(native, '_run_review_in_thread', worker)
    if kind == 'memory':
        result = stage_fixture(monkeypatch, **(flags or {}))
    else:
        result = stage_skill_fixture(monkeypatch, **(flags or {}))
    return result, receipts, seen, delivered, released, builds, captured


@pytest.mark.parametrize('kind', ['memory', 'skills'])
def test_independent_native_critic_persists_packet_bound_review_but_never_installs(monkeypatch, kind):
    result, reviews, seen, delivered, released, builds, captured = exercise(monkeypatch, kind=kind)
    assert result[0].get('staged')
    assert len(reviews) == 1, 'No durable independent review after the native creator'
    review = reviews[0]
    assert review['origin'] == 'background_review'
    assert review['metadata']['capacity_review']['decision'] == 'MODEL_PASS'
    assert review['metadata']['capacity_review']['installation_authorized'] is False
    assert len(captured['scope'].review_results) == 1
    packet = captured['packet']
    assert review['payload']['packet_sha256'] == packet['packet_sha256']
    assert all(v is True for v in review['payload']['verdict']['checks'].values())
    prompt, history = seen[0]
    assert history == [] and 'PRIVATE WINDOW MUST NOT REPLAY' not in prompt
    assert 'generator output MUST NOT reach critic' not in prompt
    assert 'No business workflow' in prompt and 'Never reproduce credentials' in prompt
    docs = packet['packet']['targets'][0]
    assert len(docs['originals']) == (1 if kind == 'memory' else 2)
    if kind == 'skills':
        assert 'scripts/must-not-run.py' in docs['source_manifest']
        assert 'BUSINESS MUST NEVER RUN' not in prompt
    candidate = wa.get_pending('memory' if kind == 'memory' else 'skills', result[0]['pending_id'])
    assert candidate['metadata']['capacity']['independent_review_passed'] is False
    if kind == 'memory':
        from tools.memory_tool import apply_memory_pending, MemoryStore
        store = MemoryStore()
        store.load_from_disk()
        assert apply_memory_pending(review['payload'], store)['success'] is False
    else:
        from tools.skill_manager_tool import apply_skill_pending
        assert json.loads(apply_skill_pending(review['payload']))['success'] is False
    assert wa.get_pending(review['subsystem'], review['id']) == review
    assert 'not installed' in delivered[0].lower() and '1 preservation review(s) passed' in delivered[0].lower()
    assert sorted(released) == [False, True]
    assert captured['run'].request_done.is_set() and not captured['children']


@pytest.mark.parametrize('damage', ['false', 'missing', 'integer', 'issues', 'hash', 'FAIL',
                                  'malformed', 'duplicate', 'pending', 'throw', 'persist'])
def test_reviewer_cannot_bypass_fidelity_schema_binding_or_durability(monkeypatch, damage):
    result, reviews, seen, delivered, released, builds, captured = exercise(monkeypatch, damage=damage)
    assert result[0].get('staged')
    assert seen, 'Independent review was never attempted'
    assert not captured['scope'].review_results
    # A failed readback may leave an orphan model assessment. Its existence is
    # not acceptance: only the trusted live scope can bind a verified record.
    assert all(r['metadata']['capacity_review']['installation_authorized'] is False for r in reviews)
    if damage != 'persist':
        assert not reviews
    assert '1 blocked' in delivered[0].lower() and 'not installed' in delivered[0].lower()
    reason = {'pending': 'stale_packet', 'throw': 'critic_unavailable',
              'persist': 'persistence_readback'}.get(damage, 'verdict_rejected')
    assert reason in delivered[0]
    assert captured['run'].request_done.is_set() and not captured['children']


@pytest.mark.parametrize('damage', ['cancelled', 'budget', 'iterations'])
def test_cancel_or_exhausted_original_budget_never_starts_critic(monkeypatch, damage):
    result, reviews, seen, delivered, released, builds, captured = exercise(monkeypatch, damage=damage)
    assert result[0].get('staged') and not reviews and not seen
    assert builds == [False] and released == [False]
    assert '1 blocked' in delivered[0].lower() and 'not installed' in delivered[0].lower()
    assert ('cancelled' if damage == 'cancelled' else 'budget_exhausted') in delivered[0]
    assert captured['run'].request_done.is_set()


@pytest.mark.parametrize('flags', [{'include_plan': False}, {'explicit': True}, {'chars': 6002}])
def test_noncapacity_explicit_or_rejected_candidate_does_not_add_critic_call(monkeypatch, flags):
    result, reviews, seen, delivered, released, builds, captured = exercise(monkeypatch, flags=flags)
    assert not reviews and not seen and builds == [False]
    assert captured['run'].request_done.is_set()


def test_verified_review_packet_rejects_later_source_change_without_overwriting_it(monkeypatch):
    from agent.background_review_capacity_critic import review_packet
    from tools.memory_tool import get_memory_dir
    _, _, _, _, _, _, captured = exercise(monkeypatch)
    source = get_memory_dir() / 'MEMORY.md'
    changed = source.read_bytes() + b' concurrent source revision'
    source.write_bytes(changed)
    with pytest.raises(ValueError, match='[Ss]ource'):
        review_packet(captured['scope'])
    assert source.read_bytes() == changed
