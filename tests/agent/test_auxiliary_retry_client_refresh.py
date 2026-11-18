"""A Codex watchdog evicts a closed client before the same-route retry."""
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent import auxiliary_retry_client as retry_client


@pytest.fixture(autouse=True)
def _deny_non_loopback_network(monkeypatch):
    """Old sync SDK stubs must never let a native async request leave the test host."""
    import ipaddress
    import socket

    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    getaddrinfo = socket.getaddrinfo

    def check(host):
        if host == 'localhost':
            return
        try:
            allowed = ipaddress.ip_address(host).is_loopback
        except ValueError:
            allowed = False
        if not allowed:
            raise AssertionError('test blocked non-loopback network destination')

    def guarded_connect(sock, address):
        check(address[0])
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        check(address[0])
        return connect_ex(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        check(host)
        return getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, 'connect', guarded_connect)
    monkeypatch.setattr(socket.socket, 'connect_ex', guarded_connect_ex)
    monkeypatch.setattr(socket, 'getaddrinfo', guarded_getaddrinfo)


def test_network_guard_blocks_external_resolution_and_connect():
    import socket
    with pytest.raises(AssertionError, match='non-loopback'):
        socket.getaddrinfo('chatgpt.com', 443)
    with socket.socket() as sock:
        with pytest.raises(AssertionError, match='non-loopback'):
            sock.connect(('203.0.113.1', 443))
        with pytest.raises(AssertionError, match='non-loopback'):
            sock.connect_ex(('203.0.113.1', 443))


class Leaf:
    def __init__(self, identity):
        self.identity = identity
        self.base_url = 'https://chatgpt.com/backend-api/codex'
        self.api_key = 'test-only-key'
        self.responses = SimpleNamespace(create=self.create)
        self.closed = False
        self.requests = 0

    def create(self, **kwargs):
        self.requests += 1
        if self.closed:
            raise ConnectionError('closed client reused')
        return iter(())

    def close(self):
        self.closed = True


@pytest.fixture
def fake_codex(monkeypatch):
    leaves = []
    monkeypatch.setattr(aux, '_client_cache', {})
    monkeypatch.setattr(aux, '_TRANSIENT_RETRY_BACKOFF_BASE', 0)
    monkeypatch.setattr(aux, '_AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS', 5.0)
    monkeypatch.setattr(aux, '_resolve_task_provider_model',
                        lambda *args: ('openai-codex', 'test-model', None, None, None))

    def build(provider, model, async_mode, **kwargs):
        leaf = Leaf(len(leaves))
        leaves.append(leaf)
        client = aux.CodexAuxiliaryClient(leaf, model)
        if async_mode:
            client, _ = aux._to_async_client(client, model)
        return client, model

    monkeypatch.setattr(aux, 'resolve_provider_client', build)

    def consume(stream, *, model, on_event, **kwargs):
        guard = on_event.__self__
        if guard._client.identity == 0:
            assert guard.timed_out.wait(3), 'watchdog did not trigger'
            raise ConnectionError('closed by guard')
        return SimpleNamespace(output=[SimpleNamespace(type='message', content=[
            SimpleNamespace(type='output_text', text='recovered')])], usage=None)

    monkeypatch.setattr('agent.codex_runtime._consume_codex_event_stream', consume)
    return leaves


@pytest.mark.asyncio
@pytest.mark.parametrize('async_mode', [False, True])
async def test_guard_evicts_real_sdk_and_retry_preserves_frozen_route(monkeypatch, async_mode):
    import asyncio
    from openai.resources.responses import AsyncResponses
    from agent.process_bootstrap import build_keepalive_http_client
    from openai import OpenAI
    monkeypatch.setattr(aux, '_client_cache', {})
    monkeypatch.setattr(aux, '_TRANSIENT_RETRY_BACKOFF_BASE', 0)
    monkeypatch.setattr(aux, '_AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS', 0.05)
    monkeypatch.setattr(aux, '_resolve_task_provider_model',
                        lambda *a: ('openai-codex', 'test-model', None, None, None))
    http = build_keepalive_http_client('https://chatgpt.com/backend-api/codex')
    http.cookies.set('test-cookie', 'test-value', domain='chatgpt.com', path='/')
    original = OpenAI(api_key='synthetic-key', base_url='https://chatgpt.com/backend-api/codex',
                      default_headers={'X-Frozen': 'yes'}, default_query={'route': 'original'},
                      timeout=19, max_retries=0, http_client=http)
    frozen = (str(original.base_url), original.api_key, dict(original._custom_headers),
              dict(original._custom_query), original.timeout, original.max_retries)
    wrapper = aux.CodexAuxiliaryClient(original, 'test-model')
    if async_mode:
        wrapper, _ = aux._to_async_client(wrapper, 'test-model')
    monkeypatch.setattr(aux, 'resolve_provider_client', lambda *a, **k: (wrapper, 'test-model'))
    sent = []
    attempts = []

    async def create(self, **kwargs):
        # Native transport has an attempt-owned async SDK leaf. The recovery
        # contract below concerns the independently cached sync template, not
        # that short-lived attempt. Capture both; never allow a real send.
        attempt = self._client
        attempts.append(attempt)
        templates = {id(entry[0]._real_client): entry[0]._real_client
                     for entry in aux._client_cache.values()
                     if hasattr(entry[0], '_real_client')}
        assert len(templates) == 1
        leaf = next(iter(templates.values()))
        sent.append(leaf)
        assert not attempt._client.is_closed
        if len(sent) == 1:
            try:
                await asyncio.sleep(60)  # actual adapter deadline must cancel this await
            except asyncio.CancelledError:
                original.api_key = 'other-synthetic-key'
                original.base_url = 'https://other.invalid/v1'
                original._custom_headers['X-Frozen'] = 'changed'
                original._custom_query['route'] = 'changed'
                raise
        return SimpleNamespace(output=[SimpleNamespace(type='message', content=[
            SimpleNamespace(type='output_text', text='recovered')])], usage=None)

    monkeypatch.setattr(AsyncResponses, 'create', create)
    request = dict(task='compression', messages=[{'role': 'user', 'content': 'test'}], timeout=2)
    result = await aux.async_call_llm(**request) if async_mode else aux.call_llm(**request)
    assert result.choices[0].message.content == 'recovered'
    fresh = sent[-1]
    assert len(sent) == 2 and fresh is not original and original._client.is_closed
    assert fresh._client is not http and not fresh._client.is_closed
    assert fresh._client._transport is not http._transport
    assert fresh._client._transport._inner is http._transport._inner
    assert next(iter(fresh._client._mounts.values())) is fresh._client._transport
    assert fresh._client.cookies.get('test-cookie') == 'test-value'
    assert fresh._client.follow_redirects == http.follow_redirects
    assert fresh._client.trust_env == http.trust_env
    assert fresh._client.event_hooks == http.event_hooks
    assert fresh._client.timeout == http.timeout
    assert (str(fresh.base_url), fresh.api_key, fresh._custom_headers, fresh._custom_query,
            fresh.timeout, fresh.max_retries) == frozen
    assert len(attempts) == 2 and all(attempt._client.is_closed for attempt in attempts)
    actual_retry = attempts[-1]
    assert (str(actual_retry.base_url), actual_retry.api_key, actual_retry._custom_headers,
            actual_retry._custom_query, actual_retry.timeout, actual_retry.max_retries) == frozen
    assert actual_retry._client.cookies.get('test-cookie') == 'test-value'
    assert actual_retry._client.cookies is not fresh._client.cookies
    fresh.close()


def test_unknown_custom_transport_is_not_cloned():
    import httpx
    from openai import OpenAI
    http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    leaf = OpenAI(api_key='synthetic-key', base_url='https://example.invalid/v1', http_client=http)
    wrapped = aux.CodexAuxiliaryClient(leaf, 'test-model')
    assert retry_client.freeze_codex_retry_client(wrapped) is None
    leaf.close()


def test_async_concurrent_rebuild_closes_loser_and_keeps_winner(monkeypatch):
    import concurrent.futures
    import threading
    import httpx
    from openai import OpenAI
    from agent.process_bootstrap import build_keepalive_http_client
    url = 'https://chatgpt.com/backend-api/codex'
    old = OpenAI(api_key='synthetic-key', base_url=url, http_client=build_keepalive_http_client(url))
    wrapped = aux.AsyncCodexAuxiliaryClient(aux.CodexAuxiliaryClient(old, 'test-model'))
    rebuild = retry_client.freeze_codex_retry_client(wrapped)
    assert rebuild is not None
    barrier = threading.Barrier(2, timeout=5)
    made = []
    def race_build():
        client = rebuild()
        made.append(client)
        barrier.wait()
        return client
    loop = SimpleNamespace(is_closed=lambda: False)
    monkeypatch.setattr(aux, '_current_event_loop', lambda: loop)
    monkeypatch.setattr(aux, '_client_cache', {})
    req = SimpleNamespace(client=wrapped, final_model='test-model')
    route = (('route',), {'async_mode': True}, race_build)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: retry_client.retry_route_client(req, route), range(2)))
    assert results[0] is results[1]
    winner = results[0]
    loser = next(client for client in made if client is not winner)
    assert loser._real_client._client.is_closed
    assert not winner._real_client._client.is_closed
    transport = winner._real_client._client._transport
    monkeypatch.setattr(transport._inner, 'handle_request', lambda request: httpx.Response(200, request=request))
    assert winner._real_client._client.get(url).status_code == 200
    winner._real_client.close()
    old.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('async_mode', [False, True])
@pytest.mark.parametrize('replacement', [False, True])
async def test_blip_reuses_healthy_client_or_concurrent_replacement(fake_codex, monkeypatch, async_mode, replacement):
    seen = []

    def consume(stream, *, model, on_event, **kwargs):
        leaf = on_event.__self__._client
        seen.append(leaf.identity)
        if len(seen) == 1:
            if replacement:
                other = Leaf(1)
                fake_codex.append(other)
                wrapper = aux.CodexAuxiliaryClient(other, 'test-model')
                if async_mode:
                    wrapper, _ = aux._to_async_client(wrapper, 'test-model')
                with aux._client_cache_lock:
                    key, (_, default, loop) = next(iter(aux._client_cache.items()))
                    aux._client_cache[key] = (wrapper, default, loop)
            raise ConnectionError('temporary network blip')
        return SimpleNamespace(output=[SimpleNamespace(type='message', content=[
            SimpleNamespace(type='output_text', text='ok')])], usage=None)

    monkeypatch.setattr('agent.codex_runtime._consume_codex_event_stream', consume)
    request = dict(task='compression', messages=[{'role': 'user', 'content': 'test'}], timeout=2)
    result = await aux.async_call_llm(**request) if async_mode else aux.call_llm(**request)
    assert result.choices[0].message.content == 'ok'
    assert seen == ([0, 1] if replacement else [0, 0])
    assert len(fake_codex) == (2 if replacement else 1)
    assert not fake_codex[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize('async_mode', [False, True])
@pytest.mark.parametrize('task', ['vision', 'compression'])
async def test_uncached_vision_and_rescued_client_remain_on_original_route(fake_codex, monkeypatch, async_mode, task):
    leaf = Leaf(42)
    client = aux.CodexAuxiliaryClient(leaf, 'test-model')
    if async_mode:
        client, _ = aux._to_async_client(client, 'test-model')
    monkeypatch.setattr(aux, '_resolve_call_client', lambda *a, **k:
                        aux._ResolvedAuxRoute(client, 'test-model', 'openai-codex', 'openai-codex'))
    attempts = []

    def consume(stream, *, model, on_event, **kwargs):
        attempts.append(on_event.__self__._client.identity)
        if len(attempts) == 1:
            raise ConnectionError('temporary network blip')
        return SimpleNamespace(output=[SimpleNamespace(type='message', content=[
            SimpleNamespace(type='output_text', text='original')])], usage=None)

    monkeypatch.setattr('agent.codex_runtime._consume_codex_event_stream', consume)
    request = dict(task=task, messages=[{'role': 'user', 'content': 'test'}], timeout=2)
    result = await aux.async_call_llm(**request) if async_mode else aux.call_llm(**request)
    assert result.choices[0].message.content == 'original'
    assert attempts == [42, 42]
    assert fake_codex == []


@pytest.mark.asyncio
@pytest.mark.parametrize('async_mode', [False, True])
async def test_changed_pool_identity_does_not_redirect_retry(fake_codex, monkeypatch, async_mode):
    pool = ['first']
    monkeypatch.setattr(aux, '_pool_cache_hint', lambda *a, **k: pool[0])
    seen = []

    def consume(stream, *, model, on_event, **kwargs):
        seen.append(on_event.__self__._client.identity)
        if len(seen) == 1:
            pool[0] = 'other-credentials'
            raise ConnectionError('network blip')
        return SimpleNamespace(output=[SimpleNamespace(type='message', content=[
            SimpleNamespace(type='output_text', text='same route')])], usage=None)

    monkeypatch.setattr('agent.codex_runtime._consume_codex_event_stream', consume)
    request = dict(task='compression', messages=[{'role': 'user', 'content': 'test'}], timeout=2)
    result = await aux.async_call_llm(**request) if async_mode else aux.call_llm(**request)
    assert result.choices[0].message.content == 'same route'
    assert seen == [0, 0]
    assert len(fake_codex) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('async_mode', [False, True])
@pytest.mark.parametrize('identity', ['profile', 'pool'])
async def test_aba_during_cache_lookup_never_sends_to_another_identity(fake_codex, monkeypatch, async_mode, identity):
    """A -> B only during lookup -> A must not admit B's cached credentials."""
    import inspect
    real_key = aux._client_cache_key
    armed = [False]
    sent = []

    def key(*args, **kwargs):
        original = real_key(*args, **kwargs)
        if armed[0] and inspect.currentframe().f_back.f_code.co_name == '_get_cached_client':
            return original + (identity, 'other')
        return original

    monkeypatch.setattr(aux, '_client_cache_key', key)

    def consume(stream, *, model, on_event, **kwargs):
        leaf = on_event.__self__._client
        sent.append(leaf.identity)
        if len(sent) == 1:
            other = Leaf(99)
            wrapper = aux.CodexAuxiliaryClient(other, 'test-model')
            if async_mode:
                wrapper, _ = aux._to_async_client(wrapper, 'test-model')
            with aux._client_cache_lock:
                original, (_, default, loop) = next(iter(aux._client_cache.items()))
                aux._client_cache[original + (identity, 'other')] = (wrapper, default, loop)
            armed[0] = True
            raise ConnectionError('network blip')
        return SimpleNamespace(output=[SimpleNamespace(type='message', content=[
            SimpleNamespace(type='output_text', text='ok')])], usage=None)

    monkeypatch.setattr('agent.codex_runtime._consume_codex_event_stream', consume)
    request = dict(task='compression', messages=[{'role': 'user', 'content': 'test'}], timeout=2)
    result = await aux.async_call_llm(**request) if async_mode else aux.call_llm(**request)
    assert result.choices[0].message.content == 'ok'
    assert sent == [0, 0], 'retry must never send the request on another identity'
