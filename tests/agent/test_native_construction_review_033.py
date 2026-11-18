"""Controlled partial-construction failure; no request IO."""
import httpx
import openai
import pytest

from agent.auxiliary_async_codex import _attempt_client


@pytest.mark.asyncio
async def test_async_openai_constructor_failure_closes_http_client(monkeypatch):
    source = openai.OpenAI(api_key='offline', base_url='http://127.0.0.1:1/v1',
                           http_client=httpx.Client(trust_env=False))
    clients = []
    original = httpx.AsyncClient

    def make_client(*args, **kwargs):
        client = original(*args, **kwargs)
        clients.append(client)
        return client

    def fail(**kwargs):
        raise ValueError('injected AsyncOpenAI constructor failure')

    monkeypatch.setattr(httpx, 'AsyncClient', make_client)
    monkeypatch.setattr(openai, 'AsyncOpenAI', fail)
    try:
        with pytest.raises(ValueError, match='injected'):
            await _attempt_client(source)
        assert clients[0].is_closed
    finally:
        source.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['transport_first', 'transport_later', 'cookie', 'http_client', 'sdk'])
async def test_all_partial_construction_boundaries_close_allocations_once(monkeypatch, phase):
    import copy
    from agent import auxiliary_async_codex as native

    mounts = {scheme: httpx.HTTPTransport(verify=False) for scheme in ('http://', 'https://')}
    source = openai.OpenAI(api_key='offline', base_url='http://127.0.0.1:1/v1',
        http_client=httpx.Client(trust_env=False, mounts=mounts))
    source._client.cookies.set('construction-cookie', 'value')
    original_transport = httpx.AsyncHTTPTransport
    original_http = httpx.AsyncClient
    original_deepcopy = copy.deepcopy
    transports, clients, closes = [], [], {}
    calls = 0

    def fail():
        raise ValueError('injected ' + phase)

    def make_transport(*args, **kwargs):
        nonlocal calls
        calls += 1
        if (phase == 'transport_first' and calls == 1) or (phase == 'transport_later' and calls == 3):
            fail()
        transport = original_transport(*args, **kwargs)
        original_close = transport.aclose
        closes[id(transport)] = 0
        async def close():
            closes[id(transport)] += 1
            await original_close()
        transport.aclose = close
        transports.append(transport)
        return transport

    def make_http(*args, **kwargs):
        if phase == 'http_client':
            fail()
        client = original_http(*args, **kwargs)
        clients.append(client)
        return client

    def copy_cookie(value, *args, **kwargs):
        if phase == 'cookie' and getattr(value, 'name', None) == 'construction-cookie':
            fail()
        return original_deepcopy(value, *args, **kwargs)

    def make_sdk(**kwargs):
        assert phase == 'sdk', 'all earlier failure boundaries must prevent SDK construction'
        fail()

    monkeypatch.setattr(httpx, 'AsyncHTTPTransport', make_transport)
    monkeypatch.setattr(httpx, 'AsyncClient', make_http)
    monkeypatch.setattr(copy, 'deepcopy', copy_cookie)
    monkeypatch.setattr(openai, 'AsyncOpenAI', make_sdk)
    try:
        with pytest.raises(ValueError, match='injected ' + phase):
            await native._attempt_client(source)
        expected = 0 if phase == 'transport_first' else 2 if phase == 'transport_later' else 3
        assert len(transports) == expected
        assert all(count == 1 for count in closes.values()), closes
        assert all(client.is_closed for client in clients)
        assert not source.is_closed()
    finally:
        source.close()


@pytest.mark.asyncio
async def test_unknown_network_backend_is_rejected_not_silently_replaced():
    from agent import auxiliary_async_codex as native
    source = openai.OpenAI(api_key='offline', base_url='http://127.0.0.1:1/v1',
                           http_client=httpx.Client(trust_env=False))
    source._client._transport._pool._network_backend = object()
    attempt = None
    try:
        with pytest.raises(ValueError, match='network backend'):
            attempt = await native._attempt_client(source)
    finally:
        if attempt is not None:
            await attempt.close()
        source.close()
