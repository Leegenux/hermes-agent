"""No-network builder integration with disposable HOME and public dummy credentials."""
import asyncio
import httpx
import pytest
from compression import zstd

from agent.process_bootstrap import build_keepalive_http_client

OFFICIAL = 'https://chatgpt.com/backend-api/codex'
RAW = '{"input":"你好 café"}'.encode('utf-8')

@pytest.fixture
def config_home(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    for name in ('HOME', 'USERPROFILE', 'HERMES_HOME'):
        monkeypatch.setenv(name, str(home))
    monkeypatch.delenv('HERMES_PROFILE', raising=False)
    return home

def configure(home, yaml_value):
    if yaml_value is not None:
        (home / 'config.yaml').write_text('agent:\n  codex_request_compression: ' + yaml_value + '\n', encoding='utf-8')

def verify_request(client, enabled, *, request=None):
    if request is None:
        request = client.build_request('POST', OFFICIAL + '/responses', content=RAW)
    for hook in client.event_hooks['request']:
        result = hook(request)
        if result is not None:
            asyncio.run(result)
    assert int(request.headers['content-length']) == len(request.content)
    if enabled:
        assert request.headers['content-encoding'] == 'zstd'
        assert zstd.decompress(request.content) == RAW
    else:
        assert 'content-encoding' not in request.headers
        assert request.content == RAW
    return request

@pytest.mark.parametrize('value,enabled', [(None, False), ('true', True), ('"true"', False), ('1', False)])
def test_actual_async_keepalive_builder(config_home, value, enabled):
    configure(config_home, value)
    client = build_keepalive_http_client(OFFICIAL, async_mode=True)
    assert isinstance(client, httpx.AsyncClient)
    try:
        verify_request(client, enabled)
    finally:
        asyncio.run(client.aclose())

@pytest.mark.parametrize('value,enabled', [(None, False), ('true', True)])
def test_actual_aux_sync_and_async_builders(config_home, value, enabled):
    configure(config_home, value)
    from agent.auxiliary_client import _create_openai_client, _to_async_client
    from openai._models import FinalRequestOptions
    sync = _create_openai_client(api_key='public-test-not-a-secret', base_url=OFFICIAL)
    async_client = None
    try:
        assert isinstance(sync._client, httpx.Client)
        options = FinalRequestOptions(method='post', url='/responses', content=RAW)
        req = verify_request(sync._client, enabled, request=sync._build_request(options))
        assert req.headers['originator'] == 'hermes-agent'
        async_client, _ = _to_async_client(sync, 'synthetic-model')
        assert isinstance(async_client._client, httpx.AsyncClient)
        req = verify_request(async_client._client, enabled, request=async_client._build_request(options))
        assert req.headers['originator'] == 'hermes-agent'
    finally:
        if async_client is not None:
            asyncio.run(async_client.close())
        sync.close()

@pytest.mark.parametrize('value,enabled', [(None, False), ('true', True)])
def test_actual_codex_aux_entry_and_async_wrapper(config_home, monkeypatch, value, enabled):
    configure(config_home, value)
    from agent import auxiliary_client as aux
    from openai._models import FinalRequestOptions
    monkeypatch.setattr(aux, '_resolve_codex_credential_and_base', lambda model: ('public-test-not-a-secret', OFFICIAL))
    wrapped, model = aux._build_codex_client('synthetic-model')
    try:
        assert model == 'synthetic-model'
        assert isinstance(wrapped, aux.CodexAuxiliaryClient)
        sdk = wrapped._real_client
        options = FinalRequestOptions(method='post', url='/responses', content=RAW)
        request = verify_request(sdk._client, enabled, request=sdk._build_request(options))
        assert request.headers['originator'] == 'hermes-agent'
        async_wrapper, _ = aux._to_async_client(wrapped, model)
        assert isinstance(async_wrapper, aux.AsyncCodexAuxiliaryClient)
        assert async_wrapper._real_client is sdk
    finally:
        wrapped.close()

@pytest.mark.parametrize('method,path', [('GET','/responses'), ('POST','/models')])
def test_opt_in_ignores_nonmatching_requests(config_home, method, path):
    configure(config_home, 'true')
    client = build_keepalive_http_client(OFFICIAL)
    try:
        request = client.build_request(method, OFFICIAL + path, content=RAW)
        for hook in client.event_hooks['request']:
            hook(request)
        assert 'content-encoding' not in request.headers
        assert request.content == RAW
        assert int(request.headers['content-length']) == len(RAW)
    finally:
        client.close()

def test_opt_in_other_host_never_gets_hook(config_home):
    configure(config_home, 'true')
    client = build_keepalive_http_client('https://proxy.example/backend-api/codex')
    try:
        request = client.build_request('POST', 'https://proxy.example/backend-api/codex/responses', content=RAW)
        for hook in client.event_hooks['request']:
            hook(request)
        assert 'content-encoding' not in request.headers
        assert request.content == RAW
        assert int(request.headers['content-length']) == len(RAW)
    finally:
        client.close()
