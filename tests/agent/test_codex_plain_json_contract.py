import pytest
import asyncio
import httpx
from compression import zstd
from agent.process_bootstrap import build_keepalive_http_client

OFFICIAL = 'https://chatgpt.com/backend-api/codex'

@pytest.mark.parametrize('async_mode', [False, True])
def test_default_official_client_has_no_compression_hook(async_mode):
    from agent.codex_headers import codex_request_compression_hooks
    assert codex_request_compression_hooks(OFFICIAL, async_mode=async_mode) is None

def test_default_real_builder_sends_unchanged_utf8_json():
    client = build_keepalive_http_client(OFFICIAL)
    try:
        raw = '{"input":"你好"}'.encode('utf-8')
        request = client.build_request('POST', OFFICIAL + '/responses', content=raw, headers={'content-type':'application/json'})
        for hook in client.event_hooks['request']:
            hook(request)
        assert 'content-encoding' not in request.headers
        assert request.content == raw
        assert int(request.headers['content-length']) == len(raw)
    finally:
        client.close()

@pytest.mark.parametrize('async_mode', [False, True])
@pytest.mark.parametrize('already_encoded', [False, True])
def test_opt_in_hook_round_trips_and_respects_existing_encoding(monkeypatch, async_mode, already_encoded):
    from agent.codex_headers import codex_request_compression_hooks
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', lambda: {'agent': {'codex_request_compression': True}})
    hooks = codex_request_compression_hooks(OFFICIAL, async_mode=async_mode)
    raw = '{"input":"你好 café"}'.encode('utf-8')
    headers = {'content-encoding': 'identity'} if already_encoded else {}
    request = httpx.Request('POST', OFFICIAL + '/responses', content=raw, headers=headers)
    if async_mode:
        asyncio.run(hooks['request'][0](request))
    else:
        hooks['request'][0](request)
    assert int(request.headers['content-length']) == len(request.content)
    if already_encoded:
        assert request.headers['content-encoding'] == 'identity'
        assert request.content == raw
    else:
        assert request.headers['content-encoding'] == 'zstd'
        assert zstd.decompress(request.content) == raw

def test_default_is_registered():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG['agent']['codex_request_compression'] is False
