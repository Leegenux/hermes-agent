"""Offline contract: preserve known Codex compression hooks, reject unknown hooks."""
import asyncio
import json

import httpx
import openai
import pytest

from agent.auxiliary_async_codex import _attempt_client
from agent.codex_headers import CODEX_AUX_BASE_URL, codex_request_compression_hooks


def test_known_official_compression_hook_has_equivalent_async_wire_encoding():
    from compression import zstd

    async def run():
        hooks = codex_request_compression_hooks(CODEX_AUX_BASE_URL)
        assert hooks and hooks['request']
        source = openai.OpenAI(api_key='offline-only', base_url=CODEX_AUX_BASE_URL,
                              http_client=httpx.Client(trust_env=False, event_hooks=hooks))
        attempt = None
        try:
            attempt = await _attempt_client(source)
            payload = {'model': 'test', 'input': 'offline payload'}
            body = json.dumps(payload).encode()
            sync_request = httpx.Request('POST', CODEX_AUX_BASE_URL + '/responses', content=body)
            async_request = httpx.Request('POST', CODEX_AUX_BASE_URL + '/responses', content=body)
            for hook in source._client.event_hooks['request']:
                hook(sync_request)
            assert attempt._client.event_hooks['request']
            for hook in attempt._client.event_hooks['request']:
                await hook(async_request)
            assert async_request.headers['content-encoding'] == 'zstd'
            assert async_request.content == sync_request.content
            assert zstd.decompress(async_request.content) == body
            assert int(async_request.headers['content-length']) == len(async_request.content)
            assert not source.is_closed()
        finally:
            if attempt is not None:
                await attempt.close()
            source.close()
    asyncio.run(run())


def test_unknown_request_hook_is_not_silently_dropped():
    def unknown_hook(request):
        request.headers['x-custom-route'] = 'must-not-drop'

    source = openai.OpenAI(api_key='offline-only', base_url=CODEX_AUX_BASE_URL,
                          http_client=httpx.Client(trust_env=False,
                                                   event_hooks={'request': [unknown_hook]}))
    try:
        with pytest.raises(ValueError, match='unsupported Codex async'):
            asyncio.run(_attempt_client(source))
    finally:
        source.close()
