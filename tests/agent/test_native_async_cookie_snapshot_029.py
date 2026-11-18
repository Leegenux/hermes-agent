import copy

import httpx
import pytest
from openai import OpenAI

from agent.auxiliary_async_codex import _attempt_client


@pytest.mark.asyncio
async def test_attempt_copies_cookie_objects_and_preserves_scope():
    http = httpx.Client(trust_env=False)
    http.cookies.set('visible', 'original', domain='example.test', path='/api')
    http.cookies.set('private', 'secret', domain='other.test', path='/')
    secure = next(c for c in http.cookies.jar if c.name == 'visible')
    secure.secure = True
    template = OpenAI(api_key='dummy', base_url='https://example.test/api', http_client=http)
    attempt = await _attempt_client(template)
    try:
        original = {c.name: c for c in http.cookies.jar}
        cloned = {c.name: c for c in attempt._client.cookies.jar}
        assert cloned.keys() == original.keys()
        assert all(cloned[k] is not original[k] for k in original)
        assert (cloned['visible'].domain, cloned['visible'].path, cloned['visible'].secure) == ('example.test', '/api', True)
        http.cookies.set('visible', 'changed', domain='example.test', path='/api')
        def cookie(url):
            return attempt._client.build_request('GET', url).headers.get('cookie', '')
        assert cookie('https://example.test/api/responses') == 'visible=original'
        assert not cookie('http://example.test/api/responses')
        assert not cookie('https://example.test/elsewhere')
        assert 'private' not in cookie('https://example.test/api/responses')
        assert cookie('https://other.test/') == 'private=secret'
    finally:
        await attempt.close()
        template.close()
