"""Native Codex auxiliary attempts must cancel their own real HTTP request."""
import asyncio
import json
import time

import httpx
import openai
import pytest

from agent.auxiliary_client import AsyncCodexAuxiliaryClient, CodexAuxiliaryClient


def event(kind, **data):
    return ('data: ' + json.dumps({'type': kind, **data}) + '\n\n').encode()


@pytest.mark.parametrize('retries', [0, 2])
@pytest.mark.parametrize('phase', ['preheaders', 'postheaders'])
def test_cancelled_attempt_does_not_interrupt_other_requests(retries, phase):
    async def run():
        seen_a, seen_b, closed_a = asyncio.Event(), asyncio.Event(), asyncio.Event()
        handlers = set()
        counts = {'A': 0, 'B': 0, 'C': 0}
        async def handle(reader, writer):
            handlers.add(asyncio.current_task())
            try:
                headers = await reader.readuntil(b'\r\n\r\n')
                length = next(int(x.split(b':', 1)[1]) for x in headers.split(b'\r\n') if x.lower().startswith(b'content-length:'))
                payload = await reader.readexactly(length)
                label = next(x for x in counts if ('probe-' + x).encode() in payload)
                counts[label] += 1
                if label == 'A':
                    if phase == 'postheaders':
                        writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n')
                        writer.write(event('response.created', response={'id': 'A', 'status': 'in_progress', 'output': []}))
                        await writer.drain()
                    seen_a.set()
                    await reader.read()
                    closed_a.set()
                    return
                writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n')
                if label == 'B':
                    seen_b.set()
                    await closed_a.wait()
                writer.write(event('response.output_item.done', output_index=0, item={'id': 'msg', 'type': 'message', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': label, 'annotations': []}]}))
                writer.write(event('response.completed', response={'id': label, 'status': 'completed', 'output': [], 'usage': {'input_tokens': 3, 'output_tokens': 1, 'total_tokens': 4}}))
                await writer.drain()
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, OSError):
                    pass
                handlers.discard(asyncio.current_task())
        server = await asyncio.start_server(handle, '127.0.0.1', 0)
        port = server.sockets[0].getsockname()[1]
        sync = openai.OpenAI(api_key='test-only', base_url=f'http://127.0.0.1:{port}/v1', max_retries=retries, http_client=httpx.Client(trust_env=False))
        adapter = AsyncCodexAuxiliaryClient(CodexAuxiliaryClient(sync, 'test'))
        async def call(label):
            return await adapter.chat.completions.create(model='test', messages=[{'role': 'user', 'content': 'probe-' + label}], timeout=20)
        a = b = None
        try:
            a = asyncio.create_task(call('A'))
            await asyncio.wait_for(seen_a.wait(), 3)
            b = asyncio.create_task(call('B'))
            await asyncio.wait_for(seen_b.wait(), 3)
            start = time.monotonic()
            a.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(a, 3)
            await asyncio.wait_for(closed_a.wait(), 2)
            assert time.monotonic() - start < 2
            result_b = await asyncio.wait_for(b, 3)
            result_c = await asyncio.wait_for(call('C'), 3)
            assert result_b.choices[0].message.content == 'B'
            assert result_c.choices[0].message.content == 'C'
            assert result_c.usage.total_tokens == 4
            assert counts == {'A': 1, 'B': 1, 'C': 1}
            assert not sync.is_closed()
        finally:
            if a and not a.done(): a.cancel()
            if b and not b.done(): b.cancel()
            sync.close()
            server.close()
            await server.wait_closed()
            for task in list(handlers): task.cancel()
            await asyncio.gather(*list(handlers), return_exceptions=True)
    asyncio.run(run())


def test_standard_shared_transport_is_rebuilt_without_closing_template():
    from agent.auxiliary_async_codex import _attempt_client
    from agent.process_bootstrap import build_keepalive_http_client

    async def run():
        http = build_keepalive_http_client('http://127.0.0.1:1/v1')
        source = openai.OpenAI(api_key='test', base_url='http://127.0.0.1:1/v1', http_client=http)
        try:
            attempt = await _attempt_client(source)
            await attempt.close()
            assert not source.is_closed()
        finally:
            source.close()
    asyncio.run(run())


def test_attempt_preserves_callable_token_and_request_defaults():
    from agent.auxiliary_async_codex import _attempt_client

    async def run():
        source = openai.OpenAI(api_key=lambda: 'rotating-token', base_url='http://127.0.0.1:1/v1',
                               max_retries=2, default_headers={'x-route': 'original'},
                               default_query={'route': 'original'}, http_client=httpx.Client(trust_env=False))
        try:
            attempt = await _attempt_client(source)
            try:
                assert attempt.max_retries == 2
                assert attempt._custom_headers['x-route'] == 'original'
                assert attempt._custom_query['route'] == 'original'
                assert await attempt._api_key_provider() == 'rotating-token'
            finally:
                await attempt.close()
            assert not source.is_closed()
        finally:
            source.close()
    asyncio.run(run())
