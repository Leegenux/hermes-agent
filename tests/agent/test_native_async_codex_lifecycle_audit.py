"""Focused native adapter lifecycle fault-injection contracts."""
import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from agent.auxiliary_client import AsyncCodexAuxiliaryClient, CodexAuxiliaryClient


def test_stream_close_failure_does_not_mask_iteration_error(monkeypatch):
    import agent.auxiliary_async_codex as native

    class BrokenStream:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise ValueError('primary iteration error')
        async def close(self):
            raise RuntimeError('secondary stream close error')

    class Attempt:
        def __init__(self):
            self.responses = self
            self.closed = False
        async def create(self, **kwargs):
            return BrokenStream()
        async def close(self):
            self.closed = True

    async def run():
        attempt = Attempt()
        async def make_attempt(source):
            return attempt
        monkeypatch.setattr(native, '_attempt_client', make_attempt)
        source = openai.OpenAI(api_key='offline', base_url='http://127.0.0.1:1/v1',
                               http_client=httpx.Client(trust_env=False))
        try:
            adapter = AsyncCodexAuxiliaryClient(CodexAuxiliaryClient(source, 'test'))
            with pytest.raises(ValueError, match='primary iteration error'):
                await adapter.chat.completions.create(model='test', messages=[{'role': 'user', 'content': 'test'}], timeout=10)
            assert attempt.closed
            assert not source.is_closed()
        finally:
            source.close()
    asyncio.run(run())


@pytest.mark.parametrize('retries', [0, 2])
def test_real_sdk_503_obeys_template_retry_count(retries):
    async def run():
        count = 0
        async def handle(reader, writer):
            nonlocal count
            try:
                headers = await reader.readuntil(b'\r\n\r\n')
                length = next(int(x.split(b':', 1)[1]) for x in headers.split(b'\r\n')
                              if x.lower().startswith(b'content-length:'))
                await reader.readexactly(length)
                count += 1
                writer.write(b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 2\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{}')
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
        server = await asyncio.start_server(handle, '127.0.0.1', 0)
        source = openai.OpenAI(api_key='offline', base_url=f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1',
                               max_retries=retries, http_client=httpx.Client(trust_env=False))
        try:
            adapter = AsyncCodexAuxiliaryClient(CodexAuxiliaryClient(source, 'test'))
            with pytest.raises(openai.InternalServerError):
                await adapter.chat.completions.create(model='test', messages=[{'role': 'user', 'content': 'retry'}], timeout=20)
            assert count == retries + 1
            assert not source.is_closed()
        finally:
            source.close()
            server.close()
            await server.wait_closed()
    asyncio.run(run())
