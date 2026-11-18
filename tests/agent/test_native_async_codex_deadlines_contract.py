"""Real loopback checks for native async deadlines; not a live-provider test."""
import asyncio
import json
import time

import httpx
import openai
import pytest

from agent.auxiliary_client import AsyncCodexAuxiliaryClient, CodexAuxiliaryClient


@pytest.mark.parametrize('phase', ['preheaders', 'keepalive', 'substantive'])
def test_native_deadline_closes_request_without_waiting_for_read_timeout(monkeypatch, phase):
    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, '_aux_stream_total_ceiling', lambda _: 0.7)

    async def run():
        # Controlled transport-deadline check: preload AnyIO's backend before the
        # 250ms request budget. Cold-start behavior remains a separate open check.
        import anyio
        await anyio.sleep(0)
        import agent.auxiliary_async_codex as native
        trace = {}
        original_factory = native._attempt_client
        async def timed_factory(template):
            trace['factory_start'] = time.monotonic()
            try:
                return await original_factory(template)
            finally:
                trace['factory_end'] = time.monotonic()
        monkeypatch.setattr(native, '_attempt_client', timed_factory)
        seen, eof = asyncio.Event(), asyncio.Event()
        stamps = {}
        handlers = set()

        async def handle(reader, writer):
            handlers.add(asyncio.current_task())
            reader_task = None
            try:
                head = await reader.readuntil(b'\r\n\r\n')
                length = next(int(x.split(b':', 1)[1]) for x in head.split(b'\r\n')
                              if x.lower().startswith(b'content-length:'))
                body = await reader.readexactly(length)
                if b'contract-warmup' in body:
                    response = json.dumps({'id': 'warm', 'object': 'response',
                                           'created_at': 0, 'status': 'completed', 'output': []}).encode()
                    writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: '
                                 + str(len(response)).encode() + b'\r\n\r\n' + response)
                    await writer.drain()
                    return
                stamps['request'] = time.monotonic()
                seen.set()
                reader_task = asyncio.create_task(reader.read())
                if phase != 'preheaders':
                    writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n')
                    await writer.drain()
                    while not reader_task.done():
                        event = ({'type': 'response.output_text.delta', 'delta': 'x'}
                                 if phase == 'substantive' else {'type': 'response.created'})
                        writer.write(('data: ' + json.dumps(event) + '\n\n').encode())
                        await writer.drain()
                        await asyncio.sleep(0.04)
                await reader_task
                stamps['eof'] = time.monotonic()
                eof.set()
            finally:
                if reader_task and not reader_task.done():
                    reader_task.cancel()
                    await asyncio.gather(reader_task, return_exceptions=True)
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, OSError):
                    pass
                handlers.discard(asyncio.current_task())

        server = await asyncio.start_server(handle, '127.0.0.1', 0)
        port = server.sockets[0].getsockname()[1]
        source = openai.OpenAI(api_key='offline-only', base_url=f'http://127.0.0.1:{port}/v1',
                              max_retries=2, http_client=httpx.Client(trust_env=False))
        adapter = AsyncCodexAuxiliaryClient(CodexAuxiliaryClient(source, 'test'))
        task = None
        try:
            # Separate a real SDK warmup from the unchanged transport deadline.
            # This controls first-use serialization/import cost, not cold-start reliability.
            warm = await native._attempt_client(source)
            try:
                await asyncio.wait_for(warm.responses.create(model='test', input='contract-warmup'), 3)
            finally:
                await warm.close()
            trace.clear()
            task = asyncio.create_task(adapter.chat.completions.create(
                model='test', messages=[{'role': 'user', 'content': 'deadline probe'}],
                timeout=20, no_progress_timeout=0.25))
            await asyncio.wait_for(seen.wait(), 3)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), 3)
            assert task.done(), 'outer wait timeout must not masquerade as the adapter deadline'
            assert isinstance(task.exception(), TimeoutError)
            elapsed = time.monotonic() - stamps['request']
            assert elapsed < 1.5, (phase, elapsed)
            if phase == 'substantive':
                assert elapsed > 0.4, 'substantive events must extend the no-progress deadline'
            else:
                assert elapsed < 0.65, 'non-content must not extend the no-progress window'
            await asyncio.wait_for(eof.wait(), 1)
            assert not source.is_closed()
        finally:
            if task:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                trace['task_done'] = time.monotonic()
                trace['task_exception'] = repr(task.exception()) if not task.cancelled() else 'cancelled'
            print('DEADLINE_TRACE', phase, trace, stamps, flush=True)
            source.close()
            server.close()
            await server.wait_closed()
            remaining = list(handlers)
            for handler in remaining:
                handler.cancel()
            await asyncio.gather(*remaining, return_exceptions=True)
    asyncio.run(run())
