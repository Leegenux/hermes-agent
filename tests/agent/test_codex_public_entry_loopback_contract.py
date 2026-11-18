"""Public auxiliary entry points, real SDK/Responses parser, local-only provider."""
import asyncio
import concurrent.futures
import contextvars
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import hermes_yaml as yaml

from agent import auxiliary_client as aux


@pytest.fixture
def endpoint(monkeypatch, tmp_path):
    aux.shutdown_cached_clients()
    aux._reset_aux_unhealthy_cache()
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('NO_PROXY', '*')
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'auxiliary': {
        'compression': {'provider': 'openai-codex', 'model': 'public-model'},
        'transient_retries': 0,
    }}), encoding='utf-8')
    original_connect = socket.socket.connect
    def local_only(sock, address):
        if not isinstance(address, tuple) or address[0] not in ('127.0.0.1', '::1'):
            raise AssertionError('non-loopback connection refused by test')
        return original_connect(sock, address)
    monkeypatch.setattr(socket.socket, 'connect', local_only)
    state = {'requests': [], 'seen': threading.Event(), 'eof': threading.Event(),
             'b_seen': threading.Event(), 'release_b': threading.Event(), 'start': time.monotonic()}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            label = next(x for x in ('warm', 'A', 'B', 'C') if 'probe-' + x in json.dumps(payload))
            state['requests'].append((label, self.path, payload['model'], time.monotonic() - state['start']))
            if label == 'A':
                state['seen'].set()
                self.connection.settimeout(15)
                try:
                    if self.connection.recv(1) == b'': state['eof'].set()
                except OSError: pass
                return
            if label == 'B':
                state['b_seen'].set()
                state['release_b'].wait(12)
            events = [
                {'type':'response.output_item.done','output_index':0,'item':{'id':'msg','type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':label,'annotations':[]}]}},
                {'type':'response.completed','response':{'id':label,'status':'completed','output':[], 'usage':{'input_tokens':3,'output_tokens':1,'total_tokens':4}}},
            ]
            data = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            try: self.wfile.write(data)
            except OSError: pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(aux, '_resolve_codex_credential_and_base',
                        lambda model=None: ('synthetic-loopback-token', f'http://127.0.0.1:{server.server_port}/v1'))
    try:
        yield state
    finally:
        state['release_b'].set()
        aux.shutdown_cached_clients()
        server.shutdown()
        server.server_close()
        thread.join(3)
        print('PUBLIC_LOOPBACK_TRACE', state['requests'], flush=True)


def request(label, timeout=20):
    return dict(task='compression', messages=[{'role':'user','content':'probe-' + label}], timeout=timeout)


def cached(async_mode):
    matches = [(k, v) for k, v in aux._client_cache.items() if k[1] == 'openai-codex' and k[2] is async_mode]
    assert len(matches) == 1
    key, value = matches[0]
    assert key[10] == 'public-model'
    return key, value[0]


def test_public_protected_cancel_preserves_concurrent_and_cached_client(endpoint):
    state = endpoint
    assert aux.call_llm(**request('warm')).choices[0].message.content == 'warm'
    key, client = cached(False)
    cancel = threading.Event()
    def a_call():
        with aux.aux_interrupt_protection(cancel_event=cancel):
            return aux.call_llm(**request('A'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(a_call)
        assert state['seen'].wait(8)
        b = pool.submit(aux.call_llm, **request('B'))
        assert state['b_seen'].wait(8)
        cancel.set()
        with pytest.raises(aux.AuxiliaryExplicitCancellation): a.result(timeout=4)
        assert state['eof'].wait(3)
        state['release_b'].set()
        assert b.result(timeout=5).choices[0].message.content == 'B'
        c = aux.call_llm(**request('C'))
        assert c.choices[0].message.content == 'C'
        assert c.usage.total_tokens == 4
        assert cached(False) == (key, client)
    assert [r[0] for r in state['requests']] == ['warm', 'A', 'B', 'C']
    assert all(r[1:3] == ('/v1/responses', 'public-model') for r in state['requests'])


def test_public_async_cache_and_parser(endpoint):
    async def run():
        a = await aux.async_call_llm(**request('warm'))
        key, client = cached(True)
        c = await aux.async_call_llm(**request('C'))
        assert a.choices[0].message.content == 'warm'
        assert c.choices[0].message.content == 'C'
        assert c.usage.total_tokens == 4
        assert cached(True) == (key, client)
    asyncio.run(run())
    assert [r[0] for r in endpoint['requests']] == ['warm', 'C']
    assert all(r[1:3] == ('/v1/responses', 'public-model') for r in endpoint['requests'])


def test_public_sync_timeout_then_cached_healthy_call(endpoint):
    state = endpoint
    assert aux.call_llm(**request('warm')).choices[0].message.content == 'warm'
    key, client = cached(False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(aux.call_llm, **request('A', timeout=4))
        assert state['seen'].wait(8)
        b = pool.submit(aux.call_llm, **request('B'))
        assert state['b_seen'].wait(8)
        with pytest.raises(Exception) as error:
            a.result(timeout=12)
        assert not isinstance(error.value, concurrent.futures.TimeoutError) or a.done()
        assert 'timeout' in type(error.value).__name__.lower() or 'timed out' in str(error.value).lower()
        assert state['eof'].wait(3)
        state['release_b'].set()
        assert b.result(timeout=5).choices[0].message.content == 'B'
        assert aux.call_llm(**request('C')).choices[0].message.content == 'C'
        key_after, client_after = cached(False)
        assert key_after == key and client_after is not client
        print('TIMEOUT_CACHE_REBUILT', type(error.value).__name__, flush=True)
    assert [r[0] for r in state['requests']] == ['warm', 'A', 'B', 'C']


def test_public_sync_from_running_loop_preserves_host_deadline_and_context(endpoint):
    marker = contextvars.ContextVar('public_test_marker', default='missing')
    calls = []
    async def run():
        assert aux.call_llm(**request('warm')).choices[0].message.content == 'warm'
        owner = threading.get_ident()
        token = marker.set('caller-context')
        deadline = time.monotonic() + 4
        def host_deadline():
            calls.append((threading.get_ident(), marker.get()))
            return deadline
        try:
            with aux.aux_stream_deadline(host_deadline):
                with pytest.raises(Exception):
                    aux.call_llm(**request('A'))
        finally:
            marker.reset(token)
        assert endpoint['seen'].is_set()
        assert endpoint['eof'].wait(3)
        assert any(thread != owner and value == 'caller-context' for thread, value in calls), calls
        assert aux.call_llm(**request('C')).choices[0].message.content == 'C'
    asyncio.run(run())
    assert [r[0] for r in endpoint['requests']] == ['warm', 'A', 'C']
