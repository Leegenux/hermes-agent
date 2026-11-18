"""Explicit host cancellation must stop native IO, not merely its relay waiter."""
import concurrent.futures
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import openai
import pytest

from agent.auxiliary_client import (AuxiliaryExplicitCancellation, CodexAuxiliaryClient,
                                    aux_interrupt_protection)


def test_protected_sync_cancel_closes_blocked_request_and_preserves_template():
    seen, eof, cancel = threading.Event(), threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            if b'blocked-A' in body:
                seen.set()
                self.connection.settimeout(6)
                try:
                    if self.connection.recv(1) == b'':
                        eof.set()
                except OSError:
                    pass
                return
            data = ('data: ' + json.dumps({'type': 'response.output_item.done', 'output_index': 0,
                    'item': {'id': 'B', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                             'content': [{'type': 'output_text', 'text': 'healthy', 'annotations': []}]}})
                    + '\n\ndata: ' + json.dumps({'type': 'response.completed', 'response':
                        {'id': 'B', 'status': 'completed', 'output': []}}) + '\n\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    sdk = openai.OpenAI(api_key='offline-only', base_url=f'http://127.0.0.1:{server.server_port}/v1',
                        max_retries=0, http_client=httpx.Client(trust_env=False))
    client = CodexAuxiliaryClient(sdk, 'test')

    def call_a():
        with aux_interrupt_protection(cancel_event=cancel):
            return client.chat.completions.create(model='test', timeout=20,
                messages=[{'role': 'user', 'content': 'blocked-A'}])

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
            a = worker.submit(call_a)
            assert seen.wait(10), 'request must be active before host cancellation'
            start = time.monotonic()
            cancel.set()
            with pytest.raises(AuxiliaryExplicitCancellation):
                a.result(timeout=2)
            assert time.monotonic() - start < 2
            assert eof.wait(1), 'server must observe connection EOF, not just a released waiter'
            healthy = client.chat.completions.create(model='test', timeout=10,
                messages=[{'role': 'user', 'content': 'healthy-B'}])
            assert healthy.choices[0].message.content == 'healthy'
            assert not sdk.is_closed()
    finally:
        cancel.set()
        sdk.close()
        server.shutdown()
        server.server_close()
        serving.join(timeout=2)
