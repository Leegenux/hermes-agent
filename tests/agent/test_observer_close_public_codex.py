"""Public Codex compression remains independent of observed main-agent close."""
import concurrent.futures
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openai
from agent import auxiliary_client as aux, process_bootstrap as pb
from agent.usage_observer import SyncTransport
from run_agent import AIAgent
from tests.agent.test_codex_public_entry_loopback_contract import endpoint, request


def test_public_compression_survives_observed_main_close(endpoint, monkeypatch):
    arrived = threading.Event()
    release = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args): pass
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'x'); self.wfile.flush()
            arrived.set(); release.wait(12)
            try: self.wfile.write(b'y'); self.wfile.flush()
            except OSError: pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    http = pb.build_keepalive_http_client(url)
    http._transport = SyncTransport(http._transport)
    http._mounts = {p: SyncTransport(t) if t else None for p, t in http._mounts.items()}
    sdk = openai.OpenAI(api_key='dummy', base_url=url, http_client=http, max_retries=0)
    ready = threading.Event(); gate = threading.Event()
    def main_request():
        try:
            with http.stream('GET', url + '/main') as response:
                ready.set(); assert gate.wait(10)
                return response.read()
        except Exception as exc:
            return type(exc).__name__
    obj = AIAgent.__new__(AIAgent); obj.client = sdk
    for name in ('shutdown_memory_provider', '_close_task_resources', '_close_active_children', '_close_codex_session', '_trim_process_memory', '_finalize_owned_session_row', '_close_request_clients'):
        setattr(obj, name, lambda *args, **kwargs: None)
    import agent.prepared_compaction as pc
    monkeypatch.setattr(pc, 'release_for_agent', lambda _: None)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            main = pool.submit(main_request)
            assert arrived.wait(8) and ready.wait(8)
            b = pool.submit(aux.call_llm, **request('B'))
            assert endpoint['b_seen'].wait(8)
            obj.close()
            gate.set(); release.set()
            assert main.result(timeout=8) == 'ReadError'
            endpoint['release_b'].set()
            result = b.result(timeout=10)
            assert result.choices[0].message.content == 'B'
            assert result.usage.total_tokens == 4
    finally:
        gate.set(); release.set(); endpoint['release_b'].set()
        sdk.close(); pb.close_shared_transports(); server.shutdown(); server.server_close(); thread.join(3)
