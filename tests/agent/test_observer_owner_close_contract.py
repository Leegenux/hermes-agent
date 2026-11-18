import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
import openai
from agent import process_bootstrap as pb
from agent.agent_runtime_helpers import _iter_pool_sockets
from agent.usage_observer import SyncTransport
from run_agent import AIAgent

@pytest.mark.parametrize('wrapped', [False, True])
def test_agent_close_cross_owner(wrapped, monkeypatch):
    for n in ('HTTPS_PROXY','HTTP_PROXY','ALL_PROXY','https_proxy','http_proxy','all_proxy','NO_PROXY','no_proxy'):
        monkeypatch.delenv(n, raising=False)
    pb.close_shared_transports()
    arrived={k:threading.Event() for k in 'ab'}; ready={k:threading.Event() for k in 'ab'}
    release=threading.Event(); gate=threading.Event(); results={}; events=[]
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def do_GET(self):
            self.send_response(200); self.send_header('Content-Length','2'); self.end_headers()
            self.wfile.write(b'x'); self.wfile.flush(); arrived[self.path[1:]].set(); release.wait(10)
            try: self.wfile.write(b'y'); self.wfile.flush()
            except OSError: pass
        def log_message(self,*a): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler); server.daemon_threads=True
    threading.Thread(target=server.serve_forever,daemon=True).start()
    url=f'http://127.0.0.1:{server.server_address[1]}'
    a=pb.build_keepalive_http_client(url); b=pb.build_keepalive_http_client(url)
    view=lambda c: next(t for p,t in c._mounts.items() if p.pattern=='http://')
    va,vb=view(a),view(b); assert va is not vb and va._pool is vb._pool
    # Apply the real observer transport exactly around each transport slot.
    if wrapped:
        for c in (a,b):
            c._transport=SyncTransport(c._transport)
            c._mounts={p:SyncTransport(t) if t else None for p,t in c._mounts.items()}
    sdk_a=openai.OpenAI(api_key='dummy',base_url=url,http_client=a,max_retries=0)
    sdk_b=openai.OpenAI(api_key='dummy',base_url=url,http_client=b,max_retries=0)
    if wrapped:
        import agent.usage_observer as observer
        monkeypatch.setattr(observer, '_enabled', lambda: True)
        monkeypatch.setattr(observer, '_fire', lambda event, **payload: events.append((event, payload)))
    def work(k,c):
        try:
            with c.stream('GET',url+'/'+k) as r:
                ready[k].set(); assert gate.wait(10); results[k]=r.read()
        except Exception as e: results[k]=type(e).__name__+': '+str(e)
    ts=[threading.Thread(target=work,args=(k,c)) for k,c in [('a',a),('b',b)]]
    try:
        for t in ts:t.start()
        assert all(x.wait(8) for x in arrived.values()) and all(x.wait(8) for x in ready.values())
        # Reveal real request owners/sockets independent of sweep's wrapper-dependent filter.
        requests=list(va._pool._requests)
        from agent.process_bootstrap import HERMES_TRANSPORT_OWNER_EXT
        owners=[r.request.extensions.get(HERMES_TRANSPORT_OWNER_EXT) for r in requests]
        from agent.agent_runtime_helpers import _connection_candidates,_socket_from_candidate
        socks=[]
        for r in requests:
            found=[_socket_from_candidate(x) for x in _connection_candidates(r.connection)]
            s=next((x for x in found if x is not None),None)
            socks.append((id(s),s.fileno()) if s else None)
        print('IDENTITY',{'wrapped':wrapped,'sdk_a':id(sdk_a),'sdk_b':id(sdk_b),'http_a':id(a),'http_b':id(b),'view_a':id(va),'view_b':id(vb),'pool':id(va._pool),'owners':owners,'sockets':socks,'a_sweep_sockets':[id(s) for s in _iter_pool_sockets(sdk_a)]},flush=True)
        assert set(owners)=={id(va),id(vb)} and len(set(socks))==2 and all(socks)
        socket_by_owner = dict(zip(owners, socks))
        assert [id(s) for s in _iter_pool_sockets(sdk_a)] == [socket_by_owner[id(va)][0]]
        obj=AIAgent.__new__(AIAgent); obj.client=sdk_a
        for name in ('shutdown_memory_provider','_close_task_resources','_close_active_children','_close_codex_session','_trim_process_memory','_finalize_owned_session_row','_close_request_clients'):
            setattr(obj,name,lambda *a,**kw:None)
        import agent.prepared_compaction as pc
        monkeypatch.setattr(pc,'release_for_agent',lambda obj:None)
        obj.close()
        b_request = next(r for r in va._pool._requests if r.request.extensions.get(HERMES_TRANSPORT_OWNER_EXT) == id(vb))
        b_socket = next(_socket_from_candidate(x) for x in _connection_candidates(b_request.connection) if _socket_from_candidate(x) is not None)
        assert (id(b_socket), b_socket.fileno()) == socket_by_owner[id(vb)]
        gate.set(); release.set()
        for t in ts:t.join(8)
        print('RESULT',wrapped,results,flush=True)
        assert not any(t.is_alive() for t in ts)
        assert isinstance(results['a'], str) and 'ReadError' in results['a']
        assert results['b'] == b'xy'
        if wrapped:
            assert sum(event == 'http_attempt_start' for event, _ in events) == 2
            assert sum(event == 'http_attempt_headers' for event, _ in events) == 2
            assert any(event == 'http_attempt_end' and payload['transport_outcome'] == 'body_exhausted' for event, payload in events)
    finally:
        gate.set(); release.set()
        for t in ts:t.join(8)
        sdk_a.close(); sdk_b.close(); pb.close_shared_transports(); server.shutdown(); server.server_close()
