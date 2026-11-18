"""Ownership boundaries for transport socket discovery."""
from types import SimpleNamespace

import httpx

from agent import process_bootstrap as pb
from agent.agent_runtime_helpers import _iter_httpx_pools_with_owner
from agent.usage_observer import SyncTransport


def pools(default, *mounts):
    return list(_iter_httpx_pools_with_owner(SimpleNamespace(_transport=default, _mounts=dict(enumerate(mounts)))))


def test_distinct_views_and_repeated_view_in_mounts():
    inner = httpx.HTTPTransport()
    try:
        a, b = pb._shared_transport_cls()(inner), pb._shared_transport_cls()(inner)
        assert pools(SyncTransport(a), SyncTransport(a), SyncTransport(b)) == [(inner._pool, id(a)), (inner._pool, id(b))]
    finally:
        inner.close()


def test_direct_http_transport_retains_owned_pool():
    transport = httpx.HTTPTransport()
    try:
        assert pools(transport) == [(transport._pool, None)]
        assert pools(SyncTransport(transport)) == [(transport._pool, None)]
    finally:
        transport.close()


def test_unknown_wrapper_must_not_claim_shared_pool():
    inner = httpx.HTTPTransport()
    class Unknown(httpx.BaseTransport):
        def __init__(self, wrapped): self.inner = wrapped
        @property
        def _pool(self): return self.inner._pool
    try:
        view = pb._shared_transport_cls()(inner)
        assert pools(Unknown(view)) == []
        assert pools(SyncTransport(Unknown(view))) == []
    finally:
        inner.close()
