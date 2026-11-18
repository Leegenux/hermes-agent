"""Attempt-scoped auxiliary retry client identity and transport lifecycle."""
from typing import Any, Optional
import logging

logger = logging.getLogger("agent.auxiliary_client")

def freeze_codex_retry_client(client: Any):
    """Capture a Codex SDK client and Hermes shared-view transport before its first request.

    Unknown transports fail closed: reconstructing them from current profile or proxy state
    could send this request with another route's identity.
    """
    import copy
    import httpx
    import openai
    from agent.auxiliary_client import CodexAuxiliaryClient, AsyncCodexAuxiliaryClient
    from agent.process_bootstrap import _shared_transport_cls
    from agent.usage_observer import SyncTransport

    if not isinstance(client, (CodexAuxiliaryClient, AsyncCodexAuxiliaryClient)):
        return None
    leaf = client._real_client
    if not isinstance(leaf, openai.OpenAI) or type(leaf._client) is not httpx.Client:
        return None
    http = leaf._client
    shared_type = _shared_transport_cls()
    views = {}

    def capture(transport):
        if transport is None:
            return None
        if id(transport) in views:
            return views[id(transport)]
        observed = type(transport) is SyncTransport
        view = transport.inner if observed else transport
        if type(view) is not shared_type:
            raise ValueError("unsupported auxiliary retry transport")
        descriptor = (view._inner, observed)
        views[id(transport)] = descriptor
        return descriptor

    try:
        default = capture(http._transport)
        mounts = [(pattern.pattern, capture(transport)) for pattern, transport in http._mounts.items()]
        # OpenAI.copy without an explicit HTTP client shares the old client. This snapshot
        # retains the original SDK options, but is never used to make a request.
        frozen_sdk = leaf.copy(set_default_headers=copy.deepcopy(leaf._custom_headers),
                               set_default_query=copy.deepcopy(leaf._custom_query))
        cookies = httpx.Cookies()
        for cookie in http.cookies.jar:
            cookies.jar.set_cookie(copy.copy(cookie))
        config = dict(auth=http.auth, params=http.params, headers=http.headers.copy(),
                      cookies=cookies,
                      timeout=http.timeout, follow_redirects=http.follow_redirects,
                      max_redirects=http.max_redirects,
                      event_hooks={k: list(v) for k, v in http.event_hooks.items()},
                      default_encoding=http._default_encoding)
        trust_env = http.trust_env
        model = client.chat.completions._sync._model if isinstance(client, AsyncCodexAuxiliaryClient) else client.chat.completions._model
    except Exception:
        return None

    def rebuild():
        made = {}
        def view(desc):
            if desc is None:
                return None
            ident = id(desc)
            if ident not in made:
                inner, observed = desc
                fresh = shared_type(inner)
                made[ident] = SyncTransport(fresh) if observed else fresh
            return made[ident]
        fresh_http = httpx.Client(**config, transport=view(default),
                                 mounts={pattern: view(desc) for pattern, desc in mounts}, trust_env=False)
        fresh_http._trust_env = trust_env
        try:
            fresh_leaf = frozen_sdk.copy(http_client=fresh_http)
        except Exception:
            fresh_http.close()
            raise
        wrapped = CodexAuxiliaryClient(fresh_leaf, model)
        return AsyncCodexAuxiliaryClient(wrapped) if isinstance(client, AsyncCodexAuxiliaryClient) else wrapped
    return rebuild


def retry_cache_route(req: Any, task: Optional[str], runtime: dict, async_mode: bool):
    """Capture only the primary cache entry, never a vision or rescued client."""
    from agent.auxiliary_client import _client_cache_key, _client_cache_lock, _client_cache
    if task == "vision":
        return None
    args = dict(async_mode=async_mode, base_url=req.resolved_base_url, api_key=req.resolved_api_key,
                api_mode=req.resolved_api_mode, main_runtime=runtime, task=task)
    key = _client_cache_key(req.resolved_provider, model=req.resolved_model, **args)
    with _client_cache_lock:
        entry = _client_cache.get(key)
        if entry is None or entry[0] is not req.client:
            return None
    return key, args, freeze_codex_retry_client(req.client)


def retry_route_client(req: Any, route: Any) -> Any:
    from agent.auxiliary_client import (
        AsyncCodexAuxiliaryClient, _client_cache, _client_cache_lock, _current_event_loop, _close_cached_client,
    )
    if route is None:
        return req.client
    key, args, rebuild = route
    # Read only the captured slot. Re-resolving after eviction can silently bind a new
    # profile, credential pool or endpoint to this request, including under ABA races.
    with _client_cache_lock:
        entry = _client_cache.get(key)
        if entry is not None:
            client, _, loop = entry
            if not args["async_mode"] or (loop is _current_event_loop() and loop is not None and not loop.is_closed()):
                return client
            return req.client
    if rebuild is None:
        return req.client
    try:
        fresh = rebuild()
    except Exception:
        logger.debug("Could not rebuild frozen Codex retry client", exc_info=True)
        return req.client
    with _client_cache_lock:
        entry = _client_cache.get(key)
        if entry is None:
            _client_cache[key] = (fresh, req.final_model, _current_event_loop() if args["async_mode"] else None)
            return fresh
    if isinstance(fresh, AsyncCodexAuxiliaryClient):
        fresh._real_client.close()
    else:
        _close_cached_client(fresh, close_async=args["async_mode"])
    client, _, loop = entry
    if args["async_mode"] and (loop is not _current_event_loop() or loop is None or loop.is_closed()):
        return req.client
    return client


