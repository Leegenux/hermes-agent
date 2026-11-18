"""Attempt-owned native async Codex Responses transport."""
import asyncio
import copy
from types import SimpleNamespace


class _CapturedCancellationWatch:
    """Bridge a captured host cancel source to this loop's request task only."""

    def __init__(self, guard):
        self._guard = guard
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        self._handle = None
        self.host_cancelled = False
        self._closed = False
        if guard._protected_cancel_check is not None:
            self._check()

    def _check(self):
        if self._closed or self._task.done():
            return
        if self._guard.cancel_requested():
            self.host_cancelled = True
            self._task.cancel()
        else:
            self._handle = self._loop.call_later(0.02, self._check)

    def close(self):
        self._closed = True
        if self._handle is not None:
            self._handle.cancel()


def _async_event_hooks(http, base_url):
    """Translate only the known Codex compression hook; never discard user hooks."""
    from agent.codex_headers import codex_request_compression_hooks

    hooks = http.event_hooks
    if not any(hooks.values()):
        return {}
    expected = codex_request_compression_hooks(str(base_url))
    if not expected or hooks.get('response'):
        raise ValueError('unsupported Codex async event hooks')
    actual = hooks.get('request', [])
    known = expected['request']
    if len(actual) != len(known):
        raise ValueError('unsupported Codex async event hooks')
    for current, reference in zip(actual, known):
        current_cells = getattr(current, '__closure__', None) or ()
        reference_cells = reference.__closure__ or ()
        if (getattr(current, '__code__', None) is not reference.__code__
                or getattr(current, '__globals__', None) is not reference.__globals__
                or len(current_cells) != len(reference_cells)
                or any(a.cell_contents is not b.cell_contents
                       for a, b in zip(current_cells, reference_cells))):
            raise ValueError('unsupported Codex async event hooks')
    return codex_request_compression_hooks(str(base_url), async_mode=True)


async def _attempt_client(template):
    import httpcore
    import httpx
    import openai
    from agent.process_bootstrap import _shared_transport_cls
    from agent.usage_observer import AsyncTransport, SyncTransport

    http = template._client
    if type(http) is not httpx.Client or http.auth:
        raise ValueError('unsupported Codex async transport configuration')
    event_hooks = _async_event_hooks(http, template.base_url)
    mounts = {pattern.pattern: transport for pattern, transport in http._mounts.items()}
    if mounts and set(mounts) != {'http://', 'https://'}:
        raise ValueError('unsupported Codex async transport configuration')
    if not mounts and http.trust_env:
        raise ValueError('unsupported Codex async transport configuration')

    allocated_transports = []
    async_http = None

    def rebuild(transport):
        observed = type(transport) is SyncTransport
        if observed:
            transport = transport.inner
        if type(transport) is _shared_transport_cls():
            transport = transport._inner
        if type(transport) is not httpx.HTTPTransport or type(transport._pool) is not httpcore.ConnectionPool:
            raise ValueError('unsupported Codex async transport configuration')
        pool = transport._pool
        from agent.process_bootstrap import _HappyEyeballsSyncBackend
        if type(pool._network_backend) not in (httpcore.SyncBackend, _HappyEyeballsSyncBackend):
            raise ValueError('unsupported Codex async network backend')
        if pool._proxy is not None or pool._uds is not None or pool._local_address is not None or pool._socket_options is not None or pool._retries:
            raise ValueError('unsupported Codex async transport configuration')
        result = httpx.AsyncHTTPTransport(verify=pool._ssl_context, http1=pool._http1, http2=pool._http2,
            limits=httpx.Limits(max_connections=pool._max_connections, max_keepalive_connections=pool._max_keepalive_connections,
                                keepalive_expiry=pool._keepalive_expiry))
        allocated_transports.append(result)
        return AsyncTransport(result) if observed else result

    if template.workload_identity is not None or template._provider is not None:
        raise ValueError('unsupported Codex async authentication configuration')
    from agent.auxiliary_async_rebuild import async_api_key
    api_key = async_api_key(template)
    try:
        async_transport = rebuild(http._transport)
        async_mounts = {pattern: rebuild(transport) for pattern, transport in mounts.items()}
        # Preserve complete cookie metadata, but never share a mutable cookie jar.
        cookies = httpx.Cookies()
        for cookie in http.cookies.jar:
            cookies.jar.set_cookie(copy.deepcopy(cookie))
        async_http = httpx.AsyncClient(transport=async_transport, mounts=async_mounts, trust_env=False, headers=http.headers,
                                       cookies=cookies, params=http.params, timeout=http.timeout, follow_redirects=http.follow_redirects,
                                       max_redirects=http.max_redirects, event_hooks=event_hooks)
        return openai.AsyncOpenAI(api_key=api_key, base_url=str(template.base_url), timeout=template.timeout,
            max_retries=template.max_retries, organization=template.organization, project=template.project,
            default_headers=copy.deepcopy(template._custom_headers), default_query=copy.deepcopy(template._custom_query),
            http_client=async_http, _strict_response_validation=template._strict_response_validation)
    except BaseException:
        # Before a client takes ownership, close each successfully allocated
        # transport. Afterwards close only the client, which owns its mounts.
        resources = [async_http] if async_http is not None else list(reversed(allocated_transports))
        for resource in resources:
            try:
                await resource.aclose()
            except Exception:
                # Constructor failure remains the primary verdict; attempt all
                # resources even if one rollback operation itself fails.
                pass
        raise


class AsyncCodexCompletionsAdapter:
    def __init__(self, sync):
        self._sync = sync

    async def create(self, **kwargs):
        from agent.auxiliary_client import (_CodexStreamGuard, _notify_aux_provider_response,
                                            _notify_aux_timing_response)
        from agent.auxiliary_codex_response import _parse_codex_final_response
        from agent.codex_runtime import _CodexResponseAssembler, _codex_event_has_content
        from agent.sdk_transform_bypass import bypass_sdk_request_transform
        from agent.usage_observer import response_fields
        from hermes_cli.providers import is_actual_route

        template = self._sync._client
        if is_actual_route(getattr(template, '_hermes_aux_effective_provider', ''), str(template.base_url)):
            raise ValueError('Actual requests require Chat Completions; refusing to call /responses.')
        payload, model, timeout = self._sync._build_responses_kwargs(kwargs)
        aliases = payload.pop('_wire_aliases', None) or {}
        issuer = payload.pop('_issuer_kind', None)
        issuer_model = str(payload.get('model') or model)
        total = timeout if isinstance(timeout, (int, float)) and timeout > 0 else None
        guard = _CodexStreamGuard(template, total, kwargs.get('no_progress_timeout'))
        client = await _attempt_client(template)
        stream = None
        cancellation = _CapturedCancellationWatch(guard)
        try:
            async with asyncio.timeout_at(guard.effective_deadline()) as deadline:
                stream = await client.responses.create(**bypass_sdk_request_transform({**payload, 'stream': True}))
                if hasattr(stream, 'output'):
                    final = stream
                else:
                    assembler = _CodexResponseAssembler(model=issuer_model, on_text_delta=None, on_reasoning_delta=None,
                                                        on_commentary_message=None, on_first_delta=None)
                    async for event in stream:
                        if _codex_event_has_content(event):
                            guard.record_progress()
                            guard.saw_content.set()
                            _notify_aux_provider_response()
                            deadline.reschedule(guard.effective_deadline())
                        else:
                            _notify_aux_timing_response()
                        if assembler.feed(event):
                            break
                    final = assembler.result()
            text, calls, usage, reason = _parse_codex_final_response(final, issuer_kind=issuer, issuer_model=issuer_model)
            for call in calls or ():
                if call.function.name in aliases:
                    call.function.name = aliases[call.function.name]
            message = SimpleNamespace(role='assistant', content=''.join(text).strip() or None, tool_calls=calls or None)
            return SimpleNamespace(choices=[SimpleNamespace(index=0, message=message, finish_reason=reason)],
                                   model=model, usage=usage, **response_fields(final))
        except asyncio.CancelledError:
            if cancellation.host_cancelled:
                from agent.auxiliary_client import AuxiliaryExplicitCancellation
                raise AuxiliaryExplicitCancellation() from None
            raise
        except TimeoutError:
            # The retry ladder reuses an occupied cache slot. Retire this exact
            # template so its already-frozen route, rather than mutable current
            # SDK defaults, is used to rebuild the next attempt. Native attempts
            # own independent transports; closing this template cannot close B.
            from agent.auxiliary_client import _evict_cached_client_instance, _close_quietly
            if _evict_cached_client_instance(template):
                _close_quietly(template, 'native Codex timeout template close failed')
            raise TimeoutError(guard.timeout_message()) from None
        finally:
            import sys
            cancellation.close()
            primary = sys.exc_info()[1]
            loop = asyncio.get_running_loop()
            end = loop.time() + 1.0
            task = asyncio.current_task()
            cancel_count = task.cancelling() if task is not None else 0
            external_cancel = None

            async def close_owned():
                error = None
                try:
                    if stream is not None and hasattr(stream, 'close'):
                        # Reserve half the absolute budget for closing the client.
                        await asyncio.wait_for(stream.close(), max(0, (end - loop.time()) / 2))
                except BaseException as exc:
                    error = exc
                try:
                    await asyncio.wait_for(client.close(), max(0, end - loop.time()))
                except BaseException as exc:
                    if error is None:
                        error = exc
                return error

            cleanup = asyncio.create_task(close_owned())
            cleanup_error = None
            while True:
                try:
                    cleanup_error = await asyncio.wait_for(asyncio.shield(cleanup), max(0, end - loop.time()))
                    break
                except asyncio.CancelledError as exc:
                    if task is not None and task.cancelling() > cancel_count:
                        cancel_count = task.cancelling()
                        external_cancel = exc
                        continue
                    cleanup_error = exc
                    break
                except TimeoutError as exc:
                    cleanup.cancel()
                    cleanup_error = exc
                    break
            if external_cancel is not None:
                raise external_cancel
            if primary is None and cleanup_error is not None:
                raise cleanup_error
