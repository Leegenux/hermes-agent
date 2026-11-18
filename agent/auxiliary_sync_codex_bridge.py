"""Short-lived synchronous bridge to attempt-owned native async Codex IO."""
import asyncio
import contextvars


def create_sync_codex_attempt(adapter, kwargs):
    import openai
    if not isinstance(adapter._client, openai.OpenAI):
        return adapter._create_attempt(**kwargs)

    from agent.auxiliary_async_codex import AsyncCodexCompletionsAdapter
    from agent import auxiliary_client as aux

    async def call():
        return await AsyncCodexCompletionsAdapter(adapter).create(**kwargs)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(call())

    # Context variables propagate separately from the request's thread-local hooks.
    locals_ = (aux._aux_interrupt_protection, aux._aux_stream_deadline,
               aux._aux_progress, aux._aux_dispatch, aux._aux_provider_response)
    snapshots = [local.__dict__.copy() for local in locals_]
    context = contextvars.copy_context()

    def run():
        for local, values in zip(locals_, snapshots):
            local.__dict__.update(values)
        return context.run(asyncio.run, call())

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(run).result()
