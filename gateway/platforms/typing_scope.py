"""Topic-local typing pause state.

Scope and compatibility helpers adapted from NousResearch/hermes-agent PR #136696,
head 21e6dc3c2ded95aa5620889d90083feb0eea1a7a (original patch archived with local receipt).
"""
import inspect
from gateway.config import Platform


def typing_scope(platform, chat_id, metadata=None):
    thread_id = (metadata or {}).get('thread_id') or (metadata or {}).get('message_thread_id')
    if platform == Platform.TELEGRAM and thread_id is not None:
        return (str(chat_id), str(thread_id))
    return str(chat_id)


def set_typing_paused(adapter, chat_id, *, paused, metadata=None):
    """Keep single-argument overrides compatible while carrying topic metadata where supported."""
    method = adapter.pause_typing_for_chat if paused else adapter.resume_typing_for_chat
    params = inspect.signature(method).parameters
    supports_metadata = 'metadata' in params or any(p.kind == p.VAR_KEYWORD for p in params.values())
    method(chat_id, **({'metadata': metadata} if supports_metadata else {}))


class TypingPauseMixin:
    def typing_is_paused(self, chat_id, metadata=None):
        paused = getattr(self, '_typing_paused', ())
        return str(chat_id) in paused or typing_scope(getattr(self, 'platform', None), chat_id, metadata) in paused

    def pause_typing_for_chat(self, chat_id: str, metadata=None) -> None:
        self._typing_paused.add(typing_scope(self.platform, chat_id, metadata))

    def resume_typing_for_chat(self, chat_id: str, metadata=None) -> None:
        self._typing_paused.discard(typing_scope(self.platform, chat_id, metadata))
