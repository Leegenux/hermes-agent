"""Scoped completion output must not expose streamed draft text."""
from types import SimpleNamespace


def test_completion_scope_holds_stream_callbacks():
    from gateway.completion_report_scope import completion_scope
    from gateway.run_turn_runner import TurnRunner
    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(mute_notification_reply=False)
    token = completion_scope.set(('trusted',))
    try:
        assert runner._setup_stream_consumer('telegram') == (None, None, None, False)
    finally:
        completion_scope.reset(token)


def test_completion_scope_never_starts_streaming_tts():
    from gateway.completion_report_scope import completion_scope
    from gateway.run_turn import GatewayTurnMixin
    runner = object.__new__(GatewayTurnMixin)
    token = completion_scope.set(('trusted',))
    try:
        runner._run_agent_start_streaming_tts(None, 'voice', None, [None])
    finally:
        completion_scope.reset(token)
