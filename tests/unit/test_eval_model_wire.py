"""Owned model envelopes round-trip through the pinned native API parsers."""

import json

import pytest

pytest.importorskip("verifiers.v1.dialects")
from verifiers.v1.dialects import AnthropicDialect, ChatDialect, ResponsesDialect
from verifiers.v1.types import Response

from reliquary.eval.model_wire import (
    completion_events,
    native_completion,
    restore_local_thinking,
)


@pytest.mark.parametrize("dialect", [ChatDialect(), AnthropicDialect(), ResponsesDialect()])
@pytest.mark.parametrize("finish", ["stop", "length", "tool_calls"])
def test_generated_native_and_streamed_envelopes_preserve_text_tools_usage(dialect, finish):
    message = {"content": "answer", "reasoning_content": "reason"}
    if finish == "tool_calls":
        message["tool_calls"] = [{"id": "call_1", "name": "signed_bash", "arguments": '{"command":"true"}'}]
    response = Response.model_validate({"id": "evaluation-turn-1", "created": 0, "model": "fixture/model@" + "a" * 40,
        "message": message, "finish_reason": finish, "usage": {"prompt_tokens": 13, "completion_tokens": 7}})
    raw = native_completion(response, dialect)
    native = dialect.parse_response(dialect.validate_response(raw))
    parser = dialect.stream_parser()
    events = list(completion_events(raw, dialect))
    for event in events:
        parser.feed(event)
    streamed = parser.finish()
    for actual in (native, streamed):
        assert actual.model == response.model and actual.message.content == "answer"
        assert actual.message.reasoning_content == "reason"
        assert actual.finish_reason == finish
        assert actual.usage.input_tokens == 13 and actual.usage.completion_tokens == 7
        assert actual.usage.reasoning_tokens is None
        if finish == "tool_calls":
            assert actual.message.tool_calls[0].name == "signed_bash"
            assert json.loads(actual.message.tool_calls[0].arguments) == {"command": "true"}
        else:
            assert not actual.message.tool_calls
    assert any(dialect.is_terminal_event(event) for event in events)


def test_local_thinking_restores_full_text_without_rewriting_input():
    from verifiers.v1.clients.train import message_to_wire

    body = {"model": "fixture/model", "max_tokens": 64, "messages": [
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "complete thought", "signature": ""},
            {"type": "tool_use", "id": "call_1", "name": "signed_bash", "input": {"command": "true"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "signed output"}]}]}
    messages = [message_to_wire(message) for message in AnthropicDialect().parse_request(body).messages]
    restored = restore_local_thinking(messages, AnthropicDialect())
    assert restored[0]["reasoning_content"] == "complete thought"
    assert "reasoning_details" not in restored[0]
    assert messages[0]["reasoning_details"][0]["thinking"] == "complete thought"
    assert restored[0]["tool_calls"] == messages[0]["tool_calls"] and restored[1] == messages[1]


@pytest.mark.parametrize("state", [
    [], [{"type": "thinking", "thinking": "thought", "signature": "provider-signature"}],
    [{"type": "thinking", "thinking": "thought", "signature": "", "extra": True}],
    [{"type": "redacted_thinking", "data": "opaque"}],
    [{"type": "thinking", "thinking": "thought"}],
    [{"type": "thinking", "thinking": "", "signature": ""}],
    [{"type": "thinking", "thinking": "one", "signature": ""},
     {"type": "thinking", "thinking": "two", "signature": ""}],
])
def test_local_thinking_refuses_unknown_or_unemitted_provider_state(state):
    with pytest.raises(ValueError, match="unsupported local thinking state"):
        restore_local_thinking([{"role": "assistant", "reasoning_details": state}], AnthropicDialect())
