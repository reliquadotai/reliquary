"""Owned model envelopes round-trip through the pinned native API parsers."""

import json

import pytest

pytest.importorskip("verifiers.v1.dialects")
from verifiers.v1.dialects import AnthropicDialect, ChatDialect, ResponsesDialect
from verifiers.v1.types import Response

from reliquary.eval.model_wire import completion_events, native_completion


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
