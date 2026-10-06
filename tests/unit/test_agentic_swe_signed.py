"""What a signed-sandbox job renders: the job's tools, verifiers' system prompt for
them, the env's sandbox prompt with no network notice, and the one pinned parser."""

import asyncio
import os
from types import SimpleNamespace

import pytest

from reliquary.corpus.job import parse_job
from reliquary.environment import agentic_swe
from reliquary.environment.agentic_swe import (
    BASH_HARNESS_TOOLS, BASH_SENTENCE, BASH_SYSTEM_PROMPT, EDIT_SENTENCE, SignedSweSource,
    harness_system_prompt, harness_tools, pinned_tool_calls,
)
from tests.unit.test_corpus_job_episode import _manifest
from tests.unit.test_corpus_job_signed_sandbox import ENV_PACKAGE, signed_episode

TOKENIZER = os.environ.get("RELIQUARY_QWEN38_TOKENIZER")


def test_the_system_prompt_follows_the_tools():
    assert harness_system_prompt(("bash",)) == BASH_SENTENCE
    assert harness_system_prompt(("bash", "edit")) == BASH_SYSTEM_PROMPT == \
        BASH_SENTENCE + " " + EDIT_SENTENCE
    assert harness_tools(("bash",)) == BASH_HARNESS_TOOLS[:1]
    assert harness_tools(("bash", "edit")) == BASH_HARNESS_TOOLS
    with pytest.raises(ValueError):
        harness_tools(("edit",))


def test_the_sentences_are_verifiers_own():
    harness = pytest.importorskip("verifiers.v1.harnesses.bash.harness")
    assert BASH_SENTENCE == harness.BASH_SYSTEM_PROMPT
    assert EDIT_SENTENCE == harness.EDIT_SYSTEM_PROMPT


def test_the_pinned_parser_is_verifiers_filter():
    train = pytest.importorskip("verifiers.v1.clients.train")
    from renderers.base import ParsedToolCall, ToolCallParseStatus as S

    calls = [
        ParsedToolCall(raw="", name="bash", arguments={"command": "ls"}, token_span=(0, 1), status=S.OK),
        ParsedToolCall(raw="", token_span=(1, 2), status=S.MALFORMED_STRUCTURE),
        ParsedToolCall(raw="", name="bash", arguments={"command": "é"}, token_span=(2, 3),
                       status=S.INVALID_JSON),
        ParsedToolCall(raw="", name="python", arguments={}, token_span=(3, 4), status=S.UNKNOWN_TOOL),
        ParsedToolCall(raw="", name="edit", arguments='{"path": 1}', token_span=(4, 5), status=S.OK),
        ParsedToolCall(raw="", name="", arguments={}, token_span=(5, 6), status=S.MISSING_NAME),
    ]
    response = train.response_from_generate({"tool_calls": calls}, "model")
    expected = [(call.name, call.arguments) for call in response.message.tool_calls]
    assert pinned_tool_calls(calls, S.UNKNOWN_TOOL) == expected
    assert [name for name, _ in expected] == ["bash", "bash", "edit"]


class _RecordingRouter:
    """`ToolRouter.answer`'s seat: records the calls the bridge hands the router."""

    def __init__(self):
        self.seen = []

    async def answer(self, turn, calls, call_id):
        self.seen.append([(name, arguments) for _, name, arguments in calls])
        return "ok"


def _router_input(parsed_calls) -> list[tuple[str, str]]:
    """The miner's real path from the inference server's parsed calls to the router:
    verifiers' train client (`response_from_generate`, then `serialize_completion` for
    the bash program's SDK), the program's next request (`model_dump` of the parsed
    message, one tool message per call) parsed by the chat dialect, then the bridge's
    `route_request`. The trace path (the Response's own message) must agree."""
    train = pytest.importorskip("verifiers.v1.clients.train")
    bridge = pytest.importorskip("reliquary_sandbox_verifiers.task")
    from openai.types.chat import ChatCompletion
    from verifiers.v1.dialects.chat import ChatDialect
    from verifiers.v1.types import Request, ToolMessage, UserMessage

    response = train.response_from_generate({"tool_calls": parsed_calls}, "model")
    calls = response.message.tool_calls or []
    trace = Request(messages=[UserMessage(content="x"), response.message,
                              *[ToolMessage(tool_call_id=c.id, content="", name=c.name)
                                for c in calls]])
    raw = train.serialize_completion(response, "model")
    message = ChatCompletion.model_validate(raw).choices[0].message.model_dump(exclude_none=True)
    wire = ChatDialect().parse_request({"messages": [
        {"role": "user", "content": "x"}, message,
        *[{"role": "tool", "tool_call_id": c["id"], "content": "", "name": c["function"]["name"]}
          for c in message.get("tool_calls") or []]]})
    seen = []
    for request in (trace, wire):
        router = _RecordingRouter()
        asyncio.run(bridge.route_request(router, request))
        seen.append(router.seen[0] if router.seen else [])
    assert seen[0] == seen[1]
    return seen[0]


def test_the_router_receives_what_the_pinned_parser_returns():
    from renderers.base import ParsedToolCall, ToolCallParseStatus as S

    calls = [
        ParsedToolCall(raw="", name="bash", arguments={"command": "ls"}, token_span=(0, 1), status=S.OK),
        ParsedToolCall(raw="", token_span=(1, 2), status=S.UNCLOSED_BLOCK),
        ParsedToolCall(raw="", name="bash", arguments='{"command": ', token_span=(2, 3),
                       status=S.INVALID_JSON),
        ParsedToolCall(raw="", name="python", arguments={}, token_span=(3, 4), status=S.UNKNOWN_TOOL),
        ParsedToolCall(raw="", name="edit", arguments={"path": 1, "old_str": [1], "new_str": None},
                       token_span=(4, 5), status=S.OK),
        ParsedToolCall(raw="", name="", arguments={}, token_span=(5, 6), status=S.MISSING_NAME),
        ParsedToolCall(raw="", name="bash", arguments=None, token_span=(6, 7), status=S.OK),
    ]
    expected = pinned_tool_calls(calls, S.UNKNOWN_TOOL)
    assert _router_input(calls) == expected
    assert [name for name, _ in expected] == ["bash", "bash", "edit", "bash"]
    assert expected[-1] == ("bash", "{}")


def _call(function, params):
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in params)
    return f"<tool_call>\n<function={function}>\n{body}</function>\n</tool_call>"


# Completions whose calls a real model can emit: valid, a tool the job does not offer,
# no name, a name outside the harness, a non-string (coerced) argument, an unclosed and
# a malformed block, and several calls in one turn.
_COMPLETIONS = {
    "bash": _call("bash", [("command", "ls")]),
    "edit": _call("edit", [("path", "a.py"), ("old_str", "x"), ("new_str", "y")]),
    "no name": _call("", [("command", "ls")]),
    "unknown tool": _call("python", [("code", "1")]),
    "non-string arguments": _call("edit", [("path", "1"), ("old_str", "[1, 2]"), ("new_str", "null")]),
    "extra parameter": _call("bash", [("command", "ls"), ("timeout", "5")]),
    "unclosed": "<tool_call>\n<function=bash>\n<parameter=command>\nls\n",
    "json style": '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>',
    "several": (_call("bash", [("command", "ls")]) + "\n" + _call("", [])
                + "\n" + _call("edit", [("path", "p"), ("old_str", "1"), ("new_str", "2")])),
}


@pytest.mark.skipif(not TOKENIZER, reason="set RELIQUARY_QWEN38_TOKENIZER")
@pytest.mark.parametrize("tools", [("bash",), ("bash", "edit")])
def test_the_validator_parses_real_completions_as_the_miner_routes_them(tools):
    """The validator's `QwenTurnRenderer.tool_calls` against the miner's path for the
    same tokens: the train client parses with the program's tools as verifiers' chat
    dialect carries them to the wire, then the bridge routes."""
    train = pytest.importorskip("verifiers.v1.clients.train")
    program = pytest.importorskip("verifiers.v1.harnesses.bash.program")
    from verifiers.v1.dialects.chat import parse_tools

    validator = agentic_swe.load_turn_renderer(TOKENIZER, tools=tools)
    offered = [program.BASH_TOOL] + ([program.EDIT_TOOL] if "edit" in tools else [])
    wire_tools = [train.tool_to_wire(tool) for tool in parse_tools(offered)]
    miner = validator._r
    for case, text in _COMPLETIONS.items():
        ids = validator._tokenizer.encode("<think>\nx\n</think>\n\n" + text + "<|im_end|>",
                                          add_special_tokens=False)
        parsed = miner.parse_response(ids, tools=wire_tools)
        assert validator.tool_calls(ids) == _router_input(parsed.tool_calls), (tools, case)


class _StubRenderer:
    def __init__(self):
        self._tokenizer = SimpleNamespace(all_special_tokens=[], added_tokens_decoder={})
        self.rendered = []

    def get_stop_token_ids(self):
        return [1, 2]

    def _token_id(self, name):
        return {"<tool_response>": 3, "</tool_response>": 4, "<|im_start|>": 5,
                "<think>": 6, "</think>": 7}[name]

    def render(self, messages, tools, add_generation_prompt):
        self.rendered.append((messages, tools))
        return SimpleNamespace(token_ids=[9])


def test_a_bash_only_renderer_renders_bash_only():
    pytest.importorskip("renderers")
    stub = _StubRenderer()
    agentic_swe.QwenTurnRenderer(stub, tools=("bash",)).initial_ids("Fix it.")
    messages, tools = stub.rendered[0]
    assert messages[0] == {"role": "system", "content": BASH_SENTENCE}
    assert messages[1] == {"role": "user", "content": "Fix it."}
    assert tools == [dict(BASH_HARNESS_TOOLS[0])]


def test_the_default_renderer_is_unchanged():
    pytest.importorskip("renderers")
    stub = _StubRenderer()
    agentic_swe.QwenTurnRenderer(stub).initial_ids("Fix it.")
    messages, tools = stub.rendered[0]
    assert messages[0]["content"] == BASH_SYSTEM_PROMPT and len(tools) == 2


def test_a_signed_source_serves_the_sandbox_prompt_without_the_notice():
    asked = []

    def prompt_of(split, index):
        asked.append((split, index))
        return f"task {index}"

    rows = {3: SimpleNamespace(instance_id="repo__x.3")}
    source = SignedSweSource("train:20", prompt_of=prompt_of,
                             row_of=lambda split, index: (None, rows[index]))
    assert source.prompt(3) == "task 3" and source.prompt(3) == "task 3"
    assert asked == [("train:20", 3)]                      # cached
    assert agentic_swe.PINNED_NETWORK_NOTICE not in source.prompt(3)
    assert source.instance_id(3) == "repo__x.3"
    assert source.task_for(3).prompt == "task 3" and source.split == "train:20"


def test_sandbox_support_refuses_what_this_process_cannot_serve(monkeypatch):
    job = parse_job(_manifest(episode=signed_episode()))
    replay = parse_job(_manifest())
    assert "not a signed_sandbox" in agentic_swe.sandbox_support_refusal(replay.episode)
    monkeypatch.setattr("reliquary.sandbox.sandbox_commit_refusal",
                        lambda pinned: f"the job pins reliquary-sandbox {pinned}")
    assert "the job pins" in agentic_swe.sandbox_support_refusal(job.episode)
    monkeypatch.setattr("reliquary.sandbox.sandbox_commit_refusal", lambda pinned: None)
    monkeypatch.setattr(agentic_swe.importlib.metadata, "version", lambda name: "9.9")
    assert "reliquary-swe==9.9 is installed" in agentic_swe.sandbox_support_refusal(job.episode)


def test_sandbox_support_compares_the_code_digest_record_0_carries(monkeypatch):
    """The gateway writes `name==version+g<sha16 of the installed code>` into record 0:
    the same version with other code is refused, the exact identity is served."""
    job = parse_job(_manifest(episode=signed_episode()))
    version = ENV_PACKAGE.split("==", 1)[1].split("+g", 1)[0]
    monkeypatch.setattr("reliquary.sandbox.sandbox_commit_refusal", lambda pinned: None)
    monkeypatch.setattr(agentic_swe.importlib.metadata, "version", lambda name: version)
    monkeypatch.setattr(agentic_swe.importlib, "import_module", lambda name: None)
    monkeypatch.setattr(agentic_swe, "installed_env_package",
                        lambda package: f"{package}=={version}+gffffffffffffffff")
    refusal = agentic_swe.sandbox_support_refusal(job.episode)
    assert "+gffffffffffffffff is installed" in refusal and ENV_PACKAGE in refusal
    monkeypatch.setattr(agentic_swe, "installed_env_package", lambda package: ENV_PACKAGE)
    assert agentic_swe.sandbox_support_refusal(job.episode) is None
