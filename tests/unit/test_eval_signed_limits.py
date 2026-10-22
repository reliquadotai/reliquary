"""Actual generation-call and rendered-token limits at the local model boundary."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers.v1.dialects")
from verifiers.v1.dialects import AnthropicDialect, ChatDialect, ResponsesDialect

from reliquary.eval.protocol import digest
from reliquary.eval.signed_agentic import model_client

MODEL = "fixture/model@" + "a" * 40
TASK = {"model": {"repo": "fixture/model", "revision": "a" * 40},
        "protocol": {"max_turns": 48, "max_episode_tokens": 65536},
        "thinking": False, "max_new_tokens": 64, "sampling": {}}
SOURCE = {"tools_sha256": digest([]), "renderer_id": "renderers:qwen38@0.1.11"}
BODY = {"model": MODEL, "messages": [{"role": "user", "content": "question"}]}


class Generator:
    calls = 0
    rendered_tokens = 2
    tokens = [17]

    def render_messages(self, messages, **kwargs):
        return [1] * self.rendered_tokens

    def generate_turn(self, ids, **kwargs):
        self.calls += 1
        self.maximum = kwargs["max_new_tokens"]
        return {"token_ids": self.tokens, "finish_reason": "stop", "message": {"content": "answer"}}


def local(generator, **kwargs):
    return model_client(generator, task=TASK, source=SOURCE, deadline=time.monotonic() + 10, seed=7, **kwargs)


@pytest.mark.asyncio
async def test_concurrent_requests_share_one_hard_48_generation_budget():
    generator = Generator()
    client = local(generator)
    results = await asyncio.gather(*(client.get_response(ChatDialect(), BODY, None) for _ in range(49)),
                                   return_exceptions=True)
    assert sum(not isinstance(value, BaseException) for value in results) == 48
    assert isinstance(results[-1], TimeoutError)
    assert generator.calls == client.turns == 48
    assert len(client.generated_spans) == len(client.token_ids) == 48


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_tokens", [65535, 65536, 65537])
async def test_complete_rendered_context_and_output_share_65536_tokens(prompt_tokens):
    generator = Generator()
    generator.rendered_tokens = prompt_tokens
    client = local(generator)
    if prompt_tokens == 65535:
        await client.get_response(ChatDialect(), BODY, None)
        assert generator.maximum == 1 and client.transcript_tokens == 65536
    else:
        with pytest.raises(ValueError, match="budget exhausted"):
            await client.get_response(ChatDialect(), BODY, None)
        assert generator.calls == client.turns == 0


@pytest.mark.asyncio
async def test_bad_model_token_accounting_cannot_escape_remaining_context():
    generator = Generator()
    generator.rendered_tokens, generator.tokens = 65535, [17, 19]
    with pytest.raises(ValueError, match="token budget"):
        await local(generator).get_response(ChatDialect(), BODY, None)
    assert generator.maximum == 1 and generator.calls == 1


@pytest.mark.asyncio
async def test_cancelled_inference_closes_the_owned_model_process():
    started, closed = threading.Event(), threading.Event()

    class Blocked(Generator):
        def generate_turn(self, ids, **kwargs):
            started.set()
            assert closed.wait(5)
            return super().generate_turn(ids, **kwargs)

        def close(self):
            closed.set()

    task = asyncio.create_task(local(Blocked()).get_response(ChatDialect(), BODY, None))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


class Bridge:
    def __init__(self, identity):
        self.capability = SimpleNamespace(protocol_id=identity, require_model_tools=self.require_model_tools)
        self.turns = []

    def require_model_tools(self, tools):
        assert tools == []

    def validate_messages(self, messages):
        assert not any(message["role"] == "tool" for message in messages)

    def register_turn(self, turn, calls, *, assistant_message):
        assert assistant_message["content"] == "answer"
        self.turns.append((turn, calls))


@pytest.mark.asyncio
@pytest.mark.parametrize("identity,dialect,body", [
    ("signed-codex-cli-v1", ResponsesDialect(), {"input": "question"}),
    ("signed-claude-code-cli-v1", AnthropicDialect(), {"messages": BODY["messages"], "max_tokens": 64}),
])
async def test_native_streaming_uses_only_owned_model_and_counts_one_turn(identity, dialect, body):
    generator, bridge = Generator(), Bridge(identity)
    client = local(generator, cli_bridge=bridge)
    reply = await client.relay(dialect, {"model": MODEL, **body})
    parser = dialect.stream_parser()
    async for chunk in reply.chunks:
        parser.feed(chunk)
    await reply.close()
    assert parser.finish().message.content == "answer"
    assert client.turns == generator.calls == 1 and bridge.turns == [(0, [])]
    with pytest.raises(ValueError, match="another model"):
        await client.get_response(dialect, {"model": "other/model", **body}, None)
    assert generator.calls == 1


@pytest.mark.asyncio
async def test_local_token_count_never_invokes_inference_or_forwards_an_aux_request():
    generator = Generator()
    client = local(generator, cli_bridge=Bridge("signed-claude-code-cli-v1"))
    assert await client.relay_aux(AnthropicDialect(), "/v1/messages/count_tokens", BODY) == {"input_tokens": 2}
    assert generator.calls == client.turns == 0
    with pytest.raises(ValueError, match="unsupported"):
        await client.relay_aux(AnthropicDialect(), "/v1/other", BODY)


@pytest.mark.asyncio
async def test_native_custom_tools_are_refused_before_normalization_can_drop_them():
    generator = Generator()
    client = local(generator, cli_bridge=Bridge("signed-codex-cli-v1"))
    with pytest.raises(ValueError, match="unsupported native tools"):
        await client.get_response(ResponsesDialect(), {"model": MODEL, "input": "question", "tools": [
            {"type": "custom", "name": "unsigned", "format": {"type": "text"}}]}, None)
    assert generator.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("identity,dialect,path,body", [
    ("signed-codex-cli-v1", ResponsesDialect(), "/v1/responses", {"input": "question"}),
    ("signed-claude-code-cli-v1", AnthropicDialect(), "/v1/messages", {"messages": BODY["messages"], "max_tokens": 64}),
])
@pytest.mark.parametrize("stream", [False, True])
async def test_real_interception_server_routes_native_http_and_streams_locally(identity, dialect, path, body, stream):
    from aiohttp import ClientSession
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.agent import AgentConfig
    from verifiers.v1.configs.client import EvalClientConfig
    from verifiers.v1.interception import InterceptionServer
    from verifiers.v1.session import RolloutSession
    from verifiers.v1.trace import AgentInfo, Trace, TraceTask
    from verifiers.v1.types import Sampling

    generator = Generator()
    client = local(generator, cli_bridge=Bridge(identity))
    config = EvalClientConfig(base_url="http://127.0.0.1:1", api_key_var="UNUSED_EVAL_FIXTURE_KEY")
    trace = Trace(task=TraceTask(type="fixture", data={}), agent=AgentInfo(config=AgentConfig()))
    session = RolloutSession(ctx=ModelContext(model=MODEL, client=config, sampling=Sampling(max_tokens=64)), trace=trace)
    async with InterceptionServer() as server:
        server.clients[config.model_dump_json()] = client
        async with server.acquire(session) as (url, token, _):
            async with ClientSession() as http:
                async with http.post(url + path, headers=dialect.auth_headers(token),
                    json={"model": "untrusted-model-choice", **body, "stream": stream}) as reply:
                    assert reply.status == 200, await reply.text()
                    if stream:
                        parser = dialect.stream_parser()
                        for event in (await reply.read()).split(b"\n\n"):
                            if event:
                                parser.feed(event + b"\n\n")
                        response = parser.finish()
                    else:
                        response = dialect.parse_response(dialect.validate_response(await reply.json()))
    assert response.model == MODEL and response.message.content == "answer"
    assert client.turns == generator.calls == trace.num_turns == 1
