"""Actual model token evidence and shared signed task identity."""

import asyncio
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from reliquary.eval.runner import VLLMGenerator
from reliquary.eval.signed_agentic import resolve_task
from reliquary.sandbox.tasks import ResolvedTask


def test_actual_worker_response_bytes_pass_native_grant_parser_without_replay():
    import json
    import httpx
    from reliquary.eval.platform_client import PlatformClient, PlatformError
    raw = (Path(__file__).parents[1] / "fixtures/evaluation_sandbox_response.json").read_bytes()
    expected = json.loads(raw)
    assert expected["grant"]["token"] == {"fixture": True}
    calls = []
    def handle(request):
        calls.append(request)
        assert request.headers["x-task-lease"] == "fixture-lease"
        return httpx.Response(200, content=raw)
    client = PlatformClient("http://fixture", "fixture-pod-token", executor_id="fixture-worker",
        http=httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(handle)))
    client.task_id, client._lease = "task", "fixture-lease"
    assert client.sandbox_session("s", "p", 0, timeout_seconds=1) == expected
    assert len(calls) == 1
    raw = json.dumps({**expected, "grant": {**expected["grant"], "token": "not-a-signed-token-object"}}).encode()
    with pytest.raises(PlatformError, match="invalid"):
        client.sandbox_session("s", "p", 0, timeout_seconds=1)
    assert len(calls) == 2


def test_episode_model_turn_preserves_actual_token_ids_and_caps_full_context(monkeypatch):
    calls = []
    def generate(prompts, params, **kwargs):
        calls.append((prompts, params))
        return [SimpleNamespace(outputs=[SimpleNamespace(text="answer", token_ids=[17, 19], finish_reason="stop")])]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    model = VLLMGenerator()
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            return {"input_ids": [91 if text == "<|im_end|>" else 97]}
        def decode(self, ids, **kwargs):
            assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
            return "answer<eos>"
    model._tokenizer = Tokenizer()
    model._llm = SimpleNamespace(generate=generate)
    model._max_total_tokens, model._max_turn_tokens = 10, 4
    result = model.generate_turn([1, 2, 3, 4, 5, 6, 7], max_new_tokens=8, seed=3, sampling={})
    assert result == {"text": "answer<eos>", "token_ids": [17, 19], "finish_reason": "stop"}
    assert calls[0][1][0]["max_tokens"] == 3 and calls[0][1][0]["seed"] == 3
    assert calls[0][1][0]["stop_token_ids"] == [91, 97]
    with pytest.raises(ValueError, match="exhausted"):
        model.generate_turn([1] * 10, max_new_tokens=4, seed=3, sampling={})
    assert len(calls) == 1


def test_signed_set_and_issuer_share_the_actual_resolved_task_identity():
    class Source:
        image = "sha256:" + "a" * 64
        def prompt(self, index):
            return "fixture task"
        async def resolve(self, index):
            return ResolvedTask(image=self.image, limits={"wall_s": 300}, cpu_millis=1000)
    source = Source()
    first = asyncio.run(resolve_task(source, 2))
    assert first["source_index"] == 2 and first["task"]["image"] == source.image
    assert first == asyncio.run(resolve_task(source, 2))
    source.image = "sha256:" + "b" * 64
    assert asyncio.run(resolve_task(source, 2))["reference_sha256"] != first["reference_sha256"]


def test_signed_set_binds_full_registry_image_and_declared_source_policy(tmp_path, monkeypatch):
    from reliquary.eval import protocol, sets

    class Source:
        identity = "fixture==1"
        options_sha256 = "a" * 64
        lock_sha256 = "b" * 64
        image = "registry.example/fixture@sha256:" + "c" * 64
        def __len__(self):
            return 1
        def manifest_sha256(self):
            return "d" * 64
        def prompt(self, index):
            return "fixture task"
        async def resolve(self, index):
            return ResolvedTask(image=self.image, limits={"wall_s": 300}, cpu_millis=1000)
    policy = protocol.SIGNED_SOURCE_POLICIES[("reliquary-swe==0.3.0", "eval")]
    config = {"package": "reliquary-swe==0.3.0", "split": "eval", "options": {},
        "image_lock_sha256": "b" * 64, "renderer_sha256": "e" * 64,
        "tools_version": "fixture", "tools": ["bash"], "max_turns": 40,
        "source_policy_sha256": policy}
    monkeypatch.setattr(protocol, "renderer_sha256", lambda: "e" * 64)
    monkeypatch.setattr(protocol, "harness_sha256", lambda *_: "f" * 64)
    source = Source()
    def build(name, **changes):
        return sets.build_episode_set("fixture-swe", out=tmp_path / name,
            interaction="signed_episode", count=1, seed=0,
            episode_config={**config, **changes}, protocol={"data": {"environment.toml": policy}},
            open_episode=lambda *_: source)
    card = build("valid")
    assert card["protocol"]["source"]["episode"]["runtime_image_sha256"] == "c" * 64
    assert card["protocol"]["harness"]["max_turns"] == 40
    assert protocol.validate_set(card) == card["protocol_sha256"]
    for changes in ({"max_turns": 48}, {"source_policy_sha256": "0" * 64},
                    {"package": "unknown==1"}, {"split": "train"}):
        with pytest.raises(ValueError):
            build("refused", **changes)
    source.image = "registry.example/fixture:latest"
    with pytest.raises(ValueError, match="exact digest"):
        build("unpinned")


def test_serialized_episode_row_limit_preserves_evidence_or_refuses():
    import json
    from reliquary.eval.runner import EvalRunner, Completion

    row = {"problem_id": "p", "samples": 1}
    completion = Completion("", 2, "stop", {"state_base64": "eHl6", "transcript": {"proof": "intact"}})
    body = EvalRunner._chunk_body([row], [[completion]])
    assert EvalRunner._chunk_body([row], [[completion]], row_limit=len(body)) == body
    assert json.loads(body)["episode"] == completion.episode
    with pytest.raises(ValueError, match="frozen byte allowance"):
        EvalRunner._chunk_body([row], [[completion]], row_limit=len(body) - 1)


def test_signed_grading_binds_each_set_and_server_accepted_worker_fence():
    import base64
    from collections import defaultdict
    import pyarrow as pa
    import pyarrow.parquet as pq
    from reliquary.eval.grading import _grade_batch, _graded_schema

    provenance = {"task_id": "t", "model": "fixture/model", "revision": "a" * 40,
                  "protocol_sha256": "b" * 64, "worker_id": "accepted-worker", "lease_fence": "current-fence"}
    batch, verified = [], []
    for set_id in ("first", "second", "stale"):
        binding = {**{key: provenance[key] for key in ("task_id", "protocol_sha256", "worker_id", "lease_fence")},
            "eval_id": "e", "model": {"repo": provenance["model"], "revision": provenance["revision"]},
            "set_id": set_id, "problem_id": "shared", "sample_index": 0}
        if set_id == "stale":
            binding["lease_fence"] = "replaced-fence"
        row = {"problem_id": "shared", "sample_index": 0, "episode": {"binding": binding,
            "transcript": {}, "state_base64": base64.b64encode(b"state").decode()}}
        card = {"set_id": set_id, "env": "fixture", "interaction": "signed_episode"}
        batch.append((row, (set_id, card, {"problem_id": "shared"}, 1)))
    async def verify(binding, transcript, **kwargs):
        assert kwargs["final_state"] == b"state"
        verified.append(binding["set_id"])
        return float(binding["set_id"] == "first")
    output = pa.BufferOutputStream()
    writer = pq.ParquetWriter(output, _graded_schema())
    asyncio.run(_grade_batch(batch, None, defaultdict(list), writer, pa,
        verify_signed=verify, provenance=provenance, eval_id="e"))
    writer.close()
    rows = pq.read_table(pa.BufferReader(output.getvalue())).to_pylist()
    assert [row["score"] for row in rows] == [1, 0, None]
    assert verified == ["first", "second"]
    assert rows[-1]["grader_detail"] == "signed_receipt_error: ValueError"


@pytest.mark.asyncio
async def test_pinned_interception_returns_real_local_model_completion_not_null():
    pytest.importorskip("verifiers.v1.interception")
    from aiohttp import ClientSession
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import EvalClientConfig
    from verifiers.v1.configs.agent import AgentConfig
    from verifiers.v1.interception import InterceptionServer
    from verifiers.v1.session import RolloutSession
    from verifiers.v1.trace import Trace, TraceTask, AgentInfo
    from verifiers.v1.types import Sampling
    from reliquary.eval.protocol import digest
    from reliquary.eval.signed_agentic import model_client

    calls = []
    class Generator:
        def render_messages(self, messages, **kwargs):
            calls.append(messages)
            return [1, 2]
        def generate_turn(self, ids, **kwargs):
            return {"text": "answer", "token_ids": [11, 13], "finish_reason": "stop",
                    "message": {"content": "answer", "reasoning_content": None, "tool_calls": None}}
    model = "fixture/model@" + "a" * 40
    config = EvalClientConfig(base_url="http://127.0.0.1:1", api_key_var="UNUSED_EVAL_FIXTURE_KEY")
    context = ModelContext(model=model, client=config, sampling=Sampling(max_tokens=32))
    task = {"model": {"repo": "fixture/model", "revision": "a" * 40},
            "protocol": {"max_turns": 2}, "thinking": False, "max_new_tokens": 32, "sampling": {}}
    client = model_client(Generator(), task=task, source={"tools_sha256": digest([]),
        "renderer_id": "renderers:qwen38@0.1.11"}, deadline=time.monotonic() + 10, seed=1)
    trace = Trace(task=TraceTask(type="fixture", data={}), agent=AgentInfo(config=AgentConfig()))
    session = RolloutSession(ctx=context, trace=trace)
    async with InterceptionServer() as server:
        server.clients[config.model_dump_json()] = client
        async with server.acquire(session) as (url, token, _):
            async with ClientSession() as http:
                async with http.post(url + "/v1/chat/completions", headers={"Authorization": "Bearer " + token},
                    json={"model": model, "messages": [{"role": "user", "content": "question"}],
                          "max_tokens": 32}) as response:
                    assert response.status == 200, await response.text()
                    body = await response.json()
    assert body["model"] == model and body["choices"][0]["message"]["content"] == "answer"
    assert body["usage"]["completion_tokens"] == 2
    assert calls == [[{"role": "user", "content": "question"}]]
    assert client.token_ids == (11, 13) and trace.num_turns == 1
