"""Pinned token transport refuses drift without issuing another inference call."""

import copy
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from reliquary.eval import remote_model
from reliquary.eval.remote_model import RemoteVLLMGenerator


class Tokenizer:
    chat_template = "fixture template"

    def __len__(self):
        return 1000

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2]}

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "answer<eos>"


@pytest.fixture
def model(tmp_path, monkeypatch):
    snapshot = tmp_path / "models--org--model" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
        "max_position_embeddings": 262144, "vocab_size": 248320}))
    tokenizer_calls = []

    def tokenizer(path, **kwargs):
        tokenizer_calls.append((path, kwargs))
        return Tokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer)))
    monkeypatch.setattr(remote_model, "version", lambda name: "0.1.11")
    calls, responses = [], []

    def handle(request):
        calls.append(request)
        body = json.loads(request.content)
        if responses:
            return responses.pop(0)(request, body)
        return httpx.Response(200, json=answer(body))

    original_client = httpx.Client
    constructor_calls = []

    def client(**kwargs):
        constructor_calls.append(kwargs.copy())
        kwargs["transport"] = httpx.MockTransport(handle)
        return original_client(**kwargs)

    monkeypatch.setattr(remote_model.httpx, "Client", client)
    constructor = {"endpoint": "http://127.0.0.1:19001", "api_key": "fixture-secret",
        "repo": "org/model", "revision": "a" * 40, "snapshot": snapshot,
        "expected_gpu": "1x expected fixture GPU", "expected_max_model_len": 65536,
        "timeout_seconds": 10}
    generator = RemoteVLLMGenerator(**constructor)
    load = {"repo": "org/model", "revision": "a" * 40, "gpu_count": 1,
        "max_new_tokens": 4, "max_total_tokens": 10, "thinking": False,
        "rows": [{"text": "fixture"}]}
    generator.load(**load)
    generator._episode_renderer = SimpleNamespace(get_stop_token_ids=lambda: [91, 97],
        parse_response=lambda ids, **kwargs: SimpleNamespace(content="answer",
            reasoning_content=None, tool_calls=[]))
    generator._episode_tools = []
    yield SimpleNamespace(generator=generator, calls=calls, responses=responses,
        constructor=constructor, load=load, client_calls=constructor_calls,
        tokenizer_calls=tokenizer_calls)
    generator.close()


def answer(body):
    return {"request_id": "generate-tokens-" + body["request_id"], "model": body["model"],
        "choices": [{"index": 0, "token_ids": [17, 19], "finish_reason": "stop",
            "logprobs": {"content": [{"token": "token_id:17", "logprob": -0.2},
                                      {"token": "token_id:19", "logprob": -0.1}]}}]}


def turn(model, **kwargs):
    return model.generator.generate_turn([1] * 7,
        **{"max_new_tokens": 8, "seed": 3, "sampling": {}, **kwargs})


def test_pinned_token_roundtrip_caps_history_and_never_loads_cpu_weights(model):
    metadata = model.generator.load(**model.load)
    assert metadata["vllm_version"] == "expected:0.29.0"
    assert metadata["gpu"] == "expected:" + model.constructor["expected_gpu"]
    assert metadata["remote_launch_identity"] == "expected; requires executor qualification"
    result = turn(model)
    assert result == {"text": "answer<eos>", "token_ids": [17, 19], "finish_reason": "stop",
        "message": {"content": "answer", "reasoning_content": None, "tool_calls": None}}
    request = model.calls[0]
    body = json.loads(request.content)
    assert str(request.url) == "http://127.0.0.1:19001/inference/v1/generate"
    assert request.headers["authorization"] == "Bearer fixture-secret"
    assert request.headers["x-request-id"] == body["request_id"]
    assert body["model"] == "org/model@" + "a" * 40
    assert body["token_ids"] == [1] * 7
    assert body["sampling_params"] == {"n": 1, "seed": 3, "max_tokens": 3,
        "temperature": 1, "top_p": 1, "top_k": -1, "stop_token_ids": [91, 97],
        "logprobs": 1, "skip_special_tokens": False}
    options = model.client_calls[0]
    assert options["trust_env"] is options["follow_redirects"] is False
    assert model.tokenizer_calls == [(model.constructor["snapshot"],
        {"trust_remote_code": False, "local_files_only": True})] * 2
    assert len(model.calls) == 1


@pytest.mark.parametrize("endpoint", ["https://public.example", "http://localhost:19001",
    "http://127.0.0.2:19001", "http://127.0.0.1", "http://user:pass@127.0.0.1:19001",
    "http://127.0.0.1:19001/another", "http://127.0.0.1:19001?forward=public.example"])
def test_endpoint_must_be_owned_literal_loopback(model, endpoint):
    with pytest.raises(ValueError, match="loopback"):
        RemoteVLLMGenerator(**{**model.constructor, "endpoint": endpoint})
    assert not model.calls


@pytest.mark.parametrize("changes", [{"repo": "org/other"}, {"revision": "b" * 40},
    {"gpu_count": 2}, {"gpu_count": True}, {"max_total_tokens": 65537},
    {"max_total_tokens": 2}, {"max_new_tokens": True}])
def test_loader_refuses_checkpoint_or_budget_drift(model, changes):
    with pytest.raises(ValueError):
        model.generator.load(**{**model.load, **changes})
    assert not model.calls


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(model="org/other@" + "a" * 40),
    lambda d: d.update(request_id="another-request"),
    lambda d: d["choices"].append(copy.deepcopy(d["choices"][0])),
    lambda d: d["choices"][0].update(index=True),
    lambda d: d["choices"][0].update(token_ids=[True, 19]),
    lambda d: d["choices"][0].update(token_ids=[-1, 19]),
    lambda d: d["choices"][0].update(token_ids=[1000, 19]),
    lambda d: d["choices"][0].update(token_ids=[17] * 4),
    lambda d: d["choices"][0].update(finish_reason="error"),
    lambda d: d["choices"][0]["logprobs"]["content"].pop(),
    lambda d: d["choices"][0]["logprobs"]["content"][0].update(token="token_id:18"),
    lambda d: d["choices"][0]["logprobs"]["content"][0].update(logprob=-9999),
    lambda d: d["choices"][0]["logprobs"]["content"][0].update(logprob="-0.1"),
])
def test_response_binding_and_actual_sampled_tokens_are_required(model, mutation):
    def malformed(request, body):
        document = answer(body)
        mutation(document)
        return httpx.Response(200, json=document)
    model.responses.append(malformed)
    with pytest.raises(ValueError, match="sampled-token evidence"):
        turn(model)
    assert len(model.calls) == 1


@pytest.mark.parametrize("status", [301, 307, 401, 409, 429, 500])
def test_refused_or_redirected_request_is_never_retried(model, status):
    model.responses.append(lambda *_: httpx.Response(status,
        headers={"location": "http://public.example"}, text="fixture-secret"))
    with pytest.raises(RuntimeError, match="refused") as error:
        turn(model)
    assert "fixture-secret" not in str(error.value)
    assert len(model.calls) == 1


def test_ambiguous_transport_timeout_is_not_retried(model):
    def timeout(request, body):
        raise httpx.ReadTimeout("fixture-secret", request=request)
    model.responses.append(timeout)
    with pytest.raises(RuntimeError, match="not retried") as error:
        turn(model)
    assert "fixture-secret" not in str(error.value)
    assert len(model.calls) == 1


@pytest.mark.parametrize("reason, expected", [("stop", "tool_calls"), ("length", "length")])
def test_valid_tool_call_keeps_model_identity_and_length_remains_terminal(model, reason, expected):
    from renderers.base import ToolCallParseStatus

    call = SimpleNamespace(id="model-call-1", name="signed_tool",
        arguments={"command": "pwd"}, status=ToolCallParseStatus.OK)
    model.generator._episode_renderer.parse_response = lambda *args, **kwargs: SimpleNamespace(
        content="", reasoning_content=None, tool_calls=[call])
    def response(request, body):
        document = answer(body)
        document["choices"][0]["finish_reason"] = reason
        return httpx.Response(200, json=document)
    model.responses.append(response)
    result = turn(model)
    assert result["finish_reason"] == expected
    assert result["message"]["tool_calls"] == [{"id": "model-call-1", "name": "signed_tool",
        "arguments": '{"command": "pwd"}'}]


def test_response_is_bounded_and_compressed_payload_is_refused(model, monkeypatch):
    monkeypatch.setattr(remote_model, "MAX_RESPONSE_BYTES", 64)
    with pytest.raises(ValueError, match="bound"):
        turn(model)
    model.responses.append(lambda *_: httpx.Response(200,
        headers={"content-encoding": "gzip"}, content=b""))
    with pytest.raises(ValueError, match="uncompressed"):
        turn(model)
    assert len(model.calls) == 2


def test_exhausted_budget_does_not_call_engine(model):
    with pytest.raises(ValueError, match="exhausted"):
        model.generator.generate_turn([1] * 10, max_new_tokens=1, seed=0, sampling={})
    assert not model.calls


def test_full_64k_transcript_is_enforced_without_truncating_history(model):
    model.generator.load(**{**model.load, "max_total_tokens": 65536})
    result = model.generator.generate_turn([1] * 65534,
        max_new_tokens=4, seed=0, sampling={})
    body = json.loads(model.calls[0].content)
    assert len(body["token_ids"]) == 65534 and body["sampling_params"]["max_tokens"] == 2
    assert result["token_ids"] == [17, 19]
    with pytest.raises(ValueError, match="exhausted"):
        model.generator.generate_turn([1] * 65536, max_new_tokens=1, seed=0, sampling={})
    assert len(model.calls) == 1


def test_unpinned_renderer_or_insufficient_checkpoint_context_is_refused(model, monkeypatch):
    monkeypatch.setattr(remote_model, "version", lambda *_: "0.1.10")
    with pytest.raises(ValueError, match="pinned renderer"):
        model.generator.load(**model.load)
    (model.constructor["snapshot"] / "config.json").write_text(
        json.dumps({"max_position_embeddings": 9}))
    with pytest.raises(ValueError, match="context"):
        model.generator.load(**model.load)
    assert not model.calls


def test_only_one_turn_may_be_in_flight(model):
    entered, release = threading.Event(), threading.Event()
    errors = []
    def pending(request, body):
        entered.set()
        assert release.wait(5)
        return httpx.Response(200, json=answer(body))
    model.responses.append(pending)
    def generate():
        try:
            turn(model)
        except Exception as error:
            errors.append(error)
    worker = threading.Thread(target=generate)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="in flight"):
            turn(model)
        assert len(model.calls) == 1
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not errors


def test_closed_transport_and_unsigned_batch_route_fail_closed(model):
    with pytest.raises(ValueError, match="only for signed"):
        model.generator.generate([], samples=[], seeds=[], sampling={}, max_new_tokens=1)
    model.generator.close()
    with pytest.raises(ValueError):
        turn(model)
    assert not model.calls


class PendingRemoteModel(RemoteVLLMGenerator):
    """Only the TCP transport is real; this check does not qualify a model server."""

    def __init__(self):
        super().__init__(endpoint=os.environ["EVAL_TEST_LOOPBACK_ENDPOINT"], api_key="fixture-secret",
            repo="org/model", revision="a" * 40, snapshot="unused",
            expected_gpu="fixture", expected_max_model_len=10, timeout_seconds=30)
        self._tokenizer = Tokenizer()
        self._vocab_size = 1000
        self._max_total_tokens, self._max_turn_tokens = 10, 4
        self._episode_renderer = SimpleNamespace(get_stop_token_ids=lambda: [91])

    def load(self):
        return {}


@pytest.mark.parametrize("cancelled", [False, True])
def test_model_process_cutoff_closes_actual_inflight_http_connection(monkeypatch, cancelled):
    from reliquary.eval.model_process import ModelProcess
    from reliquary.eval.platform_client import LeaseLost

    entered, disconnected, stop = threading.Event(), threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            entered.set()
            self.connection.settimeout(5)
            if self.connection.recv(1) == b"":
                disconnected.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    service = threading.Thread(target=server.serve_forever, daemon=True)
    service.start()
    monkeypatch.setenv("EVAL_TEST_LOOPBACK_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    process = ModelProcess(PendingRemoteModel, timeout_seconds=10 if cancelled else 0.3,
        cancelled=stop.is_set)
    errors = []
    def generate():
        try:
            process.generate_turn([1], max_new_tokens=4, seed=0, sampling={},
                absolute_deadline=time.monotonic() + 20)
        except Exception as error:
            errors.append(error)
    worker = threading.Thread(target=generate)
    try:
        process.load()
        worker.start()
        assert entered.wait(5)
        if cancelled:
            stop.set()
        worker.join(5)
        assert not worker.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], LeaseLost if cancelled else TimeoutError)
        assert disconnected.wait(5)
        assert not process._process.is_alive()
    finally:
        process.close()
        server.shutdown()
        server.server_close()
        service.join(5)
