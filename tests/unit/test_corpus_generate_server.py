"""The loopback generate endpoint, over a fake turn core (no GPU)."""

import pytest
from fastapi.testclient import TestClient

from reliquary.miner.corpus_generate_server import (
    Finished,
    GenerateEngine,
    build_generate_app,
)

TERM = 1


class FakeCore:
    """Completes every request in one step: `n % 7 + 100` repeated, then TERM."""

    def __init__(self, fail=False):
        self.added, self._queue, self._fail = [], [], fail

    def add(self, request_id, prompt_ids, max_tokens):
        self.added.append((request_id, list(prompt_ids), max_tokens))
        self._queue.append((request_id, len(prompt_ids), max_tokens))

    def has_unfinished(self):
        return bool(self._queue)

    def step(self):
        out = []
        for request_id, n, cap in self._queue:
            if self._fail:
                out.append(Finished(request_id, error="recomputed"))
                continue
            tokens = tuple([100 + n % 7] * (min(cap, 3) - 1) + [TERM])
            out.append(Finished(request_id, tokens, tuple(-0.5 for _ in tokens), "stop", ("P",)))
        self._queue = []
        return out


@pytest.fixture
def served():
    core = FakeCore()
    engine = GenerateEngine(core, max_total_tokens=40, max_tokens_per_turn=8)
    engine.start()
    yield engine, core, TestClient(build_generate_app(engine, model_name="Qwen/Qwen3.8-27B"))
    engine.stop()


def _post(client, ids, session="s1", max_tokens=None):
    params = {} if max_tokens is None else {"max_tokens": max_tokens}
    headers = {} if session is None else {"X-Session-ID": session}
    return client.post("/inference/v1/generate", headers=headers,
                       json={"model": "Qwen/Qwen3.8-27B", "token_ids": ids, "sampling_params": params})


def test_a_turn_answers_in_the_renderers_shape(served):
    _, _, client = served
    choice = _post(client, [5, 6, 7]).json()["choices"][0]
    assert choice["token_ids"] == [103, 103, TERM] and choice["finish_reason"] == "stop"
    assert choice["logprobs"]["content"][0] == {"token": "token_id:103", "logprob": -0.5}
    renderers_client = pytest.importorskip("renderers.client")
    assert renderers_client._parse_completion_logprobs(choice, choice["token_ids"]) == [-0.5] * 3


def test_max_tokens_is_clamped_to_the_turn_and_total_budgets(served):
    _, core, client = served
    _post(client, [5] * 10, max_tokens=100)
    _post(client, [5] * 36, session="s2")
    assert [cap for _, _, cap in core.added] == [8, 4]


def test_a_prompt_with_no_room_left_is_refused(served):
    _, _, client = served
    assert _post(client, [5] * 40).status_code == 400


def test_a_turn_must_name_its_session(served):
    _, _, client = served
    assert _post(client, [5, 6], session=None).status_code == 400


def test_the_session_log_keeps_every_turn_and_notices_a_rewrite(served):
    engine, _, client = served
    first = _post(client, [5, 6, 7]).json()["choices"][0]["token_ids"]
    _post(client, [5, 6, 7] + first + [40, 41])
    log = engine.take_session("s1")
    assert [len(t.prompt_ids) for t in log.turns] == [3, 8] and log.linear
    assert log.turns[0].proofs == ("P",) and engine.take_session("s1") is None
    _post(client, [5, 6, 7], session="s3")
    _post(client, [9, 9, 9, 9, 9], session="s3")
    assert engine.take_session("s3").linear is False


def test_a_turn_that_cannot_be_proven_is_a_server_error():
    engine = GenerateEngine(FakeCore(fail=True), max_total_tokens=40, max_tokens_per_turn=8)
    engine.start()
    try:
        client = TestClient(build_generate_app(engine, model_name="Qwen/Qwen3.8-27B"))
        assert _post(client, [5, 6]).status_code == 500
    finally:
        engine.stop()


def test_models_reports_the_context_budget(served):
    _, _, client = served
    card = client.get("/v1/models").json()["data"][0]
    assert (card["id"], card["max_model_len"]) == ("Qwen/Qwen3.8-27B", 40)


# -- fix round 1 ---------------------------------------------------------------

import asyncio
import time
from types import SimpleNamespace

import torch

from reliquary.miner.corpus_generate_server import VllmTurnCore
from reliquary.miner.vllm_hidden_capture import HiddenStateCapture


def _wait(predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class _Engine:
    def __init__(self, outputs):
        self.outputs, self.aborted = outputs, []

    def step(self):
        out, self.outputs = self.outputs, []
        return out

    def add_request(self, *a, **k):
        pass

    def abort_request(self, ids):
        self.aborted += list(ids)


def _core(engine, capture, prompt_len):
    core = VllmTurnCore.__new__(VllmTurnCore)
    core._engine, core._capture, core._prompt_len = engine, capture, dict(prompt_len)
    core._proof = SimpleNamespace(chunk_tokens=8, topk=2)
    core._stops = frozenset({1})
    return core


def _output(request_id, tokens=(7, 1)):
    comp = SimpleNamespace(token_ids=list(tokens), logprobs=None)
    return SimpleNamespace(request_id=request_id, finished=True, outputs=[comp], num_cached_tokens=0)


def _rows(capture, request_id, n):
    capture.record([request_id], {request_id: n}, torch.randn(n, 8))


def test_one_bad_output_does_not_fail_its_batch_mates():
    capture = HiddenStateCapture()
    _rows(capture, "a-1", 3 + 2 - 1)
    bad = _output("b-1")
    bad.outputs = None  # AttributeError / TypeError inside _finish
    core = _core(_Engine([bad, _output("a-1")]), capture, {"a": 3, "b": 3})
    results = core.step()
    by_id = {r.request_id: r for r in results}
    assert by_id["a"].error is None and by_id["a"].completion_ids == (7, 1)
    assert by_id["b"].error is not None


def test_orphan_capture_rows_are_swept_and_counted():
    capture = HiddenStateCapture()
    _rows(capture, "gone-1", 4)  # a request nobody tracks any more
    _rows(capture, "live-1", 4)  # still decoding
    core = _core(_Engine([]), capture, {"live": 4})
    assert core.pending_capture_count() == 2
    core.step()
    assert core.pending_capture_count() == 1
    core.abort("live")
    assert core.pending_capture_count() == 0 and core._engine.aborted == ["live"]


def test_a_refused_add_leaves_no_prompt_length():
    class Refusing(_Engine):
        def add_request(self, *a, **k):
            raise ValueError("too long")

    core = _core(Refusing([]), HiddenStateCapture(), {})
    core._params = lambda **k: None
    core._sampling = {}
    with pytest.raises(Exception):
        core.add("r", [1, 2, 3], 4)
    assert core._prompt_len == {}


class SlowCore(FakeCore):
    """Never finishes until released; records aborts."""

    def __init__(self):
        super().__init__()
        self.aborted, self.release = [], False

    def step(self):
        return super().step() if self.release else []

    def has_unfinished(self):
        return bool(self._queue)

    def abort(self, request_id):
        self.aborted.append(request_id)
        self._queue = [q for q in self._queue if q[0] != request_id]


def test_dropping_a_session_aborts_its_turn_and_it_stays_dropped():
    core = SlowCore()
    engine = GenerateEngine(core, max_total_tokens=40, max_tokens_per_turn=8)
    engine.start()
    try:
        async def scenario():
            task = asyncio.ensure_future(engine.generate("s9", [5, 6], None))
            assert await asyncio.get_running_loop().run_in_executor(
                None, _wait, lambda: bool(core._queue))
            engine.drop_session("s9")
            with pytest.raises(RuntimeError):
                await task
            with pytest.raises(ValueError):
                await engine.generate("s9", [5, 6], None)
        asyncio.run(scenario())
        assert _wait(lambda: len(core.aborted) == 1)
        assert engine.take_session("s9") is None
    finally:
        engine.stop()


def test_a_cancelled_waiter_aborts_its_request():
    core = SlowCore()
    engine = GenerateEngine(core, max_total_tokens=40, max_tokens_per_turn=8)
    engine.start()
    try:
        async def scenario():
            task = asyncio.ensure_future(engine.generate("s8", [5, 6], None))
            assert await asyncio.get_running_loop().run_in_executor(
                None, _wait, lambda: bool(core._queue))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        asyncio.run(scenario())
        assert _wait(lambda: len(core.aborted) == 1)
    finally:
        engine.stop()


def test_a_request_whose_waiter_is_gone_is_never_added():
    core = FakeCore()
    engine = GenerateEngine(core, max_total_tokens=40, max_tokens_per_turn=8)
    engine._inbox.put(("add", "ghost", [1, 2], 4))  # no pending waiter
    engine.start()
    try:
        time.sleep(0.3)
        assert core.added == []
    finally:
        engine.stop()


class BrokenCore(FakeCore):
    def step(self):
        raise RuntimeError("cuda died")


def test_a_broken_engine_goes_unhealthy_and_refuses_fast():
    engine = GenerateEngine(BrokenCore(), max_total_tokens=40, max_tokens_per_turn=8)
    engine.start()
    try:
        client = TestClient(build_generate_app(engine, model_name="Qwen/Qwen3.8-27B"))
        for _ in range(6):
            assert _post(client, [5, 6]).status_code == 500
            if not engine.healthy:
                break
        assert _wait(lambda: not engine.healthy)
        response = _post(client, [5, 6])
        assert response.status_code == 500 and "unhealthy" in response.text
    finally:
        engine.stop()


def test_a_boolean_max_tokens_is_refused(served):
    _, _, client = served
    assert _post(client, [5, 6], max_tokens=True).status_code == 400
