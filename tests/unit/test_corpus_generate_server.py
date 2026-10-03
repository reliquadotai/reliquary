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
