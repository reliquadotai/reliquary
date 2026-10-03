"""Row attribution for the vLLM capture, tested on a fake runner: the real one
needs a GPU, and the logic that can go wrong is the bookkeeping."""

from types import SimpleNamespace

import pytest
import torch

from reliquary.miner.vllm_hidden_capture import (
    attribute_rows,
    capture_hidden_states,
    completion_rows,
    turn_rows,
)


def test_rows_follow_the_batch_order_and_scheduled_counts():
    hidden = torch.arange(6).float().unsqueeze(1)
    rows = attribute_rows(["b", "a", "c"], {"a": 1, "b": 3, "c": 0}, hidden)
    assert rows["b"].flatten().tolist() == [0, 1, 2]
    assert rows["a"].flatten().tolist() == [3]
    assert "c" not in rows


def test_more_scheduled_rows_than_produced_is_refused():
    with pytest.raises(ValueError):
        attribute_rows(["a"], {"a": 5}, torch.zeros(3, 1))


def test_completion_rows_start_at_the_last_prompt_position():
    rows = torch.arange(9).float().unsqueeze(1)          # total_len 10 -> 9 rows
    assert completion_rows(rows, 4, 10).flatten().tolist() == [3, 4, 5, 6, 7, 8]


def test_one_surplus_row_from_async_scheduling_is_dropped():
    rows = torch.arange(10).float().unsqueeze(1)
    assert completion_rows(rows, 4, 10).flatten().tolist() == [3, 4, 5, 6, 7, 8]


def test_missing_rows_are_refused():
    with pytest.raises(ValueError, match="prefix caching"):
        completion_rows(torch.zeros(5, 1), 4, 10)


class _FakeRunner:
    def __init__(self, steps):
        self._steps = list(steps)
        self.input_batch = SimpleNamespace(req_ids=[], num_reqs=0)

    def execute_model(self, scheduler_output):
        return self._model_forward()

    def _model_forward(self):
        order, hidden = self._steps.pop(0)
        self.input_batch.req_ids = order
        self.input_batch.num_reqs = len(order)
        return hidden


def _env(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")


def test_the_patch_records_every_step_and_is_removed_after(monkeypatch):
    _env(monkeypatch)
    runner = _FakeRunner([
        (["0-x", "1-y"], torch.tensor([[1.0], [2.0], [3.0]])),
        (["0-x", "1-y"], torch.tensor([[4.0], [5.0]])),
    ])
    original = _FakeRunner.execute_model
    with capture_hidden_states(_FakeRunner) as capture:
        runner.execute_model(SimpleNamespace(num_scheduled_tokens={"0-x": 2, "1-y": 1}))
        runner.execute_model(SimpleNamespace(num_scheduled_tokens={"0-x": 1, "1-y": 1}))
    assert capture.for_request("0").flatten().tolist() == [1.0, 2.0, 4.0]
    assert capture.for_request("1").flatten().tolist() == [3.0, 5.0]
    assert _FakeRunner.execute_model is original


@pytest.mark.parametrize(
    "variable,value",
    [("VLLM_ENABLE_V1_MULTIPROCESSING", "1"), ("VLLM_USE_V2_MODEL_RUNNER", "1")],
)
def test_an_unsupported_engine_mode_is_refused(monkeypatch, variable, value):
    _env(monkeypatch)
    monkeypatch.setenv(variable, value)
    with pytest.raises(RuntimeError):
        with capture_hidden_states(_FakeRunner):
            pass


def test_pop_returns_and_forgets_the_request(monkeypatch):
    _env(monkeypatch)
    runner = _FakeRunner([(["0-x", "1-y"], torch.tensor([[1.0], [2.0], [3.0]]))])
    with capture_hidden_states(_FakeRunner) as capture:
        runner.execute_model(SimpleNamespace(num_scheduled_tokens={"0-x": 2, "1-y": 1}))
        popped = capture.pop("0")
        assert popped.flatten().tolist() == [1.0, 2.0]
        with pytest.raises(KeyError):
            capture.pop("0")
        # The other request is untouched by popping the first.
        assert capture.for_request("1").flatten().tolist() == [3.0]


def test_turn_rows_take_the_last_rows_when_the_prompt_was_cached():
    # 40-token prompt, 37 served from cache: 3 prompt rows + 5 completion tokens
    # -> 3 + (5 - 1) = 7 rows; the completion's rows are the last 5.
    rows = torch.arange(7).float().unsqueeze(1)
    assert turn_rows(rows, 5).flatten().tolist() == [2, 3, 4, 5, 6]


def test_turn_rows_with_only_the_last_prompt_token_recomputed():
    rows = torch.arange(5).float().unsqueeze(1)     # 1 prompt row + 4 decode rows
    assert turn_rows(rows, 5).flatten().tolist() == [0, 1, 2, 3, 4]


def test_turn_rows_drop_one_surplus_async_row():
    rows = torch.arange(8).float().unsqueeze(1)     # 7 expected + 1 extra step
    assert turn_rows(rows, 5, prompt_rows=3).flatten().tolist() == [2, 3, 4, 5, 6]


def test_turn_rows_refuse_too_few_rows():
    with pytest.raises(ValueError, match="rows"):
        turn_rows(torch.zeros(4, 1), 5)


def test_turn_rows_refuse_more_than_one_surplus_row():
    # Preemption recomputes a request: the rows no longer line up with tokens.
    with pytest.raises(ValueError, match="recomputed"):
        turn_rows(torch.zeros(10, 1), 5, prompt_rows=3)


def test_turn_rows_with_prompt_rows_from_the_cache_count():
    prompt_len, cached, completion_len = 40, 37, 5
    rows = torch.arange((prompt_len - cached) + completion_len - 1).float().unsqueeze(1)
    got = turn_rows(rows, completion_len, prompt_rows=prompt_len - cached)
    assert got.flatten().tolist() == [2, 3, 4, 5, 6]
