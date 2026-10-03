"""Final hidden states per request from an in-process vLLM engine.

Decode steps replay as CUDA graphs, so hooks inside the model never fire; the
return value of GPUModelRunner._model_forward survives replay, and its rows are
laid out in input_batch.req_ids order with each request's scheduled token
count. Measured on an H100 (2026-09-22): proofs built from these rows verify
against an HF prefill inside the prefill-against-prefill band.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import contextlib
import os

import torch


def attribute_rows(
    order: Sequence[str], scheduled: Mapping[str, int], hidden: torch.Tensor
) -> dict[str, torch.Tensor]:
    rows: dict[str, torch.Tensor] = {}
    offset = 0
    for request_id in order:
        count = int(scheduled.get(request_id, 0))
        if count:
            rows[request_id] = hidden[offset : offset + count]
            offset += count
    if offset > hidden.shape[0]:
        raise ValueError(f"{offset} rows scheduled, {hidden.shape[0]} produced")
    return rows


def completion_rows(rows: torch.Tensor, prompt_len: int, total_len: int) -> torch.Tensor:
    """The rows that produced the completion. Async scheduling may run one extra
    step for a request its last token ended; anything else is refused."""
    expected = total_len - 1
    if rows.shape[0] not in (expected, expected + 1):
        raise ValueError(
            f"{rows.shape[0]} rows for {total_len} tokens: rows are missing "
            "(prefix caching must be off) or were attributed to the wrong request"
        )
    return rows[prompt_len - 1 : expected]


def turn_rows(rows: torch.Tensor, completion_len: int, prompt_rows: int) -> torch.Tensor:
    """The rows that produced one completion when the prompt may have come from
    the prefix cache.

    ``prompt_rows`` is the number of prompt tokens the engine actually
    scheduled (prompt length minus cached tokens). A cached prompt contributes
    no rows, but vLLM always recomputes at least its last token, so
    ``prompt_rows >= 1``; that last prompt row predicts the first completion
    token and every decode step after it adds one row. The completion's rows
    are therefore ``rows[prompt_rows - 1 : prompt_rows - 1 + completion_len]``,
    and a single trailing surplus row (async scheduling) is ignored. Any other
    row count -- a request preempted and recomputed, or rows attributed to the
    wrong request -- is refused rather than silently shifted. There is no
    fallback without ``prompt_rows``: taking the last ``completion_len`` rows
    would keep the async surplus row and drop the first real one.
    """
    if completion_len < 1:
        raise ValueError(f"completion_len must be >= 1, got {completion_len}")
    if isinstance(prompt_rows, bool) or not isinstance(prompt_rows, int) or prompt_rows < 1:
        raise ValueError(f"prompt_rows must be an int >= 1 (vLLM recomputes the last prompt token), got {prompt_rows!r}")
    count = rows.shape[0]
    if count < completion_len:
        raise ValueError(f"{count} rows for a {completion_len}-token completion")
    expected = prompt_rows + completion_len - 1
    if count not in (expected, expected + 1):
        raise ValueError(
            f"{count} rows where {expected} were scheduled: the request was "
            "recomputed or rows were attributed to the wrong request"
        )
    return rows[prompt_rows - 1 : prompt_rows - 1 + completion_len]


class HiddenStateCapture:
    def __init__(self) -> None:
        self._rows: dict[str, list[torch.Tensor]] = {}

    def record(self, order: Sequence[str], scheduled: Mapping[str, int], hidden: torch.Tensor) -> None:
        for request_id, rows in attribute_rows(order, scheduled, hidden).items():
            self._rows.setdefault(request_id, []).append(rows.detach().to("cpu", torch.bfloat16))

    def for_request(self, request_id: str) -> torch.Tensor:
        # The runner suffixes engine ids ("0" becomes "0-ae415201").
        matches = [r for r in self._rows if r == request_id or r.startswith(request_id + "-")]
        if len(matches) != 1:
            raise KeyError(f"{len(matches)} captured requests match {request_id!r}")
        return torch.cat(self._rows[matches[0]], 0)

    def pop(self, request_id: str) -> torch.Tensor:
        """Like ``for_request``, but forgets the rows: a long-lived miner process
        must not keep every completion's activations resident forever."""
        matches = [r for r in self._rows if r == request_id or r.startswith(request_id + "-")]
        if len(matches) != 1:
            raise KeyError(f"{len(matches)} captured requests match {request_id!r}")
        return torch.cat(self._rows.pop(matches[0]), 0)


def _check_engine_mode() -> None:
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("hidden-state capture needs VLLM_ENABLE_V1_MULTIPROCESSING=0")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise RuntimeError("hidden-state capture is hooked on the V1 runner: set VLLM_USE_V2_MODEL_RUNNER=0")


@contextlib.contextmanager
def capture_hidden_states(runner_cls=None) -> Iterator[HiddenStateCapture]:
    _check_engine_mode()
    if runner_cls is None:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as runner_cls
    capture = HiddenStateCapture()
    scheduled: dict[str, int] = {}
    original_execute = runner_cls.execute_model
    original_forward = runner_cls._model_forward

    def execute_model(self, scheduler_output, *args, **kwargs):
        scheduled.clear()
        scheduled.update(scheduler_output.num_scheduled_tokens)
        return original_execute(self, scheduler_output, *args, **kwargs)

    def _model_forward(self, *args, **kwargs):
        output = original_forward(self, *args, **kwargs)
        if scheduled:
            hidden = output[0] if isinstance(output, tuple) else output
            batch = self.input_batch
            capture.record(batch.req_ids[: batch.num_reqs], dict(scheduled), hidden)
            scheduled.clear()
        return output

    runner_cls.execute_model = execute_model
    runner_cls._model_forward = _model_forward
    try:
        yield capture
    finally:
        runner_cls.execute_model = original_execute
        runner_cls._model_forward = original_forward
