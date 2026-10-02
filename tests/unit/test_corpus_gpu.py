"""The GPU process: one FIFO for every judge, merged forwards, isolated
failures, back-pressure; and the judges' client, which waits out a GPU
process that is down instead of failing an audit."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from reliquary.protocol.toploc import ChunkResult
from reliquary.validator.corpus_gpu import (
    GpuBatcher,
    GpuBusy,
    GpuScoreError,
    GpuScorer,
    build_gpu_app,
    decode_request,
    encode_request,
    scores_from_wire,
    scores_to_wire,
    serve_unix,
)
from tests.unit import corpus_split_fakes as fakes


def test_scores_cross_the_wire_unchanged():
    scores = [("ok", (ChunkResult(3, 0.1 + 0.2, 1 / 3), ChunkResult(0, 1e-9, 7.25))),
              ("proof_undecodable", ())]
    back = scores_from_wire(json.loads(json.dumps(scores_to_wire(scores))))
    assert back == scores
    assert all(type(r.mant_err_mean) is float for r in back[0][1])


def test_a_request_round_trips():
    rows = [([1, 2, 3], 1, ["a", "b"]), ([9] * 5, 2, [])]
    assert decode_request(encode_request(rows, chunk_tokens=32, topk=128)) == (rows, 32, 128)


def _rows(n, *, length=10, forged=False):
    return [([1] * length + ([fakes.FORGED] if forged else []), 1, ["p"]) for _ in range(n)]


def test_queued_requests_share_one_forward_in_arrival_order():
    calls = []

    def score(rows, chunk_tokens, topk):
        calls.append(len(rows))
        return fakes.score_rows(rows), 1.0, 0.5

    async def scenario():
        batcher = GpuBatcher(score, merge_tokens_limit=10_000, queue_tokens_limit=10_000)
        jobs = [asyncio.ensure_future(batcher.submit(_rows(k + 1), 32, 128)) for k in range(3)]
        await asyncio.sleep(0)
        worker = asyncio.ensure_future(batcher.run())
        out = await asyncio.gather(*jobs)
        worker.cancel()
        return out

    out = asyncio.run(scenario())
    assert calls == [6]
    assert [len(scores) for scores, _, _ in out] == [1, 2, 3]
    # Each request is billed its share of the forward.
    assert sum(f for _, f, _ in out) == pytest.approx(1.0)


def test_merging_stops_at_the_token_budget():
    calls = []

    def score(rows, chunk_tokens, topk):
        calls.append(len(rows))
        return fakes.score_rows(rows), 0.0, 0.0

    async def scenario():
        batcher = GpuBatcher(score, merge_tokens_limit=25, queue_tokens_limit=10_000)
        jobs = [asyncio.ensure_future(batcher.submit(_rows(1), 32, 128)) for _ in range(5)]
        await asyncio.sleep(0)
        worker = asyncio.ensure_future(batcher.run())
        await asyncio.gather(*jobs)
        worker.cancel()

    asyncio.run(scenario())
    assert calls == [2, 2, 1]


def test_one_judges_bad_rows_never_fail_another_judges():
    def score(rows, chunk_tokens, topk):
        if any(fakes.FORGED in tokens for tokens, _, _ in rows):
            raise RuntimeError("CUDA error: device-side assert")
        return fakes.score_rows(rows), 0.0, 0.0

    freed = []

    async def scenario():
        batcher = GpuBatcher(score, merge_tokens_limit=10_000, queue_tokens_limit=10_000,
                             on_error=lambda: freed.append(1))
        good = asyncio.ensure_future(batcher.submit(_rows(2), 32, 128))
        bad = asyncio.ensure_future(batcher.submit(_rows(1, forged=True), 32, 128))
        await asyncio.sleep(0)
        worker = asyncio.ensure_future(batcher.run())
        done = await asyncio.gather(good, bad, return_exceptions=True)
        worker.cancel()
        return done

    good, bad = asyncio.run(scenario())
    assert len(good[0]) == 2
    assert isinstance(bad, RuntimeError)
    assert freed == [1, 1]


def test_a_full_queue_refuses_until_it_drains():
    async def scenario():
        batcher = GpuBatcher(lambda *a: ([], 0.0, 0.0), queue_tokens_limit=15)
        first = asyncio.ensure_future(batcher.submit(_rows(1), 32, 128))
        await asyncio.sleep(0)
        with pytest.raises(GpuBusy):
            await batcher.submit(_rows(1), 32, 128)
        first.cancel()

    asyncio.run(scenario())


def _serve(tmp_path, score, *, start_after=0.0):
    """The GPU app on a unix socket, in a thread with its own loop."""
    path = tmp_path / "gpu.sock"
    loop = asyncio.new_event_loop()

    async def main():
        await asyncio.sleep(start_after)
        batcher = GpuBatcher(score, queue_tokens_limit=10**9)
        await asyncio.gather(serve_unix(build_gpu_app(batcher, {"vocab_size": 7}), path),
                             batcher.run())

    task = loop.create_task(main())

    def run():
        try:
            loop.run_until_complete(task)
        except BaseException as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def stop():
        loop.call_soon_threadsafe(task.cancel)
        thread.join(5)

    return path, stop


def test_the_scorer_gets_the_gpu_process_scores(tmp_path):
    path, stop = _serve(tmp_path, lambda rows, c, k: (fakes.score_rows(rows), 0.25, 0.5))
    try:
        scorer = GpuScorer(path, chunk_tokens=32, topk=128, retry_seconds=0.05)
        rows = _rows(2) + _rows(1, forged=True)
        scores, forward, verify = asyncio.run(scorer(rows))
    finally:
        stop()
    assert scores == fakes.score_rows(rows)
    assert (forward, verify) == (0.25, 0.5)


def test_an_audit_error_comes_back_as_the_forward_raised_it(tmp_path):
    def score(rows, chunk_tokens, topk):
        if len(rows) > 1:
            raise ValueError("prompt_len 0 leaves no completion")
        raise KeyError("bug")

    path, stop = _serve(tmp_path, score)
    try:
        scorer = GpuScorer(path, chunk_tokens=32, topk=128, retry_seconds=0.05)
        with pytest.raises(ValueError, match="prompt_len"):
            asyncio.run(scorer(_rows(2)))
        with pytest.raises(GpuScoreError, match="KeyError"):
            asyncio.run(scorer(_rows(1)))
    finally:
        stop()


def test_a_gpu_process_that_is_down_is_waited_for(tmp_path):
    path, stop = _serve(tmp_path, lambda rows, c, k: (fakes.score_rows(rows), 0.0, 0.0),
                        start_after=1.0)
    try:
        scorer = GpuScorer(path, chunk_tokens=32, topk=128, retry_seconds=0.1,
                           max_retry_seconds=0.2)
        scores, _, _ = asyncio.run(scorer(_rows(1)))
    finally:
        stop()
    assert scores == fakes.score_rows(_rows(1)) and scorer.retries >= 3
