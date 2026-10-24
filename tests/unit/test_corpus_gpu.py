"""The GPU process: one FIFO for every judge, merged forwards, isolated
failures, back-pressure; and the judges' client, which waits out a GPU
process that is down instead of failing an audit."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

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
    rows = [([1, 2, 151645], 1, ["QUJD", "RA=="]), ([9] * 5, 2, []), ([2**31 - 1, 0], 1, ["x"])]
    assert decode_request(encode_request(rows, chunk_tokens=32, topk=128)) == (rows, 32, 128)
    assert decode_request(encode_request([], chunk_tokens=32, topk=128)) == ([], 32, 128)


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


@pytest.mark.parametrize("request_count", [1, 3])
def test_failed_forward_tensors_are_freed_before_cleanup_and_retries(request_count, monkeypatch):
    import torch

    class _SimulatedOOM(RuntimeError):
        pass

    tensors, errors, calls, freed, fatal, forwards = [], [], [], [], [], []

    def score(rows, chunk_tokens, topk):
        assert all(ref() is None for ref in tensors)
        calls.append(len(rows))
        if len(calls) <= (2 if request_count > 1 else 1):
            # Both the failing frame and its chained cause can own activations.
            try:
                inner_tensor = torch.zeros(1)
                tensors.append(weakref.ref(inner_tensor))
                raise ValueError("forward allocation failed")
            except ValueError as cause:
                tensor = torch.zeros(1)
                tensors.append(weakref.ref(tensor))
                error = _SimulatedOOM("CUDA out of memory")
                errors.append(error)
                raise error from cause
        return fakes.score_rows(rows), 0.25, 0.5

    def free_cuda():
        assert all(ref() is None for ref in tensors)
        assert sys.exc_info() == (None, None, None)
        freed.append(1)

    async def scenario(executor):
        loop = asyncio.get_running_loop()
        run_in_executor = loop.run_in_executor

        def retain_forward(*args):
            future = run_in_executor(*args)
            forwards.append(future)
            return future

        # The awaiting Future can retain its own copy of the traceback.
        monkeypatch.setattr(loop, "run_in_executor", retain_forward)
        batcher = GpuBatcher(score, executor=executor, merge_tokens_limit=10_000,
                             queue_tokens_limit=10_000, on_error=free_cuda,
                             on_fatal=fatal.append)
        jobs = [asyncio.create_task(batcher.submit(_rows(k + 1), 32, 128))
                for k in range(request_count)]
        await asyncio.sleep(0)
        worker = asyncio.create_task(batcher.run())
        try:
            gathered = asyncio.gather(*jobs, return_exceptions=True)
            done, _ = await asyncio.wait((worker, gathered), timeout=5,
                                         return_when=asyncio.FIRST_COMPLETED)
            if worker in done:
                await worker
            assert gathered in done
            results = await gathered
            # Keep the failed Future and original exception alive while the
            # next request scores: neither may retain the failed activations.
            assert jobs[0].exception() is errors[-1]
            assert results[0] is errors[-1]
            assert type(results[0]) is _SimulatedOOM
            assert str(results[0]) == "CUDA out of memory"
            assert results[0].__cause__ is results[0].__context__ is None
            later = await asyncio.wait_for(batcher.submit(_rows(1), 32, 128), timeout=5)
            return results, later
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    with ThreadPoolExecutor(1) as executor:
        results, later = asyncio.run(scenario(executor))
    assert calls == ([1, 1] if request_count == 1 else [6, 1, 2, 3, 1])
    assert len(forwards) == len(calls) and all(future.done() for future in forwards)
    assert freed == [1] * len(errors)
    assert fatal == []
    assert all(ref() is None for ref in tensors)
    assert later == (fakes.score_rows(_rows(1)), 0.25, 0.5)
    if request_count > 1:
        assert [result[0] for result in results[1:]] == [
            fakes.score_rows(_rows(2)), fakes.score_rows(_rows(3))]


@pytest.mark.parametrize("explicit_cause", [False, True])
def test_forward_cleanup_releases_chained_frames_without_garbage_collection(explicit_cause):
    import gc
    import torch

    tensors, freed = [], []

    def allocate():
        tensor = torch.zeros(1)
        tensors.append(weakref.ref(tensor))
        error = ValueError("forward allocation failed")
        raise error  # Its traceback and this local form a reference cycle.

    def score(rows, chunk_tokens, topk):
        try:
            allocate()
        except ValueError as cause:
            if explicit_cause:
                raise RuntimeError("CUDA out of memory") from cause
            raise RuntimeError("CUDA out of memory")

    async def scenario(executor):
        batcher = GpuBatcher(score, executor=executor,
                             on_error=lambda: freed.append(all(ref() is None for ref in tensors)))
        worker = asyncio.create_task(batcher.run())
        try:
            failed = asyncio.create_task(batcher.submit(_rows(1), 32, 128))
            (error,) = await asyncio.wait_for(
                asyncio.gather(failed, return_exceptions=True), timeout=5)
            assert type(error) is RuntimeError and str(error) == "CUDA out of memory"
            assert failed.exception() is error
            assert freed == [True]
            assert len(tensors) == 1 and tensors[0]() is None
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    enabled = gc.isenabled()
    gc.disable()  # Incidental collection must not hide retained activations.
    try:
        with ThreadPoolExecutor(1) as executor:
            asyncio.run(scenario(executor))
    finally:
        if enabled:
            gc.enable()


def test_info_reports_forward_failure_until_scoring_recovers():
    import httpx

    calls = []
    info = {"vocab_size": 7, "model_id": "org/Frozen", "model_revision": "abc123"}

    def score(rows, chunk_tokens, topk):
        if not rows:
            return [], 0.0, 0.0
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return fakes.score_rows(rows), 0.25, 0.5

    async def scenario(executor):
        batcher = GpuBatcher(score, executor=executor)
        worker = asyncio.create_task(batcher.run())
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(
                    app=build_gpu_app(batcher, info)), base_url="http://gpu") as client:
                assert (await client.get("/info")).json() == {**info, "ready": True}
                body = encode_request(_rows(1), chunk_tokens=32, topk=128)
                failed = await client.post("/score", content=body)
                assert failed.status_code == 200
                assert failed.json() == {"error": "CUDA out of memory", "kind": "RuntimeError",
                                         "value_error": False, "runtime_error": True}
                assert (await client.get("/info")).json() == {**info, "ready": False}
                empty = await client.post("/score", content=encode_request([], chunk_tokens=32, topk=128))
                assert empty.status_code == 200 and empty.json()["scores"] == []
                assert (await client.get("/info")).json() == {**info, "ready": False}
                recovered = await client.post("/score", content=body)
                assert scores_from_wire(recovered.json()["scores"]) == fakes.score_rows(_rows(1))
                assert (await client.get("/info")).json() == {**info, "ready": True}
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    with ThreadPoolExecutor(1) as executor:
        asyncio.run(scenario(executor))
    assert "ready" not in info


def test_info_recovers_after_a_merged_forward_falls_back_successfully():
    import httpx

    calls, failed_ready = [], []

    def score(rows, chunk_tokens, topk):
        calls.append(len(rows))
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return fakes.score_rows(rows), 0.25, 0.5

    async def scenario(executor):
        batcher = GpuBatcher(score, executor=executor, merge_tokens_limit=10_000,
                             on_error=lambda: failed_ready.append(batcher.ready))
        jobs = [asyncio.create_task(batcher.submit(_rows(k), 32, 128)) for k in (1, 2)]
        await asyncio.sleep(0)
        worker = asyncio.create_task(batcher.run())
        try:
            results = await asyncio.wait_for(asyncio.gather(*jobs), timeout=5)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(
                    app=build_gpu_app(batcher, {})), base_url="http://gpu") as client:
                assert (await client.get("/info")).json() == {"ready": True}
            return results
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    with ThreadPoolExecutor(1) as executor:
        results = asyncio.run(scenario(executor))
    assert calls == [3, 1, 2]
    assert failed_ready == [False]
    assert [result[0] for result in results] == [fakes.score_rows(_rows(1)), fakes.score_rows(_rows(2))]


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
        except BaseException:  # noqa: BLE001
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


# -- review fixes: sticky CUDA faults, error kinds, empty proofs ---------------


class AcceleratorError(RuntimeError):
    """What torch raises after an illegal memory access (a RuntimeError subclass)."""


def test_an_empty_proof_crosses_as_one_proof():
    rows = [([1, 2, 3], 1, [""]), ([4, 5], 1, ["", "QQ=="])]
    assert decode_request(encode_request(rows, chunk_tokens=32, topk=128)) == (rows, 32, 128)


def test_a_body_whose_proof_count_does_not_match_is_refused():
    body = encode_request([([1, 2], 1, ["a", "b"])], chunk_tokens=32, topk=128)
    with pytest.raises(ValueError):
        decode_request(body.replace(b'"items": [[2, 1, 2]]', b'"items": [[2, 1, 3]]'))


def test_a_cuda_fault_makes_the_gpu_process_exit_for_a_reload():
    fatal = []

    def score(rows, chunk_tokens, topk):
        raise AcceleratorError("CUDA error: an illegal memory access was encountered")

    async def scenario():
        batcher = GpuBatcher(score, queue_tokens_limit=10_000, on_fatal=fatal.append)
        worker = asyncio.ensure_future(batcher.run())
        with pytest.raises(AcceleratorError):
            await batcher.submit(_rows(1), 32, 128)
        worker.cancel()

    asyncio.run(scenario())
    assert len(fatal) == 1 and "illegal memory access" in fatal[0]


def test_repeated_forward_failures_make_it_exit_too_but_not_out_of_memory():
    fatal = []
    errors = iter([RuntimeError("CUDA out of memory")] * 5 + [RuntimeError("cuBLAS failed")] * 3)

    def score(rows, chunk_tokens, topk):
        raise next(errors)

    async def scenario():
        batcher = GpuBatcher(score, queue_tokens_limit=10_000, on_fatal=fatal.append,
                             fatal_after=3)
        worker = asyncio.ensure_future(batcher.run())
        for _ in range(8):
            with pytest.raises(RuntimeError):
                await batcher.submit(_rows(1), 32, 128)
            if _ == 4:
                assert fatal == []          # five OOMs: a batch too big, not a broken card
        worker.cancel()

    asyncio.run(scenario())
    assert len(fatal) == 1


def test_a_runtime_error_subclass_is_an_audit_error_on_the_judge(tmp_path):
    def score(rows, chunk_tokens, topk):
        raise AcceleratorError("CUDA error: unspecified launch failure")

    path, stop = _serve(tmp_path, score)
    try:
        scorer = GpuScorer(path, chunk_tokens=32, topk=128, retry_seconds=0.05)
        with pytest.raises(RuntimeError) as raised:
            asyncio.run(scorer(_rows(1)))
    finally:
        stop()
    # Counted by the auditor as a validator-side error (5 in a row halt it),
    # never the silent crash-and-retry of GpuScoreError.
    assert not isinstance(raised.value, GpuScoreError)
    assert "launch failure" in str(raised.value)


def test_a_request_cancelled_while_queued_does_not_kill_the_worker():
    calls = []

    def score(rows, chunk_tokens, topk):
        calls.append(len(rows))
        return fakes.score_rows(rows), 0.0, 0.0

    async def scenario():
        batcher = GpuBatcher(score, merge_tokens_limit=10_000, queue_tokens_limit=10_000)
        gone = asyncio.ensure_future(batcher.submit(_rows(1), 32, 128))
        kept = asyncio.ensure_future(batcher.submit(_rows(2), 32, 128))
        await asyncio.sleep(0)
        gone.cancel()
        worker = asyncio.ensure_future(batcher.run())
        scores, _, _ = await kept
        later = await batcher.submit(_rows(1), 32, 128)
        worker.cancel()
        return scores, later

    scores, later = asyncio.run(scenario())
    assert len(scores) == 2 and len(later[0]) == 1


# -- I3: an episode request is retried on 503 with backoff, and fits the queue ----


def test_a_64k_episode_request_fits_a_queue_a_2m_one_does_not():
    from reliquary.validator.corpus_auditor import EPISODE_GPU_REQUEST_TOKENS

    async def scenario():
        limit = 16 * 131_072
        batcher = GpuBatcher(lambda *a: ([], 0.0, 0.0), queue_tokens_limit=limit)
        # Single-turn judges keep about 1.9M tokens queued.
        held = [asyncio.ensure_future(batcher.submit([([1] * 95_000, 1, ["p"])], 32, 128))
                for _ in range(20)]
        await asyncio.sleep(0)
        episode = [([1] * (EPISODE_GPU_REQUEST_TOKENS - 1), 64, ["p"], [(64, 1000)])]
        accepted = asyncio.ensure_future(batcher.submit(episode, 32, 128))
        await asyncio.sleep(0)
        assert not accepted.done()                     # queued, not refused
        with pytest.raises(GpuBusy):                   # what a 2M-token pass got
            await batcher.submit([([1] * 2_000_000, 64, ["p"], [(64, 1000)])], 32, 128)
        for job in held + [accepted]:
            job.cancel()

    asyncio.run(scenario())


def test_an_episode_request_refused_503_is_retried_with_backoff():
    import httpx

    answers = [httpx.Response(503), httpx.Response(503),
               httpx.Response(200, json={"scores": [["ok", [[0, 0.0125, 0.01]]]],
                                         "forward_seconds": 0.5, "verify_seconds": 0.1})]
    bodies, sleeps = [], []

    class _Client:
        async def post(self, path, content, headers):
            bodies.append(content)
            return answers.pop(0)

    async def sleep(seconds):
        sleeps.append(seconds)

    scorer = GpuScorer("/nonexistent", chunk_tokens=32, topk=128, retry_seconds=1.0,
                       max_retry_seconds=10.0, sleep=sleep)
    scorer._http = lambda: _Client()
    rows = [([1] * 200, 64, ["p"], [(64, 120), (130, 200)])]
    scores, forward, _ = asyncio.run(scorer(rows))
    assert scores == [("ok", (ChunkResult(0, 0.0125, 0.01),))] and forward == 0.5
    assert sleeps == [1.0, 2.0] and scorer.retries == 2      # backoff, never a failure
    # Every retry carries the trajectory's spans.
    assert all(decode_request(b)[0] == rows for b in bodies)
