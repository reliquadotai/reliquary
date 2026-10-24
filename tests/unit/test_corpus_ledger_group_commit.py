"""Group commit on the corpus ledger: submissions and skips that queue behind
a ledger turn share the next one's read and its one compare-and-swap.

Money-path, so the claim tested is sequential equivalence: the same requests,
decided one turn each (the pre-batching path) or queued into shared turns in
the same order, get the same verdicts and leave a byte-identical ledger and
the same sealed segments."""

from __future__ import annotations

import asyncio
import json
import math

import pytest
from fastapi import HTTPException

from reliquary.corpus.checks import completion_digest
from reliquary.corpus.walk import job_walk_index
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.protocol.corpus_submission import CorpusSkipRequest, CorpusSubmissionRequest
from tests.unit.test_corpus_route_skip import _declare, _router, _seed
from tests.unit.test_corpus_service import (  # noqa: F401  (fixtures)
    CHECKPOINT,
    EOS,
    _CountingStore,
    _faithful_prompt,
    _r2_client,
    _text_for,
    fake_r2,
    seeded_job,
)

FREE = "free-v1"
WALK = "walk-v1"
LEDGER_KEY = "reliquary/corpus/jobs/{}/ledgers.json"


def _tokens(filler):
    return [filler] * 16 + [EOS]


def _submit(job_id, hotkey, prompt_index, filler, *, cursor=0, checkpoint=CHECKPOINT):
    tokens = _tokens(filler)
    return CorpusSubmissionRequest(
        job_id=job_id, miner_hotkey=hotkey, cursor=cursor, prompt_index=prompt_index,
        checkpoint_sha256=checkpoint, rendered_prompt=_faithful_prompt(prompt_index),
        completions=[{"tokens": tokens, "text": _text_for(tokens)}], signature="ok",
    )


def _skip(job, hotkey, cursor, to_cursor):
    return CorpusSkipRequest(
        job_id=job.job_id, miner_hotkey=hotkey, cursor=cursor,
        prompt_index=job_walk_index(job, hotkey, cursor), to_cursor=to_cursor, signature="ok",
    )


async def _call(router, request):
    try:
        if isinstance(request, CorpusSkipRequest):
            return ("skip", (await router.skip_corpus(request)).model_dump(mode="json"))
        return ("submit", (await router.submit_corpus(request)).model_dump(mode="json"))
    except HTTPException as exc:
        return ("error", exc.status_code, exc.detail)
    except Exception as exc:  # a decision that raised, as the old path raised it
        return ("raised", type(exc).__name__, str(exc))


async def _one_at_a_time(router, requests):
    return [await _call(router, request) for request in requests]


async def _queued_in_order(router, requests):
    """Hold the ledger turn, queue every request in this order, then let the
    turns run: what a burst arriving during one slow write does."""
    await router.ledger_lock.acquire()
    tasks = []
    try:
        for request in requests:
            waiting = router.ledger_waiting()
            task = asyncio.ensure_future(_call(router, request))
            tasks.append(task)
            # Queued, or answered before the turn (a refusal that never queues).
            while router.ledger_waiting() == waiting and not task.done():
                await asyncio.sleep(0.001)
    finally:
        router.ledger_lock.release()
    return list(await asyncio.gather(*tasks))


def _bucket(client):
    """Everything the bucket holds but the manifests: the ledger bytes and
    every sealed segment."""
    return {key: body for key, (body, _) in client.objects.items() if not key.endswith("job.json")}


def _run(fake_r2, client, seeded_job, setup, requests, mode, **router_kwargs):
    client.objects.clear()
    setup()
    router = _router(seeded_job, router_kwargs.pop("job_id"), **router_kwargs)
    runner = _one_at_a_time if mode == "sequential" else _queued_in_order
    before = seeded_job.ledger_writes()
    verdicts = asyncio.run(runner(router, requests))
    _run.writes = seeded_job.ledger_writes() - before
    return verdicts, _bucket(client)


# --------------------------------------------------------------------------
# a free job: accepts, duplicates, full prompts, seals inside the turn
# --------------------------------------------------------------------------


def _free_setup(fake_r2):
    def setup():
        _declare(fake_r2, FREE, prompt_order="free", slots_per_prompt=2)
        # Sealed before the turn, so a duplicate of it is found in a segment.
        sealed = completion_digest(900, _tokens(9))
        snapshot, etag = asyncio.run(job_store.read_ledgers(FREE, **fake_r2))
        segment = asyncio.run(job_store.write_seen_segment(FREE, [sealed], **fake_r2))
        asyncio.run(job_store.write_ledgers(FREE, {
            "schema": "reliquary/corpus-ledgers/v2", "slots": {"900": 1}, "cursors": {},
            "seen_pending": [completion_digest(901, _tokens(9))],
            "seen_segments": [{"id": segment, "count": 1}],
        }, etag, **fake_r2))
    return setup


def _free_requests():
    requests = [
        _submit(FREE, "5A", 0, 1),                      # accepted
        _submit(FREE, "5B", 0, 2),                      # accepted: prompt 0 now full
        _submit(FREE, "5C", 0, 3),                      # prompt_full, moves nothing
        _submit(FREE, "5C", 0, 1),                      # duplicate of a pending digest
        _submit(FREE, "5D", 900, 9),                    # duplicate of a segment's digest
        _submit(FREE, "5D", 901, 9),                    # duplicate pending before the turn
        _submit(FREE, "5E", 5, 1, checkpoint="f" * 64),  # checkpoint_mismatch
    ]
    # Enough accepts to cross the seal threshold several times inside a turn.
    requests += [_submit(FREE, f"5H{i % 3}", 10 + i, 4) for i in range(11)]
    # A digest sealed earlier in the same turn, sent again.
    requests.append(_submit(FREE, "5Z", 10, 4))
    requests += [_submit(FREE, "5Y", 30, 5), _submit(FREE, "5Y", 30, 5)]
    return requests


@pytest.mark.parametrize("batch_max", [64, 3, 1])
def test_a_free_job_decides_a_queued_burst_exactly_as_one_turn_each(
    fake_r2, _r2_client, seeded_job, batch_max
):
    kwargs = dict(job_id=FREE, seal_threshold=4, segment_max=3)
    setup, requests = _free_setup(fake_r2), _free_requests()
    one, one_bucket = _run(fake_r2, _r2_client, seeded_job, setup, requests, "sequential", **kwargs)
    sequential_writes = _run.writes
    many, many_bucket = _run(fake_r2, _r2_client, seeded_job, setup, requests, "queued",
                             ledger_batch_max=batch_max, **kwargs)

    assert many == one
    assert many_bucket == one_bucket
    # One write per moving request alone; at most one per turn batched (a
    # turn of refusals that move nothing writes nothing).
    if batch_max == 1:
        assert _run.writes == sequential_writes > 10
    else:
        assert 1 <= _run.writes <= math.ceil(len(requests) / batch_max)
    # The scenario really covers what it claims.
    reasons = [verdict[1]["reason"] for verdict in one]
    for reason in ("accepted", "prompt_full", "hash_duplicate", "checkpoint_mismatch"):
        assert reason in reasons
    assert reasons.count("hash_duplicate") == 5
    ledger = json.loads(one_bucket[LEDGER_KEY.format(FREE)])
    assert len(ledger["seen_segments"]) >= 4


# --------------------------------------------------------------------------
# a walk job: cursors, prompt_full that moves one, skips
# --------------------------------------------------------------------------


def _walk_run(walk, hotkey, length):
    return [job_walk_index(walk, hotkey, cursor) for cursor in range(length)]


def _walk_setup(fake_r2):
    def setup():
        walk = _declare(fake_r2, WALK)
        full = {}
        # 5S's first three positions are full: its skip crosses them.
        for index in _walk_run(walk, "5S", 3):
            full[index] = 1
        # 5F's first position is full: its submission there is prompt_full.
        full[job_walk_index(walk, "5F", 0)] = 1
        _seed(fake_r2, WALK, slots=full, cursors={"5S": 0})
    return setup


def _walk_requests(walk):
    a = _walk_run(walk, "5A", 3)
    s = _walk_run(walk, "5S", 4)
    if len(set(a) | set(s)) < 7 or job_walk_index(walk, "5F", 0) in set(a) | set(s):
        pytest.skip("walks collide")
    return [
        _submit(WALK, "5A", a[0], 1, cursor=0),           # accepted, cursor 0 -> 1
        _submit(WALK, "5A", a[0], 2, cursor=0),           # bad_cursor
        _submit(WALK, "5A", a[1], 1, cursor=1),           # accepted, cursor 1 -> 2
        _submit(WALK, "5F", job_walk_index(walk, "5F", 0), 1),  # prompt_full, cursor moves
        _skip(walk, "5S", 0, 3),                          # accepted skip, cursor 0 -> 3
        _submit(WALK, "5S", s[3], 1, cursor=3),           # accepted after the skip
        _skip(walk, "5T", 0, 1),                          # prompt_not_full
        _skip(walk, "5T", 0, 10_000),                     # malformed (past the bound)
        _submit(WALK, "5A", a[2], 1, cursor=2),           # accepted, seals at threshold
        _submit(WALK, "5A", a[2], 1, cursor=5),           # bad_cursor
    ]


@pytest.mark.parametrize("batch_max", [64, 2])
def test_a_walk_job_decides_a_queued_burst_exactly_as_one_turn_each(
    fake_r2, _r2_client, seeded_job, batch_max
):
    walk = _declare(fake_r2, WALK)
    kwargs = dict(job_id=WALK, seal_threshold=3, segment_max=2)
    setup, requests = _walk_setup(fake_r2), _walk_requests(walk)
    one, one_bucket = _run(fake_r2, _r2_client, seeded_job, setup, requests, "sequential", **kwargs)
    many, many_bucket = _run(fake_r2, _r2_client, seeded_job, setup, requests, "queued",
                             ledger_batch_max=batch_max, **kwargs)

    assert many == one
    assert many_bucket == one_bucket
    assert 1 <= _run.writes <= math.ceil(len(requests) / batch_max)
    reasons = [verdict[1]["reason"] for verdict in one]
    assert reasons == ["accepted", "bad_cursor", "accepted", "prompt_full", "accepted",
                       "accepted", "prompt_not_full", "malformed_submission", "accepted",
                       "bad_cursor"]
    assert one[4][1]["skipped"] is True and one[4][1]["cursor"] == 3
    ledger = json.loads(one_bucket[LEDGER_KEY.format(WALK)])
    assert ledger["cursors"] == {"5A": 3, "5F": 1, "5S": 4}
    assert ledger["seen_segments"]


def test_a_decision_that_raises_fails_alone_and_moves_nothing(
    fake_r2, _r2_client, seeded_job, monkeypatch
):
    """It consumed a slot before raising: the turn decides the others again on
    a fresh copy, as the old path's next read would have."""
    from reliquary.validator import corpus_service

    real_admit = corpus_service.admit

    def admit(job, **kwargs):
        verdict = real_admit(job, **kwargs)
        if kwargs["hotkey"] == "5Boom":
            raise RuntimeError("boom")
        return verdict

    monkeypatch.setattr(corpus_service, "admit", admit)
    kwargs = dict(job_id=FREE, seal_threshold=4, segment_max=3)
    setup = _free_setup(fake_r2)
    requests = _free_requests()
    boom = _submit(FREE, "5Boom", 0, 7)
    with_boom = requests[:1] + [boom] + requests[1:]
    one, one_bucket = _run(fake_r2, _r2_client, seeded_job, setup, with_boom, "sequential",
                           **kwargs)
    many, many_bucket = _run(fake_r2, _r2_client, seeded_job, setup, with_boom, "queued",
                             **kwargs)

    assert one[1] == many[1] == ("raised", "RuntimeError", "boom")
    assert many == one
    assert many_bucket == one_bucket
    # And it is as if it was never sent.
    clean, clean_bucket = _run(fake_r2, _r2_client, seeded_job, setup, requests, "sequential",
                               **kwargs)
    assert one[:1] + one[2:] == clean
    assert one_bucket == clean_bucket


# --------------------------------------------------------------------------
# what batching buys, and what it keeps
# --------------------------------------------------------------------------


class _SlowStore(_CountingStore):
    """A bucket round trip that takes real time, so concurrent requests pile
    up behind a turn as they do in production."""

    async def read_ledgers(self, job_id):
        await asyncio.sleep(0.002)
        return await super().read_ledgers(job_id)

    async def write_ledgers(self, job_id, snapshot, etag):
        await asyncio.sleep(0.02)
        return await super().write_ledgers(job_id, snapshot, etag)


@pytest.mark.parametrize("batch_max", [64, 8])
def test_fifty_concurrent_submissions_share_a_handful_of_writes(
    fake_r2, seeded_job, batch_max
):
    _declare(fake_r2, FREE, prompt_order="free", slots_per_prompt=2)
    store = _SlowStore(fake_r2)
    seeded_job.store = store
    router = _router(seeded_job, FREE, ledger_batch_max=batch_max)
    # 50 submissions over 25 prompts of two slots, plus 5 that find theirs full.
    requests = [_submit(FREE, f"5H{i}", i // 2, i + 1) for i in range(50)]
    requests += [_submit(FREE, f"5X{i}", i, 100 + i) for i in range(5)]

    async def burst():
        first = await asyncio.gather(*(_call(router, r) for r in requests[:50]))
        late = await asyncio.gather(*(_call(router, r) for r in requests[50:]))
        return first + late

    verdicts = asyncio.run(burst())

    assert [v[1]["reason"] for v in verdicts] == ["accepted"] * 50 + ["prompt_full"] * 5
    assert sorted(v[1]["slots_remaining"] for v in verdicts[:50]) == [0] * 25 + [1] * 25
    # One write per turn, not per submission; the full ones cost none.
    assert store.ledger_write_attempts <= math.ceil(50 / batch_max) + 2
    ledger, _ = asyncio.run(job_store.read_ledgers(FREE, **fake_r2))
    assert ledger["slots"] == {str(i): 2 for i in range(25)}
    assert len(ledger["seen_pending"]) == 50


def test_a_lost_compare_and_swap_decides_the_whole_batch_again(fake_r2, seeded_job):
    _declare(fake_r2, FREE, prompt_order="free", slots_per_prompt=1)
    store = _CountingStore(fake_r2)
    seeded_job.store = store
    router = _router(seeded_job, FREE)
    # The competing write takes prompt 1, which this batch also wants.
    competing = {"schema": "reliquary/corpus-ledgers/v2", "slots": {"1": 1}, "cursors": {},
                 "seen_pending": [], "seen_segments": []}
    seeded_job.fail_next_ledger_write_with_conflict(competing_snapshot=competing)
    requests = [_submit(FREE, "5A", 0, 1), _submit(FREE, "5B", 1, 2), _submit(FREE, "5C", 2, 3)]

    verdicts = asyncio.run(_queued_in_order(router, requests))

    assert [v[1]["reason"] for v in verdicts] == ["accepted", "prompt_full", "accepted"]
    assert store.ledger_write_attempts == 2
    ledger, _ = asyncio.run(job_store.read_ledgers(FREE, **fake_r2))
    assert ledger["slots"] == {"1": 1, "0": 1, "2": 1}


def test_a_batch_that_stays_contended_answers_every_member_503(fake_r2, seeded_job):
    _declare(fake_r2, FREE, prompt_order="free", slots_per_prompt=1)
    seeded_job.store = _CountingStore(fake_r2)
    router = _router(seeded_job, FREE, max_write_attempts=2)
    seeded_job.fail_next_ledger_write_with_conflict(times=100)
    requests = [_submit(FREE, f"5H{i}", i, i + 1) for i in range(4)]

    verdicts = asyncio.run(_queued_in_order(router, requests))

    assert verdicts == [("error", 503, "corpus_ledger_contention")] * 4
    assert seeded_job.store.ledger_write_attempts == 2
    ledger, _ = asyncio.run(job_store.read_ledgers(FREE, **fake_r2))
    assert not ledger.get("slots")


def test_a_batch_of_refusals_that_move_nothing_writes_nothing(fake_r2, seeded_job):
    _declare(fake_r2, FREE, prompt_order="free", slots_per_prompt=1)
    _seed(fake_r2, FREE, slots={0: 1})
    seeded_job.store = _CountingStore(fake_r2)
    router = _router(seeded_job, FREE)
    requests = [_submit(FREE, f"5H{i}", 0, i + 1) for i in range(5)]
    requests.append(_submit(FREE, "5Z", 3, 1, checkpoint="f" * 64))

    verdicts = asyncio.run(_queued_in_order(router, requests))

    assert [v[1]["reason"] for v in verdicts] == ["prompt_full"] * 5 + ["checkpoint_mismatch"]
    assert seeded_job.store.ledger_write_attempts == 0


def test_a_request_queued_past_the_timeout_is_withdrawn_not_decided_later(
    fake_r2, seeded_job
):
    _declare(fake_r2, FREE, prompt_order="free", slots_per_prompt=1)
    seeded_job.store = _CountingStore(fake_r2)
    router = _router(seeded_job, FREE, ledger_lock_timeout=0.05)

    async def scenario():
        await router.ledger_lock.acquire()
        try:
            answer = await _call(router, _submit(FREE, "5A", 0, 1))
            assert router.ledger_waiting() == 0
        finally:
            router.ledger_lock.release()
        # Let any turn that would still run, run.
        await asyncio.sleep(0.05)
        return answer

    assert asyncio.run(scenario()) == ("error", 503, "corpus_ledger_contention")
    assert seeded_job.store.ledger_write_attempts == 0


def test_a_cancelled_taken_submission_blocks_retirement_until_its_record_is_durable(
    fake_r2, seeded_job
):
    """A disconnect does not let retirement overtake either durable write."""
    from reliquary.infrastructure import corpus_record_store
    from tests.unit.test_corpus_hot_jobs import _Harness, _hot_entry

    _declare(fake_r2, FREE, prompt_order="free", slots_per_prompt=1)
    writing, release = asyncio.Event(), asyncio.Event()
    recording, record_release, durable = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class _HeldStore(_CountingStore):
        async def write_ledgers(self, job_id, snapshot, etag):
            writing.set()
            await release.wait()
            return await super().write_ledgers(job_id, snapshot, etag)

    class _HeldRecords:
        written = []

        async def write_submission(self, job_id, submission_id, record):
            recording.set()
            await record_release.wait()
            result = await corpus_record_store.write_submission(
                job_id, submission_id, record, **fake_r2,
            )
            self.written.append(submission_id)
            durable.set()
            return result

    seeded_job.store = _HeldStore(fake_r2)
    records, announced = _HeldRecords(), []
    router = _router(seeded_job, FREE, records=records, on_accepted=announced.append)

    async def scenario():
        h = _Harness([_hot_entry(job_id=FREE)])
        h.set._router_for = lambda wiring: router

        async def final_status(job_id, *, drained=False):
            return {"job_id": job_id, "state": "drained" if drained else "open"}

        h.set._compute_status = final_status
        await h.set.refresh()
        await asyncio.sleep(0)
        async with h.client() as client:
            handler = asyncio.create_task(client.post(
                f"/corpus/jobs/{FREE}/submit",
                json=_submit(FREE, "5A", 0, 1).model_dump(mode="json"),
            ))
            await asyncio.wait_for(writing.wait(), 2)
            handler.cancel()
            with pytest.raises(asyncio.CancelledError):
                await handler
            assert h.routes.in_flight[FREE] == 0
            assert h.routes.admission_pending(FREE)
            # Even a drain read that sees no records cannot release the job.
            h.drained[FREE] = True
            h.entries = [_hot_entry(job_id=FREE, status="retired")]
            await h.set.refresh()
            await h.set.refresh()
            assert FREE in h.set.served and h.cancelled == []

            release.set()
            await asyncio.wait_for(recording.wait(), 2)
            ledger, _ = await job_store.read_ledgers(FREE, **fake_r2)
            assert ledger["slots"] == {"0": 1} and records.written == []
            await h.set.refresh()
            assert h.routes.admission_pending(FREE)
            assert FREE in h.set.served and h.cancelled == []

            record_release.set()
            await asyncio.wait_for(durable.wait(), 2)
            for _ in range(200):
                if not h.routes.admission_pending(FREE):
                    break
                await asyncio.sleep(0.005)
            assert not h.routes.admission_pending(FREE)
            stored = await corpus_record_store.read_submission(FREE, records.written[0], **fake_r2)
            assert stored["prompt_index"] == 0
            await h.set.refresh()
            await asyncio.sleep(0)
            assert FREE not in h.set.served and FREE in h.set.finished
            assert sorted(h.cancelled) == [f"audit:{FREE}", f"settle:{FREE}"]

    asyncio.run(scenario())
    ledger, _ = asyncio.run(job_store.read_ledgers(FREE, **fake_r2))
    assert ledger["slots"] == {"0": 1}
    assert len(records.written) == 1 and announced == records.written


def test_a_corrupt_ledger_found_inside_a_decision_is_named(fake_r2, seeded_job, monkeypatch):
    from reliquary.validator import corpus_service

    real_admit = corpus_service.admit

    def admit(job, **kwargs):
        if kwargs["hotkey"] == "5Bad":
            raise corpus_service.LedgerSnapshotError("corrupt")
        return real_admit(job, **kwargs)

    monkeypatch.setattr(corpus_service, "admit", admit)
    _declare(fake_r2, FREE, prompt_order="free", slots_per_prompt=1)
    router = _router(seeded_job, FREE)
    requests = [_submit(FREE, "5A", 0, 1), _submit(FREE, "5Bad", 1, 2), _submit(FREE, "5C", 2, 3)]

    verdicts = asyncio.run(_queued_in_order(router, requests))

    assert verdicts[1] == ("error", 500, "corpus_ledger_corrupt")
    assert [verdicts[0][1]["reason"], verdicts[2][1]["reason"]] == ["accepted", "accepted"]
