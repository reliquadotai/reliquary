"""The submit route shares one event loop with the auditor: its ledger work
must not block that loop, and its own concurrent requests must not race each
other's compare-and-swap."""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from fastapi import FastAPI

from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
from tests.unit.test_corpus_service import (  # noqa: F401 - fixtures
    CHECKPOINT,
    EOS,
    _CountingStore,
    _faithful_prompt,
    _r2_client,
    _Renderer,
    _Tokenizer,
    _text_for,
    fake_r2,
    seeded_job,
)


class _YieldingStore(_CountingStore):
    """Suspends inside every store call, as a real bucket round trip does, so
    concurrent requests genuinely interleave; counts lost races."""

    def __init__(self, client_kwargs):
        super().__init__(client_kwargs)
        self.conflicts = 0

    async def read_ledgers(self, job_id):
        await asyncio.sleep(0.001)
        return await super().read_ledgers(job_id)

    async def write_ledgers(self, job_id, snapshot, etag):
        await asyncio.sleep(0.001)
        try:
            return await super().write_ledgers(job_id, snapshot, etag)
        except job_store.CorpusStoreConflict:
            self.conflicts += 1
            raise


def _app(seeded_job, store):
    from reliquary.validator.corpus_service import build_corpus_router

    app = FastAPI()
    app.include_router(
        build_corpus_router(
            job_id="swe-v1",
            store=store,
            tokenizer=_Tokenizer(),
            renderer=_Renderer(),
            verify_signature=lambda request: True,
            prompt_job_for=seeded_job.prompt_job_for,
        )
    )
    return app


def _request(hotkey, prompt_index):
    tokens = [7] * 16 + [EOS]
    return CorpusSubmissionRequest(
        job_id="swe-v1",
        miner_hotkey=hotkey,
        cursor=0,
        prompt_index=prompt_index,
        checkpoint_sha256=CHECKPOINT,
        rendered_prompt=_faithful_prompt(prompt_index),
        completions=[{"tokens": tokens, "text": _text_for(tokens)}],
        signature="ok",
    ).model_dump()


async def _post_all(app, bodies):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://v") as http:
        responses = await asyncio.gather(
            *(http.post("/corpus/submit", json=body) for body in bodies)
        )
    return [r.json() for r in responses]


def test_concurrent_submissions_do_not_race_each_other(fake_r2, seeded_job):
    store = _YieldingStore(fake_r2)
    app = _app(seeded_job, store)
    bodies = [_request(f"5Hot{i}", i) for i in range(6)]

    results = asyncio.run(_post_all(app, bodies))

    assert [r["accepted"] for r in results] == [True] * 6
    # One writer, one process: its own requests take turns on the ledger
    # instead of spending the retry budget beating each other, and the ones
    # that queue behind a turn share the next one's write (group commit).
    assert store.conflicts == 0
    assert 1 <= store.ledger_write_attempts < 6
    snapshot, _ = asyncio.run(job_store.read_ledgers("swe-v1", **fake_r2))
    assert sorted(snapshot["slots"]) == [str(i) for i in range(6)]


def test_a_real_conflict_is_still_retried_under_the_lock(fake_r2, seeded_job):
    store = _YieldingStore(fake_r2)
    seeded_job.store = store
    seeded_job.fail_next_ledger_write_with_conflict(
        competing_snapshot={"slots": {"9": 1}}, times=1
    )
    app = _app(seeded_job, store)

    (result,) = asyncio.run(_post_all(app, [_request("5Hot", 0)]))

    assert result["accepted"] is True
    snapshot, _ = asyncio.run(job_store.read_ledgers("swe-v1", **fake_r2))
    # The competing write survives: the retry re-read it before admitting.
    assert snapshot["slots"] == {"0": 1, "9": 1}


def test_the_ledger_rebuild_runs_off_the_event_loop(fake_r2, seeded_job, monkeypatch):
    from reliquary.validator import corpus_service

    on_loop = []
    real = corpus_service.rebuild_ledgers

    def spying(job, snapshot):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return real(job, snapshot)

    monkeypatch.setattr(corpus_service, "rebuild_ledgers", spying)
    app = _app(seeded_job, seeded_job.store)
    (result,) = asyncio.run(_post_all(app, [_request("5Hot", 0)]))

    assert result["accepted"] is True
    assert on_loop == [False]


def test_the_ledger_bytes_are_encoded_and_decoded_off_the_event_loop(
    fake_r2, monkeypatch
):
    seen = []
    real_encode, real_decode = job_store._encode, job_store._decode

    def where(name):
        try:
            asyncio.get_running_loop()
            seen.append((name, "loop"))
        except RuntimeError:
            seen.append((name, "thread"))

    def encode(document):
        where("encode")
        return real_encode(document)

    def decode(body):
        where("decode")
        return real_decode(body)

    monkeypatch.setattr(job_store, "_encode", encode)
    monkeypatch.setattr(job_store, "_decode", decode)

    async def scenario():
        await job_store.write_ledgers("swe-v1", {"slots": {"1": 1}}, None)
        await job_store.read_ledgers("swe-v1")

    asyncio.run(scenario())
    assert seen == [("encode", "thread"), ("decode", "thread")]


def test_an_accepted_submission_logs_where_its_time_went(
    fake_r2, seeded_job, caplog
):
    app = _app(seeded_job, seeded_job.store)
    with caplog.at_level(logging.INFO, logger="reliquary.validator.corpus_service"):
        (result,) = asyncio.run(_post_all(app, [_request("5Hot", 0)]))

    assert result["accepted"] is True
    lines = [r.getMessage() for r in caplog.records if "corpus submission timing" in r.getMessage()]
    assert len(lines) == 1
    for part in ("job_read=", "checks=", "lock_wait=", "ledger_read=", "admit=",
                 "ledger_write=", "record_write=", "attempts=1", "total="):
        assert part in lines[0]


def test_the_record_store_encodes_and_decodes_off_the_event_loop(monkeypatch):
    # Submission records carry every token and proof; miners.json grows with
    # every hotkey. Both are read and written by the auditor on the shared loop.
    from reliquary.infrastructure import corpus_record_store as records
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

    client = _FakeMultiObjectR2()
    monkeypatch.setattr(records, "get_s3_client", lambda **kw: client)
    seen = []
    real_encode, real_decode = records._encode, records._decode

    def on_loop():
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def encode(document):
        seen.append(("encode", on_loop()))
        return real_encode(document)

    def decode(body):
        seen.append(("decode", on_loop()))
        return real_decode(body)

    monkeypatch.setattr(records, "_encode", encode)
    monkeypatch.setattr(records, "_decode", decode)
    store = BucketRecordStore()
    sid = "a" * 64

    async def scenario():
        await store.write_submission("swe-v1", sid, {"hotkey": "h"})
        await store.read_submission("swe-v1", sid)
        await store.write_miners("swe-v1", {"h": {}}, None)
        await store.read_miners("swe-v1")
        await store.write_settlement("swe-v1", {"paid": []}, None)
        await store.read_settlement("swe-v1")

    asyncio.run(scenario())
    assert len(seen) == 6
    assert not any(loop for _, loop in seen)


def test_a_hung_ledger_write_makes_the_next_submission_retryable_not_stuck(
    fake_r2, seeded_job
):
    from reliquary.validator.corpus_service import build_corpus_router

    release = asyncio.Event()

    class _HangingStore(_YieldingStore):
        async def write_ledgers(self, job_id, snapshot, etag):
            if not release.is_set():
                await release.wait()
            return await super().write_ledgers(job_id, snapshot, etag)

    store = _HangingStore(fake_r2)
    app = FastAPI()
    app.include_router(
        build_corpus_router(
            job_id="swe-v1",
            store=store,
            tokenizer=_Tokenizer(),
            renderer=_Renderer(),
            verify_signature=lambda request: True,
            prompt_job_for=seeded_job.prompt_job_for,
            ledger_lock_timeout=0.05,
        )
    )

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://v") as http:
            first = asyncio.create_task(http.post("/corpus/submit", json=_request("5HotA", 0)))
            await asyncio.sleep(0.02)
            second = await http.post("/corpus/submit", json=_request("5HotB", 1))
            release.set()
            return (await first), second

    first, second = asyncio.run(scenario())
    assert second.status_code == 503
    assert second.json() == {"detail": "corpus_ledger_contention"}
    assert first.json()["accepted"] is True


class _SegmentCountingStore(_CountingStore):
    def __init__(self, client_kwargs):
        super().__init__(client_kwargs)
        self.segment_gets = 0
        self.segment_puts = 0

    async def read_seen_segment(self, job_id, segment_id):
        self.segment_gets += 1
        return await super().read_seen_segment(job_id, segment_id)

    async def write_seen_segment(self, job_id, digests):
        self.segment_puts += 1
        return await super().write_seen_segment(job_id, digests)


def _sealing_app(seeded_job, store, threshold):
    from reliquary.validator.corpus_service import build_corpus_router

    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id="swe-v1", store=store, tokenizer=_Tokenizer(), renderer=_Renderer(),
        verify_signature=lambda request: True, prompt_job_for=seeded_job.prompt_job_for,
        seal_threshold=threshold,
    ))
    return app


def _many(count):
    # Distinct work on distinct prompts, one slot each.
    return [_request(f"5Hot{i % 7}", i) for i in range(count)]


def test_ledger_body_bounded_after_many_accepts(fake_r2, seeded_job, _r2_client):
    store = _SegmentCountingStore(fake_r2)
    app = _sealing_app(seeded_job, store, threshold=64)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://v") as http:
            for body in _many(600):
                assert (await http.post("/corpus/submit", json=body)).json()["accepted"]

    asyncio.run(scenario())
    body, _ = _r2_client.objects["reliquary/corpus/jobs/swe-v1/ledgers.json"]
    import json

    ledger = json.loads(body)
    seen_bytes = len(job_store._encode(
        {"seen_pending": ledger["seen_pending"], "seen_segments": ledger["seen_segments"]}
    ))
    # v1 would carry all 600 digests (~40 KB); v2 carries under one threshold
    # of them plus one small reference per sealed segment.
    assert len(ledger["seen_pending"]) < 64
    assert seen_bytes < 64 * 67 + len(ledger["seen_segments"]) * 100
    assert seen_bytes < 600 * 67 / 4
    assert store.segment_puts == 600 // 64


def test_warm_path_does_no_segment_gets(fake_r2, seeded_job):
    store = _SegmentCountingStore(fake_r2)
    app = _sealing_app(seeded_job, store, threshold=4)

    results = asyncio.run(_post_all(app, _many(30)))

    assert all(r["accepted"] for r in results)
    assert store.segment_puts == 30 // 4
    # Every segment this process sealed it already holds; none is fetched.
    assert store.segment_gets == 0
