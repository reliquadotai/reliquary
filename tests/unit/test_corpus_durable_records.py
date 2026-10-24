"""Accepted slots and their immutable receipts survive ambiguous storage writes."""

from __future__ import annotations

import asyncio
import copy

import pytest
from fastapi import HTTPException

from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure import corpus_record_store as record_store
from reliquary.protocol.signatures import corpus_submission_id
from reliquary.validator import corpus_service as service
from reliquary.validator.corpus_job_status import stored_job_counts
from tests.unit.test_corpus_ledger_group_commit import _submit
from tests.unit.test_corpus_route_skip import _router
from tests.unit.test_corpus_service import _r2_client, fake_r2, seeded_job  # noqa: F401


@pytest.fixture
def records(_r2_client, monkeypatch):
    # Reuse the native store's conditional-write fixture, including its pool.
    get = _r2_client.get_object

    async def ranged_get(**kwargs):
        kwargs.pop("Range", None)
        return await get(**kwargs)

    monkeypatch.setattr(_r2_client, "get_object", ranged_get)
    monkeypatch.setattr(record_store, "get_s3_client", lambda **kw: _r2_client)
    return record_store.BucketRecordStore()


def router(seeded_job, records, **kwargs):
    return _router(seeded_job, "swe-v1", records=records, durable_records=True, **kwargs)


async def state(seeded_job):
    raw, _ = await seeded_job.store.read_ledgers("swe-v1")
    return raw, service.rebuild_ledgers(seeded_job.job, raw)


def test_unresolved_staging_conflict_is_retryable_and_consumes_nothing(
    seeded_job, records, monkeypatch,
):
    async def go():
        async def in_flight(*args, **kwargs):
            raise job_store.CorpusStoreConflict("conditional stage write in flight")

        monkeypatch.setattr(record_store, "_put", in_flight)
        arrivals = {}
        front = router(seeded_job, records, pending_record_arrivals=arrivals)
        with pytest.raises(HTTPException, match="503"):
            await front.submit_corpus(_submit("swe-v1", "5A", 0, 1))
        assert not arrivals and not front.admission_pending()
        assert (await state(seeded_job))[1].slots.filled == 0
        assert await records.list_submission_ids("swe-v1") == []

    asyncio.run(go())


@pytest.mark.parametrize("fault", ["publish", "publish_ack", "ledger_ack", "cleanup_ack"])
def test_committed_receipt_recovers_after_storage_failure_without_another_slot(
    seeded_job, records, monkeypatch, fault,
):
    async def go():
        clock = [100.0]
        monkeypatch.setattr(service.time, "time", lambda: clock[0])
        request = _submit("swe-v1", "5A", 0, 1)
        sid = corpus_submission_id(request)
        write = seeded_job.store.write_ledgers
        promote = record_store.BucketRecordStore.promote_submission
        failures = [1]

        async def faulty_write(job_id, raw, etag):
            result = await write(job_id, raw, etag)
            if failures[0] and ((fault == "ledger_ack" and raw.get("pending_records"))
                               or (fault == "cleanup_ack" and raw.get("schema") == service.LEDGER_SCHEMA_V3
                                   and not raw["pending_records"])):
                failures[0] -= 1
                raise OSError("write acknowledgment lost")
            return result

        async def faulty_promote(self, job_id, ref):
            if fault == "publish" and failures[0]:
                failures[0] -= 1
                raise OSError("publication unavailable")
            result = await promote(self, job_id, ref)
            if fault == "publish_ack" and failures[0]:
                failures[0] -= 1
                raise OSError("publication acknowledgment lost")
            return result

        monkeypatch.setattr(seeded_job.store, "write_ledgers", faulty_write)
        monkeypatch.setattr(record_store.BucketRecordStore, "promote_submission", faulty_promote)
        arrivals = {}
        first = router(seeded_job, records, pending_record_arrivals=arrivals)
        with pytest.raises(HTTPException, match="503"):
            await first.submit_corpus(request)
        raw, ledger = await state(seeded_job)
        assert ledger.slots.filled == 1
        assert len(ledger.records) == (0 if fault == "cleanup_ack" else 1)
        if ledger.records:
            frozen = await records.read_staged_submission("swe-v1", ledger.records[0])
            assert arrivals == {sid: 100.0}
        else:
            frozen = await records.read_submission("swe-v1", sid)
        assert frozen["received_at"] == 100.0
        clock[0] = 900.0
        recovered = []
        fresh = router(seeded_job, records, on_accepted=recovered.append)
        await fresh.recover_records()
        assert (await fresh.submit_corpus(request)).accepted
        assert (await fresh.submit_corpus(request)).accepted
        assert (await records.read_submission("swe-v1", sid)) == frozen
        assert len(recovered) == (0 if fault == "cleanup_ack" else 1)
        raw, ledger = await state(seeded_job)
        assert ledger.slots.filled == 1 and ledger.records == ()
        assert raw["schema"] == service.LEDGER_SCHEMA_V3

    asyncio.run(go())


def test_two_concurrent_retries_publish_once_and_announce_once(seeded_job, records):
    async def go():
        accepted, arrivals = [], {}
        front = router(seeded_job, records, on_accepted=accepted.append,
                       pending_record_arrivals=arrivals)
        request = _submit("swe-v1", "5A", 0, 1)
        sid = corpus_submission_id(request)
        await front.ledger_lock.acquire()
        tasks = [asyncio.create_task(front.submit_corpus(request)) for _ in range(2)]
        while front.ledger_waiting() != 2:
            await asyncio.sleep(0)
        front.ledger_lock.release()
        assert all(response.accepted for response in await asyncio.gather(*tasks))
        assert accepted == [sid] and not arrivals
        assert await records.list_submission_ids("swe-v1") == [sid]
        _, ledger = await state(seeded_job)
        assert ledger.slots.filled == 1 and not ledger.records

    asyncio.run(go())


def test_completed_receipts_leave_no_per_submission_bookkeeping(seeded_job, records, _r2_client):
    import inspect

    async def go():
        arrivals, accepted = {}, []
        front = router(seeded_job, records, pending_record_arrivals=arrivals,
                       on_accepted=accepted.append)
        for index in range(20):
            assert (await front.submit_corpus(_submit("swe-v1", "5A", index, 1))).accepted
        held = inspect.getclosurevars(front.recover_records).nonlocals
        assert held["active_records"] == {} and held["announced"] == set() and not arrivals
        assert len(accepted) == len(set(accepted)) == 20
        # A different signed SID with an already accepted completion is
        # refused before staging, not retained as an uncommitted body.
        refused = await front.submit_corpus(_submit("swe-v1", "5Other", 0, 1))
        assert not refused.accepted
        assert len(await records.list_submission_ids("swe-v1")) == 20
        assert sum("/staged-submissions/" in key for key in _r2_client.objects) == 20
        refused = await front.submit_corpus(_submit("swe-v1", "5Other", 0, 2, checkpoint="f" * 64))
        assert not refused.accepted
        assert sum("/staged-submissions/" in key for key in _r2_client.objects) == 20
        assert held["active_records"] == {} and held["announced"] == set() and not arrivals

    asyncio.run(go())


def test_a_cancelled_queued_request_is_never_staged(seeded_job, records, _r2_client):
    async def go():
        front = router(seeded_job, records)
        await front.ledger_lock.acquire()
        task = asyncio.create_task(front.submit_corpus(_submit("swe-v1", "5A", 0, 1)))
        while not front.ledger_waiting():
            await asyncio.sleep(0)
        assert not any("/staged-submissions/" in key for key in _r2_client.objects)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        front.ledger_lock.release()
        assert not any("/staged-submissions/" in key for key in _r2_client.objects)
        assert (await state(seeded_job))[1].slots.filled == 0

    asyncio.run(go())


def test_cancelled_http_handler_cannot_release_the_taken_turn_coverage_fence(
    seeded_job, records, monkeypatch,
):
    async def go():
        writing, finish = asyncio.Event(), asyncio.Event()
        write = seeded_job.store.write_ledgers

        async def held_write(job_id, raw, etag):
            if raw.get("pending_records"):
                writing.set()
                await finish.wait()
            return await write(job_id, raw, etag)

        monkeypatch.setattr(seeded_job.store, "write_ledgers", held_write)
        arrivals, accepted = {}, []
        front = router(seeded_job, records, pending_record_arrivals=arrivals,
                       on_accepted=accepted.append)
        request = _submit("swe-v1", "5A", 0, 1)
        task = asyncio.create_task(front.submit_corpus(request))
        await writing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert arrivals and front.admission_pending()
        finish.set()
        async def finished():
            while front.admission_pending():
                await asyncio.sleep(0)

        await asyncio.wait_for(finished(), 2)
        assert not arrivals and not front.admission_pending()
        assert accepted == [corpus_submission_id(request)]
        assert (await state(seeded_job))[1].slots.filled == 1

    asyncio.run(go())


def test_queued_publication_outage_cannot_accumulate_more_than_one_batch(
    seeded_job, records, monkeypatch, _r2_client,
):
    async def go():
        async def unavailable(*args):
            raise OSError("publication unavailable")

        monkeypatch.setattr(record_store.BucketRecordStore, "promote_submission", unavailable)
        front = router(seeded_job, records, ledger_batch_max=2, seal_threshold=1)
        await front.ledger_lock.acquire()
        tasks = [asyncio.create_task(front.submit_corpus(_submit("swe-v1", f"5A{i}", i, 1)))
                 for i in range(7)]
        while front.ledger_waiting() != 7:
            await asyncio.sleep(0)
        front.ledger_lock.release()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(response, HTTPException) and response.status_code == 503
                   for response in responses)
        _, ledger = await state(seeded_job)
        assert ledger.slots.filled == len(ledger.records) == 2
        assert len(ledger.segments) == 2 and not ledger.pending
        assert await records.list_submission_ids("swe-v1") == []
        assert sum("/staged-submissions/" in key for key in _r2_client.objects) == 2

    asyncio.run(go())


def test_prompt_failure_and_skip_preserve_committed_refs(fake_r2, seeded_job, records):
    from reliquary.corpus.walk import job_walk_index
    from tests.unit.test_corpus_route_skip import _declare, _skip_body
    from tests.unit.test_eval_prompt_source import _job
    from reliquary.protocol.corpus_submission import CorpusSkipRequest

    walk = _declare(fake_r2, "walk-v1")
    eval_job = _job("eval-set:s:4:" + "0" * 64, count=4)
    ref = {"submission_id": "a" * 64, "sha256": "b" * 64, "received_at": 123.0}

    async def seed(job, index):
        await job_store.write_ledgers(job.job_id, {
            "schema": service.LEDGER_SCHEMA_V3, "slots": {str(index): 1}, "cursors": {},
            "seen_pending": ["c" * 64], "seen_segments": [], "pending_records": [ref],
        }, None, **fake_r2)

    async def go():
        await seed(eval_job, 2)
        assert await service.record_prompt_failure(seeded_job.store, eval_job, 2, "d" * 64)
        changed, _ = await seeded_job.store.read_ledgers(eval_job.job_id)
        assert changed["pending_records"] == [ref] and changed["failed"]
        assert await service.ensure_ledgers_v2(seeded_job.store, eval_job) == "v3"
        assert (await seeded_job.store.read_ledgers(eval_job.job_id))[0] == changed
        index = job_walk_index(walk, "5Hot", 0)
        await seed(walk, index)
        front = _router(seeded_job, "walk-v1", records=records, durable_records=True)
        request = CorpusSkipRequest(**_skip_body(walk))
        assert (await front.skip_corpus(request)).skipped
        changed, _ = await seeded_job.store.read_ledgers(walk.job_id)
        assert changed["pending_records"] == [ref] and changed["cursors"] == {"5Hot": 1}

    asyncio.run(go())


def test_stale_ref_or_changed_canonical_body_never_clears_the_committed_ref(
    seeded_job, records, monkeypatch,
):
    async def go():
        promote = record_store.BucketRecordStore.promote_submission

        async def unavailable(*args):
            raise OSError("publication unavailable")

        monkeypatch.setattr(record_store.BucketRecordStore, "promote_submission", unavailable)
        request = _submit("swe-v1", "5A", 0, 1)
        with pytest.raises(HTTPException):
            await router(seeded_job, records).submit_corpus(request)
        raw, _ = await state(seeded_job)
        ref = raw["pending_records"][0]
        original = await records.read_staged_submission("swe-v1", ref)
        monkeypatch.setattr(record_store.BucketRecordStore, "promote_submission", promote)
        changed = copy.deepcopy(original)
        changed["received_at"] += 1
        await records.write_submission("swe-v1", ref["submission_id"], changed)
        with pytest.raises(ValueError):
            await service.recover_pending_records(seeded_job.store, records, seeded_job.job)
        assert (await state(seeded_job))[0] == raw
        assert await records.read_submission("swe-v1", ref["submission_id"]) == changed
        stale = {**ref, "sha256": "f" * 64}
        bad, etag = await seeded_job.store.read_ledgers("swe-v1")
        bad["pending_records"] = [stale]
        await seeded_job.store.write_ledgers("swe-v1", bad, etag)
        with pytest.raises(ValueError):
            await service.recover_pending_records(seeded_job.store, records, seeded_job.job)
        assert (await state(seeded_job))[0]["pending_records"] == [stale]

    asyncio.run(go())


def test_status_and_downgrade_refuse_a_pending_receipt_even_with_empty_lists(
    seeded_job, records, monkeypatch,
):
    async def go():
        async def unavailable(*args):
            raise OSError("publication unavailable")

        monkeypatch.setattr(record_store.BucketRecordStore, "promote_submission", unavailable)
        with pytest.raises(HTTPException):
            await router(seeded_job, records).submit_corpus(_submit("swe-v1", "5A", 0, 1))
        counts = await stored_job_counts(records, "swe-v1")
        assert counts["submissions"] == counts["verdicts"] == 0
        assert counts["pending_records"] == 1 and counts["drained"] is False
        raw, _ = await state(seeded_job)
        assert await service.ensure_ledgers_v2(seeded_job.store, seeded_job.job) == "v3"
        with pytest.raises(service.LedgerSnapshotError):
            await service.downgrade_ledgers_v1(seeded_job.store, seeded_job.job)
        assert (await state(seeded_job))[0] == raw
        # A reader-only router must not erase the v3 ref via another admission.
        off = _router(seeded_job, "swe-v1", records=records, durable_records=False)
        with pytest.raises(HTTPException, match="503"):
            await off.submit_corpus(_submit("swe-v1", "5B", 1, 1))
        assert (await state(seeded_job))[0] == raw

    asyncio.run(go())


def test_status_never_certifies_an_unknown_ledger_without_its_manifest(records, _r2_client):
    async def go():
        await job_store.write_ledgers("missing-job", {"schema": "reliquary/corpus-ledgers/v4"}, None)
        with pytest.raises(ValueError, match="manifest"):
            await stored_job_counts(records, "missing-job")

    asyncio.run(go())
