"""The bound job store remembers what it last read or wrote: the manifest is
immutable per job id, and this validator is the only writer of its job's
ledgers. The compare-and-swap is unchanged, so a stale memory can cost a
conflict and a re-read, never a write that would not land today."""

from __future__ import annotations

import asyncio

import pytest

from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure.corpus_job_store import BucketJobStore, CorpusStoreConflict
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2, _manifest

LEDGERS_KEY = "reliquary/corpus/jobs/swe-v1/ledgers.json"


class _CountingR2(_FakeMultiObjectR2):
    def __init__(self):
        super().__init__()
        self.gets: list[str] = []
        self.fail_next_put: BaseException | None = None

    async def get_object(self, Bucket, Key):
        self.gets.append(Key)
        return await super().get_object(Bucket, Key)

    async def put_object(self, Bucket, Key, Body, **condition):
        if self.fail_next_put is not None:
            exc, self.fail_next_put = self.fail_next_put, None
            raise exc
        return await super().put_object(Bucket, Key, Body, **condition)


@pytest.fixture
def bucket(monkeypatch):
    client = _CountingR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: client)
    return client


def _run(coro):
    return asyncio.run(coro)


def test_the_manifest_is_read_once(bucket):
    _run(job_store.write_job(_manifest(), None))
    store = BucketJobStore()

    async def scenario():
        first = await store.read_job("swe-v1")
        second = await store.read_job("swe-v1")
        return first, second

    first, second = _run(scenario())
    assert first[0].job_id == second[0].job_id == "swe-v1"
    assert bucket.gets.count("reliquary/corpus/jobs/swe-v1.json") == 1


def test_an_absent_manifest_is_asked_for_again(bucket):
    store = BucketJobStore()

    async def scenario():
        assert (await store.read_job("swe-v1")) == (None, None)
        await job_store.write_job(_manifest(), None)
        return await store.read_job("swe-v1")

    job, _ = _run(scenario())
    assert job is not None


def test_ledgers_this_store_wrote_are_not_read_back(bucket):
    store = BucketJobStore()

    async def scenario():
        snapshot, etag = await store.read_ledgers("swe-v1")
        etag = await store.write_ledgers("swe-v1", {"slots": {"3": 1}}, etag)
        again, again_etag = await store.read_ledgers("swe-v1")
        return etag, again, again_etag

    etag, again, again_etag = _run(scenario())
    assert again == {"slots": {"3": 1}}
    assert again_etag == etag
    # Only the first, absent read reached the bucket.
    assert bucket.gets == [LEDGERS_KEY]


def test_a_read_ledger_is_served_from_memory_after_the_first_read(bucket):
    _run(job_store.write_ledgers("swe-v1", {"slots": {"1": 1}}, None))
    store = BucketJobStore()

    async def scenario():
        for _ in range(4):
            snapshot, _ = await store.read_ledgers("swe-v1")
        return snapshot

    assert _run(scenario()) == {"slots": {"1": 1}}
    assert bucket.gets == [LEDGERS_KEY]


def test_a_stale_memory_conflicts_and_the_next_read_goes_to_the_bucket(bucket):
    store = BucketJobStore()

    async def scenario():
        _, etag = await store.read_ledgers("swe-v1")
        etag = await store.write_ledgers("swe-v1", {"slots": {"1": 1}}, etag)
        # Another writer lands behind this store's back.
        await job_store.write_ledgers("swe-v1", {"slots": {"2": 1}}, etag)
        cached, cached_etag = await store.read_ledgers("swe-v1")
        with pytest.raises(CorpusStoreConflict):
            await store.write_ledgers("swe-v1", {"slots": {"1": 2}}, cached_etag)
        fresh, fresh_etag = await store.read_ledgers("swe-v1")
        await store.write_ledgers("swe-v1", {"slots": {"1": 1, "2": 1}}, fresh_etag)
        return fresh

    assert _run(scenario()) == {"slots": {"2": 1}}
    stored, _ = _run(job_store.read_ledgers("swe-v1"))
    assert stored == {"slots": {"1": 1, "2": 1}}


def test_a_write_that_failed_in_transit_forgets_the_ledgers(bucket):
    # It may or may not have landed: only the bucket knows.
    store = BucketJobStore()

    async def scenario():
        _, etag = await store.read_ledgers("swe-v1")
        await store.write_ledgers("swe-v1", {"slots": {"1": 1}}, etag)
        bucket.fail_next_put = OSError("reset")
        with pytest.raises(OSError):
            await store.write_ledgers("swe-v1", {"slots": {"1": 2}}, "anything")
        await store.read_ledgers("swe-v1")

    _run(scenario())
    assert bucket.gets == [LEDGERS_KEY, LEDGERS_KEY]


def test_the_ledgers_are_remembered_per_job(bucket):
    store = BucketJobStore()

    async def scenario():
        await store.write_ledgers("swe-v1", {"slots": {"1": 1}}, None)
        return await store.read_ledgers("other-v1")

    assert _run(scenario()) == ({}, None)
