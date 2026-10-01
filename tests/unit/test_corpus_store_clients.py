"""The bound stores keep one S3 client alive instead of building one per call:
building an aiobotocore client re-reads botocore's service model, and that
alone held the corpus validator's event loop at one full core."""

from __future__ import annotations

import asyncio

import pytest

from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure import corpus_record_store as records
from reliquary.infrastructure.corpus_job_store import BucketJobStore, _ClientPool
from reliquary.infrastructure.corpus_record_store import BucketRecordStore
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2, _manifest


class _CountingFactory:
    """Hands out a fresh fake client per call, sharing one bucket, and counts
    how many clients were built and closed."""

    def __init__(self, bucket=None):
        self.bucket = bucket or _FakeMultiObjectR2()
        self.built = 0
        self.closed = 0
        self.fail_next_get: list[BaseException] = []

    def __call__(self, **_kw):
        factory = self
        bucket = self.bucket
        self.built += 1

        class _Client:
            closed = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                self.closed = True
                factory.closed += 1
                return False

            async def get_object(self, **kw):
                if factory.fail_next_get:
                    raise factory.fail_next_get.pop(0)
                return await bucket.get_object(**kw)

            async def put_object(self, **kw):
                return await bucket.put_object(**kw)

            def get_paginator(self, name):
                return bucket.get_paginator(name)

        return _Client()


@pytest.fixture
def factory(monkeypatch):
    made = _CountingFactory()
    monkeypatch.setattr(job_store, "get_s3_client", made)
    monkeypatch.setattr(records, "get_s3_client", made)
    return made


def test_a_bound_job_store_builds_one_client_for_many_calls(factory):
    store = BucketJobStore()

    async def scenario():
        await job_store.write_job(_manifest(), None)
        built_by_write = factory.built
        for _ in range(5):
            await store.read_ledgers("swe-v1")
        _, etag = await store.read_ledgers("swe-v1")
        etag = await store.write_ledgers("swe-v1", {"slots": {}}, etag)
        await store.write_ledgers("swe-v1", {"slots": {"1": 1}}, etag)
        return built_by_write

    built_by_write = asyncio.run(scenario())
    assert factory.built - built_by_write == 1


def test_a_bound_record_store_builds_one_client_for_many_calls(factory):
    store = BucketRecordStore()
    sid = "a" * 64

    async def scenario():
        await store.write_submission("swe-v1", sid, {"hotkey": "h"})
        await store.read_submission("swe-v1", sid)
        await store.list_submission_ids("swe-v1")
        await store.read_miners("swe-v1")
        await store.read_settlement("swe-v1")
        await store.list_verdict_ids("swe-v1")

    asyncio.run(scenario())
    # One for the calls, one on the listing thread's own loop: listings are
    # parsed off the serving loop, and reuse their client too.
    assert factory.built == 2


def test_a_connection_error_retires_the_client_and_the_next_call_builds_one(factory):
    from botocore.exceptions import EndpointConnectionError

    store = BucketRecordStore()
    sid = "a" * 64

    async def scenario():
        await store.write_submission("swe-v1", sid, {"hotkey": "h"})
        factory.fail_next_get.append(EndpointConnectionError(endpoint_url="https://r2"))
        with pytest.raises(EndpointConnectionError):
            await store.read_submission("swe-v1", sid)
        return await store.read_submission("swe-v1", sid)

    assert asyncio.run(scenario()) == {"hotkey": "h"}
    assert factory.built == 2
    # The poisoned client is closed, not leaked.
    assert factory.closed == 1


def test_a_server_answer_is_not_a_connection_error(factory):
    # An absent key or a lost race is R2 answering: the connection is fine.
    store = BucketJobStore()

    async def scenario():
        await store.read_ledgers("swe-v1")
        await store.write_ledgers("swe-v1", {"slots": {}}, None)
        with pytest.raises(job_store.CorpusStoreConflict):
            await store.write_ledgers("swe-v1", {"slots": {}}, None)
        await store.read_job("swe-v1")

    asyncio.run(scenario())
    assert factory.built == 1
    assert factory.closed == 0


def test_each_event_loop_gets_its_own_client(factory):
    # A client's connections belong to the loop that opened them.
    store = BucketRecordStore()
    asyncio.run(store.read_settlement("swe-v1"))
    asyncio.run(store.read_settlement("swe-v1"))
    assert factory.built == 2


def test_a_retired_client_closes_only_once_its_last_user_is_done():
    built, closed = [], []

    class _Ctx:
        def __init__(self):
            built.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            closed.append(self)
            return False

    clock = [0.0]
    pool = _ClientPool(_Ctx, max_age_seconds=10.0, clock=lambda: clock[0])

    async def scenario():
        async with pool.client() as first:
            clock[0] = 11.0
            # Past its age: the next caller gets a new client, but the old one
            # is still in use here and must not be closed under it.
            async with pool.client() as second:
                assert second is not first
            assert closed == []
        assert closed == [first]

    asyncio.run(scenario())
    assert len(built) == 2


def test_concurrent_first_calls_build_a_single_client():
    built = []

    class _Ctx:
        def __init__(self):
            built.append(self)

        async def __aenter__(self):
            await asyncio.sleep(0)
            return self

        async def __aexit__(self, *exc):
            return False

    pool = _ClientPool(_Ctx)

    async def use():
        async with pool.client():
            await asyncio.sleep(0)

    async def scenario():
        await asyncio.gather(*(use() for _ in range(8)))

    asyncio.run(scenario())
    assert len(built) == 1
