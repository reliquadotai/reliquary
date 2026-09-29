"""Sealed seen-digest segments: immutable, named by the sha256 of their bytes,
written create-only. A reader trusts a segment only if its bytes still hash to
its name, and an absent one is corruption, never an empty set."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure.corpus_job_store import (
    BucketJobStore,
    CorpusSegmentCorrupt,
)
from tests.unit.test_corpus_store_cache import _CountingR2

DIGESTS = sorted(hashlib.sha256(str(i).encode()).hexdigest() for i in range(5))


@pytest.fixture
def bucket(monkeypatch):
    client = _CountingR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: client)
    return client


def _segment_key(segment_id):
    return f"reliquary/corpus/jobs/swe-v1/seen/{segment_id}.json"


def test_segment_id_is_sha256_of_stored_bytes(bucket):
    segment_id = asyncio.run(job_store.write_seen_segment("swe-v1", DIGESTS))

    body, _ = bucket.objects[_segment_key(segment_id)]
    assert segment_id == hashlib.sha256(body).hexdigest()
    assert json.loads(body) == {
        "schema": job_store.SEEN_SEGMENT_SCHEMA,
        "digests": DIGESTS,
    }
    # The listing of jobs never mistakes a segment for a manifest.
    assert asyncio.run(job_store.list_jobs()) == []


def test_write_seen_segment_create_only_idempotent(bucket):
    store = BucketJobStore()

    async def scenario():
        first = await store.write_seen_segment("swe-v1", DIGESTS)
        before = bucket.objects[_segment_key(first)]
        second = await store.write_seen_segment("swe-v1", DIGESTS)
        return first, second, before

    first, second, before = asyncio.run(scenario())
    assert first == second
    # Not overwritten: the second PUT was refused by the create-only condition.
    assert bucket.objects[_segment_key(first)] == before
    assert asyncio.run(store.read_seen_segment("swe-v1", first)) == tuple(DIGESTS)


def test_write_seen_segment_refuses_unsorted_or_repeated_digests(bucket):
    with pytest.raises(ValueError):
        asyncio.run(job_store.write_seen_segment("swe-v1", list(reversed(DIGESTS))))
    with pytest.raises(ValueError):
        asyncio.run(job_store.write_seen_segment("swe-v1", [DIGESTS[0], DIGESTS[0]]))
    with pytest.raises(ValueError):
        asyncio.run(job_store.write_seen_segment("swe-v1", []))
    assert bucket.objects == {}


def test_read_segment_hash_mismatch_is_corrupt(bucket):
    segment_id = asyncio.run(job_store.write_seen_segment("swe-v1", DIGESTS))
    tampered = job_store._encode(
        {"schema": job_store.SEEN_SEGMENT_SCHEMA, "digests": DIGESTS[:-1]}
    )
    bucket.objects[_segment_key(segment_id)] = (tampered, '"x"')

    with pytest.raises(CorpusSegmentCorrupt):
        asyncio.run(job_store.read_seen_segment("swe-v1", segment_id))


def test_read_segment_absent_is_corrupt_not_empty(bucket):
    with pytest.raises(CorpusSegmentCorrupt):
        asyncio.run(job_store.read_seen_segment("swe-v1", "0" * 64))


def test_read_segment_refuses_an_id_that_is_not_a_digest(bucket):
    # The id is interpolated into a key, so a traversal must never reach it.
    with pytest.raises(ValueError):
        asyncio.run(job_store.read_seen_segment("swe-v1", "../ledgers"))


def test_read_segment_with_the_right_hash_but_unsorted_body_is_corrupt(bucket):
    body = job_store._encode(
        {"schema": job_store.SEEN_SEGMENT_SCHEMA, "digests": list(reversed(DIGESTS))}
    )
    segment_id = hashlib.sha256(body).hexdigest()
    bucket.objects[_segment_key(segment_id)] = (body, '"x"')

    with pytest.raises(CorpusSegmentCorrupt):
        asyncio.run(job_store.read_seen_segment("swe-v1", segment_id))


def test_the_v1_backup_is_written_once(bucket):
    store = BucketJobStore()

    async def scenario():
        first = await store.write_ledgers_backup("swe-v1", {"seen": ["a"]})
        second = await store.write_ledgers_backup("swe-v1", {"seen": ["b"]})
        return first, second, await store.read_ledgers_backup("swe-v1")

    first, second, kept = asyncio.run(scenario())
    assert (first, second) == (True, False)
    assert kept == {"seen": ["a"]}
