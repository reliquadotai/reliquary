"""Each miner's audit state lives in one CAS object beside the job records;
an ``update`` must re-read and re-apply on a conflict rather than ever
writing a stale document over a concurrent writer's."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure import corpus_record_store as records
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2


@pytest.fixture
def r2(monkeypatch):
    client = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: client)
    monkeypatch.setattr(records, "get_s3_client", lambda **kw: client)
    return client


def test_an_update_preserves_other_hotkeys_entries(r2):
    # Three sequential updates (A, then B, then A again): this does not put a
    # writer between another update's own read and write, so it proves the
    # merge is scoped to one hotkey, not the CAS retry itself — that race is
    # covered by test_a_conflict_is_re_read_and_the_change_reapplied below.
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    states = MinerStates(BucketRecordStore(), "job-x")
    asyncio.run(states.update("A", lambda m: replace(m, banned_until=99.0)))
    other = MinerStates(BucketRecordStore(), "job-x")
    asyncio.run(other.update("B", lambda m: replace(m, audited_passed=1)))
    asyncio.run(states.update("A", lambda m: replace(m, audited_passed=5)))
    a = asyncio.run(states.get("A"))
    assert a.banned_until == 99.0 and a.audited_passed == 5
    assert asyncio.run(states.get("B")).audited_passed == 1


def test_an_unknown_hotkey_is_in_probation(r2):
    from reliquary.corpus.audit_policy import AuditParams, effective_state
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    m = asyncio.run(MinerStates(BucketRecordStore(), "job-x").get("new"))
    assert effective_state(m, now=0.0, params=AuditParams()) == "probation"


class _FlakyMinersStore:
    """Wraps a real ``BucketRecordStore``, injecting ``write_miners`` failures
    without touching the underlying object (``__slots__`` forbids patching it
    directly)."""

    def __init__(self, inner, *, fail_first_n=0, always_fail=False, concurrent=None):
        self._inner = inner
        self._fail_first_n = fail_first_n
        self._always_fail = always_fail
        # (hotkey, fields) a concurrent writer lands before each lost race.
        self._concurrent = concurrent
        self.write_attempts = 0

    async def read_miners(self, job_id):
        return await self._inner.read_miners(job_id)

    async def write_miners(self, job_id, state, etag):
        from reliquary.infrastructure.corpus_record_store import CorpusStoreConflict

        self.write_attempts += 1
        if self._always_fail or self.write_attempts <= self._fail_first_n:
            # A concurrent writer wins the race; this call's write is dropped.
            if self._concurrent is not None:
                hotkey, fields = self._concurrent
                document, current = await self._inner.read_miners(job_id)
                entry = {**document.get(hotkey, {}), **fields}
                await self._inner.write_miners(job_id, {**document, hotkey: entry}, current)
            raise CorpusStoreConflict("stale etag")
        return await self._inner.write_miners(job_id, state, etag)


def test_a_conflict_is_re_read_and_the_change_reapplied(r2):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    # The race is lost to a writer banning the same hotkey: the retry must
    # re-read it, or a write built from the stale read would lift the ban.
    store = _FlakyMinersStore(BucketRecordStore(), fail_first_n=1,
                              concurrent=("A", {"banned_until": 99.0}))
    states = MinerStates(store, "job-x")

    result = asyncio.run(states.update("A", lambda m: replace(m, audited_passed=m.audited_passed + 1)))

    assert store.write_attempts == 2
    assert result.audited_passed == 1 and result.banned_until == 99.0
    stored = asyncio.run(states.get("A"))
    assert stored.audited_passed == 1 and stored.banned_until == 99.0


def test_attempts_exhausted_raises_conflict(r2):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore, CorpusStoreConflict
    from reliquary.validator.corpus_miner_states import MinerStates

    store = _FlakyMinersStore(BucketRecordStore(), always_fail=True)
    states = MinerStates(store, "job-x")

    with pytest.raises(CorpusStoreConflict):
        asyncio.run(states.update("A", lambda m: replace(m, audited_passed=1), attempts=2))
    assert store.write_attempts == 2


def _seed_miners(r2, job_id, document):
    # Writes straight into the fake bucket, bypassing the store's own
    # validated write path — the only way to plant a hand-edited/corrupt
    # miners.json for these tests.
    import json

    from reliquary.infrastructure.corpus_record_store import _miners_key

    r2.objects[_miners_key(job_id)] = (json.dumps(document).encode(), '"seed"')


def test_a_non_dict_document_raises_naming_the_job(r2):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    _seed_miners(r2, "job-x", ["not", "an", "object"])
    with pytest.raises(ValueError, match="job-x"):
        asyncio.run(MinerStates(BucketRecordStore(), "job-x").get("A"))


@pytest.mark.parametrize(
    "entry",
    [
        "banned",
        [1, 2, 3],
        {"audited_passed": "five"},
        {"audited_passed": -1},
        {"audited_passed": True},
        {"confirmed_failures": "nope"},
        {"confirmed_failures": [1, float("nan")]},
        {"mant_mean_history": {"x": 1}},
        {"suspect_until": "soon"},
        {"banned_until": float("inf")},
        {"failure_ids": "d" * 64},
        {"failure_ids": ["D" * 64]},
        {"failure_ids": ["d" * 63]},
        {"failure_ids": [7]},
    ],
)
def test_a_malformed_entry_raises_naming_job_and_hotkey(r2, entry):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    _seed_miners(r2, "job-x", {"A": entry})
    with pytest.raises(ValueError) as caught:
        asyncio.run(MinerStates(BucketRecordStore(), "job-x").get("A"))
    assert "job-x" in str(caught.value)
    assert "A" in str(caught.value)


def test_update_also_validates_the_entry_it_reads(r2):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    _seed_miners(r2, "job-x", {"A": "banned"})
    with pytest.raises(ValueError, match="job-x"):
        asyncio.run(MinerStates(BucketRecordStore(), "job-x").update("A", lambda m: m))


def test_a_well_formed_entry_still_reads(r2):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    _seed_miners(
        r2,
        "job-x",
        {
            "A": {
                "audited_passed": 3,
                "confirmed_failures": [1.0, 2.0],
                "suspect_until": None,
                "banned_until": None,
                "mant_mean_history": [0.5],
                "failure_ids": ["d" * 64],
            }
        },
    )
    m = asyncio.run(MinerStates(BucketRecordStore(), "job-x").get("A"))
    assert m.audited_passed == 3
    assert m.confirmed_failures == [1.0, 2.0]
    assert m.failure_ids == ["d" * 64]


class _ThrottledMinersStore(_FlakyMinersStore):
    """R2 answers a burst of writes to one key with a throttling error, which
    is a transport error, not a conflict."""

    def __init__(self, inner, *, throttle_first_n):
        super().__init__(inner)
        self._throttle_first_n = throttle_first_n

    async def write_miners(self, job_id, state, etag):
        from botocore.exceptions import ClientError

        self.write_attempts += 1
        if self.write_attempts <= self._throttle_first_n:
            raise ClientError({"Error": {"Code": "SlowDown", "Message": "reduce your rate"}},
                              "PutObject")
        return await self._inner.write_miners(job_id, state, etag)


def test_a_throttled_write_is_retried_after_a_backoff(r2):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    store = _ThrottledMinersStore(BucketRecordStore(), throttle_first_n=2)
    slept = []

    async def _sleep(seconds):
        slept.append(seconds)

    states = MinerStates(store, "job-x", sleep=_sleep)
    result = asyncio.run(states.update("A", lambda m: replace(m, audited_passed=m.audited_passed + 1)))
    assert store.write_attempts == 3 and len(slept) == 2 and all(s > 0 for s in slept)
    assert result.audited_passed == 1
    assert asyncio.run(states.get("A")).audited_passed == 1


def test_throttling_that_never_ends_raises_after_bounded_attempts(r2):
    from botocore.exceptions import ClientError

    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    store = _ThrottledMinersStore(BucketRecordStore(), throttle_first_n=100)

    async def _sleep(seconds):
        pass

    states = MinerStates(store, "job-x", sleep=_sleep)
    with pytest.raises(ClientError):
        asyncio.run(states.update("A", lambda m: replace(m, audited_passed=1), attempts=3))
    assert store.write_attempts == 3


def test_update_many_writes_every_hotkey_in_one_write(r2):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    store = _FlakyMinersStore(BucketRecordStore())
    states = MinerStates(store, "job-x")
    out = asyncio.run(states.update_many({
        "A": lambda m: replace(m, audited_passed=2),
        "B": lambda m: replace(m, banned_until=5.0),
    }))
    assert store.write_attempts == 1
    assert out["A"].audited_passed == 2 and out["B"].banned_until == 5.0
    assert asyncio.run(states.get("B")).banned_until == 5.0
