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


def test_an_update_survives_a_concurrent_writer(r2):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    states = MinerStates(BucketRecordStore(), "job-x")
    asyncio.run(states.update("A", lambda m: replace(m, banned_until=99.0)))
    # A second writer lands between our read and write: the ban must survive.
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

    def __init__(self, inner, *, fail_first_n=0, always_fail=False):
        self._inner = inner
        self._fail_first_n = fail_first_n
        self._always_fail = always_fail
        self.write_attempts = 0

    async def read_miners(self, job_id):
        return await self._inner.read_miners(job_id)

    async def write_miners(self, job_id, state, etag):
        from reliquary.infrastructure.corpus_record_store import CorpusStoreConflict

        self.write_attempts += 1
        if self._always_fail or self.write_attempts <= self._fail_first_n:
            # Simulate a concurrent writer winning the race: the underlying
            # store is untouched by this call.
            raise CorpusStoreConflict("stale etag")
        return await self._inner.write_miners(job_id, state, etag)


def test_a_conflict_is_re_read_and_the_change_reapplied(r2):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    store = _FlakyMinersStore(BucketRecordStore(), fail_first_n=1)
    states = MinerStates(store, "job-x")

    result = asyncio.run(states.update("A", lambda m: replace(m, audited_passed=m.audited_passed + 1)))

    assert store.write_attempts == 2
    assert result.audited_passed == 1
    assert asyncio.run(states.get("A")).audited_passed == 1


def test_attempts_exhausted_raises_conflict(r2):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore, CorpusStoreConflict
    from reliquary.validator.corpus_miner_states import MinerStates

    store = _FlakyMinersStore(BucketRecordStore(), always_fail=True)
    states = MinerStates(store, "job-x")

    with pytest.raises(CorpusStoreConflict):
        asyncio.run(states.update("A", lambda m: replace(m, audited_passed=1), attempts=2))
    assert store.write_attempts == 2
