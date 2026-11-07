"""A hotkey's probation progress is kept across the orders that share a trust
document: a new order does not put a trusted hotkey back in probation, and a
confirmed failure anywhere resets it everywhere. Suspicion and bans stay per order."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from reliquary.corpus.audit_policy import AuditParams, after_confirmed_failure, after_pass, effective_state
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure import corpus_record_store as records
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2

TRUST = "miner-trust-orders"
PARAMS = AuditParams(q=0.15, probation_submissions=3)
SID = "{:064x}".format


@pytest.fixture
def r2(monkeypatch):
    client = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: client)
    monkeypatch.setattr(records, "get_s3_client", lambda **kw: client)
    return client


def _states(job_id):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    return MinerStates(BucketRecordStore(), job_id, trust_id=TRUST,
                       trust_cap=PARAMS.probation_submissions)


def _pass(i):
    return lambda m: after_pass(m, PARAMS, 0.0, SID(i))


def _fail(i, now):
    return lambda m: after_confirmed_failure(m, PARAMS, now, SID(i))


def _trust_document():
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore

    return asyncio.run(BucketRecordStore().read_miners(TRUST))[0]


def test_probation_passed_in_one_order_holds_in_the_next(r2):
    a, b = _states("order-gen-a"), _states("order-gen-b")
    for i in range(3):
        asyncio.run(a.update("hk", _pass(i)))
    assert effective_state(asyncio.run(b.get("hk")), 0.0, PARAMS) == "sampled"
    assert effective_state(asyncio.run(b.get_many(["hk"]))["hk"], 0.0, PARAMS) == "sampled"


def test_a_ban_stays_in_its_order(r2):
    a, b = _states("order-gen-a"), _states("order-gen-b")
    for i in range(3):
        asyncio.run(b.update("hk", _pass(i)))
    for i in range(PARAMS.ban_after_failures):
        asyncio.run(a.update("hk", _fail(10 + i, now=100.0 + i)))
    assert effective_state(asyncio.run(a.get("hk")), 200.0, PARAMS) == "banned"
    # The other order sees only the reset probation.
    state, _ = asyncio.run(b.state_at_most("hk", 0.0))
    assert effective_state(state, 200.0, PARAMS) == "probation"


def test_a_failure_in_one_order_resets_the_standing_everywhere(r2):
    a, b = _states("order-gen-a"), _states("order-gen-b")
    for i in range(3):
        asyncio.run(a.update("hk", _pass(i)))
    asyncio.run(b.update("hk", _fail(10, now=100.0)))
    seen = asyncio.run(a.get("hk"))
    assert seen.audited_passed == 0 and effective_state(seen, 150.0, PARAMS) == "probation"
    assert effective_state(asyncio.run(b.get("hk")), 150.0, PARAMS) == "suspect"


def test_the_trust_document_holds_only_the_standing_capped_at_probation(r2):
    a = _states("order-gen-a")
    for i in range(10):
        asyncio.run(a.update("hk", lambda m, i=i: after_pass(m, PARAMS, 1.5, SID(i))))
    entry = _trust_document()["hk"]
    assert entry["audited_passed"] == PARAMS.probation_submissions
    assert set(entry) == {"audited_passed"}
    # The order's own document keeps its full count and history.
    own = asyncio.run(a.get("hk"))
    assert own.audited_passed >= PARAMS.probation_submissions and len(own.pass_ids) == 10


def test_a_trusted_hotkeys_passes_no_longer_rewrite_the_trust_document(r2):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore

    writes = []

    class _Counting(BucketRecordStore):
        async def write_miners(self, job_id, state, etag):
            writes.append(job_id)
            return await super().write_miners(job_id, state, etag)

    from reliquary.validator.corpus_miner_states import MinerStates

    a = MinerStates(_Counting(), "order-gen-a", trust_id=TRUST, trust_cap=PARAMS.probation_submissions)
    for i in range(8):
        asyncio.run(a.update("hk", _pass(i)))
    assert writes.count(TRUST) == PARAMS.probation_submissions
    assert writes.count("order-gen-a") == 8


def test_without_a_trust_document_an_order_keeps_its_own_standing(r2):
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator.corpus_miner_states import MinerStates

    a = MinerStates(BucketRecordStore(), "job-a")
    b = MinerStates(BucketRecordStore(), "job-b")
    asyncio.run(a.update("hk", lambda m: replace(m, banned_until=1e12)))
    assert asyncio.run(b.get("hk")).banned_until is None
