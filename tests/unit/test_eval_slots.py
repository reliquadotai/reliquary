"""Eval jobs stay completable: a failed submission reopens its prompt's slot,
up to 3 x V attempts, then the prompt is exhausted (v2 cross-side fix 2)."""

from __future__ import annotations

import asyncio

import pytest

from reliquary.corpus.slots import ATTEMPTS_PER_SLOT, SlotExhausted, SlotLedger


def test_a_failure_reopens_a_slot_until_every_attempt_is_used():
    slots = SlotLedger(2, 2)
    slots.consume(0)
    slots.consume(0)
    assert slots.remaining(0) == 0 and slots.prompt_state(0) == "complete"
    assert slots.record_failure(0, "a" * 64) is True
    assert slots.record_failure(0, "a" * 64) is None  # idempotent
    assert slots.remaining(0) == 1 and slots.prompt_state(0) == "open"
    slots.consume(0)
    assert slots.prompt_state(0) == "complete"
    # Attempts run out at 3 x V = 6 slots for the prompt.
    for k in range(4):
        slots.record_failure(0, str(k) * 64)
    while slots.remaining(0):
        slots.consume(0)
    assert slots.capacity(0) == ATTEMPTS_PER_SLOT * 2 == 6
    assert slots.record_failure(0, "f" * 64) is False
    assert slots.prompt_state(0) == "exhausted"
    with pytest.raises(SlotExhausted):
        slots.consume(0)
    assert slots.prompt_counts() == {"complete": 0, "exhausted": 1, "open": 1}
    assert slots.total == 2 * 2 + 4


def test_failures_round_trip_through_the_snapshot():
    slots = SlotLedger(3, 1, prompt_start=10)
    slots.consume(10)
    slots.record_failure(10, "b" * 64)
    again = SlotLedger.from_snapshot(3, 1, slots.snapshot(), prompt_start=10,
                                     failed=slots.failed_snapshot())
    assert again.remaining(10) == 1 and again.failed_snapshot() == {10: ["b" * 12]}
    with pytest.raises(ValueError):
        SlotLedger.from_snapshot(3, 1, {}, prompt_start=10, failed={99: ["x"]})


def test_a_ledger_without_failures_is_byte_identical():
    from reliquary.validator.corpus_service import CursorLedger, ledger_snapshot

    slots = SlotLedger(2, 2)
    slots.consume(1)
    plain = ledger_snapshot(slots, CursorLedger(), ())
    assert "failed" not in plain
    slots.record_failure(1, "c" * 64)
    assert ledger_snapshot(slots, CursorLedger(), ())["failed"] == {"1": ["c" * 12]}


def test_the_ledger_records_a_failure_under_its_compare_and_swap(monkeypatch):
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.validator.corpus_service import record_prompt_failure, rebuild_ledgers
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2
    from tests.unit.test_eval_prompt_source import _job

    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: fake)
    store = BucketJobStore()
    job = _job("eval-set:s:4:" + "0" * 64, count=4)

    async def go():
        assert await record_prompt_failure(store, job, 2, "d" * 64) is True
        assert await record_prompt_failure(store, job, 2, "d" * 64) is None
        snapshot, _ = await store.read_ledgers(job.job_id)
        state = rebuild_ledgers(job, snapshot)
        assert state.slots.capacity(2) == job.slots_per_prompt + 1

    asyncio.run(go())
