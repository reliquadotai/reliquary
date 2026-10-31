"""A forced-draw sample's draw never changes: a failed slot reopens with the same draw, for anyone but the hotkey
that failed it, up to ATTEMPTS_PER_SLOT attempts; an honest redo is not a duplicate."""
from __future__ import annotations

import asyncio
import hashlib

from reliquary.corpus import eval_draw
from reliquary.corpus.slots import ATTEMPTS_PER_SLOT, SlotLedger
from tests.unit.test_eval_draw import EOS, SHA, _job

FAILER = "5Failer"
OTHER = "5Other"


def _failed_ledger(job, prompt=5, slot=0, hotkeys=(FAILER,)):
    slots = SlotLedger(job.prompt_count, job.slots_per_prompt)
    for k, hotkey in enumerate(hotkeys):
        slots.consume(prompt)
        assert slots.record_failure(prompt, f"{k}" * 64, slot=slot, hotkey=hotkey) is True
    return slots


def test_a_failure_is_recorded_on_its_slot_and_bars_its_hotkey():
    slots = _failed_ledger(_job())
    assert slots.slot_attempt(5, 0) == 1
    assert slots.slot_attempt(5, 1) == 0
    assert slots.barred(5, 0, FAILER) and not slots.barred(5, 0, OTHER) and not slots.barred(5, 1, FAILER)
    assert slots.record_failure(5, "0" * 64, slot=0, hotkey=FAILER) is None    # idempotent
    assert slots.slot_attempt(5, 0) == 1


def test_slot_failures_round_trip_through_the_snapshot():
    job = _job()
    slots = _failed_ledger(job)
    again = SlotLedger.from_snapshot(job.prompt_count, job.slots_per_prompt, slots.snapshot(),
                                     failed=slots.failed_snapshot())
    assert again.slot_attempt(5, 0) == 1 and again.barred(5, 0, FAILER)
    assert again.capacity(5) == slots.capacity(5)


def test_a_legacy_failure_reopens_capacity_but_names_no_slot():
    slots = SlotLedger(30, 4)
    slots.consume(5)
    assert slots.record_failure(5, "a" * 64) is True
    assert slots.capacity(5) == 5
    assert slots.slot_attempt(5, 0) == 0 and not slots.barred(5, 0, FAILER)


def test_a_slot_that_used_every_attempt_exhausts_a_prompt_that_cannot_complete():
    job = _job(slots_per_prompt=2)
    slots = _failed_ledger(job, slot=0, hotkeys=[f"5H{k}" for k in range(ATTEMPTS_PER_SLOT)])
    assert slots.slot_exhausted(5, 0) and not slots.slot_exhausted(5, 1)
    assert slots.prompt_state(5) == "open"            # slot 1 can still be filled
    slots.consume(5)                                   # slot 1 passes (pending counts as passing)
    assert slots.prompt_state(5) == "exhausted"


def test_the_slot_key_of_a_first_attempt_is_unchanged():
    legacy = hashlib.sha256(f"{eval_draw.EVAL_DRAW_DOMAIN}/slot/5/2".encode()).hexdigest()
    assert eval_draw.slot_key(5, 2) == eval_draw.slot_key(5, 2, 0) == legacy
    assert eval_draw.slot_key(5, 2, 1) not in (legacy, eval_draw.slot_key(5, 2, 2))


def test_a_draw_job_has_one_cursor_per_logical_slot():
    job = _job()
    assert eval_draw.slot_count(job) == job.slots_per_prompt


def _admit(job, seen, slots, *, hotkey, cursor, prompt_index=5):
    from reliquary.corpus.admission import admit
    from reliquary.corpus.walk import CursorLedger

    return admit(job, hotkey=hotkey, cursor=cursor, prompt_index=prompt_index, checkpoint_sha256=SHA,
                 token_counts=[10, 12], last_token_ids=[EOS, EOS], digests=["d0", "d1"],
                 slots=slots, cursors=CursorLedger(), seen=seen)


def test_a_failed_slot_reopens_with_the_same_cursor_for_another_hotkey():
    job = _job()
    slots = _failed_ledger(job)
    seen = frozenset({eval_draw.slot_key(5, 0, 0)})
    assert _admit(job, seen, slots, hotkey=OTHER, cursor=0).accepted
    barred = _admit(job, seen, slots, hotkey=FAILER, cursor=0)
    assert (barred.accepted, barred.reason) == (False, eval_draw.REASON_DRAW_SLOT_TAKEN)
    retaken = _admit(job, seen | {eval_draw.slot_key(5, 0, 1)}, slots, hotkey=OTHER, cursor=0)
    assert (retaken.accepted, retaken.reason) == (False, eval_draw.REASON_DRAW_SLOT_TAKEN)


def test_a_cursor_past_the_logical_slots_is_refused():
    job = _job()
    slots = SlotLedger(job.prompt_count, job.slots_per_prompt)
    refused = _admit(job, frozenset(), slots, hotkey=OTHER, cursor=job.slots_per_prompt)
    assert (refused.accepted, refused.reason) == (False, eval_draw.REASON_BAD_CURSOR)


def test_an_exhausted_slot_is_taken_for_everyone():
    job = _job()
    slots = _failed_ledger(job, hotkeys=[f"5H{k}" for k in range(ATTEMPTS_PER_SLOT)])
    refused = _admit(job, frozenset(), slots, hotkey=OTHER, cursor=0)
    assert (refused.accepted, refused.reason) == (False, eval_draw.REASON_DRAW_SLOT_TAKEN)


def test_an_honest_redo_is_not_a_duplicate_of_the_failed_attempt():
    first = eval_draw.attempt_digests(5, 0, 0, ["d0", "d1"])
    redo = eval_draw.attempt_digests(5, 0, 1, ["d0", "d1"])
    assert first == eval_draw.attempt_digests(5, 0, 0, ["d0", "d1"])
    assert not set(first) & set(redo)
    assert len(set(first)) == 2


def test_the_ledger_records_the_failed_slot_and_hotkey(monkeypatch):
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.validator.corpus_service import rebuild_ledgers, record_prompt_failure
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2
    from tests.unit.test_eval_prompt_source import _job as eval_job

    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: fake)
    store = BucketJobStore()
    job = eval_job("eval-set:s:4:" + "0" * 64, count=4)

    async def go():
        assert await record_prompt_failure(store, job, 2, "d" * 64, slot=1, hotkey=FAILER) is True
        snapshot, _ = await store.read_ledgers(job.job_id)
        state = rebuild_ledgers(job, snapshot)
        assert state.slots.slot_attempt(2, 1) == 1 and state.slots.barred(2, 1, FAILER)

    asyncio.run(go())


def test_a_ledger_turn_records_the_attempt_keys_and_admits_the_honest_redo():
    from reliquary.corpus.admission import admit
    from reliquary.validator.corpus_service import (_TurnEntry, draw_seen_keys, ledger_snapshot, plan_turn,
                                                    rebuild_ledgers)

    job = _job()
    raw = ["d0", "d1"]          # the same tokens every time: the draw fixes them

    def entry(hotkey):
        def decide(slots, cursors, seen):
            return admit(job, hotkey=hotkey, cursor=0, prompt_index=5, checkpoint_sha256=SHA,
                         token_counts=[10, 12], last_token_ids=[EOS, EOS],
                         digests=eval_draw.attempt_digests(5, 0, slots.slot_attempt(5, 0), raw),
                         slots=slots, cursors=cursors, seen=seen)
        return _TurnEntry(job, decide, draw_seen_keys(5, 0, raw), None, 0.0)

    def turn(snapshot, hotkey):
        state = rebuild_ledgers(job, snapshot)
        plan = plan_turn(job, state, snapshot, [entry(hotkey)], frozenset())
        return plan.outcomes[0], plan.after

    first, after = turn({}, FAILER)
    assert first.accepted
    assert set(after["seen_pending"]) == {*eval_draw.attempt_digests(5, 0, 0, raw), eval_draw.slot_key(5, 0, 0)}

    state = rebuild_ledgers(job, after)
    state.slots.record_failure(5, "f" * 64, slot=0, hotkey=FAILER)
    failed = ledger_snapshot(state.slots, state.cursors, state.pending, state.segments)

    barred, _ = turn(failed, FAILER)
    assert (barred.accepted, barred.reason) == (False, eval_draw.REASON_DRAW_SLOT_TAKEN)
    redo, after = turn(failed, OTHER)
    assert redo.accepted
    assert {*eval_draw.attempt_digests(5, 0, 1, raw), eval_draw.slot_key(5, 0, 1)} <= set(after["seen_pending"])


def test_both_auditors_record_the_failed_slot_and_its_hotkey(monkeypatch):
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.validator.corpus_service import rebuild_ledgers
    from tests.unit.test_corpus_job_store import _FakeMultiObjectR2
    from tests.unit.test_eval_draw import _draw_auditor, _Verdicts

    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: _FakeMultiObjectR2())
    store, job = BucketJobStore(), _job()
    bad = "b" * 64
    records = _Verdicts({bad: {"prompt_index": 2, "cursor": 1, "hotkey": FAILER}})
    auditor = _draw_auditor(job, records, store)

    async def go():
        await auditor._write(bad, {"passed": False, "reason": eval_draw.REASON_DRAW_MISMATCH})
        snapshot, _ = await store.read_ledgers(job.job_id)
        slots = rebuild_ledgers(job, snapshot).slots
        assert slots.slot_attempt(2, 1) == 1 and slots.barred(2, 1, FAILER)

    asyncio.run(go())


def test_a_record_without_a_cursor_or_hotkey_reopens_capacity_only():
    from reliquary.validator.corpus_service import draw_failure_slot

    assert draw_failure_slot({"prompt_index": 2, "cursor": 1, "hotkey": FAILER}) == {"slot": 1, "hotkey": FAILER}
    assert draw_failure_slot({"prompt_index": 2}) == {}
    assert draw_failure_slot({"prompt_index": 2, "cursor": True, "hotkey": FAILER}) == {}


def test_a_miner_forgets_a_taken_slot_so_a_reopened_one_is_mined_again():
    from reliquary.miner.forced_draw import TakenSlots

    taken = TakenSlots(ttl_s=300.0)
    taken.add(5, 0, now=1000.0)
    assert taken.view(now=1200.0) == {5: {0}}
    assert taken.view(now=1301.0) == {}
