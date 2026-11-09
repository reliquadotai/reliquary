"""A job's records are final (grading and delivery may read them) once every
submission is audited and nothing can reopen: pay settlement is not waited for."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from reliquary.corpus.slots import SlotLedger
from reliquary.validator.corpus_job_status import stored_job_counts
from reliquary.validator.corpus_service import CursorLedger, ledger_snapshot
from tests.unit.test_eval_prompt_source import _job

SID = {k: k * 64 for k in "abcd"}


class _Records:
    def __init__(self, job, ledgers, verdicts, settlement=None):
        self.job, self.ledgers, self.verdicts = job, ledgers, verdicts
        self.settlement = settlement or {"settled": [], "pending": None}
        self.verdict_reads = 0

    async def read_job(self, job_id):
        return self.job, "etag-job"

    async def read_ledgers(self, job_id):
        return self.ledgers, "etag-ledgers"

    async def list_submission_ids(self, job_id):
        return list(self.verdicts)

    async def list_verdict_ids(self, job_id):
        return list(self.verdicts)

    async def read_settlement(self, job_id):
        return self.settlement, None

    async def read_verdict(self, job_id, sid):
        self.verdict_reads += 1
        return self.verdicts[sid]


def _eval_records(*, failed_recorded=True, open_prompt=False, settled=()):
    """A 2-prompt V=1 eval job: prompt 0 passed (a), prompt 1 failed (b) then passed (c)."""
    job = _job("eval-set:s:2:" + "0" * 64, count=2)
    slots = SlotLedger(job.prompt_count, job.slots_per_prompt, prompt_start=job.prompt_start)
    slots.consume(0)
    slots.consume(1)
    if failed_recorded:
        slots.record_failure(1, SID["b"])
        if not open_prompt:
            slots.consume(1)
    verdicts = {SID["a"]: {"passed": True}, SID["b"]: {"passed": False}}
    if failed_recorded and not open_prompt:
        verdicts[SID["c"]] = {"passed": True}
    return _Records(job, ledger_snapshot(slots, CursorLedger(), ()), verdicts,
                    {"settled": list(settled), "pending": None})


def test_an_audited_eval_job_is_final_before_its_period_settles():
    counts = asyncio.run(stored_job_counts(_eval_records(), "j"))
    assert counts["unsettled"] == 3 and counts["drained"] is False
    assert counts["final"] is True


def test_a_failed_verdict_not_yet_recorded_in_the_ledger_is_not_final():
    # The auditor writes the verdict, then reopens the slot: in between, the job looks complete.
    records = _eval_records(failed_recorded=False)
    counts = asyncio.run(stored_job_counts(records, "j"))
    assert counts["unaudited"] == 0 and counts["final"] is False


def test_an_open_prompt_is_not_final():
    counts = asyncio.run(stored_job_counts(_eval_records(open_prompt=True), "j"))
    assert counts["final"] is False


def test_verdict_outcomes_are_read_once_across_calls():
    records = _eval_records(failed_recorded=False)
    cache: dict[str, bool] = {}
    assert asyncio.run(stored_job_counts(records, "j", verdict_passed=cache))["final"] is False
    reads = records.verdict_reads
    asyncio.run(stored_job_counts(records, "j", verdict_passed=cache))
    assert records.verdict_reads == reads


def test_a_drained_job_is_final_and_reads_no_verdict():
    records = _eval_records(settled=[SID["a"], SID["b"], SID["c"]])
    counts = asyncio.run(stored_job_counts(records, "j"))
    assert counts["drained"] is True and counts["final"] is True and records.verdict_reads == 0


def test_an_episode_job_waits_for_its_drain():
    records = _eval_records()
    records.job = SimpleNamespace(**{k: getattr(records.job, k) for k in (
        "job_id", "prompt_count", "prompt_start", "slots_per_prompt", "prompt_source")},
        episode=object())
    counts = asyncio.run(stored_job_counts(records, "j"))
    assert counts["final"] is False


def test_the_settler_reports_verdicts_unsettled_at_boot_to_the_status():
    from tests.unit.test_corpus_settlement import _Archives, _Records as SettleRecords, _settler, _v

    records = SettleRecords({"1" * 64: _v("A", 10), "2" * 64: _v("B", 30, ok=False)})
    settler = _settler(records, _Archives())
    seen: dict[str, dict] = {}
    settler.on_read = seen.__setitem__
    settler.totals = {"verdicts": 0, "passed": 0, "verified_tokens": 0}
    assert settler.totals["complete"] is False
    asyncio.run(settler._verdict_ids(set()))
    assert sorted(seen) == ["1" * 64, "2" * 64]
    assert "complete" not in settler.totals
