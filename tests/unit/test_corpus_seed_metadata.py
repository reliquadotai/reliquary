"""A judge restarted on a big backlog learns each pending record's hotkey,
receipt and size from the record's last bytes (the stored JSON is sorted,
so they follow the completions), not from the whole record: on 2026-10-02 the
math judge read 122k full records (12-24 GB) before its first pass."""

from __future__ import annotations

import asyncio
import json

import pytest

from reliquary.infrastructure import corpus_record_store as record_store
from reliquary.infrastructure.corpus_job_store import _encode
from reliquary.validator import corpus_auditor
from tests.unit import corpus_judge_sim as sim

SID = "ab" * 32


def _record(**overrides):
    record = {"schema": "reliquary/corpus-record/v1", "submission_id": SID, "job_id": "math-v1",
              "hotkey": "5HkAlice", "cursor": 3, "prompt_index": 10, "rendered_prompt": "p" * 300,
              "received_at": 1790000123.25, "token_count": 4200,
              "completions": [{"tokens": list(range(4200)),
                               "text": 'say "hotkey":"5Mallory" and "token_count":1',
                               "proofs": ["A" * 344] * 132}]}
    record.update(overrides)
    return _encode(record)


def test_the_metadata_is_read_from_the_tail_of_a_stored_record():
    body = _record()
    assert len(body) > 50_000
    meta = record_store.submission_meta(body[-4096:])
    assert meta == {"hotkey": "5HkAlice", "received_at": 1790000123.25, "token_count": 4200}


def test_text_that_spells_the_keys_never_counts():
    meta = record_store.submission_meta(_record(rendered_prompt='"hotkey":"5Eve"')[-4096:])
    assert meta["hotkey"] == "5HkAlice"


def test_a_tail_that_does_not_reach_the_keys_asks_for_the_whole_record():
    body = _record(rendered_prompt="q" * 20_000)
    assert record_store.submission_meta(body[-4096:]) is None
    assert record_store.submission_meta(body)["token_count"] == 4200


def test_an_older_record_without_receipt_reads_as_none():
    body = json.loads(_record())
    del body["received_at"]
    assert record_store.submission_meta(_encode(body)[-4096:])["received_at"] is None


class _RangeClient:
    """get_object with an HTTP suffix range, as R2 answers it."""

    def __init__(self, objects):
        self.objects, self.ranges = objects, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_object(self, Bucket, Key, Range=None):
        from botocore.exceptions import ClientError

        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body = self.objects[Key]
        self.ranges.append(Range)
        if Range is not None:
            body = body[-int(Range.removeprefix("bytes=-")):]

        class _Body:
            async def read(self):
                return body

            def close(self):
                pass

        return {"Body": _Body(), "ETag": '"e"'}


@pytest.mark.parametrize("prompt,ranges", [("p" * 300, ["bytes=-8192"]),
                                           ("q" * 20_000, ["bytes=-8192", None])])
def test_the_store_reads_a_tail_and_falls_back_to_the_whole_record(monkeypatch, prompt, ranges):
    key = record_store._key("math-v1", "submissions", SID)
    client = _RangeClient({key: _record(rendered_prompt=prompt)})
    monkeypatch.setattr(record_store, "get_s3_client", lambda **kw: client)
    store = record_store.BucketRecordStore()
    meta = asyncio.run(store.read_submission_meta("math-v1", SID))
    assert meta == {"hotkey": "5HkAlice", "received_at": 1790000123.25, "token_count": 4200}
    assert client.ranges == ranges
    assert asyncio.run(store.read_submission_meta("math-v1", "cd" * 32)) is None


# -- a 120k backlog: the first verdicts within minutes ------------------------------


class _BigRecords(sim.Store):
    """Full records of ~100 KB at 17 MB/s shared; a tail read costs the latency."""

    def __init__(self, *, metadata: bool, **kw):
        super().__init__(**kw)
        self.full_reads = 0
        if not metadata:
            self.read_submission_meta = None

    async def read_submission(self, job_id, sid):
        self.full_reads += 1
        await asyncio.sleep(0.1 * 16 / 17)   # 100 KB of a 17 MB/s link shared by 16 reads
        return await super().read_submission(job_id, sid)

    async def read_submission_meta(self, job_id, sid):
        await self._io("read_meta")
        record = self.submissions.get(sid)
        if record is None:
            return None
        return {k: record.get(k) for k in ("hotkey", "received_at", "token_count")}

    async def read_settlement(self, job_id):
        return {}, None


def _first_verdict_after_restart(metadata: bool, backlog: int = 120_000) -> tuple[float, int]:
    store = _BigRecords(metadata=metadata, latency=(0.05, 0.10), write_latency=(0.2, 0.4))
    result = sim.simulate(corpus_auditor, rate_per_hour=6900.0, hours=0.75, store=store,
                          start_backlog=backlog, backlog_age=15 * 3600.0, sample_every=60.0)
    firsts = [hours for hours, _, _, verdicts in result.pending_series if verdicts]
    return (firsts[0] * 60 if firsts else float("inf")), store.full_reads


def test_a_120k_backlog_of_large_records_gets_verdicts_within_minutes():
    minutes, full_reads = _first_verdict_after_restart(metadata=True)
    assert minutes <= 8.0, minutes   # measured 6 (whole records: 25)
    # Whole records are read only for the audits passes make, not to seed.
    assert full_reads < 20_000, full_reads


def test_seeding_from_whole_records_takes_tens_of_minutes():
    """The failure itself (prod 2026-10-02 21:50 -> 22:14+): no pass before the seed ends."""
    minutes, _ = _first_verdict_after_restart(metadata=False)
    assert minutes > 15.0, minutes
