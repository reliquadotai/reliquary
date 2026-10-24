"""A completion is delivered once: within a job, and against the jobs named
(a v2 job on the same prompts as its v1, where a miner may resubmit)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from reliquary.corpus.delivery import LocalDirectorySink, delivery_rows, export_delivery
from reliquary.corpus.export import (
    completion_fingerprint,
    export_rows,
    job_fingerprints,
)

EOS = 7


def _sid(i: int) -> str:
    return f"{i:064x}"


def _c(text, tokens):
    return {"text": text, "tokens": [*tokens, EOS]}


class _Records:
    """Job id -> {prompt_index: [completions]}; one passing submission each,
    and one failing submission per job whose completions were never delivered."""

    def __init__(self, jobs, failing=None):
        self.subs, self.verdicts = {}, {}
        self.voided = set()
        for job_id, by_prompt in jobs.items():
            for i, completions in by_prompt.items():
                sid = _sid(i)
                self.subs[(job_id, sid)] = {"hotkey": "HK", "prompt_index": i,
                                            "rendered_prompt": f"q{i}", "completions": completions}
                self.verdicts[(job_id, sid)] = {"passed": True}
        for job_id, completions in (failing or {}).items():
            sid = _sid(0xFFFFF)
            self.subs[(job_id, sid)] = {"hotkey": "HK", "prompt_index": 99,
                                        "rendered_prompt": "q99", "completions": completions}
            self.verdicts[(job_id, sid)] = {"passed": False}

    async def list_verdict_ids(self, job_id):
        return sorted(sid for j, sid in self.verdicts if j == job_id)

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get((job_id, sid))

    async def read_submission(self, job_id, sid):
        return self.subs.get((job_id, sid))

    async def list_voided_ids(self, job_id):
        return sorted(sid for j, sid in self.voided if j == job_id)

    async def read_voided(self, job_id, sid):
        return {} if (job_id, sid) in self.voided else None


V1 = {0: [_c("alpha", [1]), _c("beta", [2])]}
V2 = {0: [_c("alpha", [1]), _c("gamma", [3])],     # alpha resubmitted from v1
      1: [_c("gamma", [3]), _c("delta", [4])]}     # gamma twice within v2
FAILING = {"v1": [_c("delta", [4])]}               # a failed v1 submission delivered nothing

JOB = SimpleNamespace(job_id="v2", filter=None, renderer_id="chat-template-v1",
                      eos_token_id=EOS, episode=None,
                      to_contract=lambda: {"job_id": "v2"})


def test_the_fingerprint_is_the_token_ids_else_the_text():
    assert completion_fingerprint(_c("x", [1])) == completion_fingerprint(_c("y", [1]))
    assert completion_fingerprint(_c("x", [1])) != completion_fingerprint(_c("x", [2]))
    assert completion_fingerprint({"text": "x"}) == completion_fingerprint({"text": "x"})
    assert completion_fingerprint({"text": "x"}) != completion_fingerprint({"text": "y"})


def test_fingerprints_of_other_jobs_are_their_passing_completions():
    records = _Records({"v1": V1, "v2": V2}, failing=FAILING)
    seen = asyncio.run(job_fingerprints(records, ["v1"]))
    assert seen == {completion_fingerprint(c) for c in V1[0]}


@pytest.mark.parametrize("invalidate_after_read", [False, True])
def test_invalidated_history_does_not_suppress_a_later_valid_completion(
    monkeypatch, invalidate_after_read,
):
    records = _Records({"v1": V1, "v2": V2})
    if invalidate_after_read:
        read_submission = records.read_submission

        async def read(job_id, sid):
            record = await read_submission(job_id, sid)
            if job_id == "v1":
                records.voided.add((job_id, sid))
            return record

        monkeypatch.setattr(records, "read_submission", read)
    else:
        records.voided.add(("v1", _sid(0)))

    seen = asyncio.run(job_fingerprints(records, ["v1"]))
    rows, counts = _rows(records, seen)

    assert seen == {completion_fingerprint(c) for group in V2.values() for c in group}
    assert [row["completion"] for row in rows] == ["alpha", "gamma", "delta"]
    assert counts["duplicate_completion"] == 1


def _rows(records, seen):
    async def go():
        counts = {}
        rows = [r async for r in delivery_rows(job=JOB, records=records, counts=counts,
                                               seen=seen)]
        return rows, counts

    return asyncio.run(go())


def test_no_dedupe_keeps_every_copy():
    rows, counts = _rows(_Records({"v2": V2}), None)
    assert [r["completion"] for r in rows] == ["alpha", "gamma", "gamma", "delta"]
    assert counts["duplicate_completion"] is None


def test_dedupe_within_the_job_keeps_the_first_copy():
    rows, counts = _rows(_Records({"v2": V2}), set())
    assert [r["completion"] for r in rows] == ["alpha", "gamma", "delta"]
    assert counts["duplicate_completion"] == 1 and counts["rows"] == 3


def test_dedupe_against_v1_drops_what_v1_delivered():
    records = _Records({"v1": V1, "v2": V2}, failing=FAILING)
    seen = asyncio.run(job_fingerprints(records, ["v1"]))
    rows, counts = _rows(records, seen)
    assert [r["completion"] for r in rows] == ["gamma", "delta"]
    assert counts["duplicate_completion"] == 2


def test_a_duplicate_is_never_graded():
    graded = []

    async def go():
        counts = {}
        return [r async for r in delivery_rows(
            job=JOB, records=_Records({"v2": V2}), counts=counts, seen=set(),
            grade=lambda i, text: graded.append(text) or (True, 1.0))]

    asyncio.run(go())
    assert sorted(graded) == ["alpha", "delta", "gamma"]


def test_the_delivery_report_says_what_was_deduplicated(tmp_path):
    records = _Records({"v1": V1, "v2": V2}, failing=FAILING)
    sink = LocalDirectorySink(tmp_path / "bucket")
    manifest = asyncio.run(export_delivery(job=JOB, records=records, sink=sink,
                                           delivery_id="d1", work_dir=tmp_path / "work",
                                           clock=lambda: 1.0, dedupe_against=["v1"]))
    report = json.loads((tmp_path / "bucket" / manifest["report"]).read_text())
    assert manifest["rows"] == 2
    assert report["counts"]["duplicate_completion"] == 2
    assert report["dedupe"] == {"within_job": True, "against": ["v1"],
                                "fingerprints_against": 2, "key": "sha256(token_ids)"}


def test_a_delivery_cannot_dedupe_against_itself(tmp_path):
    with pytest.raises(ValueError, match="itself"):
        asyncio.run(export_delivery(job=JOB, records=_Records({"v2": V2}),
                                    sink=LocalDirectorySink(tmp_path), delivery_id="d1",
                                    dedupe_against=["v2"]))


def test_export_rows_dedupes_too():
    async def go():
        counts = {}
        rows = [r async for r in export_rows(job=JOB, records=_Records({"v2": V2}),
                                             counts=counts, seen=set())]
        return rows, counts

    rows, counts = asyncio.run(go())
    assert [r["completion"] for r in rows] == ["alpha", "gamma", "delta"]
    assert counts["duplicate_completion"] == 1
