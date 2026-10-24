"""The grader of a signed-sandbox episode job (plan 3 ruling 11). The reward is the
final record's, so the grade document is written as soon as the accepted record is
read: no executor, no replay, no drand draw. Everything else (what is ready to pay,
the oldest ungraded arrival a period settler waits for, rescans, the void hook) is
CorpusGrader's own bookkeeping: only `grade_one` differs.

The transcript (up to 8 MiB) is parsed on the grader's parse executor under its parse
gate, never on the event loop. The grade carries `replay: {"signed": true, "status":
"ok", "certified": true}` so `delivery.certified` exports it like a certified replay;
its only grader is `sandbox:<machine_id>`, never a grade executor, so a grade-executor
quarantine never holds it."""

from __future__ import annotations

import asyncio
import logging
import time

from reliquary.corpus.signed_parse import signed_records
from reliquary.infrastructure.corpus_record_store import RECORD_SCHEMA_V2
from reliquary.validator.corpus_grading import GRADE_SCHEMA, CorpusGrader
from reliquary.validator.corpus_judge_threads import run_in

logger = logging.getLogger(__name__)


class SignedEpisodeGrader(CorpusGrader):
    def __init__(self, *, job, records, source, clock=time.time, **kwargs) -> None:
        super().__init__(job=job, records=records, dispatcher=None, renderer=None, source=source,
                         params=None, clock=clock, **kwargs)

    async def grade_one(self, submission_id: str, *, regrade: bool = False) -> dict | None:
        job_id = self._job.job_id
        record = await self._records.read_submission(job_id, submission_id)
        if record is None or record.get("schema") != RECORD_SCHEMA_V2:
            return None
        self._learn_arrival(submission_id, record.get("received_at"))
        base = {"schema": GRADE_SCHEMA, "submission_id": submission_id, "hotkey": record["hotkey"],
                "prompt_index": record["prompt_index"]}
        verdict = await self._records.read_verdict(job_id, submission_id)
        if verdict is not None and not verdict.get("passed"):
            return await self._write(submission_id, {
                **base, "status": "audit_failed", "graded_success": False, "replay": None,
                "replay_certified": False, "graded_at": self._clock()}, regrade=regrade)
        if self._parse_gate is None:
            self._parse_gate = asyncio.Semaphore(self._parse_concurrency)
        try:
            transcript = record["completions"][0].get("transcript")
            async with self._parse_gate:
                final = (await run_in(self._parse_executor, signed_records, transcript)).final
            claims = transcript["token"]["claims"]
            machine = str(claims["machine_id"])
            session_id = claims["session_id"]
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            logger.error("accepted signed trajectory %s has no readable transcript", submission_id[:12])
            return await self._write(submission_id, {
                **base, "status": "unparseable", "reason": "no_transcript", "graded_success": False,
                "replay": None, "replay_certified": False, "graded_at": self._clock()}, regrade=regrade)
        return await self._write(submission_id, {
            **base, "status": "ok", "instance_id": self._source.instance_id(record["prompt_index"]),
            "graded_success": final.reward == 1.0,
            "grade": {"reward": final.reward, "facts": final.grading,
                      "session_id": session_id, "machine_id": machine,
                      "state_sha256": final.state_sha256, "cpu_total_ms": final.cpu_total_ms},
            "graded_by": [f"sandbox:{machine}"], "replay": {"signed": True, "status": "ok", "certified": True},
            "replay_certified": True, "graded_at": self._clock()}, regrade=regrade)


__all__ = ["SignedEpisodeGrader"]
