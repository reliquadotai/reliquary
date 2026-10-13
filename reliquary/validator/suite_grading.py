"""The grader of a suite episode job (design 2026-10-13): the replay, here.

A SWE job is graded by remote executors because its grade runs the repository
tests in a box. A suite episode is graded by its environment, which the
validator already ran at intake: the replay that admitted the trajectory is
deterministic, so this grader runs it again from the stored record (a restart
loses nothing) and writes the same grade document a SWE grade writes, with
the replay's verdict as the grade. The settler, the payment gate and the
export read that document unchanged: ``status`` ``ok``, ``graded_success``
the environment's success, and a certified replay.

Everything the settler and the job set use comes from ``CorpusGrader``
(``ready``, ``oldest_unready_received_at``, ``enqueue``, ``run``, ``status``);
only ``grade_one`` differs. There are no executors, so nothing is ever decided
alone, held or regraded: a quarantine is not this grader's business.
"""

from __future__ import annotations

import logging

from reliquary.infrastructure.corpus_record_store import RECORD_SCHEMA_V2
from reliquary.validator.corpus_grading import GRADE_SCHEMA, CorpusGrader
from reliquary.validator.suite_intake import replay_kwargs

logger = logging.getLogger(__name__)

# What a grade document says graded it.
LOCAL_GRADER = "validator-replay"


class LocalEpisodeGrader(CorpusGrader):
    def __init__(self, *, job, records, replayer, **kwargs) -> None:
        super().__init__(job=job, records=records, dispatcher=None, renderer=None, source=None,
                         params=None, **kwargs)
        self._replayer = replayer

    def status(self) -> dict:
        return {**super().status(), "grader": LOCAL_GRADER}

    def hold_executor(self, executor_id: str) -> None:
        """No executor grades here: nothing to hold."""

    async def regrade_executor(self, executor_id: str) -> list[str]:
        return []

    async def grade_one(self, submission_id: str, *, regrade: bool = False) -> dict | None:
        job_id = self._job.job_id
        record = await self._records.read_submission(job_id, submission_id)
        if record is None or record.get("schema") != RECORD_SCHEMA_V2:
            return None
        self._learn_arrival(submission_id, record.get("received_at"))
        base = {"schema": GRADE_SCHEMA, "submission_id": submission_id, "hotkey": record["hotkey"],
                "prompt_index": record["prompt_index"], "graded_by": [LOCAL_GRADER]}
        verdict = await self._records.read_verdict(job_id, submission_id)
        if verdict is not None and not verdict.get("passed"):
            return await self._write(submission_id, {
                **base, "status": "audit_failed", "graded_success": False, "replay": None,
                "replay_certified": False, "graded_at": self._clock()}, regrade=regrade)
        trajectory = record["completions"][0]
        outcome = await self._replayer.areplay(self._job.episode.env.name, **replay_kwargs(
            self._job, prompt_index=record["prompt_index"], prompt_ids=trajectory["prompt_tokens"],
            tokens=trajectory["tokens"],
            spans=[(turn["start"], turn["end"]) for turn in trajectory["turns"]],
            stop=trajectory["stop"]))
        if outcome.refusal is not None:
            # Intake replayed it already: only another environment or
            # tokenizer than the one that admitted it can land here. Nobody is
            # sanctioned; the row certifies nothing.
            reason, detail = outcome.refusal
            logger.error("accepted suite trajectory %s no longer replays: %s %s",
                         submission_id[:12], reason, detail)
            return await self._write(submission_id, {
                **base, "status": "unreplayable", "reason": reason, "graded_success": False,
                "replay": None, "replay_certified": False, "graded_at": self._clock()},
                regrade=regrade)
        replay = {"drawn": True, "status": "ok", "certified": True, "graded_by": [LOCAL_GRADER],
                  "reward": outcome.reward, "termination_reason": outcome.termination_reason,
                  "trace_digest": outcome.trace_digest}
        return await self._write(submission_id, {
            **base, "status": "ok", "instance_id": outcome.task_id,
            "graded_success": bool(outcome.success),
            "grade": {"reward": outcome.reward, "success": bool(outcome.success)},
            "replay": replay, "replay_certified": True, "graded_at": self._clock()},
            regrade=regrade)


__all__ = ["LOCAL_GRADER", "LocalEpisodeGrader"]
