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
# Deadline misses of one submission before it is graded unreplayable.
DEADLINE_RETRIES = 3


class LocalEpisodeGrader(CorpusGrader):
    def __init__(self, *, job, records, replayer, source, **kwargs) -> None:
        """``source`` is the job's ``SuiteSource`` (the intake's): the task a
        row replays is looked up through its row map."""
        # As many grades in flight as there are replay workers: more would
        # only queue in the pool (``GRADE_CONCURRENCY`` is sized for remote
        # executors), and the intake shares that pool.
        kwargs.setdefault("concurrency", max(1, int(getattr(replayer, "workers", 2))))
        super().__init__(job=job, records=records, dispatcher=None, renderer=None, source=None,
                         params=None, **kwargs)
        self._replayer = replayer
        self._suite_source = source
        self._deadline_misses: dict[str, int] = {}

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
            self._job, self._suite_source, prompt_index=record["prompt_index"],
            prompt_ids=trajectory["prompt_tokens"], tokens=trajectory["tokens"],
            spans=[(turn["start"], turn["end"]) for turn in trajectory["turns"]],
            stop=trajectory["stop"]))
        # A pool that cannot answer (``ReplayUnavailable``) raises: logged and
        # retried at the next rescan, as a grade that failed to run.
        if outcome.deadline:
            # Past its deadline on a worker's own clock, after intake replayed
            # it within that deadline: the host was starved, not the
            # trajectory wrong. Left ungraded (unpaid meanwhile) and retried,
            # a bounded number of times so one submission never holds every
            # period: then graded as it is, unreplayable.
            misses = self._deadline_misses.get(submission_id, 0) + 1
            self._deadline_misses[submission_id] = misses
            if misses < DEADLINE_RETRIES:
                logger.warning("suite replay of %s ran past its deadline (%d/%d); retried at "
                               "the next rescan", submission_id[:12], misses, DEADLINE_RETRIES)
                return None
            logger.error("suite replay of %s ran past its deadline %d times; graded "
                         "unreplayable", submission_id[:12], misses)
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
