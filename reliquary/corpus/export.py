"""The dataset a corpus job delivers: every completion of every audited-and-passed
submission. The job's filter is applied here, as an annotation, because it
never decides payment."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def export_rows(*, job, records, grade=None):
    for submission_id in await records.list_verdict_ids(job.job_id):
        verdict = await records.read_verdict(job.job_id, submission_id)
        if not verdict or not verdict.get("passed"):
            continue
        record = await records.read_submission(job.job_id, submission_id)
        if record is None:
            # R2 is not transactional: a verdict can be visible before its
            # submission object is (or the object can be gone). Export pays
            # nothing, so skipping is safe -- crashing here would blank the
            # whole run over one bad row.
            logger.warning(
                "corpus export: submission %s has a passing verdict but no "
                "record, skipping", submission_id[:12]
            )
            continue
        for completion in record["completions"]:
            row = {
                "prompt": record["rendered_prompt"],
                "completion": completion["text"],
                "prompt_index": record["prompt_index"],
                "hotkey": record["hotkey"],
                "submission_id": submission_id,
            }
            if grade is not None:
                accepted, score = grade(record["prompt_index"], completion["text"])
                row["accepted"], row["score"] = bool(accepted), float(score)
            yield row


def job_grader(job):
    """The grader a job's filter annotates with: its own prompt source, at its
    own threshold. Raises ValueError for a job with no filter, or an
    episode-mode source, which cannot grade a single completion text."""
    from reliquary.environment.registry import ENVIRONMENT_SPECS
    from reliquary.validator.corpus_service import _owned_position

    if job.filter is None:
        raise ValueError(f"job {job.job_id!r} has no filter to apply")
    spec = ENVIRONMENT_SPECS[job.prompt_source]
    if spec.interaction_mode == "episode":
        raise ValueError(
            f"prompt source {job.prompt_source!r} is episode-mode; "
            "a filter cannot grade a single completion text against it"
        )
    environment = spec.create()
    threshold = job.filter.threshold

    def grade(prompt_index: int, text: str) -> tuple[bool, float]:
        problem = environment.get_problem(_owned_position(job, prompt_index))
        reward = environment.compute_reward(problem, text)
        return reward >= threshold, reward

    return grade
