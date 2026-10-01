"""The dataset a corpus job delivers: every completion of every audited-and-passed
submission. The job's filter is applied here, as an annotation, because it
never decides payment."""

from __future__ import annotations

import logging
import os

from reliquary.environment.grader import GRADER_SOCKET_PATH

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


def reward_scorer(spec, environment):
    """``score(problem, text) -> float`` as admission scores it. A source whose
    reward comes from materials (code) is scored by its admission scorer, which
    sends the cases to the sandboxed grading service (runsc, no network); the
    package's own runner, which would execute model-written code on this host,
    is never called. An unreachable service raises, it never scores 0."""
    method = getattr(spec, "reward_materializer_method", None)
    if method is None:
        return environment.compute_reward
    from reliquary.environment.registry import _import_attribute

    scorer = _import_attribute(spec.scorer_path)
    materials_of = getattr(environment, method)

    def score(problem, text: str) -> float:
        return float(scorer(problem, [text], list(materials_of(problem)))[0])

    return score


def require_code_sandbox(spec) -> None:
    """Refuse to grade a code source where the grading service is not running."""
    if getattr(spec, "reward_materializer_method", None) is not None and not os.path.exists(
        GRADER_SOCKET_PATH
    ):
        raise ValueError(
            f"{spec.name!r} runs model-written code: grading it needs the sandboxed grading "
            f"service at {GRADER_SOCKET_PATH}, which is not running here"
        )


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
    require_code_sandbox(spec)
    environment = spec.create()
    score = reward_scorer(spec, environment)
    threshold = job.filter.threshold

    def grade(prompt_index: int, text: str) -> tuple[bool, float]:
        problem = environment.get_problem(_owned_position(job, prompt_index))
        reward = score(problem, text)
        return reward >= threshold, reward

    return grade
