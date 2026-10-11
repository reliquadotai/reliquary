"""The dataset a corpus job delivers: every finished completion of every
audited-and-passed submission. The job's filter is applied here, as an
annotation, because it never decides payment.

A completion is finished when its last token is the job's EOS and, in a
thinking render (the prompt opens ``<think>``), its text closes the block
exactly once. A completion cut at ``max_new_tokens`` passes TOPLOC and may
even grade right (a boxed grader reads the last box of the reasoning), but it
is no training example: it is left out before any grading and counted by
reason."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os

from reliquary.environment.grader import GRADER_SOCKET_PATH

logger = logging.getLogger(__name__)

THINK_CLOSE = "</think>"
UNFINISHED_REASONS = ("unfinished_no_eos", "unclosed_think", "multiple_think_close")


def thinking_render(job) -> bool:
    """Whether the job's renderer opens a reasoning block in the prompt. A
    legacy renderer is not a chat template and opens none."""
    from reliquary.validator.corpus_service import CHAT_TEMPLATE_RENDERERS

    return bool(CHAT_TEMPLATE_RENDERERS.get(getattr(job, "renderer_id", None), False))


def unfinished_reason(completion, *, eos_token_id: int, thinking: bool) -> str | None:
    """Why ``completion`` is not a finished trace, or None when it is."""
    tokens = completion.get("tokens") or ()
    if not tokens or int(tokens[-1]) != int(eos_token_id):
        return "unfinished_no_eos"
    if thinking:
        closes = completion["text"].count(THINK_CLOSE)
        if closes == 0:
            return "unclosed_think"
        if closes > 1:
            return "multiple_think_close"
    return None


def answer_text(text: str, *, thinking: bool) -> str:
    """The part a filter grades: after ``</think>`` in a thinking render (an
    unclosed block has no answer), the whole text otherwise."""
    if not thinking:
        return text
    return text.rsplit(THINK_CLOSE, 1)[1] if THINK_CLOSE in text else ""


def new_unfinished_counts() -> dict:
    return dict.fromkeys(UNFINISHED_REASONS, 0)


DEDUPE_KEY = "sha256(token_ids)"


def completion_fingerprint(completion) -> bytes:
    """What makes two completions the same one: their token ids (the proven
    form; one tokenizer per model), else their text."""
    tokens = completion.get("tokens")
    body = ("t:" + ",".join(str(int(t)) for t in tokens) if tokens
            else "s:" + completion.get("text", ""))
    return hashlib.sha256(body.encode()).digest()


async def job_fingerprints(records, job_ids, *, concurrency: int = 16) -> set[bytes]:
    """The fingerprints of every completion of the passing submissions of
    ``job_ids``: what their deliveries could hold, for a later job on the same
    prompts to leave out."""
    gate = asyncio.Semaphore(concurrency)

    async def read(call):
        async with gate:
            return await call

    seen: set[bytes] = set()
    for job_id in job_ids:
        ids = list(await records.list_verdict_ids(job_id))
        verdicts = await asyncio.gather(*(read(records.read_verdict(job_id, s)) for s in ids))
        passing = [s for s, v in zip(ids, verdicts) if v and v.get("passed")]
        found = await asyncio.gather(*(read(records.read_submission(job_id, s))
                                       for s in passing))
        for record in found:
            for completion in (record or {}).get("completions") or ():
                seen.add(completion_fingerprint(completion))
    return seen


def is_duplicate(completion, seen: set[bytes] | None) -> bool:
    """With ``seen`` (None: no dedupe), whether ``completion`` was seen; a new
    one is added, so a later copy is the duplicate."""
    if seen is None:
        return False
    fingerprint = completion_fingerprint(completion)
    if fingerprint in seen:
        return True
    seen.add(fingerprint)
    return False


async def export_rows(*, job, records, grade=None, counts: dict | None = None,
                      seen: set[bytes] | None = None):
    """One row per finished completion of every passing submission. ``counts``,
    when given, is filled in as the rows go. ``seen`` (``is_duplicate``)
    dedupes: a completion already in it is left out."""
    thinking = thinking_render(job)
    if counts is None:
        counts = {}
    counts.update(completions=0, rows=0, rows_accepted=0 if grade is not None else None,
                  unfinished=new_unfinished_counts(),
                  duplicate_completion=0 if seen is not None else None)
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
            counts["completions"] += 1
            reason = unfinished_reason(completion, eos_token_id=job.eos_token_id,
                                       thinking=thinking)
            if reason is not None:
                counts["unfinished"][reason] += 1
                continue
            if is_duplicate(completion, seen):
                counts["duplicate_completion"] += 1
                continue
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
                counts["rows_accepted"] += int(row["accepted"])
            counts["rows"] += 1
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
    own threshold, on the answer (``answer_text``): in a thinking render the
    text after ``</think>``. Every boxed grader (math, science, hard-math) reads
    the last box of what it is given, and code the last fenced block defining
    the entry point, so a box or a block left in the reasoning no longer grades
    a wrong final answer right; hard-math already reads past ``</think>``, so
    its verdict on a finished trace is unchanged. Raises ValueError for a job
    with no filter, or an episode-mode source, which cannot grade a single
    completion text."""
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
    thinking = thinking_render(job)

    def grade(prompt_index: int, text: str) -> tuple[bool, float]:
        problem = environment.get_problem(_owned_position(job, prompt_index))
        reward = score(problem, answer_text(text, thinking=thinking))
        return reward >= threshold, reward

    return grade
