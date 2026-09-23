"""The HTTP face of a corpus generation job.

This module is where the validator's own view of a submission is built.
``admit()`` decides, but it decides on numbers, and those numbers are this
module's obligation: ``token_counts``, ``terminations``, ``last_token_ids``
and ``digests`` are all derived here from the submitted token arrays. Nothing
the miner *declares* about its completions is forwarded — forward the
termination label and ``check_termination`` collapses back into the label
check that was deliberately removed, forward a digest and the duplicate check
becomes decoration.

Its own module rather than a handler inside ``server.py``: the corpus path
shares no state with the RL window machinery, and mounting is a separate,
reversible step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from typing import Any, Protocol

from fastapi import APIRouter, HTTPException

from reliquary.corpus.admission import Verdict, admit
from reliquary.corpus.checks import CheckResult, completion_digest
from reliquary.corpus.job import JobError, JobSpec
from reliquary.corpus.slots import SlotLedger
from reliquary.corpus.walk import CursorLedger
from reliquary.environment.agentic.types import EpisodeTask
from reliquary.environment.registry import ENVIRONMENT_SPECS
from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict
from reliquary.protocol.corpus_submission import (
    CorpusRejectReason,
    CorpusSubmissionRequest,
    CorpusSubmissionResponse,
)
from reliquary.validator.corpus_text import (
    Renderer,
    check_prompt_fidelity,
    check_text_matches_tokens,
)

logger = logging.getLogger(__name__)

SUBMIT_PATH = "/corpus/submit"

# The ledger object's whole vocabulary. An unknown field is refused rather
# than ignored: a snapshot this binary cannot fully read may describe slots it
# is about to sell a second time.
LEDGER_FIELDS = frozenset({"slots", "cursors", "seen"})

# Contention is a two-writer race, not a queue, so a handful of rounds is
# plenty; past that the miner is better served by a retryable failure than by
# a request that never returns.
DEFAULT_WRITE_ATTEMPTS = 4


class LedgerSnapshotError(Exception):
    """The stored ledgers do not describe this job.

    The job store moves snapshots as raw dicts and has no job in scope to
    check them against, so this is the first layer that can tell a corrupt
    ledger object from a valid one.
    """


class CorpusPromptSourceError(Exception):
    """The job's ``prompt_source`` cannot be resolved to the rows it claims."""


class CorpusJobStore(Protocol):
    """The three store calls the endpoint makes, bound to their bucket."""

    async def read_job(self, job_id: str) -> tuple[JobSpec | None, str | None]: ...

    async def read_ledgers(self, job_id: str) -> tuple[dict, str | None]: ...

    async def write_ledgers(
        self, job_id: str, snapshot: Mapping[str, Any], etag: str | None
    ) -> str | None: ...


class Tokenizer(Protocol):
    def decode(self, ids: Sequence[int], **kwargs: Any) -> str: ...


# --------------------------------------------------------------------------
# The prompt source
# --------------------------------------------------------------------------


class EnvironmentPromptJob:
    """A ``JobSpec`` plus the environment its ``prompt_source`` names, in the
    shape ``check_prompt_fidelity`` takes.

    ``JobSpec`` carries the manifest and no prompt text, so it cannot satisfy
    ``PromptJob`` by itself; the environment holds the rows and the manifest
    says how many of them this job owns.
    """

    __slots__ = ("_job", "_environment")

    def __init__(self, job: JobSpec, environment: Any) -> None:
        self._job = job
        self._environment = environment

    def task_for(self, prompt_index: int) -> EpisodeTask:
        position = int(prompt_index)
        if position < 0 or position >= self._job.prompt_count:
            raise CorpusPromptSourceError(
                f"job {self._job.job_id!r} has {self._job.prompt_count} prompts; "
                f"{position} is outside it"
            )
        return self._environment.get_task(position)


def prompt_job_for_spec(
    job: JobSpec, *, environments: Mapping[str, Any] | None = None
) -> EnvironmentPromptJob:
    """Resolve a job's prompt source to the rows a fidelity check needs.

    Builds the environment, which for a real source reads a dataset — so
    callers hold the result for the life of the job rather than per request.
    """
    specs = ENVIRONMENT_SPECS if environments is None else environments
    try:
        spec = specs[job.prompt_source]
    except KeyError:
        raise CorpusPromptSourceError(
            f"job {job.job_id!r} names prompt source {job.prompt_source!r}, "
            "which is not an installed environment"
        ) from None
    if getattr(spec, "interaction_mode", None) != "episode":
        # A single-turn environment answers `get_problem`, not `get_task`, and
        # its prompt is rendered by `encode_prompt` rather than by an episode
        # renderer; that is a second fidelity path, not this one.
        raise CorpusPromptSourceError(
            f"prompt source {job.prompt_source!r} is not an episode environment"
        )
    environment = spec.create()
    rows = len(environment)
    if rows < job.prompt_count:
        raise CorpusPromptSourceError(
            f"job {job.job_id!r} claims {job.prompt_count} prompts but "
            f"{job.prompt_source!r} has {rows}"
        )
    return EnvironmentPromptJob(job, environment)


class PromptFidelity:
    """Spec §7's prompt-fidelity check, bound to a job's renderer and source.

    NOT called by the submission handler: ``CorpusSubmissionRequest`` carries
    completions only, so there is no rendered prompt to compare the job's own
    rendering against. It is bound here because this is the only place holding
    both halves the check needs, and it is exercised directly by its tests.
    """

    __slots__ = ("_renderer", "_prompt_job_for", "_jobs")

    def __init__(self, *, renderer: Renderer, prompt_job_for) -> None:
        self._renderer = renderer
        self._prompt_job_for = prompt_job_for
        self._jobs: dict[str, Any] = {}

    def __call__(
        self, rendered: str, *, job: JobSpec, prompt_index: int
    ) -> CheckResult:
        return check_prompt_fidelity(
            rendered,
            job=self._prompt_job(job),
            prompt_index=prompt_index,
            renderer=self._renderer,
        )

    def _prompt_job(self, job: JobSpec):
        # Resolving a source builds its environment, so it is done once per
        # job rather than once per submission.
        cached = self._jobs.get(job.job_id)
        if cached is None:
            cached = self._prompt_job_for(job)
            self._jobs[job.job_id] = cached
        return cached


# --------------------------------------------------------------------------
# The ledgers
# --------------------------------------------------------------------------


def rebuild_ledgers(
    job: JobSpec, snapshot: Any
) -> tuple[SlotLedger, CursorLedger, set[str]]:
    """The stored snapshot as live ledgers, or a named refusal.

    Rebuilt fresh on every attempt, because ``admit()`` mutates what it is
    given: a write that loses its compare-and-swap must not leave a consumed
    slot behind in the copy the retry then admits against.
    """
    if not isinstance(snapshot, Mapping):
        raise LedgerSnapshotError(
            f"job {job.job_id!r} has a ledger object that is not an object"
        )
    unknown = sorted(set(snapshot) - LEDGER_FIELDS)
    if unknown:
        raise LedgerSnapshotError(
            f"job {job.job_id!r} has ledger fields this binary cannot read: {unknown}"
        )
    try:
        slots = SlotLedger.from_snapshot(
            job.prompt_count, job.slots_per_prompt, snapshot.get("slots") or {}
        )
        cursors = CursorLedger.from_snapshot(snapshot.get("cursors") or {})
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise LedgerSnapshotError(
            f"job {job.job_id!r} has an unusable ledger snapshot: {exc}"
        ) from exc
    seen = snapshot.get("seen") or []
    if not isinstance(seen, (list, tuple)) or any(
        not isinstance(digest, str) for digest in seen
    ):
        raise LedgerSnapshotError(
            f"job {job.job_id!r} has a seen-digest list that is not a list of digests"
        )
    return slots, cursors, set(seen)


def ledger_snapshot(
    slots: SlotLedger, cursors: CursorLedger, seen: set[str]
) -> dict[str, Any]:
    """The JSON-native form the store persists. Prompt indices are stringified
    here rather than by the encoder, so a snapshot compares equal to the one
    that comes back out of the bucket."""
    return {
        "slots": {str(index): count for index, count in slots.snapshot().items()},
        "cursors": cursors.snapshot(),
        "seen": sorted(seen),
    }


# --------------------------------------------------------------------------
# The endpoint
# --------------------------------------------------------------------------


def _refuse(
    reason: CorpusRejectReason, detail: Mapping[str, Any] | None = None
) -> CorpusSubmissionResponse:
    return CorpusSubmissionResponse(
        reason=reason, accepted=False, detail=dict(detail or {})
    )


def _respond(verdict: Verdict) -> CorpusSubmissionResponse:
    try:
        reason = CorpusRejectReason(verdict.reason)
    except ValueError:
        # A verdict the wire has no name for must not turn a submission the
        # ledgers have already recorded into a 500.
        logger.error("corpus verdict %r has no wire reason", verdict.reason)
        return CorpusSubmissionResponse(
            reason=CorpusRejectReason.MALFORMED_SUBMISSION,
            accepted=verdict.accepted,
            slots_remaining=verdict.slots_remaining,
            detail={**verdict.detail, "verdict": verdict.reason},
        )
    return CorpusSubmissionResponse(
        reason=reason,
        accepted=verdict.accepted,
        slots_remaining=verdict.slots_remaining,
        detail=dict(verdict.detail),
    )


def build_corpus_router(
    *,
    store: CorpusJobStore,
    tokenizer: Tokenizer,
    renderer: Renderer,
    verify_signature,
    prompt_job_for=prompt_job_for_spec,
    max_write_attempts: int = DEFAULT_WRITE_ATTEMPTS,
) -> APIRouter:
    """The corpus submission endpoint, over an already-bound job store."""

    router = APIRouter()
    # The fidelity seam, bound where both the renderer and the prompt source
    # are in scope. See `PromptFidelity` for why the handler cannot call it.
    router.prompt_fidelity = PromptFidelity(
        renderer=renderer, prompt_job_for=prompt_job_for
    )

    @router.post(SUBMIT_PATH, response_model=CorpusSubmissionResponse)
    async def submit_corpus(
        request: CorpusSubmissionRequest,
    ) -> CorpusSubmissionResponse:
        # Before anything reads or writes: an unsigned submission must not
        # reach the ledgers, or a spoofed hotkey consumes another miner's work.
        if not verify_signature(request):
            return _refuse(CorpusRejectReason.BAD_SIGNATURE)

        try:
            job, _ = await store.read_job(request.job_id)
        except JobError:
            # A manifest in the bucket that no longer parses is an operator
            # fault, and calling it "unknown job" would hide it.
            raise
        except ValueError:
            # The id is not one the store could ever have written, so it names
            # no job; it must not become a 500 on a hostile request.
            job = None
        if job is None:
            return _refuse(
                CorpusRejectReason.JOB_UNKNOWN, {"job_id": request.job_id}
            )

        # Derived here, from the tokens alone. See this module's docstring.
        arrays = [completion.tokens for completion in request.completions]
        token_counts = [len(tokens) for tokens in arrays]
        last_token_ids = [tokens[-1] for tokens in arrays]
        terminations = [
            "eos" if tokens[-1] == job.eos_token_id else "cap" for tokens in arrays
        ]
        digests = [
            completion_digest(request.prompt_index, tokens) for tokens in arrays
        ]

        for completion in request.completions:
            text = check_text_matches_tokens(
                completion.tokens,
                completion.text,
                tokenizer=tokenizer,
                eos_token_id=job.eos_token_id,
            )
            if not text.ok:
                return _refuse(CorpusRejectReason(text.reason), text.detail)

        for _ in range(max_write_attempts):
            snapshot, etag = await store.read_ledgers(request.job_id)
            slots, cursors, seen = rebuild_ledgers(job, snapshot)
            before = ledger_snapshot(slots, cursors, seen)

            verdict = admit(
                job,
                hotkey=request.miner_hotkey,
                cursor=request.cursor,
                prompt_index=request.prompt_index,
                checkpoint_sha256=request.checkpoint_sha256,
                token_counts=token_counts,
                terminations=terminations,
                last_token_ids=last_token_ids,
                digests=digests,
                slots=slots,
                cursors=cursors,
                seen=seen,
            )
            if verdict.accepted:
                # `admit` reads `seen`, it does not grow it: recording what was
                # paid for is the caller's half of the duplicate check.
                seen.update(digests)

            after = ledger_snapshot(slots, cursors, seen)
            if after == before:
                # A refusal that moved nothing costs no write, so a miner
                # spraying junk cannot bill us a bucket write per attempt.
                return _respond(verdict)
            try:
                await store.write_ledgers(request.job_id, after, etag)
            except CorpusStoreConflict:
                continue
            return _respond(verdict)

        logger.warning(
            "corpus ledgers for %s stayed contended over %d attempts (miner %s)",
            request.job_id,
            max_write_attempts,
            request.miner_hotkey[:12],
        )
        # Nothing was consumed, so the same work resubmits cleanly.
        raise HTTPException(status_code=503, detail="corpus_ledger_contention")

    return router


__all__ = [
    "CorpusPromptSourceError",
    "EnvironmentPromptJob",
    "LedgerSnapshotError",
    "PromptFidelity",
    "SUBMIT_PATH",
    "build_corpus_router",
    "ledger_snapshot",
    "prompt_job_for_spec",
    "rebuild_ledgers",
]
