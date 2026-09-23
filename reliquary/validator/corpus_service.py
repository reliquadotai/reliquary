"""The HTTP face of a corpus generation job.

This module is where the validator's own view of a submission is built.
``admit()`` decides, but it decides on numbers, and those numbers are this
module's obligation: ``token_counts``, ``last_token_ids`` and ``digests`` are
all derived here from the submitted token arrays. Nothing the miner *declares*
about its completions is forwarded — forward a digest and the duplicate check
becomes decoration. ``CorpusCompletion.termination`` is carried on the wire and
deliberately never read: the label the miner puts on its own work is a claim,
and ``check_termination`` derives the truth from the tokens instead.

Its own module rather than a handler inside ``server.py``: the corpus path
shares no state with the RL window machinery, and mounting is a separate,
reversible step.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
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

# Validators at different versions share one ledger object, and this repo ships
# `:latest` behind Watchtower, so an older reader that IGNORED a field it did
# not know would DELETE it on its next read-modify-write — silent loss on the
# money object. So an unknown field is refused. `schema` is what lets a field be
# added later without hard-failing every older validator on that job, and it can
# only be introduced before a real job exists.
LEDGER_SCHEMA = "reliquary/corpus-ledgers/v1"
LEDGER_FIELDS = frozenset({"schema", "slots", "cursors", "seen"})

# Contention is a two-writer race, not a queue, so a handful of rounds is
# plenty; past that the miner is better served by a retryable failure than by
# a request that never returns.
DEFAULT_WRITE_ATTEMPTS = 4

# Each resolved source holds a built environment, and a validator serves only a
# handful of live jobs at once, so the cache is bounded rather than growing with
# every job this process has ever seen.
MAX_RESOLVED_PROMPT_SOURCES = 8


class LedgerSnapshotError(Exception):
    """The stored ledgers do not describe this job.

    The job store moves snapshots as raw dicts and has no job in scope to
    check them against, so this is the first layer that can tell a corrupt
    ledger object from a valid one.
    """


class CorpusPromptSourceError(ValueError):
    """The job's ``prompt_source`` cannot be resolved to the rows it claims.

    A ``ValueError`` so that `jobs create`, which already refuses a manifest on
    ``ValueError``, rejects such a source at declaration.
    """


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


def resolve_prompt_source(
    prompt_source: str, *, environments: Mapping[str, Any] | None = None
) -> Any:
    """The environment spec a prompt source names, or a named refusal.

    Separate from building it, because `jobs create` applies the same rule: a
    job whose source cannot be rendered refuses every submission it is ever
    paid for, and the operator should learn that at declaration rather than
    from a reject-reason counter.
    """
    specs = ENVIRONMENT_SPECS if environments is None else environments
    try:
        spec = specs[prompt_source]
    except KeyError:
        raise CorpusPromptSourceError(
            f"prompt source {prompt_source!r} is not an installed environment"
        ) from None
    mode = getattr(spec, "interaction_mode", None)
    if mode != "episode":
        # A single-turn environment answers `get_problem`, not `get_task`, and
        # renders through `encode_prompt` rather than an episode renderer;
        # that is a second fidelity path, and it does not exist yet.
        raise CorpusPromptSourceError(
            f"prompt source {prompt_source!r} is {mode!r}; prompt fidelity "
            "needs an episode environment, whose rows render through "
            "`initial_text`"
        )
    return spec


def prompt_job_for_spec(
    job: JobSpec, *, environments: Mapping[str, Any] | None = None
) -> EnvironmentPromptJob:
    """Resolve a job's prompt source to the rows a fidelity check needs.

    Builds the environment, which for a real source reads a dataset — so
    callers hold the result for the life of the job rather than per request.
    """
    spec = resolve_prompt_source(job.prompt_source, environments=environments)
    try:
        environment = spec.create()
        rows = len(environment)
    except CorpusPromptSourceError:
        raise
    except Exception as exc:
        # A missing corpus directory or a broken wheel would otherwise reach
        # the handler as a bare 500, which is the anonymous failure the named
        # ledger and manifest errors already removed.
        raise CorpusPromptSourceError(
            f"prompt source {job.prompt_source!r} could not be built for job "
            f"{job.job_id!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if rows < job.prompt_count:
        raise CorpusPromptSourceError(
            f"job {job.job_id!r} claims {job.prompt_count} prompts but "
            f"{job.prompt_source!r} has {rows}"
        )
    return EnvironmentPromptJob(job, environment)


class PromptFidelity:
    """Spec §7's prompt-fidelity check, bound to a job's renderer and source.

    Bound here because this is the only place that holds both halves the check
    needs: the job's renderer and the environment its prompt source names.
    """

    __slots__ = ("_renderer", "_prompt_job_for", "_jobs", "_max_jobs", "_lock")

    def __init__(
        self,
        *,
        renderer: Renderer,
        prompt_job_for,
        max_jobs: int = MAX_RESOLVED_PROMPT_SOURCES,
    ) -> None:
        self._renderer = renderer
        self._prompt_job_for = prompt_job_for
        self._jobs: OrderedDict[str, Any] = OrderedDict()
        self._max_jobs = max_jobs
        self._lock = asyncio.Lock()

    async def __call__(
        self, rendered: str, *, job: JobSpec, prompt_index: int
    ) -> CheckResult:
        prompts = await self._prompt_job(job)
        # The comparison itself is a render and a string equality, so it stays
        # on the loop; only the build behind it does not.
        return check_prompt_fidelity(
            rendered,
            job=prompts,
            prompt_index=prompt_index,
            renderer=self._renderer,
        )

    async def _prompt_job(self, job: JobSpec):
        cached = self._jobs.get(job.job_id)
        if cached is not None:
            self._jobs.move_to_end(job.job_id)
            return cached
        # Resolving a source BUILDS its environment, and a dataset-backed one
        # reads from disk: doing that on the loop would stall every other
        # request this validator is serving, which is how `/state` froze once.
        async with self._lock:
            # Re-checked under the lock, so two first submissions for one job
            # build it once rather than racing.
            cached = self._jobs.get(job.job_id)
            if cached is not None:
                self._jobs.move_to_end(job.job_id)
                return cached
            cached = await asyncio.to_thread(self._prompt_job_for, job)
            self._jobs[job.job_id] = cached
            while len(self._jobs) > self._max_jobs:
                # Eviction costs one rebuild and never correctness: resolution
                # is a pure function of the manifest.
                self._jobs.popitem(last=False)
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
    # Absent on the empty read that precedes a job's first write, and on any
    # object written before the marker existed.
    schema = snapshot.get("schema", LEDGER_SCHEMA)
    if schema != LEDGER_SCHEMA:
        raise LedgerSnapshotError(
            f"job {job.job_id!r} has ledgers under schema {schema!r}, "
            f"not {LEDGER_SCHEMA!r}"
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
        "schema": LEDGER_SCHEMA,
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
    prompt_fidelity = PromptFidelity(renderer=renderer, prompt_job_for=prompt_job_for)
    # Also exposed, so the mount can reach the check without the handler.
    router.prompt_fidelity = prompt_fidelity

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
        except JobError as exc:
            # A manifest in the bucket that no longer parses is an operator
            # fault. `JobError` subclasses `ValueError`, so without this clause
            # the one below would disguise it as an unknown job — and since it
            # is caught here rather than left bare, the operator gets a name
            # instead of "Internal Server Error".
            logger.error("corpus job %s has an unreadable manifest: %s", request.job_id, exc)
            raise HTTPException(
                status_code=500, detail="corpus_job_manifest_corrupt"
            ) from exc
        except ValueError:
            # The id is not one the store could ever have written, so it names
            # no job; it must not become a 500 on a hostile request.
            job = None
        if job is None:
            return _refuse(
                CorpusRejectReason.JOB_UNKNOWN, {"job_id": request.job_id}
            )

        # The fidelity check indexes the prompt source, so a miner-controlled
        # index is bounded before it can raise on the operator's behalf. This
        # is the same rule `admit` applies, reached earlier.
        if request.prompt_index >= job.prompt_count:
            return _refuse(
                CorpusRejectReason.PROMPT_MISMATCH,
                {"prompt_count": job.prompt_count, "got": request.prompt_index},
            )
        try:
            fidelity = await prompt_fidelity(
                request.rendered_prompt, job=job, prompt_index=request.prompt_index
            )
        except CorpusPromptSourceError as exc:
            # The manifest names a source this binary cannot serve, so every
            # submission to this job fails identically. `jobs create` refuses
            # such a source, so reaching here means this validator does not
            # have the environments the declaring operator had.
            logger.error(
                "corpus job %s has an unusable prompt source: %s", request.job_id, exc
            )
            raise HTTPException(
                status_code=500, detail="corpus_prompt_source_unusable"
            ) from exc
        if not fidelity.ok:
            return _refuse(CorpusRejectReason(fidelity.reason), fidelity.detail)

        # Derived here, from the tokens alone. See this module's docstring.
        arrays = [completion.tokens for completion in request.completions]
        token_counts = [len(tokens) for tokens in arrays]
        last_token_ids = [tokens[-1] for tokens in arrays]
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
            try:
                slots, cursors, seen = rebuild_ledgers(job, snapshot)
            except LedgerSnapshotError as exc:
                # The blast radius is every miner on this job, and there is no
                # circuit breaker: the object stays corrupt until an operator
                # repairs it, so the refusal has to be named, not a bare 500.
                logger.error("corpus ledgers for %s are unreadable: %s", request.job_id, exc)
                raise HTTPException(
                    status_code=500, detail="corpus_ledger_corrupt"
                ) from exc
            before = ledger_snapshot(slots, cursors, seen)

            verdict = admit(
                job,
                hotkey=request.miner_hotkey,
                cursor=request.cursor,
                prompt_index=request.prompt_index,
                checkpoint_sha256=request.checkpoint_sha256,
                token_counts=token_counts,
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
    "LEDGER_SCHEMA",
    "MAX_RESOLVED_PROMPT_SOURCES",
    "LedgerSnapshotError",
    "PromptFidelity",
    "SUBMIT_PATH",
    "build_corpus_router",
    "ledger_snapshot",
    "prompt_job_for_spec",
    "rebuild_ledgers",
    "resolve_prompt_source",
]
