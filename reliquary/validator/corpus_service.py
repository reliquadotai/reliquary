"""The HTTP face of a corpus generation job.

This module is where the validator's own view of a submission is built.
``admit()`` decides, but it decides on numbers, and those numbers are this
module's obligation: ``token_counts``, ``last_token_ids`` and ``digests`` are
all derived here from the submitted token arrays. Nothing the miner *declares*
about its completions is forwarded — forward a digest and the duplicate check
becomes decoration. The wire carries no termination label either: the miner's
word about how its own completion ended is a claim, ``check_termination``
derives the truth from the tokens, and a field nothing reads could only refuse
a miner that spells its label differently.

Its own module rather than a handler inside ``server.py``: the corpus path
shares no state with the RL window machinery, and mounting is a separate,
reversible step.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Mapping, Sequence
import logging
import time
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
JOB_PATH = "/corpus/job"
CURSOR_PATH = "/corpus/cursor/{hotkey}"

# The record's own schema tag, so a reader of the bucket can tell what shape
# to expect before it parses the rest of the document.
RECORD_SCHEMA = "reliquary/corpus-submission-record/v1"
# Mirrors `DEFAULT_WRITE_ATTEMPTS` below: a handful of rounds against a
# transient bucket fault, not a queue a miner's request should block behind.
RECORD_WRITE_ATTEMPTS = 3

# Validators at different versions share one ledger object, and this repo ships
# `:latest` behind Watchtower, so an older reader that IGNORED a field it did
# not know would DELETE it on its next read-modify-write — silent loss on the
# money object. So an unknown field is refused. `schema` does NOT buy
# compatibility: `LEDGER_FIELDS` and `rebuild_ledgers` still hard-fail an older
# reader on any new field. What it buys is a NAMED failure — "ledgers under
# schema X, not Y" instead of a field list — and it can only be introduced
# before a real job exists.
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


class CorpusSignatureUnavailable(Exception):
    """This validator cannot check any corpus signature, whatever it carries.

    Raised by a verifier instead of returning False, because the two are
    different facts: False says the miner's signature did not check out, and
    this says nothing about the miner at all.
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
        return self._environment.get_task(_owned_position(self._job, prompt_index))


class SingleTurnPromptJob:
    """The same shape over a single-turn environment, whose rows answer
    ``get_problem`` and come back already rendered.

    Both jobs satisfy one ``PromptJob``, so ``check_prompt_fidelity`` never
    learns which mode it is serving: the difference between an episode prompt
    and a single-turn one is entirely here and in the renderer beside it.
    """

    __slots__ = ("_job", "_environment")

    def __init__(self, job: JobSpec, environment: Any) -> None:
        self._job = job
        self._environment = environment

    def task_for(self, prompt_index: int) -> EpisodeTask:
        # `get_problem` WRAPS its index with modulo, so an out-of-range index
        # returns a valid prompt for a row this job does not own rather than
        # raising. Bounding before the call is what makes that impossible.
        position = _owned_position(self._job, prompt_index)
        problem = self._environment.get_problem(position)
        prompt = problem.get("prompt") if isinstance(problem, Mapping) else None
        if not isinstance(prompt, str) or not prompt:
            raise CorpusPromptSourceError(
                f"prompt source {self._job.prompt_source!r} returned no prompt "
                f"text for row {position}"
            )
        # The row's identity here is its index: fidelity compares the prompt
        # text, and carrying the environment's own id would only add a way for
        # a source to hand back something `EpisodeTask` refuses.
        return EpisodeTask(
            id=f"{self._job.prompt_source}#{position}", prompt=prompt, tools=()
        )


class SingleTurnPromptRenderer:
    """The renderer half of the single-turn path: the prompt is already
    rendered when the environment hands it over, so this hands it back.

    It exists so both modes present the same two pieces — a task and a
    renderer — to one check, rather than the check growing a branch.
    """

    @staticmethod
    def initial_text(task: EpisodeTask) -> str:
        return task.prompt


def _owned_position(job: JobSpec, prompt_index: int) -> int:
    """The index, or a refusal naming the job's own bound."""
    position = int(prompt_index)
    if position < 0 or position >= job.prompt_count:
        raise CorpusPromptSourceError(
            f"job {job.job_id!r} has {job.prompt_count} prompts; "
            f"{position} is outside it"
        )
    return position


def _declared_prompt_template_id(prompt_source: str, profile: Any | None) -> str:
    """The id of the prompt template a profile renders this environment with.

    ``None`` means the active profile, which under task isolation IS the corpus
    task's contract. A profile id is accepted too, so `jobs create` can ask
    about the contract it is declaring rather than the one its own process
    happens to run.
    """
    from reliquary.protocol import profiles

    if profile is None:
        resolved = profiles.ACTIVE_PROTOCOL_PROFILE
    elif isinstance(profile, str):
        resolved = profiles.resolve_protocol_profile(profile)
    else:
        resolved = profile
    profile_id = getattr(resolved, "profile_id", "?")
    try:
        environment_profile = resolved.environments[prompt_source]
    except KeyError:
        raise CorpusPromptSourceError(
            f"profile {profile_id!r} declares no environment {prompt_source!r}, "
            "so it says nothing about how that source's prompts are rendered"
        ) from None
    template = getattr(environment_profile, "prompt_template", None)
    if template is None:
        # Legacy profiles leave the prompt to environment-local concatenation,
        # which has no id: there would be nothing for the manifest to name and
        # nothing to check it against.
        raise CorpusPromptSourceError(
            f"profile {profile_id!r} declares no prompt template for "
            f"{prompt_source!r}, so its prompts have no rendering rule a "
            "manifest could pin"
        )
    return template.template_id


def resolve_prompt_source(
    prompt_source: str,
    *,
    environments: Mapping[str, Any] | None = None,
    renderer_id: str | None = None,
    profile: Any | None = None,
) -> Any:
    """The environment spec a prompt source names, or a named refusal.

    Separate from building it, because `jobs create` applies the same rule: a
    job whose source cannot be rendered refuses every submission it is ever
    paid for, and the operator should learn that at declaration rather than
    from a reject-reason counter.

    ``renderer_id`` is the manifest's, and for a single-turn source it is
    checked rather than trusted. Which of the two is authoritative has one
    answer: the PROFILE is, because the environment renders its own rows
    through it (`get_problem` -> `render_active_prompt`) and the manifest has
    no say in that. So the manifest may only NAME that rendering, and a
    manifest that names another one is refused here -- at declaration against
    the contract being declared, and again wherever the job is served, against
    the profile that validator actually runs. Two sources of truth for what the
    miner was asked cannot be left to agree by construction.
    """
    specs = ENVIRONMENT_SPECS if environments is None else environments
    try:
        spec = specs[prompt_source]
    except KeyError:
        raise CorpusPromptSourceError(
            f"prompt source {prompt_source!r} is not an installed environment"
        ) from None
    mode = getattr(spec, "interaction_mode", None)
    if mode == "episode":
        # An episode job's renderer IS its manifest's: both sides build it from
        # `renderer_id` and render through it, so there is no second authority
        # to disagree with.
        return spec
    if mode != "single_turn":
        raise CorpusPromptSourceError(
            f"prompt source {prompt_source!r} is {mode!r}, which is neither an "
            "episode environment nor a single-turn one"
        )
    if renderer_id is None:
        raise CorpusPromptSourceError(
            f"prompt source {prompt_source!r} is single-turn, so resolving it "
            "needs the job's renderer_id: without it the profile's rendering "
            "would go unchecked, which is the disagreement this refuses"
        )
    declared = _declared_prompt_template_id(prompt_source, profile)
    if renderer_id != declared:
        raise CorpusPromptSourceError(
            f"prompt source {prompt_source!r} renders through prompt template "
            f"{declared!r}, but the job declares renderer {renderer_id!r}; a "
            "job whose renderer is not the one its prompts are rendered with "
            "fails fidelity on every submission it is ever paid for"
        )
    return spec


def renderer_for_job(
    job: JobSpec,
    encode,
    *,
    environments: Mapping[str, Any] | None = None,
    profile: Any | None = None,
) -> Any:
    """The renderer this job's prompts are compared through.

    The mode decides it, not the caller: an episode job renders through the
    renderer its manifest names, while a single-turn job's rows arrive already
    rendered and the only faithful renderer is the one that changes nothing.
    """
    from reliquary.environment.agentic.renderers import renderer_for

    spec = resolve_prompt_source(
        job.prompt_source,
        environments=environments,
        renderer_id=job.renderer_id,
        profile=profile,
    )
    if getattr(spec, "interaction_mode", None) == "episode":
        return renderer_for(job.renderer_id, encode)
    return SingleTurnPromptRenderer()


def prompt_job_for_spec(
    job: JobSpec,
    *,
    environments: Mapping[str, Any] | None = None,
    profile: Any | None = None,
) -> EnvironmentPromptJob | SingleTurnPromptJob:
    """Resolve a job's prompt source to the rows a fidelity check needs.

    Builds the environment, which for a real source reads a dataset — so
    callers hold the result for the life of the job rather than per request.
    """
    spec = resolve_prompt_source(
        job.prompt_source,
        environments=environments,
        renderer_id=job.renderer_id,
        profile=profile,
    )
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
    if getattr(spec, "interaction_mode", None) == "episode":
        return EnvironmentPromptJob(job, environment)
    return SingleTurnPromptJob(job, environment)


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


def refuse_unsigned_corpus_submissions(request: CorpusSubmissionRequest) -> bool:
    """Refuse every submission, because nothing can yet verify one.

    ``protocol/signatures.py`` binds a GRPO window envelope -- window, merkle
    root, drand round -- and carries no binding over a corpus submission's job,
    cursor or tokens; no miner signs one either. Until that binding exists this
    is what the mount wires in, so the route is reachable and unusable rather
    than open. Not a placeholder to be quietly replaced by ``True``: replacing
    it means writing the binding.

    Raises rather than returning False so the miner is told this validator
    cannot verify, not that its signature was wrong.
    """
    del request
    raise CorpusSignatureUnavailable(
        "this binary carries no corpus signature binding"
    )


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
    job_id: str,
    store: CorpusJobStore,
    tokenizer: Tokenizer,
    renderer: Renderer,
    verify_signature,
    prompt_job_for=prompt_job_for_spec,
    max_write_attempts: int = DEFAULT_WRITE_ATTEMPTS,
    records=None,
    on_accepted=None,
    proof_chunk_tokens: int | None = None,
) -> APIRouter:
    """The corpus submission endpoint, over an already-bound job store.

    ``job_id`` is the job this validator is paid to serve — the one its task
    entry names. It is not a default: a router that would serve whatever job a
    submission names spends this task's bucket writes, and eventually this
    task's share, on work declared under somebody else's cap.
    """

    router = APIRouter()
    prompt_fidelity = PromptFidelity(renderer=renderer, prompt_job_for=prompt_job_for)
    # Also exposed, so the mount can reach the check without the handler.
    router.prompt_fidelity = prompt_fidelity

    async def _record_accepted(request: CorpusSubmissionRequest, served: str) -> None:
        # After the ledger write, never before: a record without its slot would
        # be paid for work the ledgers say never happened.
        if records is None:
            return
        from reliquary.protocol.signatures import corpus_submission_id

        submission_id = corpus_submission_id(request)
        record = {
            "schema": RECORD_SCHEMA,
            "submission_id": submission_id,
            "job_id": served,
            "hotkey": request.miner_hotkey,
            "cursor": request.cursor,
            "prompt_index": request.prompt_index,
            "rendered_prompt": request.rendered_prompt,
            "received_at": time.time(),
            "token_count": sum(len(c.tokens) for c in request.completions),
            "completions": [c.model_dump() for c in request.completions],
        }
        written = False
        for attempt in range(RECORD_WRITE_ATTEMPTS):
            try:
                written = await records.write_submission(served, submission_id, record)
                break
            except Exception:
                logger.warning(
                    "corpus record %s write attempt %d failed",
                    submission_id[:12],
                    attempt + 1,
                )
        else:
            # The slot is consumed and the tokens go unpaid: the one loss this
            # design accepts rather than paying for a record it cannot keep.
            logger.critical(
                "corpus record %s could not be written; its tokens go unpaid",
                submission_id[:12],
            )
            return
        if not written:
            # Create-only store: False means this id already exists, almost
            # always a resend of the same signed submission whose record is
            # already queued -- not a fault, and not a second announcement.
            logger.info("corpus record %s already recorded", submission_id[:12])
            return
        if on_accepted is not None:
            # The slot and the record are both already durable: a subscriber's
            # own bug must not turn that into a bare 500, which would send the
            # miner a retry that is then refused as a duplicate submission.
            try:
                on_accepted(submission_id)
            except Exception:
                logger.exception(
                    "corpus on_accepted callback failed for %s", submission_id[:12]
                )

    @router.get(JOB_PATH)
    async def corpus_job() -> dict:
        job = await _read_job_checked()
        if job is None:
            raise HTTPException(status_code=404, detail="corpus_job_unknown")
        return job.to_contract()

    @router.get(CURSOR_PATH)
    async def corpus_cursor(hotkey: str) -> dict:
        job = await _read_job_checked()
        if job is None:
            raise HTTPException(status_code=404, detail="corpus_job_unknown")
        snapshot, _ = await store.read_ledgers(job_id)
        _, cursors, _ = _rebuild_ledgers_checked(job, snapshot)
        return {"hotkey": hotkey, "cursor": cursors.expected(hotkey)}

    @router.post(SUBMIT_PATH, response_model=CorpusSubmissionResponse)
    async def submit_corpus(
        request: CorpusSubmissionRequest,
    ) -> CorpusSubmissionResponse:
        # First, and before the store is touched at all: another job's work is
        # not this validator's to admit, record or eventually pay for.
        if request.job_id != job_id:
            return _refuse(
                CorpusRejectReason.JOB_NOT_SERVED,
                {"job_id": request.job_id, "serves": job_id},
            )

        # Before anything reads or writes: an unsigned submission must not
        # reach the ledgers, or a spoofed hotkey consumes another miner's work.
        try:
            verified = verify_signature(request)
        except CorpusSignatureUnavailable:
            return _refuse(CorpusRejectReason.SIGNATURE_UNVERIFIABLE)
        if not verified:
            return _refuse(CorpusRejectReason.BAD_SIGNATURE)

        # `JobError` subclasses `ValueError`, so `_read_job_checked` catches it
        # first: a manifest in the bucket that no longer parses is an operator
        # fault, and left uncaught here it would disguise itself as an unknown
        # job instead of naming the corrupt one.
        job = await _read_job_checked()
        if job is None:
            return _refuse(
                CorpusRejectReason.JOB_UNKNOWN, {"job_id": job_id}
            )

        # The fidelity check indexes the prompt source, so a miner-controlled
        # index is bounded before it can raise on the operator's behalf. On a
        # `free` job this is the bound `admit` applies, reached earlier; on
        # `miner_walk` `admit` compares against `walk_index` instead, which is
        # a stricter rule inside this one.
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
                "corpus job %s has an unusable prompt source: %s", job_id, exc
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
            snapshot, etag = await store.read_ledgers(job_id)
            # The blast radius of a corrupt snapshot is every miner on this
            # job, and there is no circuit breaker: the object stays corrupt
            # until an operator repairs it, so `_rebuild_ledgers_checked`
            # names the refusal rather than leaving a bare 500.
            slots, cursors, seen = _rebuild_ledgers_checked(job, snapshot)
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
                proof_counts=[len(c.proofs) for c in request.completions],
                proof_chunk_tokens=proof_chunk_tokens,
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
                await store.write_ledgers(job_id, after, etag)
            except CorpusStoreConflict:
                continue
            if verdict.accepted:
                await _record_accepted(request, job_id)
            return _respond(verdict)

        logger.warning(
            "corpus ledgers for %s stayed contended over %d attempts (miner %s)",
            job_id,
            max_write_attempts,
            request.miner_hotkey[:12],
        )
        # Nothing was consumed, so the same work resubmits cleanly.
        raise HTTPException(status_code=503, detail="corpus_ledger_contention")

    async def _read_job_checked() -> JobSpec | None:
        """The manifest, or the same named 500 ``submit_corpus`` raises on one
        that no longer parses -- shared so the GET routes, which read the
        identical object, fail the same way an operator's corrupt manifest.
        """
        try:
            job, _ = await store.read_job(job_id)
        except JobError as exc:
            logger.error(
                "corpus job %s has an unreadable manifest: %s", job_id, exc
            )
            raise HTTPException(
                status_code=500, detail="corpus_job_manifest_corrupt"
            ) from exc
        except ValueError:
            # The id is not one the store could ever have written, so it names
            # no job; it must not become a 500 on a hostile request.
            return None
        return job

    def _rebuild_ledgers_checked(
        job: JobSpec, snapshot: Any
    ) -> tuple[SlotLedger, CursorLedger, set[str]]:
        """``rebuild_ledgers``, translated the same way ``submit_corpus``
        translates it -- shared with the cursor route, which reads the same
        snapshot and must not turn a corrupt one into a bare lookup error.
        """
        try:
            return rebuild_ledgers(job, snapshot)
        except LedgerSnapshotError as exc:
            logger.error("corpus ledgers for %s are unreadable: %s", job_id, exc)
            raise HTTPException(
                status_code=500, detail="corpus_ledger_corrupt"
            ) from exc

    return router


__all__ = [
    "CURSOR_PATH",
    "CorpusPromptSourceError",
    "CorpusSignatureUnavailable",
    "EnvironmentPromptJob",
    "JOB_PATH",
    "LEDGER_SCHEMA",
    "MAX_RESOLVED_PROMPT_SOURCES",
    "LedgerSnapshotError",
    "PromptFidelity",
    "RECORD_SCHEMA",
    "SUBMIT_PATH",
    "SingleTurnPromptJob",
    "SingleTurnPromptRenderer",
    "build_corpus_router",
    "ledger_snapshot",
    "prompt_job_for_spec",
    "rebuild_ledgers",
    "refuse_unsigned_corpus_submissions",
    "renderer_for_job",
    "resolve_prompt_source",
]
