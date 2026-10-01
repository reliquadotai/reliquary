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
import collections
import contextlib
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
import logging
import re
import time
from typing import Any, NamedTuple, Protocol

from fastapi import APIRouter, HTTPException

from reliquary.corpus.admission import (
    Verdict,
    admit,
    out_of_range_detail,
    skip,
    skip_refusal,
    skip_target,
)
from reliquary.corpus.checks import CheckResult, completion_digest
from reliquary.corpus.job import PROMPT_ORDER_MINER_WALK, JobError, JobSpec
from reliquary.corpus.slots import SlotLedger
from reliquary.corpus.walk import CursorLedger, job_walk_index
from reliquary.environment.agentic.types import EpisodeTask
from reliquary.environment.registry import ENVIRONMENT_SPECS
from reliquary.infrastructure.corpus_job_store import (
    CorpusSegmentCorrupt,
    CorpusStoreConflict,
)
from reliquary.protocol.corpus_submission import (
    CorpusRejectReason,
    CorpusSkipRequest,
    CorpusSkipResponse,
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
# Job-scoped reads, for a validator serving several jobs on one model.
JOBS_PATH = "/corpus/jobs"
JOB_SCOPED_PATH = "/corpus/jobs/{job_id}/job"
CURSOR_SCOPED_PATH = "/corpus/jobs/{job_id}/cursor/{hotkey}"
# Check before generating: where a hotkey's walk stands and whether that prompt
# still has a slot, and the signed step over it when it has none.
NEXT_PATH = "/corpus/next/{hotkey}"
NEXT_SCOPED_PATH = "/corpus/jobs/{job_id}/next/{hotkey}"
SKIP_PATH = "/corpus/skip"
SKIP_SCOPED_PATH = "/corpus/jobs/{job_id}/skip"

# The record's own schema tag, so a reader of the bucket can tell what shape
# to expect before it parses the rest of the document.
RECORD_SCHEMA = "reliquary/corpus-submission-record/v1"
# Mirrors `DEFAULT_WRITE_ATTEMPTS` below: a handful of rounds against a
# transient bucket fault, not a queue a miner's request should block behind.
RECORD_WRITE_ATTEMPTS = 3

# Validators at different versions share one ledger object, and this repo ships
# `:latest` behind Watchtower, so an older reader that IGNORED a field it did
# not know would DELETE it on its next read-modify-write — silent loss on the
# money object. So each schema is read under its own field set and anything
# else is refused. v2 moved the seen set into sealed segments under NEW field
# names, never `seen`: a pre-v2 binary refuses a v2 object by its field check
# (a 500, before any write), so rolling back to a pre-v2 image without
# `corpus ledgers downgrade` is a corpus outage, not corruption. For v3 and
# later the schema marker gives the named failure instead.
LEDGER_SCHEMA_V1 = "reliquary/corpus-ledgers/v1"
LEDGER_SCHEMA_V2 = "reliquary/corpus-ledgers/v2"
# The schema this binary writes.
LEDGER_SCHEMA = LEDGER_SCHEMA_V2
LEDGER_FIELDS = {
    LEDGER_SCHEMA_V1: frozenset({"schema", "slots", "cursors", "seen"}),
    LEDGER_SCHEMA_V2: frozenset(
        # `failed` only on an eval job's ledger, written only once a
        # submission failed: every other ledger is byte-identical.
        {"schema", "slots", "cursors", "seen_pending", "seen_segments", "failed"}
    ),
}

# Pending digests are sealed into a segment once this many accumulate, so the
# ledger rewritten on every submission stays small.
SEAL_THRESHOLD = 1024
# The most digests one segment holds (~270 KB), and how many segment GETs or
# PUTs run at once when many move together (startup, migration).
SEGMENT_MAX = 4096
SEGMENT_PARALLELISM = 8

# Contention is a two-writer race, not a queue, so a handful of rounds is
# plenty; past that the miner is better served by a retryable failure than by
# a request that never returns.
DEFAULT_WRITE_ATTEMPTS = 4

# How long a submission waits for its turn on the ledger. One hung PUT can hold
# the turn for botocore's full ~135 s; past this the miner gets the retryable 503.
LEDGER_LOCK_TIMEOUT_SECONDS = 30.0

# Each resolved source holds a built environment, and a validator serves only a
# handful of live jobs at once, so the cache is bounded rather than growing with
# every job this process has ever seen.
MAX_RESOLVED_PROMPT_SOURCES = 8

try:
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:  # pragma: no cover - botocore ships with the store
    BotoCoreError = ClientError = OSError

# A bucket that cannot be reached right now (throttled, 5xx, reset, timeout).
# Answered 503 so a miner retries instead of halting on a permanent 500; the
# corrupt manifest/ledger cases keep their named 500s.
_STORE_TRANSPORT_ERRORS = (ClientError, BotoCoreError, OSError, asyncio.TimeoutError)


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
    """The store calls the endpoint makes, bound to their bucket."""

    async def read_job(self, job_id: str) -> tuple[JobSpec | None, str | None]: ...

    async def read_ledgers(self, job_id: str) -> tuple[dict, str | None]: ...

    async def write_ledgers(
        self, job_id: str, snapshot: Mapping[str, Any], etag: str | None
    ) -> str | None: ...

    async def write_seen_segment(self, job_id: str, digests: Sequence[str]) -> str: ...

    async def read_seen_segment(self, job_id: str, segment_id: str) -> tuple[str, ...]: ...


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


# Renderer ids that wrap a single-turn row in the checkpoint's own chat
# template; the value is whether the template's thinking mode is on.
CHAT_TEMPLATE_RENDERERS = {
    "chat-template-v1": False,
    "chat-template-thinking-v1": True,
}


class ChatTemplatePromptRenderer:
    """The row as one user turn of the model's chat template, generation prompt
    appended. Miner and validator hold the same tokenizer (the job's revision),
    so both render the same text; ``tokenizer`` may be a zero-argument callable
    when the renderer is built before the tokenizer is loaded."""

    __slots__ = ("_tokenizer", "_thinking")

    def __init__(self, tokenizer: Any, *, thinking: bool) -> None:
        self._tokenizer = tokenizer
        self._thinking = thinking

    def initial_text(self, task: EpisodeTask) -> str:
        tokenizer = self._tokenizer() if callable(self._tokenizer) and not hasattr(
            self._tokenizer, "apply_chat_template"
        ) else self._tokenizer
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": task.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._thinking,
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
    """The SOURCE index, or a refusal naming the job's own bounds. The index is
    already a row of the source: a job starting at S owns rows [S, S+N)."""
    position = int(prompt_index)
    if not job.owns(position):
        owned = (
            f"owns source rows [{job.prompt_start}, {job.prompt_end})"
            if job.prompt_start
            else f"has {job.prompt_count} prompts"
        )
        raise CorpusPromptSourceError(
            f"job {job.job_id!r} {owned}; {position} is outside it"
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
    from reliquary.eval.prompt_source import EvalSetSpec, is_eval_source

    if is_eval_source(prompt_source):
        # An eval set's rows are already rendered by its catalog template; only
        # the model's own chat template wraps them.
        if renderer_id not in CHAT_TEMPLATE_RENDERERS:
            raise CorpusPromptSourceError(
                f"an eval-set prompt source renders through the model's chat template, "
                f"not {renderer_id!r}"
            )
        try:
            return EvalSetSpec(prompt_source)
        except ValueError as exc:
            raise CorpusPromptSourceError(str(exc)) from exc
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
    if renderer_id in CHAT_TEMPLATE_RENDERERS:
        # The model's own template wraps the row the contract rendered, so the
        # contract must still render it; the template itself is pinned by the
        # checkpoint revision, not by the contract.
        return spec
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
    tokenizer: Any | None = None,
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
    if job.renderer_id in CHAT_TEMPLATE_RENDERERS:
        if tokenizer is None:
            raise CorpusPromptSourceError(
                f"job {job.job_id!r} renders through the model's chat template, "
                "so building its renderer needs the checkpoint's tokenizer"
            )
        return ChatTemplatePromptRenderer(
            tokenizer, thinking=CHAT_TEMPLATE_RENDERERS[job.renderer_id]
        )
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
    if rows < job.prompt_end:
        raise CorpusPromptSourceError(
            f"job {job.job_id!r} claims {job.prompt_count} prompts"
            + (f" from row {job.prompt_start} (through row {job.prompt_end})"
               if job.prompt_start else "")
            + f" but {job.prompt_source!r} has {rows}"
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


_SEGMENT_ID_RE = re.compile(r"^[0-9a-f]{64}$")


class SegmentRef(NamedTuple):
    """A sealed segment as a ledger names it: its content hash and size."""

    id: str
    count: int


@dataclass
class LedgerState:
    """One ledger version, parsed. ``pending`` is this state's own copy; the
    segments' contents live in a ``SeenIndex``."""

    schema: str
    slots: SlotLedger
    cursors: CursorLedger
    pending: set[str]
    segments: tuple[SegmentRef, ...]


def _digest_list(job: JobSpec, field: str, value: Any, *, unique: bool) -> list[str]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(digest, str) for digest in value
    ):
        raise LedgerSnapshotError(
            f"job {job.job_id!r} has a {field} list that is not a list of digests"
        )
    if unique and len(set(value)) != len(value):
        raise LedgerSnapshotError(f"job {job.job_id!r} repeats a digest in {field}")
    return list(value)


def _segment_refs(job: JobSpec, value: Any) -> tuple[SegmentRef, ...]:
    if not isinstance(value, (list, tuple)):
        raise LedgerSnapshotError(f"job {job.job_id!r} has seen_segments that is not a list")
    refs = []
    for item in value:
        count = item.get("count") if isinstance(item, Mapping) else None
        if (
            not isinstance(item, Mapping)
            or set(item) != {"id", "count"}
            or not isinstance(item["id"], str)
            or not _SEGMENT_ID_RE.match(item["id"])
            or type(count) is not int
            or count < 1
        ):
            raise LedgerSnapshotError(
                f"job {job.job_id!r} has a seen segment reference that is not one: {item!r}"
            )
        refs.append(SegmentRef(item["id"], count))
    if len({ref.id for ref in refs}) != len(refs):
        # The same segment twice would count its digests twice (I3).
        raise LedgerSnapshotError(f"job {job.job_id!r} names a seen segment twice")
    return tuple(refs)


def rebuild_ledgers(job: JobSpec, snapshot: Any) -> LedgerState:
    """The stored snapshot as live ledgers, or a named refusal.

    Rebuilt fresh on every attempt, because ``admit()`` mutates what it is
    given: a write that loses its compare-and-swap must not leave a consumed
    slot behind in the copy the retry then admits against. A v1 object reads
    as all of its seen set pending and no segments.
    """
    if not isinstance(snapshot, Mapping):
        raise LedgerSnapshotError(
            f"job {job.job_id!r} has a ledger object that is not an object"
        )
    # Absent on the empty read that precedes a job's first write, and on any
    # object written before the marker existed.
    schema = snapshot.get("schema", LEDGER_SCHEMA_V1)
    fields = LEDGER_FIELDS.get(schema) if isinstance(schema, str) else None
    if fields is None:
        raise LedgerSnapshotError(
            f"job {job.job_id!r} has ledgers under schema {schema!r}, "
            f"not one of {sorted(LEDGER_FIELDS)}"
        )
    unknown = sorted(set(snapshot) - fields)
    if unknown:
        raise LedgerSnapshotError(
            f"job {job.job_id!r} has ledger fields this binary cannot read "
            f"under {schema!r}: {unknown}"
        )
    try:
        slots = SlotLedger.from_snapshot(
            job.prompt_count,
            job.slots_per_prompt,
            snapshot.get("slots") or {},
            prompt_start=job.prompt_start,
            failed=snapshot.get("failed") or {},
        )
        cursors = CursorLedger.from_snapshot(snapshot.get("cursors") or {})
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise LedgerSnapshotError(
            f"job {job.job_id!r} has an unusable ledger snapshot: {exc}"
        ) from exc
    if schema == LEDGER_SCHEMA_V1:
        seen = _digest_list(job, "seen-digest", snapshot.get("seen") or [], unique=False)
        return LedgerState(schema, slots, cursors, set(seen), ())
    for field in ("seen_pending", "seen_segments"):
        if field not in snapshot:
            # Missing would silently read as a smaller seen set.
            raise LedgerSnapshotError(f"job {job.job_id!r} has v2 ledgers without {field}")
    pending = _digest_list(job, "seen_pending", snapshot["seen_pending"], unique=True)
    return LedgerState(
        schema, slots, cursors, set(pending), _segment_refs(job, snapshot["seen_segments"])
    )


def ledger_snapshot(
    slots: SlotLedger,
    cursors: CursorLedger,
    pending: Iterable[str],
    segments: Sequence[SegmentRef] = (),
) -> dict[str, Any]:
    """The JSON-native v2 form the store persists. Prompt indices are
    stringified here rather than by the encoder, so a snapshot compares equal
    to the one that comes back out of the bucket."""
    snapshot = {
        "schema": LEDGER_SCHEMA_V2,
        "slots": {str(index): count for index, count in slots.snapshot().items()},
        "cursors": cursors.snapshot(),
        "seen_pending": sorted(pending),
        "seen_segments": [{"id": ref.id, "count": ref.count} for ref in segments],
    }
    failed = slots.failed_snapshot()
    if failed:
        snapshot["failed"] = {str(index): count for index, count in failed.items()}
    return snapshot


async def record_prompt_failure(store: Any, job: JobSpec, prompt_index: int,
                                submission_id: str, *,
                                attempts: int = DEFAULT_WRITE_ATTEMPTS) -> bool | None:
    """An eval job's submission failed its audit: record it in the ledger
    (idempotent per submission), which reopens the prompt's slot until its
    attempts run out. True: reopened; False: exhausted; None: already recorded.
    Compare-and-swap against the route's own writes, which retry on conflict."""
    for _ in range(attempts):
        snapshot, etag = await store.read_ledgers(job.job_id)
        state = await asyncio.to_thread(rebuild_ledgers, job, snapshot)
        if snapshot and state.schema != LEDGER_SCHEMA_V2:
            raise LedgerSnapshotError(f"job {job.job_id!r} ledgers are not v2")
        outcome = state.slots.record_failure(prompt_index, submission_id)
        if outcome is None:
            return None
        after = ledger_snapshot(state.slots, state.cursors, state.pending, state.segments)
        try:
            await store.write_ledgers(job.job_id, after, etag)
        except CorpusStoreConflict:
            continue
        return outcome
    raise CorpusStoreConflict(f"job {job.job_id!r}: the failure of {submission_id[:12]} "
                              "kept losing its ledger race")


def seal_chunks(digests: Iterable[str], segment_max: int = SEGMENT_MAX) -> list[list[str]]:
    """Sorted, unique chunks of at most ``segment_max`` digests, each one a
    segment body."""
    ordered = sorted(set(digests))
    return [ordered[i:i + segment_max] for i in range(0, len(ordered), segment_max)]


async def _all_or_cancel(awaitables: Iterable[Awaitable[Any]]) -> list[Any]:
    """``gather``, except that the first failure cancels and awaits the rest,
    so no segment call outlives the request and no exception goes unretrieved."""
    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class SeenIndex:
    """The union of the segments one ledger version names, held in memory.

    Segments never change, so their contents are cached forever; the digest
    map is built only from the references a ledger version actually carries,
    so an orphan segment (sealed, then its ledger write lost) never counts.
    Callers serialize ``ensure`` against the reads that follow it.
    """

    def __init__(
        self, store: Any, job_id: str, *, parallelism: int = SEGMENT_PARALLELISM
    ) -> None:
        self._store = store
        self._job_id = job_id
        self._parallelism = parallelism
        self._contents: dict[str, tuple[str, ...]] = {}
        self._refs: tuple[SegmentRef, ...] = ()
        self._owner: dict[str, str] = {}

    def __contains__(self, digest: object) -> bool:
        return digest in self._owner

    def __len__(self) -> int:
        return len(self._owner)

    def __iter__(self):
        return iter(self._owner)

    @property
    def refs(self) -> tuple[SegmentRef, ...]:
        return self._refs

    def remember(self, segment_id: str, digests: Sequence[str]) -> None:
        """Contents this process just sealed, so a later ``ensure`` needs no
        GET. Not counted until a ledger version names the segment."""
        self._contents.setdefault(segment_id, tuple(digests))

    def check_pending(self, pending: Iterable[str]) -> None:
        """I3 between pending and the segments: a digest in both is corrupt."""
        for digest in pending:
            if digest in self._owner:
                raise LedgerSnapshotError(
                    f"job {self._job_id!r} has digest {digest[:12]} both pending "
                    f"and in segment {self._owner[digest][:12]}"
                )

    async def ensure(self, refs: Sequence[SegmentRef]) -> None:
        """Make the index hold exactly ``refs``. Raises ``LedgerSnapshotError``
        on a missing, altered, miscounted or overlapping segment (I2, I3),
        leaving the index as it was; transport errors propagate."""
        refs = tuple(refs)
        if refs == self._refs:
            return
        extends = refs[: len(self._refs)] == self._refs
        todo = refs[len(self._refs):] if extends else refs
        await self._load([ref.id for ref in todo if ref.id not in self._contents])
        base = self._owner if extends else {}
        added = await asyncio.to_thread(self._merge, base, todo)
        if extends:
            self._owner.update(added)
        else:
            self._owner = added
        self._refs = refs

    async def _load(self, ids: list[str]) -> None:
        if not ids:
            return
        gate = asyncio.Semaphore(self._parallelism)

        async def one(segment_id: str) -> None:
            async with gate:
                try:
                    digests = await self._store.read_seen_segment(self._job_id, segment_id)
                except CorpusSegmentCorrupt as exc:
                    raise LedgerSnapshotError(str(exc)) from exc
            self._contents[segment_id] = tuple(digests)

        await _all_or_cancel(one(segment_id) for segment_id in ids)

    def _merge(
        self, base: Mapping[str, str], refs: Sequence[SegmentRef]
    ) -> dict[str, str]:
        added: dict[str, str] = {}
        for ref in refs:
            digests = self._contents[ref.id]
            if len(digests) != ref.count:
                raise LedgerSnapshotError(
                    f"job {self._job_id!r} seen segment {ref.id[:12]} holds "
                    f"{len(digests)} digests, its reference says {ref.count}"
                )
            for digest in digests:
                if digest in base or digest in added:
                    raise LedgerSnapshotError(
                        f"job {self._job_id!r} has digest {digest[:12]} in two seen segments"
                    )
                added[digest] = ref.id
        return added


class SeenView(AbstractSet):
    """Pending plus the index, as the one set ``admit`` reads. Nothing is
    copied: ``check_duplicates`` only asks membership."""

    __slots__ = ("_index", "_pending")

    def __init__(self, index: Any, pending: AbstractSet[str]) -> None:
        self._index = index
        self._pending = pending

    def __contains__(self, digest: object) -> bool:
        return digest in self._pending or digest in self._index

    def __len__(self) -> int:
        return len(self._pending) + len(self._index)

    def __iter__(self):
        yield from self._pending
        yield from self._index


async def _write_segments(
    store: Any, job_id: str, chunks: Sequence[Sequence[str]], parallelism: int
) -> list[SegmentRef]:
    gate = asyncio.Semaphore(parallelism)

    async def one(chunk: Sequence[str]) -> SegmentRef:
        async with gate:
            return SegmentRef(await store.write_seen_segment(job_id, chunk), len(chunk))

    return await _all_or_cancel(one(chunk) for chunk in chunks)


# Migration and downgrade race at most one other writer; past this many lost
# compare-and-swaps something keeps rewriting the ledger and a human should look.
LEDGER_REWRITE_ATTEMPTS = 8


async def ensure_ledgers_v2(
    store: Any,
    job: JobSpec,
    *,
    segment_max: int = SEGMENT_MAX,
    parallelism: int = SEGMENT_PARALLELISM,
) -> str:
    """Rewrite a v1 ledger as v2: back it up (create-only, first one kept),
    seal its whole seen set, then swap the ledger under its ETag. Returns
    "absent", "v2" (nothing to do) or "migrated". A conflict re-reads and
    starts again; a chunk whose content is unchanged gets the same name, so
    its segment is not written twice."""
    for _ in range(LEDGER_REWRITE_ATTEMPTS):
        snapshot, etag = await store.read_ledgers(job.job_id)
        if etag is None:
            return "absent"
        state = await asyncio.to_thread(rebuild_ledgers, job, snapshot)
        if state.schema == LEDGER_SCHEMA_V2:
            return "v2"
        await store.write_ledgers_backup(job.job_id, snapshot)
        chunks = await asyncio.to_thread(seal_chunks, state.pending, segment_max)
        refs = await _write_segments(store, job.job_id, chunks, parallelism)
        after = ledger_snapshot(state.slots, state.cursors, (), refs)
        try:
            await store.write_ledgers(job.job_id, after, etag)
        except CorpusStoreConflict:
            continue
        logger.info(
            "corpus ledgers for %s migrated to v2: %d digests in %d segments",
            job.job_id, len(state.pending), len(refs),
        )
        return "migrated"
    raise CorpusStoreConflict(f"ledgers of {job.job_id!r} kept changing during migration")


# How long startup waits on the migration and the segment load before serving
# anyway; the route then migrates inline and loads what the index lacks.
STARTUP_LEDGER_TIMEOUT_SECONDS = 120.0


async def migrate_ledgers_at_startup(
    store: Any, job: JobSpec, *, timeout: float = STARTUP_LEDGER_TIMEOUT_SECONDS
) -> SeenIndex:
    """``ensure_ledgers_v2`` before a route serves, then the segments the
    ledger names loaded into the index handed to that route, so its first
    submission does not load them under the ledger lock.

    Best effort: the route migrates a v1 ledger inline on its first write and
    loads whatever the index lacks, and a corrupt or unreachable ledger is
    answered there by name, so a failure here must not keep the validator down.
    """
    index = SeenIndex(store, job.job_id)

    async def prepare() -> str:
        outcome = await ensure_ledgers_v2(store, job)
        snapshot, _ = await store.read_ledgers(job.job_id)
        state = await asyncio.to_thread(rebuild_ledgers, job, snapshot)
        await index.ensure(state.segments)
        return outcome

    try:
        outcome = await asyncio.wait_for(prepare(), timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "corpus ledgers for %s: startup preparation timed out after %.0f s; "
            "the route migrates and loads on its first submission",
            job.job_id, timeout,
        )
        return index
    except Exception:
        logger.exception("corpus ledgers for %s could not be prepared at startup", job.job_id)
        return index
    logger.info(
        "corpus ledgers for %s at startup: %s, %d sealed digests loaded",
        job.job_id, outcome, len(index),
    )
    return index


async def _loaded(store: Any, job: JobSpec) -> tuple[dict, str | None, LedgerState, SeenIndex]:
    """The ledger, parsed, with every segment it names loaded and checked
    (I2, I3). Raises ``LedgerSnapshotError`` on any violation."""
    snapshot, etag = await store.read_ledgers(job.job_id)
    state = await asyncio.to_thread(rebuild_ledgers, job, snapshot)
    index = SeenIndex(store, job.job_id)
    await index.ensure(state.segments)
    index.check_pending(state.pending)
    return snapshot, etag, state, index


async def downgrade_ledgers_v1(store: Any, job: JobSpec) -> str:
    """Rewrite a v2 ledger as v1 with its full seen set, under its ETag, so a
    pre-v2 image can serve the job again. Run with every validator stopped.
    Returns "absent", "v1" (nothing to do) or "downgraded"; segments stay in
    the bucket for a later re-migration."""
    for _ in range(LEDGER_REWRITE_ATTEMPTS):
        _, etag, state, index = await _loaded(store, job)
        if etag is None:
            return "absent"
        if state.schema == LEDGER_SCHEMA_V1:
            return "v1"
        base = ledger_snapshot(state.slots, state.cursors, ())
        v1 = {
            "schema": LEDGER_SCHEMA_V1,
            "slots": base["slots"],
            "cursors": base["cursors"],
            "seen": await asyncio.to_thread(sorted, set(index) | state.pending),
        }
        try:
            await store.write_ledgers(job.job_id, v1, etag)
        except CorpusStoreConflict:
            continue
        return "downgraded"
    raise CorpusStoreConflict(f"ledgers of {job.job_id!r} kept changing during downgrade")


async def verify_ledgers(store: Any, job: JobSpec) -> dict[str, Any]:
    """Sizes of the stored ledger, after checking every segment it names
    (I2, I3). ``problems`` names an I1 violation by count: each accepted
    submission consumed one slot and added ``sampling.n`` digests, so the
    seen set must hold exactly ``filled * n``."""
    from reliquary.infrastructure.corpus_job_store import _encode

    snapshot, etag, state, index = await _loaded(store, job)
    seen = len(index) + len(state.pending)
    expected = state.slots.filled * job.sampling.n
    problems = []
    if seen != expected:
        problems.append(
            f"seen holds {seen} digests but {state.slots.filled} filled slots "
            f"imply {expected}"
        )
    reader = getattr(store, "read_ledgers_backup", None)
    backup = (await reader(job.job_id)) is not None if reader is not None else None
    return {
        "job_id": job.job_id,
        "schema": state.schema if etag is not None else None,
        "etag": etag,
        "ledger_bytes": len(await asyncio.to_thread(_encode, dict(snapshot))),
        "filled": state.slots.filled,
        "prompts_touched": len(state.slots.snapshot()),
        "hotkeys": len(state.cursors.snapshot()),
        "pending": len(state.pending),
        "segments": len(state.segments),
        "seen": seen,
        "expected_seen": expected,
        "backup": backup,
        "problems": problems,
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


def _refuse_skip(
    reason: CorpusRejectReason, detail: Mapping[str, Any] | None = None
) -> CorpusSkipResponse:
    return CorpusSkipResponse(reason=reason, skipped=False, detail=dict(detail or {}))


def _skip_refused(verdict: Verdict) -> CorpusSkipResponse:
    return CorpusSkipResponse(
        reason=CorpusRejectReason(verdict.reason),
        skipped=False,
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
    verify_skip_signature=None,
    prompt_job_for=prompt_job_for_spec,
    max_write_attempts: int = DEFAULT_WRITE_ATTEMPTS,
    records=None,
    on_accepted=None,
    proof_chunk_tokens: int | None = None,
    vocab_size: int | None = None,
    is_banned: Callable[[str], Awaitable[bool]] | None = None,
    registration: Callable[[str], Awaitable[str | None]] | None = None,
    ledger_lock_timeout: float = LEDGER_LOCK_TIMEOUT_SECONDS,
    seal_threshold: int = SEAL_THRESHOLD,
    segment_max: int = SEGMENT_MAX,
    seen_index: SeenIndex | None = None,
) -> APIRouter:
    """The corpus submission endpoint, over an already-bound job store.

    ``job_id`` is the job this validator is paid to serve — the one its task
    entry names. It is not a default: a router that would serve whatever job a
    submission names spends this task's bucket writes, and eventually this
    task's share, on work declared under somebody else's cap.

    ``is_banned`` is optional: a caller with no ban state to consult (a test,
    or a validator not yet wired to one) leaves every hotkey admitted.
    ``registration`` likewise: it answers None for a hotkey registered on the
    subnet, else ``corpus_registration.NOT_REGISTERED`` or ``UNAVAILABLE``.
    ``verify_skip_signature`` checks a skip's own binding; without one every
    skip is refused ``signature_unverifiable`` and miners generate as before.
    """

    router = APIRouter()
    prompt_fidelity = PromptFidelity(renderer=renderer, prompt_job_for=prompt_job_for)
    # This process's submissions take turns on the ledger: interleaved, each
    # would read the same ETag and all but one would lose the compare-and-swap.
    ledger_lock = asyncio.Lock()
    # The sealed part of the seen set, touched only under `ledger_lock`; the
    # startup path hands in one it has already loaded.
    if seen_index is None:
        seen_index = SeenIndex(store, job_id)
    # One rebuilt ledger state, keyed by the ETag it was read under (`_read_state`).
    read_cache: dict[str, Any] = {}

    async def _from_store(call, what: str):
        try:
            return await call
        except _STORE_TRANSPORT_ERRORS as exc:
            logger.warning("corpus store %s for job %s failed: %r", what, job_id, exc)
            raise HTTPException(status_code=503, detail="corpus_store_unavailable") from exc
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

    async def seal(chunks: list[list[str]]) -> list[SegmentRef]:
        """Seal pending digests into segments; shared by submit and skip."""
        # Written before any ledger names them (I2); a lost ledger write
        # leaves them orphaned and uncounted (I4); resealing the same
        # pending set later lands on the same names.
        gate = asyncio.Semaphore(SEGMENT_PARALLELISM)

        async def one(chunk: list[str]) -> SegmentRef:
            async with gate:
                try:
                    segment_id = await _from_store(
                        store.write_seen_segment(job_id, chunk), "seen segment write"
                    )
                except CorpusStoreConflict as exc:
                    # A create that stayed contended with the key still
                    # absent: nothing is named yet, so the miner retries.
                    logger.warning("corpus seen segment for %s: %s", job_id, exc)
                    raise HTTPException(
                        status_code=503, detail="corpus_store_unavailable"
                    ) from exc
                except CorpusSegmentCorrupt as exc:
                    raise _ledger_corrupt(LedgerSnapshotError(str(exc))) from exc
            seen_index.remember(segment_id, chunk)
            return SegmentRef(segment_id, len(chunk))

        return await _all_or_cancel(one(chunk) for chunk in chunks)

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
        state = await _read_state(job)
        return {"hotkey": hotkey, "cursor": state.cursors.expected(hotkey)}

    @router.post(SUBMIT_PATH, response_model=CorpusSubmissionResponse)
    async def submit_corpus(
        request: CorpusSubmissionRequest,
    ) -> CorpusSubmissionResponse:
        # First, and before the store is touched at all: another job's work is
        # not this validator's to admit, record or eventually pay for.
        if request.job_id != job_id:
            return _refuse(
                CorpusRejectReason.JOB_NOT_SERVED,
                {"job_id": request.job_id, "serves": [job_id]},
            )

        # Before anything reads or writes: an unsigned submission must not
        # reach the ledgers, or a spoofed hotkey consumes another miner's work.
        try:
            verified = verify_signature(request)
        except CorpusSignatureUnavailable:
            return _refuse(CorpusRejectReason.SIGNATURE_UNVERIFIABLE)
        if not verified:
            return _refuse(CorpusRejectReason.BAD_SIGNATURE)

        gated = await _hotkey_refusal(request.miner_hotkey)
        if gated is not None:
            return _refuse(gated)

        # `JobError` subclasses `ValueError`, so `_read_job_checked` catches it
        # first: a manifest in the bucket that no longer parses is an operator
        # fault, and left uncaught here it would disguise itself as an unknown
        # job instead of naming the corrupt one.
        started = time.perf_counter()
        job = await _read_job_checked()
        if job is None:
            return _refuse(
                CorpusRejectReason.JOB_UNKNOWN, {"job_id": job_id}
            )
        timing = {"job_read": time.perf_counter() - started}

        # The fidelity check indexes the prompt source, so a miner-controlled
        # index is bounded before it can raise on the operator's behalf. On a
        # `free` job this is the bound `admit` applies, reached earlier; on
        # `miner_walk` `admit` compares against `job_walk_index` instead, which
        # is a stricter rule inside this one.
        # The index is a SOURCE index: a job starting at S owns [S, S+N).
        if not job.owns(request.prompt_index):
            return _refuse(
                CorpusRejectReason.PROMPT_MISMATCH,
                out_of_range_detail(job, request.prompt_index),
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

        if vocab_size is not None:
            # Before the text check, which cannot see these (decode drops
            # unknown ids), and before any write: the auditor's prefill would
            # otherwise be the first thing to trip on them.
            for index, tokens in enumerate(arrays):
                if max(tokens) >= vocab_size:
                    return _refuse(
                        CorpusRejectReason.TOKEN_OUT_OF_VOCAB,
                        {"completion": index, "vocab_size": vocab_size},
                    )

        for completion in request.completions:
            text = check_text_matches_tokens(
                completion.tokens,
                completion.text,
                tokenizer=tokenizer,
                eos_token_id=job.eos_token_id,
            )
            if not text.ok:
                return _refuse(CorpusRejectReason(text.reason), text.detail)

        timing["checks"] = time.perf_counter() - started - timing["job_read"]

        def admit_against(
            state: LedgerState,
        ) -> tuple[Verdict, tuple[dict[str, Any], list[list[str]]] | None]:
            # Pure and CPU-bound, so it runs in a thread; the lock keeps two of
            # these from ever working on one state, and `seen_index` already
            # holds exactly the segments `state` names.
            seen_index.check_pending(state.pending)
            slots, cursors, pending = state.slots, state.cursors, state.pending
            before = (slots.snapshot(), cursors.snapshot())

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
                seen=SeenView(seen_index, pending),
                proof_counts=[len(c.proofs) for c in request.completions],
                proof_chunk_tokens=proof_chunk_tokens,
            )
            if verdict.accepted:
                # `admit` reads `seen`, it does not grow it: recording what was
                # paid for is the caller's half of the duplicate check.
                pending.update(digests)
            elif (slots.snapshot(), cursors.snapshot()) == before:
                return verdict, None
            chunks: list[list[str]] = []
            if len(pending) >= seal_threshold:
                chunks = seal_chunks(pending, segment_max)
                pending = set()
            # Everything but the segment list, which the seal completes.
            return verdict, (ledger_snapshot(slots, cursors, pending, state.segments), chunks)

        for key in ("ledger_read", "segments", "admit", "ledger_write"):
            timing[key] = 0.0
        waited = time.perf_counter()
        written: Verdict | None = None
        attempts = 0
        try:
            await asyncio.wait_for(ledger_lock.acquire(), ledger_lock_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "corpus ledger turn for %s not granted within %.0f s (miner %s)",
                job_id, ledger_lock_timeout, request.miner_hotkey[:12],
            )
            # Nothing was consumed, so the same work resubmits cleanly.
            raise HTTPException(status_code=503, detail="corpus_ledger_contention") from None
        try:
            timing["lock_wait"] = time.perf_counter() - waited
            for attempts in range(1, max_write_attempts + 1):
                mark = time.perf_counter()
                snapshot, etag = await _from_store(store.read_ledgers(job_id), "ledger read")
                state = await asyncio.to_thread(_rebuild_ledgers_checked, job, snapshot)
                timing["ledger_read"] += time.perf_counter() - mark

                mark = time.perf_counter()
                await _ensure_seen(state.segments)
                timing["segments"] += time.perf_counter() - mark

                mark = time.perf_counter()
                try:
                    verdict, planned = await asyncio.to_thread(admit_against, state)
                except LedgerSnapshotError as exc:
                    raise _ledger_corrupt(exc) from exc
                timing["admit"] += time.perf_counter() - mark
                if planned is None:
                    # A refusal that moved nothing costs no write, so a miner
                    # spraying junk cannot bill us a bucket write per attempt.
                    return _respond(verdict)
                after, chunks = planned
                mark = time.perf_counter()
                if chunks:
                    sealed = await seal(chunks)
                    after["seen_segments"] = [
                        *after["seen_segments"],
                        *({"id": ref.id, "count": ref.count} for ref in sealed),
                    ]
                timing["segments"] += time.perf_counter() - mark
                mark = time.perf_counter()
                try:
                    await _from_store(store.write_ledgers(job_id, after, etag), "ledger write")
                except CorpusStoreConflict:
                    continue
                finally:
                    timing["ledger_write"] += time.perf_counter() - mark
                written = verdict
                break
        finally:
            ledger_lock.release()

        if written is not None:
            # Outside the lock: the record is create-only and keyed by its own
            # id, so it needs no turn on the ledger.
            mark = time.perf_counter()
            if written.accepted:
                await _record_accepted(request, job_id)
                timing["record_write"] = time.perf_counter() - mark
                timing["total"] = time.perf_counter() - started
                logger.info(
                    "corpus submission timing %s: %s attempts=%d",
                    request.miner_hotkey[:12],
                    " ".join(f"{k}={v:.3f}" for k, v in timing.items()),
                    attempts,
                )
            return _respond(written)

        logger.warning(
            "corpus ledgers for %s stayed contended over %d attempts (miner %s)",
            job_id,
            max_write_attempts,
            request.miner_hotkey[:12],
        )
        # Nothing was consumed, so the same work resubmits cleanly.
        raise HTTPException(status_code=503, detail="corpus_ledger_contention")

    async def _hotkey_refusal(hotkey: str) -> CorpusRejectReason | None:
        """The registration and ban gates, shared by submit and skip; called
        only once a signature has verified."""
        # An unregistered hotkey is never paid; refuse it before it costs a
        # store read or an audit. Unknown registrations are retried, not refused.
        if registration is not None:
            from reliquary.validator.corpus_registration import NOT_REGISTERED

            reason = await registration(hotkey)
            if reason == NOT_REGISTERED:
                return CorpusRejectReason.HOTKEY_NOT_REGISTERED
            if reason is not None:
                raise HTTPException(status_code=503, detail="corpus_registration_unavailable")

        # Right after the signature check and before anything is read or
        # written: an unsigned request must not be able to probe ban status.
        if is_banned is not None:
            try:
                banned = await is_banned(hotkey)
            except Exception as exc:
                # Same response `_from_store` gives a bucket transport error:
                # the miners document is unreachable, not that this hotkey
                # was cleared to submit.
                logger.warning("corpus ban check for %s failed: %r", hotkey[:12], exc)
                raise HTTPException(
                    status_code=503, detail="corpus_store_unavailable"
                ) from exc
            if banned:
                return CorpusRejectReason.MINER_BANNED
        return None

    @router.get(NEXT_PATH)
    async def corpus_next(hotkey: str) -> dict:
        """Where this hotkey's walk stands, how many slots that prompt has
        left, and ``skip_to``: the first later cursor whose prompt has a free
        slot (at most ``MAX_SKIP_STEPS`` on), so a miner crosses a run of full
        prompts in one skip. A read, exposed like the cursor: no signature, no
        write."""
        job = await _read_job_checked()
        if job is None:
            raise HTTPException(status_code=404, detail="corpus_job_unknown")
        if job.prompt_order != PROMPT_ORDER_MINER_WALK:
            # A free job has no walk: the miner chooses its prompts itself.
            raise HTTPException(status_code=409, detail="corpus_job_not_miner_walk")
        state = await _read_state(job)
        cursor = state.cursors.expected(hotkey)
        prompt_index = job_walk_index(job, hotkey, cursor)
        return {
            "cursor": cursor,
            "prompt_index": prompt_index,
            "slots_remaining": state.slots.remaining(prompt_index),
            "skip_to": skip_target(job, hotkey, cursor, state.slots),
        }

    @router.post(SKIP_PATH, response_model=CorpusSkipResponse)
    async def skip_corpus(request: CorpusSkipRequest) -> CorpusSkipResponse:
        """Step this hotkey's cursor from ``cursor`` to ``to_cursor`` over walk
        positions that are ALL full, exactly as that many ``prompt_full``
        refusals would (``admission.skip``), in one ledger write under the same
        turn and compare-and-swap as submit. Nothing is paid or recorded and
        no digest is seen."""
        if request.job_id != job_id:
            return _refuse_skip(
                CorpusRejectReason.JOB_NOT_SERVED,
                {"job_id": request.job_id, "serves": [job_id]},
            )
        if verify_skip_signature is None:
            return _refuse_skip(CorpusRejectReason.SIGNATURE_UNVERIFIABLE)
        try:
            verified = verify_skip_signature(request)
        except CorpusSignatureUnavailable:
            return _refuse_skip(CorpusRejectReason.SIGNATURE_UNVERIFIABLE)
        if not verified:
            return _refuse_skip(CorpusRejectReason.BAD_SIGNATURE)
        gated = await _hotkey_refusal(request.miner_hotkey)
        if gated is not None:
            return _refuse_skip(gated)
        job = await _read_job_checked()
        if job is None:
            return _refuse_skip(CorpusRejectReason.JOB_UNKNOWN, {"job_id": job_id})

        # Decided first against the shared read state: a refusal writes
        # nothing, so it need not queue behind submissions for the ledger. An
        # accept is only a candidate; it is decided again under the lock.
        cached = await _read_state(job)
        refused = skip_refusal(
            job,
            hotkey=request.miner_hotkey,
            cursor=request.cursor,
            prompt_index=request.prompt_index,
            to_cursor=request.to_cursor,
            slots=cached.slots,
            cursors=cached.cursors,
        )
        if refused is not None:
            return _skip_refused(refused)

        try:
            await asyncio.wait_for(ledger_lock.acquire(), ledger_lock_timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="corpus_ledger_contention") from None
        try:
            for _ in range(max_write_attempts):
                snapshot, etag = await _from_store(store.read_ledgers(job_id), "ledger read")
                state = await asyncio.to_thread(_rebuild_ledgers_checked, job, snapshot)
                # The integrity step submit runs before it decides (I2, I3).
                await _ensure_seen(state.segments)
                try:
                    await asyncio.to_thread(seen_index.check_pending, state.pending)
                except LedgerSnapshotError as exc:
                    raise _ledger_corrupt(exc) from exc
                verdict = skip(
                    job,
                    hotkey=request.miner_hotkey,
                    cursor=request.cursor,
                    prompt_index=request.prompt_index,
                    to_cursor=request.to_cursor,
                    slots=state.slots,
                    cursors=state.cursors,
                )
                if not verdict.accepted:
                    # Moved nothing, so it costs no write.
                    return _skip_refused(verdict)
                # A skip adds no digest, but a ledger it rewrites may already
                # hold a pending set past the threshold (a v1 object not yet
                # migrated): sealed exactly as submit seals it.
                pending, chunks = state.pending, []
                if len(pending) >= seal_threshold:
                    chunks = await asyncio.to_thread(seal_chunks, pending, segment_max)
                    pending = set()
                after = ledger_snapshot(state.slots, state.cursors, pending, state.segments)
                if chunks:
                    sealed = await seal(chunks)
                    after["seen_segments"] = [
                        *after["seen_segments"],
                        *({"id": ref.id, "count": ref.count} for ref in sealed),
                    ]
                try:
                    await _from_store(store.write_ledgers(job_id, after, etag), "ledger write")
                except CorpusStoreConflict:
                    continue
                moved = state.cursors.expected(request.miner_hotkey)
                logger.debug(
                    "corpus skip %s: cursor %d -> %d over full prompts",
                    request.miner_hotkey[:12], request.cursor, moved,
                )
                return CorpusSkipResponse(
                    reason=CorpusRejectReason.ACCEPTED,
                    skipped=True,
                    cursor=moved,
                    slots_remaining=0,
                )
        finally:
            ledger_lock.release()
        raise HTTPException(status_code=503, detail="corpus_ledger_contention")

    async def _read_state(job: JobSpec) -> LedgerState:
        """The ledgers for READING: rebuilt once per ETag and shared by the
        cursor and next routes (and the skip's pre-check), so a miner polling
        does not parse the ledger again. Callers must not mutate it; every
        write path rebuilds its own copy under the lock."""
        snapshot, etag = await _from_store(store.read_ledgers(job_id), "ledger read")
        if etag is not None and read_cache.get("etag") == etag:
            return read_cache["state"]
        state = await asyncio.to_thread(_rebuild_ledgers_checked, job, snapshot)
        if etag is not None:
            read_cache.update(etag=etag, state=state)
        return state

    async def _read_job_checked() -> JobSpec | None:
        """The manifest, or the same named 500 ``submit_corpus`` raises on one
        that no longer parses -- shared so the GET routes, which read the
        identical object, fail the same way an operator's corrupt manifest.
        """
        try:
            job, _ = await _from_store(store.read_job(job_id), "manifest read")
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

    def _ledger_corrupt(exc: LedgerSnapshotError) -> HTTPException:
        # A corrupt ledger or segment refuses every miner on this job until an
        # operator repairs it, so the refusal is named rather than a bare 500.
        logger.error("corpus ledgers for %s are unreadable: %s", job_id, exc)
        return HTTPException(status_code=500, detail="corpus_ledger_corrupt")

    async def _ensure_seen(refs: Sequence[SegmentRef]) -> None:
        """Load the segments a ledger version names: a missing or altered one
        is the named 500, a bucket that cannot answer the retryable 503."""
        try:
            await _from_store(seen_index.ensure(refs), "seen segment read")
        except LedgerSnapshotError as exc:
            raise _ledger_corrupt(exc) from exc

    def _rebuild_ledgers_checked(job: JobSpec, snapshot: Any) -> LedgerState:
        """``rebuild_ledgers``, translated the same way ``submit_corpus``
        translates it -- shared with the cursor route, which reads the same
        snapshot and must not turn a corrupt one into a bare lookup error.
        """
        try:
            return rebuild_ledgers(job, snapshot)
        except LedgerSnapshotError as exc:
            raise _ledger_corrupt(exc) from exc

    async def ledger_state(job: JobSpec | None = None) -> tuple[JobSpec | None, LedgerState | None]:
        """The manifest (read unless given) and the ledgers as the reads see
        them, for the status route."""
        if job is None:
            job = await _read_job_checked()
        if job is None:
            return None, None
        return job, await _read_state(job)

    router.ledger_state = ledger_state
    # The handlers themselves, so `build_corpus_jobs_router` can dispatch to
    # this job without a second copy of any of them.
    router.corpus_job = corpus_job
    router.corpus_cursor = corpus_cursor
    router.submit_corpus = submit_corpus
    router.corpus_next = corpus_next
    router.skip_corpus = skip_corpus
    router.ledger_lock = ledger_lock
    return router


class CorpusJobRoutes:
    """The jobs one app serves, changeable while it serves: a hot-added job's
    router joins, a retired job stops admitting and later leaves.

    ``default`` is the job the legacy paths answer for: the first one wired
    at boot, kept for the life of the process.
    """

    def __init__(self, routers: Mapping[str, APIRouter] | None = None, *,
                 default: str | None = None) -> None:
        self.routers: dict[str, APIRouter] = dict(routers or {})
        self.contracts: dict[str, Any] = {}
        self.retired: set[str] = set()
        # Admissions (submit, skip) past the retired check and not yet returned:
        # a job is unwired only once none is left.
        self.in_flight: collections.Counter = collections.Counter()
        self.default = default if default is not None else next(iter(self.routers), None)

    def add(self, job_id: str, router: APIRouter, *, contract: Any = None) -> None:
        self.routers[job_id] = router
        self.contracts[job_id] = contract
        self.retired.discard(job_id)
        if self.default is None:
            self.default = job_id

    def retire(self, job_id: str) -> None:
        self.retired.add(job_id)

    def remove(self, job_id: str) -> None:
        # Stays in `retired`: its miners keep hearing 410, not an unknown job.
        self.retired.add(job_id)
        self.routers.pop(job_id, None)
        self.contracts.pop(job_id, None)

    def open_jobs(self) -> list[str]:
        return sorted(j for j in self.routers if j not in self.retired)

    @contextlib.asynccontextmanager
    async def admission(self, job_id: str):
        """The job's router for one write, refused once retired, counted while it runs."""
        if job_id in self.retired:
            raise HTTPException(status_code=410, detail=JOB_RETIRED)
        router = self.routers.get(job_id)
        if router is None:
            raise HTTPException(status_code=404, detail="corpus_job_not_served")
        self.in_flight[job_id] += 1
        try:
            yield router
        finally:
            self.in_flight[job_id] -= 1


JOB_RETIRED = "job_retired"
SUBMIT_SCOPED_PATH = "/corpus/jobs/{job_id}/submit"


def build_corpus_jobs_router(routers: Mapping[str, APIRouter] | CorpusJobRoutes, *,
                             legacy: bool | None = None) -> APIRouter:
    """The job-scoped reads over one ``build_corpus_router`` per served job.

    With several jobs (or ``legacy=True``) it also owns the legacy paths:
    submit dispatches on the request's ``job_id``, and the legacy reads answer
    for the default job (the FIRST in ``routers``: the job live miners already
    mine, so adding a job never halts them). ``routers`` may be a
    ``CorpusJobRoutes`` the caller keeps changing; a retired job's next, skip
    and submit answer 410 ``job_retired``.
    """
    routes = routers if isinstance(routers, CorpusJobRoutes) else CorpusJobRoutes(routers)
    if legacy is None:
        legacy = len(routes.routers) > 1
    router = APIRouter()

    def _served(job_id: str) -> APIRouter:
        served = routes.routers.get(job_id)
        if served is not None:
            return served
        if job_id in routes.retired:
            raise HTTPException(status_code=410, detail=JOB_RETIRED)
        raise HTTPException(status_code=404, detail="corpus_job_not_served")

    def _admitting(job_id: str) -> APIRouter:
        # Before any store is touched: a retired job admits nothing more.
        if job_id in routes.retired:
            raise HTTPException(status_code=410, detail=JOB_RETIRED)
        return _served(job_id)

    @router.get(JOBS_PATH)
    async def corpus_jobs() -> dict:
        return {"jobs": routes.open_jobs()}

    @router.get(JOB_SCOPED_PATH)
    async def corpus_job_scoped(job_id: str) -> dict:
        return await _served(job_id).corpus_job()

    @router.get(CURSOR_SCOPED_PATH)
    async def corpus_cursor_scoped(job_id: str, hotkey: str) -> dict:
        return await _served(job_id).corpus_cursor(hotkey)

    @router.get(NEXT_SCOPED_PATH)
    async def corpus_next_scoped(job_id: str, hotkey: str) -> dict:
        return await _admitting(job_id).corpus_next(hotkey)

    @router.post(SKIP_SCOPED_PATH, response_model=CorpusSkipResponse)
    async def skip_corpus_scoped(job_id: str, request: CorpusSkipRequest) -> CorpusSkipResponse:
        # The path picks the job's router; that router still refuses a body
        # naming another job, so the two can never disagree silently.
        async with routes.admission(job_id) as served:
            return await served.skip_corpus(request)

    @router.post(SUBMIT_SCOPED_PATH, response_model=CorpusSubmissionResponse)
    async def submit_corpus_scoped(
        job_id: str, request: CorpusSubmissionRequest
    ) -> CorpusSubmissionResponse:
        async with routes.admission(job_id) as served:
            return await served.submit_corpus(request)

    if not legacy:
        return router

    def _default() -> str:
        if routes.default is None:
            raise HTTPException(status_code=404, detail="corpus_job_not_served")
        return routes.default

    @router.get(JOB_PATH)
    async def corpus_job_legacy() -> dict:
        return await _served(_default()).corpus_job()

    @router.get(CURSOR_PATH)
    async def corpus_cursor_legacy(hotkey: str) -> dict:
        return await _served(_default()).corpus_cursor(hotkey)

    @router.get(NEXT_PATH)
    async def corpus_next_legacy(hotkey: str) -> dict:
        return await _admitting(_default()).corpus_next(hotkey)

    @router.post(SKIP_PATH, response_model=CorpusSkipResponse)
    async def skip_corpus_legacy(request: CorpusSkipRequest) -> CorpusSkipResponse:
        # Dispatched on the body, as submit is.
        if request.job_id in routes.retired:
            raise HTTPException(status_code=410, detail=JOB_RETIRED)
        if request.job_id not in routes.routers:
            return _refuse_skip(
                CorpusRejectReason.JOB_NOT_SERVED,
                {"job_id": request.job_id, "serves": routes.open_jobs()},
            )
        async with routes.admission(request.job_id) as target:
            return await target.skip_corpus(request)

    @router.post(SUBMIT_PATH, response_model=CorpusSubmissionResponse)
    async def submit_corpus(request: CorpusSubmissionRequest) -> CorpusSubmissionResponse:
        # As a single job's route does: refused before any store is touched.
        if request.job_id in routes.retired:
            raise HTTPException(status_code=410, detail=JOB_RETIRED)
        if request.job_id not in routes.routers:
            return _refuse(
                CorpusRejectReason.JOB_NOT_SERVED,
                {"job_id": request.job_id, "serves": routes.open_jobs()},
            )
        async with routes.admission(request.job_id) as target:
            return await target.submit_corpus(request)

    return router


__all__ = [
    "CURSOR_PATH",
    "CURSOR_SCOPED_PATH",
    "CorpusJobRoutes",
    "CorpusPromptSourceError",
    "JOB_RETIRED",
    "CorpusSignatureUnavailable",
    "EnvironmentPromptJob",
    "JOBS_PATH",
    "NEXT_PATH",
    "NEXT_SCOPED_PATH",
    "SKIP_PATH",
    "SKIP_SCOPED_PATH",
    "JOB_PATH",
    "JOB_SCOPED_PATH",
    "LEDGER_SCHEMA",
    "LEDGER_SCHEMA_V1",
    "LEDGER_SCHEMA_V2",
    "LedgerState",
    "MAX_RESOLVED_PROMPT_SOURCES",
    "LedgerSnapshotError",
    "PromptFidelity",
    "RECORD_SCHEMA",
    "SEAL_THRESHOLD",
    "SEGMENT_MAX",
    "SUBMIT_PATH",
    "SUBMIT_SCOPED_PATH",
    "SeenIndex",
    "SeenView",
    "SegmentRef",
    "SingleTurnPromptJob",
    "SingleTurnPromptRenderer",
    "ChatTemplatePromptRenderer",
    "CHAT_TEMPLATE_RENDERERS",
    "build_corpus_jobs_router",
    "build_corpus_router",
    "downgrade_ledgers_v1",
    "ensure_ledgers_v2",
    "migrate_ledgers_at_startup",
    "verify_ledgers",
    "ledger_snapshot",
    "prompt_job_for_spec",
    "rebuild_ledgers",
    "seal_chunks",
    "refuse_unsigned_corpus_submissions",
    "renderer_for_job",
    "resolve_prompt_source",
]
