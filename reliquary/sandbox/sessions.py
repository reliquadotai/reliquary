"""Session tokens for signed episodes (spec §4.0 and §7; plan 3 rulings 1-3, 9).

A miner asks for a token naming its engagement. The issuer checks the per-hotkey caps,
asks the engagement's book for its terms (corpus: the job is served and signed, the
prompt is the job's, a slot is free once live reservations are counted; RL: not yet
served), picks a machine, signs the claims with the validator key, stores the session
(without the token's signature) and reserves the slot. Only then is the token returned.

Tokens are bearer secrets until their episode closes: they live in this process's
memory only (to answer a resent request with the same token) and are never logged or
written. Requests are idempotent per (hotkey, request_id): the same request id with
another engagement (by digest) is refused `request_conflict`.

States. A session is `live` until the first of:
* `closed_graded`: a verified `graded` final was reported. It still HOLDS its slot,
  because the transcript is still submittable: freeing it would let another miner take
  the slot this episode may yet consume. It ends `submitted`, `lapsed`, or `closed`
  when the miner withdraws it;
* `submitted`: its submission was accepted. Terminal; reached only from `live` or
  `closed_graded` (`session_submittable`);
* `closed`: an unpaid final (expired, box_failed, budget_exhausted including
  `transcript_bytes`), a failed open, or a withdrawal. Terminal for payment: its slot
  was freed, so a submission after it could take a slot another miner now holds.

Closes. Reason `final` reports the machine's final record with its transcript (a live
session only); `open_failed` reports a box that never opened, with no transcript (a
live session only); `withdraw` gives up a transcript the miner will not submit (its
precheck failed, its state is not UTF-8, its submission was refused): it needs the
session's own transcript, verified exactly as for `final`, moves a `live` or
`closed_graded` session to `closed` (status `withdrawn`) and so frees its slot and its
live caps, never to be paid. A withdrawal and a claim exclude each other under the
issuer lock: a claim taken first (even during the withdrawal's verification) wins
and the withdrawal answers the claimed state; a withdrawal settled first wins and the
claim is refused `session_not_submittable`.

The intake claims a session (`SessionIssuer.claim`, under the issuer lock) BEFORE its
ledger write: only a `live` or `closed_graded` session (`session_submittable`) of the
submitting hotkey, not already claimed; or a `lapsed` one whose submission was received
by its deadline (it lapsed while that submission was being checked). A claimed session
is frozen: no close, drain or lapse moves it until `submitted` (after the write) or
`release_claim` (the write did not accept it). A claim is bounded by `claim_ttl_s`:
past it (a leaked claim), drain, lapse and close act again, with an ALERT log, and a
new submission may claim it.
* `aborted`: void, refunded from the open rate, counted against the aborted cap;
  never paid;
* `voided`: its machine was drained. Our fault, not the miner's: not counted in the
  open rate, so the miner opens again; never paid (plan ruling 3 as amended
  2026-10-06: a late graded submission of a voided session is refused, which leaves
  no room for oversubscription);
* `lapsed`: past `expires_at + GRADING_GRACE_S`, when submissions are refused anyway;
  never paid.
The moves allowed are `sandbox_store.SESSION_TRANSITIONS`, used both by the book and by
the store's compare-and-swap, so a stale write never overwrites a later state. A
failed write of a state is retried with backoff and alerts on final failure.

Squatting. A hotkey holds at most one session per prompt and `max_live_per_hotkey_job`
per job (and `max_live_per_hotkey` overall), so it cannot reserve a prompt's slots to
starve others. Releasing early a session whose box never opened would bound squatting
further, but the validator cannot tell an unopened session from one mid-episode until
the machine's signed heartbeat lists its opened session ids (a reliquary-sandbox
change, not made yet); until then a never-opened token holds its slot until its close,
its machine's drain, or its lapse.

While the fleet's machine directory is stale (`fleet.directory_ready()` is false) no
token is issued and no close is verified: both are refused `directory_unavailable`
with a retry delay, never judged against an empty directory (which would read as
`unknown_key`). Ledger reads, task resolution and store writes are bounded by
`io_timeout_s`; a close's transcript is verified outside the issuer's lock and its
state re-checked under it.

Assumptions. One validator issues sessions for a given job ledger: reservations are
this process's memory (restored from R2), not a shared lock. `restore()` must succeed
before the first open: a validator that cannot read its sessions back would overbook
prompts and reset caps, so a failed restore aborts startup (enforced by the wiring).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Protocol

from reliquary_sandbox.attest import (
    GRADING_GRACE_S, STATUS_ABORTED, STATUS_GRADED, Budgets, Expected, SessionClaims,
    issue_session_token, token_sha256, verify_transcript,
)

from reliquary.corpus.job import is_signed_sandbox
from reliquary.corpus.signed_reasons import corpus_engagement
from reliquary.eval.rows import RowMap, job_row_map
from reliquary.infrastructure import sandbox_store as _store
from reliquary.infrastructure.sandbox_store import (
    SESSION_TRANSITIONS, SessionStoreConflict, session_transition_allowed,
)
from reliquary.sandbox.capacity_refusals import CAPACITY_REFUSALS  # one set with the RL yield ledger
from reliquary.sandbox.tasks import ResolvedTask

logger = logging.getLogger(__name__)

SESSION_SCHEMA = "reliquary/sandbox-session/v1"
LIVE, CLOSED_GRADED, SUBMITTED = _store.SESSION_LIVE, _store.SESSION_CLOSED_GRADED, _store.SESSION_SUBMITTED
CLOSED, ABORTED, VOIDED, LAPSED = (_store.SESSION_CLOSED, _store.SESSION_ABORTED,
                                   _store.SESSION_VOIDED, _store.SESSION_LAPSED)
HANDED_BACK = _store.SESSION_HANDED_BACK          # a ``submitted`` session's closed status
STATES = frozenset(SESSION_TRANSITIONS)
HOLDING = frozenset({LIVE, CLOSED_GRADED})
WITHDRAW, WITHDRAWN = "withdraw", "withdrawn"         # the close reason, the closed status
HOUR, DAY = 3600, 86400
MINUTE = 60
PERSIST_ATTEMPTS = 4
PERSIST_BACKOFF_S = 0.5
_sleep = asyncio.sleep


SUBMITTABLE = frozenset({LIVE, CLOSED_GRADED})


def session_submittable(state: Any) -> bool:
    """Whether a session in `state` may still be paid. The intake calls this before its
    ledger write; every other state (closed, aborted, voided, lapsed, submitted) is
    refused."""
    return isinstance(state, str) and state in SUBMITTABLE


@dataclass(frozen=True)
class SandboxPolicy:
    max_live_per_hotkey: int = 8
    max_live_per_hotkey_job: int = 4
    max_opens_per_hour: int = 120
    max_aborted_per_day: int = 20
    open_window_s: int = 900
    request_skew_s: int = 120
    retry_after_s: int = 10
    io_timeout_s: int = 10
    # A ledger turn's wait (corpus_service.LEDGER_LOCK_TIMEOUT_SECONDS, 30 s), the
    # record write's retries (RECORD_WRITE_ATTEMPTS x io_timeout_s) and 60 s of margin,
    # rounded up generously: no honest submission holds its claim longer.
    claim_ttl_s: int = 300
    # How long an intake's claim waits for the issuer lock (held only for in-memory
    # checks and one store write) before it is refused `session_busy`, retryably.
    claim_wait_s: int = 5
    # Refused opens per hotkey per rolling minute before `open_refused_rate` (429): a
    # refused open costs ledger reads and is free to send.
    max_refused_opens_per_minute: int = 60

    _ENV = {"max_live_per_hotkey": "RELIQUARY_SANDBOX_MAX_LIVE_PER_HOTKEY",
            "max_live_per_hotkey_job": "RELIQUARY_SANDBOX_MAX_LIVE_PER_HOTKEY_JOB",
            "max_opens_per_hour": "RELIQUARY_SANDBOX_MAX_OPENS_PER_HOUR",
            "max_aborted_per_day": "RELIQUARY_SANDBOX_MAX_ABORTED_PER_DAY",
            "open_window_s": "RELIQUARY_SANDBOX_OPEN_WINDOW_S",
            "request_skew_s": "RELIQUARY_SANDBOX_REQUEST_SKEW_S",
            "retry_after_s": "RELIQUARY_SANDBOX_RETRY_AFTER_S",
            "io_timeout_s": "RELIQUARY_SANDBOX_IO_TIMEOUT_S",
            "claim_ttl_s": "RELIQUARY_SANDBOX_CLAIM_TTL_S",
            "claim_wait_s": "RELIQUARY_SANDBOX_CLAIM_WAIT_S",
            "max_refused_opens_per_minute": "RELIQUARY_SANDBOX_MAX_REFUSED_OPENS_PER_MINUTE"}

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> SandboxPolicy:
        values = {}
        for name, variable in cls._ENV.items():
            raw = environ.get(variable)
            if raw is not None:
                value = int(raw)
                if value <= 0:
                    raise ValueError(f"{variable} must be a positive integer")
                values[name] = value
        return cls(**values)


@dataclass(frozen=True)
class Refusal:
    reason: str
    detail: dict = field(default_factory=dict)
    retry_after: int | None = None


@dataclass(frozen=True)
class Grant:
    session_id: str
    token: dict = field(repr=False)          # a bearer secret: never in a repr or a log
    gateway_url: str = ""
    expires_at: int = 0


@dataclass(frozen=True)
class EngagementTerms:
    engagement: str
    env: str
    split: str
    index: int
    checkpoint: str
    image: str
    env_package: str
    budgets: dict[str, int]
    job_id: str | None = None
    # The digest of the env options this validator resolved the task with: a machine
    # must publish the same one for the env (`SandboxFleet.pick`). None: not checked.
    env_options_sha256: str | None = None
    # The image-lock digest the task was resolved under: a machine reporting another lock for the env is not
    # one to place it on (`SandboxFleet._serves_env`). None: not checked.
    env_lock_sha256: str | None = None
    prompt_index: int | None = None
    # The prompt's free slots as read (before the issuer lock), reservations not
    # deducted: the issuer re-checks them against its reservations under the lock.
    slots_remaining: int | None = None
    # Reads the free slots again (bounded; a Refusal when it fails): the issuer calls
    # it under the lock when the prompt's reservations moved since the first read.
    refresh_slots: Callable[[], Awaitable[int | None | Refusal]] | None = field(
        default=None, repr=False, compare=False)
    # At most one session per engagement (an RL seed) unless the only earlier one was aborted
    # (machine-signed, at most once); a drain never frees it. Checked under the issuer lock. Corpus: False.
    exclusive: bool = False
    # The terms' cheap preconditions read again under the issuer lock (the RL precommit's
    # window may have turned while the task resolved): a Refusal refuses the open. Corpus: None.
    still_valid: Callable[[], Refusal | None] | None = field(default=None, repr=False, compare=False)
    # The task's prepare timeout as its gateway will read it: the token must outlive it (``token_validity_s``).
    # None: the sandbox's default.
    prepare_timeout_s: float | None = None
    # RL: what the session reserves on its machine, {"cpu_millis": int | None, "memory_bytes": int} (the gateway
    # reserves its agent box); placement then goes by CPU and memory. None: slots only (corpus).
    reserve: dict[str, Any] | None = None


class EngagementBook(Protocol):
    kind: str

    async def terms(self, hotkey: str, engagement: Mapping[str, Any]) -> EngagementTerms | Refusal: ...


class SessionEvents(Protocol):
    """What the RL wiring hears of its sessions (the yield ledger and the quota's index). Called on the
    event loop after the state change; must not block. A raise is logged and dropped."""

    def opened(self, record: SessionRecord) -> None: ...

    def closed(self, record: SessionRecord, final: Mapping[str, Any] | None) -> None: ...


class FanOutEvents:
    """Several ``SessionEvents`` listeners as one (the yield ledger and the quota's index). A listener that
    raises is logged and the others still hear of the event."""

    def __init__(self, *listeners) -> None:
        self.listeners = tuple(listener for listener in listeners if listener is not None)

    def opened(self, record: SessionRecord) -> None:
        for listener in self.listeners:
            try:
                listener.opened(record)
            except Exception:
                logger.exception("sandbox session listener opened failed")

    def closed(self, record: SessionRecord, final: Mapping[str, Any] | None) -> None:
        for listener in self.listeners:
            try:
                listener.closed(record, final)
            except Exception:
                logger.exception("sandbox session listener closed failed")

    def open_refused(self, reason: str, env: str | None, engagement: str | None, image: str | None = None) -> None:
        """Every refused open (the saturation metrics): its reason, and its env, engagement and image when the terms
        were read."""
        for listener in self.listeners:
            method = getattr(listener, "open_refused", None)   # optional: the saturation monitor has it
            if method is None:
                continue
            try:
                method(reason, env, engagement, image)
            except Exception:
                logger.exception("sandbox session listener open_refused failed")

    def refused(self, hotkey: str, engagement: Mapping[str, Any], refusal: Refusal) -> None:
        for listener in self.listeners:
            method = getattr(listener, "refused", None)        # optional: the yield ledger has it
            if method is None:
                continue
            try:
                method(hotkey, engagement, refusal)
            except Exception:
                logger.exception("sandbox session listener refused failed")


def runtime_start_fault(final: Mapping[str, Any] | None) -> str | None:
    """``runtime_exec_failed`` when the machine's signed final proves the runtime failed to start a
    command (``grading._grader.runtime_exec_failed``), else None: a fault of the machine, not the miner's."""
    grading = final.get("grading") if isinstance(final, Mapping) else None
    label = grading.get("_grader") if isinstance(grading, Mapping) else None
    return "runtime_exec_failed" if isinstance(label, Mapping) and label.get("runtime_exec_failed") is True else None


GRADING_POOL_ABORT = "the grading pool stayed full"
"""The gateway's abort reason when other miners' gradings held the machine's grading pool (its own
hotkey's burst has another reason: an ordinary abort)."""
LOST_OPEN_WINDOW_S = 60
"""An open answer a validator restart can have lost: issued at most this long before the last activity
preceding the restart (the miner's sessions HTTP timeout)."""


def pool_abort_fault(final: Mapping[str, Any] | None) -> str | None:
    """``grading_pool`` when the machine's signed final is an abort because other miners' gradings held its
    grading pool: it counts toward no abort cap and frees its engagement."""
    if (isinstance(final, Mapping) and final.get("status") == STATUS_ABORTED
            and final.get("reason") == GRADING_POOL_ABORT):
        return "grading_pool"
    return None


def _open_refused_fault(receipt: Mapping[str, Any]) -> str:
    """The fault of a verified signed open refusal: ``open_refused:<the gateway's reason>``."""
    document = receipt.get("document") if isinstance(receipt, Mapping) else None
    reason = document.get("reason") if isinstance(document, Mapping) else None
    if not isinstance(reason, str) or not reason or len(reason) > 48:
        reason = "unnamed"
    return f"open_refused:{reason}"


GATEWAY_PREPARE_SLACK_S = 60
"""The gateway refuses (401, unsigned) a token with no more than its task's prepare timeout plus this left
at the open."""
DEFAULT_PREPARE_TIMEOUT_S = 600
"""A sandbox task's prepare timeout when it declares none."""
OPEN_TRANSIT_S = 60
"""The time a miner may take to present a fresh token to its gateway (its sessions HTTP timeout)."""


def token_validity_s(open_window_s: int, wall_s: int, prepare_timeout_s: float | None = None) -> int:
    """How long a session token is valid: its open window and its episode's ``wall_s``, the window
    lengthened when it would not cover the task's preparation, the gateway's slack and the token's transit.
    A token shorter than that is refused unsigned by every gateway, which no miner could prove."""
    prepare = DEFAULT_PREPARE_TIMEOUT_S if prepare_timeout_s is None else prepare_timeout_s
    floor = math.ceil(prepare) + GATEWAY_PREPARE_SLACK_S + OPEN_TRANSIT_S
    return max(int(open_window_s), floor) + int(wall_s)


class JobNotReady(Exception):
    """A served job's view is registered but its submit router is not adopted yet (a
    hot add in progress): the open is refused retryably."""


@dataclass(frozen=True)
class SignedJobView:
    """What the corpus engagement book needs of a served signed job. ``rows`` is the
    job's ``RowMap`` (``reliquary.eval.rows.job_row_map``), resolved once, when the
    view is built, never per request."""

    job: Any
    resolve_task: Callable[[int], Awaitable[ResolvedTask]] | None
    slots_remaining: Callable[[int], Awaitable[int | None]] | None
    is_banned: Callable[[str], Awaitable[bool]] | None = None
    rows: RowMap | None = None

    def __post_init__(self) -> None:
        if self.rows is None:
            object.__setattr__(self, "rows", job_row_map(self.job))


def engagement_digest(engagement: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(engagement, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


_FIELD_CHECKS: dict[str, Callable[[Any], bool]] = {
    "session_id": _is_str, "hotkey": _is_str, "request_id": _is_str,
    "engagement_sha256": _is_str, "kind": _is_str, "engagement": _is_str, "env": _is_str,
    "split": _is_str, "index": _is_int, "checkpoint": _is_str,
    "job_id": lambda v: v is None or _is_str(v),
    "prompt_index": lambda v: v is None or _is_int(v),
    "machine_id": _is_str, "issued_at": _is_int, "expires_at": _is_int,
    "token_sha256": _is_str, "state": lambda v: v in STATES,
    "closed_status": lambda v: v is None or isinstance(v, str),
    "closed_at": lambda v: v is None or _is_int(v),
    "fault": lambda v: v is None or (_is_str(v) and 0 < len(v) <= 64),
    "ended_at": lambda v: v is None or _is_int(v),
    "ended_status": lambda v: v is None or (_is_str(v) and 0 < len(v) <= 64),
}
_OPTIONAL_FIELDS = ("fault", "ended_at", "ended_status")      # absent from a document when None


@dataclass
class SessionRecord:
    session_id: str
    hotkey: str
    request_id: str
    engagement_sha256: str
    kind: str
    engagement: str
    env: str
    split: str
    index: int
    checkpoint: str
    job_id: str | None
    prompt_index: int | None
    machine_id: str
    issued_at: int
    expires_at: int
    token_sha256: str
    state: str = LIVE
    closed_status: str | None = None
    closed_at: int | None = None
    # RL only, set when the issuer records faults: why this session is neither useful nor wasted for its
    # miner's yield. Absent from the document when None (older documents keep their bytes).
    fault: str | None = None
    # RL only: when the session first closed with its machine's signed final (its box's end; None for a failed
    # open). A submission or a withdrawal later moves ``closed_at`` (a withdrawal also ``closed_status``) but
    # never this, so the yield weight never counts an unsubmitted tail, and a box that ran is known to have run.
    ended_at: int | None = None
    # RL only: the status of that machine's signed final (``graded``, ``expired``...), set with ``ended_at`` when the
    # session was closed with it (``final``), never when the miner withdrew it live (it refused its own episode). A
    # later withdrawal rewrites ``closed_status`` to ``withdrawn``, never this: the yield ledger judges by it whether
    # the session's box ended graded (a group that could not form for capacity keeps its graded sessions neutral).
    ended_status: str | None = None

    def to_document(self) -> dict:
        document = {"schema": SESSION_SCHEMA, **asdict(self)}
        for optional in _OPTIONAL_FIELDS:
            if document[optional] is None:
                del document[optional]
        return document

    @classmethod
    def from_document(cls, document: Any) -> SessionRecord:
        """Every field is type-checked: a malformed document raises ValueError."""
        if not isinstance(document, Mapping) or document.get("schema") != SESSION_SCHEMA:
            raise ValueError("not a session document")
        values = {}
        for f in fields(cls):
            if f.name in _OPTIONAL_FIELDS and f.name not in document:
                continue                                   # older documents: no fault, no end
            if f.name not in document or not _FIELD_CHECKS[f.name](document[f.name]):
                raise ValueError(f"session document field {f.name} is missing or malformed")
            values[f.name] = document[f.name]
        return cls(**values)


class CorpusEngagements:
    kind = "corpus"

    def __init__(self, jobs: Callable[[str], SignedJobView | None], book: SessionBook,
                 clock: Callable[[], float] = time.time) -> None:
        self._jobs, self._book, self._clock = jobs, book, clock

    async def terms(self, hotkey: str, engagement: Mapping[str, Any]) -> EngagementTerms | Refusal:
        timeout = self._book.policy.io_timeout_s
        retry = self._book.policy.retry_after_s
        job_id, index = engagement.get("job_id"), engagement.get("prompt_index")
        view = self._jobs(job_id) if isinstance(job_id, str) else None
        if view is None:
            return Refusal("job_not_served", {"job_id": job_id})
        job = view.job
        if not is_signed_sandbox(job):
            return Refusal("job_not_signed", {"job_id": job_id})
        if isinstance(index, bool) or not isinstance(index, int) or not job.owns(index):
            return Refusal("prompt_mismatch", {"prompt_index": index})
        try:
            if view.is_banned is not None and await asyncio.wait_for(view.is_banned(hotkey),
                                                                     timeout):
                return Refusal("miner_banned", {})
            remaining = await asyncio.wait_for(view.slots_remaining(index), timeout)
        except JobNotReady:
            return Refusal("job_not_ready", {"job_id": job_id}, retry_after=retry)
        except Exception as exc:
            # A timeout, the store's 503 (an HTTPException) or any transport error: the
            # validator cannot read its ledger now, the miner retries (never a 500).
            logger.warning("sandbox open: job %s ledger unreadable (%s)", job_id,
                           type(exc).__name__)
            return Refusal("ledger_unavailable", {"job_id": job_id}, retry_after=retry)
        if remaining is None:
            return Refusal("job_complete", {"job_id": job_id})
        reserved = self._book.reserved(job.job_id, index, int(self._clock()))
        if remaining - reserved <= 0:
            return Refusal("prompt_unavailable", {"prompt_index": index,
                                                  "slots_remaining": remaining, "reserved": reserved})
        async def refresh_slots() -> int | None | Refusal:
            try:
                return await asyncio.wait_for(view.slots_remaining(index), timeout)
            except JobNotReady:
                return Refusal("job_not_ready", {"job_id": job_id}, retry_after=retry)
            except Exception as exc:
                logger.warning("sandbox open: job %s ledger unreadable on re-read (%s)",
                               job_id, type(exc).__name__)
                return Refusal("ledger_unavailable", {"job_id": job_id}, retry_after=retry)

        try:
            task = await asyncio.wait_for(view.resolve_task(index), timeout)
        except Exception as exc:
            logger.warning("sandbox open: task %d of job %s unresolved (%s)", index, job_id,
                           type(exc).__name__)
            return Refusal("task_unavailable", {"prompt_index": index}, retry_after=retry)
        spec = job.episode.sandbox
        budgets = spec.budgets.to_contract()
        for name, value in task.limits.items():
            if name in budgets:
                budgets[name] = max(budgets[name], int(value))
        # The engagement and the reservation name the row; the token names its task.
        ref = view.rows.ref(index)
        return EngagementTerms(engagement=corpus_engagement(job.job_id, index), env=spec.env,
                               split=ref.split, index=ref.index,
                               checkpoint=job.checkpoint_sha256, image=task.image,
                               env_package=spec.env_package, budgets=budgets,
                               job_id=job.job_id, prompt_index=index,
                               env_options_sha256=getattr(task, "env_options_sha256", None),
                               env_lock_sha256=getattr(task, "env_lock_sha256", None),
                               slots_remaining=int(remaining), refresh_slots=refresh_slots)


class RlPrecommitEngagements:
    """The episodic RL task's engagement (spec §4.0): not served yet; every request is
    refused `engagement_kind_unsupported`. The mapping the RL task implements here:

    * `engagement` = `rl:{window}:{precommit_sha256}`, the precommit that already fixes
      the prompt, the checkpoint and the drand round;
    * `index` = the precommit's prompt index; `checkpoint` = the window checkpoint's
      sha256; `env`/`split`/budgets from the RL env's signed-sandbox spec;
    * the reservation is the precommit itself (no slot ledger): one live session per
      precommit; `aborted` voids it (the miner may open again), every other final
      consumes it; the intake core (`validator.signed_intake.verify_signed_episode`) is
      shared, with its `Expected` built from the precommit;
    * a close with reason `open_failed` must NOT void the precommit (unlike corpus,
      where it only releases a reservation): the miner could otherwise discard a
      precommitted draw it dislikes by claiming its box failed to open."""

    kind = "rl_precommit"

    async def terms(self, hotkey: str, engagement: Mapping[str, Any]) -> Refusal:
        return Refusal("engagement_kind_unsupported",
                       {"kind": self.kind, "why": "the episodic RL task is not served yet"})


class SessionBook:
    """Sessions this validator issued, in memory (restored from the store at start)."""

    def __init__(self, policy: SandboxPolicy) -> None:
        self.policy = policy
        self._sessions: dict[str, SessionRecord] = {}
        self._requests: dict[tuple[str, str], str] = {}
        self._claimed: dict[str, int] = {}          # session id -> claimed at
        # Replaced, never mutated: read from the intake's thread while the loop
        # changes the book.
        self._submitted: frozenset[str] = frozenset()
        # (job id, prompt index) -> bumped whenever one of its sessions changes state
        # (submitted, closed, withdrawn, aborted, lapsed, voided): an open whose free-
        # slot read predates a bump reads the ledger again under the issuer lock.
        self._generations: dict[tuple[Any, Any], int] = {}
        # Precommit sha256 -> ids of the RL sessions whose engagement names it (the group
        # claim's one-paid-group-per-precommit check reads it under the issuer lock, no full scan).
        self._by_precommit: dict[str, set[str]] = {}
        # Hotkey -> ids of its sessions (the per-hotkey caps read it under the issuer lock, no full scan).
        self._by_hotkey: dict[str, set[str]] = {}
        # Machine id -> ids of the sessions issued on it (who holds a machine is read from it, no full scan).
        self._by_machine: dict[str, set[str]] = {}
        # How much the per-hotkey abort and refused-open rates scale for a hotkey (the RL quota sets it to
        # the hotkey's allocation in groups; 1 everywhere else).
        self.cap_scale: Callable[[str], float] = lambda hotkey: 1.0

    def generation(self, job_id: Any, prompt_index: Any) -> int:
        return self._generations.get((job_id, prompt_index), 0)

    def claim(self, session_id: str, now: int) -> bool:
        """Mark a session as being paid; False while a fresh claim already holds it."""
        if self.claim_fresh(session_id, now):
            return False
        self._claimed[session_id] = int(now)
        return True

    def release_claim(self, session_id: str) -> None:
        self._claimed.pop(session_id, None)

    def is_claimed(self, session_id: str) -> bool:
        return session_id in self._claimed

    def claim_fresh(self, session_id: str, now: int) -> bool:
        at = self._claimed.get(session_id)
        return at is not None and now - at <= self.policy.claim_ttl_s

    def _frozen(self, record: SessionRecord, now: int, action: str) -> bool:
        """Whether a claim keeps `action` off this session; a stale claim does not, loudly."""
        if record.session_id not in self._claimed:
            return False
        if self.claim_fresh(record.session_id, now):
            return True
        logger.error("ALERT sandbox session %s claimed %d s ago, past its %d s ttl: %s "
                     "proceeds", record.session_id, now - self._claimed[record.session_id],
                     self.policy.claim_ttl_s, action)
        return False

    def add(self, record: SessionRecord) -> None:
        previous = self._sessions.get(record.session_id)
        if previous is not None:
            self._unindex(previous)
        self._sessions[record.session_id] = record
        self._by_hotkey.setdefault(record.hotkey, set()).add(record.session_id)
        self._by_machine.setdefault(record.machine_id, set()).add(record.session_id)
        sha = _rl_precommit_of(record)
        if sha is not None:
            self._by_precommit.setdefault(sha, set()).add(record.session_id)
        self._requests[(record.hotkey, record.request_id)] = record.session_id
        if record.state == SUBMITTED:
            self._submitted = self._submitted | {record.session_id}

    def restore(self, records) -> None:
        for record in records:
            self.add(record)

    def get(self, session_id: str) -> SessionRecord | None:
        return self._sessions.get(session_id)

    def records(self) -> tuple[SessionRecord, ...]:
        return tuple(self._sessions.values())

    def _unindex(self, record: SessionRecord) -> None:
        sha = _rl_precommit_of(record)
        ids = self._by_precommit.get(sha) if sha is not None else None
        if ids is not None:
            ids.discard(record.session_id)
            if not ids:
                del self._by_precommit[sha]
        ids = self._by_hotkey.get(record.hotkey)
        if ids is not None:
            ids.discard(record.session_id)
            if not ids:
                del self._by_hotkey[record.hotkey]
        on = self._by_machine.get(record.machine_id)
        if on is not None:
            on.discard(record.session_id)
            if not on:
                del self._by_machine[record.machine_id]

    def holding_on(self, machine_ids, now: int) -> list[SessionRecord]:
        """The held sessions on these machines (an index, never a scan of the book)."""
        found = []
        for machine_id in set(machine_ids):
            for session_id in self._by_machine.get(machine_id, ()):
                record = self._sessions.get(session_id)
                if record is not None and self._holds(record, now):
                    found.append(record)
        return found

    def held_on(self, machine_ids, now: int) -> dict[str, int]:
        """Held sessions per hotkey on these machines (an index, never a scan of the book)."""
        counts: dict[str, int] = {}
        for machine_id in set(machine_ids):
            for session_id in self._by_machine.get(machine_id, ()):
                record = self._sessions.get(session_id)
                if record is not None and self._holds(record, now):
                    counts[record.hotkey] = counts.get(record.hotkey, 0) + 1
        return counts

    def hotkeys(self) -> tuple[str, ...]:
        """The hotkeys with a session in the book."""
        return tuple(self._by_hotkey)

    def held_by(self, hotkeys, now: int) -> int:
        """Held sessions of these hotkeys, fleet-wide (from the per-hotkey index)."""
        return sum(1 for hotkey in set(hotkeys) for s in self._by_hotkey.get(hotkey, ())
                   if s in self._sessions and self._holds(self._sessions[s], now))

    def held_on_precommit(self, machine_ids, now: int, engagement: str) -> int:
        """Held sessions on these machines of the precommit the RL ``engagement`` names (its own group); 0 for
        an engagement that is not RL."""
        sha = _rl_precommit_engagement(engagement)
        if sha is None:
            return 0
        machines = set(machine_ids)
        return sum(1 for s in self._by_precommit.get(sha, ())
                   if s in self._sessions and self._sessions[s].machine_id in machines
                   and self._holds(self._sessions[s], now))

    def of_precommit(self, precommit_sha256: str) -> tuple[SessionRecord, ...]:
        """The RL sessions whose engagement names this precommit (an index, not a scan)."""
        ids = self._by_precommit.get(precommit_sha256, ())
        return tuple(self._sessions[s] for s in sorted(ids) if s in self._sessions)

    def by_request(self, hotkey: str, request_id: str) -> SessionRecord | None:
        session_id = self._requests.get((hotkey, request_id))
        return None if session_id is None else self._sessions.get(session_id)

    @staticmethod
    def _holds(record: SessionRecord, now: int) -> bool:
        return record.state in HOLDING and now <= record.expires_at + GRADING_GRACE_S

    def reserved(self, job_id: str, prompt_index: int, now: int) -> int:
        return sum(1 for r in self._sessions.values()
                   if r.job_id == job_id and r.prompt_index == prompt_index and self._holds(r, now))

    def engagement_held(self, engagement: str) -> bool:
        """Whether an exclusive (RL) engagement is consumed. Only a machine-signed ``aborted``
        frees it, and at most ONCE: a ``voided`` session (a drain frees what the miner left ``live``, so
        it would be a selective re-roll) and every other state keep it taken, and so does a second
        ``aborted`` record of the same engagement. A pool abort caused by other miners never counts."""
        aborted = 0
        sha = _rl_precommit_engagement(engagement)
        if sha is None:
            candidates = self._sessions.values()
        else:                               # an RL engagement: only its precommit's sessions (an index, no scan)
            candidates = [self._sessions[s] for s in self._by_precommit.get(sha, ()) if s in self._sessions]
        for r in candidates:
            if r.engagement != engagement:
                continue
            if r.state != ABORTED:
                return True
            if r.fault == "grading_pool":
                continue                # other miners' gradings held the pool: never counted
            aborted += 1
        return aborted >= 2

    def precommits_held(self, now: int) -> frozenset[str]:
        """The precommits a session still needs (one of its sessions is held, or claimed by a
        group in flight): their runtime rows must stay."""
        return frozenset(sha for sha, ids in self._by_precommit.items()
                         if any(s in self._claimed or (s in self._sessions and self._holds(self._sessions[s], now))
                                for s in ids))

    def submitted_ids(self) -> frozenset[str]:
        """Safe from any thread: an immutable set, kept up to date by `settle`."""
        return self._submitted

    def open_refusal(self, hotkey: str, now: int) -> Refusal | None:
        mine = [self._sessions[s] for s in self._by_hotkey.get(hotkey, ()) if s in self._sessions]
        policy = self.policy
        live = sum(1 for r in mine if self._holds(r, now))
        if live >= policy.max_live_per_hotkey:
            return Refusal("live_cap", {"live": live, "max": policy.max_live_per_hotkey})
        # A session handed back still counts: it was opened (only the yield accounting excludes it).
        counted = sorted(r.issued_at for r in mine
                         if r.issued_at > now - HOUR and r.state not in (ABORTED, VOIDED))
        opens = len(counted)
        if opens >= policy.max_opens_per_hour:     # retry when enough of them leave the hour
            return Refusal("open_rate_cap", {"opens": opens, "max": policy.max_opens_per_hour},
                           retry_after=counted[opens - policy.max_opens_per_hour] + HOUR - now)
        closed = sorted(r.closed_at or 0 for r in mine
                        if r.state == ABORTED and r.fault != "grading_pool" and (r.closed_at or 0) > now - DAY)
        aborted = len(closed)
        cap = int(policy.max_aborted_per_day * self.cap_scale(hotkey))      # scale 1: the policy's own cap
        if aborted >= cap:
            return Refusal("aborted_cap", {"aborted": aborted, "max": cap},
                           retry_after=closed[aborted - cap] + DAY - now)
        return None

    def engagement_refusal(self, hotkey: str, job_id: str | None, prompt_index: int | None,
                           now: int) -> Refusal | None:
        """Squatting caps: one held session per (hotkey, prompt), a few per (hotkey, job)."""
        if job_id is None:
            return None
        held = [r for r in self._sessions.values()
                if r.hotkey == hotkey and r.job_id == job_id and self._holds(r, now)]
        if any(r.prompt_index == prompt_index for r in held):
            return Refusal("prompt_live_cap", {"job_id": job_id, "prompt_index": prompt_index})
        if len(held) >= self.policy.max_live_per_hotkey_job:
            return Refusal("job_live_cap", {"job_id": job_id, "live": len(held),
                                            "max": self.policy.max_live_per_hotkey_job})
        return None

    def settle(self, session_id: str, state: str, *, now: int,
               status: str | None = None) -> SessionRecord | None:
        """Move a session to `state` if SESSION_TRANSITIONS allows it; None otherwise."""
        record = self._sessions.get(session_id)
        if record is None or not session_transition_allowed(record.state, state):
            return None
        record.state, record.closed_status, record.closed_at = state, status, int(now)
        if record.job_id is not None:
            key = (record.job_id, record.prompt_index)
            self._generations[key] = self._generations.get(key, 0) + 1
        if state == SUBMITTED:
            self._submitted = self._submitted | {session_id}
        return record

    def lapse(self, now: int) -> list[SessionRecord]:
        lapsed = [r for r in self._sessions.values()
                  if r.state in HOLDING and now > r.expires_at + GRADING_GRACE_S
                  and not self._frozen(r, now, "lapse")]
        for record in lapsed:
            self.settle(record.session_id, LAPSED, now=now, status=record.closed_status)
        return lapsed

    def void_machine(self, machine_id: str, now: int) -> list[SessionRecord]:
        """Only `live` sessions: a `closed_graded` one already holds its signed transcript,
        and a claimed one is being paid."""
        voided = [r for r in self._sessions.values()
                  if r.machine_id == machine_id and r.state == LIVE and self._holds(r, now)
                  and not self._frozen(r, now, "drain")]
        for record in voided:
            self.settle(record.session_id, VOIDED, now=now, status="machine_drained")
        return voided

    def prune(self, now: int) -> None:
        old = [s for s, r in self._sessions.items()
               if r.state not in HOLDING and (r.closed_at or r.issued_at) < now - DAY - HOUR]
        for session_id in old:
            record = self._sessions.pop(session_id)
            self._unindex(record)
            self._requests.pop((record.hotkey, record.request_id), None)
            self._claimed.pop(session_id, None)
        if old:
            self._submitted = self._submitted - set(old)
            # Only prompts with no session left (their last change is a day old: no
            # open in flight can have read before it).
            live_keys = {(r.job_id, r.prompt_index) for r in self._sessions.values()}
            for key in [k for k in self._generations if k not in live_keys]:
                del self._generations[key]


def _rl_precommit_of(record: SessionRecord) -> str | None:
    """The precommit an RL session's engagement names, or None (not an RL engagement)."""
    return _rl_precommit_engagement(record.engagement)


def _rl_precommit_engagement(engagement: str) -> str | None:
    """The precommit an RL engagement names, or None (not an RL engagement)."""
    # Only an RL engagement is parsed: a corpus engagement never loads the RL episode module.
    if not isinstance(engagement, str) or not engagement.startswith("rl:"):
        return None
    from reliquary.protocol.service_episode import EpisodeWireError, parse_rl_engagement

    try:
        return parse_rl_engagement(engagement)[1]
    except EpisodeWireError:
        return None


@dataclass(frozen=True)
class _Issued:
    grant: Grant
    machine_id: str
    engagement: str


_NO_CAPACITY = object()
"""``_issue_locked``'s answer when no machine has room now (or a waiter of the open's class is ahead)."""
_NEVER = object()
"""``_issue_locked``'s answer when no fresh machine could ever take the open (queued issuers only)."""


def _tier(value: Any) -> Any:
    """A reservation rounded up to its tier (the next power of two), so that per-task reservations of one env
    fall into a few classes rather than one each."""
    if type(value) is not int or value <= 0:
        return value
    return 1 << (value - 1).bit_length()


def _reserve_of(terms) -> tuple:
    reserve = getattr(terms, "reserve", None) or {}
    return (reserve.get("cpu_millis"), reserve.get("memory_bytes"))


def _reserving(terms) -> dict:
    """The ``reserve`` and ``env_lock_sha256`` keywords of a placement call: only when the terms carry them (a
    corpus call keeps its exact keywords)."""
    reserve = getattr(terms, "reserve", None)
    lock = getattr(terms, "env_lock_sha256", None)
    return {**({} if reserve is None else {"reserve": reserve}),
            **({} if lock is None else {"env_lock_sha256": lock})}


def _placement_class(terms) -> tuple:
    """The open queue's class: the env and the reservation's tier, NOT the image (one image per task would make
    every open its own class and bypass the queue); each waiter's own attempt filters on its image and exact
    reservation."""
    cpu_millis, memory_bytes = _reserve_of(terms)
    return (terms.env, _tier(cpu_millis), _tier(memory_bytes))


_UNTALLIED_REFUSALS = frozenset({"sandbox_saturated", "open_queue_full", "directory_unavailable",
                                 "environment_unserved"})
"""A queued issuer's refusals that are our capacity or our outage: never counted toward a hotkey's refused-open
rate."""
_NEVER_TALLIED = frozenset({"environment_unserved"})
"""Refusals that are our state whether or not the issuer queues (an env no machine serves): never tallied."""


class SessionIssuer:
    def __init__(self, *, book: SessionBook, store, fleet, signer, token_verifier,
                 engagements: Mapping[str, EngagementBook], policy: SandboxPolicy,
                 clock: Callable[[], float] = time.time,
                 new_session_id: Callable[[], str] = lambda: uuid.uuid4().hex,
                 events: SessionEvents | None = None, record_faults: bool = False,
                 open_refusal_check: Callable[[Mapping[str, Any], SessionRecord], bool | None] | None = None,
                 queue=None, operator_of: Callable[[str], str | None] | None = None,
                 entitled: Callable[[str], int] | None = None) -> None:
        self.book = book
        self._store, self._fleet, self._signer = store, fleet, signer
        self._token_verifier = token_verifier
        self._engagements = dict(engagements)
        self._policy, self._clock, self._new_id = policy, clock, new_session_id
        self._tokens: dict[str, tuple[dict, str]] = {}
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._refused_opens: dict[str, list[int]] = {}     # hotkey -> refusal times
        self._unserved_alerted: set[tuple] = set()          # (env, package, options) already ALERTed
        # Sessions whose ``submitted`` record is stored ahead of the book (``persist_submitted``).
        self._stored_submitted: set[str] = set()
        # The RL wiring's listeners and fault records (a corpus issuer has neither: its records keep their bytes).
        self._events = events
        self._record_faults = bool(record_faults)
        # A gateway's signed open refusal of a session: True verified, False not, None cannot judge now.
        self._open_refusal_check = open_refusal_check
        # The RL open queue (``open_queue.OpenQueue``): opens finding no room wait there, outside the lock. A corpus
        # issuer has none: no room is refused at once.
        self._queue = queue
        # The chain operator of a hotkey (the registration's coldkey): a saturation is judged by operator.
        self._operator_of = operator_of
        # The sessions a hotkey's allocation entitles it to (the RL quota's): within it, a saturation is ours.
        self._entitled = entitled
        self.attribution_failures = 0          # saturation attributions that raised (a bug of ours: no verdict)
        # RL session -> the share of its machine it reserves (``Fleet.session_share`` at issue), while it holds:
        # the saturation attribution weighs sessions by their box size.
        self._weights: dict[str, float] = {}
        self._restored: set[str] = set()       # sessions issued before this process started (their tokens lost)
        self._last_activity = 0                # the last issue or close before the restart
        self._lost: set[str] = set()           # restored sessions whose open answer our restart lost

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    async def restore(self) -> int:
        """Read the recent sessions back. A store failure raises: the caller must abort
        startup rather than serve opens with empty reservations and caps. A malformed
        document is skipped with a log."""
        records = []
        for document in await self._store.list_recent(int(self._clock())):
            try:
                records.append(SessionRecord.from_document(document))
            except (KeyError, TypeError, ValueError):
                logger.error("a sandbox session document is unreadable; skipped")
        self.book.restore(records)
        self._restored = {record.session_id for record in records}
        # The lost-open window is anchored on the last activity before the restart, not on the reload (a
        # restart takes minutes; the opens of the minute before the crash are the ones whose answers we lost).
        self._last_activity = max((max(int(record.issued_at), int(record.closed_at or 0)) for record in records),
                                  default=0)
        return len(records)

    def _event(self, name: str, *args) -> None:
        method = getattr(self._events, name, None) if self._events is not None else None
        if method is None:                       # a listener may lack an optional method (``refused``)
            return
        try:
            method(*args)
        except Exception:
            logger.exception("sandbox session listener %s failed", name)

    def _owner(self, hotkey: str) -> str:
        """Who a hotkey's load belongs to: its chain operator when known, else the hotkey itself."""
        try:
            operator = self._operator_of(hotkey) if self._operator_of is not None else None
        except Exception:
            logger.exception("operator lookup failed; the hotkey stands for itself")
            operator = None
        return str(operator) if operator else str(hotkey)

    def _own_load(self, hotkey: str, machine_ids, now: int, engagement: str,
                  by_share: bool = False) -> tuple[float, float]:
        """``(own, total)`` held sessions on these machines. ``own``: those of the hotkey's operator (all its hotkeys)
        less the refused open's own group (the sessions of the precommit ``engagement`` names: a servable task's
        machines host at least a whole group, so only the operator's OTHER load can make a saturation its own
        doing). ``total``: every session on them, never less than what their reports show (corpus sessions and
        anything else this book does not hold count too)."""
        share = getattr(self._fleet, "session_share", None) if by_share else None
        load = getattr(self._fleet, "load_share", None) if by_share else None
        owner = self._owner(hotkey)
        if share is None or load is None:            # by session counts (and always for a corpus session)
            held = self.book.held_on(machine_ids, now)
            own = sum(count for other, count in held.items() if self._owner(other) == owner)
            own -= self.book.held_on_precommit(machine_ids, now, engagement)
            reported = getattr(self._fleet, "reported_load", None)
            total = max(sum(held.values()), int(reported(machine_ids)) if reported is not None else 0)
            return max(0, own), total
        # By the share of its machine each session takes (its box's CPU or memory, the larger; a slot on a machine
        # placed by slots): one 4-CPU session weighs four 1-CPU ones.
        group = _rl_precommit_engagement(engagement)
        own = total = 0.0
        for record in self.book.holding_on(machine_ids, now):
            weight = self._weights.get(record.session_id)
            if weight is None:
                weight = float(share(record.machine_id, None) or 0.0)      # not issued by this process: a slot
            total += weight
            if self._owner(record.hotkey) == owner and (group is None or _rl_precommit_of(record) != group):
                own += weight
        return own, max(total, float(load(machine_ids)))

    def _over_entitlement(self, hotkey: str, now: int) -> bool:
        """Whether the hotkey's operator holds more sessions, fleet-wide, than its hotkeys' allocations entitle it
        to. Without an allocation (no RL quota) every operator is taken as over it (the share rule alone)."""
        if self._entitled is None:
            return True
        owner = self._owner(hotkey)
        hotkeys = {str(hotkey)} | {other for other in self.book.hotkeys() if self._owner(other) == owner}
        held = self.book.held_by(hotkeys, now)
        return held > sum(max(0, int(self._entitled(other))) for other in hotkeys)

    def _self_caused(self, hotkey: str, terms, refusal: Refusal) -> bool:
        """Whether the refused miner's own operator caused the saturation it was refused for. Only a saturation or
        a full queue can be; an open no machine could ever take, or an env no machine serves, never is. Both of:
        * its operator (by coldkey whatever the quota's keying: another hotkey of the same coldkey is the same
          miner) holds more sessions fleet-wide than its hotkeys' allocations entitle it to (within them, the
          capacity was ours to provide);
        * it held at least half of the sessions on the machines that could take this open (beyond the open's own
          group; the total never below the machines' reported load), or, for a full queue, at least half of the
          waiting tickets (beyond its own group's)."""
        if refusal.reason not in ("sandbox_saturated", "open_queue_full"):
            return False
        now = int(self._clock())
        if not self._over_entitlement(hotkey, now):
            return False
        servable = getattr(self._fleet, "servable_machines", None)
        machines = () if servable is None else servable(
            image=terms.image, env=terms.env, env_package=terms.env_package, budgets=terms.budgets,
            validity_s=self._validity(terms), now=now, env_options_sha256=terms.env_options_sha256,
            **_reserving(terms))
        if not machines and refusal.reason == "sandbox_saturated":
            logger.info("saturation of %s judged with no servable machine in view: a capacity verdict",
                        str(hotkey)[:12])
        own, total = self._own_load(hotkey, machines, now, terms.engagement,
                                    by_share=getattr(terms, "reserve", None) is not None)
        if total and 2 * own >= total:
            return True
        if refusal.reason == "open_queue_full" and self._queue is not None:
            tickets = self._queue.tickets()
            owner = self._owner(hotkey)
            group = _rl_precommit_engagement(terms.engagement)
            mine = sum(1 for key, engagement in tickets if self._owner(key) == owner
                       and (group is None or _rl_precommit_engagement(engagement) != group))
            return bool(tickets) and 2 * mine >= len(tickets)
        return False

    def _capacity_refused(self, hotkey: str, engagement: Mapping[str, Any], outcome, terms=None) -> None:
        if not isinstance(outcome, Refusal) or outcome.reason not in CAPACITY_REFUSALS:
            return
        if terms is not None:
            try:
                self_caused = self._self_caused(hotkey, terms, outcome)
            except Exception:
                self.attribution_failures += 1
                logger.exception("the saturation attribution failed; no capacity verdict")
                self_caused = True                  # never a neutral verdict we could not justify
            if self_caused:
                logger.info("capacity refusal %s of %s was its own operator's load: no capacity verdict",
                            outcome.reason, str(hotkey)[:12])
                return
        self._event("refused", hotkey, engagement, outcome)

    def _wake(self) -> None:
        if self._queue is not None:
            self._queue.notify()

    def _settled(self, records) -> None:
        """Sessions left the book's live states: the fleet drops its hold on their reservations."""
        settled = getattr(self._fleet, "note_settled", None)
        for record in records:
            if record is not None:
                self._weights.pop(record.session_id, None)
        if settled is None:
            return
        for record in records:
            if record is not None:
                try:
                    settled(record.session_id)
                except Exception:
                    logger.exception("the fleet failed to drop the hold of session %s", record.session_id)

    def _reported(self, session_id: str) -> bool:
        """Whether a machine report accepted since the restart listed this session live on its gateway
        (``Fleet.session_reported``, when the fleet has it): its miner held the token, so its open answer was
        not lost."""
        reported = getattr(self._fleet, "session_reported", None)
        return bool(reported(session_id)) if callable(reported) else False

    def _resent(self, hotkey: str, request_id: str, digest: str) -> Grant | Refusal | None:
        """The answer to a request already granted (the same token), or None."""
        existing = self.book.by_request(hotkey, request_id)
        if existing is None:
            return None
        if existing.engagement_sha256 != digest:
            return Refusal("request_conflict", {"session_id": existing.session_id})
        cached = self._tokens.get(existing.session_id)
        if cached is None or existing.state != LIVE:
            if (cached is None and existing.state == LIVE and existing.session_id in self._restored
                    and existing.issued_at >= self._last_activity - LOST_OPEN_WINDOW_S):
                # Our restart lost this open's answer (issued in the minute before the last activity that
                # preceded the crash; the miner resent it).
                self._lost.add(existing.session_id)
            return Refusal("request_reused", {"session_id": existing.session_id,
                                              "state": existing.state})
        token, address = cached
        return Grant(existing.session_id, token, address, existing.expires_at)

    def _refused_rate(self, hotkey: str, now: int) -> Refusal | None:
        recent = [t for t in self._refused_opens.get(hotkey, ()) if t > now - MINUTE]
        if recent:
            self._refused_opens[hotkey] = recent
        else:
            self._refused_opens.pop(hotkey, None)
        limit = int(self._policy.max_refused_opens_per_minute * self.book.cap_scale(hotkey))
        if len(recent) < limit:
            return None
        return Refusal("open_refused_rate", {"refused": len(recent), "max": limit},
                       retry_after=recent[len(recent) - limit] + MINUTE - now)

    async def open(self, *, hotkey: str, request_id: str,
                   engagement: Mapping[str, Any]) -> Grant | Refusal:
        """The ban check, the ledger read and the task resolution run BEFORE the issuer
        lock (an intake's claim never waits behind them); under it, the in-memory caps
        and the prompt's free slots (against the reservations made meanwhile) are
        re-checked, then the session is stored. When a session of the prompt changed
        state since the free-slot read (the book's per-prompt generation moved: a
        submission consumed a slot, a close freed one), that read is stale and the
        ledger is read once more under the lock (bounded by io_timeout_s; a failure is
        a retryable `ledger_unavailable`). Refused opens count toward a per-hotkey
        rate (`max_refused_opens_per_minute`), checked before any read."""
        if not isinstance(engagement, Mapping):
            return Refusal("engagement_kind_unsupported", {"kind": None})
        try:
            digest = engagement_digest(engagement)
        except (TypeError, ValueError):
            return Refusal("engagement_kind_unsupported", {"why": "not JSON"})
        now = int(self._clock())
        resent = self._resent(hotkey, request_id, digest)
        if resent is not None:
            return resent
        throttled = self._refused_rate(hotkey, now)
        if throttled is not None:
            return throttled
        outcome = await self._open(hotkey, request_id, digest, engagement)
        if isinstance(outcome, Refusal):
            if outcome.reason not in _NEVER_TALLIED and (
                    self._queue is None or outcome.reason not in _UNTALLIED_REFUSALS):
                self._refused_opens.setdefault(hotkey, []).append(now)
            return outcome
        if isinstance(outcome, Grant):           # the same request, granted meanwhile
            return outcome
        logger.info("sandbox session %s issued to %s on %s for %s", outcome.grant.session_id,
                    hotkey[:12], outcome.machine_id, outcome.engagement)
        return outcome.grant

    async def _open(self, hotkey: str, request_id: str, digest: str,
                    engagement: Mapping[str, Any]):
        now = int(self._clock())
        refusal = self.book.open_refusal(hotkey, now)
        if refusal is not None:
            self._event("open_refused", refusal.reason, None, None, None)
            return refusal
        if not self._fleet.directory_ready(now):
            refusal = self._directory_unavailable()
            self._event("open_refused", refusal.reason, None, None, None)
            return refusal
        book = self._engagements.get(engagement.get("kind"))
        if book is None:
            return Refusal("engagement_kind_unsupported", {"kind": engagement.get("kind")})
        # Taken before the slot read: any session of the prompt changing state after it
        # makes that read stale (a submission consumed a slot and ended a reservation).
        try:
            generation = self.book.generation(engagement.get("job_id"),
                                              engagement.get("prompt_index"))
        except TypeError:                                       # unhashable: terms refuses it
            generation = None
        terms = await book.terms(hotkey, engagement)            # reads: outside the lock
        if isinstance(terms, Refusal):
            self._event("open_refused", terms.reason, None, None, None)
            self._capacity_refused(hotkey, engagement, terms)
            return terms
        outcome = await self._issue(hotkey, request_id, digest, engagement, terms, generation)
        if isinstance(outcome, Refusal):
            self._event("open_refused", outcome.reason, getattr(terms, "env", None),
                        getattr(terms, "engagement", None), getattr(terms, "image", None))
        self._capacity_refused(hotkey, engagement, outcome, terms)
        return outcome

    async def _issue(self, hotkey: str, request_id: str, digest: str, engagement: Mapping[str, Any], terms,
                     generation):
        """Issue under the lock. Without a queue (corpus), no room is refused at once, exactly as before. With an
        open queue (RL): an open no machine could ever take is refused at once (``task_unplaceable``, final for
        the seed; one that cannot be judged now is refused retryably); one finding no room waits for a turn
        OUTSIDE the lock and tries again, until its deadline (``sandbox_saturated``); a new open joins rather than
        placing first only when a waiter of its own placement class waits. A turn first looks for room without
        the lock (and not at all when its room key already missed in this sweep); a head waiter that keeps
        missing gets a machine held for it, which nobody else takes meanwhile."""
        queue = self._queue
        if queue is None:
            async with self._lock:
                outcome = await self._issue_locked(hotkey, request_id, digest, engagement, terms, generation,
                                                   may_place=True)
            if outcome is _NO_CAPACITY:
                return Refusal("sandbox_capacity", {"image": terms.image},
                               retry_after=self._policy.retry_after_s)
            return outcome
        ticket, deadline, placed = None, 0.0, False
        cls = _placement_class(terms)
        try:
            while True:
                granted = ticket is not None and ticket.granted
                exclude = queue.held_for_others(ticket)
                room = (terms.env, terms.image, _reserve_of(terms), exclude)
                if granted and (queue.room_missed(room) or self._no_room_now(terms, exclude)):
                    outcome = _NO_CAPACITY                    # passed over without the lock: nothing freed for it
                else:
                    may_place = granted or (ticket is None and not queue.waiting_in(cls))
                    async with self._lock:
                        outcome = await self._issue_locked(hotkey, request_id, digest, engagement, terms,
                                                           generation, may_place=may_place, exclude=exclude)
                if outcome is _NEVER:
                    return Refusal("task_unplaceable", {"image": terms.image, "why": "no machine can serve it"})
                if outcome is not _NO_CAPACITY:
                    placed = isinstance(outcome, _Issued)
                    return outcome
                if ticket is None:
                    joined = queue.join(hotkey, cls=cls, engagement=terms.engagement)
                    if isinstance(joined, Refusal):
                        return joined
                    ticket, deadline = joined, queue.deadline()
                    if ticket.granted:
                        continue                                  # the turn was free: try at once
                elif ticket.granted:
                    self._missed(ticket, terms, room)             # pass the turn on; wait on a fresh event
                if not await queue.turn(ticket, deadline):
                    return Refusal("sandbox_saturated", {"waiting": queue.waiting,
                                                         "waited_s": round(queue.waited(ticket), 1)},
                                   retry_after=self._policy.retry_after_s)
        finally:
            if ticket is not None:
                queue.done(ticket, placed=placed)               # a cancelled request hands its turn on too

    def _validity(self, terms) -> int:
        return token_validity_s(self._policy.open_window_s, int(terms.budgets["wall_s"]), terms.prepare_timeout_s)

    def _no_room_now(self, terms, exclude: frozenset) -> bool:
        """Whether a turn can be passed over without the lock: no machine has room for it now, and some machine
        surely could take it later (anything else is answered under the lock)."""
        now = int(self._clock())
        place = dict(image=terms.image, env=terms.env, env_package=terms.env_package, budgets=terms.budgets,
                     validity_s=self._validity(terms), now=now, env_options_sha256=terms.env_options_sha256,
                     **_reserving(terms))
        if self._fleet.pick(**place, **({"exclude": exclude} if exclude else {})) is not None:
            return False
        return self._fleet.could_place(**place) is True

    def _missed(self, ticket, terms, room) -> None:
        queue = self._queue
        if queue.should_hold(ticket):
            nearest = getattr(self._fleet, "nearest", None)
            if nearest is not None:
                queue.hold(ticket, nearest(image=terms.image, env=terms.env, env_package=terms.env_package,
                                           budgets=terms.budgets, validity_s=self._validity(terms),
                                           now=int(self._clock()), env_options_sha256=terms.env_options_sha256,
                                           **_reserving(terms)))
        queue.missed(ticket, room=room)

    async def _issue_locked(self, hotkey: str, request_id: str, digest: str, engagement: Mapping[str, Any],
                            terms, generation, *, may_place: bool, exclude: frozenset = frozenset()):
        """The checks and the issue under the issuer lock (the caller holds it): a Grant (resent), an
        ``_Issued``, a Refusal, ``_NO_CAPACITY`` or ``_NEVER``."""
        now = int(self._clock())
        resent = self._resent(hotkey, request_id, digest)    # the same request, raced
        if resent is not None:
            return resent
        refusal = (self.book.open_refusal(hotkey, now)
                   or self.book.engagement_refusal(hotkey, terms.job_id,
                                                   terms.prompt_index, now))
        if refusal is not None:
            return refusal
        if terms.still_valid is not None:
            refusal = terms.still_valid()
            if refusal is not None:
                return refusal
        if terms.exclusive and self.book.engagement_held(terms.engagement):
            return Refusal("engagement_taken", {"engagement": terms.engagement})
        if terms.slots_remaining is not None:
            remaining = terms.slots_remaining
            moved = generation != self.book.generation(terms.job_id, terms.prompt_index)
            if moved and terms.refresh_slots is not None:
                # One bounded read under the lock (a claim waits at most claim_wait_s).
                remaining = await terms.refresh_slots()
                if isinstance(remaining, Refusal):
                    return remaining
                if remaining is None:
                    return Refusal("job_complete", {"job_id": terms.job_id})
                now = int(self._clock())
            reserved = self.book.reserved(terms.job_id, terms.prompt_index, now)
            if remaining - reserved <= 0:
                return Refusal("prompt_unavailable", {
                    "prompt_index": terms.prompt_index,
                    "slots_remaining": remaining, "reserved": reserved})
        validity = token_validity_s(self._policy.open_window_s, int(terms.budgets["wall_s"]),
                                    terms.prepare_timeout_s)
        placement = None
        if may_place:
            placement = self._fleet.pick(image=terms.image, env=terms.env,
                                         env_package=terms.env_package, budgets=terms.budgets,
                                         validity_s=validity, now=now,
                                         env_options_sha256=terms.env_options_sha256,
                                         **({"exclude": exclude} if exclude else {}), **_reserving(terms))
        if placement is None:
            if self._queue is not None:
                verdict = self._fleet.could_place(
                    image=terms.image, env=terms.env, env_package=terms.env_package, budgets=terms.budgets,
                    validity_s=validity, now=now, env_options_sha256=terms.env_options_sha256,
                    **_reserving(terms))
                if verdict is None:                  # our outage or a start: retryable, never final for the seed
                    return Refusal("directory_unavailable", {"why": "the sandbox fleet cannot be judged now"},
                                   retry_after=self._policy.retry_after_s)
                if verdict is False:
                    unserved = self._unserved(terms, now)
                    return _NEVER if unserved is None else unserved
            return _NO_CAPACITY
        session_id = self._new_id()
        claims = SessionClaims(
            session_id=session_id, hotkey=hotkey, engagement=terms.engagement, env=terms.env,
            split=terms.split, index=terms.index, image=terms.image,
            checkpoint=terms.checkpoint, machine_id=placement.machine_id, issued_at=now,
            expires_at=now + validity, budgets=Budgets(**terms.budgets))
        token = issue_session_token(claims, self._signer)
        record = SessionRecord(
            session_id=session_id, hotkey=hotkey, request_id=request_id,
            engagement_sha256=digest, kind=str(engagement.get("kind")),
            engagement=terms.engagement, env=terms.env,
            split=terms.split, index=terms.index, checkpoint=terms.checkpoint,
            job_id=terms.job_id, prompt_index=terms.prompt_index,
            machine_id=placement.machine_id, issued_at=now, expires_at=claims.expires_at,
            token_sha256=token_sha256(token))
        try:
            await asyncio.wait_for(self._store.create(record.to_document()),
                                   self._policy.io_timeout_s)
        except Exception as exc:
            logger.warning("sandbox session %s not stored (%s): not issued", session_id,
                           type(exc).__name__)
            return Refusal("store_unavailable", {}, retry_after=self._policy.retry_after_s)
        self.book.add(record)
        reserve = getattr(terms, "reserve", None)
        if reserve is None:
            self._fleet.note_issued(placement.machine_id, now)
        else:
            from reliquary.sandbox.rl_capacity import PRESENT_WINDOW_S   # RL only: corpus never loads it

            # held until a report lists it, at most the token's present window (the gateway refuses it later)
            self._fleet.note_issued(placement.machine_id, now, reserve=reserve,
                                    hold_s=min(validity, PRESENT_WINDOW_S), session_id=session_id)
            share = getattr(self._fleet, "session_share", None)
            if share is not None:
                try:
                    weight = share(placement.machine_id, reserve)
                except Exception:
                    logger.exception("the share of session %s on its machine is unknown", session_id)
                    weight = None
                if weight is not None:
                    self._weights[session_id] = float(weight)
                if len(self._weights) > 2 * len(self.book._sessions) + 1024:     # a record left without settling
                    self._weights = {sid: w for sid, w in self._weights.items() if self.book.get(sid) is not None}
        self._tokens[session_id] = (token, placement.address)
        self._event("opened", record)
        return _Issued(Grant(session_id, token, placement.address, claims.expires_at),
                       placement.machine_id, terms.engagement)

    def unserved_recovered(self, env: str) -> None:
        """A machine serves ``env`` again: its next loss is ALERTed again."""
        self._unserved_alerted = {key for key in self._unserved_alerted if key[0] != env}

    def _unserved(self, terms, now: int) -> Refusal | None:
        """An open no machine could take because no fresh machine serves its env at its exact package and options
        (the fleet lost the env, a rolling upgrade, options drift): our state, refused retryably
        (``environment_unserved``, an ALERT once per identity), never final for the seed. None when some machine
        serves the env: then this task itself fits nowhere (``task_unplaceable``)."""
        serving = getattr(self._fleet, "serving", None)
        if serving is None:
            return None
        if serving(terms.env, terms.env_package, env_options_sha256=terms.env_options_sha256, now=now):
            return None
        key = (terms.env, terms.env_package, terms.env_options_sha256)
        if key not in self._unserved_alerted:
            self._unserved_alerted.add(key)
            reported = getattr(self._fleet, "reported_packages", lambda env: [])(terms.env)
            logger.error("ALERT no sandbox machine serves %s at %s (options sha256 %s); machines report %s: its "
                         "opens are refused (environment_unserved, retryable) until one does", terms.env,
                         terms.env_package, terms.env_options_sha256, ", ".join(reported) or "nothing")
        return Refusal("environment_unserved", {"environment": terms.env}, retry_after=self._policy.retry_after_s)

    @staticmethod
    def _state_of(record: SessionRecord) -> dict:
        return {"session_id": record.session_id, "state": record.state,
                "status": record.closed_status}

    async def close(self, *, hotkey: str, session_id: str, reason: str,
                    transcript: Mapping[str, Any] | None,
                    open_refusal: Mapping[str, Any] | None = None) -> dict | Refusal:
        """Settle a live session from its machine's final (or a failed open), or
        withdraw a live or graded-closed one, and return the state the session is
        actually in afterwards (a claim or submission that landed meanwhile wins). The
        transcript is verified outside the lock.

        With ``record_faults`` (the RL issuer), a fault that is not the miner's is written on the record
        before it is persisted: a failed open with the gateway's verified signed refusal
        (``open_refused:<reason>``; a receipt that cannot be judged now is refused ``directory_unavailable``,
        retryably), a failed open whose answer our restart lost (``validator_restarted``), a proven runtime
        start failure (``runtime_exec_failed``) and a grading-pool abort caused by others (``grading_pool``).
        A withdrawal never records one."""
        withdraw = reason == WITHDRAW
        closable = (LIVE, CLOSED_GRADED) if withdraw else (LIVE,)
        async with self._lock:
            now = int(self._clock())
            record = self.book.get(session_id)
            if record is None or record.hotkey != hotkey:
                return Refusal("session_unknown", {"session_id": session_id})
            if record.state not in closable or self.book._frozen(record, now, "close"):
                return self._state_of(record)
            if reason == "open_failed":
                if transcript is not None:
                    return Refusal("transcript_invalid", {"why": "a failed open has no transcript"})
            else:
                if transcript is None:
                    return Refusal("transcript_invalid",
                                   {"why": f"a {reason} close carries its transcript"})
                if not self._fleet.directory_ready(now):
                    return self._directory_unavailable()
                directory = self._fleet.directory()
                expected = Expected(hotkey=record.hotkey, engagement=record.engagement,
                                    env=record.env, split=record.split, index=record.index,
                                    checkpoint=record.checkpoint,
                                    seen_session_ids=self.book.submitted_ids(),
                                    require_graded=False)
        final, fault = None, None
        if reason == "open_failed":
            status = "open_failed"
            if self._record_faults:
                verdict = (None if open_refusal is None or self._open_refusal_check is None
                           else self._open_refusal_check(open_refusal, record))
                if open_refusal is not None and self._open_refusal_check is not None and verdict is None:
                    return self._directory_unavailable()     # cannot be judged now: resent, never lost
                if verdict:
                    fault = _open_refused_fault(open_refusal)
                elif session_id in self._lost and not self._reported(session_id):
                    fault = "validator_restarted"    # a machine that listed it says the miner had the token
        else:
            result = await asyncio.to_thread(verify_transcript, transcript, directory,
                                             self._token_verifier, expected)
            if not result.ok or result.claims.session_id != session_id:
                return Refusal("transcript_invalid", {
                    "reasons": [r.value for r in result.reasons] or ["session_mismatch"]})
            status = WITHDRAWN if withdraw else result.final.status
            final = result.final.to_dict()
            if self._record_faults and not withdraw:
                fault = runtime_start_fault(final) or pool_abort_fault(final)
        state = (CLOSED if withdraw else ABORTED if status == STATUS_ABORTED
                 else CLOSED_GRADED if status == STATUS_GRADED else CLOSED)
        async with self._lock:
            now = int(self._clock())
            current = self.book.get(session_id)
            if (current is None or current.state not in closable
                    or self.book._frozen(current, now, "close")):
                return (self._state_of(current) if current is not None
                        else Refusal("session_unknown", {"session_id": session_id}))
            settled = self.book.settle(session_id, state, now=now, status=status)
            self._tokens.pop(session_id, None)
            if settled is not None and fault is not None:
                settled.fault = fault                    # before the record is persisted
            if settled is not None and self._record_faults and settled.ended_at is None and final is not None:
                settled.ended_at = now                   # its box's end, never moved by a submission or withdrawal
                if not withdraw:                         # only a session that reached its machine's own close:
                    settled.ended_status = str(final.get("status") or status)[:64]   # never one the miner refused
        if settled is not None:
            self._event("closed", settled, final)
            self._settled((settled,))
            self._wake()
        await self._persist(settled)
        logger.info("sandbox session %s of %s closed: %s", session_id, hotkey[:12], status)
        return self._state_of(settled)

    async def claim(self, session_id: str, *, hotkey: str,
                    received: float | None = None) -> Refusal | None:
        """The intake's hold on a session it is about to pay, taken BEFORE its ledger
        write: None, or why not. `received` is when the validator received the
        submission (its own clock). Only a `live` or `closed_graded` session of this
        hotkey, or a `lapsed` one received by its deadline (it lapsed during the check);
        received after the deadline is `session_expired`. `session_claimed` is retryable:
        another submission of it is in flight; so is `session_busy`, when the issuer lock
        is not free within `claim_wait_s`. The claim ends with `submitted` or
        `release_claim`, or goes stale after `claim_ttl_s`."""
        try:
            await asyncio.wait_for(self._lock.acquire(), self._policy.claim_wait_s)
        except TimeoutError:
            logger.warning("sandbox session %s: claim waited %d s for the issuer lock; busy",
                           session_id, self._policy.claim_wait_s)
            return Refusal("session_busy", {"session_id": session_id},
                           retry_after=self._policy.retry_after_s)
        try:
            return self._claim_locked(session_id, hotkey, received, int(self._clock()))
        finally:
            self._lock.release()

    def _claim_locked(self, session_id: str, hotkey: str, received: float | None,
                      now: int) -> Refusal | None:
        """``claim``'s checks and hold; the caller holds the issuer lock."""
        record = self.book.get(session_id)
        if record is None or record.hotkey != hotkey:
            return Refusal("session_unknown", {"session_id": session_id})
        if record.state == SUBMITTED:
            return Refusal("session_submitted", {"session_id": session_id})
        deadline = record.expires_at + GRADING_GRACE_S
        if received is not None and received > deadline:
            return Refusal("session_expired", {"session_id": session_id,
                                               "deadline": deadline})
        on_time_lapse = record.state == LAPSED and received is not None
        if not (session_submittable(record.state) or on_time_lapse):
            return Refusal("session_not_submittable", {"session_id": session_id,
                                                       "state": record.state})
        if not self.book.claim(session_id, now):
            return Refusal("session_claimed", {"session_id": session_id},
                           retry_after=self._policy.retry_after_s)
        return None

    async def claim_all(self, session_ids: Sequence[str], *, hotkey: str, received: float | None = None,
                        precommit_sha256: str | None = None) -> tuple[str, Refusal] | None:
        """An episode group's sessions, all or none, under ONE hold of the issuer lock (no
        close, drain, lapse or other claim interleaves): None when every one is now claimed, else the
        first session refused and why (each session ``claim`` checks), with nothing left claimed by
        this call. A session named twice is refused (``session_claimed``) on its second mention.

        ``precommit_sha256`` (at most ONE paid group per precommit): the precommit is taken with
        its sessions. Any OTHER session of that precommit (an RL engagement naming it) already
        ``submitted`` refuses the group (``precommit_submitted``: another group of it was paid, even on
        disjoint seeds); one freshly claimed by a group in flight refuses it retryably
        (``precommit_claimed``). Durable: submitted sessions are persisted and restored."""
        ids = [str(session_id) for session_id in session_ids]
        first = ids[0] if ids else ""
        try:
            await asyncio.wait_for(self._lock.acquire(), self._policy.claim_wait_s)
        except TimeoutError:
            logger.warning("sandbox sessions of %s: a group claim waited %d s for the issuer lock; busy",
                           hotkey[:12], self._policy.claim_wait_s)
            return first, Refusal("session_busy", {"session_id": first},
                                  retry_after=self._policy.retry_after_s)
        try:
            now = int(self._clock())
            if precommit_sha256 is not None:
                taken = self._precommit_taken(precommit_sha256, set(ids), now, hotkey)
                if taken is not None:
                    return taken
            held: list[str] = []
            for session_id in ids:
                refusal = self._claim_locked(session_id, hotkey, received, now)
                if refusal is not None:
                    for done in held:
                        self.book.release_claim(done)
                    return session_id, refusal
                held.append(session_id)
            return None
        finally:
            self._lock.release()

    def _precommit_taken(self, precommit_sha256: str, own: set[str], now: int,
                         hotkey: str) -> tuple[str, Refusal] | None:
        """Another group's hold on this precommit (caller holds the issuer lock), or None. Only this
        hotkey's sessions count (an RL session opens only on its opener's own precommit, so no other
        hotkey's session can name it; never let one block it)."""
        for record in self.book.of_precommit(precommit_sha256):
            if record.session_id in own or record.hotkey != hotkey:
                continue
            if record.state == SUBMITTED:
                return record.session_id, Refusal("precommit_submitted",
                                                  {"precommit_sha256": precommit_sha256})
            if self.book.claim_fresh(record.session_id, now):
                return record.session_id, Refusal("precommit_claimed",
                                                  {"precommit_sha256": precommit_sha256},
                                                  retry_after=self._policy.retry_after_s)
        return None

    async def release_claim(self, session_id: str) -> None:
        """The claimed submission was not accepted (refused, contended, or its write
        failed): the session is as it was before the claim."""
        async with self._lock:
            self.book.release_claim(session_id)

    async def submitted(self, session_id: str) -> None:
        """The intake accepted this session's submission (called after the ledger write,
        so the slot is never counted twice): its claim and its reservation end."""
        async with self._lock:
            claimed = self.book.is_claimed(session_id)
            self.book.release_claim(session_id)
            current = self.book.get(session_id)
            if current is not None and current.state == LAPSED and not claimed:
                return                 # a lapsed session is paid only through a claim
            record = self.book.settle(session_id, SUBMITTED, now=int(self._clock()),
                                      status=STATUS_GRADED)
            self._tokens.pop(session_id, None)
        if record is not None:
            await self._persist(record)

    async def persist_submitted(self, session_ids: Sequence[str]) -> int:
        """Before the batcher takes an episode group: each claimed session's ``submitted``
        record is written (together), the book left as it is. Returns how many are stored (a session
        stored so by an earlier attempt counts without a write). One is enough to keep the precommit
        taken across a restart (``claim_all``'s precommit check). If the batcher then refuses the
        group, the book releases the claims while the store keeps ``submitted``: a restart reads those
        sessions as paid (the miner may lose that group), never the reverse. A caller that stops
        waiting (its admission deadline) does not cut the writes: each one still notes its session
        stored when it lands, so a retry counts it rather than rewriting it."""
        documents, stored = {}, 0
        async with self._lock:
            now = int(self._clock())
            for session_id in dict.fromkeys(session_ids):
                record = self.book.get(session_id)
                if record is None or not self.book.is_claimed(session_id):
                    continue
                if session_id in self._stored_submitted:
                    stored += 1
                elif session_transition_allowed(record.state, SUBMITTED):
                    documents[session_id] = replace(record, state=SUBMITTED, closed_status=STATUS_GRADED,
                                                    closed_at=now)
        writes = [self._spawn(self._persist_submitted(documents[i])) for i in documents]
        written = await asyncio.shield(asyncio.gather(*writes)) if writes else []
        return stored + sum(1 for ok in written if ok)

    async def _persist_submitted(self, record: SessionRecord) -> bool:
        ok = await self._persist(record)
        if ok:
            self._stored_submitted.add(record.session_id)
        return ok

    def _spawn(self, coroutine) -> asyncio.Task:
        """A write as a task of its own, kept until it ends (``drain`` waits for it at shutdown)."""
        task = asyncio.get_running_loop().create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def submitted_all(self, session_ids: Sequence[str]) -> None:
        """An accepted episode group's sessions, all moved to ``submitted`` under ONE hold of the
        issuer lock (no reader sees part of the group paid), then every record not already stored by
        ``persist_submitted`` written before this returns (the writes run together). A restart reads
        them back, so the precommit stays taken (``claim_all``)."""
        records = []
        async with self._lock:
            now = int(self._clock())
            for session_id in session_ids:
                claimed = self.book.is_claimed(session_id)
                self.book.release_claim(session_id)
                current = self.book.get(session_id)
                if current is not None and current.state == LAPSED and not claimed:
                    continue           # a lapsed session is paid only through a claim
                record = self.book.settle(session_id, SUBMITTED, now=now, status=STATUS_GRADED)
                self._tokens.pop(session_id, None)
                if record is not None and session_id not in self._stored_submitted:
                    records.append(record)
                self._stored_submitted.discard(session_id)
        await asyncio.gather(*(self._persist(record) for record in records))

    async def hand_back(self, precommit_sha256: str, *, hotkey: str) -> tuple[str, ...]:
        """The paid group of this precommit was not judged by the proof, for a reason of the
        validator's own. Its ``submitted`` sessions stay ``submitted`` (never paid, never claimed
        again: the precommit stays taken, no same-window retry) and are marked handed back (closed
        status), in the book and the store, so that the yield accounting and the quota do not count them against
        the miner (the hourly open cap still does: they were opened).
        Returns the sessions marked now (none twice)."""
        marked: list[SessionRecord] = []
        async with self._lock:
            now = int(self._clock())
            for record in self.book.of_precommit(precommit_sha256):
                if record.hotkey != hotkey or record.state != SUBMITTED or record.closed_status == HANDED_BACK:
                    continue
                record.closed_status, record.closed_at = HANDED_BACK, now
                marked.append(record)
        writes = [self._spawn(self._persist(replace(record))) for record in marked]
        if writes:
            await asyncio.shield(asyncio.gather(*writes))
        return tuple(record.session_id for record in marked)

    def void_machine(self, machine_id: str) -> None:
        records = self.book.void_machine(machine_id, int(self._clock()))
        for record in records:
            self._tokens.pop(record.session_id, None)
            self._stored_submitted.discard(record.session_id)
            task = asyncio.get_running_loop().create_task(self._persist(record))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        if records:
            self._settled(records)
            self._wake()
            logger.warning("machine %s drained: %d live sessions voided, no fault to their miners",
                           machine_id, len(records))

    async def drain(self, timeout: float) -> int:
        """Wait up to `timeout` seconds for the state writes started in the background
        (a drained machine's voids); cancel what is left. Returns how many were cut."""
        pending = [task for task in self._tasks if not task.done()]
        if not pending:
            return 0
        _, late = await asyncio.wait(pending, timeout=timeout)
        for task in late:
            task.cancel()
        if late:
            await asyncio.gather(*late, return_exceptions=True)
            logger.error("ALERT %d sandbox session writes cut at shutdown; a restart reads "
                         "their older stored state", len(late))
        return len(late)

    async def maintain(self) -> None:
        async with self._lock:
            now = int(self._clock())
            lapsed = self.book.lapse(now)
            for record in lapsed:
                self._tokens.pop(record.session_id, None)
                self._stored_submitted.discard(record.session_id)
            self.book.prune(now)
            self._stored_submitted = {i for i in self._stored_submitted if self.book.get(i) is not None}
            for hotkey in [h for h, times in self._refused_opens.items()
                           if not times or times[-1] <= now - MINUTE]:
                del self._refused_opens[hotkey]
        if lapsed:
            self._settled(lapsed)
            self._wake()
        for record in lapsed:
            await self._persist(record)

    async def maintain_forever(self, every_s: float = 60.0) -> None:
        while True:
            try:
                await self.maintain()
            except Exception:
                logger.exception("sandbox session maintenance failed")
            await asyncio.sleep(every_s)

    def _directory_unavailable(self) -> Refusal:
        return Refusal("directory_unavailable", {"why": "the machine directory is stale"},
                       retry_after=self._policy.retry_after_s)

    async def _persist(self, record: SessionRecord | None) -> bool:
        """Write the record's state (a snapshot taken now). A refused transition means a
        later state is already stored: nothing to do. Other failures are retried with
        backoff; the last one alerts (the in-memory state stays authoritative until a
        restart, which would read the older stored state). True only when this state is stored."""
        if record is None:
            return False
        document = record.to_document()
        for attempt in range(PERSIST_ATTEMPTS):
            if attempt:
                await _sleep(PERSIST_BACKOFF_S * 2 ** (attempt - 1))
            try:
                await asyncio.wait_for(self._store.update(document), self._policy.io_timeout_s)
                return True
            except SessionStoreConflict as exc:
                logger.warning("sandbox session %s state %s not stored: %s",
                               record.session_id, document["state"], exc)
                return False
            except Exception as exc:
                failure = type(exc).__name__
        logger.error("ALERT sandbox session %s state %s not stored after %d attempts (%s)",
                     record.session_id, document["state"], PERSIST_ATTEMPTS, failure)
        return False


__all__ = ["ABORTED", "CAPACITY_REFUSALS", "CLOSED", "CLOSED_GRADED", "DEFAULT_PREPARE_TIMEOUT_S",
           "GATEWAY_PREPARE_SLACK_S", "GRADING_POOL_ABORT", "HANDED_BACK", "LAPSED", "LIVE", "LOST_OPEN_WINDOW_S", "OPEN_TRANSIT_S",
           "SUBMITTED", "VOIDED", "CorpusEngagements", "EngagementBook", "EngagementTerms", "FanOutEvents", "Grant",
           "JobNotReady", "Refusal", "RlPrecommitEngagements", "SandboxPolicy", "SessionBook", "SessionEvents",
           "SessionIssuer", "SessionRecord", "SignedJobView", "engagement_digest", "pool_abort_fault",
           "runtime_start_fault", "session_submittable", "token_validity_s"]
