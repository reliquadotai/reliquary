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
  the slot this episode may yet consume. It ends `submitted` or `lapsed`;
* `submitted`: its submission was accepted. Terminal and dominant: it overrides
  `closed`, `voided`, `lapsed` and `closed_graded`, since the ledger already consumed
  the slot (a late graded submission of a voided session is admitted when its prompt
  has a free slot);
* `closed`: an unpaid final (expired, box_failed, budget_exhausted including
  `transcript_bytes`) or a failed open;
* `aborted`: void, refunded from the open rate, counted against the aborted cap;
* `voided`: its machine was drained (no fault; not counted in the open rate);
* `lapsed`: past `expires_at + GRADING_GRACE_S`, when submissions are refused anyway.
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
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Protocol

from reliquary_sandbox.attest import (
    GRADING_GRACE_S, STATUS_ABORTED, STATUS_GRADED, Budgets, Expected, SessionClaims,
    issue_session_token, token_sha256, verify_transcript,
)

from reliquary.corpus.job import is_signed_sandbox, sandbox_split
from reliquary.corpus.signed_reasons import corpus_engagement
from reliquary.infrastructure import sandbox_store as _store
from reliquary.infrastructure.sandbox_store import (
    SESSION_TRANSITIONS, SessionStoreConflict, session_transition_allowed,
)
from reliquary.sandbox.tasks import ResolvedTask

logger = logging.getLogger(__name__)

SESSION_SCHEMA = "reliquary/sandbox-session/v1"
LIVE, CLOSED_GRADED, SUBMITTED = _store.SESSION_LIVE, _store.SESSION_CLOSED_GRADED, _store.SESSION_SUBMITTED
CLOSED, ABORTED, VOIDED, LAPSED = (_store.SESSION_CLOSED, _store.SESSION_ABORTED,
                                   _store.SESSION_VOIDED, _store.SESSION_LAPSED)
STATES = frozenset(SESSION_TRANSITIONS)
HOLDING = frozenset({LIVE, CLOSED_GRADED})
HOUR, DAY = 3600, 86400
PERSIST_ATTEMPTS = 4
PERSIST_BACKOFF_S = 0.5
_sleep = asyncio.sleep


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

    _ENV = {"max_live_per_hotkey": "RELIQUARY_SANDBOX_MAX_LIVE_PER_HOTKEY",
            "max_live_per_hotkey_job": "RELIQUARY_SANDBOX_MAX_LIVE_PER_HOTKEY_JOB",
            "max_opens_per_hour": "RELIQUARY_SANDBOX_MAX_OPENS_PER_HOUR",
            "max_aborted_per_day": "RELIQUARY_SANDBOX_MAX_ABORTED_PER_DAY",
            "open_window_s": "RELIQUARY_SANDBOX_OPEN_WINDOW_S",
            "request_skew_s": "RELIQUARY_SANDBOX_REQUEST_SKEW_S",
            "retry_after_s": "RELIQUARY_SANDBOX_RETRY_AFTER_S",
            "io_timeout_s": "RELIQUARY_SANDBOX_IO_TIMEOUT_S"}

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
    prompt_index: int | None = None


class EngagementBook(Protocol):
    kind: str

    async def terms(self, hotkey: str, engagement: Mapping[str, Any]) -> EngagementTerms | Refusal: ...


@dataclass(frozen=True)
class SignedJobView:
    """What the corpus engagement book needs of a served signed job."""

    job: Any
    resolve_task: Callable[[int], Awaitable[ResolvedTask]] | None
    slots_remaining: Callable[[int], Awaitable[int | None]] | None
    is_banned: Callable[[str], Awaitable[bool]] | None = None


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
}


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

    def to_document(self) -> dict:
        return {"schema": SESSION_SCHEMA, **asdict(self)}

    @classmethod
    def from_document(cls, document: Any) -> SessionRecord:
        """Every field is type-checked: a malformed document raises ValueError."""
        if not isinstance(document, Mapping) or document.get("schema") != SESSION_SCHEMA:
            raise ValueError("not a session document")
        values = {}
        for f in fields(cls):
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
        except TimeoutError:
            return Refusal("ledger_unavailable", {"job_id": job_id}, retry_after=retry)
        if remaining is None:
            return Refusal("job_complete", {"job_id": job_id})
        reserved = self._book.reserved(job.job_id, index, int(self._clock()))
        if remaining - reserved <= 0:
            return Refusal("prompt_unavailable", {"prompt_index": index,
                                                  "slots_remaining": remaining, "reserved": reserved})
        try:
            task = await asyncio.wait_for(view.resolve_task(index), timeout)
        except TimeoutError:
            return Refusal("task_unavailable", {"prompt_index": index}, retry_after=retry)
        spec = job.episode.sandbox
        budgets = spec.budgets.to_contract()
        for name, value in task.limits.items():
            if name in budgets:
                budgets[name] = max(budgets[name], int(value))
        return EngagementTerms(engagement=corpus_engagement(job.job_id, index), env=spec.env,
                               split=sandbox_split(job.episode), index=index,
                               checkpoint=job.checkpoint_sha256, image=task.image,
                               env_package=spec.env_package, budgets=budgets,
                               job_id=job.job_id, prompt_index=index)


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

    def add(self, record: SessionRecord) -> None:
        self._sessions[record.session_id] = record
        self._requests[(record.hotkey, record.request_id)] = record.session_id

    def restore(self, records) -> None:
        for record in records:
            self.add(record)

    def get(self, session_id: str) -> SessionRecord | None:
        return self._sessions.get(session_id)

    def by_request(self, hotkey: str, request_id: str) -> SessionRecord | None:
        session_id = self._requests.get((hotkey, request_id))
        return None if session_id is None else self._sessions.get(session_id)

    @staticmethod
    def _holds(record: SessionRecord, now: int) -> bool:
        return record.state in HOLDING and now <= record.expires_at + GRADING_GRACE_S

    def reserved(self, job_id: str, prompt_index: int, now: int) -> int:
        return sum(1 for r in self._sessions.values()
                   if r.job_id == job_id and r.prompt_index == prompt_index and self._holds(r, now))

    def submitted_ids(self) -> frozenset[str]:
        return frozenset(s for s, r in self._sessions.items() if r.state == SUBMITTED)

    def open_refusal(self, hotkey: str, now: int) -> Refusal | None:
        mine = [r for r in self._sessions.values() if r.hotkey == hotkey]
        policy = self.policy
        live = sum(1 for r in mine if self._holds(r, now))
        if live >= policy.max_live_per_hotkey:
            return Refusal("live_cap", {"live": live, "max": policy.max_live_per_hotkey})
        opens = sum(1 for r in mine
                    if r.issued_at > now - HOUR and r.state not in (ABORTED, VOIDED))
        if opens >= policy.max_opens_per_hour:
            return Refusal("open_rate_cap", {"opens": opens, "max": policy.max_opens_per_hour})
        aborted = sum(1 for r in mine if r.state == ABORTED and (r.closed_at or 0) > now - DAY)
        if aborted >= policy.max_aborted_per_day:
            return Refusal("aborted_cap", {"aborted": aborted, "max": policy.max_aborted_per_day})
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
        return record

    def lapse(self, now: int) -> list[SessionRecord]:
        lapsed = [r for r in self._sessions.values()
                  if r.state in HOLDING and now > r.expires_at + GRADING_GRACE_S]
        for record in lapsed:
            self.settle(record.session_id, LAPSED, now=now, status=record.closed_status)
        return lapsed

    def void_machine(self, machine_id: str, now: int) -> list[SessionRecord]:
        """Only `live` sessions: a `closed_graded` one already holds its signed transcript."""
        voided = [r for r in self._sessions.values()
                  if r.machine_id == machine_id and r.state == LIVE and self._holds(r, now)]
        for record in voided:
            self.settle(record.session_id, VOIDED, now=now, status="machine_drained")
        return voided

    def prune(self, now: int) -> None:
        old = [s for s, r in self._sessions.items()
               if r.state not in HOLDING and (r.closed_at or r.issued_at) < now - DAY - HOUR]
        for session_id in old:
            record = self._sessions.pop(session_id)
            self._requests.pop((record.hotkey, record.request_id), None)


class SessionIssuer:
    def __init__(self, *, book: SessionBook, store, fleet, signer, token_verifier,
                 engagements: Mapping[str, EngagementBook], policy: SandboxPolicy,
                 clock: Callable[[], float] = time.time,
                 new_session_id: Callable[[], str] = lambda: uuid.uuid4().hex) -> None:
        self.book = book
        self._store, self._fleet, self._signer = store, fleet, signer
        self._token_verifier = token_verifier
        self._engagements = dict(engagements)
        self._policy, self._clock, self._new_id = policy, clock, new_session_id
        self._tokens: dict[str, tuple[dict, str]] = {}
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

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
        return len(records)

    async def open(self, *, hotkey: str, request_id: str,
                   engagement: Mapping[str, Any]) -> Grant | Refusal:
        if not isinstance(engagement, Mapping):
            return Refusal("engagement_kind_unsupported", {"kind": None})
        try:
            digest = engagement_digest(engagement)
        except (TypeError, ValueError):
            return Refusal("engagement_kind_unsupported", {"why": "not JSON"})
        async with self._lock:
            now = int(self._clock())
            existing = self.book.by_request(hotkey, request_id)
            if existing is not None:
                if existing.engagement_sha256 != digest:
                    return Refusal("request_conflict", {"session_id": existing.session_id})
                cached = self._tokens.get(existing.session_id)
                if cached is None or existing.state != LIVE:
                    return Refusal("request_reused", {"session_id": existing.session_id,
                                                      "state": existing.state})
                token, address = cached
                return Grant(existing.session_id, token, address, existing.expires_at)
            refusal = self.book.open_refusal(hotkey, now)
            if refusal is not None:
                return refusal
            if not self._fleet.directory_ready(now):
                return self._directory_unavailable()
            book = self._engagements.get(engagement.get("kind"))
            if book is None:
                return Refusal("engagement_kind_unsupported", {"kind": engagement.get("kind")})
            terms = await book.terms(hotkey, engagement)
            if isinstance(terms, Refusal):
                return terms
            refusal = self.book.engagement_refusal(hotkey, terms.job_id, terms.prompt_index, now)
            if refusal is not None:
                return refusal
            validity = self._policy.open_window_s + int(terms.budgets["wall_s"])
            placement = self._fleet.pick(image=terms.image, env=terms.env,
                                         env_package=terms.env_package, budgets=terms.budgets,
                                         validity_s=validity, now=now)
            if placement is None:
                return Refusal("sandbox_capacity", {"image": terms.image},
                               retry_after=self._policy.retry_after_s)
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
            self._fleet.note_issued(placement.machine_id, now)
            self._tokens[session_id] = (token, placement.address)
        logger.info("sandbox session %s issued to %s on %s for %s", session_id, hotkey[:12],
                    placement.machine_id, terms.engagement)
        return Grant(session_id, token, placement.address, claims.expires_at)

    @staticmethod
    def _state_of(record: SessionRecord) -> dict:
        return {"session_id": record.session_id, "state": record.state,
                "status": record.closed_status}

    async def close(self, *, hotkey: str, session_id: str, reason: str,
                    transcript: Mapping[str, Any] | None) -> dict | Refusal:
        """Settle a live session from its machine's final (or a failed open) and return
        the state the session is actually in afterwards (a submission that landed
        meanwhile wins). The transcript is verified outside the lock."""
        async with self._lock:
            now = int(self._clock())
            record = self.book.get(session_id)
            if record is None or record.hotkey != hotkey:
                return Refusal("session_unknown", {"session_id": session_id})
            if record.state != LIVE:
                return self._state_of(record)
            if reason == "open_failed":
                if transcript is not None:
                    return Refusal("transcript_invalid", {"why": "a failed open has no transcript"})
            else:
                if transcript is None:
                    return Refusal("transcript_invalid", {"why": "a final close carries its transcript"})
                if not self._fleet.directory_ready(now):
                    return self._directory_unavailable()
                directory = self._fleet.directory()
                expected = Expected(hotkey=record.hotkey, engagement=record.engagement,
                                    env=record.env, split=record.split, index=record.index,
                                    checkpoint=record.checkpoint,
                                    seen_session_ids=self.book.submitted_ids(),
                                    require_graded=False)
        if reason == "open_failed":
            status = "open_failed"
        else:
            result = await asyncio.to_thread(verify_transcript, transcript, directory,
                                             self._token_verifier, expected)
            if not result.ok or result.claims.session_id != session_id:
                return Refusal("transcript_invalid", {
                    "reasons": [r.value for r in result.reasons] or ["session_mismatch"]})
            status = result.final.status
        state = (ABORTED if status == STATUS_ABORTED
                 else CLOSED_GRADED if status == STATUS_GRADED else CLOSED)
        async with self._lock:
            now = int(self._clock())
            current = self.book.get(session_id)
            if current is None or current.state != LIVE:
                return (self._state_of(current) if current is not None
                        else Refusal("session_unknown", {"session_id": session_id}))
            settled = self.book.settle(session_id, state, now=now, status=status)
            self._tokens.pop(session_id, None)
        await self._persist(settled)
        logger.info("sandbox session %s of %s closed: %s", session_id, hotkey[:12], status)
        return self._state_of(settled)

    async def submitted(self, session_id: str) -> None:
        """The intake accepted this session's submission (called after the ledger write,
        so the slot is never counted twice): the reservation ends, whatever state the
        session had reached (`submitted` is dominant)."""
        async with self._lock:
            record = self.book.settle(session_id, SUBMITTED, now=int(self._clock()),
                                      status=STATUS_GRADED)
            self._tokens.pop(session_id, None)
        if record is not None:
            await self._persist(record)

    def void_machine(self, machine_id: str) -> None:
        records = self.book.void_machine(machine_id, int(self._clock()))
        for record in records:
            self._tokens.pop(record.session_id, None)
            task = asyncio.get_running_loop().create_task(self._persist(record))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        if records:
            logger.warning("machine %s drained: %d live sessions voided, no fault to their miners",
                           machine_id, len(records))

    async def maintain(self) -> None:
        async with self._lock:
            now = int(self._clock())
            lapsed = self.book.lapse(now)
            for record in lapsed:
                self._tokens.pop(record.session_id, None)
            self.book.prune(now)
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

    async def _persist(self, record: SessionRecord | None) -> None:
        """Write the record's state (a snapshot taken now). A refused transition means a
        later state is already stored: nothing to do. Other failures are retried with
        backoff; the last one alerts (the in-memory state stays authoritative until a
        restart, which would read the older stored state)."""
        if record is None:
            return
        document = record.to_document()
        for attempt in range(PERSIST_ATTEMPTS):
            if attempt:
                await _sleep(PERSIST_BACKOFF_S * 2 ** (attempt - 1))
            try:
                await asyncio.wait_for(self._store.update(document), self._policy.io_timeout_s)
                return
            except SessionStoreConflict as exc:
                logger.warning("sandbox session %s state %s not stored: %s",
                               record.session_id, document["state"], exc)
                return
            except Exception as exc:
                failure = type(exc).__name__
        logger.error("ALERT sandbox session %s state %s not stored after %d attempts (%s)",
                     record.session_id, document["state"], PERSIST_ATTEMPTS, failure)


__all__ = ["ABORTED", "CLOSED", "CLOSED_GRADED", "LAPSED", "LIVE", "SUBMITTED", "VOIDED",
           "CorpusEngagements", "EngagementBook", "EngagementTerms", "Grant", "Refusal",
           "RlPrecommitEngagements", "SandboxPolicy", "SessionBook", "SessionIssuer",
           "SessionRecord", "SignedJobView", "engagement_digest"]
