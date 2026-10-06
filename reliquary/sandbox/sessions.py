"""Session tokens for signed episodes (spec §4.0 and §7; plan 3 rulings 1-3, 9).

A miner asks for a token naming its engagement. The issuer checks the per-hotkey caps,
asks the engagement's book for its terms (corpus: the job is served and signed, the
prompt is the job's, a slot is free once live reservations are counted; RL: not yet
served), picks a machine, signs the claims with the validator key, stores the session
(without the token's signature) and reserves the slot. Only then is the token returned.

Tokens are bearer secrets until their episode closes: they live in this process's
memory only (to answer a resent request with the same token) and are never logged or
written. A session ends `submitted` (its submission was accepted), `closed` (an unpaid
final reported: expired, box_failed, budget_exhausted including `transcript_bytes`, a
graded episode not submitted, or a failed open), `aborted` (void: refunded from the
open rate, counted against the aborted cap), `voided` (its machine was drained) or
`lapsed` (past `expires_at + GRADING_GRACE_S`, when submissions are refused anyway).

While the fleet's machine directory is stale (`fleet.directory_ready()` is false) no
token is issued and no close is verified: both are refused `directory_unavailable`
with a retry delay, never judged against an empty directory (which would read as
`unknown_key`).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Protocol

from reliquary_sandbox.attest import (
    GRADING_GRACE_S, STATUS_ABORTED, Budgets, Expected, SessionClaims, issue_session_token,
    token_sha256, verify_transcript,
)

from reliquary.corpus.job import is_signed_sandbox, sandbox_split
from reliquary.corpus.signed_reasons import corpus_engagement
from reliquary.sandbox.tasks import ResolvedTask

logger = logging.getLogger(__name__)

SESSION_SCHEMA = "reliquary/sandbox-session/v1"
LIVE, SUBMITTED, CLOSED, ABORTED, VOIDED, LAPSED = (
    "live", "submitted", "closed", "aborted", "voided", "lapsed")
HOUR, DAY = 3600, 86400


@dataclass(frozen=True)
class SandboxPolicy:
    max_live_per_hotkey: int = 8
    max_opens_per_hour: int = 120
    max_aborted_per_day: int = 20
    open_window_s: int = 900
    request_skew_s: int = 120
    retry_after_s: int = 10

    _ENV = {"max_live_per_hotkey": "RELIQUARY_SANDBOX_MAX_LIVE_PER_HOTKEY",
            "max_opens_per_hour": "RELIQUARY_SANDBOX_MAX_OPENS_PER_HOUR",
            "max_aborted_per_day": "RELIQUARY_SANDBOX_MAX_ABORTED_PER_DAY",
            "open_window_s": "RELIQUARY_SANDBOX_OPEN_WINDOW_S",
            "request_skew_s": "RELIQUARY_SANDBOX_REQUEST_SKEW_S",
            "retry_after_s": "RELIQUARY_SANDBOX_RETRY_AFTER_S"}

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


@dataclass
class SessionRecord:
    session_id: str
    hotkey: str
    request_id: str
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
    def from_document(cls, document: Mapping[str, Any]) -> SessionRecord:
        if document.get("schema") != SESSION_SCHEMA:
            raise ValueError("not a session document")
        return cls(**{f.name: document[f.name] for f in fields(cls)})


class CorpusEngagements:
    kind = "corpus"

    def __init__(self, jobs: Callable[[str], SignedJobView | None], book: SessionBook,
                 clock: Callable[[], float] = time.time) -> None:
        self._jobs, self._book, self._clock = jobs, book, clock

    async def terms(self, hotkey: str, engagement: Mapping[str, Any]) -> EngagementTerms | Refusal:
        job_id, index = engagement.get("job_id"), engagement.get("prompt_index")
        view = self._jobs(job_id) if isinstance(job_id, str) else None
        if view is None:
            return Refusal("job_not_served", {"job_id": job_id})
        job = view.job
        if not is_signed_sandbox(job):
            return Refusal("job_not_signed", {"job_id": job_id})
        if isinstance(index, bool) or not isinstance(index, int) or not job.owns(index):
            return Refusal("prompt_mismatch", {"prompt_index": index})
        if view.is_banned is not None and await view.is_banned(hotkey):
            return Refusal("miner_banned", {})
        remaining = await view.slots_remaining(index)
        if remaining is None:
            return Refusal("job_complete", {"job_id": job_id})
        reserved = self._book.reserved(job.job_id, index, int(self._clock()))
        if remaining - reserved <= 0:
            return Refusal("prompt_unavailable", {"prompt_index": index,
                                                  "slots_remaining": remaining, "reserved": reserved})
        task = await view.resolve_task(index)
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
      shared, with its `Expected` built from the precommit."""

    kind = "rl_precommit"

    async def terms(self, hotkey: str, engagement: Mapping[str, Any]) -> Refusal:
        return Refusal("engagement_kind_unsupported",
                       {"kind": self.kind, "why": "the episodic RL task is not served yet"})


class SessionBook:
    """Sessions this validator issued, in memory (restored from the store at start)."""

    def __init__(self, policy: SandboxPolicy) -> None:
        self._policy = policy
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
        return record.state == LIVE and now <= record.expires_at + GRADING_GRACE_S

    def reserved(self, job_id: str, prompt_index: int, now: int) -> int:
        return sum(1 for r in self._sessions.values()
                   if r.job_id == job_id and r.prompt_index == prompt_index and self._holds(r, now))

    def submitted_ids(self) -> frozenset[str]:
        return frozenset(s for s, r in self._sessions.items() if r.state == SUBMITTED)

    def open_refusal(self, hotkey: str, now: int) -> Refusal | None:
        mine = [r for r in self._sessions.values() if r.hotkey == hotkey]
        policy = self._policy
        live = sum(1 for r in mine if self._holds(r, now))
        if live >= policy.max_live_per_hotkey:
            return Refusal("live_cap", {"live": live, "max": policy.max_live_per_hotkey})
        opens = sum(1 for r in mine if r.issued_at > now - HOUR and r.state != ABORTED)
        if opens >= policy.max_opens_per_hour:
            return Refusal("open_rate_cap", {"opens": opens, "max": policy.max_opens_per_hour})
        aborted = sum(1 for r in mine if r.state == ABORTED and (r.closed_at or 0) > now - DAY)
        if aborted >= policy.max_aborted_per_day:
            return Refusal("aborted_cap", {"aborted": aborted, "max": policy.max_aborted_per_day})
        return None

    def settle(self, session_id: str, state: str, *, now: int,
               status: str | None = None) -> SessionRecord | None:
        record = self._sessions.get(session_id)
        if record is None or record.state != LIVE:
            return None
        record.state, record.closed_status, record.closed_at = state, status, int(now)
        return record

    def lapse(self, now: int) -> list[SessionRecord]:
        lapsed = [r for r in self._sessions.values()
                  if r.state == LIVE and now > r.expires_at + GRADING_GRACE_S]
        for record in lapsed:
            self.settle(record.session_id, LAPSED, now=now)
        return lapsed

    def void_machine(self, machine_id: str, now: int) -> list[SessionRecord]:
        voided = [r for r in self._sessions.values()
                  if r.machine_id == machine_id and self._holds(r, now)]
        for record in voided:
            self.settle(record.session_id, VOIDED, now=now, status="machine_drained")
        return voided

    def prune(self, now: int) -> None:
        old = [s for s, r in self._sessions.items()
               if r.state != LIVE and (r.closed_at or r.issued_at) < now - DAY - HOUR]
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
        async with self._lock:
            now = int(self._clock())
            existing = self.book.by_request(hotkey, request_id)
            if existing is not None:
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
                kind=str(engagement.get("kind")), engagement=terms.engagement, env=terms.env,
                split=terms.split, index=terms.index, checkpoint=terms.checkpoint,
                job_id=terms.job_id, prompt_index=terms.prompt_index,
                machine_id=placement.machine_id, issued_at=now, expires_at=claims.expires_at,
                token_sha256=token_sha256(token))
            try:
                await self._store.create(record.to_document())
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

    async def close(self, *, hotkey: str, session_id: str, reason: str,
                    transcript: Mapping[str, Any] | None) -> dict | Refusal:
        async with self._lock:
            now = int(self._clock())
            record = self.book.get(session_id)
            if record is None or record.hotkey != hotkey:
                return Refusal("session_unknown", {"session_id": session_id})
            if record.state != LIVE:
                return {"session_id": session_id, "state": record.state,
                        "status": record.closed_status}
            if reason == "open_failed":
                if transcript is not None:
                    return Refusal("transcript_invalid", {"why": "a failed open has no transcript"})
                status = "open_failed"
            else:
                if transcript is None:
                    return Refusal("transcript_invalid", {"why": "a final close carries its transcript"})
                if not self._fleet.directory_ready(now):
                    return self._directory_unavailable()
                expected = Expected(hotkey=record.hotkey, engagement=record.engagement,
                                    env=record.env, split=record.split, index=record.index,
                                    checkpoint=record.checkpoint,
                                    seen_session_ids=self.book.submitted_ids(),
                                    require_graded=False)
                result = await asyncio.to_thread(verify_transcript, transcript,
                                                 self._fleet.directory(), self._token_verifier,
                                                 expected)
                if not result.ok or result.claims.session_id != session_id:
                    return Refusal("transcript_invalid", {
                        "reasons": [r.value for r in result.reasons] or ["session_mismatch"]})
                status = result.final.status
            state = ABORTED if status == STATUS_ABORTED else CLOSED
            settled = self.book.settle(session_id, state, now=now, status=status)
            self._tokens.pop(session_id, None)
        await self._persist(settled)
        logger.info("sandbox session %s of %s closed: %s", session_id, hotkey[:12], status)
        return {"session_id": session_id, "state": state, "status": status}

    async def submitted(self, session_id: str) -> None:
        """The intake accepted this session's submission (called after the ledger write,
        so the slot is never counted twice): the reservation ends."""
        record = self.book.settle(session_id, SUBMITTED, now=int(self._clock()), status="graded")
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
        now = int(self._clock())
        for record in self.book.lapse(now):
            self._tokens.pop(record.session_id, None)
            await self._persist(record)
        self.book.prune(now)

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
        if record is None:
            return
        try:
            await self._store.update(record.to_document())
        except Exception as exc:
            logger.warning("sandbox session %s state not stored: %s", record.session_id,
                           type(exc).__name__)


__all__ = ["ABORTED", "CLOSED", "LAPSED", "LIVE", "SUBMITTED", "VOIDED", "CorpusEngagements",
           "EngagementBook", "EngagementTerms", "Grant", "Refusal", "RlPrecommitEngagements",
           "SandboxPolicy", "SessionBook", "SessionIssuer", "SessionRecord", "SignedJobView"]
