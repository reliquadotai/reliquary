"""The control's half of remote auditing: executor tokens, leases, rechecks.

An executor pulls a lease (token ids, prompt length, committed proofs), returns
each item's chunk comparisons, and never writes a verdict. Its results stay
provisional until a recheck on the control's own GPU agrees with one of them;
that recheck then vouches for everything the executor scored before it. A
disagreement quarantines the executor and re-queues every batch it scored since
its last passed recheck. With no executor connected, work is scored locally.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import hmac
import itertools
import logging
import random
import secrets
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from reliquary.protocol.toploc import ChunkResult
from reliquary.validator.corpus_audit_protocol import (
    AUDIT_PROTOCOL,
    ITEM_ERROR,
    ITEM_OK,
    MAX_LEASE_ITEMS,
    MAX_LEASE_TOKENS,
    AuditResult,
    ClaimRequest,
    HeartbeatRequest,
)

logger = logging.getLogger(__name__)

AUDIT_LEASE_SECONDS = 300.0
RECHECK_FRACTION = 0.05
# An executor not heard from for this long is not connected.
EXECUTOR_LIVE_SECONDS = 90.0
REGISTRY_REFRESH_SECONDS = 30.0
# A provisional result older than this forces a recheck of a random one.
PROVISIONAL_MAX_SECONDS = 120.0
# Remote attempts at one batch before the control scores it itself.
REMOTE_ATTEMPTS = 2
SWEEP_SECONDS = 2.0
HEARTBEAT_WRITE_SECONDS = 30.0

Score = tuple[str, tuple[ChunkResult, ...]]


class LeaseRefused(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class ExecutorDirectory:
    """The executor registry as this control last read it, re-read every 30 s.

    A token is accepted only if it hashes to an active, unexpired registration
    for this process's model; one quarantined here is refused at once, before
    the registry shows it.
    """

    def __init__(self, *, model_id: str, model_revision: str,
                 list_documents: Callable[[], Awaitable[list[dict]]] | None = None,
                 clock: Callable[[], float] = time.time,
                 refresh_seconds: float = REGISTRY_REFRESH_SECONDS) -> None:
        if list_documents is None:
            from reliquary.infrastructure.corpus_executor_store import list_executors

            list_documents = list_executors
        self.model_id, self.model_revision = model_id, model_revision
        self._list = list_documents
        self._clock = clock
        self._refresh_every = refresh_seconds
        self._by_hash: dict[str, dict] = {}
        self._by_id: dict[str, dict] = {}
        self._read_at: float | None = None
        self._revoked: set[str] = set()

    async def refresh(self) -> None:
        documents = await self._list()
        self._by_hash = {d["token_sha256"]: d for d in documents if d.get("token_sha256")}
        self._by_id = {d["executor_id"]: d for d in documents if d.get("executor_id")}
        self._read_at = self._clock()

    async def maybe_refresh(self) -> None:
        if self._read_at is None or self._clock() - self._read_at >= self._refresh_every:
            try:
                await self.refresh()
            except Exception:
                logger.exception("executor registry unreadable; keeping the last read")

    def revoke_locally(self, executor_id: str) -> None:
        self._revoked.add(executor_id)

    def _refusal(self, document: dict | None) -> str | None:
        if document is None:
            return "unknown_token"
        if document.get("status") != "active" or document["executor_id"] in self._revoked:
            return "revoked"
        if float(document.get("expires_at") or 0) <= self._clock():
            return "expired"
        if (document.get("model_id"), document.get("model_revision")) != (
                self.model_id, self.model_revision):
            return "wrong_model"
        return None

    def authenticate(self, token: str | None, executor_id: str | None = None):
        """``(document, None)`` for a token this control accepts, else ``(None, why)``."""
        if not token:
            return None, "missing_token"
        digest = token_sha256(token)
        document = None
        for known, candidate in self._by_hash.items():
            if hmac.compare_digest(known, digest):
                document = candidate
        refusal = self._refusal(document)
        if refusal is not None:
            return None, refusal
        if executor_id is not None and executor_id != document["executor_id"]:
            return None, "wrong_executor"
        return document, None

    def is_authorized(self, executor_id: str) -> bool:
        return self._refusal(self._by_id.get(executor_id)) is None


@dataclass
class _Work:
    id: int
    items: list[dict]
    future: asyncio.Future
    queued_at: float
    attempts: int = 0


@dataclass
class _Lease:
    lease_id: str
    work: _Work
    executor_id: str
    expires_at: float


@dataclass
class _Scored:
    seq: int
    work: _Work
    scores: list[Score]
    at: float
    rechecking: bool = field(default=False)


def _lease_units(items: Sequence[dict]) -> list[list[int]]:
    """Item indexes grouped into leases under the item and token bounds."""
    units: list[list[int]] = []
    current: list[int] = []
    tokens = 0
    for k, item in enumerate(items):
        size = len(item["tokens"])
        if current and (len(current) >= MAX_LEASE_ITEMS or tokens + size > MAX_LEASE_TOKENS):
            units.append(current)
            current, tokens = [], 0
        current.append(k)
        tokens += size
    if current:
        units.append(current)
    return units


def scores_agree(remote: Sequence[Score], local: Sequence[Score], thresholds) -> bool:
    """Within the proof tolerance: same status per item, and every chunk measure
    of the remote within the threshold of the control's own."""
    if len(remote) != len(local):
        return False
    for (r_status, r_chunks), (l_status, l_chunks) in zip(remote, local):
        if r_status != l_status or len(r_chunks) != len(l_chunks):
            return False
        for r, l in zip(r_chunks, l_chunks):
            if abs(r.exp_mismatches - l.exp_mismatches) > thresholds.exp_mismatch:
                return False
            if abs(r.mant_err_mean - l.mant_err_mean) > thresholds.mant_mean:
                return False
            if abs(r.mant_err_median - l.mant_err_median) > thresholds.mant_median:
                return False
    return True


class RemoteAuditDispatcher:
    """Batches the auditors hand over, leased to executors or scored locally.

    ``local_scores(items)`` is the trusted verifier (the control's GPU), used
    for rechecks and whenever no executor is connected; ``quarantine`` and
    ``record_heartbeat`` write the executor registry.
    """

    def __init__(self, *, directory: ExecutorDirectory, proof, local_scores,
                 quarantine: Callable[[str, str], Awaitable[Any]] | None = None,
                 record_heartbeat: Callable[[str, float, dict], Awaitable[Any]] | None = None,
                 clock: Callable[[], float] = time.time, rng: random.Random | None = None,
                 recheck_fraction: float = RECHECK_FRACTION,
                 lease_seconds: float = AUDIT_LEASE_SECONDS,
                 live_seconds: float = EXECUTOR_LIVE_SECONDS,
                 provisional_max_seconds: float = PROVISIONAL_MAX_SECONDS) -> None:
        self._directory = directory
        self._proof = proof
        self._local_scores = local_scores
        self._quarantine_write = quarantine
        self._heartbeat_write = record_heartbeat
        self._clock = clock
        self._rng = rng or random.SystemRandom()
        self._fraction = recheck_fraction
        self._lease_seconds = lease_seconds
        self._live = live_seconds
        self._provisional_max = provisional_max_seconds
        self._ids = itertools.count()
        self._seq = itertools.count()
        self._queue: collections.deque[_Work] = collections.deque()
        self._local: collections.deque[_Work] = collections.deque()
        self._leases: dict[str, _Lease] = {}
        self._provisional: dict[str, list[_Scored]] = {}
        self._seen: dict[str, float] = {}
        self._detail: dict[str, dict] = {}
        self._written: dict[str, float] = {}
        self._background: set[asyncio.Task] = set()
        self.quarantined: set[str] = set()
        self.stats = collections.Counter()

    # -- the auditor's side ------------------------------------------------

    def connected(self) -> bool:
        now = self._clock()
        return any(now - seen <= self._live and self._directory.is_authorized(eid)
                   and eid not in self.quarantined for eid, seen in self._seen.items())

    async def score(self, items: Sequence[dict]) -> list[Score]:
        """Chunk scores for ``items`` (``tokens``, ``prompt_len``, ``proofs``),
        each one either vouched for by a recheck or computed here."""
        loop = asyncio.get_running_loop()
        units = []
        for indexes in _lease_units(items):
            work = _Work(id=next(self._ids), items=[items[k] for k in indexes],
                         future=loop.create_future(), queued_at=self._clock())
            units.append((indexes, work))
            self._queue.append(work)
        scored: list = [None] * len(items)
        for indexes, work in units:
            for k, score in zip(indexes, await work.future):
                scored[k] = score
        return scored

    # -- the executor's side -----------------------------------------------

    def _contact(self, executor_id: str) -> None:
        self._seen[executor_id] = self._clock()

    def heartbeat(self, executor_id: str, detail: dict | None = None) -> None:
        self._contact(executor_id)
        if detail is not None:
            self._detail[executor_id] = dict(detail)

    def claim(self, executor_id: str) -> dict | None:
        self._contact(executor_id)
        while self._queue:
            work = self._queue.popleft()
            if work.future.done():
                continue
            lease = _Lease(lease_id=secrets.token_hex(16), work=work, executor_id=executor_id,
                           expires_at=self._clock() + self._lease_seconds)
            self._leases[lease.lease_id] = lease
            self.stats["leased"] += 1
            return {
                "protocol": AUDIT_PROTOCOL, "lease_id": lease.lease_id,
                "model_id": self._directory.model_id,
                "model_revision": self._directory.model_revision,
                "chunk_tokens": self._proof.chunk_tokens, "topk": self._proof.topk,
                "expires_at": lease.expires_at,
                "items": [{"tokens": list(i["tokens"]), "prompt_len": int(i["prompt_len"]),
                           "proofs": list(i["proofs"])} for i in work.items],
            }
        return None

    def result(self, executor_id: str, lease_id: str, result: AuditResult) -> str:
        """Take an executor's scores for its lease; raises ``LeaseRefused``."""
        self._contact(executor_id)
        lease = self._leases.get(lease_id)
        if lease is None or lease.executor_id != executor_id:
            raise LeaseRefused(410, "lease_unknown")
        del self._leases[lease_id]
        work = lease.work
        if lease.expires_at <= self._clock():
            self._requeue(work)
            raise LeaseRefused(410, "lease_expired")
        scores = result.scores
        if len(scores) != len(work.items) or any(
                s.status == ITEM_OK and len(s.chunks) != len(item["proofs"])
                for s, item in zip(scores, work.items)):
            self._requeue(work)
            raise LeaseRefused(422, "result_does_not_fit_the_lease")
        if any(s.status == ITEM_ERROR for s in scores):
            # The executor's own fault: the batch goes back, nobody is judged.
            self.stats["executor_errors"] += 1
            self._requeue(work)
            return "requeued"
        converted = [(s.status, tuple(ChunkResult(int(e), float(m), float(d))
                                      for e, m, d in s.chunks)) for s in scores]
        entry = _Scored(seq=next(self._seq), work=work, scores=converted, at=self._clock())
        self._provisional.setdefault(executor_id, []).append(entry)
        self.stats["scored"] += 1
        if self._rng.random() < self._fraction:
            self._spawn(self._recheck(executor_id, entry))
        return "accepted"

    # -- rechecks, expiry, fallback ------------------------------------------

    def _spawn(self, coroutine) -> None:
        task = asyncio.ensure_future(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _requeue(self, work: _Work) -> None:
        if work.future.done():
            return
        work.attempts += 1
        if work.attempts >= REMOTE_ATTEMPTS:
            self._local.append(work)
        else:
            # Its wait for an executor starts over.
            work.queued_at = self._clock()
            self._queue.appendleft(work)

    @staticmethod
    def _resolve(work: _Work, scores: list[Score]) -> None:
        if not work.future.done():
            work.future.set_result(scores)

    async def _recheck(self, executor_id: str, entry: _Scored) -> None:
        entry.rechecking = True
        try:
            local = await self._local_scores(entry.work.items)
        except Exception:
            # Ours, not the executor's: it stays provisional for a later recheck.
            logger.exception("corpus audit recheck of executor %s failed locally", executor_id)
            entry.rechecking = False
            return
        self.stats["rechecks"] += 1
        pending = self._provisional.get(executor_id, [])
        if executor_id in self.quarantined:
            self._resolve(entry.work, local)
            return
        if scores_agree(entry.scores, local, self._proof.thresholds()):
            vouched = [e for e in pending if e.seq <= entry.seq]
            self._provisional[executor_id] = [e for e in pending if e.seq > entry.seq]
            for e in vouched:
                self._resolve(e.work, local if e is entry else e.scores)
            return
        await self._quarantine(executor_id, entry, local)

    async def _quarantine(self, executor_id: str, entry: _Scored, local: list[Score]) -> None:
        reason = f"recheck of batch {entry.work.id} diverged beyond the proof tolerance"
        logger.error("corpus audit executor %s quarantined: %s", executor_id, reason)
        self.quarantined.add(executor_id)
        self._directory.revoke_locally(executor_id)
        self.stats["quarantined"] += 1
        self._resolve(entry.work, local)
        # Everything it scored since its last passed recheck, and what it holds now.
        for e in self._provisional.pop(executor_id, []):
            if e is not entry:
                self._requeue(e.work)
        for lease_id, lease in list(self._leases.items()):
            if lease.executor_id == executor_id:
                del self._leases[lease_id]
                self._requeue(lease.work)
        if self._quarantine_write is not None:
            try:
                await self._quarantine_write(executor_id, reason)
            except Exception:
                logger.exception("executor %s quarantine not written to the registry",
                                 executor_id)

    async def sweep(self) -> None:
        """One pass: expire leases, force due rechecks, score locally what no
        executor will take."""
        now = self._clock()
        for lease_id, lease in list(self._leases.items()):
            if lease.expires_at <= now:
                del self._leases[lease_id]
                logger.warning("corpus audit lease %s of executor %s expired; re-queued",
                               lease_id[:8], lease.executor_id)
                self._requeue(lease.work)
        for executor_id, entries in list(self._provisional.items()):
            if not entries or any(e.rechecking for e in entries):
                continue
            busy = any(lease.executor_id == executor_id for lease in self._leases.values())
            if now - entries[0].at >= self._provisional_max or not (self._queue or busy):
                # Random among them: an executor cannot tell which batch is checked.
                self._spawn(self._recheck(executor_id, self._rng.choice(entries)))
        if not self.connected():
            while self._queue:
                self._local.append(self._queue.popleft())
        else:
            while self._queue and now - self._queue[0].queued_at > self._lease_seconds:
                self._local.append(self._queue.popleft())
        while self._local:
            work = self._local.popleft()
            if work.future.done():
                continue
            try:
                self._resolve(work, await self._local_scores(work.items))
                self.stats["local"] += 1
            except Exception as exc:
                if not work.future.done():
                    work.future.set_exception(exc)

    async def write_heartbeats(self) -> None:
        if self._heartbeat_write is None:
            return
        for executor_id, seen in list(self._seen.items()):
            if self._written.get(executor_id) == seen:
                continue
            if seen - self._written.get(executor_id, float("-inf")) < HEARTBEAT_WRITE_SECONDS \
                    and executor_id in self._written:
                continue
            try:
                await self._heartbeat_write(executor_id, seen, self._detail.get(executor_id, {}))
                self._written[executor_id] = seen
            except Exception:
                logger.exception("heartbeat of executor %s not written", executor_id)

    async def run(self, *, sweep_seconds: float = SWEEP_SECONDS) -> None:
        while True:
            try:
                await self._directory.maybe_refresh()
                await self.sweep()
                await self.write_heartbeats()
            except Exception:
                logger.exception("corpus audit dispatcher sweep failed; retrying")
            await asyncio.sleep(sweep_seconds)


def build_audit_executor_router(dispatcher: RemoteAuditDispatcher,
                                directory: ExecutorDirectory) -> APIRouter:
    """``/corpus/internal/audit/...``: claim, result, heartbeat, each behind
    ``Authorization: Bearer <executor token>``."""
    router = APIRouter()

    def _authenticated(request: Request, executor_id: str | None = None) -> dict:
        header = request.headers.get("authorization", "")
        token = header[len("Bearer "):] if header.startswith("Bearer ") else None
        document, refusal = directory.authenticate(token, executor_id)
        if document is None:
            raise HTTPException(status_code=401, detail=refusal)
        return document

    @router.post("/corpus/internal/audit/claim")
    async def claim(body: ClaimRequest, request: Request):
        document = _authenticated(request, body.executor_id)
        if (body.model_id, body.model_revision) != (document["model_id"],
                                                    document["model_revision"]):
            raise HTTPException(status_code=409, detail="wrong_model")
        lease = dispatcher.claim(document["executor_id"])
        if lease is None:
            return Response(status_code=204)
        return lease

    @router.post("/corpus/internal/audit/heartbeat")
    async def heartbeat(body: HeartbeatRequest, request: Request) -> dict:
        document = _authenticated(request, body.executor_id)
        dispatcher.heartbeat(document["executor_id"], body.detail)
        return {"executor_id": document["executor_id"], "model_id": document["model_id"],
                "model_revision": document["model_revision"]}

    @router.post("/corpus/internal/audit/{lease_id}/result")
    async def result(lease_id: str, body: AuditResult, request: Request) -> dict:
        document = _authenticated(request)
        try:
            outcome = dispatcher.result(document["executor_id"], lease_id, body)
        except LeaseRefused as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
        return {"lease_id": lease_id, "outcome": outcome}

    return router


__all__ = [
    "AUDIT_LEASE_SECONDS",
    "ExecutorDirectory",
    "LeaseRefused",
    "RECHECK_FRACTION",
    "RemoteAuditDispatcher",
    "build_audit_executor_router",
    "scores_agree",
    "token_sha256",
]
