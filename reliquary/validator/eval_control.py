"""The GPU-less eval control: every evaluation job, whatever its model, in one process.

It loads no model. Each job has its own tokenizer (CPU) and its own executor
pool keyed by ``model@revision``. Every audit batch is scored by two executors
on distinct providers (``provider_id`` and ``host`` both differ):
- agreement (the same decision per item, every measure within R3's drift
  tolerance) decides the batch;
- disagreement sends it to a third executor; whoever agrees with nobody is
  quarantined, and the passes it co-signed are re-audited by fresh pairs;
- one executor alone never decides: the batch waits.

Executors reach it on ``/corpus/internal/eval-audit/...``; miners on
``/corpus/jobs/order-eval-.../...``. The corpus control's behaviour is untouched:
it never wires an ``order-eval-`` job, and this process serves nothing else.
"""

from __future__ import annotations

import asyncio
import collections
import functools
import hmac
import json
import itertools
import logging
import secrets
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from reliquary.eval.qualify_protocol import EvalClaimRequest, QualifyResult
from reliquary.protocol.toploc import ChunkResult
from reliquary.validator.corpus_audit_protocol import (
    AUDIT_PROTOCOL,
    ITEM_ERROR,
    ITEM_OK,
    AuditResult,
    HeartbeatRequest,
)
from reliquary.validator.corpus_audit_remote import (
    AUDIT_LEASE_SECONDS,
    EXECUTOR_LIVE_SECONDS,
    LEASE_EXPIRY_STRIKES,
    MAX_LEASES_PER_EXECUTOR,
    REGISTRY_REFRESH_SECONDS,
    LeaseRefused,
    _lease_units,
    scores_agree,
    token_sha256,
)

logger = logging.getLogger(__name__)

EVAL_AUDIT_PREFIX = "/corpus/internal/eval-audit"
# Scorers of one batch: a pair, and a third on disagreement. No agreement then:
# the batch is parked (its records stay pending) and its job needs attention.
MAX_SCORERS = 3
STATUS_SECONDS = 30.0
HEARTBEAT_WRITE_SECONDS = 30.0
EVAL_CONTROL_STATUS_KEY = "reliquary/eval/control/status.json"


class BatchParked(Exception):
    """No two scorers of a batch agree (or two agreeing pairs conflict)."""

Score = tuple[str, tuple[ChunkResult, ...]]
ModelKey = tuple[str, str]


class EvalExecutorDirectory:
    """Every active, unexpired executor registration, whatever its model. A
    lease is only handed to an executor for its own model's pool."""

    def __init__(self, *, list_documents: Callable[[], Awaitable[list[dict]]] | None = None,
                 clock: Callable[[], float] = time.time,
                 refresh_seconds: float = REGISTRY_REFRESH_SECONDS) -> None:
        if list_documents is None:
            from reliquary.infrastructure.corpus_executor_store import list_executors

            list_documents = list_executors
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
        if document.get("scope") != "eval":
            # A corpus executor serves the corpus control, never this one.
            return "wrong_scope"
        if document.get("status") != "active" or document["executor_id"] in self._revoked:
            return "revoked"
        if float(document.get("expires_at") or 0) <= self._clock():
            return "expired"
        return None

    def authenticate(self, token: str | None, executor_id: str | None = None):
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

    def document(self, executor_id: str) -> dict | None:
        return self._by_id.get(executor_id)


def _placement(document: dict) -> tuple[str, str] | None:
    """Where an executor runs; None when it cannot be told apart from others."""
    provider, host = document.get("provider_id"), document.get("host")
    if not provider or not host:
        return None
    return str(provider), str(host)


@dataclass
class _Batch:
    id: int
    model: ModelKey
    proof: Any
    items: list[dict]
    future: asyncio.Future
    scores: dict[str, list[Score]] = field(default_factory=dict)
    places: dict[str, tuple[str, str]] = field(default_factory=dict)
    leased: set[str] = field(default_factory=set)
    needed: int = 2
    job_id: str = ""


@dataclass
class _Lease:
    lease_id: str
    batch: _Batch
    executor_id: str
    expires_at: float


class PairedAuditDispatcher:
    """Audit batches of every model, each decided by two agreeing executors on
    distinct providers. ``quarantine``/``record_heartbeat`` write the registry."""

    def __init__(self, *, directory: EvalExecutorDirectory,
                 quarantine: Callable[[str, str], Awaitable[Any]] | None = None,
                 record_heartbeat: Callable[[str, float, dict], Awaitable[Any]] | None = None,
                 clock: Callable[[], float] = time.time,
                 lease_seconds: float = AUDIT_LEASE_SECONDS,
                 live_seconds: float = EXECUTOR_LIVE_SECONDS,
                 max_leases_per_executor: int = MAX_LEASES_PER_EXECUTOR,
                 expiry_strikes: int = LEASE_EXPIRY_STRIKES) -> None:
        self._directory = directory
        self._quarantine_write = quarantine
        self._heartbeat_write = record_heartbeat
        self._clock = clock
        self._lease_seconds = lease_seconds
        self._live = live_seconds
        self._max_leases = max_leases_per_executor
        self._strikes_limit = expiry_strikes
        self._ids = itertools.count()
        self._queues: dict[ModelKey, collections.deque[_Batch]] = collections.defaultdict(
            collections.deque)
        self._leases: dict[str, _Lease] = {}
        self._strikes: collections.Counter = collections.Counter()
        self._seen: dict[str, float] = {}
        self._listeners: list[Callable[[str], Awaitable[Any]]] = []
        self._background: set[asyncio.Task] = set()
        self._unwritten: dict[str, str] = {}
        self._documents: dict[str, dict] = {}
        self._detail: dict[str, dict] = {}
        self._written: dict[str, tuple[float, dict]] = {}
        # job id -> why it needs attention (parked batches), for the status.
        self.attention: dict[str, list[str]] = collections.defaultdict(list)
        self.parked: collections.Counter = collections.Counter()
        self.quarantined: set[str] = set()
        self.stats = collections.Counter()

    # -- the auditors' side ------------------------------------------------

    def view(self, model_id: str, model_revision: str, proof, job_id: str = "") -> "PoolView":
        return PoolView(self, (model_id, model_revision), proof, job_id)

    def subscribe(self, listener: Callable[[str], Awaitable[Any]]) -> None:
        self._listeners.append(listener)

    def unsubscribe(self, listener) -> None:
        self._listeners = [other for other in self._listeners if other != listener]

    def forget_job(self, job_id: str) -> None:
        """A job unwired: its waiting batches and flags go with it."""
        for queue in self._queues.values():
            for batch in list(queue):
                if batch.job_id == job_id:
                    queue.remove(batch)
                    if not batch.future.done():
                        batch.future.cancel()
        self.attention.pop(job_id, None)
        self.parked.pop(job_id, None)

    async def score(self, model: ModelKey, proof, items: Sequence[dict], job_id: str = ""):
        """``(status, chunks, scored_by)`` per item; ``scored_by`` is the
        tuple of executors whose agreement decided it."""
        loop = asyncio.get_running_loop()
        units = []
        for indexes in _lease_units(items):
            batch = _Batch(id=next(self._ids), model=model, proof=proof,
                           items=[items[k] for k in indexes], future=loop.create_future(),
                           job_id=job_id)
            units.append((indexes, batch))
            self._queues[model].append(batch)
        scored: list = [None] * len(items)
        for indexes, batch in units:
            scores, scored_by = await batch.future
            for k, (status, chunks) in zip(indexes, scores):
                scored[k] = (status, chunks, scored_by)
        return scored

    def pending(self, model: ModelKey | None = None) -> int:
        queues = [self._queues[model]] if model is not None else list(self._queues.values())
        return sum(1 for queue in queues for batch in queue if not batch.future.done())

    # -- the executors' side -----------------------------------------------

    def heartbeat(self, executor_id: str, document: dict | None = None,
                  detail: dict | None = None) -> None:
        self._seen[executor_id] = self._clock()
        if document is not None:
            self._documents[executor_id] = document
        if detail is not None:
            self._detail[executor_id] = dict(detail)

    def heartbeat_detail(self, executor_id: str) -> dict:
        """What the registry records of an executor's last heartbeat:
        ``{"leases": int, "loaded": bool}``, as the corpus control writes it."""
        detail = self._detail.get(executor_id, {})
        held = sum(1 for lease in self._leases.values() if lease.executor_id == executor_id)
        leases = detail.get("leases", held)
        return {"leases": int(leases) if isinstance(leases, (int, float)) else held,
                "loaded": bool(detail.get("loaded", False))}

    def _eligible(self, batch: _Batch, executor_id: str, place: tuple[str, str]) -> bool:
        if batch.future.done() or executor_id in batch.scores or executor_id in batch.leased:
            return False
        if len(batch.scores) + len(batch.leased) >= batch.needed:
            return False
        involved = [batch.places[e] for e in batch.scores] + [
            batch.places[e] for e in batch.leased]
        # A distinct provider AND a distinct host from everyone already on it.
        return all(place[0] != p and place[1] != h for p, h in involved)

    def claim(self, document: dict) -> dict | None:
        """An audit lease for this executor's model, or None. Raises
        ``LeaseRefused(409)`` for an executor whose placement is unknown."""
        executor_id = document["executor_id"]
        self.heartbeat(executor_id, document)
        place = _placement(document)
        if place is None:
            raise LeaseRefused(409, "executor_provider_unknown")
        if sum(1 for lease in self._leases.values() if lease.executor_id == executor_id) \
                >= self._max_leases:
            return None
        queue = self._queues.get((document["model_id"], document["model_revision"]))
        for batch in list(queue or ()):
            if batch.future.done():
                queue.remove(batch)
                continue
            if not self._eligible(batch, executor_id, place):
                continue
            lease = _Lease(lease_id=secrets.token_hex(16), batch=batch, executor_id=executor_id,
                           expires_at=self._clock() + self._lease_seconds)
            self._leases[lease.lease_id] = lease
            batch.leased.add(executor_id)
            batch.places[executor_id] = place
            self.stats["leased"] += 1
            return {
                "protocol": AUDIT_PROTOCOL, "lease_id": lease.lease_id,
                "model_id": batch.model[0], "model_revision": batch.model[1],
                "chunk_tokens": batch.proof.chunk_tokens, "topk": batch.proof.topk,
                "expires_at": lease.expires_at,
                "items": [{"tokens": list(i["tokens"]), "prompt_len": int(i["prompt_len"]),
                           "proofs": list(i["proofs"])} for i in batch.items],
            }
        return None

    def lease_of(self, lease_id: str) -> _Lease | None:
        return self._leases.get(lease_id)

    async def result(self, executor_id: str, lease_id: str, result: AuditResult) -> str:
        self.heartbeat(executor_id)
        lease = self._leases.get(lease_id)
        if lease is None or lease.executor_id != executor_id:
            raise LeaseRefused(410, "lease_unknown")
        del self._leases[lease_id]
        batch = lease.batch
        batch.leased.discard(executor_id)
        if lease.expires_at <= self._clock():
            raise LeaseRefused(410, "lease_expired")
        self._strikes[executor_id] = 0
        scores = result.scores
        if len(scores) != len(batch.items) or any(
                s.status == ITEM_OK and len(s.chunks) != len(item["proofs"])
                for s, item in zip(scores, batch.items)):
            raise LeaseRefused(422, "result_does_not_fit_the_lease")
        if any(s.status == ITEM_ERROR for s in scores) or batch.future.done():
            # The executor's own fault, or no longer needed: nobody is judged on it.
            self.stats["executor_errors" if not batch.future.done() else "late"] += 1
            return "requeued" if not batch.future.done() else "unneeded"
        batch.scores[executor_id] = [
            (s.status, tuple(ChunkResult(int(e), float(m), float(d)) for e, m, d in s.chunks))
            for s in scores]
        self.stats["scored"] += 1
        await self._decide(batch)
        return "accepted"

    async def _decide(self, batch: _Batch) -> None:
        """Two agreeing scorers decide. Three: exactly one decision held by an
        agreeing pair decides, whoever agrees with nobody is quarantined. Never
        an id-ordered pick: conflicting pairs, or no pair, park the batch."""
        from reliquary.validator.corpus_audit import outcome_from_scores

        if len(batch.scores) < 2:
            return
        ids = sorted(batch.scores)
        pairs = [(e, o) for k, e in enumerate(ids) for o in ids[k + 1:]
                 if scores_agree(batch.scores[e], batch.scores[o], batch.proof)]

        def decision(executor_id: str) -> tuple:
            return tuple(outcome_from_scores(status, chunks, batch.proof).passed
                         for status, chunks in batch.scores[executor_id])

        decisions = {decision(e) for pair in pairs for e in pair}
        members = sorted({e for pair in pairs for e in pair})
        if len(decisions) == 1:
            for minority in (e for e in ids if e not in members):
                await self.quarantine(minority, f"disagreed with executors {members} on batch "
                                                f"{batch.id}")
            if not batch.future.done():
                batch.future.set_result((batch.scores[members[0]], tuple(members)))
                self.stats["agreed" if len(ids) == 2 else "settled_by_majority"] += 1
            return
        if len(ids) < MAX_SCORERS and not pairs:
            # One more executor, on yet another provider and host.
            self.stats["disagreements"] += 1
            batch.needed = len(ids) + 1
            return
        self._park(batch, "no agreeing pair" if not pairs else "conflicting agreeing pairs")

    def _park(self, batch: _Batch, why: str) -> None:
        self.stats["parked"] += 1
        self.parked[batch.job_id] += 1
        reason = (f"batch {batch.id}: {why} among {sorted(batch.scores)}; its records stay "
                  "pending")
        logger.error("eval audit of job %s parked: %s", batch.job_id, reason)
        self.attention[batch.job_id].append(reason)
        del self.attention[batch.job_id][:-10]
        if not batch.future.done():
            batch.future.set_exception(BatchParked(reason))

    def live_executors(self, model: ModelKey) -> list[dict]:
        now = self._clock()
        return [document for eid, document in self._documents.items()
                if now - self._seen.get(eid, float("-inf")) <= self._live
                and eid not in self.quarantined
                and (document.get("model_id"), document.get("model_revision")) == model
                and _placement(document) is not None]

    def status(self) -> dict:
        """Per model, what the fleet must know: ``executors_needed`` more
        executors (on providers and hosts distinct from the live ones) to
        decide the batches waiting; per job, whether it needs attention."""
        models = {}
        for model, queue in self._queues.items():
            waiting = [b for b in queue if not b.future.done()]
            live = self.live_executors(model)
            providers = {_placement(d)[0] for d in live}
            hosts = {_placement(d)[1] for d in live}
            required = max((b.needed for b in waiting), default=0)
            models[f"{model[0]}@{model[1]}"] = {
                "executors_needed": max(0, required - min(len(providers), len(hosts))),
                "live_executors": len(live), "live_providers": len(providers),
                "waiting_batches": len(waiting),
                "awaiting_third_scorer": sum(1 for b in waiting if b.needed > 2)}
        jobs = {job_id: {"needs_attention": True, "parked_batches": self.parked[job_id],
                         "reasons": list(reasons)}
                for job_id, reasons in self.attention.items() if reasons}
        return {"schema": "reliquary/eval-control-status/v1", "updated_at": self._clock(),
                "models": models, "jobs": jobs,
                "quarantined": sorted(self.quarantined), "stats": dict(self.stats)}

    async def quarantine(self, executor_id: str, reason: str) -> None:
        if executor_id in self.quarantined:
            return
        logger.error("eval audit executor %s quarantined: %s", executor_id, reason)
        self.quarantined.add(executor_id)
        self._directory.revoke_locally(executor_id)
        self.stats["quarantined"] += 1
        for lease_id, lease in list(self._leases.items()):
            if lease.executor_id == executor_id:
                del self._leases[lease_id]
                lease.batch.leased.discard(executor_id)
        for queue in self._queues.values():
            for batch in queue:
                if not batch.future.done() and executor_id in batch.scores:
                    # Its score no longer counts towards an agreement.
                    del batch.scores[executor_id]
        self._unwritten[executor_id] = reason
        await self._write_quarantines()
        for listener in self._listeners:
            task = asyncio.ensure_future(self._notify(listener, executor_id))
            self._background.add(task)
            task.add_done_callback(self._background.discard)

    @staticmethod
    async def _notify(listener, executor_id: str) -> None:
        try:
            await listener(executor_id)
        except Exception:
            logger.exception("re-audit after quarantining %s failed", executor_id)

    async def _write_quarantines(self) -> None:
        if self._quarantine_write is None:
            self._unwritten.clear()
            return
        for executor_id, reason in list(self._unwritten.items()):
            try:
                await self._quarantine_write(executor_id, reason)
                del self._unwritten[executor_id]
            except Exception:
                logger.exception("executor %s quarantine not written yet; retrying", executor_id)

    async def sweep(self) -> None:
        """Expire leases (striking their executor) and retry unwritten quarantines.
        There is no local fallback: an unscored batch waits."""
        now = self._clock()
        for lease_id, lease in list(self._leases.items()):
            if lease.expires_at <= now:
                del self._leases[lease_id]
                lease.batch.leased.discard(lease.executor_id)
                self._strikes[lease.executor_id] += 1
                if self._strikes[lease.executor_id] >= self._strikes_limit:
                    await self.quarantine(lease.executor_id,
                                          f"{self._strikes_limit} leases expired in a row")
        for queue in self._queues.values():
            while queue and queue[0].future.done():
                queue.popleft()
        if self._unwritten:
            await self._write_quarantines()

    async def write_heartbeats(self) -> None:
        """Each executor's last contact and ``{"leases", "loaded"}`` into the
        registry: at once when ``loaded`` or ``leases`` changed, otherwise at
        most once per ``HEARTBEAT_WRITE_SECONDS``, as the corpus control does."""
        if self._heartbeat_write is None:
            return
        for executor_id, seen in list(self._seen.items()):
            detail = self.heartbeat_detail(executor_id)
            written = self._written.get(executor_id)
            if written is not None and (written[0] == seen or (
                    seen - written[0] < HEARTBEAT_WRITE_SECONDS and written[1] == detail)):
                continue
            try:
                await self._heartbeat_write(executor_id, seen, detail)
                self._written[executor_id] = (seen, detail)
            except Exception:
                logger.exception("heartbeat of executor %s not written", executor_id)

    async def run(self, *, sweep_seconds: float = 2.0, write_status=None, status_of=None,
                  status_seconds: float = STATUS_SECONDS) -> None:
        last_status = float("-inf")
        while True:
            try:
                await self._directory.maybe_refresh()
                await self.sweep()
                await self.write_heartbeats()
                if write_status is not None and self._clock() - last_status >= status_seconds:
                    document = await status_of() if status_of is not None else self.status()
                    await write_status(document)
                    last_status = self._clock()
            except Exception:
                logger.exception("eval audit dispatcher sweep failed; retrying")
            await asyncio.sleep(sweep_seconds)


class PoolView:
    """One job's window on the dispatcher: its model's pool and its own proof,
    in the shape ``CorpusAuditor`` uses a remote dispatcher."""

    def __init__(self, dispatcher: PairedAuditDispatcher, model: ModelKey, proof,
                 job_id: str = "") -> None:
        self._dispatcher, self._model, self._proof = dispatcher, model, proof
        self._job_id = job_id
        self._listeners: list = []

    def connected(self) -> bool:
        # Always remote: there is no local GPU to fall back on.
        return True

    def subscribe(self, listener) -> None:
        self._listeners.append(listener)
        self._dispatcher.subscribe(listener)

    def close(self) -> None:
        """The job is gone: no re-audit of it ever starts, its batches leave."""
        for listener in self._listeners:
            self._dispatcher.unsubscribe(listener)
        self._listeners.clear()
        self._dispatcher.forget_job(self._job_id)

    async def score(self, items: Sequence[dict]):
        return await self._dispatcher.score(self._model, self._proof, items, self._job_id)


class _VocabularyOnly:
    """What the auditor reads of a model when it has none: the vocabulary size."""

    def __init__(self, vocab_size: int) -> None:
        self._embeddings = SimpleNamespace(num_embeddings=int(vocab_size))

    def get_input_embeddings(self):
        return self._embeddings


def eval_auditor(**kwargs):
    """A ``CorpusAuditor`` that never touches a GPU: every batch, re-audits
    included, goes to an executor pair; ``scored_by`` lists both executors."""
    from reliquary.validator.corpus_audit import outcome_from_scores
    from reliquary.validator.corpus_auditor import CorpusAuditor

    class EvalAuditor(CorpusAuditor):
        async def _forward(self, records, *, local: bool = False):
            results, items = await asyncio.to_thread(self._prepare, records)
            scores = await self._remote.score(
                [{"tokens": tokens, "prompt_len": n, "proofs": proofs}
                 for _, _, tokens, n, proofs in items])
            outcomes, scored_by = {}, {}
            for (i, c_idx, *_), (status, chunks, executors) in zip(items, scores):
                outcomes[i, c_idx] = outcome_from_scores(status, chunks, self._proof)
                scored_by.setdefault(i, set()).update(executors or ())
            judged = self._aggregate(records, results, outcomes)
            for i, executors in scored_by.items():
                judged[i] = {**judged[i], "scored_by": sorted(executors)}
            return judged

        async def _audit_outcomes(self, records, *, local: bool = False):
            # A parked batch is neither the miner's fault nor a validator
            # error: its records stay pending, untried, and its job is flagged.
            try:
                return await self._forward(records, local=local)
            except BatchParked as exc:
                for record in records:
                    for sid, known in zip(self._current_ids, self._current_records):
                        if known is record:
                            self.parked_ids.add(sid)
                return [_Parked(str(exc)) for _ in records]
            except (ValueError, RuntimeError) as exc:
                return [str(exc) for _ in records]

        async def _audit_records(self, ids, records, draws):
            self._current_ids, self._current_records = list(ids), list(records)
            before = set(self.parked_ids)
            try:
                return await super()._audit_records(ids, records, draws)
            finally:
                # Parked records were counted as validator errors above: undone.
                parked_now = len(self.parked_ids - before)
                self._validator_errors = max(0, self._validator_errors - parked_now)
                self._current_ids, self._current_records = [], []

        async def judge_many(self, submission_ids):
            # A parked record is not tried again until an operator acts.
            await super().judge_many([s for s in submission_ids if s not in self.parked_ids])

        # -- an eval job stays completable: a failure reopens its slot ------

        async def _record_failure(self, submission_id: str) -> None:
            from reliquary.validator.corpus_service import record_prompt_failure

            if self.job is None or self.job_store is None:
                return
            try:
                record = await self._records.read_submission(self._job_id, submission_id)
                if record is None:
                    return
                outcome = await record_prompt_failure(
                    self.job_store, self.job, int(record["prompt_index"]), submission_id)
                if outcome is False:
                    logger.warning("eval job %s: prompt %s exhausted its attempts",
                                   self._job_id, record["prompt_index"])
            except Exception:
                # Retried at the next start's reconcile; never stops the auditor.
                logger.exception("eval job %s: failure of %s not recorded in the ledger",
                                 self._job_id, submission_id[:12])

        async def _write(self, submission_id, verdict):
            stored, written = await super()._write(submission_id, verdict)
            if not (stored or verdict).get("passed"):
                await self._record_failure(submission_id)
            return stored, written

        async def reaudit_executor(self, executor_id):
            failed = await super().reaudit_executor(executor_id)
            for submission_id in failed:
                await self._record_failure(submission_id)
            return failed

        async def reconcile_failures(self) -> int:
            """Every failed or voided submission recorded in the ledger (a crash
            between a verdict and its ledger write would otherwise leave the
            prompt without its slot)."""
            ids = list(await self._records.list_verdict_ids(self._job_id))
            lister = getattr(self._records, "list_voided_ids", None)
            voided = set(await lister(self._job_id)) if lister is not None else set()
            failed = []
            for submission_id in ids:
                verdict = await self._records.read_verdict(self._job_id, submission_id)
                if verdict and (not verdict.get("passed") or submission_id in voided):
                    failed.append(submission_id)
            for submission_id in failed:
                await self._record_failure(submission_id)
            return len(failed)

        async def run(self):
            try:
                await self.reconcile_failures()
            except Exception:
                logger.exception("eval job %s: failure reconcile failed", self._job_id)
            await super().run()

    vocab_size = kwargs.pop("vocab_size")
    job = kwargs.pop("job", None)
    job_store = kwargs.pop("job_store", None)
    auditor = EvalAuditor(model=_VocabularyOnly(vocab_size), **kwargs)
    auditor.parked_ids = set()
    auditor._current_ids, auditor._current_records = [], []
    auditor.job, auditor.job_store = job, job_store
    return auditor


class _Parked(str):
    """An outcome that is neither a verdict nor a validator error."""


def build_eval_executor_router(*, dispatcher: PairedAuditDispatcher,
                               directory: EvalExecutorDirectory,
                               qualifications=None) -> APIRouter:
    """``/corpus/internal/eval-audit/...``: claim (an audit lease, or with
    ``kind`` a qualification), result, heartbeat, behind the executor token."""
    router = APIRouter()

    def authenticated(request: Request, executor_id: str | None = None) -> dict:
        header = request.headers.get("authorization", "")
        token = header[len("Bearer "):] if header.startswith("Bearer ") else None
        document, refusal = directory.authenticate(token, executor_id)
        if document is None:
            raise HTTPException(status_code=401, detail=refusal)
        return document

    @router.post(f"{EVAL_AUDIT_PREFIX}/claim")
    async def claim(body: EvalClaimRequest, request: Request):
        document = authenticated(request, body.executor_id)
        if (body.model_id, body.model_revision) != (document["model_id"],
                                                    document["model_revision"]):
            raise HTTPException(status_code=409, detail="wrong_model")
        # Every contact counts as one, qualification included.
        dispatcher.heartbeat(document["executor_id"], document)
        if body.kind in ("qualify", "any") and qualifications is not None:
            lease = await qualifications.claim(document)
            if lease is not None:
                return lease
        if body.kind in ("audit", "any"):
            try:
                lease = dispatcher.claim(document)
            except LeaseRefused as exc:
                raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
            if lease is not None:
                return lease
        return Response(status_code=204)

    @router.post(f"{EVAL_AUDIT_PREFIX}/heartbeat")
    async def heartbeat(body: HeartbeatRequest, request: Request) -> dict:
        document = authenticated(request, body.executor_id)
        dispatcher.heartbeat(document["executor_id"], document, body.detail or {})
        return {"executor_id": document["executor_id"], "model_id": document["model_id"],
                "model_revision": document["model_revision"]}

    @router.post(f"{EVAL_AUDIT_PREFIX}/{{lease_id}}/result")
    async def result(lease_id: str, request: Request) -> dict:
        document = authenticated(request)
        dispatcher.heartbeat(document["executor_id"], document)
        body = await request.json()
        try:
            if qualifications is not None and qualifications.lease_of(lease_id) is not None:
                record = await qualifications.result(
                    document, lease_id, QualifyResult.model_validate(body))
                return {"lease_id": lease_id, "outcome": record["status"]}
            outcome = await dispatcher.result(document["executor_id"], lease_id,
                                              AuditResult.model_validate(body))
        except LeaseRefused as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)[:500]) from exc
        return {"lease_id": lease_id, "outcome": outcome}

    return router


# ---------------------------------------------------------------------------
# The process
# ---------------------------------------------------------------------------


class EvalArchives:
    """The settler's archives for eval tasks: written only under a task this
    process wired and named order-eval- (RELIQUARY_TASK_ID lists no eval task:
    they are all wired hot)."""

    def __init__(self, *, served: Callable[[], Any], upload=None, other_max=None) -> None:
        from reliquary.validator.corpus_settlement import R2Archives

        self._served = served
        self._other_max = other_max or R2Archives().other_max
        self._upload = upload

    async def other_max(self, task_id: str) -> int | None:
        return await self._other_max(task_id)

    async def write(self, task_id: str, window: int, data: dict) -> None:
        from reliquary.eval.prompt_source import is_eval_job_id

        if not is_eval_job_id(task_id) or task_id not in set(self._served()):
            raise RuntimeError(f"task {task_id!r} is not an eval task this process serves; "
                               "refusing to archive")
        if self._upload is None:
            from reliquary.infrastructure import storage

            await storage.upload_window_dataset(window, data, task_id=task_id)
        else:
            await self._upload(window, data, task_id)


async def write_control_status(document: dict, **client_kwargs) -> None:
    """The eval control's status (``executors_needed`` per model, jobs needing
    attention), overwritten in the subnet bucket for the admin route."""
    from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict, _get, _put

    body = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    for _ in range(3):
        _, etag = await _get(EVAL_CONTROL_STATUS_KEY, **dict(client_kwargs))
        try:
            await _put(EVAL_CONTROL_STATUS_KEY, body, etag, **dict(client_kwargs))
            return
        except CorpusStoreConflict:
            continue


async def read_control_status(**client_kwargs) -> dict | None:
    from reliquary.infrastructure.corpus_job_store import _get

    body, _ = await _get(EVAL_CONTROL_STATUS_KEY, **dict(client_kwargs))
    return None if body is None else json.loads(body)


def eval_job_refusal(entry, job) -> str | None:
    """Why the eval control will not serve a registry entry, or None."""
    from reliquary.eval.prompt_source import is_eval_job_id, is_eval_source
    from reliquary.protocol.profiles import profile_from_contract, toploc_proof

    if not is_eval_job_id(entry.job_id):
        return "not an evaluation job"
    if not is_eval_source(job.prompt_source):
        return f"prompt source {job.prompt_source!r} is not an eval set"
    if getattr(entry, "contract", None) is None:
        return "it carries no contract"
    profile = profile_from_contract(entry.contract)
    proof = toploc_proof(profile)
    if proof is None or proof.mode != "enforce":
        return "its contract names no enforced toploc proof"
    if (profile.model_id, profile.model_revision) != (job.checkpoint_repo,
                                                      job.checkpoint_revision):
        return "its contract's model is not the job's checkpoint"
    return None


# A tokenizer's files: never the weights, never a stray large text file.
TOKENIZER_PATTERNS = ["*.json", "*.model", "*.tiktoken", "merges.txt", "vocab.txt", "*.jinja"]


def _model_files(repo: str, revision: str) -> str:
    """The model's small files at the pinned revision (public repos only: a
    gated or missing repo is a refusal, never retried as transient)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

    try:
        return snapshot_download(repo, revision=revision, token=False,
                                 allow_patterns=TOKENIZER_PATTERNS)
    except (GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError) as exc:
        raise ValueError(f"{repo}@{revision} is not a public model at that revision: "
                         f"{type(exc).__name__}") from exc


def load_cpu_tokenizer(repo: str, revision: str):
    """A model's tokenizer and vocabulary size (the embedding rows the config
    declares, the auditor's bound on token ids), without its weights."""
    import json
    import os

    from reliquary.shared.modeling import load_tokenizer

    directory = _model_files(repo, revision)
    config = json.loads(open(os.path.join(directory, "config.json")).read())
    vocab = config.get("vocab_size") or (config.get("text_config") or {}).get("vocab_size")
    if not vocab:
        raise ValueError(f"{repo}@{revision} declares no vocab_size")
    return load_tokenizer(directory), int(vocab)


def model_facts(repo: str, revision: str) -> dict:
    """The model's architecture and eos, read here from its own files (CPU):
    never taken from an executor."""
    import json
    import os

    from reliquary.shared.modeling import load_tokenizer

    directory = _model_files(repo, revision)
    config = json.loads(open(os.path.join(directory, "config.json")).read())
    architectures = config.get("architectures") or []
    eos = load_tokenizer(directory).eos_token_id
    if not architectures or eos is None:
        raise ValueError(f"{repo}@{revision} names no architecture or eos")
    return {"architecture": str(architectures[0]), "eos_token_id": int(eos)}


def build_eval_control(*, store, records, dispatcher: PairedAuditDispatcher,
                       directory: EvalExecutorDirectory, verify_signature,
                       verify_skip_signature=None, tokenizer_for=load_cpu_tokenizer,
                       qualifications=None, registration=None, settle_archives=None,
                       read_entries=None, clock: Callable[[], float] = time.time,
                       refresh_every_seconds: float = 60.0):
    """The app and the job set of the eval control. Each wired job gets its own
    tokenizer, renderer, router, auditor (an executor pair per batch) and
    settler; ``read_entries`` makes the set hot."""
    from fastapi import FastAPI

    from reliquary.eval.prompt_source import job_prompt_lines, parse_eval_source
    from reliquary.protocol.profiles import profile_from_contract, toploc_proof
    from reliquary.validator.corpus_hot_jobs import OTHER_MODEL, CorpusJobSet, job_drained
    from reliquary.validator.corpus_job_status import JobStats
    from reliquary.validator.corpus_service import (
        CorpusJobRoutes,
        build_corpus_jobs_router,
        build_corpus_router,
        migrate_ledgers_at_startup,
        prompt_job_for_spec,
        renderer_for_job,
    )
    from reliquary.validator.corpus_settlement import CorpusSettler
    from reliquary.validator.corpus_validator import build_corpus_audit_wiring

    tokenizers: dict[ModelKey, tuple[Any, int]] = {}
    served: dict[str, Any] = {}

    def router_for(w):
        return build_corpus_router(
            job_id=str(w.entry.job_id), store=store, tokenizer=w.tokenizer,
            renderer=w.renderer, verify_signature=verify_signature,
            verify_skip_signature=verify_skip_signature, prompt_job_for=w.prompt_job_for,
            records=records, on_accepted=w.on_accepted,
            proof_chunk_tokens=w.proof.chunk_tokens, vocab_size=w.vocab_size,
            is_banned=w.is_banned, registration=registration, seen_index=w.seen_index)

    async def wire(entry, cap, job):
        refusal = eval_job_refusal(entry, job)
        if refusal is not None:
            raise ValueError(refusal)
        key = (job.checkpoint_repo, job.checkpoint_revision)
        if key not in tokenizers:
            tokenizers[key] = await asyncio.to_thread(tokenizer_for, *key)
        tokenizer, vocab_size = tokenizers[key]
        profile = profile_from_contract(entry.contract)
        proof = toploc_proof(profile)

        def encode(text: str) -> list[int]:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            return list(getattr(encoded, "ids", encoded))

        renderer = renderer_for_job(job, encode, tokenizer=tokenizer, profile=profile)
        prompt_job_for = functools.partial(prompt_job_for_spec, profile=profile)
        await asyncio.to_thread(prompt_job_for, job)  # the set's prompts, read and checked
        seen_index = await migrate_ledgers_at_startup(store, job)
        params, miner_states, is_banned, beacon, round_at = build_corpus_audit_wiring(
            entry=entry, job=job, records=records)
        w = SimpleNamespace(entry=entry, cap=cap, job=job, tokenizer=tokenizer,
                            vocab_size=vocab_size, proof=proof, renderer=renderer,
                            prompt_job_for=prompt_job_for, seen_index=seen_index,
                            is_banned=is_banned, stats=JobStats())
        w.auditor = eval_auditor(
            job_id=job.job_id, records=records, tokenizer=tokenizer, proof=proof,
            params=params, miner_states=miner_states, beacon=beacon, round_at=round_at,
            on_verdict=w.stats.observe, vocab_size=vocab_size, job=job, job_store=store,
            remote=dispatcher.view(job.checkpoint_repo, job.checkpoint_revision, proof,
                                   job.job_id))
        w.settler = CorpusSettler(task_id=entry.task_id, job_id=job.job_id, cap=cap,
                                  records=records, archives=settle_archives,
                                  on_settled=w.stats.settled)

        def on_accepted(submission_id: str) -> None:
            w.stats.accepted()
            w.auditor.enqueue(submission_id)

        w.on_accepted = on_accepted
        served[job.job_id] = w
        return w

    async def settle_forever(w, every: float = 60.0) -> None:
        while True:
            try:
                await w.settler.settle_once()
            except Exception:
                logger.exception("eval settlement of %s failed; retrying", w.job.job_id)
            await asyncio.sleep(every)

    async def read_job(job_id):
        job, _ = await store.read_job(job_id)
        return job

    def admit(entry, job):
        refusal = eval_job_refusal(entry, job)
        return None if refusal is None else (OTHER_MODEL, refusal)

    def screen(entry):
        from reliquary.eval.prompt_source import is_eval_job_id

        # Not an eval job: never even read its manifest.
        if not is_eval_job_id(getattr(entry, "job_id", "")):
            return OTHER_MODEL, "not an evaluation job"
        return None

    def unwired(w) -> None:
        """A drained job releases everything it held: its wiring, its pool view
        and listeners, its prompts, and its model's tokenizer once unused."""
        from reliquary.eval.prompt_source import forget_eval_prompts, parse_eval_source

        served.pop(w.job.job_id, None)
        w.auditor._remote.close()
        forget_eval_prompts(parse_eval_source(w.job.prompt_source))
        key = (w.job.checkpoint_repo, w.job.checkpoint_revision)
        if not any((o.job.checkpoint_repo, o.job.checkpoint_revision) == key
                   for o in served.values()):
            tokenizers.pop(key, None)

    async def guarded(w, coroutine):
        """One job's background work: a failure flags the job, never the process."""
        try:
            await coroutine
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("eval job %s: its background work failed", w.job.job_id)
            dispatcher.attention[w.job.job_id].append(
                f"background work stopped: {type(exc).__name__}: {exc}"[:300])

    routes = CorpusJobRoutes()
    app = FastAPI()
    app.include_router(build_corpus_jobs_router(routes, legacy=False))

    @app.get("/corpus/jobs/{job_id}/eval-prompts")
    async def eval_prompts(job_id: str) -> Response:
        w = served.get(job_id)
        if w is None or job_id not in routes.routers:
            raise HTTPException(status_code=404, detail="corpus_job_not_served")
        body = await asyncio.to_thread(job_prompt_lines, parse_eval_source(w.job.prompt_source))
        return Response(content=body, media_type="application/x-ndjson")

    @app.get("/corpus/jobs/{job_id}/contract")
    async def eval_job_contract(job_id: str) -> dict:
        w = served.get(job_id)
        if w is None or job_id not in routes.routers:
            raise HTTPException(status_code=404, detail="corpus_job_not_served")
        return w.entry.contract

    @app.get("/corpus/jobs/{job_id}/status")
    async def eval_job_status(job_id: str) -> dict:
        status = await job_set.status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="corpus_job_not_served")
        return status

    app.include_router(build_eval_executor_router(dispatcher=dispatcher, directory=directory,
                                                  qualifications=qualifications))
    job_set = CorpusJobSet(
        routes=routes, router_for=router_for, wire=wire,
        jobs_of=lambda w: [guarded(w, w.auditor.run()), guarded(w, settle_forever(w))],
        read_entries=read_entries, read_job=read_job, admit=admit, screen=screen,
        on_unwired=unwired, refresh_every_seconds=refresh_every_seconds,
        drained=lambda w: job_drained(auditor=w.auditor, records=records,
                                      job_id=w.job.job_id),
        clock=clock)
    async def control_status() -> dict:
        """The dispatcher's status, with each served job's completeness."""
        document = dispatcher.status()
        for job_id in list(served):
            try:
                status = await job_set.status(job_id)
            except Exception:
                logger.warning("eval status of %s unavailable", job_id, exc_info=True)
                continue
            if status is None:
                continue
            entry = document["jobs"].setdefault(
                job_id, {"needs_attention": False, "parked_batches": 0, "reasons": []})
            entry.update({key: status.get(key) for key in (
                "state", "prompts_total", "prompts_full", "prompts_complete",
                "prompts_exhausted", "complete")})
        return document

    app.state.control_status = control_status
    app.state.corpus_jobs = job_set
    app.state.eval_served = served
    app.state.eval_tokenizers = tokenizers
    app.state.dispatcher = dispatcher
    return app, job_set


async def run_eval_control(*, netuid: int, http_host: str, http_port: int,
                           registration_gate: bool = True,
                           refresh_every_seconds: float | None = None) -> None:
    """Serve every active ``order-eval-`` corpus task of the registry, hot."""
    import uvicorn

    from reliquary.eval.qualification import QualificationQueue, QualificationStore
    from reliquary.eval.storage import SubnetEvalStore, subnet_key
    from reliquary.infrastructure import corpus_executor_store as executor_store
    from reliquary.infrastructure import task_registry_store as registry_store
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.protocol.signatures import (
        verify_corpus_signature,
        verify_corpus_skip_signature,
    )
    registered = None
    if registration_gate:
        from reliquary.validator.corpus_registration import (
            RegisteredHotkeys,
            load_registered_hotkeys,
        )

        registered = RegisteredHotkeys(load=lambda: load_registered_hotkeys(netuid))
        await registered.refresh()
    directory = EvalExecutorDirectory()
    dispatcher = PairedAuditDispatcher(
        directory=directory,
        quarantine=lambda executor_id, reason: executor_store.set_executor_status(
            executor_id, "quarantined", reason=reason, scope="eval"),
        record_heartbeat=lambda executor_id, at, detail: executor_store.record_heartbeat(
            executor_id, at=at, detail=detail))
    subnet = SubnetEvalStore()

    async def read_prompts(set_id):
        return await subnet.get_bytes(subnet_key(set_id, "prompts.jsonl"))

    async def facts(repo, revision):
        return await asyncio.to_thread(model_facts, repo, revision)

    qualifications = QualificationQueue(store=QualificationStore(), read_prompts=read_prompts,
                                        model_facts=facts)

    async def read_entries():
        entries, _ = await registry_store.read_registry(strict=True)
        return entries

    job_set = None
    archives = EvalArchives(served=lambda: job_set.task_ids() if job_set else ())
    app, job_set = build_eval_control(
        store=BucketJobStore(), records=BucketRecordStore(), dispatcher=dispatcher,
        directory=directory, verify_signature=verify_corpus_signature,
        verify_skip_signature=verify_corpus_skip_signature, qualifications=qualifications,
        registration=registered.reason if registered is not None else None,
        settle_archives=archives, read_entries=read_entries,
        refresh_every_seconds=refresh_every_seconds or 60.0)

    async def refresh_qualifications():
        while True:
            try:
                await qualifications.refresh()
            except Exception:
                logger.exception("qualification queue unreadable; retrying")
            await asyncio.sleep(30.0)

    background = [dispatcher.run(write_status=write_control_status,
                                 status_of=app.state.control_status), refresh_qualifications(),
                  job_set.run()]
    if registered is not None:
        background.append(registered.refresh_forever())
    server = uvicorn.Server(uvicorn.Config(app, host=http_host, port=http_port, log_level="info"))
    await asyncio.gather(server.serve(), *background)


__all__ = [
    "BatchParked",
    "EVAL_AUDIT_PREFIX",
    "EVAL_CONTROL_STATUS_KEY",
    "EvalArchives",
    "EvalExecutorDirectory",
    "MAX_SCORERS",
    "PairedAuditDispatcher",
    "PoolView",
    "build_eval_control",
    "build_eval_executor_router",
    "eval_auditor",
    "eval_job_refusal",
    "load_cpu_tokenizer",
    "model_facts",
    "read_control_status",
    "run_eval_control",
    "write_control_status",
]
