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
# Scorers of one batch before it is given up as undecidable (2, a third, then
# two more chances to find a majority).
MAX_SCORERS = 5

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
        self.quarantined: set[str] = set()
        self.stats = collections.Counter()

    # -- the auditors' side ------------------------------------------------

    def view(self, model_id: str, model_revision: str, proof) -> "PoolView":
        return PoolView(self, (model_id, model_revision), proof)

    def subscribe(self, listener: Callable[[str], Awaitable[Any]]) -> None:
        self._listeners.append(listener)

    async def score(self, model: ModelKey, proof, items: Sequence[dict]):
        """``(status, chunks, scored_by)`` per item; ``scored_by`` is the
        tuple of executors whose agreement decided it."""
        loop = asyncio.get_running_loop()
        units = []
        for indexes in _lease_units(items):
            batch = _Batch(id=next(self._ids), model=model, proof=proof,
                           items=[items[k] for k in indexes], future=loop.create_future())
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

    def heartbeat(self, executor_id: str) -> None:
        self._seen[executor_id] = self._clock()

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
        self.heartbeat(executor_id)
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
        if len(batch.scores) < 2:
            return
        ids = sorted(batch.scores)
        agreeing = {e: {o for o in ids if o != e and scores_agree(
            batch.scores[e], batch.scores[o], batch.proof)} for e in ids}
        majority = sorted(e for e, others in agreeing.items() if others)
        if len(ids) == 2 and majority:
            batch.future.set_result((batch.scores[ids[0]], tuple(ids)))
            self.stats["agreed"] += 1
            return
        if len(ids) >= 3 and majority:
            for minority in (e for e in ids if e not in majority):
                await self.quarantine(minority, f"disagreed with executors {majority} on batch "
                                                f"{batch.id}")
            if not batch.future.done():
                batch.future.set_result((batch.scores[majority[0]], tuple(majority)))
                self.stats["settled_by_majority"] += 1
            return
        # Nobody agrees yet: one more executor, up to a bound.
        self.stats["disagreements"] += 1
        batch.needed = len(ids) + 1
        if batch.needed > MAX_SCORERS and not batch.future.done():
            batch.future.set_exception(RuntimeError(
                f"batch {batch.id}: {len(ids)} executors and no two agree"))

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
        if self._heartbeat_write is None:
            return
        for executor_id, seen in list(self._seen.items()):
            try:
                await self._heartbeat_write(executor_id, seen, {})
            except Exception:
                logger.exception("heartbeat of executor %s not written", executor_id)

    async def run(self, *, sweep_seconds: float = 2.0) -> None:
        while True:
            try:
                await self._directory.maybe_refresh()
                await self.sweep()
            except Exception:
                logger.exception("eval audit dispatcher sweep failed; retrying")
            await asyncio.sleep(sweep_seconds)


class PoolView:
    """One job's window on the dispatcher: its model's pool and its own proof,
    in the shape ``CorpusAuditor`` uses a remote dispatcher."""

    def __init__(self, dispatcher: PairedAuditDispatcher, model: ModelKey, proof) -> None:
        self._dispatcher, self._model, self._proof = dispatcher, model, proof

    def connected(self) -> bool:
        # Always remote: there is no local GPU to fall back on.
        return True

    def subscribe(self, listener) -> None:
        self._dispatcher.subscribe(listener)

    async def score(self, items: Sequence[dict]):
        return await self._dispatcher.score(self._model, self._proof, items)


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

    vocab_size = kwargs.pop("vocab_size")
    return EvalAuditor(model=_VocabularyOnly(vocab_size), **kwargs)


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
        dispatcher.heartbeat(document["executor_id"])
        return {"executor_id": document["executor_id"], "model_id": document["model_id"],
                "model_revision": document["model_revision"]}

    @router.post(f"{EVAL_AUDIT_PREFIX}/{{lease_id}}/result")
    async def result(lease_id: str, request: Request) -> dict:
        document = authenticated(request)
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
        from reliquary.eval.prompt_source import EVAL_JOB_PREFIX

        if not task_id.startswith(EVAL_JOB_PREFIX) or task_id not in set(self._served()):
            raise RuntimeError(f"task {task_id!r} is not an eval task this process serves; "
                               "refusing to archive")
        if self._upload is None:
            from reliquary.infrastructure import storage

            await storage.upload_window_dataset(window, data, task_id=task_id)
        else:
            await self._upload(window, data, task_id)


def eval_job_refusal(entry, job) -> str | None:
    """Why the eval control will not serve a registry entry, or None."""
    from reliquary.eval.prompt_source import EVAL_JOB_PREFIX, is_eval_source
    from reliquary.protocol.profiles import profile_from_contract, toploc_proof

    if not str(entry.job_id or "").startswith(EVAL_JOB_PREFIX):
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


def load_cpu_tokenizer(repo: str, revision: str):
    """A model's tokenizer and vocabulary size, without its weights."""
    import json
    import os

    from huggingface_hub import snapshot_download

    from reliquary.shared.modeling import load_tokenizer

    directory = snapshot_download(repo, revision=revision, token=False,
                                  allow_patterns=["*.json", "*.model", "*.tiktoken", "*.txt",
                                                  "*.jinja"])
    config = json.loads(open(os.path.join(directory, "config.json")).read())
    vocab = config.get("vocab_size") or (config.get("text_config") or {}).get("vocab_size")
    if not vocab:
        raise ValueError(f"{repo}@{revision} declares no vocab_size")
    return load_tokenizer(directory), int(vocab)


def build_eval_control(*, store, records, dispatcher: PairedAuditDispatcher,
                       directory: EvalExecutorDirectory, verify_signature,
                       verify_skip_signature=None, tokenizer_for=load_cpu_tokenizer,
                       qualifications=None, registration=None, settle_archives=None,
                       read_entries=None, clock: Callable[[], float] = time.time):
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
            on_verdict=w.stats.observe, vocab_size=vocab_size,
            remote=dispatcher.view(job.checkpoint_repo, job.checkpoint_revision, proof))
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
        jobs_of=lambda w: [w.auditor.run(), settle_forever(w)],
        read_entries=read_entries, read_job=read_job, admit=admit,
        drained=lambda w: job_drained(auditor=w.auditor, records=records,
                                      job_id=w.job.job_id),
        clock=clock)
    app.state.corpus_jobs = job_set
    app.state.eval_served = served
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
            executor_id, "quarantined", reason=reason),
        record_heartbeat=lambda executor_id, at, detail: executor_store.record_heartbeat(
            executor_id, at=at, detail=detail))
    subnet = SubnetEvalStore()

    async def read_prompts(set_id):
        return await subnet.get_bytes(subnet_key(set_id, "prompts.jsonl"))

    qualifications = QualificationQueue(store=QualificationStore(), read_prompts=read_prompts)

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
        settle_archives=archives, read_entries=read_entries)

    async def refresh_qualifications():
        while True:
            try:
                await qualifications.refresh()
            except Exception:
                logger.exception("qualification queue unreadable; retrying")
            await asyncio.sleep(30.0)

    background = [dispatcher.run(), refresh_qualifications(), job_set.run()]
    if registered is not None:
        background.append(registered.refresh_forever())
    server = uvicorn.Server(uvicorn.Config(app, host=http_host, port=http_port, log_level="info"))
    await asyncio.gather(server.serve(), *background)


__all__ = [
    "EVAL_AUDIT_PREFIX",
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
    "run_eval_control",
]
