"""The GPU-less eval control: every evaluation job, whatever its model, in one process.

It loads no model. Each job has its own tokenizer (CPU) and its own executor
pool keyed by ``model@revision``. Every audit batch is scored by one executor
(``RELIQUARY_ORDER_AUDIT_SCORERS``, default 1), whose scores decide it, as the
validator's own GPU decides the corpus audits. With two or more scorers, on
distinct providers and hosts:
- agreement (the same decision per item, every measure within R3's drift
  tolerance) decides the batch;
- disagreement sends it to one more executor; whoever agrees with nobody is
  quarantined, and the passes it co-signed are re-audited;
- fewer scorers than required never decide: the batch waits.

Executors reach it on ``/corpus/internal/eval-audit/...``; miners on
``/corpus/jobs/order-eval-.../...``. The corpus control's behaviour is untouched:
it never wires an ``order-eval-`` job, and this process serves nothing else.
"""

from __future__ import annotations

import asyncio
import collections
import contextvars
import functools
import hmac
import json
import itertools
import logging
import math
import random
import secrets
import statistics
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from reliquary.corpus.eval_draw import has_draw
from reliquary.corpus.job import is_signed_sandbox
from reliquary.eval.qualify_protocol import EvalClaimRequest, QualifyResult
from reliquary.protocol.toploc import ChunkResult
from reliquary.validator.corpus_audit_protocol import (
    AUDIT_PROTOCOL,
    AUDIT_PROTOCOL_V3,
    AUDIT_PROTOCOL_V4,
    DRAW_PROTOCOLS,
    ITEM_ERROR,
    ITEM_OK,
    AuditLease,
    AuditResult,
    HeartbeatRequest,
    draw_protocol,
)
from reliquary.validator.corpus_audit_remote import (
    AUDIT_LEASE_SECONDS,
    EXECUTOR_LIVE_SECONDS,
    MAX_LEASES_PER_EXECUTOR,
    REGISTRY_REFRESH_SECONDS,
    LeaseRefused,
    _lease_units,
    _misfit,
    converted_scores,
    scores_agree,
    token_sha256,
)
from reliquary.validator.corpus_periods import PERIOD_SECONDS

logger = logging.getLogger(__name__)

EVAL_AUDIT_PREFIX = "/corpus/internal/eval-audit"
# Scorers of one batch: one by default; with a pair, a third on disagreement. No agreement then:
# the batch is parked (its records stay pending) and its job needs attention.
MAX_SCORERS = 3
SCORERS_ENV = "RELIQUARY_ORDER_AUDIT_SCORERS"
DEFAULT_SCORERS = 1


def job_audit_protocol(job) -> str | None:
    """The lease protocol a job's samples are audited in: v4 for a greedy draw (top_k 1), v3 for another draw,
    None (any) without one."""
    if not has_draw(job):
        return None
    from reliquary.corpus.eval_draw import job_warp

    return AUDIT_PROTOCOL_V4 if job_warp(job).top_k == 1 else AUDIT_PROTOCOL_V3


def order_audit_scorers(env=None) -> int:
    """How many executors score each audit batch (1, 2 or 3)."""
    import os

    raw = (os.environ if env is None else env).get(SCORERS_ENV, "").strip()
    value = int(raw) if raw else DEFAULT_SCORERS
    if not 1 <= value <= MAX_SCORERS:
        raise ValueError(f"{SCORERS_ENV} must be 1 to {MAX_SCORERS}, got {value}")
    return value
# Store connections shared by the miners' intake and every order's auditors (each pass reads READ_CONCURRENCY
# records and writes WRITE_CONCURRENCY verdicts at once). At 64 on 2026-11-19 a record write took ~1.7 s and a
# pass's read ~1.7 s against 0.2 s alone.
ORDER_POOL_CONNECTIONS = 256
STATUS_SECONDS = 30.0
# A judge process's slot reopen that lost its ledger race: tried again after this, doubling.
REOPEN_RETRY_SECONDS = 1.0
MAX_REOPEN_RETRY_SECONDS = 30.0
# The retry task of failures recorded together, among those of single submissions.
_REOPEN_BACKLOG = "*backlog"
# Why a job is still kept from miners (control status ``jobs[...]["held"]``).
HELD_NO_GATEWAY = "no_gateway"
HELD_AWAITING_AUDIT = "awaiting_audit"
# Bound on resolving an episode order's first task for its options digest.
GATEWAY_DIGEST_TIMEOUT_S = 60.0
# Audit throughput is measured over this window: items scored per second an executor held a lease.
THROUGHPUT_WINDOW_SECONDS = 900.0
# Enough executors that the expected audits of a model's orders drain within this (env: DRAIN_ENV).
DRAIN_ENV = "RELIQUARY_ORDER_AUDIT_DRAIN_SECONDS"
DEFAULT_DRAIN_SECONDS = 1800.0
MAX_EXECUTORS_ENV = "RELIQUARY_ORDER_MAX_AUDIT_EXECUTORS"
DEFAULT_MAX_EXECUTORS = 3
# Fewer executors are wanted only once fewer were enough for this long (no rent/release flapping).
SCALE_DOWN_SECONDS = 600.0
# Busy seconds measured before a rate is trusted.
MIN_MEASURED_BUSY_SECONDS = 120.0
# A card that lost this many results in a row (leases expired) is set aside; a scored result clears the count
# only after CLEAN_STREAK of them, so a card failing one call in two does not keep its work.
LOST_RESULT_STRIKES = 2
CLEAN_STREAK = 5
# An expired lease is remembered this long. A card that answers it late, with a scored result, is slow, not
# lost: that lease's strike is taken back, unless the lease was already the longest one (2026-11-25: leases
# measured on faster cards were too short for the 9B cards, every card was set aside about 98 s after loading,
# and its late results were never timed, so the next card died the same way).
LATE_ANSWER_SECONDS = 900.0
# A batch no card can answer (2026-11-24: agentic trajectories of 95 and 70 turns, past the 64 spans an audit item
# carries; every executor refused the lease, every 9B card was set aside after two lost leases and destroyed, and the
# model's audits stopped for a day):
# - each item is checked against the executors' `AuditLease` when its call is made; a record with an item they would
#   refuse is parked alone, before any lease (`ItemsUnreadable`), and the rest of its call goes on;
# - a batch whose leases are lost (expired unanswered, or answered with a result that does not fit) is parked only on
#   evidence against the batch: witnessed losses on UNANSWERED_PROVIDERS_PARK distinct providers (one more per extra
#   scorer it needs). A loss is witnessed when its card delivered a measured result, for an undecided batch, on another
#   lease after this one was granted: that card works, the batch does not. An unwitnessed loss (a dead or slow card)
#   is no evidence. Every loss strikes its card, witnessed or not, as #493 does; once the batch is parked its witnessed
#   losses are taken back. Losses of a card quarantined since no longer count, and a provider's losses count towards
#   at most PARKS_PER_EXECUTOR parks per PARK_WINDOW_SECONDS: past that they only strike.
UNANSWERED_PROVIDERS_PARK = 2
PARKS_PER_EXECUTOR = 2
PARK_WINDOW_SECONDS = 3600.0
# A card that lost a batch's lease while it scored other work (a witnessed loss), or answered it with an executor
# error or a result that does not fit, is not leased that batch again for BENCH_SECONDS, doubling at each repeat up to
# MAX_BENCH_SECONDS: it cannot hold one batch for ever by answering errors and claiming it again, and a batch it cannot
# answer does not cost it a strike at each of its claims. A card that answers nothing is not benched: it is set aside
# after its second lost lease, as before.
BENCH_SECONDS = 300.0
MAX_BENCH_SECONDS = 3600.0
# A call of at most this many tokens is checked on the loop (one full lease: about 8 ms); a bigger one in a thread.
INLINE_CHECK_TOKENS = 262_144
# An audit lease lives twice the time the model's recent results took per token (or the card's own pace when it
# is slower), for its tokens and those the card already holds, plus LEASE_OVERHEAD_SECONDS, within
# [AUDIT_LEASE_FLOOR_SECONDS, the dispatcher's lease seconds]; the dispatcher's lease seconds until
# LEASE_TIMES_NEEDED results were timed within LEASE_TIMES_MAX_AGE_SECONDS. Only a scored result is timed, a
# late one included.
AUDIT_LEASE_FLOOR_SECONDS = 60.0
LEASE_OVERHEAD_SECONDS = 30.0
LEASE_TIMES_KEPT = 20
LEASE_TIMES_NEEDED = 3
LEASE_TIMES_MAX_AGE_SECONDS = 3600.0
# A parked record (neither a verdict nor a validator error: an item no executor can read, or a batch parked) is tried
# again after PARK_RETRY_SECONDS, doubling up to MAX_PARK_RETRY_SECONDS, or at once when its batch gets a measured
# answer; each retry audits it alone, in the background, so no judge pass waits on it. It holds its period's close two
# periods after its receipt at most, as a record whose arrival cannot be read (corpus_auditor.
# UNKNOWN_ARRIVAL_GIVE_UP_SECONDS); one parked with other records' items, until its first retry alone had
# SOLO_RETRY_SECONDS to decide it too, so no honest batch-mate leaves first. Then its period closes without it and it no
# longer holds its operator's unaudited passes as a sibling. At q < 1 its operator's verdicts of its period received up
# to its receipt still wait for its decision (`settle_hold`, the stake guarantee); its own verdict, decided after its
# period was settled without it, is paid within what is left of the period's cap (`settle_capped`).
PARK_RETRY_SECONDS = 1800.0
MAX_PARK_RETRY_SECONDS = 4 * 3600.0
PARKED_CLOSE_GIVE_UP_SECONDS = 2.0 * PERIOD_SECONDS
SOLO_RETRY_SECONDS = 600.0
PARK_UNREADABLE = "unreadable"
PARK_LOST = "parked"
# Records without a submission id in them (tests' fakes) are told apart by identity, the last KNOWN_RECORDS read.
KNOWN_RECORDS = 8192
HEARTBEAT_WRITE_SECONDS = 30.0
EVAL_CONTROL_STATUS_KEY = "reliquary/eval/control/status.json"
# The miners document every order shares: a hotkey's probation progress carries
# from one order to the next (no `order-` prefix: it is no order).
ORDER_TRUST_ID = "miner-trust-orders"
# How stale a hotkey's audit state may be when a lease request asks whether it is suspect (a lease request is
# frequent; a failure is acted on in the audit itself, this only stops new work).
SUSPECT_READ_MAX_AGE = 30.0


def _suspect_reader(miner_states, params):
    """``hotkey``'s suspicion end (unix seconds) while it is suspect on the job, else None."""
    from reliquary.corpus.audit_policy import effective_state

    async def suspect_until(hotkey: str) -> float | None:
        state, _ = await miner_states.state_at_most(hotkey, SUSPECT_READ_MAX_AGE)
        if effective_state(state, time.time(), params) == "suspect":
            return state.suspect_until
        return None

    return suspect_until


def order_audit_drain_seconds(env=None) -> float:
    import os

    raw = (os.environ if env is None else env).get(DRAIN_ENV, "").strip()
    value = float(raw) if raw else DEFAULT_DRAIN_SECONDS
    if not value > 0:
        raise ValueError(f"{DRAIN_ENV} must be positive, got {raw}")
    return value


def order_max_audit_executors(env=None) -> int:
    import os

    raw = (os.environ if env is None else env).get(MAX_EXECUTORS_ENV, "").strip()
    value = int(raw) if raw else DEFAULT_MAX_EXECUTORS
    if value < 1:
        raise ValueError(f"{MAX_EXECUTORS_ENV} must be at least 1, got {raw}")
    return value


def order_audit_executors(declared: dict | None) -> int:
    """The most audit executors an order allows for its model: its record's ``audit_executors`` (its chosen
    speed), one without."""
    value = (declared or {}).get("audit_executors")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else 1


def executors_for_backlog(backlog: float, items_per_second: float | None, *, drain_seconds: float,
                          max_executors: int) -> int | None:
    """Executors that audit ``backlog`` items within ``drain_seconds`` at the measured rate of one: 0 with
    nothing to audit, None while the rate is unmeasured (the fleet keeps what it has)."""
    if backlog <= 0:
        return 0
    if not items_per_second:
        return None
    return max(1, min(max_executors, math.ceil(backlog / (items_per_second * drain_seconds))))


class ExecutorSizing:
    """Per model, the executors wanted: more at once, fewer only once fewer were enough for
    SCALE_DOWN_SECONDS, so a backlog that dips between passes does not release a card."""

    def __init__(self, *, clock: Callable[[], float] = time.time,
                 scale_down_seconds: float = SCALE_DOWN_SECONDS) -> None:
        self._clock = clock
        self._down = scale_down_seconds
        self._wanted: dict[str, int] = {}
        self._lower_since: dict[str, float] = {}

    def update(self, model: str, target: int | None) -> int | None:
        current = self._wanted.get(model)
        if target is None:
            return current
        now = self._clock()
        if current is None or target >= current:
            self._wanted[model] = target
            self._lower_since.pop(model, None)
        else:
            since = self._lower_since.setdefault(model, now)
            if now - since >= self._down:
                self._wanted[model] = target
                self._lower_since.pop(model, None)
        return self._wanted[model]

    def forget(self, keep: set[str]) -> None:
        for model in [m for m in self._wanted if m not in keep]:
            self._wanted.pop(model, None)
            self._lower_since.pop(model, None)


class BatchParked(Exception):
    """A batch of the call was parked: no two of its scorers agree (or two agreeing pairs conflict), or the evidence
    says no card can answer it. ``indexes``: the items of the call it held (None: all of them); ``batch_ids``: the
    batches parked; ``scored``: rows of the call's other batches decided before it, by item index. Its other batches
    still undecided are cancelled."""

    def __init__(self, message: str = "", *, indexes=None, batch_ids=(), scored=None) -> None:
        super().__init__(message)
        self.indexes = None if indexes is None else sorted(indexes)
        self.batch_ids = list(batch_ids)
        self.scored = dict(scored or {})


class ItemsUnreadable(Exception):
    """Items of a call an executor would refuse as it parses their lease (``AuditLease``), by index, with why. No batch
    of the call was made."""

    def __init__(self, reasons: dict[int, str]) -> None:
        super().__init__("; ".join(f"item {k}: {why}" for k, why in sorted(reasons.items()))[:500])
        self.reasons = dict(reasons)


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

    def quarantined_ids(self) -> list[str]:
        """The eval executors the registry, as last read, says were quarantined (a status is never set back)."""
        from reliquary.infrastructure.corpus_executor_store import WAS_QUARANTINED

        return sorted(eid for eid, d in self._by_id.items() if d.get("scope") == "eval"
                      and (d.get("status") == "quarantined" or d.get(WAS_QUARANTINED) is True))


def _placement(document: dict) -> tuple[str, str] | None:
    """Where an executor runs; None when it cannot be told apart from others."""
    provider, host = document.get("provider_id"), document.get("host")
    if not provider or not host:
        return None
    return str(provider), str(host)


def _tokens(items: Sequence[dict]) -> int:
    return sum(len(item.get("tokens") or ()) for item in items)


def _scored(result: AuditResult, items: Sequence[dict]) -> bool:
    """Whether ``result`` fits ``items`` and scored every one (no executor error)."""
    scores = result.scores
    return (len(scores) == len(items)
            and not any(_misfit(s, item) for s, item in zip(scores, items))
            and not any(s.status == ITEM_ERROR for s in scores))


def _measured(result: AuditResult, items: Sequence[dict]) -> bool:
    """Whether ``result`` fits ``items``, errs on none, and measured at least one (ok, with its chunks or its draw's
    diagnostics): what witnesses that its card scores. An answer of statuses with no measure (proof_undecodable)
    costs no GPU; one undecodable proof beside measured items is the miner's, not the card's."""
    return _scored(result, items) and any(s.status == ITEM_OK and (s.chunks or s.draw is not None)
                                          for s in result.scores)


def _wire_item(item: dict) -> dict:
    """One item of a lease as an executor receives it (``AuditItem``)."""
    return {"tokens": list(item["tokens"]), "prompt_len": int(item["prompt_len"]),
            "proofs": list(item.get("proofs") or ()),
            **({"spans": [list(s) for s in item["spans"]]} if item.get("spans") is not None else {}),
            **({"draw": item["draw"]} if item.get("draw") is not None else {})}


# A well-formed lease id, to check a lease before any executor gets it.
_PROBE_LEASE_ID = "0" * 32


def _lease_body(protocol: str, model: ModelKey, proof, lease_id: str, expires_at: float,
                items: Sequence[dict]) -> dict:
    """The lease an executor receives (``AuditLease`` on its side)."""
    return {"protocol": protocol, "lease_id": lease_id, "model_id": model[0], "model_revision": model[1],
            "chunk_tokens": proof.chunk_tokens, "topk": proof.topk, "expires_at": expires_at,
            "items": [_wire_item(i) for i in items]}


def _unreadable_items(model: ModelKey, proof, items: Sequence[dict]) -> dict[int, str]:
    """Each item an executor would refuse as it parses its lease, with why: checked alone in a lease of its protocol.
    The why names the fields and the rule, never their values (miner tokens)."""
    from pydantic import ValidationError

    unreadable = {}
    for k, item in enumerate(items):
        protocol = AUDIT_PROTOCOL if item.get("draw") is None else draw_protocol(item["draw"])
        try:
            AuditLease.model_validate(_lease_body(protocol, model, proof, _PROBE_LEASE_ID, 0.0, [item]))
        except ValidationError as exc:
            unreadable[k] = "; ".join(f"{'.'.join(str(part) for part in error['loc'][2:]) or 'item'}: "
                                      f"{error['msg']}" for error in exc.errors(include_input=False))[:300]
        except Exception as exc:  # noqa: BLE001 - whatever the executor would choke on is a refusal too
            unreadable[k] = f"{type(exc).__name__}: {exc}"[:300]
    return unreadable


@dataclass
class _Loss:
    """One executor's lost leases of a batch: its provider, whether one of them was witnessed (and when, last), and the
    leases that struck it."""
    provider: str
    witnessed: bool = False
    witnessed_at: float = float("-inf")
    leases: list = field(default_factory=list)


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
    # v3: a forced-draw job's samples, leased to executors that score the draw, decided by its judge.
    protocol: str = AUDIT_PROTOCOL
    judge: Any = None
    reuse_auditors_only: bool = False
    # Its lost leases per executor (`_Loss`); who may not lease it before when, and how often each was benched.
    losses: dict[str, _Loss] = field(default_factory=dict)
    benched: dict[str, float] = field(default_factory=dict)
    benches: collections.Counter = field(default_factory=collections.Counter)
    # Parked on its losses: a measured answer to one of its leases, late or not, says a card can answer it.
    parked: bool = False


@dataclass
class _Lease:
    lease_id: str
    batch: _Batch
    executor_id: str
    expires_at: float
    granted_at: float = 0.0
    # Granted the dispatcher's whole lease seconds: a card too slow for that stays a lost result, even late.
    full_length: bool = False
    # Its expiry struck its executor: a late scored answer takes that strike back, once.
    struck: bool = False


class PairedAuditDispatcher:
    """Audit batches of every model, each decided by ``scorers`` executors (one by default; two or
    more must agree, on distinct providers and hosts). ``quarantine``/``record_heartbeat`` write the
    registry."""

    def __init__(self, *, directory: EvalExecutorDirectory,
                 quarantine: Callable[[str, str], Awaitable[Any]] | None = None,
                 record_heartbeat: Callable[[str, float, dict], Awaitable[Any]] | None = None,
                 clock: Callable[[], float] = time.time,
                 lease_seconds: float = AUDIT_LEASE_SECONDS,
                 live_seconds: float = EXECUTOR_LIVE_SECONDS,
                 max_leases_per_executor: int = MAX_LEASES_PER_EXECUTOR,
                 expiry_strikes: int = LOST_RESULT_STRIKES,
                 scorers: int = DEFAULT_SCORERS) -> None:
        if not 1 <= scorers <= MAX_SCORERS:
            raise ValueError(f"scorers must be 1 to {MAX_SCORERS}")
        self._scorers = scorers
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
        # Per executor, results delivered in time since its last lost one.
        self._clean: collections.Counter = collections.Counter()
        # Per model, (when, seconds per token) its recent scored results took from grant to result.
        self._lease_times: dict[ModelKey, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=LEASE_TIMES_KEPT))
        # Leases that expired unanswered, by id, for LATE_ANSWER_SECONDS: a late answer is told apart.
        self._expired: dict[str, _Lease] = {}
        # Per executor, (when, seconds per token) of its recent scored results: a card slower than the model's
        # median gets leases at its own slowest recent pace at once, not after its late answers outnumber the
        # fast ones, and not at the pace of its last cheap batch.
        self._pace: dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=LEASE_TIMES_KEPT))
        # Per executor, when it last delivered a measured result in time for an undecided batch (`_measured`): a lease
        # of it lost after that was granted is witnessed. Per provider, when its losses counted towards a park.
        self._witness: dict[str, float] = {}
        self._parks_by: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._seen: dict[str, float] = {}
        self._listeners: list[Callable[[str], Awaitable[Any]]] = []
        # Told (job id, batch id) when a batch parked on its losses gets a measured answer: its records are tried again.
        self._unpark_listeners: list[Callable[[str, int], Any]] = []
        self._background: set[asyncio.Task] = set()
        self._unwritten: dict[str, str] = {}
        self._documents: dict[str, dict] = {}
        self._detail: dict[str, dict] = {}
        # The audit protocols each executor last claimed with: a v3 (draw) batch waits for v3 executors only.
        self._protocols: dict[str, tuple[str, ...]] = {}
        self._written: dict[str, tuple[float, dict]] = {}
        # job id -> why it needs attention (parked batches), for the status.
        self.attention: dict[str, list[str]] = collections.defaultdict(list)
        self.parked: collections.Counter = collections.Counter()
        self.quarantined: set[str] = set()
        self.stats = collections.Counter()
        # Throughput: per executor, since when it holds a lease; per model, (when, items scored, completion
        # tokens, prompt tokens) and (when it ended, busy seconds) of the last THROUGHPUT_WINDOW_SECONDS.
        self._busy_since: dict[str, tuple[float, ModelKey]] = {}
        self._scored_items: dict[ModelKey, collections.deque] = collections.defaultdict(collections.deque)
        self._busy_spans: dict[ModelKey, collections.deque] = collections.defaultdict(collections.deque)
        # Executors the fleet may release (more live than wanted): leased nothing more, so once their leases
        # end they are idle for good and no batch waits on a destroyed card. First contact, to pick the newest.
        self._releasing: set[str] = set()
        self.reuse_only_jobs: set[str] = set()
        self._first_seen: dict[str, float] = {}

    # -- the auditors' side ------------------------------------------------

    def view(self, model_id: str, model_revision: str, proof, job_id: str = "", judge=None) -> "PoolView":
        return PoolView(self, (model_id, model_revision), proof, job_id, judge)

    def subscribe(self, listener: Callable[[str], Awaitable[Any]]) -> None:
        self._listeners.append(listener)

    def unsubscribe(self, listener) -> None:
        self._listeners = [other for other in self._listeners if other != listener]

    def subscribe_unparked(self, listener: Callable[[str, int], Any]) -> None:
        """``listener(job_id, batch_id)``: a batch parked on its losses got a measured answer, late or not."""
        self._unpark_listeners.append(listener)

    def unsubscribe_unparked(self, listener) -> None:
        self._unpark_listeners = [other for other in self._unpark_listeners if other != listener]

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
        self.reuse_only_jobs.discard(job_id)

    async def score(self, model: ModelKey, proof, items: Sequence[dict], job_id: str = "", judge=None,
                    batches: list | None = None):
        """``(status, chunks, scored_by[, diagnostics])`` per item; ``scored_by`` is the
        tuple of executors whose agreement decided it. Drawn items (``draw``) go in v3
        batches, a trajectory alone, decided by ``judge`` (the job's ``DrawJudge``).
        ``batches`` collects this call's batches, for a caller that may have to cancel them.

        Raises ``ItemsUnreadable`` for items an executor would refuse (no batch is made), and ``BatchParked`` as soon
        as one of the call's batches is parked, with the rows of those decided before it; the others are cancelled,
        so no card scores them for nobody."""
        loop = asyncio.get_running_loop()
        units = []
        drawn = [k for k, item in enumerate(items) if item.get("draw") is not None]
        if any(item.get("spans") is not None and item.get("draw") is None for item in items):
            raise RuntimeError("trajectory audit is not available on remote executors")
        # Checked as each executor parses its lease (about 7 ms for 262,144 tokens): off the loop past one lease.
        unreadable = (_unreadable_items(model, proof, items) if _tokens(items) <= INLINE_CHECK_TOKENS
                      else await asyncio.to_thread(_unreadable_items, model, proof, items))
        if unreadable:
            self.stats["unreadable_items"] += len(unreadable)
            first = min(unreadable)
            logger.error("eval audit of job %s: %d of %d item(s) no executor can read (item %d: %s); their "
                         "records are parked before any lease", job_id, len(unreadable), len(items), first,
                         unreadable[first])
            self.parked[job_id] += 1
            self.stats["parked"] += 1
            self._flag(job_id, f"{len(unreadable)} item(s) no executor can read ({unreadable[first]})")
            raise ItemsUnreadable(unreadable)
        plain = [k for k, item in enumerate(items) if item.get("draw") is None]
        groups = [(AUDIT_PROTOCOL, [[plain[k] for k in unit] for unit in _lease_units([items[k] for k in plain])])]
        # Greedy draws (top_k 1) go in v4 leases: an executor that predates v4 never scores them.
        for protocol in (AUDIT_PROTOCOL_V3, AUDIT_PROTOCOL_V4):
            mine = [k for k in drawn if draw_protocol(items[k]["draw"]) == protocol]
            drawn_plain = [k for k in mine if items[k].get("spans") is None]
            groups.append((protocol, [[drawn_plain[k] for k in unit]
                                      for unit in _lease_units([items[k] for k in drawn_plain])]
                           + [[k] for k in mine if items[k].get("spans") is not None]))
        for protocol, grouped in groups:
            for indexes in grouped:
                batch = _Batch(id=next(self._ids), model=model, proof=proof,
                               items=[items[k] for k in indexes], future=loop.create_future(),
                               job_id=job_id, protocol=protocol, judge=judge,
                               needed=self._scorers,
                               reuse_auditors_only=job_id in self.reuse_only_jobs)
                units.append((indexes, batch))
                self._queues[model].append(batch)
                if batches is not None:
                    batches.append(batch)
        scored: list = [None] * len(items)
        waiting = {batch.future: (indexes, batch) for indexes, batch in units}
        try:
            while waiting:
                # The first batch parked ends the call at once, whichever batch it is.
                done, _ = await asyncio.wait(list(waiting), return_when=asyncio.FIRST_EXCEPTION)
                failed, cancelled = [], False
                for future in done:
                    indexes, batch = waiting.pop(future)
                    if future.cancelled():
                        cancelled = True
                    elif future.exception() is not None:
                        failed.append((indexes, batch, future.exception()))
                    else:
                        scores, scored_by = future.result()
                        for k, (status, chunks, *extra) in zip(indexes, scores):
                            scored[k] = (status, chunks, scored_by, *extra)
                if not (failed or cancelled):
                    continue
                for future in waiting:
                    future.cancel()
                if cancelled:
                    # Its order unwired (`forget_job`), as an awaited batch raised before.
                    raise asyncio.CancelledError(f"a batch of job {job_id} was cancelled")
                others = [exc for _, _, exc in failed if not isinstance(exc, BatchParked)]
                if others:
                    raise others[0]
                raise BatchParked("; ".join(dict.fromkeys(str(exc) for _, _, exc in failed)),
                                  indexes=[k for indexes, _, _ in failed for k in indexes],
                                  batch_ids=[batch.id for _, batch, _ in failed],
                                  scored={k: row for k, row in enumerate(scored) if row is not None})
        except asyncio.CancelledError:
            # This call was given up (its caller cancelled): no card scores what is left of it.
            for future in waiting:
                future.cancel()
            raise
        return scored

    @property
    def scorers(self) -> int:
        return self._scorers

    def pending_items(self, job_id: str) -> int:
        """The samples of ``job_id`` waiting in batches not yet decided (claimed ones included)."""
        return sum(len(batch.items) for queue in self._queues.values() for batch in queue
                   if batch.job_id == str(job_id) and not batch.future.done())

    def pending(self, model: ModelKey | None = None) -> int:
        queues = [self._queues[model]] if model is not None else list(self._queues.values())
        return sum(1 for queue in queues for batch in queue if not batch.future.done())

    # -- the executors' side -----------------------------------------------

    def heartbeat(self, executor_id: str, document: dict | None = None,
                  detail: dict | None = None) -> None:
        self._seen[executor_id] = self._clock()
        self._first_seen.setdefault(executor_id, self._seen[executor_id])
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
        if batch.benched.get(executor_id, float("-inf")) > self._clock():
            # It lost this batch or answered it with an error lately (`_bench`).
            return False
        if len(batch.scores) + len(batch.leased) >= batch.needed:
            return False
        involved = [batch.places[e] for e in batch.scores] + [
            batch.places[e] for e in batch.leased]
        # A distinct provider AND a distinct host from everyone already on it.
        return all(place[0] != p and place[1] != h for p, h in involved)

    def claim(self, document: dict, protocols=(AUDIT_PROTOCOL,)) -> dict | None:
        """An audit lease for this executor's model in one of ``protocols``, or None.
        Raises ``LeaseRefused(409)`` for an executor whose placement is unknown."""
        executor_id = document["executor_id"]
        self.heartbeat(executor_id, document)
        self._protocols[executor_id] = tuple(protocols)
        place = _placement(document)
        if place is None:
            raise LeaseRefused(409, "executor_provider_unknown")
        if executor_id in self._releasing:
            return None
        held = [lease for lease in self._leases.values() if lease.executor_id == executor_id]
        if len(held) >= self._max_leases or any(lease.batch.reuse_auditors_only for lease in held):
            return None
        queue = self._queues.get((document["model_id"], document["model_revision"]))
        for batch in sorted(queue or (), key=lambda batch: batch.reuse_auditors_only):
            if batch.future.done():
                queue.remove(batch)
                continue
            if batch.protocol not in protocols or not self._eligible(batch, executor_id, place):
                continue
            if batch.reuse_auditors_only and (held or
                    self.reusable_auditor(batch.model, batch.protocol) != executor_id):
                continue
            now = self._clock()
            seconds = self.lease_time(batch.model, _tokens(batch.items)
                                      + sum(_tokens(h.batch.items) for h in held), executor_id)
            lease = _Lease(lease_id=secrets.token_hex(16), batch=batch, executor_id=executor_id,
                           expires_at=now + seconds, granted_at=now,
                           full_length=seconds >= self._lease_seconds)
            self._leases[lease.lease_id] = lease
            self._mark_busy(executor_id, batch.model)
            batch.leased.add(executor_id)
            batch.places[executor_id] = place
            self.stats["leased"] += 1
            return _lease_body(batch.protocol, batch.model, batch.proof, lease.lease_id, lease.expires_at,
                               batch.items)
        return None

    def lease_time(self, model: ModelKey, tokens: int, executor_id: str | None = None) -> float:
        """Seconds an audit lease of ``tokens`` tokens (its own and those its card already holds) lives, at
        the model's recent median pace, or at ``executor_id``'s slowest recent pace when that is slower."""
        now = self._clock()
        recent = [per_token for when, per_token in self._lease_times.get(model, ())
                  if now - when <= LEASE_TIMES_MAX_AGE_SECONDS]
        if len(recent) < LEASE_TIMES_NEEDED:
            return self._lease_seconds
        own = [pace for when, pace in (self._pace.get(executor_id, ()) if executor_id else ())
               if now - when <= LEASE_TIMES_MAX_AGE_SECONDS]
        per_token = max([statistics.median(recent), *own])
        return min(self._lease_seconds,
                   max(AUDIT_LEASE_FLOOR_SECONDS, 2.0 * per_token * max(1, tokens) + LEASE_OVERHEAD_SECONDS))

    def _time(self, lease: _Lease) -> None:
        """Time a lease whose result scored its items, late or not."""
        tokens = _tokens(lease.batch.items)
        if tokens:
            now = self._clock()
            per_token = (now - lease.granted_at) / tokens
            self._lease_times[lease.batch.model].append((now, per_token))
            self._pace[lease.executor_id].append((now, per_token))

    def _answered_late(self, lease: _Lease, result: AuditResult) -> None:
        """A result after its lease expired: refused (the batch is scored again), but its card is slow, not
        lost. A scored one times the model's and the card's next leases, so they fit it, and takes back the
        strike that lease's expiry gave, unless the lease was already the longest one: a card too slow for
        that stays a lost result. A measured one also says the batch can be answered: its card's loss is no
        evidence against it any more, a batch parked on its losses is unparked (`_unpark`), and the card's bench
        on it is lifted (unless its lease was the longest)."""
        batch = lease.batch
        if batch.reuse_auditors_only:
            return
        if not _scored(result, batch.items):
            # An error or a misfit, late: benched from the batch, as before the sweep.
            self._bench(batch, lease.executor_id)
            return
        self.stats["answered_late"] += 1
        logger.warning("eval audit executor %s answered lease %s of batch %s late, %.0f s after its grant: "
                       "timed, %s", lease.executor_id, lease.lease_id[:8], batch.id,
                       self._clock() - lease.granted_at,
                       "still a lost result (its lease was the longest)" if lease.full_length
                       else "not a lost result")
        self._time(lease)
        measured = _measured(result, batch.items)
        if measured:
            if batch.losses.pop(lease.executor_id, None) is not None:
                self.stats["losses_taken_back"] += 1
            self._unpark(batch)
        if lease.full_length:
            return
        if measured:
            batch.benched.pop(lease.executor_id, None)
        if lease.struck:
            lease.struck = False
            self._strikes[lease.executor_id] = max(0, self._strikes[lease.executor_id] - 1)
            self.stats["strikes_taken_back"] += 1

    def _unpark(self, batch: _Batch) -> None:
        """A batch parked on its losses got a measured answer: a card can answer it. Its job's auditor is told
        (``subscribe_unparked``) and tries its records again at once."""
        if not batch.parked:
            return
        batch.parked = False
        self.stats["unparked"] += 1
        logger.warning("eval audit batch %s of job %s, parked as lost, got a measured answer: unpark sent to its "
                       "auditor", batch.id, batch.job_id)
        for listener in list(self._unpark_listeners):
            try:
                listener(batch.job_id, batch.id)
            except Exception:
                logger.exception("eval audit: unpark of batch %s not delivered", batch.id)

    def _bench(self, batch: _Batch, executor_id: str) -> None:
        """``executor_id`` leases ``batch`` again only after BENCH_SECONDS, doubled at each repeat."""
        seconds = min(MAX_BENCH_SECONDS, BENCH_SECONDS * 2 ** batch.benches[executor_id])
        batch.benches[executor_id] += 1
        batch.benched[executor_id] = self._clock() + seconds
        self.stats["benched"] += 1

    def _counted(self, provider: str, now: float) -> bool:
        """Whether ``provider``'s losses still count towards a park (PARKS_PER_EXECUTOR per PARK_WINDOW_SECONDS)."""
        parks = self._parks_by.get(provider)
        while parks and now - parks[0] > PARK_WINDOW_SECONDS:
            parks.popleft()
        return not parks or len(parks) < PARKS_PER_EXECUTOR

    async def _lost(self, lease: _Lease, *, why: str = "expired unanswered", bench: bool = False,
                    evidence: bool = True) -> None:
        """A lost result: a lease that expired unanswered, or answered with something else than its scores. Its card is
        struck, as #493 sets aside a card that loses results. A witnessed loss, or an answer that is no score
        (``bench``), also benches it from the batch (`_bench`). A witnessed loss counts against the batch (unless
        ``evidence`` is off), which is parked once the evidence says no card can answer it
        (UNANSWERED_PROVIDERS_PARK): its witnessed losses then strike nobody."""
        batch, executor_id = lease.batch, lease.executor_id
        now = self._clock()
        witnessed = self._witness.get(executor_id, float("-inf")) > lease.granted_at
        if witnessed or bench:
            self._bench(batch, executor_id)
        self._clean[executor_id] = 0
        loss, park = None, None
        if not batch.future.done() and evidence:
            place = batch.places.get(executor_id)
            loss = batch.losses.setdefault(executor_id, _Loss(provider=place[0] if place else executor_id))
            if witnessed:
                loss.witnessed, loss.witnessed_at = True, now
            # Only fresh witnessed losses are evidence (within PARK_WINDOW_SECONDS); not those of a card quarantined or
            # released since, nor past its provider's cap.
            losers = sorted((e, other) for e, other in batch.losses.items()
                            if other.witnessed and now - other.witnessed_at <= PARK_WINDOW_SECONDS
                            and e not in self.quarantined and e not in self._releasing
                            and self._counted(other.provider, now))
            providers = {other.provider for _, other in losers}
            if len(providers) >= UNANSWERED_PROVIDERS_PARK + max(0, batch.needed - 1):
                park = (losers, providers)
            if witnessed:
                self.stats["witnessed_losses"] += 1
        spared = park is not None and any(e == executor_id for e, _ in park[0])
        if not spared:
            self._strikes[executor_id] += 1
            lease.struck = True
            if loss is not None:
                loss.leases.append(lease)
        logger.warning("eval audit lease %s of batch %s (job %s, %d items, %d tokens) %s on executor %s, %.0f s "
                       "after its grant%s: strike %d of %d", lease.lease_id[:8], batch.id, batch.job_id,
                       len(batch.items), _tokens(batch.items), why, executor_id,
                       min(now, lease.expires_at) - lease.granted_at,
                       " (witnessed: it scored another lease since)" if witnessed else "",
                       self._strikes[executor_id], self._strikes_limit)
        if park is not None:
            self._park_lost(batch, *park, now)
        if self._strikes[executor_id] >= self._strikes_limit:
            await self.quarantine(executor_id, f"{self._strikes_limit} results lost (leases expired or "
                                               "answers that do not fit)")

    def _park_lost(self, batch: _Batch, losers: list, providers: set, now: float) -> None:
        """Park a batch no card can answer: its witnessed losers' strikes for it are taken back (they work, the batch
        does not), each provider's park is counted, and the reason names them."""
        for provider in providers:
            self._parks_by[provider].append(now)
        for executor_id, loss in losers:
            for lease in loss.leases:
                if lease.struck:
                    lease.struck = False
                    self._strikes[executor_id] = max(0, self._strikes[executor_id] - 1)
                    self.stats["strikes_taken_back_parked"] += 1
        batch.parked = True
        self.stats["unanswered_parked"] += 1
        self._park(batch, "its leases were lost by working cards on {} providers: {}".format(
            len(providers), ", ".join(f"{e} ({loss.provider})" for e, loss in losers)))

    def lease_of(self, lease_id: str) -> _Lease | None:
        return self._leases.get(lease_id)

    async def result(self, executor_id: str, lease_id: str, result: AuditResult) -> str:
        self.heartbeat(executor_id)
        lease = self._leases.get(lease_id)
        if lease is None or lease.executor_id != executor_id:
            late = self._expired.get(lease_id)
            if late is not None and late.executor_id == executor_id:
                del self._expired[lease_id]
                self._answered_late(late, result)
                raise LeaseRefused(410, "lease_expired")
            raise LeaseRefused(410, "lease_unknown")
        del self._leases[lease_id]
        batch = lease.batch
        batch.leased.discard(executor_id)
        self._mark_busy(executor_id, batch.model)
        if lease.expires_at <= self._clock():
            # Expired, not swept yet: struck as the sweep would strike it, so no strike depends on the 2 s between an
            # expiry and its sweep. A scored answer is slow, not lost (what the sweep's strike and its take-back add up
            # to); one to the longest lease stays struck, but an answered lease is no evidence against its batch.
            if not batch.reuse_auditors_only:
                if not _scored(result, batch.items):
                    await self._lost(lease, why="expired, then answered with no score", bench=True)
                elif lease.full_length:
                    await self._lost(lease, evidence=False)
                self._answered_late(lease, result)
            raise LeaseRefused(410, "lease_expired")
        scores = result.scores
        if len(scores) != len(batch.items) or any(_misfit(s, item) for s, item in zip(scores, batch.items)):
            # Not an answer to its lease: a lost result, as an expiry is (struck, benched, counted against the batch).
            if not batch.reuse_auditors_only:
                await self._lost(lease, why="answered with a result that does not fit", bench=True)
            raise LeaseRefused(422, "result_does_not_fit_the_lease")
        errored = any(s.status == ITEM_ERROR for s in scores)
        if not batch.reuse_auditors_only and not errored:
            # Only a scored result is clean and timed: errors answered at once would shorten every card's
            # leases on this model, and clear the strikes of a card that scores nothing.
            self._clean[executor_id] += 1
            if self._clean[executor_id] >= CLEAN_STREAK:
                self._strikes[executor_id] = 0
                # Those strikes are gone: neither a late answer to one of their leases nor a park of their batch has
                # one to take back.
                for expired in self._expired.values():
                    if expired.executor_id == executor_id:
                        expired.struck = False
                for queue in self._queues.values():
                    for queued in queue:
                        for struck in getattr(queued.losses.get(executor_id), "leases", ()):
                            struck.struck = False
            self._time(lease)
            if _measured(result, batch.items):
                # A witness only for work that counts: an answer to a batch decided or cancelled proves nothing.
                if not batch.future.done():
                    self._witness[executor_id] = self._clock()
                self._unpark(batch)
        if errored or batch.future.done():
            # The executor's own fault, or no longer needed: nobody is judged on it.
            if errored and not batch.future.done() and not batch.reuse_auditors_only:
                # Not leased to it again at once: one card cannot hold a batch by answering errors (`_bench`).
                self._bench(batch, executor_id)
            self.stats["executor_errors" if not batch.future.done() else "late"] += 1
            return "requeued" if not batch.future.done() else "unneeded"
        batch.scores[executor_id] = converted_scores(scores)
        self.stats["scored"] += 1
        if not batch.reuse_auditors_only:
            prompts = sum(int(item.get("prompt_len") or 0) for item in batch.items)
            completions = sum(len(item.get("tokens") or ()) for item in batch.items) - prompts
            self._scored_items[batch.model].append((self._clock(), len(batch.items), completions, prompts))
        await self._decide(batch)
        return "accepted"

    async def _decide(self, batch: _Batch) -> None:
        """Two agreeing scorers decide. Three: exactly one decision held by an
        agreeing pair decides, whoever agrees with nobody is quarantined. Never
        an id-ordered pick: conflicting pairs, or no pair, park the batch."""
        from reliquary.validator.corpus_audit import outcome_from_scores

        if self._scorers == 1 and batch.scores:
            # One scorer: its scores decide, as the validator's own GPU decides the corpus audits.
            if not batch.future.done():
                (executor_id, scores), = batch.scores.items()
                batch.future.set_result((scores, (executor_id,)))
                self.stats["scored_alone"] += 1
            return
        if len(batch.scores) < 2:
            return
        ids = sorted(batch.scores)

        def decision(executor_id: str) -> tuple:
            return tuple(_passes(score, batch) for score in batch.scores[executor_id])

        # A pair agrees on the measures and on the decision: two honest scorers either side of a judge's floor
        # by drift call a third, rather than parking the batch.
        pairs = [(e, o) for k, e in enumerate(ids) for o in ids[k + 1:]
                 if scores_agree(batch.scores[e], batch.scores[o], batch.proof) and decision(e) == decision(o)]

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
        self._park(batch, f"{'no agreeing pair' if not pairs else 'conflicting agreeing pairs'} among {ids}")

    def _park(self, batch: _Batch, why: str) -> None:
        if batch.future.done():
            return
        self.stats["parked"] += 1
        self.parked[batch.job_id] += 1
        reason = f"batch {batch.id} ({len(batch.items)} items): {why}; its records are parked and tried again"
        logger.error("eval audit of job %s parked: %s", batch.job_id, reason)
        self._flag(batch.job_id, reason)
        batch.future.set_exception(BatchParked(reason))

    def _flag(self, job_id: str, reason: str) -> None:
        """Why ``job_id`` needs attention, for the status (the last 10 reasons)."""
        self.attention[job_id].append(reason)
        del self.attention[job_id][:-10]

    def live_executors(self, model: ModelKey) -> list[dict]:
        now = self._clock()
        return [document for eid, document in self._documents.items()
                if now - self._seen.get(eid, float("-inf")) <= self._live
                and eid not in self.quarantined
                and (document.get("model_id"), document.get("model_revision")) == model
                and _placement(document) is not None]

    def audit_ready(self, model: ModelKey, protocol: str | None = None) -> bool:
        """Whether a live executor of ``model`` already claims audit leases (in ``protocol`` when
        given): an executor claims only once its model is loaded."""
        return any(d["executor_id"] in self._protocols
                   and (protocol is None or protocol in self._protocols[d["executor_id"]])
                   for d in self.live_executors(model))

    def _mark_busy(self, executor_id: str, model: ModelKey) -> None:
        """After a lease of ``executor_id`` was granted or ended: open or close its busy span."""
        held = any(lease.executor_id == executor_id and not lease.batch.reuse_auditors_only
                   for lease in self._leases.values())
        now = self._clock()
        if held and executor_id not in self._busy_since:
            self._busy_since[executor_id] = (now, model)
        elif not held and executor_id in self._busy_since:
            since, busy_model = self._busy_since.pop(executor_id)
            self._busy_spans[busy_model].append((now, now - since))

    def reusable_auditor(self, model: ModelKey, protocol: str) -> str | None:
        """Borrow the baseline only; a surplus executor must be free to leave."""
        if self._scorers != 1:
            return None
        live = self.live_executors(model)
        if not live:
            return None
        executor = min(live, key=lambda d: (self._first_seen.get(d["executor_id"], 0.0),
                                           d["executor_id"]))["executor_id"]
        if executor not in self._releasing and protocol in self._protocols.get(executor, ()):
            return executor
        return None

    def release_beyond(self, model: ModelKey, keep: int | None) -> list[str]:
        """The newest live executors of ``model`` beyond ``keep`` (None: none), to release."""
        live = sorted(self.live_executors(model),
                      key=lambda d: (self._first_seen.get(d["executor_id"], 0.0), d["executor_id"]))
        ids = [d["executor_id"] for d in live]
        return ids[max(1, keep):] if keep is not None else []

    def set_releasing(self, executor_ids) -> None:
        """The executors leased nothing more: exactly these (a model no longer served keeps none)."""
        self._releasing = set(executor_ids)

    def executor_ids(self, model: ModelKey) -> tuple[list[str], list[str]]:
        """The live executors of ``model``, and those of them holding no lease."""
        live = sorted(d["executor_id"] for d in self.live_executors(model))
        busy = {lease.executor_id for lease in self._leases.values()}
        return live, [e for e in live if e not in busy]

    def _scored_per_second(self, model: ModelKey) -> tuple[int, int, int, float] | None:
        """Items, completion tokens and prompt tokens one executor of ``model`` scored, and the seconds it
        held a lease, over the last THROUGHPUT_WINDOW_SECONDS; None until MIN_MEASURED_BUSY_SECONDS."""
        now = self._clock()
        start = now - THROUGHPUT_WINDOW_SECONDS
        for log in (self._scored_items[model], self._busy_spans[model]):
            while log and log[0][0] < start:
                log.popleft()
        scored = self._scored_items[model]
        items = sum(row[1] for row in scored)
        busy = sum(min(seconds, end - start) for end, seconds in self._busy_spans[model])
        busy += sum(now - max(since, start) for since, m in self._busy_since.values() if m == model)
        if busy < MIN_MEASURED_BUSY_SECONDS or items <= 0:
            return None
        return items, sum(row[2] for row in scored), sum(row[3] for row in scored), busy

    def items_per_second(self, model: ModelKey) -> float | None:
        """Items one executor of ``model`` scores per second it holds a lease (`_scored_per_second`)."""
        measured = self._scored_per_second(model)
        return None if measured is None else measured[0] / measured[3]

    def tokens_per_second(self, model: ModelKey) -> tuple[float, float] | None:
        """Completion and prompt tokens one executor of ``model`` audits per second it holds a lease: what
        caps an order's paid tokens (``admin.cap_sizing``)."""
        measured = self._scored_per_second(model)
        return None if measured is None else (measured[1] / measured[3], measured[2] / measured[3])

    def audit_pace(self, model: ModelKey) -> tuple[float, float | None] | None:
        """What the merit order fills to (``corpus.lease_allocator``): the completion tokens per second the live
        executors of ``model`` audit together (one's measured speed x how many are live), and their busy share
        over the window; None until measured."""
        measured = self._scored_per_second(model)
        live = self.live_executors(model)
        if measured is None or not live:
            return None
        busy = measured[3]
        # Busy over the time each live executor was seen within the window, not the whole window: after a restart
        # (or for a card just arrived) the window is not covered yet, and its idle share would read far too high.
        now = self._clock()
        seen = sum(min(THROUGHPUT_WINDOW_SECONDS, now - self._first_seen.get(d["executor_id"], now)) for d in live)
        return measured[1] / busy * len(live), min(1.0, busy / max(seen, MIN_MEASURED_BUSY_SECONDS))

    def status(self) -> dict:
        """Per model, what the fleet must know: ``executors_needed`` more
        executors (on providers and hosts distinct from the live ones) to
        decide the batches waiting; per job, whether it needs attention."""
        models = {}
        for model, queue in self._queues.items():
            waiting = [b for b in queue if not b.future.done()]
            live = self.live_executors(model)
            providers = {_placement(d)[0] for d in live}
            needed = 0
            # Per protocol: a draw (v3) batch is scored by v3 executors alone.
            demand = [b for b in waiting if not b.reuse_auditors_only]
            for protocol in {b.protocol for b in demand}:
                able = [d for d in live
                        if protocol in self._protocols.get(d["executor_id"], (AUDIT_PROTOCOL,))]
                required = max(b.needed for b in demand if b.protocol == protocol)
                distinct = min(len({_placement(d)[0] for d in able}), len({_placement(d)[1] for d in able}))
                needed = max(needed, required - distinct)
            models[f"{model[0]}@{model[1]}"] = {
                "executors_needed": max(0, needed),
                "live_executors": len(live), "live_providers": len(providers),
                "reusable_draw_auditor": self.reusable_auditor(model, AUDIT_PROTOCOL_V3),
                "live_draw_executors": sum(1 for d in live if AUDIT_PROTOCOL_V3 in self._protocols.get(
                    d["executor_id"], ())),
                "waiting_batches": len(waiting),
                "waiting_draw_batches": sum(1 for b in waiting if b.protocol in DRAW_PROTOCOLS),
                # Greedy draws (v4) are leased only to executors that claim v4.
                "live_greedy_executors": sum(1 for d in live if AUDIT_PROTOCOL_V4 in self._protocols.get(
                    d["executor_id"], ())),
                "waiting_greedy_batches": sum(1 for b in waiting if b.protocol == AUDIT_PROTOCOL_V4),
                "awaiting_third_scorer": sum(1 for b in waiting if b.needed > max(2, self._scorers))}
        jobs = {job_id: {"needs_attention": True, "parked_batches": self.parked[job_id],
                         "reasons": list(reasons)}
                for job_id, reasons in self.attention.items() if reasons}
        now = self._clock()
        for provider in list(self._parks_by):
            self._counted(provider, now)  # drops the parks past the window
        parks = {p: len(times) for p, times in self._parks_by.items() if times}
        return {"schema": "reliquary/eval-control-status/v1", "updated_at": self._clock(),
                "models": models, "jobs": jobs,
                "quarantined": sorted(self.quarantined), "stats": dict(self.stats),
                # Per provider, the parks its cards' losses counted towards over PARK_WINDOW_SECONDS.
                "parks_by_provider": parks}

    async def load_quarantined(self) -> list[str]:
        """Quarantines the registry holds that this process has not heard of (it forgets its own at a restart): each
        goes through ``quarantine`` once, with no registry write. A read that failed or lists none changes nothing."""
        loaded = [eid for eid in self._directory.quarantined_ids() if eid not in self.quarantined]
        for executor_id in loaded:
            await self.quarantine(executor_id, "quarantined in the registry", recorded=True)
        return loaded

    async def quarantine(self, executor_id: str, reason: str, *, recorded: bool = False) -> None:
        """``recorded``: read from the registry, so nothing is written back to it."""
        if executor_id in self.quarantined:
            return
        logger.error("eval audit executor %s quarantined: %s", executor_id, reason)
        self.quarantined.add(executor_id)
        self._directory.revoke_locally(executor_id)
        self.stats["quarantined_loaded" if recorded else "quarantined"] += 1
        for lease_id, lease in list(self._leases.items()):
            if lease.executor_id == executor_id:
                del self._leases[lease_id]
                lease.batch.leased.discard(executor_id)
                self._mark_busy(executor_id, lease.batch.model)
        for queue in self._queues.values():
            for batch in queue:
                if not batch.future.done() and executor_id in batch.scores:
                    # Its score no longer counts towards an agreement.
                    del batch.scores[executor_id]
                # Nor its losses as evidence that no card can answer a batch.
                batch.losses.pop(executor_id, None)
        if not recorded:
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
                if self._leases.pop(lease_id, None) is None:
                    # Answered, or taken back by its executor's quarantine, while an earlier one awaited.
                    continue
                lease.batch.leased.discard(lease.executor_id)
                self._mark_busy(lease.executor_id, lease.batch.model)
                if lease.batch.reuse_auditors_only:
                    logger.info("eval audit lease %s of reuse batch %s expired on executor %s (no strike)",
                                lease_id[:8], lease.batch.id, lease.executor_id)
                    continue
                # Remembered for a late answer (`_answered_late`).
                self._expired[lease_id] = lease
                await self._lost(lease)
        for lease_id, lease in list(self._expired.items()):
            if now - lease.expires_at > LATE_ANSWER_SECONDS:
                del self._expired[lease_id]
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


def _passes(score, batch: _Batch) -> bool:
    """One scorer's decision on one item: the proof's thresholds, or a drawn item's judge."""
    from reliquary.validator.corpus_audit import SCORE_OK, outcome_from_scores

    if len(score) > 2:
        from reliquary.corpus import eval_draw
        from reliquary.corpus.draw_audit import diagnostics_from_wire

        return score[0] == SCORE_OK and eval_draw.judge([diagnostics_from_wire(score[2])], batch.judge)[0]
    return outcome_from_scores(score[0], score[1], batch.proof).passed


class PoolView:
    """One job's window on the dispatcher: its model's pool and its own proof,
    in the shape ``CorpusAuditor`` uses a remote dispatcher."""

    def __init__(self, dispatcher: PairedAuditDispatcher, model: ModelKey, proof,
                 job_id: str = "", judge=None) -> None:
        self._dispatcher, self._model, self._proof = dispatcher, model, proof
        self._job_id = job_id
        self._judge = judge
        self._listeners: list = []
        self._unpark_listeners: list = []

    def connected(self) -> bool:
        # Always remote: there is no local GPU to fall back on.
        return True

    def subscribe(self, listener) -> None:
        self._listeners.append(listener)
        self._dispatcher.subscribe(listener)

    def subscribe_unparked(self, listener) -> None:
        """``listener(batch_id)``: a batch of this job parked on its losses got a measured answer."""
        job_id = self._job_id

        def mine(batch_job: str, batch_id: int) -> None:
            if batch_job == job_id:
                listener(batch_id)

        self._unpark_listeners.append(mine)
        self._dispatcher.subscribe_unparked(mine)

    def close(self) -> None:
        """The job is gone: no re-audit of it ever starts, its batches leave."""
        for listener in self._listeners:
            self._dispatcher.unsubscribe(listener)
        self._listeners.clear()
        for listener in self._unpark_listeners:
            self._dispatcher.unsubscribe_unparked(listener)
        self._unpark_listeners.clear()
        self._dispatcher.forget_job(self._job_id)

    async def score(self, items: Sequence[dict]):
        return await self._dispatcher.score(self._model, self._proof, items, self._job_id, self._judge)


class _VocabularyOnly:
    """What the auditor reads of a model when it has none: the vocabulary size."""

    def __init__(self, vocab_size: int) -> None:
        self._embeddings = SimpleNamespace(num_embeddings=int(vocab_size))

    def get_input_embeddings(self):
        return self._embeddings


def eval_auditor(**kwargs):
    """A ``CorpusAuditor`` that never touches a GPU: every batch, re-audits
    included, goes to an executor pair; ``scored_by`` lists both executors.
    ``reopen_slots`` (eval jobs only) reopens a slot on each failed audit."""
    from reliquary.validator.corpus_audit import outcome_from_scores
    from reliquary.validator.corpus_auditor import CorpusAuditor

    class EvalAuditor(CorpusAuditor):
        # Orders forfeit a caught cheat's unpaid work (#468): the slow-hotkey rule weighs that stake.
        stake_at_receipt = True

        async def _forward(self, records, *, local: bool = False):
            sids = [self._sid_of(record) for record in records]
            now = self._clock()
            # A parked record cooling down is not scored, whoever asks (a pass that met it in an audit-ahead flight).
            cooling = {i: _Parked(state.why) for i, sid in enumerate(sids)
                       if (state := self._park_state.get(sid) or self._reaudit_parked.get(sid)) is not None
                       and now < state.retry_at}
            again = [i for i, sid in enumerate(sids) if i not in cooling and sid is not None
                     and (sid in self._park_state or sid in self._reaudit_parked)]
            if _REAUDIT_PARKED.get() is None:
                # A pending record parked before is tried again alone, in the background: one record no executor can
                # score then parks nobody else's, and a retry no card answers holds no pass. Its outcome is taken by its
                # next pass (`_ahead`).
                for i in again:
                    if sids[i] not in self._retries:
                        self._retries[sids[i]] = asyncio.ensure_future(self._retry_alone(records[i], sids[i]))
                    cooling[i] = _Parked(f"tried again alone: {self._park_state[sids[i]].why}"
                                         if sids[i] in self._park_state else "tried again alone")
                again = []
            rest = [i for i in range(len(records)) if i not in cooling and i not in again]
            # A re-audit tries each pass parked before in a call of its own, and waits for it.
            parts = [[i] for i in again] + ([rest] if rest else [])
            if len(parts) == 1 and not cooling:
                return await self._forward_part(records, sids)
            done = await asyncio.gather(*(self._forward_part([records[i] for i in part], [sids[i] for i in part])
                                          for part in parts), return_exceptions=True)
            outcomes: list = [cooling.get(i) for i in range(len(records))]
            for part, out in zip(parts, done):
                if isinstance(out, (ValueError, RuntimeError)):
                    out = [str(out)] * len(part)
                elif isinstance(out, BaseException):
                    raise out
                for i, outcome in zip(part, out):
                    outcomes[i] = outcome
            return outcomes

        async def _forward_part(self, records, sids):
            """The outcomes of ``records``: a verdict each, or ``_Parked`` for a record an executor cannot read or
            whose batch was parked (recorded at once: an unpark may come while the rest is scored again), or a
            validator error's message for a record left unscored when a relaunch failed."""
            results, items = await asyncio.to_thread(self._prepare, records)
            if self._draw_job is None:
                # A draw job's trajectories go v3; proven ones have no executor here.
                self._refuse_trajectories(items)
            rows: dict[int, tuple] = {}
            parked: dict[int, str] = {}
            failure = None
            todo = list(range(len(items)))
            while todo:
                try:
                    scores = await self._remote.score(self._wire_items(records, [items[k] for k in todo]))
                except ItemsUnreadable as exc:
                    hit = [(j, f"an item no executor can read: {why}", PARK_UNREADABLE, ())
                           for j, why in exc.reasons.items()]
                except BatchParked as exc:
                    for j, row in exc.scored.items():
                        rows[todo[j]] = row
                    hit = [(j, str(exc), PARK_LOST, tuple(exc.batch_ids))
                           for j in (range(len(todo)) if exc.indexes is None else exc.indexes)]
                except (ValueError, RuntimeError) as exc:
                    if not parked:
                        raise
                    # A relaunch failed (its order unwired meanwhile): what was parked stays parked.
                    failure = str(exc)
                    break
                else:
                    rows.update(zip(todo, scores))
                    hit = []
                left = [k for k in todo if k not in rows]
                if left and not hit and len(left) == len(todo):
                    # Parked, naming none of the call's items: every record still to score was in it.
                    hit = [(j, "parked", PARK_LOST, ()) for j in range(len(todo))]
                # A park that held one record's items alone names its culprit.
                hit_records = {items[todo[j]][0] for j, *_ in hit}
                for j, why, kind, batch_ids in hit:
                    i = items[todo[j]][0]
                    if i not in parked:
                        parked[i] = why
                        self._record_park(sids[i], why, kind, batch_ids, alone=len(hit_records) == 1)
                todo = [k for k in left if items[k][0] not in parked]
            for i, why in parked.items():
                results[i] = _Parked(why)
            unscored = {item[0] for k, item in enumerate(items) if k not in rows and item[0] not in parked}
            for i in unscored:
                results[i] = failure
            kept = [item for k, item in enumerate(items) if item[0] not in parked and item[0] not in unscored]
            kept_rows = [rows[k] for k, item in enumerate(items) if item[0] not in parked and item[0] not in unscored]
            scored_by = {}
            for (i, *_), (_, _, executors, *_) in zip(kept, kept_rows):
                scored_by.setdefault(i, set()).update(executors or ())
            judged = self._verdicts(records, results, kept,
                                    [(status, chunks, *extra) for status, chunks, _, *extra in kept_rows])
            for i, executors in scored_by.items():
                judged[i] = {**judged[i], "scored_by": sorted(executors)}
            for i, sid in enumerate(sids):
                if sid is not None and isinstance(judged[i], dict):
                    # Decided: no longer parked.
                    self._reaudit_parked.pop(sid, None)
                    self._unpark_record(sid)
            if parked:
                first = next(iter(parked))
                state = self._park_state.get(sids[first]) or self._reaudit_parked.get(sids[first])
                logger.warning("eval job %s: %d record(s) parked%s (%s), e.g. %s; tried again in %.0f s at most",
                               self._job_id, len(parked), " in a re-audit" if _REAUDIT_PARKED.get() is not None
                               else "", parked[first][:200], ", ".join((sids[i] or "?")[:12] for i in list(parked)[:3]),
                               (state.retry_at - self._clock()) if state is not None else PARK_RETRY_SECONDS)
            return judged

        def _record_park(self, sid, why, kind, batch_ids, *, alone=False) -> None:
            """Park ``sid`` now (a re-audit keeps its own): its batch's unpark may come before the call ends."""
            if sid is None:
                return
            reaudit = _REAUDIT_PARKED.get()
            if reaudit is not None:
                reaudit.add(sid)
                self._park_again(self._reaudit_parked, sid, why, kind, batch_ids, alone)
            elif sid not in self._judged:
                self._park_again(self._park_state, sid, why, kind, batch_ids, alone)
                self.parked_ids.add(sid)

        def _park_again(self, states, sid, why, kind, batch_ids, alone=False) -> None:
            """Park ``sid`` (or again): tried after PARK_RETRY_SECONDS, doubling up to MAX_PARK_RETRY_SECONDS, or at
            once if its batch was unparked already."""
            now = self._clock()
            state = states.get(sid)
            if state is None:
                state = states[sid] = _ParkState(since=now, backoff=PARK_RETRY_SECONDS)
            else:
                state.backoff = min(MAX_PARK_RETRY_SECONDS, 2 * state.backoff)
                state.tries += 1
            state.retry_at = now + state.backoff
            state.why, state.kind, state.batches = why, kind, set(batch_ids)
            state.alone = state.alone or alone or kind == PARK_UNREADABLE
            for gone in [b for b, at in self._unparked_batches.items() if now - at > LATE_ANSWER_SECONDS]:
                del self._unparked_batches[gone]
            if any(b in self._unparked_batches for b in state.batches):
                # Its batch got a measured answer before this park was recorded: as `_batch_unparked` would.
                state.retry_at, state.backoff = now, PARK_RETRY_SECONDS / 2
                state.tries, state.alone, state.since = 0, False, now

        async def _retry_alone(self, record, sid) -> None:
            """A parked record tried again alone, in the background. Decided: its outcome waits for its pass in
            `_ahead`, and the pass is due now. Parked again: its cool-down doubled (`_park_again`)."""
            state = self._park_state.get(sid)
            if state is not None:
                # Not tried again while this retry is out.
                state.retry_at = self._clock() + max(state.backoff, PARK_RETRY_SECONDS)
            try:
                outcome, = await self._forward_part([record], [sid])
            except (ValueError, RuntimeError):
                logger.warning("eval job %s: retry of parked %s failed on the validator; tried again later",
                               self._job_id, sid[:12], exc_info=True)
                return
            finally:
                self._retries.pop(sid, None)
            if isinstance(outcome, dict) and sid not in self._judged:
                self._ahead[sid] = outcome
                self._schedule(sid, self._clock())

        def _unpark_record(self, sid) -> None:
            if self._park_state.pop(sid, None) is not None:
                self.parked_ids.discard(sid)
                logger.info("eval job %s: parked %s decided", self._job_id, sid[:12])
                self._wake_held(sid)

        def _note_held(self, submission_id: str, parked) -> None:
            for sid in parked:
                self._held_by.setdefault(sid, set()).add(submission_id)

        def _wake_held(self, sid) -> None:
            """A parked record decided or unparked: the passes it held as a sibling are judged again now."""
            now = self._clock()
            for other in self._held_by.pop(sid, ()):
                if other in self._judged:
                    continue
                if other in self._next_due and self._next_due[other] > now:
                    self._next_due[other] = now
                if other in self._due and self._due[other] > now:
                    self._schedule(other, now)

        def _batch_unparked(self, batch_id) -> None:
            """A batch parked on its losses got a measured answer: its records are tried again at once."""
            now = self._clock()
            self._unparked_batches[batch_id] = now
            moved = 0
            for states in (self._park_state, self._reaudit_parked):
                for sid, state in states.items():
                    if batch_id not in state.batches:
                        continue
                    moved += 1
                    # A card answered its batch: the park is no evidence against it any more. It holds the close again
                    # until its retry decides it or parks it alone again (SOLO_RETRY_SECONDS at most).
                    state.retry_at, state.backoff = now, PARK_RETRY_SECONDS / 2
                    state.tries, state.alone, state.since, state.left_out = 0, False, now, False
                    if states is self._park_state:
                        if sid in self._next_due:
                            self._next_due[sid] = now
                        if sid in self._due:
                            self._schedule(sid, now)
                        self._wake_held(sid)
            if moved:
                logger.info("eval job %s: batch %s unparked; %d parked record(s) tried again now", self._job_id,
                            batch_id, moved)

        def _sid_of(self, record) -> str | None:
            """A record's submission id: written in it by the intake; else as its pass or its read named it."""
            sid = record.get("submission_id") if isinstance(record, dict) else None
            if isinstance(sid, str) and sid:
                return sid
            known = self._ids_by_object.get(id(record))
            return known[0] if known is not None and known[1] is record else None

        def _know(self, ids, records) -> None:
            for sid, record in zip(ids, records):
                if isinstance(record, dict) and not record.get("submission_id"):
                    self._ids_by_object[id(record)] = (sid, record)
                    self._ids_by_object.move_to_end(id(record))
            while len(self._ids_by_object) > KNOWN_RECORDS:
                self._ids_by_object.popitem(last=False)

        async def _read_all(self, submission_ids):
            records = await super()._read_all(submission_ids)
            self._know(list(records), list(records.values()))
            return records

        def _mark_judged(self, submission_id: str) -> None:
            super()._mark_judged(submission_id)
            self._unpark_record(submission_id)

        def _is_parked(self, submission_id: str) -> bool:
            return submission_id in self._park_state

        def _give_up_at(self, submission_id: str) -> float | None:
            """When a parked record lets go of its period's close: PARKED_CLOSE_GIVE_UP_SECONDS after its receipt
            (durable: a restart does not start it again). One parked with other records' items, never alone, waits too
            until its first retry alone had SOLO_RETRY_SECONDS: no honest batch-mate leaves before it is decided."""
            meta, state = self._meta.get(submission_id), self._park_state.get(submission_id)
            if meta is None or state is None:
                return None
            give_up = meta[1] + PARKED_CLOSE_GIVE_UP_SECONDS
            if state.alone or state.tries >= 1:
                return give_up
            return max(give_up, state.since + PARK_RETRY_SECONDS + SOLO_RETRY_SECONDS)

        def _parked_wait(self, submission_id: str, now: float) -> float | None:
            state = self._park_state.get(submission_id)
            if state is None or now >= state.retry_at:
                return None
            give_up = self._give_up_at(submission_id)
            return state.retry_at if give_up is None or give_up <= now else min(state.retry_at, give_up)

        def _left_out_of_close(self, submission_id: str, now: float) -> bool:
            # One record no executor can score must not freeze the order's pay: it lets go of the close at its give-up.
            state = self._park_state.get(submission_id)
            give_up = self._give_up_at(submission_id)
            if give_up is None or now <= give_up:
                return False
            if not state.left_out:
                state.left_out = True
                self._left_out_ids.add(submission_id)
                logger.error("eval job %s: parked %s, received %.0f s ago, no longer holds its period's close nor "
                             "its operator's unaudited passes; it is still tried again%s", self._job_id,
                             submission_id[:12], now - self._meta[submission_id][1],
                             ", and its operator's verdicts it could forfeit wait for its decision"
                             if self._params.q < 1.0 else "")
            return True

        def settle_hold(self, verdicts, period_of) -> set[str]:
            """The stake guarantee past a give-up (owner, 2026-11-26): while a parked record left out of the close is
            undecided, the verdicts of its operator received in its period up to its own receipt (what its failure can
            forfeit: `forfeits` voids up to the failure's receipt) are not settled, so that forfeit can still void them
            (`_void_forfeited` skips what is settled). Every other verdict is settled. ``verdicts``: id -> (hotkey, work
            period, received_at). At q = 1 a failure forfeits nothing: nothing is held."""
            if self._params.q >= 1.0:
                return set()
            now = self._clock()
            until: dict[tuple[str, int], float] = {}
            for sid in list(self._park_state):
                if sid in self._meta and sid not in self._judged and self._left_out_of_close(sid, now):
                    hotkey, received_at, _ = self._meta[sid]
                    key = (self._operator(hotkey), period_of(received_at))
                    until[key] = max(until.get(key, float("-inf")), received_at)
            if not until:
                return set()
            held = set()
            for sid, (hotkey, period, received_at) in verdicts.items():
                bound = until.get((self._operator(hotkey), period)) if hotkey else None
                if bound is not None and (received_at is None or float(received_at) <= bound):
                    held.add(sid)
            return held

        def settle_capped(self, verdicts, period_of) -> set[str]:
            """The verdicts paid within what is left of their period's cap, if their period was settled without them
            (owner, 2026-11-26): parked records decided after they left the close (marked in their verdict, so a
            restart keeps it), and, at q < 1, the verdicts their stake hold kept back (their operator's, in their
            period, received up to them). Any other late verdict is paid as before."""
            marked = {sid for sid, verdict in verdicts.items()
                      if sid in self._left_out_ids or (verdict or {}).get("left_out_of_close")}
            if self._params.q >= 1.0:
                return marked
            # The left-out records, from memory and from the marked verdicts of this call (a restart forgets the first).
            until: dict[tuple[str, int], float] = {}
            anchors = [(self._meta[sid][0], self._meta[sid][1]) for sid in self._left_out_ids if sid in self._meta]
            anchors += [(verdicts[sid].get("hotkey"), verdicts[sid].get("received_at")) for sid in marked]
            for hotkey, received_at in anchors:
                if hotkey and received_at is not None:
                    key = (self._operator(hotkey), period_of(received_at))
                    until[key] = max(until.get(key, float("-inf")), float(received_at))
            held = set()
            for sid, verdict in verdicts.items():
                hotkey, received_at = (verdict or {}).get("hotkey"), (verdict or {}).get("received_at")
                if not hotkey or received_at is None:
                    continue
                bound = until.get((self._operator(hotkey), period_of(received_at)))
                if bound is not None and float(received_at) <= bound:
                    held.add(sid)
            return marked | held

        async def _audit_outcomes(self, records, *, local: bool = False):
            # A parked record is neither the miner's fault nor a validator error (`_Parked`): it stays pending, tried
            # again after a cool-down, and its job is flagged.
            try:
                return await self._forward(records, local=local)
            except (ValueError, RuntimeError) as exc:
                return [str(exc) for _ in records]

        async def _audit_records(self, ids, records, draws, *, deferred=None):
            self._know(ids, records)
            try:
                return await super()._audit_records(ids, records, draws, deferred=deferred)
            finally:
                self._reaudit_quarantined()

        def _reaudit_quarantined(self) -> None:
            """A judge process: a pass written after its executor's quarantine event has no event left to catch it."""
            quarantined = getattr(self._remote, "quarantined", None)
            if quarantined is None:
                return
            for executor_id in set(self._remote_scored) & quarantined():
                if self._remote_scored[executor_id] and executor_id not in self._late_reaudits:
                    self._late_reaudits[executor_id] = asyncio.ensure_future(self._reaudit_late(executor_id))

        async def _reaudit_late(self, executor_id: str) -> None:
            delay = REOPEN_RETRY_SECONDS
            try:
                while self._remote_scored.get(executor_id):
                    listed = list(self._remote_scored[executor_id])
                    try:
                        await self.reaudit_executor(executor_id)
                        delay = REOPEN_RETRY_SECONDS
                    except Exception:
                        logger.exception("eval job %s: re-audit of quarantined executor %s failed; retrying",
                                         self._job_id, executor_id)
                        # It took them off the list before failing.
                        known = self._remote_scored.setdefault(executor_id, [])
                        present = set(known)
                        known.extend(sid for sid in listed if sid not in present)
                        await asyncio.sleep(delay)
                        delay = min(MAX_REOPEN_RETRY_SECONDS, 2 * delay)
                        continue
                    left = self._remote_scored.get(executor_id) or []
                    if left and all(sid in self._reaudit_parked for sid in left):
                        # Only parked passes are left, listed again: tried once the first cool-down ends (an unpark
                        # moves it to now; looked at again within a minute).
                        wake = min(self._reaudit_parked[sid].retry_at for sid in left)
                        await asyncio.sleep(min(60.0, max(1.0, wake - self._clock())))
            finally:
                self._late_reaudits.pop(executor_id, None)

        # -- an eval job stays completable: a failure reopens its slot ------
        # (a generation order's never does: today's corpus semantics)

        async def _reopen_once(self, submission_id: str) -> None:
            from reliquary.corpus.eval_draw import has_draw
            from reliquary.validator.corpus_service import draw_failure_slot, record_prompt_failure

            record = await self._records.read_submission(self._job_id, submission_id)
            if record is None:
                return
            outcome = await record_prompt_failure(
                self.job_store, self.job, int(record["prompt_index"]), submission_id,
                **(draw_failure_slot(record) if has_draw(self.job) else {}))
            if outcome is False:
                logger.warning("eval job %s: prompt %s exhausted its attempts",
                               self._job_id, record["prompt_index"])

        async def _record_failure(self, submission_id: str) -> None:
            if not self.reopen_slots or self.job is None or self.job_store is None:
                return
            try:
                await self._reopen_once(submission_id)
            except Exception as exc:
                # Retried at the next start's reconcile, or now in a judge process; never stops the auditor.
                logger.exception("eval job %s: failure of %s not recorded in the ledger",
                                 self._job_id, submission_id[:12])
                if self.retry_reopens and isinstance(exc, _reopen_retried()):
                    self._retry_reopen(submission_id)

        def _retry_reopen(self, submission_id: str) -> None:
            if submission_id not in self._reopening:
                task = asyncio.ensure_future(self._reopen_until_landed(submission_id))
                self._reopening[submission_id] = task

        async def _reopen_until_landed(self, submission_id: str) -> None:
            """A failure that lost its ledger race (the front writes that ledger all the time), tried until it lands."""
            delay = REOPEN_RETRY_SECONDS
            try:
                while True:
                    await asyncio.sleep(delay * (0.5 + random.random() / 2))
                    try:
                        await self._reopen_once(submission_id)
                        logger.info("eval job %s: failure of %s recorded after retrying", self._job_id,
                                    submission_id[:12])
                        return
                    except _reopen_retried():
                        delay = min(MAX_REOPEN_RETRY_SECONDS, 2 * delay)
                    except Exception:
                        logger.exception("eval job %s: failure of %s left to the next start's reconcile",
                                         self._job_id, submission_id[:12])
                        return
            finally:
                self._reopening.pop(submission_id, None)

        async def _reopen_many(self, submission_ids) -> None:
            from reliquary.corpus.eval_draw import has_draw
            from reliquary.validator.corpus_service import record_prompt_failures

            entries = await self._failure_entries(submission_ids, draw=has_draw(self.job))
            outcomes = await record_prompt_failures(self.job_store, self.job, entries)
            exhausted = sum(1 for outcome in outcomes.values() if outcome is False)
            if exhausted:
                logger.warning("eval job %s: %d failure(s) exhausted their prompt's attempts",
                               self._job_id, exhausted)

        async def _reopen_slots(self, submission_ids) -> None:
            """A forfeit's or a reconcile's failures in one ledger write (2026-11-23: 6,819 voided passes never
            reopened, and the order was delivered short). Never raises."""
            await super()._reopen_slots(submission_ids)
            if not submission_ids or not self.reopen_slots or self.job is None or self.job_store is None:
                return
            try:
                await self._reopen_many(submission_ids)
            except Exception as exc:
                logger.exception("eval job %s: %d failure(s) not recorded in the ledger",
                                 self._job_id, len(submission_ids))
                if self.retry_reopens and isinstance(exc, _reopen_retried()):
                    self._reopen_backlog.update(submission_ids)
                    if _REOPEN_BACKLOG not in self._reopening:
                        self._reopening[_REOPEN_BACKLOG] = asyncio.ensure_future(self._backlog_until_landed())

        async def _backlog_until_landed(self) -> None:
            """Every failure of a batch that lost its ledger race, together, until they land."""
            delay = REOPEN_RETRY_SECONDS
            try:
                while self._reopen_backlog:
                    await asyncio.sleep(delay * (0.5 + random.random() / 2))
                    batch = sorted(self._reopen_backlog)
                    try:
                        await self._reopen_many(batch)
                        self._reopen_backlog.difference_update(batch)
                        delay = REOPEN_RETRY_SECONDS
                        logger.info("eval job %s: %d failure(s) recorded after retrying", self._job_id, len(batch))
                    except _reopen_retried():
                        delay = min(MAX_REOPEN_RETRY_SECONDS, 2 * delay)
                    except Exception:
                        logger.exception("eval job %s: %d failure(s) left to the next start's reconcile",
                                         self._job_id, len(batch))
                        self._reopen_backlog.difference_update(batch)
            finally:
                self._reopening.pop(_REOPEN_BACKLOG, None)

        async def _write(self, submission_id, verdict):
            if submission_id in self._left_out_ids:
                # Durable: its late pay stays within the cap after a restart too (`settle_capped`).
                verdict = {**verdict, "left_out_of_close": True}
            stored, written = await super()._write(submission_id, verdict)
            if not (stored or verdict).get("passed"):
                await self._record_failure(submission_id)
            return stored, written

        async def _reopen_reaudited(self, submission_ids, *, drift):
            # Every one the re-audit voids, in its own ledger: the drifts once every void is written, or as soon as one
            # raised, so neither their slow ledger writes hold back the next voids nor a void write that raises leaves
            # them taken until the next start.
            await super()._reopen_reaudited(submission_ids, drift=drift)
            for submission_id in submission_ids:
                await self._record_failure(submission_id)

        async def reaudit_executor(self, executor_id):
            # A pass whose re-audit was parked stays on its executor's list, tried again after its cool-down by
            # `_reaudit_late`: a parked re-audit never leaves a quarantined executor's passes standing in silence.
            # Each failure's slot reopens in the ledger as the re-audit voids it (`_reopen_reaudited`), once.
            listed = list(self._remote_scored.get(executor_id, ()))
            now = self._clock()
            cooling = {sid for sid in listed
                       if sid in self._reaudit_parked and now < self._reaudit_parked[sid].retry_at}
            if cooling:
                self._remote_scored[executor_id] = [sid for sid in listed if sid not in cooling]
            parked: set[str] = set()
            token = _REAUDIT_PARKED.set(parked)
            try:
                failed = await super().reaudit_executor(executor_id)
            finally:
                _REAUDIT_PARKED.reset(token)
                back = [sid for sid in listed if sid in cooling or sid in parked]
                if back:
                    known = self._remote_scored.setdefault(executor_id, [])
                    present = set(known)
                    known.extend(sid for sid in back if sid not in present)
            if parked:
                logger.warning("eval job %s: %d pass(es) of quarantined executor %s parked in their re-audit, kept "
                               "on its list (e.g. %s)", self._job_id, len(parked), executor_id, min(parked)[:12])
            if back and executor_id not in self._late_reaudits:
                self._late_reaudits[executor_id] = asyncio.ensure_future(self._reaudit_late(executor_id))
            return failed

        async def reconcile_failures(self) -> int:
            """Every failed or voided submission recorded in the ledger (a crash
            between a verdict and its ledger write would otherwise leave the
            prompt without its slot). Only those the ledger may lack are read
            (``unrecorded_failures``)."""
            if not self.reopen_slots:
                return 0
            from reliquary.validator.corpus_auditor import unrecorded_failures

            # The job's ledger, the one its failures are written to; with a slot store's as well, every one is read.
            ledger = self.job_store if getattr(self, "_slot_store", None) is None else None
            failed, unrecorded = await unrecorded_failures(self._records, self._job_id, ledger, self.job)
            await self._reopen_slots(unrecorded)
            return len(failed)

        async def restore_scored(self) -> int:
            """A judge process that restarted: which executor scored each pass a quarantine could still void
            (unsettled, or settled inside the hold, as ``reaudit_executor`` chooses), read back from the verdicts,
            then the re-audit of those scored by an executor already quarantined, in slices, a few at once
            (`REAUDIT_SLICES_IN_FLIGHT`). A pass that re-audit confirmed is re-audited again at every start, until it
            is settled and its audit older than the hold: nothing durable records the confirmation (its verdict is
            create-only). Its count is not added to again (`_count_draw`).

            On a forced-draw job, the draw counts first take every audit written before this process as the judging
            passes and re-audits that wrote them counted it (`_count_restored`): every verdict, voided or not, scored
            by any executor or audited here, and the void of each voided audited pass an executor scored. Each id is
            counted as it is read, through the one guarded count (`_count_draw`), so an id a judging pass or an
            earlier attempt counted is counted once. No pass is listed before every verdict is counted: a re-audit
            (this one's, or one a judging pass starts, `_reaudit_late`) then counts its drifts as refusals of passes
            already counted and judges them on whole counts, as in a process that never restarted, and until then no
            refusal is charged for frequent refusals (`_draw_restoring`). Counted afresh instead, that re-audit
            counted every pass it confirmed and none of the refusals written beside them, so the job's rate read as
            no refusal at all and honest hotkeys were charged for frequent refusals (2,000 restored passes: a fresh
            hotkey charged for 3 drifts in 50 audits). The settler waits for this (``scores_restored``)."""
            from reliquary.validator.corpus_auditor import RECONCILE_READS

            ids = list(await self._records.list_verdict_ids(self._job_id))
            lister = getattr(self._records, "list_voided_ids", None)
            voided = set(await lister(self._job_id)) if lister is not None else set()
            state, _ = await self._records.read_settlement(self._job_id)
            settled = set((state or {}).get("settled") or ())
            gate = asyncio.Semaphore(RECONCILE_READS)
            draw = self._draw_job is not None
            void_reader = getattr(self._records, "read_voided", None) if draw else None

            async def read(submission_id):
                async with gate:
                    return submission_id, await self._records.read_verdict(self._job_id, submission_id)

            async def read_void(submission_id):
                async with gate:
                    return submission_id, await void_reader(self._job_id, submission_id)

            now, restored = self._clock(), 0
            # Per executor, the passes to list once every verdict is read (and counted).
            scored: dict[str, list[str]] = {}
            # Voided verdicts too on a forced-draw job: a judging pass counted them; only an unvoided pass is listed.
            wanted = ids if draw else [sid for sid in ids if sid not in voided]
            for start in range(0, len(wanted), RECONCILE_READS):
                verdicts = await asyncio.gather(*map(read, wanted[start:start + RECONCILE_READS]))
                voids = {}
                if void_reader is not None:
                    reaudited = [sid for sid, verdict in verdicts if sid in voided and verdict
                                 and verdict.get("passed") and verdict.get("audited") and verdict.get("scored_by")]
                    voids = dict(await asyncio.gather(*map(read_void, reaudited)))
                for sid, verdict in verdicts:
                    if draw and verdict:
                        self._count_restored(sid, verdict, voids.get(sid))
                    if sid in voided or not verdict or not verdict.get("passed") or not verdict.get("audited"):
                        continue
                    if sid in settled and now - float(verdict.get("audited_at") or 0.0) > self._params.hold_seconds:
                        continue
                    for executor_id in verdict.get("scored_by") or ():
                        scored.setdefault(executor_id, []).append(sid)
            # Every audit written before this process is counted: the frequent-refusals rule reads whole counts.
            self._draw_restoring = False
            for executor_id, sids in scored.items():
                known = self._remote_scored.setdefault(executor_id, [])
                present = set(known)
                for sid in sids:
                    if sid not in present:
                        present.add(sid)
                        known.append(sid)
                        restored += 1
            # Last: a quarantine heard while reading took only what was listed by then.
            quarantined = self._remote.quarantined()
            for executor_id in sorted(set(self._remote_scored) & quarantined):
                await self.reaudit_executor(executor_id)
            return restored

        async def _restore_behind(self):
            await self.seeded.wait()
            delay = REOPEN_RETRY_SECONDS
            while True:
                try:
                    restored = await self.restore_scored()
                    logger.info("eval job %s: %d executor score(s) restored for re-audit", self._job_id, restored)
                    self.scores_restored.set()
                    return
                except Exception:
                    logger.exception("eval job %s: executor scores not restored; retrying", self._job_id)
                    await asyncio.sleep(delay)
                    delay = min(MAX_REOPEN_RETRY_SECONDS, 2 * delay)

        async def _reconcile_behind(self):
            try:
                await self.reconcile_failures()
            except Exception:
                logger.exception("eval job %s: failure reconcile failed", self._job_id)

        async def run(self):
            # Behind the audit, not before it: a restarted control must not leave the job unaudited while
            # its verdicts are read (2026-11-11: ~10 minutes for 10k verdicts).
            # The base run reconciles a forced-draw job's failures itself (`_reconcile_quietly`).
            reconcile = (None if getattr(self, "_slot_store", None) is not None
                         else asyncio.ensure_future(self._reconcile_behind()))
            restore = asyncio.ensure_future(self._restore_behind()) if self.restores_scored else None
            try:
                await super().run()
            finally:
                if reconcile is not None:
                    reconcile.cancel()
                if restore is not None:
                    restore.cancel()
                    await asyncio.gather(restore, return_exceptions=True)
                late = list(self._late_reaudits.values()) + list(self._retries.values())
                for task in late:
                    task.cancel()
                await asyncio.gather(*late, return_exceptions=True)
                retries = list(self._reopening.values())
                for task in retries:
                    task.cancel()
                await asyncio.gather(*retries, return_exceptions=True)

    vocab_size = kwargs.pop("vocab_size")
    job = kwargs.pop("job", None)
    job_store = kwargs.pop("job_store", None)
    # Off unless asked: only an eval job reopens a slot on a failed audit.
    reopen_slots = kwargs.pop("reopen_slots", False)
    retry_reopens = kwargs.pop("retry_reopens", False)
    restores_scored = kwargs.pop("restores_scored", False)
    auditor = EvalAuditor(model=_VocabularyOnly(vocab_size), **kwargs)
    auditor.reopen_slots = bool(reopen_slots)
    # A judge process: a reopen that lost every ledger attempt is tried again until it lands.
    auditor.retry_reopens = bool(retry_reopens)
    auditor._reopening = {}
    auditor._reopen_backlog = set()
    # A judge process: who scored what survives its restart (`restore_scored`), and on a forced-draw job the draw
    # counts too, no refusal charged for frequent refusals until they are whole.
    auditor.restores_scored = bool(restores_scored)
    auditor._draw_restoring = auditor.restores_scored and auditor._draw_job is not None
    auditor.scores_restored = asyncio.Event()
    auditor._late_reaudits = {}
    # Pending records parked (`_ParkState`), and passes parked in a re-audit, by submission id.
    auditor.parked_ids = set()
    auditor._park_state = {}
    auditor._reaudit_parked = {}
    # Parked records once left out of their close (their late verdict is paid within the cap), and batches unparked
    # lately (a park recorded after its batch's unpark is tried at once).
    auditor._left_out_ids = set()
    auditor._unparked_batches = {}
    # Retries of parked records running alone, and the passes each parked record holds as a sibling.
    auditor._retries = {}
    auditor._held_by = {}
    auditor._ids_by_object = collections.OrderedDict()
    if callable(getattr(auditor._remote, "subscribe_unparked", None)):
        auditor._remote.subscribe_unparked(auditor._batch_unparked)
    auditor.job, auditor.job_store = job, job_store
    return auditor


def _reopen_retried() -> tuple:
    """What a slot reopen is tried again after: a lost ledger race or a store fault."""
    from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict
    from reliquary.validator.corpus_miner_states import _TRANSIENT_ERRORS

    return (CorpusStoreConflict, *_TRANSIENT_ERRORS)


class _Parked(str):
    """An outcome that is neither a verdict nor a validator error: the record stays pending, parked, and counts
    nowhere towards the auditor's halt (``undecided``, read by ``CorpusAuditor``)."""

    undecided = True


@dataclass
class _ParkState:
    """A parked record: since when, why and how (PARK_UNREADABLE / PARK_LOST), the batches parked with it, when it is
    tried again, its current cool-down, how often it was parked again, whether a park held its items alone (no other
    record's: it is the culprit), and whether its give-up was logged."""
    since: float
    backoff: float
    retry_at: float = 0.0
    why: str = ""
    kind: str = ""
    batches: set = field(default_factory=set)
    tries: int = 0
    alone: bool = False
    left_out: bool = False


# Set while `reaudit_executor` runs: the passes its audits park, which it keeps on its executor's list.
_REAUDIT_PARKED: contextvars.ContextVar[set | None] = contextvars.ContextVar("reaudit_parked", default=None)


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
                lease = dispatcher.claim(document, tuple(body.protocols))
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


class OrderArchives:
    """The guard on the settler's period archives for order tasks
    (``R2PeriodArchives(guard=...)``): written only under a task this process
    wired and named order-eval- or order-gen- (RELIQUARY_TASK_ID lists no order
    task: they are all wired hot)."""

    def __init__(self, *, served: Callable[[], Any]) -> None:
        self._served = served

    def refuse_unserved(self, task_id: str) -> None:
        from reliquary.eval.prompt_source import is_order_job_id

        if not is_order_job_id(task_id) or task_id not in set(self._served()):
            raise RuntimeError(f"task {task_id!r} is not an order task this process serves; "
                               "refusing to archive")


EvalArchives = OrderArchives


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


def order_job_refusal(entry, job) -> str | None:
    """Why the order control will not serve a registry entry, or None: an eval
    job draws from its eval set, a generation order from a packaged catalog
    source through the model's chat template; both carry an enforced TOPLOC
    proof on the job's own checkpoint."""
    from reliquary.eval.prompt_source import is_eval_job_id, is_eval_source, is_gen_job_id
    from reliquary.protocol import profiles

    if is_eval_job_id(entry.job_id):
        if not is_eval_source(job.prompt_source):
            return f"prompt source {job.prompt_source!r} is not an eval set"
        episode = getattr(job, "episode", None)
        if episode is not None and not (is_signed_sandbox(job) or getattr(episode, "is_suite", False)):
            # A replayed SWE episode's steps would be trusted to the miner's box. A suite episode is replayed
            # here, its every observation recomputed (``suite_intake``).
            return "an order plays an episode only in the signed sandbox or as a replayed suite episode"
    elif is_gen_job_id(entry.job_id):
        from reliquary.eval.qualification import order_environment_refusal
        from reliquary.validator.corpus_service import CHAT_TEMPLATE_RENDERERS

        declared = list(((getattr(entry, "contract", None) or {}).get("environments") or {}))
        if is_eval_source(job.prompt_source):
            # A frozen set (ours or a customer's upload): a forced-draw order, one declared environment.
            if not has_draw(job):
                return "a generation order draws from a set only under a forced draw"
            if getattr(entry, "contract", None) is not None and len(declared) != 1:
                return f"its contract declares environments {declared}, not one"
        else:
            refusal = order_environment_refusal(job.prompt_source)
            if refusal is not None:
                return refusal
            if getattr(entry, "contract", None) is not None and declared != [job.prompt_source]:
                return f"its contract declares environments {declared}, not {job.prompt_source!r} alone"
        if job.renderer_id not in CHAT_TEMPLATE_RENDERERS:
            return f"renderer {job.renderer_id!r} is not the model's chat template"
        if getattr(job, "submit", None) != "scoped":
            # Its miners would post to /corpus/submit, which this control lacks.
            return "its manifest does not send submissions to its scoped route"
    else:
        return "not an order job"
    if getattr(entry, "contract", None) is None:
        return "it carries no contract"
    from reliquary.constants import SUPPORTED_ARCHITECTURES

    try:
        profile = profiles.profile_from_contract(entry.contract)
    except ValueError as exc:
        return f"its contract cannot be read: {exc}"
    proof = profiles.toploc_proof(profile)
    if proof is None or proof.mode != "enforce":
        return "its contract names no enforced toploc proof"
    architecture = getattr(profile, "model_architecture", None)
    if architecture not in SUPPORTED_ARCHITECTURES:
        # Whatever wrote the entry: this table, never a self-declared string.
        return f"its architecture {architecture!r} is not supported"
    if getattr(job, "episode", None) is not None:
        from reliquary.eval.episode_jobs import episode_renderer_refusal

        refusal = episode_renderer_refusal(architecture, job.episode.renderer,
                                           suite=getattr(job.episode, "is_suite", False))
        if refusal is not None:
            return refusal
    if (profile.model_id, profile.model_revision) != (job.checkpoint_repo,
                                                      job.checkpoint_revision):
        return "its contract's model is not the job's checkpoint"
    return None


eval_job_refusal = order_job_refusal


def qualification_refusal(entry, job, declared: dict | None, record: dict | None) -> str | None:
    """Why the qualification behind an order job does not back it, or None.
    ``declared`` is the job's order record (written by the admin), ``record``
    its qualification: a job the admin did not declare from a qualified
    record of a supported architecture, under exactly its conditions, is
    never served."""
    from reliquary.constants import SUPPORTED_ARCHITECTURES
    from reliquary.eval import qualification as qual
    from reliquary.eval.prompt_source import is_gen_job_id, parse_eval_source
    from reliquary.validator import corpus_service

    if declared is None or declared.get("job_id") != job.job_id:
        return "no order record names this job: only the admin declares order jobs"
    if declared.get("schema") == qual.ORDER_JOB_SCHEMA_V2:
        return pinned_order_refusal(entry, job, declared)
    if getattr(job, "episode", None) is not None:
        # A qualification measured single-turn completions: an episode order is a forced-draw one.
        return "an episode order is a forced-draw order, never a qualified one"
    if record is None or record.get("qualification_id") != declared.get("qualification_id"):
        return f"its qualification {declared.get('qualification_id')!r} is not readable"
    if record.get("status") != qual.QUALIFIED:
        return f"its qualification is {record.get('status')!r}, not qualified"
    generation = is_gen_job_id(job.job_id)
    if (record.get("kind") == qual.GENERATION) != generation \
            or (declared.get("kind") == qual.GENERATION) != generation:
        return "its qualification is of another kind"
    if (record.get("model"), record.get("revision")) != (job.checkpoint_repo,
                                                         job.checkpoint_revision):
        return "its qualification is for another model"
    result = record.get("result") or {}
    contract = getattr(entry, "contract", None) or {}
    if result.get("architecture") not in SUPPORTED_ARCHITECTURES \
            or result.get("architecture") != contract.get("model_architecture"):
        return f"its architecture {result.get('architecture')!r} is not the supported one declared"
    if result.get("checkpoint_sha256") != job.checkpoint_sha256:
        return "its checkpoint fingerprint is not the qualified one"
    if result.get("eos_token_id") != job.eos_token_id:
        return "its eos is not the qualified one"
    toploc = [p for p in contract.get("proofs") or () if p.get("scheme") == "toploc-v1"]
    measured = result.get("thresholds") or {}
    if not toploc or any(toploc[0].get(k) != v for k, v in measured.items()) or not measured:
        return "its toploc thresholds are not the qualified ones"
    sampling = {"temperature": job.sampling.temperature, "top_p": job.sampling.top_p,
                "top_k": job.sampling.top_k}
    wanted = {"sampling": {"top_p": 1.0, "top_k": 0, **(record.get("sampling") or {})},
              "max_new_tokens": record.get("max_new_tokens"),
              "thinking": record.get("thinking"), "samples": declared.get("samples")}
    given = {"sampling": sampling, "max_new_tokens": job.sampling.max_new_tokens,
             "thinking": corpus_service.CHAT_TEMPLATE_RENDERERS.get(job.renderer_id),
             "samples": job.slots_per_prompt}
    if generation:
        wanted.update(env=record.get("env"), prompt_start=record.get("prompt_start"),
                      problems=record.get("problems"))
        given.update(env=job.prompt_source, prompt_start=job.prompt_start,
                     problems=job.prompt_count)
    else:
        source = parse_eval_source(job.prompt_source)
        wanted.update(set_id=record.get("set_id"), problems=record.get("problems"))
        given.update(set_id=source.set_id, problems=source.count)
    differs = sorted(k for k in wanted if wanted[k] != given[k]
                     or (k in declared and k != "samples" and declared[k] != given[k]))
    if differs:
        return f"its conditions differ from its qualification: {differs}"
    if generation:
        # The package this control loads is the one the order was qualified on.
        spec = corpus_service.ENVIRONMENT_SPECS.get(job.prompt_source)
        here = getattr(spec, "environment_manifest_sha256", None)
        in_contract = ((contract.get("environments") or {}).get(job.prompt_source) or {}).get(
            "environment_manifest_sha256")
        if not here or here != record.get("environment_manifest_sha256") or here != in_contract:
            return (f"the {job.prompt_source!r} package here ({str(here)[:12]}) is not the "
                    "qualified one or the contract's")
    return None


def pinned_order_refusal(entry, job, declared: dict) -> str | None:
    """Why a forced-draw order's job does not match the order record the admin wrote, or None. No qualification
    backs it: its model pin, source and conditions are the record's, and its thresholds its judge."""
    from reliquary.constants import SUPPORTED_ARCHITECTURES
    from reliquary.eval import qualification as qual
    from reliquary.eval.prompt_source import is_eval_source, is_gen_job_id, parse_eval_source
    from reliquary.validator import corpus_service

    generation = is_gen_job_id(job.job_id)
    if (declared.get("kind") == qual.GENERATION) != generation:
        return "its order record is of another kind"
    if not has_draw(job) or declared.get("draw") != job.sampling.draw:
        return "an order without a qualification is a forced-draw order"
    pin = declared.get("pin") or {}
    contract = getattr(entry, "contract", None) or {}
    if (pin.get("model"), pin.get("revision")) != (job.checkpoint_repo, job.checkpoint_revision):
        return "its order record pins another model"
    if pin.get("architecture") not in SUPPORTED_ARCHITECTURES \
            or pin.get("architecture") != contract.get("model_architecture"):
        return f"its architecture {pin.get('architecture')!r} is not the supported one declared"
    if pin.get("checkpoint_sha256") != job.checkpoint_sha256:
        return "its checkpoint fingerprint is not the pinned one"
    if pin.get("eos_token_id") != job.eos_token_id:
        return "its eos is not the pinned one"
    from reliquary.corpus.job import draw_judge_from

    sampling = {"temperature": job.sampling.temperature, "top_p": job.sampling.top_p,
                "top_k": job.sampling.top_k, **({"min_p": job.sampling.min_p} if job.sampling.min_p else {})}
    # Every condition the admin declared, compared whole: a job unlike its record is never served.
    keys = ("sampling", "max_new_tokens", "thinking", "episode", "samples", "seed", "n", "min_new_tokens",
            "max_prompt_tokens", "judge", "audit_q")
    wanted = {k: declared.get(k) for k in keys}
    try:
        wanted["judge"] = draw_judge_from(declared.get("judge") or {}).to_contract()
    except ValueError:
        return "its order record's judge cannot be read"
    params = getattr(entry, "params", None) or {}
    given = {"sampling": sampling, "max_new_tokens": job.sampling.max_new_tokens,
             "thinking": corpus_service.CHAT_TEMPLATE_RENDERERS.get(job.renderer_id),
             "episode": job.episode.to_contract() if getattr(job, "episode", None) is not None else None,
             "samples": job.slots_per_prompt, "seed": job.seed, "n": job.sampling.n,
             "min_new_tokens": job.sampling.min_new_tokens,
             "max_prompt_tokens": getattr(job, "max_prompt_tokens", None), "judge": job.judge.to_contract(),
             "audit_q": params.get("audit_q")}
    source = declared.get("source") or {}
    if is_eval_source(job.prompt_source):
        parsed = parse_eval_source(job.prompt_source)
        wanted["source"] = {"set_id": source.get("set_id"), "problems": source.get("problems"), "prompt_start": 0}
        given["source"] = {"set_id": parsed.set_id, "problems": parsed.count, "prompt_start": job.prompt_start}
        wanted["environments"] = [declared.get("contract_environment")]
        given["environments"] = list((contract.get("environments") or {}))
    else:
        wanted["source"] = {k: source.get(k) for k in ("env", "prompt_start", "problems")}
        given["source"] = {"env": job.prompt_source, "prompt_start": job.prompt_start,
                           "problems": job.prompt_count}
    if not generation and wanted["audit_q"] != 1.0:
        return "an evaluation order audits every submission"
    differs = sorted(k for k in wanted if wanted[k] != given[k])
    if differs:
        return f"its conditions differ from its order record: {differs}"
    if not is_eval_source(job.prompt_source):
        spec = corpus_service.ENVIRONMENT_SPECS.get(job.prompt_source)
        here = getattr(spec, "environment_manifest_sha256", None)
        in_contract = ((contract.get("environments") or {}).get(job.prompt_source) or {}).get(
            "environment_manifest_sha256")
        if not here or here != source.get("environment_manifest_sha256") or here != in_contract:
            return (f"the {job.prompt_source!r} package here ({str(here)[:12]}) is not the "
                    "ordered one or the contract's")
    return None


# A tokenizer's files: never the weights, never a stray large text file.
TOKENIZER_PATTERNS = ["*.json", "*.model", "*.tiktoken", "merges.txt", "vocab.txt", "*.jinja"]


def _model_files(repo: str, revision: str) -> str:
    """The model's small files at the pinned revision (public repos only: a
    gated or missing repo is a refusal, never retried as transient; the hub
    being unreachable is transient)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        GatedRepoError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        return snapshot_download(repo, revision=revision, token=False,
                                 allow_patterns=TOKENIZER_PATTERNS)
    except LocalEntryNotFoundError as exc:
        # Nothing cached and the hub unreachable: try again later.
        raise ConnectionError(f"{repo}@{revision}: the hub is unreachable") from exc
    except (GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError,
            EntryNotFoundError) as exc:
        raise ValueError(f"{repo}@{revision} is not a public model at that revision: "
                         f"{type(exc).__name__}") from exc


def _model_config(directory: str, repo: str, revision: str) -> dict:
    import json
    import os

    try:
        with open(os.path.join(directory, "config.json")) as handle:
            return json.loads(handle.read())
    except (OSError, ValueError) as exc:
        # An adapter-only or GGUF-only repo: deterministic, refused.
        raise ValueError(f"{repo}@{revision} has no readable config.json") from exc


def _cpu_tokenizer(directory: str, repo: str, revision: str):
    from reliquary.shared.modeling import load_tokenizer

    try:
        return load_tokenizer(directory)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"{repo}@{revision}: no tokenizer loads from its files "
                         f"({type(exc).__name__})") from exc


def load_cpu_tokenizer(repo: str, revision: str):
    """A model's tokenizer and vocabulary size (the embedding rows the config
    declares, the auditor's bound on token ids), without its weights."""
    directory = _model_files(repo, revision)
    config = _model_config(directory, repo, revision)
    vocab = config.get("vocab_size") or (config.get("text_config") or {}).get("vocab_size")
    if not vocab:
        raise ValueError(f"{repo}@{revision} declares no vocab_size")
    return _cpu_tokenizer(directory, repo, revision), int(vocab)


def _declared_stops(directory: str) -> list[int]:
    """The stop tokens the model's own ``generation_config.json`` declares (none when it has no such file)."""
    import json
    from pathlib import Path

    try:
        value = json.loads((Path(directory) / "generation_config.json").read_text()).get("eos_token_id")
    except (OSError, ValueError, AttributeError):
        return []
    values = value if isinstance(value, list) else [value]
    return [int(v) for v in values if isinstance(v, int) and not isinstance(v, bool)]


def model_facts(repo: str, revision: str) -> dict:
    """The model's architecture and eos (``turn_end_token_id``), read here from its own files (CPU):
    never taken from an executor. ValueError: the model can never be served
    (no config, no tokenizer, not public); any other error is transient."""
    from reliquary.constants import architecture_of
    from reliquary.shared.modeling import turn_end_token_id

    directory = _model_files(repo, revision)
    config = _model_config(directory, repo, revision)
    architecture = architecture_of(config)
    eos = turn_end_token_id(_cpu_tokenizer(directory, repo, revision), _declared_stops(directory))
    if architecture is None or eos is None:
        raise ValueError(f"{repo}@{revision} names no architecture or eos")
    return {"architecture": architecture, "eos_token_id": int(eos)}


def _also(first, second):
    """A ``(submission_id, verdict)`` hook calling ``first`` then ``second``; ``second`` failing never breaks it."""

    def both(submission_id, verdict) -> None:
        first(submission_id, verdict)
        try:
            second(submission_id, verdict)
        except Exception:
            logger.exception("verdict hook failed for %s", submission_id)

    return both


def wire_order_judge(w, *, records, job_store, remote, params, miner_states, beacon, round_at,
                     settle_archives=None, operator_of=None, arrivals_covered=None, threads=None,
                     settle_executor=None, retry_reopens: bool = False,
                     restores_scored: bool = False) -> None:
    """One order's miner book, settler and auditor on ``w`` (``entry``, ``cap``, ``job``, ``tokenizer``,
    ``proof``, ``vocab_size``, ``stats``, and ``grader`` for an episode order), in whichever process judges it."""
    from reliquary.eval.prompt_source import is_eval_source
    from reliquary.infrastructure.corpus_period_store import R2PeriodArchives
    from reliquary.validator import corpus_periods as cp
    from reliquary.validator.corpus_miner_status import (
        MinerBook, feed, proof_thresholds, read_recent_windows,
    )
    from reliquary.validator.corpus_period_settlement import CorpusPeriodSettler, oldest_of
    from reliquary.validator.corpus_settlement import SETTLE_FULL_LIST_SECONDS, settler_fed

    entry, job, cap = w.entry, w.job, w.cap
    # The miner status route's view (`/corpus/jobs/{job}/miners/{hotkey}`), fed by the same reports.
    # A draw job carries no TOPLOC proof: its judge is in the contract.
    w.miners = MinerBook(job_id=job.job_id, task_id=entry.task_id, records=records,
                         read_windows=read_recent_windows,
                         thresholds=None if has_draw(job) else proof_thresholds(w.proof))
    on_verdict, on_settled = feed(w.stats, w.miners)
    on_read = w.stats.observe
    if job.episode is not None:
        # Its audited tokens per episode, for the status's `sandbox` (`order_sandbox_status`).
        from reliquary.validator.order_sandbox_status import EpisodeTokens

        w.episode_tokens = EpisodeTokens()
        on_verdict, on_read = _also(on_verdict, w.episode_tokens.observe), _also(on_read, w.episode_tokens.observe)
    # Paid by period, as every corpus task (design 2026-10-03): a period
    # closes once the auditor holds nothing undecided received in it -- and, for an
    # episode order, its grader nothing ungraded (ruling P21): never paid ungraded.
    grader = getattr(w, "grader", None)
    pending = [lambda: w.auditor.oldest_pending_received_at()]
    if grader is not None:
        pending.append(grader.oldest_unready_received_at)
        grader.on_voided = w.miners.voided
    w.settler = CorpusPeriodSettler(
        task_id=entry.task_id, job_id=job.job_id, cap=cap, records=records,
        archives=R2PeriodArchives(guard=settle_archives),
        oldest_pending=oldest_of(*pending), ready=grader.ready if grader is not None else None,
        on_settled=on_settled, full_list_every_seconds=SETTLE_FULL_LIST_SECONDS, executor=settle_executor,
        max_price=cp.max_price(entry.params), max_input_price=cp.max_input_price(entry.params),
        # The stake guarantee past a parked record's give-up, and its late verdict paid within the cap.
        hold=lambda verdicts, period_of: w.auditor.settle_hold(verdicts, period_of),
        capped=lambda verdicts, period_of: w.auditor.settle_capped(verdicts, period_of))
    if (getattr(entry, "params", None) or {}).get("order_contract", {}).get("mode") == "observation":
        from reliquary.corpus.contract_observation import ContractObservation

        try:
            observation = ContractObservation.from_env()
            if observation is not None:
                w.settler.on_observed_archive = observation.recorder(entry)
        except Exception:
            logger.exception("could not attach the local archive observer to %s", entry.job_id)
    w.auditor = eval_auditor(
        job_id=job.job_id, records=records, tokenizer=w.tokenizer, proof=w.proof,
        params=params, miner_states=miner_states, beacon=beacon, round_at=round_at,
        on_verdict=settler_fed(w.settler, on_verdict), on_voided=w.miners.voided, vocab_size=w.vocab_size,
        job=job, job_store=job_store, reopen_slots=is_eval_source(job.prompt_source) or has_draw(job),
        draw_job=job if has_draw(job) else None, operator_of=operator_of, remote=remote,
        arrivals_covered=arrivals_covered, threads=threads, retry_reopens=retry_reopens,
        restores_scored=restores_scored)
    w.settler.on_window = w.miners.window
    w.settler.on_read = on_read
    # The stake weighs a record's prompt as the settler pays it (`corpus_periods.work_value`).
    output_price, input_price = cp.max_price(entry.params), cp.max_input_price(entry.params)
    if input_price and output_price:
        w.auditor.stake_input_weight = float(input_price) / float(output_price)
    elif input_price:
        # Prompts paid at a price no output price compares with: the corpus rule, never a guessed stake.
        w.auditor.stake_at_receipt = False


def build_eval_control(*, store, records, dispatcher: PairedAuditDispatcher,
                       directory: EvalExecutorDirectory, verify_signature,
                       verify_skip_signature=None, tokenizer_for=load_cpu_tokenizer,
                       qualifications=None, registration=None, settle_archives=None,
                       read_entries=None, clock: Callable[[], float] = time.time,
                       refresh_every_seconds: float = 60.0, qualification_store=None,
                       order_jobs=None, sandbox=None, sandbox_down: str | None = None,
                       model_dir_for=None, signed_wiring=None, suite_replayer_for=None,
                       operator_of: Callable[[str], str | None] | None = None, order_judge=None,
                       contract_observation=None, current_owners=None):
    """The app and the job set of the eval control. Each wired job gets its own
    tokenizer, renderer, router, auditor (an executor pair per batch) and
    settler; ``read_entries`` makes the set hot.

    ``sandbox`` (``sandbox_wiring.SandboxServices``, started) serves signed-sandbox episode
    orders as the corpus validator does: the session route, each job's signed intake and
    its grader, whose grades gate the job's pay. Without it (``sandbox_down`` says why) an
    episode order is refused. ``model_dir_for(repo, revision)``: the model's tokenizer files
    the turn renderer loads; ``signed_wiring``: extra ``wire_signed_job`` arguments (tests).
    A suite episode order needs no sandbox: replayed here, by ``suite_replayer_for(tokenizer)``
    (``suite_intake.PoolSuiteReplayer`` by default). ``operator_of(hotkey)``: its registration's coldkey (None:
    unknown), whose hotkeys every auditor stakes and forfeits together (``CorpusAuditor._operator``).
    ``order_judge`` (``order_judge.OrderJudgeFront``): non-episode orders are judged and paid in its process."""
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
        recover_pending_records,
        prompt_job_for_spec,
        renderer_for_job,
    )
    from reliquary.validator.corpus_validator import (
        WINDOW_SETTLED,
        add_public_corpus_routes,
        build_corpus_audit_wiring,
        window_settled,
    )

    tokenizers: dict[ModelKey, tuple[Any, int]] = {}
    served: dict[str, Any] = {}
    suite_replayers: dict[ModelKey, Any] = {}
    if suite_replayer_for is None:
        def suite_replayer_for(tokenizer):
            from reliquary.validator.suite_intake import PoolSuiteReplayer

            return PoolSuiteReplayer.for_tokenizer(tokenizer)
    model_dir_for = model_dir_for or _model_files
    grade_parse_threads = None
    if sandbox is not None:
        from concurrent.futures import ThreadPoolExecutor

        from reliquary.validator.corpus_grading import GRADE_PARSE_THREADS

        # The graders' trajectory parses: never the default executor the submit routes use.
        grade_parse_threads = ThreadPoolExecutor(GRADE_PARSE_THREADS, thread_name_prefix="order-grade-parse")
    if qualification_store is None or order_jobs is None:
        from reliquary.eval.qualification import EvalJobStore, QualificationStore

        qualification_store = qualification_store or QualificationStore()
        order_jobs = order_jobs or EvalJobStore()

    def refused(job_id: str, why: str) -> None:
        # Served by nobody: said in the status, not only in a log line.
        dispatcher.attention[str(job_id)].append(f"refused: {why}"[:300])

    def router_for(w):
        from reliquary.validator.corpus_validator import _lease_verifier

        router = build_corpus_router(
            job_id=str(w.entry.job_id), store=store, tokenizer=w.tokenizer,
            renderer=w.renderer, verify_signature=verify_signature,
            verify_skip_signature=verify_skip_signature, prompt_job_for=w.prompt_job_for,
            verify_lease_signature=_lease_verifier(verify_skip_signature),
            records=records, on_accepted=w.on_accepted,
            # A draw job's samples carry no proof: the draw is their check.
            proof_chunk_tokens=None if has_draw(w.job) else w.proof.chunk_tokens, vocab_size=w.vocab_size,
            is_banned=w.is_banned, registration=registration, seen_index=w.seen_index,
            suspect_until=getattr(w, "suspect_until", None),
            job=getattr(w, "job", None), durable_records=False, auction=_auction_of(w),
            lease_fault=_lease_fault_of(w),
            # Held by this control for its audit (a restart's seed, a judge down): no lease runs out meanwhile.
            lease_held=functools.partial(routes.lease_clock_stopped, str(w.entry.job_id)),
            episode_intake=getattr(w, "episode_intake", None))
        w.sync_lease_clock = router.sync_lease_clock
        return router

    def _lease_fault_of(w):
        """A signed episode order's draw leases our sandbox failed: released uncharged
        (``sandbox.corpus_lease_faults``)."""
        job = getattr(w, "job", None)
        if sandbox is None or job is None or job.episode is None or not is_signed_sandbox(job):
            return None
        from reliquary.sandbox.corpus_lease_faults import corpus_lease_fault

        return corpus_lease_fault(sandbox.issuer, str(job.job_id))

    def _auction_of(w):
        """The job's token auction when its entry prices it so (``corpus.token_auction``), its leases opened by
        what waits for the audit: the auditor's queue and the batches sent to executors."""
        from reliquary.corpus.token_auction import job_auction, state_path_for

        if not hasattr(w, "auction"):
            job_id = str(w.entry.job_id)
            w.auction = job_auction(
                getattr(w.entry, "params", None),
                backlog=lambda: w.auditor.audit_backlog() + dispatcher.pending_items(job_id),
                state_path=state_path_for(job_id), job=getattr(w, "job", None))
            job = getattr(w, "job", None)
            if w.auction is not None and w.auction.measures and job is not None:
                # The merit order fills what the model's audit cards verify, at the share this job audits.
                model = (job.checkpoint_repo, job.checkpoint_revision)
                w.auction.bind_merit(audit=lambda: dispatcher.audit_pace(model),
                                     q=lambda: w.auditor.audit_share(),
                                     q_weight=lambda: w.auditor.audit_decisions())
        return w.auction

    async def wire(entry, cap, job):
        refusal = order_job_refusal(entry, job)
        if refusal is None and window_settled(entry):
            refusal = WINDOW_SETTLED
        if refusal is None:
            declared = await order_jobs.read(job.job_id)
            record = None
            if declared is not None and declared.get("qualification_id"):
                record, _ = await qualification_store.read(declared["qualification_id"])
            refusal = qualification_refusal(entry, job, declared, record)
        if refusal is None and job.episode is not None and is_signed_sandbox(job) and sandbox is None:
            refusal = ("a signed-sandbox episode order and this order control serves no sandbox: "
                       + (sandbox_down or "RELIQUARY_ORDER_SANDBOX_KEY_FILE/_KEY_ID are not set"))
        if refusal is not None:
            refused(entry.job_id, refusal)
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
        # The set's prompts (or the catalog source's rows), read and checked.
        await asyncio.to_thread(prompt_job_for, job)
        seen_index = await migrate_ledgers_at_startup(store, job)
        # This lane has no shared sampled-coverage fence, so it cannot enable
        # v3 writers. Finish any committed bodies before constructing readers.
        await recover_pending_records(store, records, job)
        params, miner_states, is_banned, beacon, round_at = build_corpus_audit_wiring(
            entry=entry, job=job, records=records, trust_id=ORDER_TRUST_ID)
        # The drand chain once, off the loop, before the job's first pass: left to that pass,
        # a fetch still in flight has it audit every record it decides (sampling off).
        from reliquary.validator.corpus_validator import warm_drand_chain

        await warm_drand_chain(None)
        w = SimpleNamespace(entry=entry, cap=cap, job=job, tokenizer=tokenizer,
                            audit_executors=order_audit_executors(declared),
                            vocab_size=vocab_size, proof=proof, renderer=renderer,
                            prompt_job_for=prompt_job_for, seen_index=seen_index,
                            is_banned=is_banned, suspect_until=_suspect_reader(miner_states, params),
                            stats=JobStats())
        if job.episode is not None:
            # Published to the session issuer only at the end, once the whole job is wired.
            await wire_episode(w)
        w.audit_params, w.miner_states = params, miner_states
        judged_apart = order_judge is not None and job.episode is None
        grader = getattr(w, "grader", None)
        if judged_apart:
            # Its auditor, settler and book run in the judge process.
            order_judge.attach(w, records=records, store=store)
        else:
            wire_order_judge(
                w, records=records, job_store=store, params=params, miner_states=miner_states, beacon=beacon,
                round_at=round_at, settle_archives=settle_archives, operator_of=operator_of,
                remote=dispatcher.view(job.checkpoint_repo, job.checkpoint_revision, proof,
                                       job.job_id, job.judge if has_draw(job) else None))

            def on_accepted(submission_id: str) -> None:
                w.stats.accepted()
                w.auditor.enqueue(submission_id)
                if grader is not None:
                    grader.enqueue(submission_id)

            w.on_accepted = on_accepted
        if job.episode is not None and is_signed_sandbox(job):
            # Only now: a job still being wired is never visible to the session issuer.
            sandbox.publish(w)
        # Miners wait for the audit: what they would send before an executor of the model has
        # loaded it would only wait unaudited, or be lost on an order that never gets one.
        protocol = job_audit_protocol(job)
        # And until its auditor has read the job back (a restart): what miners send before would pile up unjudged.
        # An episode order also waits for a gateway serving its env: miners refused sessions give up on it.
        gateway_ready = episode_gateway_ready(w) if job.episode is not None and is_signed_sandbox(job) else None

        def why_held() -> str | None:
            if gateway_ready is not None and not gateway_ready():
                return HELD_NO_GATEWAY
            if not (w.auditor.seeded.is_set() and dispatcher.audit_ready(key, protocol)):
                return HELD_AWAITING_AUDIT
            return None

        routes.stop_lease_clock_on(str(job.job_id), HELD_AWAITING_AUDIT)
        if judged_apart:
            # Open once its audit is first ready, as below; held again whenever its judge is not up and seeded.
            opened = False

            def why_held_apart() -> str | None:
                nonlocal opened
                if not w.auditor.seeded.is_set():
                    return HELD_AWAITING_AUDIT
                if not opened and not dispatcher.audit_ready(key, protocol):
                    return HELD_AWAITING_AUDIT
                opened = True
                return None

            routes.hold(str(job.job_id), lambda: why_held_apart() is None, why=why_held_apart, latch=False)
        else:
            routes.hold(str(job.job_id), lambda: why_held() is None, why=why_held)
        if (declared or {}).get("reuse_auditors_only") is True:
            dispatcher.reuse_only_jobs.add(job.job_id)
        if contract_observation is not None and entry.params.get("order_contract"):
            try:
                owners = current_owners() if current_owners else None
                await asyncio.to_thread(contract_observation.open, entry, owners)
            except Exception:
                logger.exception("could not observe opening of %s", job.job_id)
        served[job.job_id] = w
        return w

    async def wire_episode(w) -> None:
        """A signed episode order, as the corpus validator wires one: its signed intake and
        session view (published once the whole job is wired), its grader. A suite one: its
        replaying intake and the grader that replays it again (``suite_grading``)."""
        if w.job.episode.is_suite:
            await wire_suite_episode(w)
            return
        from reliquary.validator.sandbox_wiring import wire_signed_grader, wire_signed_job

        job = w.job
        model_dir = await asyncio.to_thread(model_dir_for, job.checkpoint_repo, job.checkpoint_revision)
        await asyncio.to_thread(functools.partial(
            wire_signed_job, w, services=sandbox, routes=lambda: routes, checkpoint_dir=model_dir,
            tokenizer=w.tokenizer, vocab_size=w.vocab_size, chunk_tokens=w.proof.chunk_tokens,
            publish=False, **(signed_wiring or {})))
        wire_signed_grader(w, judge_records=records, parse_executor=grade_parse_threads)
        # The options digest its sessions are placed under (its first task's), for the gateway hold.
        w.env_options_sha256 = None
        try:
            task = await asyncio.wait_for(w.sandbox_view.resolve_task(0), GATEWAY_DIGEST_TIMEOUT_S)
            w.env_options_sha256 = getattr(task, "env_options_sha256", None)
        except Exception as exc:
            logger.warning("episode order %s: its options digest is unresolved (%s); held for a gateway "
                           "at its package only", job.job_id, type(exc).__name__)

    async def wire_suite_episode(w) -> None:
        """A suite episode order, as the corpus validator wires a suite job: no sandbox, no gateway. One
        replay pool per checkpoint tokenizer, shared by its orders' intakes and graders."""
        from reliquary.validator import suite_intake
        from reliquary.validator.suite_grading import LocalEpisodeGrader

        job = w.job
        key = (job.checkpoint_repo, job.checkpoint_revision)
        if key not in suite_replayers:
            suite_replayers[key] = suite_replayer_for(w.tokenizer)
        w.episode_intake = await asyncio.to_thread(
            suite_intake.build_suite_intake, job, tokenizer=w.tokenizer, vocab_size=w.vocab_size,
            chunk_tokens=w.proof.chunk_tokens, replayer=suite_replayers[key])
        w.grader = LocalEpisodeGrader(job=job, records=records, replayer=w.episode_intake.replayer,
                                      source=w.episode_intake.source)

    def episode_gateway_ready(w) -> Callable[[], bool]:
        """Whether a fresh, active gateway serves the episode's env at its package and options (the
        placement's own check) and has slots for it. Once released, a job stays open (`routes.hold`)."""
        spec = w.job.episode.sandbox
        options = getattr(w, "env_options_sha256", None)

        def ready() -> bool:
            return sandbox.fleet.slots_serving_env(str(spec.env), str(spec.env_package),
                                                   env_options_sha256=options) > 0
        return ready

    async def settle_forever(w, every: float = 60.0) -> None:
        # After the auditor's start: its listing and reads come first after a restart.
        await w.auditor.seeded.wait()
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
        from reliquary.validator.corpus_hot_jobs import REFUSED

        refusal = order_job_refusal(entry, job)
        if refusal is None:
            return None
        refused(entry.job_id, refusal)
        return REFUSED, refusal

    def screen(entry):
        from reliquary.eval.prompt_source import is_order_job_id

        # Not an order job: never even read its manifest.
        if not is_order_job_id(getattr(entry, "job_id", "")):
            return OTHER_MODEL, "not an order job"
        return None

    async def unwired(w) -> None:
        """A drained job releases everything it held: its wiring, its pool view
        and listeners, its prompts, and its model's tokenizer once unused."""
        from reliquary.eval.prompt_source import (
            forget_eval_prompts, is_eval_source, parse_eval_source,
        )

        served.pop(w.job.job_id, None)
        if sandbox is not None and w.job.episode is not None:
            sandbox.forget(str(w.job.job_id))
        w.auditor._remote.close()
        if is_eval_source(w.job.prompt_source):
            forget_eval_prompts(parse_eval_source(w.job.prompt_source))
        key = (w.job.checkpoint_repo, w.job.checkpoint_revision)
        if not any((o.job.checkpoint_repo, o.job.checkpoint_revision) == key
                   for o in served.values()):
            tokenizers.pop(key, None)
        if contract_observation is not None and w.entry.params.get("order_contract"):
            await asyncio.to_thread(contract_observation.close, w.job.job_id, int(clock()),
                                    job_set.finished.get(w.job.job_id, {}))

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
    if contract_observation is not None:
        from reliquary.validator.standing_offers import standing_offers_router

        app.include_router(standing_offers_router(contract_observation.book, contract_observation.markets,
                                                 registration=registration, operator_of=operator_of, clock=clock))
    if sandbox is not None:
        routes.sandbox_hotkey = getattr(sandbox, "validator_hotkey", None)
        # Read by an open before any signed job is wired: a job about to be is `job_not_ready`, not unknown.
        sandbox.routes = lambda: routes
        # /corpus/jobs/{job_id}/sandbox/sessions: the episode orders' sessions, under this control's key.
        app.include_router(sandbox.router)
        app.state.corpus_sandbox = sandbox

    @app.get("/corpus/jobs/{job_id}/eval-prompts")
    async def eval_prompts(job_id: str) -> Response:
        from reliquary.eval.prompt_source import is_eval_source

        w = served.get(job_id)
        if w is None or job_id not in routes.routers:
            raise routes.unserved(job_id)
        if not is_eval_source(w.job.prompt_source):
            # A generation order's rows are the catalog source's, as every miner builds them.
            raise HTTPException(status_code=404, detail="not_an_eval_job")
        body = await asyncio.to_thread(job_prompt_lines, parse_eval_source(w.job.prompt_source))
        return Response(content=body, media_type="application/x-ndjson")

    @app.get("/corpus/jobs/{job_id}/contract")
    async def eval_job_contract(job_id: str) -> dict:
        w = served.get(job_id)
        if w is None or job_id not in routes.routers:
            raise routes.unserved(job_id)
        return w.entry.contract

    @app.get("/corpus/jobs/{job_id}/status")
    async def eval_job_status(job_id: str) -> dict:
        from reliquary.eval.prompt_source import is_order_job_id

        # Only an order job's status is ever looked up (stored ones included).
        try:
            status = await job_set.status(job_id) if is_order_job_id(job_id) else None
        except Exception as exc:
            from reliquary.validator.order_judge import JudgeUnavailable

            if isinstance(exc, JudgeUnavailable):
                raise HTTPException(status_code=503, detail="order_judge_unavailable") from exc
            raise
        if status is None:
            raise routes.unserved(job_id)
        return status

    add_public_corpus_routes(app, routes)
    app.include_router(build_eval_executor_router(dispatcher=dispatcher, directory=directory,
                                                  qualifications=qualifications))
    job_set = CorpusJobSet(
        routes=routes, router_for=router_for, wire=wire,
        jobs_of=lambda w: ([] if getattr(w, "judge_link", None) is not None else [
            guarded(w, w.auditor.run()), guarded(w, settle_forever(w)),
            *([guarded(w, w.grader.run())] if getattr(w, "grader", None) is not None else [])]),
        read_entries=(order_judge.watching(read_entries) if order_judge is not None and read_entries
                      else read_entries),
        read_job=read_job, admit=admit, screen=screen,
        on_unwired=unwired, refresh_every_seconds=refresh_every_seconds,
        # A drained order's status outlives this process; a retired one is
        # finished after a restart.
        finals=records if hasattr(records, "write_final_status") else None,
        wire_retired=True,
        drained=lambda w: job_drained(auditor=w.auditor, records=records,
                                      job_id=w.job.job_id),
        clock=clock)
    drain_seconds = order_audit_drain_seconds()
    max_executors = order_max_audit_executors()
    sizing = ExecutorSizing(clock=clock)
    sandbox_models: dict[str, tuple[str, str]] = {}   # signed episode job -> its model, while it may forecast

    async def control_status() -> dict:
        """The dispatcher's status, with each served job's completeness, and per model the executors
        its orders' expected audits want (`executors_wanted`)."""
        document = dispatcher.status()
        document["reuse_auditors_only"] = True
        document["at"] = time.time()
        # Per model served, its orders' expected audit items; None while one of them is unknown.
        backlog: dict[tuple[str, str], float | None] = {}
        # Per model, the most executors its orders allow (the fastest order's choice), within the control's.
        allowed: dict[tuple[str, str], int] = {}
        released: list[str] = []
        statuses: dict[str, dict] = {}
        for w in list(served.values()):
            if w.job.job_id in dispatcher.reuse_only_jobs:
                continue
            model = (w.job.checkpoint_repo, w.job.checkpoint_revision)
            backlog.setdefault(model, 0.0)
            allowed[model] = max(allowed.get(model, 1), getattr(w, "audit_executors", 1))
        for job_id in list(served):
            try:
                status = await job_set.status(job_id)
            except Exception:
                logger.warning("eval status of %s unavailable", job_id, exc_info=True)
                if job_id not in dispatcher.reuse_only_jobs:
                    job = served[job_id].job
                    backlog[(job.checkpoint_repo, job.checkpoint_revision)] = None
                continue
            if status is None:
                continue
            statuses[job_id] = status
            entry = document["jobs"].setdefault(
                job_id, {"needs_attention": False, "parked_batches": 0, "reasons": []})
            entry.update({key: status.get(key) for key in (
                "state", "prompts_total", "prompts_full", "prompts_complete",
                "prompts_exhausted", "complete")})
            # Kept from miners, and why (``no_gateway``, ``awaiting_audit``); absent once open.
            held = routes.held_reason(job_id)
            if held is not None:
                entry["held"] = held
            else:
                entry.pop("held", None)
            # What sizes its cap (``admin.cap_sizing``): its model, cards, and what it paid and has accrued.
            settler = getattr(served.get(job_id), "settler", None)
            # Judged apart and its snapshot stale: no sizing, rather than a cap resized on zeros.
            if settler is not None and getattr(settler, "sizing_known", True):
                job = served[job_id].job
                entry["sizing"] = {"model": f"{job.checkpoint_repo}@{job.checkpoint_revision}",
                                   "audit_executors": getattr(served[job_id], "audit_executors", 1),
                                   "recent_paid": list(settler.recent_paid),
                                   "open_value": settler.open_value()}
            # What the fleet needs to keep an audit executor for it: none once nothing left may need one.
            auditor = getattr(served.get(job_id), "auditor", None)
            if auditor is not None:
                pending = auditor.audits_pending()
                entry["audits_pending"] = pending
                # Why the records judged since this process started were audited (`CorpusAuditor._audit_reason`).
                entry["audit_reasons"] = dict(getattr(auditor, "audit_reasons", {}))
                # Outcomes scored ahead and thrown away: card work no verdict used (`CorpusAuditor.ahead_wasted`).
                entry["ahead_wasted"] = int(getattr(auditor, "ahead_wasted", 0))
                entry["audits_idle_since"] = auditor.audits_idle_since
                if job_id in dispatcher.reuse_only_jobs:
                    continue
                job = served[job_id].job
                model = (job.checkpoint_repo, job.checkpoint_revision)
                expected = auditor.expected_audits(pending)
                if expected is None or backlog.get(model) is None:
                    backlog[model] = None
                else:
                    backlog[model] += expected
        for model, expected in backlog.items():
            key = f"{model[0]}@{model[1]}"
            live, idle = dispatcher.executor_ids(model)
            rate = dispatcher.items_per_second(model)
            tokens = dispatcher.tokens_per_second(model)
            # Unknown backlog (a job seeding, a status unreadable): what is wanted stays as it was.
            target = None if expected is None else executors_for_backlog(
                expected, rate, drain_seconds=drain_seconds,
                max_executors=min(max_executors, allowed.get(model, 1)))
            wanted = sizing.update(key, target)
            entry = document["models"].setdefault(key, {"executors_needed": 0, "live_executors": len(live)})
            entry["reusable_draw_auditor"] = dispatcher.reusable_auditor(model, AUDIT_PROTOCOL_V3)
            # Beyond what is wanted (at least the scorers a batch needs), the newest executors lease nothing more;
            # the fleet releases one only once it holds no lease (`releasing` and `idle_executors`).
            releasing = dispatcher.release_beyond(
                model, None if wanted is None else max(wanted, dispatcher.scorers))
            released.extend(releasing)
            entry.update({"executors_allowed": min(max_executors, allowed.get(model, 1)),
                          "expected_audits": None if expected is None else round(expected),
                          "items_per_second": rate, "executors_wanted": wanted,
                          "audit_tokens_per_second": None if tokens is None else tokens[0],
                          "audit_prompt_tokens_per_second": None if tokens is None else tokens[1],
                          "executors": live, "idle_executors": idle, "releasing": releasing})
            if wanted:
                entry["executors_needed"] = max(entry.get("executors_needed", 0), wanted - len(live))
        sizing.forget({f"{m[0]}@{m[1]}" for m in backlog})
        dispatcher.set_releasing(released)
        sandbox_jobs(document, statuses)
        return document

    def sandbox_jobs(document: dict, statuses: dict[str, dict]) -> None:
        """Per served signed episode order, what its sandbox gateways are sized from (`order_sandbox_status`)."""
        from reliquary.validator.order_sandbox_status import (
            FLEET_FULL_WINDOW_S, activity_by_hotkey, book_sessions, held_by_hotkey, job_demand, sandbox_status,
        )

        episodes = [w for w in list(served.values()) if w.job.episode is not None and is_signed_sandbox(w.job)]
        if sandbox is None or not episodes:
            return
        now = int(clock())
        try:
            sessions = book_sessions(sandbox.book, now)
            holders = held_by_hotkey(sandbox.book, now)
            shown = activity_by_hotkey(sandbox.book, now)
        except Exception:
            logger.warning("sandbox status: the session book is unreadable", exc_info=True)
            return
        quota = getattr(sandbox, "quota", None)
        declarations: dict = {}
        if quota is not None and hasattr(quota, "declarations"):
            try:
                declarations = quota.declarations(now)
            except Exception:
                logger.warning("sandbox status: the declared wants are unreadable", exc_info=True)
        for w in episodes:
            sandbox_models[str(w.job.job_id)] = (w.job.checkpoint_repo, w.job.checkpoint_revision)
        for job_id in [j for j in sandbox_models if j not in declarations and j not in served]:
            del sandbox_models[job_id]                    # nothing left to forecast from
        serving = {str(w.job.job_id) for w in episodes}

        def low_yield(hotkey: str) -> bool:
            try:
                return quota is not None and quota.lane(hotkey) == "low_yield"
            except Exception:
                return False

        for w in episodes:
            job_id, job = str(w.job.job_id), w.job
            model = (job.checkpoint_repo, job.checkpoint_revision)
            held, finished = sessions.get(job_id, (0, []))
            tokens = getattr(w, "episode_tokens", None)
            try:
                entry = document["jobs"].setdefault(
                    job_id, {"needs_attention": False, "parked_batches": 0, "reasons": []})
                sizing_of = document["models"].get(f"{model[0]}@{model[1]}", {})
                share = getattr(getattr(w, "auditor", None), "audit_share", None)
                entry["sandbox"] = sandbox_status(
                    job=job, status=statuses.get(job_id),
                    reopens=bool(getattr(w.auditor, "reopen_slots", False)), held=held, finished=finished,
                    tokens_per_episode=None if tokens is None else tokens.mean(),
                    tokens_per_second=dispatcher.tokens_per_second(model),
                    executors_wanted=sizing_of.get("executors_wanted"),
                    fleet_refusals=0 if quota is None else quota.fleet_refusals(job_id, now, FLEET_FULL_WINDOW_S),
                    demand=job_demand(job_id, now=now, declarations=declarations, held=holders, activity=shown,
                                      model_of=sandbox_models, served=serving, excluded=low_yield),
                    executors_allowed=sizing_of.get("executors_allowed"),
                    audit_share=share() if callable(share) else None)
            except Exception:
                logger.warning("sandbox status of %s unavailable", job_id, exc_info=True)

    app.state.control_status = control_status
    app.state.corpus_jobs = job_set
    app.state.eval_served = served
    app.state.eval_tokenizers = tokenizers
    app.state.dispatcher = dispatcher
    return app, job_set


async def run_eval_control(*, netuid: int, http_host: str, http_port: int,
                           registration_gate: bool = True,
                           refresh_every_seconds: float | None = None) -> None:
    """Serve every active order task (``order-eval-`` and ``order-gen-``) of
    the registry, hot."""
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
        directory=directory, scorers=order_audit_scorers(),
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

    sandbox, sandbox_down = await order_sandbox_services(
        registration=registered.reason if registered is not None else None)
    job_set = None
    archives = OrderArchives(served=lambda: job_set.task_ids() if job_set else ())
    # RELIQUARY_ORDER_SPLIT_JUDGE=1: non-episode orders are judged and paid in a child process.
    from reliquary.validator.order_judge import OrderJudgeFront, order_judge_enabled

    order_judge = (OrderJudgeFront.from_env(
        dispatcher=dispatcher, coldkeys=registered.coldkeys if registered is not None else None)
        if order_judge_enabled() else None)
    from reliquary.corpus.contract_observation import ContractObservation

    try:
        contract_observation = ContractObservation.from_env()
    except Exception:
        logger.exception("local contract observation is unavailable")
        contract_observation = None
    app, job_set = build_eval_control(
        # Every order's auditors and the miners' intake share these connections (`ORDER_POOL_CONNECTIONS`).
        store=BucketJobStore(), records=BucketRecordStore(max_pool_connections=ORDER_POOL_CONNECTIONS),
        dispatcher=dispatcher,
        directory=directory, verify_signature=verify_corpus_signature,
        verify_skip_signature=verify_corpus_skip_signature, qualifications=qualifications,
        registration=registered.reason if registered is not None else None,
        operator_of=registered.coldkey if registered is not None else None,
        settle_archives=archives, read_entries=read_entries,
        refresh_every_seconds=refresh_every_seconds or 60.0, sandbox=sandbox, sandbox_down=sandbox_down,
        order_judge=order_judge, contract_observation=contract_observation,
        current_owners=registered.fresh_coldkeys if registered is not None else None)

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
    if sandbox is not None:
        background.extend(sandbox.background())
    if order_judge is not None:
        background.append(order_judge.run())
    server = uvicorn.Server(uvicorn.Config(app, host=http_host, port=http_port, log_level="info"))
    try:
        await asyncio.gather(server.serve(), *background)
    finally:
        if order_judge is not None:
            order_judge.stop()
        if sandbox is not None:
            await sandbox.stop()


# The order control's own sandbox key settings: never the corpus validator's, so neither verifies nor restores
# the other's sessions. The other RELIQUARY_SANDBOX_* settings (policy, directory, machines) are read as there.
ORDER_SANDBOX_KEY_SETTINGS = {"key_file": "RELIQUARY_ORDER_SANDBOX_KEY_FILE",
                              "key_id": "RELIQUARY_ORDER_SANDBOX_KEY_ID",
                              "retired_keys": "RELIQUARY_ORDER_SANDBOX_RETIRED_KEYS",
                              "hotkey": "RELIQUARY_ORDER_SANDBOX_HOTKEY"}


async def order_sandbox_services(*, registration=None, environ=None, build=None,
                                 session_store=None) -> tuple[Any, str | None]:
    """The signed-sandbox services episode orders are served with, started: ``(services, None)``; ``(None,
    None)`` when this control has no sandbox key settings; ``(None, why)`` when they fail (episode orders are
    then refused with ``why``, every other order is served). Its sessions live under their own prefix."""
    import os

    from reliquary.validator.corpus_validator import start_sandbox_services

    from reliquary.protocol.sandbox_session import JOB_SCOPED_PREFIX

    environ = os.environ if environ is None else environ
    try:
        from reliquary.validator.sandbox_wiring import SandboxValidatorConfig, build_sandbox_services

        config = SandboxValidatorConfig.from_env(environ, key_settings=ORDER_SANDBOX_KEY_SETTINGS)
        if config is None:
            return None, None
        if not config.validator_hotkey:
            # No wallet here: the ss58 address miners sign session requests to.
            raise ValueError(f"set {ORDER_SANDBOX_KEY_SETTINGS['hotkey']}")
        if session_store is None:
            from reliquary.infrastructure.sandbox_store import ORDER_SESSION_PREFIX, R2SessionStore

            session_store = R2SessionStore(prefix=ORDER_SESSION_PREFIX)
        services = (build or build_sandbox_services)(
            config, validator_hotkey=config.validator_hotkey, registration=registration,
            session_store=session_store, route_prefix=JOB_SCOPED_PREFIX)
    except Exception as exc:  # noqa: BLE001 - isolated: single-turn orders still serve
        # The key loader's errors name the path and its mode, never the key.
        logger.error("ALERT: order sandbox services not built: %r; episode orders are refused", exc)
        return None, f"its sandbox services were not built ({type(exc).__name__}: {str(exc)[:120]})"
    if not await start_sandbox_services(services, []):
        return None, "its sandbox services failed to start"
    return services, None


# What the order control imports beyond the base install: the drand draw of
# the partial audit, the model files read on CPU.
ORDER_CONTROL_IMPORTS = ("bittensor_drand", "huggingface_hub", "transformers")


def _import(name: str) -> None:
    import importlib

    importlib.import_module(name)


def _verify_environment(env: str) -> None:
    """The env's package installed, its artifact verified, and its digest the
    catalog's (the one orders are qualified against)."""
    from reliquary.environment.agentic.external import verify_external_artifact
    from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG
    from reliquary.validator import corpus_service

    spec = corpus_service.ENVIRONMENT_SPECS[env]
    verify_external_artifact(spec)
    pinned = getattr(ENVIRONMENT_CATALOG.get(env), "environment_manifest_sha256", None)
    if spec.environment_manifest_sha256 != pinned:
        raise ValueError(f"{env}: installed {spec.environment_manifest_sha256} is not the "
                         f"catalog's {pinned}")


def _verify_rl_environment(env: str) -> None:
    """An env a training order may name, loadable by its RL controller: an
    external one's package installed at the version and artifact the registry
    pins; a built-in one is checked against its manifest at import."""
    from reliquary.environment.agentic.external import verify_external_artifact
    from reliquary.environment.registry import ENVIRONMENT_SPECS

    spec = ENVIRONMENT_SPECS[env]
    if spec.external_distribution is not None:
        verify_external_artifact(spec)


def order_control_runtime_check() -> dict:
    """Everything the order control needs at runtime, checked without a GPU
    or a network: ``{"imports", "environments", "rl_environments", "ok"}``,
    each entry "ok" or why not (`reliquary corpus order-control-check`, run
    at image build). ``rl_environments``: every env a training order may
    name (``admin.training.trainable_environments``), for the RL controllers
    `reliquary rl supervise` runs from the same image."""
    from reliquary.admin.training import trainable_environments
    from reliquary.eval.qualification import ORDER_ENVIRONMENTS

    report: dict[str, Any] = {"imports": {}, "environments": {}, "rl_environments": {}}
    for name in ORDER_CONTROL_IMPORTS:
        try:
            _import(name)
            report["imports"][name] = "ok"
        except Exception as exc:
            report["imports"][name] = f"{type(exc).__name__}: {exc}"
    for env in sorted(ORDER_ENVIRONMENTS):
        try:
            _verify_environment(env)
            report["environments"][env] = "ok"
        except Exception as exc:
            report["environments"][env] = f"{type(exc).__name__}: {exc}"
    for env in trainable_environments():
        try:
            _verify_rl_environment(env)
            report["rl_environments"][env] = "ok"
        except Exception as exc:
            report["rl_environments"][env] = f"{type(exc).__name__}: {exc}"
    report["ok"] = all(v == "ok" for part in ("imports", "environments", "rl_environments")
                       for v in report[part].values())
    return report


build_order_control = build_eval_control
run_order_control = run_eval_control


__all__ = [
    "ORDER_SANDBOX_KEY_SETTINGS",
    "BatchParked",
    "ItemsUnreadable",
    "OrderArchives",
    "build_order_control",
    "wire_order_judge",
    "order_control_runtime_check",
    "qualification_refusal",
    "pinned_order_refusal",
    "order_audit_scorers",
    "order_job_refusal",
    "order_sandbox_services",
    "run_order_control",
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
