"""The control's half of grading (spec §5 N5): grade executor tokens, leases,
and agreement.

The control never runs a container, so the audit dispatcher's local recheck
becomes agreement between executors: a result drawn for a recheck (5%, drawn
from the OS) and every replay that would fail the miner is repeated by a
second, distinct executor; a disagreement goes to a third, the majority
stands, and each executor that disagreed with it is quarantined. An executor
alone can never fail a miner. Timeouts, executor errors and expired leases
re-lease the item elsewhere and never judge anyone: two timeouts (an expired
lease counts as one) resolve as ``timeout``, three errors as ``error``.
A box the trajectory itself lost or ran past its deadline (``box_lost``,
``box_timeout``, ruling P23) is a vote, not an error: it needs a second
distinct provider, and two agreeing resolve the item ``unjudgeable``.

Executors count by provider (ruling P17): each grade executor registers its
``provider_id``, an item is never leased to a provider that already voted on
it, and agreement counts distinct providers, so one operator's boxes are one
vote. Every wait for a next distinct executor is bounded (ruling P16): an item
that holds a vote and finds no further executor within ``dispute_seconds``
resolves ``disputed``, which sanctions nobody and certifies nothing. That
clock runs only while no live eligible executor exists (F3), and an item
holding a vote goes back to the front of the queue, so a grading backlog
never turns a failing replay into a dispute.

An executor that failed an attempt on an item (its lease expired, it
answered ``error``/``timeout``, or its result did not fit the lease) is kept
off that item so the next attempt runs elsewhere. When no other live
executor could take it (one executor, or one provider), that exclusion lapses
after ``GRADE_RETRY_EXCLUDED_SECONDS``: the item is leased to it again, so an
item with no vote can never wait forever (no dispute clock runs without a
vote). The attempt counters are unchanged (two timeouts, three errors), and
an executor that voted on an item is never leased it again.

A claim whose reply never reached the executor (the connection dropped)
leaves a lease nobody works, until it expires (3.3 h for a replay). A grade
executor's heartbeat reports the leases it holds: any lease of it older than
``GRADE_LEASE_REPORT_GRACE_SECONDS`` and missing from the report is taken
back as an executor error (no vote, the item requeued, no strike). A
heartbeat without the report (an older executor) takes nothing back.

A recheck is only drawn while the executors registered on this control's env
(not quarantined) span two providers or more: with one provider the second
vote could never come, and every drawn item would wait out the dispute clock
and resolve ``disputed`` (paid, not certified, not exported). A failing replay
still needs two providers whatever the draw: with one it resolves
``uncertified`` after the dispute wait, as before.

An item no lease can carry (actions or observations beyond the
``GradeLease`` bounds) is never leased: every executor would refuse the lease
and it would cycle forever. It resolves at once as ``ungradeable``, the
validator's own limit, never evidence against the miner. Since ruling P23 the
intake (and the miner's precheck) refuse such a trajectory before it takes a
slot (``grade_item_bounds_refusal``), so this outcome is only defensive.

Executor liveness, lease caps, expiry strikes, quarantine and heartbeat writes
are the audit dispatcher's (``ExecutorLeases``).
"""

from __future__ import annotations

import asyncio
import collections
import logging
import math
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import ValidationError

from reliquary.corpus.replay_compare import ReplayReport, within_tolerance
from reliquary.validator.corpus_audit_protocol import HeartbeatRequest
from reliquary.validator.corpus_audit_remote import (
    EXECUTOR_LIVE_SECONDS,
    LEASE_EXPIRY_STRIKES,
    ExecutorDirectory,
    ExecutorLeases,
    LeaseRefused,
    _bounded_env,
    bearer_authenticator,
)
from reliquary.validator.corpus_grade_protocol import (
    GRADE_PROTOCOL,
    GradeClaimRequest,
    GradeItem,
    GradeItemResult,
    GradeResult,
)
from reliquary.validator.lease_executor import ERROR_BACKOFF_SECONDS, REQUEST_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

GRADE_PREFIX = "/corpus/internal/grade"
# A lease's life per mode. The executor does not renew a lease while it works,
# so each covers the work's own bound plus the box's start and the corpus
# load: a grade runs under its 1800 s scoring timeout; a replay under its
# setup deadline (SWE-smith: 900 + 600 s) then its trajectory budget
# (2 x (3600 + 900) = 9000 s, ruling P25), 10 500 s, plus a margin.
GRADE_LEASE_SECONDS = {
    "grade": _bounded_env("RELIQUARY_CORPUS_GRADE_LEASE_SECONDS", 2400.0, 2100.0, 7200.0),
    "replay": _bounded_env("RELIQUARY_CORPUS_REPLAY_LEASE_SECONDS", 12000.0, 11000.0, 28800.0),
}
# What a replay lease must leave beyond the replay's own deadlines: the box's
# start, the corpus load and the result's post (ruling P26).
REPLAY_LEASE_MARGIN_SECONDS = 600.0


def replay_lease_refusal(task, lease_seconds: float | None = None) -> str | None:
    """Why the replay lease is too short for ``task``'s replay (its setup
    deadline + trajectory budget, ``agentic_replay.replay_deadlines``, plus
    a margin), or None. Too short, every long honest replay expires twice
    and resolves unjudged."""
    from reliquary.validator.agentic_replay import replay_deadlines

    lease = GRADE_LEASE_SECONDS["replay"] if lease_seconds is None else float(lease_seconds)
    work = sum(replay_deadlines(task))
    if lease >= work + REPLAY_LEASE_MARGIN_SECONDS:
        return None
    return (f"the replay lease ({lease:.0f} s, RELIQUARY_CORPUS_REPLAY_LEASE_SECONDS) is shorter "
            f"than the task's replay work {work:.0f} s + margin {REPLAY_LEASE_MARGIN_SECONDS:.0f} s; "
            f"raise it to at least {work + REPLAY_LEASE_MARGIN_SECONDS:.0f}")


def _job_task(job, index: int):
    from reliquary.environment.agentic_swe import load_swe_source
    from reliquary.validator.agentic_replay import swesmith_task

    source = load_swe_source(job.episode.env.num_images)
    return swesmith_task(source.instance_id(index))


def check_replay_lease(job, *, task_for=None, lease_seconds: float | None = None) -> None:
    """At control start (blocking: builds one of the job's tasks): raise
    ``RuntimeError`` when the replay lease cannot cover the job's replays.
    Every task of a task set shares its phase timeouts, so one is checked."""
    task = (task_for or _job_task)(job, 0)
    refusal = replay_lease_refusal(task, lease_seconds)
    if refusal:
        raise RuntimeError(f"job {job.job_id!r}: {refusal}")


GRADE_RECHECK_FRACTION = 0.05
MAX_RESULTS_PER_ITEM = 3
MAX_TIMEOUTS = 2
MAX_ERRORS = 3
# Leases one executor holds at once. A grade executor runs that many boxes at
# most whatever its own --concurrency (production 2026-10-06: grade-01 set to 12
# held 8, a third of its CPU idle while the SWE backlog grew): set it to the
# executors' concurrency.
MAX_LEASES_PER_EXECUTOR = int(_bounded_env("RELIQUARY_CORPUS_GRADE_MAX_LEASES_PER_EXECUTOR",
                                           8.0, 1.0, 64.0))
# How long an item holding a vote waits for a next distinct-provider executor.
GRADE_DISPUTE_SECONDS = _bounded_env("RELIQUARY_CORPUS_GRADE_DISPUTE_SECONDS", 1800.0, 60.0, 86400.0)
# Ruling P26: an executor serves the dispute clock only while it claims (a
# claim request this recent) or holds a lease; heartbeats alone do not. The
# default outlasts two of an executor's slowest claim cycles (a request that
# times out, then the error backoff; ruling P27): 260 s.
GRADE_CLAIM_LIVE_SECONDS = _bounded_env("RELIQUARY_CORPUS_GRADE_CLAIM_LIVE_SECONDS",
                                        2 * (REQUEST_TIMEOUT_SECONDS + ERROR_BACKOFF_SECONDS),
                                        30.0, 3600.0)
# How long an executor that failed an attempt on an item stays off it when no
# other executor could take it (production 2026-10-04: one executor, a replay
# lease expired, the item waited forever and no period settled).
GRADE_RETRY_EXCLUDED_BOUNDS = (60.0, 86400.0)
GRADE_RETRY_EXCLUDED_SECONDS = _bounded_env("RELIQUARY_CORPUS_GRADE_RETRY_EXCLUDED_SECONDS",
                                            600.0, *GRADE_RETRY_EXCLUDED_BOUNDS)
# How old an unreported lease must be before a heartbeat that omits it takes
# it back: three of the executor's 20 s heartbeat intervals.
GRADE_LEASE_REPORT_GRACE_SECONDS = _bounded_env("RELIQUARY_CORPUS_GRADE_LEASE_REPORT_GRACE_SECONDS",
                                                60.0, 30.0, 3600.0)
UNGRADEABLE = "ungradeable"
DISPUTED = "disputed"
# Ruling P23: the box failed or the deadline passed once the trajectory's
# actions (or its applied patch) ran. Each such result is a vote like an
# "ok" one; two distinct providers agreeing resolve the item UNJUDGEABLE.
TRAJECTORY_STATUSES = frozenset({"box_lost", "box_timeout"})
UNJUDGEABLE = "unjudgeable"
# Ruling P27: a replay whose votes, without any agreement, include no
# certifying one (failed and/or box failures): voided unpaid, no sanction.
UNCERTIFIED = "uncertified"

# The facts an "ok" result must carry for its mode.
_MODE_FACTS = {"grade": ("diff_applied", "tests_passed"), "replay": ("replay_diff_equal",)}


@dataclass(frozen=True)
class GradeDecision:
    # "ok", "error", "timeout", "ungradeable", "disputed", "unjudgeable" or
    # "uncertified"; only "ok" judges the miner, "unjudgeable" (two providers)
    # and "uncertified" (a replay no vote certifies) void it unpaid.
    status: str
    result: dict | None                 # the agreed result, when "ok"
    graded_by: tuple[str, ...]
    # The distinct providers of the agreeing executors, when "ok": a sanction
    # needs two (ruling P17), which the grader checks again on its side.
    providers: tuple[str, ...] = ()


def replay_certified(result: dict) -> bool:
    return within_tolerance(ReplayReport(
        compared=int(result.get("observations_compared") or 0),
        mismatched=list(result.get("observations_mismatched") or []),
        diff_equal=bool(result.get("replay_diff_equal"))))


def decision_key(mode: str, result: dict) -> tuple:
    """What two executors must agree on: the facts the control decides from.
    A box lost and a box timed out are one outcome: the trajectory's."""
    if result.get("status") in TRAJECTORY_STATUSES:
        return (UNJUDGEABLE,)
    if mode == "grade":
        return (result.get("diff_applied"), result.get("tests_passed"))
    # Whether it certifies, only (M2): two executors that both fail an episode
    # agree, even if one saw another diff; the diff stays in the document.
    return (replay_certified(result),)


@dataclass
class _Work:
    id: int
    item: dict
    future: asyncio.Future
    queued_at: float
    results: dict[str, dict] = field(default_factory=dict)
    providers: dict[str, str] = field(default_factory=dict)   # voter -> its provider
    excluded: set[str] = field(default_factory=set)      # voted on it: never again
    # Failed an attempt on it (expiry, error, misfit) -> when it last did:
    # off it while another executor could take it, else after the retry wait.
    failed_at: dict[str, float] = field(default_factory=dict)
    drawn: bool | None = None           # recheck draw, made at the first result
    timeouts: int = 0
    errors: int = 0
    # The dispute clock (F3): seconds this voted item spent queued while no
    # live eligible executor existed, as of the sweep at ``swept_at``.
    unserved: float = 0.0
    swept_at: float | None = None

    @property
    def mode(self) -> str:
        return self.item["mode"]

    @property
    def agree_needed(self) -> int:
        unjudgeable = any(r.get("status") in TRAJECTORY_STATUSES for r in self.results.values())
        failing = self.mode == "replay" and any(
            not replay_certified(r) for r in self.results.values())
        return 2 if self.drawn or failing or unjudgeable else 1


@dataclass
class _Lease:
    lease_id: str
    work: _Work
    executor_id: str
    expires_at: float
    leased_at: float = 0.0


class RemoteGradeDispatcher(ExecutorLeases):
    kind = "grade executor"

    def __init__(self, *, directory: ExecutorDirectory, env_package: str, env_version: str,
                 quarantine: Callable[[str, str], Awaitable[Any]] | None = None,
                 record_heartbeat: Callable[[str, float, dict], Awaitable[Any]] | None = None,
                 clock: Callable[[], float] = time.time, rng=None,
                 recheck_fraction: float = GRADE_RECHECK_FRACTION,
                 live_seconds: float = EXECUTOR_LIVE_SECONDS,
                 max_leases_per_executor: int = MAX_LEASES_PER_EXECUTOR,
                 expiry_strikes: int = LEASE_EXPIRY_STRIKES,
                 lease_seconds: dict[str, float] | None = None,
                 dispute_seconds: float = GRADE_DISPUTE_SECONDS,
                 claim_live_seconds: float = GRADE_CLAIM_LIVE_SECONDS,
                 retry_excluded_seconds: float = GRADE_RETRY_EXCLUDED_SECONDS,
                 report_grace_seconds: float = GRADE_LEASE_REPORT_GRACE_SECONDS) -> None:
        super().__init__(directory=directory, quarantine=quarantine,
                         record_heartbeat=record_heartbeat, clock=clock,
                         live_seconds=live_seconds,
                         max_leases_per_executor=max_leases_per_executor,
                         expiry_strikes=expiry_strikes)
        self._env = {"package": env_package, "version": env_version}
        # Each result is drawn on its own, from the OS: an executor cannot predict it.
        self._rng = rng or secrets.SystemRandom()
        self._fraction = recheck_fraction
        self._lease_seconds = {**GRADE_LEASE_SECONDS, **(lease_seconds or {})}
        self._dispute_seconds = float(dispute_seconds)
        self._queue: collections.deque[_Work] = collections.deque()
        self._holders: list[Callable[[str], Any]] = []
        # Executors whose last claim was refused 409 wrong_env: live by their
        # heartbeats, but they never take a lease, so never eligible (N2).
        self._wrong_env: set[str] = set()
        # Box failures (box_lost/box_timeout) each executor reported: never a
        # penalty (ruling P26), shown on the status route for the operator.
        self.box_failure_votes: collections.Counter[str] = collections.Counter()
        # When each executor last asked for a lease (ruling P26).
        self._claimed_at: dict[str, float] = {}
        self._claim_live = float(claim_live_seconds)
        self._retry_excluded = float(retry_excluded_seconds)
        self._report_grace = float(report_grace_seconds)
        # Whether recheck draws were last possible (two providers), to log changes once.
        self._draws_possible: bool | None = None

    def hold_on_quarantine(self, holder: Callable[[str], Any]) -> None:
        """``holder(executor_id)`` runs synchronously the moment an executor is
        quarantined, before any registry write or listener: what it decided
        alone can be held from payment at once."""
        self._holders.append(holder)

    def load_quarantined(self) -> list[str]:
        """Quarantines the registry already holds (a restart forgets its own):
        refused and held at once, with no new registry write. The listeners
        hear of them like any quarantine."""
        loaded = []
        for executor_id in self._directory.quarantined_ids():
            if self._mark_quarantined(executor_id, "quarantined in the registry"):
                self._unwritten_quarantines.pop(executor_id, None)
                loaded.append(executor_id)
                for listener in self._listeners:
                    self._spawn(self._notify(listener, executor_id))
        return loaded

    async def sweep(self) -> None:
        # A quarantine written by another control (or before a restart) counts here too.
        self.load_quarantined()
        await self._sweep()

    @property
    def env_pin(self) -> tuple[str, str]:
        """The env package and commit every lease of this dispatcher runs."""
        return self._env["package"], self._env["version"]

    # -- the grader's side ------------------------------------------------------

    async def decide(self, item: dict) -> GradeDecision:
        """The agreed facts for one grade or replay item."""
        try:
            item = GradeItem.model_validate(item).model_dump()
        except ValidationError as exc:
            # Every executor would refuse this lease: never lease it.
            self.stats[UNGRADEABLE] += 1
            logger.error("grade item %s (%s) exceeds the lease bounds; ungradeable: %s",
                         str(item.get("submission_id"))[:12], item.get("mode"),
                         str(exc).splitlines()[0][:300])
            return GradeDecision(UNGRADEABLE, None, ())
        loop = asyncio.get_running_loop()
        work = _Work(id=next(self._ids), item=item, future=loop.create_future(),
                     queued_at=self._clock())
        self._queue.append(work)
        return await work.future

    # -- the executor's side ----------------------------------------------------

    def heartbeat(self, executor_id: str, detail: dict | None = None,
                  held_leases: list[str] | None = None) -> None:
        super().heartbeat(executor_id, detail)
        if held_leases is None:
            return                                # an older executor: nothing reported
        held, now = set(held_leases), self._clock()
        for lease_id, lease in list(self._leases.items()):
            if (lease.executor_id != executor_id or lease_id in held
                    or now - lease.leased_at < self._report_grace):
                continue
            if lease.expires_at <= now:
                # Expired: the sweep counts it (a timeout and a strike), so
                # leaving a lease out of a report never dodges its expiry.
                continue
            # Leased, but the executor does not hold it: the claim's reply
            # was lost (or its result could not be delivered). Its fault,
            # never the miner's, and no strike: the network's as often.
            del self._leases[lease_id]
            self.stats["leases_unreported"] += 1
            logger.warning("grade lease %s (item %d, %s) of %s is not held by it %.0f s after "
                           "the claim; taken back", lease_id[:8], lease.work.id,
                           lease.work.mode, executor_id, now - lease.leased_at)
            lease.work.failed_at[executor_id] = now
            self._failed_attempt(lease.work, "errors")

    def refused_env(self, executor_id: str) -> None:
        """A claim of ``executor_id`` was refused for its env (409): it
        counts for no item until it claims on this control's env again."""
        self._wrong_env.add(executor_id)

    def claim(self, executor_id: str) -> dict | None:
        self._contact(executor_id)
        self._claimed_at[executor_id] = self._clock()
        self._wrong_env.discard(executor_id)          # it claims on this control's env
        if executor_id in self.quarantined or self._held(executor_id) >= self._max_leases:
            return None
        if self._provider(executor_id) is None:
            logger.error("grade executor %s has no provider_id; never leased", executor_id)
            return None
        for work in list(self._queue):
            if work.future.done():
                self._queue.remove(work)
                continue
            if not self._eligible(executor_id, work):
                continue                         # one vote per executor, and per provider
            self._queue.remove(work)
            lease = _Lease(lease_id=secrets.token_hex(16), work=work, executor_id=executor_id,
                           expires_at=self._clock() + self._lease_seconds[work.mode],
                           leased_at=self._clock())
            self._leases[lease.lease_id] = lease
            self.stats["leased"] += 1
            return {"protocol": GRADE_PROTOCOL, "lease_id": lease.lease_id,
                    "expires_at": lease.expires_at, "env": dict(self._env), "items": [work.item]}
        return None

    def _provider(self, executor_id: str) -> str | None:
        document = self._directory.document(executor_id) or {}
        provider = document.get("provider_id")
        # Normalized as at registration, for any document written before it was.
        provider = str(provider).strip().lower() if provider else ""
        return provider or None

    def _recheck_possible(self) -> bool:
        """Two or more distinct providers among the executors registered on
        this control's env and not quarantined: a recheck can be served."""
        providers = {self._provider(eid) for eid in self._directory.executor_ids()
                     if eid not in self.quarantined and self._directory.is_authorized(eid)
                     and self._on_pinned_env(eid)}
        providers.discard(None)
        possible = len(providers) >= 2
        if possible != self._draws_possible:
            if possible:
                if self._draws_possible is not None:
                    logger.info("grade rechecks enabled: grade executors on %s@%s span %d "
                                "providers", self._env["package"], self._env["version"][:12],
                                len(providers))
            else:
                logger.warning("grade rechecks disabled: grade executors on %s@%s span %d "
                               "provider(s) %s; a passing result needs no second vote "
                               "until a second provider registers", self._env["package"],
                               self._env["version"][:12], len(providers), sorted(providers))
            self._draws_possible = possible
        return possible

    def _on_pinned_env(self, executor_id: str) -> bool:
        document = self._directory.document(executor_id) or {}
        return ((document.get("model_id"), document.get("model_revision"))
                == (self._env["package"], self._env["version"])
                and executor_id not in self._wrong_env)

    def _working(self, now: float) -> list[str]:
        """Live executors that can actually take a next vote: claiming
        lately, or busy with a lease (ruling P26)."""
        return [eid for eid in self._live_executors()
                if now - self._claimed_at.get(eid, -math.inf) <= self._claim_live
                or self._held(eid) > 0]

    def _eligible(self, executor_id: str, work: _Work, *, retry: bool = True) -> bool:
        """It may take ``work``'s next vote: registered for this control's
        env and not refused for it (N2), not excluded, of a provider that has
        not voted on it. One that failed an attempt on it comes back only
        after the retry wait, and only while no executor that never failed
        it could take it instead."""
        provider = self._provider(executor_id)
        if not (executor_id not in work.excluded and provider is not None
                and provider not in work.providers.values()
                and self._on_pinned_env(executor_id)):
            return False
        failed_at = work.failed_at.get(executor_id)
        if failed_at is None:
            return True
        now = self._clock()
        if not retry or now - failed_at < self._retry_excluded:
            return False
        return not any(self._eligible(other, work, retry=False)
                       for other in self._working(now) if other != executor_id)

    @staticmethod
    def _misfit(work: _Work, answer: GradeItemResult) -> str | None:
        if answer.submission_id != work.item["submission_id"]:
            return "for another submission"
        if answer.status == "ok" and any(getattr(answer, fact) is None
                                         for fact in _MODE_FACTS[work.mode]):
            return f"an ok {work.mode} result without {_MODE_FACTS[work.mode]}"
        if answer.status == "ok" and work.mode == "replay":
            # M1: the counts must be the lease's own: every action with an
            # observation compared, mismatches only among those, once each.
            compared = {i for i, action in enumerate(work.item["actions"])
                        if action.get("observation") is not None}
            mismatched = answer.observations_mismatched
            if answer.observations_compared != len(compared):
                return (f"a replay comparing {answer.observations_compared} observations "
                        f"where the lease has {len(compared)}")
            if len(set(mismatched)) != len(mismatched) or not set(mismatched) <= compared:
                return "a replay whose mismatched indices are not the lease's compared ones"
        return None

    def result(self, executor_id: str, lease_id: str, result: GradeResult) -> str:
        """Take an executor's facts for its lease; raises ``LeaseRefused``."""
        self._contact(executor_id)
        lease = self._leases.get(lease_id)
        if lease is None or lease.executor_id != executor_id:
            raise LeaseRefused(410, "lease_unknown")
        del self._leases[lease_id]
        work = lease.work
        if lease.expires_at <= self._clock():
            # Counted like a sweep would have: an expiry, then elsewhere.
            self._take_back(lease, expired=True)
            if self._strike(executor_id):
                self._spawn(self.quarantine(executor_id,
                                            f"{self._strikes_limit} grade leases expired in a row"))
            raise LeaseRefused(410, "lease_expired")
        answer = result.results[0]
        work.failed_at[executor_id] = self._clock()   # until its vote counts
        misfit = self._misfit(work, answer)
        if misfit is not None:
            # Never an honest executor's answer (it echoes the item it ran).
            logger.error("grade executor %s answered lease %s with %s; refused",
                         executor_id, lease_id[:8], misfit)
            self.stats["misfit_results"] += 1
            self._failed_attempt(work, "errors")
            if self._strike(executor_id):
                self._spawn(self.quarantine(executor_id, f"{self._strikes_limit} strikes, "
                                            f"the last a result {misfit}"))
            raise LeaseRefused(422, "result_does_not_fit_the_lease")
        self._strikes[executor_id] = 0
        provider = self._provider(executor_id)
        if provider is None:
            # Its registry entry lost its provider since the claim: a vote that
            # could not be told apart from another provider's never counts.
            logger.error("grade executor %s has no provider_id at result time; refused",
                         executor_id)
            self.stats["providerless_results"] += 1
            self._requeue(work)
            raise LeaseRefused(403, "executor_has_no_provider")
        if answer.status in ("error", "timeout"):
            # The executor's or the box's, never the miner's.
            self.stats[f"executor_{answer.status}s"] += 1
            self._failed_attempt(work, f"{answer.status}s")
            return "requeued"
        if answer.status in TRAJECTORY_STATUSES:
            self.box_failure_votes[executor_id] += 1
        if work.drawn is None:
            work.drawn = self._recheck_possible() and self._rng.random() < self._fraction
        work.results[executor_id] = answer.model_dump()
        work.providers[executor_id] = provider
        work.excluded.add(executor_id)              # a vote is never repeated by it
        work.failed_at.pop(executor_id, None)
        self.stats["graded"] += 1
        self._settle(work)
        return "accepted"

    def _failed_attempt(self, work: _Work, kind: str) -> None:
        """An attempt that judged nobody; enough of them resolve the item unjudged."""
        count = getattr(work, kind) + 1
        setattr(work, kind, count)
        limit, status = (MAX_TIMEOUTS, "timeout") if kind == "timeouts" else (MAX_ERRORS, "error")
        if count >= limit:
            if work.mode == "replay" and work.results and not self._certifying(work):
                # Ruling P28/P28b: out of attempts while holding votes, none
                # certifying it (failed or box failures): void unpaid. With
                # no vote at all only executors failed, before any recorded
                # action (a failure after it is a box vote): unjudged, as before.
                self._uncertified(work, f"{count} {kind}")
                return
            logger.warning("grade item %d (%s): %d %s; resolved unjudged", work.id, work.mode,
                           count, kind)
            self._resolve(work, GradeDecision(status, None, ()))
        else:
            self._requeue(work)

    @staticmethod
    def _certifying(work: _Work) -> bool:
        return any(r.get("status") == "ok" and replay_certified(r) for r in work.results.values())

    def _uncertified(self, work: _Work, why: str) -> None:
        """Ruling P27/P28: a replay item no vote certifies, resolved without an
        agreement; its votes (and their providers) kept, nobody penalized."""
        self.stats[UNCERTIFIED] += 1
        voters = tuple(sorted(work.results))
        logger.warning("grade item %d (replay, submission %s) uncertified after %s: no vote "
                       "certifies it (%s); void unpaid, no sanction", work.id,
                       work.item["submission_id"][:12], why,
                       {e: decision_key(work.mode, r) for e, r in sorted(work.results.items())})
        self._resolve(work, GradeDecision(
            UNCERTIFIED, dict(work.results[voters[0]]) if voters else None, voters,
            tuple(sorted({work.providers[e] for e in voters}))))

    def _settle(self, work: _Work) -> None:
        if work.future.done():
            return
        if not work.results:
            self._requeue(work)
            return
        groups: dict[tuple, list[str]] = collections.defaultdict(list)
        for executor_id, answer in work.results.items():
            groups[decision_key(work.mode, answer)].append(executor_id)
        # Agreement counts providers, not executors: one operator is one vote.
        key, voters = max(groups.items(),
                          key=lambda kv: len({work.providers[e] for e in kv[1]}))
        if len({work.providers[e] for e in voters}) >= work.agree_needed:
            agreeing = tuple(sorted(voters))
            status = UNJUDGEABLE if key == (UNJUDGEABLE,) else "ok"
            self._resolve(work, GradeDecision(
                status, dict(work.results[agreeing[0]]), agreeing,
                tuple(sorted({work.providers[e] for e in agreeing}))))
            if status == UNJUDGEABLE:
                self.stats[UNJUDGEABLE] += 1
            for dissenter in sorted(set(work.results) - set(agreeing)):
                if (status == UNJUDGEABLE
                        or work.results[dissenter].get("status") in TRAJECTORY_STATUSES):
                    # Ruling P26: a split between box failures and facts
                    # penalizes nobody on either side. A trajectory can kill
                    # boxes at random (`[ $((RANDOM%2)) = 0 ] && kill ...`),
                    # so a box that died, or one that survived where others
                    # died, is no evidence against its executor.
                    logger.info("grade item %d (%s): executor %s's vote %s outvoted by a box "
                                "failure split; not penalized", work.id, work.mode, dissenter,
                                decision_key(work.mode, work.results[dissenter]))
                    continue
                # Refused at once; the registry write and listeners follow.
                if self._mark_quarantined(dissenter, f"grade item {work.id} ({work.mode}) "
                                                     f"disagreed with {list(agreeing)}"):
                    self._spawn(self._publish_quarantine(dissenter))
            return
        if len(work.results) >= MAX_RESULTS_PER_ITEM:
            logger.error("grade item %d: %d executors without two agreeing; unjudged",
                         work.id, len(work.results))
            self._resolve(work, GradeDecision("error", None, tuple(sorted(work.results))))
            return
        self._requeue(work)

    # -- expiry, quarantine, background ----------------------------------------

    def _requeue(self, work: _Work) -> None:
        if not work.future.done() and work not in self._queue:
            work.queued_at = self._clock()
            work.swept_at = work.queued_at
            if work.results:
                # It holds a vote and waits for the next one: before any new
                # item, so a backlog never runs out its dispute clock.
                self._queue.appendleft(work)
            else:
                self._queue.append(work)

    def _take_back(self, lease: _Lease, *, expired: bool) -> None:
        lease.work.failed_at[lease.executor_id] = self._clock()
        if expired:
            self._failed_attempt(lease.work, "timeouts")
        else:
            self._requeue(lease.work)

    @staticmethod
    def _resolve(work: _Work, decision: GradeDecision) -> None:
        if not work.future.done():
            work.future.set_result(decision)

    def _on_quarantined(self, executor_id: str) -> None:
        """Its votes on undecided items stop counting before anything awaits,
        so no result arriving meanwhile can agree with a quarantined executor."""
        for holder in self._holders:
            try:
                holder(executor_id)
            except Exception:
                logger.exception("quarantine hold for %s failed", executor_id)
        leased = {id(lease.work) for lease in self._leases.values()}
        for work in list(self._queue) + [lease.work for lease in self._leases.values()]:
            if work.future.done() or executor_id not in work.results:
                continue
            work.results.pop(executor_id, None)
            work.providers.pop(executor_id, None)
            if id(work) not in leased:
                self._settle(work)               # a leased one settles when its lease answers

    async def _sweep(self) -> None:
        await self._expire_leases()
        now = self._clock()
        live = self._live_executors()
        # Who can actually take a next vote: claiming lately, or busy with a lease.
        working = self._working(now)
        waiting = 0
        for work in list(self._queue):
            if work.future.done():
                continue
            served = any(self._eligible(eid, work) for eid in working)
            if work.results:
                # Only time with no live eligible executor counts (F3): one that
                # is merely busy takes this item next (voted items go first).
                since = work.swept_at if work.swept_at is not None else now
                if not served:
                    work.unserved += max(0.0, now - since)
                work.swept_at = now
            if work.results and work.unserved >= self._dispute_seconds:
                if work.mode == "replay" and not self._certifying(work):
                    # Ruling P27: no vote certifies it. Nobody is sanctioned
                    # (no agreement), but it is not paid as a dispute either.
                    self._uncertified(work, f"{work.unserved:.0f} s without a next executor")
                    continue
                # No distinct executor came for the next vote: nobody is judged.
                self.stats[DISPUTED] += 1
                logger.warning(
                    "grade item %d (%s, submission %s) disputed: no distinct-provider executor "
                    "was available for %.0f s after %s; no sanction, not certified", work.id,
                    work.mode, work.item["submission_id"][:12], work.unserved,
                    {e: decision_key(work.mode, r) for e, r in sorted(work.results.items())})
                self._resolve(work, GradeDecision(DISPUTED, None, tuple(sorted(work.results))))
                continue
            waiting += 1
            if live and not served:
                self.stats["stranded"] += 1
                logger.warning("grade item %d (%s) waits: every live grade executor is excluded "
                               "from it (voted, failed it and still inside the retry wait, "
                               "same provider as a voter, not on this control's env, or "
                               "neither claiming nor holding a lease)",
                               work.id, work.mode)
        self.stats["waiting"] = waiting
        if waiting and not live:
            logger.warning("%d grade items wait and no grade executor is connected", waiting)
        if self._unwritten_quarantines:
            await self._write_quarantines()


def build_grade_executor_router(dispatcher: RemoteGradeDispatcher,
                                directory: ExecutorDirectory) -> APIRouter:
    """``/corpus/internal/grade/...``: claim, result, heartbeat, each behind
    ``Authorization: Bearer <grade executor token>``."""
    router = APIRouter()
    _authenticated = bearer_authenticator(directory)

    @router.post(f"{GRADE_PREFIX}/claim")
    async def claim(body: GradeClaimRequest, request: Request):
        document = _authenticated(request, body.executor_id)
        if (body.env_package, body.env_version) != (document["model_id"], document["model_revision"]):
            dispatcher.refused_env(document["executor_id"])
            raise HTTPException(status_code=409, detail="wrong_env")
        lease = dispatcher.claim(document["executor_id"])
        if lease is None:
            return Response(status_code=204)
        return lease

    @router.post(f"{GRADE_PREFIX}/heartbeat")
    async def heartbeat(body: HeartbeatRequest, request: Request) -> dict:
        document = _authenticated(request, body.executor_id)
        dispatcher.heartbeat(document["executor_id"], body.detail, body.held_leases)
        # The executor learns its env pin here (GradeExecutor.start).
        return {"executor_id": document["executor_id"], "model_id": document["model_id"],
                "model_revision": document["model_revision"]}

    @router.post(GRADE_PREFIX + "/{lease_id}/result")
    async def result(lease_id: str, body: GradeResult, request: Request) -> dict:
        document = _authenticated(request)
        try:
            outcome = dispatcher.result(document["executor_id"], lease_id, body)
        except LeaseRefused as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
        return {"lease_id": lease_id, "outcome": outcome}

    return router


__all__ = ["DISPUTED", "REPLAY_LEASE_MARGIN_SECONDS", "check_replay_lease", "replay_lease_refusal",
           "GRADE_CLAIM_LIVE_SECONDS", "GRADE_DISPUTE_SECONDS", "GRADE_LEASE_SECONDS", "GRADE_PREFIX",
           "GRADE_LEASE_REPORT_GRACE_SECONDS", "GRADE_RETRY_EXCLUDED_SECONDS",
           "TRAJECTORY_STATUSES", "UNCERTIFIED", "UNGRADEABLE", "UNJUDGEABLE", "GradeDecision",
           "RemoteGradeDispatcher", "build_grade_executor_router", "decision_key",
           "replay_certified"]
