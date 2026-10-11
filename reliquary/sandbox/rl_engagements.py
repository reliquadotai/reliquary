"""The RL validator's sandbox engagement. On that validator it replaces
the refusing ``sessions.RlPrecommitEngagements`` stub, which the corpus validator keeps.

A session of an RL episode group names ``rl:{window}:{precommit}:{seed}``. Its terms:

* the precommit is this hotkey's, recorded by the runtime, for the CURRENT service window (checked
  when the terms are read and again under the issuer lock, ``EngagementTerms.still_valid``);
* the seed is one of the task's public pool (``0 <= seed < pool_seeds``, i.e. 2M);
* at most one session per (precommit, seed) (``EngagementTerms.exclusive``, under the issuer lock):
  only a machine-signed ``aborted`` final frees a seed, once per precommit seed; a drain (``voided``
  frees what the miner left live: a selective re-roll), a failed open, an expiry, a box failure, a lapse
  or a withdrawal consume it, so a miner cannot re-roll a draw;
* a precommit whose window started more than ``FILL_CLOSED_MAX_SECONDS`` + ``WINDOW_AGE_MARGIN_S`` ago
  (or whose start this validator does not know) opens nothing (``precommit_stale``): a stalled window
  never outlives the book's memory of its taken seeds;
* a precommit recorded more than 24 h ago opens nothing (``precommit_stale``) and a session's
  ``wall_s + open_window_s`` stays within 24 h, so the book's taken seeds (kept ~25 h) outlive every
  session of a precommit;
* the image and the declared limits come from the env's task source; the budgets from the
  contract, raised to the task's limits;
* the session quota is asked at every open (``SessionQuota.admit_open``) and again under the issuer lock when
  it offers ``admit_locked`` (the RL yield quota, ``rl_quota.YieldQuota``); a request whose seed is taken or
  that the session book's caps refuse never reaches the quota (it is no demand).

The token binds hotkey, engagement (precommit + seed), env, split, index (the task) and checkpoint (the
window's); admission checks every one with ``verify_transcript``.

Sessions are stored under ``sandbox_store.RL_SESSION_PREFIX`` (the wiring passes it), so a restart of
the RL validator reads its taken seeds back and a corpus validator never restores them.

Seed exclusivity holds within ONE RL validator process: the check runs on its in-memory book under its
issuer lock, and the store has no create-only marker per (precommit, seed). Never run two replicas of an
RL validator on one session store."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from reliquary.sandbox.sessions import (
    EngagementTerms,
    Refusal,
    SessionBook,
    token_validity_s,
)
from reliquary.sandbox.tasks import ResolvedTask

logger = logging.getLogger(__name__)

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ENGAGEMENT_KEYS = frozenset({"kind", "precommit"})
_PRECOMMIT_KEYS = frozenset({"precommit_sha256", "seed_index"})
MAX_PRECOMMIT_AGE_S = 86_400          # the book forgets a taken seed ~25 h after its session closed
MAX_SESSION_SPAN_S = 86_400           # the token's validity (sessions.token_validity_s)
WINDOW_AGE_MARGIN_S = 900             # past the longest window (FILL_CLOSED_MAX_SECONDS)


class SessionQuota(Protocol):
    """The per-hotkey session quota, asked at every RL open: None admits, a Refusal refuses.

    A quota may also define ``admit_locked(hotkey, now, *, precommit_sha256=None, seed_index=None,
    environment=None) -> Refusal | None``: the same check, run under the issuer lock against the sessions as
    they are then (it may take a pool grant). Such a quota's ``admit_open`` also takes ``seed_index`` (the
    request it records as demand), and ``forget(hotkey, precommit_sha256)``, when defined, withdraws that
    demand when the precommit's task then fails."""

    async def admit_open(self, hotkey: str, *, environment: str,
                         precommit_sha256: str) -> Refusal | None: ...


class OpenQuota:
    """No quota: every open the other caps admit is admitted."""

    async def admit_open(self, hotkey: str, *, environment: str,
                         precommit_sha256: str) -> Refusal | None:
        return None


class SessionOutcomes(Protocol):
    """Yield accounting: how every episode group ended. ``session_ids`` are the group's
    sessions (empty when the group failed before its transcripts were verified). ``stage``: the refusal's
    stage (None when accepted, or when the caller has none); a stage of the validator's own making makes the
    refusal neutral for the miner's yield."""

    def group_settled(self, *, hotkey: str, precommit_sha256: str, session_ids: tuple[str, ...],
                      accepted: bool, stage: str | None = None) -> None: ...

    def group_handed_back(self, *, hotkey: str, precommit_sha256: str, session_ids: tuple[str, ...],
                          stage: str) -> None:
        """An accepted group the proof could not judge for a reason of the validator's own: unpaid,
        unsanctioned; its sessions must not count against the miner's yield or quota."""


class NoOutcomes:
    def group_settled(self, *, hotkey: str, precommit_sha256: str, session_ids: tuple[str, ...],
                      accepted: bool, stage: str | None = None) -> None:
        return None

    def group_handed_back(self, *, hotkey: str, precommit_sha256: str, session_ids: tuple[str, ...],
                          stage: str) -> None:
        return None


@dataclass(frozen=True)
class EpisodeEnvironmentView:
    """What the engagement book needs of one served episode env."""

    policy: Any                                          # service_contract.EpisodeEnvPolicy
    resolve_task: Callable[[int], Awaitable[ResolvedTask]]


class RlEpisodeEngagements:
    kind = "rl_precommit"

    def __init__(self, *, precommits: Callable[[str], Any],
                 environments: Callable[[str], EpisodeEnvironmentView | None],
                 current_window: Callable[[], int | None], book: SessionBook,
                 recorded_at: Callable[[str], float | None],
                 quota: SessionQuota | None = None, clock: Callable[[], float] = time.time,
                 window_started_at: Callable[[int], float | None] | None = None) -> None:
        """``recorded_at(sha256)``: when the runtime recorded the precommit (its row's ``at``), or None
        (then nothing opens: fail closed). ``window_started_at(window)``: when this validator opened the
        window (in memory), or None (then nothing opens); None as a callable skips the check (tests)."""
        self._precommits = precommits
        self._window_started_at = window_started_at
        self._recorded_at = recorded_at
        self._clock = clock
        self._environments = environments
        self._current_window = current_window
        self._book = book
        self._quota = quota or OpenQuota()
        # set by the RL wiring: a degraded (unserved) episode env is refused retryably before any placement
        self.serving_refusal: Callable[[str], Refusal | None] | None = None

    def _stale(self, window: int) -> Refusal | None:
        from reliquary.constants import FILL_CLOSED_MAX_SECONDS

        current = self._current_window()
        if current is None or window != current:
            return Refusal("precommit_stale", {"window": current})
        if self._window_started_at is not None:
            started = self._window_started_at(window)
            if started is None or float(self._clock()) - float(started) > FILL_CLOSED_MAX_SECONDS + WINDOW_AGE_MARGIN_S:
                return Refusal("precommit_stale", {"window": current, "why": "the window is too old"})
        return None

    def _forget(self, hotkey: str, sha: str) -> None:
        """The quota admitted a request whose precommit's task then failed: it withdraws the precommit's demand."""
        forget = getattr(self._quota, "forget", None)
        if forget is not None:
            forget(hotkey, sha)

    async def terms(self, hotkey: str, engagement: Mapping[str, Any]) -> EngagementTerms | Refusal:
        from reliquary.protocol.service_episode import EpisodeWireError, rl_engagement

        timeout = self._book.policy.io_timeout_s
        retry = self._book.policy.retry_after_s
        value = engagement.get("precommit")
        if (set(engagement) != _ENGAGEMENT_KEYS or not isinstance(value, Mapping)
                or set(value) != _PRECOMMIT_KEYS):
            return Refusal("engagement_kind_unsupported",
                           {"why": "an RL engagement names its precommit and its seed, nothing else"})
        sha, seed = value["precommit_sha256"], value["seed_index"]
        if not isinstance(sha, str) or not _SHA.fullmatch(sha) or type(seed) is not int or seed < 0:
            return Refusal("engagement_kind_unsupported", {"why": "malformed precommit digest or seed"})
        try:
            precommit = await asyncio.wait_for(asyncio.to_thread(self._precommits, sha), timeout)
        except Exception as exc:
            logger.warning("rl session open: precommit %s unreadable (%s)", sha[:12], type(exc).__name__)
            return Refusal("ledger_unavailable", {}, retry_after=retry)
        if precommit is None or precommit.hotkey != hotkey or precommit.sha256 != sha:
            return Refusal("precommit_unknown", {"precommit_sha256": sha})
        stale = self._stale(precommit.window)
        if stale is not None:
            return stale
        try:
            recorded = await asyncio.wait_for(asyncio.to_thread(self._recorded_at, sha), timeout)
        except Exception as exc:
            logger.warning("rl session open: precommit %s age unreadable (%s)", sha[:12], type(exc).__name__)
            return Refusal("ledger_unavailable", {}, retry_after=retry)
        if recorded is None or float(self._clock()) - float(recorded) > MAX_PRECOMMIT_AGE_S:
            return Refusal("precommit_stale", {"why": "recorded more than 24 h ago"})
        view = self._environments(precommit.environment)
        if view is None:
            return Refusal("environment_not_served", {"environment": precommit.environment})
        if self.serving_refusal is not None:
            refusal = self.serving_refusal(precommit.environment)      # a degraded env: no machine serves it
            if refusal is not None:
                return refusal
        policy = view.policy
        if seed >= policy.pool_seeds:
            return Refusal("seed_out_of_pool", {"seed_index": seed, "pool_seeds": policy.pool_seeds})
        try:
            name = rl_engagement(precommit.window, sha, seed)
        except EpisodeWireError:
            return Refusal("seed_out_of_pool", {"seed_index": seed, "pool_seeds": policy.pool_seeds})
        if self._book.engagement_held(name):
            return Refusal("engagement_taken", {"engagement": name})       # a used seed is no demand
        refused = self._book.open_refusal(hotkey, int(self._clock()))
        if refused is not None:
            return refused                                                    # the book's caps first: no demand
        locked = getattr(self._quota, "admit_locked", None)
        seeded = {} if locked is None else {"seed_index": seed}             # OpenQuota keeps its signature
        try:
            refused = await asyncio.wait_for(
                self._quota.admit_open(hotkey, environment=precommit.environment, precommit_sha256=sha,
                                       **seeded),
                timeout)
        except Exception as exc:                 # fail closed, retryably: the quota cannot answer now
            logger.warning("rl session open: quota unanswered (%s)", type(exc).__name__)
            return Refusal("ledger_unavailable", {}, retry_after=retry)
        if refused is not None:
            return refused
        try:
            task = await asyncio.wait_for(view.resolve_task(precommit.task_index), timeout)
        except Exception as exc:
            logger.warning("rl session open: task %d of %s unresolved (%s)", precommit.task_index,
                           precommit.environment, type(exc).__name__)
            self._forget(hotkey, sha)
            return Refusal("task_unavailable", {"task_index": precommit.task_index}, retry_after=retry)
        budgets = session_budgets(policy, task)
        prepare = getattr(task, "prepare_timeout_s", None)
        if token_validity_s(self._book.policy.open_window_s, int(budgets["wall_s"]), prepare) > MAX_SESSION_SPAN_S:
            self._forget(hotkey, sha)
            return Refusal("session_too_long", {"wall_s": int(budgets["wall_s"]),
                                                "max_span_s": MAX_SESSION_SPAN_S})
        window, environment = precommit.window, precommit.environment

        def still_valid() -> Refusal | None:
            # Under the issuer lock: the window again, then the quota against the sessions as they are now (a
            # pooled hotkey's first open of this precommit takes its pool grant here).
            stale = self._stale(window)
            if stale is not None or locked is None:
                return stale
            # no seed: the pre-check recorded the request (a seed taken meanwhile must leave no demand)
            return locked(hotkey, int(self._clock()), precommit_sha256=sha, environment=environment)

        return EngagementTerms(
            engagement=name, env=policy.sandbox_env, split=policy.split, index=precommit.task_index,
            checkpoint=precommit.checkpoint, image=task.image, env_package=policy.env_package,
            budgets=budgets, exclusive=True, still_valid=still_valid, prepare_timeout_s=prepare,
            env_options_sha256=getattr(task, "env_options_sha256", None),
            env_lock_sha256=getattr(task, "env_lock_sha256", None),
            # the session's agent box on its machine: the task's CPUs (None: the machine's box) and memory budget
            reserve={"cpu_millis": getattr(task, "cpu_millis", None), "memory_bytes": int(budgets["memory_bytes"])})


def session_budgets(policy, task) -> dict[str, int]:
    """The session's budgets: the policy's, raised to the task's declared limits (what ``terms`` grants; the
    precommit route's servable check uses the same)."""
    budgets = policy.budgets_dict()
    for limit_name, limit in task.limits.items():
        if limit_name in budgets:
            budgets[limit_name] = max(budgets[limit_name], int(limit))
    return budgets


__all__ = ["WINDOW_AGE_MARGIN_S", "EpisodeEnvironmentView", "NoOutcomes", "OpenQuota", "RlEpisodeEngagements", "SessionOutcomes",
           "SessionQuota", "session_budgets"]
