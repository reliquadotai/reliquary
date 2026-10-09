"""The RL validator's sandbox engagement (phase 2, plan 2C; spec §4.1.3). On that validator it replaces
the refusing ``sessions.RlPrecommitEngagements`` stub, which the corpus validator keeps.

A session of an RL episode group names ``rl:{window}:{precommit}:{seed}``. Its terms:

* the precommit is this hotkey's, recorded by the runtime, for the CURRENT service window (checked
  when the terms are read and again under the issuer lock, ``EngagementTerms.still_valid``);
* the seed is one of the task's public pool (``0 <= seed < pool_seeds``, i.e. 2M);
* at most one session per (precommit, seed) (``EngagementTerms.exclusive``, under the issuer lock):
  only a machine-signed ``aborted`` final frees a seed, once per precommit seed; a drain (``voided``
  frees what the miner left live: a selective re-roll), a failed open, an expiry, a box failure, a lapse
  or a withdrawal consume it, so a miner cannot re-roll a draw;
* a precommit recorded more than 24 h ago opens nothing (``precommit_stale``) and a session's
  ``wall_s + open_window_s`` stays within 24 h, so the book's taken seeds (kept ~25 h) outlive every
  session of a precommit;
* the image and the declared limits come from the env's task source (plan 2A); the budgets from the
  contract, raised to the task's limits;
* plan 2D's quota is asked at every open (``SessionQuota``; permissive until then).

The token binds hotkey, engagement (precommit + seed), env, split, index (the task) and checkpoint (the
window's); admission checks every one with ``verify_transcript``.

Sessions are stored under ``sandbox_store.RL_SESSION_PREFIX`` (the wiring passes it), so a restart of
the RL validator reads its taken seeds back and a corpus validator never restores them."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from reliquary.sandbox.sessions import EngagementTerms, Refusal, SessionBook
from reliquary.sandbox.tasks import ResolvedTask

logger = logging.getLogger(__name__)

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ENGAGEMENT_KEYS = frozenset({"kind", "precommit"})
_PRECOMMIT_KEYS = frozenset({"precommit_sha256", "seed_index"})
MAX_PRECOMMIT_AGE_S = 86_400          # the book forgets a taken seed ~25 h after its session closed
MAX_SESSION_SPAN_S = 86_400           # wall_s + open_window_s


class SessionQuota(Protocol):
    """Plan 2D's per-hotkey session quota, asked at every RL open: None admits, a Refusal refuses."""

    async def admit_open(self, hotkey: str, *, environment: str,
                         precommit_sha256: str) -> Refusal | None: ...


class OpenQuota:
    """No quota (until plan 2D): every open the other caps admit is admitted."""

    async def admit_open(self, hotkey: str, *, environment: str,
                         precommit_sha256: str) -> Refusal | None:
        return None


class SessionOutcomes(Protocol):
    """Plan 2D's yield accounting: how every episode group ended. ``session_ids`` are the group's
    sessions (empty when the group failed before its transcripts were verified)."""

    def group_settled(self, *, hotkey: str, precommit_sha256: str, session_ids: tuple[str, ...],
                      accepted: bool) -> None: ...


class NoOutcomes:
    def group_settled(self, *, hotkey: str, precommit_sha256: str, session_ids: tuple[str, ...],
                      accepted: bool) -> None:
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
                 quota: SessionQuota | None = None, clock: Callable[[], float] = time.time) -> None:
        """``recorded_at(sha256)``: when the runtime recorded the precommit (its row's ``at``), or None
        (then nothing opens: fail closed)."""
        self._precommits = precommits
        self._recorded_at = recorded_at
        self._clock = clock
        self._environments = environments
        self._current_window = current_window
        self._book = book
        self._quota = quota or OpenQuota()

    def _stale(self, window: int) -> Refusal | None:
        current = self._current_window()
        if current is None or window != current:
            return Refusal("precommit_stale", {"window": current})
        return None

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
        policy = view.policy
        if seed >= policy.pool_seeds:
            return Refusal("seed_out_of_pool", {"seed_index": seed, "pool_seeds": policy.pool_seeds})
        try:
            name = rl_engagement(precommit.window, sha, seed)
        except EpisodeWireError:
            return Refusal("seed_out_of_pool", {"seed_index": seed, "pool_seeds": policy.pool_seeds})
        try:
            refused = await asyncio.wait_for(
                self._quota.admit_open(hotkey, environment=precommit.environment, precommit_sha256=sha),
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
            return Refusal("task_unavailable", {"task_index": precommit.task_index}, retry_after=retry)
        budgets = policy.budgets_dict()
        for limit_name, limit in task.limits.items():
            if limit_name in budgets:
                budgets[limit_name] = max(budgets[limit_name], int(limit))
        if int(budgets["wall_s"]) + self._book.policy.open_window_s > MAX_SESSION_SPAN_S:
            return Refusal("session_too_long", {"wall_s": int(budgets["wall_s"]),
                                                "max_span_s": MAX_SESSION_SPAN_S})
        window = precommit.window
        return EngagementTerms(
            engagement=name, env=policy.sandbox_env, split=policy.split, index=precommit.task_index,
            checkpoint=precommit.checkpoint, image=task.image, env_package=policy.env_package,
            budgets=budgets, exclusive=True, still_valid=lambda: self._stale(window))


__all__ = ["EpisodeEnvironmentView", "NoOutcomes", "OpenQuota", "RlEpisodeEngagements", "SessionOutcomes",
           "SessionQuota"]
