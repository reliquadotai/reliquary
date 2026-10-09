"""The parent-side half of an episode group's admission (phase 2, plan 2C).

The isolated admission child parsed the body, ran the service policy (the request's checkpoint is the
window's announced one) and checked the miner's signatures; the transcripts are verified here, off the
event loop, because this process holds the machine directory and the session issuer (as the corpus
signed intake does). Then the M sessions are claimed, all or none under one hold of the issuer lock
(each one then held against close, drain and lapse until the batcher has answered) and, once it has,
``settle`` marks them ``submitted`` (accepted: never paid twice) or releases them. Any refusal or
exception after the claim releases every claim. Plan 2D's yield accounting is told how every group
ended (``SessionOutcomes``)."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

from reliquary.constants import CODE_ADMISSION_WALL_SECONDS, MATH_ADMISSION_WALL_SECONDS
from reliquary.protocol.submission import RejectReason
from reliquary.sandbox.rl_engagements import NoOutcomes, SessionOutcomes
from reliquary.validator.episode_admission import EpisodeGroupChecker, EpisodeRefusal, finish_prepared

logger = logging.getLogger(__name__)

# The issuer's claim refusals; any other (``session_unknown``: not this hotkey's or not a session this
# validator issued, ``session_not_submittable``) is a transcript the issuer's book contradicts.
#
# Retryable and refunded (WORKER_DROPPED) only when the validator itself could not answer (``session_busy``:
# the issuer lock was not free in time). A session or a precommit held by another group of the same
# hotkey in flight is the miner's own doing: retryable, never refunded (RATE_LIMITED).
_CLAIM_REFUSALS = {
    "session_claimed": (RejectReason.RATE_LIMITED, "episode_group_in_flight"),
    "session_busy": (RejectReason.WORKER_DROPPED, "episode_session_busy"),
    "session_submitted": (RejectReason.HASH_DUPLICATE, "episode_session_reused"),
    "session_expired": (RejectReason.PRECOMMIT_EXPIRED, "episode_deadline"),
    # Ruling: one paid group per precommit, even on disjoint seeds.
    "precommit_submitted": (RejectReason.HASH_DUPLICATE, "episode_precommit_used"),
    "precommit_claimed": (RejectReason.RATE_LIMITED, "episode_group_in_flight"),
}
# Refusals the miner may retry at once: the server reserves no (operator, prompt) identity for them.
# Every other episode refusal (a bad transcript, ``episode_rate``, ``episode_timeout``: the miner's own
# group took too long) keeps it reserved for the window like any refused group.
RETRYABLE_STAGES = frozenset({"episode_group_in_flight", "episode_session_busy", "episode_directory",
                              "episode_checker_busy", "episode_persist_failed"})
# Per hotkey: transcript checks (signatures, a renderer parse of M episodes) started per minute, and in
# flight at once (until the check's thread returns, even when the admission stopped waiting for it: one
# hotkey never holds more than ``max_checks_in_flight`` of the global slots). A refusal before the
# checker (unserved env, stale directory) does not count.
DEFAULT_MAX_CHECKS_PER_MINUTE = 30
DEFAULT_MAX_CHECKS_IN_FLIGHT = 2
# Every intake together: transcript checks running at once (threads of their own). A check is not
# interruptible: a cancelled admission's check keeps its slot until its thread returns. Saturated, a
# group is refused retryably (WORKER_DROPPED, refunded: the validator's capacity, not the miner's doing).
MAX_CHECKS_RUNNING = 3
_CHECKS_RUNNING = threading.BoundedSemaphore(MAX_CHECKS_RUNNING)
_CHECK_THREADS = ThreadPoolExecutor(max_workers=MAX_CHECKS_RUNNING, thread_name_prefix="episode-check")
# The longest admission deadline the server gives a group (``ValidatorServer._admission_wall_seconds``)
# and how many times over a session claim must outlive it: a claim then spans the check, the ingress-order
# wait and the batcher's answer, and only a leaked claim reaches its ttl.
ADMISSION_DEADLINE_S = max(MATH_ADMISSION_WALL_SECONDS, CODE_ADMISSION_WALL_SECONDS)
CLAIM_TTL_MARGIN = 4


@dataclass(frozen=True)
class EpisodeClaim:
    hotkey: str
    precommit_sha256: str
    session_ids: tuple[str, ...]


def _precommit_sha(request) -> str:
    try:
        value = request.rollouts[0].commit["rollout"]["episode"]["precommit_sha256"]
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""
    return value if isinstance(value, str) else ""


def build_episode_checker(policy, *, checkpoint_dir: str, source, chunk_tokens: int) -> EpisodeGroupChecker:
    """The checker of one episode env: the turn renderer of the policy's checkpoint over the CONTRACT's
    tools (``policy.tools``), the env's task source. Rebuild it (``EpisodeGroupIntake.set_checkers``)
    whenever the contract or the task source changes: the checker caches rendered prompts per task."""
    from reliquary.environment import agentic_swe

    return EpisodeGroupChecker(policy=policy,
                               renderer=agentic_swe.load_turn_renderer(checkpoint_dir, tools=tuple(policy.tools)),
                               source=source, chunk_tokens=chunk_tokens)


class EpisodeGroupIntake:
    """``directory(now)``: the machine directory, or None while it is stale (``fleet.directory_if_ready``);
    ``token_verifier``: THIS validator's own session-token keys only (the tokens it issued; never a key
    read from the directory or the request); ``sessions``: the RL session issuer (``claim_all``,
    ``release_claim``, ``submitted``); ``seen()``: an immutable snapshot of the paid session ids
    (``SessionBook.submitted_ids``: restored from the durable store at start); ``precommits(sha)``:
    ``ServiceRuntime.episode_precommit`` (thread-safe: called off the loop). A checker whose policy is
    not the window contract's refuses its groups (``episode_policy_stale``) until it is rebuilt."""

    def __init__(self, *, checkers: Mapping[str, EpisodeGroupChecker], precommits: Callable[[str], Any],
                 directory: Callable[[float], Any | None], token_verifier, sessions,
                 seen: Callable[[], Collection[str]], outcomes: SessionOutcomes | None = None,
                 max_checks_per_minute: int = DEFAULT_MAX_CHECKS_PER_MINUTE,
                 max_checks_in_flight: int = DEFAULT_MAX_CHECKS_IN_FLIGHT,
                 clock: Callable[[], float] = time.monotonic,
                 admission_deadline_s: float = ADMISSION_DEADLINE_S) -> None:
        ttl = getattr(getattr(sessions, "policy", None), "claim_ttl_s", None)
        if not isinstance(ttl, (int, float)) or ttl < CLAIM_TTL_MARGIN * float(admission_deadline_s):
            raise ValueError(f"the session claim ttl ({ttl!r} s) must be at least {CLAIM_TTL_MARGIN}x the "
                             f"admission deadline ({admission_deadline_s} s)")
        self._checkers = dict(checkers)
        self._precommits = precommits
        self._directory = directory
        self._tokens = token_verifier
        self._sessions = sessions
        self._seen = seen
        self._outcomes = outcomes or NoOutcomes()
        self._max_per_minute = int(max_checks_per_minute)
        self._max_in_flight = int(max_checks_in_flight)
        self._clock = clock
        self._recent: dict[str, list[float]] = {}
        self._in_flight: dict[str, int] = {}
        self._in_flight_lock = threading.Lock()         # a check's thread ends its count
        self._tasks: set[asyncio.Task] = set()

    @property
    def environments(self) -> tuple[str, ...]:
        return tuple(sorted(self._checkers))

    def set_checkers(self, checkers: Mapping[str, EpisodeGroupChecker]) -> None:
        """Replace every checker (a new contract or task source)."""
        self._checkers = dict(checkers)

    def _rate_refusal(self, hotkey: str) -> dict | None:
        """None, and one more check of ``hotkey`` in flight (``_check_ended`` ends it); else why not."""
        now = float(self._clock())
        recent = [t for t in self._recent.get(hotkey, ()) if t > now - 60.0]
        with self._in_flight_lock:
            running = self._in_flight.get(hotkey, 0)
            if running >= self._max_in_flight:
                self._recent[hotkey] = recent
                return {"in_flight": running, "max": self._max_in_flight}
            if len(recent) >= self._max_per_minute:
                self._recent[hotkey] = recent
                return {"per_minute": len(recent), "max": self._max_per_minute}
            self._in_flight[hotkey] = running + 1
        recent.append(now)
        self._recent[hotkey] = recent
        if len(self._recent) > 4096:            # forget idle hotkeys
            for key in [k for k, v in self._recent.items() if not v or v[-1] <= now - 60.0]:
                self._recent.pop(key, None)
        return None

    def _report(self, hotkey: str, precommit_sha256: str, session_ids, *, accepted: bool) -> None:
        try:
            self._outcomes.group_settled(hotkey=hotkey, precommit_sha256=precommit_sha256,
                                         session_ids=tuple(session_ids), accepted=accepted)
        except Exception:
            logger.exception("the episode outcome hook failed")

    def _refuse(self, prepared, reason: RejectReason, stage: str, detail: dict | None = None,
                session_ids=()):
        request = prepared.request
        prepared.episode_pending = False
        prepared.reject_reason = reason
        prepared.reject_stage = stage
        logger.info("episode group of %s refused at %s: %s", str(request.miner_hotkey)[:12], stage, detail or {})
        self._report(str(request.miner_hotkey), _precommit_sha(request), session_ids, accepted=False)
        return prepared, None

    def _check_ended(self, hotkey: str) -> None:
        """Called from the event loop or from the check's own thread."""
        with self._in_flight_lock:
            left = self._in_flight.get(hotkey, 1) - 1
            if left > 0:
                self._in_flight[hotkey] = left
            else:
                self._in_flight.pop(hotkey, None)

    def _forget_check(self, hotkey: str) -> None:
        """A check refused for capacity does not count against the hotkey's per-minute budget."""
        recent = self._recent.get(hotkey)
        if recent:
            recent.pop()

    def _release_if_claimed(self, claiming: asyncio.Future, session_ids) -> None:
        if claiming.cancelled() or claiming.exception() is not None or claiming.result() is not None:
            return
        self._spawn_release(session_ids)

    def _spawn_release(self, session_ids) -> asyncio.Task:
        """The release as a task of its own, kept until it ends (a cancelled caller never cuts it)."""
        task = asyncio.get_running_loop().create_task(self._release(tuple(session_ids)))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _release(self, session_ids) -> None:
        for session_id in session_ids:
            try:
                await self._sessions.release_claim(session_id)
            except Exception:
                logger.exception("episode session %s claim not released (its claim ttl ends it)", session_id)

    async def admit(self, *, environment: str, prepared, received: float, contract):
        """Verify, claim, complete ``prepared`` in place. Returns ``(prepared, claim)``: a claim only for a
        group the batcher may now take (the caller must ``settle`` it); None with ``prepared`` refused."""
        request = prepared.request
        checker = self._checkers.get(environment)
        if checker is None:
            return self._refuse(prepared, RejectReason.GENERATION_CONTRACT_MISMATCH, "episode_unserved")
        try:
            current = contract.episode_policy(environment)
        except ValueError:
            current = None
        if current is None or checker.policy != current:
            logger.error("episode checker of %s is not the window contract's: rebuild it", environment)
            return self._refuse(prepared, RejectReason.GENERATION_CONTRACT_MISMATCH, "episode_policy_stale")
        directory = self._directory(received)
        if directory is None:
            return self._refuse(prepared, RejectReason.WORKER_DROPPED, "episode_directory")
        hotkey = str(request.miner_hotkey)
        limited = self._rate_refusal(hotkey)
        if limited is not None:
            return self._refuse(prepared, RejectReason.RATE_LIMITED, "episode_rate", limited)
        sha = _precommit_sha(request)
        handed_off = False                  # to the check's thread, which then ends the hotkey's count
        try:
            precommit = await asyncio.to_thread(self._precommits, sha) if sha else None
            if not _CHECKS_RUNNING.acquire(blocking=False):
                self._forget_check(hotkey)
                return self._refuse(prepared, RejectReason.WORKER_DROPPED, "episode_checker_busy",
                                    {"running": MAX_CHECKS_RUNNING})
            try:
                # Off the event loop: signatures and a renderer parse of M episodes. The slot is
                # released when the check's thread returns (or never starts), not when we stop waiting.
                future = _CHECK_THREADS.submit(checker.check, request, precommit=precommit, directory=directory,
                                               token_verifier=self._tokens, seen=self._seen(), received=received)
            except BaseException:
                _CHECKS_RUNNING.release()
                raise
            handed_off = True

            def ended(_done) -> None:
                try:
                    _CHECKS_RUNNING.release()
                finally:
                    self._check_ended(hotkey)

            future.add_done_callback(ended)
            try:
                outcome = await asyncio.wrap_future(future)
            except Exception:
                logger.error("episode group of %s: the transcript check raised; refused", hotkey[:12],
                             exc_info=True)
                return self._refuse(prepared, RejectReason.BAD_SCHEMA, "episode_transcript",
                                    {"check": "raised"})
        finally:
            if not handed_off:
                self._check_ended(hotkey)
        if isinstance(outcome, EpisodeRefusal):
            return self._refuse(prepared, outcome.reason, outcome.stage, outcome.detail)
        # The checker passed, so ``sha`` is the precommit's: the precommit is taken with its sessions.
        claiming = asyncio.ensure_future(self._sessions.claim_all(
            outcome.session_ids, hotkey=request.miner_hotkey, received=received, precommit_sha256=sha))
        try:
            refused = await asyncio.shield(claiming)
        except asyncio.CancelledError:
            # The admission's deadline (or a drain) cancelled us mid-claim: whatever the claim takes is
            # released as soon as it has taken it.
            claiming.add_done_callback(lambda done: self._release_if_claimed(done, outcome.session_ids))
            raise
        if refused is not None:
            session_id, refusal = refused
            reason, stage = _CLAIM_REFUSALS.get(refusal.reason,
                                                (RejectReason.REWARD_MISMATCH, "episode_transcript"))
            return self._refuse(prepared, reason, stage, {"session_id": session_id, "claim": refusal.reason},
                                outcome.session_ids)
        claim = EpisodeClaim(hotkey, sha, tuple(outcome.session_ids))
        try:
            finish_prepared(prepared, outcome, contract)
        except BaseException:
            await asyncio.shield(self._spawn_release(claim.session_ids))
            self._report(hotkey, sha, claim.session_ids, accepted=False)
            raise
        if prepared.reject_reason is not None:
            await asyncio.shield(self._spawn_release(claim.session_ids))
            self._report(hotkey, sha, claim.session_ids, accepted=False)
            return prepared, None
        return prepared, claim

    async def persist(self, claim: EpisodeClaim) -> bool:
        """Before the batcher takes the group: the sessions' ``submitted`` records written (awaited).
        False when not one could be: the caller must not let the batcher take the group (a restart
        would not know these sessions paid). One stored record is enough: after a restart it keeps the
        precommit taken (``claim_all`` refuses any other group of it), and its own session is paid."""
        try:
            return await self._sessions.persist_submitted(claim.session_ids) > 0
        except Exception:
            logger.exception("episode sessions %s: submitted records not written", ",".join(claim.session_ids))
            return False

    async def drain(self, timeout: float) -> int:
        """Clean shutdown: wait up to ``timeout`` s for the releases and settlements in flight, then
        cancel what is left (a cut claim lapses at its ttl). Returns how many were cut."""
        pending = [task for task in self._tasks if not task.done()]
        if not pending:
            return 0
        _, late = await asyncio.wait(pending, timeout=timeout)
        for task in late:
            task.cancel()
        if late:
            await asyncio.gather(*late, return_exceptions=True)
            logger.error("ALERT %d episode settlements cut at shutdown; their claims lapse at the ttl", len(late))
        return len(late)

    async def settle(self, claim: EpisodeClaim, *, accepted: bool) -> None:
        """After the batcher's answer: accepted sessions end ``submitted``; others are released (the
        miner may submit them again). Shielded: a cancellation of the caller never leaves an accepted
        group's sessions unpaid-for in the book (a stale claim would let them be claimed again)."""
        task = asyncio.get_running_loop().create_task(self._settle(claim, accepted=accepted))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        await asyncio.shield(task)

    async def _settle(self, claim: EpisodeClaim, *, accepted: bool) -> None:
        if accepted:
            # Every session of the group moved together; a record ``persist`` could not write is retried.
            try:
                await self._sessions.submitted_all(claim.session_ids)
            except Exception:
                logger.exception("episode sessions %s not settled", ",".join(claim.session_ids))
        else:
            for session_id in claim.session_ids:
                try:
                    await self._sessions.release_claim(session_id)
                except Exception:
                    logger.exception("episode session %s not settled", session_id)
        self._report(claim.hotkey, claim.precommit_sha256, claim.session_ids, accepted=accepted)


__all__ = ["ADMISSION_DEADLINE_S", "CLAIM_TTL_MARGIN", "DEFAULT_MAX_CHECKS_IN_FLIGHT",
           "DEFAULT_MAX_CHECKS_PER_MINUTE", "MAX_CHECKS_RUNNING", "RETRYABLE_STAGES", "EpisodeClaim",
           "EpisodeGroupIntake", "build_episode_checker"]
