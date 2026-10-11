"""The parent-side half of an episode group's admission.

The isolated admission child parsed the body, ran the service policy (the request's checkpoint is the
window's announced one) and checked the miner's signatures; the transcripts are verified here, off the
event loop, because this process holds the machine directory and the session issuer (as the corpus
signed intake does). Then the M sessions are claimed, all or none under one hold of the issuer lock
(each one then held against close, drain and lapse until the batcher has answered) and, once it has,
``settle`` marks them ``submitted`` (accepted: never paid twice) or releases them. Any refusal or
exception after the claim releases every claim. The yield accounting is told how every group
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
from reliquary.protocol.episode_retry import RETRYABLE_STAGES
from reliquary.protocol.service_episode import GROUPS_IN_FLIGHT
from reliquary.protocol.submission import RejectReason
from reliquary.sandbox.rl_engagements import NoOutcomes, SessionOutcomes
from reliquary.sandbox.rl_yield import VALIDATOR_STAGES as _AFTER_CLAIM_STAGES
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
    # One paid group per precommit, even on disjoint seeds.
    "precommit_submitted": (RejectReason.HASH_DUPLICATE, "episode_precommit_used"),
    "precommit_claimed": (RejectReason.RATE_LIMITED, "episode_group_in_flight"),
}
# Refusals the miner may retry at once: the server reserves no (operator, prompt) identity for them.
# ``episode_checks_in_flight`` (the operator's in-flight cap) is retryable but not refunded (RATE_LIMITED).
# Every other episode refusal (a bad transcript, ``episode_rate``: the per-minute budget,
# ``episode_timeout``: the miner's own check ran past the deadline) keeps it reserved for the window
# like any refused group.
# The set lives in ``reliquary.protocol.episode_retry`` (the miner resends on the same stages).
# Every intake together: transcript checks running at once (threads of their own). A check is not
# interruptible: a cancelled admission's check keeps its slot until its thread returns. Saturated, a
# group is refused retryably (WORKER_DROPPED, refunded: the validator's capacity, not the miner's doing).
MAX_CHECKS_RUNNING = GROUPS_IN_FLIGHT + 1
# Transcript checks (signatures, a renderer parse of M episodes): per hotkey, started per minute; per
# OPERATOR (all its hotkeys together), in flight at once (until the check's thread returns, even when
# the admission stopped waiting for it: one operator never holds more than ``max_checks_in_flight`` of
# the global slots, one is always left to the others). A refusal before the checker (unserved env, stale
# directory) does not count.
DEFAULT_MAX_CHECKS_PER_MINUTE = 30
DEFAULT_MAX_CHECKS_IN_FLIGHT = GROUPS_IN_FLIGHT
_CHECKS_RUNNING = threading.BoundedSemaphore(MAX_CHECKS_RUNNING)
_CHECK_THREADS = ThreadPoolExecutor(max_workers=MAX_CHECKS_RUNNING, thread_name_prefix="episode-check")
# The longest admission deadline the server gives a group (``ValidatorServer._admission_wall_seconds``)
# and how many times over a session claim must outlive it: a claim spans the check (to the deadline),
# the ingress-order wait and the records' write (to the deadline plus the predecessor's own wall: two
# walls in all), then the batcher's answer; only a leaked claim reaches its ttl.
ADMISSION_DEADLINE_S = max(MATH_ADMISSION_WALL_SECONDS, CODE_ADMISSION_WALL_SECONDS)
CLAIM_TTL_MARGIN = 4
# The longest a server holds a claim before settling it, in admission walls (see above).
CLAIM_HOLD_WALLS = 2
assert CLAIM_TTL_MARGIN > CLAIM_HOLD_WALLS


class CheckProgress:
    """Where an ``admit`` got to, for a caller that cuts it at a deadline: did the transcript check
    itself run past it (the miner's group took too long), or was the cut elsewhere (the validator's
    lock wait, precommit lookup or claim)? Times are ``time.monotonic``; ``ended`` is set from the
    check's own thread when it returns."""

    def __init__(self) -> None:
        self.started: float | None = None
        self.ended: float | None = None

    def ran_past(self, deadline: float) -> bool:
        if self.started is None:
            return False
        ended = self.ended
        return ended is None or ended >= deadline


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


class EpisodeGroupIntake:
    """``directory(now)``: the machine directory, or None while it is stale (``fleet.directory_if_ready``);
    ``token_verifier``: THIS validator's own session-token keys only (the tokens it issued; never a key
    read from the directory or the request); ``sessions``: the RL session issuer (``claim_all``,
    ``release_claim``, ``submitted``); ``seen()``: an immutable snapshot of the paid session ids
    (``SessionBook.submitted_ids``: restored from the durable store at start); ``precommits(sha)``:
    ``ServiceRuntime.episode_precommit`` (thread-safe: called off the loop). A checker whose policy is
    not the window contract's refuses its groups (``episode_policy_stale``) until it is rebuilt.
    ``window_open(window)``: whether this validator takes episode groups for that window (False for a
    window it resumed after a restart: what the batcher accepted before is lost with its memory, so no
    group of it is taken again: ``episode_window_resumed``); None takes every window (tests).

    ``checks_allowed(hotkey, received)``: with the RL yield quota, the precommits of the groups the hotkey can
    submit (for a group received at ``received``, judged as the session claim judges it). A group naming any
    other precommit is refused before it takes a check slot (no verdict: it may be submittable a moment
    later), and the in-flight check cap is the hotkey's, equal to how many there are (proportional: more
    hotkeys gain nothing, a hotkey with nothing to submit gets no check slot); one operator's hotkeys
    together hold at most ``max_checks_running - 1`` checks, so one slot is always left to the others. A
    hook that raises refuses the group the same way (retryable, no verdict, no slot). None keeps the
    per-operator ``max_checks_in_flight``.
    ``max_checks_running``: this intake's own transcript-check slots (a semaphore and threads of that size);
    None shares the module's ``MAX_CHECKS_RUNNING``. ``checks_scale(hotkey)``: how much the per-hotkey
    per-minute check budget scales (the quota's ``cap_scale``: with the hotkey's allocation); None keeps
    ``max_checks_per_minute`` for every hotkey. The budget holds no slot: the in-flight caps above bound what
    any hotkey or operator holds at once, whatever its budget."""

    def __init__(self, *, checkers: Mapping[str, EpisodeGroupChecker], precommits: Callable[[str], Any],
                 directory: Callable[[float], Any | None], token_verifier, sessions,
                 seen: Callable[[], Collection[str]], outcomes: SessionOutcomes | None = None,
                 max_checks_per_minute: int = DEFAULT_MAX_CHECKS_PER_MINUTE,
                 max_checks_in_flight: int = DEFAULT_MAX_CHECKS_IN_FLIGHT,
                 clock: Callable[[], float] = time.monotonic,
                 admission_deadline_s: float = ADMISSION_DEADLINE_S,
                 window_open: Callable[[int], bool] | None = None,
                 checks_allowed: Callable[[str, float], Collection[str]] | None = None,
                 max_checks_running: int | None = None,
                 checks_scale: Callable[[str], float] | None = None,
                 check_metrics: Any = None) -> None:
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
        self._window_open = window_open
        self._checks_allowed = checks_allowed
        self._checks_scale = checks_scale
        # Told each transcript check's time (``check_time(seconds)``) and each group refused for want of a check
        # slot (``checker_busy(hotkey, precommit_sha256)``): the saturation metrics size MAX_CHECKS_RUNNING.
        self._check_metrics = check_metrics
        if max_checks_running is not None and int(max_checks_running) < 2:
            raise ValueError("max_checks_running must be at least 2 (one slot is always left to the others)")
        self._max_running = MAX_CHECKS_RUNNING if max_checks_running is None else int(max_checks_running)
        # This intake's own slots, or None: the module's (read at each check, so tests can replace them).
        self._slots: threading.BoundedSemaphore | None = None
        self._threads: ThreadPoolExecutor | None = None
        if max_checks_running is not None:
            self._slots = threading.BoundedSemaphore(self._max_running)
            self._threads = ThreadPoolExecutor(max_workers=self._max_running, thread_name_prefix="episode-check")
        # With ``checks_allowed``: checks in flight per operator (all its hotkeys), and, per in-flight key,
        # the operators of its running checks (in start order: ``_check_ended`` ends the last).
        self._op_in_flight: dict[str, int] = {}
        self._op_of: dict[str, list[str]] = {}

    @property
    def environments(self) -> tuple[str, ...]:
        return tuple(sorted(self._checkers))

    def _metric(self, name: str, *args) -> None:
        method = getattr(self._check_metrics, name, None) if self._check_metrics is not None else None
        if method is None:
            return
        try:
            method(*args)
        except Exception:
            logger.exception("the episode check metric %s failed", name)

    def _owner(self, hotkey: str, operator: str | None) -> str:
        """The in-flight cap's key: the hotkey with ``checks_allowed``, else the operator."""
        if self._checks_allowed is not None:
            return str(hotkey)
        return str(operator) if operator else str(hotkey)

    def _rate_refusal(self, hotkey: str, owner: str, *, operator: str | None = None,
                      submittable: int | None = None) -> tuple[str, dict] | None:
        """None, and one more check of ``owner`` in flight (``_check_ended`` ends it); else the
        refusal's stage and why. The in-flight cap reserves no identity (``episode_checks_in_flight``,
        retryable); the per-minute budget does (``episode_rate``). ``operator``: the receipt's (the
        hotkey when unknown), and ``submittable`` (how many groups the hotkey can submit), read only with
        ``checks_allowed``."""
        now = float(self._clock())
        recent = [t for t in self._recent.get(hotkey, ()) if t > now - 60.0]
        if self._checks_allowed is None:
            cap = self._max_in_flight
        else:
            cap = max(0, min(int(submittable or 0), self._max_running - 1))
        limit = self._max_per_minute
        if self._checks_scale is not None:
            try:                    # the quota's memory, on the event loop, before the intake's lock
                limit = max(1, int(self._max_per_minute * float(self._checks_scale(hotkey))))
            except Exception:
                logger.exception("the episode check budget's scale failed for %s; the base budget applies",
                                 hotkey[:12])
        op_key = str(operator) if operator else str(hotkey)
        with self._in_flight_lock:
            running = self._in_flight.get(owner, 0)
            if running >= cap:
                self._recent[hotkey] = recent
                return "episode_checks_in_flight", {"in_flight": running, "max": cap}
            if self._checks_allowed is not None:
                op_running = self._op_in_flight.get(op_key, 0)
                if op_running >= self._max_running - 1:          # one slot always left to other operators
                    self._recent[hotkey] = recent
                    return "episode_checks_in_flight", {"operator_in_flight": op_running,
                                                        "max": self._max_running - 1}
            if len(recent) >= limit:
                self._recent[hotkey] = recent
                return "episode_rate", {"per_minute": len(recent), "max": limit}
            self._in_flight[owner] = running + 1
            if self._checks_allowed is not None:
                self._op_in_flight[op_key] = self._op_in_flight.get(op_key, 0) + 1
                self._op_of.setdefault(owner, []).append(op_key)
        recent.append(now)
        self._recent[hotkey] = recent
        if len(self._recent) > 4096:            # forget idle hotkeys
            for key in [k for k, v in self._recent.items() if not v or v[-1] <= now - 60.0]:
                self._recent.pop(key, None)
        return None

    def _report(self, hotkey: str, precommit_sha256: str, session_ids, *, accepted: bool,
                stage: str | None = None) -> None:
        try:
            self._outcomes.group_settled(hotkey=hotkey, precommit_sha256=precommit_sha256,
                                         session_ids=tuple(session_ids), accepted=accepted, stage=stage)
        except Exception:
            logger.exception("the episode outcome hook failed")

    def _refuse(self, prepared, reason: RejectReason, stage: str, detail: dict | None = None,
                session_ids=(), *, verdict: bool = True):
        request = prepared.request
        prepared.episode_pending = False
        prepared.reject_reason = reason
        prepared.reject_stage = stage
        logger.info("episode group of %s refused at %s: %s", str(request.miner_hotkey)[:12], stage, detail or {})
        # Before the claim the precommit is unverified: a stage of the validator's own making (neutral for the
        # miner's yield) is never reported from here, only from a claimed group's settlement; nor is a refusal
        # whose cause is ours (``verdict=False``). Either gives no verdict: the miner is charged nothing.
        if verdict and stage not in _AFTER_CLAIM_STAGES:
            self._report(str(request.miner_hotkey), _precommit_sha(request), session_ids, accepted=False,
                         stage=stage)
        return prepared, None

    def _check_ended(self, owner: str) -> None:
        """Called from the event loop or from the check's own thread."""
        with self._in_flight_lock:
            left = self._in_flight.get(owner, 1) - 1
            if left > 0:
                self._in_flight[owner] = left
            else:
                self._in_flight.pop(owner, None)
            ops = self._op_of.get(owner)
            if ops:
                op_key = ops.pop()
                if not ops:
                    self._op_of.pop(owner, None)
                op_left = self._op_in_flight.get(op_key, 1) - 1
                if op_left > 0:
                    self._op_in_flight[op_key] = op_left
                else:
                    self._op_in_flight.pop(op_key, None)

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

    async def admit(self, *, environment: str, prepared, received: float, contract,
                    operator: str | None = None, progress: CheckProgress | None = None):
        """Verify, claim, complete ``prepared`` in place. Returns ``(prepared, claim)``: a claim only for a
        group the batcher may now take (the caller must ``settle`` it); None with ``prepared`` refused.
        ``operator``: the receipt's (the in-flight cap's key; the hotkey when unknown). ``progress``:
        filled in for a caller that may cut this call (``CheckProgress.ran_past``)."""
        request = prepared.request
        if self._window_open is not None and not self._window_open(getattr(request, "window_start", None)):
            return self._refuse(prepared, RejectReason.WINDOW_NOT_ACTIVE, "episode_window_resumed")
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
        owner = self._owner(hotkey, operator)
        sha = _precommit_sha(request)
        submittable = None
        if self._checks_allowed is not None:
            # Before any check slot: a group of a precommit the hotkey cannot submit (junk, or sessions not
            # graded yet) costs no check. Retryable, no verdict (the hotkey owns no such precommit, or it may
            # become submittable).
            try:
                groups = frozenset(self._checks_allowed(hotkey, received))
            except Exception:
                logger.exception("the episode check cap's hook failed for %s; group refused retryably", hotkey[:12])
                return self._refuse(prepared, RejectReason.RATE_LIMITED, "episode_checks_in_flight",
                                    {"submittable": "unknown"}, verdict=False)
            if sha not in groups:
                return self._refuse(prepared, RejectReason.RATE_LIMITED, "episode_checks_in_flight",
                                    {"precommit": "not_submittable", "submittable": len(groups)}, verdict=False)
            submittable = len(groups)
        limited = self._rate_refusal(hotkey, owner, operator=operator, submittable=submittable)
        if limited is not None:
            return self._refuse(prepared, RejectReason.RATE_LIMITED, *limited)
        handed_off = False                  # to the check's thread, which then ends the hotkey's count
        try:
            precommit = await asyncio.to_thread(self._precommits, sha) if sha else None
            slots = self._slots if self._slots is not None else _CHECKS_RUNNING
            threads = self._threads if self._threads is not None else _CHECK_THREADS
            if not slots.acquire(blocking=False):
                self._forget_check(hotkey)
                self._metric("checker_busy", hotkey, sha)
                return self._refuse(prepared, RejectReason.WORKER_DROPPED, "episode_checker_busy",
                                    {"running": self._max_running})
            try:
                # Off the event loop: signatures and a renderer parse of M episodes. The slot is
                # released when the check's thread returns (or never starts), not when we stop waiting.
                if progress is not None:
                    progress.started = time.monotonic()
                check_started = time.monotonic()
                future = threads.submit(checker.check, request, precommit=precommit, directory=directory,
                                               token_verifier=self._tokens, seen=self._seen(), received=received)
            except BaseException:
                slots.release()
                raise
            handed_off = True

            def ended(_done) -> None:
                if progress is not None:
                    progress.ended = time.monotonic()
                try:
                    slots.release()
                finally:
                    self._check_ended(owner)

            future.add_done_callback(ended)
            try:
                try:
                    outcome = await asyncio.wrap_future(future)
                finally:
                    if future.done():
                        self._metric("check_time", time.monotonic() - check_started)
            except Exception:
                logger.error("episode group of %s: the transcript check raised; refused", hotkey[:12],
                             exc_info=True)
                # A raising check is a validator bug (the renderer, the task source): no verdict.
                return self._refuse(prepared, RejectReason.BAD_SCHEMA, "episode_transcript",
                                    {"check": "raised"}, verdict=False)
        finally:
            if not handed_off:
                self._check_ended(owner)
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
            # A validator bug after the check: ours, never the miner's refusal.
            self._report(hotkey, sha, claim.session_ids, accepted=False, stage="admission_worker")
            raise
        if prepared.reject_reason is not None:
            await asyncio.shield(self._spawn_release(claim.session_ids))
            self._report(hotkey, sha, claim.session_ids, accepted=False, stage=prepared.reject_stage)
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

    async def settle(self, claim: EpisodeClaim, *, accepted: bool, stage: str | None = None) -> None:
        """After the batcher's answer: accepted sessions end ``submitted``; others are released (the
        miner may submit them again). Shielded: a cancellation of the caller never leaves an accepted
        group's sessions unpaid-for in the book (a stale claim would let them be claimed again). ``stage``: the
        server's refusal stage of a group not accepted (the yield accounting reads it)."""
        task = asyncio.get_running_loop().create_task(self._settle(claim, accepted=accepted, stage=stage))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        await asyncio.shield(task)

    async def _settle(self, claim: EpisodeClaim, *, accepted: bool, stage: str | None = None) -> None:
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
        self._report(claim.hotkey, claim.precommit_sha256, claim.session_ids, accepted=accepted,
                     stage=None if accepted else stage)


__all__ = ["ADMISSION_DEADLINE_S", "CLAIM_HOLD_WALLS", "CLAIM_TTL_MARGIN", "CheckProgress", "DEFAULT_MAX_CHECKS_IN_FLIGHT",
           "DEFAULT_MAX_CHECKS_PER_MINUTE", "MAX_CHECKS_RUNNING", "RETRYABLE_STAGES", "EpisodeClaim",
           "EpisodeGroupIntake"]
