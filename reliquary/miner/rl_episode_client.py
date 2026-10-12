"""The miner side of the RL validator's episode routes: the signed precommit, the RL
session open (engagement ``rl_precommit``) and a signed-episode runner for ONE precommit, whose episode
index is the SEED (every seed plays the same task prompt). Episodes that end graded are closed ``final``
(``closed_graded``: they keep their seed), then either submitted in the group or withdrawn by their
``release`` (not waste for a submitted group, spec decision 6).

The runner's ``EpisodeResult.stop`` is the harness's raw stop condition: before an episode is committed,
``episode_commit.episode_stop`` maps it to the stop the validator admits (verifiers labels a turn cut at
the episode's token cap ``agent_completed``; the validator wants ``context_length``)."""
from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

from reliquary.miner.signed_episode import (
    GRADING_GRACE_S, MAX_UNAVAILABLE, OPEN_WINDOW_S, THROTTLED, VALIDATOR_THROTTLES, HttpSandboxSessions,
    SignedSweEpisodeRunner, _claims_of, _compact, _NoKeys, _Unconfirmed,
)
from reliquary.protocol.sandbox_session import SessionRefused, sandbox_open_path
from reliquary.protocol.signatures import build_episode_precommit_binding, build_sandbox_open_binding

logger = logging.getLogger(__name__)

RL_PREFIX = "/rl"
RL_THROTTLES = VALIDATOR_THROTTLES | {"open_busy", "environment_not_served", "sandbox_saturated", "open_queue_full",
                                      "environment_unserved"}
"""The RL routes' named 503s: the validator is serving (an unserved env may be loading, the sandbox fleet may be
saturated); the seed is reopened, and they never count as "not serving"."""
RL_FINAL_REFUSALS = frozenset({"precommit_unknown", "precommit_stale", "seed_out_of_pool", "engagement_taken",
                               "session_too_long", "task_unplaceable"})
"""The RL engagement book's own 4xx, and an open no machine could ever take: the seed is the miner's to change,
never reopened."""
RL_TASK_FINAL_REFUSALS = frozenset({"task_unplaceable", "precommit_unknown", "precommit_stale", "session_too_long"})
"""Final for every seed of the precommit (they all play its one task, under one precommit): after one, the runner
sends no other open of that precommit, so it costs one refused open, not one per seed (refused opens count toward
the hotkey's refused-open rate)."""
RL_START_REFUSALS = frozenset({"quota_env"})
"""Refusals that end a group which has not started (no seed granted): the hotkey holds its share of that env, so the
group's slot goes to another env this window rather than wait (``EpisodeGroupMiner.mine_task`` raises it)."""

RL_OVERGENERATION_REFUSALS = frozenset({"quota_live", "quota_pool"})
"""The quota's answer to a seed beyond M of a group that holds its M sessions (over-generation): no room for one more
in the hotkey's allocation, its pooled entity or the fleet (a group under M is never refused them). Once M episodes of
the precommit ended graded and usable, one ends the group's opens: it is played from the seeds it has. Waiting instead
would hold its graded sessions for room that only its own submission frees: every group at M would wait on the others
until its window's end. Before that the seed is reopened as any throttle: a session still playing may fail, and the
quota then admits the group's next seed."""

RL_OPENS_SIDE_BY_SIDE = 4
"""Opens of one started group in flight at once (each may wait in the validator's open queue): enough to keep the
group's seeds queued under saturation, few enough not to burst the validator's open path."""
RL_OPENS_IN_FLIGHT = 16
"""Opens of the whole miner process in flight at once, all groups together. They run on their own executor (of this
many threads), so an open waiting in the validator's queue never holds a thread that closes, precommits, proofs or
transcript checks need."""

_OPEN_EXECUTOR: ThreadPoolExecutor | None = None
_OPEN_SLOTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()      # event loop -> its opens' semaphore
_EXECUTOR_LOCK = threading.Lock()


def _open_executor() -> ThreadPoolExecutor:
    global _OPEN_EXECUTOR
    with _EXECUTOR_LOCK:
        if _OPEN_EXECUTOR is None:
            _OPEN_EXECUTOR = ThreadPoolExecutor(max_workers=RL_OPENS_IN_FLIGHT, thread_name_prefix="rl-open")
        return _OPEN_EXECUTOR


def _open_slots() -> asyncio.Semaphore:
    """The process's open slots (``RL_OPENS_IN_FLIGHT``), one semaphore per event loop."""
    loop = asyncio.get_running_loop()
    slots = _OPEN_SLOTS.get(loop)
    if slots is None:
        slots = _OPEN_SLOTS[loop] = asyncio.Semaphore(RL_OPENS_IN_FLIGHT)
    return slots

_FANNED: contextvars.ContextVar[bool] = contextvars.ContextVar("rl_open_fanned", default=False)
"""Set (per seed task) while an open is sent side by side with the precommit's other opens."""


def _not_serving(refused: SessionRefused) -> bool:
    """A refusal that says the validator is not serving (no answer, 404, an unnamed 5xx): the runner's consecutive
    count decides when to stop (``validator_unavailable``); until then the seed is reopened."""
    return (refused.status is None or refused.status == 404
            or (refused.status >= 500 and refused.reason not in RL_THROTTLES))


def check_engine_caps(policy, *, max_total_tokens: int, max_tokens_per_turn: int, max_model_len: int) -> None:
    """At startup: the generate engine's caps and the harness's context length are the contract's episode
    policy, or ValueError. An engine cap above the policy plays turns the validator refuses; one below it
    cuts episodes the validator reads as a miner's early stop (its limit rules use the policy's caps)."""
    wrong = {name: (int(got), int(want)) for name, got, want in (
        ("max_total_tokens", max_total_tokens, policy.max_episode_tokens),
        ("max_tokens_per_turn", max_tokens_per_turn, policy.max_tokens_per_turn),
        ("max_model_len", max_model_len, policy.max_episode_tokens),
    ) if isinstance(got, bool) or int(got) != int(want)}
    if wrong:
        raise ValueError("the episode engine's caps are not the contract's episode policy: "
                         + ", ".join(f"{name} {got} != {want}" for name, (got, want) in wrong.items()))


def signed_precommit_body(*, precommit, sign_binding: Callable[[bytes], str], now: float,
                          validator_hotkey: str, path: str) -> dict:
    """The body of ``POST {prefix}/episodes/precommit``, signed for this validator and route."""
    body = {"miner_hotkey": precommit.hotkey, "at": int(now), "precommit": precommit.to_dict(), "signature": ""}
    body["signature"] = sign_binding(build_episode_precommit_binding(
        body["precommit"], at=body["at"], validator_hotkey=validator_hotkey, path=path))
    return body


def signed_rl_open_request(*, hotkey: str, precommit_sha256: str, seed_index: int,
                           sign_binding: Callable[[bytes], str], now: float, request_id: str,
                           validator_hotkey: str, path: str) -> dict:
    body = {"miner_hotkey": hotkey, "request_id": request_id, "at": int(now),
            "engagement": {"kind": "rl_precommit",
                           "precommit": {"precommit_sha256": precommit_sha256, "seed_index": int(seed_index)}},
            "signature": ""}
    body["signature"] = sign_binding(build_sandbox_open_binding(body, validator_hotkey=validator_hotkey, path=path))
    return body


class HttpRlEpisodes(HttpSandboxSessions):
    """The RL validator's session and precommit routes (audience: its hotkey and the ``/rl`` paths)."""

    def __init__(self, http, *, validator_hotkey: str, prefix: str = RL_PREFIX) -> None:
        super().__init__(http, validator_hotkey=validator_hotkey, prefix=prefix)

    def precommit_path(self) -> str:
        from reliquary.sandbox.rl_routes import episode_precommit_path

        return episode_precommit_path(self.prefix)

    def precommit(self, body: dict) -> dict:
        """``{"precommit_sha256", "created"}``; SessionRefused (reason, status, Retry-After) otherwise."""
        return self._post(self.precommit_path(), body)


def rl_transcript_refusal(transcript: Any, *, policy, precommit, seed_index: int,
                          now: float) -> tuple[str, dict] | None:
    """The episode admission's transcript checks a miner can run (all but the machine and validator
    signatures and the paid-session snapshot): the transcript size, ``verify_transcript`` with the RL
    binding and a graded final, the grading deadline, record 0's tools version, tools and env package.
    ``(reason, detail)`` (the admission's stage names) or None."""
    from reliquary_sandbox.attest import Expected, Reason, verify_transcript
    from reliquary_sandbox.observation import TOOLS_VERSIONS

    from reliquary.protocol.corpus_submission import MAX_TRANSCRIPT_BYTES
    from reliquary.protocol.service_episode import rl_engagement

    if not isinstance(transcript, Mapping):
        return "malformed_submission", {"transcript": None}
    size = len(_compact(transcript))
    if size > MAX_TRANSCRIPT_BYTES:
        return "trajectory_too_large", {"transcript_bytes": size}
    expected = Expected(hotkey=precommit.hotkey,
                        engagement=rl_engagement(precommit.window, precommit.sha256, int(seed_index)),
                        env=policy.sandbox_env, split=policy.split, index=precommit.task_index,
                        checkpoint=precommit.checkpoint, require_graded=True)
    result = verify_transcript(transcript, _NoKeys(), _claims_of, expected)
    reasons = [reason.value for reason in result.reasons if reason is not Reason.UNKNOWN_KEY]
    if reasons:
        return "episode_transcript", {"reasons": reasons}
    deadline = result.claims.expires_at + GRADING_GRACE_S
    if now > deadline:
        return "episode_deadline", {"now": int(now), "deadline": deadline}
    opened = result.open
    if (opened.tools_version not in TOOLS_VERSIONS or tuple(opened.tools) != tuple(policy.tools)
            or opened.env_package != policy.env_package):
        return "episode_record0", {"tools_version": opened.tools_version, "tools": list(opened.tools),
                                   "env_package": opened.env_package}
    return None


class _TaskPrompt:
    """Every seed of a precommit plays the task's one prompt."""

    def __init__(self, text: str) -> None:
        self._text = text

    def prompt(self, index: int) -> str:
        return self._text


def _policy_job(policy, precommit) -> SimpleNamespace:
    """The fields of a corpus job the signed runner reads, from the contract's episode policy."""
    sandbox = SimpleNamespace(env=policy.sandbox_env, env_package=policy.env_package, tools=policy.tools,
                              budgets=SimpleNamespace(**policy.budgets_dict()))
    episode = SimpleNamespace(max_turns=policy.max_turns, max_tokens_per_turn=policy.max_tokens_per_turn,
                              sandbox=sandbox)
    return SimpleNamespace(job_id=f"rl-{precommit.sha256[:16]}", checkpoint_sha256=precommit.checkpoint,
                           episode=episode)


class RlSignedEpisodeRunner(SignedSweEpisodeRunner):
    """``SignedSweEpisodeRunner`` for one precommit: ``run(seed_index, on_session)`` opens that seed's
    session (engagement ``rl_precommit``), plays the task prompt through the harness and returns the
    episode. Its graded state is not a diff (``requires_text_state`` False); its transcript is checked
    with the RL binding before it is kept.

    Opens. Until a seed of the precommit is granted, its opens go one at a time (a probe): the validator's yield
    quota counts a whole group of demand from one request, so a group that cannot start yet costs one refused open
    per ``Retry-After``, never one per seed (refused opens count toward the hotkey's refused-open rate). Once a seed
    is granted (the group started: the quota never cuts it), the other seeds are sent side by side
    (``RL_OPENS_SIDE_BY_SIDE`` at once; ``RL_OPENS_IN_FLIGHT`` for the whole process, on their own executor), each
    waiting in the validator's open queue for its own turn. Any throttle makes the next open a probe again: the
    decision to send side by side is taken under the runner's lock, after its gate (the last ``Retry-After``), and a
    grant only re-opens side-by-side sending when no throttle came since its own send started.

    A refusal that is not the seed's to change is retried for the SAME seed until the window ends: a throttle (429,
    503, a named throttle: the yield quota's ``quota_*``, the open queue's, an unserved env, a stale directory) after
    its ``Retry-After``; a "not serving" answer (404, an unnamed 5xx) and a transport failure (the validator
    unreachable: the SAME request id is resent, so a grant it got is the one returned) after the back-off, until the
    consecutive count stops the runner (only a probe's answer counts: side-by-side opens fail together). So a group
    whose first seeds opened is never given up for a refusal of ours. A refusal about the precommit itself
    (``RL_TASK_FINAL_REFUSALS``) ends every seed of it; ``quota_env`` before the group started ends the group, its
    slot going to another env."""

    requires_text_state = False
    validator_throttles = RL_THROTTLES
    forwards_open_refusal = True          # the RL validators read the gateway's signed receipt

    def __init__(self, *, policy, precommit, prompt: str, hotkey: str, sign_binding, sessions, model_name: str,
                 renderer_model_dir: str, generate_url: str, sampling, harness_env: dict | None = None,
                 runner_factory=None, clock: Callable[[], float] = time.time, max_live: int | None = None,
                 open_until: float | None = None, **kwargs) -> None:
        if precommit.hotkey != hotkey:
            raise ValueError("the precommit names another hotkey than the runner's")
        if precommit.environment != policy.environment:
            raise ValueError("the precommit names another environment than the policy's")
        super().__init__(job=_policy_job(policy, precommit), hotkey=hotkey, sign_binding=sign_binding,
                         sessions=sessions, model_name=model_name, renderer_model_dir=renderer_model_dir,
                         generate_url=generate_url, sampling=sampling, harness_env=harness_env,
                         source=_TaskPrompt(prompt), runner_factory=runner_factory, clock=clock,
                         max_live=max_live or policy.pool_seeds, **kwargs)
        self._policy, self._precommit = policy, precommit
        self._open_until = open_until
        self._default_until: float | None = None
        self._fanning = False                 # a seed was granted and no throttle since: opens go side by side
        self._throttles = 0                   # throttles so far (a grant re-opens side by side only if none since)
        self._started = False                 # a seed of the precommit was granted
        self._usable = 0                      # episodes of the precommit that ended graded and usable
        self._side_by_side = asyncio.Semaphore(RL_OPENS_SIDE_BY_SIDE)
        self._task_refused: SessionRefused | None = None
        self.start_refused: SessionRefused | None = None    # the group ended before it started (RL_START_REFUSALS)

    def _until(self) -> float:
        """When opens stop: the window's end (``open_until``), by default OPEN_WINDOW_S after the first send."""
        if self._open_until is not None:
            return float(self._open_until)
        if self._default_until is None:
            self._default_until = self._clock() + OPEN_WINDOW_S
        return self._default_until

    async def _open(self, index: int, slot) -> dict:
        """The seed's open; a throttled one (429, 503, a named throttle) or a "not serving" one is sent again for
        the SAME seed once its wait has passed, while that is before ``_until()``. A final refusal (the seed's, the
        precommit's, a group that could not start) or the runner's stop (``validator_unavailable``) ends the
        seed."""
        until = self._until()
        while True:
            try:
                return await self._open_once(index, slot)
            except SessionRefused as refused:
                if refused.reason in RL_TASK_FINAL_REFUSALS and self._task_refused is None:
                    self._task_refused = refused
                if refused.reason in RL_START_REFUSALS and not self._started:
                    self.start_refused = self.start_refused or refused
                    if self._task_refused is None:
                        self._task_refused = refused
                    raise
                if refused.reason in RL_OVERGENERATION_REFUSALS and self._usable >= self._policy.pool_seeds // 2:
                    if self._task_refused is None:
                        logger.info("precommit %s holds %d usable episodes and the validator has no room for more "
                                    "(%s): played from the seeds it has", self._precommit.sha256[:12], self._usable,
                                    refused.reason)
                        self._task_refused = refused
                    raise
                reopen = (not slot.keep and refused.reason not in RL_FINAL_REFUSALS
                          and refused.reason != "validator_unavailable"
                          and (refused.status in THROTTLED or refused.reason in RL_THROTTLES
                               or _not_serving(refused)))
                wait = max(0.0, self._not_before - self._monotonic())
                if not reopen or self._clock() + wait >= until:
                    raise
                logger.info("seed %d open throttled (%s): reopened in %.0f s", index, refused.reason, wait)

    def _task_refusal(self) -> SessionRefused | None:
        """What ends every open of the precommit without sending it: a refusal about the precommit, a group that
        could not start, or the validator stopped serving (the consecutive count reached MAX_UNAVAILABLE)."""
        if self._unavailable >= MAX_UNAVAILABLE:
            return SessionRefused("validator_unavailable", {"consecutive": self._unavailable})
        refused = self._task_refused
        if refused is None:
            return None
        return SessionRefused(refused.reason, refused.detail, status=refused.status)

    def _granted(self, throttles: int) -> None:
        self._started = True
        if self._throttles == throttles:      # no throttle since this open was sent: side by side again
            self._fanning = True

    async def _open_once(self, index: int, slot) -> dict:
        """One open of the seed: behind the runner's lock, its gate and its lost opens' recovery, then sent alone
        (a probe, under the lock) or, once the group started and no throttle since, side by side."""
        refused = self._task_refusal()
        if refused is not None:
            raise refused
        lock = self._open_lock
        await lock.acquire()
        held = True
        try:
            while True:
                await self._wait_gate()
                await self._recover()
                refused = self._task_refusal()
                if refused is not None:
                    raise refused
                throttles = self._throttles
                if not self._fanning:
                    grant = await self._send_attempts(index, slot)
                    self._granted(throttles)
                    return grant
                await self._side_by_side.acquire()
                # Decided again after the wait for a slot: a throttle meanwhile (a pause, a probe) comes first.
                if (self._fanning and self._throttles == throttles
                        and self._not_before <= self._monotonic()):
                    break
                self._side_by_side.release()
            lock.release()
            held = False
        finally:
            if held:
                lock.release()
        fanned = _FANNED.set(True)
        try:
            grant = await self._send_attempts(index, slot)
            self._granted(throttles)
            return grant
        finally:
            _FANNED.reset(fanned)
            self._side_by_side.release()

    async def _settle(self, index: int, session_id: str, slot, result, final: dict, closing: str):
        settled = await super()._settle(index, session_id, slot, result, final, closing)
        if settled.ok:
            self._usable += 1
        return settled

    def _open_call(self, body: dict):
        return asyncio.get_running_loop().run_in_executor(_open_executor(), self._sessions.open, body)

    async def _send_attempts(self, index: int, slot) -> dict:
        """One signed open of ``index`` (one request id). A refusal moves the gate and is raised. A transport
        failure counts as "not serving" (``_note_refusal``) and the SAME request is sent again after the back-off
        (a grant it got is returned, never a second one), until the runner stops or the window ends: then the open
        is kept unconfirmed (its slot held, a grant it got is closed later) and the failure raised."""
        request_id = self._new_request_id()
        resent = False
        while True:
            try:
                async with _open_slots():
                    grant = await self._send_open(self._open_body(index, request_id))
            except SessionRefused as refused:
                if resent:
                    await self._close_reused(refused)
                raise self._note_refusal(refused) from None
            except Exception as exc:
                logger.warning("sandbox session open for seed %d failed (%s)", index, type(exc).__name__)
                refused = self._note_refusal(SessionRefused("validator_unreachable"))
                _FANNED.set(False)            # the same open, sent again alone: its next failure counts
                wait = max(0.0, self._not_before - self._monotonic())
                if refused.reason == "validator_unavailable" or self._clock() + wait >= self._until():
                    slot.keep = True
                    self._unconfirmed.append(_Unconfirmed(index, request_id, slot,
                                                          self._monotonic() + self.deadline(index)))
                    raise refused from None
                await self._wait_gate()
                resent = True
                continue
            self._backoff = self._unavailable = 0
            return grant

    def _note_refusal(self, refused: SessionRefused) -> SessionRefused:
        # A 409 is about this seed (taken, out of the pool, a stale precommit): the other seeds' opens
        # are not held behind it.
        if refused.status == 409 and not refused.retry_after:
            self._unavailable = 0
            return refused
        self._throttles += 1
        self._fanning = False                 # the next open is a probe
        if _FANNED.get():
            # Sent side by side: the other opens in flight fail with it, so it neither counts toward the stop nor
            # grows the back-off; the probe that follows does.
            self._hold_opens(refused.retry_after or 1.0)
            return refused
        return super()._note_refusal(refused)

    def _open_body(self, index: int, request_id: str) -> dict:
        return signed_rl_open_request(hotkey=self._hotkey, precommit_sha256=self._precommit.sha256,
                                      seed_index=index, sign_binding=self._sign, now=self._clock(),
                                      request_id=request_id, validator_hotkey=self._validator,
                                      path=sandbox_open_path(self._prefix))

    def _transcript_refusal(self, transcript, index: int, now: float):
        return rl_transcript_refusal(transcript, policy=self._policy, precommit=self._precommit,
                                     seed_index=index, now=now)


__all__ = [
    "RL_FINAL_REFUSALS",
    "RL_OPENS_IN_FLIGHT",
    "RL_OPENS_SIDE_BY_SIDE",
    "RL_OVERGENERATION_REFUSALS",
    "RL_PREFIX",
    "RL_START_REFUSALS",
    "RL_TASK_FINAL_REFUSALS",
    "RL_THROTTLES",
    "HttpRlEpisodes",
    "RlSignedEpisodeRunner",
    "check_engine_caps",
    "rl_transcript_refusal",
    "signed_precommit_body",
    "signed_rl_open_request",
]
