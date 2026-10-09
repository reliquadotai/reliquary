"""The miner side of the RL validator's episode routes (phase 2, plan 2C): the signed precommit, the RL
session open (engagement ``rl_precommit``) and a signed-episode runner for ONE precommit, whose episode
index is the SEED (every seed plays the same task prompt). Episodes that end graded are closed ``final``
(``closed_graded``: they keep their seed), then either submitted in the group or withdrawn by their
``release`` (not waste for a submitted group, spec decision 6).

The runner's ``EpisodeResult.stop`` is the harness's raw stop condition: before an episode is committed,
``episode_commit.episode_stop`` maps it to the stop the validator admits (verifiers labels a turn cut at
the episode's token cap ``agent_completed``; the validator wants ``context_length``)."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

from reliquary.miner.signed_episode import (
    GRADING_GRACE_S, OPEN_WINDOW_S, THROTTLED, VALIDATOR_THROTTLES, HttpSandboxSessions, SignedSweEpisodeRunner,
    _claims_of, _compact, _NoKeys,
)
from reliquary.protocol.sandbox_session import SessionRefused, sandbox_open_path
from reliquary.protocol.signatures import build_episode_precommit_binding, build_sandbox_open_binding

logger = logging.getLogger(__name__)

RL_PREFIX = "/rl"
RL_THROTTLES = VALIDATOR_THROTTLES | {"open_busy", "environment_not_served"}
"""The RL routes' named 503s: the validator is serving (an unserved env may be loading); the seed is reopened."""
RL_FINAL_REFUSALS = frozenset({"precommit_unknown", "precommit_stale", "seed_out_of_pool", "engagement_taken",
                               "session_too_long"})
"""The RL engagement book's own 4xx: the seed is the miner's to change, never reopened."""


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
    with the RL binding before it is kept."""

    requires_text_state = False
    validator_throttles = RL_THROTTLES

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

    async def _open(self, index: int, slot) -> dict:
        """The seed's open; a throttled one (429, 503, a named throttle) is sent again for the SAME seed once
        its Retry-After has passed, while that is before ``open_until`` (the window; by default
        OPEN_WINDOW_S from the first send). Any other refusal ends the seed."""
        until = self._open_until if self._open_until is not None else self._clock() + OPEN_WINDOW_S
        while True:
            try:
                return await super()._open(index, slot)
            except SessionRefused as refused:
                reopen = (not slot.keep and refused.reason not in RL_FINAL_REFUSALS
                          and refused.reason != "validator_unavailable"
                          and (refused.status in THROTTLED or refused.reason in RL_THROTTLES))
                wait = max(0.0, self._not_before - self._monotonic())
                if not reopen or self._clock() + wait >= until:
                    raise
                logger.info("seed %d open throttled (%s): reopened in %.0f s", index, refused.reason, wait)

    def _note_refusal(self, refused: SessionRefused) -> SessionRefused:
        # A 409 is about this seed (taken, out of the pool, a stale precommit): the other seeds' opens
        # are not held behind it.
        if refused.status == 409 and not refused.retry_after:
            self._unavailable = 0
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


__all__ = ["HttpRlEpisodes", "RL_FINAL_REFUSALS", "RL_PREFIX", "RL_THROTTLES", "RlSignedEpisodeRunner",
           "check_engine_caps", "rl_transcript_refusal", "signed_precommit_body", "signed_rl_open_request"]
