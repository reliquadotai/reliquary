"""Replayed episodes on the async RL path: deterministic tool worlds (registry ``interaction_mode="episode"``).

Two kinds of multi-turn env reach a lease. A signed-episode env (SWE, TMax: a real container) has its tool
observations signed by the sandbox (``rl_episode.LeaseEpisodeChecker``). A replayed env (EnvScaler, telecom: a
pinned Python world) has none: the validator recomputes every observation itself, by replaying the miner's model
turns through the environment, as a corpus suite job does at intake (``corpus.suite_episode``, live for telecom).
A miner cannot forge an observation: its tokens are compared with the replay's, token for token.

What a lease of a replayed env carries, per rollout (``protocol/rl_submit``): its completion (everything after
the validator's prompt render: model turns, observations, renderer markup), its model spans, its stop and an
empty ``transcript`` (no machine signs anything). The checks, in order:

* the generic turn checks of a suite trajectory (``corpus.checks``): spans in order, one turn per span (no stop
  or turn markup inside, spelled out or not), the per-turn and episode budgets, each turn ending on its
  terminator or exactly at the cap the engine applies. No TOPLOC rules (the lease path verifies model tokens by
  the forced draw, in the trainer's forward and the audits, as for a signed episode);
* the replay (``lease_replay_outcome``), in a CPU-only process pool under a wall deadline and a memory bound
  (``LeaseSuiteReplayer``, the corpus intake's workers): the tokens must be the replay's up to the end of the
  last span, the stop the replay's;
* the reward is the replay's (validator authoritative).

Stops (as a signed episode's): ``agent_completed`` (the world ended it) and ``max_turns`` are graded;
``context_length`` (the next turn has no room under the episode's token budget) is a cut, scored 0 and trained,
flagged truncated. A cut is checked, not claimed: the replay of exactly the submitted turns must leave the world
running and the context at the budget, so a miner cannot cut an episode early to reshape its group.

Any refusal rejects the whole group (no observation, quota halved), as for a signed episode. A replay the pool
cannot answer (a broken pool, a queue past its wait, a replay past its deadline on the worker's own clock) is the
validator's outage (``GraderInfrastructureError``: retried, the lease stays open), never a refusal.

The env's episode policy is its profile's ``EpisodeProfile`` (``suite_lease_policy``): renderer, turns, per-turn
and episode token caps, observation bound. Every replayed env of one miner engine shares the engine's caps.
"""
from __future__ import annotations

import concurrent.futures
import functools
import hashlib
import logging
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any

from reliquary.environment.grader_client import GraderInfrastructureError

logger = logging.getLogger(__name__)

CONTEXT_LENGTH = "context_length"
GRADED_STOPS = ("agent_completed", "max_turns")
LEASE_STOPS = (*GRADED_STOPS, CONTEXT_LENGTH)
# A world starts an interpreter per episode and runs up to its turn limit of tool calls: seconds, not telecom's
# ~30 ms. This bounds a pathological replay on the worker's own clock.
LEASE_REPLAY_DEADLINE_S = float(os.environ.get("RELIQUARY_RL_SUITE_REPLAY_DEADLINE_S", "120"))
LEASE_REPLAY_WAIT_S = float(os.environ.get("RELIQUARY_RL_SUITE_REPLAY_WAIT_S", "600"))
LEASE_REPLAY_WORKERS = int(os.environ.get("RELIQUARY_RL_SUITE_REPLAY_WORKERS", "4"))


def is_replayed_env(name: str) -> bool:
    """A registry env whose episodes the validator replays (``interaction_mode="episode"``)."""
    from reliquary.environment.registry import find_environment_spec

    spec = find_environment_spec(name)
    return spec is not None and spec.interaction_mode == "episode"


@dataclass(frozen=True)
class SuiteLeasePolicy:
    """A replayed env's episode policy on the lease path. ``tools`` and ``harness`` are empty: the world's tools
    are its tasks', rendered by the env's own renderer (fields kept so callers reading a signed policy's do not
    branch)."""
    environment: str
    renderer_id: str
    max_turns: int
    max_tokens_per_turn: int
    max_episode_tokens: int
    max_observation_bytes: int
    tools: tuple = ()
    harness: tuple = ()


def suite_lease_policy(name: str, *, profile=None, environment=None) -> SuiteLeasePolicy:
    """The policy of replayed env ``name`` from the protocol profile's ``EpisodeProfile`` (ValueError when the env
    is not a replayed env of the profile). ``max_turns`` is at most the environment's own (``environment``: an
    instance, when one is built already)."""
    from reliquary.environment.registry import get_environment_spec

    if profile is None:
        from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE as profile
    spec = get_environment_spec(name)
    if spec.interaction_mode != "episode":
        raise ValueError(f"{name}: not a replayed episode environment ({spec.interaction_mode})")
    entry = profile.environments.get(name)
    episode = getattr(entry, "episode", None)
    if episode is None:
        raise ValueError(f"{name}: the profile {profile.profile_id} declares no episode policy for it")
    if episode.renderer_id != spec.renderer_id:
        raise ValueError(f"{name}: the profile renders with {episode.renderer_id!r}, the env with "
                         f"{spec.renderer_id!r}")
    turns = int(episode.max_turns)
    if environment is not None:
        turns = min(turns, int(environment.max_turns))
    return SuiteLeasePolicy(environment=name, renderer_id=episode.renderer_id, max_turns=turns,
                            max_tokens_per_turn=int(episode.max_action_tokens),
                            max_episode_tokens=int(episode.max_episode_tokens),
                            max_observation_bytes=int(episode.max_observation_bytes))


def lease_episode_seed(lease) -> int:
    """The seed every episode of a lease is played with: the first 8 bytes, big-endian, of
    sha256(``rl-lease-episode:{nonce}:{prompt_idx}``). Never sent: both sides derive it from the lease (the 16
    rollouts of a group start from one world; they differ by what the policy draws)."""
    digest = hashlib.sha256(f"rl-lease-episode:{lease.nonce}:{int(lease.prompt_idx)}".encode()).hexdigest()
    return int(digest[:16], 16)


# --- the replay ---------------------------------------------------------------------------------------------------
def turn_refusal(policy: SuiteLeasePolicy, *, prompt_len: int, tokens: Sequence[int],
                 spans: Sequence[tuple[int, int]], encode, decode) -> tuple[str, dict] | None:
    """The suite trajectory's generic checks on a lease (no TOPLOC rules): spans, one turn per span, budgets,
    termination. ``(reason, detail)`` or None. The miner runs the same before submitting."""
    from reliquary.corpus.checks import check_turn_budget, check_turn_spans, check_turn_termination
    from reliquary.corpus.suite_episode import suite_span_refusal, turn_ids

    spans = [(int(s), int(e)) for s, e in spans]
    result = check_turn_spans(spans, len(tokens), policy.max_turns)
    if not result.ok:
        return result.reason or "bad_turns", dict(result.detail)
    inside = suite_span_refusal(tokens, spans, encode=encode, decode=decode)
    if inside is not None:
        return inside[0], dict(inside[1])
    terminator, stops = turn_ids(encode)
    for result in (
        check_turn_budget(spans, prompt_len=prompt_len, length=len(tokens),
                          max_tokens_per_turn=policy.max_tokens_per_turn,
                          max_total_tokens=policy.max_episode_tokens),
        check_turn_termination(tokens, spans, prompt_len=prompt_len, terminator_id=terminator, stop_ids=stops,
                               max_tokens_per_turn=policy.max_tokens_per_turn,
                               max_total_tokens=policy.max_episode_tokens),
    ):
        if not result.ok:
            return result.reason or "bad_turns", dict(result.detail)
    return None


def lease_replay_outcome(environment, *, renderer_id: str, task_index: int, seed: int, prompt_ids, tokens, spans,
                         stop: str, encode, decode, max_turns: int, max_episode_tokens: int,
                         max_observation_bytes: int = 64 * 1024):
    """Replay one lease episode and compare (``corpus.suite_episode.replay_outcome`` for a graded stop). A
    ``context_length`` cut replays exactly its turns: the world must still be running after the last one and the
    context (with the observation that answered it) must have reached ``max_episode_tokens``; its reward is 0. A
    deadline (``TimeoutError``) is the caller's to handle."""
    from reliquary.corpus.suite_episode import (
        SuiteReplayOutcome, compare_replay, replay_outcome, replay_suite,
    )

    if stop != CONTEXT_LENGTH:
        if stop not in GRADED_STOPS:
            return SuiteReplayOutcome(("bad_stop", {"stop": stop, "allowed": list(LEASE_STOPS)}))
        return replay_outcome(environment, renderer_id=renderer_id, task_index=task_index, seed=seed,
                              prompt_ids=prompt_ids, tokens=tokens, spans=spans, stop=stop, encode=encode,
                              decode=decode, max_turns=max_turns)
    turns = len(spans)
    if not 0 < turns <= int(max_turns):
        return SuiteReplayOutcome(("bad_turns", {"why": f"{turns} turns, at most {max_turns}"}))
    try:
        trace = replay_suite(environment, renderer_id=renderer_id, task_index=task_index, seed=seed,
                             prompt_ids=prompt_ids, tokens=tokens, spans=spans, encode=encode, decode=decode,
                             max_turns=turns, max_observation_bytes=max_observation_bytes)
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001 - whatever the environment raises refuses it
        return SuiteReplayOutcome(("bad_observation", {
            "why": f"replay failed: {type(exc).__name__}: {str(exc)[:200]}"}))
    # compare_replay checks tokens, spans and markup; its stop rule is the graded one: the cut is checked here.
    refusal = compare_replay(trace, prompt_ids=prompt_ids, tokens=tokens, spans=spans, stop="max_turns")
    if refusal is not None and refusal[0] != "bad_stop":
        return SuiteReplayOutcome(refusal)
    if trace.termination_reason != "turn_limit":
        return SuiteReplayOutcome(("bad_stop", {"stop": stop, "termination": trace.termination_reason,
                                                "why": "the world ended the episode: not a cut"}))
    if len(trace.tokens) < int(max_episode_tokens):
        return SuiteReplayOutcome(("bad_stop", {"stop": stop, "tokens": len(trace.tokens),
                                                "max_episode_tokens": int(max_episode_tokens),
                                                "why": "the next turn had room: not a cut"}))
    return SuiteReplayOutcome(None, task_id=str(trace.task_id), reward=0.0, success=False,
                              termination_reason=CONTEXT_LENGTH, trace_digest=str(trace.trace_digest))


def lease_replay_in_worker(env_name: str, deadline_seconds: float, **kwargs):
    """Top-level (picklable): one lease replay in a corpus replay worker (``suite_intake``: CPU only, memory
    bounded, the checkpoint's tokenizer). Past the deadline the outcome is the deadline's."""
    from reliquary.corpus.suite_episode import SuiteReplayOutcome, TokenizerCodec
    from reliquary.environment.registry import get_environment_spec
    from reliquary.validator import suite_intake

    if suite_intake._WORKER_TOKENIZER is None:
        raise RuntimeError("suite replay worker has no tokenizer")
    codec = TokenizerCodec(suite_intake._WORKER_TOKENIZER)
    outcome = None
    try:
        with suite_intake._deadline(deadline_seconds):
            outcome = lease_replay_outcome(get_environment_spec(env_name).create(), encode=codec.encode,
                                           decode=codec.decode, **kwargs)
    except suite_intake._ReplayTimeout:
        outcome = None
    if suite_intake._DEADLINE_HIT or outcome is None:
        return SuiteReplayOutcome(suite_intake.DEADLINE_REFUSAL, deadline=True)
    return outcome


class LeaseSuiteReplayer:
    """The corpus intake's replay pool (``suite_intake.PoolSuiteReplayer``: spawned CPU-only workers holding the
    tokenizer, RLIMIT_AS, a broken pool rebuilt once) running ``lease_replay_in_worker``. ``replay`` raises
    ``GraderInfrastructureError`` when no answer comes (the validator's outage: the miner resends)."""

    def __init__(self, tokenizer_json: str, *, workers: int = LEASE_REPLAY_WORKERS,
                 deadline_seconds: float = LEASE_REPLAY_DEADLINE_S, wait_seconds: float = LEASE_REPLAY_WAIT_S):
        from reliquary.validator.suite_intake import PoolSuiteReplayer

        self.pool = PoolSuiteReplayer(tokenizer_json, workers=workers, deadline_seconds=deadline_seconds,
                                      wait_seconds=wait_seconds)
        self.deadline_seconds, self.wait_seconds = float(deadline_seconds), float(wait_seconds)

    @property
    def workers(self) -> int:
        return self.pool.workers

    @classmethod
    def for_tokenizer(cls, tokenizer, **kwargs) -> "LeaseSuiteReplayer":
        serializer = getattr(getattr(tokenizer, "backend_tokenizer", None), "to_str", None)
        if not callable(serializer):
            raise RuntimeError("a replayed episode env needs a fast tokenizer to replay with")
        return cls(str(serializer()), **kwargs)

    def replay(self, env_name: str, **kwargs):
        """``PoolSuiteReplayer.replay`` with the lease replay; blocks the calling thread."""
        pool = self.pool
        for _ in range(2):
            executor = pool._executor()
            try:
                future = executor.submit(lease_replay_in_worker, env_name, self.deadline_seconds, **kwargs)
                outcome = pool._settled(executor, future.result(timeout=self.wait_seconds))
            except BrokenProcessPool as exc:
                pool._drop(executor, f"{type(exc).__name__}: {exc}: a worker died")
                continue
            except FutureTimeout:
                raise GraderInfrastructureError(str(pool._timed_out(executor, future))) from None
            if outcome.deadline:
                raise GraderInfrastructureError("an episode replay ran past its deadline")
            return outcome
        raise GraderInfrastructureError("the episode replay pool broke twice in a row")

    def close(self) -> None:
        self.pool.close()


class InlineLeaseReplayer:
    """In this process, on the caller's environment and codec: tests and the miner's precheck."""

    def __init__(self, environment, *, encode, decode) -> None:
        self.environment, self._encode, self._decode = environment, encode, decode
        self.workers = 1

    def replay(self, env_name: str, **kwargs):
        del env_name
        return lease_replay_outcome(self.environment, encode=self._encode, decode=self._decode, **kwargs)

    def close(self) -> None:
        pass


# --- the checker --------------------------------------------------------------------------------------------------
class LeaseSuiteChecker:
    """One per replayed env, the grader's ``episodes[env]`` (``rl_submit.GroupGrader.grade_episodes``):
    ``policy`` (``SuiteLeasePolicy``), ``environment`` (an instance, for the task prompts), ``codec`` (the
    checkpoint tokenizer's ``TokenizerCodec``: the replay workers' coding) and ``replayer`` (``LeaseSuiteReplayer``;
    ``InlineLeaseReplayer`` in tests). The 16 replays of a lease run side by side, up to the pool's workers."""

    def __init__(self, *, policy: SuiteLeasePolicy, environment, codec, replayer) -> None:
        from reliquary.environment.agentic.renderers import renderer_for

        self.policy, self.environment, self.codec, self.replayer = policy, environment, codec, replayer
        self.renderer = renderer_for(policy.renderer_id, codec.encode)
        self._prompt = functools.lru_cache(maxsize=4096)(self._render)

    def _render(self, prompt_idx: int) -> tuple[int, ...]:
        return tuple(int(t) for t in self.renderer.encode_initial(self.environment.get_task(int(prompt_idx))))

    def prompt_ids(self, prompt_idx: int) -> tuple[int, ...]:
        return self._prompt(int(prompt_idx))

    def grade(self, lease, rollouts: Sequence[Any], *, received: float):
        """``rl_submit.Graded`` of a lease's replayed episodes (``Rollout`` with ``episode``)."""
        del received
        from reliquary.validator.rl_submit import Graded

        policy = self.policy
        prompt = self.prompt_ids(lease.prompt_idx)
        for index, rollout in enumerate(rollouts):
            episode = rollout.episode
            if episode is None or rollout.force_span is not None or dict(episode.get("transcript") or {}):
                return Graded(prompt, reject="episode_schema")      # a replayed episode carries no transcript
            if episode["stop"] not in LEASE_STOPS:
                return Graded(prompt, reject="bad_stop")
            refusal = turn_refusal(policy, prompt_len=len(prompt), tokens=list(rollout.tokens), spans=rollout.spans,
                                   encode=self.codec.encode, decode=self.codec.decode)
            if refusal is not None:
                logger.info("replayed lease %s (%s) rollout %d refused: %s %s", lease.lease_id[:12],
                            policy.environment, index, refusal[0], refusal[1])
                return Graded(prompt, reject=refusal[0])
        seed = lease_episode_seed(lease)

        def one(rollout):
            return self.replayer.replay(
                policy.environment, renderer_id=policy.renderer_id, task_index=int(lease.prompt_idx), seed=seed,
                prompt_ids=list(prompt), tokens=[int(t) for t in rollout.tokens],
                spans=[(int(s), int(e)) for s, e in rollout.spans], stop=str(rollout.episode["stop"]),
                max_turns=policy.max_turns, max_episode_tokens=policy.max_episode_tokens,
                max_observation_bytes=policy.max_observation_bytes)

        workers = max(1, min(len(rollouts), int(getattr(self.replayer, "workers", 1))))
        with concurrent.futures.ThreadPoolExecutor(workers, thread_name_prefix="rl-suite-replay") as threads:
            outcomes = list(threads.map(one, rollouts))
        rewards, truncated = [], []
        for index, outcome in enumerate(outcomes):
            if outcome.refusal is not None:
                reason, detail = outcome.refusal
                logger.info("replayed lease %s (%s) rollout %d refused: %s %s", lease.lease_id[:12],
                            policy.environment, index, reason, detail)
                return Graded(prompt, reject=reason)
            cut = outcome.termination_reason == CONTEXT_LENGTH
            rewards.append(0.0 if cut else float(outcome.reward))
            truncated.append(cut)
        return Graded(prompt, tuple(rewards), tuple(truncated), spans=tuple(tuple(r.spans) for r in rollouts))


def build_suite_checkers(names: Sequence[str], tokenizer, *, profile=None, replayer=None,
                         environments: Mapping[str, Any] | None = None) -> dict[str, LeaseSuiteChecker]:
    """The checkers of the run's replayed envs, over one replay pool (``replayer``: tests). Blocking: each env is
    built (an EnvScaler corpus is read from its data directory)."""
    from reliquary.corpus.suite_episode import TokenizerCodec
    from reliquary.environment import load_episode_environment

    if not names:
        return {}
    codec = TokenizerCodec(tokenizer)
    replayer = replayer or LeaseSuiteReplayer.for_tokenizer(tokenizer)
    out = {}
    for name in names:
        environment = (environments or {}).get(name) or load_episode_environment(name)
        policy = suite_lease_policy(name, profile=profile, environment=environment)
        out[name] = LeaseSuiteChecker(policy=policy, environment=environment, codec=codec, replayer=replayer)
        logger.info("replayed episode env %s: %d turns, %d tokens a turn, %d an episode", name, policy.max_turns,
                    policy.max_tokens_per_turn, policy.max_episode_tokens)
    return out


__all__ = [
    "CONTEXT_LENGTH", "GRADED_STOPS", "InlineLeaseReplayer", "LEASE_STOPS", "LeaseSuiteChecker",
    "LeaseSuiteReplayer", "SuiteLeasePolicy", "build_suite_checkers", "is_replayed_env", "lease_episode_seed",
    "lease_replay_in_worker", "lease_replay_outcome", "suite_lease_policy", "turn_refusal",
]
