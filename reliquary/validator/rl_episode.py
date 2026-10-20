"""Episodes on the async RL path (rl-async-design.md §4.10; implementation map §1.15, §9).

Leases, mandatory submission, quota and lifecycle apply unchanged: a lease of an episode env is answered by its
16 episodes, every index, like any lease. What changes is the grading, done here with the window path's own checks
(``episode_admission``) on a lease binding instead of a precommit:

* each rollout carries its completion (model turns + tool observations), its model spans and its gateway-signed
  transcript (``protocol/rl_submit.py``);
* the transcript is verified with ``verified_transcript`` (signatures, chain, graded final, deadline, record 0)
  against the lease: hotkey, engagement ``rl-lease:{lease_id}:{index}`` (``rl_lease_engagement``), the env's
  sandbox env and split, the task (= the leased prompt) and checkpoint ``k<k>`` (``lease_checkpoint_tag``);
* the token level with ``episode_shape_refusal`` (the signed parse: every observation is the rendering of its
  signed call record, token for token; turn shape, budgets, termination, limit stops), after the prompt fidelity
  check (the prompt is the validator's own render), all in the lease's surface variant
  (``agentic_swe.lease_variant``, drawn here from the lease's nonce: a miner that played another variant has
  other tokens, and is refused);
* one session twice in the group is refused; the reward is the final record's (validator authoritative).

No TOPLOC proof per span (dropped with the windows): the episode's model tokens are verified like any rollout's,
by the forced draw (audits and the trainer's forward) over the model positions, keyed per model-token position
across turns (``protocol/rl_draw``). Sandbox time counts in the lease's latency, never in the match.

Any refusal rejects the whole group (no observation, quota halved: ``RlSubmissions``), as a bad termination does
for a single-turn group: an episode cannot be withheld, so the 16 draws stay the miner's only choice-free sample.
A stale machine directory raises ``GraderInfrastructureError`` (retry later, the lease stays open).
"""
from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Collection, Sequence
from typing import Any

from reliquary.environment.grader_client import GraderInfrastructureError

logger = logging.getLogger(__name__)


class LeaseEpisodeChecker:
    """One per episode env: its policy (``service_contract.EpisodeEnvPolicy`` fields: sandbox_env, split,
    env_package, tools, max_turns, max_tokens_per_turn, max_episode_tokens), the turn renderer of the policy's
    tokenizer, its task source (``prompt(index)``), ``directory(now)`` (the machine directory, None while stale),
    ``token_verifier`` (this validator's own session-token keys) and ``seen()`` (session ids already paid; the
    lease engagement already makes a transcript single-use, so empty by default)."""

    def __init__(self, *, policy, renderer, source, directory: Callable[[float], Any | None], token_verifier,
                 seen: Callable[[], Collection[str]] = frozenset, min_chunk_tokens: int | None = None,
                 renderer_of: Callable[[int], Any] | None = None) -> None:
        from reliquary.protocol.toploc import MIN_CHUNK_TOKENS

        self.policy, self.renderer, self.source = policy, renderer, source
        # The renderer of a surface variant other than 0 (a harness with axes): the lease's, computed here.
        self.renderer_of = renderer_of
        self._variants: dict[int, Any] = {0: renderer}
        self.directory, self.token_verifier, self.seen = directory, token_verifier, seen
        self.min_chunk_tokens = MIN_CHUNK_TOKENS if min_chunk_tokens is None else int(min_chunk_tokens)
        self._prompt = functools.lru_cache(maxsize=256)(self._render)

    def _render(self, prompt_idx: int, variant: int = 0) -> tuple[int, ...]:
        renderer = self.variant_renderer(variant)
        return tuple(int(t) for t in renderer.initial_ids(self.source.prompt(int(prompt_idx))))

    def prompt_ids(self, prompt_idx: int, variant: int = 0) -> tuple[int, ...]:
        return self._prompt(int(prompt_idx), int(variant))

    def variant_renderer(self, variant: int):
        if variant not in self._variants:
            if self.renderer_of is None:
                raise ValueError(f"no renderer for surface variant {variant}")
            self._variants[variant] = self.renderer_of(int(variant))
        return self._variants[variant]

    def lease_variant(self, lease) -> int:
        """The surface variant the lease's episodes must have played (the validator's own draw)."""
        from reliquary.environment.agentic_swe import harness_of, lease_variant, policy_harness_pin

        return lease_variant(harness_of(policy_harness_pin(self.policy)), lease)

    def grade(self, lease, rollouts: Sequence[Any], *, received: float):
        """``rl_submit.Graded`` of a lease's episodes (``Rollout`` with ``episode``)."""
        graded = self._grade(lease, rollouts, received=received)
        variant = self.lease_variant(lease)
        if variant or getattr(graded, "reject", None):
            logger.info("episode lease %s (%s, surface variant %d): %s", lease.lease_id[:12], self.policy.environment,
                        variant, f"rejected {graded.reject}" if getattr(graded, "reject", None) else "admitted")
        return graded

    def _grade(self, lease, rollouts: Sequence[Any], *, received: float):
        from reliquary_sandbox.attest import Expected

        from reliquary.protocol.rl_draw import lease_checkpoint_tag
        from reliquary.protocol.service_episode import rl_lease_engagement
        from reliquary.validator.episode_admission import EpisodeRefusal, episode_shape_refusal, verified_transcript
        from reliquary.validator.rl_submit import Graded

        policy = self.policy
        variant = self.lease_variant(lease)
        renderer = self.variant_renderer(variant)
        prompt = self.prompt_ids(lease.prompt_idx, variant)
        directory = self.directory(received)
        if directory is None:
            raise GraderInfrastructureError("the machine directory is stale")
        seen = frozenset(self.seen())
        verified = []
        for index, rollout in enumerate(rollouts):
            episode = rollout.episode
            if episode is None or rollout.force_span is not None:
                return Graded(prompt, reject="episode_schema")
            if len(prompt) + len(rollout.tokens) > policy.max_episode_tokens:
                return Graded(prompt, reject="episode_length")
            expected = Expected(hotkey=lease.hotkey, engagement=rl_lease_engagement(lease.lease_id, index),
                                env=policy.sandbox_env, split=policy.split, index=int(lease.prompt_idx),
                                checkpoint=lease_checkpoint_tag(int(lease.checkpoint)), seen_session_ids=seen,
                                require_graded=True)
            result = verified_transcript(policy, episode["transcript"], expected, directory=directory,
                                         token_verifier=self.token_verifier, received=received,
                                         where={"rollout": index})
            if isinstance(result, EpisodeRefusal):
                return Graded(prompt, reject=result.stage)
            verified.append(result)
        sessions = [result.claims.session_id for result in verified]
        if len(set(sessions)) != len(sessions):
            return Graded(prompt, reject="episode_session_reused")
        offset = len(prompt)
        for rollout, result in zip(rollouts, verified):
            spans = [(offset + start, offset + end) for start, end in rollout.spans]
            refusal = episode_shape_refusal(
                policy, renderer, prompt_ids=list(prompt), tokens=[*prompt, *rollout.tokens], spans=spans,
                stop=rollout.episode["stop"], calls=tuple(result.calls), opened=result.open, final=result.final,
                min_chunk_tokens=self.min_chunk_tokens, lease=True)
            if refusal is not None:
                return Graded(prompt, reject=refusal.stage)
        rewards = tuple(float(result.final.reward) for result in verified)
        return Graded(prompt, rewards, (False,) * len(rollouts), spans=tuple(r.spans for r in rollouts))


__all__ = ["LeaseEpisodeChecker"]
