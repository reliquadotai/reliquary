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
* one session twice in the group is refused; the reward is the final record's (validator authoritative), 0 for an
  episode the model's first malformed turn ended (``format_error``: the turn markup it wrote, or a turn with calls
  ended on the end-of-text token), which stays in the group instead of refusing the lease.

No TOPLOC proof per span (dropped with the windows): the episode's model tokens are verified like any rollout's,
by the forced draw (audits and the trainer's forward) over the model positions, keyed per model-token position
across turns (``protocol/rl_draw``). Sandbox time counts in the lease's latency, never in the match.

Any refusal rejects the whole group (no observation, quota halved: ``RlSubmissions``), as a bad termination does
for a single-turn group: an episode cannot be withheld, so the 16 draws stay the miner's only choice-free sample.
A stale machine directory raises ``GraderInfrastructureError`` (retry later, the lease stays open).
"""
from __future__ import annotations

import collections
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
                 renderer_of: Callable[..., Any] | None = None) -> None:
        from reliquary.environment.agentic_swe import policy_default_harness, policy_harness_choices
        from reliquary.protocol.toploc import MIN_CHUNK_TOKENS

        self.policy, self.renderer, self.source = policy, renderer, source
        # The renderer of another surface than the default (a variant other than 0 of a harness with axes:
        # ``renderer_of(variant)``; another harness of a mix: ``renderer_of(variant, spec=spec)``): the lease's,
        # computed here.
        self.renderer_of = renderer_of
        self._mixed = len(policy_harness_choices(policy)) > 1
        self._default = policy_default_harness(policy)
        self._surfaces: dict[tuple[str, int], Any] = {(self._default.key, 0): renderer}
        self.directory, self.token_verifier, self.seen = directory, token_verifier, seen
        self.min_chunk_tokens = MIN_CHUNK_TOKENS if min_chunk_tokens is None else int(min_chunk_tokens)
        self._prompt = functools.lru_cache(maxsize=256)(self._render)
        # Leases graded, per (harness, verdict): admitted, or the refusal's stage.
        self.outcomes: collections.Counter = collections.Counter()

    def _render(self, prompt_idx: int, variant: int, key: str) -> tuple[int, ...]:
        renderer = self._surfaces[key, variant]
        return tuple(int(t) for t in renderer.initial_ids(self.source.prompt(int(prompt_idx))))

    def prompt_ids(self, prompt_idx: int, variant: int = 0, spec=None) -> tuple[int, ...]:
        self.surface_renderer(spec, variant)               # built (and cached) before the prompt cache keys it
        return self._prompt(int(prompt_idx), int(variant), (spec or self._default).key)

    def variant_renderer(self, variant: int):
        return self.surface_renderer(None, variant)

    def surface_renderer(self, spec, variant: int):
        """The renderer of harness ``spec`` (None: the default one) in surface ``variant``."""
        spec = spec or self._default
        key = (spec.key, int(variant))
        if key not in self._surfaces:
            if self.renderer_of is None:
                raise ValueError(f"no renderer for {spec.key} surface variant {variant}")
            self._surfaces[key] = (self.renderer_of(int(variant)) if spec is self._default
                                   else self.renderer_of(int(variant), spec=spec))
        return self._surfaces[key]

    def lease_surface(self, lease):
        """(spec, variant) the lease's episodes must have played (the validator's own draw)."""
        from reliquary.environment.agentic_swe import lease_surface

        return lease_surface(self.policy, lease)

    def lease_variant(self, lease) -> int:
        """The surface variant the lease's episodes must have played (the validator's own draw)."""
        return self.lease_surface(lease)[1]

    def grade(self, lease, rollouts: Sequence[Any], *, received: float):
        """``rl_submit.Graded`` of a lease's episodes (``Rollout`` with ``episode``). Every lease is logged with
        its harness, surface variant and verdict (a refusal with its detail) and counted in ``outcomes``."""
        spec, variant = self.lease_surface(lease)
        graded, detail = self._grade(lease, rollouts, received=received, surface=(spec, variant))
        reject = getattr(graded, "reject", None)
        self.outcomes[spec.key, reject or "admitted"] += 1
        logger.info("episode lease %s (%s, %s, surface variant %d): %s", lease.lease_id[:12],
                    self.policy.environment, spec.key, variant,
                    f"rejected {reject} {detail}" if reject else "admitted")
        return graded

    def _grade(self, lease, rollouts: Sequence[Any], *, received: float, surface):
        from reliquary_sandbox.attest import Expected

        from reliquary.protocol.rl_draw import lease_checkpoint_tag
        from reliquary.protocol.service_episode import rl_lease_engagement
        from reliquary.validator.episode_admission import EpisodeRefusal, episode_shape_refusal, verified_transcript
        from reliquary.validator.rl_submit import Graded

        policy = self.policy
        spec, variant = surface
        spec = None if spec is self._default else spec
        renderer = self.surface_renderer(spec, variant)
        prompt = self.prompt_ids(lease.prompt_idx, variant, spec)
        directory = self.directory(received)
        if directory is None:
            raise GraderInfrastructureError("the machine directory is stale")
        seen = frozenset(self.seen())
        verified = []
        for index, rollout in enumerate(rollouts):
            episode = rollout.episode
            if episode is None or rollout.force_span is not None:
                return Graded(prompt, reject="episode_schema"), {"rollout": index}
            if len(prompt) + len(rollout.tokens) > policy.max_episode_tokens:
                return Graded(prompt, reject="episode_length"), {"rollout": index}
            expected = Expected(hotkey=lease.hotkey, engagement=rl_lease_engagement(lease.lease_id, index),
                                env=policy.sandbox_env, split=policy.split, index=int(lease.prompt_idx),
                                checkpoint=lease_checkpoint_tag(int(lease.checkpoint)), seen_session_ids=seen,
                                require_graded=True)
            result = verified_transcript(policy, episode["transcript"], expected, directory=directory,
                                         token_verifier=self.token_verifier, received=received,
                                         where={"rollout": index})
            if isinstance(result, EpisodeRefusal):
                return Graded(prompt, reject=result.stage), result.detail
            verified.append(result)
        sessions = [result.claims.session_id for result in verified]
        if len(set(sessions)) != len(sessions):
            return Graded(prompt, reject="episode_session_reused"), {}
        offset = len(prompt)
        for rollout, result in zip(rollouts, verified):
            spans = [(offset + start, offset + end) for start, end in rollout.spans]
            refusal = episode_shape_refusal(
                policy, renderer, prompt_ids=list(prompt), tokens=[*prompt, *rollout.tokens], spans=spans,
                stop=rollout.episode["stop"], calls=tuple(result.calls), opened=result.open, final=result.final,
                min_chunk_tokens=self.min_chunk_tokens, lease=True)
            if refusal is not None:
                return Graded(prompt, reject=refusal.stage), refusal.detail
        from reliquary.corpus.trajectory_parse import STOP_FORMAT_ERROR

        # A format_error episode (the model's first malformed turn ended it) scores 0, whatever the box's grade: it
        # stays in the group, so the lease and its other episodes count.
        rewards = tuple(0.0 if rollout.episode["stop"] == STOP_FORMAT_ERROR else float(result.final.reward)
                        for rollout, result in zip(rollouts, verified))
        return Graded(prompt, rewards, (False,) * len(rollouts), spans=tuple(r.spans for r in rollouts)), None


__all__ = ["LeaseEpisodeChecker"]
