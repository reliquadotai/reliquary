"""Episodes for the async reference miner (rl-async-design.md §4.10; implementation map §1.15, §9).

``EpisodeLeaseGenerator`` is a ``GroupGenerator`` with ``agenerate``: per lease of a signed-episode env it plays
the lease's ``ROLLOUTS`` episodes at once through the window path's machinery, unchanged but for the binding:

* the session is opened for rollout i of the lease (``rl_episode_client.LeaseEpisodeRunner``: engagement
  ``rl-lease:{lease_id}:{i}``; the validator's sandbox issuer grants at most one per rollout);
* every generate session of the episode is bound to ``forced_draw.LeaseDrawBinding`` (nonce, prompt, k, i): the
  forced engine (``corpus_generate_server.GenerateEngine``) starts each turn's draw at the model tokens generated so
  far, so the draw is keyed per model-token position across turns (``protocol/rl_draw``);
* an ended episode is kept only when the validator would admit it (``episode_commit.episode_admissible``: graded,
  usable, its stop, length, prompt fidelity and turn shape) and becomes an ``rl_async.EpisodeRollout``.

Submission is mandatory and an episode cannot be withheld: when one of the lease's episodes is unusable the whole
lease fails (its graded sessions are withdrawn, the lease expires unsubmitted); the miner never chooses among
episodes. A machine-signed ``aborted`` final is played again once (the issuer frees that engagement once), as in
the window miner.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

ABORT_REPLAYS = 1


class EpisodeLeaseFailed(RuntimeError):
    """A lease whose episodes cannot all be submitted."""


class EpisodeLeaseGenerator:
    """``policies[env]``: the env's episode policy; ``runner_factory(policy, lease, prompt)``: an async context
    manager running episodes (``LeaseEpisodeRunner``); ``engine``: the forced generate engine (``take_session``,
    ``drop_session``); ``draws``: its ``ForcedDraws``; ``renderer_for(policy)``: the turn renderer the validator
    builds; ``task_prompt(env, prompt_idx)``: the task's prompt text; ``load_weights(state, version)``: how the
    engine takes a new checkpoint (None: refused, the miner restarts on new weights)."""

    def __init__(self, *, policies: Mapping[str, Any], runner_factory: Callable[..., Any], engine, draws,
                 renderer_for: Callable[[Any], Any], task_prompt: Callable[[str, int], str],
                 load_weights: Callable[[Mapping[str, Any], int], None] | None = None,
                 rollouts: int = 16) -> None:
        self.policies, self._factory, self._engine, self._draws = dict(policies), runner_factory, engine, draws
        self._renderer_for, self._task_prompt, self._load = renderer_for, task_prompt, load_weights
        self.rollouts = int(rollouts)
        self._renderers: dict[str, Any] = {}

    def generate(self, leases) -> list:
        raise RuntimeError("episodes are played on the event loop: call agenerate")

    async def agenerate(self, leases: Sequence[Any]) -> list[list[Any]]:
        """Every lease's ``ROLLOUTS`` episodes (``EpisodeRollout``), leases played side by side. Raises when any
        lease fails (the miner's generation retry, then the lease is given up)."""
        results = await asyncio.gather(*(self.play_lease(lease) for lease in leases), return_exceptions=True)
        for lease, result in zip(leases, results):
            if isinstance(result, BaseException):
                raise EpisodeLeaseFailed(f"lease {lease.lease_id[:12]}: {result}") from result
        return list(results)

    def load_weights(self, state: Mapping[str, Any], version: int) -> None:
        if self._load is None:
            raise RuntimeError("this episode engine takes no new weights in place: restart the miner")
        self._load(state, version)

    def _renderer(self, policy):
        name = policy.environment
        if name not in self._renderers:
            self._renderers[name] = self._renderer_for(policy)
        return self._renderers[name]

    async def play_lease(self, lease) -> list[Any]:
        from reliquary.miner.episode_commit import episode_admissible
        from reliquary.miner.episode_group_miner import play_bound
        from reliquary.miner.forced_draw import LeaseDrawBinding
        from reliquary.miner.rl_async import EpisodeRollout

        policy = self.policies.get(lease.environment)
        if policy is None:
            raise EpisodeLeaseFailed(f"{lease.environment} is not an episode environment of this miner")
        renderer = self._renderer(policy)
        prompt = self._task_prompt(lease.environment, int(lease.prompt_idx))
        prompt_len = len(renderer.initial_ids(prompt))
        runner = self._factory(policy, lease, prompt)
        label = f"lease {lease.lease_id[:12]}"

        async def one(index: int):
            binding = LeaseDrawBinding(lease.nonce, int(lease.prompt_idx), int(lease.checkpoint), index)
            for replay in range(ABORT_REPLAYS + 1):
                played = await play_bound(runner, self._engine, self._draws, index, binding,
                                          getattr(runner, "deadline", None), label=label)
                if played is None:
                    return None
                result, session = played
                if (not result.ok and getattr(result, "final_status", None) == "aborted"
                        and replay < ABORT_REPLAYS):
                    logger.info("rollout %d of %s aborted by its machine: played again", index, label)
                    continue
                return SimpleNamespace(seed_index=index, result=result, session=session)
            return None

        async with runner:
            outcomes = await asyncio.gather(*(one(i) for i in range(self.rollouts)), return_exceptions=True)
            kept, why = [], None
            for index, outcome in enumerate(outcomes):
                if isinstance(outcome, BaseException) or outcome is None:
                    why = why or f"rollout {index}: no episode ({outcome!r})"
                    kept.append(None)
                    continue
                admitted, refusal = episode_admissible(outcome, policy=policy, renderer=renderer, prompt=prompt)
                if admitted is None:
                    why = why or f"rollout {index}: {refusal}"
                kept.append(admitted)
            if why is not None:
                await _release([o for o in outcomes if not isinstance(o, BaseException) and o is not None])
                raise EpisodeLeaseFailed(why)
            for outcome in outcomes:
                done = getattr(outcome.result, "submitted", None)
                if done is not None:
                    done()                    # always submitted: the live slot is free, the session kept graded
        return [EpisodeRollout(tuple(int(t) for t in value.tokens[prompt_len:]),
                               tuple((int(s) - prompt_len, int(e) - prompt_len) for s, e in value.spans),
                               str(value.stop), dict(outcome.result.transcript))
                for value, outcome in zip(kept, outcomes)]


async def _release(outcomes) -> None:
    for outcome in outcomes:
        release = getattr(outcome.result, "release", None)
        if release is None:
            continue
        try:
            await release()
        except Exception:
            logger.exception("withdrawing the episode of rollout %s failed", outcome.seed_index)


__all__ = ["ABORT_REPLAYS", "EpisodeLeaseFailed", "EpisodeLeaseGenerator"]
