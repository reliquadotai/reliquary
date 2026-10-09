"""The forced draw of an RL signed episode, on the miner (phase 2, plan 2C; spec §4.1.4).

Every model token of an episode is drawn from ``u(pool, seed_index, position)`` where ``position`` counts
the episode's MODEL tokens only, across turns: tool outputs and renderer markup take no position. The
validator checks the same stream over the episode's policy positions (``batcher.seed_uniforms``). A
generate session (the verifiers trace id of one episode) is bound to its (pool, seed); each turn's draw
starts at the model tokens the session has generated so far."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DrawBinding:
    pool: Any                 # protocol.seed_pool.SeedPool
    seed_index: int

    def __post_init__(self) -> None:
        self.pool.uniform(self.seed_index, 0)            # refuses a seed outside the pool

    def uniforms(self, base_offset: int, count: int) -> list[float]:
        return [self.pool.uniform(self.seed_index, base_offset + j) for j in range(count)]

    def extra_args(self, base_offset: int) -> dict:
        """The vLLM request's ``extra_args`` for ``ForcedSeedVLLMProcessor``: the public pool, the seed
        and the turn's first model-token position."""
        from reliquary.miner.vllm_generation import forced_seed_extra_args

        return forced_seed_extra_args(randomness=self.pool.randomness, prompt_idx=self.pool.prompt_idx,
                                      checkpoint_hash=self.pool.checkpoint_hash, rollout_index=0,
                                      base_offset=int(base_offset), seed_pool=self.pool,
                                      seed_index=self.seed_index)


class ForcedDraws:
    """Session id -> binding; thread-safe (the generate endpoint reads it from its request handlers)."""

    def __init__(self) -> None:
        self._bindings: dict[str, DrawBinding] = {}
        self._lock = threading.Lock()

    def bind(self, session_id: str, binding: DrawBinding) -> None:
        with self._lock:
            current = self._bindings.get(session_id)
            if current is not None and current != binding:
                raise ValueError(f"session {session_id} is already bound to another seed")
            self._bindings[session_id] = binding

    def get(self, session_id: str) -> DrawBinding | None:
        with self._lock:
            return self._bindings.get(session_id)

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._bindings.pop(session_id, None)


def forced_sampling_params(base: dict, draw: dict) -> dict:
    """vLLM ``SamplingParams`` kwargs for a forced turn: the processor one-hots the forced pick (its own
    warp is the protocol's), so the engine takes the argmax; the job's stop ids, cap and logprobs stay."""
    params = dict(base)
    params.update(temperature=0.0, top_p=1.0, top_k=-1, extra_args=dict(draw))
    return params


__all__ = ["DrawBinding", "ForcedDraws", "forced_sampling_params"]
