"""How a lease keys the forced draw (rl-async-design.md §4.1-4.2): one definition for the miner, the validator
(audits) and the trainer (verification and the substitution loss).

The draw itself is the profile's (``forced_sampling``: ``sampling.draw``, Gumbel-max from v10). Its public uniform is
``u_at(randomness, prompt_idx, checkpoint_hash, rollout_index, t)``; under leases the stream is keyed by the lease:

* ``randomness``: the lease nonce as the validator hands it out, lowercase hex of ``H(NONCE_DOMAIN, S_k, env,
  prompt, k)`` (``validator/rl_leases.lease_nonce``). It already binds the env, the prompt and k;
* ``prompt_idx``: the leased prompt;
* ``checkpoint_hash``: ``k<k>``, the lease's checkpoint (the window path's revision string has no meaning here);
* ``t``: the model-token position. Single-turn: a rollout's policy positions are its completion offsets, the
  runner-keyed processor's ``num_tokens - num_prompt_tokens``. Episodes (multi-turn, §4.10): ``t`` counts the
  episode's MODEL tokens only, across turns (tool observations and renderer markup take no position), as the
  window path's ``miner/forced_draw``: a turn's request carries ``base_offset`` = the model tokens of the turns
  before it, so the processor's position is ``base_offset + num_tokens - num_prompt_tokens``. The validator and
  the trainer read the same stream over the episode's model spans (``model_positions``).

Every lease of one (env, prompt, k) therefore draws the same 16 streams, whoever holds it.
"""
from __future__ import annotations

from reliquary.environment.forced_sampling import u_at


def lease_checkpoint_tag(checkpoint: int) -> str:
    if type(checkpoint) is not int or checkpoint < 0:
        raise ValueError("checkpoint: a non-negative integer is required")
    return f"k{checkpoint}"


def _nonce(nonce: str) -> str:
    text = str(nonce)
    if not text or text != text.lower() or any(c not in "0123456789abcdef" for c in text):
        raise ValueError("lease nonce: lowercase hex required")
    return text


def lease_draw_context(nonce: str, prompt_idx: int, checkpoint: int) -> dict:
    """The request fields of the existing forced-draw plumbing (``forced_seed_extra_args``,
    ``VLLMRolloutGenerator.generate``) for one lease."""
    return {"randomness": _nonce(nonce), "prompt_idx": int(prompt_idx),
            "checkpoint_hash": lease_checkpoint_tag(int(checkpoint))}


def lease_extra_args(nonce: str, prompt_idx: int, checkpoint: int, rollout_index: int, base_offset: int = 0) -> dict:
    """The vLLM request's ``extra_args`` for rollout ``rollout_index`` of a lease, its draw starting at model-token
    position ``base_offset`` (a later turn of an episode; 0 for a single-turn rollout)."""
    from reliquary.miner.vllm_generation import forced_seed_extra_args

    return forced_seed_extra_args(**lease_draw_context(nonce, prompt_idx, checkpoint),
                                  rollout_index=int(rollout_index), base_offset=int(base_offset))


def model_positions(completion_length: int, spans=None) -> list[int]:
    """The completion-relative positions the policy sampled, in draw order: every position of a single-turn
    completion, else the positions of the episode's model spans (completion-relative). Draw ``t`` of the j-th
    entry is ``j``."""
    if spans is None:
        return list(range(int(completion_length)))
    return [p for start, end in spans for p in range(int(start), int(end))]


def lease_uniforms(nonce: str, prompt_idx: int, checkpoint: int, rollout_index: int, positions: int) -> list[float]:
    """The public uniforms of rollout ``rollout_index`` at model-token positions ``0..positions-1``."""
    context = lease_draw_context(nonce, prompt_idx, checkpoint)
    return [u_at(context["randomness"], context["prompt_idx"], context["checkpoint_hash"], int(rollout_index), t)
            for t in range(int(positions))]


__all__ = ["lease_checkpoint_tag", "lease_draw_context", "lease_extra_args", "lease_uniforms", "model_positions"]
