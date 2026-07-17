"""Miner-side glue: force HF ``generate`` to sample from the protocol's forced
per-position draw instead of a local RNG.

The validator verifies the same draw by teacher-forcing (see
``reliquary.validator.verifier``); both sides call the identical
``warp`` + ``pick`` primitives in ``reliquary.environment.forced_sampling`` so an
honest miner running this processor scores ~1.0 on the seed-consistency gate.

A ``LogitsProcessor`` is the only clean hook: it keeps the batched, fast
``model.generate`` path (a Python decode loop over M rollouts would be far
slower) and merely replaces the per-step sample with the forced inverse-CDF pick.
Drive it with ``do_sample=False`` and NO temperature/top_k/top_p in
``generate`` — this processor does the full protocol warp itself, so HF's own
warpers must not run.
"""
from __future__ import annotations

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import pick, u_at, warp

# HF sampling-warper kwargs the forced path must NOT pass: the processor already
# applies the protocol warp (T_PROTO/top_k/top_p), so leaving these on generate()
# would warp twice and drift the miner off the validator's forced pick.
_WARPER_KWARGS = ("temperature", "top_p", "top_k")

# HF builds these logits processors from the model's ``generation_config`` and
# runs them BEFORE our processor -- they are NOT do_sample-gated, so a checkpoint
# whose config sets e.g. ``repetition_penalty`` would mutate the scores our
# processor then warps, drifting the miner off the validator's raw-logits forced
# pick (a systematic honest false-mismatch). Pin each to its inert value so only
# the forced processor acts; explicit generate() kwargs override generation_config.
_NEUTRAL_PROCESSOR_KWARGS = {
    "repetition_penalty": 1.0,
    "encoder_repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "encoder_no_repeat_ngram_size": 0,
    "min_length": 0,
    "min_new_tokens": 0,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "bad_words_ids": None,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "exponential_decay_length_penalty": None,
    "sequence_bias": None,
}


class ForcedSeedLogitsProcessor(LogitsProcessor):
    """Replace each row's sampled token with the forced inverse-CDF pick.

    Batch row ``r`` is completion of rollout ``rollout_indices[r]``; its first
    sampled token (step s=0) sits at completion offset ``base_offsets[r]`` and
    advances by one each step. ``start_len`` is ``input_ids.shape[1]`` at s=0
    (prompt length for phase-1; the left-padded width for BFT phase-2), so
    ``s = input_ids.shape[1] - start_len`` recovers the step index uniformly
    across left-padded rows.
    """

    def __init__(self, *, randomness: str, hotkey: str, prompt_idx: int,
                 checkpoint_hash: str, rollout_indices: list[int],
                 base_offsets: list[int], start_len: int,
                 temperature: float = T_PROTO, top_k: int = TOP_K_PROTO,
                 top_p: float = TOP_P_PROTO) -> None:
        self.randomness = randomness
        self.hotkey = hotkey
        self.prompt_idx = int(prompt_idx)
        self.checkpoint_hash = checkpoint_hash
        self.rollout_indices = [int(i) for i in rollout_indices]
        self.base_offsets = [int(o) for o in base_offsets]
        self.start_len = int(start_len)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        s = int(input_ids.shape[1]) - self.start_len
        out = torch.full_like(scores, float("-inf"))
        for r in range(scores.shape[0]):
            t = self.base_offsets[r] + s
            u = u_at(self.randomness, self.prompt_idx,
                     self.checkpoint_hash, self.rollout_indices[r], t)
            probs = warp(scores[r], t=self.temperature,
                         top_k=self.top_k, top_p=self.top_p)
            out[r, pick(probs, u)] = 0.0
        return out


def forced_seed_generate_kwargs(base_kwargs: dict, processor: LogitsProcessor) -> dict:
    """Return a copy of ``base_kwargs`` wired for forced-seed generation:
    strip HF's own sampling warpers, neutralize the generation_config-sourced
    logit processors (so the forced processor sees raw logits, matching the
    validator), force greedy selection of the processor's one-hot token, and
    attach the processor. ``base_kwargs`` is not mutated."""
    kw = dict(base_kwargs)
    for k in _WARPER_KWARGS:
        kw.pop(k, None)
    kw.update(_NEUTRAL_PROCESSOR_KWARGS)
    kw["do_sample"] = False
    kw["logits_processor"] = LogitsProcessorList([processor])
    return kw


def phase2_base_offsets(primed_lengths: list[int], prompt_length: int) -> list[int]:
    """Completion offset of the first phase-2 sampled token for each BFT row:
    the row resumes from its primed sequence, so the offset is
    ``primed_len - prompt_length`` (clamped at 0)."""
    return [max(0, int(L) - int(prompt_length)) for L in primed_lengths]
