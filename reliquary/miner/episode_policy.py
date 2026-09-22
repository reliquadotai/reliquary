"""Hugging Face policy adapter for canonical Reliquary Episode v1 rollouts."""

from __future__ import annotations

from typing import Any

from reliquary.environment.agentic.types import (
    MAX_ACTION_BYTES,
    AssistantAction,
    GeneratedAction,
)


class JsonActionStoppingCriteria:
    """Stop as soon as the generated suffix is one complete action object."""

    def __init__(self, tokenizer: Any, start_length: int) -> None:
        self.tokenizer = tokenizer
        self.start_length = int(start_length)

    def __call__(self, input_ids, scores, **kwargs):
        del scores, kwargs
        import torch

        verdicts = []
        for row in input_ids:
            text = self.tokenizer.decode(
                row[self.start_length:].tolist(), skip_special_tokens=False
            )
            try:
                AssistantAction.from_json(text)
                verdicts.append(True)
            except (RecursionError, TypeError, ValueError):
                verdicts.append(False)
        return torch.tensor(verdicts, device=input_ids.device, dtype=torch.bool)


def _within_action_bytes(tokenizer, generated: list[int]) -> tuple[list[int], str]:
    """The longest prefix of a turn that fits the action byte budget.

    A turn that overruns it used to raise out of the policy — past the runner's
    own handling — so one rambling turn ended the miner's episode with a
    traceback instead of an invalid action worth zero. The cut is on tokens, so
    the validator replaying the trace reads exactly the same turn.
    """

    def decode(count: int) -> str:
        return tokenizer.decode(generated[:count], skip_special_tokens=False)

    text = decode(len(generated))
    if len(text.encode("utf-8")) <= MAX_ACTION_BYTES:
        return generated, text
    low, high = 1, len(generated)
    while low < high:
        middle = (low + high + 1) // 2
        if len(decode(middle).encode("utf-8")) <= MAX_ACTION_BYTES:
            low = middle
        else:
            high = middle - 1
    return generated[:low], decode(low)


class HFEpisodePolicy:
    """Single-rollout forced-seed policy used by ``MiningEngine``."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        randomness: str,
        hotkey: str,
        prompt_idx: int,
        checkpoint_hash: str,
        rollout_index: int,
        max_action_tokens: int,
        max_episode_tokens: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.randomness = randomness
        self.hotkey = hotkey
        self.prompt_idx = int(prompt_idx)
        self.checkpoint_hash = checkpoint_hash
        self.rollout_index = int(rollout_index)
        self.max_action_tokens = int(max_action_tokens)
        self.max_episode_tokens = int(max_episode_tokens)
        self.sampled_offset = 0

    def generate(self, *, tokens: tuple[int, ...], **_: object) -> GeneratedAction:
        import torch
        from transformers import StoppingCriteriaList

        from reliquary.miner.forced_seed_sampler import (
            ForcedSeedLogitsProcessor,
            forced_seed_generate_kwargs,
        )
        from reliquary.shared.modeling import resolve_eos_token_ids

        context = list(tokens)
        remaining = self.max_episode_tokens - len(context)
        if remaining <= 0:
            raise RuntimeError("episode token budget exhausted before termination")
        action_budget = min(self.max_action_tokens, remaining)
        device = getattr(self.model, "device", "cpu")
        input_ids = torch.tensor([context], device=device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        eos_ids = resolve_eos_token_ids(self.model, self.tokenizer)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None and eos_ids:
            pad_token_id = min(eos_ids)
        processor = ForcedSeedLogitsProcessor(
            randomness=self.randomness,
            hotkey=self.hotkey,
            prompt_idx=self.prompt_idx,
            checkpoint_hash=self.checkpoint_hash,
            rollout_indices=[self.rollout_index],
            base_offsets=[self.sampled_offset],
            start_len=len(context),
        )
        kwargs = {
            "max_new_tokens": action_budget,
            "attention_mask": attention_mask,
            "pad_token_id": pad_token_id,
            "stopping_criteria": StoppingCriteriaList([
                JsonActionStoppingCriteria(self.tokenizer, len(context))
            ]),
        }
        if eos_ids:
            kwargs["eos_token_id"] = sorted(eos_ids)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                **forced_seed_generate_kwargs(kwargs, processor),
            )
        generated = output[0, len(context):].tolist()
        if not generated:
            raise RuntimeError("episode policy produced no action tokens")
        generated, text = _within_action_bytes(self.tokenizer, generated)
        self.sampled_offset += len(generated)
        return GeneratedAction(
            text=text,
            tokens=tuple(int(token) for token in generated),
        )
