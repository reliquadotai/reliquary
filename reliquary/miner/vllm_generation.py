"""Generate the protocol's forced draw on vLLM instead of transformers.

The draw is public: token t of rollout r is the inverse-CDF pick of ``u_at`` over
the warped policy, and the validator teacher-forces the same pick from its own
forward. Any engine may produce it, so long as it reads the same logits — which
is why this processor computes warp + pick itself and hands vLLM a one-hot,
with the request sampled greedily so it can select nothing else.

Measured on an H200 (20-09, Teutonic-I, code prompts): sequences generated here
and verified by the transformers validator agree on 0.964-0.980 of stochastic
positions per group, worst rollout 0.941, no group or rollout under its floor
and no terminal pick refused — level with the batched transformers path it
replaces (0.970-0.974). Throughput at 256 concurrent rollouts is 6,050 tok/s
against 507 for a 16-wide transformers batch, because a transformers decode step
costs ~31 ms whatever the batch, so the win is concurrency, not the engine.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import _warp_batch, u_at
from reliquary.shared.modeling import first_eos_index

logger = logging.getLogger(__name__)

try:  # vLLM is an optional miner dependency; the class must stay importable.
    from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
except ImportError:  # pragma: no cover - exercised on hosts without vLLM
    BatchUpdate = Any  # type: ignore[assignment,misc]
    LogitsProcessor = object  # type: ignore[assignment,misc]

FORCED_SEED_KEY = "forced_seed"


class ForcedSeedVLLMProcessor(LogitsProcessor):  # type: ignore[misc,valid-type]
    """One-hot the forced pick for every running request, in one batched pass.

    vLLM addresses requests by their slot in the persistent batch, and reuses a
    slot as soon as a request leaves it, so the per-slot state follows the
    engine's own removed/added/moved order rather than a request id.
    """

    def __init__(self, vllm_config: Any = None, device: Any = None,
                 is_pin_memory: bool = False) -> None:
        self._slots: dict[int, dict[str, Any]] = {}

    def is_argmax_invariant(self) -> bool:
        """The pick is not the argmax: this processor decides the token."""
        return False

    def update_state(self, batch_update: "BatchUpdate | None") -> None:
        if batch_update is None:
            return
        for index in getattr(batch_update, "removed", ()):
            self._slots.pop(_slot_index(index), None)
        for added in getattr(batch_update, "added", ()):
            index, params = added[0], added[1]
            slot = _slot_index(index)
            self._slots.pop(slot, None)
            forced = dict(getattr(params, "extra_args", None) or {}).get(FORCED_SEED_KEY)
            if forced:
                self._slots[slot] = {
                    "randomness": str(forced["randomness"]),
                    "prompt_idx": int(forced["prompt_idx"]),
                    "checkpoint_hash": str(forced["checkpoint_hash"]),
                    "rollout_index": int(forced["rollout_index"]),
                    "offset": int(forced.get("base_offset", 0)),
                }
        for moved in getattr(batch_update, "moved", ()):
            source, destination = _slot_index(moved[0]), _slot_index(moved[1])
            swap = "SWAP" in str(moved[2]).upper() if len(moved) > 2 else False
            from_state = self._slots.pop(source, None)
            to_state = self._slots.pop(destination, None)
            if from_state is not None:
                self._slots[destination] = from_state
            if swap and to_state is not None:
                self._slots[source] = to_state

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        slots = [slot for slot in sorted(self._slots) if slot < logits.shape[0]]
        if not slots:
            return logits
        draws = [
            u_at(state["randomness"], state["prompt_idx"], state["checkpoint_hash"],
                 state["rollout_index"], state["offset"])
            for state in (self._slots[slot] for slot in slots)
        ]
        rows = torch.tensor(slots, device=logits.device)
        probs = _warp_batch(logits[rows], t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
        cdf = torch.cumsum(probs, dim=-1)
        uniforms = torch.tensor(draws, device=cdf.device, dtype=cdf.dtype).unsqueeze(1)
        picks = torch.searchsorted(cdf, uniforms, right=True).squeeze(1)
        picks = picks.clamp(max=probs.shape[-1] - 1)
        forced = torch.full_like(logits, float("-inf"))
        forced[rows, picks] = 0.0
        for slot in slots:
            self._slots[slot]["offset"] += 1
        return forced


def _slot_index(value: Any) -> int:
    return int(value[0] if isinstance(value, (tuple, list)) else value)


def forced_seed_extra_args(
    *, randomness: str, prompt_idx: int, checkpoint_hash: str,
    rollout_index: int, base_offset: int = 0,
) -> dict[str, Any]:
    """What one request tells the processor about its place in the draw."""
    return {
        FORCED_SEED_KEY: {
            "randomness": randomness,
            "prompt_idx": int(prompt_idx),
            "checkpoint_hash": checkpoint_hash,
            "rollout_index": int(rollout_index),
            "base_offset": int(base_offset),
        }
    }


class VLLMRolloutGenerator:
    """A group of rollouts for one prompt, generated concurrently on vLLM.

    The engine is built once and reused; a checkpoint change rebuilds it, since
    vLLM holds its weights for the process's life.
    """

    def __init__(
        self, model_path: str, *, revision: str | None = None,
        max_model_len: int | None = None, gpu_memory_utilization: float = 0.85,
        max_num_seqs: int = 256, enforce_eager: bool = False,
        engine: Any | None = None, sampling_params_class: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.revision = revision
        self._settings = {
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
        }
        self._sampling_params_class = sampling_params_class
        self._llm = engine if engine is not None else self._build(model_path, revision)

    def _build(self, model_path: str, revision: str | None) -> Any:
        from vllm import LLM  # local import: optional dependency

        engine = LLM(
            model=model_path, revision=revision, dtype="bfloat16",
            logits_processors=[ForcedSeedVLLMProcessor], **self._settings,
        )
        logger.info(
            "vLLM generation backend ready (%s%s)", model_path,
            f"@{revision}" if revision else "",
        )
        return engine

    def reload(self, model_path: str, revision: str | None = None) -> None:
        """Point the generator at new weights.

        vLLM holds its weights for the engine's life, so this rebuilds it. That
        costs a model load per checkpoint, which is why the miner pulls a
        checkpoint between windows rather than during one.
        """
        del self._llm
        self.model_path, self.revision = model_path, revision
        self._llm = self._build(model_path, revision)

    def generate(
        self, prompt_tokens: list[int], *, randomness: str, prompt_idx: int,
        checkpoint_hash: str, rollouts: int, max_new_tokens: int,
        eos_ids: list[int],
    ) -> list[list[int]]:
        """Completion token ids per rollout, truncated at their first stop token.

        vLLM keeps the token that stopped a request (measured 20-09), which the
        protocol needs: the terminal pick is checked on it. The truncation is the
        transformers path's, so a stop token appearing mid-batch cannot carry
        padding downstream either way.
        """
        sampling_params = self._sampling_params_class
        if sampling_params is None:
            from vllm import SamplingParams as sampling_params

        stop_ids = sorted(int(token) for token in eos_ids)
        requests = [{"prompt_token_ids": list(prompt_tokens)} for _ in range(rollouts)]
        sampling = [
            sampling_params(
                temperature=0.0, max_tokens=int(max_new_tokens), detokenize=False,
                stop_token_ids=stop_ids or None,
                extra_args=forced_seed_extra_args(
                    randomness=randomness, prompt_idx=prompt_idx,
                    checkpoint_hash=checkpoint_hash, rollout_index=index,
                ),
            )
            for index in range(rollouts)
        ]
        outputs = self._llm.generate(requests, sampling, use_tqdm=False)
        completions: list[list[int]] = []
        for output in outputs:
            completion = list(output.outputs[0].token_ids)
            end = first_eos_index(completion, set(stop_ids))
            if end is not None:
                completion = completion[: end + 1]
            completions.append(completion)
        return completions
