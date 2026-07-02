"""Standalone VRAM smoke-test for the GRPO training step.

Loads the configured base model, builds B_BATCH × M_ROLLOUTS synthetic
rollouts at worst-case sequence length, runs ``train_step`` once (cold
peak) then once more (warm peak), prints peak VRAM at each phase.

No validator, no R2, no bittensor — pure torch + transformers + the
``train_step`` function as it runs in production.

Usage (on the H100 box, inside the validator Docker image):

    python scripts/smoke_train_vram.py
    python scripts/smoke_train_vram.py --completion-len 4096
    python scripts/smoke_train_vram.py --model Qwen/Qwen3.5-2B

Exits non-zero on OOM. Use ``CUDA_VISIBLE_DEVICES=0`` to pin a GPU.
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from dataclasses import dataclass, field

import torch

from reliquary.constants import (
    ATTN_IMPLEMENTATION, B_BATCH, DEFAULT_BASE_MODEL,
    M_ROLLOUTS, MAX_NEW_TOKENS_PROTOCOL_CAP,
)
from reliquary.validator.training import reset_training_state, train_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smoke_vram")


@dataclass
class _Rollout:
    tokens: list
    reward: float
    commit: dict = field(default_factory=dict)


@dataclass
class _Group:
    rollouts: list
    prompt_idx: int = 0


def _build_batch(prompt_len: int, completion_len: int, n_groups: int, m_rollouts: int) -> list:
    """Build n_groups × m_rollouts synthetic rollouts, all at the worst-case
    seq length. Uses token id 1 throughout — the actual ids don't matter for
    memory; only the shapes do.
    """
    seq_len = prompt_len + completion_len
    batch = []
    for g in range(n_groups):
        rollouts = []
        # Spread rewards so advantages aren't all-zero (otherwise the group
        # is skipped and we'd never exercise the loss path).
        for i in range(m_rollouts):
            tokens = [1] * seq_len
            rollouts.append(_Rollout(
                tokens=tokens,
                reward=float(i),
                commit={
                    "tokens": tokens,
                    "rollout": {
                        "prompt_length": prompt_len,
                        # arbitrary but length-correct — train_step doesn't validate values
                        "token_logprobs": [-1.0] * completion_len,
                    },
                },
            ))
        batch.append(_Group(rollouts=rollouts, prompt_idx=g))
    return batch


def _gb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.2f} GB"


def _print_mem(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    logger.info(
        "[%s] alloc=%s peak=%s reserved=%s",
        tag, _gb(alloc), _gb(peak), _gb(reserved),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--prompt-len", type=int, default=256,
                   help="Synthetic prompt length in tokens")
    p.add_argument("--completion-len", type=int, default=MAX_NEW_TOKENS_PROTOCOL_CAP,
                   help="Synthetic completion length (defaults to protocol cap)")
    p.add_argument("--n-groups", type=int, default=B_BATCH)
    p.add_argument("--m-rollouts", type=int, default=M_ROLLOUTS)
    args = p.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA not available — this smoke test requires a GPU")
        return 1

    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory
    logger.info("GPU: %s (%s total)", device_name, _gb(total_mem))
    logger.info(
        "Config: model=%s prompt_len=%d completion_len=%d n_groups=%d m_rollouts=%d → %d rollouts/step",
        args.model, args.prompt_len, args.completion_len,
        args.n_groups, args.m_rollouts, args.n_groups * args.m_rollouts,
    )

    reset_training_state()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _print_mem("startup")

    logger.info("Loading train model in bf16 with attn=%s …", ATTN_IMPLEMENTATION)
    from reliquary.shared.modeling import load_text_generation_model

    model = load_text_generation_model(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).to("cuda:0").eval()
    try:
        model.gradient_checkpointing_enable()
        logger.info("gradient_checkpointing enabled")
    except (AttributeError, NotImplementedError):
        logger.warning("model does not support gradient_checkpointing_enable")
    _print_mem("after train model load")

    logger.info("Loading frozen reference model in bf16 with attn=%s …", ATTN_IMPLEMENTATION)
    ref_model = load_text_generation_model(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
    ).to("cuda:0").eval()
    ref_model.requires_grad_(False)
    _print_mem("after ref model load")

    batch = _build_batch(
        prompt_len=args.prompt_len,
        completion_len=args.completion_len,
        n_groups=args.n_groups,
        m_rollouts=args.m_rollouts,
    )
    _print_mem("after batch build (CPU only)")

    # ── Cold step — exercises _lazy_init (allocates optimiser + ref model) ──
    logger.info("=== COLD train_step (initialises optimiser + ref model) ===")
    torch.cuda.reset_peak_memory_stats()
    try:
        train_step(model, [batch], ref_model=ref_model, window_index=0)
    except torch.cuda.OutOfMemoryError as e:
        _print_mem("OOM (cold)")
        logger.error("OOM during cold train_step: %s", e)
        return 2
    _print_mem("after cold train_step")

    # ── Warm step — steady-state, optimiser + ref already allocated ──
    logger.info("=== WARM train_step (steady state) ===")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        train_step(model, [batch], ref_model=ref_model, window_index=1)
    except torch.cuda.OutOfMemoryError as e:
        _print_mem("OOM (warm)")
        logger.error("OOM during warm train_step: %s", e)
        return 2
    _print_mem("after warm train_step")

    logger.info("OK — train_step fits on this GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
