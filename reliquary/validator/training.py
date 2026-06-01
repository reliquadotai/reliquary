"""GRPO training step for Reliquary v2.1.

Single-step-per-window GRPO implementation: group-relative advantages
computed from the rewards in each ValidSubmission, PPO-clipped surrogate
loss, KL penalty against a frozen reference model (the validator's
starting checkpoint). Linear warmup + cosine LR schedule.

Uses miner-provided token log-probs (from the GRAIL commit) as π_old —
saves one forward pass per rollout.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.utils.checkpoint

from reliquary.validator import telemetry
from reliquary.constants import (
    GRAD_CLIP_NORM, KL_BETA, LEARNING_RATE, LR_COSINE_MAX_WINDOWS,
    LR_WARMUP_WINDOWS, PPO_CLIP_EPSILON,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-global state — persists across train_step calls for the same model
# ---------------------------------------------------------------------------

_optimizer: Optional[torch.optim.Optimizer] = None
_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
_optimizer_model_id: Optional[int] = None


def _build_optimizer(params) -> torch.optim.Optimizer:
    """Prefer bitsandbytes PagedAdamW8bit on CUDA — quantised optimiser
    state (~4× smaller than fp32 / ~2× smaller than bf16) plus unified
    memory paging that spills to host RAM under pressure. Falls back to
    plain AdamW when CUDA or bitsandbytes is unavailable (CPU tests, dev
    boxes without a GPU).
    """
    if torch.cuda.is_available():
        try:
            import bitsandbytes as bnb  # type: ignore[import-not-found]
            logger.info("Using bitsandbytes PagedAdamW8bit")
            return bnb.optim.PagedAdamW8bit(
                params,
                lr=LEARNING_RATE,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
            )
        except ImportError:
            logger.warning("bitsandbytes not available — falling back to torch.optim.AdamW")
    return torch.optim.AdamW(
        params,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )


def _lazy_init(model) -> bool:
    """Create optimizer + scheduler on first call for a given model. No-op
    on subsequent calls with the same model. The reference model used for
    KL is no longer built here — it's passed in by the caller (typically
    ``ValidationService.verify_model``) and refreshed externally on each
    publish.
    """
    global _optimizer, _scheduler, _optimizer_model_id
    if _optimizer_model_id == id(model):
        return True

    try:
        params = list(model.parameters())
    except (AttributeError, TypeError):
        logger.warning("_lazy_init: model has no .parameters(); skipping init")
        return False
    if not params:
        logger.warning("_lazy_init: model.parameters() is empty; skipping init")
        return False

    _optimizer = _build_optimizer(params)

    def _lr_lambda(step: int) -> float:
        if step < LR_WARMUP_WINDOWS:
            return (step + 1) / LR_WARMUP_WINDOWS
        progress = (step - LR_WARMUP_WINDOWS) / max(
            1, LR_COSINE_MAX_WINDOWS - LR_WARMUP_WINDOWS
        )
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    _scheduler = torch.optim.lr_scheduler.LambdaLR(_optimizer, _lr_lambda)
    _optimizer_model_id = id(model)
    logger.info("Training state initialised (optimizer, scheduler)")
    return True


def reset_training_state() -> None:
    """Clear the module-level singletons. Used by tests to start fresh.

    Production code should never call this — it throws away optimiser
    momentum.
    """
    global _optimizer, _scheduler, _optimizer_model_id
    _optimizer = None
    _scheduler = None
    _optimizer_model_id = None


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without a model)
# ---------------------------------------------------------------------------

def _compute_advantages(rewards: list[float]) -> list[float]:
    """Group-relative normalized advantages.

    mean = mean(rewards); std = pop-std(rewards); return (r - mean) / std.
    Degenerate group (std == 0) → all zeros (no signal, group will be skipped).
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = variance ** 0.5
    if std < 1e-8:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


# ---------------------------------------------------------------------------
# Per-rollout loss (forward-pass heavy — uses the model)
# ---------------------------------------------------------------------------

# Row-chunk for selected-logprob streaming. With Qwen3 vocab=152064:
#   chunk × vocab × 4 bytes = 64 × 152064 × 4 ≈ 39 MiB peak fp32 alloc per chunk.
_LOGPROB_CHUNK = 64


def _logprob_block(logits_slice: torch.Tensor, indices_slice: torch.Tensor) -> torch.Tensor:
    """log p(idx | row) for one chunk, in fp32. Equivalent to
    ``log_softmax(logits_slice.float(), dim=-1).gather(1, idx).squeeze(1)``.
    """
    logits_f = logits_slice.float()
    lse = torch.logsumexp(logits_f, dim=-1)
    gathered = logits_f.gather(1, indices_slice.unsqueeze(1)).squeeze(1)
    return gathered - lse


def _selected_logprobs(
    logits: torch.Tensor,
    indices: torch.Tensor,
    chunk: int = _LOGPROB_CHUNK,
) -> torch.Tensor:
    """Streaming, fp32-stable equivalent of
    ``log_softmax(logits.float(), dim=-1).gather(1, indices.unsqueeze(1)).squeeze(1)``.

    Materialises at most ``chunk × vocab × 4`` bytes of fp32 at a time
    instead of the full ``N × vocab × 4`` tensor. When ``logits.requires_grad``
    is True, each chunk is wrapped in ``torch.utils.checkpoint`` so backward
    also peaks at one chunk's worth of memory (recompute on demand) rather
    than holding the full fp32 cast for the backward pass.
    """
    n = logits.shape[0]
    use_ckpt = logits.requires_grad
    parts = []
    for i in range(0, n, chunk):
        end = i + chunk
        if use_ckpt:
            part = torch.utils.checkpoint.checkpoint(
                _logprob_block, logits[i:end], indices[i:end],
                use_reentrant=False,
            )
        else:
            part = _logprob_block(logits[i:end], indices[i:end])
        parts.append(part)
    return torch.cat(parts, dim=0)


def _rollout_loss(
    model,
    ref_model,
    rollout,
    advantage: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute (ppo_loss, kl_term, n_completion_tokens) for one rollout.

    ``ppo_loss`` and ``kl_term`` are scalars averaged over completion tokens;
    ``n_completion_tokens`` is the count, which lets ``train_step`` apply DAPO
    token-level normalisation (weigh every token equally) by recovering the
    per-token sum as ``mean * n``. Forward passes run in bf16 autocast;
    softmax / log-softmax cast back to fp32 for numerical stability.

    π_old comes from the miner's GRAIL commit (rollout.commit["rollout"]
    ["token_logprobs"]) — saves an extra forward pass.
    """
    tokens_list = rollout.commit["tokens"]
    prompt_length = rollout.commit.get("rollout", {}).get("prompt_length", 0)
    old_logprobs_list = rollout.commit.get("rollout", {}).get("token_logprobs", [])

    if prompt_length <= 0 or not old_logprobs_list:
        raise ValueError("rollout missing prompt_length or token_logprobs")

    tokens = torch.tensor([tokens_list], device=device)  # [1, T]

    # Current model forward pass (with grad). use_cache=False is required
    # for gradient_checkpointing to actually take effect — Qwen defaults
    # use_cache=True which silently disables checkpointing under HF.
    dtype_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) \
        if device.type in ("cuda", "cpu") else torch.autocast(device_type="cpu", enabled=False)
    with dtype_ctx:
        logits = model(tokens, use_cache=False).logits[0]  # [T, vocab]

    # log π_new(token_{t+1} | context_<=t) computed in fp32 but streamed
    # over rows — see _selected_logprobs.
    next_tokens = tokens[0, 1:]  # [T-1]
    new_logprobs = _selected_logprobs(logits[:-1], next_tokens)

    # Slice to completion tokens only: logits[prompt_length-1] predicts
    # tokens[prompt_length] (first completion token).
    new_logprobs_c = new_logprobs[prompt_length - 1:]

    # Reference model forward pass (no grad)
    with torch.no_grad():
        with dtype_ctx:
            ref_logits = ref_model(tokens, use_cache=False).logits[0]
        ref_logprobs = _selected_logprobs(ref_logits[:-1], next_tokens)
    ref_logprobs_c = ref_logprobs[prompt_length - 1:]

    # π_old from miner (same completion slice)
    old_logprobs = torch.tensor(
        old_logprobs_list, device=device, dtype=new_logprobs_c.dtype,
    )
    if len(old_logprobs) != len(new_logprobs_c):
        raise ValueError(
            f"log-prob length mismatch: miner reported {len(old_logprobs)}, "
            f"model predicts {len(new_logprobs_c)} completion tokens"
        )

    # PPO clipped surrogate
    log_ratio = new_logprobs_c - old_logprobs
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * advantage
    ppo_loss = -torch.min(surr1, surr2).mean()

    # KL(π_new || π_ref) — Schulman's k3 estimator:
    #   kl ≈ exp(ref - new) - 1 - (ref - new)
    # Unbiased, low-variance, always ≥ 0.
    kl_log_ratio = ref_logprobs_c - new_logprobs_c
    kl = (torch.exp(kl_log_ratio) - 1 - kl_log_ratio).mean()

    return ppo_loss, kl, int(new_logprobs_c.shape[0])


# ---------------------------------------------------------------------------
# Main entry point — one GRPO step per call
# ---------------------------------------------------------------------------

def train_step(
    model,
    batches: list,
    *,
    ref_model,
    window_index: int | None = None,
) -> Any:
    """Run one GRPO step over the union of *batches*.

    All rollouts across every batch contribute backward calls before a
    single optimizer.step(). *batches* is a list of batches, where each
    batch is a list of group objects (ValidSubmission). Pass ``[batch]``
    for the legacy mono-batch case.

    *ref_model* is the frozen reference policy for the KL penalty. The
    caller is responsible for keeping it up to date (in production:
    ``ValidationService.verify_model``, refreshed at every successful
    publish).

    *window_index* is used as the wandb step when telemetry is enabled.
    Safe to omit in tests.
    """
    if not batches or all(not b for b in batches):
        logger.info("train_step: empty batch, skipping")
        return model

    if not _lazy_init(model):
        logger.info("train_step: model not initializable (non-torch?), skipping")
        return model
    assert _optimizer is not None and _scheduler is not None

    model.train()
    device = next(model.parameters()).device

    _optimizer.zero_grad()

    n_total_rollouts = sum(len(g.rollouts) for batch in batches for g in batch)
    total_ppo = 0.0
    total_kl = 0.0
    n_processed = 0
    n_skipped = 0

    # Pass 1 (metadata only, no forward): group advantages + total completion-
    # token budget. DAPO token-level normalisation weighs every completion
    # token equally, vs the old sample-level mean which averaged per rollout
    # and so under-weighted (and under-penalised) tokens in long responses.
    # Completion length is known from the miner-committed token_logprobs, so
    # this pre-pass costs no extra forward.
    plan: list[tuple[Any, list[float]]] = []
    n_total_tokens = 0
    for batch in batches:
        for group in batch:
            advantages = _compute_advantages([r.reward for r in group.rollouts])
            if all(a == 0.0 for a in advantages):
                n_skipped += 1
                logger.debug("skipping degenerate group on prompt_idx=%d", group.prompt_idx)
                continue
            plan.append((group, advantages))
            for rollout in group.rollouts:
                meta = (rollout.commit or {}).get("rollout", {}) or {}
                n_total_tokens += len(meta.get("token_logprobs", []) or [])

    if n_total_tokens == 0:
        logger.info("train_step: no trainable completion tokens")
        return model

    # Pass 2: forward/backward one rollout at a time (peak memory = one
    # sequence) and accumulate the token-level-normalised loss.
    for group, advantages in plan:
        for rollout, adv in zip(group.rollouts, advantages):
            try:
                ppo_loss, kl, n_tok = _rollout_loss(
                    model=model, ref_model=ref_model,
                    rollout=rollout, advantage=adv, device=device,
                )
            except ValueError as e:
                logger.warning("rollout skipped: %s", e)
                continue
            # ppo_loss / kl are per-token means; ``* n_tok`` recovers the token
            # sum and ``/ n_total_tokens`` applies equal per-token weight.
            loss = (ppo_loss + KL_BETA * kl) * n_tok / n_total_tokens
            loss.backward()
            total_ppo += ppo_loss.item()
            total_kl += kl.item()
            n_processed += 1

    if n_processed == 0:
        logger.info("train_step: no valid rollouts processed")
        return model

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    _optimizer.step()
    _scheduler.step()
    lr = _scheduler.get_last_lr()[0]

    logger.info(
        "train_step: lr=%.2e ppo=%.4f kl=%.4f grad_norm=%.3f rollouts=%d/%d",
        lr, total_ppo / n_processed, total_kl / n_processed,
        float(grad_norm), n_processed, n_total_rollouts,
    )

    # Emit structured metrics to wandb (no-op if telemetry disabled).
    all_rewards = [r.reward for batch in batches for g in batch for r in g.rollouts]
    n_rewards = len(all_rewards)
    reward_mean = sum(all_rewards) / n_rewards
    reward_var = sum((r - reward_mean) ** 2 for r in all_rewards) / n_rewards
    reward_std = reward_var ** 0.5
    n_groups = sum(len(batch) for batch in batches)
    metrics = {
        "train/lr": lr,
        "train/ppo_loss": total_ppo / n_processed,
        "train/kl": total_kl / n_processed,
        "train/grad_norm": float(grad_norm),
        "train/rollouts_processed": n_processed,
        "train/rollouts_total": n_total_rollouts,
        "train/valid_rollout_ratio": n_processed / n_total_rollouts,
        "rewards/mean": reward_mean,
        "rewards/std": reward_std,
        "rewards/min": min(all_rewards),
        "rewards/max": max(all_rewards),
        "batch/n_groups": n_groups,
        "batch/n_degenerate_groups": n_skipped,
        "batch/degenerate_ratio": n_skipped / n_groups,
    }
    telemetry.log_training_step(metrics, step=window_index)

    return model
