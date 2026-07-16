"""GRPO training step for Reliquary v2.1.

Single-step-per-window GRPO implementation: group-relative advantages
computed from the rewards in each ValidSubmission, PPO-clipped surrogate
loss, KL penalty against a frozen reference model (the validator's
starting checkpoint). Linear warmup + cosine LR schedule.

By default, uses miner-provided token log-probs (from the GRAIL commit) as
π_old. Production can instead pass the published behavior model and recompute
π_old independently; the KL reference remains a separate policy.
"""

from __future__ import annotations

import gc
import logging
import math
from typing import Any, Optional

import torch
import torch.utils.checkpoint

from reliquary.validator import telemetry
from reliquary.constants import (
    GRAD_CLIP_NORM, GRAD_NORM_SKIP_THRESHOLD, KL_BETA, LEARNING_RATE,
    LR_COSINE_MAX_WINDOWS, LR_WARMUP_WINDOWS,
    MICROBATCH_MAX_PADDED_TOKENS, PPO_CLIP_EPSILON,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-global state — persists across train_step calls for the same model
# ---------------------------------------------------------------------------

_optimizer: Optional[torch.optim.Optimizer] = None
_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
_optimizer_model_id: Optional[int] = None


class TrainingStepSkipped(RuntimeError):
    """A deliberately rejected optimizer step that must not be published."""

    def __init__(self, reason: str, grad_norm: float) -> None:
        super().__init__(f"{reason}: grad_norm={grad_norm}")
        self.reason = reason
        self.grad_norm = grad_norm


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


def _shape_advantages(rollouts, advantages):
    """Two-sided length shaping, on ADVANTAGES only (the σ-gate stays on raw
    correctness). Overrides an advantage to −SHAPE_PENALTY when:
      * overlong       — the rollout is cap-truncated; or
      * under-thinking — a non-forced rollout finished early
        (completion_length < SHAPE_LEN_FRAC·BFT_THINKING_BUDGET) and is wrong.
    All inputs are validator-determined (reward re-graded, completion_length
    schema-checked, forced/truncated validator-set). SHAPE_PENALTY == 0 disables
    it."""
    from reliquary.constants import (
        BFT_THINKING_BUDGET,
        SHAPE_LEN_FRAC,
        SHAPE_PENALTY,
    )

    if SHAPE_PENALTY <= 0:
        return advantages
    early_cap = SHAPE_LEN_FRAC * BFT_THINKING_BUDGET
    shaped = list(advantages)
    for i, r in enumerate(rollouts):
        meta = (getattr(r, "commit", None) or {}).get("rollout", {}) or {}
        if meta.get("truncated"):
            shaped[i] = -SHAPE_PENALTY          # overlong (cap-truncated)
            continue
        if meta.get("forced"):
            continue                            # forced rollouts untouched
        correct = float(getattr(r, "reward", 0.0)) > 0.5
        clen = int(meta.get("completion_length", 0))
        if clen < early_cap and not correct:
            shaped[i] = -SHAPE_PENALTY          # under-thinking (early + wrong)
    return shaped


def _batch_loss_weights(
    token_counts: list[float],
    raw_weights: Optional[list[float]] = None,
) -> list[float]:
    """Per-token loss scale ``s_b = w_b / N_b`` for token-level-per-env loss.

    ``token_counts[b]`` is batch b's surviving completion-token count (N_b). A
    batch with N_b == 0 is dropped (scale 0). ``raw_weights`` are the relative
    env weights w_e (default equal); they are renormalized over the *present*
    batches so ``Σ_b s_b·N_b == 1``, hence each env contributes exactly its
    weight regardless of token mass. A single present batch gives ``s = 1/N`` —
    the plain global token-level (DAPO) normalization the loss used before.
    """
    n = len(token_counts)
    if raw_weights is None:
        raw_weights = [1.0] * n
    present = [b for b in range(n) if token_counts[b] > 0]
    total_w = sum(raw_weights[b] for b in present)
    scales = [0.0] * n
    if total_w <= 0:
        return scales
    for b in present:
        scales[b] = (raw_weights[b] / total_w) / token_counts[b]
    return scales


def _env_weight_for_batch(batch, env_weights: dict) -> float:
    """Relative loss weight for a batch, read from the ``env_name`` on its first
    rollout (every rollout in a batch shares one env). Unknown/absent → 1.0."""
    for group in batch:
        for rollout in group.rollouts:
            name = getattr(rollout, "env_name", "") or ""
            return float(env_weights.get(name, 1.0))
    return 1.0


def _completion_keep_list(rollout, prompt_length: int, n_completion: int):
    """Return a boolean keep list over completion positions.

    BFT forced-close tokens are validator-accepted but not policy-sampled, so
    they must be excluded from every policy-gradient path and from token-count
    normalization denominators.  Only the batcher's private, validator-derived
    span is trusted; wire metadata is never a training input. ``None`` means
    every completion token is kept.
    """
    force_span = getattr(rollout, "_validated_force_span", None)
    if not force_span:
        return None
    keep = [True] * max(0, int(n_completion))
    fs = max(0, int(force_span[0]) - int(prompt_length))
    fe = min(len(keep), int(force_span[1]) - int(prompt_length))
    if fs < fe:
        keep[fs:fe] = [False] * (fe - fs)
    return keep


def _completion_token_logprobs(rollout) -> list:
    """Normalize either protocol-supported log-prob layout to completion-only."""
    commit = rollout.commit or {}
    tokens = commit.get("tokens", []) or []
    meta = commit.get("rollout", {}) or {}
    prompt_length = int(meta.get("prompt_length", 0) or 0)
    old = list(meta.get("token_logprobs", []) or [])
    if tokens and len(old) == len(tokens):
        return old[prompt_length:]
    return old


def _trainable_completion_count(
    rollout,
    prompt_length: int,
    n_completion: int,
) -> int:
    keep = _completion_keep_list(rollout, prompt_length, n_completion)
    if keep is None:
        return max(0, int(n_completion))
    return sum(1 for flag in keep if flag)


def _plan_from_batches(batches, env_weights: Optional[dict] = None):
    """Pass 1 (metadata only, no forward): group-relative advantages + per-batch
    token-level loss weights. ``batches`` is one batch per env (env_mix order,
    see service.py). Returns ``(plan, n_skipped)`` where each plan entry is
    ``(group, advantages, scale)`` and ``scale`` is the per-token weight w_b/N_b
    for the group's env. Degenerate groups (std==0 → zero advantage → no signal)
    are dropped and counted in ``n_skipped``.
    """
    from reliquary.constants import ENV_LOSS_WEIGHTS

    if env_weights is None:
        env_weights = ENV_LOSS_WEIGHTS
    n_batches = len(batches)
    surviving: list[tuple[Any, list[float], int]] = []
    batch_tokens = [0.0] * n_batches
    n_skipped = 0
    for b_idx, batch in enumerate(batches):
        for group in batch:
            advantages = _shape_advantages(
                group.rollouts,
                _compute_advantages([r.reward for r in group.rollouts]),
            )
            if all(a == 0.0 for a in advantages):
                n_skipped += 1
                logger.debug("skipping degenerate group on prompt_idx=%s",
                             getattr(group, "prompt_idx", "?"))
                continue
            surviving.append((group, advantages, b_idx))
            for rollout in group.rollouts:
                meta = (rollout.commit or {}).get("rollout", {}) or {}
                old = _completion_token_logprobs(rollout)
                prompt_length = int(meta.get("prompt_length", 0) or 0)
                batch_tokens[b_idx] += _trainable_completion_count(
                    rollout, prompt_length, len(old),
                )

    raw_weights = None
    if env_weights:
        raw_weights = [_env_weight_for_batch(b, env_weights) for b in batches]
    scales = _batch_loss_weights(batch_tokens, raw_weights)
    plan = [(g, a, scales[b]) for g, a, b in surviving]
    return plan, n_skipped


def _bft_training_metrics(plan) -> dict[str, float | int]:
    """Summarize actual BFT exposure in the surviving training plan.

    ``abs_adv_weighted_tokens`` includes the plan's environment scale and is a
    gradient-exposure proxy, not an exact gradient norm. It is sufficient to
    detect a forced path becoming disproportionately influential over time.
    """
    by_path: dict[str, dict[str, float]] = {}
    total_rollouts = 0
    total_raw_tokens = 0
    total_trainable_tokens = 0
    total_injected_tokens = 0
    forced_rollouts = 0
    forced_trainable_tokens = 0
    total_abs_adv_weighted_tokens = 0.0
    forced_abs_adv_weighted_tokens = 0.0

    for group, advantages, scale in plan:
        for rollout, advantage in zip(group.rollouts, advantages):
            meta = (rollout.commit or {}).get("rollout", {}) or {}
            prompt_length = int(meta.get("prompt_length", 0) or 0)
            raw_tokens = len(_completion_token_logprobs(rollout))
            trainable_tokens = _trainable_completion_count(
                rollout, prompt_length, raw_tokens,
            )
            span = getattr(rollout, "_validated_force_span", None)
            injected_tokens = max(0, raw_tokens - trainable_tokens)
            path = str(
                getattr(rollout, "_validated_termination_path", None)
                or "unknown"
            )
            path_metrics = by_path.setdefault(
                path,
                {
                    "rollouts": 0.0,
                    "trainable_tokens": 0.0,
                    "injected_tokens_masked": 0.0,
                    "abs_adv_weighted_tokens": 0.0,
                },
            )
            exposure = (
                abs(float(advantage)) * float(scale) * trainable_tokens
            )
            path_metrics["rollouts"] += 1
            path_metrics["trainable_tokens"] += trainable_tokens
            path_metrics["injected_tokens_masked"] += injected_tokens
            path_metrics["abs_adv_weighted_tokens"] += exposure
            total_abs_adv_weighted_tokens += exposure

            total_rollouts += 1
            total_raw_tokens += raw_tokens
            total_trainable_tokens += trainable_tokens
            total_injected_tokens += injected_tokens
            if span is not None:
                forced_rollouts += 1
                forced_trainable_tokens += trainable_tokens
                forced_abs_adv_weighted_tokens += exposure

    metrics: dict[str, float | int] = {
        "bft/plan_rollouts": total_rollouts,
        "bft/forced_rollouts": forced_rollouts,
        "bft/forced_rollout_ratio": (
            forced_rollouts / total_rollouts if total_rollouts else 0.0
        ),
        "bft/raw_completion_tokens": total_raw_tokens,
        "bft/trainable_completion_tokens": total_trainable_tokens,
        "bft/injected_tokens_masked": total_injected_tokens,
        "bft/injected_token_ratio": (
            total_injected_tokens / total_raw_tokens if total_raw_tokens else 0.0
        ),
        "bft/forced_trainable_token_ratio": (
            forced_trainable_tokens / total_trainable_tokens
            if total_trainable_tokens
            else 0.0
        ),
        "bft/abs_adv_weighted_tokens": total_abs_adv_weighted_tokens,
        "bft/forced_abs_adv_weighted_tokens": (
            forced_abs_adv_weighted_tokens
        ),
        "bft/forced_abs_adv_weighted_token_ratio": (
            forced_abs_adv_weighted_tokens / total_abs_adv_weighted_tokens
            if total_abs_adv_weighted_tokens
            else 0.0
        ),
    }
    for path, values in sorted(by_path.items()):
        prefix = f"bft/path/{path}"
        metrics[f"{prefix}/rollouts"] = int(values["rollouts"])
        metrics[f"{prefix}/trainable_tokens"] = int(
            values["trainable_tokens"]
        )
        metrics[f"{prefix}/injected_tokens_masked"] = int(
            values["injected_tokens_masked"]
        )
        metrics[f"{prefix}/abs_adv_weighted_tokens"] = float(
            values["abs_adv_weighted_tokens"]
        )
    return metrics


# ---------------------------------------------------------------------------
# Per-rollout loss (forward-pass heavy — uses the model)
# ---------------------------------------------------------------------------

# Row-chunk for selected-logprob streaming. With Qwen3.5 vocab=248320:
#   chunk × vocab × 4 bytes = 64 × 248320 × 4 ≈ 61 MiB peak fp32 alloc per chunk.
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


def _hidden_logprob_block(
    hidden_slice: torch.Tensor,
    indices_slice: torch.Tensor,
    lm_head,
) -> torch.Tensor:
    return _logprob_block(lm_head(hidden_slice), indices_slice)


def _selected_logprobs_from_hidden(
    hidden_rows: torch.Tensor,
    indices: torch.Tensor,
    lm_head,
    chunk: int = _LOGPROB_CHUNK,
) -> torch.Tensor:
    """Compute selected token logprobs from hidden states in row chunks.

    This avoids materialising the full ``sequence × vocab`` logits tensor that
    HF ``model(...).logits`` returns. Qwen3.5 has a 248k-token vocab, so the
    full-logits path is the difference between fitting long rollouts and
    getting killed by memory pressure. Checkpointing keeps backward memory at
    one chunk by recomputing the LM-head/logsumexp block as needed.
    """
    n = hidden_rows.shape[0]
    use_ckpt = hidden_rows.requires_grad
    parts = []
    for i in range(0, n, chunk):
        end = i + chunk

        def _block(h, idx):
            return _hidden_logprob_block(h, idx, lm_head)

        if use_ckpt:
            part = torch.utils.checkpoint.checkpoint(
                _block, hidden_rows[i:end], indices[i:end],
                use_reentrant=False,
            )
        else:
            part = _block(hidden_rows[i:end], indices[i:end])
        parts.append(part)
    return torch.cat(parts, dim=0)


def _base_model_and_lm_head(model):
    base = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if base is None or lm_head is None or not callable(lm_head):
        return None, None
    return base, lm_head


def _last_hidden_state(outputs):
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    try:
        return outputs[0]
    except (TypeError, IndexError):
        return None


def _selected_logprobs_for_tokens(model, tokens: torch.Tensor, next_tokens: torch.Tensor) -> torch.Tensor:
    """Selected next-token logprobs without full-sequence logits when possible."""
    base, lm_head = _base_model_and_lm_head(model)
    if base is not None and lm_head is not None:
        try:
            base_out = base(tokens, use_cache=False)
            hidden = _last_hidden_state(base_out)
            if hidden is not None:
                return _selected_logprobs_from_hidden(hidden[0, :-1], next_tokens, lm_head)
        except TypeError:
            # Some tiny test doubles / legacy models don't expose a compatible
            # base forward. Fall back to the standard HF logits contract below.
            pass

    logits = model(tokens, use_cache=False).logits[0]
    return _selected_logprobs(logits[:-1], next_tokens)


def _completion_keep_mask(rollout, prompt_length, n_completion, device):
    """Boolean keep-mask over completion positions with the BFT ``force_span``
    excluded. Returns ``None`` when the rollout is not forced (train every
    completion token, identical to the pre-BFT path)."""
    keep = _completion_keep_list(rollout, prompt_length, n_completion)
    if keep is None:
        return None
    return torch.tensor(keep, dtype=torch.bool, device=device)


def _rollout_loss(
    model,
    ref_model,
    rollout,
    advantage: float,
    device,
    *,
    behavior_model=None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute (ppo_loss, kl_term, n_completion_tokens) for one rollout.

    ``ppo_loss`` and ``kl_term`` are scalars averaged over completion tokens;
    ``n_completion_tokens`` is the count, which lets ``train_step`` apply DAPO
    token-level normalisation (weigh every token equally) by recovering the
    per-token sum as ``mean * n``. Forward passes run in bf16 autocast;
    softmax / log-softmax cast back to fp32 for numerical stability.

    π_old comes from ``behavior_model`` when supplied, otherwise from the
    miner's GRAIL commit. ``ref_model`` is used only for the KL term.
    """
    tokens_list = rollout.commit["tokens"]
    prompt_length = rollout.commit.get("rollout", {}).get("prompt_length", 0)
    old_logprobs_list = _completion_token_logprobs(rollout)

    if prompt_length <= 0 or not old_logprobs_list:
        raise ValueError("rollout missing prompt_length or token_logprobs")

    tokens = torch.tensor([tokens_list], device=device)  # [1, T]

    # Current model forward pass (with grad). use_cache=False is required
    # for gradient_checkpointing to actually take effect — Qwen defaults
    # use_cache=True which silently disables checkpointing under HF.
    dtype_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) \
        if device.type in ("cuda", "cpu") else torch.autocast(device_type="cpu", enabled=False)
    next_tokens = tokens[0, 1:]  # [T-1]
    with dtype_ctx:
        new_logprobs = _selected_logprobs_for_tokens(model, tokens, next_tokens)

    # Slice to completion tokens only: logits[prompt_length-1] predicts
    # tokens[prompt_length] (first completion token).
    new_logprobs_c = new_logprobs[prompt_length - 1:]

    # Reference model forward pass (no grad)
    with torch.no_grad():
        with dtype_ctx:
            ref_logprobs = _selected_logprobs_for_tokens(ref_model, tokens, next_tokens)
    ref_logprobs_c = ref_logprobs[prompt_length - 1:]

    behavior_logprobs_c = None
    if behavior_model is ref_model:
        behavior_logprobs_c = ref_logprobs_c
    elif behavior_model is not None:
        with torch.no_grad():
            with dtype_ctx:
                behavior_logprobs = _selected_logprobs_for_tokens(
                    behavior_model, tokens, next_tokens
                )
        behavior_logprobs_c = behavior_logprobs[prompt_length - 1:]

    # π_old from miner (same completion slice)
    old_logprobs = torch.tensor(
        old_logprobs_list, device=device, dtype=new_logprobs_c.dtype,
    )
    if len(old_logprobs) != len(new_logprobs_c):
        raise ValueError(
            f"log-prob length mismatch: miner reported {len(old_logprobs)}, "
            f"model predicts {len(new_logprobs_c)} completion tokens"
        )
    if behavior_logprobs_c is not None:
        if len(behavior_logprobs_c) != len(new_logprobs_c):
            raise ValueError("behavior-model completion length mismatch")
        old_logprobs = behavior_logprobs_c.detach()

    # Mask the validator-injected FORCE span out of the loss. Those actions were
    # not sampled by the policy, so policy-gradient on them is invalid and their
    # tiny probability would blow up the ratio; train thinking-before and the
    # answer-after only. This is injected-action masking, not DAPO overlong
    # filtering (which drops whole overlong samples).
    keep = _completion_keep_mask(
        rollout, prompt_length,
        int(new_logprobs_c.shape[0]), device,
    )

    # PPO clipped surrogate
    log_ratio = new_logprobs_c - old_logprobs
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * advantage
    ppo_per_token = -torch.min(surr1, surr2)

    # KL(π_new || π_ref) — Schulman's k3 estimator:
    #   kl ≈ exp(ref - new) - 1 - (ref - new)
    # Unbiased, low-variance, always ≥ 0.
    kl_log_ratio = ref_logprobs_c - new_logprobs_c
    kl_per_token = torch.exp(kl_log_ratio) - 1 - kl_log_ratio

    if keep is not None:
        ppo_per_token = ppo_per_token[keep]
        kl_per_token = kl_per_token[keep]

    n_keep = int(ppo_per_token.shape[0])
    if n_keep == 0:
        # Whole completion masked (degenerate) → no training signal.
        return ppo_per_token.sum(), kl_per_token.sum(), 0

    return ppo_per_token.mean(), kl_per_token.mean(), n_keep


# ---------------------------------------------------------------------------
# Micro-batched forward/backward — pack short rollouts into one forward.
# Numerically ~equivalent to the per-rollout path (bf16 attention-kernel noise
# only); ~2.6× faster on a realistic length mix. See the equivalence tests.
# ---------------------------------------------------------------------------

def _pack_by_token_budget(lengths: list[int], budget: int) -> list[list[int]]:
    """Greedy length-sorted bin packing. Each bin's padded cost (n_seqs ×
    longest_seq) stays ≤ budget; a sequence longer than budget bins alone (=
    the legacy one-at-a-time path), so peak memory never exceeds one such bin.
    Returns lists of indices into ``lengths``.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    bins: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        L = lengths[i]
        if cur and (len(cur) + 1) * max(cur_max, L) > budget:
            bins.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        bins.append(cur)
    return bins


def _batched_completion_logprobs(model, input_ids, attention_mask, prompt_lengths, lengths):
    """Selected next-token logprobs over every rollout's completion tokens in a
    right-padded batch, concatenated in row order. Same per-token values as
    ``_selected_logprobs_for_tokens`` (each row depends only on its own hidden
    state) — only the completion-predicting rows are gathered. Returns
    ``(logprobs[N], seg)`` where ``seg[i]`` is rollout i's completion-token count.
    """
    device = input_ids.device
    b_rows: list[int] = []
    p_rows: list[int] = []
    seg: list[int] = []
    for i, (p, L) in enumerate(zip(prompt_lengths, lengths)):
        seg.append(L - p)
        for t in range(p - 1, L - 1):
            b_rows.append(i)
            p_rows.append(t)
    b_idx = torch.tensor(b_rows, device=device, dtype=torch.long)
    p_idx = torch.tensor(p_rows, device=device, dtype=torch.long)
    targets = input_ids[b_idx, p_idx + 1]
    base, lm_head = _base_model_and_lm_head(model)
    if base is not None and lm_head is not None:
        try:
            out = base(input_ids, attention_mask=attention_mask, use_cache=False)
        except TypeError:
            out = base(input_ids, use_cache=False)  # doubles without a mask kwarg
        hidden = _last_hidden_state(out)
        if hidden is not None:
            return _selected_logprobs_from_hidden(hidden[b_idx, p_idx], targets, lm_head), seg
    logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
    return _selected_logprobs(logits[b_idx, p_idx], targets), seg


def _new_kl_stats() -> dict[str, float | int]:
    return {
        "token_count": 0,
        "nonfinite_count": 0,
        "gt_0_1_count": 0,
        "gt_1_count": 0,
        "gt_10_count": 0,
        "max": 0.0,
        "log_ratio_abs_max": 0.0,
        "weighted_ppo": 0.0,
        "weighted_kl": 0.0,
        "ppo_token_count": 0,
        "ppo_clip_active_count": 0,
        "ppo_ratio_below_clip_count": 0,
        "ppo_ratio_above_clip_count": 0,
        "ppo_ratio_nonfinite_count": 0,
        "ppo_log_ratio_abs_max": 0.0,
        "ppo_log_ratio_abs_gt_1_count": 0,
        "ppo_log_ratio_abs_gt_2_count": 0,
        "ppo_log_ratio_abs_gt_5_count": 0,
        "pi_old_claim_token_count": 0,
        "pi_old_claim_abs_error_sum": 0.0,
        "pi_old_claim_abs_error_max": 0.0,
        "pi_old_claim_gt_1e_3_count": 0,
    }


def _record_kl_stats(
    stats: dict[str, float | int] | None,
    *,
    ppo_tok,
    kl_tok,
    kl_log,
    scale_cat,
    ppo_ratio,
    ppo_log_ratio,
    ppo_clip_active,
    claimed_old,
    behavior_old,
) -> None:
    """Accumulate bounded policy-health telemetry after a micro-batch."""
    if stats is None:
        return
    with torch.no_grad():
        values = kl_tok.detach().float()
        log_ratio = kl_log.detach().float()
        stats["token_count"] += values.numel()
        stats["nonfinite_count"] += int((~torch.isfinite(values)).sum())
        stats["gt_0_1_count"] += int((values > 0.1).sum())
        stats["gt_1_count"] += int((values > 1.0).sum())
        stats["gt_10_count"] += int((values > 10.0).sum())
        stats["max"] = max(float(stats["max"]), float(values.max()))
        stats["log_ratio_abs_max"] = max(
            float(stats["log_ratio_abs_max"]),
            float(log_ratio.abs().max()),
        )
        stats["weighted_ppo"] += float((scale_cat * ppo_tok).sum())
        stats["weighted_kl"] += float((scale_cat * kl_tok).sum())
        stats["ppo_token_count"] += ppo_ratio.numel()
        stats["ppo_clip_active_count"] += int(ppo_clip_active.sum())
        stats["ppo_ratio_below_clip_count"] += int(
            (ppo_ratio < 1.0 - PPO_CLIP_EPSILON).sum()
        )
        stats["ppo_ratio_above_clip_count"] += int(
            (ppo_ratio > 1.0 + PPO_CLIP_EPSILON).sum()
        )
        stats["ppo_ratio_nonfinite_count"] += int(
            (~torch.isfinite(ppo_ratio)).sum()
        )
        finite_log_ratio = ppo_log_ratio.detach().float()
        finite_log_ratio = finite_log_ratio[torch.isfinite(finite_log_ratio)]
        if finite_log_ratio.numel():
            abs_log_ratio = finite_log_ratio.abs()
            stats["ppo_log_ratio_abs_max"] = max(
                float(stats["ppo_log_ratio_abs_max"]),
                float(abs_log_ratio.max()),
            )
            stats["ppo_log_ratio_abs_gt_1_count"] += int(
                (abs_log_ratio > 1.0).sum()
            )
            stats["ppo_log_ratio_abs_gt_2_count"] += int(
                (abs_log_ratio > 2.0).sum()
            )
            stats["ppo_log_ratio_abs_gt_5_count"] += int(
                (abs_log_ratio > 5.0).sum()
            )
        if behavior_old is not None:
            claim_error = (
                behavior_old.detach().float()
                - claimed_old.detach().float()
            ).abs()
            finite_claim_error = claim_error[torch.isfinite(claim_error)]
            stats["pi_old_claim_token_count"] += claim_error.numel()
            stats["pi_old_claim_gt_1e_3_count"] += int(
                (claim_error > 1e-3).sum()
            )
            if finite_claim_error.numel():
                stats["pi_old_claim_abs_error_sum"] += float(
                    finite_claim_error.sum()
                )
                stats["pi_old_claim_abs_error_max"] = max(
                    float(stats["pi_old_claim_abs_error_max"]),
                    float(finite_claim_error.max()),
                )


def _kl_telemetry_metrics(stats: dict[str, float | int]) -> dict[str, float]:
    token_count = int(stats["token_count"])
    denominator = max(1, token_count)
    weighted_ppo = float(stats["weighted_ppo"])
    weighted_kl = float(stats["weighted_kl"])
    weighted_penalty = KL_BETA * weighted_kl
    ppo_token_count = int(stats["ppo_token_count"])
    ppo_denominator = max(1, ppo_token_count)
    claim_token_count = int(stats["pi_old_claim_token_count"])
    claim_denominator = max(1, claim_token_count)
    return {
        "train/kl_beta": KL_BETA,
        "train/ppo_objective_component": weighted_ppo,
        "train/kl_objective_component": weighted_kl,
        "train/kl_penalty_objective": weighted_penalty,
        "train/kl_to_ppo_abs_ratio": (
            abs(weighted_penalty) / max(abs(weighted_ppo), 1e-12)
        ),
        "train/kl_token_count": float(token_count),
        "train/kl_token_max": float(stats["max"]),
        "train/kl_log_ratio_abs_max": float(stats["log_ratio_abs_max"]),
        "train/kl_token_nonfinite_ratio": (
            int(stats["nonfinite_count"]) / denominator
        ),
        "train/kl_token_gt_0_1_ratio": (
            int(stats["gt_0_1_count"]) / denominator
        ),
        "train/kl_token_gt_1_ratio": int(stats["gt_1_count"]) / denominator,
        "train/kl_token_gt_10_ratio": (
            int(stats["gt_10_count"]) / denominator
        ),
        "train/ppo_clip_active_ratio": (
            int(stats["ppo_clip_active_count"]) / ppo_denominator
        ),
        "train/ppo_ratio_below_clip_ratio": (
            int(stats["ppo_ratio_below_clip_count"]) / ppo_denominator
        ),
        "train/ppo_ratio_above_clip_ratio": (
            int(stats["ppo_ratio_above_clip_count"]) / ppo_denominator
        ),
        "train/ppo_ratio_outside_clip_ratio": (
            (
                int(stats["ppo_ratio_below_clip_count"])
                + int(stats["ppo_ratio_above_clip_count"])
            )
            / ppo_denominator
        ),
        "train/ppo_ratio_nonfinite_ratio": (
            int(stats["ppo_ratio_nonfinite_count"]) / ppo_denominator
        ),
        "train/ppo_log_ratio_abs_max": float(
            stats["ppo_log_ratio_abs_max"]
        ),
        "train/ppo_log_ratio_abs_gt_1_ratio": (
            int(stats["ppo_log_ratio_abs_gt_1_count"]) / ppo_denominator
        ),
        "train/ppo_log_ratio_abs_gt_2_ratio": (
            int(stats["ppo_log_ratio_abs_gt_2_count"]) / ppo_denominator
        ),
        "train/ppo_log_ratio_abs_gt_5_ratio": (
            int(stats["ppo_log_ratio_abs_gt_5_count"]) / ppo_denominator
        ),
        "train/pi_old_claim_token_count": float(claim_token_count),
        "train/pi_old_claim_abs_error_mean": (
            float(stats["pi_old_claim_abs_error_sum"]) / claim_denominator
        ),
        "train/pi_old_claim_abs_error_max": float(
            stats["pi_old_claim_abs_error_max"]
        ),
        "train/pi_old_claim_gt_1e_3_ratio": (
            int(stats["pi_old_claim_gt_1e_3_count"]) / claim_denominator
        ),
    }


def _microbatch_grad(
    model,
    ref_model,
    batch,
    device,
    *,
    atomic,
    kl_stats=None,
    behavior_model=None,
):
    """Forward + backward one micro-batch. ``batch`` is a list of
    ``(tokens, prompt_length, old_logprobs, advantage, scale, keep)`` where ``scale``
    is the per-token loss weight ``w_e/N_e`` of the rollout's env. Returns
    ``(sum_ppo_mean, sum_kl_mean, n)`` for logging (per-rollout means summed,
    matching the legacy metric).

    ``atomic=True`` commits gradients via ``torch.autograd.grad`` only after the
    backward fully succeeds, so an OOM mid-backward (gradient-checkpoint
    recompute) leaves already-accumulated ``.grad`` untouched — the failed
    micro-batch contributes nothing and a split-retry stays correct.
    """
    B = len(batch)
    T = max(len(it[0]) for it in batch)
    input_ids = torch.zeros(B, T, dtype=torch.long, device=device)
    attn = torch.zeros(B, T, dtype=torch.long, device=device)
    plens, lens, olds, advs, scales, keeps = [], [], [], [], [], []
    for j, (tokens, p, old, adv, scale, keep) in enumerate(batch):
        L = len(tokens)
        input_ids[j, :L] = torch.tensor(tokens, device=device)
        attn[j, :L] = 1
        plens.append(p)
        lens.append(L)
        olds.append(old)
        advs.append(adv)
        scales.append(scale)
        keeps.append(keep)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type in ("cuda", "cpu")):
        new_lp, seg = _batched_completion_logprobs(model, input_ids, attn, plens, lens)
        with torch.no_grad():
            ref_lp, _ = _batched_completion_logprobs(ref_model, input_ids, attn, plens, lens)
            if behavior_model is ref_model:
                behavior_lp = ref_lp
            elif behavior_model is not None:
                behavior_lp, behavior_seg = _batched_completion_logprobs(
                    behavior_model, input_ids, attn, plens, lens
                )
                if behavior_seg != seg:
                    raise ValueError("behavior-model completion segments mismatch")
            else:
                behavior_lp = None

    claimed_old_cat = torch.tensor(
        [x for old in olds for x in old],
        device=device,
        dtype=new_lp.dtype,
    )
    old_cat = claimed_old_cat
    behavior_old_cat = None
    if behavior_lp is not None:
        if behavior_lp.shape != new_lp.shape:
            raise ValueError("behavior-model completion shape mismatch")
        behavior_old_cat = behavior_lp.detach()
        old_cat = behavior_old_cat
    adv_cat = torch.tensor(
        [advs[k] for k in range(B) for _ in range(seg[k])],
        device=device,
        dtype=new_lp.dtype,
    )
    scale_cat = torch.tensor(
        [scales[k] for k in range(B) for _ in range(seg[k])],
        device=device,
        dtype=new_lp.dtype,
    )
    keep_cat = torch.tensor(
        [flag for keep in keeps for flag in keep],
        device=device,
        dtype=torch.bool,
    )
    new_lp = new_lp[keep_cat]
    ref_lp = ref_lp[keep_cat]
    old_cat = old_cat[keep_cat]
    claimed_old_cat = claimed_old_cat[keep_cat]
    if behavior_old_cat is not None:
        behavior_old_cat = behavior_old_cat[keep_cat]
    adv_cat = adv_cat[keep_cat]
    scale_cat = scale_cat[keep_cat]
    keep_seg = [sum(1 for flag in keep if flag) for keep in keeps]
    ppo_log_ratio = new_lp - old_cat
    ratio = torch.exp(ppo_log_ratio)
    unclipped_surr = ratio * adv_cat
    clipped_surr = (
        torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON)
        * adv_cat
    )
    surr = torch.min(unclipped_surr, clipped_surr)
    clip_active = clipped_surr < unclipped_surr
    ppo_tok = -surr
    kl_log = ref_lp - new_lp
    kl_tok = torch.exp(kl_log) - 1 - kl_log
    # Token-level (DAPO) normalisation, weighted per-env: scale_cat carries each
    # token's w_e/N_e, so the loss is Σ_e w_e·(token-mean over env e) — no env's
    # raw token mass dominates the shared step.
    loss = (scale_cat * (ppo_tok + KL_BETA * kl_tok)).sum()

    if atomic:
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is None:
                    continue  # param not in this micro-batch's graph -> 0 contribution
                p.grad = g if p.grad is None else p.grad + g
    else:
        loss.backward()

    _record_kl_stats(
        kl_stats,
        ppo_tok=ppo_tok,
        kl_tok=kl_tok,
        kl_log=kl_log,
        scale_cat=scale_cat,
        ppo_ratio=ratio,
        ppo_log_ratio=ppo_log_ratio,
        ppo_clip_active=clip_active,
        claimed_old=claimed_old_cat,
        behavior_old=behavior_old_cat,
    )

    sum_ppo = sum_kl = 0.0
    off = 0
    with torch.no_grad():
        for n in keep_seg:
            sum_ppo += float(ppo_tok[off:off + n].mean())
            sum_kl += float(kl_tok[off:off + n].mean())
            off += n
    return sum_ppo, sum_kl, B


def _process_microbatch(
    model, ref_model, batch, device, *, atomic, kl_stats=None,
    behavior_model=None,
):
    """One micro-batch; when atomic, on OOM halve and retry down to a single
    sequence (which, if it still OOMs, is a genuine unrecoverable OOM)."""
    if not atomic:
        return _microbatch_grad(
            model, ref_model, batch, device, atomic=False, kl_stats=kl_stats,
            behavior_model=behavior_model,
        )
    oom = False
    try:
        return _microbatch_grad(
            model, ref_model, batch, device, atomic=True, kl_stats=kl_stats,
            behavior_model=behavior_model,
        )
    except torch.cuda.OutOfMemoryError:
        if len(batch) == 1:
            raise
        oom = True
    # Reclaim OUTSIDE the except: while it is active its traceback pins the
    # failed forward's frame (and tensors), so empty_cache there frees nothing.
    if oom:
        gc.collect()
        torch.cuda.empty_cache()
        mid = len(batch) // 2
        a = _process_microbatch(
            model, ref_model, batch[:mid], device,
            atomic=True, kl_stats=kl_stats, behavior_model=behavior_model,
        )
        b = _process_microbatch(
            model, ref_model, batch[mid:], device,
            atomic=True, kl_stats=kl_stats, behavior_model=behavior_model,
        )
        return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _build_microbatch_items(plan):
    """Flatten plan -> list of (tokens, prompt_length, old_logprobs, advantage,
    scale, keep), dropping rollouts the per-rollout path would have skipped (missing
    prompt_length/token_logprobs, or a miner/model completion-length mismatch).
    ``scale`` is the per-token loss weight w_e/N_e carried from the plan entry."""
    items = []
    for group, advantages, scale in plan:
        for rollout, adv in zip(group.rollouts, advantages):
            commit = rollout.commit or {}
            tokens = commit.get("tokens")
            meta = commit.get("rollout", {}) or {}
            p = int(meta.get("prompt_length", 0))
            old = _completion_token_logprobs(rollout)
            if not tokens or p <= 0 or not old:
                logger.warning("rollout skipped: missing prompt_length or token_logprobs")
                continue
            n_completion = len(tokens) - p
            if n_completion <= 0 or n_completion != len(old):
                logger.warning("rollout skipped: log-prob length mismatch")
                continue
            keep = _completion_keep_list(rollout, p, n_completion)
            if keep is None:
                keep = [True] * n_completion
            if not any(keep):
                logger.warning("rollout skipped: force span masks all tokens")
                continue
            items.append((tokens, p, old, adv, scale, keep))
    return items


def _accumulate_grouped_grads(
    model, ref_model, plan, device, *, budget, atomic, kl_stats=None,
    behavior_model=None,
):
    """Pack the plan's rollouts into token-budget micro-batches and accumulate
    gradients. Returns ``(total_ppo, total_kl, n_processed)``."""
    items = _build_microbatch_items(plan)
    if not items:
        return 0.0, 0.0, 0
    lengths = [len(it[0]) for it in items]
    total_ppo = total_kl = 0.0
    n_processed = 0
    for idxs in _pack_by_token_budget(lengths, budget):
        sp, sk, n = _process_microbatch(
            model, ref_model, [items[i] for i in idxs], device,
            atomic=atomic, kl_stats=kl_stats,
            behavior_model=behavior_model,
        )
        total_ppo += sp
        total_kl += sk
        n_processed += n
    return total_ppo, total_kl, n_processed


# ---------------------------------------------------------------------------
# Main entry point — one GRPO step per call
# ---------------------------------------------------------------------------

def train_step(
    model,
    batches: list,
    *,
    ref_model,
    behavior_model=None,
    window_index: int | None = None,
) -> Any:
    """Run one GRPO step over the union of *batches*.

    All rollouts across every batch contribute backward calls before a
    single optimizer.step(). *batches* is a list of batches, where each
    batch is a list of group objects (ValidSubmission). Pass ``[batch]``
    for the legacy mono-batch case.

    *ref_model* is the frozen reference policy for the KL penalty. The caller
    owns its lifecycle: production uses either the rolling published
    ``verify_model`` or an explicitly pinned fixed checkpoint.

    *behavior_model*, when supplied, must be the published checkpoint miners
    generated against. It independently supplies PPO's π_old.

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
    lr_applied = float(_optimizer.param_groups[0]["lr"])

    model.train()
    device = next(model.parameters()).device

    _optimizer.zero_grad()

    n_total_rollouts = sum(len(g.rollouts) for batch in batches for g in batch)

    # Pass 1 (metadata only, no forward): group-relative advantages + per-batch
    # token-level loss weights. Each batch is one env (env_mix order, see
    # service.py), so the loss normalizes token-level *within* a batch and
    # recombines as Σ_e w_e·L_e — a long-completion env (code) cannot dominate a
    # short-completion env (math) through raw token mass. DAPO token-level
    # weighting is preserved inside each env. Degenerate groups (std==0 → zero
    # advantage) carry no signal and are dropped. Completion length comes from
    # the miner-committed token_logprobs, so this costs no extra forward.
    plan, n_skipped = _plan_from_batches(batches)

    if not plan:
        logger.info("train_step: no trainable groups")
        return model

    # Pass 2: micro-batched forward/backward (token-budget packing). The fast
    # path accumulates straight into .grad; if a micro-batch OOMs, discard the
    # partial grads and retry the whole pass with the atomic split-retry path at
    # a halved budget (atomic = an OOM mid-backward commits no gradient).
    kl_stats = _new_kl_stats()
    try:
        total_ppo, total_kl, n_processed = _accumulate_grouped_grads(
            model, ref_model, plan, device,
            budget=MICROBATCH_MAX_PADDED_TOKENS, atomic=False,
            kl_stats=kl_stats,
            behavior_model=behavior_model,
        )
    except torch.cuda.OutOfMemoryError:
        logger.warning("train_step: OOM in micro-batch — retrying atomic at halved budget")
        _optimizer.zero_grad()
        torch.cuda.empty_cache()
        kl_stats = _new_kl_stats()
        total_ppo, total_kl, n_processed = _accumulate_grouped_grads(
            model, ref_model, plan, device,
            budget=max(1, MICROBATCH_MAX_PADDED_TOKENS // 2), atomic=True,
            kl_stats=kl_stats,
            behavior_model=behavior_model,
        )

    if n_processed == 0:
        logger.info("train_step: no valid rollouts processed")
        return model

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    grad_norm_value = float(grad_norm)
    nonfinite_gradient = not math.isfinite(grad_norm_value)
    gradient_spike = (
        not nonfinite_gradient
        and grad_norm_value > GRAD_NORM_SKIP_THRESHOLD
    )
    if nonfinite_gradient or gradient_spike:
        reason = "nonfinite_gradient" if nonfinite_gradient else "gradient_spike"
        logger.warning(
            "train_step: rejected %s grad_norm=%s threshold=%s; "
            "optimizer and checkpoint cadence unchanged",
            reason,
            grad_norm_value,
            GRAD_NORM_SKIP_THRESHOLD,
        )
        failure_metrics = _kl_telemetry_metrics(kl_stats)
        failure_metrics.update({
            "train/grad_norm": grad_norm_value,
            "train/lr_applied": lr_applied,
            "train/lr_next": lr_applied,
            "train/step_skipped_nonfinite": float(nonfinite_gradient),
            "train/step_skipped_grad_spike": float(gradient_spike),
            "train/pi_old_recomputed": float(behavior_model is not None),
        })
        telemetry.log_training_step(failure_metrics, step=window_index)
        _optimizer.zero_grad(set_to_none=True)
        raise TrainingStepSkipped(reason, grad_norm_value)
    _optimizer.step()
    _scheduler.step()
    lr_next = float(_scheduler.get_last_lr()[0])

    logger.info(
        "train_step: lr_applied=%.2e lr_next=%.2e ppo=%.4f kl=%.4f "
        "grad_norm=%.3f rollouts=%d/%d",
        lr_applied, lr_next, total_ppo / n_processed, total_kl / n_processed,
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
        # Keep ``train/lr`` as the historical post-scheduler value while
        # exposing the rate that actually produced this update explicitly.
        "train/lr": lr_next,
        "train/lr_applied": lr_applied,
        "train/lr_next": lr_next,
        "train/ppo_loss": total_ppo / n_processed,
        "train/kl": total_kl / n_processed,
        "train/grad_norm": float(grad_norm),
        "train/grad_clip_ratio": float(grad_norm) / GRAD_CLIP_NORM,
        "train/grad_was_clipped": float(grad_norm > GRAD_CLIP_NORM),
        "train/step_skipped_nonfinite": 0.0,
        "train/step_skipped_grad_spike": 0.0,
        "train/pi_old_recomputed": float(behavior_model is not None),
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
    metrics.update(_kl_telemetry_metrics(kl_stats))
    metrics.update(_bft_training_metrics(plan))
    telemetry.log_training_step(metrics, step=window_index)

    return model
