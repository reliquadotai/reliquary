"""GRAIL proof verification — primitives used by GrpoWindowBatcher.

The orchestration lives in `reliquary.validator.batcher`. This module only
exposes the per-commit checks that touch the model or the signature scheme.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from reliquary.constants import (
    CHALLENGE_K,
    LAYER_INDEX,
    MIN_EOS_PROBABILITY,
    T_PROTO,
)

logger = logging.getLogger(__name__)


@dataclass
class ProofResult:
    """Return value of verify_commitment_proofs.

    ``has_sparse_outputs`` discriminates the production path from legacy
    test stubs. When True, the sparse fields below carry the validator's
    precomputed values from the forward pass — used by the behavioural
    checks (termination / logprob / distribution) instead of round-tripping
    the full logits tensor through PCIe. When False the batcher skips
    behavioural checks; this preserves the prior contract under which a
    stub returning empty logits opted the rollout out of behavioural
    enforcement.

    ``sketch_diff_max`` is the worst per-position |miner_sketch -
    validator_sketch| across the K sketch-challenge positions, surfaced
    for post-hoc threshold calibration even when the proof passed the
    current tolerance.
    """

    all_passed: bool
    passed: int
    checked: int
    sketch_diff_max: int = 0
    has_sparse_outputs: bool = False
    # Accepted for backwards-compatibility with stubs that constructed
    # ``ProofResult(..., logits=torch.empty(0))`` before the keep-on-GPU
    # refactor. Production never populates this — sparse fields below
    # carry everything the behavioural checks need. Default is a tiny
    # empty tensor; stubs that opt into behavioural enforcement should
    # set has_sparse_outputs=True and populate the sparse fields.
    logits: Any = field(default_factory=lambda: torch.empty(0))
    # Termination: EOS probability mass at logits[seq_len - 2]. None when
    # the model has no eos_token_id configured or seq_len < 2.
    p_stop: float | None = None
    # Logprob challenge: absolute token positions sampled by
    # indices_from_root_in_range, paired with the validator's
    # log-softmax(logits[idx - 1])[tokens[idx]] for each. Empty when
    # completion_length < CHALLENGE_K or any sampled index is out of
    # range — the logprob check treats that as a deterministic fail.
    challenge_lp_indices: list[int] = field(default_factory=list)
    challenge_lp_values: list[float] = field(default_factory=list)
    # Distribution check: chosen-token probability under T_PROTO at each
    # valid completion-producing position. One float per (t-1, t) pair
    # where prompt_length <= t < prompt_length + completion_length. May
    # be shorter than completion_length when boundary positions are
    # skipped (t == 0, t - 1 >= seq_len, t >= len(tokens)).
    completion_chosen_probs: list[float] = field(default_factory=list)


def verify_signature(commit: dict, hotkey: str) -> bool:
    """Hard check: verify Ed25519 signature on commit binding."""
    from reliquary.protocol.signatures import verify_commit_signature

    return verify_commit_signature(commit, hotkey)


def _eos_set_from_model(model: Any, tokenizer: Any) -> set[int]:
    """Resolve the EOS token set the way termination/p_stop both expect.

    Tries ``model.generation_config.eos_token_id`` first (production
    Qwen3 ships [151645, 151643] there), then falls back to the
    tokenizer's ``eos_token_id``. Returns an empty set when nothing is
    declared — callers should treat that as "no EOS gate available".
    """
    eos_ids: Any = None
    gen_cfg = getattr(model, "generation_config", None) if model is not None else None
    if gen_cfg is not None:
        eos_ids = getattr(gen_cfg, "eos_token_id", None)
    if eos_ids is None:
        eos_ids = getattr(tokenizer, "eos_token_id", None)
    if eos_ids is None:
        return set()
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    return {int(e) for e in eos_ids if e is not None}


def verify_termination(
    commit: dict,
    tokenizer: Any,
    proof: "ProofResult | None" = None,
    model: Any = None,
) -> bool:
    """Two paths to a valid termination, both gaming-safe:

    Path 1 — max-length termination: total token sequence (prompt +
    completion) reached the network-wide protocol cap
    ``MAX_NEW_TOKENS_PROTOCOL_CAP``. The miner ran out of context window.
    We check the *total* length rather than ``completion_length`` alone
    because honest miners running under a ``max_model_len`` ceiling
    (e.g. vLLM, where prompt + generation ≤ max_model_len) can never
    satisfy ``completion_length ≥ cap``.

    Path 2 — natural EOS termination: ``tokens[-1]`` is one of the
    configured stop tokens AND its probability mass at the previous
    position's softmax (``p_stop``) is at least ``MIN_EOS_PROBABILITY``.
    The probability gate catches sampler-forced stops at near-zero
    probability that wouldn't pass an honest decode. ``p_stop`` is
    precomputed on GPU by ``verify_commitment_proofs`` and carried on
    ``proof`` — there's no per-call softmax on a CPU logits tensor.
    """
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    tokens = commit["tokens"]
    rollout_meta = commit.get("rollout", {}) or {}
    completion_length = int(rollout_meta.get("completion_length", 0))
    prompt_length = int(rollout_meta.get("prompt_length", 0))

    if prompt_length + completion_length >= MAX_NEW_TOKENS_PROTOCOL_CAP:
        return True

    eos_set = _eos_set_from_model(model, tokenizer)
    total_length = prompt_length + completion_length
    if not eos_set:
        logger.warning(
            "termination_fail reason=no_eos_set prompt_length=%d "
            "completion_length=%d total=%d cap=%d",
            prompt_length, completion_length, total_length,
            MAX_NEW_TOKENS_PROTOCOL_CAP,
        )
        return False

    last_tok = int(tokens[-1])
    in_eos = last_tok in eos_set
    p_stop = proof.p_stop if proof is not None else None
    if p_stop is None:
        logger.warning(
            "termination_fail reason=no_p_stop prompt_length=%d "
            "completion_length=%d total=%d cap=%d last_token=%d",
            prompt_length, completion_length, total_length,
            MAX_NEW_TOKENS_PROTOCOL_CAP, last_tok,
        )
        return False

    ok = in_eos and p_stop >= MIN_EOS_PROBABILITY
    if not ok:
        logger.warning(
            "termination_fail prompt_length=%d completion_length=%d "
            "total=%d cap=%d last_token=%d in_eos=%s p_stop=%.5f "
            "min_p=%.3f eos_set=%s",
            prompt_length, completion_length, total_length,
            MAX_NEW_TOKENS_PROTOCOL_CAP,
            last_tok, in_eos, p_stop, MIN_EOS_PROBABILITY, sorted(eos_set),
        )
    return ok


def is_cap_truncation(
    commit: dict,
    tokenizer: Any,
    proof: "ProofResult | None" = None,
    model: Any = None,
) -> bool:
    """Return True when a cap-hit rollout did not naturally stop on EOS.

    ``verify_termination`` accepts the protocol cap path so one honest runaway
    rollout can remain usable. The batcher still needs to count those cap hits
    as truncations when the EOS probability gate did not pass, otherwise a
    miner can force every rollout to max length and bypass the existing
    per-submission truncation budget.
    """
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    rollout_meta = commit.get("rollout", {}) or {}
    completion_length = int(rollout_meta.get("completion_length", 0))
    prompt_length = int(rollout_meta.get("prompt_length", 0))
    if prompt_length + completion_length < MAX_NEW_TOKENS_PROTOCOL_CAP:
        return False

    eos_set = _eos_set_from_model(model, tokenizer)
    if not eos_set:
        return True

    tokens = commit.get("tokens") or []
    if not tokens:
        return True

    p_stop = proof.p_stop if proof is not None else None
    return not (
        int(tokens[-1]) in eos_set
        and p_stop is not None
        and p_stop >= MIN_EOS_PROBABILITY
    )


def has_eos_padding(
    commit: dict,
    tokenizer: Any,
    model: Any = None,
) -> bool:
    """Return True when completion tokens continue after an EOS token.

    Honest generation should stop at the first EOS; the reference miner also
    truncates there before building the proof. Keeping extra EOS tokens (or any
    tokens after EOS) manufactures long, high-probability tails that satisfy
    the logprob/distribution checks while poisoning training with stop-token
    padding.
    """
    eos_set = _eos_set_from_model(model, tokenizer)
    if not eos_set:
        return False

    tokens = list(commit.get("tokens") or [])
    rollout_meta = commit.get("rollout", {}) or {}
    prompt_length = int(rollout_meta.get("prompt_length", 0))
    completion_length = int(rollout_meta.get("completion_length", 0))
    completion = tokens[prompt_length: prompt_length + completion_length]
    eos_positions = [
        idx for idx, token in enumerate(completion) if int(token) in eos_set
    ]
    if not eos_positions:
        return False
    return len(eos_positions) > 1 or eos_positions[0] != len(completion) - 1


def verify_commitment_proofs(
    commit: dict,
    model: Any,
    window_randomness: str,
    *,
    tokenizer: Any = None,
) -> ProofResult:
    """Hard check: verify GRAIL sketch commitments against the model
    forward pass, AND precompute the sparse values the behavioural
    checks consume downstream.

    The body forward runs once on GPU. The lm_head runs once on GPU.
    Everything the behavioural checks need (p_stop for termination, the
    validator's logprob at K logprob-challenge positions, the chosen-
    token probability under T_PROTO at every completion position) is
    computed on GPU and transferred to CPU as a handful of floats —
    NOT as a [seq_len, vocab] tensor (which would dominate the wall-clock
    cost via PCIe). Only the per-position hidden states needed by the
    sketch verification move to CPU as a [seq_len, hidden_dim] tensor,
    which is two orders of magnitude smaller than the logits would be.
    """
    from reliquary.protocol.crypto import (
        indices_from_root, indices_from_root_in_range,
    )
    from reliquary.protocol.grail_verifier import GRAILVerifier
    from reliquary.shared.forward import forward_single_layer
    from reliquary.shared.hf_compat import resolve_hidden_size

    tokens = commit["tokens"]
    commitments = commit["commitments"]
    rollout_meta = commit.get("rollout", {}) or {}
    prompt_length = int(rollout_meta.get("prompt_length", 0))
    completion_length = int(rollout_meta.get("completion_length", 0))

    seq_len = len(tokens)

    # SECURITY: Always use the validator's independently-computed randomness.
    # A miner who controls the randomness can predict which positions are
    # challenged and only forge those.
    randomness = window_randomness

    hidden_dim = resolve_hidden_size(model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness)

    expected_challenges = min(CHALLENGE_K, seq_len)
    challenge_indices = indices_from_root(
        tokens, randomness, seq_len, expected_challenges
    )

    device = next(model.parameters()).device
    input_ids = torch.tensor([tokens], device=device)
    with torch.no_grad():
        hidden_states_gpu, logits_batch = forward_single_layer(
            model, input_ids, None, LAYER_INDEX
        )

    hidden_states_gpu = hidden_states_gpu[0]  # [seq_len, hidden_dim]
    logits_gpu = logits_batch[0]  # [seq_len, vocab_size], kept on GPU

    p_stop = _gpu_p_stop(
        logits_gpu, seq_len, _eos_set_from_model(model, tokenizer), device,
    )
    challenge_lp_indices, challenge_lp_values = _gpu_challenge_logprobs(
        logits_gpu, tokens, prompt_length, completion_length, randomness, device,
    )
    completion_chosen_probs = _gpu_completion_chosen_probs(
        logits_gpu, tokens, prompt_length, completion_length, seq_len, device,
    )

    hidden_states = hidden_states_gpu.detach().to("cpu")

    passed = 0
    checked = 0
    sketch_diff_max = 0
    for idx in challenge_indices:
        if idx >= seq_len:
            continue
        checked += 1
        miner_commit = commitments[idx]
        validator_hidden = hidden_states[idx]
        valid, diag = verifier.verify_commitment(
            validator_hidden, miner_commit, r_vec, seq_len, idx
        )
        sketch_diff = int((diag or {}).get("sketch_diff", 0))
        if sketch_diff > sketch_diff_max:
            sketch_diff_max = sketch_diff
        if valid:
            passed += 1

    # SECURITY: All expected challenge positions must be checked and pass.
    # A miner cannot benefit from having fewer positions verified.
    all_passed = passed == checked and checked >= expected_challenges
    return ProofResult(
        all_passed=all_passed,
        passed=passed,
        checked=checked,
        sketch_diff_max=sketch_diff_max,
        has_sparse_outputs=True,
        p_stop=p_stop,
        challenge_lp_indices=challenge_lp_indices,
        challenge_lp_values=challenge_lp_values,
        completion_chosen_probs=completion_chosen_probs,
    )


def _gpu_p_stop(
    logits_gpu: torch.Tensor,
    seq_len: int,
    eos_set: set[int],
    device: Any,
) -> float | None:
    if seq_len < 2 or not eos_set:
        return None
    probs_last = torch.softmax(logits_gpu[seq_len - 2].float(), dim=-1)
    eos_idx_tensor = torch.tensor(
        sorted(eos_set), device=device, dtype=torch.long,
    )
    return float(probs_last[eos_idx_tensor].sum().item())


def _gpu_challenge_logprobs(
    logits_gpu: torch.Tensor,
    tokens: list[int],
    prompt_length: int,
    completion_length: int,
    randomness: str,
    device: Any,
) -> tuple[list[int], list[float]]:
    """Recompute the validator's log-prob at each logprob-challenge index.

    Returns ``(indices, values)`` of equal length. Both empty when the
    completion is too short to sample CHALLENGE_K positions, when the
    sampler returns fewer than K indices (defensive), or when any
    sampled position would read out-of-range — the logprob check treats
    that as a fail at the call site.
    """
    from reliquary.protocol.crypto import indices_from_root_in_range

    if completion_length < CHALLENGE_K:
        return [], []
    challenge_idxs = indices_from_root_in_range(
        tokens, randomness,
        prompt_length, prompt_length + completion_length,
        CHALLENGE_K,
    )
    if len(challenge_idxs) != CHALLENGE_K:
        return [], []

    positions = [i - 1 for i in challenge_idxs]
    seq_len = logits_gpu.size(0)
    if any(p < 0 or p >= seq_len for p in positions):
        return [], []

    pos_tensor = torch.tensor(positions, device=device, dtype=torch.long)
    tok_tensor = torch.tensor(
        [tokens[i] for i in challenge_idxs], device=device, dtype=torch.long,
    )
    selected = logits_gpu[pos_tensor].float()
    log_probs = torch.log_softmax(selected, dim=-1)
    chosen = log_probs.gather(1, tok_tensor.unsqueeze(1)).squeeze(1)
    return list(challenge_idxs), chosen.tolist()


def _gpu_completion_chosen_probs(
    logits_gpu: torch.Tensor,
    tokens: list[int],
    prompt_length: int,
    completion_length: int,
    seq_len: int,
    device: Any,
) -> list[float]:
    """Compute chosen-token probability under T_PROTO at each valid
    completion-producing position, on GPU, vectorised.

    Mirrors the per-step loop in the legacy CPU implementation: for each
    ``t in [prompt_length, prompt_length + completion_length)``, skip
    when ``t == 0`` or ``t - 1`` lies outside ``logits``'s range or
    ``t`` lies outside ``tokens``. The remaining (t-1, tokens[t]) pairs
    are gathered in a single softmax + gather pass and shipped to CPU
    as one Python float per surviving step.
    """
    if completion_length <= 0:
        return []
    t_start = prompt_length
    t_end = min(prompt_length + completion_length, len(tokens), seq_len + 1)
    valid_t = [t for t in range(t_start, t_end) if t > 0 and t - 1 < seq_len]
    if not valid_t:
        return []

    pos_tensor = torch.tensor(
        [t - 1 for t in valid_t], device=device, dtype=torch.long,
    )
    tok_tensor = torch.tensor(
        [tokens[t] for t in valid_t], device=device, dtype=torch.long,
    )
    scaled = logits_gpu[pos_tensor].float() / float(T_PROTO)
    probs = scaled.softmax(dim=-1)
    chosen = probs.gather(1, tok_tensor.unsqueeze(1)).squeeze(1)
    return chosen.tolist()


def verify_reward_claim(
    env: Any,
    problem: dict,
    completion_text: str,
    claimed: float,
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Re-compute the env's reward on *completion_text* and compare to *claimed*.

    Miners declare the reward of each completion in their submission (saves
    validator compute when they can pre-filter out-of-zone) but the validator
    re-runs ``env.compute_reward`` to check honesty. A mismatch means the
    miner lied about reward, warranting rejection.

    Returns True iff |env_reward - claimed| <= tolerance. The small tolerance
    absorbs float64 formatting round-trip (JSON serialisation) noise.
    """
    try:
        actual = env.compute_reward(problem, completion_text)
    except Exception:
        return False
    return abs(float(actual) - float(claimed)) <= tolerance


def rewards_std(rewards: list[float]) -> float:
    """Population standard deviation of a rollout group's rewards.

    Returns 0.0 for empty or single-element lists (degenerate — no
    information). The population formula (divide by n, not n-1) is
    used because we want the std of THIS specific sample, not an
    estimator of the underlying distribution's std.
    """
    n = len(rewards)
    if n < 2:
        return 0.0
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    return variance ** 0.5


def binary_reward_correct_count(
    rewards: list[float],
    *,
    tolerance: float = 1e-6,
) -> int | None:
    """Return k for binary {0,1} reward groups, else None.

    OpenMathInstruct rewards are binary today. Keeping this as a helper lets
    the validator apply a stricter middle-frontier gate only when the reward
    shape is actually binary, while leaving future continuous-reward
    environments on the sigma filter alone.
    """
    correct = 0
    for reward in rewards:
        value = float(reward)
        if abs(value - 1.0) <= tolerance:
            correct += 1
        elif abs(value) <= tolerance:
            continue
        else:
            return None
    return correct


def binary_reward_mix_in_frontier(rewards: list[float]) -> bool:
    """True if a binary reward group is inside the middle frontier band.

    For M=8, SIGMA_MIN=0.43 admits k=2 and k=6 edge groups. Those groups are
    exploitable by local reward-oracle selection and produce imbalanced GRPO
    labels. This helper keeps binary groups in the steadier k=3..5 band.

    Non-binary groups return True here and remain governed by the sigma gate.
    """
    from reliquary.constants import (
        BINARY_REWARD_MAX_CORRECT,
        BINARY_REWARD_MIN_CORRECT,
    )

    correct = binary_reward_correct_count(rewards)
    if correct is None:
        return True
    return BINARY_REWARD_MIN_CORRECT <= correct <= BINARY_REWARD_MAX_CORRECT


def is_in_zone(sigma: float, *, bootstrap: bool = False) -> bool:
    """True iff *sigma* exceeds the minimum threshold for training signal.

    Steady state: σ ≥ SIGMA_MIN (0.43).
    Bootstrap: σ ≥ BOOTSTRAP_SIGMA_MIN (0.33).

    A group with σ below this is dropped because its rollouts cluster
    too tightly for the normalised advantage (r - μ) / σ to carry a
    usable gradient signal.
    """
    from reliquary.constants import BOOTSTRAP_SIGMA_MIN, SIGMA_MIN

    if sigma < 1e-8:
        return False   # degenerate
    return sigma >= (BOOTSTRAP_SIGMA_MIN if bootstrap else SIGMA_MIN)


def verify_logprobs_claim(
    tokens: list[int],
    prompt_length: int,
    completion_length: int,
    claimed_logprobs: list[float],
    proof: "ProofResult",
) -> tuple[bool, float]:
    """Hard check: validate miner-claimed per-token logprobs against the
    validator's precomputed log-probs at K=CHALLENGE_K challenged
    positions.

    For each challenge position ``i`` carried on ``proof``, compute
    ``dev_i = exp(|validator_lp - miner_lp|) - 1`` and compare the
    **median** across the K positions against ``LOGPROB_IS_EPS``.

    Median (not mean) is robust to the bf16 outliers honest miners see
    on cross-GPU runs.

    ``claimed_logprobs`` accepts two layouts:
    - Full-sequence (length == len(tokens)): prompt positions ignored,
      completion positions read directly by absolute index.
    - Completion-only (length == completion_length): position-j entry
      corresponds to absolute index ``prompt_length + j``.

    Returns ``(is_valid, median_dev)``. ``median_dev`` is ``inf`` when
    the check cannot be executed (completion too short, malformed
    payload, or the proof carries no challenge values).
    """
    import math
    from statistics import median

    from reliquary.constants import LOGPROB_IS_EPS

    if completion_length < CHALLENGE_K:
        return False, float("inf")
    if not proof.challenge_lp_indices or not proof.challenge_lp_values:
        return False, float("inf")
    if len(proof.challenge_lp_indices) != CHALLENGE_K:
        return False, float("inf")
    if len(proof.challenge_lp_values) != CHALLENGE_K:
        return False, float("inf")

    if len(claimed_logprobs) == len(tokens):
        def miner_lp_at(abs_idx: int) -> float:
            return float(claimed_logprobs[abs_idx])
    elif len(claimed_logprobs) == completion_length:
        def miner_lp_at(abs_idx: int) -> float:
            return float(claimed_logprobs[abs_idx - prompt_length])
    else:
        return False, float("inf")

    devs: list[float] = []
    for abs_idx, model_lp in zip(
        proof.challenge_lp_indices, proof.challenge_lp_values
    ):
        devs.append(math.exp(abs(float(model_lp) - miner_lp_at(abs_idx))) - 1.0)

    median_dev = float(median(devs))
    return median_dev <= LOGPROB_IS_EPS, median_dev


def evaluate_token_distribution(
    tokens: list[int],
    prompt_length: int,
    completion_length: int,
    proof: "ProofResult",
) -> tuple[bool | None, dict]:
    """Soft check: detect suspicious chosen-token probability distributions.

    Reads ``proof.completion_chosen_probs`` — the validator's per-step
    probability of the token the miner emitted, computed on GPU during
    the forward pass — and compares summary stats against the
    SAMPLING_* thresholds.

    Returns ``(is_valid, metrics)``:
      - ``True``   — distribution is consistent with sampling from the
                     validator's model at T_PROTO
      - ``False``  — suspicious (median or q10 collapsed below threshold)
      - ``None``   — insufficient steps (< SAMPLING_MIN_STEPS) — caller
                     defaults to accept

    ``metrics`` carries ``mean``, ``median``, ``q10``, ``low_frac``,
    ``high_frac`` regardless of the decision (empty dict only when
    there's insufficient data).
    """
    import numpy as np

    from reliquary.constants import (
        SAMPLING_HIGH_P,
        SAMPLING_LOW_P,
        SAMPLING_LOW_Q10_MAX,
        SAMPLING_MEDIAN_LOW_MAX,
        SAMPLING_MIN_STEPS,
    )

    if completion_length < SAMPLING_MIN_STEPS:
        return None, {}
    probs = proof.completion_chosen_probs
    if len(probs) < SAMPLING_MIN_STEPS:
        return None, {}

    x = np.asarray(probs, dtype=np.float64)
    metrics = {
        "mean":      float(x.mean()),
        "median":    float(np.median(x)),
        "q10":       float(np.quantile(x, 0.10)),
        "low_frac":  float((x <= SAMPLING_LOW_P).mean()),
        "high_frac": float((x >= SAMPLING_HIGH_P).mean()),
    }

    suspicious = (
        metrics["median"] < SAMPLING_MEDIAN_LOW_MAX
        or metrics["q10"] < SAMPLING_LOW_Q10_MAX
    )
    return (not suspicious), metrics
