"""GRAIL proof verification — primitives used by GrpoWindowBatcher.

The orchestration lives in `reliquary.validator.batcher`. This module only
exposes the per-commit checks that touch the model or the signature scheme.
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from reliquary.constants import (
    CHALLENGE_K,
    LAYER_INDEX,
    MIN_EOS_PROBABILITY,
    T_PROTO,
)
from reliquary.shared.modeling import resolve_eos_token_ids

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CodeSemanticSpan:
    start: int
    end: int
    label: str


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
    # Token authenticity: argmax probability and argmax token id under T_PROTO,
    # aligned 1:1 with completion_chosen_probs (same surviving steps).
    completion_argmax_probs: list[float] = field(default_factory=list)
    completion_argmax_ids: list[int] = field(default_factory=list)


def verify_signature(commit: dict, hotkey: str) -> bool:
    """Hard check: verify Ed25519 signature on commit binding."""
    from reliquary.protocol.signatures import verify_commit_signature

    return verify_commit_signature(commit, hotkey)


def _eos_set_from_model(model: Any, tokenizer: Any) -> set[int]:
    """Resolve all stop tokens used by termination and p_stop checks."""
    return resolve_eos_token_ids(model, tokenizer)


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
    (
        completion_chosen_probs,
        completion_argmax_probs,
        completion_argmax_ids,
    ) = _gpu_completion_token_stats(
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
        completion_argmax_probs=completion_argmax_probs,
        completion_argmax_ids=completion_argmax_ids,
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


def _gpu_completion_token_stats(
    logits_gpu: torch.Tensor,
    tokens: list[int],
    prompt_length: int,
    completion_length: int,
    seq_len: int,
    device: Any,
) -> tuple[list[float], list[float], list[int]]:
    """Per completion-producing position under T_PROTO, on GPU, vectorised:
    chosen-token prob, argmax prob, argmax token id. The three lists are
    aligned 1:1. Boundary positions (t == 0, t - 1 >= seq_len, t >= len(tokens))
    are skipped identically across all three.
    """
    if completion_length <= 0:
        return [], [], []
    t_start = prompt_length
    t_end = min(prompt_length + completion_length, len(tokens), seq_len + 1)
    valid_t = [t for t in range(t_start, t_end) if t > 0 and t - 1 < seq_len]
    if not valid_t:
        return [], [], []

    pos_tensor = torch.tensor(
        [t - 1 for t in valid_t], device=device, dtype=torch.long,
    )
    tok_tensor = torch.tensor(
        [tokens[t] for t in valid_t], device=device, dtype=torch.long,
    )
    scaled = logits_gpu[pos_tensor].float() / float(T_PROTO)
    probs = scaled.softmax(dim=-1)
    chosen = probs.gather(1, tok_tensor.unsqueeze(1)).squeeze(1)
    amax_probs, amax_ids = probs.max(dim=-1)
    return chosen.tolist(), amax_probs.tolist(), amax_ids.tolist()


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


def is_in_zone(sigma: float, *, bootstrap: bool = False) -> bool:
    """True iff *sigma* exceeds the minimum threshold for training signal.

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
    *,
    exempt_positions: set[int] | None = None,
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
    if exempt_positions:
        probs = [p for i, p in enumerate(probs) if i not in exempt_positions]
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


def validate_force_span(
    tokens: list[int],
    rollout_meta: dict,
    canonical_force_ids: list[int],
    prompt_length: int,
    *,
    thinking_budget: int,
    think_close_ids: set[int],
) -> tuple[bool, set[int]]:
    """BFT carve-out gate. For a forced rollout, verify the declared
    ``force_span``:
      * content  — byte-exactly the canonical FORCE ids (which begin with the
        atomic ``</think>`` id);
      * position — starts exactly ``thinking_budget`` tokens into the completion;
      * honesty  — no atomic ``</think>`` appears before it.

    Returns ``(ok, exempt)`` where ``exempt`` is the set of completion-relative
    positions to skip in the per-token authenticity / distribution checks. A
    non-forced rollout is ``(True, set())`` (no carve); an invalid span is
    ``(False, set())``.
    """
    if not rollout_meta.get("forced"):
        return True, set()
    span = rollout_meta.get("force_span")
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return False, set()
    start, end = int(span[0]), int(span[1])
    if not (prompt_length <= start < end <= len(tokens)):
        return False, set()
    if start - prompt_length != int(thinking_budget):
        return False, set()
    if any(int(t) in think_close_ids for t in tokens[prompt_length:start]):
        return False, set()
    if list(tokens[start:end]) != list(canonical_force_ids):
        return False, set()
    return True, set(range(start - prompt_length, end - prompt_length))


def evaluate_token_authenticity(
    proof: "ProofResult",
    *,
    threshold: float | None = None,
    argmax_conf: float | None = None,
    tokens: list[int] | None = None,
    prompt_length: int = 0,
    completion_length: int = 0,
    tokenizer: Any = None,
    numeric_threshold: float | None = None,
    exempt_positions: set[int] | None = None,
) -> tuple[bool, dict]:
    """Hard check on the GPU-forward token probs. Two collapse signatures:
    (1) any token below ``threshold`` is a gross injection;
    (2) an answer-bearing numeric token (digit, or sign/decimal adjacent to a
    digit) below ``numeric_threshold`` while argmax >= ``argmax_conf`` is an
    edited value (the arithmetic-result edit the boxed check misses). The
    numeric pass runs only when ``tokens`` and ``tokenizer`` are supplied.
    ``ok=True`` when no stats are available.
    """
    from reliquary.constants import TOKEN_AUTH_ARGMAX_CONF, TOKEN_AUTH_THRESHOLD

    if threshold is None:
        threshold = TOKEN_AUTH_THRESHOLD
    if argmax_conf is None:
        argmax_conf = TOKEN_AUTH_ARGMAX_CONF
    if numeric_threshold is None:
        numeric_threshold = NUMERIC_AUTH_THRESHOLD
    chosen = proof.completion_chosen_probs
    amax = proof.completion_argmax_probs
    if not chosen or not amax:
        return True, {}
    exempt = exempt_positions or set()
    comp = digit_ids = sign_ids = None
    if tokens is not None and tokenizer is not None and completion_length > 0:
        comp = list(tokens[prompt_length: prompt_length + completion_length])
        digit_ids, sign_ids = _numeric_token_ids(tokenizer)
    n = min(len(chosen), len(amax))
    for j in range(n):
        if j in exempt:
            continue
        if chosen[j] < threshold:
            ids = proof.completion_argmax_ids
            return False, {
                "pos": j,
                "p_chosen": float(chosen[j]),
                "p_argmax": float(amax[j]),
                "argmax_id": (ids[j] if j < len(ids) else None),
            }
        if comp is not None and j < len(comp):
            t = comp[j]
            numeric = t in digit_ids or (
                t in sign_ids
                and (
                    (j > 0 and comp[j - 1] in digit_ids)
                    or (j + 1 < len(comp) and comp[j + 1] in digit_ids)
                )
            )
            if numeric and chosen[j] < numeric_threshold and amax[j] >= argmax_conf:
                return False, {
                    "pos": j,
                    "kind": "numeric",
                    "p_chosen": float(chosen[j]),
                    "p_argmax": float(amax[j]),
                    "token_id": int(t),
                }
    return True, {}


def evaluate_all_token_auth_shadow(
    proof: "ProofResult",
    *,
    threshold: float | None = None,
    argmax_conf: float | None = None,
    tokens: list[int] | None = None,
    prompt_length: int = 0,
    completion_length: int = 0,
    tokenizer: Any = None,
    include_findings: bool = False,
    max_findings: int | None = None,
    context_chars: int = 80,
    exempt_positions: set[int] | None = None,
) -> tuple[bool, dict]:
    """Aggregate all-token argmax-gated authenticity shadow check.

    This observes the broader detector proposed for plausible token edits:
    a chosen completion token below ``threshold`` is suspicious only when the
    validator model was highly confident in a different argmax. It returns
    aggregate metrics by default; callers can archive counts/minima without
    exposing exact positions in public R2 data. Validator-private callers may
    opt into ``finding_details`` for local forensics.
    """
    from reliquary.constants import (
        ALL_TOKEN_AUTH_SHADOW_ARGMAX_CONF,
        ALL_TOKEN_AUTH_SHADOW_THRESHOLD,
    )

    if threshold is None:
        threshold = ALL_TOKEN_AUTH_SHADOW_THRESHOLD
    if argmax_conf is None:
        argmax_conf = ALL_TOKEN_AUTH_SHADOW_ARGMAX_CONF
    chosen = proof.completion_chosen_probs
    amax = proof.completion_argmax_probs
    if not chosen or not amax:
        return True, {}

    n = min(len(chosen), len(amax))
    if n <= 0:
        return True, {}

    exempt = exempt_positions or set()
    min_prob: float | None = None
    findings = 0
    finding_min_prob: float | None = None
    finding_details: list[dict[str, Any]] = []
    completion_tokens: list[int] = []
    argmax_ids = proof.completion_argmax_ids
    if tokens is not None and completion_length > 0:
        completion_tokens = list(
            tokens[prompt_length: prompt_length + completion_length]
        )
    for j in range(n):
        if j in exempt:
            continue
        p = float(chosen[j])
        if min_prob is None or p < min_prob:
            min_prob = p
        if p < threshold and float(amax[j]) >= argmax_conf:
            findings += 1
            if finding_min_prob is None or p < finding_min_prob:
                finding_min_prob = p
            if include_findings and (
                max_findings is None or len(finding_details) < max_findings
            ):
                token_id = (
                    int(completion_tokens[j])
                    if j < len(completion_tokens)
                    else None
                )
                argmax_id = (
                    int(argmax_ids[j])
                    if j < len(argmax_ids)
                    else None
                )
                finding_details.append({
                    "completion_pos": j,
                    "absolute_token_pos": prompt_length + j,
                    "p_chosen": p,
                    "p_argmax": float(amax[j]),
                    "token_id": token_id,
                    "token_text": _safe_decode_ids(tokenizer, [token_id]),
                    "argmax_id": argmax_id,
                    "argmax_text": _safe_decode_ids(tokenizer, [argmax_id]),
                    "completion_context": _completion_context_for_position(
                        completion_tokens,
                        j,
                        tokenizer,
                        context_chars=context_chars,
                    ),
                })

    metrics = {
        "n_tokens": n,
        "min_prob": min_prob,
        "threshold": float(threshold),
        "argmax_conf": float(argmax_conf),
        "findings": findings,
        "finding_min_prob": finding_min_prob,
    }
    if include_findings:
        metrics["finding_details"] = finding_details
    return (findings == 0), metrics


def _safe_decode_ids(tokenizer: Any, ids: list[int | None]) -> str | None:
    if tokenizer is None or any(t is None for t in ids):
        return None
    try:
        return str(tokenizer.decode(
            [int(t) for t in ids],
            skip_special_tokens=False,
        ))
    except TypeError:
        try:
            return str(tokenizer.decode([int(t) for t in ids]))
        except Exception:
            return None
    except Exception:
        return None


def _completion_context_for_position(
    completion_tokens: list[int],
    pos: int,
    tokenizer: Any,
    *,
    context_chars: int,
) -> str | None:
    if tokenizer is None or not completion_tokens or pos >= len(completion_tokens):
        return None
    radius = 32
    left = _safe_decode_ids(
        tokenizer,
        completion_tokens[max(0, pos - radius): pos],
    ) or ""
    token = _safe_decode_ids(tokenizer, [completion_tokens[pos]]) or ""
    right = _safe_decode_ids(
        tokenizer,
        completion_tokens[pos + 1: pos + 1 + radius],
    ) or ""
    chars = max(0, int(context_chars))
    return f"{left[-chars:]}{token}{right[:chars]}"


_CODE_FENCE_RE = re.compile(
    r"(```|~~~)(?:python3?|py)?\s*\n(.*?)\n\1",
    re.DOTALL,
)
_COMPARE_OP_RE = re.compile(
    r"\bnot\s+in\b|\bis\s+not\b|==|!=|<=|>=|<|>|\bis\b|\bin\b"
)
_BINOP_RE = re.compile(r"\*\*|//|<<|>>|[@+\-*/%&|^]")
_BOOLOP_RE = re.compile(r"\b(?:and|or)\b")
_UNARYOP_RE = re.compile(r"\bnot\b|[+\-~]")


def _extract_python_span(completion: str) -> tuple[str, int, int] | None:
    """Return ``(code, start, end)`` for the last Python code block.

    Falls back to the raw completion only when it parses as Python. OpenCode
    completions normally fence the answer, but accepting raw code keeps this
    shadow check useful for simpler local tests and miner outputs.
    """
    if not completion:
        return None
    matches = list(_CODE_FENCE_RE.finditer(completion))
    if matches:
        m = matches[-1]
        return m.group(2), m.start(2), m.end(2)
    try:
        ast.parse(completion)
    except SyntaxError:
        return None
    return completion, 0, len(completion)


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _node_span(
    source: str,
    offsets: list[int],
    node: ast.AST,
) -> tuple[int, int] | None:
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    col = getattr(node, "col_offset", None)
    end_col = getattr(node, "end_col_offset", None)
    if None in (lineno, end_lineno, col, end_col):
        return None
    if (
        lineno < 1
        or end_lineno < 1
        or lineno >= len(offsets)
        or end_lineno >= len(offsets)
    ):
        return None
    start = offsets[lineno - 1] + int(col)
    end = offsets[end_lineno - 1] + int(end_col)
    if 0 <= start < end <= len(source):
        return start, end
    return None


def _add_span(
    spans: list[_CodeSemanticSpan],
    seen: set[tuple[int, int, str]],
    start: int,
    end: int,
    label: str,
) -> None:
    if start < 0 or end <= start:
        return
    key = (start, end, label)
    if key not in seen:
        seen.add(key)
        spans.append(_CodeSemanticSpan(start, end, label))


def _add_between_operator_span(
    source: str,
    spans: list[_CodeSemanticSpan],
    seen: set[tuple[int, int, str]],
    start: int,
    end: int,
    pattern: re.Pattern,
    label: str,
) -> None:
    if end <= start:
        return
    fragment = source[start:end]
    match = pattern.search(fragment)
    if match is None:
        return
    _add_span(spans, seen, start + match.start(), start + match.end(), label)


def _constant_label(value: Any) -> str | None:
    if isinstance(value, bool):
        return "constant:bool"
    if value is None:
        return "constant:none"
    if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
        return "constant:number"
    if isinstance(value, str):
        return "constant:string"
    return None


def _code_semantic_spans(source: str) -> list[_CodeSemanticSpan]:
    """AST-sensitive spans where a tiny edit can flip OpenCode reward.

    These spans are deliberately broader than a final enforcement policy should
    be; the caller uses them for shadow telemetry first. Calibration can then
    decide which labels are safe to hard-reject.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    offsets = _line_offsets(source)
    spans: list[_CodeSemanticSpan] = []
    seen: set[tuple[int, int, str]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            left_span = _node_span(source, offsets, node.left)
            for comparator in node.comparators:
                right_span = _node_span(source, offsets, comparator)
                if left_span is not None and right_span is not None:
                    _add_between_operator_span(
                        source, spans, seen,
                        left_span[1], right_span[0],
                        _COMPARE_OP_RE, "compare_op",
                    )
                left_span = right_span
        elif isinstance(node, ast.BinOp):
            left_span = _node_span(source, offsets, node.left)
            right_span = _node_span(source, offsets, node.right)
            if left_span is not None and right_span is not None:
                _add_between_operator_span(
                    source, spans, seen,
                    left_span[1], right_span[0],
                    _BINOP_RE, "binary_op",
                )
        elif isinstance(node, ast.BoolOp):
            value_spans = [
                span for value in node.values
                if (span := _node_span(source, offsets, value)) is not None
            ]
            for left, right in zip(value_spans, value_spans[1:]):
                _add_between_operator_span(
                    source, spans, seen,
                    left[1], right[0],
                    _BOOLOP_RE, "bool_op",
                )
        elif isinstance(node, ast.UnaryOp):
            node_span = _node_span(source, offsets, node)
            operand_span = _node_span(source, offsets, node.operand)
            if node_span is not None and operand_span is not None:
                _add_between_operator_span(
                    source, spans, seen,
                    node_span[0], operand_span[0],
                    _UNARYOP_RE, "unary_op",
                )
        elif isinstance(node, ast.keyword):
            value_span = _node_span(source, offsets, node.value)
            if value_span is not None:
                label = f"keyword:{node.arg or 'unpack'}"
                _add_span(spans, seen, value_span[0], value_span[1], label)
        elif isinstance(node, ast.Constant):
            label = _constant_label(node.value)
            node_span = _node_span(source, offsets, node)
            if label is not None and node_span is not None:
                _add_span(spans, seen, node_span[0], node_span[1], label)
        elif isinstance(node, ast.Return) and node.value is not None:
            value_span = _node_span(source, offsets, node.value)
            if value_span is not None:
                _add_span(spans, seen, value_span[0], value_span[1], "return_expr")
        elif isinstance(node, ast.Subscript):
            slice_span = _node_span(source, offsets, node.slice)
            if slice_span is not None:
                _add_span(spans, seen, slice_span[0], slice_span[1], "subscript_slice")

    return spans


def _completion_token_offsets(
    completion_tokens: list[int],
    tokenizer: Any,
) -> tuple[str, list[tuple[int, int]]]:
    offsets: list[tuple[int, int]] = []
    text = ""
    for tok in completion_tokens:
        frag = tokenizer.decode([int(tok)], skip_special_tokens=False)
        offsets.append((len(text), len(text) + len(frag)))
        text += frag
    return text, offsets


def _span_priority(span: _CodeSemanticSpan) -> tuple[int, int, str]:
    label = span.label
    if label.startswith("keyword:") or label.endswith("_op"):
        label_priority = 0
    elif label in {"subscript_slice", "constant:bool", "constant:number"}:
        label_priority = 1
    else:
        label_priority = 2
    return (span.end - span.start, label_priority, label)


def evaluate_code_semantic_token_authenticity(
    tokens: list[int],
    prompt_length: int,
    completion_length: int,
    proof: "ProofResult",
    tokenizer: Any,
    *,
    threshold: float | None = None,
    argmax_conf: float | None = None,
    include_findings: bool = False,
    max_findings: int | None = None,
    context_chars: int = 80,
) -> tuple[bool, dict]:
    """Shadow check for post-hoc edits to OpenCode semantic tokens.

    Generic token auth is intentionally tuned for near-impossible injections.
    Code reward, however, can flip on plausible alternate tokens such as
    ``True``/``False``, comparison operators, or keyword values like
    ``reverse=True``. This helper maps AST-sensitive spans back to completion
    tokens and flags low-probability chosen tokens only when the model was
    confident of a different argmax.

    Returns ``(ok, metrics)``. ``ok=False`` means suspicious telemetry was
    found; callers may keep this in shadow mode by logging/archive only.
    """
    from reliquary.constants import (
        CODE_SEMANTIC_AUTH_ARGMAX_CONF,
        CODE_SEMANTIC_AUTH_THRESHOLD,
    )

    if threshold is None:
        threshold = CODE_SEMANTIC_AUTH_THRESHOLD
    if argmax_conf is None:
        argmax_conf = CODE_SEMANTIC_AUTH_ARGMAX_CONF
    if completion_length <= 0 or tokenizer is None:
        return True, {}
    probs = proof.completion_chosen_probs
    amax = proof.completion_argmax_probs
    if not probs or not amax:
        return True, {}

    completion_tokens = list(tokens[prompt_length: prompt_length + completion_length])
    completion_text, offsets = _completion_token_offsets(completion_tokens, tokenizer)
    extracted = _extract_python_span(completion_text)
    if extracted is None:
        return True, {}
    code, code_start, _code_end = extracted
    spans = _code_semantic_spans(code)
    if not spans:
        return True, {}

    semantic_by_pos: dict[int, str] = {}
    for span in sorted(spans, key=_span_priority):
        start = code_start + span.start
        end = code_start + span.end
        for pos, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= start or tok_start >= end:
                continue
            semantic_by_pos.setdefault(pos, span.label)

    if not semantic_by_pos:
        return True, {}

    selected: list[float] = []
    findings: list[dict[str, Any]] = []
    finding_details: list[dict[str, Any]] = []
    argmax_ids = proof.completion_argmax_ids
    n = min(len(probs), len(amax), len(completion_tokens))
    for pos in sorted(semantic_by_pos):
        if pos >= n:
            continue
        p = float(probs[pos])
        selected.append(p)
        if p < threshold and float(amax[pos]) >= argmax_conf:
            tok_start, tok_end = offsets[pos]
            label = semantic_by_pos[pos]
            token_id = int(completion_tokens[pos])
            argmax_id = (
                int(argmax_ids[pos])
                if pos < len(argmax_ids)
                else None
            )
            finding = {
                "pos": pos,
                "label": label,
                "p_chosen": p,
                "p_argmax": float(amax[pos]),
                "token_id": token_id,
                "token_text": (_safe_decode_ids(tokenizer, [token_id]) or "")[:32],
            }
            findings.append(finding)
            if include_findings and (
                max_findings is None or len(finding_details) < max_findings
            ):
                code_tok_start = max(0, tok_start - code_start)
                code_tok_end = max(code_tok_start, tok_end - code_start)
                finding_details.append({
                    "completion_pos": pos,
                    "absolute_token_pos": prompt_length + pos,
                    "label": label,
                    "p_chosen": p,
                    "p_argmax": float(amax[pos]),
                    "token_id": token_id,
                    "token_text": _safe_decode_ids(tokenizer, [token_id]),
                    "argmax_id": argmax_id,
                    "argmax_text": _safe_decode_ids(tokenizer, [argmax_id]),
                    "completion_char_start": tok_start,
                    "completion_char_end": tok_end,
                    "code_char_start": code_tok_start,
                    "code_char_end": code_tok_end,
                    "completion_context": _completion_context_for_position(
                        completion_tokens,
                        pos,
                        tokenizer,
                        context_chars=context_chars,
                    ),
                    "code_context": _text_context_for_span(
                        code,
                        code_tok_start,
                        code_tok_end,
                        context_chars=context_chars,
                    ),
                })

    if not selected:
        return True, {}
    metrics = {
        "n_spans": len(spans),
        "n_tokens": len(semantic_by_pos),
        "min_prob": min(selected),
        "threshold": float(threshold),
        "argmax_conf": float(argmax_conf),
        "findings": len(findings),
        "finding_min_prob": (
            min(f["p_chosen"] for f in findings)
            if findings
            else None
        ),
    }
    if findings:
        first = findings[0]
        metrics.update({
            "first_pos": first["pos"],
            "first_label": first["label"],
            "first_p_chosen": first["p_chosen"],
            "first_p_argmax": first["p_argmax"],
            "first_token_id": first["token_id"],
            "first_token_text": first["token_text"],
        })
    if include_findings:
        metrics["finding_details"] = finding_details
    return (len(findings) == 0), metrics


def _text_context_for_span(
    text: str,
    start: int,
    end: int,
    *,
    context_chars: int,
) -> str:
    chars = max(0, int(context_chars))
    left = max(0, start - chars)
    right = min(len(text), end + chars)
    return text[left:right]


def _find_last_boxed_token_range(
    completion_tokens: list[int],
    tokenizer: Any,
) -> tuple[int, int] | None:
    """Return ``(start, end)`` completion-relative token indices (inclusive)
    that cover the content between ``{`` and ``}`` of the last
    ``\\boxed{...}`` or ``\\fbox{...}`` in the decoded completion. ``None`` if
    no closed boxed answer is present.

    Walks token-by-token decode to map text offsets back to token positions.
    Matches ``_last_boxed_only_string`` in the OMI env so the filter targets
    the same substring the reward parser keys on.
    """
    if not completion_tokens:
        return None
    offsets: list[tuple[int, int]] = []
    cum = ""
    for tok in completion_tokens:
        frag = tokenizer.decode([int(tok)], skip_special_tokens=False)
        offsets.append((len(cum), len(cum) + len(frag)))
        cum += frag

    idx = max(cum.rfind("\\boxed{"), cum.rfind("\\fbox{"))
    if idx < 0:
        return None
    try:
        open_idx = cum.index("{", idx)
    except ValueError:
        return None
    depth = 0
    close_idx = -1
    for j in range(open_idx, len(cum)):
        c = cum[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                close_idx = j
                break
    if close_idx < 0:
        return None

    content_start = open_idx + 1
    content_end = close_idx  # exclusive

    start_tok = end_tok = None
    for i, (s, e) in enumerate(offsets):
        if start_tok is None and e > content_start:
            start_tok = i
        if s < content_end:
            end_tok = i
    if start_tok is None or end_tok is None:
        return None
    return start_tok, end_tok


def evaluate_boxed_answer_probability(
    tokens: list[int],
    prompt_length: int,
    completion_length: int,
    proof: "ProofResult",
    tokenizer: Any,
    *,
    threshold: float | None = None,
) -> tuple[bool, dict]:
    """Hard check (OMI-specific): every token inside the last ``\\boxed{...}``
    must have chosen-token probability ≥ ``threshold``.

    The OMI reward parser extracts the answer from the last ``\\boxed{...}``;
    a miner can flip a wrong rollout to right by swapping a few answer tokens
    post-hoc. The validator's forward pass on the tampered tokens shows a
    collapsed chosen probability at those positions, while honest sampling
    keeps boxed-answer probabilities high (>0.5 in measurements).

    Returns ``(ok, metrics)`` where ``metrics`` carries ``min_prob`` and
    ``n_tokens`` for telemetry. ``ok=True`` when no boxed answer is present
    or no probabilities are available — the filter only fires on a concrete
    low-probability boxed token.
    """
    from reliquary.constants import BOXED_ANSWER_MIN_PROB, TOKEN_AUTH_ARGMAX_CONF

    if threshold is None:
        threshold = BOXED_ANSWER_MIN_PROB
    if completion_length <= 0:
        return True, {}
    probs = proof.completion_chosen_probs
    if not probs:
        return True, {}
    amax = proof.completion_argmax_probs

    completion_tokens = list(tokens[prompt_length: prompt_length + completion_length])
    rng = _find_last_boxed_token_range(completion_tokens, tokenizer)
    if rng is None:
        return True, {}
    start, end = rng

    selected: list[float] = []
    tampered = False
    for i in range(start, end + 1):
        if 0 <= i < len(probs):
            p = float(probs[i])
            selected.append(p)
            # A collapsed boxed token is a swap only if the model was confident
            # of a different token there. A low prob with no confident argmax is
            # genuine sampling uncertainty — don't reject (avoids vLLM->HF drift
            # false positives near the threshold).
            if p < threshold and i < len(amax) and amax[i] >= TOKEN_AUTH_ARGMAX_CONF:
                tampered = True
    if not selected:
        return True, {}
    metrics = {"min_prob": min(selected), "n_tokens": len(selected)}
    return (not tampered), metrics


# Numeric-edit threshold (defined here, not in constants/env): an answer-bearing
# numeric token below this while argmax >= conf is an edited value. Folded into
# evaluate_token_authenticity. Set at the measured 0%-FP point.
NUMERIC_AUTH_THRESHOLD = 1e-6

_NUMERIC_ID_CACHE: dict[int, tuple[frozenset[int], frozenset[int]]] = {}


def _numeric_token_ids(tokenizer: Any) -> tuple[frozenset[int], frozenset[int]]:
    """``(digit_ids, sign_ids)`` for the vocab, built once per tokenizer.
    sign_ids = standalone ``- + . /`` tokens (numeric only when digit-adjacent)."""
    key = id(tokenizer)
    cached = _NUMERIC_ID_CACHE.get(key)
    if cached is not None:
        return cached
    digit_ids: set[int] = set()
    sign_ids: set[int] = set()
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        out = (frozenset(), frozenset())
        _NUMERIC_ID_CACHE[key] = out
        return out
    for tid in get_vocab().values():
        s = tokenizer.decode([int(tid)]).strip()
        if not s:
            continue
        if s.isdigit():
            digit_ids.add(int(tid))
        elif s in ("-", "+", ".", "/", "−"):
            sign_ids.add(int(tid))
    out = (frozenset(digit_ids), frozenset(sign_ids))
    _NUMERIC_ID_CACHE[key] = out
    return out
