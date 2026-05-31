"""Tests for the behavioural validators after the keep-logits-on-GPU
refactor.

These call the three behavioural primitives (``verify_termination``,
``verify_logprobs_claim``, ``evaluate_token_distribution``) directly with
a ``ProofResult`` carrying the sparse values the GPU path now
precomputes. The legacy logits-based path is gone; production never
materialises a CPU logits tensor for these checks.
"""

import math

import pytest
import torch

from reliquary.constants import CHALLENGE_K, LOGPROB_IS_EPS, T_PROTO
from reliquary.validator import verifier
from reliquary.validator.verifier import ProofResult


# ---------------------------------------------------------------------------
# verify_logprobs_claim
# ---------------------------------------------------------------------------


def _proof_with_challenges(indices: list[int], values: list[float]) -> ProofResult:
    return ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        challenge_lp_indices=indices,
        challenge_lp_values=values,
    )


def test_verify_logprobs_claim_honest_passes():
    """When the miner-claimed logprob at every challenge position matches
    the validator's recomputation, the median IS-deviation is 0 and the
    check passes."""
    vocab = 50
    seq_len = 40
    tokens = list(range(seq_len))
    prompt_len = 5
    completion_len = seq_len - prompt_len
    uniform_lp = -math.log(vocab)

    # K=CHALLENGE_K challenge positions inside the completion range.
    indices = list(range(prompt_len, prompt_len + CHALLENGE_K))
    values = [uniform_lp] * CHALLENGE_K
    proof = _proof_with_challenges(indices, values)

    # Miner claims the same logprob at every completion position (full-
    # sequence layout — prompt positions zeroed and ignored).
    claimed = [0.0] * prompt_len + [uniform_lp] * completion_len

    ok, median_dev = verifier.verify_logprobs_claim(
        tokens=tokens,
        prompt_length=prompt_len,
        completion_length=completion_len,
        claimed_logprobs=claimed,
        proof=proof,
    )
    assert ok is True, f"honest pair must pass, got median_dev={median_dev}"


def test_verify_logprobs_claim_cheater_fails():
    """When the miner inflates logprobs above what the validator
    recomputed, the median deviation crosses LOGPROB_IS_EPS and the
    check fails."""
    vocab = 50
    seq_len = 40
    tokens = list(range(seq_len))
    prompt_len = 5
    completion_len = seq_len - prompt_len
    uniform_lp = -math.log(vocab)

    indices = list(range(prompt_len, prompt_len + CHALLENGE_K))
    values = [uniform_lp] * CHALLENGE_K  # validator says -3.91
    proof = _proof_with_challenges(indices, values)

    # Miner claims much higher logprobs (~ -0.5) at every position.
    claimed = [0.0] * prompt_len + [-0.5] * completion_len

    ok, median_dev = verifier.verify_logprobs_claim(
        tokens=tokens,
        prompt_length=prompt_len,
        completion_length=completion_len,
        claimed_logprobs=claimed,
        proof=proof,
    )
    assert ok is False
    assert median_dev > LOGPROB_IS_EPS


def test_verify_logprobs_claim_too_short_rejects():
    """Completion shorter than CHALLENGE_K can't be challenged → reject."""
    tokens = list(range(20))
    proof = _proof_with_challenges([], [])

    ok, _ = verifier.verify_logprobs_claim(
        tokens=tokens,
        prompt_length=5,
        completion_length=15,
        claimed_logprobs=[0.0] * 20,
        proof=proof,
    )
    assert ok is False


def test_verify_logprobs_claim_completion_only_layout():
    """Miner's token_logprobs may be laid out as completion-only
    (length == completion_length, no prompt padding). The helper must
    accept that shape too."""
    vocab = 50
    seq_len = 40
    tokens = list(range(seq_len))
    prompt_len = 5
    completion_len = seq_len - prompt_len
    uniform_lp = -math.log(vocab)

    indices = list(range(prompt_len, prompt_len + CHALLENGE_K))
    values = [uniform_lp] * CHALLENGE_K
    proof = _proof_with_challenges(indices, values)

    claimed = [uniform_lp] * completion_len  # completion-only layout

    ok, _ = verifier.verify_logprobs_claim(
        tokens=tokens,
        prompt_length=prompt_len,
        completion_length=completion_len,
        claimed_logprobs=claimed,
        proof=proof,
    )
    assert ok is True


def test_verify_logprobs_claim_empty_proof_rejects():
    """A proof that didn't populate sparse outputs (e.g. the stub didn't
    opt into behavioural checks, or completion was too short to sample
    K positions on the GPU side) is treated as a deterministic fail —
    we cannot verify the claim without the validator's recomputation."""
    tokens = list(range(40))
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        challenge_lp_indices=[],
        challenge_lp_values=[],
    )
    ok, dev = verifier.verify_logprobs_claim(
        tokens=tokens,
        prompt_length=5,
        completion_length=35,
        claimed_logprobs=[0.0] * 40,
        proof=proof,
    )
    assert ok is False
    assert dev == float("inf")


# ---------------------------------------------------------------------------
# evaluate_token_distribution
# ---------------------------------------------------------------------------


def _proof_with_dist(probs: list[float]) -> ProofResult:
    return ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        completion_chosen_probs=probs,
    )


def test_evaluate_token_distribution_honest_passes():
    """Validator's chosen-token probabilities cluster high → median and
    q10 both above thresholds → accept."""
    completion_len = 45
    probs = [0.99] * completion_len
    proof = _proof_with_dist(probs)

    ok, metrics = verifier.evaluate_token_distribution(
        tokens=list(range(50)),
        prompt_length=5,
        completion_length=completion_len,
        proof=proof,
    )
    assert ok is True
    assert metrics["median"] > 0.30


def test_evaluate_token_distribution_cheater_fails():
    """Validator's chosen-token probabilities collapse to near zero →
    median and q10 below thresholds → reject."""
    completion_len = 45
    probs = [1e-6] * completion_len
    proof = _proof_with_dist(probs)

    ok, metrics = verifier.evaluate_token_distribution(
        tokens=list(range(50)),
        prompt_length=5,
        completion_length=completion_len,
        proof=proof,
    )
    assert ok is False
    assert metrics["median"] < 0.30
    assert metrics["q10"] < 0.025


def test_evaluate_token_distribution_too_short_skips():
    """Completion shorter than SAMPLING_MIN_STEPS returns (None, {}) —
    not enough data to decide, caller defaults to accept."""
    completion_len = 15
    probs = [0.5] * completion_len
    proof = _proof_with_dist(probs)

    ok, _ = verifier.evaluate_token_distribution(
        tokens=list(range(20)),
        prompt_length=5,
        completion_length=completion_len,
        proof=proof,
    )
    assert ok is None


# ---------------------------------------------------------------------------
# verify_termination
# ---------------------------------------------------------------------------


class _ModelWithEos:
    """Minimal model stub exposing ``generation_config.eos_token_id``."""

    class _GenCfg:
        eos_token_id = [99, 100]

    generation_config = _GenCfg()


def _commit_with_lengths(prompt_length: int, completion_length: int, last_token: int):
    seq_len = prompt_length + completion_length
    tokens = list(range(seq_len - 1)) + [last_token]
    return {
        "tokens": tokens,
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": completion_length,
        },
    }


def test_verify_termination_max_length_path_passes_without_p_stop():
    """Path 1: when prompt+completion meets MAX_NEW_TOKENS_PROTOCOL_CAP,
    p_stop is irrelevant — accept."""
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

    commit = _commit_with_lengths(
        prompt_length=10,
        completion_length=MAX_NEW_TOKENS_PROTOCOL_CAP - 10,
        last_token=42,  # not eos, doesn't matter for path 1
    )
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        p_stop=None,  # not consulted on path 1
    )
    assert verifier.verify_termination(
        commit, tokenizer=None, proof=proof, model=_ModelWithEos(),
    ) is True


def test_verify_termination_eos_with_strong_p_stop_passes():
    """Path 2: last token is EOS and p_stop is well above the gate."""
    from reliquary.constants import MIN_EOS_PROBABILITY

    commit = _commit_with_lengths(
        prompt_length=10, completion_length=40, last_token=99,
    )
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        p_stop=MIN_EOS_PROBABILITY * 5,
    )
    assert verifier.verify_termination(
        commit, tokenizer=None, proof=proof, model=_ModelWithEos(),
    ) is True


def test_verify_termination_eos_with_weak_p_stop_fails():
    """Path 2: last token is EOS but p_stop collapsed near zero —
    forced-stop, reject."""
    from reliquary.constants import MIN_EOS_PROBABILITY

    commit = _commit_with_lengths(
        prompt_length=10, completion_length=40, last_token=99,
    )
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        p_stop=MIN_EOS_PROBABILITY * 0.1,
    )
    assert verifier.verify_termination(
        commit, tokenizer=None, proof=proof, model=_ModelWithEos(),
    ) is False


def test_verify_termination_non_eos_last_token_fails():
    """Path 2: last token is not in the EOS set → reject."""
    commit = _commit_with_lengths(
        prompt_length=10, completion_length=40, last_token=42,
    )
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        p_stop=0.99,
    )
    assert verifier.verify_termination(
        commit, tokenizer=None, proof=proof, model=_ModelWithEos(),
    ) is False


def test_verify_termination_missing_eos_config_fails():
    """No EOS declared anywhere → can't enforce path 2 → reject."""
    commit = _commit_with_lengths(
        prompt_length=10, completion_length=40, last_token=42,
    )
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        p_stop=0.99,
    )
    assert verifier.verify_termination(
        commit, tokenizer=None, proof=proof, model=None,
    ) is False


def test_verify_termination_missing_p_stop_fails():
    """EOS configured but proof carries no p_stop (model.generation_config
    was missing at proof-build time, etc.) → reject."""
    commit = _commit_with_lengths(
        prompt_length=10, completion_length=40, last_token=99,
    )
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        p_stop=None,
    )
    assert verifier.verify_termination(
        commit, tokenizer=None, proof=proof, model=_ModelWithEos(),
    ) is False


# ---------------------------------------------------------------------------
# evaluate_boxed_answer_probability
# ---------------------------------------------------------------------------


class _CharTokenizer:
    """One-token-per-character tokenizer for boxed-answer tests.

    Token id = ord(char); decode([id]) = chr(id). The completion text is
    therefore the concatenation of completion tokens, which makes the
    text-offset → token-index mapping trivial to reason about in tests.
    """

    def decode(self, ids, *, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)


def _ords(s: str) -> list[int]:
    return [ord(c) for c in s]


def test_evaluate_boxed_no_boxed_returns_true():
    completion = "the answer is 42 (no box)"
    tokens = [0] * 5 + _ords(completion)
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        completion_chosen_probs=[0.01] * len(completion),
    )
    ok, metrics = verifier.evaluate_boxed_answer_probability(
        tokens=tokens, prompt_length=5,
        completion_length=len(completion),
        proof=proof, tokenizer=_CharTokenizer(),
    )
    assert ok is True
    assert metrics == {}


def test_evaluate_boxed_high_prob_passes():
    completion = "answer is \\boxed{42}"
    tokens = [0] * 5 + _ords(completion)
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        completion_chosen_probs=[0.99] * len(completion),
    )
    ok, metrics = verifier.evaluate_boxed_answer_probability(
        tokens=tokens, prompt_length=5,
        completion_length=len(completion),
        proof=proof, tokenizer=_CharTokenizer(),
    )
    assert ok is True
    assert metrics["min_prob"] == 0.99
    assert metrics["n_tokens"] == 2  # "4" and "2"


def test_evaluate_boxed_low_prob_rejects():
    """One token inside \\boxed{} at tampered probability (~1/vocab) → reject."""
    completion = "answer is \\boxed{42}"
    tokens = [0] * 5 + _ords(completion)
    probs = [0.99] * len(completion)
    # Position of "4" inside the boxed content
    boxed_open = completion.index("{") + 1
    probs[boxed_open] = 1e-6
    argmax = [1.0] * len(completion)  # model was confident of a different token
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        completion_chosen_probs=probs,
        completion_argmax_probs=argmax,
    )
    ok, metrics = verifier.evaluate_boxed_answer_probability(
        tokens=tokens, prompt_length=5,
        completion_length=len(completion),
        proof=proof, tokenizer=_CharTokenizer(),
    )
    assert ok is False
    assert metrics["min_prob"] == 1e-6


def test_evaluate_boxed_low_prob_no_confident_argmax_passes():
    """Low boxed prob but no confident alternative (flat dist) — genuine
    sampling uncertainty, not a post-hoc swap. With the argmax guard, accept."""
    completion = "answer is \\boxed{42}"
    tokens = [0] * 5 + _ords(completion)
    probs = [0.99] * len(completion)
    argmax = [1.0] * len(completion)
    boxed_open = completion.index("{") + 1
    probs[boxed_open] = 1e-6
    argmax[boxed_open] = 0.30  # model had no confident alternative here
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        completion_chosen_probs=probs,
        completion_argmax_probs=argmax,
    )
    ok, _ = verifier.evaluate_boxed_answer_probability(
        tokens=tokens, prompt_length=5,
        completion_length=len(completion),
        proof=proof, tokenizer=_CharTokenizer(),
    )
    assert ok is True


def test_evaluate_boxed_unclosed_returns_true():
    """Truncated completion with open \\boxed{ but no } → skip the check."""
    completion = "starting \\boxed{4"
    tokens = [0] * 5 + _ords(completion)
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        completion_chosen_probs=[0.01] * len(completion),
    )
    ok, _ = verifier.evaluate_boxed_answer_probability(
        tokens=tokens, prompt_length=5,
        completion_length=len(completion),
        proof=proof, tokenizer=_CharTokenizer(),
    )
    assert ok is True


def test_evaluate_boxed_last_boxed_wins():
    """Multiple \\boxed{...} — only the LAST one is gated (matches OMI parser)."""
    completion = "first \\boxed{1} then \\boxed{42}"
    tokens = [0] * 5 + _ords(completion)
    probs = [0.99] * len(completion)
    # Make the FIRST boxed content low — should NOT reject.
    first_boxed_open = completion.index("{") + 1
    probs[first_boxed_open] = 1e-6
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        completion_chosen_probs=probs,
    )
    ok, _ = verifier.evaluate_boxed_answer_probability(
        tokens=tokens, prompt_length=5,
        completion_length=len(completion),
        proof=proof, tokenizer=_CharTokenizer(),
    )
    assert ok is True


def test_evaluate_boxed_fbox_alias_also_checked():
    completion = "answer is \\fbox{42}"
    tokens = [0] * 5 + _ords(completion)
    probs = [0.99] * len(completion)
    probs[completion.index("{") + 1] = 1e-6
    argmax = [1.0] * len(completion)  # model was confident of a different token
    proof = ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        completion_chosen_probs=probs,
        completion_argmax_probs=argmax,
    )
    ok, _ = verifier.evaluate_boxed_answer_probability(
        tokens=tokens, prompt_length=5,
        completion_length=len(completion),
        proof=proof, tokenizer=_CharTokenizer(),
    )
    assert ok is False


def _proof_with_token_stats(chosen, argmax_probs, argmax_ids=None):
    return ProofResult(
        all_passed=True, passed=1, checked=1, has_sparse_outputs=True,
        completion_chosen_probs=chosen,
        completion_argmax_probs=argmax_probs,
        completion_argmax_ids=argmax_ids or [0] * len(chosen),
    )


def test_token_authenticity_honest_passes():
    proof = _proof_with_token_stats([0.99, 0.8, 1.0], [0.99, 0.8, 1.0])
    ok, m = verifier.evaluate_token_authenticity(proof)
    assert ok is True and m == {}


def test_token_authenticity_injection_fails():
    proof = _proof_with_token_stats(
        [1.0, 2.0e-20, 1.0], [1.0, 1.0, 1.0], argmax_ids=[5, 7, 9],
    )
    ok, m = verifier.evaluate_token_authenticity(proof)
    assert ok is False
    assert m["pos"] == 1 and m["argmax_id"] == 7


def test_token_authenticity_high_entropy_honest_passes():
    proof = _proof_with_token_stats([1.0, 1.0e-9, 1.0], [1.0, 0.30, 1.0])
    ok, _ = verifier.evaluate_token_authenticity(proof)
    assert ok is True


def test_token_authenticity_empty_skips():
    ok, m = verifier.evaluate_token_authenticity(
        ProofResult(all_passed=True, passed=1, checked=1)
    )
    assert ok is True and m == {}


def test_gpu_completion_token_stats_returns_chosen_and_argmax():
    # 2 completion positions, vocab 4. Build logits so argmax is known.
    seq_len = 3
    logits = torch.zeros(seq_len, 4)
    logits[0] = torch.tensor([0.0, 5.0, 0.0, 0.0])   # predicts token at idx1
    logits[1] = torch.tensor([0.0, 0.0, 0.0, 9.0])   # predicts token at idx2
    tokens = [0, 1, 3]  # idx1 token=1 (the argmax), idx2 token=3 (the argmax)

    chosen, amax_p, amax_id = verifier._gpu_completion_token_stats(
        logits, tokens, prompt_length=1, completion_length=2,
        seq_len=seq_len, device="cpu",
    )
    assert len(chosen) == len(amax_p) == len(amax_id) == 2
    assert amax_id == [1, 3]
    for c, a in zip(chosen, amax_p):
        assert abs(c - a) < 1e-6
        assert a > 0.9
