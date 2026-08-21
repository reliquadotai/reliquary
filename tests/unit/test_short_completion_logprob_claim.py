"""Full-coverage logprob-claim verification for completions < CHALLENGE_K.

The sampled-challenge check is a deterministic fail below 32 tokens
(measured in production: 100% of its rejections were this case, zero real
mismatches). Short completions are now verified over EVERY position via
the proof's chosen-token probs — stronger coverage — and only skipped
when genuinely non-executable, matching the other detectors' behavior.
"""
import math

import pytest

from reliquary.validator.batcher import (
    _verify_logprobs_for_training, _verify_short_logprob_claim,
)


class _Proof:
    def __init__(self, probs):
        self.completion_chosen_probs = probs


def _validator_lps(probs):
    return [math.log(p) for p in probs]


PROBS = [0.9, 0.4, 0.75, 0.2, 0.95, 0.6, 0.3, 0.85]  # L=8 < 32
PROMPT = [1, 2, 3]
TOKENS = PROMPT + [10 + i for i in range(len(PROBS))]


def test_honest_completion_only_claim_passes():
    lps = _validator_lps(PROBS)
    ok, dev = _verify_short_logprob_claim(
        lps, TOKENS, len(PROMPT), len(PROBS), list(lps),
    )
    assert ok is True
    assert dev == pytest.approx(0.0, abs=1e-9)


def test_honest_full_sequence_claim_passes():
    lps = _validator_lps(PROBS)
    claimed = [0.0] * len(PROMPT) + list(lps)  # prompt entries ignored
    ok, _ = _verify_short_logprob_claim(
        lps, TOKENS, len(PROMPT), len(PROBS), claimed,
    )
    assert ok is True


def test_forged_claim_rejected():
    lps = _validator_lps(PROBS)
    forged = [lp + 0.8 for lp in lps]  # exp(0.8)-1 ~ 1.2 >> eps on every pos
    ok, dev = _verify_short_logprob_claim(
        lps, TOKENS, len(PROMPT), len(PROBS), forged,
    )
    assert ok is False
    assert dev > 1.0


def test_single_bf16_outlier_tolerated():
    """Honest cross-GPU runs see occasional per-token outliers; one outlier
    among 8 positions must not reject (the >= half rule)."""
    lps = _validator_lps(PROBS)
    claimed = list(lps)
    claimed[3] += 0.7  # one big outlier
    ok, _ = _verify_short_logprob_claim(
        lps, TOKENS, len(PROMPT), len(PROBS), claimed,
    )
    assert ok is True


def test_majority_deviation_rejected():
    lps = _validator_lps(PROBS)
    claimed = [lp + 0.7 for lp in lps[:5]] + list(lps[5:])  # 5 of 8 off
    ok, _ = _verify_short_logprob_claim(
        lps, TOKENS, len(PROMPT), len(PROBS), claimed,
    )
    assert ok is False


def test_malformed_claim_length_rejected():
    lps = _validator_lps(PROBS)
    ok, dev = _verify_short_logprob_claim(
        lps, TOKENS, len(PROMPT), len(PROBS), lps[:-2],
    )
    assert ok is False
    assert dev == float("inf")


def test_nonfinite_claim_rejected():
    lps = _validator_lps(PROBS)
    claimed = list(lps)
    claimed[0] = float("nan")
    ok, dev = _verify_short_logprob_claim(
        lps, TOKENS, len(PROMPT), len(PROBS), claimed,
    )
    assert ok is False


def test_unverifiable_returns_none_so_caller_keeps_reject():
    """No validator coverage (T != 1, degenerate) -> (None, None). The
    caller admits a short rollout ONLY on a positive (True) verdict, so
    None keeps the original reject — never an accept."""
    ok, dev = _verify_short_logprob_claim(
        None, TOKENS, len(PROMPT), len(PROBS), _validator_lps(PROBS),
    )
    assert ok is None and dev is None


def test_derivation_gates_still_apply(monkeypatch):
    import reliquary.constants as constants

    monkeypatch.setattr(constants, "T_PROTO", 0.6)
    assert _verify_logprobs_for_training(_Proof(PROBS), len(PROBS)) is None
    monkeypatch.setattr(constants, "T_PROTO", 1.0)
    lps = _verify_logprobs_for_training(_Proof(PROBS), len(PROBS))
    assert lps == pytest.approx(_validator_lps(PROBS))


def test_single_token_completion():
    lps = _validator_lps([0.5])
    ok, _ = _verify_short_logprob_claim(lps, [1, 2, 10], 2, 1, list(lps))
    assert ok is True
    ok, _ = _verify_short_logprob_claim(lps, [1, 2, 10], 2, 1, [lps[0] + 0.9])
    assert ok is False


def test_overflow_claim_does_not_raise():
    """A crafted -1e30 logprob must NOT raise OverflowError (which would
    fault the whole proof plane). The median tolerates a single outlier by
    design; a MAJORITY of such values rejects."""
    lps = _validator_lps(PROBS)
    # single crafted value: no crash; median ignores the lone outlier
    claimed = list(lps)
    claimed[0] = -1e30
    ok_single, _ = _verify_short_logprob_claim(
        lps, TOKENS, len(PROMPT), len(PROBS), claimed,
    )
    assert ok_single is True  # robust to one, and crucially did not raise
    # majority crafted: rejected, still no crash
    claimed2 = [-1e30] * 5 + list(lps[5:])
    ok_many, dev_many = _verify_short_logprob_claim(
        lps, TOKENS, len(PROMPT), len(PROBS), claimed2,
    )
    assert ok_many is False
    assert dev_many == float("inf")


def test_l2_single_large_forge_rejected():
    """Median rule (same as the sampled path): at L=2 one large forge pulls
    the median past eps -> reject. No 50% free pass."""
    lps = _validator_lps([0.5, 0.5])
    tokens = [1, 2, 10, 11]
    forged = [lps[0], lps[1] + 5.0]
    ok, _ = _verify_short_logprob_claim(lps, tokens, 2, 2, forged)
    assert ok is False


def test_l2_small_bf16_outlier_tolerated():
    """A small cross-GPU deviation under eps at L=2 still passes."""
    from reliquary.constants import LOGPROB_IS_EPS
    lps = _validator_lps([0.5, 0.5])
    tokens = [1, 2, 10, 11]
    import math as _m
    small = _m.log1p(LOGPROB_IS_EPS) * 0.5  # dev < eps
    claimed = [lps[0], lps[1] + small]
    ok, _ = _verify_short_logprob_claim(lps, tokens, 2, 2, claimed)
    assert ok is True
