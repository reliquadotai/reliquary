"""The per-chunk comparison of poly.cpp and the per-sequence rule of
toploc-validator's validate_stage_results, reproduced exactly."""

import pytest

from reliquary.protocol.toploc import (
    NO_MANTISSA,
    ChunkResult,
    ToplocThresholds,
    compare_bf16_bits,
    sequence_verdict,
)

DEPLOYED = ToplocThresholds(exp_mismatch=60, mant_mean=40.0, mant_median=40.0)


def _bits(exponent, mantissa, sign=0):
    return (sign << 15) | (exponent << 7) | mantissa


def test_identical_bits_have_no_error():
    bits = [_bits(120, m) for m in range(10)]
    assert compare_bf16_bits(bits, bits) == ChunkResult(0, 0.0, 0.0)


def test_the_sign_bit_is_not_an_exponent_mismatch():
    assert compare_bf16_bits([_bits(120, 3, sign=1)], [_bits(120, 3)]).exp_mismatches == 0


def test_mantissa_error_counts_only_where_exponents_agree():
    proof = [_bits(120, 10), _bits(121, 0), _bits(120, 14)]
    verifier = [_bits(120, 13), _bits(99, 0), _bits(120, 14)]
    result = compare_bf16_bits(proof, verifier)
    assert result.exp_mismatches == 1
    assert result.mant_err_mean == 1.5          # errors 3 and 0
    assert result.mant_err_median == 3.0        # sorted [0, 3], upper median


def test_no_agreeing_exponent_reports_the_sentinel():
    result = compare_bf16_bits([_bits(1, 0)], [_bits(2, 0)])
    assert result == ChunkResult(1, NO_MANTISSA, NO_MANTISSA)


def test_lengths_must_agree():
    with pytest.raises(ValueError):
        compare_bf16_bits([1, 2], [1])


def test_a_clean_sequence_passes():
    assert sequence_verdict([ChunkResult(5, 2.0, 1.0)] * 4, DEPLOYED) == (True, None)


@pytest.mark.parametrize(
    "bad,reason",
    [
        (ChunkResult(61, 0.0, 0.0), "exp_mismatch"),
        (ChunkResult(0, 40.5, 0.0), "mant_err_mean"),
        (ChunkResult(0, 0.0, 41.0), "mant_err_median"),
    ],
)
def test_every_chunk_must_pass(bad, reason):
    results = [ChunkResult(0, 0.0, 0.0)] * 5 + [bad]
    assert sequence_verdict(results, DEPLOYED) == (False, reason)


def test_the_boundary_is_inclusive():
    assert sequence_verdict([ChunkResult(60, 40.0, 40.0)], DEPLOYED) == (True, None)


def test_an_empty_sequence_fails_closed():
    assert sequence_verdict([], DEPLOYED) == (False, "no_chunks")


def test_forgiveness_admits_a_bounded_number_of_failing_chunks():
    lenient = ToplocThresholds(60, 40.0, 40.0, min_allowed_failures=1)
    results = [ChunkResult(0, 0.0, 0.0)] * 3 + [ChunkResult(90, 1.0, 1.0)]
    assert sequence_verdict(results, lenient) == (True, None)
    results.append(ChunkResult(90, 1.0, 1.0))
    assert sequence_verdict(results, lenient) == (False, "too_many_exp_mismatches")


def test_forgiven_chunks_are_still_held_to_the_mantissa_bounds():
    # Faithful to the reference: a forgiven chunk with a wild mantissa still fails.
    lenient = ToplocThresholds(60, 40.0, 40.0, min_allowed_failures=1)
    results = [ChunkResult(0, 0.0, 0.0), ChunkResult(90, 50.0, 1.0)]
    assert sequence_verdict(results, lenient) == (False, "mant_err_mean")
