"""TOPLOC proof arithmetic: interpolation, evaluation and the byte format.

Ported from PrimeIntellect-ai/toploc @ 7ab7bcd (toploc/C/csrc/ndd.cpp and
poly.cpp), MIT License, Copyright (c) 2024 Prime Intellect. Every modular step
and the byte layout are kept identical, so the thresholds Prime Intellect
deploys mean the same thing here. Torch-free: this is consensus arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

# The prime field every coefficient lives in, and the ceiling of the search
# for a modulus under which a chunk's indices stay distinct.
MOD_N = 65497


def expected_chunks(completion_tokens: int, chunk_tokens: int) -> int:
    """How many proofs a completion of this length must carry."""
    if completion_tokens < 1 or chunk_tokens < 1:
        raise ValueError("token counts must be positive")
    return math.ceil(completion_tokens / chunk_tokens)


def injective_modulus(xs: Sequence[int]) -> int:
    """The largest modulus <= MOD_N under which the points stay distinct."""
    values = [int(x) for x in xs]
    if len(set(values)) != len(values):
        raise ValueError("interpolation points must be distinct")
    for modulus in range(MOD_N, 0, -1):
        if len({x % modulus for x in values}) == len(values):
            return modulus
    raise ValueError("no injective modulus found")


def _modinv(values: np.ndarray) -> np.ndarray:
    """Inverses modulo the prime MOD_N by Fermat; equal to the reference's EEA."""
    result = np.ones_like(values)
    base = values % MOD_N
    exponent = MOD_N - 2
    while exponent:
        if exponent & 1:
            result = (result * base) % MOD_N
        base = (base * base) % MOD_N
        exponent >>= 1
    return result


# Every inverse in the field, computed once: a lookup is ~4x faster than the
# per-pass exponentiation, which dominated proof building on the miner.
_INVERSES = _modinv(np.arange(MOD_N, dtype=np.int64))


def newton_coefficients(xs: Sequence[int], ys: Sequence[int]) -> list[int]:
    """Ascending coefficients of the polynomial through (xs, ys) mod MOD_N."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) == 0:
        raise ValueError("at least one point is required")
    x = np.asarray(xs, dtype=np.int64)
    dd = np.asarray(ys, dtype=np.int64) % MOD_N
    n = len(x)
    for k in range(1, n):
        denominator = (x[k:] - x[: n - k]) % MOD_N
        if (denominator == 0).any():
            raise ValueError("points collide in the field")
        # The right-hand side reads the previous pass before it is overwritten,
        # which is what the reference's descending loop achieves in place.
        dd[k:] = ((dd[k:] - dd[k - 1 : n - 1]) % MOD_N * _INVERSES[denominator]) % MOD_N

    coeffs = np.zeros(n, dtype=np.int64)
    factor = np.zeros(n, dtype=np.int64)
    factor[0] = 1
    for i in range(n):
        coeffs[: i + 1] = (coeffs[: i + 1] + dd[i] * factor[: i + 1]) % MOD_N
        if i + 1 < n:
            minus_xi = int((-x[i]) % MOD_N)
            shifted = np.zeros_like(factor)
            shifted[0] = (factor[0] * minus_xi) % MOD_N
            shifted[1 : i + 2] = (factor[0 : i + 1] + factor[1 : i + 2] * minus_xi) % MOD_N
            factor = shifted
    return [int(c) for c in coeffs]


def evaluate(coeffs: Sequence[int], xs: Sequence[int]) -> list[int]:
    """Horner's method at every x, mod MOD_N, as ``evaluate_polynomials`` does."""
    if len(coeffs) == 0:
        raise ValueError("a polynomial needs at least one coefficient")
    c = np.asarray(coeffs, dtype=np.int64)
    x = np.asarray(xs, dtype=np.int64)
    result = np.full(x.shape, c[-1], dtype=np.int64)
    for coefficient in c[-2::-1]:
        result = (result * x + coefficient) % MOD_N
    return [int(v) for v in result % MOD_N]


@dataclass(frozen=True, slots=True)
class ChunkProof:
    modulus: int
    coeffs: tuple[int, ...]

    @classmethod
    def from_points(cls, xs: Sequence[int], ys: Sequence[int]) -> ChunkProof:
        modulus = injective_modulus(xs)
        coeffs = newton_coefficients([int(x) % modulus for x in xs], ys)
        return cls(modulus, tuple(coeffs))

    def to_bytes(self) -> bytes:
        out = bytearray(int(self.modulus).to_bytes(2, "big"))
        for coefficient in self.coeffs:
            out += int(coefficient).to_bytes(2, "big")
        return bytes(out)

    @classmethod
    def from_bytes(cls, data: bytes) -> ChunkProof:
        """Stricter than the reference, which drops an odd byte and divides by a
        zero modulus: a verifier must fail closed on both."""
        if len(data) < 4 or len(data) % 2:
            raise ValueError(f"a proof is 2 + 2k bytes with k >= 1, got {len(data)}")
        modulus = int.from_bytes(data[0:2], "big")
        if modulus == 0:
            raise ValueError("a zero modulus is the reference's null proof")
        if modulus > MOD_N:
            raise ValueError(f"modulus {modulus} is above the field")
        coeffs = tuple(
            int.from_bytes(data[i : i + 2], "big") for i in range(2, len(data), 2)
        )
        # An honest builder only emits reduced coefficients; refusing the rest
        # removes a byte-level way to write the same proof twice.
        if any(c >= MOD_N for c in coeffs):
            raise ValueError("a coefficient is outside the field")
        return cls(modulus, coeffs)

    def values_at(self, xs: Sequence[int]) -> list[int]:
        return evaluate(self.coeffs, [int(x) % self.modulus for x in xs])


# The reference's value when no exponent agrees, so no mantissa can be compared.
NO_MANTISSA = float(2**64)

_BF16_EXP_MASK = 0x7F80
_BF16_MANT_MASK = 0x007F


@dataclass(frozen=True, slots=True)
class ChunkResult:
    """Three error measures: acceptance is always at or below a threshold."""

    exp_mismatches: int
    mant_err_mean: float
    mant_err_median: float


def compare_bf16_bits(
    proof_bits: Sequence[int], verifier_bits: Sequence[int]
) -> ChunkResult:
    """The per-chunk comparison of ``verify_proofs`` in poly.cpp."""
    if len(proof_bits) != len(verifier_bits):
        raise ValueError("proof and verifier value counts differ")
    mismatches = 0
    errors: list[int] = []
    for proof, verifier in zip(proof_bits, verifier_bits):
        if (proof & _BF16_EXP_MASK) != (verifier & _BF16_EXP_MASK):
            mismatches += 1
        else:
            errors.append(abs((proof & _BF16_MANT_MASK) - (verifier & _BF16_MANT_MASK)))
    if not errors:
        return ChunkResult(mismatches, NO_MANTISSA, NO_MANTISSA)
    errors.sort()
    return ChunkResult(
        mismatches, sum(errors) / len(errors), float(errors[len(errors) // 2])
    )


@dataclass(frozen=True, slots=True)
class ToplocThresholds:
    exp_mismatch: int
    mant_mean: float
    mant_median: float
    min_allowed_failures: int = 0
    ratio_allowed_failures: float = 0.0


def sequence_verdict(
    results: Sequence[ChunkResult], thresholds: ToplocThresholds
) -> tuple[bool, str | None]:
    """toploc-validator's validate_stage_results @ 55c1a23, except that an empty
    sequence fails closed instead of passing."""
    if not results:
        return False, "no_chunks"
    t = thresholds
    if t.min_allowed_failures > 0 or t.ratio_allowed_failures > 0:
        allowance = max(t.min_allowed_failures, t.ratio_allowed_failures * len(results))
        kept: list[ChunkResult] = []
        forgiven: list[ChunkResult] = []
        for result in results:
            if result.exp_mismatches <= t.exp_mismatch:
                kept.append(result)
            else:
                forgiven.append(result)
                allowance -= 1
        if allowance < 0:
            return False, "too_many_exp_mismatches"
        if any(r.mant_err_mean > t.mant_mean for r in forgiven):
            return False, "mant_err_mean"
        if any(r.mant_err_median > t.mant_median for r in forgiven):
            return False, "mant_err_median"
        results = kept
    for result in results:
        if result.exp_mismatches > t.exp_mismatch:
            return False, "exp_mismatch"
        if result.mant_err_mean > t.mant_mean:
            return False, "mant_err_mean"
        if result.mant_err_median > t.mant_median:
            return False, "mant_err_median"
    return True, None
