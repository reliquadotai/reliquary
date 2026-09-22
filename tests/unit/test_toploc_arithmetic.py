"""TOPLOC's arithmetic, ported from the reference C++. The literal port below
is the oracle: it transcribes ndd.cpp line by line, so the vectorised module
can be checked against it on random inputs."""

import random

import pytest

from reliquary.protocol.toploc import (
    MOD_N,
    ChunkProof,
    evaluate,
    expected_chunks,
    injective_modulus,
    newton_coefficients,
)


def _literal_newton(x, y):
    """ndd.cpp compute_newton_coefficients, transcribed without cleverness."""
    n = len(x)
    dd = [v % MOD_N for v in y]
    for k in range(1, n):
        for i in range(n - 1, k - 1, -1):
            num = (dd[i] - dd[i - 1]) % MOD_N
            den = (x[i] - x[i - k]) % MOD_N
            dd[i] = (num * pow(den, MOD_N - 2, MOD_N)) % MOD_N
    coeffs = [0] * n
    factor = [0] * n
    factor[0] = 1
    for i in range(n):
        for j in range(i + 1):
            coeffs[j] = (coeffs[j] + dd[i] * factor[j]) % MOD_N
        if i + 1 < n:
            minus = (-x[i]) % MOD_N
            prev = factor[0]
            factor[0] = (prev * minus) % MOD_N
            for k in range(1, i + 2):
                old = factor[k]
                factor[k] = (prev + old * minus) % MOD_N
                prev = old
    return coeffs


def test_the_field_is_the_reference_prime():
    assert MOD_N == 65497
    assert all(MOD_N % d for d in range(2, int(MOD_N**0.5) + 1))


def test_vectorised_newton_matches_the_literal_port():
    rng = random.Random(7)
    for _ in range(20):
        x = rng.sample(range(0, 60000), 128)
        y = [rng.randrange(0, 65536) for _ in x]
        assert newton_coefficients(x, y) == _literal_newton(x, y)


def test_the_polynomial_passes_through_its_points():
    rng = random.Random(3)
    x = rng.sample(range(0, 60000), 128)
    y = [rng.randrange(0, MOD_N) for _ in x]
    assert evaluate(newton_coefficients(x, y), x) == y


def test_the_modulus_is_the_largest_injective_one():
    assert injective_modulus([1, 2, 3]) == 65497
    # 65497 and 0 collide under 65497, so the search must step down.
    assert injective_modulus([0, 65497]) == 65496


def test_repeated_points_have_no_modulus():
    with pytest.raises(ValueError):
        injective_modulus([5, 5])


def test_bytes_round_trip_and_layout():
    proof = ChunkProof(modulus=65497, coeffs=(1, 258, 65496))
    raw = proof.to_bytes()
    assert raw == bytes([0xFF, 0xD9, 0x00, 0x01, 0x01, 0x02, 0xFF, 0xD8])
    assert ChunkProof.from_bytes(raw) == proof


@pytest.mark.parametrize(
    "raw",
    [
        b"",                      # nothing
        b"\xff\xd9",              # modulus but no coefficient
        b"\xff\xd9\x00",          # odd length: the reference would drop a byte
        b"\x00\x00\x00\x01",      # zero modulus: the reference's null proof
    ],
)
def test_malformed_bytes_are_refused(raw):
    with pytest.raises(ValueError):
        ChunkProof.from_bytes(raw)


def test_values_at_reduces_by_the_proof_modulus():
    rng = random.Random(11)
    x = rng.sample(range(0, 200000), 128)   # indices beyond the field, as in a real chunk
    y = [rng.randrange(0, 65536) for _ in x]
    proof = ChunkProof.from_points(x, y)
    assert proof.values_at(x) == [v % MOD_N for v in y]


def test_a_128_point_proof_is_258_bytes():
    rng = random.Random(1)
    x = rng.sample(range(0, 81920), 128)
    y = [rng.randrange(0, 65536) for _ in x]
    assert len(ChunkProof.from_points(x, y).to_bytes()) == 258


@pytest.mark.parametrize("tokens,expected", [(1, 1), (32, 1), (33, 2), (70, 3), (64, 2)])
def test_expected_chunks(tokens, expected):
    assert expected_chunks(tokens, 32) == expected
