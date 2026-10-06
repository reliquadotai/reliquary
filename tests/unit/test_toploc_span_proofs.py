"""Span chunking: a short trailing chunk is merged into the previous one."""

import pytest
import torch

from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.toploc import sequence_verdict
from reliquary.protocol.toploc_proof import (
    MIN_CHUNK_TOKENS,
    build_chunk_proofs,
    build_span_proofs,
    span_chunk_bounds,
    span_chunk_count,
    verify_chunk_proofs,
    verify_span_proofs,
)


@pytest.mark.parametrize("length,bounds", [
    (1, [(0, 1)]),
    (7, [(0, 7)]),
    (32, [(0, 32)]),
    (33, [(0, 33)]),                       # 1-token tail merged
    (39, [(0, 39)]),                       # 7-token tail merged
    (40, [(0, 32), (32, 40)]),             # 8-token tail stands alone
    (70, [(0, 32), (32, 70)]),             # 6-token tail merged into the second chunk
    (96, [(0, 32), (32, 64), (64, 96)]),
])
def test_bounds(length, bounds):
    assert span_chunk_bounds(length, 32) == bounds
    assert span_chunk_count(length, 32) == len(bounds)


def test_bounds_refuse_nonsense():
    for bad in [(0, 32, 8), (10, 0, 8), (10, 32, 0), (10, 32, 33)]:
        with pytest.raises(ValueError):
            span_chunk_bounds(*bad)


def test_without_a_short_tail_span_proofs_equal_chunk_proofs():
    torch.manual_seed(0)
    hidden = torch.randn(72, 128).to(torch.bfloat16)        # 32 + 32 + 8
    assert build_span_proofs(hidden, chunk_tokens=32, topk=128) == \
        build_chunk_proofs(hidden, chunk_tokens=32, topk=128)


def test_a_merged_tail_round_trips_and_differs_from_fixed_chunks():
    torch.manual_seed(1)
    hidden = torch.randn(35, 128).to(torch.bfloat16)        # 32 + 3 -> one chunk of 35
    proofs = build_span_proofs(hidden, chunk_tokens=32, topk=128)
    assert len(proofs) == 1 and len(build_chunk_proofs(hidden, chunk_tokens=32, topk=128)) == 2
    results = verify_span_proofs(hidden, proofs, chunk_tokens=32, topk=128)
    assert [r.exp_mismatches for r in results] == [0]


def test_verify_refuses_a_proof_count_that_is_not_the_span_count():
    hidden = torch.randn(35, 128).to(torch.bfloat16)
    proofs = build_chunk_proofs(hidden, chunk_tokens=32, topk=128)  # 2, span needs 1
    with pytest.raises(ValueError):
        verify_span_proofs(hidden, proofs, chunk_tokens=32, topk=128)


@pytest.mark.parametrize("tail_tokens", range(1, MIN_CHUNK_TOKENS + 1))
def test_synthetic_tail_outlier_is_merged_only_below_the_standalone_boundary(tail_tokens):
    hidden = torch.cat((
        torch.full((PROOF.chunk_tokens, PROOF.topk), 4.0, dtype=torch.bfloat16),
        torch.full((tail_tokens, PROOF.topk), 0.25, dtype=torch.bfloat16),
    ))
    replayed = hidden.clone()
    replayed[-tail_tokens:] *= 2  # Change only the low-magnitude tail's bf16 exponent.
    shape = {"chunk_tokens": PROOF.chunk_tokens, "topk": PROOF.topk}
    fixed = verify_chunk_proofs(replayed, build_chunk_proofs(hidden, **shape), **shape)
    assert len(fixed) == 2 and fixed[0].exp_mismatches == 0
    assert fixed[1].exp_mismatches > PROOF.exp_mismatch_threshold
    assert sequence_verdict(fixed, PROOF.thresholds()) == (False, "exp_mismatch")

    spans = verify_span_proofs(replayed, build_span_proofs(hidden, **shape), **shape)
    if tail_tokens < MIN_CHUNK_TOKENS:
        assert span_chunk_bounds(len(hidden), PROOF.chunk_tokens) == [(0, len(hidden))]
        assert len(spans) == 1 and spans[0].exp_mismatches == fixed[0].exp_mismatches
        assert sequence_verdict(spans, PROOF.thresholds()) == (True, None)
    else:
        assert span_chunk_bounds(len(hidden), PROOF.chunk_tokens) == [
            (0, PROOF.chunk_tokens), (PROOF.chunk_tokens, len(hidden)),
        ]
        assert spans == fixed
        assert sequence_verdict(spans, PROOF.thresholds()) == (False, "exp_mismatch")
