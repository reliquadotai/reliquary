"""Span chunking: a short trailing chunk is merged into the previous one."""

import json
from pathlib import Path

import pytest
import torch

from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.toploc_proof import (
    MIN_CHUNK_TOKENS,
    build_chunk_proofs,
    build_span_proofs,
    span_chunk_bounds,
    span_chunk_count,
    verify_chunk_proofs,
    verify_span_proofs,
)

_M1 = Path(__file__).resolve().parents[2] / "docs/design/measurements"


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


def test_m1_short_chunks_are_the_only_ones_above_the_full_chunk_band():
    chunks = []  # (chunk length, exp mismatches) over both M1 runs
    for name in ("2026-10-03-m1-agentic-proofs.json",
                 "2026-10-03-m1-agentic-proofs-control-cache-off.json"):
        for row in json.loads((_M1 / name).read_text())["report"]:
            start, end = row["span"]
            for i, (exp, _mean, _median) in enumerate(row["chunks"]):
                chunks.append((min(PROOF.chunk_tokens, end - start - PROOF.chunk_tokens * i), exp))
    full_max = max(exp for length, exp in chunks if length == PROOF.chunk_tokens)
    outliers = {length for length, exp in chunks if exp > full_max}
    assert outliers and max(outliers) < MIN_CHUNK_TOKENS
