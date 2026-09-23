"""Proofs over real tensors. Rows are the hidden states that produced the
completion tokens; chunks of 32 rows, the last one possibly short."""

import base64 as _b64

import pytest
import torch

from reliquary.protocol.toploc import NO_MANTISSA
from reliquary.protocol.toploc_proof import (
    build_chunk_proofs,
    completion_proofs_b64,
    verify_chunk_proofs,
)

KW = {"chunk_tokens": 32, "topk": 128}


def _hidden(rows=70, width=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, width, generator=g).to(torch.bfloat16)


def test_one_proof_per_chunk_including_the_short_last_one():
    assert len(build_chunk_proofs(_hidden(rows=70), **KW)) == 3


def test_the_same_activations_verify_exactly():
    hidden = _hidden()
    results = verify_chunk_proofs(hidden, build_chunk_proofs(hidden, **KW), **KW)
    assert all(r.exp_mismatches == 0 and r.mant_err_mean == 0.0 for r in results)


def test_unrelated_activations_do_not_verify():
    proofs = build_chunk_proofs(_hidden(seed=0), **KW)
    results = verify_chunk_proofs(_hidden(seed=1), proofs, **KW)
    assert all(r.exp_mismatches > 100 for r in results)


def test_float32_activations_are_compared_as_bf16():
    hidden = _hidden()
    proofs = build_chunk_proofs(hidden.float(), **KW)
    results = verify_chunk_proofs(hidden, proofs, **KW)
    assert all(r.exp_mismatches == 0 for r in results)


def test_a_malformed_proof_fails_closed():
    hidden = _hidden()
    proofs = build_chunk_proofs(hidden, **KW)
    proofs[1] = b"\x00\x00\x00\x01"          # the reference's null proof
    results = verify_chunk_proofs(hidden, proofs, **KW)
    assert results[1].exp_mismatches == 128
    assert results[1].mant_err_mean == NO_MANTISSA
    assert results[0].exp_mismatches == 0


def test_a_wrong_number_of_proofs_is_refused():
    hidden = _hidden()
    with pytest.raises(ValueError):
        verify_chunk_proofs(hidden, build_chunk_proofs(hidden, **KW)[:-1], **KW)


def test_a_chunk_smaller_than_topk_is_refused():
    with pytest.raises(ValueError):
        build_chunk_proofs(_hidden(rows=1, width=64), **KW)


def test_a_proof_with_the_wrong_number_of_coefficients_fails_closed():
    # An honest proof has exactly topk coefficients; anything else is refused
    # before Horner, which would otherwise run once per forged coefficient.
    hidden = _hidden()
    proofs = build_chunk_proofs(hidden, **KW)
    proofs[0] = proofs[0] + b"\x00\x01" * 1000
    results = verify_chunk_proofs(hidden, proofs, **KW)
    assert results[0].exp_mismatches == 128
    assert results[0].mant_err_mean == NO_MANTISSA





def test_completion_proofs_cover_exactly_the_completion_rows():
    full = _hidden(rows=12 + 70, seed=3)        # 12 prompt tokens, 70 completion tokens
    proofs = completion_proofs_b64(full, 12, 82, **KW)
    assert len(proofs) == 3
    rows = full[11:81]
    assert [_b64.b64decode(p) for p in proofs] == build_chunk_proofs(rows, **KW)


@pytest.mark.parametrize("prompt_length,total", [(0, 10), (10, 10), (5, 99)])
def test_completion_proofs_refuse_impossible_bounds(prompt_length, total):
    with pytest.raises(ValueError):
        completion_proofs_b64(_hidden(rows=10), prompt_length, total, **KW)
