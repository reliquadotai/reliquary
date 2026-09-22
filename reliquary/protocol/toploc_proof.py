"""Build and verify TOPLOC proofs over a completion's final hidden states.

Row j is the hidden state at position prompt_len - 1 + j, the one that produced
completion token j: this is the reference's skip_prefill mode, where the prompt
is not proven. Both sides cast to bf16 first, because comparing fp32 bits with
a proof's bf16 bits would be meaningless.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from reliquary.protocol.toploc import (
    NO_MANTISSA,
    ChunkProof,
    ChunkResult,
    compare_bf16_bits,
)


def _chunks(hidden: torch.Tensor, chunk_tokens: int) -> Iterator[torch.Tensor]:
    if hidden.dim() != 2 or hidden.shape[0] == 0:
        raise ValueError(f"expected [rows, width] activations, got {tuple(hidden.shape)}")
    hidden = hidden.to(torch.bfloat16)
    for start in range(0, hidden.shape[0], chunk_tokens):
        yield hidden[start : start + chunk_tokens].reshape(-1)


def _top(flat: torch.Tensor, topk: int) -> tuple[list[int], list[int]]:
    if flat.numel() < topk:
        raise ValueError(f"a chunk of {flat.numel()} activations cannot yield top-{topk}")
    indices = flat.abs().topk(topk).indices
    values = flat[indices].detach().to("cpu").contiguous()
    bits = (values.view(torch.int16).to(torch.int32) & 0xFFFF).tolist()
    return indices.to("cpu").tolist(), bits


def build_chunk_proofs(
    hidden: torch.Tensor, *, chunk_tokens: int, topk: int
) -> list[bytes]:
    proofs = []
    for flat in _chunks(hidden, chunk_tokens):
        indices, bits = _top(flat, topk)
        proofs.append(ChunkProof.from_points(indices, bits).to_bytes())
    return proofs


def verify_chunk_proofs(
    hidden: torch.Tensor,
    proofs: Sequence[bytes],
    *,
    chunk_tokens: int,
    topk: int,
) -> list[ChunkResult]:
    """One result per chunk; an unreadable proof is a failing result, not an error."""
    flats = list(_chunks(hidden, chunk_tokens))
    if len(flats) != len(proofs):
        raise ValueError(f"{len(proofs)} proofs for {len(flats)} chunks")
    results = []
    for flat, raw in zip(flats, proofs):
        indices, bits = _top(flat, topk)
        try:
            proof = ChunkProof.from_bytes(raw)
            # Exactly topk coefficients, as an honest builder emits: checked
            # before Horner, whose cost grows with every forged coefficient.
            if len(proof.coeffs) != topk:
                raise ValueError(f"{len(proof.coeffs)} coefficients, expected {topk}")
            proof_bits = proof.values_at(indices)
        except ValueError:
            results.append(ChunkResult(topk, NO_MANTISSA, NO_MANTISSA))
            continue
        results.append(compare_bf16_bits(proof_bits, bits))
    return results
