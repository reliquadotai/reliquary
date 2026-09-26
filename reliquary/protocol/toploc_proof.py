"""Build and verify TOPLOC proofs over a completion's final hidden states.

Row j is the hidden state at position prompt_len - 1 + j, the one that produced
completion token j: this is the reference's skip_prefill mode, where the prompt
is not proven. Both sides cast to bf16 first, because comparing fp32 bits with
a proof's bf16 bits would be meaningless.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence

import torch

from reliquary.protocol.toploc import (
    NO_MANTISSA,
    ChunkProof,
    ChunkResult,
    compare_bf16_bits,
    evaluate_batch,
    injective_modulus,
    newton_coefficients_batch,
)


def _chunk_tops(
    hidden: torch.Tensor, chunk_tokens: int, topk: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Each chunk's top-k indices by magnitude and the bf16 bits there."""
    if hidden.dim() != 2 or hidden.shape[0] == 0:
        raise ValueError(f"expected [rows, width] activations, got {tuple(hidden.shape)}")
    hidden = hidden.to(torch.bfloat16)
    # One abs over the whole tensor: per chunk it cost ~60x more on CPU.
    magnitude = hidden.abs()
    all_indices, all_bits = [], []
    for start in range(0, hidden.shape[0], chunk_tokens):
        flat = hidden[start : start + chunk_tokens].reshape(-1)
        if flat.numel() < topk:
            raise ValueError(f"a chunk of {flat.numel()} activations cannot yield top-{topk}")
        indices = magnitude[start : start + chunk_tokens].reshape(-1).topk(topk).indices
        values = flat[indices].detach().to("cpu").contiguous()
        all_bits.append((values.view(torch.int16).to(torch.int32) & 0xFFFF).tolist())
        all_indices.append(indices.to("cpu").tolist())
    return all_indices, all_bits


def build_chunk_proofs(
    hidden: torch.Tensor, *, chunk_tokens: int, topk: int
) -> list[bytes]:
    all_indices, all_bits = _chunk_tops(hidden, chunk_tokens, topk)
    moduli = [injective_modulus(indices) for indices in all_indices]
    reduced = [[i % m for i in indices] for indices, m in zip(all_indices, moduli)]
    coeffs = newton_coefficients_batch(reduced, all_bits)
    return [
        ChunkProof(m, tuple(int(c) for c in row)).to_bytes()
        for m, row in zip(moduli, coeffs)
    ]


def verify_chunk_proofs(
    hidden: torch.Tensor,
    proofs: Sequence[bytes],
    *,
    chunk_tokens: int,
    topk: int,
) -> list[ChunkResult]:
    """One result per chunk; an unreadable proof is a failing result, not an error."""
    all_indices, all_bits = _chunk_tops(hidden, chunk_tokens, topk)
    if len(all_indices) != len(proofs):
        raise ValueError(f"{len(proofs)} proofs for {len(all_indices)} chunks")
    parsed: dict[int, ChunkProof] = {}
    for chunk, raw in enumerate(proofs):
        try:
            proof = ChunkProof.from_bytes(raw)
        except ValueError:
            continue
        # Exactly topk coefficients, as an honest builder emits: checked
        # before Horner, whose cost grows with every forged coefficient.
        if len(proof.coeffs) == topk:
            parsed[chunk] = proof
    readable = sorted(parsed)
    values = (
        evaluate_batch(
            [parsed[c].coeffs for c in readable],
            [[i % parsed[c].modulus for i in all_indices[c]] for c in readable],
        )
        if readable
        else []
    )
    proof_bits = {c: [int(v) for v in row] for c, row in zip(readable, values)}
    return [
        compare_bf16_bits(proof_bits[c], all_bits[c])
        if c in proof_bits
        else ChunkResult(topk, NO_MANTISSA, NO_MANTISSA)
        for c in range(len(proofs))
    ]


def completion_proofs_b64(
    hidden: torch.Tensor,
    prompt_length: int,
    total_length: int,
    *,
    chunk_tokens: int,
    topk: int,
) -> list[str]:
    """Wire-ready proofs over the rows of a full-sequence forward that produced
    the completion: positions prompt_length - 1 .. total_length - 2."""
    if not 0 < prompt_length < total_length or hidden.shape[0] < total_length - 1:
        raise ValueError(
            f"no completion rows for prompt {prompt_length} in {total_length} tokens "
            f"over {hidden.shape[0]} rows"
        )
    rows = hidden[prompt_length - 1 : total_length - 1]
    proofs = build_chunk_proofs(rows, chunk_tokens=chunk_tokens, topk=topk)
    return [base64.b64encode(proof).decode() for proof in proofs]
