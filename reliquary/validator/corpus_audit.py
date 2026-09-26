"""The corpus audit's identity check: re-run the pinned model over a sampled
submission and verify the miner's TOPLOC proofs against its own activations.

A drawn submission that fails is paid nothing and voids the miner's epoch
credit (spec section 9); this module only returns the verdict.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from reliquary.protocol.profiles import PROOF_SCHEME_TOPLOC, ProofProfile
from reliquary.protocol.toploc import ChunkResult, sequence_verdict
from reliquary.protocol.toploc_proof import verify_chunk_proofs


@dataclass(frozen=True, slots=True)
class AuditOutcome:
    passed: bool
    reason: str | None
    results: tuple[ChunkResult, ...] = ()


def _decoder(model):
    # The base model returns only the last (normed) hidden state; asking the LM
    # head model for all of them costs ~20 GB at 32k tokens on a 27B model.
    return model.model if hasattr(model, "model") else model.get_decoder()


@torch.no_grad()
def completion_hidden_states(model, tokens: Sequence[int], prompt_len: int) -> torch.Tensor:
    """Final hidden state at every position that produced a completion token."""
    return batch_completion_hidden_states(model, [(list(tokens), prompt_len)])[0]


@torch.no_grad()
def batch_completion_hidden_states(
    model, sequences: Sequence[tuple[Sequence[int], int]]
) -> list[torch.Tensor]:
    """The same rows for several sequences, right-padded into one forward pass."""
    vocabulary = model.get_input_embeddings().num_embeddings
    for tokens, prompt_len in sequences:
        if not 0 < prompt_len < len(tokens):
            raise ValueError(f"prompt_len {prompt_len} leaves no completion in {len(tokens)} tokens")
        if min(tokens) < 0 or max(tokens) >= vocabulary:
            # On CUDA an out-of-range embedding index kills the device context.
            raise ValueError(f"a token id is outside the vocabulary of {vocabulary}")
    device = next(model.parameters()).device
    width = max(len(tokens) for tokens, _ in sequences)
    ids = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, (tokens, _) in enumerate(sequences):
        ids[row, : len(tokens)] = torch.tensor(tokens, device=device)
        mask[row, : len(tokens)] = 1
    hidden = _decoder(model)(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    return [hidden[row, n - 1 : len(tokens) - 1] for row, (tokens, n) in enumerate(sequences)]


def audit_completion(
    hidden: torch.Tensor, proofs_b64: Sequence[str], proof: ProofProfile
) -> AuditOutcome:
    if proof.scheme != PROOF_SCHEME_TOPLOC:
        raise ValueError(f"the corpus audit verifies toploc, not {proof.scheme!r}")
    # Raised, not returned as a verdict: these are the validator's own errors,
    # and a failed audit would void an honest miner's epoch credit.
    if hidden.dim() != 2:
        raise ValueError(f"configuration: expected [rows, width] activations, got {tuple(hidden.shape)}")
    if proof.topk > hidden.shape[1]:
        raise ValueError(
            f"configuration: topk {proof.topk} exceeds the model width {hidden.shape[1]}"
        )
    try:
        raw = [base64.b64decode(p, validate=True) for p in proofs_b64]
    except (binascii.Error, ValueError):
        return AuditOutcome(False, "proof_undecodable")
    try:
        results = verify_chunk_proofs(
            hidden, raw, chunk_tokens=proof.chunk_tokens, topk=proof.topk
        )
    except ValueError:
        return AuditOutcome(False, "bad_proof_shape")
    passed, reason = sequence_verdict(results, proof.thresholds())
    return AuditOutcome(passed, reason, tuple(results))
