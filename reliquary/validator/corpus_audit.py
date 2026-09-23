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


@torch.no_grad()
def completion_hidden_states(model, tokens: Sequence[int], prompt_len: int) -> torch.Tensor:
    """Final hidden state at every position that produced a completion token."""
    if not 0 < prompt_len < len(tokens):
        raise ValueError(f"prompt_len {prompt_len} leaves no completion in {len(tokens)} tokens")
    device = next(model.parameters()).device
    ids = torch.tensor([list(tokens)], device=device)
    output = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return output.hidden_states[-1][0, prompt_len - 1 : len(tokens) - 1]


def audit_completion(
    hidden: torch.Tensor, proofs_b64: Sequence[str], proof: ProofProfile
) -> AuditOutcome:
    if proof.scheme != PROOF_SCHEME_TOPLOC:
        raise ValueError(f"the corpus audit verifies toploc, not {proof.scheme!r}")
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
