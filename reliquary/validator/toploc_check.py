"""TOPLOC beside GRAIL in the RL proof: one verdict from the hidden states the
GRAIL check already computed.

The thresholds come from the ``toploc_spec`` the validator put in its own copy
of the commit, from its task contract; a miner-sent spec never reaches here,
because CommitModel refuses it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from reliquary.protocol.profiles import ProofProfile
from reliquary.validator.corpus_audit import audit_completion


@dataclass(frozen=True, slots=True)
class ToplocVerdict:
    passed: bool
    reason: str | None
    worst_exp: int
    worst_mant_mean: float
    worst_mant_median: float


def toploc_verdict(
    hidden: torch.Tensor, commit: Mapping, prompt_length: int
) -> ToplocVerdict | None:
    spec = commit.get("toploc_spec")
    if spec is None:
        return None
    try:
        proof = ProofProfile(**spec)
        proofs = commit.get("toploc_proofs")
        if proofs is None:
            return ToplocVerdict(False, "missing", 0, 0.0, 0.0)
        rows = hidden[prompt_length - 1 : hidden.shape[0] - 1] if prompt_length > 0 else hidden[:0]
        outcome = audit_completion(rows, proofs, proof)
    except Exception as exc:
        # The validator's own error. Enforced, it must stay loud rather than
        # fail an honest miner; in shadow it must not cost the GRAIL proof.
        if spec.get("mode") != "shadow":
            raise
        return ToplocVerdict(False, f"error:{type(exc).__name__}", 0, 0.0, 0.0)
    results = outcome.results
    return ToplocVerdict(
        outcome.passed,
        outcome.reason,
        max((r.exp_mismatches for r in results), default=0),
        max((r.mant_err_mean for r in results), default=0.0),
        max((r.mant_err_median for r in results), default=0.0),
    )
