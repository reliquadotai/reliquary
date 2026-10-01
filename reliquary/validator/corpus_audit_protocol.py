"""Versioned JSON boundary between the corpus control and a remote audit executor.

A lease carries what the auditor feeds the model (token ids, the prompt length
and the committed proofs); a result carries, per item, the chunk comparisons
``score_sequences`` computes. The pass/fail decision never crosses: the control
applies the proof's thresholds itself.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reliquary.protocol.toploc_wire import MAX_PROOF_B64_CHARS

AUDIT_PROTOCOL = "reliquary.corpus-audit/v1"
# Bounds of one lease: what keeps a request and a response a few megabytes.
MAX_LEASE_ITEMS = 64
MAX_LEASE_TOKENS = 262_144
MAX_SEQUENCE_TOKENS = 65_536
MAX_ITEM_PROOFS = 4096

ITEM_OK = "ok"
ITEM_ERROR = "error"
ITEM_STATUSES = ("ok", "proof_undecodable", "bad_proof_shape", "error")

ExecutorId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")]
LeaseId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
Proof = Annotated[str, Field(max_length=MAX_PROOF_B64_CHARS)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaimRequest(_Strict):
    executor_id: ExecutorId
    model_id: str = Field(min_length=1, max_length=256)
    model_revision: str = Field(min_length=1, max_length=256)


class HeartbeatRequest(_Strict):
    executor_id: ExecutorId
    detail: dict[str, int | float | str | bool] | None = Field(default=None, max_length=32)


class AuditItem(_Strict):
    tokens: list[Annotated[int, Field(ge=0)]] = Field(min_length=2, max_length=MAX_SEQUENCE_TOKENS)
    prompt_len: int = Field(ge=1)
    proofs: list[Proof] = Field(max_length=MAX_ITEM_PROOFS)


class AuditLease(_Strict):
    protocol: Literal["reliquary.corpus-audit/v1"] = AUDIT_PROTOCOL
    lease_id: LeaseId
    model_id: str
    model_revision: str
    chunk_tokens: int = Field(gt=0)
    topk: int = Field(gt=0)
    expires_at: float
    items: list[AuditItem] = Field(min_length=1, max_length=MAX_LEASE_ITEMS)


class ItemScore(_Strict):
    status: Literal["ok", "proof_undecodable", "bad_proof_shape", "error"]
    # Per chunk: exp_mismatches, mant_err_mean, mant_err_median.
    chunks: list[tuple[Annotated[int, Field(ge=0)], float, float]] = Field(
        default_factory=list, max_length=MAX_ITEM_PROOFS)
    detail: str | None = Field(default=None, max_length=512)

    @field_validator("chunks")
    @classmethod
    def _finite(cls, chunks):
        for _, mean, median in chunks:
            if not (math.isfinite(mean) and math.isfinite(median)) or mean < 0 or median < 0:
                raise ValueError("chunk measures must be finite and non-negative")
        return chunks


class AuditResult(_Strict):
    scores: list[ItemScore] = Field(min_length=1, max_length=MAX_LEASE_ITEMS)


__all__ = [
    "AUDIT_PROTOCOL",
    "AuditItem",
    "AuditLease",
    "AuditResult",
    "ClaimRequest",
    "HeartbeatRequest",
    "ITEM_ERROR",
    "ITEM_OK",
    "ITEM_STATUSES",
    "ItemScore",
    "MAX_LEASE_ITEMS",
    "MAX_LEASE_TOKENS",
]
