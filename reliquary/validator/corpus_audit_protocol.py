"""Versioned JSON boundary between the corpus control and a remote audit executor.

A lease carries what the auditor feeds the model (token ids, the prompt length
and the committed proofs); a result carries, per item, the chunk comparisons
``score_sequences`` computes. The pass/fail decision never crosses: the control
applies the proof's thresholds itself.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reliquary.protocol.toploc_wire import MAX_PROOF_B64_CHARS

AUDIT_PROTOCOL = "reliquary.corpus-audit/v1"
# Agentic trajectories: items carry their assistant spans (spec §5 N4).
AUDIT_PROTOCOL_V2 = "reliquary.corpus-audit/v2"
MAX_ITEM_SPANS = 64
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
    # What the executor can score; an executor that predates v2 sends none.
    protocols: list[Literal["reliquary.corpus-audit/v1", "reliquary.corpus-audit/v2"]] = Field(
        default_factory=lambda: [AUDIT_PROTOCOL], min_length=1, max_length=2)


class HeartbeatRequest(_Strict):
    executor_id: ExecutorId
    detail: dict[str, int | float | str | bool] | None = Field(default=None, max_length=32)
    # The leases the executor holds right now (grade executors). Absent (an
    # older executor, an audit one): the control takes nothing back on it.
    held_leases: list[LeaseId] | None = Field(default=None, max_length=256)


class AuditItem(_Strict):
    tokens: list[Annotated[int, Field(ge=0)]] = Field(min_length=2, max_length=MAX_SEQUENCE_TOKENS)
    prompt_len: int = Field(ge=1)
    proofs: list[Proof] = Field(max_length=MAX_ITEM_PROOFS)
    # v2: assistant spans in `tokens` coordinates; `proofs` are their lists concatenated.
    spans: list[tuple[int, int]] | None = Field(default=None, max_length=MAX_ITEM_SPANS)

    @model_validator(mode="after")
    def _spans_fit(self) -> "AuditItem":
        if self.spans is not None:
            edge = self.prompt_len
            for start, end in self.spans:
                if not (edge <= start < end <= len(self.tokens)):
                    raise ValueError("spans must be ordered, disjoint, after the prompt and inside tokens")
                edge = end
        return self


class AuditLease(_Strict):
    protocol: Literal["reliquary.corpus-audit/v1", "reliquary.corpus-audit/v2"] = AUDIT_PROTOCOL
    lease_id: LeaseId
    model_id: str
    model_revision: str
    chunk_tokens: int = Field(gt=0)
    topk: int = Field(gt=0)
    expires_at: float
    items: list[AuditItem] = Field(min_length=1, max_length=MAX_LEASE_ITEMS)
    min_chunk_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _v2_iff_spans(self) -> "AuditLease":
        spanned = any(item.spans is not None for item in self.items)
        if spanned != (self.protocol == AUDIT_PROTOCOL_V2) or spanned != (self.min_chunk_tokens is not None):
            raise ValueError("a v2 lease carries spans and min_chunk_tokens, a v1 lease neither")
        return self


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
    "AUDIT_PROTOCOL_V2",
    "MAX_ITEM_SPANS",
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
