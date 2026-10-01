"""The eval control's executor boundary: claims, qualify leases and results.

Audit leases are R3's (``AuditLease``/``AuditResult``) unchanged; a qualify lease
is told apart by its ``type``. Every model is strict: an unknown field is refused.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reliquary.validator.corpus_audit_protocol import AUDIT_PROTOCOL, ExecutorId, LeaseId

MAX_QUALIFY_PROMPTS = 1024
MAX_QUALIFY_CHUNKS = 1_000_000


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalClaimRequest(_Strict):
    executor_id: ExecutorId
    model_id: str = Field(min_length=1, max_length=256)
    model_revision: str = Field(min_length=1, max_length=256)
    # What the executor will take: an audit lease (R3's executors ask for
    # nothing else), a qualification, or either.
    kind: Literal["any", "audit", "qualify"] = "audit"


class QualifyPrompt(_Strict):
    problem_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1)


class QualifyLease(_Strict):
    protocol: Literal["reliquary.corpus-audit/v1"] = AUDIT_PROTOCOL
    type: Literal["qualify"] = "qualify"
    lease_id: LeaseId
    qualification_id: str
    model_id: str
    model_revision: str
    chunk_tokens: int = Field(gt=0)
    topk: int = Field(gt=0)
    expires_at: float
    prompts: list[QualifyPrompt] = Field(min_length=1, max_length=MAX_QUALIFY_PROMPTS)
    completions: int = Field(gt=0, le=64)
    sampling: dict[str, float | int]
    max_new_tokens: int = Field(gt=0)
    thinking: bool


class QualifyResult(_Strict):
    type: Literal["qualify"] = "qualify"
    # Per chunk of every honest completion: exp_mismatches, mant_err_mean, mant_err_median.
    chunks: list[tuple[Annotated[int, Field(ge=0)], float, float]] = Field(
        min_length=1, max_length=MAX_QUALIFY_CHUNKS)
    completions: int = Field(gt=0)
    # Completions whose own proofs did not verify (status other than ok).
    failed_completions: int = Field(ge=0)
    completion_tokens: int = Field(gt=0)
    decode_seconds: float = Field(gt=0)
    gpu_count: int = Field(gt=0)
    gpu: str = Field(min_length=1, max_length=256)
    vllm_version: str = Field(min_length=1, max_length=64)
    # The fingerprint of what this executor downloaded; both qualifiers must match.
    # The model's eos and architecture are read by the control, never sent.
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("chunks")
    @classmethod
    def _finite(cls, chunks):
        for _, mean, median in chunks:
            if not (math.isfinite(mean) and math.isfinite(median)) or mean < 0 or median < 0:
                raise ValueError("chunk measures must be finite and non-negative")
        return chunks


__all__ = ["EvalClaimRequest", "QualifyLease", "QualifyPrompt", "QualifyResult"]
