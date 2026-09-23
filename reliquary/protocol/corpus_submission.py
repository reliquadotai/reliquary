"""Pydantic v2 models for the miner->validator corpus generation protocol.

Named ``corpus`` rather than ``batch``: ``BatchSubmissionRequest`` in
``submission.py`` is the GRPO training batch, and the two must not be confused.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reliquary.protocol.toploc_wire import (
    MAX_PROOF_B64_CHARS,
    MAX_PROOF_BYTES,
    MAX_PROOF_CHARS_PER_TOKEN,
    PROOF_CHARS_SLACK,
    ProofB64,
    proof_volume_error,
)


# Ceilings the parser enforces before any check can run. The text bound is the
# token ceiling at a generous 8 characters per token: `text` carries the most
# bytes of any field, so leaving it unbounded leaves the others decorative.
MAX_COMPLETION_TOKENS = 131072
MAX_COMPLETION_TEXT_CHARS = MAX_COMPLETION_TOKENS * 8
MAX_COMPLETIONS_PER_SUBMISSION = 64


class CorpusRejectReason(str, Enum):
    """Canonical verdicts. ``ACCEPTED`` is the one success value.

    The cheap-check reasons are duplicated as plain strings in
    ``reliquary.corpus.checks`` and ``reliquary.corpus.admission`` so those
    modules stay free of pydantic; the schema test pins them equal.
    """

    ACCEPTED = "accepted"
    BAD_SIGNATURE = "bad_signature"
    MALFORMED_SUBMISSION = "malformed_submission"
    JOB_UNKNOWN = "job_unknown"
    JOB_COMPLETE = "job_complete"
    CHECKPOINT_MISMATCH = "checkpoint_mismatch"
    BAD_CURSOR = "bad_cursor"
    PROMPT_MISMATCH = "prompt_mismatch"
    PROMPT_FULL = "prompt_full"
    BAD_COMPLETION_COUNT = "bad_completion_count"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    TOKEN_BUDGET_UNDERRUN = "token_budget_underrun"
    BAD_TERMINATION = "bad_termination"
    HASH_DUPLICATE = "hash_duplicate"
    DEGENERATE = "degenerate"
    BAD_PROOF_SHAPE = "bad_proof_shape"
    PROOF_FAIL = "proof_fail"


class CorpusCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(min_length=1, max_length=MAX_COMPLETION_TOKENS)
    text: str = Field(max_length=MAX_COMPLETION_TEXT_CHARS)
    termination: Literal["eos", "cap"]
    proofs: list[ProofB64] = Field(default_factory=list, max_length=MAX_COMPLETION_TOKENS)

    @field_validator("tokens")
    @classmethod
    def _token_ids_are_not_negative(cls, value: list[int]) -> list[int]:
        if any(token < 0 for token in value):
            raise ValueError("token ids must not be negative")
        return value

    @model_validator(mode="after")
    def _proofs_fit_the_completion(self) -> "CorpusCompletion":
        error = proof_volume_error(self.proofs, len(self.tokens))
        if error:
            raise ValueError(error)
        return self


class CorpusSubmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1)
    miner_hotkey: str = Field(min_length=1)
    cursor: int = Field(ge=0)
    prompt_index: int = Field(ge=0)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completions: list[CorpusCompletion] = Field(
        min_length=1, max_length=MAX_COMPLETIONS_PER_SUBMISSION
    )
    signature: str = Field(min_length=1)


class CorpusSubmissionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: CorpusRejectReason
    accepted: bool
    slots_remaining: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
