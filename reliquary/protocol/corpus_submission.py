"""Pydantic v2 models for the miner->validator corpus generation protocol.

Named ``corpus`` rather than ``batch``: ``BatchSubmissionRequest`` in
``submission.py`` is the GRPO training batch, and the two must not be confused.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CorpusRejectReason(str, Enum):
    """Canonical verdicts. ``ACCEPTED`` is the one success value.

    The four cheap-check reasons are duplicated as plain strings in
    ``reliquary.corpus.checks`` so that module stays free of pydantic; the
    schema test pins them equal.
    """

    ACCEPTED = "accepted"
    BAD_SIGNATURE = "bad_signature"
    JOB_UNKNOWN = "job_unknown"
    JOB_COMPLETE = "job_complete"
    CHECKPOINT_MISMATCH = "checkpoint_mismatch"
    BAD_CURSOR = "bad_cursor"
    PROMPT_MISMATCH = "prompt_mismatch"
    PROMPT_FULL = "prompt_full"
    BAD_COMPLETION_COUNT = "bad_completion_count"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    BAD_TERMINATION = "bad_termination"
    HASH_DUPLICATE = "hash_duplicate"
    DEGENERATE = "degenerate"


class CorpusCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(min_length=1)
    text: str
    termination: Literal["eos", "cap"]

    @field_validator("tokens")
    @classmethod
    def _token_ids_are_not_negative(cls, value: list[int]) -> list[int]:
        if any(token < 0 for token in value):
            raise ValueError("token ids must not be negative")
        return value


class CorpusSubmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1)
    miner_hotkey: str = Field(min_length=1)
    cursor: int = Field(ge=0)
    prompt_index: int = Field(ge=0)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completions: list[CorpusCompletion] = Field(min_length=1)
    signature: str = Field(min_length=1)


class CorpusSubmissionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: CorpusRejectReason
    accepted: bool
    slots_remaining: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
