"""Pydantic v2 models for the miner->validator corpus generation protocol.

Named ``corpus`` rather than ``batch``: ``BatchSubmissionRequest`` in
``submission.py`` is the GRPO training batch, and the two must not be confused.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Ceilings the parser enforces before any check can run. The text bound is the
# token ceiling at a generous 8 characters per token: `text` carries the most
# bytes of any field, so leaving it unbounded leaves the others decorative.
MAX_COMPLETION_TOKENS = 131072
MAX_COMPLETION_TEXT_CHARS = MAX_COMPLETION_TOKENS * 8
MAX_COMPLETIONS_PER_SUBMISSION = 64
# The prompt the miner conditioned on. Same ceiling as one completion's text:
# a prompt longer than a full generation would not fit the context either.
MAX_RENDERED_PROMPT_CHARS = MAX_COMPLETION_TEXT_CHARS


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
    # The validator-side text check (`validator/corpus_text.py`): payment
    # counts tokens, the corpus is made of text, and a completion whose text
    # is not its tokens is paid for nothing.
    TEXT_MISMATCH = "text_does_not_match_tokens"
    # The prompt the miner says it conditioned on is not the source row the
    # job assigned to that slot, rendered by the job's own renderer.
    PROMPT_NOT_FAITHFUL = "prompt_not_faithful"
    DEGENERATE = "degenerate"


class CorpusCompletion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(min_length=1, max_length=MAX_COMPLETION_TOKENS)
    text: str = Field(max_length=MAX_COMPLETION_TEXT_CHARS)
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
    # Required, not optional: prompt fidelity is a free-tier check, and a field
    # a miner may omit is a check a miner may switch off.
    rendered_prompt: str = Field(min_length=1, max_length=MAX_RENDERED_PROMPT_CHARS)
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
