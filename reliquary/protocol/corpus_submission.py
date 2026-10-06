"""Pydantic v2 models for the miner->validator corpus generation protocol.

Named ``corpus`` rather than ``batch``: ``BatchSubmissionRequest`` in
``submission.py`` is the GRPO training batch, and the two must not be confused.
"""

from __future__ import annotations

import json
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
# The prompt the miner conditioned on. Same ceiling as one completion's text:
# a prompt longer than a full generation would not fit the context either.
MAX_RENDERED_PROMPT_CHARS = MAX_COMPLETION_TEXT_CHARS

# One agentic trajectory (spec §6): the job's max_total_tokens is at most
# 60,000 over prompt and tokens (gate M3), so the tokens alone stay under it.
# Duplicated from `corpus.job` the way the reject reasons are; the wire test
# pins them equal.
MAX_TRAJECTORY_TOKENS = 60000
MAX_TRAJECTORY_TURNS = 64
MAX_FINAL_DIFF_CHARS = 1_048_576
# The binding writes token ids as 4 bytes.
MAX_TOKEN_ID = 2**32 - 1

# A signed-sandbox trajectory's transcript (records 0 to final, signed by our machine).
# Observations shown to the model are in the tokens already (<= 60k), so an honest
# transcript stays far below this; the last turn's unanswered outputs are the rest.
MAX_TRANSCRIPT_BYTES = 8 * 1024 * 1024


class CorpusRejectReason(str, Enum):
    """Canonical verdicts. ``ACCEPTED`` is the one success value.

    The cheap-check reasons are duplicated as plain strings in
    ``reliquary.corpus.checks`` and ``reliquary.corpus.admission`` so those
    modules stay free of pydantic; the schema test pins them equal.
    """

    ACCEPTED = "accepted"
    BAD_SIGNATURE = "bad_signature"
    # Not the miner's fault: this validator has no way to check a signature at
    # all, and saying "bad_signature" would send it debugging its own keys.
    SIGNATURE_UNVERIFIABLE = "signature_unverifiable"
    # Checked right after the signature, so a spoofed hotkey cannot probe ban
    # status and a banned one never reaches the ledgers.
    MINER_BANNED = "miner_banned"
    # An unregistered hotkey is never paid, so its work would only spend audit GPU.
    HOTKEY_NOT_REGISTERED = "hotkey_not_registered"
    MALFORMED_SUBMISSION = "malformed_submission"
    JOB_UNKNOWN = "job_unknown"
    # Distinct from JOB_UNKNOWN on purpose: "this job exists but this
    # validator is not the one paid for it" and "no such job" send a miner to
    # two different places.
    JOB_NOT_SERVED = "job_not_served"
    JOB_COMPLETE = "job_complete"
    CHECKPOINT_MISMATCH = "checkpoint_mismatch"
    BAD_CURSOR = "bad_cursor"
    PROMPT_MISMATCH = "prompt_mismatch"
    PROMPT_FULL = "prompt_full"
    # A skip names a prompt that still has a slot: only a full prompt may be
    # stepped over without answering it.
    PROMPT_NOT_FULL = "prompt_not_full"
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
    # A token id at or above the job model's vocabulary size.
    TOKEN_OUT_OF_VOCAB = "token_out_of_vocab"
    # RESERVED, no producer yet: spec §7 lists degeneracy/repetition as a
    # free-tier check over `validator/rollout_patterns.py`, and the name is
    # pinned here so the check lands under it rather than inventing a second.
    DEGENERATE = "degenerate"
    # Agentic trajectories (spec §5 N3): turn spans malformed or too many,
    # too many turns too short to prove, a tool segment that is not the pinned
    # renderer's, a tool call no observation answers, a stop its tokens deny.
    BAD_TURNS = "bad_turns"
    SHORT_TURNS = "short_turns"
    BAD_OBSERVATION = "bad_observation"
    UNANSWERED_TOOL_CALL = "unanswered_tool_call"
    TRAJECTORY_TOO_LARGE = "trajectory_too_large"
    BAD_STOP = "bad_stop"
    BAD_PROOF_SHAPE = "bad_proof_shape"
    PROOF_FAIL = "proof_fail"
    # Signed-sandbox trajectories (plan 3, reliquary.corpus.signed_reasons): a
    # transcript that does not verify or bind to this submission, one that arrived
    # after its session's deadline, a session already paid, calls or state that are
    # not the signed ones.
    SANDBOX_TRANSCRIPT_INVALID = "sandbox_transcript_invalid"
    SANDBOX_SESSION_EXPIRED = "sandbox_session_expired"
    SANDBOX_SESSION_REUSED = "sandbox_session_reused"
    SANDBOX_CALL_MISMATCH = "sandbox_call_mismatch"
    SANDBOX_STATE_MISMATCH = "sandbox_state_mismatch"


class CorpusCompletion(BaseModel):
    """Tokens and their text, and nothing the miner says ABOUT them.

    There is no ``termination`` field: the validator derives that label from
    the tokens (``check_termination``) and ``admit`` has no parameter it could
    travel through, so a required enum here could only refuse an honest miner
    that spells its own label ``"stop"`` or ``"length"``.
    """

    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(min_length=1, max_length=MAX_COMPLETION_TOKENS)
    text: str = Field(max_length=MAX_COMPLETION_TEXT_CHARS)
    proofs: list[ProofB64] = Field(default_factory=list, max_length=MAX_COMPLETION_TOKENS)

    @field_validator("tokens")
    @classmethod
    def _token_ids_are_not_negative(cls, value: list[int]) -> list[int]:
        if any(token < 0 or token > MAX_TOKEN_ID for token in value):
            raise ValueError("token ids must fit in 32 bits and not be negative")
        return value

    @model_validator(mode="after")
    def _proofs_fit_the_completion(self) -> "CorpusCompletion":
        error = proof_volume_error(self.proofs, len(self.tokens))
        if error:
            raise ValueError(error)
        return self


class CorpusTurn(BaseModel):
    """One assistant span of a trajectory, in `tokens` coordinates (the first
    turn starts at 0, right after the initial prompt), and its proofs."""

    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(ge=1)
    proofs: list[ProofB64] = Field(default_factory=list, max_length=MAX_TRAJECTORY_TOKENS)

    @model_validator(mode="after")
    def _a_turn_has_tokens_and_fitting_proofs(self) -> "CorpusTurn":
        if self.end <= self.start:
            raise ValueError(f"turn [{self.start}, {self.end}) is empty")
        error = proof_volume_error(self.proofs, self.end - self.start)
        if error:
            raise ValueError(error)
        return self


class CorpusTrajectory(BaseModel):
    """A multi-turn episode: every token after the initial prompt, the
    assistant spans with their proofs, the diff the env collected, and why it
    stopped. Nothing else the miner says about it is read."""

    model_config = ConfigDict(extra="forbid")

    tokens: list[int] = Field(min_length=1, max_length=MAX_TRAJECTORY_TOKENS)
    turns: list[CorpusTurn] = Field(min_length=1, max_length=MAX_TRAJECTORY_TURNS)
    final_diff: str = Field(max_length=MAX_FINAL_DIFF_CHARS)
    # No "episode_closed" (the signed bridge's mid-turn close): such an episode ends
    # unpaid, and `corpus.signed_parse` refuses it with a graded final anyway. Kept
    # out of the wire as defense in depth.
    stop: Literal["agent_completed", "max_turns", "context_length"]
    # Signed-sandbox jobs only: the gateway's transcript `{"token", "records"}`.
    transcript: dict[str, Any] | None = None

    @field_validator("transcript")
    @classmethod
    def _transcript_is_bounded(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return value
        if set(value) != {"token", "records"} or not isinstance(value["records"], list):
            raise ValueError("the transcript is not {\"token\", \"records\": [...]}")
        # allow_nan=False: a NaN or infinity is not JSON, and the signed binding
        # (`transcript_digest`) refuses it too, so it is a malformed body, not a crash.
        size = len(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                              allow_nan=False).encode("utf-8"))
        if size > MAX_TRANSCRIPT_BYTES:
            raise ValueError(f"the transcript is {size} bytes, over {MAX_TRANSCRIPT_BYTES}")
        return value

    @field_validator("tokens")
    @classmethod
    def _token_ids_are_not_negative(cls, value: list[int]) -> list[int]:
        if any(token < 0 or token > MAX_TOKEN_ID for token in value):
            raise ValueError("token ids must fit in 32 bits and not be negative")
        return value

    @model_validator(mode="after")
    def _turns_and_proofs_are_bounded_by_the_tokens(self) -> "CorpusTrajectory":
        # `turn.end` alone bounds nothing: without these, one turn could claim
        # a 10**9-token span and carry tens of thousands of proofs.
        previous_end = 0
        for turn in self.turns:
            if turn.start < previous_end:
                raise ValueError("turns overlap or are out of order")
            if turn.end > len(self.tokens):
                raise ValueError("a turn ends past the trajectory's tokens")
            previous_end = turn.end
        error = proof_volume_error(
            [proof for turn in self.turns for proof in turn.proofs], len(self.tokens)
        )
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
    # Required, not optional: prompt fidelity is a free-tier check, and a field
    # a miner may omit is a check a miner may switch off.
    rendered_prompt: str = Field(min_length=1, max_length=MAX_RENDERED_PROMPT_CHARS)
    completions: list[CorpusCompletion] = Field(
        default_factory=list, max_length=MAX_COMPLETIONS_PER_SUBMISSION
    )
    # An episode job's one trajectory, instead of completions.
    trajectory: CorpusTrajectory | None = None
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def _completions_or_a_trajectory(self) -> "CorpusSubmissionRequest":
        if (self.trajectory is None) == (not self.completions):
            # Neither: nothing to pay. Both: two works under one signature.
            raise ValueError("a submission carries completions or one trajectory, exactly one")
        return self


class CorpusSubmissionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: CorpusRejectReason
    accepted: bool
    slots_remaining: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class CorpusSkipRequest(BaseModel):
    """A signed request to step this hotkey's walk from ``cursor`` to
    ``to_cursor``, granted only when every prompt in between has no slot left. It carries no
    work, so nothing is paid and nothing is recorded; its signature is bound
    under its own domain, so it can never stand in for a submission's."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1)
    miner_hotkey: str = Field(min_length=1)
    cursor: int = Field(ge=0)
    prompt_index: int = Field(ge=0)
    # Where the cursor lands: every walk position in [cursor, to_cursor) must
    # be full, and the validator bounds the length.
    to_cursor: int = Field(ge=1)
    signature: str = Field(min_length=1)


class CorpusSkipResponse(BaseModel):
    """``skipped`` with ``reason`` ``accepted`` when the cursor moved;
    ``cursor`` is then where this hotkey now stands."""

    model_config = ConfigDict(extra="forbid")

    reason: CorpusRejectReason
    skipped: bool
    cursor: int | None = None
    slots_remaining: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
