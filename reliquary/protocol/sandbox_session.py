"""Wire models of the sandbox session routes (plan 3). A miner asks the validator for
a session token naming its engagement, and reports how a session ended when it will
not submit it. Pure: pydantic only, so a miner without reliquary-sandbox can import it.

A signed request names its audience: the validator's hotkey and the HTTP path it is
posted to are part of its binding (`signatures.build_sandbox_open_binding`), so a
request signed for one validator, or for the close of one session, never verifies at
another. A signed open body is still a bearer credential for its freshness window
(±120 s): a client must never log it, and the endpoint is only reached over TLS or the
validator's tunnel."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reliquary.protocol.corpus_submission import MAX_TRANSCRIPT_BYTES

MAX_OPEN_REFUSAL_BYTES = 4096


def sandbox_open_path(prefix: str = "/corpus") -> str:
    return f"{prefix}/sandbox/sessions"


def sandbox_close_path(prefix: str, session_id: str) -> str:
    return f"{prefix}/sandbox/sessions/{session_id}/close"


class SessionRefused(RuntimeError):
    """The validator refused to open or close a session (raised on the miner side)."""

    def __init__(self, reason: str, detail: dict | None = None, *,
                 retry_after: float | None = None, status: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = dict(detail or {})
        self.retry_after = retry_after
        self.status = status


class SandboxEngagement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["corpus", "rl_precommit", "rl_lease"]
    job_id: str | None = Field(default=None, min_length=1, max_length=64)
    prompt_index: int | None = Field(default=None, ge=0, strict=True)
    precommit: dict[str, Any] | None = None
    lease: dict[str, Any] | None = None      # an async RL lease's rollout: {lease_id, rollout_index} (rl_engagements)


class SandboxSessionOpenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    miner_hotkey: str = Field(min_length=1, max_length=64)
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    at: int = Field(ge=0, strict=True)
    engagement: SandboxEngagement
    signature: str = Field(min_length=1, max_length=256)


class SandboxSessionCloseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    miner_hotkey: str = Field(min_length=1, max_length=64)
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    at: int = Field(ge=0, strict=True)
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    # final: the machine's final record (with its transcript); open_failed: no box, no
    # transcript; withdraw: a verified transcript the miner will not submit, releasing
    # its slot for good (the session is never paid afterwards).
    reason: Literal["final", "open_failed", "withdraw"]
    transcript: dict[str, Any] | None = None
    # The gateway's signed refusal of this session's open (its X-Reliquary-Open-Refusal header), only with
    # reason open_failed. The machine signed it, bound to the token: the miner's binding need not cover it,
    # and a validator that cannot verify it counts the close as the miner's word.
    open_refusal: dict[str, Any] | None = None
    signature: str = Field(min_length=1, max_length=256)

    @field_validator("transcript")
    @classmethod
    def _bounded(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None and len(json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                                       .encode("utf-8")) > MAX_TRANSCRIPT_BYTES:
            raise ValueError(f"the transcript is over {MAX_TRANSCRIPT_BYTES} bytes")
        return value

    @model_validator(mode="after")
    def _refusal_with_a_failed_open(self):
        if self.open_refusal is not None:
            if self.reason != "open_failed":
                raise ValueError("an open refusal goes with reason open_failed only")
            if len(json.dumps(self.open_refusal, separators=(",", ":")).encode("utf-8")) > MAX_OPEN_REFUSAL_BYTES:
                raise ValueError(f"the open refusal is over {MAX_OPEN_REFUSAL_BYTES} bytes")
        return self


__all__ = ["SandboxEngagement", "SandboxSessionCloseRequest", "SandboxSessionOpenRequest",
           "SessionRefused", "sandbox_close_path", "sandbox_open_path"]
