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

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reliquary.protocol.corpus_submission import MAX_TRANSCRIPT_BYTES


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

    kind: Literal["corpus", "rl_precommit"]
    job_id: str | None = Field(default=None, min_length=1, max_length=64)
    prompt_index: int | None = Field(default=None, ge=0, strict=True)
    precommit: dict[str, Any] | None = None


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
    signature: str = Field(min_length=1, max_length=256)

    @field_validator("transcript")
    @classmethod
    def _bounded(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None and len(json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                                       .encode("utf-8")) > MAX_TRANSCRIPT_BYTES:
            raise ValueError(f"the transcript is over {MAX_TRANSCRIPT_BYTES} bytes")
        return value


__all__ = ["SandboxEngagement", "SandboxSessionCloseRequest", "SandboxSessionOpenRequest",
           "SessionRefused", "sandbox_close_path", "sandbox_open_path"]
