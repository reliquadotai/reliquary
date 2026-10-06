"""Wire models of the sandbox session routes (plan 3). A miner asks the validator for
a session token naming its engagement, and reports how a session ended when it will
not submit it. Pure: pydantic only, so a miner without reliquary-sandbox can import it."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reliquary.protocol.corpus_submission import MAX_TRANSCRIPT_BYTES


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
    prompt_index: int | None = Field(default=None, ge=0)
    precommit: dict[str, Any] | None = None


class SandboxSessionOpenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    miner_hotkey: str = Field(min_length=1, max_length=64)
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    at: int = Field(ge=0)
    engagement: SandboxEngagement
    signature: str = Field(min_length=1, max_length=256)


class SandboxSessionCloseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    miner_hotkey: str = Field(min_length=1, max_length=64)
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    at: int = Field(ge=0)
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    reason: Literal["final", "open_failed"]
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
           "SessionRefused"]
