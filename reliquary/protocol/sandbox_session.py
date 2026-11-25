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

# A miner's bound on one signed episode (``signed_episode_deadline``): its open window, the
# agent's wall budget, grading inside the verification window, submit. The order control's
# leases on such a job last at least this long (``corpus.draw_leases``).
OPEN_WINDOW_S = 900
GRADING_GRACE_S = 1800
SUBMIT_MARGIN_S = 120


def signed_episode_deadline(wall_s: float) -> float:
    """Seconds a miner gives one signed episode whose sandbox wall budget is ``wall_s``."""
    return float(OPEN_WINDOW_S + wall_s + GRADING_GRACE_S + SUBMIT_MARGIN_S)


# A corpus open's declared want (``SandboxEngagement.want``): the live sessions this hotkey wants on the job at once
# (its batch), at most this. The validator sizes the sandbox fleet from it (design 2026-11-13-sandbox-sizing-miner-
# demand); it is never an entitlement: the quota's allowance still bounds what the hotkey holds. A validator that
# reads it says so on its grants and quota refusals (``WANT_ACCEPTED``); a miner sends it only after that, since
# an older validator refuses the unknown field.
MAX_SESSION_WANT = 512
WANT_ACCEPTED = "declares_want"


# The session routes of a job served by the order control: under the job's own
# path, which the order jobs' nginx location already sends there.
JOB_SCOPED_PREFIX = "/corpus/jobs/{job_id}"


def job_scoped_prefix(job_id: str) -> str:
    return JOB_SCOPED_PREFIX.format(job_id=job_id)


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
    # A corpus open's declared want (MAX_SESSION_WANT): signed with the engagement, absent from an older miner's.
    want: int | None = Field(default=None, ge=1, le=MAX_SESSION_WANT, strict=True)


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


__all__ = ["JOB_SCOPED_PREFIX", "MAX_SESSION_WANT", "WANT_ACCEPTED", "SandboxEngagement", "SandboxSessionCloseRequest",
           "SandboxSessionOpenRequest", "SessionRefused", "job_scoped_prefix", "sandbox_close_path",
           "sandbox_open_path"]
