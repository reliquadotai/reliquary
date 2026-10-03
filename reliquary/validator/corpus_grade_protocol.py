"""Versioned JSON boundary between the corpus control and a grade executor
(spec §5 N5). A lease carries one trajectory's grade or replay; a result
carries the facts the executor observed. The decisions (certified, failed,
confirmed) stay on the control.

The actions a replay lease carries are the ones ``trajectory_parse`` read from
the trajectory's proven tokens, never a miner-supplied trace; an action that
is not exactly one of the harness's calls cannot be put in a lease at all
(``harness_call_refusal``)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reliquary.corpus.replay_compare import harness_call_refusal
from reliquary.protocol.corpus_submission import MAX_FINAL_DIFF_CHARS
from reliquary.validator.corpus_audit_protocol import ExecutorId, LeaseId

GRADE_PROTOCOL = "reliquary.corpus-grade/v1"
MAX_ACTIONS = 4096
MAX_ARGUMENT_CHARS = 1_048_576
MAX_OBSERVATION_CHARS = 4_194_304


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GradeClaimRequest(_Strict):
    executor_id: ExecutorId
    env_package: str = Field(min_length=1, max_length=64)
    env_version: str = Field(pattern=r"^[0-9a-f]{40}$")


class GradeAction(_Strict):
    tool: str = Field(min_length=1, max_length=64)
    arguments: str = Field(max_length=MAX_ARGUMENT_CHARS)
    # None: replayed and not compared (a final turn cut by context_length).
    observation: str | None = Field(default=None, max_length=MAX_OBSERVATION_CHARS)

    @model_validator(mode="after")
    def _a_harness_call(self) -> GradeAction:
        refusal = harness_call_refusal(self.tool, self.arguments)
        if refusal:
            raise ValueError(f"not a harness call: {refusal}")
        return self


class GradeItem(_Strict):
    submission_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_index: int = Field(ge=0)
    instance_id: str = Field(min_length=1, max_length=256)
    mode: Literal["grade", "replay"]
    final_diff: str = Field(max_length=MAX_FINAL_DIFF_CHARS)
    actions: list[GradeAction] = Field(default_factory=list, max_length=MAX_ACTIONS)


class GradeEnv(_Strict):
    package: str = Field(min_length=1, max_length=64)
    version: str = Field(pattern=r"^[0-9a-f]{40}$")


class GradeLease(_Strict):
    protocol: Literal["reliquary.corpus-grade/v1"] = GRADE_PROTOCOL
    lease_id: LeaseId
    expires_at: float
    env: GradeEnv
    items: list[GradeItem] = Field(min_length=1, max_length=1)


class GradeItemResult(_Strict):
    status: Literal["ok", "error", "timeout"]
    diff_applied: bool | None = None
    tests_passed: bool | None = None
    replay_diff_equal: bool | None = None
    observations_compared: int = Field(default=0, ge=0)
    observations_mismatched: list[int] = Field(default_factory=list, max_length=MAX_ACTIONS)
    detail: str | None = Field(default=None, max_length=512)


class GradeResult(_Strict):
    results: list[GradeItemResult] = Field(min_length=1, max_length=1)


__all__ = ["GRADE_PROTOCOL", "GradeAction", "GradeClaimRequest", "GradeEnv", "GradeItem",
           "GradeItemResult", "GradeLease", "GradeResult"]
