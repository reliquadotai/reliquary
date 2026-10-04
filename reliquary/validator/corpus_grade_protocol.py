"""Versioned JSON boundary between the corpus control and a grade executor
(spec §5 N5). A lease carries one trajectory's grade or replay; a result
carries the facts the executor observed. The decisions (certified, failed,
confirmed) stay on the control.

The actions a replay lease carries are the ones ``trajectory_parse`` read from
the trajectory's proven tokens, never a miner-supplied trace. Any call is
replayed as the harness would answer it (ruling P15): an unknown tool gets the
harness's "error: unknown tool" text, extra argument keys are ignored and
missing ones defaulted, exactly as ``agentic_replay.TOOL_PROGRAM`` does."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reliquary.protocol.corpus_submission import MAX_FINAL_DIFF_CHARS
from reliquary.validator.corpus_audit_protocol import ExecutorId, LeaseId

GRADE_PROTOCOL = "reliquary.corpus-grade/v1"
MAX_ACTIONS = 4096
# Size bounds only: any name is replayed (an unknown one as the harness's error).
MAX_TOOL_NAME_CHARS = 1024
MAX_ARGUMENT_CHARS = 1_048_576
# Also the SWE env's per-observation bound at intake (spec N3 check 6): the
# pinned bash harness truncates nothing, so this lease bound is the limit.
MAX_OBSERVATION_CHARS = 4_194_304
# One item's text in all (diff, names, arguments, observations): a lease stays
# a few megabytes. An honest 60k-token trajectory is well under 1 MiB of text.
MAX_GRADE_ITEM_CHARS = 16 * 2 ** 20


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GradeClaimRequest(_Strict):
    executor_id: ExecutorId
    env_package: str = Field(min_length=1, max_length=64)
    env_version: str = Field(pattern=r"^[0-9a-f]{40}$")


class GradeAction(_Strict):
    tool: str = Field(max_length=MAX_TOOL_NAME_CHARS)
    arguments: str = Field(max_length=MAX_ARGUMENT_CHARS)
    # None: replayed and not compared (the final turn of a context_length or max_turns stop).
    observation: str | None = Field(default=None, max_length=MAX_OBSERVATION_CHARS)


class GradeItem(_Strict):
    submission_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_index: int = Field(ge=0)
    instance_id: str = Field(min_length=1, max_length=256)
    mode: Literal["grade", "replay"]
    final_diff: str = Field(max_length=MAX_FINAL_DIFF_CHARS)
    actions: list[GradeAction] = Field(default_factory=list, max_length=MAX_ACTIONS)

    @model_validator(mode="after")
    def _bounded_in_all(self) -> "GradeItem":
        total = _item_chars(self.actions, self.final_diff)
        if total > MAX_GRADE_ITEM_CHARS:
            raise ValueError(f"the item holds {total} chars, more than {MAX_GRADE_ITEM_CHARS}")
        return self


def _item_chars(actions, final_diff: str) -> int:
    return len(final_diff) + sum(len(a.tool) + len(a.arguments) + len(a.observation or "")
                                 for a in actions)


def grade_item_bounds_refusal(actions: Sequence, final_diff: str) -> dict | None:
    """Why no grade lease could carry a trajectory's parsed actions (each with
    ``tool``, ``arguments``, ``observation``) and diff, or None. The intake and
    the miner's precheck both apply it (ruling P23 d): such a trajectory is
    refused before it takes a slot, never accepted and left ungradeable. Reads
    the bounds at call time."""
    if len(actions) > MAX_ACTIONS:
        return {"why": "actions", "count": len(actions), "max": MAX_ACTIONS}
    for index, action in enumerate(actions):
        for why, text, bound in (("tool_name", action.tool, MAX_TOOL_NAME_CHARS),
                                 ("arguments", action.arguments, MAX_ARGUMENT_CHARS),
                                 ("observation", action.observation or "", MAX_OBSERVATION_CHARS)):
            if len(text) > bound:
                return {"why": why, "action": index, "chars": len(text), "max": bound}
    if len(final_diff) > MAX_FINAL_DIFF_CHARS:
        return {"why": "final_diff", "chars": len(final_diff), "max": MAX_FINAL_DIFF_CHARS}
    total = _item_chars(actions, final_diff)
    if total > MAX_GRADE_ITEM_CHARS:
        return {"why": "total", "chars": total, "max": MAX_GRADE_ITEM_CHARS}
    return None


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
    # "error"/"timeout": the executor's own failure (re-leased elsewhere).
    # "box_lost"/"box_timeout": the trajectory's (ruling P23), a vote like "ok".
    status: Literal["ok", "error", "timeout", "box_lost", "box_timeout"]
    # Echoed from the item, so the control can check the facts are for the
    # trajectory it leased (defence in depth beside the lease id).
    submission_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    diff_applied: bool | None = None
    tests_passed: bool | None = None
    replay_diff_equal: bool | None = None
    observations_compared: int = Field(default=0, ge=0)
    observations_mismatched: list[int] = Field(default_factory=list, max_length=MAX_ACTIONS)
    detail: str | None = Field(default=None, max_length=512)


class GradeResult(_Strict):
    results: list[GradeItemResult] = Field(min_length=1, max_length=1)


__all__ = ["GRADE_PROTOCOL", "MAX_GRADE_ITEM_CHARS", "MAX_TOOL_NAME_CHARS", "grade_item_bounds_refusal", "GradeAction", "GradeClaimRequest", "GradeEnv", "GradeItem",
           "GradeItemResult", "GradeLease", "GradeResult"]
