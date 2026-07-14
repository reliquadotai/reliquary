"""Strict data contracts for the eval-dashboard producer."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SHA256_PATTERN = r"^[0-9a-f]{64}$"
REVISION_PATTERN = r"^[0-9a-f]{40,64}$"


def _timestamps_match(timestamp: float, iso: str) -> bool:
    parsed = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return parsed.tzinfo is not None and abs(parsed.timestamp() - timestamp) < 0.001


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ModelRevision(FrozenModel):
    repo_id: str = Field(min_length=3)
    revision: str = Field(pattern=REVISION_PATTERN)
    checkpoint_n: int = Field(ge=0)


class LineageSpec(FrozenModel):
    lineage_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    base: ModelRevision
    checkpoint_repo_id: str = Field(min_length=3)


class TokenizerSource(FrozenModel):
    repo_id: str = Field(min_length=3)
    revision: str = Field(pattern=REVISION_PATTERN)


class HoldoutSpec(FrozenModel):
    domain: Literal["math", "code"]
    dataset_repo_id: str = Field(min_length=3)
    dataset_revision: str = Field(pattern=REVISION_PATTERN)
    split: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    task_ids_sha256: str = Field(pattern=SHA256_PATTERN)
    contamination_review_sha256: str = Field(pattern=SHA256_PATTERN)
    n_prompts: int = Field(gt=0)
    grader_id: str = Field(min_length=3)
    grader_revision: str = Field(pattern=REVISION_PATTERN)
    format_version: Literal["1"] = "1"


class GenerationSpec(FrozenModel):
    protocol_parity: bool = True
    samples_per_prompt: int = Field(ge=1, le=16)
    temperature: float = Field(gt=0.0, le=5.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(ge=0)
    presence_penalty: float = 0.0
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    batch_size: Literal[1] = 1
    seed_salt: str = Field(min_length=16)
    math_max_new_tokens: int = Field(gt=0)
    math_bft_enabled: bool
    math_thinking_budget: int = Field(gt=0)
    math_answer_budget: int = Field(gt=0)
    math_force_template: str = Field(min_length=1)
    code_max_new_tokens: int = Field(gt=0)

    @field_validator("presence_penalty")
    @classmethod
    def _finite_presence_penalty(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("presence_penalty must be finite")
        return value

    @model_validator(mode="after")
    def _supported_generation_contract(self):
        if self.presence_penalty != 0.0:
            raise ValueError(
                "the canonical HF evaluator does not implement presence_penalty; "
                "set it explicitly to 0.0"
            )
        if (
            self.math_thinking_budget + self.math_answer_budget
            > self.math_max_new_tokens
        ):
            raise ValueError(
                "math thinking + answer budgets exceed math_max_new_tokens"
            )
        return self


class ScheduleSpec(FrozenModel):
    owner: str = Field(min_length=3)
    cadence_seconds: int = Field(ge=300)
    overdue_seconds: int = Field(ge=900)
    retry_attempts: int = Field(default=4, ge=1, le=20)
    retry_base_seconds: int = Field(default=30, ge=1, le=3600)

    @model_validator(mode="after")
    def _overdue_exceeds_cadence(self):
        if self.overdue_seconds <= self.cadence_seconds:
            raise ValueError("overdue_seconds must exceed cadence_seconds")
        return self


class EvalConfig(FrozenModel):
    schema_version: Literal["1"] = "1"
    lineage: LineageSpec
    tokenizer_source: TokenizerSource
    math_holdout: HoldoutSpec
    code_holdout: HoldoutSpec
    generation: GenerationSpec
    publish_interval_windows: int = Field(gt=0)
    schedule: ScheduleSpec

    @model_validator(mode="after")
    def _domain_contract(self):
        if self.math_holdout.domain != "math":
            raise ValueError("math_holdout.domain must be 'math'")
        if self.code_holdout.domain != "code":
            raise ValueError("code_holdout.domain must be 'code'")
        return self


class TrainingArtifact(FrozenModel):
    sha256: str = Field(pattern=SHA256_PATTERN)
    n_prompts: int = Field(gt=0)


class TrainingSource(FrozenModel):
    repo_id: str = Field(min_length=3)
    revision: str = Field(pattern=REVISION_PATTERN)
    prompt_field: str = Field(min_length=1)
    artifacts: list[TrainingArtifact] = Field(min_length=1)


class ContaminationReview(FrozenModel):
    schema_version: Literal["1"] = "1"
    domain: Literal["math", "code"]
    holdout_sha256: str = Field(pattern=SHA256_PATTERN)
    task_ids_sha256: str = Field(pattern=SHA256_PATTERN)
    reviewed_at: str = Field(min_length=20)
    reviewer: str = Field(min_length=3)
    method: str = Field(min_length=8)
    training_sources: list[TrainingSource] = Field(min_length=1)
    exact_overlap_count: int = Field(ge=0)
    near_duplicate_count: int = Field(ge=0)
    decision: Literal["approved", "rejected"]
    notes: str = ""

    @field_validator("reviewed_at")
    @classmethod
    def _reviewed_at_is_timezone_aware(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("reviewed_at must include a timezone")
        return value


class MathTask(FrozenModel):
    task_id: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1)
    ground_truth: str = Field(min_length=1)


class CodeTask(FrozenModel):
    task_id: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1)
    cases: list[dict[str, Any]] = Field(min_length=1)


class SampleResult(FrozenModel):
    sample_index: int = Field(ge=0)
    seed: int = Field(ge=0)
    reward: float = Field(ge=0.0, le=1.0)
    completion_length: int = Field(ge=0)
    terminated: bool
    forced: bool
    completion_sha256: str = Field(pattern=SHA256_PATTERN)
    duration_seconds: float = Field(ge=0.0)


class TaskResult(FrozenModel):
    task_id: str = Field(min_length=1, max_length=256)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    samples: list[SampleResult] = Field(min_length=1)

    @model_validator(mode="after")
    def _sample_indices_are_canonical(self):
        expected = list(range(len(self.samples)))
        if [sample.sample_index for sample in self.samples] != expected:
            raise ValueError("sample indices must be ordered and contiguous from zero")
        return self


class DomainResult(FrozenModel):
    domain: Literal["math", "code"]
    n_prompts: int = Field(gt=0)
    samples_per_prompt: int = Field(gt=0)
    pass_at_1: float = Field(ge=0.0, le=1.0)
    pass_at_k: float = Field(ge=0.0, le=1.0)
    pass_avg: float = Field(ge=0.0, le=1.0)
    trunc_pct: float = Field(ge=0.0, le=100.0)
    forced_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    pass1_ci95: tuple[float, float]
    tasks: list[TaskResult]

    @model_validator(mode="after")
    def _result_counts_match(self):
        if len(self.tasks) != self.n_prompts:
            raise ValueError("n_prompts does not match task results")
        if any(len(task.samples) != self.samples_per_prompt for task in self.tasks):
            raise ValueError("samples_per_prompt does not match task results")
        task_ids = [task.task_id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("domain result contains duplicate task ids")
        low, high = self.pass1_ci95
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("pass1_ci95 must be ordered and bounded")
        return self


class CheckpointResult(FrozenModel):
    schema_version: Literal["1"] = "1"
    config_hash: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    lineage_id: str
    model_repo_id: str
    model_revision: str = Field(pattern=REVISION_PATTERN)
    checkpoint_n: int = Field(ge=0)
    observed_window: int = Field(ge=0)
    started_at: float = Field(ge=0.0)
    completed_at: float = Field(ge=0.0)
    completed_at_iso: str
    duration_seconds: float = Field(ge=0.0)
    hardware: dict[str, Any]
    runtime: dict[str, Any]
    domains: dict[Literal["math", "code"], DomainResult]

    @model_validator(mode="after")
    def _checkpoint_contract(self):
        if set(self.domains) != {"math", "code"}:
            raise ValueError("checkpoint results must contain math and code")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at precedes started_at")
        if not _timestamps_match(self.completed_at, self.completed_at_iso):
            raise ValueError("completed_at and completed_at_iso disagree")
        return self


class ArtifactReference(FrozenModel):
    key: str = Field(min_length=3)
    sha256: str = Field(pattern=SHA256_PATTERN)


class DomainCoverage(FrozenModel):
    n_prompts: int = Field(gt=0)
    samples_per_prompt: int = Field(gt=0)
    pass_at_1: float = Field(ge=0.0, le=1.0)
    pass_at_k: float = Field(ge=0.0, le=1.0)
    pass_avg: float = Field(ge=0.0, le=1.0)
    trunc_pct: float = Field(ge=0.0, le=100.0)
    forced_pct: float | None = Field(default=None, ge=0.0, le=100.0)


class PublishedCheckpointArtifact(FrozenModel):
    result: ArtifactReference
    checkpoint_n: int = Field(ge=0)
    model_repo_id: str = Field(min_length=3)
    model_revision: str = Field(pattern=REVISION_PATTERN)
    observed_window: int = Field(ge=0)
    started_at: float = Field(ge=0.0)
    completed_at: float = Field(ge=0.0)
    completed_at_iso: str = Field(min_length=20)
    duration_seconds: float = Field(ge=0.0)
    hardware: dict[str, Any]
    runtime: dict[str, Any]
    domains: dict[Literal["math", "code"], DomainCoverage]

    @model_validator(mode="after")
    def _coverage_is_complete(self):
        if set(self.domains) != {"math", "code"}:
            raise ValueError("published checkpoint coverage must contain math and code")
        if not _timestamps_match(self.completed_at, self.completed_at_iso):
            raise ValueError("published checkpoint timestamps disagree")
        return self


class EvalConfigManifest(FrozenModel):
    schema_version: Literal["2"] = "2"
    kind: Literal["eval_config"] = "eval_config"
    config_hash: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    lineage_id: str = Field(min_length=1)
    effective_config: dict[str, Any]


class EvalPublicationManifest(FrozenModel):
    schema_version: Literal["2"] = "2"
    kind: Literal["eval_publication"] = "eval_publication"
    publication_id: str = Field(pattern=SHA256_PATTERN)
    config_hash: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    lineage_id: str = Field(min_length=1)
    generated_at: float = Field(ge=0.0)
    generated_at_iso: str = Field(min_length=20)
    evidence_completed_at: float = Field(ge=0.0)
    evidence_completed_at_iso: str = Field(min_length=20)
    base_checkpoint_n: int = Field(ge=0)
    latest_checkpoint_n: int = Field(ge=0)
    latest_checkpoint_revision: str = Field(pattern=REVISION_PATTERN)
    latest_model_repo_id: str = Field(min_length=3)
    config_manifest: ArtifactReference
    dashboard: ArtifactReference
    checkpoints: list[PublishedCheckpointArtifact] = Field(min_length=1)

    @model_validator(mode="after")
    def _publication_is_consistent(self):
        if not _timestamps_match(self.generated_at, self.generated_at_iso):
            raise ValueError("publication timestamps disagree")
        if not _timestamps_match(
            self.evidence_completed_at, self.evidence_completed_at_iso
        ):
            raise ValueError("publication evidence timestamps disagree")
        if self.publication_id != self.dashboard.sha256:
            raise ValueError("publication_id must equal the dashboard digest")
        checkpoint_numbers = [item.checkpoint_n for item in self.checkpoints]
        if len(set(checkpoint_numbers)) != len(checkpoint_numbers):
            raise ValueError("publication contains duplicate checkpoint numbers")
        latest = max(self.checkpoints, key=lambda item: item.checkpoint_n)
        if (
            latest.checkpoint_n != self.latest_checkpoint_n
            or latest.model_revision != self.latest_checkpoint_revision
            or latest.model_repo_id != self.latest_model_repo_id
            or latest.completed_at != self.evidence_completed_at
            or latest.completed_at_iso != self.evidence_completed_at_iso
        ):
            raise ValueError("publication latest-checkpoint metadata disagrees")
        return self
