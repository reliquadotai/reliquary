"""The minimal protocol implemented by all Reliquary episode environments."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from reliquary.environment.agentic.types import (
    AssistantAction,
    EpisodeTask,
    EpisodeTrace,
    ResetResult,
    RewardReport,
    StepResult,
)


@runtime_checkable
class EpisodeEnvironment(Protocol):
    """A deterministic, replayable, single-policy multi-turn environment."""

    name: str
    validator_authoritative_reward: bool
    max_turns: int

    def __len__(self) -> int: ...

    def get_task(self, index: int) -> EpisodeTask: ...

    def reset(self, task: EpisodeTask, seed: int) -> ResetResult: ...

    def step(
        self,
        task: EpisodeTask,
        state: Any,
        action: AssistantAction,
    ) -> StepResult: ...

    def grade(
        self,
        task: EpisodeTask,
        state: Any,
        trace: EpisodeTrace,
    ) -> RewardReport: ...

    def close(self, state: Any) -> None: ...
