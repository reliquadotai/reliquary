"""Explicit Episode v1 suite catalog and validator replay entry point."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from reliquary.environment.agentic.base import EpisodeEnvironment
from reliquary.environment.agentic.replay import replay_wire_actions
from reliquary.environment.agentic.types import EpisodeTrace


BUILTIN_EPISODE_ENVIRONMENTS = (
    "reliquary_stateful_tools_v1",
    "reliquary_retrieval_tools_v1",
    "reliquary_workspace_tools_v1",
)


def episode_score_many_not_supported(
    problem: dict[str, Any],
    completion_texts: list[str],
    reward_materials: Any = None,
) -> list[float]:
    del problem, completion_texts, reward_materials
    raise TypeError("episode environments must be scored by deterministic replay")


def replay_submission(
    environment: EpisodeEnvironment,
    *,
    task_index: int,
    seed: int,
    actions: Iterable[Mapping[str, Any]],
) -> EpisodeTrace:
    """Top-level, process-picklable replay function used by validator workers."""

    return replay_wire_actions(
        environment,
        task_index=task_index,
        seed=seed,
        actions=[dict(action) for action in actions],
    )
