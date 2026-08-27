"""Compatibility helpers for Reliquary's existing prompt control plane."""

from __future__ import annotations

from typing import Any

from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer


def episode_problem(environment: Any, index: int) -> dict[str, Any]:
    task = environment.get_task(index)
    value: dict[str, Any] = {
        "prompt": task.prompt,
        "ground_truth": "",
        "task_id": task.id,
        "task_family": task.metadata.get("task_family", environment.name),
        "generator_version": task.metadata.get("generator_version", "v1"),
    }
    for key in ("operation_id", "difficulty"):
        if key in task.metadata:
            value[key] = task.metadata[key]
    return value


def rendered_episode_prompt(environment: Any, index: int) -> str:
    return CanonicalEpisodeRenderer.initial_text(environment.get_task(index))


def encode_episode_prompt(tokenizer: Any, environment: Any, index: int) -> list[int]:
    encoded = tokenizer.encode(
        rendered_episode_prompt(environment, index),
        add_special_tokens=False,
    )
    return list(getattr(encoded, "ids", encoded))
