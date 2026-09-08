"""Stable Episode v1 rows used by fixtures and cross-host qualification."""

from __future__ import annotations

from typing import Any

from reliquary.environment.agentic.renderer import CanonicalEpisodeRenderer
from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy
from reliquary.environment.agentic.types import AssistantAction, sha256_json
from reliquary.environment.registry import get_environment_spec


GOLDEN_SCHEMA = "reliquary/episode-golden/v1"


def _byte_encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def episode_golden_row(environment: str, index: int) -> dict[str, Any]:
    """Return tokenizer-neutral consensus evidence for one fixed task index."""

    spec = get_environment_spec(environment)
    env = spec.create()
    task = env.get_task(index)
    renderer = CanonicalEpisodeRenderer(_byte_encode)
    reference = EpisodeRunner(renderer=renderer).run(
        env,
        task,
        seed=2026,
        policy=ScriptedPolicy(task.private["reference_actions"]),
    )
    rejected = EpisodeRunner(renderer=renderer).run(
        spec.create(),
        task,
        seed=2026,
        policy=ScriptedPolicy([AssistantAction.final("incorrect")]),
    )
    if reference.reward is None or rejected.reward is None:
        raise RuntimeError("golden episode did not produce a reward report")
    return {
        "schema": GOLDEN_SCHEMA,
        "environment": environment,
        "index": int(index),
        "task_id": task.id,
        "public_task_sha256": sha256_json(task.to_public_wire()),
        "reference_actions_sha256": sha256_json([
            action.to_wire() for action in task.private["reference_actions"]
        ]),
        "reference_tokens_sha256": sha256_json(list(reference.tokens)),
        "reference_trace_digest": reference.trace_digest,
        "reference_state_digest": reference.reward.state_digest,
        "reference_reward": reference.reward.reward,
        "rejected_tokens_sha256": sha256_json(list(rejected.tokens)),
        "rejected_trace_digest": rejected.trace_digest,
        "rejected_state_digest": rejected.reward.state_digest,
        "rejected_reward": rejected.reward.reward,
    }


__all__ = ["GOLDEN_SCHEMA", "episode_golden_row"]
