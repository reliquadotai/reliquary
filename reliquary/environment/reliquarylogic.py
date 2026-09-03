"""Reliquary-authored deterministic logic environment."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
from typing import Any, ClassVar

from reliquary.environment.logic_tasks import (
    CHECKER_VERSION,
    GENERATOR_VERSION,
    TASK_FAMILY,
    VIRTUAL_LENGTH,
    GeneratedLogicTask,
    check_answer,
    generate_logic_task,
    verifier_spec,
)
from reliquary.environment.structured_output import extract_json_answer


# The batcher calls ``get_problem`` several times per submission with the
# same index; generation is pure, so cache the task and rebuild the dict.
@lru_cache(maxsize=4096)
def _task_for_index(index: int) -> GeneratedLogicTask:
    return generate_logic_task(index)


ENVIRONMENT_NAME = "reliquarylogic_v1"


def _render_prompt(puzzle: str) -> str:
    """Wrap the generated puzzle in the active profile's prompt template.

    A profile that does not declare this environment falls back to the bare
    puzzle rather than raising: every live Math+Code profile omits logic, and
    offline scoring runs under whichever profile happens to be active. The
    import is local because the profile module is read at call time, so a
    test can swap the active profile.
    """
    from reliquary.protocol import profiles

    if ENVIRONMENT_NAME not in profiles.ACTIVE_PROTOCOL_PROFILE.environments:
        return puzzle
    rendered = profiles.render_active_prompt(ENVIRONMENT_NAME, problem=puzzle)
    return puzzle if rendered is None else rendered


def problem_from_task(
    task: GeneratedLogicTask, generator_index: int
) -> dict[str, Any]:
    """Problem dict for a task.

    Split out so a family that the roster does not currently draw can still
    be scored offline, without activating it and remapping the index space.
    """
    return {
        # Identity is the puzzle, never the envelope: prompt_content_sha256
        # burns an index for the content cooldown, so hashing the rendered
        # prompt would resurrect every consumed index on a prompt change.
        "id": sha256(task.prompt.encode("utf-8")).hexdigest()[:16],
        "prompt": _render_prompt(task.prompt),
        "ground_truth": verifier_spec(task),
        "task_family": TASK_FAMILY,
        "family": task.family,
        "difficulty": task.difficulty,
        "operation_id": task.operation_id,
        "generator_version": GENERATOR_VERSION,
        "generator_index": generator_index,
    }


class ReliquaryLogicEnvironment:
    """Procedural logic puzzles with an exact JSON reward."""

    name: ClassVar[str] = "reliquarylogic_v1"
    validator_authoritative_reward: ClassVar[bool] = True
    generator_version: ClassVar[str] = GENERATOR_VERSION
    checker_version: ClassVar[str] = CHECKER_VERSION
    task_family: ClassVar[str] = TASK_FAMILY

    def __len__(self) -> int:
        return VIRTUAL_LENGTH

    def get_problem(self, index: int) -> dict[str, Any]:
        normalized = int(index) % VIRTUAL_LENGTH
        return problem_from_task(_task_for_index(normalized), normalized)

    def compute_reward(self, problem: dict, completion: str) -> float:
        try:
            raw_spec = problem.get("ground_truth")
            if not isinstance(raw_spec, str):
                return 0.0
            spec = json.loads(raw_spec)
            if not isinstance(spec, dict):
                return 0.0
            if spec.get("schema") != "reliquary/logic-verifier/v1":
                return 0.0
            if spec.get("checker_version") != CHECKER_VERSION:
                return 0.0
            answer = extract_json_answer(completion)
            if set(answer) != {"result"}:
                return 0.0
            return float(check_answer(spec, answer["result"]))
        except Exception:
            return 0.0

    def source_health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "environment": self.name,
            "task_family": TASK_FAMILY,
            "generator_version": GENERATOR_VERSION,
            "checker_version": CHECKER_VERSION,
            "virtual_length": VIRTUAL_LENGTH,
            "external_dependencies": [],
        }


def score_reliquarylogic(
    problem: dict[str, Any],
    completion_texts: list[str],
    reward_materials: Any = None,
) -> list[float]:
    del reward_materials
    environment = ReliquaryLogicEnvironment()
    return [
        environment.compute_reward(problem, text) for text in completion_texts
    ]


__all__ = [
    "ReliquaryLogicEnvironment",
    "problem_from_task",
    "score_reliquarylogic",
]
