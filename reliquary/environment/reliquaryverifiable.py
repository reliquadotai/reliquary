"""Reliquary-authored deterministic verifiable Records environment."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, ClassVar

from reliquary.environment.records_tasks import (
    CHECKER_VERSION,
    GENERATOR_VERSION,
    TASK_FAMILY,
    VIRTUAL_LENGTH,
    generate_records_task,
    verifier_spec,
)
from reliquary.environment.structured_output import extract_json_answer


class ReliquaryVerifiableEnvironment:
    """Generated structured-record transformations with exact JSON reward."""

    name: ClassVar[str] = "reliquaryverifiable_v1"
    validator_authoritative_reward: ClassVar[bool] = True
    generator_version: ClassVar[str] = GENERATOR_VERSION
    checker_version: ClassVar[str] = CHECKER_VERSION
    task_family: ClassVar[str] = TASK_FAMILY

    def __len__(self) -> int:
        return VIRTUAL_LENGTH

    def get_problem(self, index: int) -> dict[str, Any]:
        normalized = int(index) % VIRTUAL_LENGTH
        task = generate_records_task(normalized)
        prompt = task.prompt
        return {
            "id": sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "prompt": prompt,
            "ground_truth": verifier_spec(task),
            "task_family": TASK_FAMILY,
            "difficulty": task.difficulty,
            "operation_id": task.operation_id,
            "generator_version": GENERATOR_VERSION,
            "generator_index": normalized,
        }

    def compute_reward(self, problem: dict, completion: str) -> float:
        try:
            raw_spec = problem.get("ground_truth")
            if not isinstance(raw_spec, str):
                return 0.0
            spec = json.loads(raw_spec)
            if not isinstance(spec, dict):
                return 0.0
            if spec.get("schema") != "reliquary/records-verifier/v1":
                return 0.0
            if spec.get("checker_version") != CHECKER_VERSION:
                return 0.0
            answer = extract_json_answer(completion)
            if set(answer) != {"result"}:
                return 0.0
            return float(answer["result"] == spec.get("expected"))
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


def score_reliquaryverifiable(
    problem: dict[str, Any],
    completion_texts: list[str],
    reward_materials: Any = None,
) -> list[float]:
    del reward_materials
    environment = ReliquaryVerifiableEnvironment()
    return [
        environment.compute_reward(problem, completion)
        for completion in completion_texts
    ]


__all__ = [
    "ReliquaryVerifiableEnvironment",
    "score_reliquaryverifiable",
]
