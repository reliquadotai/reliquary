from dataclasses import replace
import pickle

import pytest

from reliquary.environment.agentic.external import (
    ExternalAnswerEnvironment, load_external_backend, score_external_answers,
)
from reliquary.environment.registry import get_environment_spec


class AnswerBackend:
    name = "reliquary_logic_v2"
    validator_authoritative_reward = True
    max_turns = 1

    def __len__(self):
        return 3

    def task(self, index):
        return {"id": f"task-{index}", "prompt": f"Answer {index}.",
                "metadata": {"family": "test", "split": "train"}}

    def grade(self, index, completion):
        success = completion == str(index)
        return {"reward": float(success), "success": success, "state_digest": "a" * 64}


def test_answer_adapter_binds_problem_identity_before_scoring():
    spec = get_environment_spec("reliquary_logic_v2")
    environment = ExternalAnswerEnvironment(AnswerBackend(), spec)
    problem = environment.get_problem(2)
    assert environment.get_problem(5) == problem
    assert environment.compute_reward(problem, "2") == 1.0
    assert environment.compute_reward(problem, "malformed") == 0.0
    assert problem["family"] == "test"
    for field, forged in (("generator_index", True), ("generator_index", -1),
                          ("generator_index", 1), ("id", "different"),
                          ("prompt", "different"), ("ground_truth", "2"),
                          ("environment", "opencodeinstruct")):
        assert environment.compute_reward({**problem, field: forged}, "2") == 0.0
    assert environment.compute_reward(problem, None) == 0.0
    assert pickle.loads(pickle.dumps(score_external_answers)) is score_external_answers
    with pytest.raises(ValueError, match="environment mismatch"):
        spec.score_many({**problem, "environment": "opencodeinstruct"}, ["2"])


@pytest.mark.parametrize("reward", [True, float("nan"), float("inf"), 0.5, -1, 2, "1"])
def test_answer_adapter_rejects_invalid_backend_reward(reward):
    backend = AnswerBackend()
    backend.grade = lambda index, completion: {
        "reward": reward, "success": True, "state_digest": "a" * 64,
    }
    environment = ExternalAnswerEnvironment(backend, get_environment_spec(backend.name))
    with pytest.raises((ValueError, TypeError)):
        environment.compute_reward(environment.get_problem(0), "0")


@pytest.mark.parametrize("field,value", [
    ("name", "another"), ("max_turns", True), ("max_turns", 2),
    ("validator_authoritative_reward", False),
])
def test_answer_adapter_rejects_incompatible_backend(field, value):
    backend = AnswerBackend()
    setattr(backend, field, value)
    with pytest.raises(ValueError):
        ExternalAnswerEnvironment(backend, get_environment_spec("reliquary_logic_v2"))


def test_external_single_turn_requires_reviewed_contract_and_fixed_split():
    spec = get_environment_spec("reliquary_logic_v2")
    for changes in ({"environment_manifest_sha256": None},
                    {"contract_version": "unreviewed"},
                    {"validator_authoritative_reward": False},
                    {"attainable_rewards": (0.0, 0.5, 1.0)}):
        with pytest.raises(ValueError):
            replace(spec, **changes)
    with pytest.raises(ValueError, match="unsupported.*split"):
        load_external_backend(spec, split="unbound-prompt-config")
