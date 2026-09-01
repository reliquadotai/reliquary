from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from reliquary.environment.base import Environment
from reliquary.environment.registry import get_environment_spec
from reliquary.environment.logic_tasks import (
    VIRTUAL_LENGTH,
    generate_logic_task,
)
from reliquary.protocol.profiles import PROFILES
from reliquary.environment.reliquarylogic import (
    ReliquaryLogicEnvironment,
    score_reliquarylogic,
)


SAMPLE = 4000


def _completion(value) -> str:
    return "reasoning\n```json\n" + json.dumps({"result": value}) + "\n```"


def _spec(problem: dict) -> dict:
    return json.loads(problem["ground_truth"])


def _reference(index: int):
    """Reference solution. Constraint specs deliberately omit it on the wire."""
    return generate_logic_task(index % VIRTUAL_LENGTH).expected


def test_satisfies_environment_protocol():
    assert isinstance(ReliquaryLogicEnvironment(), Environment)


def test_index_to_problem_is_deterministic_and_wraps():
    environment = ReliquaryLogicEnvironment()
    for index in (0, 1, 2, 7, 12345, 999_999):
        first = environment.get_problem(index)
        assert first == environment.get_problem(index)
        assert first == environment.get_problem(index + VIRTUAL_LENGTH)


def test_generator_is_total_over_a_wide_index_range():
    """No retry loops: every index must yield a task, always."""
    for index in range(0, SAMPLE * 977, 977):
        task = generate_logic_task(index)
        assert task.prompt
        assert task.check in (
            "equality", "numbrix_path", "cryptarithm_sum",
        )


FAMILIES = {
    "boolean_expressions", "cipher", "cryptarithm",
    "dyck_language", "numbrix", "web_of_lies",
}


def test_families_are_represented():
    families = {
        generate_logic_task(index).family for index in range(SAMPLE)
    }
    assert families == FAMILIES


def test_generation_stays_inside_the_hot_path_budget():
    """``get_problem`` sits in the batcher's per-submission path."""
    for index in range(200):
        generate_logic_task(index)
    started = time.perf_counter()
    for index in range(SAMPLE):
        generate_logic_task(index + 10**6)
    per_call_ms = (time.perf_counter() - started) / SAMPLE * 1000
    assert per_call_ms < 5.0, f"{per_call_ms:.3f} ms/call"


def test_reference_answer_is_accepted_for_every_family():
    environment = ReliquaryLogicEnvironment()
    seen = set()
    for index in range(SAMPLE):
        problem = environment.get_problem(index)
        spec = _spec(problem)
        assert environment.compute_reward(
            problem, _completion(_reference(index))
        ) == 1.0
        seen.add(spec["family"])
    assert seen == FAMILIES


def test_wrong_answers_are_rejected():
    environment = ReliquaryLogicEnvironment()
    for index in range(400):
        problem = environment.get_problem(index)
        expected = _reference(index)
        if isinstance(expected, bool):
            wrong = not expected
        elif isinstance(expected, str):
            wrong = expected + "zz"
        elif isinstance(expected, dict):
            wrong = {key: 0 for key in expected}
        else:
            wrong = [[0]]
        assert environment.compute_reward(problem, _completion(wrong)) == 0.0


def test_boolean_family_rejects_the_int_bool_conflation():
    environment = ReliquaryLogicEnvironment()
    for index in range(SAMPLE):
        problem = environment.get_problem(index)
        spec = _spec(problem)
        if spec["family"] != "boolean_expressions":
            continue
        numeric = 1 if _reference(index) else 0
        assert environment.compute_reward(problem, _completion(numeric)) == 0.0
        return
    raise AssertionError("no boolean task in sample")


def _first_numbrix(environment: ReliquaryLogicEnvironment):
    for index in range(SAMPLE):
        problem = environment.get_problem(index)
        spec = _spec(problem)
        if spec["family"] == "numbrix":
            assert "expected" not in spec, "constraint spec must not ship an answer"
            return problem, spec, _reference(index)
    raise AssertionError("no numbrix task in sample")


def test_numbrix_checker_enforces_each_invariant():
    environment = ReliquaryLogicEnvironment()
    problem, spec, reference = _first_numbrix(environment)
    grid = [list(row) for row in reference]
    size = spec["constraints"]["size"]

    # Swapping two values breaks adjacency while keeping the value multiset.
    broken = [list(row) for row in grid]
    broken[0][0], broken[size - 1][size - 1] = (
        broken[size - 1][size - 1], broken[0][0],
    )
    assert environment.compute_reward(problem, _completion(broken)) == 0.0

    duplicated = [list(row) for row in grid]
    duplicated[0][0] = duplicated[0][1]
    assert environment.compute_reward(problem, _completion(duplicated)) == 0.0

    out_of_range = [list(row) for row in grid]
    out_of_range[0][0] = size * size + 1
    assert environment.compute_reward(
        problem, _completion(out_of_range)
    ) == 0.0

    assert environment.compute_reward(
        problem, _completion(grid[:-1])
    ) == 0.0


def test_numbrix_clues_are_consistent_with_the_reference_solution():
    environment = ReliquaryLogicEnvironment()
    _problem, spec, solution = _first_numbrix(environment)
    clues = spec["constraints"]["clues"]
    revealed = 0
    for row_index, row in enumerate(clues):
        for column_index, given in enumerate(row):
            if given:
                revealed += 1
                assert given == solution[row_index][column_index]
    assert 0 < revealed < len(solution) ** 2


def test_malformed_completions_never_raise():
    environment = ReliquaryLogicEnvironment()
    problem = environment.get_problem(0)
    for completion in (
        "",
        "no json here",
        "```json\n{}\n```",
        '```json\n{"result": 1, "extra": 2}\n```',
        '```json\n{"other": 1}\n```',
        '```json\n{"result": 1.5}\n```',
        '```json\n{"result": NaN}\n```',
        '```json\n{"result": {"result": 1}}\n```',
        "```json\n[1, 2, 3]\n```",
        "```json\n{\"result\": " + "[" * 200 + "]" * 200 + "}\n```",
    ):
        assert environment.compute_reward(problem, completion) == 0.0


def test_batch_scorer_matches_single_rewards():
    environment = ReliquaryLogicEnvironment()
    problem = environment.get_problem(3)
    texts = [_completion(_reference(3)), "garbage"]
    assert score_reliquarylogic(problem, texts) == [1.0, 0.0]


def test_source_health_declares_no_external_dependency():
    health = ReliquaryLogicEnvironment().source_health()
    assert health["status"] == "ok"
    assert health["external_dependencies"] == []
    assert health["virtual_length"] == VIRTUAL_LENGTH


def _goldens():
    path = (
        Path(__file__).resolve().parents[2]
        / "tests/fixtures/reliquarylogic_v1.jsonl"
    )
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_golden_problem_and_reward_contract():
    environment = ReliquaryLogicEnvironment()
    goldens = _goldens()
    assert {golden["family"] for golden in goldens} == FAMILIES
    for golden in goldens:
        problem = environment.get_problem(golden["index"])
        assert problem["id"] == golden["id"]
        assert problem["operation_id"] == golden["operation_id"]
        assert problem["difficulty"] == golden["difficulty"]
        assert hashlib.sha256(
            problem["prompt"].encode()
        ).hexdigest() == golden["prompt_sha256"]
        assert hashlib.sha256(
            problem["ground_truth"].encode()
        ).hexdigest() == golden["target_sha256"]
        assert environment.compute_reward(
            problem, golden["accepted_completion"]
        ) == 1.0
        for rejected in golden["rejected_completions"]:
            assert environment.compute_reward(problem, rejected) == 0.0


def test_registry_binds_the_environment_and_its_manifest():
    spec = get_environment_spec("reliquarylogic_v1")
    assert spec.admission_resource_class == "cpu"
    assert spec.final_answer_policy == "json"
    assert spec.validator_authoritative_reward is True
    assert spec.attainable_rewards == (0.0, 1.0)
    assert spec.interaction_mode == "single_turn"
    # Constructing the catalog already validated the manifest, golden and
    # implementation digests; creating the environment proves the factory.
    assert spec.create().name == "reliquarylogic_v1"


def test_profile_and_registry_pin_the_same_manifest():
    profile = PROFILES["qwen3-4b-reliquary-logic-v8-dev1"]
    environment = profile.environments["reliquarylogic_v1"]
    spec = get_environment_spec("reliquarylogic_v1")
    assert (
        environment.environment_manifest_sha256
        == spec.environment_manifest_sha256
    )
    assert environment.environment_contract_id == spec.contract_version


def test_backbite_always_yields_a_hamiltonian_path():
    """The move must be total: no retry loop can exist behind get_problem."""
    from reliquary.environment.logic_tasks import _backbite, _serpentine
    from reliquary.environment.records_tasks import HashCounterRng

    for seed in range(200):
        rng = HashCounterRng(hashlib.sha256(str(seed).encode()).digest())
        size = 4 + seed % 2
        path = _backbite(rng, _serpentine(size), size, 4 * size * size)
        assert sorted(path) == sorted(
            (row, column) for row in range(size) for column in range(size)
        )
        assert all(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            for a, b in zip(path, path[1:])
        )


def test_prompts_do_not_collide_across_a_wide_index_range():
    """Duplicate prompt content is burned silently at seal, so measure it."""
    environment = ReliquaryLogicEnvironment()
    by_family: dict[str, list[str]] = {}
    for index in range(SAMPLE):
        problem = environment.get_problem(index)
        family = _spec(problem)["family"]
        by_family.setdefault(family, []).append(problem["id"])
    for family, ids in by_family.items():
        duplicate_rate = 1 - len(set(ids)) / len(ids)
        assert duplicate_rate < 0.02, f"{family}: {duplicate_rate:.2%}"


def _first_of(environment, family):
    for index in range(SAMPLE):
        problem = environment.get_problem(index)
        spec = _spec(problem)
        if spec["family"] == family:
            return problem, spec, _reference(index)
    raise AssertionError(f"no {family} task in sample")


def test_cryptarithm_checker_enforces_each_invariant():
    environment = ReliquaryLogicEnvironment()
    problem, spec, reference = _first_of(environment, "cryptarithm")
    assert "expected" not in spec, "constraint spec must not ship an answer"

    letters = sorted(reference)
    collided = dict(reference)
    collided[letters[1]] = collided[letters[0]]
    assert environment.compute_reward(problem, _completion(collided)) == 0.0

    dropped = {k: v for k, v in reference.items() if k != letters[0]}
    assert environment.compute_reward(problem, _completion(dropped)) == 0.0

    out_of_range = dict(reference)
    out_of_range[letters[0]] = 10
    assert environment.compute_reward(
        problem, _completion(out_of_range)
    ) == 0.0

    leading = dict(reference)
    leading[spec["constraints"]["sum"][0]] = 0
    assert environment.compute_reward(problem, _completion(leading)) == 0.0

    assert environment.compute_reward(problem, _completion(reference)) == 1.0


def test_cryptarithm_puzzle_is_arithmetically_sound():
    environment = ReliquaryLogicEnvironment()
    _problem, spec, reference = _first_of(environment, "cryptarithm")
    addends = spec["constraints"]["addends"]
    total = spec["constraints"]["sum"]
    decode = lambda word: int("".join(str(reference[c]) for c in word))
    assert decode(addends[0]) + decode(addends[1]) == decode(total)
    assert len(set(reference.values())) == len(reference)
