from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from reliquary.environment.records_tasks import VIRTUAL_LENGTH
from reliquary.environment.registry import get_environment_spec
from reliquary.environment.reliquaryverifiable import (
    ReliquaryVerifiableEnvironment,
)
from reliquary.environment.structured_output import (
    StructuredOutputError,
    extract_json_answer,
)


ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / "tests/fixtures/reliquaryverifiable_v1.jsonl"
MANIFEST = (
    ROOT
    / "reliquary/environment/manifests/reliquaryverifiable_v1.json"
)


def _goldens():
    return [json.loads(line) for line in GOLDENS.read_text().splitlines()]


def test_golden_problem_and_reward_contract():
    environment = ReliquaryVerifiableEnvironment()
    for golden in _goldens():
        problem = environment.get_problem(golden["index"])
        assert problem["id"] == golden["id"]
        assert problem["operation_id"] == golden["operation_id"]
        assert problem["difficulty"] == golden["difficulty"]
        assert hashlib.sha256(problem["prompt"].encode()).hexdigest() == golden[
            "prompt_sha256"
        ]
        assert hashlib.sha256(problem["ground_truth"].encode()).hexdigest() == golden[
            "target_sha256"
        ]
        assert environment.compute_reward(
            problem, golden["accepted_completion"]
        ) == 1.0
        for rejected in golden["rejected_completions"]:
            assert environment.compute_reward(problem, rejected) == 0.0


def test_last_fenced_json_answer_wins():
    assert extract_json_answer(
        "reasoning\n```json\n{\"result\":0}\n```\n"
        "revision\n```json\n{\"result\":1}\n```"
    ) == {"result": 1}


@pytest.mark.parametrize(
    "completion",
    [
        "",
        "not json",
        "[]",
        '{"result":1,"result":2}',
        '{"result":1.0}',
        '{"result":NaN}',
        '{"result":9007199254740992}',
        '{"result":{"a":{"b":{"c":{"d":{"e":{"f":{"g":{"h":{"i":1}}}}}}}}}}',
    ],
)
def test_structured_output_rejects_ambiguous_or_unbounded_values(completion):
    with pytest.raises(StructuredOutputError):
        extract_json_answer(completion)


def test_oversized_completion_is_rejected():
    with pytest.raises(StructuredOutputError, match="byte limit"):
        extract_json_answer('{"result":"' + ("x" * 20_000) + '"}')


def test_problem_wrap_and_source_health():
    environment = ReliquaryVerifiableEnvironment()
    assert environment.get_problem(0) == environment.get_problem(VIRTUAL_LENGTH)
    assert len(environment) == 1 << 25
    health = environment.source_health()
    assert health["status"] == "ok"
    assert health["external_dependencies"] == []


def test_compute_reward_never_raises_on_bad_inputs():
    environment = ReliquaryVerifiableEnvironment()
    problem = environment.get_problem(0)
    malformed = [None, "", "x", "{" * 100, "\x00", "[]"]
    malformed.extend(f"garbled-{index}" for index in range(10_000))
    assert all(environment.compute_reward(problem, value) == 0.0 for value in malformed)
    assert environment.compute_reward({}, "{}") == 0.0
    assert environment.compute_reward({"ground_truth": "garbled"}, "{}") == 0.0


def test_first_ten_thousand_indices_are_deterministic_and_unique():
    environment = ReliquaryVerifiableEnvironment()
    ids: set[str] = set()
    operation_ids: set[str] = set()
    for index in range(10_000):
        first = environment.get_problem(index)
        second = environment.get_problem(index)
        assert first == second
        assert first["id"] not in ids
        ids.add(first["id"])
        operation_ids.add(first["operation_id"])
    assert operation_ids == {
        "filter-sort-project-v1",
        "active-sort-rename-v1",
        "deduplicate-project-v1",
        "group-count-sum-v1",
        "top-project-v1",
    }


def test_generation_is_stable_across_process_hash_seeds():
    code = (
        "import json; "
        "from reliquary.environment.reliquaryverifiable import "
        "ReliquaryVerifiableEnvironment; "
        "environment=ReliquaryVerifiableEnvironment(); "
        "print(json.dumps([environment.get_problem(i) for i in "
        "(0,2,3,12,19,9999)],sort_keys=True,separators=(',',':')))"
    )
    outputs = []
    for hash_seed in ("1", "987654321"):
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]


def test_manifest_digest_matches_registered_consensus_identity():
    canonical = json.dumps(
        json.loads(MANIFEST.read_text()),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == (
        get_environment_spec("reliquaryverifiable_v1").environment_manifest_sha256
    )
