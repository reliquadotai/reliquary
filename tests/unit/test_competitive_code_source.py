"""reliquary_competitive_code_v1 as a corpus prompt source.

The package (reliquary-environments, `reliquary-competitive-code`) serves
competitive-programming statements graded on hidden stdin/stdout tests. Its
train split is shared by problem identity: rows [0, 4181) are for SFT, rows
[4181, 6899) for RL, and no corpus job may ever be served an RL row.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reliquary.environment.agentic.types import canonical_json
from reliquary.environment.registry import ENVIRONMENT_SPECS, EnvironmentSpec

NAME = "reliquary_competitive_code_v1"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "competitive_code"
SFT = (0, 4181)
RL = (4181, 2718)


def test_the_source_is_registered_as_a_sandboxed_binary_program() -> None:
    spec = ENVIRONMENT_SPECS[NAME]
    assert spec.factory_path == (
        "reliquary_competitive_code.environment:CompetitiveCodeEnvironment"
    )
    assert spec.contract_version == "reliquary/stdio-program/v1"
    assert spec.admission_resource_class == "sandbox"
    assert spec.final_answer_policy == "fenced_python"
    assert (spec.reward_lattice_policy, spec.attainable_rewards) == (
        "binary-v1", (0.0, 1.0))
    # The package hands over its tests; this repository runs them in gVisor.
    assert spec.reward_materializer_method == "admission_reward_cases"
    assert not spec.scorer_path.endswith(":score_external_answers")
    assert spec.external_distribution == "reliquary-competitive-code"


def test_the_spec_pins_the_packaged_artifact() -> None:
    artifact = json.loads((FIXTURES / "artifact.json").read_text())
    spec = ENVIRONMENT_SPECS[NAME]
    assert artifact["environment"] == NAME
    assert artifact["contract"] == spec.contract_version
    assert artifact["entrypoints"]["replay"] == spec.factory_path
    digest = hashlib.sha256(canonical_json(artifact).encode()).hexdigest()
    assert digest == spec.environment_manifest_sha256


@pytest.mark.parametrize("name", ["guest.py", "compare.py", "extraction.py"])
def test_the_judge_fixtures_are_the_pinned_package_files(name) -> None:
    """The sandbox tests below run these copies; they must be the bytes the
    artifact pins, or they would test something the fleet never runs."""
    artifact = json.loads((FIXTURES / "artifact.json").read_text())
    path = {"guest.py": "judge/guest.py", "compare.py": "judge/compare.py",
            "extraction.py": "extraction.py"}[name]
    pinned = artifact["files"][f"reliquary_competitive_code/{path}"]
    assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == pinned


def _spec(**overrides) -> EnvironmentSpec:
    base = dict(
        name="stdio_code_v1",
        factory_path="reliquary_competitive_code.environment:CompetitiveCodeEnvironment",
        scorer_path="reliquary.environment.stdio_program:score_stdio_program",
        validator_authoritative_reward=True,
        admission_resource_class="sandbox",
        termination_policy="eos_or_cap",
        final_answer_policy="fenced_python",
        reward_lattice_policy="binary-v1",
        attainable_rewards=(0.0, 1.0),
        contract_version="reliquary/stdio-program/v1",
        reward_materializer_method="admission_reward_cases",
        environment_manifest_sha256="0" * 64,
        external_distribution="reliquary-competitive-code",
        external_artifact_resource="reliquary_competitive_code/artifact.json",
    )
    base.update(overrides)
    return EnvironmentSpec(**base)


def test_binary_materials_are_admitted_only_when_scored_here_in_the_sandbox() -> None:
    assert _spec().attainable_rewards == (0.0, 1.0)
    with pytest.raises(ValueError, match="scored here"):
        _spec(scorer_path="reliquary.environment.agentic.external:score_external_answers")
    with pytest.raises(ValueError, match="sandbox"):
        _spec(admission_resource_class="cpu")


def test_the_rl_share_can_never_be_declared_by_a_corpus_job() -> None:
    from reliquary.eval.sets import refuse_held_out_overlap

    refuse_held_out_overlap(NAME, *SFT)
    for start, count in ((RL[0], 1), (4180, 2), (0, 4182), (6000, 10)):
        with pytest.raises(ValueError, match="RL"):
            refuse_held_out_overlap(NAME, start, count)


class _Environment:
    """Rows as the package's task metadata marks them."""

    def __init__(self, uses) -> None:
        self._uses = uses

    def __len__(self) -> int:
        return len(self._uses)

    def get_problem(self, index: int) -> dict:
        problem = {"prompt": f"problem {index}", "environment": NAME}
        if self._uses[index] is not None:
            problem["use"] = self._uses[index]
        return problem


def _job(start: int, count: int):
    from tests.unit.test_corpus_export import _job_spec

    return _job_spec(prompt_source=NAME, prompt_start=start, prompt_count=count)


def test_a_row_the_source_marks_for_rl_is_never_served() -> None:
    """Declaration refuses the RL range; serving refuses an RL row whatever
    the job says, by the identity the package derives the share from."""
    from reliquary.validator.corpus_service import (
        CorpusPromptSourceError,
        SingleTurnPromptJob,
    )

    prompts = SingleTurnPromptJob(_job(0, 4), _Environment(["sft", "sft", "rl", None]))
    assert prompts.task_for(1).prompt == "problem 1"
    assert prompts.task_for(3).prompt == "problem 3"  # a source with no shares
    with pytest.raises(CorpusPromptSourceError, match="RL"):
        prompts.task_for(2)
