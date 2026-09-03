"""Protocol-v5 reasoning prompt cutover invariants."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from reliquary.protocol import profiles


V4_PROFILE_ID = "qwen3-4b-base-dapo-v4"
V5_PROFILE_ID = "qwen3-4b-base-dapo-reasoning-v5"

MATH_TEMPLATE = (
    "Solve the following math problem step by step.\n\n"
    "$problem\n\n"
    "Put your final answer within \\boxed{}."
)
CODE_TEMPLATE = (
    "Solve the following programming problem step by step.\n\n"
    "$problem$contract\n\n"
    "After your reasoning, provide the final implementation in the last "
    "fenced Python code block."
)


def test_v5_is_a_prompt_only_fork_of_v4():
    v4 = profiles.PROFILES[V4_PROFILE_ID]
    v5 = profiles.PROFILES[V5_PROFILE_ID]

    assert v4.protocol_version == 4
    assert v5.protocol_version == 5
    assert v5.model_id == v4.model_id
    assert v5.model_revision == v4.model_revision
    assert v5.collection_seconds == v4.collection_seconds
    assert v5.upload_grace_seconds == v4.upload_grace_seconds
    assert v5.prompt_encoding == v4.prompt_encoding == "raw"
    assert v5.sampling == v4.sampling
    assert v5.throughput_tiebreak == v4.throughput_tiebreak

    for environment in v4.environments:
        before = v4.environments[environment]
        after = v5.environments[environment]
        assert after.max_new_tokens == before.max_new_tokens
        assert after.bft == before.bft
        assert after.answer_format == before.answer_format
        assert before.prompt_template is None
        assert after.prompt_template is not None


@pytest.mark.parametrize(
    ("environment", "template_id", "template"),
    [
        (
            "openmathinstruct",
            "openmathinstruct-step-by-step-v1",
            MATH_TEMPLATE,
        ),
        (
            "opencodeinstruct",
            "opencodeinstruct-step-by-step-v1",
            CODE_TEMPLATE,
        ),
    ],
)
def test_v5_generation_contract_pins_exact_prompt_template(
    environment,
    template_id,
    template,
):
    contract = profiles.to_generation_contract(V5_PROFILE_ID)
    prompt_contract = contract["environments"][environment]["prompt_template"]

    assert prompt_contract == {
        "id": template_id,
        "renderer": "dollar-substitution-v1",
        "template": template,
        "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }


def test_v5_prompt_templates_are_immutable():
    template = profiles.PROFILES[V5_PROFILE_ID].environments[
        "openmathinstruct"
    ].prompt_template
    assert template is not None

    with pytest.raises(FrozenInstanceError):
        template.template = "changed"


def test_v5_checkpoint_lineage_binds_generation_contract():
    script = """
import json
from reliquary.constants import PROTOCOL_GENERATION_CONTRACT
from reliquary.validator.checkpoint_profile import active_checkpoint_profile
print(json.dumps({
    "checkpoint_profile": active_checkpoint_profile(),
    "generation_contract": PROTOCOL_GENERATION_CONTRACT,
}))
"""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("RELIQUARY_")
    }
    env["RELIQUARY_PROTOCOL_PROFILE"] = V5_PROFILE_ID
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(completed.stdout)
    checkpoint_profile = payload["checkpoint_profile"]
    canonical_contract = json.dumps(
        payload["generation_contract"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert checkpoint_profile["schema_version"] == 2
    assert checkpoint_profile["generation_contract_sha256"] == (
        hashlib.sha256(canonical_contract).hexdigest()
    )


def test_v4_generation_contract_remains_byte_compatible():
    contract = profiles.to_generation_contract(V4_PROFILE_ID)

    assert all(
        "prompt_template" not in environment
        for environment in contract["environments"].values()
    )


def test_v5_templates_render_exact_canonical_prompts():
    v5 = profiles.PROFILES[V5_PROFILE_ID]
    function_contract = (
        "\n\nWrite your solution as a Python function named `add` that takes "
        "2 arguments and returns the result; do not read from stdin or print."
    )

    math = v5.environments["openmathinstruct"].prompt_template
    code = v5.environments["opencodeinstruct"].prompt_template
    assert math is not None
    assert code is not None
    assert math.render(problem="What is 2+2?") == (
        "Solve the following math problem step by step.\n\n"
        "What is 2+2?\n\n"
        "Put your final answer within \\boxed{}."
    )
    assert code.render(
        problem="Implement addition.",
        contract=function_contract,
    ) == (
        "Solve the following programming problem step by step.\n\n"
        "Implement addition."
        + function_contract
        + "\n\nAfter your reasoning, provide the final implementation in the "
        "last fenced Python code block."
    )


@pytest.mark.parametrize(
    "template,match",
    [
        ("no input", r"must contain \$problem"),
        ("$problem $unknown", "unknown placeholders"),
        ("$problem $", "is not valid"),
    ],
)
def test_prompt_templates_fail_closed(template, match):
    with pytest.raises(ValueError, match=match):
        profiles.PromptTemplateProfile(
            template_id="bad-template",
            template=template,
        )


def test_math_environment_uses_v5_profile_prompt(monkeypatch):
    from reliquary.environment.openmathinstruct import (
        OpenMathInstructEnvironment,
    )

    env = OpenMathInstructEnvironment.__new__(OpenMathInstructEnvironment)
    env._dataset = [{"problem": "What is 2+2?", "expected_answer": "4"}]
    monkeypatch.setattr(
        profiles,
        "ACTIVE_PROTOCOL_PROFILE",
        profiles.PROFILES[V5_PROFILE_ID],
    )

    assert env.get_problem(0)["prompt"] == (
        "Solve the following math problem step by step.\n\n"
        "What is 2+2?\n\n"
        "Put your final answer within \\boxed{}."
    )


def test_code_environment_uses_v5_profile_prompt(monkeypatch):
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment

    cases = [
        {
            "entry": {"kind": "function", "name": "add"},
            "args": [1, 2],
            "kwargs": {},
            "expected": 3,
            "compare": "exact",
        }
    ]
    env = OpenCodeInstructEnvironment.__new__(OpenCodeInstructEnvironment)
    env._dataset = [
        {
            "input": "Implement addition.",
            "structured_cases": cases,
        }
    ]
    env._cases_by_id = {}
    monkeypatch.setattr(
        profiles,
        "ACTIVE_PROTOCOL_PROFILE",
        profiles.PROFILES[V5_PROFILE_ID],
    )

    prompt = env.get_problem(0)["prompt"]
    assert prompt.startswith(
        "Solve the following programming problem step by step.\n\n"
        "Implement addition."
    )
    assert "Python function named `add`" in prompt
    assert prompt.endswith(
        "After your reasoning, provide the final implementation in the last "
        "fenced Python code block."
    )


# ── reliquarylogic_v1 ────────────────────────────────────────────────────
#
# The logic profile declared a prompt template from the start, but the
# environment never rendered it: ``problem_from_task`` returned the bare
# generated puzzle. The identity template hid the gap, because rendering it
# and skipping it produced the same bytes.

LOGIC_PROFILE_ID = "qwen3-4b-reliquary-logic-v8-dev1"
LOGIC_TEMPLATE = (
    "Solve the following problem step by step.\n\n"
    "$problem\n\n"
    "After your reasoning, give the final answer in the last fenced JSON "
    "code block."
)


def _logic_profile():
    return profiles.PROFILES[LOGIC_PROFILE_ID]


def test_logic_generation_contract_pins_exact_prompt_template():
    contract = profiles.to_generation_contract(LOGIC_PROFILE_ID)
    prompt_contract = contract["environments"]["reliquarylogic_v1"][
        "prompt_template"
    ]

    assert prompt_contract == {
        "id": "reliquary-logic-step-by-step-v1",
        "renderer": "dollar-substitution-v1",
        "template": LOGIC_TEMPLATE,
        "sha256": hashlib.sha256(LOGIC_TEMPLATE.encode("utf-8")).hexdigest(),
    }


def test_logic_template_renders_the_exact_canonical_prompt():
    template = _logic_profile().environments[
        "reliquarylogic_v1"
    ].prompt_template
    assert template is not None

    assert template.render(problem="Is A true?") == (
        "Solve the following problem step by step.\n\n"
        "Is A true?\n\n"
        "After your reasoning, give the final answer in the last fenced JSON "
        "code block."
    )


def test_logic_environment_uses_profile_prompt(monkeypatch):
    from reliquary.environment.logic_tasks import generate_logic_task
    from reliquary.environment.reliquarylogic import ReliquaryLogicEnvironment

    monkeypatch.setattr(
        profiles, "ACTIVE_PROTOCOL_PROFILE", _logic_profile()
    )
    puzzle = generate_logic_task(0).prompt
    prompt = ReliquaryLogicEnvironment().get_problem(0)["prompt"]

    assert prompt == (
        "Solve the following problem step by step.\n\n"
        + puzzle
        + "\n\nAfter your reasoning, give the final answer in the last "
        "fenced JSON code block."
    )
    # The generated answer shape must survive the wrapper untouched: it is
    # the only statement of the channel the grader reads.
    assert '```json\n{"result": <final value>}\n```' in prompt


def test_logic_environment_falls_back_to_the_bare_puzzle_off_profile(
    monkeypatch,
):
    """A profile that does not declare logic must not break the generator.

    The live Math+Code profiles have no logic environment, and offline
    scoring runs under whichever profile happens to be active.
    """
    from reliquary.environment.logic_tasks import generate_logic_task
    from reliquary.environment.reliquarylogic import ReliquaryLogicEnvironment

    monkeypatch.setattr(
        profiles,
        "ACTIVE_PROTOCOL_PROFILE",
        profiles.PROFILES["qwen35-2b-auction-v2"],
    )

    assert (
        ReliquaryLogicEnvironment().get_problem(0)["prompt"]
        == generate_logic_task(0).prompt
    )


def test_logic_problem_id_is_stable_across_the_prompt_contract(monkeypatch):
    """Identity is the puzzle, not the envelope.

    ``prompt_content_sha256`` burns a problem for the content cooldown, so an
    id derived from the rendered prompt would silently resurrect every
    already-consumed index on any future prompt change.
    """
    from reliquary.environment.reliquarylogic import ReliquaryLogicEnvironment

    environment = ReliquaryLogicEnvironment()
    monkeypatch.setattr(
        profiles,
        "ACTIVE_PROTOCOL_PROFILE",
        profiles.PROFILES["qwen35-2b-auction-v2"],
    )
    bare = environment.get_problem(7)
    monkeypatch.setattr(
        profiles, "ACTIVE_PROTOCOL_PROFILE", _logic_profile()
    )
    rendered = environment.get_problem(7)

    assert rendered["prompt"] != bare["prompt"]
    assert rendered["id"] == bare["id"]


def test_logic_budget_matches_the_reasoning_profile():
    """One base model, one budget to think in.

    v8 pins the same base revision as v4/v5, so a logic rollout has no reason
    to be allowed less room than a math or code rollout on the same weights —
    least of all now that its prompt asks for the reasoning first.
    """
    v5 = profiles.PROFILES[V5_PROFILE_ID]
    logic = _logic_profile()

    assert logic.model_id == v5.model_id
    assert logic.model_revision == v5.model_revision
    assert logic.environments["reliquarylogic_v1"].max_new_tokens == (
        v5.environments["openmathinstruct"].max_new_tokens
    )
    assert logic.throughput_tiebreak == v5.throughput_tiebreak
