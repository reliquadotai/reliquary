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
