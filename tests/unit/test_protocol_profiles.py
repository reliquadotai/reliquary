import importlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from reliquary.protocol import profiles


@pytest.mark.parametrize(
    (
        "profile_id",
        "model_id",
        "model_revision",
        "protocol_version",
        "collection_seconds",
        "math_cap",
        "thinking_budget",
    ),
    [
        (
            "qwen35-2b-auction-v2",
            "Qwen/Qwen3.5-2B",
            "15852e8c16360a2fea060d615a32b45270f8a8fc",
            2,
            100,
            32768,
            2048,
        ),
        (
            "qwen35-4b-auction-v3",
            "Qwen/Qwen3.5-4B",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            3,
            300,
            16384,
            15616,
        ),
    ],
)
def test_versioned_profile_contracts(
    profile_id,
    model_id,
    model_revision,
    protocol_version,
    collection_seconds,
    math_cap,
    thinking_budget,
):
    profile = profiles.PROFILES[profile_id]

    assert profile.profile_id == profile_id
    assert profile.model_id == model_id
    assert profile.model_revision == model_revision
    assert profile.protocol_version == protocol_version
    assert profile.collection_seconds == collection_seconds
    assert profile.upload_grace_seconds == 33
    assert profile.sampling == profiles.SamplingProfile(
        rollouts=8,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        do_sample=False,
    )

    math = profile.environments["openmathinstruct"]
    assert math.max_new_tokens == math_cap
    assert math.bft == profiles.BFTProfile(
        thinking_budget=thinking_budget,
        answer_budget=512,
        force_answer=True,
    )

    code = profile.environments["opencodeinstruct"]
    assert code.max_new_tokens == 32768
    assert code.bft is None


def test_default_resolver_and_environment_override(monkeypatch):
    monkeypatch.delenv("RELIQUARY_PROTOCOL_PROFILE", raising=False)
    assert profiles.DEFAULT_PROFILE_ID == "qwen35-2b-auction-v2"
    assert (
        profiles.resolve_protocol_profile()
        is profiles.PROFILES[profiles.DEFAULT_PROFILE_ID]
    )

    monkeypatch.setenv(
        "RELIQUARY_PROTOCOL_PROFILE",
        "qwen35-4b-auction-v3",
    )
    assert (
        profiles.resolve_protocol_profile()
        is profiles.PROFILES["qwen35-4b-auction-v3"]
    )
    assert (
        profiles.resolve_protocol_profile("qwen35-2b-auction-v2")
        is profiles.PROFILES["qwen35-2b-auction-v2"]
    )


def test_active_profile_is_resolved_at_import(monkeypatch):
    with monkeypatch.context() as env:
        env.delenv("RELIQUARY_PROTOCOL_PROFILE", raising=False)
        reloaded = importlib.reload(profiles)
        assert (
            reloaded.ACTIVE_PROTOCOL_PROFILE
            is reloaded.PROFILES[reloaded.DEFAULT_PROFILE_ID]
        )

        env.setenv(
            "RELIQUARY_PROTOCOL_PROFILE",
            "qwen35-4b-auction-v3",
        )
        reloaded = importlib.reload(profiles)
        assert (
            reloaded.ACTIVE_PROTOCOL_PROFILE
            is reloaded.PROFILES["qwen35-4b-auction-v3"]
        )

    importlib.reload(profiles)


@pytest.mark.parametrize("profile_id", ["", "unknown", "QWEN35-2B-AUCTION-V2"])
def test_unknown_profile_ids_fail_closed(monkeypatch, profile_id):
    monkeypatch.setenv("RELIQUARY_PROTOCOL_PROFILE", profile_id)
    with pytest.raises(ValueError, match="unknown protocol profile"):
        profiles.resolve_protocol_profile()
    with pytest.raises(ValueError, match="unknown protocol profile"):
        profiles.resolve_protocol_profile(profile_id)


def test_generation_contract_is_json_safe_and_detached():
    profile = profiles.PROFILES["qwen35-4b-auction-v3"]

    contract = profile.to_generation_contract()
    assert json.loads(json.dumps(contract)) == contract
    assert profiles.to_generation_contract(profile) == contract
    assert profiles.to_generation_contract(profile.profile_id) == contract

    contract["sampling"]["rollouts"] = 999
    contract["environments"]["openmathinstruct"]["bft"][
        "thinking_budget"
    ] = 1
    assert profile.sampling.rollouts == 8
    assert profile.environments["openmathinstruct"].bft.thinking_budget == 15616


def test_profiles_are_frozen_slotted_and_recursively_immutable():
    profile = profiles.PROFILES[profiles.DEFAULT_PROFILE_ID]
    math = profile.environments["openmathinstruct"]
    bft = math.bft
    assert bft is not None

    for value in (profile, profile.sampling, math, bft):
        assert is_dataclass(value)
        assert not hasattr(value, "__dict__")

    with pytest.raises(FrozenInstanceError):
        profile.protocol_version = 99
    with pytest.raises(FrozenInstanceError):
        profile.sampling.temperature = 1.0
    with pytest.raises(FrozenInstanceError):
        math.max_new_tokens = 1
    with pytest.raises(FrozenInstanceError):
        bft.force_answer = False
    with pytest.raises(TypeError):
        profile.environments["new"] = math
    with pytest.raises(TypeError):
        profiles.PROFILES["new"] = profile


@pytest.mark.parametrize(
    "profile_id,expected",
    [
        (
            "qwen35-2b-auction-v2",
            {
                "version": 2,
                "math_cap": 32768,
                "code_cap": 32768,
                "thinking": 2048,
                "window": 100.0,
                "domain": "reliquary-forced-seed-v2",
                "lr": 5e-6,
                "kl": 0.04,
                "mask_math": False,
                "proof_attempts": 64,
            },
        ),
        (
            "qwen35-4b-auction-v3",
            {
                "version": 3,
                "math_cap": 16384,
                "code_cap": 32768,
                "thinking": 15616,
                "window": 300.0,
                "domain": "reliquary-forced-seed-v3",
                "lr": 1e-6,
                "kl": 0.01,
                "mask_math": True,
                "proof_attempts": 16,
            },
        ),
    ],
)
def test_profile_atomically_drives_runtime_constants(profile_id, expected):
    script = """
import json
from reliquary import constants as c
print(json.dumps({
    "version": c.PROTOCOL_VERSION,
    "math_cap": c.max_new_tokens_for_environment("openmathinstruct"),
    "code_cap": c.max_new_tokens_for_environment("opencodeinstruct"),
    "thinking": c.BFT_THINKING_BUDGET,
    "window": c.WINDOW_COLLECTION_SECONDS,
    "domain": c.FORCED_SEED_DOMAIN,
    "lr": c.LEARNING_RATE,
    "kl": c.KL_BETA,
    "mask_math": c.MASK_MATH_FORCED_FROM_LOSS,
    "proof_attempts": c.MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
}))
"""
    env = dict(os.environ)
    env["RELIQUARY_PROTOCOL_PROFILE"] = profile_id
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(completed.stdout) == expected
