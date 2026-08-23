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
        "code_cap",
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
    code_cap,
    thinking_budget,
):
    profile = profiles.PROFILES[profile_id]

    assert profile.profile_id == profile_id
    assert profile.model_id == model_id
    assert profile.model_revision == model_revision
    assert profile.protocol_version == protocol_version
    assert profile.collection_seconds == collection_seconds
    assert profile.upload_grace_seconds == 33
    assert profile.prompt_encoding == "chat_template"
    assert profile.sampling == profiles.SamplingProfile(
        rollouts=8,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        do_sample=False,
    )

    math = profile.environments["openmathinstruct"]
    assert math.max_new_tokens == math_cap
    assert math.answer_format == "boxed_or_trailing_number"
    assert math.bft == profiles.BFTProfile(
        thinking_budget=thinking_budget,
        answer_budget=512,
        force_answer=True,
    )

    code = profile.environments["opencodeinstruct"]
    assert code.max_new_tokens == code_cap
    assert code.bft is None
    assert code.answer_format is None


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
    assert contract["prompt_encoding"] == "chat_template"
    assert (
        contract["environments"]["openmathinstruct"]["answer_format"]
        == "boxed_or_trailing_number"
    )

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
        profile.prompt_encoding = "raw"
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
                "overlong_factor": 0.0,
                "forced_zero": False,
                "proof_attempts": 64,
                "ranked_proof_attempts": 64,
                "model": "Qwen/Qwen3.5-2B",
                "t": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "bft_enabled": True,
                "raw_prompts": False,
                "omi_train_only": False,
                "clip_low": 0.2,
                "clip_high": 0.2,
                "opt_8bit": True,
                "rollouts": 8,
                "b_batch": 8,
                "max_submissions": 8,
                "publish_interval": 4,
                "prompt_encoding": "chat_template",
                "answer_format": "boxed_or_trailing_number",
            },
        ),
        (
            "qwen35-4b-auction-v3",
            {
                "version": 3,
                "math_cap": 16384,
                "code_cap": 16384,
                "thinking": 15616,
                "window": 300.0,
                "domain": "reliquary-forced-seed-v3",
                "lr": 1e-6,
                "kl": 0.01,
                "mask_math": False,
                "overlong_factor": 0.5,
                "forced_zero": True,
                "proof_attempts": 64,
                "ranked_proof_attempts": 16,
                "model": "Qwen/Qwen3.5-4B",
                "t": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "bft_enabled": True,
                "raw_prompts": False,
                "omi_train_only": False,
                "clip_low": 0.2,
                "clip_high": 0.2,
                "opt_8bit": True,
                "rollouts": 8,
                "b_batch": 8,
                "max_submissions": 8,
                "publish_interval": 4,
                "prompt_encoding": "chat_template",
                "answer_format": "boxed_or_trailing_number",
            },
        ),
        (
            # v4: DAPO-faithful restart from a true base model. No BFT (the base
            # emits no <think>), DAPO/verl rollout sampling (which also makes
            # warp() the identity, so the PPO ratio lives in the space the
            # samples came from), KL removed, raw-completion prompts, and the
            # OMI corpus restricted to its canonical shards.
            "qwen3-4b-base-dapo-v4",
            {
                "version": 4,
                # Length-curriculum start point: half of v3's 16384 cap / 300s
                # window, sized to ck0's short reasoning, ramps up with it.
                "math_cap": 8192,
                "code_cap": 8192,
                "thinking": 0,
                "window": 100.0,
                "domain": "reliquary-forced-seed-v4",
                "lr": 1e-6,
                "kl": 0.0,
                "mask_math": False,
                "overlong_factor": 0.5,
                "forced_zero": True,
                "proof_attempts": 64,
                "ranked_proof_attempts": 32,
                "model": "Qwen/Qwen3-4B-Base",
                "t": 1.0,
                "top_p": 1.0,
                # 0, not None: warp() guards on `top_k and top_k > 0` but the
                # miner's ForcedSeedLogitsProcessor coerces with int(top_k),
                # which raises on None.
                "top_k": 0,
                "bft_enabled": False,
                "raw_prompts": True,
                "omi_train_only": True,
                "clip_low": 0.2,
                "clip_high": 0.28,
                "opt_8bit": False,
                "rollouts": 16,
                "b_batch": 16,
                # 2·B_BATCH: cover every slot + one retry/improvement margin.
                "max_submissions": 32,
                "publish_interval": 16,
                "prompt_encoding": "raw",
                "answer_format": "boxed",
            },
        ),
        (
            # v5 is a prompt-only protocol fork: every generation/training knob
            # remains at its v4 value while the signed environment templates
            # restore the missing step-by-step reasoning cue.
            "qwen3-4b-base-dapo-reasoning-v5",
            {
                "version": 5,
                "math_cap": 8192,
                "code_cap": 8192,
                "thinking": 0,
                "window": 100.0,
                "domain": "reliquary-forced-seed-v5",
                "lr": 1e-6,
                "kl": 0.0,
                "mask_math": False,
                "overlong_factor": 0.5,
                "forced_zero": True,
                "proof_attempts": 64,
                "ranked_proof_attempts": 32,
                "model": "Qwen/Qwen3-4B-Base",
                "t": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "bft_enabled": False,
                "raw_prompts": True,
                "omi_train_only": True,
                "clip_low": 0.2,
                "clip_high": 0.28,
                "opt_8bit": False,
                "rollouts": 16,
                "b_batch": 16,
                "max_submissions": 32,
                "publish_interval": 16,
                "prompt_encoding": "raw",
                "answer_format": "boxed",
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
    "overlong_factor": c.OVERLONG_PENALTY_FACTOR,
    "forced_zero": c.TRAIN_FORCED_REWARD_ZERO,
    "proof_attempts": c.MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW,
    "ranked_proof_attempts": c.MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW,
    "model": c.PROTOCOL_MODEL_ID,
    "t": c.T_PROTO,
    "top_p": c.TOP_P_PROTO,
    "top_k": c.TOP_K_PROTO,
    "bft_enabled": c.BFT_ENABLED,
    "raw_prompts": c.RAW_COMPLETION_PROMPTS,
    "omi_train_only": c.OMI_TRAIN_SHARDS_ONLY,
    "clip_low": c.PPO_CLIP_EPSILON_LOW,
    "clip_high": c.PPO_CLIP_EPSILON_HIGH,
    "opt_8bit": c.OPTIMIZER_STATE_8BIT,
    "rollouts": c.M_ROLLOUTS,
    "b_batch": c.B_BATCH,
    "max_submissions": c.MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
    "publish_interval": c.CHECKPOINT_PUBLISH_INTERVAL_WINDOWS,
    "prompt_encoding": c.ACTIVE_PROTOCOL_PROFILE.prompt_encoding,
    "answer_format": c.MATH_ANSWER_FORMAT,
}))
"""
    # Drop inherited RELIQUARY_* overrides — several pinned constants are
    # env-overridable and a leaked override would flip them.
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("RELIQUARY_")
    }
    env["RELIQUARY_PROTOCOL_PROFILE"] = profile_id
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(completed.stdout) == expected
