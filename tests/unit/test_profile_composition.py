"""A profile splits into model, run policy and environment bodies, and back."""

import json
from pathlib import Path

import pytest

from reliquary.protocol.composition import (
    MODEL_IDENTITY_FIELDS,
    RUN_POLICIES,
    ModelSpec,
    RunPolicy,
    check_profile_invariants,
    compose_profile,
    model_spec_of,
    run_policy_of,
)
from reliquary.protocol.profiles import PROFILES, profile_from_contract
from reliquary.protocol.release_contract import canonical_sha256

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _recompose(profile):
    return compose_profile(
        profile_id=profile.profile_id,
        model=model_spec_of(profile),
        run=run_policy_of(profile),
        environments=sorted(profile.environments),
        catalog=profile.environments,
    )


@pytest.mark.parametrize("profile_id", sorted(PROFILES))
def test_every_compiled_profile_decomposes_and_recomposes(profile_id):
    profile = PROFILES[profile_id]
    composed = _recompose(profile)
    assert composed == profile
    assert canonical_sha256(composed.to_generation_contract()) == canonical_sha256(
        profile.to_generation_contract()
    )


@pytest.mark.parametrize("profile_id", sorted(PROFILES))
def test_compiled_profiles_satisfy_the_invariants(profile_id):
    check_profile_invariants(PROFILES[profile_id])


def test_model_spec_carries_the_model_fields():
    spec = model_spec_of(PROFILES["teutonic-9b-reliquary-suite-v9-dev1"])
    assert spec == ModelSpec(
        model_id="ReliquaryForge/teutonic-i-graft-sft-cot-v2",
        model_revision="d5256c5ccc2c06d8f9bf3133b37ab2a5b95a224e",
        model_architecture=None,
        prompt_encoding="chat_template",
    )
    assert spec.proofs == ()
    # prompt_encoding is on the spec but not the identity: corpus ignores it.
    assert MODEL_IDENTITY_FIELDS == (
        "model_id", "model_revision", "model_architecture", "proofs",
    )


def test_run_policies_are_derived_from_compiled_profiles():
    assert RUN_POLICIES["dapo-v6"] == run_policy_of(
        PROFILES["qwen3-4b-base-dapo-reliquary-v1"]
    )
    assert RUN_POLICIES["suite-v9"] == run_policy_of(
        PROFILES["teutonic-9b-reliquary-suite-v9-dev1"]
    )
    assert RUN_POLICIES["episode-v7"] == run_policy_of(
        PROFILES["qwen3-4b-reliquary-episode-v7-dev1"]
    )
    assert all(isinstance(p, RunPolicy) for p in RUN_POLICIES.values())


def test_corpus_v1_policy_is_the_live_corpus_code_v1_top_level():
    contract = json.loads((FIXTURES / "corpus_code_v1_contract.json").read_text())
    assert RUN_POLICIES["corpus-v1"] == run_policy_of(profile_from_contract(contract))
    # The live task was seeded from teutonic-v9, the only profile it matches.
    matches = [
        pid for pid, p in PROFILES.items()
        if run_policy_of(p) == RUN_POLICIES["corpus-v1"]
    ]
    assert matches == ["teutonic-9b-reliquary-suite-v9-dev1"]


def test_live_corpus_code_v1_recomposes_from_its_parts():
    contract = json.loads((FIXTURES / "corpus_code_v1_contract.json").read_text())
    live = profile_from_contract(contract)
    composed = compose_profile(
        profile_id="corpus-code-v1",
        model=model_spec_of(live),
        run=RUN_POLICIES["corpus-v1"],
        environments=["reliquary_code_v1"],
        catalog=live.environments,
    )
    assert composed.to_generation_contract() == contract
    assert canonical_sha256(composed.to_generation_contract()) == canonical_sha256(contract)


def _with_math(profile, **changes):
    from dataclasses import replace

    environments = dict(profile.environments)
    environments["openmathinstruct"] = replace(environments["openmathinstruct"], **changes)
    return replace(profile, environments=environments)


def test_invariants_refuse_an_unknown_prompt_encoding():
    from dataclasses import replace

    profile = replace(PROFILES["qwen3-4b-base-dapo-reliquary-v1"], prompt_encoding="chatml")
    with pytest.raises(ValueError, match="prompt encoding"):
        check_profile_invariants(profile)


@pytest.mark.parametrize("answer_format", [None, "text", "last_json_object_v1"])
def test_invariants_refuse_a_math_answer_format_constants_would_refuse(answer_format):
    profile = _with_math(
        PROFILES["qwen3-4b-base-dapo-reliquary-v1"], answer_format=answer_format
    )
    with pytest.raises(ValueError, match="answer format"):
        check_profile_invariants(profile)


def test_invariants_refuse_a_math_cap_that_cannot_hold_the_bft_budgets():
    # v2 forces 2048 + 512; a cap of exactly that leaves no force-span room.
    profile = _with_math(PROFILES["qwen35-2b-auction-v2"], max_new_tokens=2048 + 512)
    with pytest.raises(ValueError, match="BFT"):
        check_profile_invariants(profile)
