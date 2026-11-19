"""The forced draw and the LM-head precision as protocol fields (``sampling.draw``, ``sampling.lm_head_fp32``)."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from reliquary.protocol.profiles import PROFILES, SamplingProfile, profile_from_contract

GUMBEL = "teutonic-9b-reliquary-suite-v10-gumbel-dev1"
API_TRAINING = ("qwen35-4b-reliquary-suite-v10-gumbel-dev1", "qwen35-9b-reliquary-suite-v10-gumbel-dev1",
                "qwen38-27b-reliquary-suite-v10-gumbel-dev1")
V9 = "teutonic-9b-reliquary-suite-v9-dev1"
ROOT = Path(__file__).resolve().parents[2]


def test_sampling_defaults_are_the_inverse_cdf_on_the_model_head():
    sampling = SamplingProfile(rollouts=16, temperature=1.0, top_p=1.0, top_k=0, do_sample=False)
    assert sampling.draw == "inverse_cdf"
    assert sampling.lm_head_fp32 is False


@pytest.mark.parametrize("profile_id", sorted(set(PROFILES) - {GUMBEL, *API_TRAINING}))
def test_earlier_profiles_keep_the_inverse_cdf_and_omit_the_fields(profile_id):
    profile = PROFILES[profile_id]
    assert profile.sampling.draw == "inverse_cdf" and profile.sampling.lm_head_fp32 is False
    sampling = profile.to_generation_contract()["sampling"]
    assert set(sampling) == {"rollouts", "temperature", "top_p", "top_k", "do_sample"}


def test_gumbel_profile_is_v9_with_the_gumbel_draw_and_a_new_version():
    gumbel, v9 = PROFILES[GUMBEL], PROFILES[V9]
    assert gumbel.sampling.draw == "gumbel_max"
    assert gumbel.sampling.lm_head_fp32 is True                         # decided 2026-10-15
    assert gumbel.protocol_version == 10
    assert replace(gumbel.sampling, draw="inverse_cdf", lm_head_fp32=False) == v9.sampling
    long = {"reliquary_dapo_math_v1", "reliquary_hard_math_v1",                 # 64k there (2026-10-19/20 probes)
            "reliquary_competitive_code_v1"}
    added = {"envscaler_tools_v2"}                      # a replayed tool world on the lease path (2026-10-21)
    assert added.isdisjoint(v9.environments) and added <= set(gumbel.environments)
    assert {n: e for n, e in gumbel.environments.items() if n not in long | added} == {
        n: e for n, e in v9.environments.items() if n not in long}
    assert all(replace(gumbel.environments[n], max_new_tokens=v9.environments[n].max_new_tokens)
               == v9.environments[n] and gumbel.environments[n].max_new_tokens == 65536 for n in long)
    assert (gumbel.model_id, gumbel.model_revision) == (v9.model_id, v9.model_revision)
    sampling = gumbel.to_generation_contract()["sampling"]
    assert sampling["draw"] == "gumbel_max" and sampling["lm_head_fp32"] is True


def test_gumbel_profile_contract_round_trips():
    contract = PROFILES[GUMBEL].to_generation_contract()
    assert profile_from_contract(contract) == PROFILES[GUMBEL]


def test_fp32_head_round_trips_through_a_contract():
    contract = PROFILES[V9].to_generation_contract()
    contract["sampling"]["lm_head_fp32"] = True
    profile = profile_from_contract(contract)
    assert profile.sampling.lm_head_fp32 is True
    assert profile.to_generation_contract() == contract


@pytest.mark.parametrize("field,value", [("draw", "inverse_cdf"), ("lm_head_fp32", False)])
def test_a_default_value_spelled_out_is_refused(field, value):
    """One encoding per profile: the defaults are carried by absence, as every earlier contract does."""
    contract = PROFILES[V9].to_generation_contract()
    contract["sampling"][field] = value
    with pytest.raises(ValueError, match=field):
        profile_from_contract(contract)


@pytest.mark.parametrize("field,value", [("draw", "top1"), ("draw", 3), ("lm_head_fp32", 1)])
def test_malformed_values_are_refused(field, value):
    contract = PROFILES[GUMBEL].to_generation_contract()
    contract["sampling"][field] = value
    with pytest.raises(ValueError):
        profile_from_contract(contract)


@pytest.mark.parametrize("kw", [{"temperature": 0.6}, {"top_k": 20}, {"top_p": 0.95}])
def test_gumbel_profile_must_sample_the_raw_softmax(kw):
    base = dict(rollouts=16, temperature=1.0, top_p=1.0, top_k=0, do_sample=False, draw="gumbel_max")
    base.update(kw)
    with pytest.raises(ValueError, match="gumbel_max"):
        SamplingProfile(**base)


def test_gumbel_run_policy_is_declared():
    from reliquary.protocol.composition import RUN_POLICIES

    assert RUN_POLICIES["suite-v10-gumbel"].sampling.draw == "gumbel_max"
    assert RUN_POLICIES["suite-v10-gumbel"].protocol_version == 10


def _constants_under(profile_id: str) -> dict:
    env = dict(os.environ, RELIQUARY_PROTOCOL_PROFILE=profile_id)
    env.pop("RELIQUARY_TASK_CONTRACT", None)
    code = ("import json, reliquary.constants as c; print(json.dumps([c.FORCED_DRAW, c.LM_HEAD_FP32,"
            " c.FORCED_SEED_DOMAIN]))")
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True,
                         text=True, check=True)
    draw, fp32, domain = json.loads(out.stdout.strip().splitlines()[-1])
    return {"draw": draw, "fp32": fp32, "domain": domain}


def test_constants_follow_the_active_profile():
    assert _constants_under(GUMBEL) == {"draw": "gumbel_max", "fp32": True,
                                        "domain": "reliquary-forced-seed-v10"}
    assert _constants_under(V9) == {"draw": "inverse_cdf", "fp32": False,
                                    "domain": "reliquary-forced-seed-v9"}


CAPS_4B = {"reliquary_dapo_math_v1": 16384, "reliquary_science_v1": 16384, "reliquary_hard_math_v1": 16384,
           "reliquary_instruction_following_v1": 8192, "reliquary_code_v1": 8192,
           "reliquary_competitive_code_v1": 16384, "reliquary_general_v1": 16384,
           "reliquary_telecom_solo_v1": 49152, "envscaler_tools_v2": 65536}
# The larger models reason at the Teutonic v10 caps where it raised them to 64k (2026-10-19/20 budget probes).
LONG = ("reliquary_dapo_math_v1", "reliquary_hard_math_v1", "reliquary_competitive_code_v1")


def _caps(profile_id):
    return {name: env.max_new_tokens for name, env in PROFILES[profile_id].environments.items()}


def test_the_4b_caps_single_turn_envs_at_16k():
    assert _caps(API_TRAINING[0]) == CAPS_4B


@pytest.mark.parametrize("profile_id", API_TRAINING[1:])
def test_the_larger_models_take_the_teutonic_long_caps_and_the_4b_elsewhere(profile_id):
    teutonic = _caps(GUMBEL)
    assert {name for name, cap in teutonic.items() if cap == 65536 and PROFILES[GUMBEL].environments[name].episode
            is None} == set(LONG)
    assert _caps(profile_id) == {**CAPS_4B, **{name: teutonic[name] for name in LONG}}


@pytest.mark.parametrize("profile_id", API_TRAINING[1:])
def test_the_larger_models_differ_from_the_4b_by_their_model_and_long_caps_only(profile_id):
    base, profile = PROFILES[API_TRAINING[0]], PROFILES[profile_id]
    assert profile.sampling.draw == "gumbel_max" and profile.protocol_version == 10
    envs = {name: (replace(env, max_new_tokens=base.environments[name].max_new_tokens) if name in LONG else env)
            for name, env in profile.environments.items()}
    assert replace(profile, profile_id=base.profile_id, model_id=base.model_id,
                   model_revision=base.model_revision, environments=envs) == base
    assert len(profile.model_revision) == 40


def test_the_api_training_profiles_pin_their_models():
    assert {p: (PROFILES[p].model_id, PROFILES[p].model_revision) for p in API_TRAINING} == {
        "qwen35-4b-reliquary-suite-v10-gumbel-dev1": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
        "qwen35-9b-reliquary-suite-v10-gumbel-dev1": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
        "qwen38-27b-reliquary-suite-v10-gumbel-dev1": ("Qwen/Qwen3.8-27B",
                                                        "37561816e66b5b9153b1b64c10968c18c1e6f4a0"),
    }
