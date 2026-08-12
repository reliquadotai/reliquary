"""Cross-cutting invariants of the v4 profile, checked in one live process.

test_protocol_profiles.py locks each knob's *value*. This file checks the
things no single knob can express: that reliquary.constants imports at all
under v4 (it carries import-time asserts, e.g. the math cap must exceed the BFT
budgets), that BFT is off across every one of its knobs rather than just the
headline flag, and — the one that spans modules — that warp() is the identity
at *the profile's own* sampling values.

That last one is the premise clip-higher rests on. Reading the values from the
profile instead of hardcoding them is the point: if someone edits the sampling
profile, this fails, whereas a test with literals would keep passing while the
PPO ratio silently drifted out of the space the samples came from.
"""
import json
import os
import subprocess
import sys

_SCRIPT = """
import json
import torch

from reliquary import constants as c
from reliquary.environment.forced_sampling import warp

logits = torch.randn(512, generator=torch.Generator().manual_seed(0)) * 4.0
warped = warp(logits, t=c.T_PROTO, top_k=c.TOP_K_PROTO, top_p=c.TOP_P_PROTO)
plain = torch.softmax(logits.float(), dim=-1)

print(json.dumps({
    "profile_id": c.ACTIVE_PROTOCOL_PROFILE.profile_id,
    "version": c.PROTOCOL_VERSION,
    "bft_enabled": c.BFT_ENABLED,
    "bft_thinking": c.BFT_THINKING_BUDGET,
    "bft_answer": c.BFT_ANSWER_BUDGET,
    "mask_math_forced": c.MASK_MATH_FORCED_FROM_LOSS,
    "mask_code_truncated": c.MASK_CODE_TRUNCATED_FROM_LOSS,
    "mask_budget_ended": c.MASK_BUDGET_ENDED_FROM_LOSS,
    "kl_beta": c.KL_BETA,
    "clip_low": c.PPO_CLIP_EPSILON_LOW,
    "clip_high": c.PPO_CLIP_EPSILON_HIGH,
    "overlong_factor": c.OVERLONG_PENALTY_FACTOR,
    "overlong_cache": c.OVERLONG_PENALTY_CACHE_TOKENS,
    "raw_prompts": c.RAW_COMPLETION_PROMPTS,
    "omi_train_only": c.OMI_TRAIN_SHARDS_ONLY,
    "opt_8bit": c.OPTIMIZER_STATE_8BIT,
    "seed_floor": c.FORCED_SEED_CONSISTENCY_FLOOR,
    "seed_rollout_floor": c.FORCED_SEED_ROLLOUT_FLOOR,
    "min_eos_prob": c.MIN_EOS_PROBABILITY,
    "sampling_median_max": c.SAMPLING_MEDIAN_LOW_MAX,
    "sampling_q10_max": c.SAMPLING_LOW_Q10_MAX,
    # Cross-module: is the sampling the profile declares actually the identity?
    "warp_is_identity": bool(torch.allclose(warped, plain, atol=1e-6)),
    "warp_full_support": bool(torch.all(warped > 0)),
}))
"""


def _boot(profile_id):
    env = dict(os.environ)
    env["RELIQUARY_PROTOCOL_PROFILE"] = profile_id
    done = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        check=True, capture_output=True, text=True, env=env,
    )
    return json.loads(done.stdout)


def test_v4_profile_is_internally_coherent():
    got = _boot("qwen3-4b-base-dapo-v4")

    assert got == {
        "profile_id": "qwen3-4b-base-dapo-v4",
        "version": 4,
        # BFT off across every knob, not just the headline flag.
        "bft_enabled": False,
        "bft_thinking": 0,
        "bft_answer": 0,
        # DAPO shapes overlong rollouts, it does not mask them out.
        "mask_math_forced": False,
        "mask_code_truncated": False,
        "mask_budget_ended": False,
        "overlong_factor": 0.5,
        "overlong_cache": 4096,
        # No KL, asymmetric clip, full-precision optimizer state.
        "kl_beta": 0.0,
        "clip_low": 0.2,
        "clip_high": 0.28,
        "opt_8bit": False,
        # Behavioural-validator thresholds recalibrated for the full-support
        # envelope (H100, 2026-08-12, 40 honest 16k-cap OMI rollouts).
        "seed_floor": 0.70,
        "seed_rollout_floor": 0.65,
        "min_eos_prob": 0.001,
        "sampling_median_max": 0.05,
        "sampling_q10_max": 0.0002,
        "raw_prompts": True,
        "omi_train_only": True,
        # The premise clip-higher rests on.
        "warp_is_identity": True,
        "warp_full_support": True,
    }


def test_v3_profile_keeps_its_own_coherent_shape():
    """The live deployment must be untouched by everything above."""
    got = _boot("qwen35-4b-auction-v3")

    assert got["bft_enabled"] is True
    assert got["kl_beta"] == 0.01
    assert got["clip_high"] == 0.2
    assert got["raw_prompts"] is False
    assert got["omi_train_only"] is False
    assert got["opt_8bit"] is True
    # v3 behavioural-validator thresholds stay at their calibrated values.
    assert got["seed_floor"] == 0.80
    assert got["seed_rollout_floor"] == 0.75
    assert got["min_eos_prob"] == 0.01
    assert got["sampling_median_max"] == 0.30
    assert got["sampling_q10_max"] == 0.025
    # v3 samples at T=0.6 / top_k=20, so warp() truncates and rescales.
    assert got["warp_is_identity"] is False
    assert got["warp_full_support"] is False
