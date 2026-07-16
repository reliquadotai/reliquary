"""Sanity checks on v2 constants — catches accidental edits."""

import importlib
import os

from reliquary import constants as C


def test_v2_sigma_bounds():
    assert C.SIGMA_MIN == 0.43
    assert C.BOOTSTRAP_SIGMA_MIN == 0.33
    assert C.BOOTSTRAP_SIGMA_MIN < C.SIGMA_MIN
    assert C.MAX_TRUNCATED_PER_SUBMISSION == 1
    assert C.BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION == 1


def test_bft_constants_present_and_within_cap():
    assert isinstance(C.BFT_ENABLED, bool)
    assert C.BFT_THINKING_BUDGET > 0
    assert C.BFT_ANSWER_BUDGET > 0
    # Forced thinking + forced answer must fit under the hard generation cap.
    assert C.BFT_THINKING_BUDGET + C.BFT_ANSWER_BUDGET <= C.MAX_NEW_TOKENS_PROTOCOL_CAP
    assert C.BFT_FORCE_TEMPLATE.startswith("</think>")
    assert "\\boxed{" in C.BFT_FORCE_TEMPLATE


def test_v2_group_sizes():
    assert C.M_ROLLOUTS == 8
    assert C.B_BATCH == 8
    assert C.MAX_POST_TRIGGER_PROOF_CANDIDATES == 8
    assert C.MAX_SEAL_QUEUE_DRAIN_SECONDS == 60.0
    assert C.SPARSE_VALID_IDLE_SEAL_SECONDS == 300.0
    assert C.SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS == 4
    assert C.SPARSE_VALID_MAX_WINDOW_SECONDS == 900.0


def test_v2_temperature_fixed_nonzero():
    assert 0.5 < C.T_PROTO <= 1.0


def test_v2_cooldown_values():
    assert C.BATCH_PROMPT_COOLDOWN_WINDOWS == 1_000_000
    assert C.BOOTSTRAP_WINDOWS == 100


def test_hash_dedup_retention_decoupled_from_cooldown():
    """Hash retention is independent of prompt cooldown.

    v2.3 + 1M cooldown: BATCH_PROMPT_COOLDOWN_WINDOWS now exceeds
    HASH_DEDUP_RETENTION_WINDOWS, so a prompt is locked by cooldown long
    before the hash horizon would catch a duplicate token sequence. The
    hash dedup remains in place as a defense-in-depth (e.g. for cases
    where the cooldown map is partially rebuilt after a long restart
    gap) — its purpose shifted from "cooldown-extender" to
    "post-cooldown safety net". The two values just need to be sensible
    and explicit; no ordering invariant.
    """
    assert C.HASH_DEDUP_RETENTION_WINDOWS == 300
    assert C.BATCH_PROMPT_COOLDOWN_WINDOWS == 1_000_000


def test_cooldown_rebuild_lookback_bounded():
    """The gap-replay / no-snapshot fallback scan must stay small enough for a
    fast startup, comfortably exceed the snapshot cadence (so a normal gap is
    always covered), and stay below the cooldown horizon."""
    assert C.COOLDOWN_REBUILD_LOOKBACK == 2000
    assert C.COOLDOWN_REBUILD_LOOKBACK > C.COOLDOWN_SNAPSHOT_INTERVAL_WINDOWS
    assert C.COOLDOWN_REBUILD_LOOKBACK < C.BATCH_PROMPT_COOLDOWN_WINDOWS


def test_startup_rebuild_horizon_env_overrides():
    """Operators can widen startup replay horizons without a code deploy."""
    prior_cooldown = os.environ.get("COOLDOWN_REBUILD_LOOKBACK")
    prior_hash = os.environ.get("HASH_DEDUP_RETENTION_WINDOWS")
    os.environ["COOLDOWN_REBUILD_LOOKBACK"] = "720"
    os.environ["HASH_DEDUP_RETENTION_WINDOWS"] = "1440"
    try:
        importlib.reload(C)
        assert C.COOLDOWN_REBUILD_LOOKBACK == 720
        assert C.HASH_DEDUP_RETENTION_WINDOWS == 1440
    finally:
        if prior_cooldown is None:
            os.environ.pop("COOLDOWN_REBUILD_LOOKBACK", None)
        else:
            os.environ["COOLDOWN_REBUILD_LOOKBACK"] = prior_cooldown
        if prior_hash is None:
            os.environ.pop("HASH_DEDUP_RETENTION_WINDOWS", None)
        else:
            os.environ["HASH_DEDUP_RETENTION_WINDOWS"] = prior_hash
        importlib.reload(C)


def test_policy_ratio_skip_threshold_env_override():
    env_name = "RELIQUARY_PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD"
    prior = os.environ.get(env_name)
    os.environ[env_name] = "0.05"
    try:
        importlib.reload(C)
        assert C.PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD == 0.05
    finally:
        if prior is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = prior
        importlib.reload(C)


def test_v2_bootstrap_sigma_lower_than_steady():
    # Bootstrap accepts groups with lower σ (σ ≥ 0.33) vs steady (σ ≥ 0.43)
    assert C.BOOTSTRAP_SIGMA_MIN < C.SIGMA_MIN


def test_wandb_constants_present():
    assert C.WANDB_PROJECT == "reliquary-validator"
    assert C.WANDB_TRAINING_VERSION == "v1"


def test_min_eos_probability_constant_present():
    from reliquary.constants import MIN_EOS_PROBABILITY
    assert 0.0 < MIN_EOS_PROBABILITY < 1.0
    assert MIN_EOS_PROBABILITY == 0.01
