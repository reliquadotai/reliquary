"""Sanity checks on v2 constants — catches accidental edits."""

from reliquary import constants as C


def test_v2_sigma_bounds():
    assert C.SIGMA_MIN == 0.43
    assert C.BOOTSTRAP_SIGMA_MIN == 0.33
    assert C.BOOTSTRAP_SIGMA_MIN < C.SIGMA_MIN


def test_v2_group_sizes():
    assert C.M_ROLLOUTS == 8
    assert C.B_BATCH == 8


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
    assert C.HASH_DEDUP_RETENTION_WINDOWS == 10000
    assert C.BATCH_PROMPT_COOLDOWN_WINDOWS == 1_000_000


def test_cooldown_rebuild_lookback_bounded():
    """The R2 rebuild cap must stay small enough for a startup scan to
    complete in seconds even when BATCH_PROMPT_COOLDOWN_WINDOWS is set to
    an astronomical value for one-shot semantics."""
    assert C.COOLDOWN_REBUILD_LOOKBACK == 10_000
    assert C.COOLDOWN_REBUILD_LOOKBACK < C.BATCH_PROMPT_COOLDOWN_WINDOWS


def test_v2_bootstrap_sigma_lower_than_steady():
    # Bootstrap accepts groups with lower σ (σ ≥ 0.33) vs steady (σ ≥ 0.43)
    assert C.BOOTSTRAP_SIGMA_MIN < C.SIGMA_MIN


def test_wandb_constants_present():
    assert C.WANDB_PROJECT == "reliquary-validator"
    assert C.WANDB_TRAINING_VERSION == "v1"


def test_min_eos_probability_constant_present():
    from reliquary.constants import MIN_EOS_PROBABILITY
    assert 0.0 < MIN_EOS_PROBABILITY < 1.0
    assert MIN_EOS_PROBABILITY == 0.02
