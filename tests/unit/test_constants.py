"""Sanity checks on v2 constants — catches accidental edits."""

import importlib
import os

from reliquary import constants as C


def test_bft_constants_present_and_within_cap():
    assert isinstance(C.BFT_ENABLED, bool)
    assert C.BFT_THINKING_BUDGET > 0
    assert C.BFT_ANSWER_BUDGET > 0
    # Forced thinking + forced answer must fit under the hard generation cap.
    assert C.BFT_THINKING_BUDGET + C.BFT_ANSWER_BUDGET <= C.MAX_NEW_TOKENS_PROTOCOL_CAP
    assert C.BFT_FORCE_TEMPLATE.startswith("</think>")
    assert "\\boxed{" in C.BFT_FORCE_TEMPLATE


def test_v2_temperature_fixed_nonzero():
    assert 0.5 < C.T_PROTO <= 1.0


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


def test_checkpoint_publish_interval_defaults_and_env_override():
    env_name = "RELIQUARY_CHECKPOINT_PUBLISH_INTERVAL_WINDOWS"
    prior = os.environ.pop(env_name, None)
    try:
        importlib.reload(C)
        assert C.CHECKPOINT_PUBLISH_INTERVAL_WINDOWS == 4
        os.environ[env_name] = "7"
        importlib.reload(C)
        assert C.CHECKPOINT_PUBLISH_INTERVAL_WINDOWS == 7
    finally:
        if prior is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = prior
        importlib.reload(C)


def test_shaping_defaults_disabled():
    env_name = "RELIQUARY_SHAPE_PENALTY"
    prior = os.environ.pop(env_name, None)
    try:
        importlib.reload(C)
        assert C.SHAPE_PENALTY == 0.0
    finally:
        if prior is not None:
            os.environ[env_name] = prior
        importlib.reload(C)


def test_wandb_training_version_env_override():
    env_name = "RELIQUARY_WANDB_VERSION"
    prior = os.environ.get(env_name)
    os.environ[env_name] = "training-recovery-test"
    try:
        importlib.reload(C)
        assert C.WANDB_TRAINING_VERSION == "training-recovery-test"
    finally:
        if prior is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = prior
        importlib.reload(C)


def test_wandb_constants_present():
    assert C.WANDB_PROJECT == "reliquary-validator"
    assert C.WANDB_TRAINING_VERSION == "v1"


def test_recompute_pi_old_defaults_on():
    """pi_old comes from the validator's own verify_model forward by default.

    Free with the rolling KL reference (the forward aliases), and it keeps
    miner-fidelity noise out of the PPO ratio. Operators can still disable
    via RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY=false.
    """
    from reliquary.constants import RECOMPUTE_PI_OLD_FROM_VERIFY

    assert RECOMPUTE_PI_OLD_FROM_VERIFY is True
