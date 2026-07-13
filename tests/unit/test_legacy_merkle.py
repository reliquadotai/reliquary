from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from reliquary.miner.engine import _compute_merkle_root
from reliquary.protocol.legacy_merkle import (
    compute_legacy_rollouts_merkle_root,
    legacy_submission_merkle_matches,
)


def _rollout(**changes):
    values = {
        "tokens": [1, 2, 3],
        "reward": 0.5,
        "commit": {
            "tokens": [1, 2, 3],
            "rollout": {"prompt_length": 1, "completion_length": 2},
            "z": 2,
            "a": [True, None],
        },
        "env_name": "openmathinstruct",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_legacy_helper_matches_current_miner_implementation():
    rollouts = [_rollout(), _rollout(tokens=[1, 2, 4])]

    assert compute_legacy_rollouts_merkle_root(rollouts) == (
        _compute_merkle_root(rollouts)
    )


def test_legacy_root_golden_vector_is_frozen():
    assert compute_legacy_rollouts_merkle_root([_rollout()]) == (
        "3cb66970da317b8a6b65779b136fbd15"
        "f57ca7d99e739958900e35d2ecff7dbe"
    )


def test_legacy_root_binds_order_tokens_reward_and_complete_commit():
    baseline = compute_legacy_rollouts_merkle_root(
        [_rollout(), _rollout(tokens=[4, 5, 6])]
    )

    assert compute_legacy_rollouts_merkle_root(
        [_rollout(tokens=[4, 5, 6]), _rollout()]
    ) != baseline
    assert compute_legacy_rollouts_merkle_root(
        [_rollout(), _rollout(tokens=[4, 5, 7])]
    ) != baseline
    assert compute_legacy_rollouts_merkle_root(
        [_rollout(), _rollout(tokens=[4, 5, 6], reward=0.75)]
    ) != baseline
    changed_commit = deepcopy(_rollout(tokens=[4, 5, 6]).commit)
    changed_commit["rollout"]["completion_length"] = 3
    assert compute_legacy_rollouts_merkle_root(
        [_rollout(), _rollout(tokens=[4, 5, 6], commit=changed_commit)]
    ) != baseline


def test_legacy_root_intentionally_does_not_bind_environment():
    math = _rollout(env_name="openmathinstruct")
    code = _rollout(env_name="opencodeinstruct")

    assert compute_legacy_rollouts_merkle_root([math]) == (
        compute_legacy_rollouts_merkle_root([code])
    )


def test_legacy_submission_match_is_case_insensitive():
    rollouts = [_rollout()]
    root = compute_legacy_rollouts_merkle_root(rollouts)
    request = SimpleNamespace(rollouts=rollouts, merkle_root=root.upper())

    assert legacy_submission_merkle_matches(request) == (True, root)


def test_legacy_submission_mismatch_returns_computed_root():
    rollouts = [_rollout()]
    request = SimpleNamespace(rollouts=rollouts, merkle_root="00" * 32)

    matches, computed = legacy_submission_merkle_matches(request)

    assert matches is False
    assert computed == compute_legacy_rollouts_merkle_root(rollouts)


def test_legacy_root_rejects_empty_groups():
    with pytest.raises(ValueError, match="empty group"):
        compute_legacy_rollouts_merkle_root([])
