"""Merkle root determinism — regression test for repr() → JSON fix."""

from dataclasses import dataclass

from reliquary.miner.engine import _compute_merkle_root
from reliquary.protocol.merkle import (
    compute_rollouts_merkle_root,
    compute_rollouts_selection_digest,
    submission_merkle_matches,
)


@dataclass
class _R:
    tokens: list
    reward: float
    commit: dict


def test_merkle_root_stable_across_dict_order():
    """Two rollout batches identical except for commit dict insertion order
    must produce the same Merkle root."""
    r1 = [_R(tokens=[1, 2], reward=1.0,
            commit={"proof_version": "v7", "tokens": [1, 2]})]
    r2 = [_R(tokens=[1, 2], reward=1.0,
            commit={"tokens": [1, 2], "proof_version": "v7"})]  # different order
    assert _compute_merkle_root(r1) == _compute_merkle_root(r2)


def test_merkle_root_64_hex_chars():
    r = [_R(tokens=[1], reward=0.0, commit={}) for _ in range(8)]
    root = _compute_merkle_root(r)
    assert len(root) == 64
    assert all(c in "0123456789abcdef" for c in root)


def test_merkle_root_changes_when_tokens_differ():
    r1 = [_R(tokens=[1, 2], reward=1.0, commit={"x": 1})]
    r2 = [_R(tokens=[1, 3], reward=1.0, commit={"x": 1})]  # different tokens
    assert _compute_merkle_root(r1) != _compute_merkle_root(r2)


def test_merkle_root_changes_when_reward_differs():
    r1 = [_R(tokens=[1], reward=1.0, commit={})]
    r2 = [_R(tokens=[1], reward=0.0, commit={})]
    assert _compute_merkle_root(r1) != _compute_merkle_root(r2)


def test_miner_wrapper_uses_protocol_canonical_root():
    rollouts = [
        _R(tokens=[1, 2], reward=1.0, commit={"tokens": [1, 2]}),
        _R(tokens=[3, 4], reward=0.0, commit={"tokens": [3, 4]}),
    ]
    assert _compute_merkle_root(rollouts) == compute_rollouts_merkle_root(rollouts)


def test_submission_root_match_detects_claim_mutation():
    from types import SimpleNamespace

    rollouts = [_R(tokens=[1], reward=1.0, commit={"tokens": [1]})]
    request = SimpleNamespace(
        rollouts=rollouts,
        merkle_root=compute_rollouts_merkle_root(rollouts),
    )
    assert submission_merkle_matches(request) is True
    request.merkle_root = "00" * 32
    assert submission_merkle_matches(request) is False


def test_merkle_root_binds_environment_name():
    rollout = {
        "tokens": [1],
        "reward": 1.0,
        "commit": {"tokens": [1]},
        "env_name": "openmathinstruct",
    }
    math_root = compute_rollouts_merkle_root([rollout])
    rollout["env_name"] = "opencodeinstruct"
    assert compute_rollouts_merkle_root([rollout]) != math_root


def test_merkle_root_has_stable_domain_separated_vector():
    rollout = {
        "tokens": [1, 2],
        "reward": 1.0,
        "commit": {"tokens": [1, 2], "proof_version": "v7"},
        "env_name": "openmathinstruct",
    }
    assert compute_rollouts_merkle_root([rollout]) == (
        "3cc22f466eddcc9848d80a57635a05d0ec7827e15f9675fd32908bf54816b4ad"
    )


def test_selection_digest_ignores_non_generation_metadata():
    rollout = {
        "tokens": [1, 2],
        "reward": 0.0,
        "commit": {"tokens": [1, 2], "advantage": 0.0},
        "env_name": "opencodeinstruct",
    }
    before_root = compute_rollouts_merkle_root([rollout])
    before_selection = compute_rollouts_selection_digest([rollout])

    rollout["reward"] = 1.0
    rollout["commit"]["advantage"] = 999.0

    assert compute_rollouts_merkle_root([rollout]) != before_root
    assert compute_rollouts_selection_digest([rollout]) == before_selection


def test_selection_digest_binds_tokens_order_and_environment():
    first = {
        "tokens": [1, 2],
        "reward": 0.0,
        "commit": {},
        "env_name": "openmathinstruct",
    }
    baseline = compute_rollouts_selection_digest([first])
    first["tokens"] = [2, 1]
    assert compute_rollouts_selection_digest([first]) != baseline
    first["tokens"] = [1, 2]
    first["env_name"] = "opencodeinstruct"
    assert compute_rollouts_selection_digest([first]) != baseline
