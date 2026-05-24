"""Merkle root determinism — regression test for repr() → JSON fix."""

from dataclasses import dataclass

from reliquary.miner.engine import _compute_merkle_root


@dataclass
class _R:
    tokens: list
    reward: float
    commit: dict


def test_merkle_root_stable_across_dict_order():
    """Two rollout batches identical except for commit dict insertion order
    must produce the same Merkle root."""
    r1 = [_R(tokens=[1, 2], reward=1.0,
            commit={"proof_version": "v6", "tokens": [1, 2]})]
    r2 = [_R(tokens=[1, 2], reward=1.0,
            commit={"tokens": [1, 2], "proof_version": "v6"})]  # different order
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
