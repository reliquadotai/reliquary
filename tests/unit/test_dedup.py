"""Unit tests for hash-dedup primitives."""

from reliquary.protocol.submission import RejectReason


def test_hash_duplicate_reject_reason_exists():
    assert RejectReason.HASH_DUPLICATE.value == "hash_duplicate"


def test_compute_rollout_hash_returns_32_bytes():
    from reliquary.validator.dedup import compute_rollout_hash
    h = compute_rollout_hash([1, 2, 3, 4])
    assert isinstance(h, bytes)
    assert len(h) == 32


def test_compute_rollout_hash_deterministic():
    from reliquary.validator.dedup import compute_rollout_hash
    h1 = compute_rollout_hash([100, 200, 300, 400, 500])
    h2 = compute_rollout_hash([100, 200, 300, 400, 500])
    assert h1 == h2


def test_compute_rollout_hash_differs_on_single_token_change():
    from reliquary.validator.dedup import compute_rollout_hash
    a = compute_rollout_hash([10, 20, 30, 40, 50])
    b = compute_rollout_hash([10, 20, 31, 40, 50])  # single token diff
    assert a != b


def test_compute_rollout_hash_rejects_negative_tokens():
    import pytest
    from reliquary.validator.dedup import compute_rollout_hash
    with pytest.raises(ValueError):
        compute_rollout_hash([1, -2, 3])
