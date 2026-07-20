import pytest
import torch
from reliquary.protocol.crypto import (
    prf,
    r_vec_from_randomness,
    indices_from_root,
    indices_from_root_in_range,
)


class TestPrf:
    def test_deterministic(self):
        a = prf(b"test", b"data", out_bytes=32)
        b = prf(b"test", b"data", out_bytes=32)
        assert a == b

    def test_different_labels_differ(self):
        a = prf(b"label1", b"data", out_bytes=32)
        b = prf(b"label2", b"data", out_bytes=32)
        assert a != b

    def test_different_parts_differ(self):
        a = prf(b"label", b"part_a", out_bytes=32)
        b = prf(b"label", b"part_b", out_bytes=32)
        assert a != b

    def test_correct_length(self):
        for n in [0, 1, 16, 32, 64, 128, 256]:
            assert len(prf(b"test", out_bytes=n)) == n

    def test_empty_output(self):
        assert prf(b"test", out_bytes=0) == b""

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            prf(b"test", out_bytes=-1)

    def test_rejects_too_large(self):
        with pytest.raises(ValueError):
            prf(b"test", out_bytes=2**16 + 1)

    def test_rejects_non_bytes_label(self):
        with pytest.raises(TypeError):
            prf("not bytes", out_bytes=32)  # type: ignore

    def test_rejects_non_bytes_parts(self):
        with pytest.raises(TypeError):
            prf(b"ok", "not bytes", out_bytes=32)  # type: ignore

    def test_multiple_parts(self):
        result = prf(b"label", b"a", b"b", b"c", out_bytes=32)
        assert len(result) == 32


class TestRVecFromRandomness:
    def test_shape(self):
        vec = r_vec_from_randomness("abcdef1234567890" * 4, 4096)
        assert vec.shape == (4096,)
        assert vec.dtype == torch.int32

    def test_deterministic(self):
        a = r_vec_from_randomness("aabb", 128)
        b = r_vec_from_randomness("aabb", 128)
        assert torch.equal(a, b)

    def test_different_randomness_differs(self):
        a = r_vec_from_randomness("aabb", 128)
        b = r_vec_from_randomness("ccdd", 128)
        assert not torch.equal(a, b)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            r_vec_from_randomness("", 128)

    def test_rejects_invalid_d_model(self):
        with pytest.raises(ValueError):
            r_vec_from_randomness("aabb", 0)

    def test_rejects_too_large_d_model(self):
        with pytest.raises(ValueError):
            r_vec_from_randomness("aabb", 100001)

    def test_handles_0x_prefix(self):
        a = r_vec_from_randomness("0xaabb", 64)
        b = r_vec_from_randomness("aabb", 64)
        assert torch.equal(a, b)

    def test_handles_odd_length_hex(self):
        # Should pad with leading zero
        vec = r_vec_from_randomness("abc", 64)
        assert vec.shape == (64,)


class TestIndicesFromRoot:
    def test_correct_count(self):
        tokens = list(range(100))
        idxs = indices_from_root(tokens, "abcd1234", 100, 10)
        assert len(idxs) == 10

    def test_sorted(self):
        tokens = list(range(200))
        idxs = indices_from_root(tokens, "abcd1234", 200, 32)
        assert idxs == sorted(idxs)

    def test_deterministic(self):
        tokens = list(range(100))
        a = indices_from_root(tokens, "abcd1234", 100, 10)
        b = indices_from_root(tokens, "abcd1234", 100, 10)
        assert a == b

    def test_within_range(self):
        tokens = list(range(50))
        idxs = indices_from_root(tokens, "ffff", 50, 10)
        assert all(0 <= i < 50 for i in idxs)

    def test_unique_indices(self):
        tokens = list(range(100))
        idxs = indices_from_root(tokens, "aabb", 100, 20)
        assert len(idxs) == len(set(idxs))

    def test_rejects_k_gt_seq_len(self):
        with pytest.raises(ValueError):
            indices_from_root([1, 2, 3], "abcd", 3, 5)

    def test_rejects_k_zero(self):
        with pytest.raises(ValueError):
            indices_from_root([1, 2], "abcd", 2, 0)

    def test_rejects_empty_tokens(self):
        with pytest.raises(ValueError):
            indices_from_root([], "abcd", 10, 5)

    def test_large_k_relative_to_seq(self):
        """When k > 10% of seq_len, uses shuffle approach."""
        tokens = list(range(20))
        idxs = indices_from_root(tokens, "aabb", 20, 15)
        assert len(idxs) == 15
        assert idxs == sorted(idxs)


class TestIndicesFromRootInRange:
    def test_within_range(self):
        tokens = list(range(100))
        idxs = indices_from_root_in_range(tokens, "aabb", 10, 50, 5)
        assert all(10 <= i < 50 for i in idxs)

    def test_empty_range(self):
        tokens = list(range(10))
        assert indices_from_root_in_range(tokens, "aabb", 5, 5, 3) == []

    def test_rejects_negative_start(self):
        with pytest.raises(ValueError):
            indices_from_root_in_range([1], "aa", -1, 5, 2)
