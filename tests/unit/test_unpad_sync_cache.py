"""Unpad sync-cache — bit-identical memoization of flash-attention unpad
metadata (the ~1200 hidden nonzero-syncs per train_step).

The cache must be pure memoization: same tensors out, invalidated on
in-place mutation, keyed per thread. Numerical equality is exact — any
tolerance here would mean the patch is not a memoization and must not ship.
"""
import threading

import pytest
import torch

pytest.importorskip("transformers")
import transformers.modeling_flash_attention_utils as mfa

import reliquary.validator.training as training


def _mask(rows):
    return torch.tensor(rows, dtype=torch.int64)


@pytest.fixture()
def cached_fn():
    fn = mfa._get_unpad_data
    if not getattr(fn, "_reliquary_unpad_cache", False):
        assert training._install_unpad_sync_cache()
        fn = mfa._get_unpad_data
    return fn


def test_installed_at_import(cached_fn):
    assert getattr(cached_fn, "_reliquary_unpad_cache", False)


def test_bit_identical_to_original(cached_fn):
    original = cached_fn._reliquary_original
    for rows in (
        [[1, 1, 1, 0], [1, 1, 0, 0]],
        [[1] * 7],
        [[1, 0, 0], [1, 1, 1], [1, 1, 0]],
    ):
        mask = _mask(rows)
        got_idx, got_cu, got_max = cached_fn(mask)
        exp_idx, exp_cu, exp_max = original(mask)
        assert torch.equal(got_idx, exp_idx)
        assert torch.equal(got_cu, exp_cu)
        # int materialization is part of the contract (docstring says int);
        # the VALUE must match the original's exactly.
        assert isinstance(got_max, int)
        assert got_max == int(exp_max)


def test_memoizes_same_tensor(cached_fn):
    mask = _mask([[1, 1, 0], [1, 0, 0]])
    first = cached_fn(mask)
    second = cached_fn(mask)
    # same result OBJECT: the original ran once (identity proves the hit)
    assert second is first


def test_inplace_mutation_invalidates(cached_fn):
    mask = _mask([[1, 1, 1, 1], [1, 1, 0, 0]])
    first = cached_fn(mask)
    mask[1, 2] = 1  # in-place -> _version bump
    second = cached_fn(mask)
    assert second is not first
    exp_idx, exp_cu, exp_max = cached_fn._reliquary_original(mask)
    assert torch.equal(second[0], exp_idx)
    assert torch.equal(second[1], exp_cu)
    assert second[2] == int(exp_max)


def test_distinct_tensor_same_content_misses(cached_fn):
    a = _mask([[1, 1, 0]])
    b = _mask([[1, 1, 0]])
    ra = cached_fn(a)
    rb = cached_fn(b)
    assert ra is not rb  # identity-keyed, not content-keyed
    assert torch.equal(ra[0], rb[0]) and torch.equal(ra[1], rb[1])
    assert ra[2] == rb[2]


def test_thread_local_slots(cached_fn):
    mask = _mask([[1, 1, 0], [1, 1, 1]])
    main_result = cached_fn(mask)
    other: dict = {}

    def worker():
        other["result"] = cached_fn(mask)

    t = threading.Thread(target=worker)
    t.start(); t.join()
    # other thread has its own slot: recomputed (different object), equal data
    assert other["result"] is not main_result
    assert torch.equal(other["result"][0], main_result[0])
    # and the main slot survived the other thread's traffic
    assert cached_fn(mask) is main_result


def test_kill_switch_blocks_install(monkeypatch):
    monkeypatch.setattr(training, "UNPAD_SYNC_CACHE", False)
    assert training._install_unpad_sync_cache() is False


def test_reinstall_is_idempotent(cached_fn):
    before = mfa._get_unpad_data
    assert training._install_unpad_sync_cache() is True
    assert mfa._get_unpad_data is before  # no double wrap


def test_bool_view_of_base_invalidates(cached_fn):
    """Production always caches a bool VIEW (masking_utils slices then casts);
    the design rests on views sharing the base's version counter."""
    base = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    view = base[:, -3:]
    first = cached_fn(view)
    base[1, 3] = True  # mutate the BASE
    second = cached_fn(view)
    assert second is not first
    exp = cached_fn._reliquary_original(view)
    assert torch.equal(second[0], exp[0])
    assert torch.equal(second[1], exp[1])
    assert second[2] == int(exp[2])


def test_resize_invalidates(cached_fn):
    mask = torch.ones(2, 3, dtype=torch.int64)
    first = cached_fn(mask)
    mask.resize_(1, 3)
    second = cached_fn(mask)
    assert second is not first
    assert second[2] == 3


def test_upad_input_end_to_end_stock_vs_cached(cached_fn):
    """The consumer path: _upad_input outputs must be equal stock vs cached
    (only max_seqlen's TYPE may differ: int vs 0-dim tensor)."""
    torch.manual_seed(0)
    B, T, H, D = 3, 7, 2, 4
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]]
    )
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)
    v = torch.randn(B, T, H, D)
    got = mfa._upad_input(q, k, v, mask, T, mfa._unpad_input)
    orig_fn, mfa._get_unpad_data = mfa._get_unpad_data, cached_fn._reliquary_original
    try:
        exp = mfa._upad_input(q, k, v, mask, T, mfa._unpad_input)
    finally:
        mfa._get_unpad_data = orig_fn
    for g, e in zip(got[:4], exp[:4]):
        assert torch.equal(g, e)
    (g_cu_q, g_cu_k), (e_cu_q, e_cu_k) = got[4], exp[4]
    assert torch.equal(g_cu_q, e_cu_q) and torch.equal(g_cu_k, e_cu_k)
    (g_mq, g_mk), (e_mq, e_mk) = got[5], exp[5]
    assert int(g_mq) == int(e_mq) and int(g_mk) == int(e_mk)


def test_clear_releases_slot(cached_fn):
    mask = _mask([[1, 1, 0]])
    first = cached_fn(mask)
    training._clear_unpad_sync_cache()
    second = cached_fn(mask)
    assert second is not first  # slot was dropped -> recompute


def test_kill_switch_subprocess_leaves_stock():
    """With the env flag off, a fresh process must keep the stock function."""
    import subprocess, sys
    code = (
        "import reliquary.validator.training as t; "
        "import transformers.modeling_flash_attention_utils as m; "
        "assert not getattr(m._get_unpad_data, '_reliquary_unpad_cache', False); "
        "print('stock ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={"PATH": "/usr/bin:/bin", "RELIQUARY_UNPAD_SYNC_CACHE": "0",
             "PYTHONPATH": "."},
        capture_output=True, text=True, cwd=".",
    )
    assert "stock ok" in out.stdout, out.stderr[-500:]
