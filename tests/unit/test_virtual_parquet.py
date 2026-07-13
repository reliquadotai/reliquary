"""Unit tests for VirtualParquetDataset — mapping, lazy fetch, LRU, modulo.

Uses real (tiny) parquet files on a local-FS shim, so the row-group mapping and
fetch logic are exercised with no network.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reliquary.environment.virtual_parquet import (
    PromptSourceUnavailable,
    VirtualParquetDataset,
)


def _make_parquet(path, values, rg_size):
    pq.write_table(
        pa.table({"v": values, "structured_cases": [str(x) for x in values]}),
        str(path),
        row_group_size=rg_size,
    )


class _LocalFS:
    """Minimal fsspec-like shim: ls returns fixed local paths, open opens them."""

    def __init__(self, files):
        self._files = [str(f) for f in files]
        self.row_group_reads = 0

    def ls(self, base, detail=False):
        return list(self._files)

    def open(self, path):
        return open(path, "rb")


def _dataset(tmp_path, cache_row_groups=64):
    # file A: 5 rows, rg_size 2 -> row-groups [0,1][2,3][4]
    # file B: 3 rows, rg_size 2 -> row-groups [5,6][7]
    _make_parquet(tmp_path / "a.parquet", [0, 1, 2, 3, 4], rg_size=2)
    _make_parquet(tmp_path / "b.parquet", [5, 6, 7], rg_size=2)
    fs = _LocalFS([tmp_path / "a.parquet", tmp_path / "b.parquet"])
    ds = VirtualParquetDataset(
        "owner/repo", "rev", columns=["v"], fs=fs, cache_row_groups=cache_row_groups,
    )
    return ds, fs


def test_len_sums_all_row_groups(tmp_path):
    ds, _ = _dataset(tmp_path)
    assert len(ds) == 8


def test_get_row_maps_within_and_across_files(tmp_path):
    ds, _ = _dataset(tmp_path)
    assert [ds.get_row(i)["v"] for i in range(8)] == [0, 1, 2, 3, 4, 5, 6, 7]


def test_get_row_modulo_wraps(tmp_path):
    ds, _ = _dataset(tmp_path)
    assert ds.get_row(8)["v"] == 0
    assert ds.get_row(8 + 5)["v"] == 5


def test_only_touched_row_groups_are_materialized(tmp_path):
    ds, _ = _dataset(tmp_path)
    ds.get_row(0)
    ds.get_row(1)  # same row-group as idx 0
    assert len(ds._cache) == 1
    ds.get_row(2)  # second row-group
    assert len(ds._cache) == 2
    # the far file-B row-groups were never fetched
    assert all(loc[0] == 0 for loc in ds._cache)


def test_lru_evicts_beyond_budget(tmp_path):
    ds, _ = _dataset(tmp_path, cache_row_groups=2)
    for i in range(8):  # touches all 5 row-groups
        ds.get_row(i)
    assert len(ds._cache) == 2  # bounded


def test_columns_subset_is_respected(tmp_path):
    ds, _ = _dataset(tmp_path)
    row = ds.get_row(3)
    assert row == {"v": 3}  # only the requested column


def test_deterministic_across_instances(tmp_path):
    ds1, _ = _dataset(tmp_path)
    ds2, _ = _dataset(tmp_path)
    assert [ds1.get_row(i)["v"] for i in range(8)] == [ds2.get_row(i)["v"] for i in range(8)]


def test_thread_safe_concurrent_get_row(tmp_path):
    """get_row is hit from the validator's submit preflight (event loop, via
    to_thread) AND its worker thread on the same shared instance. Concurrent
    access with heavy LRU eviction must never crash or return a wrong row."""
    import threading

    ds, _ = _dataset(tmp_path, cache_row_groups=1)  # tiny cache -> max eviction churn
    errors: list = []

    def hammer():
        for i in range(300):
            idx = (i * 3) % 8
            try:
                if ds.get_row(idx)["v"] != idx:
                    errors.append(("wrong-row", idx))
            except Exception as e:  # OrderedDict mutated-during-iteration, etc.
                errors.append((type(e).__name__, str(e)[:40]))

    threads = [threading.Thread(target=hammer) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


class _FailingRangeFS:
    def __init__(self, path):
        self.path = path

    def ls(self, base, detail=False):
        return [self.path]

    def open(self, path):
        raise OSError("range backend unavailable")


def test_exact_full_file_fallback_survives_range_outage(
    tmp_path, monkeypatch,
):
    import huggingface_hub

    local_path = tmp_path / "fallback.parquet"
    _make_parquet(local_path, [10, 11, 12], rg_size=2)
    remote_path = "datasets/owner/repo@rev/data/fallback.parquet"
    downloaded = False

    def _cache_lookup(*args, **kwargs):
        return str(local_path) if downloaded else None

    def _download(*args, **kwargs):
        nonlocal downloaded
        downloaded = True
        assert kwargs["repo_id"] == "owner/repo"
        assert kwargs["revision"] == "rev"
        assert kwargs["filename"] == "data/fallback.parquet"
        assert kwargs["repo_type"] == "dataset"
        return str(local_path)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _cache_lookup)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    ds = VirtualParquetDataset(
        "owner/repo",
        "rev",
        columns=["v"],
        fs=_FailingRangeFS(remote_path),
        full_file_fallback=True,
    )

    assert len(ds) == 3
    assert ds.get_row(2) == {"v": 12}
    health = ds.source_health()
    assert health["status"] == "ready"
    assert health["source_failures_total"] == 1
    assert health["full_file_fallbacks_total"] == 1
    assert health["local_full_file_hits_total"] == 1


def test_source_failure_is_typed_and_visible(tmp_path, monkeypatch):
    import huggingface_hub

    remote_path = "datasets/owner/repo@rev/data/fallback.parquet"
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None
    )

    def _download_failure(*args, **kwargs):
        raise ConnectionError("full download unavailable")

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", _download_failure
    )
    ds = VirtualParquetDataset(
        "owner/repo",
        "rev",
        fs=_FailingRangeFS(remote_path),
        full_file_fallback=True,
    )

    with pytest.raises(PromptSourceUnavailable):
        len(ds)
    health = ds.source_health()
    assert health["status"] == "degraded"
    assert health["source_failures_total"] == 2
    assert health["last_error_type"] == "ConnectionError"
