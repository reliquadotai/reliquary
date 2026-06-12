"""Lazy, deterministic parquet dataset over a pinned HF repo.

Reads only the row-groups actually touched (HTTP range requests), so a
multi-GB dataset costs no bulk download and ~no RAM — just a small LRU of
fetched row-groups. Both validator and miner pin the same ``(repo, revision)``
so ``len()`` and ``get_row()`` are byte-identical (required for GRAIL token
binding and the per-window prompt range). Pairs naturally with the prompt
range: a window's contiguous slice maps to a handful of adjacent row-groups.

``fs`` is injectable so unit tests exercise the mapping/LRU with a fake
filesystem and no network.
"""
from __future__ import annotations

import bisect
import threading
from collections import OrderedDict
from typing import Any, Optional


class VirtualParquetDataset:
    """Index space of a pinned parquet dataset, materialized row-group by
    row-group on demand. Supports ``len()`` and ``ds[idx]`` (modulo-wrapped),
    matching the slice of the HF ``datasets`` API the environments touch.
    """

    def __init__(
        self,
        repo: str,
        revision: str,
        *,
        columns: Optional[list[str]] = None,
        data_dir: str = "data",
        cache_row_groups: int = 64,
        fs: Any = None,
    ) -> None:
        self._repo = repo
        self._revision = revision
        self._columns = columns
        self._data_dir = data_dir
        self._cache_cap = cache_row_groups
        self._fs = fs  # injectable for tests; HfFileSystem when None
        self._files: Optional[list[str]] = None
        self._rg_start: Optional[list[int]] = None  # global start idx per row-group
        self._rg_loc: Optional[list[tuple[int, int]]] = None  # (file_idx, rg_idx)
        self._total: Optional[int] = None
        self._cache: "OrderedDict[tuple[int, int], list[dict]]" = OrderedDict()
        # Cache open ParquetFile handles so adjacent row-group reads from the
        # same shard don't re-fetch the footer each time (bounded LRU).
        self._pf: "OrderedDict[int, Any]" = OrderedDict()
        # get_row is called from the validator's submit-worker thread AND (via
        # asyncio.to_thread) the submit preflight, on the same shared instance,
        # so access must be thread-safe. _lock guards the LRU cache dict (held
        # only briefly, never across I/O); _io_lock serializes the manifest
        # build and the row-group reads + the _pf handle cache, which share a
        # per-file handle that is not concurrency-safe.
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()

    # -- manifest (footers only; no row data) --------------------------------
    def _filesystem(self):
        if self._fs is None:
            from huggingface_hub import HfFileSystem
            self._fs = HfFileSystem()
        return self._fs

    def _ensure_manifest(self) -> None:
        if self._total is not None:
            return
        import pyarrow.parquet as pq

        with self._io_lock:
            if self._total is not None:  # built by another thread while we waited
                return
            fs = self._filesystem()
            base = f"datasets/{self._repo}@{self._revision}/{self._data_dir}"
            files = sorted(
                p for p in fs.ls(base, detail=False) if str(p).endswith(".parquet")
            )
            if not files:
                raise RuntimeError(f"no parquet files under {base}")
            rg_start: list[int] = []
            rg_loc: list[tuple[int, int]] = []
            total = 0
            for fi, path in enumerate(files):
                with fs.open(path) as fh:
                    md = pq.ParquetFile(fh).metadata
                    for rg in range(md.num_row_groups):
                        rg_start.append(total)
                        rg_loc.append((fi, rg))
                        total += md.row_group(rg).num_rows
            self._files, self._rg_start, self._rg_loc = files, rg_start, rg_loc
            self._total = total  # assigned last: the double-check sentinel

    # -- access --------------------------------------------------------------
    def __len__(self) -> int:
        self._ensure_manifest()
        assert self._total is not None
        return self._total

    def __getitem__(self, idx: int) -> dict:
        return self.get_row(idx)

    def get_row(self, idx: int) -> dict:
        """Return row ``idx % len`` as a dict, fetching its row-group lazily.

        Thread-safe: cache hits take only the brief ``_lock``; a miss serializes
        the I/O on ``_io_lock`` (with a double-check) so concurrent callers never
        read the shared parquet handle at once or corrupt the LRU.
        """
        self._ensure_manifest()
        assert self._total and self._rg_start is not None and self._rg_loc is not None
        idx %= self._total
        gi = bisect.bisect_right(self._rg_start, idx) - 1
        key = self._rg_loc[gi]
        rg_start = self._rg_start[gi]
        with self._lock:
            rows = self._cache.get(key)
            if rows is not None:
                self._cache.move_to_end(key)
                return rows[idx - rg_start]
        # Miss: fetch under the I/O lock — the per-file handle is shared and not
        # concurrency-safe. Re-check the cache in case a peer fetched it first.
        with self._io_lock:
            with self._lock:
                rows = self._cache.get(key)
            if rows is None:
                rows = self._fetch_row_group(*key)
                with self._lock:
                    self._cache[key] = rows
                    self._cache.move_to_end(key)
                    while len(self._cache) > self._cache_cap:
                        self._cache.popitem(last=False)
        return rows[idx - rg_start]

    def _fetch_row_group(self, file_idx: int, rg_idx: int) -> list[dict]:
        pf = self._parquet_file(file_idx)
        table = pf.read_row_group(rg_idx, columns=self._columns)
        return table.to_pylist()

    def _parquet_file(self, file_idx: int):
        pf = self._pf.get(file_idx)
        if pf is not None:
            self._pf.move_to_end(file_idx)
            return pf
        import pyarrow.parquet as pq

        assert self._files is not None
        pf = pq.ParquetFile(self._filesystem().open(self._files[file_idx]))
        self._pf[file_idx] = pf
        if len(self._pf) > 4:  # keep a few shards open (1 for the curated set)
            _, old = self._pf.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return pf
