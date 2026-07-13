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
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Optional


logger = logging.getLogger(__name__)


class PromptSourceUnavailable(RuntimeError):
    """The pinned prompt source cannot serve an exact dataset row."""


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
        full_file_fallback: Optional[bool] = None,
    ) -> None:
        self._repo = repo
        self._revision = revision
        self._columns = columns
        self._data_dir = data_dir
        self._cache_cap = cache_row_groups
        self._fs = fs  # injectable for tests; HfFileSystem when None
        if full_file_fallback is None:
            full_file_fallback = (
                fs is None
                and os.environ.get(
                    "RELIQUARY_PARQUET_FULL_FILE_FALLBACK", "true"
                ).lower()
                in {"1", "true", "yes"}
            )
        self._full_file_fallback = bool(full_file_fallback)
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
        self._health_lock = threading.Lock()
        self._source_successes = 0
        self._source_failures = 0
        self._local_full_file_hits = 0
        self._full_file_fallbacks = 0
        self._last_success_ts: Optional[float] = None
        self._last_failure_ts: Optional[float] = None
        self._last_error_type: Optional[str] = None

    # -- manifest (footers only; no row data) --------------------------------
    def _filesystem(self):
        if self._fs is None:
            from huggingface_hub import HfFileSystem
            self._fs = HfFileSystem()
        return self._fs

    def _repo_filename(self, path: str) -> str:
        prefix = f"datasets/{self._repo}@{self._revision}/"
        if path.startswith(prefix):
            return path[len(prefix):]
        marker = f"/{self._data_dir}/"
        if marker in path:
            return f"{self._data_dir}/{path.split(marker, 1)[1]}"
        return path

    def _record_success(self) -> None:
        with self._health_lock:
            self._source_successes += 1
            self._last_success_ts = time.time()

    def _record_failure(self, exc: BaseException) -> None:
        with self._health_lock:
            self._source_failures += 1
            self._last_failure_ts = time.time()
            self._last_error_type = type(exc).__name__

    def _cached_file(self, path: str) -> Optional[str]:
        if not self._full_file_fallback:
            return None
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(
                repo_id=self._repo,
                filename=self._repo_filename(path),
                revision=self._revision,
                repo_type="dataset",
            )
        except Exception:
            return None
        return cached if isinstance(cached, str) else None

    def _download_exact_file(self, path: str) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=self._repo,
            filename=self._repo_filename(path),
            revision=self._revision,
            repo_type="dataset",
        )

    def _open_parquet_file(self, path: str):
        """Open one pinned Parquet blob, preferring the persistent HF cache.

        Hugging Face range reads and full-file downloads use different data
        paths. If the range backend is unavailable, ``hf_hub_download`` still
        verifies and stores the exact revision-pinned blob in the normal HF
        cache. Subsequent windows then stay independent of that remote outage.
        """
        import pyarrow.parquet as pq

        cached = self._cached_file(path)
        if cached is not None:
            try:
                parquet_file = pq.ParquetFile(cached)
                _ = parquet_file.metadata
            except Exception as exc:
                self._record_failure(exc)
            else:
                with self._health_lock:
                    self._local_full_file_hits += 1
                self._record_success()
                return parquet_file

        remote_handle = None
        try:
            remote_handle = self._filesystem().open(path)
            parquet_file = pq.ParquetFile(remote_handle)
            _ = parquet_file.metadata
        except Exception as remote_exc:
            self._record_failure(remote_exc)
            if remote_handle is not None:
                try:
                    remote_handle.close()
                except Exception:
                    pass
            if not self._full_file_fallback:
                raise PromptSourceUnavailable(
                    f"prompt source unavailable: {self._repo}@{self._revision} "
                    f"({type(remote_exc).__name__})"
                ) from remote_exc
            try:
                local_path = self._download_exact_file(path)
                parquet_file = pq.ParquetFile(local_path)
                _ = parquet_file.metadata
            except Exception as fallback_exc:
                self._record_failure(fallback_exc)
                raise PromptSourceUnavailable(
                    f"prompt source unavailable: {self._repo}@{self._revision} "
                    f"({type(fallback_exc).__name__})"
                ) from fallback_exc
            with self._health_lock:
                self._full_file_fallbacks += 1
            logger.warning(
                "prompt_source_full_file_fallback repo=%s revision=%s file=%s",
                self._repo,
                self._revision,
                self._repo_filename(path),
            )
            self._record_success()
            return parquet_file
        self._record_success()
        return parquet_file

    def source_health(self) -> dict[str, Any]:
        """Return secret-free prompt-source readiness telemetry."""
        with self._health_lock:
            last_success = self._last_success_ts
            last_failure = self._last_failure_ts
            status = "ready" if self._total is not None else "initializing"
            if last_failure is not None and (
                last_success is None or last_failure > last_success
            ):
                status = "degraded"
            return {
                "status": status,
                "repo": self._repo,
                "revision": self._revision,
                "manifest_ready": self._total is not None,
                "files": len(self._files or ()),
                "row_groups": len(self._rg_loc or ()),
                "cached_row_groups": len(self._cache),
                "source_successes_total": self._source_successes,
                "source_failures_total": self._source_failures,
                "local_full_file_hits_total": self._local_full_file_hits,
                "full_file_fallbacks_total": self._full_file_fallbacks,
                "last_success_ts": last_success,
                "last_failure_ts": last_failure,
                "last_error_type": self._last_error_type,
            }

    def _ensure_manifest(self) -> None:
        if self._total is not None:
            return

        with self._io_lock:
            if self._total is not None:  # built by another thread while we waited
                return
            fs = self._filesystem()
            base = f"datasets/{self._repo}@{self._revision}/{self._data_dir}"
            try:
                files = sorted(
                    str(p)
                    for p in fs.ls(base, detail=False)
                    if str(p).endswith(".parquet")
                )
            except Exception as exc:
                self._record_failure(exc)
                raise PromptSourceUnavailable(
                    f"prompt source manifest unavailable: "
                    f"{self._repo}@{self._revision} ({type(exc).__name__})"
                ) from exc
            if not files:
                exc = RuntimeError(f"no parquet files under {base}")
                self._record_failure(exc)
                raise PromptSourceUnavailable(str(exc)) from exc
            rg_start: list[int] = []
            rg_loc: list[tuple[int, int]] = []
            total = 0
            for fi, path in enumerate(files):
                parquet_file = self._open_parquet_file(path)
                try:
                    md = parquet_file.metadata
                    for rg in range(md.num_row_groups):
                        rg_start.append(total)
                        rg_loc.append((fi, rg))
                        total += md.row_group(rg).num_rows
                finally:
                    try:
                        parquet_file.close()
                    except Exception:
                        pass
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
        assert self._files is not None
        pf = self._open_parquet_file(self._files[file_idx])
        self._pf[file_idx] = pf
        if len(self._pf) > 4:  # keep a few shards open (1 for the curated set)
            _, old = self._pf.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return pf


__all__ = ["PromptSourceUnavailable", "VirtualParquetDataset"]
