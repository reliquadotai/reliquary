# OMI Virtual Dataset — Row-Group Streaming Design

- **Date:** 2026-06-11
- **Status:** Draft for review
- **Topic:** Use the full OpenMathInstruct-2 dataset (~14M prompts) without
  downloading or materializing it, by addressing a virtual 14M index space and
  fetching only the parquet row-groups each window actually touches.

## Problem

`OpenMathInstructEnvironment` downloads the first `RELIQUARY_OMI_SHARDS` (default
4) of 32 parquet shards up front and materializes them. Consequences:

- Only the first ~1.76M of 14M prompts are ever reachable; 28 shards are dead.
- The shard count is a hardcoded magic number that caps the universe.
- Loading all 32 shards is not an option: it costs ~2GB+ download at startup
  (on top of the model download) and materializes 14M rows in RAM.

The two binding constraints are **RAM** (do not hold 14M rows) and **miner
startup download time** (do not add a large dataset download next to the model).

## Goals

- `len(env)` reflects the **true full dataset** (~13,972,791 rows), so every
  prompt is logically addressable.
- **No bulk data download** at startup — only tiny parquet footers.
- **Bounded RAM/disk** at runtime — only the working set a window touches.
- **Byte-identical content** on miner and validator (token-binding +
  prompt-range require identical `get_problem(idx)` and identical `len(env)`).
- Remove the `RELIQUARY_OMI_SHARDS` cap.

## Non-Goals

- Changing `OpenCodeInstructEnvironment` (it loads a small curated, pinned
  subset — not the 14M-scale problem). The lazy loader is written so it *could*
  generalize later, but that is out of scope here.
- Changing the prompt-range, cooldown, batcher, miner, or any consumer of the
  `Environment` interface. This change is confined to the OMI env internals.

## Key Insight (measured)

A parquet shard is internally split into **row-groups**; a small footer records
each row-group's byte offset and row count. With HTTP range requests
(pyarrow + `huggingface_hub.HfFileSystem`) a single row-group can be read
**without downloading the whole shard**.

Measured structure of `nvidia/OpenMathInstruct-2`:

- 1 shard = **437 row-groups × 1000 rows** (~1MB uncompressed each).
- 32 shards → **13,972,791** total rows.
- A prompt-range slice of 5000 contiguous rows = **exactly ~5 adjacent
  row-groups ≈ a few MB**.

Measured costs:

| Operation | Cold | Notes |
|---|---|---|
| Manifest: read 32 footers | ~16s | parallelizable to ~2-3s; cacheable → ~0 on restart |
| 1 row-group (1000 rows) | ~1s | first access |
| Window slice (5 row-groups) | ~0.45s | warm; negligible vs 3-5 min window |

So startup data download ≈ 0, per-window fetch ≈ a few MB / sub-second.

## Architecture

One new internal component, plus a rewrite of the OMI env's data backing. The
public `Environment` surface (`name`, `__len__`, `get_problem`) is unchanged.

### `VirtualParquetDataset` (new, `reliquary/environment/virtual_parquet.py`)

A dependency-light, deterministic, lazy reader over a set of remote parquet
shards pinned to a revision.

Responsibilities:

1. **Manifest build.** On first use, read the footer of each shard
   (concurrently) to get per-shard row counts → cumulative offsets. Total =
   `len`. Persist the manifest to a local cache keyed by `(repo, revision)` so
   subsequent processes/restarts skip the network entirely.
2. **Index mapping.** `global_idx → (shard, local_row)` via the cumulative
   offsets; `local_row → (row_group = local_row // RG_SIZE, offset)`. Pure and
   deterministic. `RG_SIZE` (1000) is read from the manifest, not assumed.
3. **Lazy fetch + cache.** `get_row(global_idx)` reads the containing row-group
   (range request, only `problem` + `expected_answer` columns), holds it in an
   LRU cache of row-groups bounded by a configurable budget; evicts oldest.
   Fetched row-groups are also persisted to a local disk cache for reuse across
   restarts.

### `OpenMathInstructEnvironment` (rewrite of the data backing)

- Drops `RELIQUARY_OMI_SHARDS` and the eager `load_dataset`.
- Holds a `VirtualParquetDataset` pinned to a **fixed revision** (new constant,
  required for both-sides determinism — today the env loads `main` unpinned).
- `__len__` → manifest total. `get_problem(idx)` → `dataset.get_row(idx % len)`,
  then the same prompt/ground-truth shaping as today.

## Data Flow

**Startup:** build/load manifest (footers or cached) → `len(env)` known. No data
rows fetched.

**Per window:** prompt-range yields slice `[lo, hi)` → consumers call
`get_problem(idx)` for idx in the slice → first call in a row-group triggers a
range fetch (~1MB), subsequent calls in the same group hit cache. A whole 5000
slice = ~5 fetches ≈ a few MB, sub-second, overlappable with GRAIL verify.

## Determinism & Both-Sides Agreement (critical)

Token-binding and prompt-range require miner and validator to compute identical
`len(env)` and identical `get_problem(idx)` bytes. Guaranteed by:

- **Pinned revision** of the OMI repo (new constant). Both sides read the same
  immutable parquet files → identical footers → identical manifest → identical
  `len` and mapping → identical rows.
- Manifest derived only from footer metadata (counts), not from any local state.
- `RG_SIZE` and offsets read from the files, never hardcoded.

If the manifest cannot be built (network down at cold start with no cached
manifest), the env fails closed rather than silently using a partial universe
(a partial `len` would desync from peers).

## Caching & Eviction

- **Row-group LRU**, budget configurable (default e.g. 64 row-groups ≈ ~64MB).
  A window touches ~5; the budget comfortably covers several recent windows.
- **Disk persistence** of the manifest and (optionally) fetched row-groups under
  the HF cache dir, keyed by `(repo, revision)`. Survives restarts.

## Resilience / Error Handling

- **Manifest:** retry with backoff; once cached on disk, cold network is only
  needed the very first time per machine/revision.
- **Row-group fetch:** retry with backoff; on persistent failure raise so the
  caller skips that prompt/window rather than serving wrong content.
- **Optional resident socle (default OFF):** a config to keep the first K
  row-groups always resident as a degraded-mode fallback if HF is unreachable.
  Off by default (YAGNI); the disk cache already covers the common case. Can be
  enabled if prod observes HF flakiness.

## Interaction with Prompt-Range

- Universe grows 1.76M → ~14M. With `PROMPT_RANGE_SIZE=5000` the per-window
  slice fraction drops from ~0.28% to ~0.036% — **stronger** anti-pre-curation
  dilution, and honest miners still get 5000 eligible prompts/window/env (ample).
- `PROMPT_RANGE_SIZE` stays 5000 for now; it is already env-configurable and can
  be retuned independently. No change required by this design.
- The prompt-range slice is contiguous → maps to adjacent row-groups → the
  cheapest possible fetch pattern. The two features reinforce each other.

## Configuration / Rollout

- New constant: pinned OMI revision (commit sha).
- New optional envs: row-group cache budget; resident-socle toggle (default off).
- `RELIQUARY_OMI_SHARDS` removed. (Cutover note: both sides must upgrade
  together — `len(env)` changes from ~1.76M to ~14M, which changes the
  prompt-index space. Sequence this like the prompt-range cutover: ship the
  client, then flip, so miners and validator agree on `len`.)

## Testing Strategy

- **Pure mapping tests:** `global_idx ↔ (shard, row_group, offset)` round-trips,
  boundary crossings between shards, modulo wrap — no network.
- **Manifest tests:** cumulative-offset math against a fake footer set; total
  equals sum; cache hit avoids refetch.
- **Lazy-fetch tests:** inject a fake fetcher (local tiny parquet fixture or a
  stub) — assert only the touched row-groups are fetched, LRU eviction works,
  same idx served from cache without refetch.
- **Determinism test:** two independent `VirtualParquetDataset` instances over
  the same pinned fixture return identical rows for the same indices.
- No live-HF dependency in unit tests (fetch layer is injectable).

## Open Decisions (resolve during review)

1. Resident socle: keep default OFF? (recommended yes — disk cache + fail-closed
   is enough; add socle only if prod shows HF flakiness.)
2. Pinned revision: pick a current OMI commit sha to freeze.
3. Row-group cache budget default (proposed 64 row-groups).
