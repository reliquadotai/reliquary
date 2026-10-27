"""Publish one checkpoint per optimizer step as lossless sparse deltas.

docs/rl-async-design.md §4.8. Per step the publisher writes to the
``reliquary-public`` bucket:

- the delta from the previous step (bf16 positions whose bits changed);
- a cumulative delta to this step from each running base (the last anchor and
  every ``cumulative_stride``-th step, at most ``max_cumulative_bases``), for
  miners that lag; a base whose cumulative grows past ``max_cumulative_bytes``
  is dropped, and those miners take the anchor;
- every ``anchor_interval`` steps (and on the first publication) a full anchor
  as safetensors shards, with the model's non-weight files (``model_dir``: HF
  config, generation config, tokenizer, chat template) so a miner can load it
  from the bucket alone;
- ``index.json`` last, so a reader never sees an entry whose object is missing.
  Pruned objects leave the index first and are deleted ``prune_grace_s`` later
  (the validator's presigned URL lifetime): a miner that just fetched the old
  index can still download them. The pending deletions are memory only: a
  restart leaves them in the bucket.

A cumulative base keeps one bit per weight (the union of changed positions),
1.1 GB on a 9B model; the values come from the current step. The publisher
holds a CPU copy of the last published checkpoint (the delta's base), so the
trainer is free to keep updating its own weights. It never reads deltas back.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reliquary.shared.ckpt_delta import (
    ANCHOR_SHARD_BYTES,
    DEFAULT_ZSTD_LEVEL,
    changed_masks,
    checkpoint_sha256,
    delta_from_masks,
    encode_delta,
    write_anchor_shards,
)
from reliquary.shared.ckpt_index import (
    INDEX_FORMAT,
    MAX_MODEL_FILE_BYTES,
    anchor_key,
    delta_key,
    index_key,
    model_file_key,
    model_file_name_ok,
)
from reliquary.shared.strict_json import strict_json_loads

logger = logging.getLogger(__name__)

PRUNE_GRACE_S = 900.0   # rl_checkpoint.PRESIGN_TTL_SECONDS: a URL handed out before a prune stays good

def model_files(model_dir: str | Path) -> list[Path]:
    """The non-weight files of an HF model directory an anchor carries (config, tokenizer, chat template)."""
    folder = Path(model_dir)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a model directory")
    files = sorted(p for p in folder.iterdir() if p.is_file() and model_file_name_ok(p.name))
    if not any(p.name == "config.json" for p in files):
        raise ValueError(f"{folder} has no config.json")
    too_big = [p.name for p in files if p.stat().st_size > MAX_MODEL_FILE_BYTES]
    if too_big:
        raise ValueError(f"model files over {MAX_MODEL_FILE_BYTES} bytes: {too_big}")
    return files


class R2ObjectStore:
    """The few object operations the publisher needs, over a sync boto3 client."""

    def __init__(self, client: Any, bucket: str) -> None:
        self._client, self.bucket = client, bucket

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def put_file(self, key: str, path: Path) -> None:
        from reliquary.trainer.publisher import _multipart_transfer_config

        self._client.upload_file(str(path), self.bucket, key, Config=_multipart_transfer_config())

    def get_bytes(self, key: str) -> bytes | None:
        try:
            return self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self._client.exceptions.NoSuchKey:
            return None

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class _Base:
    version: int
    sha256: str
    bits: dict[str, np.ndarray]  # packed union of changed positions since version
    sizes: dict[str, int]


class _UnpackedMasks(Mapping):
    """A base's union as bool masks, unpacked one tensor at a time."""

    def __init__(self, base: _Base) -> None:
        self._base = base

    def __getitem__(self, name: str) -> torch.Tensor:
        bits = np.unpackbits(self._base.bits[name], count=self._base.sizes[name])
        return torch.from_numpy(bits.view(bool))

    def __iter__(self):
        return iter(self._base.bits)

    def __len__(self) -> int:
        return len(self._base.bits)


_NORM_CHUNK = 1 << 24


def update_metrics(old: Mapping[str, torch.Tensor], new: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Size of one optimizer step on the published (CPU) weights: global and per-group relative L2 norm
    ||new - old|| / ||old||, the largest per-tensor one, and the share of elements that changed."""
    groups: dict[str, list[float]] = {}
    worst, changed, total = 0.0, 0, 0
    for name, b in new.items():
        a = old[name]
        if not (a.is_floating_point() and b.is_floating_point()) or a.shape != b.shape:
            continue
        fa, fb = a.reshape(-1), b.reshape(-1)
        diff_sq = base_sq = 0.0
        for i in range(0, fa.numel(), _NORM_CHUNK):
            xa, xb = fa[i:i + _NORM_CHUNK].float(), fb[i:i + _NORM_CHUNK].float()
            d = xb - xa
            diff_sq += float(d.square().sum())
            base_sq += float(xa.square().sum())
            changed += int((d != 0).sum())
        total += fa.numel()
        if base_sq > 0:
            worst = max(worst, (diff_sq / base_sq) ** 0.5)
        for key in ("all", _param_group(name)):
            acc = groups.setdefault(key, [0.0, 0.0])
            acc[0] += diff_sq
            acc[1] += base_sq
    out = {f"async/update/rel_norm_{key}": (d / b) ** 0.5 for key, (d, b) in groups.items() if b > 0}
    out["async/update/rel_norm_tensor_max"] = worst
    out["async/update/changed_ratio"] = changed / total if total else 0.0
    return out


def _param_group(name: str) -> str:
    """lm_head | embed | layer_<index> | norm | other."""
    lowered = name.lower()
    if "lm_head" in lowered:
        return "lm_head"
    if "embed" in lowered:
        return "embed"
    parts = lowered.split(".")
    if "layers" in parts:
        i = parts.index("layers")
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            return f"layer_{int(parts[i + 1]):03d}"
    if "norm" in lowered:
        return "norm"
    return "other"


class DeltaPublisher:
    def __init__(
        self,
        *,
        run: str,
        store: Any,
        staging_dir: str | Path,
        anchor_interval: int = 50,
        cumulative_stride: int = 4,
        max_cumulative_bases: int = 3,
        max_cumulative_bytes: int | None = None,
        keep_anchors: int = 2,
        anchor_shard_bytes: int = ANCHOR_SHARD_BYTES,
        zstd_level: int = DEFAULT_ZSTD_LEVEL,
        model_dir: str | Path | None = None,
        clock: Callable[[], float] = time.time,
        prune_grace_s: float = PRUNE_GRACE_S,
        writer_id: str | None = None,
        publish_index: Callable[[dict], None] | None = None,
        assert_owner: Callable[[], None] | None = None,
    ) -> None:
        if anchor_interval < 1 or cumulative_stride < 1 or keep_anchors < 1:
            raise ValueError("intervals and keep_anchors must be positive")
        self.run, self._store = run, store
        if writer_id is not None and not re.fullmatch(r"[1-9][0-9]*-[0-9a-f]{32}", writer_id):
            raise ValueError("invalid checkpoint writer identity")
        self._writer_id, self._publish_index, self._assert_owner = writer_id, publish_index, assert_owner
        if writer_id is not None and (publish_index is None or assert_owner is None):
            raise ValueError("a checkpoint writer requires fenced index publication")
        self._staging = Path(staging_dir)
        self.anchor_interval, self.cumulative_stride = anchor_interval, cumulative_stride
        self.max_cumulative_bases = max_cumulative_bases
        self.max_cumulative_bytes = max_cumulative_bytes
        self.keep_anchors, self._shard_bytes = keep_anchors, anchor_shard_bytes
        self._level, self._clock = zstd_level, clock
        self._prune_grace_s = float(prune_grace_s)
        self._doomed: list[tuple[float, list[str]]] = []   # (delete after, keys) pruned from the index
        self._model_files = model_files(model_dir) if model_dir is not None else []
        self._prev: dict[str, torch.Tensor] | None = None
        self._prev_version: int | None = None
        self._prev_sha: str | None = None
        self._last_anchor: int | None = None
        self._bases: list[_Base] = []
        self.last_update: dict[str, float] = {}     # update-size telemetry of the last delta (``update_metrics``)
        self._update_pair: tuple | None = None
        self.anchors: list[dict] = []
        self.deltas: list[dict] = []
        # Entries already published by an earlier process stay listed (and pruned
        # in turn); the next publication is an anchor since the base state is gone.
        existing = store.get_bytes(index_key(run))
        if existing:
            old = strict_json_loads(existing)
            if old.get("format") == INDEX_FORMAT and old.get("run") == run:
                self.anchors, self.deltas = list(old["anchors"]), list(old["deltas"])

    @property
    def latest_snapshot(self) -> Mapping[str, torch.Tensor] | None:
        """The last published checkpoint (CPU copy, never mutated: each step
        replaces it), e.g. for a background Hugging Face backup."""
        return self._prev

    def publish(self, version: int, state: Mapping[str, torch.Tensor]) -> dict:
        if self._assert_owner is not None:
            self._assert_owner()
        version = int(version)
        if self._prev_version is not None and version <= self._prev_version:
            raise ValueError(f"version {version} does not follow {self._prev_version}")
        snapshot = {n: t.detach().to("cpu", copy=True).contiguous() for n, t in state.items()}
        sha = checkpoint_sha256(snapshot)
        if self._prev is not None:
            masks = changed_masks(self._prev, snapshot)
            self._put_delta(snapshot, masks, self._prev_version, self._prev_sha, version, sha)
            self._extend_cumulatives(snapshot, masks, version, sha)
        anchor = self._prev is None or self._last_anchor is None or (
            version - self._last_anchor >= self.anchor_interval
            or version % self.anchor_interval == 0)
        if anchor:
            self._put_anchor(snapshot, version, sha)
            self._bases = [self._new_base(version, sha, snapshot)]
        elif version % self.cumulative_stride == 0:
            self._bases.append(self._new_base(version, sha, snapshot))
            del self._bases[:-self.max_cumulative_bases]
        dropped = self._prune(version)
        index = {
            "format": INDEX_FORMAT, "run": self.run,
            "latest": {"version": version, "sha256": sha},
            "anchors": self.anchors, "deltas": self.deltas,
            "published_at": round(float(self._clock()), 3),
        }
        if self._publish_index is None:
            self._store.put_bytes(index_key(self.run), json.dumps(index, sort_keys=True).encode())
        else:
            self._publish_index(index)
        self._update_pair = (self._prev, snapshot) if self._prev is not None else None
        self._prev, self._prev_version, self._prev_sha = snapshot, version, sha
        if dropped:
            self._doomed.append((float(self._clock()) + self._prune_grace_s, dropped))
        self._delete_due()
        return index

    def measure_update(self) -> dict[str, float]:
        """``update_metrics`` of the last publication, run after it (it keeps the previous weights until then)."""
        pair, self._update_pair = self._update_pair, None
        if pair is None:
            return {}
        try:
            self.last_update = update_metrics(*pair)
        except Exception:                                # telemetry never fails a publication
            logger.exception("update-size telemetry failed")
            self.last_update = {}
        return self.last_update

    # ------------------------------------------------------------------ steps

    def _object_key(self, key: str) -> str:
        if self._assert_owner is not None:
            self._assert_owner()
        if self._writer_id is None:
            return key
        prefix = index_key(self.run).rsplit("/", 1)[0] + "/"
        return prefix + "writers/" + self._writer_id + "/" + key[len(prefix):]

    def _put_delta(self, state, masks, from_version: int, base_sha: str, to_version: int,
                   sha: str) -> dict | None:
        delta = delta_from_masks(state, masks, from_version=from_version, to_version=to_version,
                                 base_sha256=base_sha, target_sha256=sha)
        blob = encode_delta(delta, level=self._level)
        if (self.max_cumulative_bytes is not None and to_version - from_version > 1
                and len(blob) > self.max_cumulative_bytes):
            return None
        key = self._object_key(delta_key(self.run, from_version, to_version))
        self._store.put_bytes(key, blob)
        entry = {"from": from_version, "to": to_version, "key": key, "size": len(blob),
                 "sha256": hashlib.sha256(blob).hexdigest(), "target_sha256": sha}
        self.deltas.append(entry)
        logger.info("checkpoint delta %d->%d: %d changed bf16 elements, %d bytes",
                    from_version, to_version, delta.changed_elements, len(blob))
        return entry

    @staticmethod
    def _new_base(version: int, sha: str, state) -> _Base:
        sizes = {n: (t.numel() if t.dtype == torch.bfloat16 else 1) for n, t in state.items()}
        bits = {n: np.zeros((s + 7) // 8, dtype=np.uint8) for n, s in sizes.items()}
        return _Base(version, sha, bits, sizes)

    def _extend_cumulatives(self, state, masks, version: int, sha: str) -> None:
        kept = []
        for base in self._bases:
            for name, mask in masks.items():
                base.bits[name] |= np.packbits(mask.numpy())
            if version - base.version < 2:
                kept.append(base)
                continue
            if self._put_delta(state, _UnpackedMasks(base), base.version, base.sha256, version, sha) is None:
                logger.info("cumulative from %d passed the size cap: dropped", base.version)
                continue
            kept.append(base)
        self._bases = kept

    def _put_anchor(self, state, version: int, sha: str) -> None:
        folder = self._staging / f"anchor-{version:08d}"
        shutil.rmtree(folder, ignore_errors=True)
        try:
            files = []
            for path in write_anchor_shards(state, folder, shard_bytes=self._shard_bytes):
                key = self._object_key(anchor_key(self.run, version, path.name))
                self._store.put_file(key, path)
                files.append({"key": key, "size": path.stat().st_size,
                              "sha256": _sha256_file(path)})
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        anchor = {"version": version, "sha256": sha, "files": files}
        if self._model_files:
            anchor["model_files"] = []
            for path in self._model_files:
                data = path.read_bytes()
                key = self._object_key(model_file_key(self.run, version, path.name))
                self._store.put_bytes(key, data)
                anchor["model_files"].append({"name": path.name, "key": key, "size": len(data),
                                              "sha256": hashlib.sha256(data).hexdigest()})
        self.anchors.append(anchor)
        self._last_anchor = version

    def _prune(self, version: int) -> list[str]:
        """Keep ``keep_anchors`` anchors, the deltas starting at or after the
        oldest of them, and cumulatives ending at this step or the one before
        (a miner may still be downloading those). Returns the dropped entries'
        keys: deleted once the index without them is written, after the grace."""
        drop_anchors = self.anchors[:-self.keep_anchors]
        self.anchors = self.anchors[-self.keep_anchors:]
        floor = self.anchors[0]["version"] if self.anchors else 0
        keep, drop = [], []
        for d in self.deltas:
            stale_cumulative = d["to"] - d["from"] > 1 and d["to"] < version - 1
            (drop if d["from"] < floor or stale_cumulative else keep).append(d)
        self.deltas = keep
        dropped = [f["key"] for a in drop_anchors for f in [*a["files"], *a.get("model_files", [])]]
        return dropped + [d["key"] for d in drop]

    def _delete_due(self) -> None:
        now = float(self._clock())
        while self._doomed and self._doomed[0][0] <= now:
            for key in self._doomed.pop(0)[1]:
                try:
                    if self._assert_owner is not None:
                        self._assert_owner()
                    self._store.delete(key)
                except Exception:
                    logger.warning("could not delete %s; left in the bucket", key, exc_info=True)
