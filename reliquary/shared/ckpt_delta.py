"""Lossless sparse checkpoint deltas (docs/rl-async-design.md §4.8).

A delta from version a to version b lists, for each tensor, the flat positions
whose bf16 BITS changed and their new bf16 bits (never arithmetic differences),
plus the SHA-256 of the full checkpoint at b. Applying it is an overwrite, so a
delta a->b is also valid from any version in [a, b): a position changed between
a and the holder's version, and not after, already holds its value at b. That
makes the cumulative delta a->c (union of positions, latest values) usable by
every miner that lags behind c by no more than c - a steps.

Tensors that are not bf16 are stored whole when any byte changed. A tensor added,
removed or reshaped is refused: that needs a full anchor.

Wire format (one file per delta)::

    b"RQCKDLT1" | u64 header length | header JSON | sections

Each tensor has one section (whole tensor) or two (indices, values), each
compressed with zstd after a byte shuffle (bytes of equal significance grouped,
which is what makes small index gaps and shared bf16 exponents compress).
Indices are sorted gaps (u32, or u64 for tensors of 2**32 elements or more), or a
packed bitmask when that is smaller (more than one position in 32 changed).
Compression goes through pyarrow, already a dependency.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import struct
from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa
import torch

MAGIC = b"RQCKDLT1"
FORMAT_VERSION = 1
HASH_DOMAIN = b"reliquary-ckpt-sha256/1\n"
DEFAULT_ZSTD_LEVEL = 3
_U32_LIMIT = 2**32
ANCHOR_SHARD_BYTES = 5 * 1024**3


class DeltaError(ValueError):
    """The delta does not fit the checkpoint (structure, versions or format)."""


class ChecksumMismatch(RuntimeError):
    """A rebuilt checkpoint does not hash to the published SHA-256."""


@dataclass(frozen=True)
class TensorPatch:
    name: str
    dtype: str
    shape: tuple[int, ...]
    # Sparse (bf16): sorted int64 flat positions and their new uint16 bits.
    # Whole: indices is None and values holds the tensor's raw bytes (uint8).
    indices: np.ndarray | None
    values: np.ndarray

    @property
    def whole(self) -> bool:
        return self.indices is None


@dataclass(frozen=True)
class CheckpointDelta:
    from_version: int
    to_version: int
    base_sha256: str
    target_sha256: str
    patches: tuple[TensorPatch, ...]

    @property
    def changed_elements(self) -> int:
        return sum(p.values.size for p in self.patches if not p.whole)


# --------------------------------------------------------------------- hashing

def _dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


def _raw_bytes(t: torch.Tensor) -> np.ndarray:
    flat = t.detach().contiguous().reshape(-1)
    if flat.numel() == 0:
        return np.empty(0, dtype=np.uint8)
    return flat.view(torch.uint8).cpu().numpy()


def _tensor_digest(t: torch.Tensor) -> bytes:
    return hashlib.sha256(memoryview(_raw_bytes(t))).digest()


def checkpoint_sha256(state: Mapping[str, torch.Tensor], *, workers: int = 8) -> str:
    """SHA-256 of a checkpoint, independent of sharding and file layout.

    sha256(domain || for each name in sorted order: len(meta) meta sha256(bytes)),
    meta being the JSON [name, dtype, shape]. Tensors are hashed in parallel
    (hashlib releases the GIL), which matters on a 9B checkpoint.
    """
    names = sorted(state)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        digests = list(pool.map(lambda n: _tensor_digest(state[n]), names))
    h = hashlib.sha256(HASH_DOMAIN)
    for name, digest in zip(names, digests):
        t = state[name]
        meta = json.dumps([name, _dtype_name(t), list(t.shape)], separators=(",", ":")).encode()
        h.update(struct.pack("<I", len(meta)) + meta + digest)
    return h.hexdigest()


def verify_checkpoint(state: Mapping[str, torch.Tensor], expected_sha256: str) -> None:
    actual = checkpoint_sha256(state)
    if actual != expected_sha256:
        raise ChecksumMismatch(f"checkpoint sha256 {actual} != published {expected_sha256}")


# ----------------------------------------------------------------------- build

def _bits16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().reshape(-1).view(torch.int16)


def _check_same_structure(old: Mapping[str, torch.Tensor], new: Mapping[str, torch.Tensor]) -> None:
    if set(old) != set(new):
        added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
        raise DeltaError(f"tensor set changed (added {added[:3]}, removed {removed[:3]}): needs an anchor")
    for name in old:
        a, b = old[name], new[name]
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            raise DeltaError(f"{name}: dtype or shape changed: needs an anchor")


def changed_masks(
    old: Mapping[str, torch.Tensor], new: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Per tensor, which positions changed: a flat bool mask for bf16 tensors,
    a one-element bool for the others (changed at all). Masks OR together, which
    is how a publisher tracks the union of changes since some base version."""
    _check_same_structure(old, new)
    masks: dict[str, torch.Tensor] = {}
    for name in sorted(new):
        a, b = old[name], new[name]
        if b.dtype == torch.bfloat16:
            masks[name] = (_bits16(a) != _bits16(b)).cpu()
        else:
            masks[name] = torch.tensor([not np.array_equal(_raw_bytes(a), _raw_bytes(b))])
    return masks


def delta_from_masks(
    new: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    *,
    from_version: int,
    to_version: int,
    base_sha256: str,
    target_sha256: str,
) -> CheckpointDelta:
    """The delta that writes ``new``'s values at every masked position."""
    if not to_version > from_version:
        raise DeltaError("a delta must move forward")
    patches = []
    for name in sorted(masks):
        mask, t = masks[name], new[name]
        if t.dtype == torch.bfloat16:
            if mask.numel() != t.numel():
                raise DeltaError(f"{name}: mask does not fit the tensor")
            idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                continue
            vals = _bits16(t).cpu()[idx]
            patches.append(TensorPatch(name, "bfloat16", tuple(t.shape),
                                       idx.numpy().astype(np.int64, copy=False),
                                       vals.numpy().view(np.uint16)))
        elif bool(mask.any()):
            patches.append(TensorPatch(name, _dtype_name(t), tuple(t.shape), None,
                                       _raw_bytes(t).copy()))
    return CheckpointDelta(int(from_version), int(to_version), base_sha256, target_sha256,
                           tuple(patches))


def build_delta(
    old: Mapping[str, torch.Tensor],
    new: Mapping[str, torch.Tensor],
    *,
    from_version: int,
    to_version: int,
    base_sha256: str | None = None,
    target_sha256: str | None = None,
) -> CheckpointDelta:
    """Delta between two checkpoints. Pass known hashes to skip rehashing."""
    masks = changed_masks(old, new)
    return delta_from_masks(
        new, masks, from_version=from_version, to_version=to_version,
        base_sha256=base_sha256 or checkpoint_sha256(old),
        target_sha256=target_sha256 or checkpoint_sha256(new),
    )


# ----------------------------------------------------------------------- apply

def apply_delta(
    state: dict[str, torch.Tensor], delta: CheckpointDelta, *, verify: bool = True
) -> None:
    """Overwrite ``state`` in place (tensors must be contiguous), then check the
    rebuilt checkpoint's SHA-256 unless ``verify`` is False."""
    for p in delta.patches:
        t = state.get(p.name)
        if t is None or _dtype_name(t) != p.dtype or tuple(t.shape) != p.shape:
            raise DeltaError(f"{p.name}: not in the checkpoint with dtype {p.dtype} shape {p.shape}")
        if not t.is_contiguous():
            raise DeltaError(f"{p.name}: tensor must be contiguous to patch in place")
        if p.whole:
            src = torch.from_numpy(np.ascontiguousarray(p.values))
            t.reshape(-1).view(torch.uint8).copy_(src.to(t.device))
        else:
            idx = torch.from_numpy(p.indices).to(t.device)
            vals = torch.from_numpy(p.values.view(np.int16)).to(t.device)
            t.reshape(-1).view(torch.int16)[idx] = vals
    if verify:
        verify_checkpoint(state, delta.target_sha256)


# --------------------------------------------------------------------- compose

def _merge_sparse(a: TensorPatch, b: TensorPatch) -> TensorPatch:
    keep = ~np.isin(a.indices, b.indices, assume_unique=True)
    idx = np.concatenate([a.indices[keep], b.indices])
    vals = np.concatenate([a.values[keep], b.values])
    order = np.argsort(idx, kind="stable")
    return TensorPatch(b.name, b.dtype, b.shape, idx[order], vals[order])


def _apply_to_whole(a: TensorPatch, b: TensorPatch) -> TensorPatch:
    raw = a.values.copy()
    raw.view(np.uint16)[b.indices] = b.values
    return TensorPatch(b.name, b.dtype, b.shape, None, raw)


def compose_deltas(first: CheckpointDelta, second: CheckpointDelta) -> CheckpointDelta:
    """The cumulative delta first.from -> second.to (union, latest values)."""
    if first.to_version != second.from_version or first.target_sha256 != second.base_sha256:
        raise DeltaError("deltas are not consecutive")
    a_by_name = {p.name: p for p in first.patches}
    b_by_name = {p.name: p for p in second.patches}
    patches = []
    for name in sorted(set(a_by_name) | set(b_by_name)):
        a, b = a_by_name.get(name), b_by_name.get(name)
        if a is not None and b is not None and (a.dtype, a.shape) != (b.dtype, b.shape):
            raise DeltaError(f"{name}: dtype or shape differs between deltas")
        if a is None or (b is not None and b.whole):
            patches.append(b)
        elif b is None:
            patches.append(a)
        elif a.whole:
            patches.append(_apply_to_whole(a, b))
        else:
            patches.append(_merge_sparse(a, b))
    return CheckpointDelta(first.from_version, second.to_version, first.base_sha256,
                           second.target_sha256, tuple(patches))


# ------------------------------------------------------------------ wire format

def _shuffle(arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    width = arr.dtype.itemsize
    return arr.view(np.uint8).reshape(-1, width).T.tobytes() if width > 1 else arr.tobytes()


def _unshuffle(raw: bytes, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    planes = np.frombuffer(raw, dtype=np.uint8)
    if dtype.itemsize == 1:
        return planes.copy()
    return planes.reshape(dtype.itemsize, -1).T.copy().view(dtype).reshape(-1)


def _numel(shape) -> int:
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def _encode_indices(indices: np.ndarray, numel: int) -> tuple[str, bytes]:
    if 32 * indices.size > numel:
        mask = np.zeros(numel, dtype=bool)
        mask[indices] = True
        return "bitmask", np.packbits(mask, bitorder="little").tobytes()
    gaps = np.diff(indices, prepend=0)
    if numel <= _U32_LIMIT:
        return "gaps32", _shuffle(gaps.astype(np.uint32))
    return "gaps64", _shuffle(gaps.astype(np.uint64))


def _decode_indices(encoding: str, raw: bytes, numel: int, count: int) -> np.ndarray:
    if encoding == "bitmask":
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), count=numel, bitorder="little")
        idx = np.flatnonzero(bits).astype(np.int64)
    elif encoding in ("gaps32", "gaps64"):
        gaps = _unshuffle(raw, np.uint32 if encoding == "gaps32" else np.uint64)
        idx = np.cumsum(gaps, dtype=np.int64)
    else:
        raise DeltaError(f"unknown index encoding {encoding!r}")
    if idx.size != count or (count and (idx[-1] >= numel or np.any(np.diff(idx) <= 0))):
        raise DeltaError("corrupt index section")
    return idx


def encode_delta(delta: CheckpointDelta, *, level: int = DEFAULT_ZSTD_LEVEL) -> bytes:
    codec = pa.Codec("zstd", compression_level=level)
    entries, blobs, offset = [], [], 0

    def section(raw: bytes) -> list[int]:
        nonlocal offset
        packed = codec.compress(raw, asbytes=True) if raw else b""
        blobs.append(packed)
        offset += len(packed)
        return [offset - len(packed), len(packed), len(raw)]

    for p in delta.patches:
        entry = {"name": p.name, "dtype": p.dtype, "shape": list(p.shape)}
        if p.whole:
            entry.update(kind="whole", sections=[section(p.values.tobytes())])
        else:
            encoding, raw_idx = _encode_indices(p.indices, _numel(p.shape))
            entry.update(kind="sparse", count=int(p.values.size), index_encoding=encoding,
                         sections=[section(raw_idx), section(_shuffle(p.values))])
        entries.append(entry)
    header = json.dumps({
        "format": FORMAT_VERSION, "codec": "zstd",
        "from_version": delta.from_version, "to_version": delta.to_version,
        "base_sha256": delta.base_sha256, "target_sha256": delta.target_sha256,
        "tensors": entries,
    }, separators=(",", ":"), sort_keys=True).encode()
    return b"".join([MAGIC, struct.pack("<Q", len(header)), header, *blobs])


def decode_delta(data: bytes) -> CheckpointDelta:
    if data[:8] != MAGIC or len(data) < 16:
        raise DeltaError("not a checkpoint delta")
    (header_len,) = struct.unpack("<Q", data[8:16])
    try:
        header = json.loads(data[16:16 + header_len])
    except ValueError as exc:
        raise DeltaError("corrupt delta header") from exc
    if header.get("format") != FORMAT_VERSION or header.get("codec") != "zstd":
        raise DeltaError("unsupported delta format")
    payload = memoryview(data)[16 + header_len:]
    codec = pa.Codec("zstd")

    def section(spec) -> bytes:
        start, length, raw_length = spec
        if start < 0 or start + length > len(payload):
            raise DeltaError("delta section out of bounds")
        if raw_length == 0:
            return b""
        raw = codec.decompress(payload[start:start + length], decompressed_size=raw_length,
                               asbytes=True)
        if len(raw) != raw_length:
            raise DeltaError("delta section has the wrong size")
        return raw

    patches = []
    for entry in header["tensors"]:
        shape = tuple(int(d) for d in entry["shape"])
        if entry["kind"] == "whole":
            values = np.frombuffer(section(entry["sections"][0]), dtype=np.uint8).copy()
            patches.append(TensorPatch(entry["name"], entry["dtype"], shape, None, values))
        elif entry["kind"] == "sparse" and entry["dtype"] == "bfloat16":
            count = int(entry["count"])
            idx = _decode_indices(entry["index_encoding"], section(entry["sections"][0]),
                                  _numel(shape), count)
            values = _unshuffle(section(entry["sections"][1]), np.uint16)
            if values.size != count:
                raise DeltaError("corrupt value section")
            patches.append(TensorPatch(entry["name"], "bfloat16", shape, idx, values))
        else:
            raise DeltaError(f"unknown tensor entry kind {entry.get('kind')!r}")
    return CheckpointDelta(int(header["from_version"]), int(header["to_version"]),
                           str(header["base_sha256"]), str(header["target_sha256"]),
                           tuple(patches))


# --------------------------------------------------------------------- anchors

def write_anchor_shards(
    state: Mapping[str, torch.Tensor], folder: Path, *, shard_bytes: int = ANCHOR_SHARD_BYTES
) -> list[Path]:
    """Full checkpoint as safetensors shards (tensors in name order)."""
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, torch.Tensor]] = [{}]
    size = 0
    for name in sorted(state):
        t = state[name]
        nbytes = t.numel() * t.element_size()
        if shards[-1] and size + nbytes > shard_bytes:
            shards.append({})
            size = 0
        shards[-1][name] = t
        size += nbytes
    paths = []
    for i, shard in enumerate(shards, start=1):
        path = folder / f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(path))
        paths.append(path)
    return paths


def load_anchor_shards(paths: list[Path]) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for path in paths:
        state.update(load_file(str(path)))
    return state
