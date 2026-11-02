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

Wire format (one file per delta, format 2)::

    b"RQCKDLT1" | u64 header length | header JSON | sections

A whole tensor has one section. A sparse (bf16) tensor is cut into chunks of
``chunk_elements`` positions (2**24 by default, in the header); each chunk with a
change has its own two sections (indices local to the chunk, values), so chunks
compress and decode in parallel, which is what makes a 9B delta fast to build on
the trainer and to apply on a miner. Sections are zstd-compressed (through
pyarrow, already a dependency) after a byte shuffle (bytes of equal significance
grouped, which is what makes small index gaps and shared bf16 exponents
compress). Chunk indices are sorted u32 gaps, or a packed bitmask when more than
one position in five changed. The bytes do not depend on the thread count.
Format 1 (one index and one value section per tensor) is no longer read.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa
import torch

MAGIC = b"RQCKDLT1"
FORMAT_VERSION = 2
HASH_DOMAIN = b"reliquary-ckpt-sha256/1\n"
DEFAULT_ZSTD_LEVEL = 3
_U32_LIMIT = 2**32
DEFAULT_CHUNK_ELEMENTS = 2**24
_APPLY_SLICE = 2**21
_BITMASK_DENSITY = 5  # bitmask when more than 1 position in 5 changed
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

def _workers(workers: int | None) -> int:
    return max(1, workers if workers is not None else min(32, os.cpu_count() or 1))


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
    old: Mapping[str, torch.Tensor],
    new: Mapping[str, torch.Tensor],
    *,
    workers: int | None = None,
) -> dict[str, torch.Tensor]:
    """Per tensor, which positions changed: a flat bool mask for bf16 tensors,
    a one-element bool for the others (changed at all). Masks OR together, which
    is how a publisher tracks the union of changes since some base version."""
    _check_same_structure(old, new)

    def one(name: str) -> torch.Tensor:
        a, b = old[name], new[name]
        if b.dtype == torch.bfloat16:
            return (_bits16(a) != _bits16(b)).cpu()
        return torch.tensor([not np.array_equal(_raw_bytes(a), _raw_bytes(b))])

    names = sorted(new)
    with ThreadPoolExecutor(max_workers=_workers(workers)) as pool:
        return dict(zip(names, pool.map(one, names)))


def delta_from_masks(
    new: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    *,
    from_version: int,
    to_version: int,
    base_sha256: str,
    target_sha256: str,
    workers: int | None = None,
) -> CheckpointDelta:
    """The delta that writes ``new``'s values at every masked position."""
    if not to_version > from_version:
        raise DeltaError("a delta must move forward")
    def one(name: str) -> TensorPatch | None:
        mask, t = masks[name], new[name]
        if t.dtype == torch.bfloat16:
            if mask.numel() != t.numel():
                raise DeltaError(f"{name}: mask does not fit the tensor")
            idx = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                return None
            vals = _bits16(t).cpu()[idx]
            return TensorPatch(name, "bfloat16", tuple(t.shape),
                               idx.numpy().astype(np.int64, copy=False),
                               vals.numpy().view(np.uint16))
        if bool(mask.any()):
            return TensorPatch(name, _dtype_name(t), tuple(t.shape), None, _raw_bytes(t).copy())
        return None

    names = sorted(masks)
    with ThreadPoolExecutor(max_workers=_workers(workers)) as pool:
        patches = [p for p in pool.map(one, names) if p is not None]
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
    state: dict[str, torch.Tensor],
    delta: CheckpointDelta,
    *,
    verify: bool = True,
    workers: int | None = None,
) -> None:
    """Overwrite ``state`` in place (tensors must be contiguous), then check the
    rebuilt checkpoint's SHA-256 unless ``verify`` is False."""
    jobs = []
    for p in delta.patches:
        t = state.get(p.name)
        if t is None or _dtype_name(t) != p.dtype or tuple(t.shape) != p.shape:
            raise DeltaError(f"{p.name}: not in the checkpoint with dtype {p.dtype} shape {p.shape}")
        if not t.is_contiguous():
            raise DeltaError(f"{p.name}: tensor must be contiguous to patch in place")
        if p.whole:
            jobs.append(lambda t=t, p=p: t.reshape(-1).view(torch.uint8).copy_(
                torch.from_numpy(np.ascontiguousarray(p.values)).to(t.device)))
            continue
        flat = t.reshape(-1).view(torch.int16)
        idx = torch.from_numpy(p.indices)
        vals = torch.from_numpy(p.values.view(np.int16))
        # Positions are distinct, so slices of one tensor patch concurrently.
        for a in range(0, idx.numel(), _APPLY_SLICE):
            jobs.append(lambda flat=flat, i=idx[a:a + _APPLY_SLICE], v=vals[a:a + _APPLY_SLICE]:
                        flat.index_put_((i.to(flat.device),), v.to(flat.device)))
    with ThreadPoolExecutor(max_workers=_workers(workers)) as pool:
        for fut in [pool.submit(job) for job in jobs]:
            fut.result()
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

def _shuffle(arr: np.ndarray) -> np.ndarray:
    """Byte planes of ``arr`` (all first bytes, then all second bytes, ...)."""
    arr = np.ascontiguousarray(arr)
    width = arr.dtype.itemsize
    if width == 1:
        return arr.view(np.uint8)
    src = arr.view(np.uint8).reshape(-1, width)
    out = np.empty(arr.nbytes, dtype=np.uint8)
    planes = out.reshape(width, -1)
    for k in range(width):
        planes[k] = src[:, k]
    return out


def _unshuffle_into(raw: bytes, out: np.ndarray) -> None:
    """Inverse of ``_shuffle``, written into the contiguous array ``out``."""
    width = out.dtype.itemsize
    planes = np.frombuffer(raw, dtype=np.uint8)
    if planes.size != out.nbytes:
        raise DeltaError("delta section has the wrong size")
    if width == 1:
        out.view(np.uint8)[:] = planes
        return
    dst = out.view(np.uint8).reshape(-1, width)
    planes = planes.reshape(width, -1)
    for k in range(width):
        dst[:, k] = planes[k]


def _numel(shape) -> int:
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def _encode_indices(indices: np.ndarray, numel: int) -> tuple[str, np.ndarray]:
    """Sorted positions inside one chunk of ``numel`` (<= 2**32) elements.

    Gaps compress smaller and decode faster up to about one position in five
    (measured on uniform positions); a bitmask wins above that."""
    if _BITMASK_DENSITY * indices.size > numel:
        mask = np.zeros(numel, dtype=bool)
        mask[indices] = True
        return "bitmask", np.packbits(mask, bitorder="little")
    gaps = np.empty(indices.size, dtype=np.uint32)
    if indices.size:
        gaps[0] = indices[0]
        np.subtract(indices[1:], indices[:-1], out=gaps[1:], casting="unsafe")
    return "gaps32", _shuffle(gaps)


def _decode_indices_into(encoding: str, raw: bytes, numel: int, out: np.ndarray) -> None:
    """Positions of one chunk, written into the int64 array ``out`` (its size is
    the declared count). Checks they are strictly increasing and in range."""
    count = out.size
    if encoding == "bitmask":
        packed = np.frombuffer(raw, dtype=np.uint8)
        if packed.size != (numel + 7) // 8:
            raise DeltaError("corrupt index section")
        tail = numel % 8
        if tail and packed[-1] >> tail:
            raise DeltaError("corrupt index section")
        bits = np.unpackbits(packed, count=numel, bitorder="little").view(bool)
        idx = np.flatnonzero(bits)
        if idx.size != count:
            raise DeltaError("corrupt index section")
        out[:] = idx
    elif encoding == "gaps32":
        if len(raw) != 4 * count:
            raise DeltaError("corrupt index section")
        gaps = np.empty(count, dtype=np.uint32)
        _unshuffle_into(raw, gaps)
        if count and (np.any(gaps[1:] == 0)):
            raise DeltaError("corrupt index section")
        np.cumsum(gaps, dtype=np.int64, out=out)
        if count and out[-1] >= numel:
            raise DeltaError("corrupt index section")
    else:
        raise DeltaError(f"unknown index encoding {encoding!r}")


def _decode_indices(encoding: str, raw: bytes, numel: int, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.int64)
    _decode_indices_into(encoding, bytes(raw), numel, out)
    return out


def encode_delta(
    delta: CheckpointDelta,
    *,
    level: int = DEFAULT_ZSTD_LEVEL,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
    workers: int | None = None,
) -> bytes:
    """Wire bytes of ``delta``. Sparse tensors are cut into chunks of
    ``chunk_elements`` positions, each with its own index and value sections,
    compressed in parallel; the bytes do not depend on ``workers``."""
    if not 0 < chunk_elements <= _U32_LIMIT:
        raise DeltaError("chunk_elements must be in (0, 2**32]")
    codec = pa.Codec("zstd", compression_level=level)

    def compress(raw: np.ndarray) -> tuple[bytes, int]:
        raw = raw.view(np.uint8)
        return (codec.compress(raw, asbytes=True) if raw.size else b""), int(raw.size)

    jobs = []  # in output order: one per whole tensor, one per sparse chunk

    def whole_job(p):
        return lambda: [compress(np.ascontiguousarray(p.values))]

    def chunk_job(p, base, n, a, b):
        def run():
            encoding, raw_idx = _encode_indices(p.indices[a:b] - base, n)
            return [encoding, compress(raw_idx), compress(_shuffle(p.values[a:b]))]
        return run

    layouts = []
    for p in delta.patches:
        if p.whole:
            layouts.append(None)
            jobs.append(whole_job(p))
            continue
        numel = _numel(p.shape)
        starts = np.arange(0, numel, chunk_elements, dtype=np.int64)
        cuts = np.searchsorted(p.indices, np.append(starts, numel))
        chunks = []
        for c, base in enumerate(starts.tolist()):
            a, b = int(cuts[c]), int(cuts[c + 1])
            if a == b:
                continue
            chunks.append((c, b - a))
            jobs.append(chunk_job(p, base, min(chunk_elements, numel - base), a, b))
        layouts.append(chunks)

    with ThreadPoolExecutor(max_workers=_workers(workers)) as pool:
        results = iter(list(pool.map(lambda job: job(), jobs)))

    entries, blobs, offset = [], [], 0

    def section(packed_and_raw) -> list[int]:
        nonlocal offset
        packed, raw_length = packed_and_raw
        blobs.append(packed)
        offset += len(packed)
        return [offset - len(packed), len(packed), raw_length]

    for p, chunks in zip(delta.patches, layouts):
        entry = {"name": p.name, "dtype": p.dtype, "shape": list(p.shape)}
        if chunks is None:
            entry.update(kind="whole", sections=[section(next(results)[0])])
        else:
            rows = []
            for c, count in chunks:
                encoding, idx_sec, val_sec = next(results)
                rows.append([c, count, encoding, section(idx_sec), section(val_sec)])
            entry.update(kind="sparse", count=int(p.values.size), chunks=rows)
        entries.append(entry)
    header = json.dumps({
        "format": FORMAT_VERSION, "codec": "zstd", "chunk_elements": int(chunk_elements),
        "from_version": delta.from_version, "to_version": delta.to_version,
        "base_sha256": delta.base_sha256, "target_sha256": delta.target_sha256,
        "tensors": entries,
    }, separators=(",", ":"), sort_keys=True).encode()
    return b"".join([MAGIC, struct.pack("<Q", len(header)), header, *blobs])


def decode_delta(data: bytes, *, workers: int | None = None) -> CheckpointDelta:
    """Parse wire bytes; chunks are decompressed and decoded in parallel."""
    if data[:8] != MAGIC or len(data) < 16:
        raise DeltaError("not a checkpoint delta")
    (header_len,) = struct.unpack("<Q", data[8:16])
    try:
        header = json.loads(data[16:16 + header_len])
    except ValueError as exc:
        raise DeltaError("corrupt delta header") from exc
    if header.get("format") != FORMAT_VERSION or header.get("codec") != "zstd":
        raise DeltaError("unsupported delta format")
    chunk_elements = header.get("chunk_elements")
    if not isinstance(chunk_elements, int) or not 0 < chunk_elements <= _U32_LIMIT:
        raise DeltaError("corrupt delta header")
    payload = memoryview(data)[16 + header_len:]
    codec = pa.Codec("zstd")

    def section(spec) -> bytes:
        start, length, raw_length = (int(x) for x in spec)
        if start < 0 or length < 0 or raw_length < 0 or start + length > len(payload):
            raise DeltaError("delta section out of bounds")
        if raw_length == 0:
            return b""
        raw = codec.decompress(payload[start:start + length], decompressed_size=raw_length,
                               asbytes=True)
        if len(raw) != raw_length:
            raise DeltaError("delta section has the wrong size")
        return raw

    def whole_job(out_list, slot, spec):
        def run():
            out_list[slot] = np.frombuffer(section(spec), dtype=np.uint8).copy()
        return run

    def chunk_job(idx, vals, base, n, a, b, encoding, idx_spec, val_spec):
        def run():
            _decode_indices_into(encoding, section(idx_spec), n, idx[a:b])
            if base:
                idx[a:b] += base
            _unshuffle_into(section(val_spec), vals[a:b])
        return run

    built, jobs, wholes = [], [], {}
    for entry in header["tensors"]:
        shape = tuple(int(d) for d in entry["shape"])
        if entry["kind"] == "whole":
            built.append((entry["name"], entry["dtype"], shape, None, len(built)))
            jobs.append(whole_job(wholes, len(built) - 1, entry["sections"][0]))
        elif entry["kind"] == "sparse" and entry["dtype"] == "bfloat16":
            numel, count = _numel(shape), int(entry["count"])
            if not 0 <= count <= numel:
                raise DeltaError("corrupt delta header")
            idx = np.empty(count, dtype=np.int64)
            vals = np.empty(count, dtype=np.uint16)
            a, last = 0, -1
            for c, n_c, encoding, idx_spec, val_spec in entry["chunks"]:
                c, n_c = int(c), int(n_c)
                base = c * chunk_elements
                if c <= last or base >= numel or n_c <= 0 or a + n_c > count:
                    raise DeltaError("corrupt chunk table")
                last = c
                jobs.append(chunk_job(idx, vals, base, min(chunk_elements, numel - base),
                                      a, a + n_c, encoding, idx_spec, val_spec))
                a += n_c
            if a != count:
                raise DeltaError("corrupt chunk table")
            built.append((entry["name"], "bfloat16", shape, idx, vals))
        else:
            raise DeltaError(f"unknown tensor entry kind {entry.get('kind')!r}")

    with ThreadPoolExecutor(max_workers=_workers(workers)) as pool:
        for fut in [pool.submit(job) for job in jobs]:
            fut.result()
    patches = []
    for slot, (name, dtype, shape, idx, vals) in enumerate(built):
        if idx is None:
            patches.append(TensorPatch(name, dtype, shape, None, wholes[slot]))
        else:
            patches.append(TensorPatch(name, dtype, shape, idx, vals))
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
