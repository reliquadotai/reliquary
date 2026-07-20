"""Cryptographic primitives for GRAIL protocol.

Pure, deterministic functions for:
- Pseudorandom function (PRF) with domain separation
- Random sketch vector generation from randomness
- Deterministic index selection for proof challenges
- Modular inner product for hidden state sketches
- Proof package creation

These functions have no side effects and are easily unit-testable.
"""

from __future__ import annotations

import hashlib
import logging
import random
import struct
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore

from reliquary.constants import RNG_LABEL

logger = logging.getLogger(__name__)


def prf(label: bytes, *parts: bytes, out_bytes: int) -> bytes:
    """Pseudorandom function using SHA-256 in counter mode for arbitrary output length.

    Args:
        label: Domain separation label
        *parts: Variable number of byte strings to include in PRF input
        out_bytes: Number of output bytes required

    Returns:
        Deterministic pseudorandom bytes of length out_bytes

    Raises:
        ValueError: If out_bytes is negative or too large
        TypeError: If inputs are not bytes
    """
    if out_bytes < 0:
        raise ValueError(f"out_bytes must be non-negative, got {out_bytes}")
    if out_bytes > 2**16:
        raise ValueError(f"out_bytes too large: {out_bytes} (max 65536)")
    if out_bytes == 0:
        return b""

    if not isinstance(label, bytes):
        raise TypeError(f"label must be bytes, got {type(label).__name__}")
    for i, part in enumerate(parts):
        if not isinstance(part, bytes):
            raise TypeError(f"parts[{i}] must be bytes, got {type(part).__name__}")

    # SECURITY: Use ONLY SHAKE256 — no fallback. Having two code paths
    # (SHAKE256 vs SHA256 counter mode) that produce different outputs is
    # dangerous: if miner and validator take different paths, all proofs
    # break. SHAKE256 is available in Python 3.6+ via hashlib.
    #
    # Each part is length-prefixed (4-byte big-endian) to prevent ambiguity.
    # Without length-prefixing, prf(label, b"a||b") and
    # prf(label, b"a", b"", b"b") could collide.
    shake = hashlib.shake_256()
    shake.update(len(label).to_bytes(4, "big"))
    shake.update(label)
    for part in parts:
        shake.update(len(part).to_bytes(4, "big"))
        shake.update(part)
    return shake.digest(out_bytes)


def r_vec_from_randomness(rand_hex: str, d_model: int) -> torch.Tensor:  # type: ignore[misc]
    """Generate random projection vector from drand randomness.

    Takes drand randomness (32 bytes hex) and expands it deterministically
    into a d_model-dimensional vector using a PRF.
    """
    if torch is None:
        raise ImportError("torch is required for r_vec_from_randomness")

    if not hasattr(r_vec_from_randomness, "_cache"):
        r_vec_from_randomness._cache = {}  # type: ignore[attr-defined]

    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    if d_model > 100000:
        raise ValueError(f"d_model too large: {d_model} (max 100000)")
    if not rand_hex:
        raise ValueError("rand_hex cannot be empty")

    clean_hex = rand_hex.strip().replace("0x", "").replace("0X", "")
    if not clean_hex:
        raise ValueError(f"Empty randomness hex string after cleaning: '{rand_hex}'")
    if len(clean_hex) % 2 != 0:
        clean_hex = "0" + clean_hex

    cache_key = (clean_hex, d_model)
    cache: dict[tuple[str, int], torch.Tensor] = cast(
        dict[tuple[str, int], torch.Tensor],
        getattr(r_vec_from_randomness, "_cache", {}),
    )
    if cache_key in cache:
        return cache[cache_key].clone()

    try:
        raw = prf(
            RNG_LABEL["sketch"],
            bytes.fromhex(clean_hex),
            out_bytes=4 * d_model,
        )
    except ValueError as e:
        raise ValueError(
            f"Invalid hex string for randomness: '{rand_hex}' -> '{clean_hex}': {e}"
        ) from e

    try:
        import numpy as np

        ints_array = np.frombuffer(raw, dtype=">i4").astype(np.int32, copy=False)
        tensor = torch.from_numpy(ints_array.copy())
    except ImportError:
        ints = struct.unpack(">" + "i" * d_model, raw)
        tensor = torch.tensor(ints, dtype=torch.int32)

    if len(cache) < 100:
        cache[cache_key] = tensor.clone()
        r_vec_from_randomness._cache = cache  # type: ignore[attr-defined]

    return tensor


def indices_from_root(tokens: list[int], rand_hex: str, seq_len: int, k: int) -> list[int]:
    """Generate deterministic indices for proof verification."""
    from reliquary.protocol.tokens import int_to_bytes

    if k > seq_len:
        raise ValueError(f"Cannot sample {k} indices from sequence of length {seq_len}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not tokens:
        raise ValueError("tokens list cannot be empty")

    tokens_bytes = b"".join(int_to_bytes(token) for token in tokens)
    tokens_hash = hashlib.sha256(tokens_bytes).digest()

    clean_hex = rand_hex.strip().replace("0x", "").replace("0X", "")
    if not clean_hex:
        raise ValueError(f"Empty randomness hex string: '{rand_hex}'")
    if len(clean_hex) % 2 != 0:
        clean_hex = "0" + clean_hex

    try:
        material = prf(
            RNG_LABEL["open"],
            tokens_hash,
            bytes.fromhex(clean_hex),
            out_bytes=32,
        )
    except ValueError as e:
        raise ValueError(
            f"Invalid hex string for randomness: '{rand_hex}' -> '{clean_hex}': {e}"
        ) from e

    rnd = random.Random(material)

    if k < seq_len * 0.1:
        idxs = sorted(rnd.sample(range(seq_len), k))
    else:
        all_indices = list(range(seq_len))
        rnd.shuffle(all_indices)
        idxs = sorted(all_indices[:k])

    return idxs


def indices_from_root_in_range(
    tokens: list[int], rand_hex: str, start: int, end: int, k: int
) -> list[int]:
    """Generate deterministic indices restricted to a half-open range [start, end)."""
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    if end < start:
        raise ValueError(f"end must be >= start, got start={start}, end={end}")

    length = end - start
    if length <= 0:
        return []

    k_eff = min(k, length)
    rel = indices_from_root(tokens, rand_hex, length, k_eff)
    return [start + i for i in rel]
