"""Wire format of an async RL submission (rl-async-design.md §4.4; implementation map §3).

A group is the lease's ``GROUP_ROLLOUTS`` rollouts, every index present, each carrying only its completion token
ids (the validator renders the prompt itself) as ``base64(zlib(uint32 little-endian))``, plus the BFT force span
(completion-relative) where the environment forces one. No rewards, logprobs or proofs: the validator grades and
verifies on its own. ``rollouts_digest`` (over the rollouts exactly as sent) is what the miner signs
(``signatures.build_rl_submit_binding``), so the validator checks the signature before it inflates anything.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import sys
import zlib
from array import array

GROUP_ROLLOUTS = 16          # design §4.4: 16 draws per lease, all submitted (the new run's profile)
_TOKEN_BYTES = 4
_DIGEST_DOMAIN = b"reliquary/rl-submit-rollouts/v1"


def _le_u32(tokens) -> bytes:
    values = array("I", tokens)
    if values.itemsize != _TOKEN_BYTES:                 # pragma: no cover - every supported platform
        raise RuntimeError("uint32 array required")
    if sys.byteorder == "big":                          # pragma: no cover
        values.byteswap()
    return values.tobytes()


def encode_tokens(tokens) -> str:
    return base64.b64encode(zlib.compress(_le_u32(tokens), 6)).decode("ascii")


def decode_tokens(text: str, *, max_tokens: int) -> tuple[int, ...]:
    """ValueError on anything but 1..``max_tokens`` uint32 ids (bounded inflate: no decompression bomb)."""
    try:
        packed = base64.b64decode(text.encode("ascii"), validate=True)
        inflater = zlib.decompressobj()
        raw = inflater.decompress(packed, max_tokens * _TOKEN_BYTES + 1)
    except (binascii.Error, UnicodeEncodeError, zlib.error) as exc:
        raise ValueError("tokens are not base64 zlib") from exc
    if not inflater.eof or inflater.unused_data or len(raw) > max_tokens * _TOKEN_BYTES:
        raise ValueError(f"tokens: at most {max_tokens} ids, one zlib stream")
    if not raw or len(raw) % _TOKEN_BYTES:
        raise ValueError("tokens: a non-empty uint32 array")
    values = array("I")
    values.frombytes(raw)
    if sys.byteorder == "big":                          # pragma: no cover
        values.byteswap()
    return tuple(values)


def rollouts_digest(rollouts) -> bytes:
    """What the miner signs: ``rollouts`` as sent, (index, encoded tokens, force span or None) in index order."""
    h = hashlib.sha256(_DIGEST_DOMAIN)
    h.update(len(rollouts).to_bytes(4, "big"))
    for index, tokens, span in rollouts:
        body = str(tokens).encode("ascii")
        h.update(int(index).to_bytes(4, "big") + len(body).to_bytes(8, "big") + body)
        if span is None:
            h.update(b"\x00")
        else:
            h.update(b"\x01" + int(span[0]).to_bytes(8, "big") + int(span[1]).to_bytes(8, "big"))
    return h.digest()


__all__ = ["GROUP_ROLLOUTS", "decode_tokens", "encode_tokens", "rollouts_digest"]
