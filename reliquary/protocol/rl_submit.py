"""Wire format of an async RL submission (rl-async-design.md §4.4; implementation map §3).

A group is the lease's ``GROUP_ROLLOUTS`` rollouts, every index present, each carrying only its completion token
ids (the validator renders the prompt itself) as ``base64(zlib(uint32 little-endian))``, plus the BFT force span
(completion-relative) where the environment forces one. No rewards, logprobs or proofs: the validator grades and
verifies on its own. ``rollouts_digest`` (over the rollouts exactly as sent) is what the miner signs
(``signatures.build_rl_submit_binding``), so the validator checks the signature before it inflates anything.

Episodes (multi-turn, signed sandboxes; design §4.10). A rollout of an episode env also carries ``episode``:
``{"spans": [[start, end], ...], "stop": str, "transcript": {...}}``. ``tokens`` is then everything after the
validator's prompt render (model turns, tool observations and renderer markup), ``spans`` the model turns
(completion-relative, ordered, non-overlapping), ``stop`` the stop the harness admits (``episode_commit``) and
``transcript`` the gateway-signed session transcript whose token names the engagement
``rl-lease:{lease_id}:{index}``. The digest binds the episode by ``episode_digest``; a single-turn rollout's
digest is unchanged.

Body size (``/rl/submit``). A single-turn group's body is at most ``MAX_SUBMIT_BODY_BYTES`` (8 MiB). An episode
group carries 16 signed transcripts (each up to the gateway's ``MAX_TRANSCRIPT_BYTES``): its body may reach
``MAX_EPISODE_SUBMIT_BODY_BYTES`` (64 MiB) once inflated, on a validator serving episode envs. The miner sends an
episode group gzipped (``Content-Encoding: gzip``, ``gzip_body``; transcripts are JSON and compress well); the
validator inflates it within the cap (``inflate_body``: no decompression bomb). The signature is over the
rollouts, never over the encoding.

Every submission names its hotkey in the ``HOTKEY_HEADER`` header (the body's ``miner_hotkey``): the validator
checks its registration before it reads or inflates the body, and refuses a body whose hotkey differs.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import sys
import zlib
from array import array

GROUP_ROLLOUTS = 16          # design §4.4: 16 draws per lease, all submitted (the new run's profile)
MAX_SUBMIT_BODY_BYTES = 8 * 1024 * 1024
MAX_EPISODE_SUBMIT_BODY_BYTES = 64 * 1024 * 1024
GZIP = "gzip"
HOTKEY_HEADER = "X-Reliquary-Hotkey"
_TOKEN_BYTES = 4
_DIGEST_DOMAIN = b"reliquary/rl-submit-rollouts/v1"
_EPISODE_DOMAIN = b"reliquary/rl-submit-episode/v1"
EPISODE_KEYS = frozenset({"spans", "stop", "transcript"})


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


def episode_digest(episode) -> bytes:
    """SHA-256 of an episode as sent (canonical JSON: sorted keys, no spaces, ASCII)."""
    body = json.dumps(episode, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(_EPISODE_DOMAIN + body.encode("ascii")).digest()


def episode_spans(episode, completion_length: int) -> tuple[tuple[int, int], ...]:
    """The model spans of an episode as sent, read strictly (ValueError otherwise): ``episode`` has exactly
    ``EPISODE_KEYS``, a string stop, a mapping transcript and integer pairs, ordered, non-overlapping, non-empty,
    inside the completion, the first starting at 0 (the first model turn follows the prompt) and the last ending
    at its end (an episode ends on a model turn)."""
    if not isinstance(episode, dict) or set(episode) != EPISODE_KEYS:
        raise ValueError(f"an episode is exactly {sorted(EPISODE_KEYS)}")
    if not isinstance(episode["stop"], str) or not isinstance(episode["transcript"], dict):
        raise ValueError("episode: a string stop and a transcript object")
    raw = episode["spans"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("episode: at least one model span")
    spans, previous = [], 0
    for pair in raw:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(type(v) is int for v in pair)
                or pair[0] < previous or pair[1] <= pair[0] or pair[1] > completion_length):
            raise ValueError("episode spans: ordered, non-overlapping integer pairs inside the completion")
        spans.append((pair[0], pair[1]))
        previous = pair[1]
    if spans[0][0] != 0 or spans[-1][1] != completion_length:
        raise ValueError("episode spans: the completion starts and ends on a model turn")
    return tuple(spans)


def gzip_body(raw: bytes) -> bytes:
    import gzip

    return gzip.compress(raw, compresslevel=6, mtime=0)


def inflate_body(raw: bytes, *, cap: int) -> bytes:
    """A gzip body inflated, ValueError when it is not one gzip stream or inflates past ``cap`` bytes."""
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out = inflater.decompress(raw, cap + 1)
    except zlib.error as exc:
        raise ValueError("the body is not gzip") from exc
    if len(out) > cap or inflater.unconsumed_tail:
        raise OverflowError(f"the body inflates past {cap} bytes")
    if not inflater.eof or inflater.unused_data:
        raise ValueError("the body is not exactly one gzip stream")
    return out


def rollouts_digest(rollouts) -> bytes:
    """What the miner signs: ``rollouts`` as sent, (index, encoded tokens, force span or None[, episode or None])
    in index order."""
    h = hashlib.sha256(_DIGEST_DOMAIN)
    h.update(len(rollouts).to_bytes(4, "big"))
    for index, tokens, span, *rest in rollouts:
        body = str(tokens).encode("ascii")
        h.update(int(index).to_bytes(4, "big") + len(body).to_bytes(8, "big") + body)
        if span is None:
            h.update(b"\x00")
        else:
            h.update(b"\x01" + int(span[0]).to_bytes(8, "big") + int(span[1]).to_bytes(8, "big"))
        episode = rest[0] if rest else None
        if episode is not None:
            h.update(b"\x02" + episode_digest(episode))
    return h.digest()


__all__ = ["EPISODE_KEYS", "GROUP_ROLLOUTS", "GZIP", "HOTKEY_HEADER", "MAX_EPISODE_SUBMIT_BODY_BYTES", "MAX_SUBMIT_BODY_BYTES",
           "decode_tokens", "encode_tokens", "episode_digest", "episode_spans", "gzip_body", "inflate_body",
           "rollouts_digest"]
