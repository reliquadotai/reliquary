"""RolloutHashSet — per-rollout content dedup across a cooldown horizon.

A miner that re-submits a rollout whose token content matches one already
entered in a sealed batch within the retention window is rejected with
``RejectReason.HASH_DUPLICATE``. Mirrors the lifecycle of
``reliquary.validator.cooldown.CooldownMap``: in-memory set, rebuilt at
validator startup from the recent R2 archive payloads.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable


_LOGICAL_GROUP_DOMAIN = b"reliquary/logical-group/v1\x00"


def _uint_bytes(value: Any, width: int, field: str) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        return value.to_bytes(width, "big", signed=False)
    except OverflowError as exc:
        raise ValueError(f"{field} does not fit in {width} bytes") from exc


def _length_prefixed(value: bytes, field: str) -> bytes:
    return _uint_bytes(len(value), 4, f"{field} length") + value


def compute_logical_group_hash(request: Any) -> bytes:
    """Hash the validator-owned economic identity of one submitted group.

    The digest binds the prompt, ordered environments and ordered committed
    token streams. It intentionally excludes miner-controlled wrappers such as
    nonce, Merkle root, claimed rewards, commitment metadata and signatures:
    changing those fields must not mint another claim on the same generation.
    Reservation scope is applied by ``GrpoWindowBatcher``. Legacy selection
    scopes this digest per hotkey. Difficulty-auction selection instead allows
    one economic claim per operator and prompt, independent of token/wrapper
    variation, so additional hotkeys cannot mint additional auction tickets.
    """
    h = hashlib.sha256()
    h.update(_LOGICAL_GROUP_DOMAIN)
    h.update(_uint_bytes(request.prompt_idx, 8, "prompt_idx"))
    h.update(_uint_bytes(len(request.rollouts), 4, "rollout count"))

    for index, rollout in enumerate(request.rollouts):
        h.update(_uint_bytes(index, 4, "rollout index"))
        env_name = rollout.env_name
        if not isinstance(env_name, str):
            raise ValueError("env_name must be a string")
        h.update(_length_prefixed(env_name.encode("utf-8"), "env_name"))

        try:
            tokens = rollout.commit["tokens"]
        except (KeyError, TypeError) as exc:
            raise ValueError("commit.tokens must be present") from exc
        if not isinstance(tokens, (list, tuple)):
            raise ValueError("commit.tokens must be a sequence")
        h.update(_uint_bytes(len(tokens), 4, "token count"))
        for token in tokens:
            h.update(_uint_bytes(token, 4, "token id"))

    return h.digest()


def compute_rollout_hash(tokens: Iterable[int]) -> bytes:
    """Return SHA256 digest of *tokens* packed as big-endian uint32.

    Deterministic over Python implementations: each int is serialised as a
    fixed 4-byte big-endian unsigned integer and concatenated before
    hashing. Rejects negative values (vocab token ids are always
    non-negative; a negative slipping in here means upstream corruption).
    """
    h = hashlib.sha256()
    for t in tokens:
        if t < 0:
            raise ValueError(f"compute_rollout_hash: negative token id {t}")
        h.update(int(t).to_bytes(4, "big", signed=False))
    return h.digest()


class RolloutHashSet:
    """Per-rollout content set with a sliding retention horizon.

    Membership tested via ``__contains__``. Entries older than
    ``retention_windows`` are dropped via ``prune``.
    """

    def __init__(self, retention_windows: int) -> None:
        if retention_windows < 0:
            raise ValueError("retention_windows must be non-negative")
        self._retention_windows = retention_windows
        self._entries: dict[bytes, int] = {}

    def add(self, h: bytes, window: int) -> None:
        if window < 0:
            raise ValueError("window must be non-negative")
        # Keep the most recent window for any given hash.
        prev = self._entries.get(h, -1)
        if window > prev:
            self._entries[h] = window

    def __contains__(self, h: bytes) -> bool:
        return h in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def prune(self, current_window: int) -> None:
        """Drop entries whose window is older than the retention horizon.

        An entry at ``window=W`` survives while
        ``current_window - W < retention_windows``. At equality the entry
        is dropped — same half-open interval semantics as ``CooldownMap``.
        """
        if self._retention_windows == 0:
            self._entries.clear()
            return
        horizon = current_window - self._retention_windows
        self._entries = {
            h: w for h, w in self._entries.items() if w > horizon
        }

    def rebuild_from_history(
        self, archives: list[dict], current_window: int,
    ) -> None:
        """Replace state from a list of archived window payloads.

        Each archive must carry ``window_start`` (int) and ``batch`` (list
        of selected submissions). Batch submissions carry ``rollouts``. Each
        rollout either has an explicit ``hash`` (hex string) — used directly
        — or only ``tokens`` (list[int]), in which case the hash is recomputed
        via :func:`compute_rollout_hash`. Newer archives may also include
        rewarded ``runners_up`` entries with ``rollout_hashes``; those are
        indexed too because rewarded runner content was paid by the window.

        Archives whose ``window_start`` is older than the retention horizon
        relative to ``current_window`` are skipped — same semantics as
        :meth:`prune`.
        """
        self._entries.clear()
        horizon = current_window - self._retention_windows
        for archive in archives:
            w = int(archive["window_start"])
            if w <= horizon:
                continue
            for sub in archive.get("batch", []):
                for rollout in sub.get("rollouts", []):
                    h_hex = rollout.get("hash")
                    if h_hex is not None:
                        h = bytes.fromhex(h_hex)
                    else:
                        h = compute_rollout_hash(rollout["tokens"])
                    self.add(h, w)
            for sub in archive.get("runners_up", []):
                if not sub.get("rewarded", False):
                    continue
                for h_hex in sub.get("rollout_hashes", []):
                    self.add(bytes.fromhex(h_hex), w)
