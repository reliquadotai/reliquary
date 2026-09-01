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
    hashing. Rejects non-integer, negative, and wider-than-uint32 values
    because accepting a normalized representation would make durable replay
    ambiguous.
    """
    h = hashlib.sha256()
    for t in tokens:
        h.update(_uint_bytes(t, 4, "token id"))
    return h.digest()


def _canonical_rollout_digest(value: object, field: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be 64 lowercase hex characters")
    return bytes.fromhex(value)


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
        Every batch rollout must include its archived ``tokens``. If it also
        carries a stored ``hash``, that digest must be canonical lowercase
        SHA-256 and match a fresh digest of those tokens before it is indexed.
        Older archives without ``hash`` remain supported by recomputing it.
        Newer archives may also include rewarded ``runners_up`` entries with
        ``rollout_hashes``; those legacy metadata-only digests have no archived
        tokens to cross-check, so they are accepted only in canonical form.

        Archives whose ``window_start`` is older than the retention horizon
        relative to ``current_window`` are skipped — same semantics as
        :meth:`prune`.
        """
        restored: dict[bytes, int] = {}

        def remember(digest: bytes, window: int) -> None:
            previous = restored.get(digest, -1)
            if window > previous:
                restored[digest] = window

        horizon = current_window - self._retention_windows
        for archive in archives:
            if not isinstance(archive, dict):
                raise ValueError("archive must be an object")
            w = archive.get("window_start")
            if type(w) is not int or w < 0:
                raise ValueError(
                    "archive window_start must be a non-negative integer"
                )
            if w <= horizon:
                continue
            batch = archive.get("batch", [])
            if not isinstance(batch, list):
                raise ValueError("archive batch must be a list")
            for sub in batch:
                if not isinstance(sub, dict):
                    raise ValueError("archived batch submission must be an object")
                rollouts = sub.get("rollouts", [])
                if not isinstance(rollouts, list):
                    raise ValueError("archived rollouts must be a list")
                for rollout in rollouts:
                    if not isinstance(rollout, dict):
                        raise ValueError("archived rollout must be an object")
                    tokens = rollout.get("tokens")
                    if not isinstance(tokens, list):
                        raise ValueError("archived rollout tokens must be a list")
                    computed = compute_rollout_hash(tokens)
                    if "hash" in rollout:
                        stored = _canonical_rollout_digest(
                            rollout["hash"], "archived rollout hash"
                        )
                        if stored != computed:
                            raise ValueError(
                                "archived rollout hash does not match tokens"
                            )
                    else:
                        stored = computed
                    remember(stored, w)
            runners_up = archive.get("runners_up", [])
            if not isinstance(runners_up, list):
                raise ValueError("archive runners_up must be a list")
            for sub in runners_up:
                if not isinstance(sub, dict):
                    raise ValueError("archived runner must be an object")
                if not sub.get("rewarded", False):
                    continue
                rollout_hashes = sub.get("rollout_hashes", [])
                if not isinstance(rollout_hashes, list):
                    raise ValueError("runner rollout_hashes must be a list")
                for h_hex in rollout_hashes:
                    remember(
                        _canonical_rollout_digest(h_hex, "runner rollout hash"),
                        w,
                    )
        self._entries = restored
