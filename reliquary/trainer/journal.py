"""Strictly-ordered consumption of the validator's training journal.

The trainer NEVER advances on a timeout: absence of both the payload and
the tombstone for cursor+stride means wait. A skipped update is always an
explicit tombstone, never a race.
"""

from __future__ import annotations

from typing import Any, Callable

from reliquary.constants import (
    FILL_CLOSED_EMISSIONS_PER_WINDOW,
    FILL_CLOSED_ENABLED,
)
from reliquary.infrastructure.training_payload_queue import (
    payload_key,
    tombstone_key,
)
from reliquary.shared.training_payload import (
    decode_tombstone,
    decode_training_payload,
    validate_training_identity,
)


RAW_JOURNAL_KEY_SPACE = "raw"
FILL_CLOSED_JOURNAL_KEY_SPACE = "fill_closed"


def active_journal_key_space() -> str:
    """Which key space THIS process reads and writes the journal in."""
    return (
        FILL_CLOSED_JOURNAL_KEY_SPACE
        if FILL_CLOSED_ENABLED
        else RAW_JOURNAL_KEY_SPACE
    )


# Module-level convenience for writers that stamp the marker beside the
# cursor. Read through ``active_journal_key_space()`` where the gate may be
# patched at runtime (tests).
ACTIVE_JOURNAL_KEY_SPACE = active_journal_key_space()


def migrate_journal_cursor(
    cursor: int, key_space: str | None,
) -> tuple[int, str]:
    """Translate a stored cursor into the ACTIVE journal key space (R25).

    Returns ``(cursor, key_space)`` in the active space, so the caller can
    persist the marker it was migrated into and a second resume from the
    same checkpoint is a no-op.

    The cutover is the whole point. Before v6 the cursor IS a window number;
    under v6 the journal is keyed ``window * FILL_CLOSED_EMISSIONS_PER_WINDOW
    + batch_index``, so resuming a raw cursor against the encoded space would
    park the trainer inside a long-finished window forever, or raise on
    ``next_entry``'s window-start comparison. That must not be a manual
    runbook step: the kind of step that gets skipped.

    A missing marker reads as ``"raw"`` -- every checkpoint published before
    this field existed is raw-keyed. An unrecognised marker raises rather
    than guessing which multiplication to apply.

    The bootstrap cursor (``RELIQUARY_TRAINER_BOOTSTRAP_CURSOR``) is NOT
    migrated: it carries no marker and is an operator's explicit statement of
    where the journal starts, so it is taken as already being in the active
    space.
    """
    if type(cursor) is not int or cursor < 0:
        raise ValueError("journal cursor must be a non-negative integer")
    if key_space is None:
        stored = RAW_JOURNAL_KEY_SPACE
    elif type(key_space) is not str or not key_space:
        raise ValueError(
            "journal key space must be absent or a non-empty string"
        )
    else:
        stored = key_space
    active = active_journal_key_space()
    known = {RAW_JOURNAL_KEY_SPACE, FILL_CLOSED_JOURNAL_KEY_SPACE}
    if stored not in known:
        raise ValueError(
            f"unknown journal key space {stored!r}; expected one of "
            + ", ".join(sorted(known))
        )
    if stored == active:
        return cursor, active
    if active == FILL_CLOSED_JOURNAL_KEY_SPACE:
        return cursor * FILL_CLOSED_EMISSIONS_PER_WINDOW, active
    return cursor // FILL_CLOSED_EMISSIONS_PER_WINDOW, active


class WindowJournal:
    """Reads the next journal entry via an injected key fetcher.

    ``fetch_fn(key) -> bytes | None`` — production uses ``r2_fetch_fn``,
    tests use ``dict.get``.
    """

    def __init__(
        self,
        fetch_fn: Callable[[str], bytes | None],
        *,
        expected_identity: dict[str, Any] | None = None,
    ) -> None:
        self._fetch = fetch_fn
        self._expected_identity = dict(expected_identity or {})

    def next_entry(
        self, cursor: int, *, stride: int
    ) -> tuple[str, Any] | None:
        target = int(cursor) + int(stride)
        # v6 only (R13). Under FILL_CLOSED_ENABLED, ``target`` is not a
        # window number -- it is the ENCODED journal key
        # (window_start * FILL_CLOSED_EMISSIONS_PER_WINDOW + batch_index,
        # see encoded_window_journal_key), because a still-open window
        # writes up to FILL_CLOSED_EMISSIONS_PER_WINDOW payloads under
        # consecutive keys. The payload's own ``window_start`` field
        # still carries the REAL window number (train_step's LR/step
        # bookkeeping needs the real number, not an inflated encoded
        # one), so the two are compared through the SAME encoding rather
        # than for raw equality. The fetch key itself needs no
        # translation either way: ``payload_key``/``tombstone_key`` are
        # opaque addressing, and the writer already computed this exact
        # ``target`` value via ``encoded_window_journal_key``.
        expected_window_start = (
            target // FILL_CLOSED_EMISSIONS_PER_WINDOW
            if FILL_CLOSED_ENABLED else target
        )
        data = self._fetch(payload_key(target))
        if data is not None:
            decoded = decode_training_payload(data)
            if decoded.window_start != expected_window_start:
                raise ValueError("training payload window differs from journal key")
            if self._expected_identity:
                validate_training_identity(
                    decoded.training_identity,
                    self._expected_identity,
                    artifact=f"training payload for window {target}",
                )
            return "payload", decoded
        data = self._fetch(tombstone_key(target))
        if data is not None:
            tombstone = decode_tombstone(data)
            tombstone_window = tombstone.get("window_start")
            if (
                type(tombstone_window) is not int
                or tombstone_window < 0
                or tombstone_window != expected_window_start
            ):
                raise ValueError("training tombstone window differs from journal key")
            if self._expected_identity:
                validate_training_identity(
                    tombstone,
                    self._expected_identity,
                    artifact=f"training tombstone for window {target}",
                )
            return "tombstone", tombstone
        return None



def r2_fetch_fn(client: Any, bucket: str) -> Callable[[str], bytes | None]:
    """Production fetcher: boto3 GetObject, None on a missing key."""

    def fetch(key: str) -> bytes | None:
        try:
            return client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except client.exceptions.NoSuchKey:
            return None

    return fetch
