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
    epoch_marker_key,
    payload_key,
    tombstone_key,
)
from reliquary.shared.training_payload import (
    decode_checkpoint_epoch_marker,
    decode_tombstone,
    decode_training_payload,
    validate_training_identity,
)


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
            if int(tombstone.get("window_start", -1)) != expected_window_start:
                raise ValueError("training tombstone window differs from journal key")
            if self._expected_identity:
                validate_training_identity(
                    tombstone,
                    self._expected_identity,
                    artifact=f"training tombstone for window {target}",
                )
            return "tombstone", tombstone
        return None

    def checkpoint_epoch_status(self, binding: Any) -> str | None:
        """Return the durable all-lanes terminal status, or wait for it."""
        raw = self._fetch(epoch_marker_key(binding.epoch_id))
        if raw is None:
            return None
        marker = decode_checkpoint_epoch_marker(raw)
        marked = marker["checkpoint_epoch"]
        expected = (
            binding.epoch_id,
            binding.manifest_sha256,
            binding.training_run_id,
            binding.training_mode,
            binding.first_window,
            binding.window_count,
            binding.target_groups_per_environment_lane,
        )
        actual = (
            marked.epoch_id,
            marked.manifest_sha256,
            marked.training_run_id,
            marked.training_mode,
            marked.first_window,
            marked.window_count,
            marked.target_groups_per_environment_lane,
        )
        if actual != expected:
            raise ValueError("checkpoint epoch marker binding differs")
        return str(marker["status"])


def r2_fetch_fn(client: Any, bucket: str) -> Callable[[str], bytes | None]:
    """Production fetcher: boto3 GetObject, None on a missing key."""

    def fetch(key: str) -> bytes | None:
        try:
            return client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except client.exceptions.NoSuchKey:
            return None

    return fetch
