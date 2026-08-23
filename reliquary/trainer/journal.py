"""Strictly-ordered consumption of the validator's training journal.

The trainer NEVER advances on a timeout: absence of both the payload and
the tombstone for cursor+stride means wait. A skipped update is always an
explicit tombstone, never a race.
"""

from __future__ import annotations

from typing import Any, Callable

from reliquary.infrastructure.training_payload_queue import (
    payload_key,
    tombstone_key,
)
from reliquary.shared.training_payload import (
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
        data = self._fetch(payload_key(target))
        if data is not None:
            decoded = decode_training_payload(data)
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
