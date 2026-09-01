"""JSON decoding helpers for durable and protocol-bound records."""

from __future__ import annotations

import json
from typing import Any


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = member
    return value


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_json_loads(raw: str | bytes | bytearray) -> Any:
    """Decode canonical-value JSON for durable protocol records.

    Python's default decoder silently keeps the last value for a duplicate
    key and accepts the non-standard constants NaN and Infinity. Both are
    unsafe for durable protocol records because readers can bind or validate
    different meanings for the same bytes.
    """

    return json.loads(
        raw,
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
    )


__all__ = ["strict_json_loads"]
