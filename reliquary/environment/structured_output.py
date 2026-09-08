"""Bounded deterministic JSON answer extraction for verifiable tasks."""

from __future__ import annotations

import json
import re
from typing import Any


MAX_COMPLETION_BYTES = 16 * 1024
MAX_CONTAINER_ITEMS = 64
MAX_DEPTH = 8
MAX_INTEGER_MAGNITUDE = (1 << 53) - 1

_JSON_FENCE = re.compile(
    r"```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```",
    flags=re.IGNORECASE | re.DOTALL,
)


class StructuredOutputError(ValueError):
    """A model answer is not in the bounded canonical JSON channel."""


def _reject_float(value: str) -> Any:
    raise StructuredOutputError(f"floating-point values are not allowed: {value}")


def _reject_constant(value: str) -> Any:
    raise StructuredOutputError(f"non-finite values are not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StructuredOutputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_json_value(value: Any, *, depth: int = 0) -> int:
    if depth > MAX_DEPTH:
        raise StructuredOutputError("JSON answer is nested too deeply")
    if value is None or isinstance(value, (str, bool)):
        return 0
    if isinstance(value, int):
        if abs(value) > MAX_INTEGER_MAGNITUDE:
            raise StructuredOutputError("JSON integer exceeds the safe bound")
        return 0
    if isinstance(value, float):
        raise StructuredOutputError("floating-point values are not allowed")
    if isinstance(value, list):
        count = len(value)
        for item in value:
            count += _validate_json_value(item, depth=depth + 1)
            if count > MAX_CONTAINER_ITEMS:
                raise StructuredOutputError("JSON answer contains too many items")
        return count
    if isinstance(value, dict):
        count = len(value)
        for key, item in value.items():
            if not isinstance(key, str):
                raise StructuredOutputError("JSON object keys must be strings")
            count += _validate_json_value(item, depth=depth + 1)
            if count > MAX_CONTAINER_ITEMS:
                raise StructuredOutputError("JSON answer contains too many items")
        return count
    raise StructuredOutputError(f"unsupported JSON value: {type(value).__name__}")


def extract_json_answer(completion: str | None) -> dict[str, Any]:
    """Extract the final fenced JSON object, or one bare whole object.

    Explanatory reasoning may precede a final fenced answer. If no JSON fence
    exists, the entire trimmed completion must be a JSON object. Duplicate
    keys, floats, non-finite constants, excessive depth/size, and trailing
    content fail closed.
    """

    if not isinstance(completion, str):
        raise StructuredOutputError("completion must be a string")
    if len(completion.encode("utf-8")) > MAX_COMPLETION_BYTES:
        raise StructuredOutputError("completion exceeds the byte limit")

    matches = list(_JSON_FENCE.finditer(completion))
    payload = matches[-1].group("body") if matches else completion.strip()
    if not payload:
        raise StructuredOutputError("JSON answer is empty")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except StructuredOutputError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StructuredOutputError("invalid JSON answer") from exc
    if not isinstance(value, dict):
        raise StructuredOutputError("JSON answer must be an object")
    _validate_json_value(value)
    return value


def canonical_json(value: Any) -> str:
    """Canonical compact representation after applying the same bounds."""

    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


__all__ = [
    "MAX_COMPLETION_BYTES",
    "StructuredOutputError",
    "canonical_json",
    "extract_json_answer",
]
