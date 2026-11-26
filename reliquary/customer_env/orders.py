"""A qualified customer env version as an order's prompt sets.

Its frozen rows (``prompts.jsonl`` in the subnet bucket) are written as an OpenAI Batch file and compiled by
``orders.uploads``: the same rendering with the model's own template, the same sets, the same serving and
audit. For an evaluation each row's reference names its frozen task and the grader is the version itself,
scored on the reward host (``customer_env.scoring``).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from reliquary.customer_env import parse_env_id
from reliquary.customer_env.registry import subnet_key
from reliquary.customer_env.scoring import reference

_SAFE_ID = __import__("re").compile(r"\A[A-Za-z0-9][A-Za-z0-9._:-]{0,63}\Z")


def batch_file(rows: list[dict], sampling: Mapping, *, purpose: str) -> bytes:
    """The rows as OpenAI Batch JSONL; a key that is not a safe custom id is named by its position."""
    lines = []
    for index, row in enumerate(rows):
        messages = ([{"role": "system", "content": row["system"]}] if row.get("system") else []) + [
            {"role": "user", "content": row["prompt"]}]
        line = {"custom_id": row["key"] if _SAFE_ID.match(row["key"]) else f"row-{index}",
                "method": "POST", "url": "/v1/chat/completions", "body": {"messages": messages, **sampling}}
        if purpose == "eval":
            line["reference"] = reference(row["key"], row["prompt_sha256"])
        lines.append(json.dumps(line, sort_keys=True, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode()


async def read_rows(store, env_id: str, expected_sha256: str) -> list[dict]:
    body = await store.get_bytes(subnet_key(parse_env_id(env_id), "prompts.jsonl"))
    if body is None or hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError("customer_env_rows_unavailable: the frozen rows are missing or changed")
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def select(rows: list[dict], start: int | None, end: int | None) -> list[dict]:
    """The order's rows ``[start, end)`` of the frozen order (all of them by default)."""
    start, end = start or 0, len(rows) if end is None else end
    if not 0 <= start < end <= len(rows):
        raise ValueError(f"rows [{start}, {end}) are outside the version's {len(rows)} rows")
    return rows[start:end]


__all__ = ["batch_file", "read_rows", "select"]
