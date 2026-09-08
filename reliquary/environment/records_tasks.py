"""Deterministic structured-record tasks for ``reliquaryverifiable_v1``."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, TypeVar

from reliquary.environment.structured_output import canonical_json


ENVIRONMENT_NAME = "reliquaryverifiable_v1"
TASK_FAMILY = "records_v1"
GENERATOR_VERSION = "records-v1"
CHECKER_VERSION = "records-checker-v1"
VIRTUAL_LENGTH = 1 << 25

_CATEGORIES = ("amber", "blue", "coral", "jade")
_T = TypeVar("_T")


class HashCounterRng:
    """Small SHA-256 counter stream with stable rejection sampling."""

    __slots__ = ("_seed", "_counter", "_buffer")

    def __init__(self, seed: bytes):
        self._seed = bytes(seed)
        self._counter = 0
        self._buffer = b""

    def _take(self, count: int) -> bytes:
        while len(self._buffer) < count:
            block = hashlib.sha256(
                self._seed + self._counter.to_bytes(8, "big")
            ).digest()
            self._counter += 1
            self._buffer += block
        result, self._buffer = self._buffer[:count], self._buffer[count:]
        return result

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper bound must be positive")
        byte_count = max(1, (upper.bit_length() + 7) // 8)
        limit = (1 << (8 * byte_count)) - ((1 << (8 * byte_count)) % upper)
        while True:
            candidate = int.from_bytes(self._take(byte_count), "big")
            if candidate < limit:
                return candidate % upper

    def choice(self, values: tuple[_T, ...] | list[_T]) -> _T:
        return values[self.randbelow(len(values))]

    def shuffle(self, values: list[_T]) -> None:
        for index in range(len(values) - 1, 0, -1):
            other = self.randbelow(index + 1)
            values[index], values[other] = values[other], values[index]


@dataclass(frozen=True, slots=True)
class GeneratedRecordsTask:
    prompt: str
    expected: Any
    operation_id: str
    difficulty: int


def _rng_for_index(index: int) -> HashCounterRng:
    normalized = int(index) % VIRTUAL_LENGTH
    seed = hashlib.sha256(
        f"{ENVIRONMENT_NAME}\0{GENERATOR_VERSION}\0{normalized}".encode("ascii")
    ).digest()
    return HashCounterRng(seed)


def _records(rng: HashCounterRng) -> list[dict[str, Any]]:
    count = 8 + rng.randbelow(3)
    records: list[dict[str, Any]] = []
    for index in range(count):
        records.append(
            {
                "id": f"r{index + 1:02d}",
                "category": rng.choice(_CATEGORIES),
                "score": rng.randbelow(91) + 5,
                "active": bool(rng.randbelow(2)),
            }
        )
    # Guarantee filter templates always have both matching and non-matching
    # rows, independent of the pseudorandom draw.
    records[0]["category"] = "amber"
    records[1]["category"] = "blue"
    records[0]["active"] = True
    records[1]["active"] = False
    rng.shuffle(records)
    return records


def _render_prompt(
    records: list[dict[str, Any]],
    instructions: list[str],
) -> str:
    numbered = "\n".join(
        f"{index}. {instruction}"
        for index, instruction in enumerate(instructions, start=1)
    )
    source = json.dumps(records, separators=(",", ":"), ensure_ascii=True)
    return (
        "Transform the JSON records below. Apply every operation in order.\n\n"
        f"Input records:\n```json\n{source}\n```\n\n"
        f"Operations:\n{numbered}\n\n"
        "Return the final value in exactly this answer shape, using JSON types "
        "and no additional keys:\n```json\n{\"result\": <final value>}\n```"
    )


def _filter_sort_project(
    records: list[dict[str, Any]], rng: HashCounterRng
) -> GeneratedRecordsTask:
    category = rng.choice(_CATEGORIES)
    # Ensure this chosen category is represented without making the filter a
    # no-op. At most one row is rewritten.
    if not any(row["category"] == category for row in records):
        records[0]["category"] = category
    if all(row["category"] == category for row in records):
        records[-1]["category"] = next(
            value for value in _CATEGORIES if value != category
        )
    descending = bool(rng.randbelow(2))
    selected = [row for row in records if row["category"] == category]
    selected.sort(key=lambda row: row["score"], reverse=descending)
    expected = [{"id": row["id"], "score": row["score"]} for row in selected]
    direction = "descending" if descending else "ascending"
    instructions = [
        f'Keep only records whose "category" is "{category}".',
        f'Stable-sort the remaining records by integer "score" in {direction} order.',
        'Project each record to exactly the keys "id" and "score".',
    ]
    return GeneratedRecordsTask(
        prompt=_render_prompt(records, instructions),
        expected=expected,
        operation_id="filter-sort-project-v1",
        difficulty=2,
    )


def _active_sort_rename(
    records: list[dict[str, Any]], rng: HashCounterRng
) -> GeneratedRecordsTask:
    active = bool(rng.randbelow(2))
    selected = [row for row in records if row["active"] is active]
    selected.sort(key=lambda row: (row["category"], row["id"]))
    expected = [
        {
            "id": row["id"],
            "category": row["category"],
            "value": row["score"],
        }
        for row in selected
    ]
    instructions = [
        f'Keep only records whose "active" value is {str(active).lower()}.',
        'Sort by "category" ascending, then by "id" ascending.',
        'Project to "id", "category", and "score".',
        'Rename the projected "score" key to "value".',
    ]
    return GeneratedRecordsTask(
        prompt=_render_prompt(records, instructions),
        expected=expected,
        operation_id="active-sort-rename-v1",
        difficulty=3,
    )


def _deduplicate_project(
    records: list[dict[str, Any]], rng: HashCounterRng
) -> GeneratedRecordsTask:
    del rng
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in records:
        category = str(row["category"])
        if category not in seen:
            seen.add(category)
            unique.append(row)
    expected = [
        {"category": row["category"], "score": row["score"]}
        for row in unique
    ]
    instructions = [
        'Deduplicate by "category", keeping the first input record for each category.',
        'Preserve the surviving input order.',
        'Project each surviving record to exactly "category" and "score".',
    ]
    return GeneratedRecordsTask(
        prompt=_render_prompt(records, instructions),
        expected=expected,
        operation_id="deduplicate-project-v1",
        difficulty=2,
    )


def _group_aggregate(
    records: list[dict[str, Any]], rng: HashCounterRng
) -> GeneratedRecordsTask:
    del rng
    totals: dict[str, dict[str, int]] = {}
    for row in records:
        bucket = totals.setdefault(
            str(row["category"]), {"count": 0, "score_sum": 0}
        )
        bucket["count"] += 1
        bucket["score_sum"] += int(row["score"])
    expected = [
        {
            "category": category,
            "count": totals[category]["count"],
            "score_sum": totals[category]["score_sum"],
        }
        for category in sorted(totals)
    ]
    instructions = [
        'Group records by "category".',
        'For each category compute integer "count" and integer "score_sum".',
        'Return one object per category with keys "category", "count", and "score_sum".',
        'Sort the output by "category" ascending.',
    ]
    return GeneratedRecordsTask(
        prompt=_render_prompt(records, instructions),
        expected=expected,
        operation_id="group-count-sum-v1",
        difficulty=3,
    )


def _top_project(
    records: list[dict[str, Any]], rng: HashCounterRng
) -> GeneratedRecordsTask:
    limit = 2 + rng.randbelow(3)
    ordered = sorted(records, key=lambda row: (-int(row["score"]), row["id"]))
    expected = [
        {"id": row["id"], "score": row["score"]}
        for row in ordered[:limit]
    ]
    instructions = [
        'Sort by "score" descending and use "id" ascending to break ties.',
        f"Keep only the first {limit} records.",
        'Project each retained record to exactly "id" and "score".',
    ]
    return GeneratedRecordsTask(
        prompt=_render_prompt(records, instructions),
        expected=expected,
        operation_id="top-project-v1",
        difficulty=2,
    )


_GENERATORS = (
    _filter_sort_project,
    _active_sort_rename,
    _deduplicate_project,
    _group_aggregate,
    _top_project,
)


def generate_records_task(index: int) -> GeneratedRecordsTask:
    rng = _rng_for_index(index)
    records = _records(rng)
    generator = _GENERATORS[rng.randbelow(len(_GENERATORS))]
    return generator(records, rng)


def verifier_spec(task: GeneratedRecordsTask) -> str:
    return canonical_json(
        {
            "schema": "reliquary/records-verifier/v1",
            "checker_version": CHECKER_VERSION,
            "operation_id": task.operation_id,
            "difficulty": task.difficulty,
            "expected": task.expected,
        }
    )


__all__ = [
    "CHECKER_VERSION",
    "ENVIRONMENT_NAME",
    "GENERATOR_VERSION",
    "GeneratedRecordsTask",
    "TASK_FAMILY",
    "VIRTUAL_LENGTH",
    "generate_records_task",
    "verifier_spec",
]
