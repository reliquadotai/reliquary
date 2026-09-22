"""What one corpus generation job declares, and the one rule for reading it.

A job is not an ``Environment``: that protocol couples prompt generation to a
reward because in RL the reward is the product. Here the grader only decides
corpus membership, never payment, so the two are independent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

JOB_SCHEMA = "reliquary/corpus-job/v1"

PROMPT_ORDER_MINER_WALK = "miner_walk"
PROMPT_ORDER_FREE = "free"
PROMPT_ORDERS = frozenset({PROMPT_ORDER_MINER_WALK, PROMPT_ORDER_FREE})

JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_new_tokens", "max_new_tokens", "n")
_FILTER_FIELDS = ("grader_id", "threshold")
_JOB_FIELDS = (
    "schema",
    "job_id",
    "checkpoint_repo",
    "checkpoint_revision",
    "checkpoint_sha256",
    "prompt_source",
    "prompt_count",
    "renderer_id",
    "sampling",
    "slots_per_prompt",
    "filter",
    "prompt_order",
    "deadline_round",
)


class JobError(ValueError):
    """The manifest does not describe a job this code can run."""


@dataclass(frozen=True, slots=True)
class Sampling:
    """Imposed by the job, never chosen by the miner."""

    temperature: float
    top_p: float
    top_k: int
    min_new_tokens: int
    max_new_tokens: int
    n: int


@dataclass(frozen=True, slots=True)
class Filter:
    """The grader that decides corpus membership. It never decides payment."""

    grader_id: str
    threshold: float


@dataclass(frozen=True, slots=True)
class JobSpec:
    job_id: str
    checkpoint_repo: str
    checkpoint_revision: str
    checkpoint_sha256: str
    prompt_source: str
    prompt_count: int
    renderer_id: str
    sampling: Sampling
    slots_per_prompt: int
    filter: Filter | None
    prompt_order: str
    deadline_round: int | None

    @property
    def total_slots(self) -> int:
        return self.prompt_count * self.slots_per_prompt

    @property
    def rejection_sampling(self) -> bool:
        """The flag: a job with a filter keeps only what the grader accepts."""
        return self.filter is not None

    def to_contract(self) -> dict[str, Any]:
        """A detached, JSON-native view, stable enough to hash and sign."""
        return {
            "schema": JOB_SCHEMA,
            "job_id": self.job_id,
            "checkpoint_repo": self.checkpoint_repo,
            "checkpoint_revision": self.checkpoint_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "prompt_source": self.prompt_source,
            "prompt_count": self.prompt_count,
            "renderer_id": self.renderer_id,
            "sampling": {
                "temperature": self.sampling.temperature,
                "top_p": self.sampling.top_p,
                "top_k": self.sampling.top_k,
                "min_new_tokens": self.sampling.min_new_tokens,
                "max_new_tokens": self.sampling.max_new_tokens,
                "n": self.sampling.n,
            },
            "slots_per_prompt": self.slots_per_prompt,
            "filter": (
                None
                if self.filter is None
                else {
                    "grader_id": self.filter.grader_id,
                    "threshold": self.filter.threshold,
                }
            ),
            "prompt_order": self.prompt_order,
            "deadline_round": self.deadline_round,
        }


def _text(raw: Mapping[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value or value.strip() != value:
        raise JobError(f"{field} must be non-empty trimmed text, got {value!r}")
    return value


def _positive_int(raw: Mapping[str, Any], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise JobError(f"{field} must be a positive whole number, got {value!r}")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JobError(f"{field} must be a number, got {value!r}")
    return float(value)


def _parse_sampling(raw: Any) -> Sampling:
    if not isinstance(raw, Mapping):
        raise JobError("sampling must be an object")
    unknown = set(raw) - set(_SAMPLING_FIELDS)
    if unknown:
        raise JobError(f"sampling has unknown fields: {sorted(unknown)}")
    missing = [f for f in _SAMPLING_FIELDS if f not in raw]
    if missing:
        raise JobError(f"sampling is missing: {', '.join(missing)}")
    temperature = _number(raw["temperature"], "sampling.temperature")
    if temperature <= 0:
        raise JobError(f"sampling.temperature must be positive, got {temperature}")
    top_p = _number(raw["top_p"], "sampling.top_p")
    if not 0.0 < top_p <= 1.0:
        raise JobError(f"sampling.top_p must be in (0, 1], got {top_p}")
    top_k = raw["top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise JobError(f"sampling.top_k must be a non-negative whole number, got {top_k!r}")
    max_new_tokens = _positive_int(raw, "max_new_tokens")
    # The floor is what a cursor step costs: a job that wants no floor sets 1.
    min_new_tokens = _positive_int(raw, "min_new_tokens")
    if min_new_tokens > max_new_tokens:
        raise JobError(
            f"sampling.min_new_tokens {min_new_tokens} exceeds max_new_tokens {max_new_tokens}"
        )
    return Sampling(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        n=_positive_int(raw, "n"),
    )


def _parse_filter(raw: Any) -> Filter | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise JobError("filter must be an object or null")
    unknown = set(raw) - set(_FILTER_FIELDS)
    if unknown:
        raise JobError(f"filter has unknown fields: {sorted(unknown)}")
    missing = [f for f in _FILTER_FIELDS if f not in raw]
    if missing:
        raise JobError(f"filter is missing: {', '.join(missing)}")
    return Filter(
        grader_id=_text(raw, "grader_id"),
        threshold=_number(raw["threshold"], "filter.threshold"),
    )


def parse_job(raw: Mapping[str, Any]) -> JobSpec:
    """Read one manifest, or refuse it naming exactly what is wrong."""
    if not isinstance(raw, Mapping):
        raise JobError("a job manifest must be an object")
    unknown = set(raw) - set(_JOB_FIELDS)
    if unknown:
        raise JobError(f"unknown job fields: {sorted(unknown)}")
    missing = [f for f in _JOB_FIELDS if f not in raw]
    if missing:
        raise JobError(f"job is missing: {', '.join(missing)}")
    if raw["schema"] != JOB_SCHEMA:
        raise JobError(f"unsupported job schema {raw['schema']!r}")

    job_id = _text(raw, "job_id")
    if not JOB_ID_RE.match(job_id):
        raise JobError(f"unusable job id {job_id!r}")

    checkpoint_sha256 = _text(raw, "checkpoint_sha256")
    if not _SHA256_RE.match(checkpoint_sha256):
        raise JobError("checkpoint_sha256 must be 64 lowercase hex characters")

    prompt_order = _text(raw, "prompt_order")
    if prompt_order not in PROMPT_ORDERS:
        raise JobError(f"unknown prompt order {prompt_order!r}")

    deadline_round = raw["deadline_round"]
    if deadline_round is not None and (
        isinstance(deadline_round, bool)
        or not isinstance(deadline_round, int)
        or deadline_round < 0
    ):
        raise JobError(f"deadline_round must be a non-negative round or null, got {deadline_round!r}")

    return JobSpec(
        job_id=job_id,
        checkpoint_repo=_text(raw, "checkpoint_repo"),
        checkpoint_revision=_text(raw, "checkpoint_revision"),
        checkpoint_sha256=checkpoint_sha256,
        prompt_source=_text(raw, "prompt_source"),
        prompt_count=_positive_int(raw, "prompt_count"),
        renderer_id=_text(raw, "renderer_id"),
        sampling=_parse_sampling(raw["sampling"]),
        slots_per_prompt=_positive_int(raw, "slots_per_prompt"),
        filter=_parse_filter(raw["filter"]),
        prompt_order=prompt_order,
        deadline_round=deadline_round,
    )
