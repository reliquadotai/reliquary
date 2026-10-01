"""Per-model qualification: the TOPLOC thresholds an eval job is audited with.

An executor decodes the job's own eval prompts with vLLM, verifies them with the
auditor's HF prefill, and reports every chunk's measures. The control turns that
honest band into thresholds: ``max(p99 × 1.5, floor)`` per measure, the floors
being the deployed 60/40/40. A model whose band itself exceeds the hard ceiling
is refused. The record lives in the subnet bucket, where the admin service reads
it when the job is created: the platform never hands the thresholds over itself.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Sequence
from typing import Any

QUALIFICATION_SCHEMA = "reliquary/eval-qualification/v1"
QUALIFICATION_PREFIX = "reliquary/eval/qualifications/"
QUALIFY_COMPLETIONS = 32
THRESHOLD_MARGIN = 1.5
PENDING, LEASED, QUALIFIED, REFUSED, FAILED = "pending", "leased", "qualified", "refused", "failed"
TERMINAL = frozenset({QUALIFIED, REFUSED, FAILED})
_ID_RE = re.compile(r"\A[a-z0-9][a-z0-9-]{0,62}\Z")


def _ceiling_env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def ceilings() -> dict[str, float]:
    """The honest band above which a model is refused (four times the floors
    by default; ``RELIQUARY_QUALIFY_CEILING_*`` override)."""
    return {"exp_mismatch": _ceiling_env("RELIQUARY_QUALIFY_CEILING_EXP", 240),
            "mant_mean": _ceiling_env("RELIQUARY_QUALIFY_CEILING_MANT_MEAN", 160.0),
            "mant_median": _ceiling_env("RELIQUARY_QUALIFY_CEILING_MANT_MEDIAN", 160.0)}


def validated_qualification_id(value: Any) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(f"qualification id {value!r} is not a name")
    return value


def qualification_key(qualification_id: str) -> str:
    return f"{QUALIFICATION_PREFIX}{validated_qualification_id(qualification_id)}.json"


def p99(values: Sequence[float]) -> float:
    """The nearest-rank 99th percentile."""
    if not values:
        raise ValueError("no measures")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(round(0.99 * len(ordered), 9)) - 1)]


def thresholds_from_band(chunks: Sequence[Sequence[float]], *, floor=None,
                         ceiling: dict[str, float] | None = None) -> dict:
    """``chunks`` are ``(exp_mismatches, mant_err_mean, mant_err_median)`` of
    honest completions. ``{"band", "thresholds", "refused"}``: ``refused`` names
    the measure over the ceiling, or is None."""
    if floor is None:
        from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as floor
    ceiling = ceiling or ceilings()
    if not chunks:
        raise ValueError("qualification measured no chunk")
    band = {"exp_mismatch": p99([float(c[0]) for c in chunks]),
            "mant_mean": p99([float(c[1]) for c in chunks]),
            "mant_median": p99([float(c[2]) for c in chunks]),
            "chunks": len(chunks)}
    thresholds = {
        "exp_mismatch_threshold": max(int(math.ceil(band["exp_mismatch"] * THRESHOLD_MARGIN)),
                                      int(floor.exp_mismatch_threshold)),
        "mant_mean_threshold": max(round(band["mant_mean"] * THRESHOLD_MARGIN, 6),
                                   float(floor.mant_mean_threshold)),
        "mant_median_threshold": max(round(band["mant_median"] * THRESHOLD_MARGIN, 6),
                                     float(floor.mant_median_threshold)),
    }
    refused = next((f"honest {name} p99 {band[name]:.3f} is over the ceiling {ceiling[name]}"
                    for name in ("exp_mismatch", "mant_mean", "mant_median")
                    if band[name] > ceiling[name]), None)
    return {"band": band, "thresholds": thresholds, "refused": refused}


def check_thresholds(thresholds: dict, *, floor=None) -> dict:
    """Thresholds a job may carry: every measure at or above the floor."""
    if floor is None:
        from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as floor
    exp = thresholds.get("exp_mismatch_threshold")
    mean = thresholds.get("mant_mean_threshold")
    median = thresholds.get("mant_median_threshold")
    if isinstance(exp, bool) or not isinstance(exp, int) or exp < floor.exp_mismatch_threshold:
        raise ValueError(f"exp_mismatch_threshold {exp!r} is under the floor")
    for name, value, low in (("mant_mean_threshold", mean, floor.mant_mean_threshold),
                             ("mant_median_threshold", median, floor.mant_median_threshold)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < low:
            raise ValueError(f"{name} {value!r} is under the floor")
    return {"exp_mismatch_threshold": int(exp), "mant_mean_threshold": float(mean),
            "mant_median_threshold": float(median)}


def new_request(*, qualification_id: str, model: str, revision: str, set_id: str,
                problems: int, sampling: dict, max_new_tokens: int, thinking: bool,
                completions: int = QUALIFY_COMPLETIONS, clock=time.time) -> dict:
    return {
        "schema": QUALIFICATION_SCHEMA, "qualification_id": qualification_id,
        "model": model, "revision": revision, "set_id": set_id, "problems": int(problems),
        "completions": int(completions), "sampling": dict(sampling),
        "max_new_tokens": int(max_new_tokens), "thinking": bool(thinking),
        "status": PENDING, "requested_at": clock(), "lease": None, "result": None,
    }


REQUEST_FIELDS = ("model", "revision", "set_id", "problems", "completions", "sampling",
                  "max_new_tokens", "thinking")


class QualificationStore:
    """Qualification records in the subnet bucket, compare-and-swap."""

    def __init__(self, **client_kwargs) -> None:
        self._client_kwargs = client_kwargs

    async def read(self, qualification_id: str) -> tuple[dict | None, str | None]:
        from reliquary.infrastructure.corpus_job_store import _get

        body, etag = await _get(qualification_key(qualification_id), **dict(self._client_kwargs))
        return (None, None) if body is None else (json.loads(body), etag)

    async def write(self, document: dict, etag: str | None) -> str | None:
        """Create (``etag`` None) or replace; ``CorpusStoreConflict`` on a race."""
        from reliquary.infrastructure.corpus_job_store import _put

        body = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        return await _put(qualification_key(document["qualification_id"]), body, etag,
                          **dict(self._client_kwargs))

    async def list_ids(self) -> list[str]:
        from reliquary.infrastructure.corpus_job_store import _bucket
        from reliquary.infrastructure.storage import get_s3_client

        kwargs = dict(self._client_kwargs)
        bucket = _bucket(kwargs)
        ids = []
        async with get_s3_client(**kwargs) as client:
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=bucket, Prefix=QUALIFICATION_PREFIX):
                for obj in page.get("Contents", []) or []:
                    name = obj["Key"][len(QUALIFICATION_PREFIX):]
                    if name.endswith(".json"):
                        ids.append(name[:-len(".json")])
        return sorted(ids)


__all__ = [
    "FAILED",
    "LEASED",
    "PENDING",
    "QUALIFICATION_SCHEMA",
    "QUALIFIED",
    "QUALIFY_COMPLETIONS",
    "QualificationStore",
    "REFUSED",
    "REQUEST_FIELDS",
    "TERMINAL",
    "THRESHOLD_MARGIN",
    "ceilings",
    "check_thresholds",
    "new_request",
    "p99",
    "qualification_key",
    "thresholds_from_band",
    "validated_qualification_id",
]
