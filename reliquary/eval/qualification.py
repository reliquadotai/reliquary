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
        from reliquary.infrastructure import corpus_job_store as jobs

        kwargs = dict(self._client_kwargs)
        bucket = jobs._bucket(kwargs)
        ids = []
        async with jobs.get_s3_client(**kwargs) as client:
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=bucket, Prefix=QUALIFICATION_PREFIX):
                for obj in page.get("Contents", []) or []:
                    name = obj["Key"][len(QUALIFICATION_PREFIX):]
                    if name.endswith(".json"):
                        ids.append(name[:-len(".json")])
        return sorted(ids)


QUALIFY_LEASE_SECONDS = 7200.0


class QualificationQueue:
    """The eval control's side: pending qualifications leased to an executor
    registered for their model, and its measured band turned into a verdict."""

    def __init__(self, *, store, read_prompts, proof=None, clock=time.time,
                 lease_seconds: float = QUALIFY_LEASE_SECONDS) -> None:
        if proof is None:
            from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as proof
        self._store = store
        self._read_prompts = read_prompts
        self._proof = proof
        self._clock = clock
        self._lease_seconds = lease_seconds
        self._open: dict[str, dict] = {}

    async def refresh(self) -> None:
        for qualification_id in await self._store.list_ids():
            record, _ = await self._store.read(qualification_id)
            if record is None or record.get("status") in TERMINAL:
                self._open.pop(qualification_id, None)
            else:
                self._open[qualification_id] = record

    def _claimable(self, record: dict, executor: dict) -> bool:
        if (record["model"], record["revision"]) != (executor.get("model_id"),
                                                     executor.get("model_revision")):
            return False
        lease = record.get("lease") or {}
        return record["status"] == PENDING or (
            record["status"] == LEASED and float(lease.get("expires_at", 0)) <= self._clock())

    async def claim(self, executor: dict) -> dict | None:
        """A qualify lease for this executor's model, or None."""
        import secrets

        from reliquary.eval.prompt_source import head_lines
        from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict

        for qualification_id in sorted(self._open):
            record, etag = await self._store.read(qualification_id)
            if record is None or not self._claimable(record, executor):
                continue
            body = await self._read_prompts(record["set_id"])
            if body is None:
                continue
            rows = [json.loads(line) for line in head_lines(
                body, min(record["problems"], len(body.splitlines()))).splitlines()]
            lease = {"lease_id": secrets.token_hex(16), "executor_id": executor["executor_id"],
                     "expires_at": self._clock() + self._lease_seconds}
            leased = {**record, "status": LEASED, "lease": lease}
            try:
                await self._store.write(leased, etag)
            except CorpusStoreConflict:
                continue
            self._open[qualification_id] = leased
            return {
                "protocol": "reliquary.corpus-audit/v1", "type": "qualify",
                "lease_id": lease["lease_id"], "qualification_id": qualification_id,
                "model_id": record["model"], "model_revision": record["revision"],
                "chunk_tokens": self._proof.chunk_tokens, "topk": self._proof.topk,
                "expires_at": lease["expires_at"],
                "prompts": [{"problem_id": r["problem_id"],
                             "text": r["messages"][-1]["content"]} for r in rows],
                "completions": record["completions"], "sampling": record["sampling"],
                "max_new_tokens": record["max_new_tokens"], "thinking": record["thinking"],
            }
        return None

    def lease_of(self, lease_id: str) -> str | None:
        for qualification_id, record in self._open.items():
            if (record.get("lease") or {}).get("lease_id") == lease_id:
                return qualification_id
        return None

    async def result(self, executor: dict, lease_id: str, result) -> dict:
        """Record a measured band; raises ``LeaseRefused`` for a lease not this
        executor's or expired."""
        from reliquary.validator.corpus_audit_remote import LeaseRefused

        qualification_id = self.lease_of(lease_id)
        record, etag = (await self._store.read(qualification_id)
                        if qualification_id else (None, None))
        lease = (record or {}).get("lease") or {}
        if (record is None or record.get("status") != LEASED or lease.get("lease_id") != lease_id
                or lease.get("executor_id") != executor["executor_id"]):
            raise LeaseRefused(410, "lease_unknown")
        if float(lease["expires_at"]) <= self._clock():
            raise LeaseRefused(410, "lease_expired")
        verdict = thresholds_from_band(result.chunks)
        refused = verdict["refused"]
        if result.failed_completions:
            refused = (f"{result.failed_completions} honest completion(s) failed their own "
                       "proofs on the executor that decoded them")
        measured = {
            "thresholds": verdict["thresholds"], "band": verdict["band"],
            "refused_reason": refused,
            "tokens_per_gpu_hour": result.completion_tokens / result.decode_seconds * 3600.0
            / result.gpu_count,
            "checkpoint_sha256": result.checkpoint_sha256, "architecture": result.architecture,
            "eos_token_id": result.eos_token_id, "executor_id": executor["executor_id"],
            "provider_id": executor.get("provider_id"), "host": executor.get("host"),
            "gpu": result.gpu, "gpu_count": result.gpu_count,
            "vllm_version": result.vllm_version, "completions": result.completions,
            "failed_completions": result.failed_completions,
            "completion_tokens": result.completion_tokens,
            "decode_seconds": result.decode_seconds, "measured_at": self._clock(),
        }
        final = {**record, "status": REFUSED if refused else QUALIFIED, "result": measured}
        await self._store.write(final, etag)
        self._open.pop(qualification_id, None)
        return final


__all__ = [
    "FAILED",
    "LEASED",
    "PENDING",
    "QUALIFICATION_SCHEMA",
    "QUALIFIED",
    "QUALIFY_COMPLETIONS",
    "QUALIFY_LEASE_SECONDS",
    "QualificationQueue",
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
