"""Per-model qualification: the TOPLOC thresholds an eval job is audited with.

Two executors on distinct providers (and hosts) each decode the job's own eval
prompts with vLLM, verify them with the auditor's HF prefill, and report every
chunk's measures. The control accepts the measurement only when both bands agree
(each p99 within a ratio of the other) and both executors downloaded the same
checkpoint; otherwise another pair is tried. Thresholds are ``max(p99 × 1.5,
floor)`` of the agreed band (the larger of the two), clamped to a hard threshold
ceiling; a model whose honest p99 is itself over that ceiling is refused. The
model's eos and architecture are read by the control from the model's own small
files, never taken from an executor. The record lives in the subnet bucket and
binds the conditions it was measured under, which the job must repeat.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

QUALIFICATION_SCHEMA = "reliquary/eval-qualification/v2"
QUALIFICATION_PREFIX = "reliquary/eval/qualifications/"
QUALIFY_COMPLETIONS = 32
MAX_QUALIFY_COMPLETIONS = 64
# Completions × max_new_tokens a qualification may decode (it must fit its lease).
MAX_QUALIFY_TOKENS = 64 * 32768
QUALIFIERS = 2
THRESHOLD_MARGIN = 1.5
# Two bands agree when each p99 is within this ratio of the other (plus a slack
# for near-zero measures).
BAND_AGREEMENT_RATIO = 1.5
BAND_SLACK = {"exp_mismatch": 2.0, "mant_mean": 1.0, "mant_median": 1.0}
# Disagreeing pairs, and lease expiries, before the qualification fails.
MAX_QUALIFY_ATTEMPTS = 3
MAX_QUALIFY_EXPIRIES = 3
PENDING, QUALIFIED, REFUSED, FAILED = "pending", "qualified", "refused", "failed"
TERMINAL = frozenset({QUALIFIED, REFUSED, FAILED})
MEASURES = ("exp_mismatch", "mant_mean", "mant_median")
_ID_RE = re.compile(r"\A[a-z0-9][a-z0-9-]{0,62}\Z")


def threshold_ceilings() -> dict[str, float]:
    """The hard ceiling no job's threshold may exceed (twice the deployed
    floors by default; ``RELIQUARY_QUALIFY_THRESHOLD_CEILING_*`` override)."""
    return {"exp_mismatch": float(os.environ.get("RELIQUARY_QUALIFY_THRESHOLD_CEILING_EXP", 120)),
            "mant_mean": float(os.environ.get(
                "RELIQUARY_QUALIFY_THRESHOLD_CEILING_MANT_MEAN", 80.0)),
            "mant_median": float(os.environ.get(
                "RELIQUARY_QUALIFY_THRESHOLD_CEILING_MANT_MEDIAN", 80.0))}


def qualification_expiry_seconds() -> float:
    """A qualification still pending this long after its request fails
    (``qualification_expired``), whether or not an executor ever claimed it:
    six hours by default, ``RELIQUARY_QUALIFY_EXPIRY_SECONDS`` overrides."""
    return float(os.environ.get("RELIQUARY_QUALIFY_EXPIRY_SECONDS", 6 * 3600))


def qualify_lease_seconds(max_new_tokens: int) -> float:
    """A qualification decodes every completion in one batch, then verifies:
    half an hour plus a quarter second per token of budget."""
    return 1800.0 + 0.25 * int(max_new_tokens)


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


def band_of(chunks: Sequence[Sequence[float]]) -> dict:
    if not chunks:
        raise ValueError("qualification measured no chunk")
    return {"exp_mismatch": p99([float(c[0]) for c in chunks]),
            "mant_mean": p99([float(c[1]) for c in chunks]),
            "mant_median": p99([float(c[2]) for c in chunks]),
            "chunks": len(chunks)}


def bands_agree(a: dict, b: dict) -> bool:
    for name in MEASURES:
        low, high = sorted((float(a[name]), float(b[name])))
        if high > low * BAND_AGREEMENT_RATIO + BAND_SLACK[name]:
            return False
    return True


def thresholds_from_band(band: dict, *, floor=None,
                         ceiling: dict[str, float] | None = None) -> dict:
    """``{"thresholds", "clamped", "refused"}`` for an agreed band:
    ``max(p99 × 1.5, floor)`` clamped to the ceiling; ``refused`` names the
    measure whose honest p99 is itself over the ceiling (honest work would fail)."""
    if floor is None:
        from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as floor
    ceiling = ceiling or threshold_ceilings()
    raw = {"exp_mismatch": max(math.ceil(band["exp_mismatch"] * THRESHOLD_MARGIN),
                               int(floor.exp_mismatch_threshold)),
           "mant_mean": max(round(band["mant_mean"] * THRESHOLD_MARGIN, 6),
                            float(floor.mant_mean_threshold)),
           "mant_median": max(round(band["mant_median"] * THRESHOLD_MARGIN, 6),
                              float(floor.mant_median_threshold))}
    clamped = {name: min(raw[name], ceiling[name]) for name in MEASURES}
    thresholds = {"exp_mismatch_threshold": int(clamped["exp_mismatch"]),
                  "mant_mean_threshold": float(clamped["mant_mean"]),
                  "mant_median_threshold": float(clamped["mant_median"])}
    refused = next((f"honest {name} p99 {band[name]:.3f} is over the threshold ceiling "
                    f"{ceiling[name]}" for name in MEASURES if band[name] > ceiling[name]), None)
    return {"thresholds": thresholds, "clamped": sorted(n for n in MEASURES if clamped[n] < raw[n]),
            "refused": refused}


def check_thresholds(thresholds: dict, *, floor=None) -> dict:
    """Thresholds a job may carry: every measure between the floor and the ceiling."""
    if floor is None:
        from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as floor
    ceiling = threshold_ceilings()
    exp = thresholds.get("exp_mismatch_threshold")
    mean = thresholds.get("mant_mean_threshold")
    median = thresholds.get("mant_median_threshold")
    if isinstance(exp, bool) or not isinstance(exp, int) or exp < floor.exp_mismatch_threshold:
        raise ValueError(f"exp_mismatch_threshold {exp!r} is under the floor")
    if exp > ceiling["exp_mismatch"]:
        raise ValueError(f"exp_mismatch_threshold {exp!r} is over the ceiling")
    for name, value, low, high in (
            ("mant_mean_threshold", mean, floor.mant_mean_threshold, ceiling["mant_mean"]),
            ("mant_median_threshold", median, floor.mant_median_threshold,
             ceiling["mant_median"])):
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < low:
            raise ValueError(f"{name} {value!r} is under the floor")
        if value > high:
            raise ValueError(f"{name} {value!r} is over the ceiling")
    return {"exp_mismatch_threshold": int(exp), "mant_mean_threshold": float(mean),
            "mant_median_threshold": float(median)}


def new_request(*, qualification_id: str, model: str, revision: str, set_id: str,
                problems: int, sampling: dict, max_new_tokens: int, thinking: bool,
                completions: int = QUALIFY_COMPLETIONS, clock=time.time) -> dict:
    if not 0 < int(completions) <= MAX_QUALIFY_COMPLETIONS:
        raise ValueError(f"completions must be in [1, {MAX_QUALIFY_COMPLETIONS}]")
    if int(completions) * int(max_new_tokens) > MAX_QUALIFY_TOKENS:
        raise ValueError(f"completions x max_new_tokens is over {MAX_QUALIFY_TOKENS}")
    return {
        "schema": QUALIFICATION_SCHEMA, "qualification_id": qualification_id,
        "model": model, "revision": revision, "set_id": set_id, "problems": int(problems),
        "completions": int(completions), "sampling": dict(sampling),
        "max_new_tokens": int(max_new_tokens), "thinking": bool(thinking),
        "status": PENDING, "requested_at": clock(), "leases": {}, "measurements": {},
        "disagreements": [], "expiries": 0, "result": None,
    }


# What a qualification was measured under: a job must declare the same.
REQUEST_FIELDS = ("model", "revision", "set_id", "problems", "completions", "sampling",
                  "max_new_tokens", "thinking")

# A generation order's qualification: its catalog environment and range (the
# first ``min(problems, completions)`` rows from ``prompt_start`` are decoded),
# the package it renders through, and the same conditions.
GENERATION = "generation"
GEN_REQUEST_FIELDS = ("kind", "model", "revision", "env", "prompt_start", "problems",
                      "completions", "sampling", "max_new_tokens", "thinking",
                      "environment_manifest_sha256")


def request_fields(record: dict) -> tuple[str, ...]:
    return GEN_REQUEST_FIELDS if record.get("kind") == GENERATION else REQUEST_FIELDS


# The sources a generation order may draw from, by name: a new packaged
# source is a decision, never enabled by its shape alone.
ORDER_ENVIRONMENTS = frozenset({
    "reliquary_logic_v2", "reliquary_dapo_math_v1",
    "reliquary_instruction_following_v1", "reliquary_code_v1"})


def order_environment_refusal(env: str) -> str | None:
    """Why a generation order may not draw from ``env``, or None. Only a listed
    packaged single-turn source qualifies: it renders its own rows, so one
    order control renders every job's prompts as its miners do, whatever
    profile that process runs."""
    from reliquary.validator import corpus_service

    if env not in ORDER_ENVIRONMENTS:
        known = corpus_service.ENVIRONMENT_SPECS.get(env)
        if known is None:
            return f"env {env!r} is not an installed environment"
        if getattr(known, "interaction_mode", None) == "single_turn" \
                and getattr(known, "external_distribution", None) is None:
            return f"env {env!r} renders through the process profile, not a packaged source"
        if getattr(known, "interaction_mode", None) != "single_turn":
            return f"env {env!r} is not a single-turn source"
        return f"env {env!r} is not one of the order sources {sorted(ORDER_ENVIRONMENTS)}"
    spec = corpus_service.ENVIRONMENT_SPECS.get(env)
    if spec is None:
        return f"env {env!r} is not an installed environment"
    if getattr(spec, "interaction_mode", None) != "single_turn":
        return f"env {env!r} is not a single-turn source"
    if getattr(spec, "external_distribution", None) is None:
        return f"env {env!r} renders through the process profile, not a packaged source"
    return None


def new_generation_request(*, qualification_id: str, model: str, revision: str, env: str,
                           prompt_start: int, problems: int, sampling: dict,
                           max_new_tokens: int, thinking: bool,
                           completions: int = QUALIFY_COMPLETIONS, clock=time.time) -> dict:
    from reliquary.eval.sets import refuse_held_out_overlap
    from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG

    refusal = order_environment_refusal(env)
    if refusal is not None:
        raise ValueError(refusal)
    if int(prompt_start) < 0:
        raise ValueError("prompt_start must not be negative")
    # The held-out rows of the eval sets are never generated for an order.
    refuse_held_out_overlap(env, int(prompt_start), int(problems))
    record = new_request(qualification_id=qualification_id, model=model, revision=revision,
                         set_id="", problems=problems, sampling=sampling,
                         max_new_tokens=max_new_tokens, thinking=thinking,
                         completions=completions, clock=clock)
    del record["set_id"]
    profile = ENVIRONMENT_CATALOG.get(env)
    record.update(kind=GENERATION, env=env, prompt_start=int(prompt_start),
                  environment_manifest_sha256=getattr(profile, "environment_manifest_sha256",
                                                      None))
    return record


def catalog_prompts(env: str, start: int, count: int, *, environments=None) -> list[dict]:
    """Rows ``[start, start + count)`` of a packaged source, as a qualify
    lease carries them (``problem_id`` = ``<env>#<index>``, as the job names
    a row). ``environments`` caches built sources by name."""
    from reliquary.validator import corpus_service

    cache = environments if environments is not None else {}
    if env not in cache:
        cache[env] = corpus_service.ENVIRONMENT_SPECS[env].create()
    environment = cache[env]
    if start + count > len(environment):
        raise ValueError(f"{env!r} has {len(environment)} rows, fewer than {start + count}")
    rows = []
    for index in range(start, start + count):
        prompt = environment.get_problem(index).get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{env!r} row {index} has no prompt")
        rows.append({"problem_id": f"{env}#{index}", "text": prompt})
    return rows


def sample_sha256(prompts: Sequence[dict]) -> str:
    import hashlib

    body = json.dumps([[p["problem_id"], p["text"]] for p in prompts],
                      separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


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


def lease_prompt(row: dict) -> dict:
    """A set row as a qualify lease prompt: its user text, and its system turn
    when it has one, rendered by the executor exactly as the job's miners do."""
    from reliquary.eval.prompt_source import single_turn_messages

    system, text = single_turn_messages(row["messages"])
    prompt = {"problem_id": row["problem_id"], "text": text}
    if system is not None:
        prompt["system"] = system
    return prompt


EVAL_JOB_SCHEMA = "reliquary/eval-job/v1"
# A generation order's record, in the same store: its qualification and conditions.
ORDER_JOB_SCHEMA = "reliquary/order-job/v1"
EVAL_JOB_KEY_PREFIX = "reliquary/eval/jobs/"


class EvalJobStore:
    """What an eval job was declared from (its qualification, set and order
    conditions), create-only beside the qualifications: the report's
    provenance is built from it, never from a grade request."""

    def __init__(self, **client_kwargs) -> None:
        self._client_kwargs = client_kwargs

    @staticmethod
    def _key(job_id: str) -> str:
        from reliquary.corpus.job import JOB_ID_RE

        if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
            raise ValueError(f"unusable job id {job_id!r}")
        return f"{EVAL_JOB_KEY_PREFIX}{job_id}.json"

    async def read(self, job_id: str) -> dict | None:
        from reliquary.infrastructure.corpus_job_store import _get

        body, _ = await _get(self._key(job_id), **dict(self._client_kwargs))
        return None if body is None else json.loads(body)

    async def create(self, document: dict) -> None:
        """Written once; the same document again is a no-op, another a ValueError."""
        from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict, _put

        body = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        try:
            await _put(self._key(document["job_id"]), body, None, **dict(self._client_kwargs))
        except CorpusStoreConflict:
            if await self.read(document["job_id"]) != json.loads(body):
                raise ValueError(f"eval job {document['job_id']!r} was declared otherwise") \
                    from None


class QualificationQueue:
    """The eval control's side: each pending qualification leased to two
    executors of its model on distinct providers and hosts, their bands
    compared, the verdict recorded. ``model_facts(repo, revision)`` reads the
    model's architecture and eos from its own files (CPU)."""

    def __init__(self, *, store, read_prompts, model_facts, proof=None,
                 clock=time.time, read_catalog_prompts=None) -> None:
        if proof is None:
            from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as proof
        self._store = store
        self._read_prompts = read_prompts
        if read_catalog_prompts is None:
            import asyncio

            environments: dict = {}

            async def read_catalog_prompts(env, start, count):
                return await asyncio.to_thread(catalog_prompts, env, start, count,
                                               environments=environments)
        self._read_catalog_prompts = read_catalog_prompts
        self._model_facts = model_facts
        self._proof = proof
        self._clock = clock
        self._open: dict[str, dict] = {}
        # Ids already terminal: never read again.
        self._done: set[str] = set()
        # (repo, revision) -> the model's facts, read once.
        self._facts: dict[tuple[str, str], dict] = {}

    async def facts_of(self, model: str, revision: str) -> dict:
        key = (model, revision)
        if key not in self._facts:
            self._facts[key] = await self._model_facts(model, revision)
        return self._facts[key]

    async def model_refusal(self, record: dict) -> dict | None:
        """The result refusing a qualification before any executor is leased:
        an architecture outside ``SUPPORTED_ARCHITECTURES``, or model files
        that are not public at the revision. None: go on (a transient read
        failure raises)."""
        from reliquary.constants import SUPPORTED_ARCHITECTURES

        try:
            facts = await self.facts_of(record["model"], record["revision"])
        except ValueError as exc:
            return {"refused_reason": "model_files_unreadable", "detail": str(exc)[:300]}
        if facts["architecture"] not in SUPPORTED_ARCHITECTURES:
            return {"refused_reason": "architecture_unsupported",
                    "architecture": facts["architecture"],
                    "supported_architectures": sorted(SUPPORTED_ARCHITECTURES)}
        return None

    def _expired(self, record: dict) -> bool:
        return self._clock() - float(record.get("requested_at") or 0) \
            >= qualification_expiry_seconds()

    async def refresh(self) -> None:
        """Read every open record; one pending past its expiry fails, with or
        without an executor of its model."""
        from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict

        for qualification_id in await self._store.list_ids():
            if qualification_id in self._done:
                continue
            record, etag = await self._store.read(qualification_id)
            if record is None or record.get("status") in TERMINAL:
                self._open.pop(qualification_id, None)
                self._done.add(qualification_id)
            elif self._expired(record):
                try:
                    await self._finish(qualification_id, {**record, "status": FAILED, "result": {
                        "failed_reason": "qualification_expired"}}, etag)
                except CorpusStoreConflict:
                    continue
            else:
                self._open[qualification_id] = record

    def lease_of(self, lease_id: str) -> str | None:
        for qualification_id, record in self._open.items():
            for lease in (record.get("leases") or {}).values():
                if lease["lease_id"] == lease_id:
                    return qualification_id
        return None

    def _expire(self, record: dict) -> dict:
        now = self._clock()
        leases = dict(record.get("leases") or {})
        expired = [e for e, lease in leases.items() if float(lease["expires_at"]) <= now]
        for executor_id in expired:
            del leases[executor_id]
        record = {**record, "leases": leases, "expiries": record.get("expiries", 0) + len(expired)}
        if record["expiries"] >= MAX_QUALIFY_EXPIRIES:
            record.update(status=FAILED, result={
                "failed_reason": f"{record['expiries']} qualify leases expired"})
        return record

    @staticmethod
    def _eligible(record: dict, executor: dict) -> bool:
        from reliquary.validator.eval_control import _placement

        place = _placement(executor)
        executor_id = executor["executor_id"]
        involved = {**(record.get("leases") or {}), **(record.get("measurements") or {})}
        if place is None or executor_id in involved or len(involved) >= QUALIFIERS:
            return False
        for other_id, other in involved.items():
            if place[0] == other["provider_id"] or place[1] == other["host"]:
                return False
            if sorted([executor_id, other_id]) in record.get("disagreements", []):
                return False
        return True

    async def claim(self, executor: dict) -> dict | None:
        """A qualify lease for this executor's model, or None."""
        import secrets

        from reliquary.eval.prompt_source import head_lines
        from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict

        for qualification_id in sorted(self._open):
            record, etag = await self._store.read(qualification_id)
            if record is None or record.get("status") in TERMINAL:
                continue
            if (record["model"], record["revision"]) != (executor.get("model_id"),
                                                         executor.get("model_revision")):
                continue
            if self._expired(record):
                await self._finish(qualification_id, {**record, "status": FAILED, "result": {
                    "failed_reason": "qualification_expired"}}, etag)
                continue
            try:
                refusal = await self.model_refusal(record)
            except Exception:
                logger.warning("model facts of %s@%s unreadable; retrying", record["model"],
                               record["revision"], exc_info=True)
                continue
            if refusal is not None:
                await self._finish(qualification_id, {**record, "status": REFUSED,
                                                      "result": refusal}, etag)
                continue
            record = self._expire(record)
            if record["status"] == FAILED:
                await self._finish(qualification_id, record, etag)
                continue
            if not self._eligible(record, executor):
                continue
            if record.get("kind") == GENERATION:
                count = min(record["problems"], record["completions"])
                try:
                    prompts = await self._read_catalog_prompts(
                        record["env"], record["prompt_start"], count)
                except ValueError as exc:
                    await self._finish(qualification_id, {**record, "status": REFUSED, "result": {
                        "refused_reason": "prompts_unreadable", "detail": str(exc)[:300]}}, etag)
                    continue
                sample = sample_sha256(prompts)
                if record.get("sample_sha256", sample) != sample:
                    # The package moved under the record: never measured on two samples.
                    await self._finish(qualification_id, {**record, "status": FAILED, "result": {
                        "failed_reason": "the environment's sample changed"}}, etag)
                    continue
                record = {**record, "sample_sha256": sample}
            else:
                body = await self._read_prompts(record["set_id"])
                if body is None:
                    continue
                # Only the prompts a completion is decoded for.
                count = min(record["problems"], record["completions"], len(body.splitlines()))
                rows = [json.loads(line) for line in head_lines(body, count).splitlines()]
                prompts = [lease_prompt(r) for r in rows]
            seconds = qualify_lease_seconds(record["max_new_tokens"])
            lease = {"lease_id": secrets.token_hex(16), "expires_at": self._clock() + seconds,
                     "provider_id": executor["provider_id"], "host": executor["host"]}
            leased = {**record, "leases": {**record["leases"], executor["executor_id"]: lease}}
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
                "prompts": prompts,
                "completions": record["completions"], "sampling": record["sampling"],
                "max_new_tokens": record["max_new_tokens"], "thinking": record["thinking"],
            }
        return None

    async def _finish(self, qualification_id: str, record: dict, etag) -> dict:
        await self._store.write(record, etag)
        if record["status"] in TERMINAL:
            self._open.pop(qualification_id, None)
            self._done.add(qualification_id)
        else:
            self._open[qualification_id] = record
        return record

    async def result(self, executor: dict, lease_id: str, result) -> dict:
        """Record one executor's measurement; decide once two are in. Raises
        ``LeaseRefused`` for a lease not this executor's, expired, or raced."""
        from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict
        from reliquary.validator.corpus_audit_remote import LeaseRefused

        qualification_id = self.lease_of(lease_id)
        record, etag = (await self._store.read(qualification_id)
                        if qualification_id else (None, None))
        executor_id = executor["executor_id"]
        lease = ((record or {}).get("leases") or {}).get(executor_id)
        if record is None or record.get("status") in TERMINAL or lease is None \
                or lease["lease_id"] != lease_id:
            raise LeaseRefused(410, "lease_unknown")
        if float(lease["expires_at"]) <= self._clock():
            raise LeaseRefused(410, "lease_expired")
        measurement = {
            "band": band_of(result.chunks), "failed_completions": result.failed_completions,
            "checkpoint_sha256": result.checkpoint_sha256,
            "tokens_per_gpu_hour": result.completion_tokens / result.decode_seconds * 3600.0
            / result.gpu_count,
            "completions": result.completions, "completion_tokens": result.completion_tokens,
            "decode_seconds": result.decode_seconds, "gpu": result.gpu,
            "gpu_count": result.gpu_count, "vllm_version": result.vllm_version,
            "provider_id": lease["provider_id"], "host": lease["host"],
            "measured_at": self._clock()}
        leases = {k: v for k, v in record["leases"].items() if k != executor_id}
        record = {**record, "leases": leases,
                  "measurements": {**record["measurements"], executor_id: measurement}}
        if len(record["measurements"]) >= QUALIFIERS:
            record = await self._decide(record)
        try:
            return await self._finish(qualification_id, record, etag)
        except CorpusStoreConflict as exc:
            raise LeaseRefused(409, "qualification_changed_retry") from exc

    async def _decide(self, record: dict) -> dict:
        (a_id, a), (b_id, b) = sorted(record["measurements"].items())[:2]
        failed = [bool(a["failed_completions"]), bool(b["failed_completions"])]
        if all(failed):
            return {**record, "status": REFUSED, "result": {
                "refused_reason": "both qualifiers' honest completions failed their own proofs",
                "measurements": record["measurements"]}}
        if any(failed) or a["checkpoint_sha256"] != b["checkpoint_sha256"] \
                or not bands_agree(a["band"], b["band"]):
            disagreements = record["disagreements"] + [sorted([a_id, b_id])]
            record = {**record, "measurements": {}, "disagreements": disagreements,
                      "last_disagreement": {a_id: a, b_id: b}}
            if len(disagreements) >= MAX_QUALIFY_ATTEMPTS:
                record.update(status=FAILED, result={
                    "failed_reason": f"{len(disagreements)} qualifier pairs disagreed"})
            return record
        band = {name: max(a["band"][name], b["band"][name]) for name in MEASURES}
        band["chunks"] = a["band"]["chunks"] + b["band"]["chunks"]
        verdict = thresholds_from_band(band)
        refusal = await self.model_refusal(record)
        if refusal is not None:
            return {**record, "status": REFUSED, "result": refusal}
        facts = await self.facts_of(record["model"], record["revision"])
        result = {
            "thresholds": verdict["thresholds"], "clamped": verdict["clamped"], "band": band,
            "refused_reason": verdict["refused"],
            "checkpoint_sha256": a["checkpoint_sha256"],
            "architecture": facts["architecture"], "eos_token_id": int(facts["eos_token_id"]),
            "tokens_per_gpu_hour": min(a["tokens_per_gpu_hour"], b["tokens_per_gpu_hour"]),
            "measurements": record["measurements"], "decided_at": self._clock(),
        }
        return {**record, "status": REFUSED if verdict["refused"] else QUALIFIED,
                "result": result}


__all__ = [
    "BAND_AGREEMENT_RATIO",
    "EvalJobStore",
    "FAILED",
    "GENERATION",
    "ORDER_ENVIRONMENTS",
    "ORDER_JOB_SCHEMA",
    "qualification_expiry_seconds",
    "GEN_REQUEST_FIELDS",
    "catalog_prompts",
    "new_generation_request",
    "order_environment_refusal",
    "request_fields",
    "sample_sha256",
    "MAX_QUALIFY_ATTEMPTS",
    "PENDING",
    "QUALIFICATION_SCHEMA",
    "QUALIFIED",
    "QUALIFY_COMPLETIONS",
    "QUALIFIERS",
    "QualificationQueue",
    "QualificationStore",
    "REFUSED",
    "REQUEST_FIELDS",
    "TERMINAL",
    "THRESHOLD_MARGIN",
    "band_of",
    "bands_agree",
    "qualify_lease_seconds",
    "threshold_ceilings",
    "check_thresholds",
    "new_request",
    "p99",
    "qualification_key",
    "thresholds_from_band",
    "validated_qualification_id",
]
