"""`reliquary admin serve`: the subnet's write authority, for the platform.

Every route needs a fresh HMAC signature (``auth``). The registry is written
under its compare-and-swap with two extra limits re-checked on every retry: the
active caps stay within the one pool, and the corpus caps within
``RELIQUARY_ADMIN_POOL_MAX``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reliquary.admin.auth import (
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    HmacVerifier,
)

from reliquary.sandbox.evaluation_engagements import EvaluationSessionRequest

logger = logging.getLogger(__name__)

_SUM_TOLERANCE = 1e-9
# The only task and job ids the platform may create or touch: never an
# operator-declared prod job.
DEFAULT_TASK_PREFIX = "order-"
# ``auction``: order jobs with a token ceiling are priced by auction (``corpus.token_auction``).
ORDER_PRICING_ENV = "RELIQUARY_ORDER_PRICING"
# No admin request is larger; the bound is checked before the body is read.
MAX_BODY_BYTES = 1024 * 1024
DEFAULT_EVAL_CAP = 0.02
# How long a job's counts are awaited before its last ones answer (``job_counts``).
COUNTS_WAIT_SECONDS = 20.0
# A generation order on any model: the cap share it defaults to, and its audit:
# every submission unless the order asks for a sampled share (`audit_q`), then
# a hotkey's first 20 audited passes (counted across orders) before sampling starts.
# Held 10 min before an unaudited pass is paid: an order generates in 15-40 min and its
# last passes wait out the hold before delivery, while miners have nothing to do. The hold
# only has to outlast a drawn record's audit (1-2 min), not a corpus period; at ~2 records
# a minute a hotkey's stake still holds far more than 1/q records inside it.
DEFAULT_GEN_CAP = 0.02
GEN_AUDIT_PARAMS = {"audit_q": 1.0, "audit_probation_submissions": 20,
                    "audit_hold_seconds": 600, "audit_suspect_seconds": 86400,
                    "audit_ban_after_failures": 3, "audit_ban_window_seconds": 604800,
                    "audit_ban_seconds": 604800}
# Renderer of a catalog source's rows: the model's own chat template.
THINKING_RENDERERS = {False: "chat-template-v1", True: "chat-template-thinking-v1"}
# An order's token ceiling is given in USD and converted at job creation with
# the subnet's alpha price in TAO and TAO's USD price: this variable when set,
# else CoinGecko's, cached a few minutes.
TAO_USD_ENV = "RELIQUARY_TAO_USD"
COINGECKO_TAO_USD_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd"
TAO_USD_CACHE_SECONDS = 300.0


async def _read_subnet_pool() -> dict:
    """The subnet's alpha price in TAO and its miners' alpha per block (``chain.subnet_pool``)."""
    from reliquary.infrastructure.chain import close_subtensor, get_subtensor, subnet_pool

    subtensor = await get_subtensor()
    try:
        return await subnet_pool(subtensor)
    finally:
        await close_subtensor(subtensor)


async def _fetch_tao_usd() -> float:
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(COINGECKO_TAO_USD_URL)
        response.raise_for_status()
        return float(response.json()["bittensor"]["usd"])


def tao_usd_reader(*, fetch: Callable | None = None, clock: Callable[[], float] = time.monotonic,
                   ttl: float = TAO_USD_CACHE_SECONDS):
    """TAO's USD price: ``RELIQUARY_TAO_USD`` when set, else fetched and kept ``ttl`` seconds."""
    cached: dict[str, float] = {}

    def checked(value) -> float:
        value = float(value)
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"TAO/USD must be a positive number, got {value!r}")
        return value

    async def read() -> float:
        given = os.environ.get(TAO_USD_ENV)
        if given:
            return checked(given)
        now = clock()
        if cached and 0 <= now - cached["at"] < ttl:
            return cached["value"]
        value = checked(await (fetch or _fetch_tao_usd)())
        cached.update(value=value, at=now)
        return value

    return read


class QualifiedModel(BaseModel):
    """A model the platform may order from, as the admin host's catalog pins it."""

    model_config = ConfigDict(extra="forbid")
    revision: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    eos_token_id: int = Field(ge=0)


class TokenCeiling(BaseModel):
    """An order's price ceiling, in USD per million tokens: prompt (input) and
    generated (output) tokens. Miners are never paid past it; the rest burns."""

    model_config = ConfigDict(extra="forbid")
    input: float = Field(ge=0.0, allow_inf_nan=False)
    output: float = Field(gt=0.0, allow_inf_nan=False)


def audit_executors_of(body) -> dict:
    """The order record's ``audit_executors``, only when the order asks for more than one."""
    n = getattr(body, "audit_executors", None)
    return {"audit_executors": int(n)} if n is not None and n > 1 else {}


class CreateJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(min_length=1, max_length=128)
    task_id: str | None = Field(default=None, min_length=1, max_length=128)
    model: str = Field(min_length=1)
    env: str = Field(min_length=1)
    prompt_start: int = Field(default=0, ge=0)
    prompt_count: int = Field(gt=0)
    samples_per_prompt: int = Field(gt=0)
    max_new_tokens: int | None = Field(default=None, gt=0)
    thinking: bool = False
    # A fixed share of the pool, as every corpus job; required, except that an
    # eval job defaults to 0.02.
    cap: float | None = Field(default=None, ge=0.0, le=1.0)
    # An evaluation job: its prompts are the eval set's first prompt_count
    # problems, its model and thresholds come from the qualification record,
    # whose conditions (set, count, sampling, budget, thinking) it must repeat.
    eval_set_id: str | None = Field(default=None, min_length=1, max_length=128)
    qualification_id: str | None = Field(default=None, min_length=1, max_length=63)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    audit_q: float | None = Field(default=None, gt=0.0, le=1.0)
    # The most audit executors the order control may use for this order's model (none: one), at the order's
    # chosen speed; the control's own maximum still applies.
    audit_executors: int | None = Field(default=None, ge=1, le=8)
    reuse_auditors_only: bool = False
    sampling: "OrderSampling | None" = None
    # Eval and generation orders: every sample follows the job's public forced draw (``corpus.eval_draw``),
    # checked by the audit in place of a proof; ``judge`` sets some of its thresholds (``corpus.job.DrawJudge``).
    draw: Literal["gumbel_max"] | None = None
    judge: dict[str, float | int] | None = Field(default=None, max_length=8)
    # A long-context order: its longest prompt in tokens (miners size their engine to it).
    max_prompt_tokens: int | None = Field(default=None, gt=0, le=131072)
    # A forced-draw order without a qualification pins its model here (``orders.models``): the full revision.
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    # A reviewed contract can be pinned before the first durable write.
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    profile_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Eval and generation orders: the ceiling, converted to alpha when the task
    # is first declared (a replay keeps the first conversion).
    max_price_usd_per_mtoken: TokenCeiling | None = None
    # A forced-draw eval order on a SWE or bridged (TMax) set: the manifest's signed-sandbox
    # `episode` that plays its references (as `jobs create --episode-file`).
    episode: dict[str, Any] | None = None


class OrderSampling(BaseModel):
    """The order's sampling: what miners are told and qualification measured."""

    model_config = ConfigDict(extra="forbid")
    temperature: float = Field(gt=0.0, le=4.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    # Draw orders only (free-sampling miners do not apply it).
    min_p: float = Field(default=0.0, ge=0.0, lt=1.0)

    def conditions(self) -> dict:
        """What a qualification records and an order repeats: min_p only when set, so the conditions of every
        order without it read as before it existed."""
        return self.model_dump(exclude=None if self.min_p else {"min_p"})


class RequestQualification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qualification_id: str = Field(min_length=1, max_length=63)
    model: str = Field(min_length=1, max_length=256)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    # An evaluation names its set; a generation order its catalog env and the
    # first row of its range. Exactly one of set_id and env.
    set_id: str | None = Field(default=None, min_length=1, max_length=128)
    env: str | None = Field(default=None, min_length=1, max_length=128)
    prompt_start: int | None = Field(default=None, ge=0)
    # The order's problem count: the job will be its set's (or its range's)
    # first `problems` prompts; `completions` are decoded over the first of them.
    problems: int = Field(gt=0, le=100_000_000)
    completions: int = Field(default=32, gt=0, le=64)
    sampling: OrderSampling
    max_new_tokens: int = Field(gt=0, le=131072)
    thinking: bool = False

    @model_validator(mode="after")
    def _one_source(self):
        if (self.set_id is None) == (self.env is None):
            raise ValueError("a qualification names exactly one of set_id and env")
        if self.set_id is not None and (self.prompt_start is not None
                                        or self.problems > 1_000_000):
            raise ValueError("an eval qualification has no prompt_start and at most "
                             "1,000,000 problems")
        return self


CreateJob.model_rebuild()


class IngestUpload(BaseModel):
    """A batch file the platform holds, read as the order's prompts (``orders.uploads``)."""

    model_config = ConfigDict(extra="forbid")
    object_key: str = Field(min_length=1, max_length=512, pattern=r"^batch-inputs/[A-Za-z0-9._/-]+$")
    # None: the platform could not hash it (a large upload in parts); the admin hashes it and answers the hash.
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1, max_length=256)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    purpose: Literal["batch", "eval"] = "batch"
    # An eval upload's grading environment (its references are graded by it).
    grader: str | None = Field(default=None, min_length=1, max_length=128)
    max_prompt_tokens: int | None = Field(default=None, gt=0, le=131072)

    @model_validator(mode="after")
    def _grader_for_eval(self):
        if (self.purpose == "eval") != (self.grader is not None):
            raise ValueError("an eval upload names its grader, and only an eval upload does")
        if ".." in self.object_key:
            raise ValueError("object_key must not climb out of batch-inputs/")
        return self


class SetCap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cap: float = Field(ge=0.0, le=1.0)


class Retire(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retired_at: int | None = Field(default=None, ge=0)


class SetAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    admission: Literal["open", "paused"]


class RegisterSandboxMachine(BaseModel):
    """A rented sandbox gateway the platform registers once its box made its machine key."""
    model_config = ConfigDict(extra="forbid")
    machine_id: str = Field(min_length=1, max_length=64)
    address: str = Field(min_length=1, max_length=300)
    provider: str = Field(min_length=1, max_length=64)
    capacity: int = Field(gt=0, le=1024)
    key_id: str = Field(min_length=1, max_length=64)
    public_key_b64: str = Field(min_length=1, max_length=64)
    valid_from: int = Field(gt=0)


class SandboxMachineStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["active", "draining", "revoked"]
    reason: str | None = Field(default=None, max_length=200)


class RegisterExecutor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    executor_id: str
    token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    expires_at: float
    # Where it runs: the eval control pairs executors on distinct ones.
    provider_id: str | None = Field(default=None, min_length=1, max_length=256)
    host: str | None = Field(default=None, min_length=1, max_length=256)
    # "eval" executors serve the eval control only (provider_id and host required);
    # "grade" executors (provider_id required) grade and replay for the corpus control, bound to an env
    # pin (model_id = env package, model_revision = env commit).
    scope: Literal["corpus", "eval", "grade"] = "corpus"


class CreateDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    delivery_id: str | None = None
    apply_filter: bool = True
    # Keep the finished completions the filter rejects (annotated) in the
    # instruction shards: an SFT without the correctness filter.
    keep_rejected: bool = False
    # Deliver a completion once within the job and never one already in a
    # passing submission of these jobs (a v2 on the prompts of its v1).
    dedupe: bool = False
    dedupe_against: list[str] = Field(default_factory=list, max_length=16)
    # Delivery v3 (``corpus.delivery._export_v3``): the one delivery of every order. 2 (the default) is the
    # export the live platform reads; the fields below apply to 3 only, which ignores apply_filter.
    schema_version: Literal[2, 3] = 2
    # A length-cut completion is a sample at that budget: kept unless false.
    include_truncated: bool | None = None
    # "annotate" grades every row with the job's filter (accepted, score), "drop" leaves the rejected out.
    filter: Literal["none", "annotate", "drop"] | None = None
    format: Literal["parquet", "openai_jsonl"] | None = None

    @model_validator(mode="after")
    def _v3_fields(self):
        if self.schema_version == 2:
            if any(v is not None for v in (self.include_truncated, self.filter, self.format)):
                raise ValueError("include_truncated, filter and format are delivery v3 options "
                                 "(schema_version 3)")
        elif self.keep_rejected:
            raise ValueError("a v3 delivery keeps the rejected rows with filter 'annotate'")
        return self


class Provenance(BaseModel):
    """What the report names as having produced the completions. The core keys
    are required; the platform may add more, recorded as given."""

    model_config = ConfigDict(extra="allow")
    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    model_sha: str = Field(min_length=1)
    sampling: dict[str, Any]
    thinking: bool
    max_new_tokens: int = Field(gt=0)
    # The pod's, required when the completions are its uploads.
    vllm_version: str | None = Field(default=None, min_length=1)
    gpu: str | None = Field(default=None, min_length=1)
    pod_provider_id: str | None = Field(default=None, min_length=1)


class CombineEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The gradings of an evaluation order's sets (one job each), in report order.
    parts: list[str] = Field(min_length=1, max_length=64)


class GradeEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # "uploads": the v1 pod's completion files; "job": an eval job's passing records.
    source: Literal["uploads", "job"] = "uploads"
    job_id: str | None = Field(default=None, min_length=1, max_length=63)
    set_ids: list[str] = Field(min_length=1, max_length=64)
    completion_keys: list[str] = Field(default_factory=list, max_length=10_000)
    problems_per_set: dict[str, int]
    # Samples ordered per problem, indexed 0..samples-1.
    samples_per_set: dict[str, int]
    # Required for an uploads grading; a job grading takes its facts from the
    # job and its qualification, and only checks a provenance given against them.
    provenance: Provenance | None = None
    # Job mode: grade a job not every prompt of which holds its samples (the
    # platform, after its deadline); the report says complete=false.
    allow_incomplete: bool = False
    service_contract: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _source_fields(self):
        if self.service_contract is not None:
            from reliquary.protocol.service_contract import ServiceContract
            ServiceContract.from_dict(self.service_contract)
        if self.source == "uploads":
            if self.service_contract is not None:
                raise ValueError("uploaded completions cannot be verified against a service contract: "
                                 "their checkpoint and generation contract come from the same caller")
            if self.job_id is not None or not self.completion_keys:
                raise ValueError("an uploads grading names completion_keys and no job_id")
            if self.provenance is None:
                raise ValueError("an uploads grading needs its provenance")
            missing = [k for k in ("vllm_version", "gpu", "pod_provider_id")
                       if getattr(self.provenance, k) is None]
            if missing:
                raise ValueError(f"an uploads grading's provenance needs {missing}")
        elif self.job_id is None or self.completion_keys:
            raise ValueError("a job grading names job_id and no completion_keys")
        return self


def cap_limits(pool_max: float, *, drained_tasks=frozenset()) -> Callable[[Mapping, Mapping], None]:
    """The registry guard: paying caps within 1.0, paying corpus caps within
    ``pool_max``. Paying means active, or a retired corpus task whose job is
    not in ``drained_tasks`` (it still settles). A write that lowers a total
    is never refused by it."""
    from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION, RegistryError

    def paying(e) -> bool:
        return e.status == "active" or (e.mechanism == MECHANISM_CORPUS_GENERATION
                                        and e.task_id not in drained_tasks)

    def totals(entries: Mapping) -> tuple[float, float]:
        active = [e for e in entries.values() if paying(e)]
        return (sum(float(e.params["cap"]) for e in active),
                sum(float(e.params["cap"]) for e in active
                    if e.mechanism == MECHANISM_CORPUS_GENERATION))

    def guard(before: Mapping, after: Mapping) -> None:
        (all_before, corpus_before), (all_after, corpus_after) = totals(before), totals(after)
        if all_after > 1.0 + _SUM_TOLERANCE and all_after > all_before:
            raise RegistryError(f"active caps would total {all_after:.4f}, above 1.0")
        if corpus_after > pool_max + _SUM_TOLERANCE and corpus_after > corpus_before:
            raise RegistryError(
                f"corpus caps would total {corpus_after:.4f}, above the admin pool "
                f"of {pool_max:.4f}"
            )

    return guard


def _public_executor(document: Mapping) -> dict:
    return {k: v for k, v in document.items() if k != "token_sha256"}


def _current_round() -> int:
    """The drand round now, from the chain's published genesis and period."""
    from reliquary.infrastructure import drand
    from reliquary.validator.corpus_validator import make_round_at

    chain = drand.get_current_chain()
    if chain.get("genesis_time") is None or chain.get("period") is None:
        raise RuntimeError("drand chain parameters unknown")
    return make_round_at(chain["genesis_time"], chain["period"])(time.time()) - 1


def create_admin_app(*, secret: bytes, pool_max: float,
                     models: Mapping[str, QualifiedModel | Mapping], deliveries=None,
                     records=None, clock: Callable[[], float] = time.time,
                     current_round: Callable[[], int] = _current_round,
                     prepare=None, work_dir=None,
                     task_prefix: str = DEFAULT_TASK_PREFIX, eval_store=None,
                     open_environment=None, grade_scorer=None,
                     require_sandbox=None, qualifications=None, eval_jobs=None,
                     model_facts=None, evaluation_sessions=None, pin_model=None,
                     upload_tokenizer=None, subnet_pool=None, tao_usd=None,
                     sandbox_machine_prefix: str | None = None, sandbox_machines=None,
                     order_sessions=None) -> FastAPI:
    """The admin app. ``deliveries`` is the platform bucket's sink (None turns
    the export and grade routes off); ``records`` the subnet's record store;
    ``eval_store`` the subnet bucket holding the eval sets' grading files; ``order_sessions``
    the order control's sandbox session store (``list_recent``), read for an agentic order's
    redone slots (default: the bucket's order-session prefix).
    ``subnet_pool`` (async, ``chain.subnet_pool``'s answer) and ``tao_usd``
    (async, USD per TAO) convert an order's USD token ceiling. ``sandbox_machine_prefix``
    turns on the sandbox machine routes for the machines named with it (the platform's rented
    gateways) and only those; ``sandbox_machines`` is the machine directory (``sandbox_store``)."""
    from reliquary.infrastructure import corpus_executor_store as executors
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.infrastructure import task_registry_store as registry_store
    from reliquary.shared.task_registry import RegistryError

    if not (isinstance(pool_max, (int, float)) and math.isfinite(pool_max)
            and 0.0 <= pool_max <= 1.0):
        raise ValueError(f"the admin pool must be in [0, 1], got {pool_max!r}")
    if not task_prefix:
        raise ValueError("the admin task prefix is empty: it would reach every task")
    catalog = {name: spec if isinstance(spec, QualifiedModel) else QualifiedModel(**spec)
               for name, spec in models.items()}
    verifier = HmacVerifier(secret, clock=clock)
    # Retired corpus tasks whose job is drained: their caps no longer pay.
    drained_tasks: set[str] = set()
    if records is None:
        from reliquary.infrastructure.corpus_record_store import BucketRecordStore

        records = BucketRecordStore()
    if prepare is None:
        from reliquary.cli.main import prepare_corpus_job as prepare
    # delivery id -> (job id, the export, what it was asked for)
    exports: dict[str, tuple[str, asyncio.Task, dict]] = {}
    delivery_locks: dict[str, asyncio.Lock] = {}
    # eval id -> (request digest, the running grading)
    gradings: dict[str, tuple[str, asyncio.Task]] = {}
    # One decision at a time per eval id, so two first calls start one grading.
    grade_locks: dict[str, asyncio.Lock] = {}
    if eval_store is None:
        from reliquary.eval.storage import SubnetEvalStore

        eval_store = SubnetEvalStore()
    if open_environment is None:
        from reliquary.eval.sets import open_source as open_environment
    if qualifications is None:
        from reliquary.eval.qualification import QualificationStore

        qualifications = QualificationStore()
    if eval_jobs is None:
        from reliquary.eval.qualification import EvalJobStore

        eval_jobs = EvalJobStore()
    if model_facts is None:
        async def model_facts(repo: str, revision: str) -> dict:
            from reliquary.validator.eval_control import model_facts as read_facts

            # The model's own config and tokenizer files, on CPU (no weights).
            return await asyncio.to_thread(read_facts, repo, revision)
    if pin_model is None:
        def pin_model(repo: str, revision: str):
            from reliquary.orders import models

            return models.pin_model(repo, revision)
    if evaluation_sessions is None:
        from reliquary.sandbox.evaluation_engagements import from_environment

        evaluation_sessions = from_environment(subnet=eval_store, clock=clock)
    if subnet_pool is None:
        async def subnet_pool() -> dict:
            return await _read_subnet_pool()
    if tao_usd is None:
        tao_usd = tao_usd_reader()
    # Rows of each order source, counted once (building one reads its package).
    source_rows: dict[str, int] = {}

    async def signed(request: Request) -> None:
        if request.url.query:
            raise HTTPException(status_code=400, detail="query_not_signed")
        body = await request.body()
        refusal = verifier.refusal(request.method, request.url.path,
                                   request.headers.get(TIMESTAMP_HEADER),
                                   request.headers.get(NONCE_HEADER),
                                   request.headers.get(SIGNATURE_HEADER), body)
        if refusal is not None:
            logger.warning("admin request %s %s refused: %s", request.method,
                           request.url.path, refusal)
            raise HTTPException(status_code=401, detail=refusal)

    router = APIRouter(prefix="/admin/v1", dependencies=[Depends(signed)])

    def in_scope(name: str) -> None:
        if not name.startswith(task_prefix):
            raise HTTPException(status_code=409, detail="outside_admin_scope")

    async def entries_naming(job_id: str) -> list:
        entries, _ = await registry_store.read_registry(strict=False)
        return [e for _, e in sorted(entries.items()) if e.job_id == job_id]

    # Verdict outcomes by job, read once each (verdicts are create-only), until the job is final;
    # then only that it is: final never reverts (no prompt open, every verdict written), so its
    # verdicts are not read again by the delivery and grading that follow.
    verdict_outcomes: dict[str, dict[str, bool]] = {}
    final_jobs: set[str] = set()
    # Listing a large job takes longer than a caller waits (the platform: 30 s). One count runs at a time per
    # job, outlives the request that started it, and a caller gets the last counts after COUNTS_WAIT_SECONDS;
    # a drained job's counts never change.
    last_counts: dict[str, dict] = {}
    counting: dict[str, asyncio.Future] = {}

    async def read_counts(job_id: str) -> dict:
        from reliquary.validator.corpus_job_status import stored_job_counts

        known = job_id in final_jobs
        # Read before the listing: a job retired by then admits nothing the listing could miss.
        entries = await entries_naming(job_id)
        retired = bool(entries) and all(e.status == "retired" for e in entries)
        counts = await stored_job_counts(records, job_id, with_final=not known,
                                         verdict_passed=verdict_outcomes.setdefault(job_id, {}))
        counts["retired"] = retired
        # Kept for good only once its tasks are retired: a job still admitting is only quiet for now (2026-11-12:
        # a 12 h network cut left an open order with every record settled; its 544-record counts were kept as
        # drained and final while it went on to 7,543, and its delivery waited on them).
        if known or (counts["final"] and retired):
            counts["final"] = True
            final_jobs.add(job_id)
            verdict_outcomes.pop(job_id, None)
        last_counts[job_id] = counts
        return counts

    async def job_miner_pay(job_id: str, entries) -> dict | None:
        """What the job's miners were paid per settled period (``corpus_job_status.miner_pay``); None when the
        settlement is unreadable."""
        from reliquary.validator import corpus_periods as cp
        from reliquary.validator.corpus_job_status import miner_pay

        try:
            state, _ = await records.read_settlement(job_id)
        except Exception:
            logger.warning("admin: the settlement of %s is unreadable", job_id, exc_info=True)
            return None
        emission = next((e.params.get(cp.MAX_PRICE_EMISSION_PARAM) for e in entries
                         if e.params.get(cp.MAX_PRICE_EMISSION_PARAM) is not None), None)
        return miner_pay(state, emission)

    async def job_counts(job_id: str) -> dict:
        """The stored counts; ``final`` gates grading and delivery, ``drained`` only pay."""
        last = last_counts.get(job_id)
        if last is not None and last["drained"] and last.get("retired"):
            return dict(last)
        running = counting.get(job_id)
        if running is None:
            running = counting[job_id] = asyncio.ensure_future(read_counts(job_id))
            running.add_done_callback(lambda done, j=job_id: (
                counting.pop(j, None), done.cancelled() or done.exception()))
        try:
            return dict(await asyncio.wait_for(asyncio.shield(running), COUNTS_WAIT_SECONDS))
        except asyncio.TimeoutError:
            if last is not None:
                return dict(last)
            return dict(await asyncio.shield(running))

    async def draining_limits(new_cap: float):
        """The cap guard, counting retired corpus tasks whose job has not drained."""
        if new_cap == 0.0:
            # Adding no share or lowering one cannot increase either total;
            # historical job listings are unnecessary for this cap decision.
            return cap_limits(float(pool_max), drained_tasks=set(drained_tasks))
        from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION
        from reliquary.validator.corpus_job_status import stored_job_counts

        entries, _ = await registry_store.read_registry(strict=False)
        for entry in entries.values():
            # A retired task at cap 0 adds nothing to the totals, drained or not:
            # listing its job (minutes for a large one) cannot change the decision.
            if (entry.status == "retired" and entry.mechanism == MECHANISM_CORPUS_GENERATION
                    and entry.job_id and entry.task_id not in drained_tasks
                    and float(entry.params.get("cap") or 0.0) > 0.0):
                if (await stored_job_counts(records, entry.job_id))["drained"]:
                    # Drained is for good: never listed again.
                    drained_tasks.add(entry.task_id)
                    await release_drained(entry)
        return cap_limits(float(pool_max), drained_tasks=set(drained_tasks))

    async def release_drained(entry) -> None:
        """Every period of a drained job is settled: its cap goes to 0 (its earned tail still paid), or the
        registry's pool, which counts retired caps, would stay full of finished orders."""
        from reliquary.shared.task_registry import SETTLEMENT_PERIOD_EMA

        if entry.params.get("settlement") != SETTLEMENT_PERIOD_EMA:
            return
        try:
            await registry_store.release_task_cap(entry.task_id)
            logger.info("admin: released the cap of drained task %s", entry.task_id)
        except Exception:  # noqa: BLE001 - the next create retries; the guard still counts it until then
            logger.warning("admin: could not release the cap of drained task %s", entry.task_id, exc_info=True)

    def price_of(entry) -> dict | None:
        from reliquary.validator.corpus_periods import price_view

        return price_view(entry.params)

    def idempotent(answer: dict, entry) -> dict:
        answer = {**answer, "created": False, "status": entry.status,
                  "cap": float(entry.params["cap"])}
        price = price_of(entry)
        return answer if price is None else {**answer, "max_price": price}

    async def refuse_unservable_model(model: str, revision: str) -> None:
        """422 ``architecture_unsupported`` / ``model_files_unreadable`` from the
        model's own files; 503 ``model_files_unavailable`` when the hub cannot
        be read now."""
        from reliquary.constants import SUPPORTED_ARCHITECTURES

        try:
            facts = await model_facts(model, revision)
        except ValueError as exc:
            raise HTTPException(status_code=422,
                                detail=f"model_files_unreadable: {exc}"[:500]) from exc
        except Exception as exc:
            logger.warning("model files of %s@%s unavailable", model, revision, exc_info=True)
            raise HTTPException(status_code=503, detail="model_files_unavailable") from exc
        if facts["architecture"] not in SUPPORTED_ARCHITECTURES:
            raise HTTPException(status_code=422, detail=(
                f"architecture_unsupported: {facts['architecture']!r}; supported: "
                f"{sorted(SUPPORTED_ARCHITECTURES)}"))

    async def refuse_past_the_source(env: str, start: int, count: int) -> None:
        """The whole range inside the source, as job creation will check it."""
        from reliquary.validator import corpus_service

        if env not in source_rows:
            try:
                source_rows[env] = await asyncio.to_thread(
                    lambda: len(corpus_service.ENVIRONMENT_SPECS[env].create()))
            except Exception as exc:
                logger.warning("source %s could not be built", env, exc_info=True)
                raise HTTPException(status_code=503, detail="source_unavailable") from exc
        if start + count > source_rows[env]:
            raise HTTPException(status_code=422, detail=(
                f"rows [{start}, {start + count}) run past {env!r}, which has "
                f"{source_rows[env]} rows"))

    def refuse_unsupported(result: dict) -> None:
        from reliquary.constants import SUPPORTED_ARCHITECTURES

        # A record qualified before the table changed is not trusted for it.
        if result.get("architecture") not in SUPPORTED_ARCHITECTURES:
            raise HTTPException(status_code=409, detail=(
                f"architecture_unsupported: {result.get('architecture')!r}"))

    async def eval_job_arguments(body: CreateJob, *, record_order: bool) -> dict:
        """What an evaluation job is declared from: its set's first
        prompt_count prompts, the qualification's model and thresholds."""
        from reliquary.eval import qualification as qual
        from reliquary.eval.prompt_source import eval_source_for, register_eval_prompts
        from reliquary.eval.sets import validated_set_id
        from reliquary.eval.storage import subnet_key
        from reliquary.protocol.external_eval import contract_environment_for

        if body.qualification_id is None:
            raise HTTPException(status_code=422, detail="qualification_id_required")
        in_scope(body.qualification_id)
        if body.sampling is None or body.max_new_tokens is None:
            raise HTTPException(status_code=422, detail="an eval job names its sampling and max_new_tokens")
        if body.prompt_start != 0:
            raise HTTPException(status_code=422, detail="an eval job starts at the set's first problem")
        if body.audit_q not in (None, 1.0):
            raise HTTPException(status_code=422, detail="an eval job audits every submission (audit_q 1.0)")
        try:
            set_id = validated_set_id(body.eval_set_id)
            record, _ = await qualifications.read(body.qualification_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail="qualification_unknown")
        if record.get("kind") == qual.GENERATION:
            raise HTTPException(status_code=409, detail="qualification_is_for_another_kind")
        if record.get("status") != qual.QUALIFIED:
            raise HTTPException(status_code=409, detail=f"model_not_qualified: {record.get('status')}")
        if record["model"] != body.model:
            raise HTTPException(status_code=409, detail="qualification_is_for_another_model")
        result = record["result"]
        refuse_unsupported(result)
        card_body = await eval_store.get_bytes(subnet_key(set_id, "set.json"))
        prompts_body = await eval_store.get_bytes(subnet_key(set_id, "prompts.jsonl"))
        if card_body is None or prompts_body is None:
            raise HTTPException(status_code=404, detail="set_unknown")
        card = json.loads(card_body)
        if body.env not in (card["env"], card["source"]):
            raise HTTPException(status_code=422, detail=f"set {set_id} is a {card['env']} set")
        if body.prompt_count > int(card["count"]):
            raise HTTPException(status_code=422, detail=f"set {set_id} holds {card['count']} problems")
        # Thresholds hold only for the conditions they were measured under.
        wanted = {"set_id": set_id, "problems": body.prompt_count,
                  "sampling": body.sampling.conditions(), "max_new_tokens": body.max_new_tokens,
                  "thinking": body.thinking}
        differs = sorted(k for k, v in wanted.items() if record.get(k) != v)
        if differs:
            raise HTTPException(status_code=409,
                                detail=f"qualification_conditions_differ: {differs}")
        source = eval_source_for(set_id, prompts_body, body.prompt_count)
        register_eval_prompts(source, prompts_body)
        seed = body.seed if body.seed is not None else job_seed(body.job_id)
        try:
            if record_order:
                await eval_jobs.create({
                    "schema": qual.EVAL_JOB_SCHEMA, "job_id": body.job_id,
                    "qualification_id": body.qualification_id, "set_id": set_id,
                    "problems": body.prompt_count, "samples": body.samples_per_prompt,
                    "sampling": body.sampling.conditions(), "max_new_tokens": body.max_new_tokens,
                    "thinking": body.thinking, **audit_executors_of(body)})
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return dict(
            model_revision=record["revision"], model_architecture=result["architecture"],
            checkpoint_sha256=result["checkpoint_sha256"], eos_token_id=int(result["eos_token_id"]),
            prompt_source=source.name, contract_environment=contract_environment_for(card),
            seed=seed,
            toploc_thresholds=result["thresholds"], audit_params={"audit_q": 1.0},
            temperature=body.sampling.temperature, top_p=body.sampling.top_p,
            top_k=body.sampling.top_k, **draw_arguments(body),
        )

    def job_seed(job_id: str) -> int:
        return int(hashlib.sha256(job_id.encode()).hexdigest()[:15], 16)

    def draw_arguments(body: CreateJob) -> dict:
        """The order's forced draw, its judge and min_p (``prepare_corpus_job``); refused without a draw."""
        if body.draw is None:
            if body.judge is not None or (body.sampling is not None and body.sampling.min_p):
                raise HTTPException(status_code=422, detail="judge and min_p set a forced draw: name draw")
            return {}
        return {"draw": body.draw, "judge": body.judge, "min_p": body.sampling.min_p}

    async def gen_job_arguments(body: CreateJob, *, record_order: bool) -> dict:
        """What a generation order on any model is declared from: its catalog
        env's range, the qualification's model and thresholds, whose
        conditions (env, range, package, sampling, budget, thinking) it repeats."""
        from reliquary.eval import qualification as qual
        from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG

        if body.qualification_id is None:
            raise HTTPException(status_code=422, detail="qualification_id_required")
        in_scope(body.qualification_id)
        if body.sampling is None or body.max_new_tokens is None:
            raise HTTPException(status_code=422,
                                detail="a generation order names its sampling and max_new_tokens")
        refusal = qual.order_environment_refusal(body.env)
        if refusal is not None:
            raise HTTPException(status_code=422, detail=refusal)
        try:
            record, _ = await qualifications.read(body.qualification_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail="qualification_unknown")
        if record.get("kind") != qual.GENERATION:
            raise HTTPException(status_code=409, detail="qualification_is_for_another_kind")
        if record.get("status") != qual.QUALIFIED:
            raise HTTPException(status_code=409, detail=f"model_not_qualified: {record.get('status')}")
        if record["model"] != body.model:
            raise HTTPException(status_code=409, detail="qualification_is_for_another_model")
        result = record["result"]
        refuse_unsupported(result)
        profile = ENVIRONMENT_CATALOG.get(body.env)
        wanted = {"env": body.env, "prompt_start": body.prompt_start,
                  "problems": body.prompt_count, "sampling": body.sampling.conditions(),
                  "max_new_tokens": body.max_new_tokens, "thinking": body.thinking,
                  # The package the rows render through, as it is here now.
                  "environment_manifest_sha256": getattr(
                      profile, "environment_manifest_sha256", None)}
        differs = sorted(k for k, v in wanted.items() if record.get(k) != v)
        if differs:
            raise HTTPException(status_code=409,
                                detail=f"qualification_conditions_differ: {differs}")
        try:
            # What the order control checks the job against before serving it.
            if record_order:
                await eval_jobs.create({
                    "schema": qual.ORDER_JOB_SCHEMA, "job_id": body.job_id, "kind": qual.GENERATION,
                    "qualification_id": body.qualification_id, "env": body.env,
                    "prompt_start": body.prompt_start, "problems": body.prompt_count,
                    "samples": body.samples_per_prompt, "sampling": body.sampling.conditions(),
                    "max_new_tokens": body.max_new_tokens, "thinking": body.thinking,
                    **audit_executors_of(body)})
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return dict(
            model_revision=record["revision"], model_architecture=result["architecture"],
            checkpoint_sha256=result["checkpoint_sha256"], eos_token_id=int(result["eos_token_id"]),
            prompt_source=body.env, toploc_thresholds=result["thresholds"],
            # Its miners submit on its own route (the order control has no legacy one).
            submit="scoped",
            audit_params={**GEN_AUDIT_PARAMS,
                          **({"audit_q": body.audit_q} if body.audit_q is not None else {})},
            temperature=body.sampling.temperature, top_p=body.sampling.top_p,
            top_k=body.sampling.top_k, **draw_arguments(body),
            # A draw order's stream is keyed on its seed: the client's, else one derived from the job id.
            **({"seed": body.seed if body.seed is not None else job_seed(body.job_id)} if body.draw else {}),
        )

    async def read_set(set_id: str) -> tuple[dict, bytes]:
        from reliquary.eval.sets import validated_set_id
        from reliquary.eval.storage import subnet_key

        try:
            set_id = validated_set_id(set_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        card_body = await eval_store.get_bytes(subnet_key(set_id, "set.json"))
        prompts_body = await eval_store.get_bytes(subnet_key(set_id, "prompts.jsonl"))
        if card_body is None or prompts_body is None:
            raise HTTPException(status_code=404, detail="set_unknown")
        return json.loads(card_body), prompts_body

    def upload_sampling_differs(sampling: Mapping, body: CreateJob) -> list[str]:
        """An order on an upload's group runs the sampling its lines asked for (``orders.uploads``)."""
        given = {**body.sampling.model_dump(), "max_tokens": body.max_new_tokens, "seed": body.seed,
                 "n": body.samples_per_prompt}
        return sorted(k for k, v in sampling.items()
                      if k in given and v is not None and given[k] != v)

    async def pinned_order_arguments(body: CreateJob, *, evaluation: bool, record_order: bool) -> dict:
        """A forced-draw order (eval or generation) on any source: no qualification. Its model is pinned from
        its own files (``orders.models``), its thresholds are its judge, and its declared conditions are the
        order record the order control checks the job against."""
        from reliquary.eval import qualification as qual
        from reliquary.eval.episode_jobs import set_contract_environment
        from reliquary.eval.prompt_source import eval_source_for, register_eval_prompts
        from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG

        if body.revision is None:
            raise HTTPException(status_code=422, detail="a forced-draw order pins its model revision")
        if body.sampling is None or body.max_new_tokens is None:
            raise HTTPException(status_code=422, detail="an order names its sampling and max_new_tokens")
        try:
            pin = await asyncio.to_thread(pin_model, body.model, body.revision)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)[:500]) from exc
        except Exception as exc:
            logger.warning("model files of %s@%s unavailable", body.model, body.revision, exc_info=True)
            raise HTTPException(status_code=503, detail="model_files_unavailable") from exc
        if evaluation:
            if body.eval_set_id is None:
                raise HTTPException(status_code=422, detail="an evaluation order names its eval_set_id")
            if body.audit_q not in (None, 1.0):
                raise HTTPException(status_code=422, detail="an eval job audits every submission (audit_q 1.0)")
        if body.eval_set_id is not None:
            if body.prompt_start != 0:
                raise HTTPException(status_code=422, detail="an order on a set starts at its first problem")
            card, prompts_body = await read_set(body.eval_set_id)
            if not evaluation and card.get("source_kind") != "upload":
                # Our sets are held out: their completions are never deliverable training data.
                raise HTTPException(status_code=422, detail="a generation order draws from an upload, not an eval set")
            if card.get("source_kind") == "upload":
                if (card.get("model"), card.get("revision")) != (body.model, body.revision):
                    raise HTTPException(status_code=422, detail="the upload was rendered for another model")
                if evaluation and not card.get("graded"):
                    raise HTTPException(status_code=422, detail="an evaluation needs a set with references")
                differs = upload_sampling_differs(card.get("sampling") or {}, body)
                if differs:
                    raise HTTPException(status_code=422, detail=f"upload_sampling_differs: {differs}")
                from reliquary.miner.corpus_miner import PROMPT_ALLOWANCE_TOKENS

                # Miners size their engine to the job's longest prompt: the upload's must fit.
                longest_prompt = int(card.get("prompt_tokens_max") or 0)
                if longest_prompt > (body.max_prompt_tokens or PROMPT_ALLOWANCE_TOKENS):
                    raise HTTPException(status_code=422, detail=(
                        f"max_prompt_tokens_too_small: the upload's longest prompt is {longest_prompt} tokens"))
            elif body.env not in (card.get("env"), card.get("source")):
                raise HTTPException(status_code=422, detail=f"set {body.eval_set_id} is a {card.get('env')} set")
            if body.prompt_count > int(card["count"]):
                raise HTTPException(status_code=422, detail=f"set {body.eval_set_id} holds {card['count']} problems")
            episode = order_episode(body, card, pin, evaluation=evaluation)
            source = eval_source_for(body.eval_set_id, prompts_body, body.prompt_count)
            register_eval_prompts(source, prompts_body)
            prompt_source, contract_environment = source.name, set_contract_environment(card)
            declared_source = {"set_id": source.set_id, "problems": body.prompt_count}
        else:
            if body.episode is not None:
                raise HTTPException(status_code=422, detail="an episode order plays a SWE, TMax or suite eval set")
            episode = None
            refusal = qual.order_environment_refusal(body.env)
            if refusal is not None:
                raise HTTPException(status_code=422, detail=refusal)
            await refuse_past_the_source(body.env, body.prompt_start, body.prompt_count)
            try:
                from reliquary.eval.sets import refuse_held_out_overlap

                refuse_held_out_overlap(body.env, body.prompt_start, body.prompt_count)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            prompt_source, contract_environment = body.env, None
            declared_source = {"env": body.env, "prompt_start": body.prompt_start, "problems": body.prompt_count,
                               "environment_manifest_sha256": getattr(
                                   ENVIRONMENT_CATALOG.get(body.env), "environment_manifest_sha256", None)}
        from reliquary.miner.corpus_miner import PROMPT_ALLOWANCE_TOKENS

        longest = max(PROMPT_ALLOWANCE_TOKENS, body.max_prompt_tokens or 0) + body.max_new_tokens
        if episode is not None:
            longest = max(longest, int(episode["max_total_tokens"]))
        if pin.max_position_embeddings is not None and longest > pin.max_position_embeddings:
            raise HTTPException(status_code=422, detail=(
                f"context_too_long: {longest} tokens, the model holds {pin.max_position_embeddings}"))
        seed = body.seed if body.seed is not None else job_seed(body.job_id)
        if record_order:
            from reliquary.corpus.job import draw_judge_from

            try:
                judge = draw_judge_from(body.judge or {}).to_contract()
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            try:
                await eval_jobs.create({
                    "schema": qual.ORDER_JOB_SCHEMA_V2, "job_id": body.job_id,
                    "kind": qual.EVALUATION if evaluation else qual.GENERATION, "pin": pin.to_dict(),
                    "source": declared_source, "contract_environment": contract_environment,
                    "samples": body.samples_per_prompt, "n": 1, "min_new_tokens": 2,
                    "sampling": body.sampling.conditions(), "max_new_tokens": body.max_new_tokens,
                    "max_prompt_tokens": body.max_prompt_tokens,
                    # An episode renders through its own renderer: no thinking toggle.
                    "thinking": None if episode is not None else body.thinking, "episode": episode,
                    "draw": body.draw, "judge": judge, "seed": seed,
                    "audit_q": 1.0 if evaluation else (body.audit_q or GEN_AUDIT_PARAMS["audit_q"]),
                    **({"reuse_auditors_only": True} if body.reuse_auditors_only else {}),
                    **audit_executors_of(body)})
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        audit = {"audit_q": 1.0} if evaluation else {
            **GEN_AUDIT_PARAMS, **({"audit_q": body.audit_q} if body.audit_q is not None else {})}
        return dict(
            model_revision=pin.revision, model_architecture=pin.architecture,
            checkpoint_sha256=pin.checkpoint_sha256, eos_token_id=pin.eos_token_id,
            prompt_source=prompt_source, contract_environment=contract_environment, seed=seed,
            audit_params=audit, temperature=body.sampling.temperature, top_p=body.sampling.top_p,
            top_k=body.sampling.top_k, **draw_arguments(body),
            # Its miners submit on its own route (the order control has no legacy one).
            **({} if evaluation else {"submit": "scoped"}),
            **({"episode": episode, "renderer_id": episode["renderer"]} if episode is not None else {}),
        )

    def order_episode(body: CreateJob, card: Mapping, pin, *, evaluation: bool) -> dict | None:
        """The order's episode as its manifest carries it, None on a single-turn set: on a SWE or bridged (TMax)
        set a signed-sandbox episode, on a suite set (``episode``: a replayed tool world) a suite one, that agrees
        with the set (``eval.episode_jobs``), on a model that has its renderer."""
        from reliquary.corpus.job import episode_contract
        from reliquary.eval.episode_jobs import episode_renderer_refusal, refuse_episode_disagreeing_with_set

        interaction = card.get("interaction", "single_turn")
        if interaction == "single_turn":
            if body.episode is not None:
                raise HTTPException(status_code=422, detail=f"set {body.eval_set_id} holds single-turn problems: "
                                    "an order on it has no episode")
            return None
        if not evaluation or interaction not in ("swe", "signed_bridged", "episode"):
            raise HTTPException(status_code=422, detail=(
                f"set {body.eval_set_id} holds {interaction} tasks: an order plays only SWE or TMax sets "
                "(in the signed sandbox) or suite episode sets (replayed), as an evaluation"))
        if body.thinking:
            raise HTTPException(status_code=422, detail="an episode renders through its own renderer: thinking is false")
        try:
            refuse_episode_disagreeing_with_set(card, body.eval_set_id, body.episode,
                                                (body.episode or {}).get("renderer"))
            episode = episode_contract(body.episode, eval_job=True)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)[:500]) from exc
        refusal = episode_renderer_refusal(pin.architecture, episode["renderer"],
                                           suite=episode.get("harness") == "suite")
        if refusal is not None:
            raise HTTPException(status_code=422, detail=refusal)
        return episode

    def operator_generation_job(body: CreateJob) -> bool:
        from reliquary.eval.prompt_source import gen_job_prefix

        prefix = gen_job_prefix(task_prefix) + "ops-"
        if not (body.job_id.startswith(prefix) or (body.task_id or "").startswith(prefix)):
            return False
        if (body.task_id != body.job_id or not re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                body.job_id.removeprefix(prefix))):
            raise HTTPException(status_code=422, detail="operator_generation_ids_must_match_uuid4")
        return True

    async def prepare_job(body: CreateJob):
        from reliquary.eval.prompt_source import eval_job_prefix, gen_job_prefix

        eval_prefix = eval_job_prefix(task_prefix)
        gen_prefix = gen_job_prefix(task_prefix)
        if not (body.job_id.startswith(task_prefix)
                and (body.task_id or body.job_id).startswith(task_prefix)):
            raise HTTPException(status_code=422, detail="task_id_outside_admin_scope")
        task_id = body.task_id or body.job_id
        for order_prefix in (eval_prefix, gen_prefix):
            if task_id.startswith(order_prefix) and not body.job_id.startswith(order_prefix):
                raise HTTPException(status_code=422, detail=(
                    f"an {order_prefix!r} task id names an {order_prefix!r} job id"))
        evaluation = body.job_id.startswith(eval_prefix)
        if evaluation and not (body.task_id or body.job_id).startswith(eval_prefix):
            raise HTTPException(status_code=422, detail=(
                f"an eval job's ids start with {eval_prefix!r}, and only an eval job's"))
        # Pinned operator work has its own exact namespace and catalog path.
        # Other generation orders are qualified, or forced-draw orders pinned without one.
        operator_generation = operator_generation_job(body)
        generation = body.job_id.startswith(gen_prefix) and not operator_generation
        if generation and not (body.task_id or body.job_id).startswith(gen_prefix):
            raise HTTPException(status_code=422, detail=(
                f"a generation order's ids both start with {gen_prefix!r}"))
        # A forced-draw order needs no qualification: one path for eval and generation, on a set or an env.
        pinned = (evaluation or generation) and body.draw is not None and body.qualification_id is None
        if body.eval_set_id is not None and not (evaluation or (generation and pinned)):
            raise HTTPException(status_code=422, detail=(
                f"an eval job's ids start with {eval_prefix!r}, and only an eval job's"))
        if body.reuse_auditors_only and not (pinned and body.eval_set_id is not None and body.episode is None):
            raise HTTPException(status_code=422, detail="auditor reuse requires a pinned draw prompt set")
        if body.revision is not None and not pinned:
            raise HTTPException(status_code=422, detail="revision pins a forced-draw order without a qualification")
        if generation and body.seed is not None and body.draw is None:
            raise HTTPException(status_code=422, detail="eval fields on a generation order: a seed keys a forced draw")
        if not evaluation and not generation and any(
                v is not None for v in (body.qualification_id, body.seed, body.audit_q, body.draw, body.judge,
                                        body.audit_executors)):
            raise HTTPException(status_code=422, detail="eval fields on a non-eval job")
        if body.max_price_usd_per_mtoken is not None and not (evaluation or generation):
            raise HTTPException(status_code=422, detail=(
                "max_price_usd_per_mtoken applies to eval and generation orders"))
        cap = body.cap
        if body.episode is not None and not (pinned and evaluation):
            raise HTTPException(status_code=422, detail="an episode order is a forced-draw evaluation (draw, no "
                                "qualification)")
        if pinned:
            arguments = await pinned_order_arguments(body, evaluation=evaluation, record_order=False)
            cap = (DEFAULT_EVAL_CAP if evaluation else DEFAULT_GEN_CAP) if cap is None else cap
        elif evaluation:
            arguments = await eval_job_arguments(body, record_order=False)
            cap = DEFAULT_EVAL_CAP if cap is None else cap
        elif generation:
            arguments = await gen_job_arguments(body, record_order=False)
            cap = DEFAULT_GEN_CAP if cap is None else cap
        else:
            if cap is None:
                raise HTTPException(status_code=422, detail="cap_required")
            if body.sampling is not None:
                raise HTTPException(status_code=422, detail="eval fields on a non-eval job")
            spec = catalog.get(body.model)
            if spec is None:
                raise HTTPException(status_code=422, detail=f"model {body.model!r} is not qualified")
            from reliquary.environment.registry import ENVIRONMENT_SPECS

            env = ENVIRONMENT_SPECS.get(body.env)
            if env is None or getattr(env, "interaction_mode", None) != "single_turn":
                raise HTTPException(status_code=422,
                                    detail=f"env {body.env!r} is not a single-turn catalog source")
            arguments = dict(model_revision=spec.revision, model_architecture=spec.architecture,
                             checkpoint_sha256=spec.checkpoint_sha256,
                             eos_token_id=spec.eos_token_id, prompt_source=body.env,
                             audit_params={})
            if operator_generation:
                arguments["submit"] = "scoped"
        # An episode order's renderer is its episode's.
        renderer_id = arguments.pop("renderer_id", THINKING_RENDERERS[body.thinking])
        try:
            manifest, entry = await asyncio.to_thread(
                prepare, job_id=body.job_id, task_id=body.task_id, model=body.model,
                from_profile=None, prompt_encoding=None, prompt_count=body.prompt_count,
                prompt_start=body.prompt_start, renderer_id=renderer_id,
                slots_per_prompt=body.samples_per_prompt,
                max_new_tokens=body.max_new_tokens, cap=cap, min_incentive_share=0.0,
                **({"max_prompt_tokens": body.max_prompt_tokens} if body.max_prompt_tokens is not None else {}),
                **arguments,
            )
        except (RegistryError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        from reliquary.protocol.release_contract import canonical_sha256

        if ((body.manifest_sha256 is not None and body.manifest_sha256 != canonical_sha256(manifest))
                or (body.profile_sha256 is not None and body.profile_sha256 != entry.profile_sha256)):
            raise HTTPException(status_code=409, detail="reviewed_contract_changed")
        return manifest, entry

    async def token_price(ceiling: TokenCeiling) -> dict:
        """The registry params of an order's USD ceiling at today's rates: alpha
        per million tokens = USD / (alpha price in TAO x TAO/USD), then a pool
        share per token at the miners' emission per period (as ``tasks set-cap``)."""
        from reliquary.validator import corpus_periods as cp

        try:
            pool = await subnet_pool()
            tao = float(await tao_usd())
            alpha_usd = float(pool["alpha_price_tao"]) * tao
            emission = float(pool["miner_alpha_per_block"]) * cp.PERIOD_SECONDS / 12.0
            return cp.usd_price_params(
                usd_per_mtoken_input=ceiling.input, usd_per_mtoken_output=ceiling.output,
                alpha_usd=alpha_usd, tao_usd=tao, miner_alpha_per_period=emission,
                set_at=float(clock()))
        except Exception as exc:
            logger.warning("admin: the token ceiling could not be converted: %r", exc)
            raise HTTPException(status_code=503, detail="token_price_unavailable") from exc

    async def sized_entry(entry, job, audit_executors: int | None):
        """With ``RELIQUARY_ORDER_CAP_SIZING`` on, the entry's cap sized to what its audit lets it pay
        (``admin.cap_sizing``), the asked cap kept as ``cap_max``; else the entry as it is."""
        from dataclasses import replace

        from reliquary.admin import cap_sizing as sizing
        from reliquary.validator.eval_control import read_control_status

        if not sizing.sizing_enabled():
            return entry
        try:
            status = await read_control_status()
        except Exception:
            logger.warning("admin: the control status is unreadable; sizing on default rates", exc_info=True)
            status = None
        completion, prompt = sizing.model_rates(status, f"{job.checkpoint_repo}@{job.checkpoint_revision}")
        estimate = sizing.spend_estimate(entry.params, completion_tps=completion, prompt_tps=prompt,
                                         cards=audit_executors or 1)
        if estimate is None:
            return entry
        asked = float(entry.params["cap"])
        cap = sizing.sized_cap(asked, estimate=estimate)
        return replace(entry, params={**entry.params, "cap": cap, "floor": cap, sizing.CAP_MAX_PARAM: asked})

    async def resize_order_caps() -> int:
        """Resize every active order task with a ``cap_max`` from the control's status; the count changed."""
        from reliquary.admin import cap_sizing as sizing
        from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION, RegistryError
        from reliquary.validator.eval_control import read_control_status

        status = await read_control_status()
        if not status or clock() - float(status.get("at") or 0.0) > sizing.STATUS_MAX_AGE_SECONDS:
            return 0
        jobs = status.get("jobs") or {}
        entries, _ = await registry_store.read_registry(strict=False)
        changed = 0
        for entry in list(entries.values()):
            if (entry.status != "active" or entry.mechanism != MECHANISM_CORPUS_GENERATION
                    or sizing.CAP_MAX_PARAM not in entry.params):
                continue
            facts = (jobs.get(entry.job_id) or {}).get("sizing")
            cap = None if not facts else sizing.resized(entry.params, facts, status)
            if cap is None:
                continue
            try:
                await registry_store.set_task_cap(entry.task_id, cap, guard=await draining_limits(cap))
            except (RegistryError, registry_store.RegistryConflict) as exc:
                logger.warning("admin: cap of %s stays %s (sized %s): %s", entry.task_id,
                               entry.params["cap"], cap, exc)
                continue
            logger.info("admin: cap of %s %s -> %s", entry.task_id, entry.params["cap"], cap)
            changed += 1
        return changed

    async def resize_forever() -> None:
        from reliquary.admin.cap_sizing import RESIZE_SECONDS

        while True:
            try:
                await resize_order_caps()
            except Exception:
                logger.exception("admin: order cap resize failed; retrying")
            await asyncio.sleep(RESIZE_SECONDS)

    def order_pricing() -> dict:
        """An order job with a ceiling is priced by auction (``corpus.token_auction``) when
        ``RELIQUARY_ORDER_PRICING`` says ``auction``; else first come at the ceiling."""
        from reliquary.corpus.token_auction import PRICING_AUCTION, PRICING_PARAM

        chosen = os.environ.get(ORDER_PRICING_ENV, "").strip()
        if chosen not in ("", PRICING_AUCTION):
            raise HTTPException(status_code=503, detail="order_pricing_misconfigured")
        return {PRICING_PARAM: PRICING_AUCTION} if chosen == PRICING_AUCTION else {}

    def task_document(entry) -> dict:
        from reliquary.shared.task_registry import render_registry

        return {"task_id": entry.task_id, "admission": entry.admission,
                **json.loads(render_registry({entry.task_id: entry}))["tasks"][entry.task_id]}

    def job_contract(manifest, entry) -> dict:
        from reliquary.corpus.job import parse_job
        from reliquary.protocol.release_contract import canonical_sha256

        wanted = parse_job(manifest).to_contract()
        return {"schema": "subnet-task-contract/v1", "manifest": wanted,
                "manifest_sha256": canonical_sha256(wanted),
                "profile_sha256": entry.profile_sha256, "task": task_document(entry)}

    def refuse_changed_task(existing, expected) -> None:
        from reliquary.admin.cap_sizing import CAP_MAX_PARAM

        # A sized cap (``admin.cap_sizing``) moves; the cap the order asked for is kept beside it.
        asked = existing.params.get(CAP_MAX_PARAM, existing.params["cap"])
        if (existing.profile_sha256 != expected.profile_sha256
                or float(asked) != float(expected.params["cap"])):
            raise HTTPException(status_code=409, detail="task_exists_with_another_contract")

    @router.get("/task-catalog")
    async def task_catalog() -> dict:
        from reliquary.environment.agentic.external import verify_external_artifact
        from reliquary.environment.registry import ENVIRONMENT_SPECS
        from reliquary.eval.qualification import order_environment_refusal
        from reliquary.protocol.environment_catalog import (
            ENVIRONMENT_CATALOG, environment_body_contract,
        )
        from reliquary.protocol.release_contract import canonical_sha256
        from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION

        environments = []
        for name in sorted(set(ENVIRONMENT_CATALOG) & set(ENVIRONMENT_SPECS)):
            spec = ENVIRONMENT_SPECS[name]
            supported = spec.interaction_mode == "single_turn"
            if supported and spec.external_distribution is not None:
                try:
                    await asyncio.to_thread(verify_external_artifact, spec)
                except (OSError, TypeError, ValueError):
                    supported = False
            contract = environment_body_contract(name)
            environments.append({
                "environment": name, "interaction_mode": spec.interaction_mode,
                "contract": contract, "contract_sha256": canonical_sha256(contract),
                "legacy_generation_supported": supported,
                "qualified_generation_supported": supported and order_environment_refusal(name) is None,
            })
        return {"schema": "subnet-task-catalog/v1", "task_prefix": task_prefix,
                "pool_max": float(pool_max), "zero_cap_supported": True,
                "admission_controls_supported": True,
                "requires_matching_corpus_control": True,
                "mechanisms": [MECHANISM_CORPUS_GENERATION],
                "models": [{"model": name, **spec.model_dump()}
                           for name, spec in sorted(catalog.items())],
                "environments": environments,
                "limits": {"max_prompts": 10000, "max_samples": 16,
                           "max_new_tokens": 32768},
                "renderers": dict(THINKING_RENDERERS),
                # A create accepts max_price_usd_per_mtoken on eval and generation orders.
                "orders": {"token_ceilings": True}}

    @router.get("/tasks")
    async def list_tasks() -> dict:
        entries, _ = await registry_store.read_registry(strict=False)
        return {"schema": "subnet-task-list/v1", "tasks": [task_document(e)
                for name, e in sorted(entries.items()) if name.startswith(task_prefix)]}

    @router.get("/tasks/{task_id}")
    async def read_task(task_id: str) -> dict:
        in_scope(task_id)
        entries, _ = await registry_store.read_registry(strict=False)
        entry = entries.get(task_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="task_unknown")
        return {"schema": "subnet-task/v1", "task": task_document(entry)}

    @router.post("/jobs/validate")
    async def validate_job(body: CreateJob) -> dict:
        manifest, entry = await prepare_job(body)
        entries, _ = await registry_store.read_registry(strict=False)
        named = [e for e in entries.values() if e.job_id == body.job_id]
        existing, _ = await job_store.read_job(body.job_id)
        if existing is not None and existing.to_contract() != job_contract(manifest, entry)["manifest"]:
            raise HTTPException(status_code=409, detail="job_exists_with_another_manifest")
        if named and any(e.task_id != entry.task_id for e in named):
            raise HTTPException(status_code=409, detail="job_already_declared")
        for existing_task in named:
            refuse_changed_task(existing_task, entry)
        if not named:
            try:
                from reliquary.shared.task_registry import add_task, require_default_declared_first

                require_default_declared_first(entries, entry)
                updated = add_task(entries, entry)
                (await draining_limits(float(entry.params["cap"])))(entries, updated)
            except RegistryError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        return job_contract(manifest, entry)

    @router.post("/jobs")
    async def create_job(body: CreateJob, response: Response) -> dict:
        from reliquary.corpus.job import parse_job
        from reliquary.eval.prompt_source import gen_job_prefix

        manifest, entry = await prepare_job(body)
        cap = float(entry.params["cap"])
        from reliquary.eval.prompt_source import eval_job_prefix

        evaluation = body.job_id.startswith(eval_job_prefix(task_prefix))
        if ((evaluation or body.job_id.startswith(gen_job_prefix(task_prefix)))
                and body.draw is not None and body.qualification_id is None
                and not operator_generation_job(body)):
            await pinned_order_arguments(body, evaluation=evaluation, record_order=True)
        elif evaluation:
            await eval_job_arguments(body, record_order=True)
        elif (body.job_id.startswith(gen_job_prefix(task_prefix))
              and not operator_generation_job(body)):
            await gen_job_arguments(body, record_order=True)
        answer = {"job_id": body.job_id, "task_id": entry.task_id}
        wanted = parse_job(manifest).to_contract()

        async def same_manifest_stored() -> bool:
            existing, _ = await job_store.read_job(body.job_id)
            if existing is None:
                return False
            if existing.to_contract() != wanted:
                raise HTTPException(status_code=409, detail="job_exists_with_another_manifest")
            return True

        if not await same_manifest_stored():
            try:
                # Create-only, and never deleted: a racing call's task may name it.
                await job_store.write_job(manifest, None)
            except job_store.CorpusStoreConflict:
                if not await same_manifest_stored():
                    raise HTTPException(status_code=503, detail="job_manifest_raced") from None
        named = await entries_naming(body.job_id)
        for other in named:
            if other.task_id == entry.task_id:
                refuse_changed_task(other, entry)
                response.status_code = 200
                return idempotent(answer, other)
        if named:
            raise HTTPException(status_code=409, detail=(
                f"job {body.job_id!r} is already declared by task {named[0].task_id!r}"))
        if body.max_price_usd_per_mtoken is not None:
            # Converted once, here: a replay answers the stored task above, whatever the rate is by then.
            from dataclasses import replace

            entry = replace(entry, params={**entry.params,
                                           **await token_price(body.max_price_usd_per_mtoken),
                                           **order_pricing()})
            entry = await sized_entry(entry, parse_job(manifest), body.audit_executors)
        try:
            await registry_store.create_task(entry, guard=await draining_limits(cap))
        except (RegistryError, registry_store.RegistryConflict) as exc:
            # Lost to an identical concurrent call: its task is this answer.
            for other in await entries_naming(body.job_id):
                if other.task_id == entry.task_id:
                    refuse_changed_task(other, entry)
                    response.status_code = 200
                    return idempotent(answer, other)
            status = 409 if isinstance(exc, RegistryError) else 503
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        logger.info("admin: declared job %s as task %s, cap %s", body.job_id, entry.task_id,
                    cap)
        response.status_code = 201
        created = {**answer, "created": True, "status": "active", "cap": float(cap)}
        price = price_of(entry)
        return created if price is None else {**created, "max_price": price}

    async def corpus_entry(task_id: str):
        """The task, refused unless it is a corpus task (never ``default``, never RL)."""
        from reliquary.shared.task_id import DEFAULT_TASK_ID
        from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION

        in_scope(task_id)
        entries, _ = await registry_store.read_registry(strict=False)
        entry = entries.get(task_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="task_unknown")
        if task_id == DEFAULT_TASK_ID or entry.mechanism != MECHANISM_CORPUS_GENERATION:
            raise HTTPException(status_code=409, detail="not_a_corpus_task")
        return entry

    @router.post("/tasks/{task_id}/cap")
    async def set_cap(task_id: str, body: SetCap) -> dict:
        await corpus_entry(task_id)
        try:
            await registry_store.set_task_cap(task_id, body.cap, guard=await draining_limits(body.cap))
        except RegistryError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except registry_store.RegistryConflict as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"task_id": task_id, "cap": body.cap}

    @router.post("/tasks/{task_id}/admission")
    async def set_admission(task_id: str, body: SetAdmission) -> dict:
        await corpus_entry(task_id)
        try:
            await registry_store.set_task_admission(task_id, body.admission)
        except RegistryError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except registry_store.RegistryConflict as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"task_id": task_id, "admission": body.admission, "status": "active"}

    @router.post("/tasks/{task_id}/retire")
    async def retire(task_id: str, body: Retire) -> dict:
        entry = await corpus_entry(task_id)
        if entry.status == "retired":
            return {"task_id": task_id, "status": "retired", "retired_at": entry.retired_at}
        try:
            stamp = body.retired_at if body.retired_at is not None else current_round()
            await registry_store.retire_task_entry(task_id, stamp)
        except (RegistryError, registry_store.RegistryConflict) as exc:
            # A concurrent retire landed first: its stamp is the answer.
            entry = await corpus_entry(task_id)
            if entry.status == "retired":
                return {"task_id": task_id, "status": "retired", "retired_at": entry.retired_at}
            status = 409 if isinstance(exc, RegistryError) else 503
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"task_id": task_id, "status": "retired", "retired_at": stamp}

    @router.get("/jobs/{job_id}/status")
    async def job_status(job_id: str) -> dict:
        from reliquary.protocol.release_contract import canonical_sha256

        in_scope(job_id)
        try:
            job, _ = await job_store.read_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="job_unknown") from exc
        if job is None:
            raise HTTPException(status_code=404, detail="job_unknown")
        counts = await job_counts(job_id)
        entries = await entries_naming(job_id)
        pay = await job_miner_pay(job_id, entries)
        tasks = [{"task_id": e.task_id, "status": e.status, "cap": float(e.params["cap"]),
                  **({"max_price": price_of(e)} if price_of(e) is not None else {})}
                 for e in entries]
        manifest = job.to_contract()
        return {"job_id": job_id, **counts, "miner_pay": pay, "manifest": manifest, "tasks": tasks,
                "manifest_sha256": canonical_sha256(manifest),
                "profile_sha256": entries[0].profile_sha256 if len(entries) == 1 else None,
                "task_contracts": [task_document(e) for e in entries]}

    if sandbox_machine_prefix is not None:
        from reliquary.infrastructure import sandbox_store

        machines = sandbox_machines or sandbox_store
        machine_prefix = sandbox_store.validated_machine_prefix(sandbox_machine_prefix)

        def own_machine(machine_id: str) -> str:
            try:
                sandbox_store.validated_machine_id(machine_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail="sandbox_machine_unknown") from exc
            if not machine_id.startswith(machine_prefix):
                raise HTTPException(status_code=409, detail="outside_admin_scope")
            return machine_id

        @router.post("/sandbox-machines")
        async def register_sandbox_machine(body: RegisterSandboxMachine, response: Response) -> dict:
            own_machine(body.machine_id)
            try:
                document, created = await machines.register_machine(
                    machine_id=body.machine_id, address=body.address, provider=body.provider,
                    capacity=body.capacity, key_id=body.key_id, public_key_b64=body.public_key_b64,
                    valid_from=body.valid_from, now=clock())
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except machines.MachineConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            response.status_code = 201 if created else 200
            return document

        @router.post("/sandbox-machines/{machine_id}/status")
        async def sandbox_machine_status(machine_id: str, body: SandboxMachineStatus) -> dict:
            document = await machines.set_machine_status(own_machine(machine_id), body.status,
                                                         reason=body.reason)
            if document is None:
                raise HTTPException(status_code=404, detail="sandbox_machine_unknown")
            return document

        @router.get("/sandbox-machines/{machine_id}")
        async def read_sandbox_machine(machine_id: str) -> dict:
            # last_heartbeat: when a control last accepted the machine's signed report.
            document = await machines.read_machine(own_machine(machine_id))
            if document is None:
                raise HTTPException(status_code=404, detail="sandbox_machine_unknown")
            return document

    @router.post("/executors")
    async def register_executor(body: RegisterExecutor, response: Response) -> dict:
        try:
            document, created = await executors.register_executor(
                executor_id=body.executor_id, token_sha256=body.token_sha256,
                model_id=body.model_id, model_revision=body.model_revision,
                expires_at=body.expires_at, now=clock(), provider_id=body.provider_id,
                host=body.host, scope=body.scope)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except executors.ExecutorConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        response.status_code = 201 if created else 200
        return _public_executor(document)

    @router.delete("/executors/{executor_id}")
    async def revoke_executor(executor_id: str) -> dict:
        try:
            document = await executors.set_executor_status(executor_id, "revoked",
                                                           reason="revoked by the platform")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="executor_unknown") from exc
        if document is None:
            raise HTTPException(status_code=404, detail="executor_unknown")
        return _public_executor(document)

    @router.get("/executors/{executor_id}")
    async def read_executor(executor_id: str) -> dict:
        try:
            document = await executors.read_executor(executor_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="executor_unknown") from exc
        if document is None:
            raise HTTPException(status_code=404, detail="executor_unknown")
        return _public_executor(document)

    async def v3_delivery(job, delivery_id: str, body: CreateDelivery, grade) -> dict:
        """A v3 delivery, with the job's ledger (the prompts that used every attempt) and its frozen set's rows
        (custom ids); ``grade`` is the job's grader when a filter is asked for."""
        from reliquary.corpus.delivery import export_delivery, set_rows_for_job
        from reliquary.validator.corpus_service import rebuild_ledgers

        mode = body.filter or "none"
        snapshot, _ = await job_store.read_ledgers(job.job_id)
        slots = rebuild_ledgers(job, snapshot).slots
        set_rows = await asyncio.to_thread(set_rows_for_job, job)
        return await export_delivery(
            job=job, records=records, sink=deliveries, delivery_id=delivery_id, grade=grade,
            work_dir=work_dir, clock=clock, dedupe=body.dedupe, dedupe_against=body.dedupe_against,
            schema_version=3, include_truncated=body.include_truncated is not False, filter_mode=mode,
            output_format=body.format or "parquet", slots=slots, set_rows=set_rows)

    @router.post("/jobs/{job_id}/deliveries")
    async def create_delivery(job_id: str, body: CreateDelivery, response: Response) -> dict:
        from reliquary.corpus.delivery import (
            export_delivery, instruction_source_for_job, validate_cached_delivery,
            validated_delivery_id,
        )
        from reliquary.corpus.export import job_grader

        in_scope(job_id)
        if deliveries is None:
            raise HTTPException(status_code=503, detail="deliveries_not_configured")
        try:
            delivery_id = validated_delivery_id(body.delivery_id or job_id)
            job, _ = await job_store.read_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if job is None:
            raise HTTPException(status_code=404, detail="job_unknown")
        for other in body.dedupe_against:
            in_scope(other)
            if other == job_id:
                raise HTTPException(status_code=422, detail="dedupe_against_itself")
            try:
                known, _ = await job_store.read_job(other)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if known is None:
                raise HTTPException(status_code=422, detail=f"dedupe_against_unknown: {other}")
        accepts = getattr(deliveries, "accepts_delivery_id", None)
        if accepts is not None and not accepts(delivery_id):
            raise HTTPException(status_code=503, detail="delivery_namespace_not_configured")
        if job.episode is not None:
            raise HTTPException(status_code=422, detail="an episode job is delivered with "
                                "`reliquary jobs export JOB --sft` on a host with its renderer")

        from reliquary.corpus.delivery import DELIVERY_SCHEMA, DELIVERY_SCHEMA_V3

        # What this request asks for: a delivery under its id answers it only if it was asked the same.
        asked = {"schema": DELIVERY_SCHEMA_V3 if body.schema_version == 3 else DELIVERY_SCHEMA}
        if body.schema_version == 3:
            asked["options"] = {"format": body.format or "parquet",
                                "include_truncated": True if body.include_truncated is None else body.include_truncated,
                                "filter": body.filter or "none", "dedupe_against": sorted(set(body.dedupe_against))}

        def same_request(manifest) -> bool:
            # A v2 request takes any delivery written before v3 existed, as it always did.
            if (manifest.get("schema") == DELIVERY_SCHEMA_V3) != (asked["schema"] == DELIVERY_SCHEMA_V3):
                return False
            options = manifest.get("options") or {}
            return "options" not in asked or all(
                (sorted(options.get(k) or ()) if k == "dedupe_against" else options.get(k)) == v
                for k, v in asked["options"].items())

        def delivery_result(manifest):
            if not same_request(manifest):
                raise HTTPException(status_code=409, detail="delivery_exists_with_another_request")
            try:
                validate_cached_delivery(manifest, job=job, sink=deliveries)
            except ValueError as exc:
                detail = ("delivery_belongs_to_another_job" if manifest.get("job_id") != job_id
                          else "delivery_belongs_to_another_contract")
                raise HTTPException(status_code=409, detail=detail) from exc
            return {"state": "done", "delivery_id": delivery_id, "keys": manifest["keys"],
                    "rows": manifest["rows"]}

        lock = delivery_locks.setdefault(delivery_id, asyncio.Lock())
        async with lock:
            running_export = exports.get(delivery_id)
            if running_export is not None and running_export[0] != job_id:
                raise HTTPException(status_code=409, detail="delivery_belongs_to_another_job")
            if running_export is not None and running_export[2] != asked:
                raise HTTPException(status_code=409, detail="delivery_exists_with_another_request")
            running = running_export[1] if running_export is not None else None
            if running is not None and running.done():
                exports.pop(delivery_id)
                if running.exception() is not None:
                    raise HTTPException(status_code=500,
                                        detail=f"delivery failed: {running.exception()}")
                manifest = running.result()
                return delivery_result(manifest)
            if running is None:
                stored = await deliveries.get_json(f"deliveries/{delivery_id}/manifest.json")
                if stored is not None:
                    return delivery_result(stored)
                # A delivery is final once written: never from a job still moving.
                if not (await job_counts(job_id))["final"]:
                    raise HTTPException(status_code=409, detail="job_not_drained")
                if body.schema_version == 3:
                    grade = None
                    if (body.filter or "none") != "none":
                        if job.filter is None:
                            raise HTTPException(status_code=422, detail=(
                                f"filter {body.filter!r} needs the job's declared filter: the job declares none"))
                        try:
                            grade = await asyncio.to_thread(job_grader, job)
                        except ValueError as exc:
                            raise HTTPException(status_code=422, detail=f"filter_unavailable: {exc}") from exc
                    exports[delivery_id] = (job_id, asyncio.ensure_future(
                        v3_delivery(job, delivery_id, body, grade)), asked)
                    response.status_code = 202
                    return {"state": "running", "delivery_id": delivery_id}
                grade, note = None, None
                if body.apply_filter and job.filter is not None:
                    try:
                        grade = await asyncio.to_thread(job_grader, job)
                    except ValueError as exc:
                        note = str(exc)
                if body.keep_rejected and grade is None:
                    raise HTTPException(status_code=422, detail="keep_rejected needs the job's "
                                        "filter applied: " + (note or "apply_filter is off or the "
                                                              "job declares no filter"))
                instruction_source, instruction_note = await asyncio.to_thread(
                    instruction_source_for_job, job)
                # Long: run beside the request; the caller polls with the same id.
                exports[delivery_id] = (job_id, asyncio.ensure_future(export_delivery(
                    job=job, records=records, sink=deliveries, delivery_id=delivery_id,
                    grade=grade, filter_note=note, work_dir=work_dir, clock=clock,
                    instruction_source=instruction_source, instruction_note=instruction_note,
                    keep_rejected=body.keep_rejected, dedupe=body.dedupe,
                    dedupe_against=body.dedupe_against)), asked)
            response.status_code = 202
            return {"state": "running", "delivery_id": delivery_id}

    @router.post("/uploads")
    async def ingest_upload(body: IngestUpload) -> dict:
        """A customer's batch file (OpenAI Batch JSONL in the platform bucket) as frozen
        prompt sets, one per sampling (``orders.uploads``); idempotent on the request."""
        from reliquary.eval.storage import SetConflict
        from reliquary.orders import uploads

        if deliveries is None:
            raise HTTPException(status_code=503, detail="platform_bucket_unavailable")
        if body.grader is not None:
            from reliquary.orders.graders import REFERENCE_GRADERS

            if body.grader not in REFERENCE_GRADERS:
                raise HTTPException(status_code=422, detail=(
                    f"grader_unknown: {body.grader!r}; graders: {sorted(REFERENCE_GRADERS)}"))
        if upload_tokenizer is None:
            from reliquary.validator.eval_control import load_cpu_tokenizer as load_tokenizer
        else:
            load_tokenizer = upload_tokenizer
        try:
            writer = getattr(deliveries, "put_upload_errors", None)
            errors_sink = (SimpleNamespace(put_bytes=partial(writer, object_key=body.object_key))
                           if callable(writer) else deliveries)
            return await uploads.ingest(
                object_key=body.object_key, sha256=body.sha256, model=body.model,
                revision=body.revision, purpose=body.purpose,
                max_prompt_tokens=body.max_prompt_tokens, grader=body.grader,
                read_object=deliveries.get_bytes, load_tokenizer=load_tokenizer,
                eval_store=eval_store, errors_sink=errors_sink)
        except uploads.UploadRefused as exc:
            raise HTTPException(status_code=422, detail=str(exc)[:500]) from exc
        except SetConflict as exc:
            raise HTTPException(status_code=409, detail=f"set_conflict: {exc}"[:500]) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"model_files_unreadable: {exc}"[:500]) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("upload %s could not be ingested", body.object_key, exc_info=True)
            raise HTTPException(status_code=503, detail="upload_unavailable") from exc

    @router.get("/architectures")
    async def architectures() -> dict:
        """The model architectures orders on any model may use, with the
        evidence each entry rests on (``SUPPORTED_ARCHITECTURES``)."""
        from reliquary.constants import SUPPORTED_ARCHITECTURES

        from reliquary.orders.graders import REFERENCE_GRADERS
        from reliquary.orders.models import MAX_CHECKPOINT_BYTES

        return {"schema": "reliquary/architectures/v1",
                "architectures": sorted(SUPPORTED_ARCHITECTURES),
                "evidence": {k: list(v) for k, v in SUPPORTED_ARCHITECTURES.items()},
                # What a forced-draw order may name (``orders``): its model's weights and its set's graders.
                # token_ceilings: a create accepts max_price_usd_per_mtoken.
                "orders": {"max_checkpoint_bytes": MAX_CHECKPOINT_BYTES, "token_ceilings": True,
                           "reference_graders": {name: g.description
                                                 for name, g in sorted(REFERENCE_GRADERS.items())}}}

    @router.get("/eval-control/status")
    async def eval_control_status() -> dict:
        """The eval control's last status: per model ``executors_needed`` (the
        executors to rent, on distinct providers and hosts) and the jobs that
        need attention."""
        from reliquary.validator.eval_control import read_control_status

        document = await read_control_status()
        if document is None:
            raise HTTPException(status_code=404, detail="eval_control_status_unknown")
        return document

    @router.post("/qualifications")
    async def request_qualification(body: RequestQualification, response: Response) -> dict:
        """Queue a model's qualification for the eval control; idempotent on the id."""
        from reliquary.eval import qualification as qual
        from reliquary.eval.sets import validated_set_id
        from reliquary.eval.storage import subnet_key
        from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict

        in_scope(body.qualification_id)
        try:
            qual.validated_qualification_id(body.qualification_id)
            if body.set_id is not None:
                validated_set_id(body.set_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if body.set_id is not None and await eval_store.get_bytes(
                subnet_key(body.set_id, "prompts.jsonl")) is None:
            raise HTTPException(status_code=404, detail="set_unknown")
        try:
            if body.env is not None:
                wanted = qual.new_generation_request(
                    qualification_id=body.qualification_id, model=body.model,
                    revision=body.revision, env=body.env,
                    prompt_start=body.prompt_start or 0, problems=body.problems,
                    completions=body.completions, sampling=body.sampling.conditions(),
                    max_new_tokens=body.max_new_tokens, thinking=body.thinking, clock=clock)
            else:
                wanted = qual.new_request(
                    qualification_id=body.qualification_id, model=body.model,
                    revision=body.revision, set_id=body.set_id, problems=body.problems,
                    completions=body.completions, sampling=body.sampling.conditions(),
                    max_new_tokens=body.max_new_tokens, thinking=body.thinking, clock=clock)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if body.env is not None:
            await refuse_past_the_source(body.env, wanted["prompt_start"], body.problems)
        existing, _ = await qualifications.read(body.qualification_id)
        if existing is None:
            # Before any executor is rented: a model that can never be served.
            await refuse_unservable_model(body.model, body.revision)

        def same(existing: dict) -> dict:
            if any(existing.get(k) != wanted.get(k) for k in qual.request_fields(wanted)) \
                    or qual.request_fields(existing) != qual.request_fields(wanted):
                raise HTTPException(status_code=409,
                                    detail="qualification_exists_with_another_request")
            response.status_code = 200
            return existing

        if existing is not None:
            return same(existing)
        try:
            await qualifications.write(wanted, None)
        except CorpusStoreConflict:
            existing, _ = await qualifications.read(body.qualification_id)
            return same(existing)
        response.status_code = 201
        return wanted

    @router.get("/qualifications/{qualification_id}")
    async def read_qualification(qualification_id: str) -> dict:
        in_scope(qualification_id)
        try:
            record, _ = await qualifications.read(qualification_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="qualification_unknown") from exc
        if record is None:
            raise HTTPException(status_code=404, detail="qualification_unknown")
        return record

    async def order_redone_slots(job, source) -> dict | None:
        """``grading.redone_slots`` from the order control's session store (its last days);
        None when the store cannot be read (the report says nothing rather than zero)."""
        from reliquary.eval.grading import redone_slots

        store = order_sessions
        if store is None:
            from reliquary.infrastructure.sandbox_store import ORDER_SESSION_PREFIX, R2SessionStore

            store = R2SessionStore(prefix=ORDER_SESSION_PREFIX)
        try:
            documents = await store.list_recent(int(clock()))
        except Exception:
            logger.warning("order %s: sandbox sessions unreadable, no redo count", job.job_id, exc_info=True)
            return None
        return redone_slots(documents, job.job_id, lambda index: _problem_id(source, index))

    def _problem_id(source, index: int) -> str:
        from reliquary.eval.prompt_source import load_eval_rows

        return load_eval_rows(source)[index]["problem_id"]

    async def job_grading_source(body: GradeEvaluation):
        """An eval job's passing records as the completions, once it is drained
        and complete (or ``allow_incomplete``); the request must name exactly
        the job's set, problems and samples. The provenance is the job's and its
        qualification's; the request's is only checked against it."""
        from reliquary.eval import qualification as qual
        from reliquary.eval.grading import (
            EpisodeJobRows, JobRows, collect_job_records, exhausted_prompts, job_complete,
        )
        from reliquary.eval.prompt_source import parse_eval_source
        from reliquary.validator.corpus_service import CHAT_TEMPLATE_RENDERERS

        try:
            job, _ = await job_store.read_job(body.job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="job_unknown") from exc
        if job is None:
            raise HTTPException(status_code=404, detail="job_unknown")
        source = parse_eval_source(job.prompt_source)
        samples = job.slots_per_prompt * job.sampling.n
        expected = {"set_ids": [source.set_id], "problems_per_set": {source.set_id: source.count},
                    "samples_per_set": {source.set_id: samples}}
        given = {"set_ids": body.set_ids, "problems_per_set": body.problems_per_set,
                 "samples_per_set": body.samples_per_set}
        if given != expected:
            raise HTTPException(status_code=422, detail=f"job {job.job_id} grades as {expected}")
        declared = await eval_jobs.read(job.job_id)
        # A forced-draw order has no qualification: its record is its own.
        pinned = declared is not None and declared.get("schema") == qual.ORDER_JOB_SCHEMA_V2
        record, _ = (await qualifications.read(declared["qualification_id"])
                     if declared and not pinned else (None, None))
        if declared is None or (record is None and not pinned):
            raise HTTPException(status_code=409, detail="eval_job_record_missing")
        sampling = {"temperature": job.sampling.temperature, "top_p": job.sampling.top_p,
                    "top_k": job.sampling.top_k}
        facts = {"model": job.checkpoint_repo, "revision": job.checkpoint_revision,
                 "model_sha": job.checkpoint_revision, "sampling": sampling,
                 "thinking": CHAT_TEMPLATE_RENDERERS.get(job.renderer_id, False),
                 "max_new_tokens": job.sampling.max_new_tokens}
        claimed = facts if body.provenance is None else body.provenance.model_dump()
        differs = sorted(k for k, v in facts.items()
                         if (claimed[k] if k != "sampling" else
                             {**{"top_p": 1.0, "top_k": 0}, **claimed[k]}) != v)
        if differs:
            raise HTTPException(status_code=422, detail=f"provenance_mismatch: {differs}")
        if not (await job_counts(job.job_id))["final"]:
            raise HTTPException(status_code=409, detail="job_not_drained")
        if job.episode is not None:
            from reliquary.eval.job_grading import JobNotGradable

            # Graded already by the order control's grader, from its signed transcripts: its grade documents.
            try:
                job_rows = await EpisodeJobRows.collect(job, records, allow_incomplete=body.allow_incomplete)
            except JobNotGradable as exc:
                raise HTTPException(status_code=409, detail=f"job_not_graded: {exc}"[:500]) from exc
            collected = {"samples_by_prompt": job_rows.samples_by_prompt}
        else:
            collected = await collect_job_records(job, records, **({"include_generation_status": True} if body.service_contract is not None else {}))
            job_rows = JobRows(job=job, collected=collected)
        from reliquary.validator.corpus_service import rebuild_ledgers

        snapshot, _ = await job_store.read_ledgers(job.job_id)
        slots = rebuild_ledgers(job, snapshot).slots
        complete = job_complete(job, collected, samples, slots)
        exhausted = exhausted_prompts(job, collected, samples, slots)
        if not complete and not body.allow_incomplete:
            raise HTTPException(status_code=409, detail="job_not_complete")
        if pinned:
            from reliquary.corpus.eval_draw import draw_randomness

            # Every sample follows its slot's public draw, judged under the order's thresholds.
            verification = {"scheme": job.sampling.draw, "source": "order", "sampling_verified": True,
                            "judge": job.judge.to_contract(), "randomness": draw_randomness(job),
                            "pin": declared.get("pin"),
                            # An episode's score is its signed sandbox's final reward.
                            **({"grade": "signed-sandbox"} if job.episode is not None else {})}
            for entry in await entries_naming(job.job_id):
                if body.service_contract is not None:
                    facts["generation_contract_sha256"] = entry.profile_sha256
                verification["task_id"] = entry.task_id
        else:
            verification = {"scheme": "toploc-v1", "source": "qualification",
                            "qualification_id": declared["qualification_id"],
                            "sampling_verified": False}
            for entry in await entries_naming(job.job_id):
                if body.service_contract is not None:
                    facts["generation_contract_sha256"] = entry.profile_sha256
                proofs = (entry.contract or {}).get("proofs") or ()
                toploc = [p for p in proofs if p.get("scheme") == "toploc-v1"]
                if toploc:
                    verification["thresholds"] = {k: toploc[0][k] for k in (
                        "exp_mismatch_threshold", "mant_mean_threshold", "mant_median_threshold")}
                    verification["task_id"] = entry.task_id
            result = record.get("result") or {}
            verification["qualification"] = {
                "band": result.get("band"), "clamped": result.get("clamped"),
                "qualifiers": {eid: {"provider_id": m.get("provider_id"), "host": m.get("host"),
                                     "gpu": m.get("gpu")}
                               for eid, m in (result.get("measurements") or {}).items()},
                "status": record.get("status")}
        from reliquary.eval.job_grading import episode_protocol

        redone = await order_redone_slots(job, source) if job.episode is not None else None
        provenance = {**facts, "checkpoint_sha256": job.checkpoint_sha256, "seed": job.seed,
                      # Slots played again after an infra end (no row): count and reason, per problem.
                      **({"redone_slots": redone} if job.episode is not None else {}),
                      # The harness, profile and bounds an agentic score depends on.
                      **({"episode": episode_protocol(job.episode)} if job.episode is not None else {}),
                      "eos_token_id": job.eos_token_id, "verification": verification,
                      "job_complete": complete,
                      # Every attempt used: their missing samples are failures.
                      "prompts_exhausted": len(exhausted),
                      "exhausted_problem_ids": [_problem_id(source, i) for i in exhausted],
                      **({"allow_incomplete": True} if body.allow_incomplete else {})}
        return job_rows, provenance

    @router.post("/evaluations/{eval_id}/sandbox-session")
    async def evaluation_sandbox_session(eval_id: str, body: EvaluationSessionRequest,
                                         response: Response) -> dict:
        from reliquary.eval.prompt_source import eval_job_prefix

        in_scope(eval_id)
        if eval_id != eval_job_prefix(task_prefix) + body.task_id:
            raise HTTPException(status_code=409, detail="evaluation_identity_changed")
        response.headers["Cache-Control"] = "no-store"
        if eval_id != body.eval_id:
            raise HTTPException(status_code=409, detail="evaluation_identity_changed")
        if evaluation_sessions is None:
            raise HTTPException(status_code=503, detail="evaluation_sandboxes_not_configured")
        try:
            result = await asyncio.wait_for(evaluation_sessions.open(body), 60)
        except ValueError as exc:
            logger.warning("evaluation sandbox refused (%s)", type(exc).__name__)
            raise HTTPException(status_code=409, detail="evaluation_contract_refused") from exc
        except Exception as exc:
            logger.warning("evaluation sandbox unavailable (%s)", type(exc).__name__)
            raise HTTPException(status_code=503, detail="evaluation_sandbox_unavailable") from exc
        if "refusal" in result:
            from reliquary.sandbox.routes import REFUSAL_STATUS

            refusal = result["refusal"]
            response.status_code = REFUSAL_STATUS.get(refusal["reason"], 409)
            if response.status_code in (429, 503):
                response.headers["Retry-After"] = str(max(1, math.ceil(refusal["retry_after"] or 5)))
        return result

    @router.post("/evaluations/{eval_id}/combine")
    async def combine_evaluation(eval_id: str, body: CombineEvaluation) -> dict:
        """One report for an evaluation order on several sets: each set's grading, side by side, the macro over
        all of them (``orders.reports``)."""
        from reliquary.corpus.delivery import validated_delivery_id
        from reliquary.orders import reports

        for name in (eval_id, *body.parts):
            in_scope(name)
            try:
                validated_delivery_id(name)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        if eval_id in body.parts:
            raise HTTPException(status_code=422, detail="a combined evaluation is not one of its parts")
        if deliveries is None or not getattr(deliveries, "evaluation_supported", True):
            raise HTTPException(status_code=503, detail="deliveries_not_configured")
        running = gradings.get(eval_id)
        if running is not None:
            raise HTTPException(status_code=409, detail="a grading runs under this id")
        try:
            # One writer per evaluation id: a grading and a combination never share files.
            async with grade_locks.setdefault(eval_id, asyncio.Lock()):
                manifest = await reports.combine(deliveries, eval_id, body.parts, clock=clock)
        except reports.PartNotGraded as exc:
            raise HTTPException(status_code=409, detail=f"part_not_graded: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"state": "done", "eval_id": eval_id, "keys": manifest["keys"], "complete": manifest["complete"]}

    @router.post("/evaluations/{eval_id}/grade")
    async def grade_evaluation(eval_id: str, body: GradeEvaluation, response: Response) -> dict:
        from reliquary.corpus.delivery import validated_delivery_id
        from reliquary.eval import grading
        from reliquary.eval.prompt_source import REFERENCE_INTERACTIONS

        in_scope(eval_id)
        if deliveries is None or not getattr(deliveries, "evaluation_supported", True):
            raise HTTPException(status_code=503, detail="deliveries_not_configured")
        try:
            validated_delivery_id(eval_id)
            keys = (grading.validated_completion_keys(body.completion_keys)
                    if body.source == "uploads" else [])
            from reliquary.protocol.service_contract import ServiceContract
            service_contract = (None if body.service_contract is None else
                                ServiceContract.from_dict(body.service_contract))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if body.source == "job":
            from reliquary.eval.prompt_source import eval_job_prefix

            in_scope(body.job_id)
            if not body.job_id.startswith(eval_job_prefix(task_prefix)):
                raise HTTPException(status_code=422, detail="not_an_eval_job")
        digest = grading.request_digest(body.set_ids, keys, body.problems_per_set,
                                        body.samples_per_set, job_id=body.job_id,
                                        service_contract_sha256=None if service_contract is None else service_contract.sha256)

        def done(manifest: dict) -> dict:
            if manifest.get("request_sha256") != digest:
                raise HTTPException(status_code=409, detail="grade_exists_with_another_request")
            return {"state": "done", "eval_id": eval_id, "keys": manifest["keys"],
                    "rows": manifest["rows"], "complete": manifest["complete"]}

        lock = grade_locks.setdefault(eval_id, asyncio.Lock())
        async with lock:
            running = gradings.get(eval_id)
            if running is not None and running[0] != digest:
                raise HTTPException(status_code=409,
                                    detail="grade_exists_with_another_request")
            if running is not None and running[1].done():
                gradings.pop(eval_id)
                failure = running[1].exception()
                if failure is not None:
                    raise HTTPException(status_code=500, detail=f"grading failed: {failure}")
                return done(running[1].result())
            if running is None:
                stored = await deliveries.get_json(
                    f"{grading.evaluation_prefix(eval_id)}/manifest.json")
                if stored is not None:
                    return done(stored)
                try:
                    # Refusals the caller can act on are answered now, not polled for.
                    sets = await grading.load_sets(body.set_ids, body.problems_per_set,
                                                   body.samples_per_set, subnet=eval_store)
                    # A job on reference sets (SWE, TMax) is graded by its grader: its grades, never re-run here.
                    grading.require_gradable(sets, pregraded=body.source == "job" and any(
                        card.get("interaction") in REFERENCE_INTERACTIONS for card, _ in sets.values()))
                    grading.require_sandboxes(sets, require_sandbox)
                except grading.SetUnknown as exc:
                    raise HTTPException(status_code=404, detail="set_unknown") from exc
                except grading.GradeRequestError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                except grading.SandboxUnavailable as exc:
                    raise HTTPException(status_code=503,
                                        detail="code_sandbox_unavailable") from exc
                provenance = ({} if body.provenance is None
                              else body.provenance.model_dump(exclude_none=True))
                job_rows = None
                if body.source == "job":
                    job_rows, provenance = await job_grading_source(body)
                if service_contract is not None:
                    from reliquary.services.mapping import validate_grading_context
                    try:
                        validate_grading_context(service_contract, sets, provenance)
                    except ValueError as exc:
                        raise HTTPException(status_code=422, detail=str(exc)) from exc
                extra = {} if grade_scorer is None else {"scorer_for": grade_scorer}
                if evaluation_sessions is not None:
                    extra["verify_signed"] = evaluation_sessions.verify_result
                gradings[eval_id] = (digest, asyncio.ensure_future(grading.grade_evaluation(
                    eval_id=eval_id, set_ids=body.set_ids, completion_keys=keys,
                    problems_per_set=body.problems_per_set,
                    samples_per_set=body.samples_per_set, job_rows=job_rows,
                    provenance=provenance, platform=deliveries,
                    subnet=eval_store, open_environment=open_environment,
                    require_sandbox=require_sandbox, work_dir=work_dir, clock=clock,
                    **({"service_contract": service_contract} if service_contract is not None else {}),
                    **extra)))
        response.status_code = 202
        return {"state": "running", "eval_id": eval_id}

    GRADED_FILES = ("report.json", "manifest.json", "graded.parquet", "mapping.jsonl", "mapping-manifest.json")

    @router.get("/evaluations/{eval_id}/files/{name}")
    async def evaluation_file(eval_id: str, name: str) -> Response:
        """A graded evaluation's file, as the grading wrote it to the deliveries bucket."""
        import tempfile
        from pathlib import Path

        from reliquary.corpus.delivery import validated_delivery_id
        from reliquary.eval.grading import evaluation_prefix

        in_scope(eval_id)
        try:
            validated_delivery_id(eval_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if name not in GRADED_FILES:
            raise HTTPException(status_code=422, detail=f"a graded evaluation's files are "
                                                        f"{list(GRADED_FILES)}")
        if deliveries is None or not getattr(deliveries, "evaluation_supported", True):
            raise HTTPException(status_code=503, detail="deliveries_not_configured")
        with tempfile.TemporaryDirectory() as scratch:
            local = Path(scratch) / name
            if not await deliveries.get_file(f"{evaluation_prefix(eval_id)}/{name}", local):
                raise HTTPException(status_code=404, detail="evaluation_file_unknown")
            content = local.read_bytes()
        media = "application/json" if name.endswith(".json") else "application/octet-stream"
        return Response(content=content, media_type=media)

    @asynccontextmanager
    async def lifespan(app):
        from reliquary.admin.cap_sizing import sizing_enabled

        if evaluation_sessions is not None:
            await evaluation_sessions.start()
        resizing = asyncio.create_task(resize_forever()) if sizing_enabled() else None
        try:
            yield
        finally:
            if resizing is not None:
                resizing.cancel()
            if evaluation_sessions is not None:
                await evaluation_sessions.stop()

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.state.exports = exports
    app.state.gradings = gradings
    app.state.task_prefix = task_prefix
    app.state.resize_order_caps = resize_order_caps
    app.add_middleware(BodyLimit, limit=MAX_BODY_BYTES)
    return app


class BodyLimit:
    """Refuse a body over ``limit`` before any of it is read or parsed (pure
    ASGI: FastAPI parses JSON bodies before the signature dependency runs)."""

    def __init__(self, app, limit: int) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            length = headers.get(b"content-length")
            refusal = None
            if length is None and headers.get(b"transfer-encoding"):
                refusal = (411, b'{"detail":"length_required"}')
            elif length is not None and (not length.isdigit() or int(length) > self.limit):
                refusal = (413, b'{"detail":"body_too_large"}')
            if refusal is not None:
                await send({"type": "http.response.start", "status": refusal[0],
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": refusal[1]})
                return
        await self.app(scope, receive, send)


from reliquary.validator.corpus_audit_remote import token_sha256  # noqa: E402


__all__ = [
    "CreateJob",
    "GradeEvaluation",
    "RequestQualification",
    "Provenance",
    "QualifiedModel",
    "THINKING_RENDERERS",
    "cap_limits",
    "create_admin_app",
    "token_sha256",
]
