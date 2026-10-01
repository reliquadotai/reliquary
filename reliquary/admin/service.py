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
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reliquary.admin.auth import (
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    HmacVerifier,
)

logger = logging.getLogger(__name__)

_SUM_TOLERANCE = 1e-9
# The only task and job ids the platform may create or touch: never an
# operator-declared prod job.
DEFAULT_TASK_PREFIX = "order-"
# No admin request is larger; the bound is checked before the body is read.
MAX_BODY_BYTES = 1024 * 1024
DEFAULT_EVAL_CAP = 0.02
# Renderer of a catalog source's rows: the model's own chat template.
THINKING_RENDERERS = {False: "chat-template-v1", True: "chat-template-thinking-v1"}


class QualifiedModel(BaseModel):
    """A model the platform may order from, as the admin host's catalog pins it."""

    model_config = ConfigDict(extra="forbid")
    revision: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    eos_token_id: int = Field(ge=0)


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
    sampling: "OrderSampling | None" = None


class OrderSampling(BaseModel):
    """The order's sampling: what miners are told and qualification measured."""

    model_config = ConfigDict(extra="forbid")
    temperature: float = Field(gt=0.0, le=4.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)


class RequestQualification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qualification_id: str = Field(min_length=1, max_length=63)
    model: str = Field(min_length=1, max_length=256)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    set_id: str = Field(min_length=1, max_length=128)
    # The order's problem count: the job will be its set's first `problems`
    # prompts; `completions` are decoded over the first of them.
    problems: int = Field(gt=0, le=1_000_000)
    completions: int = Field(default=32, gt=0, le=64)
    sampling: OrderSampling
    max_new_tokens: int = Field(gt=0, le=131072)
    thinking: bool = False


CreateJob.model_rebuild()


class SetCap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cap: float = Field(ge=0.0, le=1.0)


class Retire(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retired_at: int | None = Field(default=None, ge=0)


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
    # "eval" executors serve the eval control only (provider_id and host required).
    scope: Literal["corpus", "eval"] = "corpus"


class CreateDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    delivery_id: str | None = None
    apply_filter: bool = True


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
    provenance: Provenance
    # Job mode: grade a job not every prompt of which holds its samples (the
    # platform, after its deadline); the report says complete=false.
    allow_incomplete: bool = False

    @model_validator(mode="after")
    def _source_fields(self):
        if self.source == "uploads":
            if self.job_id is not None or not self.completion_keys:
                raise ValueError("an uploads grading names completion_keys and no job_id")
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
                     require_sandbox=None, qualifications=None, eval_jobs=None) -> FastAPI:
    """The admin app. ``deliveries`` is the platform bucket's sink (None turns
    the export and grade routes off); ``records`` the subnet's record store;
    ``eval_store`` the subnet bucket holding the eval sets' grading files."""
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
    exports: dict[str, asyncio.Task] = {}
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

    async def draining_limits():
        """The cap guard, counting retired corpus tasks whose job has not drained."""
        from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION
        from reliquary.validator.corpus_job_status import stored_job_counts

        entries, _ = await registry_store.read_registry(strict=False)
        for entry in entries.values():
            if (entry.status == "retired" and entry.mechanism == MECHANISM_CORPUS_GENERATION
                    and entry.job_id and entry.task_id not in drained_tasks):
                if (await stored_job_counts(records, entry.job_id))["drained"]:
                    # Drained is for good: never listed again.
                    drained_tasks.add(entry.task_id)
        return cap_limits(float(pool_max), drained_tasks=set(drained_tasks))

    def idempotent(answer: dict, entry) -> dict:
        return {**answer, "created": False, "status": entry.status,
                "cap": float(entry.params["cap"])}

    async def eval_job_arguments(body: CreateJob) -> dict:
        """What an evaluation job is declared from: its set's first
        prompt_count prompts, the qualification's model and thresholds."""
        from reliquary.eval import qualification as qual
        from reliquary.eval.prompt_source import eval_source_for, register_eval_prompts
        from reliquary.eval.sets import validated_set_id
        from reliquary.eval.storage import subnet_key

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
        if record.get("status") != qual.QUALIFIED:
            raise HTTPException(status_code=409, detail=f"model_not_qualified: {record.get('status')}")
        if record["model"] != body.model:
            raise HTTPException(status_code=409, detail="qualification_is_for_another_model")
        result = record["result"]
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
                  "sampling": body.sampling.model_dump(), "max_new_tokens": body.max_new_tokens,
                  "thinking": body.thinking}
        differs = sorted(k for k, v in wanted.items() if record.get(k) != v)
        if differs:
            raise HTTPException(status_code=409,
                                detail=f"qualification_conditions_differ: {differs}")
        source = eval_source_for(set_id, prompts_body, body.prompt_count)
        register_eval_prompts(source, prompts_body)
        seed = body.seed if body.seed is not None else int(
            hashlib.sha256(body.job_id.encode()).hexdigest()[:15], 16)
        try:
            await eval_jobs.create({
                "schema": qual.EVAL_JOB_SCHEMA, "job_id": body.job_id,
                "qualification_id": body.qualification_id, "set_id": set_id,
                "problems": body.prompt_count, "samples": body.samples_per_prompt,
                "sampling": body.sampling.model_dump(), "max_new_tokens": body.max_new_tokens,
                "thinking": body.thinking})
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return dict(
            model_revision=record["revision"], model_architecture=result["architecture"],
            checkpoint_sha256=result["checkpoint_sha256"], eos_token_id=int(result["eos_token_id"]),
            prompt_source=source.name, contract_environment=card["source"], seed=seed,
            toploc_thresholds=result["thresholds"], audit_params={"audit_q": 1.0},
            temperature=body.sampling.temperature, top_p=body.sampling.top_p,
            top_k=body.sampling.top_k,
        )

    @router.post("/jobs")
    async def create_job(body: CreateJob, response: Response) -> dict:
        from reliquary.corpus.job import parse_job
        from reliquary.eval.prompt_source import eval_job_prefix

        eval_prefix = eval_job_prefix(task_prefix)
        if not (body.job_id.startswith(task_prefix)
                and (body.task_id or body.job_id).startswith(task_prefix)):
            raise HTTPException(status_code=422, detail="task_id_outside_admin_scope")
        evaluation = body.eval_set_id is not None
        # The eval control serves exactly the order-eval- ids (and routing
        # sends it exactly those): one is an eval job if and only if it says so.
        if evaluation != body.job_id.startswith(eval_prefix) or (
                evaluation and not (body.task_id or body.job_id).startswith(eval_prefix)):
            raise HTTPException(status_code=422, detail=(
                f"an eval job's ids start with {eval_prefix!r}, and only an eval job's"))
        if not evaluation and any(v is not None for v in (body.qualification_id, body.seed,
                                                           body.audit_q)):
            raise HTTPException(status_code=422, detail="eval fields on a non-eval job")
        cap = body.cap
        if evaluation:
            arguments = await eval_job_arguments(body)
            cap = DEFAULT_EVAL_CAP if cap is None else cap
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
        try:
            manifest, entry = await asyncio.to_thread(
                prepare, job_id=body.job_id, task_id=body.task_id, model=body.model,
                from_profile=None, prompt_encoding=None, prompt_count=body.prompt_count,
                prompt_start=body.prompt_start, renderer_id=THINKING_RENDERERS[body.thinking],
                slots_per_prompt=body.samples_per_prompt,
                max_new_tokens=body.max_new_tokens, cap=cap, min_incentive_share=0.0,
                **arguments,
            )
        except (RegistryError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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
                response.status_code = 200
                return idempotent(answer, other)
        if named:
            raise HTTPException(status_code=409, detail=(
                f"job {body.job_id!r} is already declared by task {named[0].task_id!r}"))
        try:
            await registry_store.create_task(entry, guard=await draining_limits())
        except (RegistryError, registry_store.RegistryConflict) as exc:
            # Lost to an identical concurrent call: its task is this answer.
            for other in await entries_naming(body.job_id):
                if other.task_id == entry.task_id:
                    response.status_code = 200
                    return idempotent(answer, other)
            status = 409 if isinstance(exc, RegistryError) else 503
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        logger.info("admin: declared job %s as task %s, cap %s", body.job_id, entry.task_id,
                    cap)
        response.status_code = 201
        return {**answer, "created": True, "status": "active", "cap": float(cap)}

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
            await registry_store.set_task_cap(task_id, body.cap, guard=await draining_limits())
        except RegistryError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except registry_store.RegistryConflict as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"task_id": task_id, "cap": body.cap}

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
        from reliquary.validator.corpus_job_status import stored_job_counts

        in_scope(job_id)
        try:
            job, _ = await job_store.read_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="job_unknown") from exc
        if job is None:
            raise HTTPException(status_code=404, detail="job_unknown")
        counts = await stored_job_counts(records, job_id)
        tasks = [{"task_id": e.task_id, "status": e.status, "cap": float(e.params["cap"])}
                 for e in await entries_naming(job_id)]
        return {"job_id": job_id, **counts, "manifest": job.to_contract(), "tasks": tasks}

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

    @router.post("/jobs/{job_id}/deliveries")
    async def create_delivery(job_id: str, body: CreateDelivery, response: Response) -> dict:
        from reliquary.corpus.delivery import export_delivery, validated_delivery_id
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
        running = exports.get(delivery_id)
        if running is not None and running.done():
            exports.pop(delivery_id)
            if running.exception() is not None:
                raise HTTPException(status_code=500,
                                    detail=f"delivery failed: {running.exception()}")
            manifest = running.result()
            return {"state": "done", "delivery_id": delivery_id, "keys": manifest["keys"],
                    "rows": manifest["rows"]}
        if running is None:
            stored = await deliveries.get_json(f"deliveries/{delivery_id}/manifest.json")
            if stored is not None:
                return {"state": "done", "delivery_id": delivery_id, "keys": stored["keys"],
                        "rows": stored["rows"]}
            from reliquary.validator.corpus_job_status import stored_job_counts

            # A delivery is final once written: never from a job still moving.
            if not (await stored_job_counts(records, job_id))["drained"]:
                raise HTTPException(status_code=409, detail="job_not_drained")
            grade, note = None, None
            if body.apply_filter and job.filter is not None:
                try:
                    grade = await asyncio.to_thread(job_grader, job)
                except ValueError as exc:
                    note = str(exc)
            # Long: run beside the request; the caller polls with the same id.
            exports[delivery_id] = asyncio.ensure_future(export_delivery(
                job=job, records=records, sink=deliveries, delivery_id=delivery_id,
                grade=grade, filter_note=note, work_dir=work_dir, clock=clock))
        response.status_code = 202
        return {"state": "running", "delivery_id": delivery_id}

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
            validated_set_id(body.set_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if await eval_store.get_bytes(subnet_key(body.set_id, "prompts.jsonl")) is None:
            raise HTTPException(status_code=404, detail="set_unknown")
        try:
            wanted = qual.new_request(
                qualification_id=body.qualification_id, model=body.model,
                revision=body.revision, set_id=body.set_id, problems=body.problems,
                completions=body.completions, sampling=body.sampling.model_dump(),
                max_new_tokens=body.max_new_tokens, thinking=body.thinking, clock=clock)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        def same(existing: dict) -> dict:
            if any(existing.get(k) != wanted[k] for k in qual.REQUEST_FIELDS):
                raise HTTPException(status_code=409,
                                    detail="qualification_exists_with_another_request")
            response.status_code = 200
            return existing

        existing, _ = await qualifications.read(body.qualification_id)
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
            JobRows, collect_job_records, exhausted_prompts, job_complete,
        )
        from reliquary.eval.prompt_source import parse_eval_source
        from reliquary.validator.corpus_job_status import stored_job_counts
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
        record, _ = (await qualifications.read(declared["qualification_id"])
                     if declared else (None, None))
        if declared is None or record is None:
            raise HTTPException(status_code=409, detail="eval_job_record_missing")
        sampling = {"temperature": job.sampling.temperature, "top_p": job.sampling.top_p,
                    "top_k": job.sampling.top_k}
        facts = {"model": job.checkpoint_repo, "revision": job.checkpoint_revision,
                 "model_sha": job.checkpoint_revision, "sampling": sampling,
                 "thinking": CHAT_TEMPLATE_RENDERERS.get(job.renderer_id, False),
                 "max_new_tokens": job.sampling.max_new_tokens}
        claimed = body.provenance.model_dump()
        differs = sorted(k for k, v in facts.items()
                         if (claimed[k] if k != "sampling" else
                             {**{"top_p": 1.0, "top_k": 0}, **claimed[k]}) != v)
        if differs:
            raise HTTPException(status_code=422, detail=f"provenance_mismatch: {differs}")
        if not (await stored_job_counts(records, job.job_id))["drained"]:
            raise HTTPException(status_code=409, detail="job_not_drained")
        collected = await collect_job_records(job, records)
        from reliquary.validator.corpus_service import rebuild_ledgers

        snapshot, _ = await job_store.read_ledgers(job.job_id)
        slots = rebuild_ledgers(job, snapshot).slots
        complete = job_complete(job, collected, samples, slots)
        exhausted = exhausted_prompts(job, collected, samples, slots)
        if not complete and not body.allow_incomplete:
            raise HTTPException(status_code=409, detail="job_not_complete")
        verification = {"scheme": "toploc-v1", "source": "qualification",
                        "qualification_id": declared["qualification_id"],
                        "sampling_verified": False}
        for entry in await entries_naming(job.job_id):
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
        provenance = {**facts, "checkpoint_sha256": job.checkpoint_sha256, "seed": job.seed,
                      "eos_token_id": job.eos_token_id, "verification": verification,
                      "job_complete": complete,
                      # Every attempt used: their missing samples are failures.
                      "prompts_exhausted": len(exhausted),
                      "exhausted_problem_ids": [_problem_id(source, i) for i in exhausted],
                      **({"allow_incomplete": True} if body.allow_incomplete else {})}
        return JobRows(job=job, collected=collected), provenance

    @router.post("/evaluations/{eval_id}/grade")
    async def grade_evaluation(eval_id: str, body: GradeEvaluation, response: Response) -> dict:
        from reliquary.corpus.delivery import validated_delivery_id
        from reliquary.eval import grading

        in_scope(eval_id)
        if deliveries is None:
            raise HTTPException(status_code=503, detail="deliveries_not_configured")
        try:
            validated_delivery_id(eval_id)
            keys = (grading.validated_completion_keys(body.completion_keys)
                    if body.source == "uploads" else [])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if body.source == "job":
            from reliquary.eval.prompt_source import eval_job_prefix

            in_scope(body.job_id)
            if not body.job_id.startswith(eval_job_prefix(task_prefix)):
                raise HTTPException(status_code=422, detail="not_an_eval_job")
        digest = grading.request_digest(body.set_ids, keys, body.problems_per_set,
                                        body.samples_per_set, job_id=body.job_id)

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
                    grading.require_sandboxes(sets, require_sandbox)
                except grading.SetUnknown as exc:
                    raise HTTPException(status_code=404, detail="set_unknown") from exc
                except grading.GradeRequestError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                except grading.SandboxUnavailable as exc:
                    raise HTTPException(status_code=503,
                                        detail="code_sandbox_unavailable") from exc
                provenance = body.provenance.model_dump(exclude_none=True)
                job_rows = None
                if body.source == "job":
                    job_rows, provenance = await job_grading_source(body)
                extra = {} if grade_scorer is None else {"scorer_for": grade_scorer}
                gradings[eval_id] = (digest, asyncio.ensure_future(grading.grade_evaluation(
                    eval_id=eval_id, set_ids=body.set_ids, completion_keys=keys,
                    problems_per_set=body.problems_per_set,
                    samples_per_set=body.samples_per_set, job_rows=job_rows,
                    provenance=provenance, platform=deliveries,
                    subnet=eval_store, open_environment=open_environment,
                    require_sandbox=require_sandbox, work_dir=work_dir, clock=clock,
                    **extra)))
        response.status_code = 202
        return {"state": "running", "eval_id": eval_id}

    app = FastAPI()
    app.include_router(router)
    app.state.exports = exports
    app.state.gradings = gradings
    app.state.task_prefix = task_prefix
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
