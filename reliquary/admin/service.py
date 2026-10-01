"""`reliquary admin serve`: the subnet's write authority, for the platform.

Every route needs a fresh HMAC signature (``auth``). The registry is written
under its compare-and-swap with two extra limits re-checked on every retry: the
active caps stay within the one pool, and the corpus caps within
``RELIQUARY_ADMIN_POOL_MAX``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from collections.abc import Callable, Mapping

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from reliquary.admin.auth import (
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    HmacVerifier,
)

logger = logging.getLogger(__name__)

_SUM_TOLERANCE = 1e-9
# No admin request is larger; the bound is checked before the body is read.
MAX_BODY_BYTES = 1024 * 1024
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
    cap: float = Field(ge=0.0, le=1.0)


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


class CreateDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    delivery_id: str | None = None
    apply_filter: bool = True


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
                     prepare=None, work_dir=None) -> FastAPI:
    """The admin app. ``deliveries`` is the platform bucket's sink (None turns
    the export route off); ``records`` the subnet's record store."""
    from reliquary.infrastructure import corpus_executor_store as executors
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.infrastructure import task_registry_store as registry_store
    from reliquary.shared.task_registry import RegistryError

    if not (isinstance(pool_max, (int, float)) and math.isfinite(pool_max)
            and 0.0 <= pool_max <= 1.0):
        raise ValueError(f"the admin pool must be in [0, 1], got {pool_max!r}")
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

    @router.post("/jobs")
    async def create_job(body: CreateJob, response: Response) -> dict:
        from reliquary.corpus.job import parse_job

        spec = catalog.get(body.model)
        if spec is None:
            raise HTTPException(status_code=422, detail=f"model {body.model!r} is not qualified")
        from reliquary.environment.registry import ENVIRONMENT_SPECS

        env = ENVIRONMENT_SPECS.get(body.env)
        if env is None or getattr(env, "interaction_mode", None) != "single_turn":
            raise HTTPException(status_code=422,
                                detail=f"env {body.env!r} is not a single-turn catalog source")
        try:
            manifest, entry = await asyncio.to_thread(
                prepare, job_id=body.job_id, task_id=body.task_id, model=body.model,
                model_revision=spec.revision, model_architecture=spec.architecture,
                checkpoint_sha256=spec.checkpoint_sha256, from_profile=None,
                prompt_encoding=None, prompt_source=body.env, prompt_count=body.prompt_count,
                prompt_start=body.prompt_start, renderer_id=THINKING_RENDERERS[body.thinking],
                eos_token_id=spec.eos_token_id, slots_per_prompt=body.samples_per_prompt,
                max_new_tokens=body.max_new_tokens, cap=body.cap, min_incentive_share=0.0,
                audit_params={},
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
                    body.cap)
        response.status_code = 201
        return {**answer, "created": True, "status": "active", "cap": float(body.cap)}

    async def corpus_entry(task_id: str):
        """The task, refused unless it is a corpus task (never ``default``, never RL)."""
        from reliquary.shared.task_id import DEFAULT_TASK_ID
        from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION

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
                expires_at=body.expires_at, now=clock())
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

    app = FastAPI()
    app.include_router(router)
    app.state.exports = exports
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
    "QualifiedModel",
    "THINKING_RENDERERS",
    "cap_limits",
    "create_admin_app",
    "token_sha256",
]
