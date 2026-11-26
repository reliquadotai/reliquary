"""The admin routes for customer envs (the platform's side is docs/design/2026-11-26-customer-environments-design.md).

The admin host indexes versions and orchestrates qualification; it never unpacks an archive (the builder
host does) and never runs a customer's code (the reward host does).
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from reliquary.customer_env import ACCOUNT_RE, NAME_RE, env_id, parse_env_id
from reliquary.customer_env.registry import Registry, resolve

logger = logging.getLogger(__name__)
MAX_IN_FLIGHT_PER_ACCOUNT = 3


class SubmitVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=63)
    object_key: str = Field(min_length=1, max_length=512, pattern=r"^customer-envs/[A-Za-z0-9._/-]+$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OrderSets(BaseModel):
    model_config = ConfigDict(extra="forbid")
    env_id: str = Field(min_length=1, max_length=300)
    account: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=256)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    purpose: Literal["batch", "eval"] = "eval"
    sampling: dict[str, Any] = Field(default_factory=dict)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, gt=0)
    max_prompt_tokens: int | None = Field(default=None, gt=0, le=131072)


class _SyncStore:
    """An async bucket store for the qualifier's worker thread."""

    def __init__(self, store, loop) -> None:
        self._store, self._loop = store, loop

    def create_bytes(self, key: str, body: bytes):
        create = getattr(self._store, "create_bytes", None) or self._store.put_bytes
        return asyncio.run_coroutine_threadsafe(create(key, body), self._loop).result(300)


@dataclass
class CustomerEnvs:
    """``make_qualifier(read_object, store) -> Qualifier`` for one run; ``registry`` the index."""

    registry: Registry
    make_qualifier: Any


def from_environment() -> CustomerEnvs | None:
    """``RELIQUARY_CUSTOMER_ENV_DB``, ``RELIQUARY_BUILDER_URL`` and ``RELIQUARY_REWARD_HOST_URL`` (with the
    client certificate of ``scoring.from_environment``), or None: the routes then answer 503."""
    db, builder_url = os.environ.get("RELIQUARY_CUSTOMER_ENV_DB"), os.environ.get("RELIQUARY_BUILDER_URL")
    reward_url = os.environ.get("RELIQUARY_REWARD_HOST_URL")
    if not (db and builder_url and reward_url):
        return None
    from reliquary.customer_env.host import RemoteBuilder, RemoteRewardClient
    from reliquary.customer_env.qualify import Qualifier

    cert = (os.environ.get("RELIQUARY_REWARD_HOST_CLIENT_CERT"), os.environ.get("RELIQUARY_REWARD_HOST_CLIENT_KEY"))
    tls = {"cert": cert if all(cert) else None, "ca": os.environ.get("RELIQUARY_REWARD_HOST_CA")}
    registry = Registry(db)
    builder = RemoteBuilder(builder_url, cert=tls["cert"], verify=tls["ca"] or True)

    def make_qualifier(read_object, store):
        return Qualifier(registry, read_object=read_object, builder=builder, store=store,
                         open_worker=lambda image: RemoteRewardClient(reward_url, image, fresh=True, **tls))

    return CustomerEnvs(registry, make_qualifier)


def order_resolver(envs: CustomerEnvs | None):
    """``resolve(env, account)`` for a training order (ValueError when it may not train on it), or None."""
    if envs is None:
        return None

    def resolve_for_rl(env: str, account: str):
        return resolve(envs.registry, env, account=account, product="rl")

    return resolve_for_rl


def add_routes(router: APIRouter, envs: CustomerEnvs | None, *, deliveries, eval_store,
               load_tokenizer=None) -> None:
    running: dict[str, asyncio.Task] = {}

    def need() -> CustomerEnvs:
        if envs is None:
            raise HTTPException(503, "customer_envs_unavailable")
        if deliveries is None:
            raise HTTPException(503, "platform_bucket_unavailable")
        return envs

    def start(env: str) -> None:
        if env in running and not running[env].done():
            return
        loop = asyncio.get_running_loop()

        def read_object(key: str) -> bytes:
            body = asyncio.run_coroutine_threadsafe(deliveries.get_bytes(key), loop).result(300)
            if body is None:
                raise FileNotFoundError(key)
            return body

        qualifier = envs.make_qualifier(read_object, _SyncStore(eval_store, loop))
        running[env] = asyncio.create_task(asyncio.to_thread(qualifier.run, env))

    @router.post("/customer-envs")
    async def submit(body: SubmitVersion) -> dict:
        envs_ = need()
        if not ACCOUNT_RE.match(body.account) or not NAME_RE.match(body.name):
            raise HTTPException(422, "account or name is not of the allowed form")
        if ".." in body.object_key:
            raise HTTPException(422, "object_key must not climb out of customer-envs/")
        version = env_id(body.account, body.name, body.sha256)
        existing = envs_.registry.get(str(version))
        if existing is None and envs_.registry.in_flight(body.account) >= MAX_IN_FLIGHT_PER_ACCOUNT:
            raise HTTPException(409, "too_many_qualifications")
        record, created = envs_.registry.submit(version, body.object_key)
        if created:
            start(record.env_id)
        return {"env_id": record.env_id, "status": record.status}

    @router.get("/customer-envs/accounts/{account}")
    async def for_account(account: str) -> dict:
        envs_ = need()
        if not ACCOUNT_RE.match(account):
            raise HTTPException(422, "bad account")
        return {"versions": [r.public() for r in envs_.registry.for_account(account)]}

    def _record(raw: str):
        envs_ = need()
        try:
            env = str(parse_env_id(unquote(raw)))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        record = envs_.registry.get(env)
        if record is None:
            raise HTTPException(404, "customer_env_unknown")
        return envs_, record

    @router.get("/customer-envs/{raw:path}")
    async def one(raw: str) -> dict:
        return _record(raw)[1].public()

    @router.delete("/customer-envs/{raw:path}")
    async def disable(raw: str) -> dict:
        envs_, record = _record(raw)
        if record.status in ("queued", "building", "qualifying"):
            raise HTTPException(409, "customer_env_qualifying")
        return envs_.registry.update(record.env_id, status="disabled").public()

    @router.post("/customer-envs/order-sets")
    async def order_sets(body: OrderSets) -> dict:
        from reliquary.customer_env import orders
        from reliquary.orders import uploads

        envs_ = need()
        try:
            record = resolve(envs_.registry, body.env_id, account=body.account, product=body.purpose)
            rows = orders.select(await orders.read_rows(eval_store, record.env_id, record.prompt_set["sha256"]),
                                 body.start, body.end)
        except ValueError as exc:
            raise HTTPException(422, str(exc)[:500]) from None
        file = orders.batch_file(rows, body.sampling, purpose=body.purpose)
        if load_tokenizer is None:
            from reliquary.validator.eval_control import load_cpu_tokenizer as tokenizer_loader
        else:
            tokenizer_loader = load_tokenizer

        async def read_object(_key):
            return file

        try:
            return await uploads.ingest(
                object_key=f"customer-envs/{record.account}/{record.name}/{record.sha256}/order",
                sha256=None, model=body.model, revision=body.revision, purpose=body.purpose,
                max_prompt_tokens=body.max_prompt_tokens, grader=record.env_id if body.purpose == "eval" else None,
                read_object=read_object, load_tokenizer=tokenizer_loader, eval_store=eval_store,
                errors_sink=_SubnetErrors(eval_store))
        except uploads.UploadRefused as exc:
            raise HTTPException(422, str(exc)[:500]) from None


class _SubnetErrors:
    """An order set's error lines (prompts over the model's limit) beside its sets in the subnet bucket: the
    platform bucket's sink takes only a batch file's errors."""

    def __init__(self, store) -> None:
        self._store = store

    async def put_bytes(self, key: str, body: bytes) -> None:
        create = getattr(self._store, "create_bytes", None) or self._store.put_bytes
        await create(f"reliquary/customer-envs/order-errors/{key}", body)


__all__ = ["CustomerEnvs", "MAX_IN_FLIGHT_PER_ACCOUNT", "OrderSets", "SubmitVersion", "add_routes",
           "from_environment", "order_resolver"]
