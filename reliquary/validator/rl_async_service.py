"""The async RL validator service (rl-async-design.md §3; implementation map §9), behind one switch:
``RELIQUARY_RL_MODE=async`` (default ``window``: the window service runs, unchanged).

One process on the controller, like the corpus validator: no window, batcher or proof plane.

* **Components** (``build_rl_async``), all in one SQLite file under the state dir (``rl-async.sqlite3``): the
  ``LeaseBook`` with its ``QuotaController`` and ``LeaseCollector``, ``RlSubmissions`` (grading, audits, events)
  with its ``TrainingBuffer``, the ``AsyncPriceController`` and the ``RlPayLedger``; plus the ``RlCheckpointBoard``
  over the trainer's index in the ``reliquary-public`` bucket.
* **Routes** (``mount_rl_async``, through ``ValidatorServer.mount_rl_lease_router``): ``/rl/lease``, ``/rl/quota``,
  ``/rl/lease/mine``, ``/rl/lease/commitments``, ``/rl/prompts/state``, ``/rl/submit``, ``/rl/pay`` (replaces the
  window board's), ``/rl/checkpoint`` and ``/rl/observations`` (the scan log's signed head + presigned URLs).
* **Trainer link** (``/rl/trainer/*``, bearer ``RELIQUARY_RL_TRAINER_TOKEN``) on its own listener,
  ``RELIQUARY_RL_TRAINER_BIND`` (``host:port``, default ``127.0.0.1:8889``), never on the public port. The link is
  plain HTTP and its token can mint pay: a non-loopback bind is refused unless
  ``RELIQUARY_RL_TRAINER_INSECURE_BIND=1`` says the address is on a private network (WireGuard, a private VLAN);
  otherwise reach it over an SSH tunnel: the detached async trainer (``reliquary train-async``, its own GPU box
  as today) takes groups from the buffer, reports them, adopts the checkpoints it publishes and reports its speed.
* **Episodes** (``rl_async_episodes``): the task's ``params.rl_episode_envs`` (the order's episode block and
  catalogue, per env) make those envs signed-episode envs: the sandbox side (``wire_episode_sandbox``: token key,
  machine directory and fleet, session issuer over ``RlLeaseEngagements``, ``/rl/sandbox/sessions`` routes) and one
  ``LeaseEpisodeChecker`` per env in the grader; ``GET /rl/episodes`` serves the blocks to miners; an episode env's
  lease lifetime includes the session's wall budget; ``/rl/submit`` takes gzipped bodies up to 64 MiB inflated
  for episode groups (8 MiB for single-turn ones).
* **Discovery:** with ``--external-ip`` the validator serves its axon at start, as the window validator does, so
  ``mine-async`` finds it on the metagraph (``--validator-url`` stays an override).
* **Loops** (``rl_async_loops``): the scan log's publication (``rl_observations``, a segment per 60 s into
  ``rl/<run>/observations/``); the lease expiry sweep; the out-of-zone audits (``run_rl_audits``) against the
  verify model held at the latest checkpoint (``audit_source``: the remote proof worker's when
  ``RELIQUARY_PROOF_EXECUTOR_MODE=remote``, ``remote_rl_audit``; else a local ``RELIQUARY_RL_AUDIT_DEVICE``; without
  either every audit concludes inconclusive: unpaid, unsanctioned), with ``RlSubmissions.run_audits``' retry cap;
  the ledger (ingest, settle, publish the period archives through ``R2PeriodArchives``, as the corpus periods); the price tick (``rl_price``:
  each env's queued tokens against its target share of the trainer's speed, which the trainer reports per step).
* **Settlement:** the task must be declared with ``params.settlement = "period-ema-v1"`` (``reliquary tasks create
  --rl-async``): the unchanged weight setter pays its period archives. Per-env target shares are the entry's
  ``env_split``; the async parameters are the ``rl_*`` params below.
* **Signing:** the checkpoint index and the scan log's head and pages are signed by the remote signer's narrow
  operations (``/v1/rl/index/sign``, ``/v1/rl/observations/sign``) when ``RELIQUARY_SIGNER_URL`` is set (signer-01
  needs ``RELIQUARY_SIGNER_RL_RUN`` = the run), else by the local wallet.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hmac
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

RL_MODE_ENV = "RELIQUARY_RL_MODE"
WINDOW, ASYNC = "window", "async"
TRAINER_TOKEN_ENV = "RELIQUARY_RL_TRAINER_TOKEN"
TRAINER_BIND_ENV = "RELIQUARY_RL_TRAINER_BIND"
DEFAULT_TRAINER_BIND = "127.0.0.1:8889"
TRAINER_INSECURE_BIND_ENV = "RELIQUARY_RL_TRAINER_INSECURE_BIND"   # "1": a non-loopback bind on a private network
AUDIT_DEVICE_ENV = "RELIQUARY_RL_AUDIT_DEVICE"
RUN_ENV = "RELIQUARY_RL_RUN"
DB_NAME = "rl-async.sqlite3"

# Registry params of an async task (``rl_async_params``), beside the price fields every entry carries.
PLANNED_TOKENS_PARAM = "rl_planned_tokens_per_period"   # matched tokens the trainer consumes per 72-min period
LEASE_SPEED_PARAM = "rl_lease_reference_tokens_per_s"   # a slow reference card's decode speed, per sequence
LEASE_MARGIN_PARAM = "rl_lease_expiry_margin"
PRICE_TICK_PARAM = "rl_price_tick_s"
PENDING_PARAM = "rl_max_pending_groups"         # global cap: buffered groups (queued or taken)
DEFAULT_MAX_PENDING_GROUPS = 192      # placeholder: 3 batches of 64. queue_sim 10-17: trainer wait 7 % at 128, 2 % at 160, 1 % at 192, <1 % at 256; queue wait p50 0.6 / 0.9 / 1.3 / 2.2 steps
DEFAULT_LEASE_TOKENS_PER_S = 20.0      # placeholder: the design's "3090 at max length", measured at qualification
DEFAULT_LEASE_MARGIN = 2.0

EXPIRE_EVERY_S = 5.0
AUDIT_EVERY_S = 2.0
AUDIT_LIMIT = 8
LEDGER_EVERY_S = 30.0
PRICE_EVERY_S = 30.0
CHECKPOINT_FOLLOW_EVERY_S = 10.0


def rl_mode(env: Mapping[str, str] = os.environ) -> str:
    value = str(env.get(RL_MODE_ENV, WINDOW)).strip().lower() or WINDOW
    if value not in (WINDOW, ASYNC):
        raise ValueError(f"{RL_MODE_ENV} must be {WINDOW!r} or {ASYNC!r}, got {value!r}")
    return value


def rl_async_params(*, planned_tokens_per_period: float, lease_tokens_per_s: float = DEFAULT_LEASE_TOKENS_PER_S,
                    lease_margin: float = DEFAULT_LEASE_MARGIN, tick_s: float | None = None,
                    max_pending_groups: int | None = None, episode_envs: Mapping[str, Any] | None = None,
                    lease_expiry_s: Mapping[str, float] | None = None) -> dict:
    """The params an async task adds to its registry entry (pay by period: ``settlement = period-ema-v1``).
    ``episode_envs``: the signed-episode envs' blocks (``rl_async_episodes``: episode, catalogue?, version?);
    ``lease_expiry_s``: per-env lease lifetimes overriding the computed ones."""
    from reliquary.shared.task_registry import SETTLEMENT_PERIOD_EMA
    from reliquary.validator.rl_async_episodes import (
        EPISODE_ENVS_PARAM, LEASE_EXPIRY_PARAM, episode_blocks, expiry_overrides,
    )

    if not (planned_tokens_per_period > 0 and lease_tokens_per_s > 0 and lease_margin >= 1):
        raise ValueError("planned tokens and lease speed must be positive, the margin at least 1")
    params: dict[str, Any] = {"settlement": SETTLEMENT_PERIOD_EMA,
                              PLANNED_TOKENS_PARAM: float(planned_tokens_per_period),
                              LEASE_SPEED_PARAM: float(lease_tokens_per_s), LEASE_MARGIN_PARAM: float(lease_margin)}
    if tick_s is not None:
        params[PRICE_TICK_PARAM] = float(tick_s)
    if max_pending_groups is not None:
        if max_pending_groups < 1:
            raise ValueError("max_pending_groups must be positive")
        params[PENDING_PARAM] = int(max_pending_groups)
    if episode_envs:
        params[EPISODE_ENVS_PARAM] = episode_blocks({EPISODE_ENVS_PARAM: episode_envs})
    if lease_expiry_s:
        params[LEASE_EXPIRY_PARAM] = expiry_overrides({LEASE_EXPIRY_PARAM: lease_expiry_s})
    return params


@dataclass(frozen=True)
class RlAsyncSettings:
    task_id: str
    cap: float
    run: str                                    # the checkpoint index's run (``rl/<run>/index.json``)
    targets: Mapping[str, float]                # env -> target share of the trainer's speed (priced, not enforced)
    planned_tokens_per_period: float
    lease_expiry_s: Mapping[str, float]
    price: Any = None                           # AsyncPriceParams (None: the defaults)
    max_pending_groups: int = DEFAULT_MAX_PENDING_GROUPS
    episodes: Mapping[str, Any] = field(default_factory=dict)   # env -> its validated episode block (§4.10)

    def episode_policies(self) -> dict:
        from reliquary.validator.rl_async_episodes import episode_policies

        return episode_policies(self.episodes)


def rl_async_settings(entry, cap: float, *, run: str | None = None,
                      max_tokens: Callable[[str], int] | None = None) -> RlAsyncSettings:
    """An async task's settings from its registry entry. ValueError when the entry was not declared for it."""
    from reliquary.validator import corpus_periods as cp
    from reliquary.validator.rl_price import AsyncPriceParams
    from reliquary.validator.rl_async_episodes import (
        episode_blocks, episode_lease_expiry_s, episode_policies, expiry_overrides,
    )
    from reliquary.validator.rl_leases import lease_expiry_s

    params = dict(entry.params)
    if not cp.is_period_task(entry):
        raise ValueError(f"task {entry.task_id!r} is not settled by period: declare it with "
                         "`reliquary tasks create --rl-async` (params.settlement = period-ema-v1)")
    if PLANNED_TOKENS_PARAM not in params:
        raise ValueError(f"task {entry.task_id!r} declares no {PLANNED_TOKENS_PARAM}")
    if not entry.env_split:
        raise ValueError(f"task {entry.task_id!r} declares no env_split: the async trainer's target mix")
    if max_tokens is None:
        from reliquary.constants import max_new_tokens_for_environment as max_tokens
    targets = {str(env): float(share) for env, share in entry.env_split.items() if float(share) > 0}
    speed = float(params.get(LEASE_SPEED_PARAM, DEFAULT_LEASE_TOKENS_PER_S))
    margin = float(params.get(LEASE_MARGIN_PARAM, DEFAULT_LEASE_MARGIN))
    blocks = episode_blocks(params, targets)
    policies = episode_policies(blocks)
    expiry = {env: (episode_lease_expiry_s(policies[env], reference_tokens_per_s=speed, margin=margin)
                    if env in policies else
                    lease_expiry_s(max_tokens=int(max_tokens(env)), reference_tokens_per_s=speed, margin=margin))
              for env in targets}
    overrides = expiry_overrides(params)
    unknown = sorted(set(overrides) - set(targets))
    if unknown:
        raise ValueError(f"lease expiry overrides for envs outside the env_split: {unknown}")
    expiry.update(overrides)
    price = AsyncPriceParams(**{key: cast(params[name]) for key, name, cast in (
        ("tick_s", PRICE_TICK_PARAM, float),) if name in params})
    return RlAsyncSettings(task_id=str(entry.task_id), cap=float(cap),
                           run=run or os.environ.get(RUN_ENV) or str(entry.task_id), targets=targets,
                           planned_tokens_per_period=float(params[PLANNED_TOKENS_PARAM]), lease_expiry_s=expiry,
                           price=price, episodes=blocks,
                           max_pending_groups=int(params.get(PENDING_PARAM, DEFAULT_MAX_PENDING_GROUPS)))


@dataclass
class RlAsync:
    settings: RlAsyncSettings
    book: Any
    collector: Any
    submissions: Any
    buffer: Any
    ledger: Any
    prices: Any
    board: Any = None
    observations: Any = None                    # rl_observations.ObservationBoard (None: /rl/observations 503s)
    sandbox: Any = None                         # rl_async_episodes.RlLeaseSandbox (episode envs only)
    gpu_registry: Any = None                    # optional NVIDIA hardware admission, not execution attestation

    def close(self) -> None:
        if self.gpu_registry is not None:
            self.gpu_registry.close()
        self.book.close()


def build_rl_async(settings: RlAsyncSettings, *, path: str | os.PathLike[str], grader, board=None,
                   prompt_range: Mapping[str, tuple[int, int]] | None = None, submit_config=None,
                   collector_interval_s: float | None = None, clock: Callable[[], float] = time.time) -> RlAsync:
    """Every async component on one SQLite file at ``path`` (restart-safe: open leases, buffer, ledger, prices)."""
    from reliquary.validator.rl_lease_quota import QuotaController
    from reliquary.validator.rl_leases import LeaseBook, LeaseBookConfig, LeaseCollector
    from reliquary.validator.rl_ledger import LedgerConfig, RlPayLedger
    from reliquary.validator.rl_price import AsyncPriceController, AsyncPriceParams
    from reliquary.validator.rl_submit import RlSubmissions

    book = LeaseBook(path, LeaseBookConfig(expiry_s=dict(settings.lease_expiry_s),
                                           prompt_range=None if prompt_range is None else dict(prompt_range),
                                           max_pending_groups=settings.max_pending_groups),
                     quota=QuotaController(), clock=clock)
    try:
        collector = (LeaseCollector(book) if collector_interval_s is None
                     else LeaseCollector(book, interval_s=collector_interval_s))
        submissions = RlSubmissions(book, grader, config=submit_config, clock=clock)
        prices = AsyncPriceController(book, dict(settings.targets), settings.cap, settings.planned_tokens_per_period,
                                      params=settings.price or AsyncPriceParams(), clock=clock)
        ledger = RlPayLedger(book, submissions, prices, LedgerConfig(task_id=settings.task_id, cap=settings.cap),
                             clock=clock)
    except BaseException:
        book.close()
        raise
    return RlAsync(settings, book, collector, submissions, submissions.buffer, ledger, prices, board)


# --- the trainer link -----------------------------------------------------------------------------------------
def _pack(group) -> str:
    return base64.b64encode(group.pack()).decode("ascii")


def build_rl_trainer_router(rl: RlAsync, *, token: str, prefix: str = "/rl/trainer") -> APIRouter:
    """The detached async trainer's view of the buffer, the submissions' report, the lease book's checkpoints and
    the prices (``trainer/rl_link.TrainerLink``). Every route needs ``Authorization: Bearer <token>``."""
    if len(token) < 16:
        raise ValueError(f"{TRAINER_TOKEN_ENV} must be at least 16 characters")
    expected = f"Bearer {token}".encode()
    router = APIRouter()

    def refused(status: int, reason: str, detail: str = "") -> JSONResponse:
        return JSONResponse(status_code=status, content={"reason": reason, "detail": detail},
                            headers={"Cache-Control": "no-store"})

    def authorized(request: Request) -> bool:
        return hmac.compare_digest(request.headers.get("authorization", "").encode(), expected)

    def route(method: str, path: str, op: Callable[[dict], Any]) -> None:
        async def handler(request: Request) -> JSONResponse:
            if not authorized(request):
                return refused(401, "unauthorized")
            try:
                body = await request.json() if method == "POST" else dict(request.query_params)
                if not isinstance(body, dict):
                    raise ValueError("the body is not an object")
                result = await asyncio.to_thread(op, body)
            except LookupError as exc:
                return refused(404, "unknown", str(exc))
            except (ValueError, TypeError, KeyError) as exc:
                return refused(422, "malformed_request", str(exc))
            except Exception as exc:
                logger.exception("trainer link %s failed", path)
                return refused(500, "internal_error", type(exc).__name__)
            return JSONResponse(content=result, headers={"Cache-Control": "no-store"})

        router.add_api_route(f"{prefix}/{path}", handler, methods=[method])

    def report(b: dict) -> dict:
        matched, subs = b.get("matched_tokens"), b.get("substitutions")
        rl.submissions.report_training(str(b["lease_id"]), rollout_match=[float(m) for m in b["rollout_match"]],
                                       accepted=bool(b["accepted"]), failure_class=b.get("failure_class"),
                                       matched_tokens=None if matched is None else [int(n) for n in matched],
                                       substitutions=None if subs is None else [[(int(p), int(t)) for p, t in s]
                                                                                for s in subs])
        return {"ok": True}

    def adopt(b: dict) -> dict:
        rl.book.adopt_checkpoint(int(b["k"]))
        return {"latest": rl.book.latest_checkpoint}

    route("GET", "config", lambda b: {"task_id": rl.settings.task_id, "run": rl.settings.run,
                                      "targets": dict(rl.settings.targets), "latest": rl.book.latest_checkpoint})
    route("GET", "latest", lambda b: {"latest": rl.book.latest_checkpoint})
    route("POST", "adopt", adopt)
    route("POST", "take", lambda b: {"groups": [_pack(g) for g in rl.buffer.take(int(b["limit"]))]})
    route("POST", "take_older", lambda b: {"groups": [_pack(g) for g in rl.buffer.take_older(int(b["below_k"]),
                                                                                             int(b["limit"]))]})
    route("POST", "release", lambda b: rl.buffer.release([str(i) for i in b["lease_ids"]]) or {"ok": True})
    route("POST", "requeue", lambda b: {"requeued": rl.buffer.requeue()})
    route("GET", "depth", lambda b: {"depth": rl.buffer.depth()})
    route("POST", "report", report)

    async def checkpoint(request: Request) -> JSONResponse:
        """``/rl/checkpoint``'s answer for the trainer's resume (it reaches this listener only)."""
        if not authorized(request):
            return refused(401, "unauthorized")
        answer = None if rl.board is None else await rl.board.answer()
        if answer is None:
            return refused(503, "unavailable")
        return JSONResponse(content=answer, headers={"Cache-Control": "no-store"})

    router.add_api_route(f"{prefix}/checkpoint", checkpoint, methods=["GET"])
    return router


def trainer_bind(env: Mapping[str, str] = os.environ) -> tuple[str, int]:
    """``RELIQUARY_RL_TRAINER_BIND`` as (host, port): the trainer link's own listener. ValueError for a host that
    is not loopback unless ``RELIQUARY_RL_TRAINER_INSECURE_BIND=1`` (plain HTTP: a private network only)."""
    import ipaddress

    value = str(env.get(TRAINER_BIND_ENV, "")).strip() or DEFAULT_TRAINER_BIND
    host, sep, port = value.rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"{TRAINER_BIND_ENV} must be host:port, got {value!r}")
    host = host.strip("[]")
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and str(env.get(TRAINER_INSECURE_BIND_ENV, "")).strip() != "1":
        raise ValueError(f"{TRAINER_BIND_ENV}={value!r} is not loopback: the trainer link is plain HTTP and its "
                         f"token mints pay; bind it on a private network and set {TRAINER_INSECURE_BIND_ENV}=1, "
                         "or keep it on loopback behind an SSH tunnel")
    return host, int(port)


def build_rl_trainer_app(rl: RlAsync, *, token: str):
    """The trainer link as its own app, served on ``trainer_bind()`` beside the public one."""
    from fastapi import FastAPI

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(build_rl_trainer_router(rl, token=token))
    return app


# --- routes -----------------------------------------------------------------------------------------------------
def mount_rl_async(server, rl: RlAsync, *, validator_hotkey: str, registration=None,
                   clock: Callable[[], float] = time.time) -> None:
    """Serve the public async routes on ``server`` (a ``ValidatorServer``); each replaces an app route at the same
    path. The trainer link is not among them (``build_rl_trainer_app``, its own listener)."""
    from reliquary.validator.rl_checkpoint import rl_checkpoint_response
    from reliquary.validator.rl_observations import rl_observations_response
    from reliquary.validator.rl_lease_routes import build_rl_lease_router

    router = build_rl_lease_router(book=rl.book, collector=rl.collector, validator_hotkey=validator_hotkey,
                                   registration=registration, clock=clock, submissions=rl.submissions,
                                   pay=rl.ledger, max_episode_body_bytes=episode_body_cap(rl.settings),
                                   admit=lambda env, idx: _lease_admission(rl, env, idx), gpu_registry=rl.gpu_registry)
    if rl.gpu_registry is not None:
        from reliquary.validator.rl_gpu_routes import build_gpu_router

        router.include_router(build_gpu_router(rl.gpu_registry, registration=registration))

    @router.get("/rl/checkpoint")
    async def rl_checkpoint():
        return await rl_checkpoint_response(rl.board)

    @router.get("/rl/observations")
    async def rl_observations(after: str = "0"):
        return await rl_observations_response(rl.observations, after)

    if rl.settings.episodes:
        from fastapi.responses import JSONResponse as _Json

        from reliquary.validator.rl_async_episodes import episodes_answer

        answer = episodes_answer(rl.settings.episodes, rl.settings.lease_expiry_s)

        @router.get("/rl/episodes")
        async def rl_episodes():
            """The episode envs' blocks this validator grades with (``rl_async_episodes``): the miner builds its
            policies and task prompts from them."""
            return _Json(content=answer, headers={"Cache-Control": "max-age=60"})

    server.mount_rl_lease_router(router)
    if rl.sandbox is not None:
        for sessions in rl.sandbox.routers:
            server.mount_rl_lease_router(sessions)


async def _lease_admission(rl, environment: str, prompt_idx: int) -> tuple[str, dict] | None:
    """The lease route's ``admit``: an episode env's task needs a fleet that can host its sessions now
    (``RlLeaseSandbox.lease_refusal``); every other lease is the book's alone."""
    sandbox = getattr(rl, "sandbox", None)
    if sandbox is None:
        return None
    return await sandbox.lease_refusal(environment, prompt_idx)


def episode_body_cap(settings: RlAsyncSettings) -> int | None:
    """The inflated ``/rl/submit`` body an episode group may send (None: no episode env, the single-turn cap):
    16 transcripts at their largest plus the tokens, at most ``MAX_EPISODE_SUBMIT_BODY_BYTES``."""
    if not settings.episodes:
        return None
    from reliquary.validator.rl_lease_routes import MAX_EPISODE_SUBMIT_BODY_BYTES

    return MAX_EPISODE_SUBMIT_BODY_BYTES


# --- the verify model at the latest checkpoint -----------------------------------------------------------------
@dataclass
class AuditModel:
    """The validator's verify model, kept at the latest published checkpoint (audits run against the latest, §4.4):
    ``follow_once`` rebuilds the newest version from the board's signed index (``CheckpointSync``) and loads it."""
    model: Any
    sync: Any
    version: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    _state: dict | None = None

    async def follow_once(self, board) -> int | None:
        answer = None if board is None else await board.answer()
        if not answer:
            return None
        latest = int(answer["index"]["latest"]["version"])
        if self.version is not None and latest <= self.version:
            return None
        state, version, _ = await asyncio.to_thread(self.sync.sync, answer, self._state, self.version)

        def load() -> None:
            with self.lock:
                self.model.load_state_dict(state)

        await asyncio.to_thread(load)
        self._state, self.version = state, version
        logger.info("audit model at checkpoint %d", version)
        return version

    def audit(self, latest: int | None, submissions, limit: int) -> int | None:
        """Drain audits when the model holds ``latest``; None (nothing run) while it lags."""
        from reliquary.validator.rl_audit import run_rl_audits

        if latest is None or self.version != latest:
            return None
        with self.lock:
            return run_rl_audits(self.model, submissions, limit=limit)


# --- loops ------------------------------------------------------------------------------------------------------
async def _every(step: Callable[[], Any], period: float, name: str) -> None:
    while True:
        try:
            await step()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("rl async %s failed; retrying in %.0fs", name, period)
        await asyncio.sleep(period)


def rl_async_loops(rl: RlAsync, *, archives, audit_model: AuditModel | None = None, audit_limit: int = AUDIT_LIMIT,
                   clock: Callable[[], float] = time.time, every: Mapping[str, float] | None = None,
                   publisher=None) -> list:
    """The background coroutines of the service (run until cancelled). ``publisher``: the scan log's
    ``ObservationPublisher`` (``rl_observations.scan_publisher``)."""
    from reliquary.validator import telemetry
    from reliquary.validator.rl_metrics import METRICS_EVERY_S, controller_metrics
    from reliquary.validator.rl_observations import FLUSH_EVERY_S

    periods = {"expire": EXPIRE_EVERY_S, "audit": AUDIT_EVERY_S, "ledger": LEDGER_EVERY_S, "price": PRICE_EVERY_S,
               "checkpoint": CHECKPOINT_FOLLOW_EVERY_S, "observations": FLUSH_EVERY_S, "metrics": METRICS_EVERY_S,
               **dict(every or {})}

    async def expire() -> None:
        expired = await asyncio.to_thread(rl.book.expire_due)
        if expired:
            logger.info("rl leases expired: %d", expired)

    async def audit() -> None:
        from reliquary.validator.rl_audit import run_rl_audits

        if audit_model is None:     # no verify model: inconclusive (unpaid, unsanctioned), never a stale verdict
            await asyncio.to_thread(run_rl_audits, None, rl.submissions, limit=audit_limit)
            return
        latest = await asyncio.to_thread(lambda: rl.book.latest_checkpoint)
        await asyncio.to_thread(audit_model.audit, latest, rl.submissions, audit_limit)

    async def ledger() -> None:
        await asyncio.to_thread(rl.ledger.ingest)
        settled = await asyncio.to_thread(rl.ledger.settle)
        for doc in settled:
            logger.info("rl period %d settled: %d hotkeys, scale %.3f", doc["work_period"],
                        len(doc["rewards_by_hotkey"]), doc["scale"])
        if archives is not None:
            await rl.ledger.publish(archives)

    async def price() -> None:
        now = float(clock())
        if not await asyncio.to_thread(rl.prices.due, now):
            return
        await asyncio.to_thread(lambda: rl.prices.tick(rl.buffer.drain_tick(), rl.book.drain_full(), now))

    async def follow() -> None:
        await audit_model.follow_once(rl.board)

    async def publish() -> None:
        while not publisher.disabled and await publisher.flush() is not None:
            pass

    async def metrics() -> None:
        row = await asyncio.to_thread(controller_metrics, rl, float(clock()), periods["metrics"])
        telemetry.log_training_step(row, step=None)

    loops = [_every(expire, periods["expire"], "expiry sweep"), _every(audit, periods["audit"], "audits"),
             _every(ledger, periods["ledger"], "ledger"), _every(price, periods["price"], "price tick")]
    if telemetry.is_active():
        loops.append(_every(metrics, periods["metrics"], "wandb metrics"))
    if audit_model is not None:
        loops.append(_every(follow, periods["checkpoint"], "audit checkpoint"))
    if publisher is not None:
        loops.append(_every(publish, periods["observations"], "scan log publication"))
    return loops


# --- the process --------------------------------------------------------------------------------------------------
def _grader_for(names, checkpoint: str, load_kwargs: Mapping[str, Any], episodes: Mapping[str, Any] | None = None):
    """The submission grader and the env prompt ranges, from the run's tokenizer and model config. ``episodes``:
    the signed-episode envs as loaded (``rl_async_episodes.load_episode_environment``): graded by their checkers
    (``GroupGrader.add_episodes``, once the sandbox side is built), prompt range = their whole taskset."""
    from types import SimpleNamespace

    from transformers import AutoConfig, GenerationConfig

    from reliquary.environment import load_environment
    from reliquary.protocol.rl_prompt_ranges import rl_prompt_range
    from reliquary.shared.modeling import (
        force_close_token_ids, load_tokenizer, resolve_eos_token_ids, think_close_token_ids,
    )
    from reliquary.validator.rl_submit import GroupGrader

    tokenizer = load_tokenizer(checkpoint, **load_kwargs)
    config = AutoConfig.from_pretrained(checkpoint, **load_kwargs)
    try:
        generation = GenerationConfig.from_pretrained(checkpoint, **load_kwargs)
    except OSError:
        generation = None
    eos = resolve_eos_token_ids(SimpleNamespace(config=config, generation_config=generation), tokenizer)
    try:
        canonical, think_close = force_close_token_ids(tokenizer), think_close_token_ids(tokenizer)
    except Exception:
        canonical, think_close = (), ()
    episodes = dict(episodes or {})
    environments = {name: load_environment(name) for name in names if name not in episodes}
    vocab = getattr(config, "vocab_size", None) or getattr(getattr(config, "text_config", None), "vocab_size", None)
    grader = GroupGrader(environments, tokenizer, eos_token_ids=sorted(eos), think_close_ids=think_close,
                         canonical_force_ids=canonical, vocab_size=vocab)
    ranges = {name: rl_prompt_range(name, len(env)) for name, env in environments.items()}
    ranges.update({name: (0, len(env)) for name, env in episodes.items()})
    return grader, ranges


def load_episode_environments(settings: RlAsyncSettings, *, build_source=None) -> dict:
    """Each episode env of ``settings`` as loaded (blocking: packages and tasksets)."""
    from reliquary.validator.rl_async_episodes import load_episode_environment

    return {name: load_episode_environment(name, block, build_source=build_source)
            for name, block in settings.episodes.items()}


def renderer_for_checkpoint(checkpoint: str) -> Callable[[Any], Any]:
    """The turn renderer of the run's tokenizer over a policy's tools (the window validator's, one per tool set)."""
    renderers: dict[tuple, Any] = {}

    def renderer_for(policy):
        from reliquary.environment.agentic_swe import load_turn_renderer

        tools = tuple(policy.tools)
        if tools not in renderers:
            renderers[tools] = load_turn_renderer(str(checkpoint), tools=tools)
        return renderers[tools]

    return renderer_for


def wire_episode_sandbox(rl: RlAsync, config, *, validator_hotkey: str, environments: Mapping[str, Any],
                         renderer_for: Callable[[Any], Any], registration=None, **kwargs):
    """The signed-sandbox side of the episode envs (``rl_async_episodes.build_rl_lease_sandbox`` over the lease
    book) on ``rl``: its checkers grade the episode envs' groups. ``kwargs``: the builder's (tests: fakes)."""
    from reliquary.validator.rl_async_episodes import build_rl_lease_sandbox

    sandbox = build_rl_lease_sandbox(config, validator_hotkey=validator_hotkey, lease_book=rl.book,
                                     environments=environments, policies=rl.settings.episode_policies(),
                                     renderer_for=renderer_for, registration=registration, **kwargs)
    rl.submissions.grader.add_episodes(sandbox.checkers)
    rl.sandbox = sandbox
    return sandbox


def index_signer_for(wallet, signer_client=None):
    """The checkpoint index's signer: the remote signer's index operation when ``RELIQUARY_SIGNER_URL`` gave one
    (the signer's ``RELIQUARY_SIGNER_RL_RUN`` must name this run), else the local wallet."""
    from reliquary.validator.rl_checkpoint import remote_index_signer, wallet_index_signer

    return remote_index_signer(signer_client) if signer_client is not None else wallet_index_signer(wallet)


def _public_boards(rl: RlAsync, wallet, validator_hotkey: str, signer_client=None):
    """The checkpoint board and the scan log (publisher + board), both over the ``reliquary-public`` bucket and
    signed by the remote signer when ``signer_client`` is given, else by the local wallet. Returns the publisher."""
    from reliquary.infrastructure.storage import public_checkpoint_bucket, public_checkpoint_client
    from reliquary.shared.ckpt_index import index_key
    from reliquary.trainer.ckpt_publisher import R2ObjectStore
    from reliquary.validator.rl_checkpoint import RlCheckpointBoard, presigner
    from reliquary.validator.rl_observations import ObservationBoard, bucket_io, scan_publisher

    run = rl.settings.run
    client, bucket = public_checkpoint_client(), public_checkpoint_bucket()
    store, presign = R2ObjectStore(client, bucket), presigner(client, bucket)
    rl.board = RlCheckpointBoard(load_index=lambda: store.get_bytes(index_key(run)),
                                 index_signer=index_signer_for(wallet, signer_client),
                                 validator_hotkey=validator_hotkey, presign=presign)
    put, get = bucket_io(client, bucket)
    publisher = scan_publisher(rl.book, run=run, task_id=rl.settings.task_id, wallet=wallet, put=put, get=get,
                               signer_client=signer_client)
    rl.observations = ObservationBoard(publisher, presign=presign)
    return publisher


def audit_source(env: Mapping[str, str] = os.environ) -> str | None:
    """Where the audit forward runs: ``remote`` (the proof worker, V1's GPU-less controller), ``local`` (a device of
    this process) or None (every audit inconclusive). ValueError when both are configured."""
    remote = str(env.get("RELIQUARY_PROOF_EXECUTOR_MODE", "")).strip().lower() == "remote"
    local = bool(str(env.get(AUDIT_DEVICE_ENV, "")).strip())
    if remote and local:
        raise ValueError(f"{AUDIT_DEVICE_ENV} is set while the proof executor is remote: audits run on one of them")
    return "remote" if remote else "local" if local else None


def _audit_model(device: str, checkpoint: str, load_kwargs: Mapping[str, Any], validator_hotkey: str,
                 work_dir: Path) -> AuditModel:
    import torch

    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.miner.ckpt_sync import CheckpointSync
    from reliquary.shared.modeling import load_text_generation_model

    model = load_text_generation_model(checkpoint, torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION,
                                       **load_kwargs).to(device).eval()
    return AuditModel(model, CheckpointSync(validator_hotkey=validator_hotkey, work_dir=work_dir))


async def run_rl_async_validator(*, wallet, netuid: int, signer_client, http_host: str, http_port: int, entry,
                                 cap: float, checkpoint: str, load_kwargs: Mapping[str, Any] | None = None,
                                 registration_gate: bool = True, subtensor=None, external_ip: str | None = None,
                                 external_port: int | None = None) -> None:
    """Serve the async RL task ``entry`` until interrupted (the weight setter runs beside it, as in window mode)."""
    import uvicorn

    from reliquary.infrastructure.corpus_period_store import R2PeriodArchives
    from reliquary.shared.log_levels import unmute_reliquary_loggers
    from reliquary.validator.corpus_settlement import R2Archives
    from reliquary.validator.corpus_validator import _run_corpus_services
    from reliquary.validator.server import ValidatorServer
    from reliquary.validator.service import _validator_state_dir

    unmute_reliquary_loggers()
    load_kwargs = dict(load_kwargs or {})
    settings = rl_async_settings(entry, cap)
    token = os.environ.get(TRAINER_TOKEN_ENV, "").strip()
    bind = trainer_bind() if token else None
    validator_hotkey = wallet.hotkey.ss58_address
    sandbox_config = None
    if settings.episodes:
        from reliquary.validator.sandbox_wiring import SandboxValidatorConfig

        sandbox_config = SandboxValidatorConfig.from_env()
        if sandbox_config is None:
            raise ValueError("set RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE and RELIQUARY_SANDBOX_VALIDATOR_KEY_ID to "
                             f"serve the signed-episode envs {sorted(settings.episodes)}")
    episode_envs = await asyncio.to_thread(load_episode_environments, settings)
    grader, ranges = await asyncio.to_thread(_grader_for, list(settings.targets), checkpoint, load_kwargs,
                                             episode_envs)
    state_dir = _validator_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    rl = build_rl_async(settings, path=state_dir / DB_NAME, grader=grader, prompt_range=ranges)
    from reliquary.validator.gpu_registry import registry_from_env

    try:
        rl.gpu_registry = registry_from_env(state_dir / "rl-gpu-admission.sqlite3", validator=validator_hotkey,
                                            run=settings.run)
    except BaseException:
        rl.close()
        raise
    from reliquary.validator import telemetry

    telemetry.init(validator_hotkey, {"task_id": settings.task_id, "run": settings.run, "cap": cap,
                                      "targets": dict(settings.targets),
                                      "max_pending_groups": settings.max_pending_groups},
                   role="controller", version=settings.run)
    publisher = _public_boards(rl, wallet, validator_hotkey, signer_client)
    audit_model, source = None, audit_source()
    if source == "remote":
        from reliquary.validator.remote_rl_audit import RemoteAuditModel

        audit_model = RemoteAuditModel.from_environment()
        logger.info("out-of-zone audits run on the remote proof worker %s", audit_model.worker_id)
    elif source == "local":
        audit_model = await asyncio.to_thread(_audit_model, os.environ[AUDIT_DEVICE_ENV].strip(), checkpoint,
                                              load_kwargs, validator_hotkey, state_dir / "rl-audit-ckpt")
    else:
        logger.warning("no audit model (%s unset, proof executor not remote): out-of-zone audits conclude "
                       "inconclusive (first scans unpaid, no sanction)", AUDIT_DEVICE_ENV)
    if not token:
        logger.warning("%s unset: no trainer link is served; the async trainer cannot connect", TRAINER_TOKEN_ENV)

    registered = None
    if registration_gate:
        from reliquary.validator.corpus_registration import RegisteredHotkeys, load_registered_hotkeys

        registered = RegisteredHotkeys(load=lambda: load_registered_hotkeys(netuid))
        if not await registered.refresh():
            logger.warning("subnet registrations unknown at start; miners get 503 until they load")
    if settings.episodes:
        wire_episode_sandbox(rl, sandbox_config, validator_hotkey=validator_hotkey, environments=episode_envs,
                             renderer_for=renderer_for_checkpoint(checkpoint),
                             registration=None if registered is None else registered.reason,
                             require_resources=True)
        await rl.sandbox.start()
    server = ValidatorServer(http_host, http_port, emission_cap=cap)
    mount_rl_async(server, rl, validator_hotkey=validator_hotkey,
                   registration=None if registered is None else registered.reason)
    archives = R2PeriodArchives(guard=R2Archives(served=lambda: (settings.task_id,), served_only=True))
    background = rl_async_loops(rl, archives=archives, audit_model=audit_model, publisher=publisher)
    if rl.sandbox is not None:
        background.extend(rl.sandbox.background())
    if registered is not None:
        background.append(registered.refresh_forever())
    if bind is not None:
        trainer_http = uvicorn.Server(uvicorn.Config(build_rl_trainer_app(rl, token=token), host=bind[0],
                                                     port=bind[1], log_level="warning", access_log=False))
        trainer_http.capture_signals = contextlib.nullcontext   # the public server owns the signals
        background.append(trainer_http.serve())
        logger.info("trainer link on %s:%d", *bind)
    if external_ip and (subtensor is not None or signer_client is not None):
        from reliquary.validator.axon import serve_axon_on_chain

        # Through the signer's serve-axon operation when there is one (V1's controller holds no key).
        await serve_axon_on_chain(subtensor, wallet=wallet, netuid=netuid, ip=external_ip,
                                  port=external_port or http_port, signer_client=signer_client)
    else:
        logger.warning("no --external-ip: the axon is not served; miners need --validator-url")
    logger.info("async RL validator: task %s (cap %.3f), run %s, envs %s, latest checkpoint %s",
                settings.task_id, cap, settings.run, dict(settings.targets), rl.book.latest_checkpoint)
    http = uvicorn.Server(uvicorn.Config(server.app, host=http_host, port=http_port, log_level="warning",
                                         access_log=False, timeout_keep_alive=15))

    async def cleanup() -> None:
        if rl.sandbox is not None:
            try:
                await rl.sandbox.stop()
            except Exception:
                logger.exception("rl lease sessions not drained at shutdown")
        await publisher.close()
        await asyncio.to_thread(rl.close)
        telemetry.finish()

    await _run_corpus_services(http, background, cleanup=cleanup)


__all__ = [
    "ASYNC", "AUDIT_DEVICE_ENV", "AuditModel", "DB_NAME", "LEASE_MARGIN_PARAM", "LEASE_SPEED_PARAM", "PENDING_PARAM",
    "PLANNED_TOKENS_PARAM", "PRICE_TICK_PARAM", "RL_MODE_ENV", "RUN_ENV", "RlAsync", "RlAsyncSettings",
    "TRAINER_BIND_ENV", "TRAINER_INSECURE_BIND_ENV", "TRAINER_TOKEN_ENV", "WINDOW", "audit_source", "build_rl_async", "build_rl_trainer_app",
    "build_rl_trainer_router", "index_signer_for", "mount_rl_async", "rl_async_loops", "rl_async_params",
    "episode_body_cap", "load_episode_environments", "renderer_for_checkpoint", "rl_async_settings", "rl_mode",
    "run_rl_async_validator", "trainer_bind", "wire_episode_sandbox",
]
