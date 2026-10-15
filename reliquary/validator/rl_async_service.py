"""The async RL validator service (rl-async-design.md §3; implementation map §9), behind one switch:
``RELIQUARY_RL_MODE=async`` (default ``window``: the window service runs, unchanged).

One process on the controller, like the corpus validator: no window, batcher or proof plane.

* **Components** (``build_rl_async``), all in one SQLite file under the state dir (``rl-async.sqlite3``): the
  ``LeaseBook`` with its ``QuotaController`` and ``LeaseCollector``, ``RlSubmissions`` (grading, audits, events)
  with its ``TrainingBuffer``, the ``AsyncPriceController`` and the ``RlPayLedger``; plus the ``RlCheckpointBoard``
  over the trainer's index in the ``reliquary-public`` bucket.
* **Routes** (``mount_rl_async``, through ``ValidatorServer.mount_rl_lease_router``): ``/rl/lease``, ``/rl/quota``,
  ``/rl/lease/mine``, ``/rl/lease/commitments``, ``/rl/prompts/state``, ``/rl/submit``, ``/rl/pay`` (replaces the
  window board's) and ``/rl/checkpoint``; with ``RELIQUARY_RL_TRAINER_TOKEN`` set, the trainer link
  (``/rl/trainer/*``, bearer token): the detached async trainer (``reliquary train-async``, its own GPU box as
  today) takes groups from the buffer, reports them, adopts the checkpoints it publishes and feeds the prices.
* **Loops** (``rl_async_loops``): the lease expiry sweep; the out-of-zone audits (``run_rl_audits``) against the
  verify model held at the latest checkpoint (``audit_source``: the remote proof worker's when
  ``RELIQUARY_PROOF_EXECUTOR_MODE=remote``, ``remote_rl_audit``; else a local ``RELIQUARY_RL_AUDIT_DEVICE``; without
  either every audit concludes inconclusive: unpaid, unsanctioned), with ``RlSubmissions.run_audits``' retry cap; the ledger (ingest, settle,
  publish the period archives through ``R2PeriodArchives``, as the corpus periods); the price tick when the trainer
  is idle (it ticks per step itself).
* **Settlement:** the task must be declared with ``params.settlement = "period-ema-v1"`` (``reliquary tasks create
  --rl-async``): the unchanged weight setter pays its period archives. Per-env target shares are the entry's
  ``env_split``; the async parameters are the ``rl_*`` params below.

* **Signing:** the checkpoint index is signed by the remote signer's index operation when ``RELIQUARY_SIGNER_URL``
  is set (signer-01 needs ``RELIQUARY_SIGNER_RL_RUN`` = the run), else by the local wallet.

Not here (map §9): the R2 segments of the observation log (decision D; ``rl_prompt_scans`` is served by
``/rl/prompts/state``), axon publication.
"""
from __future__ import annotations

import asyncio
import base64
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
AUDIT_DEVICE_ENV = "RELIQUARY_RL_AUDIT_DEVICE"
RUN_ENV = "RELIQUARY_RL_RUN"
DB_NAME = "rl-async.sqlite3"

# Registry params of an async task (``rl_async_params``), beside the price fields every entry carries.
PLANNED_TOKENS_PARAM = "rl_planned_tokens_per_period"   # matched tokens the trainer consumes per 72-min period
LEASE_SPEED_PARAM = "rl_lease_reference_tokens_per_s"   # a slow reference card's decode speed, per sequence
LEASE_MARGIN_PARAM = "rl_lease_expiry_margin"
PRICE_TICK_PARAM = "rl_price_tick_s"
OVERFLOW_PARAM = "rl_buffer_overflow_groups"
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
                    overflow_groups: int | None = None) -> dict:
    """The params an async task adds to its registry entry (pay by period: ``settlement = period-ema-v1``)."""
    from reliquary.shared.task_registry import SETTLEMENT_PERIOD_EMA

    if not (planned_tokens_per_period > 0 and lease_tokens_per_s > 0 and lease_margin >= 1):
        raise ValueError("planned tokens and lease speed must be positive, the margin at least 1")
    params: dict[str, Any] = {"settlement": SETTLEMENT_PERIOD_EMA,
                              PLANNED_TOKENS_PARAM: float(planned_tokens_per_period),
                              LEASE_SPEED_PARAM: float(lease_tokens_per_s), LEASE_MARGIN_PARAM: float(lease_margin)}
    if tick_s is not None:
        params[PRICE_TICK_PARAM] = float(tick_s)
    if overflow_groups is not None:
        params[OVERFLOW_PARAM] = int(overflow_groups)
    return params


@dataclass(frozen=True)
class RlAsyncSettings:
    task_id: str
    cap: float
    run: str                                    # the checkpoint index's run (``rl/<run>/index.json``)
    targets: Mapping[str, float]                # env -> target share of trained tokens (the trainer's mix)
    planned_tokens_per_period: float
    lease_expiry_s: Mapping[str, float]
    price: Any = None                           # AsyncPriceParams (None: the defaults)


def rl_async_settings(entry, cap: float, *, run: str | None = None,
                      max_tokens: Callable[[str], int] | None = None) -> RlAsyncSettings:
    """An async task's settings from its registry entry. ValueError when the entry was not declared for it."""
    from reliquary.validator import corpus_periods as cp
    from reliquary.validator.rl_price import AsyncPriceParams
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
    expiry = {env: lease_expiry_s(max_tokens=int(max_tokens(env)), reference_tokens_per_s=speed, margin=margin)
              for env in targets}
    price = AsyncPriceParams(**{key: cast(params[name]) for key, name, cast in (
        ("tick_s", PRICE_TICK_PARAM, float), ("overflow_groups", OVERFLOW_PARAM, int)) if name in params})
    return RlAsyncSettings(task_id=str(entry.task_id), cap=float(cap),
                           run=run or os.environ.get(RUN_ENV) or str(entry.task_id), targets=targets,
                           planned_tokens_per_period=float(params[PLANNED_TOKENS_PARAM]), lease_expiry_s=expiry,
                           price=price)


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

    def close(self) -> None:
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
                                           prompt_range=None if prompt_range is None else dict(prompt_range)),
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

    def route(method: str, path: str, op: Callable[[dict], Any]) -> None:
        async def handler(request: Request) -> JSONResponse:
            if not hmac.compare_digest(request.headers.get("authorization", "").encode(), expected):
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
        matched = b.get("matched_tokens")
        rl.submissions.report_training(str(b["lease_id"]), rollout_match=[float(m) for m in b["rollout_match"]],
                                       accepted=bool(b["accepted"]), failure_class=b.get("failure_class"),
                                       matched_tokens=None if matched is None else [int(n) for n in matched])
        return {"ok": True}

    def adopt(b: dict) -> dict:
        rl.book.adopt_checkpoint(int(b["k"]))
        return {"latest": rl.book.latest_checkpoint}

    def wait(b: dict) -> dict:
        rl.prices.note_wait(str(b["environment"]), float(b["seconds"]))
        return {"ok": True}

    def tick(b: dict) -> dict:
        trained = {str(env): int(n) for env, n in dict(b["trained_tokens"]).items()}
        decisions = rl.prices.tick(trained, rl.buffer.depth())
        return {env: {"price": d.price, "regime": d.regime} for env, d in decisions.items()}

    route("GET", "config", lambda b: {"task_id": rl.settings.task_id, "run": rl.settings.run,
                                      "targets": dict(rl.settings.targets), "latest": rl.book.latest_checkpoint})
    route("GET", "latest", lambda b: {"latest": rl.book.latest_checkpoint})
    route("POST", "adopt", adopt)
    route("POST", "take", lambda b: {"groups": [_pack(g) for g in rl.buffer.take(str(b["environment"]),
                                                                                 int(b["limit"]))]})
    route("POST", "take_older", lambda b: {"groups": [_pack(g) for g in rl.buffer.take_older(int(b["below_k"]),
                                                                                             int(b["limit"]))]})
    route("POST", "release", lambda b: rl.buffer.release([str(i) for i in b["lease_ids"]]) or {"ok": True})
    route("POST", "requeue", lambda b: {"requeued": rl.buffer.requeue()})
    route("GET", "depth", lambda b: {"depth": rl.buffer.depth()})
    route("POST", "report", report)
    route("POST", "wait", wait)
    route("POST", "tick", tick)
    return router


# --- routes -----------------------------------------------------------------------------------------------------
def mount_rl_async(server, rl: RlAsync, *, validator_hotkey: str, registration=None, trainer_token: str | None = None,
                   clock: Callable[[], float] = time.time) -> None:
    """Serve the async routes on ``server`` (a ``ValidatorServer``); each replaces an app route at the same path."""
    from reliquary.validator.rl_checkpoint import rl_checkpoint_response
    from reliquary.validator.rl_lease_routes import build_rl_lease_router

    router = build_rl_lease_router(book=rl.book, collector=rl.collector, validator_hotkey=validator_hotkey,
                                   registration=registration, clock=clock, submissions=rl.submissions,
                                   pay=rl.ledger)

    @router.get("/rl/checkpoint")
    async def rl_checkpoint():
        return await rl_checkpoint_response(rl.board)

    if trainer_token:
        router.include_router(build_rl_trainer_router(rl, token=trainer_token))
    server.mount_rl_lease_router(router)


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
                   clock: Callable[[], float] = time.time, every: Mapping[str, float] | None = None) -> list:
    """The background coroutines of the service (run until cancelled)."""
    periods = {"expire": EXPIRE_EVERY_S, "audit": AUDIT_EVERY_S, "ledger": LEDGER_EVERY_S, "price": PRICE_EVERY_S,
               "checkpoint": CHECKPOINT_FOLLOW_EVERY_S, **dict(every or {})}

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
            return                   # the trainer ticks per step; this tick covers an idle trainer
        await asyncio.to_thread(lambda: rl.prices.tick(rl.ledger.trained_tokens(now - rl.prices.params.tick_s, now),
                                                       rl.buffer.depth(), now))

    async def follow() -> None:
        await audit_model.follow_once(rl.board)

    loops = [_every(expire, periods["expire"], "expiry sweep"), _every(audit, periods["audit"], "audits"),
             _every(ledger, periods["ledger"], "ledger"), _every(price, periods["price"], "price tick")]
    if audit_model is not None:
        loops.append(_every(follow, periods["checkpoint"], "audit checkpoint"))
    return loops


# --- the process --------------------------------------------------------------------------------------------------
def _grader_for(names, checkpoint: str, load_kwargs: Mapping[str, Any]):
    """The submission grader and the env prompt ranges, from the run's tokenizer and model config."""
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
    environments = {name: load_environment(name) for name in names}
    vocab = getattr(config, "vocab_size", None) or getattr(getattr(config, "text_config", None), "vocab_size", None)
    grader = GroupGrader(environments, tokenizer, eos_token_ids=sorted(eos), think_close_ids=think_close,
                         canonical_force_ids=canonical, vocab_size=vocab)
    ranges = {name: rl_prompt_range(name, len(env)) for name, env in environments.items()}
    return grader, ranges


def index_signer_for(wallet, signer_client=None):
    """The checkpoint index's signer: the remote signer's index operation when ``RELIQUARY_SIGNER_URL`` gave one
    (the signer's ``RELIQUARY_SIGNER_RL_RUN`` must name this run), else the local wallet."""
    from reliquary.validator.rl_checkpoint import remote_index_signer, wallet_index_signer

    return remote_index_signer(signer_client) if signer_client is not None else wallet_index_signer(wallet)


def _checkpoint_board(run: str, index_signer, validator_hotkey: str):
    from reliquary.infrastructure.storage import public_checkpoint_bucket, public_checkpoint_client
    from reliquary.shared.ckpt_index import index_key
    from reliquary.trainer.ckpt_publisher import R2ObjectStore
    from reliquary.validator.rl_checkpoint import RlCheckpointBoard, presigner

    client, bucket = public_checkpoint_client(), public_checkpoint_bucket()
    store = R2ObjectStore(client, bucket)
    return RlCheckpointBoard(load_index=lambda: store.get_bytes(index_key(run)), index_signer=index_signer,
                             validator_hotkey=validator_hotkey, presign=presigner(client, bucket))


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
                                 registration_gate: bool = True) -> None:
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
    validator_hotkey = wallet.hotkey.ss58_address
    grader, ranges = await asyncio.to_thread(_grader_for, list(settings.targets), checkpoint, load_kwargs)
    state_dir = _validator_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    rl = build_rl_async(settings, path=state_dir / DB_NAME, grader=grader, prompt_range=ranges,
                        board=_checkpoint_board(settings.run, index_signer_for(wallet, signer_client),
                                                validator_hotkey))
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
    token = os.environ.get(TRAINER_TOKEN_ENV, "").strip()
    if not token:
        logger.warning("%s unset: no trainer link is served; the async trainer cannot connect", TRAINER_TOKEN_ENV)

    registered = None
    if registration_gate:
        from reliquary.validator.corpus_registration import RegisteredHotkeys, load_registered_hotkeys

        registered = RegisteredHotkeys(load=lambda: load_registered_hotkeys(netuid))
        if not await registered.refresh():
            logger.warning("subnet registrations unknown at start; miners get 503 until they load")
    server = ValidatorServer(http_host, http_port, emission_cap=cap)
    mount_rl_async(server, rl, validator_hotkey=validator_hotkey,
                   registration=None if registered is None else registered.reason, trainer_token=token or None)
    archives = R2PeriodArchives(guard=R2Archives(served=lambda: (settings.task_id,), served_only=True))
    background = rl_async_loops(rl, archives=archives, audit_model=audit_model)
    if registered is not None:
        background.append(registered.refresh_forever())
    logger.info("async RL validator: task %s (cap %.3f), run %s, envs %s, latest checkpoint %s",
                settings.task_id, cap, settings.run, dict(settings.targets), rl.book.latest_checkpoint)
    http = uvicorn.Server(uvicorn.Config(server.app, host=http_host, port=http_port, log_level="warning",
                                         access_log=False, timeout_keep_alive=15))

    async def cleanup() -> None:
        await asyncio.to_thread(rl.close)

    await _run_corpus_services(http, background, cleanup=cleanup)


__all__ = [
    "ASYNC", "AUDIT_DEVICE_ENV", "AuditModel", "DB_NAME", "LEASE_MARGIN_PARAM", "LEASE_SPEED_PARAM", "OVERFLOW_PARAM",
    "PLANNED_TOKENS_PARAM", "PRICE_TICK_PARAM", "RL_MODE_ENV", "RUN_ENV", "RlAsync", "RlAsyncSettings",
    "TRAINER_TOKEN_ENV", "WINDOW", "audit_source", "build_rl_async", "build_rl_trainer_router", "mount_rl_async", "rl_async_loops",
    "rl_async_params", "rl_async_settings", "rl_mode", "run_rl_async_validator",
]
