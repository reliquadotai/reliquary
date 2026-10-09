"""The RL validator's signed-sandbox side (phase 2, plan 2C): the validator's token key, the machine
directory and fleet, the RL session book and issuer (sessions stored under their own prefix), the session
and precommit routes under ``/rl``, and the episode intake (one checker per episode env). Built only when
the order has signed-episode envs; it imports reliquary-sandbox, which a validator without such an env
never loads.

Settings: the corpus validator's (``sandbox_wiring.SandboxValidatorConfig``). The RL policy is raised to
what one hotkey's groups in flight need (``rl_sandbox_policy``); plan 2D's quota (``SessionQuota``)
decides beyond that.

Windows. Episode intake (precommits, opens, groups) is open only for a window THIS process opened
(``window_started_at``): a window resumed after a restart takes no episode group, since what its batcher
accepted before lived in memory only. A window older than ``FILL_CLOSED_MAX_SECONDS`` + margin opens no
session.

Hand-back. A paid group the proof could not judge for a reason of the validator's own is handed back by
its batcher (``episode_proof_inconclusive``, possibly on the event loop inside the seal or on a worker
thread): ``hand_back`` only schedules the work on the loop and returns; the issuer then marks the group's
sessions handed back in its book and its store (still ``submitted``: no same-window retry), and plan 2D's
outcomes hear of it."""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from reliquary.sandbox.rl_engagements import (
    EpisodeEnvironmentView, NoOutcomes, OpenQuota, RlEpisodeEngagements, SessionOutcomes, SessionQuota,
)
from reliquary.sandbox.sessions import SandboxPolicy, SessionBook, SessionIssuer
from reliquary.validator.corpus_registration import NOT_REGISTERED
from reliquary.validator.episode_admission import EpisodeGroupChecker
from reliquary.validator.episode_intake import DEFAULT_MAX_CHECKS_IN_FLIGHT, EpisodeGroupIntake

logger = logging.getLogger(__name__)

RL_PREFIX = "/rl"
SESSION_MAINTAIN_SECONDS = 60.0
RESTORE_ATTEMPTS = 3
RESTORE_TIMEOUT_S = 60.0
RESTORE_BACKOFF_S = 5.0
STOP_TIMEOUT_S = 10.0
# Episode groups of one hotkey in flight at once (the intake's per-operator check cap): each holds its
# 2M pool sessions plus one being reopened after a signed abort.
RL_GROUPS_IN_FLIGHT = DEFAULT_MAX_CHECKS_IN_FLIGHT
PRECOMMIT_PRUNE_SECONDS = 3600.0
MIN_PRECOMMIT_RETENTION_S = 2 * 86_400


def rl_sandbox_policy(base: SandboxPolicy, pool_seeds: int, *,
                      groups_in_flight: int = RL_GROUPS_IN_FLIGHT) -> SandboxPolicy:
    """``base`` raised for RL groups: ``max_live_per_hotkey`` >= (2M + 1) per group in flight, and the
    hourly open and daily abort caps scaled by the same factor over the defaults (a group opens 2M
    sessions at once; the corpus defaults are sized for one session per prompt)."""
    default = SandboxPolicy()
    live = (int(pool_seeds) + 1) * int(groups_in_flight)
    if base.max_live_per_hotkey >= live:
        live = base.max_live_per_hotkey
    scale = math.ceil(live / default.max_live_per_hotkey)
    return dataclasses.replace(
        base, max_live_per_hotkey=live,
        max_opens_per_hour=max(base.max_opens_per_hour, default.max_opens_per_hour * scale),
        max_aborted_per_day=max(base.max_aborted_per_day, default.max_aborted_per_day * scale))


def stop_ids_from_metadata(config: Mapping | None, generation_config: Mapping | None, tokenizer) -> set[int]:
    """The EOS set ``shared.modeling.resolve_eos_token_ids`` derives, from a remote proof worker's
    reported metadata (plain dicts, nested ``text_config`` included) and the local tokenizer."""
    from reliquary.shared.modeling import _iter_token_ids

    eos: set[int] = set()
    for source in (generation_config, config, (config or {}).get("text_config")):
        if isinstance(source, Mapping):
            eos.update(_iter_token_ids(source.get("eos_token_id")))
    if tokenizer is not None:
        eos.update(_iter_token_ids(getattr(tokenizer, "eos_token_id", None)))
    return eos


def episode_stop_set_refusal(renderer_stops: Mapping[str, Collection[int]], proof_stops: Collection[int],
                             remote_stops: Collection[int] | None = None) -> str | None:
    """Why the validator must not serve these episode envs, or None. Every renderer stop must be in the
    proof's stop set (a turn ending on a stop outside it can never be judged: every honest group would
    go unpaid), and a remote proof worker's stop set must be the batcher's (the unchecked-turn rule)."""
    proof = {int(t) for t in proof_stops}
    if not proof:
        return "the proof's stop set is empty"
    for name, stops in sorted(renderer_stops.items()):
        missing = {int(t) for t in stops} - proof
        if missing:
            return f"env {name}: renderer stop ids {sorted(missing)} are outside the proof's stop set"
    if remote_stops is not None and {int(t) for t in remote_stops} != proof:
        return (f"the remote proof worker's stop set {sorted(int(t) for t in remote_stops)} differs from "
                f"the batcher's {sorted(proof)}")
    return None


@dataclass
class RlEpisodeServices:
    signer: Any = field(repr=False)
    token_verifier: Any = field(repr=False)
    fleet: Any
    book: SessionBook
    issuer: SessionIssuer
    routers: tuple
    intake: EpisodeGroupIntake
    environments: tuple[str, ...] = ()
    runtime: Any = field(default=None, repr=False)
    outcomes: Any = field(default_factory=NoOutcomes, repr=False)
    clock: Callable[[], float] = field(default=time.time, repr=False)
    precommit_retention_s: float = MIN_PRECOMMIT_RETENTION_S
    restore_attempts: int = RESTORE_ATTEMPTS
    restore_timeout_s: float = RESTORE_TIMEOUT_S
    restore_backoff_s: float = RESTORE_BACKOFF_S
    _loop: Any = field(default=None, repr=False)
    _tasks: set = field(default_factory=set, repr=False)

    async def start(self) -> None:
        """Before serving: the directory read once (the fleet retries on its own), then the RL sessions
        restored (bounded, retried). A restore that still fails raises: a validator that lost its sessions
        would hand a taken seed out again."""
        self._loop = asyncio.get_running_loop()
        await self.fleet.refresh_if_due()
        restored = 0
        for attempt in range(1, self.restore_attempts + 1):
            try:
                restored = await asyncio.wait_for(self.issuer.restore(), self.restore_timeout_s)
                break
            except Exception as exc:
                if attempt == self.restore_attempts:
                    logger.error("rl sandbox sessions not restored after %d attempts (%s): the validator "
                                 "does not start", attempt, type(exc).__name__)
                    raise
                await asyncio.sleep(self.restore_backoff_s * 2 ** (attempt - 1))
        logger.info("rl sandbox sessions restored: %d", restored)

    def background(self) -> list:
        return [self.fleet.run(), self.issuer.maintain_forever(SESSION_MAINTAIN_SECONDS),
                self.prune_precommits_forever(PRECOMMIT_PRUNE_SECONDS)]

    async def stop(self, timeout: float = STOP_TIMEOUT_S) -> None:
        pending = [task for task in self._tasks if not task.done()]
        if pending:
            _, late = await asyncio.wait(pending, timeout=timeout)
            for task in late:
                task.cancel()
            if late:
                await asyncio.gather(*late, return_exceptions=True)
                logger.error("ALERT %d episode hand-backs cut at shutdown", len(late))
        await self.issuer.drain(timeout)

    # --- the batcher's hand-back hook (non-blocking) ---
    def batcher_hook(self, environment: str, window: int) -> Callable[[Any, str], None]:
        """``GrpoWindowBatcher.episode_proof_inconclusive`` of one episode env's batcher of ``window``."""
        def hook(pending, stage: str) -> None:
            self.hand_back(environment=environment, window=int(window), hotkey=str(pending.hotkey),
                           task_index=int(pending.prompt_idx), stage=str(stage))
        return hook

    def hand_back(self, *, environment: str, window: int, hotkey: str, task_index: int, stage: str) -> None:
        """Schedule the hand-back on the event loop and return at once (any thread; never blocks, never
        awaits, never takes the issuer lock)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            logger.error("episode group of %s (window %d, %s task %d) not handed back: no event loop",
                         hotkey[:12], window, environment, task_index)
            return
        loop.call_soon_threadsafe(self._spawn_hand_back, environment, window, hotkey, task_index, stage)

    def _spawn_hand_back(self, environment, window, hotkey, task_index, stage) -> None:
        task = asyncio.get_running_loop().create_task(
            self._hand_back(environment, window, hotkey, task_index, stage))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _hand_back(self, environment, window, hotkey, task_index, stage) -> tuple[str, ...]:
        try:
            sha = await asyncio.wait_for(asyncio.to_thread(
                self.runtime.episode_precommit_sha, window=window, environment=environment,
                task_index=task_index, hotkey=hotkey), self.issuer.policy.io_timeout_s)
            if sha is None:
                logger.error("episode group of %s (window %d, %s task %d): no precommit to hand back",
                             hotkey[:12], window, environment, task_index)
                return ()
            ids = await self.issuer.hand_back(sha, hotkey=hotkey)
            logger.warning("episode group of %s (precommit %s, %s): %d sessions handed back", hotkey[:12],
                           sha[:12], stage, len(ids))
            report = getattr(self.outcomes, "group_handed_back", None)
            if report is not None and ids:
                report(hotkey=hotkey, precommit_sha256=sha, session_ids=ids, stage=stage)
            return ids
        except Exception:
            logger.exception("episode group of %s (window %d, %s task %d) not handed back", hotkey[:12],
                             window, environment, task_index)
            return ()

    # --- precommit rows retention ---
    async def prune_precommits(self) -> int:
        """The runtime's precommit rows of settled windows older than the retention, except those a
        session still needs (held, or claimed by a group in flight)."""
        now = float(self.clock())
        keep = self.book.precommits_held(int(now))
        gone = await asyncio.to_thread(self.runtime.prune_episode_precommits,
                                       before=now - float(self.precommit_retention_s), keep=keep)
        if gone:
            logger.info("episode precommits pruned: %d", gone)
        return gone

    async def prune_precommits_forever(self, every_s: float) -> None:
        while True:
            try:
                await self.prune_precommits()
            except Exception:
                logger.exception("episode precommit pruning failed")
            await asyncio.sleep(every_s)


def rl_registration(server) -> Callable:
    """The sandbox routes' registration check, over the RL server's registration gate."""
    from reliquary.protocol.submission import RejectReason

    async def registration(hotkey: str) -> str | None:
        reason = await server._registration_reject_reason(hotkey)
        if reason is None:
            return None
        return NOT_REGISTERED if reason is RejectReason.HOTKEY_NOT_REGISTERED else "unavailable"

    return registration


def build_rl_episode_services(
    config, *, validator_hotkey: str, runtime, environments: Mapping[str, Any],
    renderer_for: Callable[[Any], Any], current_window: Callable[[], int | None], chunk_tokens: int,
    window_started_at: Callable[[int], float | None] | None = None,
    proof_stop_ids: Collection[int] | None = None, remote_stop_ids: Collection[int] | None = None,
    precommit_retention_s: float = MIN_PRECOMMIT_RETENTION_S,
    registration=None, quota: SessionQuota | None = None, outcomes: SessionOutcomes | None = None,
    signer=None, store_kwargs=None, session_store=None, read_documents=None, fetch_report=None,
    clock: Callable[[], float] = time.time,
) -> RlEpisodeServices:
    """``environments``: the order's episode envs as loaded (``SignedEpisodeEnvironment``; their
    ``source`` is plan 2A's); ``renderer_for(policy)``: the turn renderer of the policy's tokenizer over
    the policy's tools; ``current_window()``: the service window admissions are open for (memory only:
    it is read on the event loop); ``window_started_at(window)``: when this process opened it, None for
    a window it did not open (a resumed one); ``proof_stop_ids`` / ``remote_stop_ids``: the batcher's
    stop set and a remote proof worker's (``episode_stop_set_refusal``, checked when given)."""
    from reliquary_sandbox.attest import Ed25519TokenVerifier, Signer, load_private_key

    from reliquary.infrastructure import sandbox_store
    from reliquary.sandbox import require_sandbox
    from reliquary.sandbox.fleet import Fleet, http_fetch_report
    from reliquary.sandbox.rl_routes import build_episode_precommit_router
    from reliquary.sandbox.routes import build_sandbox_sessions_router

    require_sandbox()
    contract = runtime.contract
    policies = {name: contract.episode_policy(name) for name in contract.episode_environments
                if name in environments}
    if not policies:
        raise ValueError("no episode environment of the order is loaded")
    renderers = {name: renderer_for(p) for name, p in policies.items()}
    if proof_stop_ids is not None:
        why = episode_stop_set_refusal({name: r.stop_ids for name, r in renderers.items()}, proof_stop_ids,
                                       remote_stop_ids)
        if why is not None:
            raise ValueError(f"signed episodes cannot be proven: {why}")
    policy = rl_sandbox_policy(config.policy or SandboxPolicy(), max(p.pool_seeds for p in policies.values()))
    signer = signer or Signer(config.key_id, load_private_key(config.key_file))
    token_verifier = Ed25519TokenVerifier({**config.retired_keys, signer.key_id: signer.public_key_b64})
    kw = dict(store_kwargs or {})

    async def documents():
        return await sandbox_store.list_machines(**kw)

    async def record(machine_id, *, at, summary):
        return await sandbox_store.record_machine_heartbeat(machine_id, at=at, summary=summary, **kw)

    fleet = Fleet(read_documents=read_documents or documents, fetch_report=fetch_report or http_fetch_report,
                  clock=clock, record_heartbeat=record if read_documents is None else None,
                  directory_refresh_s=config.directory_refresh_s,
                  directory_max_age_s=config.directory_max_age_s,
                  directory_read_timeout_s=config.directory_read_timeout_s)
    book = SessionBook(policy)
    views = {name: EpisodeEnvironmentView(policy=p, resolve_task=environments[name].source.resolve)
             for name, p in policies.items()}
    engagements = RlEpisodeEngagements(precommits=runtime.episode_precommit, environments=views.get,
                                       current_window=current_window, book=book,
                                       recorded_at=runtime.episode_precommit_recorded_at,
                                       quota=quota or OpenQuota(), clock=clock,
                                       window_started_at=window_started_at)
    issuer = SessionIssuer(
        book=book,
        store=session_store or sandbox_store.R2SessionStore(prefix=sandbox_store.RL_SESSION_PREFIX, **kw),
        fleet=fleet, signer=signer, token_verifier=token_verifier,
        engagements={"rl_precommit": engagements}, policy=policy, clock=clock)
    fleet.on_drained = issuer.void_machine
    sessions_router = build_sandbox_sessions_router(
        issuer, policy=policy, validator_hotkey=validator_hotkey, prefix=RL_PREFIX, registration=registration,
        clock=clock, max_concurrent_closes=config.close_concurrency,
        close_body_timeout_s=config.close_body_timeout_s, max_preauth_closes=config.close_preauth_concurrency)
    precommit_router = build_episode_precommit_router(
        record=runtime.record_episode_precommit, current_window=current_window,
        validator_hotkey=validator_hotkey, policy=policy, prefix=RL_PREFIX, registration=registration,
        clock=clock)
    checkers = {name: EpisodeGroupChecker(policy=p, renderer=renderers[name], source=environments[name].source,
                                          chunk_tokens=chunk_tokens)
                for name, p in policies.items()}
    outcomes = outcomes or NoOutcomes()
    window_open = (None if window_started_at is None
                   else (lambda window: isinstance(window, int) and window_started_at(window) is not None))
    intake = EpisodeGroupIntake(checkers=checkers, precommits=runtime.episode_precommit,
                                directory=fleet.directory_if_ready, token_verifier=token_verifier,
                                sessions=issuer, seen=book.submitted_ids, outcomes=outcomes,
                                window_open=window_open)
    logger.info("rl episode services built for %s (validator key %s)", sorted(policies), signer.key_id)
    return RlEpisodeServices(signer=signer, token_verifier=token_verifier, fleet=fleet, book=book,
                             issuer=issuer, routers=(sessions_router, precommit_router), intake=intake,
                             environments=tuple(sorted(policies)), runtime=runtime, outcomes=outcomes,
                             clock=clock, precommit_retention_s=float(precommit_retention_s))


__all__ = ["MIN_PRECOMMIT_RETENTION_S", "RL_GROUPS_IN_FLIGHT", "RL_PREFIX", "RlEpisodeServices",
           "build_rl_episode_services", "episode_stop_set_refusal", "rl_registration", "rl_sandbox_policy",
           "stop_ids_from_metadata"]
