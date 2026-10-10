"""The RL validator's signed-sandbox side: the validator's token key, the machine
directory and fleet, the RL session book and issuer (sessions stored under their own prefix), the session
and precommit routes under ``/rl``, and the episode intake (one checker per episode env). Built only when
the order has signed-episode envs; it imports reliquary-sandbox, which a validator without such an env
never loads.

Settings: the corpus validator's (``sandbox_wiring.SandboxValidatorConfig``). The RL policy is raised to
what one hotkey's groups in flight need (``rl_sandbox_policy``); the session quota (``SessionQuota``)
decides beyond that. With a yield ledger, the quota is the yield quota (``rl_quota.YieldQuota``: the fleet's
session slots shared by weighted water-filling) and the policy's per-hotkey live and hourly-open caps are
lifted (``NO_PER_KEY_CAP``); the daily abort cap keeps its size.

Windows. Episode intake (precommits, opens, groups) is open only for a window THIS process opened
(``window_started_at``): a window resumed after a restart takes no episode group, since what its batcher
accepted before lived in memory only. A window older than ``FILL_CLOSED_MAX_SECONDS`` + margin opens no
session.

Hand-back. A paid group the proof could not judge for a reason of the validator's own is handed back by
its batcher (``episode_proof_inconclusive``, possibly on the event loop inside the seal or on a worker
thread): ``hand_back`` only schedules the work on the loop and returns; the issuer then marks the group's
sessions handed back in its book and its store (still ``submitted``: no same-window retry), and the yield
accounting hears of it."""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from reliquary.protocol.service_episode import GROUPS_IN_FLIGHT
from reliquary.sandbox.rl_engagements import (
    EpisodeEnvironmentView, NoOutcomes, OpenQuota, RlEpisodeEngagements, SessionOutcomes, SessionQuota,
)
from reliquary.sandbox.rl_yield import held_seconds
from reliquary.sandbox.sessions import FanOutEvents, SandboxPolicy, SessionBook, SessionIssuer
from reliquary.validator.corpus_registration import NOT_REGISTERED
from reliquary.validator.episode_admission import EpisodeGroupChecker
from reliquary.validator.episode_intake import EpisodeGroupIntake

logger = logging.getLogger(__name__)

RL_PREFIX = "/rl"
SESSION_MAINTAIN_SECONDS = 60.0
RESTORE_ATTEMPTS = 3
RESTORE_TIMEOUT_S = 60.0
RESTORE_BACKOFF_S = 5.0
STOP_TIMEOUT_S = 10.0
RL_GROUPS_IN_FLIGHT = GROUPS_IN_FLIGHT     # the intake's per-operator check cap
PRECOMMIT_PRUNE_SECONDS = 3600.0
MIN_PRECOMMIT_RETENTION_S = 2 * 86_400
YIELD_FLUSH_SECONDS = 10.0
NO_PER_KEY_CAP = 1 << 30
"""The policy's per-hotkey live and hourly-open caps when the yield quota decides: no fixed per-key maximum."""


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
    reported metadata (plain dicts, nested ``text_config`` included) and the local tokenizer.

    Config only: the worker reports its config and generation config, not its tokenizer's
    eos, so the comparison with the batcher's set (whose proxies carry the same metadata) checks the
    configs; the real model's nested ``text_config`` eos is to be checked in remote mode before deploy."""
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
    ledger: Any = field(default=None, repr=False)
    quota: Any = field(default=None, repr=False)
    window_uses: Callable[[int], Any] | None = field(default=None, repr=False)
    policies: dict = field(default_factory=dict, repr=False)
    window_open: Callable[[int], bool] | None = field(default=None, repr=False)
    current_window: Callable[[], int | None] | None = field(default=None, repr=False)
    operator_of: Callable[[str], str | None] | None = field(default=None, repr=False)
    resumed_windows: set = field(default_factory=set, repr=False)
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
        if self.ledger is not None:
            self.ledger.reconcile(self.book.records())
            rebuild = getattr(self.quota, "rebuild", None)
            if rebuild is not None:
                rebuild(self.book.records())                 # the quota's index, from the restored book
            pending = self.ledger.pending_windows()
            if (pending and self.window_open is not None and not self.window_open(pending[-1])
                    and self.ledger.accepting_at_last_flush() == pending[-1]):
                # The window the restart interrupted (its groups are no longer taken) is neutral for its
                # unused precommits; no stage a miner can send says so. Only when that window was still
                # accepting groups at the crashed process's last flush (at most YIELD_FLUSH_SECONDS before the
                # crash), never a window that had already ended.
                self.resumed_windows = {pending[-1]}
                logger.warning("rl session yield: window %d was interrupted by the restart; its unused "
                               "precommits count for no one", pending[-1])
            await asyncio.to_thread(self.ledger.flush)

    def background(self) -> list:
        jobs = [self.fleet.run(), self.issuer.maintain_forever(SESSION_MAINTAIN_SECONDS),
                self.prune_precommits_forever(PRECOMMIT_PRUNE_SECONDS)]
        if self.ledger is not None:
            jobs.append(self.yield_forever(YIELD_FLUSH_SECONDS))
        return jobs

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
        if self.ledger is not None:
            try:
                await self.settle_yield()
            except Exception:
                logger.exception("rl session yield not flushed at shutdown")
            # After the last flush (a cancelled background flush still running in its thread holds the
            # ledger's flush lock: close waits for it).
            await asyncio.to_thread(self.ledger.close)

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

    # --- the yield ledger ---
    async def settle_yield(self) -> int:
        """Decide the precommits of every settled window (on the loop: the book is the loop's; the
        runtime is read in a worker thread), then write the journal in a worker thread. Returns how many
        sessions were decided."""
        ledger = self.ledger
        if ledger is None:
            return 0
        now = float(self.clock())
        decided = 0
        try:
            decided = await self._decide_settled(ledger, now)
        finally:
            # Whatever the decisions raised, the journal (verdicts, opens) is written and the snapshot refreshed.
            await asyncio.to_thread(ledger.flush, now)
        return decided

    async def _decide_settled(self, ledger, now: float) -> int:
        decided = 0
        if self.current_window is not None:
            ledger.note_accepting(self.current_window(), now)          # durable at this flush
        sizes = {p.sandbox_env: p.pool_seeds // 2 for p in self.policies.values()}     # M per sandbox env
        reported = getattr(self.fleet, "session_reported", None)
        if self.window_uses is not None:
            held = self.book.precommits_held(int(now))
            for window in ledger.pending_windows():
                uses = await asyncio.to_thread(self.window_uses, window)
                if uses is None:
                    continue                                   # not settled yet
                used = set()
                for environment, prompt_idx, hotkey in uses.used:
                    policy = self.policies.get(environment)
                    if policy is not None:
                        used.add((hotkey, policy.sandbox_env, policy.split, int(prompt_idx)))
                for sha in ledger.precommits_of(window):
                    if sha in held:
                        continue
                    rows = [(r.session_id, r.state, r.closed_status, getattr(r, "fault", None),
                             held_seconds(r, now, presented=_presented(reported, r.session_id)))
                            for r in self.book.of_precommit(sha)]
                    task = ledger.task_of(sha)
                    decided += ledger.decide(sha, rows, used=task in used,
                                             aborted=bool(uses.aborted), now=now,
                                             resumed=window in self.resumed_windows,
                                             group_size=None if task is None else sizes.get(task[1]))
        return decided

    async def yield_forever(self, every_s: float) -> None:
        """Settle and flush every ``every_s`` seconds: a verdict is durable at most that long after it."""
        while True:
            try:
                await self.settle_yield()
            except Exception:
                logger.exception("rl session yield not settled")
            await asyncio.sleep(every_s)


def _presented(reported: Callable[[str], bool] | None, session_id: str) -> bool | None:
    """True when a machine report listed the session (its box ran: its held seconds count even without a
    machine final); None otherwise, so ``held_seconds`` falls back to the record's own final."""
    if not callable(reported):
        return None
    try:
        return True if reported(session_id) else None
    except Exception:
        logger.exception("the fleet could not tell whether session %s was reported", session_id)
        return None


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
    clock: Callable[[], float] = time.time, open_body_timeout_s: float | None = None,
    max_preauth_opens: int | None = None,
    yield_path=None, ledger=None, operator_of: Callable[[str], str | None] | None = None,
    window_uses: Callable[[int], Any] | None = None,
) -> RlEpisodeServices:
    """``environments``: the order's episode envs as loaded (``SignedEpisodeEnvironment``; their
    ``source`` is the env's task source); ``renderer_for(policy)``: the turn renderer of the policy's tokenizer over
    the policy's tools; ``current_window()``: the service window admissions are open for (memory only:
    it is read on the event loop); ``window_started_at(window)``: when this process opened it, None for
    a window it did not open (a resumed one); ``proof_stop_ids`` / ``remote_stop_ids``: the batcher's
    stop set and a remote proof worker's (``episode_stop_set_refusal``, checked when given).

    ``yield_path``: the yield ledger's SQLite file (None: no ledger, sessions carry no fault and no ledger
    hears of them), opened here, on the caller's thread; ``ledger``: an already opened ledger instead (the
    validator opens it in a worker thread); ``operator_of(hotkey)``: the chain operator, read only to tell
    whether a saturation was the refused miner's own doing; ``window_uses``: ``ServiceRuntime.window_uses``
    (blocking; called in a worker thread)."""
    from reliquary_sandbox.attest import Ed25519TokenVerifier, Signer, load_private_key

    from reliquary.infrastructure import sandbox_store
    from reliquary.sandbox import require_sandbox
    from reliquary.sandbox.fleet import Fleet, http_fetch_report
    from reliquary.sandbox.rl_yield import LedgerEvents, SessionYieldLedger
    from reliquary.sandbox.rl_routes import BODY_TIMEOUT_S, MAX_PREAUTH, build_episode_precommit_router
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
    from reliquary.sandbox.rl_quota import YieldQuota

    pool_seeds = max(p.pool_seeds for p in policies.values())
    policy = rl_sandbox_policy(config.policy or SandboxPolicy(), pool_seeds)
    yield_quota = (yield_path is not None or ledger is not None) and (quota is None or isinstance(quota, YieldQuota))
    if yield_quota:
        # The yield quota decides how many sessions a hotkey holds and opens (no fixed per-key maximum); the
        # daily abort cap keeps its size: an abort is neutral and frees its seed.
        policy = dataclasses.replace(policy, max_live_per_hotkey=NO_PER_KEY_CAP, max_opens_per_hour=NO_PER_KEY_CAP)
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
    if yield_path is not None and ledger is not None:
        raise ValueError("pass the yield ledger's path or the ledger, not both")
    book = SessionBook(policy)
    if ledger is None and yield_path is not None:
        ledger = SessionYieldLedger(yield_path, clock=clock)
    if yield_quota and quota is None:
        # M by every name an env goes by: the env name an open passes and the sandbox env a session stores.
        group_sizes = {p.sandbox_env: p.pool_seeds // 2 for p in policies.values()}
        group_sizes.update({name: p.pool_seeds // 2 for name, p in policies.items()})
        quota = YieldQuota(ledger=ledger, pool_seeds=pool_seeds,
                           group_size=min(p.pool_seeds for p in policies.values()) // 2,
                           capacity=fleet.session_capacity, reported=getattr(fleet, "session_reported", None),
                           group_sizes=group_sizes, retry_after_s=policy.retry_after_s, clock=clock)
    if ledger is not None and isinstance(quota, YieldQuota):
        ledger.allow_capacity_verdicts(quota.groups_allowed)      # capacity verdicts: per hotkey and window
        book.cap_scale = quota.cap_scale                          # the per-hotkey rates scale with the allocation
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
        engagements={"rl_precommit": engagements}, policy=policy, clock=clock,
        events=None if ledger is None else FanOutEvents(
            LedgerEvents(ledger), quota if isinstance(quota, YieldQuota) else None),
        record_faults=ledger is not None,
        open_refusal_check=None if ledger is None else fleet.open_refusal_valid)
    fleet.on_drained = issuer.void_machine
    sessions_router = build_sandbox_sessions_router(
        issuer, policy=policy, validator_hotkey=validator_hotkey, prefix=RL_PREFIX, registration=registration,
        clock=clock, max_concurrent_closes=config.close_concurrency,
        close_body_timeout_s=config.close_body_timeout_s, max_preauth_closes=config.close_preauth_concurrency,
        # the opens are bounded like the precommits
        open_body_timeout_s=BODY_TIMEOUT_S if open_body_timeout_s is None else open_body_timeout_s,
        max_preauth_opens=MAX_PREAUTH if max_preauth_opens is None else max_preauth_opens)
    precommit_router = build_episode_precommit_router(
        record=runtime.record_episode_precommit, current_window=current_window,
        validator_hotkey=validator_hotkey, policy=policy, prefix=RL_PREFIX, registration=registration,
        clock=clock)
    checkers = {name: EpisodeGroupChecker(policy=p, renderer=renderers[name], source=environments[name].source,
                                          chunk_tokens=chunk_tokens)
                for name, p in policies.items()}
    outcomes = outcomes or ledger or NoOutcomes()
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
                             clock=clock, precommit_retention_s=float(precommit_retention_s),
                             ledger=ledger, quota=quota, window_uses=window_uses, policies=dict(policies),
                             window_open=window_open, current_window=current_window, operator_of=operator_of)


__all__ = ["MIN_PRECOMMIT_RETENTION_S", "NO_PER_KEY_CAP", "RL_GROUPS_IN_FLIGHT", "RL_PREFIX", "YIELD_FLUSH_SECONDS", "RlEpisodeServices",
           "build_rl_episode_services", "episode_stop_set_refusal", "rl_registration", "rl_sandbox_policy",
           "stop_ids_from_metadata"]
