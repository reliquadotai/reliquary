"""The RL validator's signed-sandbox side: the validator's token key, the machine
directory and fleet, the RL session book and issuer (sessions stored under their own prefix), the session
and precommit routes under ``/rl``, and the episode intake (one checker per episode env). Built only when
the order has signed-episode envs; it imports reliquary-sandbox, which a validator without such an env
never loads.

Settings: the corpus validator's (``sandbox_wiring.SandboxValidatorConfig``). The RL policy is raised to
what one hotkey's groups in flight need (``rl_sandbox_policy``); the session quota (``SessionQuota``)
decides beyond that. With a yield ledger, the quota is the yield quota (``rl_quota.YieldQuota``: the fleet's
session slots shared by weighted water-filling) and the policy's per-hotkey live and hourly-open caps are
lifted (``NO_PER_KEY_CAP``); the daily abort cap and the refused-open rate are not lifted: they scale with the
hotkey's allocation (``SessionBook.cap_scale``, 1 for a newcomer).

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
from reliquary.sandbox.sessions import FanOutEvents, Refusal, SandboxPolicy, SessionBook, SessionIssuer
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
RL_GROUPS_IN_FLIGHT = GROUPS_IN_FLIGHT     # the intake's per-operator check cap; with the yield quota the cap is the hotkey's submittable groups
PRECOMMIT_PRUNE_SECONDS = 3600.0
MIN_PRECOMMIT_RETENTION_S = 2 * 86_400
YIELD_FLUSH_SECONDS = 10.0
BOOT_FLEET_WAIT_S = 120.0
"""At start, how long the validator waits for a machine serving each episode env before starting it degraded."""
POLL_SECONDS_FOR_SERVED = 5.0
OPTIONS_PROBE_TIMEOUT_S = 30.0
"""At start, how long resolving an env's first task to learn its options digest may take."""
"""How often a degraded env is re-checked (the fleet's poll period; kept here: no fleet import at module level)."""
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


def episode_group_sizes(policies: Mapping[str, Any]) -> dict[str, int]:
    """Each episode env's group size M (half its seed pool), by every name the env goes by: the order's env name
    (what an open names) and its sandbox env (what a session record stores). A name with two sizes is refused."""
    sizes: dict[str, int] = {}
    for name, policy in policies.items():
        for key in (name, policy.sandbox_env):
            if sizes.setdefault(key, policy.pool_seeds // 2) != policy.pool_seeds // 2:
                raise ValueError(f"episode env name {key!r} has two group sizes; the yield quota needs one")
    return sizes


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
    queue: Any = field(default=None, repr=False)
    window_uses: Callable[[int], Any] | None = field(default=None, repr=False)
    policies: dict = field(default_factory=dict, repr=False)
    window_open: Callable[[int], bool] | None = field(default=None, repr=False)
    current_window: Callable[[], int | None] | None = field(default=None, repr=False)
    operator_of: Callable[[str], str | None] | None = field(default=None, repr=False)
    resumed_windows: set = field(default_factory=set, repr=False)
    restore_attempts: int = RESTORE_ATTEMPTS
    restore_timeout_s: float = RESTORE_TIMEOUT_S
    restore_backoff_s: float = RESTORE_BACKOFF_S
    served: dict = field(default_factory=dict, repr=False)
    resolvers: dict = field(default_factory=dict, repr=False)
    boot_wait_s: float | None = None
    unserved_envs: set = field(default_factory=set)
    views: dict = field(default_factory=dict, repr=False)
    saturation: Any = field(default=None, repr=False)
    tiers: Any = field(default=None, repr=False)
    task_counts: dict = field(default_factory=dict, repr=False)     # env -> its tasks (for seeding the tiers)
    raw_resolvers: dict = field(default_factory=dict, repr=False)   # env -> its source's resolve (unobserved)
    saturation_cache: Any = field(default=None, repr=False)         # the last snapshot, computed off the request
    # What an env added to the running run is checked and served with (``environment_refusal``, ``add_environment``)
    renderer_for: Any = field(default=None, repr=False)
    proof_stop_ids: Any = field(default=None, repr=False)
    remote_stop_ids: Any = field(default=None, repr=False)
    chunk_tokens: int = 0
    pool_seeds: int = 0                  # the seed pool the session policy's caps and the quota are sized for
    pool_sizes: dict = field(default_factory=dict, repr=False)      # env -> its seed pool (the allocation answer)
    renderers: dict = field(default_factory=dict, repr=False)       # tools -> turn renderer (built off the loop)
    active_envs: frozenset | None = None     # the current window's envs (``set_active_envs``); None: every env
    tier_pending: frozenset = frozenset()    # added envs whose reservation tier is not seeded yet
    _loop: Any = field(default=None, repr=False)
    _tasks: set = field(default_factory=set, repr=False)

    async def start(self) -> None:
        """Before serving: the directory read once (the fleet retries on its own), then the RL sessions
        restored (bounded, retried). A restore that still fails raises: a validator that lost its sessions
        would hand a taken seed out again. With ``boot_wait_s``, the episode envs no machine serves after that
        wait start degraded (``require_served``): the validator starts anyway."""
        self._loop = asyncio.get_running_loop()
        await self.fleet.refresh_if_due()
        if self.tiers is not None:
            await self.seed_tiers()
        if self.boot_wait_s is not None:
            await self.require_served(self.boot_wait_s)
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
        if self.boot_wait_s is not None:
            jobs.append(self.watch_served_forever(POLL_SECONDS_FOR_SERVED))
        if self.saturation is not None:
            from reliquary.sandbox.rl_capacity import SATURATION_CHECK_S

            jobs.append(self.saturation_forever(SATURATION_CHECK_S))
        return jobs

    # --- the operator's saturation view and ALERTs ---
    async def seed_tiers(self, names: Collection[str] | None = None) -> None:
        """Seed each env's per-session reservation from a sample of its own tasks (``SessionTiers.seed``): at
        most ``TIER_SEED_TASKS`` spread evenly over its index range, within ``TIER_SEED_TIMEOUT_S``. An env whose
        tasks cannot be counted or resolved stays learnt from what is resolved (logged). Never raises.
        ``names``: only these envs (an env added to the running run); None: every env."""
        from reliquary.sandbox.rl_capacity import TIER_SEED_TASKS, TIER_SEED_TIMEOUT_S

        for name in sorted(self.policies if names is None else set(names) & set(self.policies)):
            count, resolve = self.task_counts.get(name), self.raw_resolvers.get(name)
            if not count or resolve is None:
                logger.warning("episode env %s: its task count is unknown; its reservation tier is learnt from the "
                               "tasks miners resolve", name)
                continue
            indices = (range(count) if count <= TIER_SEED_TASKS
                       else sorted({i * count // TIER_SEED_TASKS for i in range(TIER_SEED_TASKS)}))
            tasks: list = []
            deadline = time.monotonic() + TIER_SEED_TIMEOUT_S

            async def sample(indices=indices, tasks=tasks, resolve=resolve, name=name) -> None:
                failed = 0
                for index in indices:
                    try:
                        tasks.append(await resolve(int(index)))
                    except (LookupError, ValueError, OSError, RuntimeError):
                        failed += 1                      # a task the source cannot resolve: not sampled
                if failed:
                    logger.warning("episode env %s: %d sampled tasks could not be resolved", name, failed)
            try:
                await asyncio.wait_for(sample(), max(0.0, deadline - time.monotonic()))
            except TimeoutError:
                logger.warning("episode env %s: %d tasks sampled for its reservation tier before the timeout",
                               name, len(tasks))
            except Exception:
                logger.exception("episode env %s: its reservation tier was not seeded", name)
                continue
            self.tiers.seed(name, tasks)
            logger.info("episode env %s: sessions counted at %s (from %d tasks)", name,
                        self.tiers.reservation(name), len(tasks))

    def reservations(self) -> dict | None:
        """{(sandbox env, package): reservation}: what the fleet size is counted in (``counted_envs`` only)."""
        return None if self.tiers is None else self.tiers.reservations(self.counted_envs())

    def free_fraction(self) -> float | None:
        """The fleet's free sessions over its sessions (None when it has none)."""
        sessions, free = self.fleet.session_room(reservations=self.reservations())
        return None if sessions <= 0 else free / sessions

    def saturation_snapshot(self) -> dict:
        """Computed by ``saturation_forever`` every ``SATURATION_CHECK_S`` (never on a request)."""
        snapshot = self.saturation.snapshot(queue=self.queue, quota=self.quota, fleet=self.fleet,
                                            reservations=self.reservations(), ledger=self.ledger, issuer=self.issuer)
        if self.tiers is not None:
            snapshot["reservations"] = self.tiers.view()
        snapshot["at"] = float(self.clock())
        return snapshot

    def cached_saturation(self) -> dict | None:
        """The last snapshot ``saturation_forever`` computed (None before the first)."""
        return self.saturation_cache

    def check_saturation(self) -> list[str]:
        """One round of the operator's ALERTs (the grow-fleet one, the check slots, envs that lost every
        machine, capacity verdicts per hotkey); returns their texts (each is logged)."""
        from reliquary.sandbox.rl_capacity import MAX_CHECKS_RUNNING

        monitor, alerts = self.saturation, []
        reservations = self.reservations()
        envs = (self.fleet.capacity_by_env() if reservations is None
                else self.fleet.capacity_by_env(reservations=reservations))
        alerts.extend(monitor.check_saturated(envs, images_of=getattr(self.fleet, "image_set_images", None)))
        checker_alert = monitor.check_checker(running=MAX_CHECKS_RUNNING)
        if checker_alert:
            alerts.append(checker_alert)
        if self.fleet.directory_ready():                # a stale directory is the fleet's own ALERT, not every env's
            served = {name: bool(self.fleet.serving(sandbox_env, package, env_options_sha256=options))
                      for name, (sandbox_env, package, options) in self.served.items()}
            lost, recovered = monitor.check_envs(served, known_lost=self.unserved_envs)
            alerts.extend(lost)
            for name in recovered:
                recover = getattr(self.issuer, "unserved_recovered", None)
                if recover is not None:
                    recover(self.served[name][0])      # its next loss is ALERTed again at the open path too
        if self.ledger is not None:
            alerts.extend(monitor.check_capacity_verdicts(self.ledger))
        return alerts

    async def saturation_forever(self, every_s: float) -> None:
        while True:
            try:
                self.check_saturation()
            except Exception:
                logger.exception("the rl saturation check failed")
            try:
                self.saturation_cache = self.saturation_snapshot()
            except Exception:
                logger.exception("the rl saturation snapshot failed")
            await asyncio.sleep(every_s)

    # --- episode envs no machine serves: a degraded start, never a refusal to start ---
    def unserved(self) -> dict[str, tuple[str, str, str | None]]:
        """The episode envs no fresh active machine serves at their pinned identity (and options)."""
        return {name: served for name, served in self.served.items()
                if not self.fleet.serving(served[0], served[1], env_options_sha256=served[2])}

    def _alert_unserved(self, missing: Mapping[str, tuple[str, str, str | None]]) -> None:
        def reported(sandbox_env: str, package: str) -> str:
            packages = self.fleet.reported_packages(sandbox_env)
            text = ", ".join(packages) or "nothing"
            if package in packages:
                text += (" (a machine names it but is stale, drained, publishes other options or no valid "
                         "resources)")
            return text

        details = "; ".join(
            f"{name}: the order pins {package} for {sandbox_env}, machines report {reported(sandbox_env, package)}"
            for name, (sandbox_env, package, _options) in sorted(missing.items()))
        logger.error("ALERT no sandbox machine serves these episode environments: %s; they are refused "
                     "(environment_unserved) until a machine serves them", details)

    async def require_served(self, timeout_s: float) -> None:
        """At start: poll the machines until every episode env is served, for at most ``timeout_s``; the envs
        still unserved then start degraded (refused retryably, an ALERT). Never raises."""
        deadline = time.monotonic() + float(timeout_s)
        await self._learn_options()
        missing: dict = {}
        while True:
            try:
                await self.fleet.poll_once()
                missing = self.unserved()
            except Exception:
                logger.exception("the served check failed at start")
                missing = dict(self.served)
            if not missing or time.monotonic() >= deadline:
                break
            await asyncio.sleep(POLL_SECONDS_FOR_SERVED)
        self.unserved_envs = set(missing)
        if missing:
            self._alert_unserved(missing)

    async def _learn_options(self) -> None:
        """The options digest each env's tasks are resolved with (its first task's, as every open will carry it),
        for the envs whose source does not state one: the boot check then compares options too. A task that
        cannot be resolved now leaves the digest unknown (logged): each open still checks its own task's."""
        for name, (sandbox_env, package, options) in sorted(self.served.items()):
            resolve = self.resolvers.get(name)
            if options is not None or resolve is None:
                continue
            try:
                task = await asyncio.wait_for(resolve(0), OPTIONS_PROBE_TIMEOUT_S)
            except Exception as exc:
                logger.warning("episode env %s: its options digest is unknown at start (%s): the boot check "
                               "compares its package only", name, type(exc).__name__)
                continue
            digest = getattr(task, "env_options_sha256", None)
            if digest is not None:
                self.served[name] = (sandbox_env, package, str(digest))

    def serving_refusal(self, environment: str) -> Refusal | None:
        """``environment_unserved`` (retryable) for a degraded env; None otherwise."""
        if environment in self.unserved_envs:
            return Refusal("environment_unserved", {"environment": environment},
                           retry_after=self.book.policy.retry_after_s)
        return None

    async def precommit_admission(self, value) -> tuple[str, dict] | None:
        """Before a precommit is recorded: its env must be served, the fleet view ready, and its task servable
        on at least ``MIN_SERVING_MACHINES`` machines by the open path's own filter, which together can host
        its group (M sessions) at once. ``environment_unserved`` and ``fleet_unavailable`` are retryable (503);
        ``task_unservable`` (409) only when fresh machines serve the env and the open path would refuse this
        task on all of them. A task the source cannot resolve is left to the runtime's own checks. Nothing is
        recorded on a refusal, so none is waste or quota. Never raises."""
        try:
            return await self._precommit_admission(value)
        except Exception as exc:
            logger.warning("episode precommit admission failed (%s): refused retryably", type(exc).__name__)
            return "fleet_unavailable", {"environment": str(getattr(value, "environment", ""))}

    async def _precommit_admission(self, value) -> tuple[str, dict] | None:
        from reliquary.sandbox.rl_capacity import MIN_SERVING_MACHINES
        from reliquary.sandbox.rl_engagements import session_budgets
        from reliquary.sandbox.sessions import token_validity_s

        environment = str(value.environment)
        refusal = self.serving_refusal(environment)
        if refusal is not None:
            return refusal.reason, dict(refusal.detail)
        view = self.views.get(environment)
        if view is None:
            return None                                # not an episode env here: the runtime refuses it
        if environment in self.tier_pending:
            # Added to the run, its reservation tier not seeded yet: not counted in the fleet size, no group yet.
            return "fleet_unavailable", {"environment": environment, "why": "tier_unseeded"}
        try:
            task = await asyncio.wait_for(view.resolve_task(int(value.task_index)), self.book.policy.io_timeout_s)
        except TimeoutError:
            # The source is slow now: never admitted unchecked, never a final answer.
            return "fleet_unavailable", {"environment": environment, "why": "task_resolve_timeout"}
        except Exception:
            return None                                # a task the source has not: the runtime's own checks
        policy = view.policy
        options = getattr(task, "env_options_sha256", None)
        if not self.fleet.serving(policy.sandbox_env, policy.env_package, env_options_sha256=options):
            # A stale directory or report, or no machine serving the env right now: never a final answer.
            return "fleet_unavailable", {"environment": environment}
        budgets = session_budgets(policy, task)
        reserve = {"cpu_millis": getattr(task, "cpu_millis", None), "memory_bytes": int(budgets["memory_bytes"])}
        placement = dict(image=task.image, env=policy.sandbox_env, env_package=policy.env_package, budgets=budgets,
                         # the open path's own token validity (its machines' caps must allow it)
                         validity_s=token_validity_s(self.book.policy.open_window_s, int(budgets["wall_s"]),
                                                     getattr(task, "prepare_timeout_s", None)),
                         env_options_sha256=options, reserve=reserve)
        lock = getattr(task, "env_lock_sha256", None)
        if lock is not None:                           # the open path's own lock filter (``_reserving``)
            placement["env_lock_sha256"] = lock
        group = policy.pool_seeds // 2
        machines = self.fleet.servable(**placement, group_size=group)     # M at once
        if machines >= MIN_SERVING_MACHINES:
            return None
        verdict = self.fleet.could_place(**placement, now=self.clock())
        if verdict is None:
            # A machine that served it when last heard is stale or drained, or one's resources cannot be read:
            # the open path itself would answer retryably.
            return "fleet_unavailable", {"environment": environment}
        if verdict and self.fleet.servable(**placement, group_size=group, include_unknown=True) >= MIN_SERVING_MACHINES:
            # Its fresh machines alone cannot host its group, but with the ones missing a heartbeat (or whose
            # resources cannot be read now) they could: a partial view never gives a final answer.
            return "fleet_unavailable", {"environment": environment}
        return "task_unservable", {"environment": environment, "task_index": int(value.task_index),
                                   "image": task.image, "cpu_millis": reserve["cpu_millis"], "machines": machines}

    async def watch_served_forever(self, every_s: float) -> None:
        """While an env is degraded: re-check every ``every_s``; serve it again once a machine reports it."""
        while True:
            try:
                if self.unserved_envs:
                    missing = self.unserved()
                    recovered = self.unserved_envs - set(missing)
                    if recovered:
                        logger.warning("episode environments served again: %s", ", ".join(sorted(recovered)))
                    self.unserved_envs = self.unserved_envs & set(missing)
            except Exception:
                logger.exception("the served check failed")
            await asyncio.sleep(every_s)

    async def stop(self, timeout: float = STOP_TIMEOUT_S) -> None:
        pending = [task for task in self._tasks if not task.done()]
        if pending:
            _, late = await asyncio.wait(pending, timeout=timeout)
            for task in late:
                task.cancel()
            if late:
                await asyncio.gather(*late, return_exceptions=True)
                logger.error("ALERT %d episode hand-backs cut at shutdown", len(late))
        # A drain that raises still leaves the yield journal flushed and the ledger closed; its error is raised
        # after them, and wins over a close that fails too (that one is logged).
        drained: BaseException | None = None
        try:
            await self.issuer.drain(timeout)
        except BaseException as exc:
            drained = exc
        if self.ledger is not None:
            try:
                await self.settle_yield()
            except Exception:
                logger.exception("rl session yield not flushed at shutdown")
            # After the last flush (a cancelled background flush still running in its thread holds the
            # ledger's flush lock: close waits for it).
            try:
                await asyncio.to_thread(self.ledger.close)
            except Exception:
                if drained is None:
                    raise
                logger.exception("rl session yield ledger not closed at shutdown")
        if drained is not None:
            raise drained

    # --- envs added to the running run ---
    def _policy_refusal(self, name: str, policy) -> str | None:
        """Why the session policy, the quota or the tiers could not take an env ``name`` with ``policy``: a seed
        pool larger than the one they were sized for, or one of its names (its name, its sandbox env) already
        known with another group size or for another env. Memory only."""
        if self.pool_seeds and int(policy.pool_seeds) > self.pool_seeds:
            return (f"its seed pool {policy.pool_seeds} is larger than the {self.pool_seeds} the RL session "
                    f"policy and the quota were sized for at start")
        try:
            episode_group_sizes({**self.policies, name: policy})
        except ValueError as exc:
            return str(exc)
        known = dict(getattr(self.quota, "_env_names", None) or {})
        for key, canonical in _env_names(self.policies).items():
            known.setdefault(key, canonical)
        for key, canonical in _env_names({name: policy}).items():
            if known.get(key, canonical) != canonical:
                return f"episode env name {key!r} already names env {known[key]!r} (one name, one env)"
        return None

    def _renderer(self, policy) -> Any:
        """The turn renderer for ``policy``'s tools, built once per tool set (``renderer_for`` may load a
        tokenizer: never call this on the event loop for a tool set not built yet)."""
        key = tuple(policy.tools)
        renderer = self.renderers.get(key)
        if renderer is None:
            renderer = self.renderers[key] = self.renderer_for(policy)
        return renderer

    def _stop_refusal(self, policy, renderer) -> str | None:
        if self.proof_stop_ids is None:
            return None
        return episode_stop_set_refusal({policy.environment: renderer.stop_ids}, self.proof_stop_ids,
                                        self.remote_stop_ids)

    def environment_refusal(self, policy) -> str | None:
        """Why these services could not serve an episode env with ``policy`` (an env a schedule request adds): a
        seed pool larger than the one they were sized for, one of its names known with another group size or for
        another env, no renderer for its tools, or its renderer's stops outside the proof's stop set; None when
        they can. BLOCKING (it may build the renderer, loading a tokenizer): call it in a worker thread; the
        renderer it builds is kept for ``add_environment``. A transient error while building the renderer (a
        tokenizer download timing out; ``catalogue_activation.transient_error``) is raised, not answered: it is
        no verdict on the env."""
        why = self._policy_refusal(policy.environment, policy)
        if why is not None or self.renderer_for is None:
            return why
        try:
            renderer = self._renderer(policy)
        except Exception as exc:
            from reliquary.validator.catalogue_activation import transient_error

            if transient_error(exc):
                raise
            return f"no renderer for its tools {list(policy.tools)} ({type(exc).__name__}: {exc})"
        return self._stop_refusal(policy, renderer)

    def set_active_envs(self, names: Collection[str] | None) -> None:
        """The envs the current window serves (its frozen pools): only they count in the fleet size, the per-env
        caps, the saturation view and the allocation answer. None (never called): every env, as before."""
        self.active_envs = None if names is None else frozenset(str(n) for n in names)

    def counted_envs(self) -> frozenset[str] | None:
        """The envs whose sessions the fleet is counted in: the active ones, less any added env whose reservation
        tier is not settled yet. None: every env (nothing restricted)."""
        if self.active_envs is None and not self.tier_pending:
            return None
        names = set(self.policies) if self.active_envs is None else set(self.active_envs) & set(self.policies)
        return frozenset(names - self.tier_pending)

    def allocation_pool_sizes(self) -> dict[str, int]:
        """Each counted env's seed pool (what the allocation route tells a miner its groups in)."""
        counted = self.counted_envs()
        return {n: p for n, p in self.pool_sizes.items() if counted is None or n in counted}

    def _prepare_add(self, name: str, environment, policy, renderer=None) -> tuple:
        """Everything ``add_environment`` builds and checks before its first change: ValueError when the
        services cannot serve the env (nothing changed); an I/O error or a timeout of the source is raised as
        it is. Memory only, apart from what the source's ``len`` and ``options_sha256`` read."""
        current = getattr(self.runtime, "contract", None) if self.runtime is not None else None
        if current is not None and name in current.environments and current.episode_policy(name) != policy:
            raise ValueError(f"episode env {name}: the policy given is not the one of the runtime's schedule")
        why = self._policy_refusal(name, policy)
        if why is not None:
            raise ValueError(f"episode env {name} cannot be served: {why}")
        if renderer is None:
            renderer = self.renderers.get(tuple(policy.tools))
        if renderer is None:
            raise ValueError(f"episode env {name}: no renderer was built for its tools {list(policy.tools)} (check "
                             f"it with environment_refusal off the event loop first, or pass the renderer)")
        why = self._stop_refusal(policy, renderer)
        if why is not None:
            raise ValueError(f"episode env {name} cannot be served: {why}")
        source = environment.source
        try:
            task_count = _task_count(source)
            pin = (policy.sandbox_env, policy.env_package, getattr(source, "options_sha256", None))
            checker = EpisodeGroupChecker(policy=policy, renderer=renderer, source=source,
                                          chunk_tokens=self.chunk_tokens)
            resolve = source.resolve if self.tiers is None else _observing(self.tiers, name, source.resolve)
            view = EpisodeEnvironmentView(policy=policy, resolve_task=resolve)
        except Exception as exc:
            from reliquary.validator.catalogue_activation import transient_error

            if transient_error(exc):
                raise
            raise ValueError(f"episode env {name}: its task source cannot be served ({type(exc).__name__}: "
                             f"{exc})") from exc
        return renderer, task_count, pin, checker, resolve, view

    def add_check(self, name: str, environment, policy) -> str | None:
        """A dry run of ``add_environment`` (worker thread: it may build the env's turn renderer): why these
        services could not take the env, or None. Nothing changes. An I/O error or a timeout (a tokenizer
        download, the source's reads) is raised: no verdict on the env."""
        known = self.views.get(name)
        if known is not None:
            return None if known.policy == policy else f"episode env {name} is already served with another policy"
        why = self.environment_refusal(policy)
        if why is not None:
            return why
        try:
            self._prepare_add(name, environment, policy)
        except ValueError as exc:
            return str(exc)
        return None

    def add_environment(self, name: str, environment, policy, *, renderer=None) -> None:
        """Serve an env added to the running run, as one of the order's: its policy (the ledger's settlement
        translation, the reservation tiers, hence the fleet size and the saturation view), its served pin (the
        degraded and lost-machine checks), its names in the yield quota (M, the per-env cap), its seed pool (the
        allocation answer), its task source observed by the tiers, its checker to the intake, and last its view
        (``views``, read by the engagement book: opens follow). ``policy`` is the schedule's view's
        (``schedule.contract_view(order).episode_policy(name)``); once the runtime's view names the env, it must
        be that one. ``renderer``: built off the loop; else the one ``environment_refusal`` built for its tools
        (this method never builds one: it runs on the event loop). Idempotent. Raises ValueError, changing
        nothing, when the services cannot serve it.

        With the reservation tiers, the env counts in the fleet size only once its tier is seeded from its own
        tasks (in the background; ``tier_pending`` meanwhile), and its precommits are refused retryably
        (``fleet_unavailable``) until then: an unseeded tier is the whole box, which would shrink the count of
        every machine it shares. Whether the env takes groups, and in which window, is the runtime's: a
        precommit needs the env in its window's frozen pools, and a group is judged by its own window's frozen
        contract (a window frozen before the add refuses it)."""
        known = self.views.get(name)
        if known is not None:
            if known.policy != policy:
                raise ValueError(f"episode env {name} is already served with another policy")
            return
        renderer, task_count, pin, checker, resolve, view = self._prepare_add(name, environment, policy, renderer)
        source = environment.source
        add_names = getattr(self.quota, "add_environment", None)
        if add_names is not None:
            add_names(episode_group_sizes({name: policy}), _env_names({name: policy}))
        if self.tiers is not None:
            self.tier_pending = self.tier_pending | {name}     # not counted before its tier is seeded
        self.policies[name] = policy                     # in place: the tiers read this dict
        self.served[name] = pin
        self.pool_sizes[name] = int(policy.pool_seeds)
        self.resolvers[name] = resolve
        self.raw_resolvers[name] = source.resolve
        self.task_counts[name] = task_count
        self.intake.add_checker(name, checker)
        self.views[name] = view
        self.environments = tuple(sorted({*self.environments, name}))
        logger.warning("episode env %s added to the rl services (sandbox env %s, %s)", name, policy.sandbox_env,
                       policy.env_package)
        self._alert_if_unserved(name)
        if self.tiers is not None:
            if self._loop is not None and not self._loop.is_closed():
                task = self._loop.create_task(self._seed_added(name))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            else:
                logger.error("episode env %s: no event loop to seed its reservation tier; it is not counted in "
                             "the fleet size and takes no group", name)

    async def _seed_added(self, name: str) -> None:
        """Seed an added env's tier, then count it (seeded, or learnt when its tasks cannot be sampled: as at
        start). Never raises."""
        try:
            await self.seed_tiers([name])
        except Exception:
            logger.exception("episode env %s: its reservation tier was not seeded", name)
        finally:
            self.tier_pending = self.tier_pending - {name}

    def _alert_if_unserved(self, name: str) -> None:
        """An env added while no fresh machine serves it: the ALERT (and, while the degraded check runs, its
        precommits refused ``environment_unserved`` until a machine serves it), as at start."""
        try:
            if not self.fleet.directory_ready():
                return                                   # a stale directory is the fleet's own ALERT
            served = self.served[name]
            if self.fleet.serving(served[0], served[1], env_options_sha256=served[2]):
                return
            if self.boot_wait_s is not None:            # ``watch_served_forever`` serves it again once reported
                self.unserved_envs = self.unserved_envs | {name}
            self._alert_unserved({name: served})
        except Exception:
            logger.exception("the served check of added env %s failed", name)

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


def _observing(tiers, name: str, resolve: Callable) -> Callable:
    """``resolve`` that tells ``tiers`` of every task it resolves (the env's largest reservation tier)."""
    async def resolve_and_observe(index):
        task = await resolve(index)
        try:
            tiers.observe(name, task)
        except Exception:
            logger.exception("the reservation tier of %s was not updated", name)
        return task
    return resolve_and_observe


def _env_names(policies: Mapping[str, Any]) -> dict[str, str]:
    """Every name an episode env goes by (its order name, its sandbox env) -> its sandbox env."""
    names = {}
    for name, p in policies.items():
        names[name] = names[p.sandbox_env] = p.sandbox_env
    return names


def _task_count(source: Any) -> int | None:
    try:
        return len(source)
    except TypeError:                                   # a source that cannot count its tasks
        return None


RL_SATURATION_TOKEN_ENV = "RELIQUARY_RL_SATURATION_TOKEN"
OBSERVATIONS_TOKEN_ENV = "RELIQUARY_SERVICE_OBSERVATIONS_TOKEN"     # server.SERVICE_OBSERVATIONS_TOKEN_ENV
"""The operator's bearer token for ``GET /rl/saturation`` (the service observations' token when unset)."""


def build_saturation_router(snapshot: Callable[[], dict | None]):
    """``GET {RL_PREFIX}/saturation``: the operator's saturation view, ADMIN-ONLY like ``/service-observations``
    (it names hotkeys and shows live scarcity): ``Authorization: Bearer <token>`` (``RELIQUARY_RL_SATURATION_TOKEN``,
    else ``RELIQUARY_SERVICE_OBSERVATIONS_TOKEN``), compared in constant time; no token configured: 404 whatever the
    request; a missing or wrong one: 401. It serves the snapshot the background job computed last (503 before the
    first): a request computes nothing."""
    import hmac
    import os

    from fastapi import APIRouter, HTTPException, Request

    router = APIRouter()

    async def saturation(request: Request) -> dict:
        token = os.environ.get(RL_SATURATION_TOKEN_ENV, "") or os.environ.get(OBSERVATIONS_TOKEN_ENV, "")
        if not token:
            raise HTTPException(status_code=404, detail="not_found")
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
            raise HTTPException(status_code=401, detail="unauthorized")
        cached = snapshot()
        if cached is None:
            raise HTTPException(status_code=503, detail="not_ready", headers={"Retry-After": "10"})
        return cached

    # The annotations are strings here (postponed): the request type is given explicitly, as FastAPI resolves them
    # from the module's globals, where this lazily imported name is not.
    saturation.__annotations__ = {"request": Request, "return": dict}
    router.add_api_route(f"{RL_PREFIX}/saturation", saturation, methods=["GET"])
    return router


def rl_registration(server) -> Callable:
    """The sandbox routes' registration check, over the RL server's registration gate."""
    from reliquary.protocol.submission import RejectReason

    async def registration(hotkey: str) -> str | None:
        reason = await server._registration_reject_reason(hotkey)
        if reason is None:
            return None
        return NOT_REGISTERED if reason is RejectReason.HOTKEY_NOT_REGISTERED else "unavailable"

    return registration


def _check_caps(quota, clock: Callable[[], float]) -> dict:
    """With the yield quota, the intake's check caps: a hotkey checks only the groups it can submit now (as the
    session claim judges them), at most as many at once as there are; one operator at most
    ``MAX_CHECKS_RUNNING - 1`` at once (one slot always left to the others), on the RL intake's own
    ``MAX_CHECKS_RUNNING`` slots; and its per-minute check budget scales with its allocation. Without it, the
    intake's defaults (a cap per operator)."""
    from reliquary.sandbox.rl_capacity import MAX_CHECKS_RUNNING
    from reliquary.sandbox.rl_quota import YieldQuota

    if not isinstance(quota, YieldQuota):
        return {}
    return {"checks_allowed": lambda hotkey, received: quota.submittable(hotkey, int(clock()), received),
            "max_checks_running": MAX_CHECKS_RUNNING, "checks_scale": quota.cap_scale}


ALLOCATION_ANSWER_TTL_S = 5.0
"""How long one hotkey's answer of the allocation route is served from its snapshot before it is recomputed."""


def _allocation_of(quota, pool_seeds: Mapping[str, int] | Callable[[], Mapping[str, int]], *,
                   clock: Callable[[], float] = time.time,
                   ttl_s: float = ALLOCATION_ANSWER_TTL_S) -> Callable[[str], dict]:
    """What the allocation route answers a hotkey about itself: its entity (``lane``), its POTENTIAL in sessions
    (its weighted share with its own demand unbounded; a pooled hotkey: its entity's maximum per member) and, per
    env, the groups that potential holds in that env's own pool size (at least one). Before the fleet can be sized,
    ``groups`` is None: the miner keeps its own number rather than drop to one group. Each hotkey's answer is a
    snapshot kept ``ttl_s`` (the route never recomputes a potential per call)."""
    cache: dict[str, tuple[float, dict]] = {}

    def allocation_of(hotkey: str) -> dict:
        now = float(clock())
        # read live: an env added later is named (a callable: the envs counted now)
        sizes = {str(env): int(p) for env, p in (pool_seeds() if callable(pool_seeds) else pool_seeds).items()}
        kept = cache.get(hotkey)
        if kept is not None and now - kept[0] < ttl_s:
            return kept[1]
        if not quota.capacity_known():
            answer = {"hotkey": hotkey, "lane": None, "sessions": None, "groups": None, "pool_seeds": sizes}
        else:
            sessions = int(quota.sessions_allowed(hotkey))
            answer = {"hotkey": hotkey, "lane": quota.lane(hotkey), "sessions": sessions,
                      "groups": {env: max(1, math.ceil(sessions / p)) for env, p in sizes.items()},
                      "pool_seeds": sizes}
        for stale in [h for h, (at, _) in cache.items() if now - at >= ttl_s]:
            del cache[stale]
        cache[hotkey] = (now, answer)
        return answer

    return allocation_of


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
    window_uses: Callable[[int], Any] | None = None, require_resources: bool = False,
    boot_wait_s: float | None = None,
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
    (blocking; called in a worker thread); ``require_resources``: RL sessions go only to machines reporting their
    CPU and memory ``resources`` (production; a machine without them is logged once and gets none);
    ``boot_wait_s``: at start, wait at most this long for a machine serving each episode env, then start the
    unserved ones degraded (None: no check)."""
    from reliquary_sandbox.attest import Ed25519TokenVerifier, Signer, load_private_key

    from reliquary.infrastructure import sandbox_store
    from reliquary.sandbox import require_sandbox
    from reliquary.sandbox.fleet import Fleet, http_fetch_report
    from reliquary.sandbox.rl_yield import LedgerEvents, SessionYieldLedger
    from reliquary.sandbox.rl_routes import (
        BODY_TIMEOUT_S, MAX_PREAUTH, build_allocation_router, build_episode_precommit_router,
    )
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
        # daily abort cap is not lifted (an abort is neutral and frees its seed): it scales with the allocation.
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
    fleet.resources_required = bool(require_resources)
    if yield_path is not None and ledger is not None:
        raise ValueError("pass the yield ledger's path or the ledger, not both")
    book = SessionBook(policy)
    if ledger is None and yield_path is not None:
        ledger = SessionYieldLedger(yield_path, clock=clock)
    tiers = None
    built: dict = {}             # the services, once built: the routers and the quota are built before them

    def counted_reservations():
        # The reservations of the envs counted now (the current window's, tiers settled); every env before the
        # services exist.
        services = built.get("services")
        return tiers.reservations() if services is None else services.reservations()

    if yield_quota and quota is None:
        from reliquary.sandbox.rl_capacity import SessionTiers

        # The fleet size, in sessions of each env at its largest task tier that the machines' slots, CPU and
        # memory hold (never the bare slots: an allocation must never exceed what placement can host).
        tiers = SessionTiers(policies)
        # M by every name an env goes by: the env name an open passes and the sandbox env a session stores.
        quota = YieldQuota(ledger=ledger, pool_seeds=pool_seeds,
                           group_size=min(p.pool_seeds for p in policies.values()) // 2,
                           capacity=lambda: fleet.session_capacity(reservations=counted_reservations()),
                           reported=getattr(fleet, "session_reported", None),
                           group_sizes=episode_group_sizes(policies), retry_after_s=policy.retry_after_s, clock=clock,
                           # a hotkey's groups on one env: within its share of that env's machines
                           env_capacities=lambda: {env: row["slots"] for env, row in fleet.capacity_by_env(
                               reservations=counted_reservations()).items()},
                           env_names=_env_names(policies))
    if ledger is not None and isinstance(quota, YieldQuota):
        ledger.allow_capacity_verdicts(quota.groups_allowed)      # capacity verdicts: per hotkey and window
        book.cap_scale = quota.cap_scale                          # the per-hotkey rates scale with the allocation
    queue = saturation = None
    if isinstance(quota, YieldQuota):
        # RL opens finding no room wait in the fair queue (outside the issuer lock) instead of being refused.
        from reliquary.sandbox.open_queue import OpenQueue
        from reliquary.sandbox.rl_metrics import SaturationMonitor

        saturation = SaturationMonitor(clock=clock)
        queue = OpenQueue(rank=quota.rank, headroom=lambda hotkey: quota.headroom(hotkey, int(clock())),
                          on_wait=saturation.record_wait)
    resolvers = {name: environments[name].source.resolve for name in policies}
    if tiers is not None:
        resolvers = {name: _observing(tiers, name, resolve) for name, resolve in resolvers.items()}
    views = {name: EpisodeEnvironmentView(policy=p, resolve_task=resolvers[name]) for name, p in policies.items()}
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
            LedgerEvents(ledger), quota if isinstance(quota, YieldQuota) else None, saturation),
        record_faults=ledger is not None,
        open_refusal_check=None if ledger is None else fleet.open_refusal_valid, queue=queue,
        operator_of=operator_of if ledger is not None else None,
        entitled=quota.sessions_allowed if ledger is not None and isinstance(quota, YieldQuota) else None)
    fleet.on_drained = issuer.void_machine
    if queue is not None:
        fleet.on_polled = queue.notify                            # each poll may show freed room
    sessions_router = build_sandbox_sessions_router(
        issuer, policy=policy, validator_hotkey=validator_hotkey, prefix=RL_PREFIX, registration=registration,
        clock=clock, max_concurrent_closes=config.close_concurrency,
        close_body_timeout_s=config.close_body_timeout_s, max_preauth_closes=config.close_preauth_concurrency,
        # the opens are bounded like the precommits
        open_body_timeout_s=BODY_TIMEOUT_S if open_body_timeout_s is None else open_body_timeout_s,
        max_preauth_opens=MAX_PREAUTH if max_preauth_opens is None else max_preauth_opens)

    async def admit(value):
        return await built["services"].precommit_admission(value)

    precommit_router = build_episode_precommit_router(
        record=runtime.record_episode_precommit, current_window=current_window,
        validator_hotkey=validator_hotkey, policy=policy, prefix=RL_PREFIX, registration=registration,
        clock=clock, admit=admit,
        precheck=lambda value: runtime.record_episode_precommit(value, dry_run=True))
    checkers = {name: EpisodeGroupChecker(policy=p, renderer=renderers[name], source=environments[name].source,
                                          chunk_tokens=chunk_tokens)
                for name, p in policies.items()}
    outcomes = outcomes or ledger or NoOutcomes()
    window_open = (None if window_started_at is None
                   else (lambda window: isinstance(window, int) and window_started_at(window) is not None))
    intake = EpisodeGroupIntake(checkers=checkers, precommits=runtime.episode_precommit,
                                directory=fleet.directory_if_ready, token_verifier=token_verifier,
                                sessions=issuer, seen=book.submitted_ids, outcomes=outcomes,
                                window_open=window_open, check_metrics=saturation, **_check_caps(quota, clock))
    logger.info("rl episode services built for %s (validator key %s)", sorted(policies), signer.key_id)
    routers = (sessions_router, precommit_router)
    pool_sizes = {name: int(p.pool_seeds) for name, p in policies.items()}
    if isinstance(quota, YieldQuota):
        routers += (build_allocation_router(
            _allocation_of(quota, lambda: built["services"].allocation_pool_sizes(), clock=clock),
            validator_hotkey=validator_hotkey, policy=policy, prefix=RL_PREFIX, registration=registration,
            clock=clock),)
    if saturation is not None:
        routers += (build_saturation_router(lambda: built["services"].cached_saturation()),)
    services = RlEpisodeServices(signer=signer, token_verifier=token_verifier, fleet=fleet, book=book,
                             issuer=issuer, routers=routers, intake=intake,
                             environments=tuple(sorted(policies)), runtime=runtime, outcomes=outcomes,
                             clock=clock, precommit_retention_s=float(precommit_retention_s),
                             ledger=ledger, quota=quota, queue=queue, window_uses=window_uses, policies=policies,
                             window_open=window_open, current_window=current_window, operator_of=operator_of,
                             served={name: (p.sandbox_env, p.env_package,
                                            getattr(environments[name].source, "options_sha256", None))
                                     for name, p in policies.items()},
                             resolvers=resolvers, boot_wait_s=boot_wait_s, views=views, saturation=saturation,
                             tiers=tiers, task_counts={name: _task_count(environments[name].source) for name in policies},
                             raw_resolvers={name: environments[name].source.resolve for name in policies},
                             renderer_for=renderer_for,
                             proof_stop_ids=None if proof_stop_ids is None else frozenset(proof_stop_ids),
                             remote_stop_ids=None if remote_stop_ids is None else frozenset(remote_stop_ids),
                             chunk_tokens=int(chunk_tokens), pool_seeds=int(pool_seeds), pool_sizes=pool_sizes,
                             renderers={tuple(p.tools): renderers[name] for name, p in policies.items()})
    engagements.serving_refusal = services.serving_refusal
    built["services"] = services
    return services


__all__ = ["BOOT_FLEET_WAIT_S", "MIN_PRECOMMIT_RETENTION_S", "NO_PER_KEY_CAP", "RL_GROUPS_IN_FLIGHT", "RL_PREFIX", "YIELD_FLUSH_SECONDS", "RlEpisodeServices",
           "build_rl_episode_services", "episode_group_sizes", "episode_stop_set_refusal", "rl_registration", "rl_sandbox_policy",
           "stop_ids_from_metadata"]
