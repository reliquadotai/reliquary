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
from collections.abc import Callable, Mapping
from typing import Any

from reliquary.protocol.service_episode import GROUPS_IN_FLIGHT
from reliquary.sandbox.sessions import SandboxPolicy

logger = logging.getLogger(__name__)

RL_PREFIX = "/rl"
SESSION_MAINTAIN_SECONDS = 60.0
RESTORE_ATTEMPTS = 3
RESTORE_TIMEOUT_S = 60.0
RESTORE_BACKOFF_S = 5.0
STOP_TIMEOUT_S = 10.0
RL_GROUPS_IN_FLIGHT = GROUPS_IN_FLIGHT     # the intake's per-operator check cap; with the yield quota the cap is the hotkey's submittable groups
"""At start, how long the validator waits for a machine serving each episode env before starting it degraded."""
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


async def placement_refusal(*, fleet, view, session_policy, environment: str, task_index: int, group: int,
                            clock: Callable[[], float]) -> tuple[str, dict] | None:
    """Whether the fleet can host ``group`` sessions of this episode task at once, on at least
    ``MIN_SERVING_MACHINES`` machines by the open path's own filter: None (yes, or a task the source has not,
    left to the runtime's own checks), ``fleet_unavailable`` (retryable: a slow source, a stale or partial view)
    or ``task_unservable`` (fresh machines serve the env and the open path would refuse this task on all of
    them). Shared by the window path's precommit admission and the async path's lease grant."""
    from reliquary.sandbox.rl_capacity import MIN_SERVING_MACHINES
    from reliquary.sandbox.rl_engagements import session_budgets
    from reliquary.sandbox.sessions import token_validity_s

    try:
        task = await asyncio.wait_for(view.resolve_task(int(task_index)), session_policy.io_timeout_s)
    except TimeoutError:
        # The source is slow now: never admitted unchecked, never a final answer.
        return "fleet_unavailable", {"environment": environment, "why": "task_resolve_timeout"}
    except Exception:
        return None                                # a task the source has not: the runtime's own checks
    policy = view.policy
    options = getattr(task, "env_options_sha256", None)
    if not fleet.serving(policy.sandbox_env, policy.env_package, env_options_sha256=options):
        # A stale directory or report, or no machine serving the env right now: never a final answer.
        return "fleet_unavailable", {"environment": environment}
    budgets = session_budgets(policy, task)
    reserve = {"cpu_millis": getattr(task, "cpu_millis", None), "memory_bytes": int(budgets["memory_bytes"])}
    placement = dict(image=task.image, env=policy.sandbox_env, env_package=policy.env_package, budgets=budgets,
                     # the open path's own token validity (its machines' caps must allow it)
                     validity_s=token_validity_s(session_policy.open_window_s, int(budgets["wall_s"]),
                                                 getattr(task, "prepare_timeout_s", None)),
                     env_options_sha256=options, reserve=reserve)
    lock = getattr(task, "env_lock_sha256", None)
    if lock is not None:                           # the open path's own lock filter (``_reserving``)
        placement["env_lock_sha256"] = lock
    machines = fleet.servable(**placement, group_size=group)     # the group at once
    if machines >= MIN_SERVING_MACHINES:
        return None
    verdict = fleet.could_place(**placement, now=clock())
    if verdict is None:
        # A machine that served it when last heard is stale or drained, or one's resources cannot be read:
        # the open path itself would answer retryably.
        return "fleet_unavailable", {"environment": environment}
    if verdict and fleet.servable(**placement, group_size=group, include_unknown=True) >= MIN_SERVING_MACHINES:
        # Its fresh machines alone cannot host its group, but with the ones missing a heartbeat (or whose
        # resources cannot be read now) they could: a partial view never gives a final answer.
        return "fleet_unavailable", {"environment": environment}
    return "task_unservable", {"environment": environment, "task_index": int(task_index),
                               "image": task.image, "cpu_millis": reserve["cpu_millis"], "machines": machines}


def _env_names(policies: Mapping[str, Any]) -> dict[str, str]:
    """Every name an episode env goes by (its order name, its sandbox env) -> its sandbox env."""
    names = {}
    for name, p in policies.items():
        names[name] = names[p.sandbox_env] = p.sandbox_env
    return names


"""The operator's bearer token for ``GET /rl/saturation`` (the service observations' token when unset)."""


"""How long one hotkey's answer of the allocation route is served from its snapshot before it is recomputed."""


__all__ = ["NO_PER_KEY_CAP", "RL_GROUPS_IN_FLIGHT", "RL_PREFIX", "rl_sandbox_policy"]
