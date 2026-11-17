"""Which RL task an async miner works for, when one validator serves several behind its front (``GET /rl/tasks``).

A task is worth its expected pay for the envs this miner serves: per env, its share of the task's batch times what a
lease pays there, the higher of the price of the moment and the miner's own ask (an ask over the env's ceiling wins
nothing). Only envs with work to give count (a budget, or an allowance of the miner's own). The best task wins, ties
to the lowest task id; a task without an auction pays nothing the miner can tell, so it is taken only when pinned
(``--rl-task``). A validator without ``/rl/tasks`` serves one task on the bare ``/rl`` routes.

The miner chooses again every ``RECHOOSE_S`` and when its task answers ``full`` or 404 (``mine_until_moved``). A move
leases nothing more on the old task and waits for its open leases to end before it goes.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

BARE_PREFIX = "/rl"
RECHOOSE_S = 600.0
SIGNAL_MIN_S = 60.0                 # a burst of ``full`` answers asks the front once a minute at most
IDLE_POLL_S = 5.0
TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")     # a path segment and a directory name


def task_prefix(task_id: str | None) -> str:
    if task_id is not None and not TASK_ID.fullmatch(task_id):
        raise ValueError(f"not an RL task id: {task_id!r}")
    return BARE_PREFIX if task_id is None else f"{BARE_PREFIX}/{task_id}"


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def expected_pay(task: Mapping[str, Any], *, served_envs: Iterable[str], asks: Mapping[str, float]) -> float | None:
    """Alpha per million matched tokens the task is expected to pay this miner, or None when none of its served envs
    has work for it (no auction, no budget nor allowance, or its ask over the ceiling)."""
    auction = task.get("auction")
    envs = auction.get("environments") if isinstance(auction, Mapping) else None
    if not isinstance(envs, Mapping):
        return None
    total, any_work = 0.0, False
    for env in served_envs:
        view = envs.get(env)
        if not isinstance(view, Mapping):
            continue
        if not ((_number(view.get("allowance")) or 0) > 0 or (_number(view.get("budget")) or 0) > 0):
            continue
        ask, ceiling = asks.get(env), _number(view.get("ceiling_alpha_per_mtoken"))
        if ask is not None and ceiling is not None and float(ask) > ceiling:
            continue
        price = max(_number(view.get("price_alpha_per_mtoken")) or 0.0, float(ask or 0.0))
        total += (_number(view.get("share")) or 0.0) * price
        any_work = True
    return total if any_work else None


def choose_task(tasks: list[dict], *, served_envs: set[str], asks: Mapping[str, float]) -> str | None:
    best: tuple[float, str] | None = None
    for task in tasks:
        if not isinstance(task, Mapping) or not TASK_ID.fullmatch(str(task.get("task_id"))):
            continue
        pay = expected_pay(task, served_envs=served_envs, asks=asks)
        if pay is None:
            continue
        task_id = str(task["task_id"])
        if best is None or pay > best[0] or (pay == best[0] and task_id < best[1]):
            best = (pay, task_id)
    return None if best is None else best[1]


class TaskListUnavailable(Exception):
    pass


async def fetch_tasks(client, validator_url: str) -> list[dict] | None:
    """``GET /rl/tasks``; None when the validator has no such route (an old validator: bare ``/rl``)."""
    response = await client.get(validator_url.rstrip("/") + "/rl/tasks")
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise TaskListUnavailable(str(response.status_code))
    body = response.json()
    if not isinstance(body, list):
        raise TaskListUnavailable("not a list")
    return body


class TaskChooser:
    """``fetch``: the task list (``fetch_tasks``). ``asks``: the miner's own offers, alpha per million tokens."""

    def __init__(self, fetch: Callable[[], Awaitable[list[dict] | None]], *, served_envs: Iterable[str],
                 asks: Mapping[str, float], pinned: str | None = None) -> None:
        self._fetch, self.pinned = fetch, pinned
        self.served_envs, self.asks = set(served_envs), dict(asks)

    async def choose(self, current: str | None = None) -> str | None:
        """The task to work for (None: the bare ``/rl`` routes). Nothing to pick or the list out of reach: the
        current task stays."""
        if self.pinned is not None:
            return self.pinned
        try:
            tasks = await self._fetch()
        except Exception as exc:
            logger.warning("rl tasks not listed (%s); staying on %s", type(exc).__name__, task_prefix(current))
            return current
        if tasks is None:
            return None
        best = choose_task(tasks, served_envs=self.served_envs, asks=self.asks)
        return current if best is None else best


async def mine_until_moved(miner, chooser: TaskChooser, current: str | None, *, period_s: float = RECHOOSE_S,
                           signal_min_s: float = SIGNAL_MIN_S, idle_poll_s: float = IDLE_POLL_S,
                           clock: Callable[[], float] = time.monotonic) -> tuple[bool, str | None]:
    """Run ``miner`` on ``current`` until another task is chosen: then lease nothing more, wait for the open leases
    to end, stop it and return ``(True, task)``. A miner that stops on its own returns ``(False, current)`` (its
    error raised)."""
    stop = asyncio.Event()
    running = asyncio.create_task(miner.run(stop))

    async def watch() -> str | None:
        last = clock()
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(miner.task_signal.wait(), timeout=max(0.0, period_s - (clock() - last)))
            if miner.task_signal.is_set() and clock() - last < signal_min_s:
                await asyncio.sleep(signal_min_s - (clock() - last))
            miner.task_signal.clear()
            last = clock()
            choice = await chooser.choose(current)
            if choice != current:
                logger.info("rl task: moving from %s to %s once the open leases end", task_prefix(current),
                            task_prefix(choice))
                miner.wind_down()
                while not miner.idle():
                    await asyncio.sleep(idle_poll_s)
                return choice

    watching = asyncio.create_task(watch())
    try:
        done, _ = await asyncio.wait({running, watching}, return_when=asyncio.FIRST_COMPLETED)
        if running in done:
            running.result()
            return False, current
        stop.set()
        await running
        return True, watching.result()
    finally:
        for task in (running, watching):
            if not task.done():
                task.cancel()
        await asyncio.gather(running, watching, return_exceptions=True)


__all__ = ["BARE_PREFIX", "RECHOOSE_S", "TaskChooser", "TaskListUnavailable", "choose_task", "expected_pay",
           "fetch_tasks", "mine_until_moved", "task_prefix"]
