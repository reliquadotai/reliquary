"""Which RL task an async miner works for, when one validator serves several behind its front (``GET /rl/tasks``).

A task is worth its expected pay for the envs this miner serves: per env, its share of the task's batch times what a
lease pays there, the higher of the price of the moment and the miner's own ask (an ask over the env's ceiling wins
nothing). Only envs with work to give count (a budget, or an allowance of the miner's own). The best task wins, ties
to the lowest task id; a task without an auction pays nothing the miner can tell, so it is taken only when pinned
(``--rl-task``). A validator without ``/rl/tasks`` serves one task on the bare ``/rl`` routes.

The engine is built once: a task naming another model than the engine's is skipped (one naming none only when it is
the current task), and a task whose first anchor's config or tokenizer is not the engine's (``ModelFingerprint``) is
denied for the process's life, as is one whose episode envs the engine cannot play (``vet``).

The miner chooses again every ``RECHOOSE_S`` and when its task answers ``full`` or 404 (``mine_until_moved``). It
moves only for ``MOVE_GAIN`` more pay, or when its task has no work left or is gone. A move leases nothing more on the
old task and waits for its open leases to end before it goes.
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
MOVE_GAIN = 1.2                     # the pay a move must bring, over the current task's
MODEL_FILES = ("config.json",)      # with every ``tokenizer*`` file: what makes the engine's model
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


def _model_fits(task: Mapping[str, Any], model: Mapping[str, Any] | None, current: str | None) -> bool:
    if model is None:
        return True                   # the engine's model is not known yet: the anchor check guards
    named = task.get("model")
    if not isinstance(named, Mapping):
        return str(task.get("task_id")) == current
    return (named.get("id"), named.get("revision")) == (model.get("id"), model.get("revision"))


def _best(tasks: list[dict], *, served_envs, asks, model, current, excluded) -> tuple[float, str] | None:
    best: tuple[float, str] | None = None
    for task in tasks:
        if not isinstance(task, Mapping) or not TASK_ID.fullmatch(str(task.get("task_id"))):
            continue
        task_id = str(task["task_id"])
        if task_id in excluded or not _model_fits(task, model, current):
            continue
        pay = expected_pay(task, served_envs=served_envs, asks=asks)
        if pay is None:
            continue
        if best is None or pay > best[0] or (pay == best[0] and task_id < best[1]):
            best = (pay, task_id)
    return best


def choose_task(tasks: list[dict], *, served_envs: set[str], asks: Mapping[str, float],
                model: Mapping[str, Any] | None = None, current: str | None = None,
                excluded: Iterable[str] = ()) -> str | None:
    """The best paying task this engine (``model``) can play, ``excluded`` aside."""
    best = _best(tasks, served_envs=served_envs, asks=asks, model=model, current=current, excluded=set(excluded))
    return None if best is None else best[1]


class ModelFingerprint:
    """The engine's model, as the first checkpoint index it syncs names it: the SHA-256 of its config and tokenizer
    files. A later index (another task's) matches only with the same files."""

    def __init__(self) -> None:
        self.files: dict[str, str] | None = None

    @staticmethod
    def of(index: Mapping[str, Any]) -> dict[str, str]:
        from reliquary.shared.ckpt_index import latest_model_files

        return {str(f.get("name")): str(f.get("sha256")) for f in latest_model_files(index)
                if f.get("name") in MODEL_FILES or str(f.get("name", "")).startswith("tokenizer")}

    def matches(self, index: Mapping[str, Any]) -> bool:
        files = self.of(index)
        if not files:
            return True
        if self.files is None:
            self.files = files
        return files == self.files


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
        self.model: Mapping[str, Any] | None = None     # the engine's, once known (``adopt_model``)
        self.denied: set[str] = set()                   # tasks the engine cannot play, for the process's life
        self.listed: dict[str, Mapping[str, Any]] = {}  # the last listing, by task id

    def deny(self, task_id: str | None) -> None:
        if task_id is not None:
            logger.warning("rl task %s denied for this process: its model or envs are not the engine's", task_id)
            self.denied.add(task_id)

    def adopt_model(self, task_id: str | None) -> None:
        """The engine was built for ``task_id``: its listed model is the engine's."""
        named = self.listed.get(task_id, {}).get("model") if task_id is not None else None
        if isinstance(named, Mapping) and self.model is None:
            self.model = dict(named)

    async def choose(self, current: str | None = None) -> str | None:
        """The task to work for (None: the bare ``/rl`` routes). Another task only for ``MOVE_GAIN`` more pay, or
        when the current one has no work left (it stays if nothing else has) or is gone (bare ``/rl`` if nothing
        else qualifies). The list out of reach: the current task stays."""
        if self.pinned is not None:
            return self.pinned
        try:
            tasks = await self._fetch()
        except Exception as exc:
            logger.warning("rl tasks not listed (%s); staying on %s", type(exc).__name__, task_prefix(current))
            return current
        if tasks is None:
            return None
        self.listed = {str(t["task_id"]): t for t in tasks if isinstance(t, Mapping) and "task_id" in t}
        best = _best(tasks, served_envs=self.served_envs, asks=self.asks, model=self.model, current=current,
                     excluded=self.denied)
        if current is None:
            return None if best is None else best[1]
        here = self.listed.get(current)
        if here is None or current in self.denied:            # gone, or not the engine's
            return None if best is None else best[1]
        pay = expected_pay(here, served_envs=self.served_envs, asks=self.asks)
        if best is None or best[1] == current:
            return current
        if pay is None or best[0] >= MOVE_GAIN * pay:
            return best[1]
        return current


async def mine_until_moved(miner, chooser: TaskChooser, current: str | None, *, period_s: float | None = None,
                           signal_min_s: float | None = None, idle_poll_s: float | None = None,
                           vet: Callable[[str | None], Awaitable[bool | None]] | None = None,
                           clock: Callable[[], float] = time.monotonic) -> tuple[bool, str | None]:
    """Run ``miner`` on ``current`` until another task is chosen: then lease nothing more, wait for the open leases
    to end, stop it and return ``(True, task)``. ``vet(task)``: whether the engine can play the candidate (False:
    denied, chosen again; None: unknown now, the miner stays). A miner whose checkpoint is not the engine's model (``model_mismatch``) denies its task. A miner
    that stops on its own returns ``(False, current)`` (its error raised)."""
    period_s = RECHOOSE_S if period_s is None else period_s
    signal_min_s = SIGNAL_MIN_S if signal_min_s is None else signal_min_s
    idle_poll_s = IDLE_POLL_S if idle_poll_s is None else idle_poll_s
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
            if getattr(miner, "model_mismatch", False):
                chooser.deny(current)
            choice = await chooser.choose(current)
            while choice != current and vet is not None:
                playable = await vet(choice)
                if playable:
                    break
                if playable is None or choice is None:     # unknown now, or the bare routes: stay
                    choice = current
                    break
                chooser.deny(choice)
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


__all__ = ["BARE_PREFIX", "MOVE_GAIN", "RECHOOSE_S", "ModelFingerprint", "TaskChooser", "TaskListUnavailable", "choose_task", "expected_pay",
           "fetch_tasks", "mine_until_moved", "task_prefix"]
