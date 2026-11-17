"""A miner's choice among the RL tasks behind one front (``GET /rl/tasks``): the task that pays best for the envs it
serves, a pinned task kept, an old validator's bare ``/rl`` routes, and the move to another task once the open
leases are done."""
from __future__ import annotations

import asyncio

import pytest

from reliquary.miner.rl_task_choice import (
    BARE_PREFIX, TaskChooser, choose_task, expected_pay, fetch_tasks, mine_until_moved, task_prefix,
)


def env(share, price, *, ceiling=10.0, budget=4, allowance=None):
    view = {"pricing": "auction", "price_alpha_per_mtoken": price, "ceiling_alpha_per_mtoken": ceiling,
            "budget": budget, "share": share}
    if allowance is not None:
        view["allowance"] = allowance
    return view


def task(task_id, **envs):
    return {"task_id": task_id, "environments": {e: v["share"] for e, v in envs.items()},
            "auction": {"lag_target": 2, "environments": envs, "stopped": None}, "draw_floor": 0.9}


def test_a_miner_picks_the_best_paying_task_it_serves():
    tasks = [
        task("a", math=env(0.5, 2.0), code=env(0.5, 9.0)),       # code pays most, but the miner serves math only
        task("b", math=env(1.0, 1.5)),
        task("c", code=env(1.0, 9.0)),                           # nothing the miner serves
    ]
    served = {"math"}
    assert expected_pay(tasks[0], served_envs=served, asks={}) == pytest.approx(1.0)
    assert expected_pay(tasks[2], served_envs=served, asks={}) is None
    assert choose_task(tasks, served_envs=served, asks={}) == "b"
    # its own ask counts when the price of the moment is under it (and under the ceiling)
    assert expected_pay(tasks[0], served_envs=served, asks={"math": 4.0}) == pytest.approx(2.0)
    assert expected_pay(tasks[1], served_envs=served, asks={"math": 4.0}) == pytest.approx(4.0)
    # an ask over an env's ceiling wins nothing there
    capped = [task("a", math=env(1.0, 1.0, ceiling=3.0)), task("b", math=env(1.0, 2.0))]
    assert choose_task(capped, served_envs=served, asks={"math": 5.0}) == "b"
    assert expected_pay(capped[0], served_envs=served, asks={"math": 5.0}) is None
    # ties go to the lowest task id; an unpriced env pays its ask
    tied = [task("z", math=env(1.0, None)), task("m", math=env(1.0, 1.0))]
    assert choose_task(tied, served_envs=served, asks={"math": 1.0}) == "m"


def test_the_ask_ordering_picks_the_task_where_the_miner_earns_more():
    tasks = [task("a", math=env(0.5, 2.0)), task("b", math=env(1.0, 1.5))]
    assert choose_task(tasks, served_envs={"math"}, asks={"math": 4.0}) == "b"


def test_a_task_with_no_budget_is_skipped():
    served = {"math"}
    tasks = [task("a", math=env(1.0, 9.0, budget=0)), task("b", math=env(1.0, 1.0))]
    assert choose_task(tasks, served_envs=served, asks={}) == "b"
    # an allowance of its own keeps a task even at a zero budget
    tasks[0]["auction"]["environments"]["math"]["allowance"] = 2
    assert choose_task(tasks, served_envs=served, asks={}) == "a"
    # a task without an auction pays nothing it can tell: never chosen unless pinned
    plain = {"task_id": "0", "environments": {}, "auction": None, "draw_floor": 0.9}
    assert choose_task([plain, task("b", math=env(1.0, 1.0, budget=0))], served_envs=served, asks={}) is None
    assert choose_task([], served_envs=served, asks={}) is None


async def test_a_pinned_task_is_kept():
    fetched = []

    async def fetch():
        fetched.append(1)
        return [task("a", math=env(1.0, 1.0)), task("b", math=env(1.0, 9.0))]

    pinned = TaskChooser(fetch, served_envs={"math"}, asks={}, pinned="a")
    assert await pinned.choose() == "a" and await pinned.choose("a") == "a"
    assert fetched == [] and task_prefix("a") == "/rl/a"
    free = TaskChooser(fetch, served_envs={"math"}, asks={})
    assert await free.choose() == "b" and await free.choose("a") == "b"
    # nothing to pick (every budget spent) or the list unreachable: the miner stays where it is
    async def spent():
        return [task("b", math=env(1.0, 9.0, budget=0))]

    async def down():
        raise OSError("down")

    assert await TaskChooser(spent, served_envs={"math"}, asks={}).choose("a") == "a"
    assert await TaskChooser(down, served_envs={"math"}, asks={}).choose("a") == "a"


async def test_an_old_validator_keeps_bare_routes():
    httpx = pytest.importorskip("httpx")

    def old(request):
        assert request.url.path == "/rl/tasks"
        return httpx.Response(404, json={"detail": "Not Found"})

    def new(request):
        return httpx.Response(200, json=[task("a", math=env(1.0, 1.0))])

    async with httpx.AsyncClient(transport=httpx.MockTransport(old)) as client:
        assert await fetch_tasks(client, "http://v") is None
        chooser = TaskChooser(lambda: fetch_tasks(client, "http://v"), served_envs={"math"}, asks={})
        assert await chooser.choose() is None and task_prefix(None) == BARE_PREFIX == "/rl"
    async with httpx.AsyncClient(transport=httpx.MockTransport(new)) as client:
        assert [t["task_id"] for t in await fetch_tasks(client, "http://v/")] == ["a"]


class FakeMiner:
    """``mine_until_moved``'s view of an ``AsyncMiner``: run until stopped, a task signal, wind down, idle."""

    def __init__(self) -> None:
        self.task_signal = asyncio.Event()
        self.leasing = True
        self.open = {"l1": object()}
        self.stopped = False

    def wind_down(self) -> None:
        self.leasing = False

    def idle(self) -> bool:
        return not self.open

    async def run(self, stop: asyncio.Event) -> None:
        await stop.wait()
        self.stopped = True


async def test_the_miner_moves_on_a_full_task_once_its_leases_are_done():
    choices = iter(["a", "b"])

    async def fetch():
        best = next(choices)
        return [task(best, math=env(1.0, 9.0))]

    miner = FakeMiner()
    chooser = TaskChooser(fetch, served_envs={"math"}, asks={})
    moving = asyncio.create_task(mine_until_moved(miner, chooser, "a", period_s=3600, signal_min_s=0,
                                                  idle_poll_s=0.01))
    await asyncio.sleep(0.02)
    miner.task_signal.set()                       # "full": a better task, if any, is looked for now
    await asyncio.sleep(0.05)
    assert miner.leasing and not moving.done()    # still "a": the miner stays
    miner.task_signal.set()
    await asyncio.sleep(0.05)
    assert not miner.leasing and not moving.done()   # "b" pays: no new lease, the open one finishes first
    miner.open.clear()
    assert await asyncio.wait_for(moving, 1) == (True, "b") and miner.stopped


async def test_a_miner_that_stops_on_its_own_is_not_moved():
    async def fetch():
        return []

    class Ending(FakeMiner):
        async def run(self, stop):
            raise RuntimeError("generator died")

    with pytest.raises(RuntimeError):
        await mine_until_moved(Ending(), TaskChooser(fetch, served_envs={"math"}, asks={}), "a")


async def test_a_full_or_vanished_task_signals_the_miner(monkeypatch):
    from reliquary.miner.rl_async import RequestRefused
    from tests.unit.test_rl_async_miner import miner_for

    miner, transport, _, planner = miner_for()
    assert not miner.task_signal.is_set()
    miner._note_refusal(planner, 1, "full")
    assert miner.task_signal.is_set()
    miner.task_signal.clear()

    async def gone(*args):
        raise RequestRefused("404", retryable=False)

    transport.lease = gone
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: real_sleep(0))      # the refusal's back-off
    await miner.refresh_quota()
    assert await miner.lease_more() == 0
    assert miner.task_signal.is_set()
    miner.wind_down()
    assert miner.lease_room() == 0


@pytest.mark.parametrize("listed,pinned,prefix", [
    ([task("a", math=env(1.0, 1.0)), task("b", math=env(1.0, 2.0))], None, "/rl/b"),
    ([task("a", math=env(1.0, 1.0)), task("b", math=env(1.0, 2.0))], "a", "/rl/a"),
    (None, None, "/rl"),
])
async def test_the_miner_mines_its_task_on_that_tasks_path(tmp_path, monkeypatch, listed, pinned, prefix):
    bt = pytest.importorskip("bittensor")
    httpx = pytest.importorskip("httpx")
    from types import SimpleNamespace

    from reliquary.miner import rl_async
    from reliquary.miner.rl_async import AsyncMinerConfig, run_async_miner

    def front(request):
        if request.url.path == "/rl/tasks" and listed is not None:
            return httpx.Response(200, json=listed)
        return httpx.Response(404, json={"detail": "Not Found"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(front), **kw))
    seen = []

    async def run(self, stop=None):
        seen.append((self.transport._prefix, self.config.state_dir))

    monkeypatch.setattr(rl_async.AsyncMiner, "run", run)
    wallet = SimpleNamespace(hotkey=bt.Keypair.create_from_uri("//Alice"))
    config = AsyncMinerConfig(environments=("math",), state_dir=str(tmp_path / "state"))
    await run_async_miner(config, validator_url="http://v", validator_hotkey="5Val", wallet=wallet,
                          work_dir=str(tmp_path), fake_generator=True, rl_task=pinned,
                          prompt_ranges={"math": (0, 10)})
    task_id = prefix.rpartition("/")[2]
    expected_state = str(tmp_path / "state") + ("" if prefix == "/rl" else f"/tasks/{task_id}")
    assert seen == [(prefix, expected_state)]


def test_the_pay_board_is_read_from_the_tasks_path():
    from reliquary.miner.rl_pay_client import PayWeights

    urls = []
    PayWeights("http://v/", prefix="/rl/a", fetch=urls.append, background=False).weights()
    PayWeights("http://v", fetch=urls.append, background=False).weights()
    assert urls == ["http://v/rl/a/pay", "http://v/rl/pay"]
