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
        return [task("a", math=env(1.0, 1.0, budget=0)), task("b", math=env(1.0, 9.0, budget=0))]

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
        self.model_mismatch = False

    def wind_down(self) -> None:
        self.leasing = False

    def idle(self) -> bool:
        return not self.open

    async def run(self, stop: asyncio.Event) -> None:
        await stop.wait()
        self.stopped = True


async def until(condition, timeout=2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        assert loop.time() < deadline, "condition not reached"
        await asyncio.sleep(0.001)


async def test_the_miner_moves_on_a_full_task_once_its_leases_are_done():
    listings = iter([[task("a", math=env(1.0, 9.0))], [task("b", math=env(1.0, 9.0))]])
    fetched = []

    async def fetch():
        fetched.append(1)
        return next(listings)

    miner = FakeMiner()
    chooser = TaskChooser(fetch, served_envs={"math"}, asks={})
    moving = asyncio.create_task(mine_until_moved(miner, chooser, "a", period_s=3600, signal_min_s=0,
                                                  idle_poll_s=0.001))
    miner.task_signal.set()                       # "full": a better task, if any, is looked for now
    await until(lambda: len(fetched) == 1 and not miner.task_signal.is_set())
    assert miner.leasing and not moving.done()    # still "a": the miner stays
    miner.task_signal.set()
    await until(lambda: not miner.leasing)        # "a" is gone, "b" has work: no new lease, the open one finishes
    assert not moving.done()
    miner.open.clear()
    assert await asyncio.wait_for(moving, 2) == (True, "b") and miner.stopped


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


M1 = {"id": "Qwen/Qwen3.5-4B", "revision": "r1"}
M2 = {"id": "Qwen/Qwen3.5-2B", "revision": "r2"}


def with_model(listed, model):
    return {**listed, "model": model}


def test_a_task_on_another_model_is_never_chosen():
    served = {"math"}
    tasks = [with_model(task("a", math=env(1.0, 1.0)), M1), with_model(task("b", math=env(1.0, 9.0)), M2),
             task("c", math=env(1.0, 5.0))]                       # c names no model
    assert choose_task(tasks, served_envs=served, asks={}, model=M1) == "a"
    assert choose_task(tasks, served_envs=served, asks={}, model=M1, current="c") == "c"   # unnamed: only as current
    assert choose_task(tasks, served_envs=served, asks={}) == "b"                          # engine model unknown
    assert choose_task(tasks, served_envs=served, asks={}, model=M1, excluded={"a"}) is None


async def test_a_move_needs_a_fifth_more_pay_unless_the_task_has_no_work_or_is_gone():
    listing = []

    async def fetch():
        return listing

    chooser = TaskChooser(fetch, served_envs={"math"}, asks={})
    listing[:] = [task("a", math=env(1.0, 1.0)), task("b", math=env(1.0, 1.19))]
    assert await chooser.choose("a") == "a"
    listing[:] = [task("a", math=env(1.0, 1.0)), task("b", math=env(1.0, 1.2))]
    assert await chooser.choose("a") == "b"
    listing[:] = [task("a", math=env(1.0, 9.0, budget=0)), task("b", math=env(1.0, 1.0))]
    assert await chooser.choose("a") == "b"                      # a has no work left
    listing[:] = [task("a", math=env(1.0, 9.0, budget=0))]
    assert await chooser.choose("a") == "a"                      # no work anywhere: stay
    listing[:] = [task("b", math=env(1.0, 9.0, budget=0))]
    assert await chooser.choose("a") is None                     # a is gone and nothing qualifies: bare /rl
    listing[:] = [task("a", math=env(1.0, 1.0)), task("b", math=env(1.0, 5.0))]
    chooser.deny("b")
    assert await chooser.choose("a") == "a" and await chooser.choose() == "a"


async def test_a_task_the_engine_cannot_play_is_denied_before_winding_down():
    async def fetch():
        return [task("a", math=env(1.0, 1.0)), task("b", math=env(1.0, 9.0)), task("c", math=env(1.0, 5.0))]

    vetted = []

    async def vet(candidate):
        vetted.append(candidate)
        return candidate != "b"                  # b serves math as a signed-episode env, the engine does not

    miner = FakeMiner()
    miner.open.clear()
    chooser = TaskChooser(fetch, served_envs={"math"}, asks={})
    moving = asyncio.create_task(mine_until_moved(miner, chooser, "a", period_s=3600, signal_min_s=0,
                                                  idle_poll_s=0.001, vet=vet))
    miner.task_signal.set()
    assert await asyncio.wait_for(moving, 2) == (True, "c")
    assert vetted == ["b", "c"] and "b" in chooser.denied


async def test_a_checkpoint_of_another_model_denies_the_task_and_chooses_again():
    async def fetch():
        return [task("a", math=env(1.0, 9.0)), task("b", math=env(1.0, 1.0))]

    miner = FakeMiner()
    miner.open.clear()
    chooser = TaskChooser(fetch, served_envs={"math"}, asks={})
    moving = asyncio.create_task(mine_until_moved(miner, chooser, "a", period_s=3600, signal_min_s=0,
                                                  idle_poll_s=0.001))
    miner.model_mismatch = True                  # a's anchor is not the engine's model
    miner.task_signal.set()
    assert await asyncio.wait_for(moving, 2) == (True, "b") and chooser.denied == {"a"}


def model_index(config_sha, tokenizer_sha="t" * 64):
    files = [{"name": "config.json", "sha256": config_sha, "size": 1, "key": "k1"},
             {"name": "tokenizer.json", "sha256": tokenizer_sha, "size": 1, "key": "k2"},
             {"name": "generation_config.json", "sha256": "g" * 64, "size": 1, "key": "k3"}]
    return {"anchors": [{"version": 0, "model_files": files}]}


def test_the_engines_model_is_the_first_anchors_config_and_tokenizer():
    from reliquary.miner.rl_task_choice import ModelFingerprint

    engine = ModelFingerprint()
    assert engine.matches(model_index("c" * 64))                # the first anchor names the engine's model
    assert engine.matches(model_index("c" * 64))
    other = model_index("c" * 64)
    other["anchors"][0]["model_files"][2]["sha256"] = "x" * 64  # a generation config is not the model
    assert engine.matches(other)
    assert not engine.matches(model_index("d" * 64))
    assert not engine.matches(model_index("c" * 64, tokenizer_sha="u" * 64))
    assert engine.matches({"anchors": []})                       # no model files listed: nothing to compare


async def test_a_miner_leases_nothing_on_a_checkpoint_of_another_model():
    from reliquary.miner.rl_task_choice import ModelFingerprint
    from tests.unit.test_rl_async_miner import miner_for

    class Sync:
        calls = 0

        def sync(self, *args):
            Sync.calls += 1
            raise AssertionError("never synced")

    engine = ModelFingerprint()
    engine.matches(model_index("c" * 64))
    miner, transport, _, _ = miner_for(sync=Sync(), loaded_version=None)
    miner.model_check = engine.matches
    transport.checkpoints.append({"index": {**model_index("d" * 64), "latest": {"version": 3}}})
    assert await miner.follow_checkpoint_once() is None
    assert Sync.calls == 0 and miner.model_mismatch and miner.task_signal.is_set() and miner.lease_room() == 0


def test_a_move_repoints_the_episode_sessions():
    from types import SimpleNamespace

    from reliquary.miner.rl_async import repoint_episode_sessions

    sessions = SimpleNamespace(prefix="/rl/a")
    generator = SimpleNamespace(episodes=SimpleNamespace(sessions=sessions))
    repoint_episode_sessions(generator, "/rl/b")
    assert sessions.prefix == "/rl/b"
    repoint_episode_sessions(SimpleNamespace(), "/rl/b")        # no episode stack: nothing to repoint


async def test_an_episode_miner_moves_to_a_task_on_its_model(tmp_path, monkeypatch):
    bt = pytest.importorskip("bittensor")
    httpx = pytest.importorskip("httpx")
    from types import SimpleNamespace

    from reliquary.miner import rl_async, rl_task_choice
    from reliquary.miner.rl_async import AsyncMinerConfig, run_async_miner

    block = {"math": {"policy": "p"}}
    listings = iter([
        [with_model(task("a", math=env(1.0, 1.0)), M1)],
        [with_model(task("a", math=env(1.0, 1.0)), M1), with_model(task("b", math=env(1.0, 9.0)), M2),
         with_model(task("c", math=env(1.0, 3.0)), M1)],
    ])

    def front(request):
        if request.url.path == "/rl/tasks":
            return httpx.Response(200, json=next(listings))
        if request.url.path.endswith("/episodes"):
            return httpx.Response(200, json={"environments": block})
        return httpx.Response(404, json={"detail": "Not Found"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(front), **kw))
    monkeypatch.setattr(rl_task_choice, "SIGNAL_MIN_S", 0.0)
    monkeypatch.setattr(rl_task_choice, "IDLE_POLL_S", 0.001)
    monkeypatch.setattr(rl_async, "planner_ranges", lambda envs, **kw: {e: (0, 10) for e in envs})
    seen = []

    async def run(self, stop=None):
        seen.append(self.transport._prefix)
        if len(seen) == 1:
            self.task_signal.set()
            await stop.wait()

    monkeypatch.setattr(rl_async.AsyncMiner, "run", run)
    wallet = SimpleNamespace(hotkey=bt.Keypair.create_from_uri("//Alice"))
    await asyncio.wait_for(run_async_miner(
        AsyncMinerConfig(environments=("math",)), validator_url="http://v", validator_hotkey="5Val", wallet=wallet,
        work_dir=str(tmp_path), fake_generator=True), 5)
    assert seen == ["/rl/a", "/rl/c"]                            # b pays most, on another model
