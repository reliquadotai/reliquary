"""The reference miner follows the validator's allocation for its hotkey: it runs as many episode groups as its
potential allows (never above its own ceiling, never below one group), starts a group with one open at a time and
sends the rest side by side once it started, and handles every refusal of the RL routes (reopen the seed, drop the
seed, drop the task for the window) without ever giving up a half-opened group for a refusal of the validator's."""
import asyncio
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

bt = pytest.importorskip("bittensor")
attest = pytest.importorskip("reliquary_sandbox.attest")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from reliquary.miner.episode_mining import (  # noqa: E402
    EpisodeMinerConfig, EpisodeMiningLoop, allocation_reader, groups_to_run,
)
from reliquary.miner.rl_episode_client import (  # noqa: E402
    RL_FINAL_REFUSALS, RL_OPENS_SIDE_BY_SIDE, RL_TASK_FINAL_REFUSALS, RL_THROTTLES, RlSignedEpisodeRunner,
)
from reliquary.miner.signed_episode import MAX_UNAVAILABLE, _Slot  # noqa: E402
from reliquary.protocol.sandbox_session import SessionRefused  # noqa: E402
from reliquary.sandbox.rl_capacity import USEFUL_HORIZON_S, YieldStats  # noqa: E402
from reliquary.sandbox.rl_quota import YieldQuota  # noqa: E402
from reliquary.sandbox.rl_routes import PRECOMMIT_STATUS, build_allocation_router  # noqa: E402
from reliquary.sandbox.routes import REFUSAL_STATUS  # noqa: E402
from reliquary.sandbox.sessions import LIVE  # noqa: E402
from reliquary.validator.rl_sandbox_wiring import _allocation_of  # noqa: E402
from tests.unit.episode_v2_fixtures import EPISODE, episode_contract, episode_precommit  # noqa: E402
from tests.unit.sandbox_fixtures import NOW  # noqa: E402
from tests.unit.test_rl_episode_wiring import _services  # noqa: E402

MINER = bt.Keypair.create_from_uri("//Alice")
VALIDATOR = bt.Keypair.create_from_uri("//Bob")
CONTRACT = episode_contract()
POLICY = CONTRACT.episode_policy(EPISODE)
P32, CAPACITY = 32, 2_160


def loop(**kwargs):
    async def read_state():
        return None

    async def build_stack(view):
        raise AssertionError("not reached")

    return EpisodeMiningLoop(environments=("e",), read_state=read_state, build_stack=build_stack,
                             sizes=lambda env: 1, **kwargs)


# --- the miner's number of groups ------------------------------------------------------------------------------

def test_more_than_two_groups_in_flight_are_allowed():
    assert loop(groups_in_flight=8)._cap == 8
    with pytest.raises(ValueError):
        loop(groups_in_flight=0)
    config = EpisodeMinerConfig(environments=(EPISODE,), validator_url="http://v", validator_hotkey="5V",
                                groups_in_flight=12)
    assert config.groups_in_flight == 12
    with pytest.raises(ValueError):
        EpisodeMinerConfig(environments=(EPISODE,), validator_url="http://v", validator_hotkey="5V",
                           groups_in_flight=0)


def test_the_miner_follows_the_announced_allocation():
    answers = [5, None, 0, 40]

    async def allocation():
        return answers.pop(0)

    miner = loop(groups_in_flight=8, allocation=allocation)
    assert miner._cap == 8                                   # its ceiling, before any answer
    asyncio.run(miner._follow_allocation())
    assert miner._cap == 5
    asyncio.run(miner._follow_allocation())
    assert miner._cap == 5                                   # no answer: keep the last one
    asyncio.run(miner._follow_allocation())
    assert miner._cap == 1                                   # never below one: its requests are its demand
    asyncio.run(miner._follow_allocation())
    assert miner._cap == 8                                   # never above its own ceiling


def test_a_failing_or_malformed_allocation_keeps_the_current_number():
    async def broken():
        raise RuntimeError("down")

    async def boolean():
        return True

    for allocation in (broken, boolean):
        miner = loop(groups_in_flight=3, allocation=allocation)
        asyncio.run(miner._follow_allocation())
        assert miner._cap == 3


def test_the_ceiling_is_never_raised_by_the_validator():
    assert groups_to_run(4, 68, 4) == 4 and groups_to_run(4, 2, 4) == 2 and groups_to_run(4, None, 3) == 3
    assert groups_to_run(4, 0, 4) == 1


def test_the_allocation_is_read_at_every_poll_before_the_groups_are_filled():
    reads = []

    async def allocation():
        reads.append(1)
        return 2

    miner = loop(groups_in_flight=6, allocation=allocation)
    filled = []
    miner._fill = lambda view: filled.append(miner._cap)
    miner.stack, miner.stack_key = SimpleNamespace(), ("r", "s")
    view = SimpleNamespace(key=("r", "s"))
    import reliquary.miner.episode_mining as em

    original = em.episode_window
    em.episode_window = lambda state: view
    try:
        asyncio.run(miner.step())
        asyncio.run(miner.step())
    finally:
        em.episode_window = original
    assert reads == [1, 1] and filled == [2, 2]


def test_the_allocation_reader_returns_the_groups_or_none():
    answers = {"5Ok": (200, {"groups": 7}), "5Busy": (503, {"reason": "x"}), "5Bad": (200, {"groups": True}),
               "5Unknown": (200, {"groups": None})}

    def handle(request):
        hotkey = request.url.path.rsplit("/", 1)[1]
        assert request.url.path == f"/rl/sessions/allocation/{hotkey}"
        status, body = answers[hotkey]
        return httpx.Response(status, json=body)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return [await allocation_reader("http://validator.test/", hotkey, client=client)()
                    for hotkey in answers]

    assert asyncio.run(go()) == [7, None, None, None]


# --- the route -------------------------------------------------------------------------------------------------

def test_the_route_serves_a_hotkeys_allocation():
    app = FastAPI()
    app.include_router(build_allocation_router(lambda hotkey: {"hotkey": hotkey, "lane": "main", "sessions": 320,
                                                               "groups": 10, "pool_seeds": 32}))
    client = TestClient(app)
    body = client.get("/rl/sessions/allocation/5Hot").json()
    assert body["groups"] == 10 and body["lane"] == "main"
    refused = client.get("/rl/sessions/allocation/0OIl")             # not base58
    assert refused.status_code == 400 and refused.json()["reason"] == "hotkey_invalid"


def test_the_wiring_serves_the_allocation_route(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    paths = [getattr(route, "path", "") for router in services.routers for route in router.routes]
    assert any(path.endswith("/sessions/allocation/{hotkey}") for path in paths)
    app = FastAPI()
    for router in services.routers:
        app.include_router(router)
    body = TestClient(app).get("/rl/sessions/allocation/5Hot").json()
    assert body["groups"] is None                                     # no machine report yet: the miner keeps its own


def test_without_a_yield_quota_no_allocation_route(tmp_path):
    services, _ = _services(tmp_path)
    paths = [getattr(route, "path", "") for router in services.routers for route in router.routes]
    assert not any("/sessions/allocation/" in path for path in paths)


class Ledger:
    def __init__(self, table):
        self.table, self.version = table, 0

    def snapshot(self):
        return self.table


def main_stats(concurrent: float) -> YieldStats:
    return YieldStats(useful=180, decided=200, useful_24h=int(concurrent * 48),
                      useful_s_24h=concurrent * USEFUL_HORIZON_S)


def test_the_first_poll_before_any_open_tells_the_potential_never_one_group():
    clock = SimpleNamespace(now=float(NOW))
    quota = YieldQuota(ledger=Ledger({"5Big": main_stats(300.0), "5Rival": main_stats(300.0)}), pool_seeds=P32,
                       group_size=P32 // 2, capacity=lambda: CAPACITY, clock=lambda: clock.now)
    answer = _allocation_of(quota, P32)("5Big")
    assert answer["lane"] == "main" and answer["groups"] > 4         # nothing asked yet: its share, not 1 group
    told = []

    async def allocation():
        told.append(answer["groups"])
        return answer["groups"]

    miner = loop(groups_in_flight=4, allocation=allocation)
    asyncio.run(miner._follow_allocation())
    assert miner._cap == 4 and told
    fresh = _allocation_of(quota, P32)("5Fresh")
    assert fresh["lane"] == "newcomer" and fresh["groups"] == 2       # a newcomer's maximum


def test_an_unsized_fleet_tells_no_number():
    quota = YieldQuota(ledger=Ledger({"5Big": main_stats(300.0)}), pool_seeds=P32, group_size=P32 // 2,
                       capacity=lambda: 0, clock=lambda: float(NOW))
    assert _allocation_of(quota, P32)("5Big")["groups"] is None


# --- the opens of one precommit --------------------------------------------------------------------------------

def grant(seed):
    return {"session_id": f"s-{seed}", "token": {}, "gateway_url": "http://g:1", "expires_at": NOW + 4500}


class Gate:
    """The RL session routes, faked: ``answer(seed, nth send of that seed, sends so far)`` -> a grant, or raises."""

    prefix, validator_hotkey = "/rl", VALIDATOR.ss58_address

    def __init__(self, answer, delay=0.0):
        self.answer, self.delay = answer, delay
        self.sent, self.live, self.most, self.most_before_grant = [], 0, 0, 0
        self.granted = 0
        self.lock = threading.Lock()

    def open(self, body):
        seed = body["engagement"]["precommit"]["seed_index"]
        with self.lock:
            self.sent.append(seed)
            nth, total = self.sent.count(seed), len(self.sent)
            self.live += 1
            self.most = max(self.most, self.live)
            if not self.granted:
                self.most_before_grant = self.most
        try:
            if self.delay:
                time.sleep(self.delay)
            answer = self.answer(seed, nth, total)
            with self.lock:
                self.granted += 1
            return answer
        finally:
            with self.lock:
                self.live -= 1

    def close(self, session_id, body):
        return {"state": "closed"}


def runner_for(gate, **kwargs):
    now = [0.0]

    async def sleep(seconds):
        now[0] += seconds
        await asyncio.sleep(0)

    precommit = episode_precommit(CONTRACT, hotkey=MINER.ss58_address)
    runner = RlSignedEpisodeRunner(
        policy=POLICY, precommit=precommit, prompt="Write 42.", hotkey=MINER.ss58_address,
        sign_binding=lambda binding: MINER.sign(binding).hex(), sessions=gate, model_name="m",
        renderer_model_dir="/m", generate_url="http://127.0.0.1:1", sampling=SimpleNamespace(temperature=1.0, top_p=1.0),
        clock=lambda: NOW + now[0], monotonic=lambda: now[0], sleep=sleep, open_until=NOW + 10 ** 6, **kwargs)
    return runner, now


def open_all(runner, seeds):
    async def go():
        async with runner:
            return await asyncio.gather(*(runner._open(seed, _Slot(runner._live)) for seed in seeds),
                                        return_exceptions=True)

    return asyncio.run(go())


def refused(reason, retry_after=5.0):
    status = REFUSAL_STATUS.get(reason, 502)
    return SessionRefused(reason, {}, status=status, retry_after=retry_after if status in (429, 503) else None)


def test_a_group_that_cannot_start_sends_one_open_at_a_time_then_the_rest_side_by_side():
    def answer(seed, nth, total):
        if total <= 4:
            raise refused("quota_live", 10.0)                       # the group cannot start yet
        return grant(seed)

    gate = Gate(answer, delay=0.05)
    runner, now = runner_for(gate)
    results = open_all(runner, range(16))
    assert all(isinstance(result, dict) for result in results)
    assert len(gate.sent) == 16 + 4                                   # four refused probes, never one per seed
    assert gate.most_before_grant == 1                                # one at a time until it started
    assert 2 <= gate.most <= RL_OPENS_SIDE_BY_SIDE                    # then side by side, bounded
    assert now[0] >= 4 * 10.0


@pytest.mark.parametrize("reason", sorted({
    "quota_live", "quota_opens", "quota_pool", "quota_pool_opens", "quota_env", "sandbox_saturated",
    "open_queue_full", "open_queue_key_full", "open_queued", "directory_unavailable", "environment_unserved",
    "open_refused_rate", "open_busy"}))
def test_a_half_opened_group_is_never_given_up_for_a_refusal_of_the_validators(reason):
    def answer(seed, nth, total):
        if seed >= 4 and nth <= MAX_UNAVAILABLE + 2:                  # more refusals in a row than the stop count
            raise refused(reason)
        return grant(seed)

    gate = Gate(answer)
    runner, _ = runner_for(gate)
    results = open_all(runner, range(16))
    assert [r["session_id"] for r in results] == [f"s-{seed}" for seed in range(16)], results
    assert runner._unavailable == 0


def test_every_retryable_refusal_of_the_rl_routes_is_reopened():
    # Every 503 an RL open can answer is a named throttle (the intake's group claim is answered by no route);
    # every 429 is reopened by its status.
    retryable = {reason for reason, status in REFUSAL_STATUS.items() if status == 503} - {"precommit_claimed"}
    assert retryable <= RL_THROTTLES
    assert {"quota_live", "quota_opens", "quota_pool", "quota_pool_opens", "quota_env", "open_queue_key_full",
            "open_queued"} <= {reason for reason, status in REFUSAL_STATUS.items() if status == 429}
    assert not RL_TASK_FINAL_REFUSALS & RL_THROTTLES and RL_TASK_FINAL_REFUSALS <= RL_FINAL_REFUSALS
    # the precommit route: its 429 / 503 are sent again until the window ends, its 409 drop the task
    assert PRECOMMIT_STATUS["environment_unserved"] == PRECOMMIT_STATUS["fleet_unavailable"] == 503
    assert PRECOMMIT_STATUS["task_unservable"] == 409


def test_an_unplaceable_task_ends_every_seed_of_the_precommit_after_one_refused_open():
    def answer(seed, nth, total):
        raise refused("task_unplaceable")

    gate = Gate(answer)
    runner, _ = runner_for(gate)
    results = open_all(runner, range(16))
    assert all(isinstance(r, SessionRefused) and r.reason == "task_unplaceable" for r in results)
    assert len(gate.sent) == 1


def test_a_seed_final_refusal_ends_that_seed_only():
    def answer(seed, nth, total):
        if seed == 3:
            raise refused("engagement_taken")
        return grant(seed)

    gate = Gate(answer)
    runner, _ = runner_for(gate)
    results = open_all(runner, range(8))
    assert isinstance(results[3], SessionRefused) and results[3].reason == "engagement_taken"
    assert sum(isinstance(r, dict) for r in results) == 7


def test_a_brief_outage_while_the_group_opens_side_by_side_loses_no_seed():
    def answer(seed, nth, total):
        # the side-by-side sends fail together, then 8 probes (more failures than the stop count in all, fewer
        # counted ones), then it is back
        if 2 <= total <= 1 + RL_OPENS_SIDE_BY_SIDE + MAX_UNAVAILABLE - 2:
            raise SessionRefused("http_502", {}, status=502)
        return grant(seed)

    gate = Gate(answer, delay=0.05)
    runner, _ = runner_for(gate)
    results = open_all(runner, range(16))
    assert all(isinstance(r, dict) for r in results), results
    assert gate.most >= 2 and runner._unavailable == 0


def test_a_validator_that_stays_down_stops_the_opens_within_the_stop_count():
    def answer(seed, nth, total):
        if seed == 0:
            return grant(seed)
        raise SessionRefused("http_502", {}, status=502)

    gate = Gate(answer)
    runner, _ = runner_for(gate)
    results = open_all(runner, range(16))
    assert isinstance(results[0], dict)
    assert all(isinstance(r, SessionRefused) and r.reason == "validator_unavailable" for r in results[1:])
    assert len(gate.sent) <= 1 + 15 + MAX_UNAVAILABLE                  # nothing is sent once the hotkey stopped


# --- the precommit ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reason", ["environment_unserved", "fleet_unavailable", "window_not_open", "precommit_rate"])
def test_a_retryable_precommit_refusal_is_sent_again(tmp_path, reason):
    from reliquary.constants import M_ROLLOUTS
    from tests.unit.test_episode_group_miner import Clock, Sessions, alternating, world

    busy = [SessionRefused(reason, status=PRECOMMIT_STATUS[reason], retry_after=10.0) for _ in range(3)]
    w = world(tmp_path, alternating(2 * M_ROLLOUTS), sessions=Sessions(refuse=busy), clock=Clock())
    assert w.mine(open_until=NOW + 3600).accepted
    assert len(w.miner._sessions.bodies) == 4


def test_an_unservable_task_is_dropped_for_the_window_and_opens_nothing(tmp_path):
    from reliquary.constants import M_ROLLOUTS
    from tests.unit.test_episode_group_miner import Clock, Sessions, alternating, world

    w = world(tmp_path, alternating(2 * M_ROLLOUTS), clock=Clock(),
              sessions=Sessions(refuse=[SessionRefused("task_unservable", status=409)]))
    with pytest.raises(SessionRefused):
        w.mine(open_until=NOW + 3600)
    assert len(w.miner._sessions.bodies) == 1 and w.runner.entered == []


# --- the closed loop: the miner's rule against the real quota, production regime -------------------------------

def closed_loop(capacity, keys, life_s, *, hours=1.0, warm_s=1_200, poll_s=5, retry_s=10):
    """Each key {hotkey: (weight, ceiling)} runs ``groups_to_run(ceiling, told, current)`` groups of P32 seeds, told
    by the allocation route's own answer; a group's first open is a probe (one seed every ``retry_s`` until one is
    granted, as the runner does), then every seed is asked at once, a refused seed asked again every ``retry_s``;
    each group ends ``life_s`` after it started. The quota is the real one (refresh 1 s, production constants).
    Returns {hotkey: [held per poll]} and {hotkey: [groups run per poll]} after the warm-up."""
    from reliquary.protocol.service_episode import rl_engagement

    clock = SimpleNamespace(now=float(NOW))
    ledger = Ledger({h: main_stats(w) for h, (w, _ceiling) in keys.items()})
    quota = YieldQuota(ledger=ledger, pool_seeds=P32, group_size=P32 // 2, capacity=lambda: capacity,
                       clock=lambda: clock.now)
    told_of = _allocation_of(quota, P32)
    state = {h: {"run": ceiling, "groups": [], "made": 0} for h, (_w, ceiling) in keys.items()}
    held, run, made = {h: [] for h in keys}, {h: [] for h in keys}, [0]
    start = clock.now
    while clock.now < start + hours * 3_600:
        now = int(clock.now)
        for h, st in state.items():
            st["run"] = groups_to_run(keys[h][1], told_of(h)["groups"], st["run"])
            for group in [g for g in st["groups"] if now >= g["start"] + life_s]:
                for record in group["records"]:
                    record.state, record.closed_status, record.closed_at = "closed", "expired", now
                quota.closed(None, None)
                st["groups"].remove(group)
            while len(st["groups"]) < st["run"]:
                st["made"] += 1
                st["groups"].append({"sha": f"{10_000 * (list(keys).index(h) + 1) + st['made']:064x}",
                                     "start": now, "seeds": list(range(P32)), "records": [], "tried": -10**9})
            for group in st["groups"]:
                if not group["seeds"] or now - group["tried"] < retry_s:
                    continue
                group["tried"] = now
                for seed in list(group["seeds"]) if group["records"] else group["seeds"][:1]:
                    if quota.admit_locked(h, now, precommit_sha256=group["sha"], seed_index=seed) is None:
                        made[0] += 1
                        record = SimpleNamespace(session_id=f"c{made[0]}", hotkey=h, env="reliquary-swe",
                                                 engagement=rl_engagement(1, group["sha"], seed), issued_at=now,
                                                 expires_at=now + life_s + 600, state=LIVE, closed_status=None,
                                                 closed_at=None)
                        quota.opened(record)
                        group["records"].append(record)
                        group["seeds"].remove(seed)
                if len(group["records"]) == 1 and group["seeds"]:
                    group["tried"] = -10**9                            # the probe started it: the rest at once
        if clock.now - start >= warm_s:
            for h, st in state.items():
                held[h].append(len(quota._counted(h, now)))
                run[h].append(st["run"])
        clock.now += poll_s
    return held, run


@pytest.mark.parametrize("life_s", [1_800, 600])
def test_the_reference_miner_reaches_the_same_total_with_one_hotkey_or_ten(life_s):
    one_held, one_run = closed_loop(CAPACITY, {"5One": (300.0, 10)}, life_s)
    ten = {f"5S{n}": (30.0, 1) for n in range(10)}
    ten_held, ten_run = closed_loop(CAPACITY, ten, life_s)
    one_total = sum(one_held["5One"]) / len(one_held["5One"])
    ten_total = sum(sum(poll) for poll in zip(*ten_held.values())) / len(ten_held["5S0"])
    assert one_total > 0.9 * 10 * P32                                  # an idle fleet: its ceiling, not one group
    assert abs(one_total - ten_total) <= 0.1 * ten_total               # the same total, one hotkey or ten
    assert set(one_run["5One"]) == {10}                                # no oscillation: its ceiling throughout
    assert all(set(runs) == {1} for runs in ten_run.values())
