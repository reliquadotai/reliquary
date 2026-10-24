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
    EpisodeMinerConfig, EpisodeMiningLoop, allocation_groups, allocation_reader, groups_to_run,
)
from reliquary.miner.rl_episode_client import (  # noqa: E402
    RL_FINAL_REFUSALS, RL_OPENS_SIDE_BY_SIDE, RL_OVERGENERATION_REFUSALS, RL_TASK_FINAL_REFUSALS, RL_THROTTLES,
    RlSignedEpisodeRunner,
)
from reliquary.miner.signed_episode import MAX_UNAVAILABLE, _Slot  # noqa: E402
from reliquary.protocol.sandbox_session import SessionRefused  # noqa: E402
from reliquary.sandbox.rl_capacity import USEFUL_HORIZON_S, YieldStats  # noqa: E402
from reliquary.sandbox.rl_quota import YieldQuota  # noqa: E402
from reliquary.sandbox.rl_routes import PRECOMMIT_STATUS, build_allocation_router  # noqa: E402
from reliquary.sandbox.routes import REFUSAL_STATUS  # noqa: E402
from reliquary.protocol.signatures import verify_rl_allocation_signature  # noqa: E402
from reliquary.sandbox.sessions import LIVE, SandboxPolicy  # noqa: E402
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


def sign(binding: bytes) -> str:
    return MINER.sign(binding).hex()


def test_the_allocation_reader_signs_its_request_and_returns_its_envs_groups():
    answers = iter([(200, {"groups": {EPISODE: 7, "other": 2}}), (200, {"groups": {EPISODE: 3, "e2": 5}}),
                    (503, {"reason": "x"}), (200, {"groups": {EPISODE: True}}), (200, {"groups": None})])
    bodies = []

    def handle(request):
        import json

        assert request.method == "POST" and request.url.path == "/rl/sessions/allocation"
        bodies.append(json.loads(request.content))
        status, body = next(answers)
        return httpx.Response(status, json=body)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            read = allocation_reader("http://validator.test/", MINER.ss58_address, client=client, sign_binding=sign,
                                     validator_hotkey=VALIDATOR.ss58_address, environments=(EPISODE, "e2"),
                                     clock=lambda: float(NOW))
            return [await read() for _ in range(5)]

    assert asyncio.run(go()) == [7, 3, None, None, None]                # its env's number; the fewest of its envs
    body = bodies[0]
    assert body["miner_hotkey"] == MINER.ss58_address and body["at"] == NOW
    assert verify_rl_allocation_signature(MINER.ss58_address, at=NOW, signature=body["signature"],
                                          validator_hotkey=VALIDATOR.ss58_address, path="/rl/sessions/allocation")


# --- the route -------------------------------------------------------------------------------------------------

def allocation_client(allocation_of, *, clock=lambda: float(NOW), registration=None):
    app = FastAPI()
    app.include_router(build_allocation_router(allocation_of, validator_hotkey=VALIDATOR.ss58_address,
                                               policy=SandboxPolicy(), clock=clock, registration=registration))
    return TestClient(app)


def body_of(keypair=MINER, *, hotkey=None, at=NOW, validator=VALIDATOR, path="/rl/sessions/allocation"):
    from reliquary.protocol.signatures import build_rl_allocation_binding

    hotkey = hotkey or keypair.ss58_address
    binding = build_rl_allocation_binding(hotkey=hotkey, at=at, validator_hotkey=validator.ss58_address, path=path)
    return {"miner_hotkey": hotkey, "at": at, "signature": keypair.sign(binding).hex()}


def test_the_route_answers_only_the_signed_callers_own_allocation():
    asked = []

    def allocation_of(hotkey):
        asked.append(hotkey)
        return {"hotkey": hotkey, "lane": "main", "sessions": 320, "groups": {EPISODE: 10}, "pool_seeds": {EPISODE: 32}}

    client = allocation_client(allocation_of)
    first = body_of()
    answer = client.post("/rl/sessions/allocation", json=first)
    assert answer.status_code == 200 and answer.json()["groups"] == {EPISODE: 10}
    assert asked == [MINER.ss58_address]
    unsigned = client.post("/rl/sessions/allocation", json={**body_of(at=NOW + 1), "signature": ""})
    assert unsigned.status_code == 401 and unsigned.json()["reason"] == "signature_required"
    other = bt.Keypair.create_from_uri("//Eve")                           # Eve signs a request naming Alice
    foreign = client.post("/rl/sessions/allocation", json=body_of(other, hotkey=MINER.ss58_address, at=NOW + 2))
    assert foreign.status_code == 403 and foreign.json()["reason"] == "bad_signature"
    elsewhere = client.post("/rl/sessions/allocation", json=body_of(at=NOW + 3, validator=other))
    assert elsewhere.status_code == 403                                   # signed for another validator
    stale = client.post("/rl/sessions/allocation", json=body_of(at=NOW - 10_000))
    assert stale.status_code == 400 and stale.json()["reason"] == "stale_request"
    replay = client.post("/rl/sessions/allocation", json=first)          # the same signed request again
    assert replay.status_code == 409 and replay.json()["reason"] == "request_reused"
    assert asked == [MINER.ss58_address]                                  # nobody else learnt anything


def test_the_route_refuses_an_unregistered_hotkey_and_bounds_each_hotkeys_minute():
    from reliquary.sandbox.rl_routes import ALLOCATION_MAX_PER_MINUTE

    from reliquary.validator.corpus_registration import NOT_REGISTERED

    async def gate(hotkey):
        return None if hotkey == MINER.ss58_address else NOT_REGISTERED

    client = allocation_client(lambda hotkey: {"groups": {}}, registration=gate)
    other = bt.Keypair.create_from_uri("//Eve")
    assert client.post("/rl/sessions/allocation", json=body_of(other)).status_code == 403
    codes = [client.post("/rl/sessions/allocation", json=body_of()).status_code
             for _ in range(ALLOCATION_MAX_PER_MINUTE + 1)]
    assert codes[:-1] == [200] * ALLOCATION_MAX_PER_MINUTE and codes[-1] == 429


def test_the_wiring_serves_the_signed_allocation_route(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    paths = [getattr(route, "path", "") for router in services.routers for route in router.routes]
    assert "/rl/sessions/allocation" in paths
    app = FastAPI()
    for router in services.routers:
        app.include_router(router)
    client = TestClient(app)
    import time as _time

    body = body_of(at=int(_time.time()))
    answer = client.post("/rl/sessions/allocation", json=body)
    assert answer.status_code == 200, answer.json()
    assert answer.json()["groups"] is None                            # no machine report yet: the miner keeps its own
    assert client.get(f"/rl/sessions/allocation/{MINER.ss58_address}").status_code in (404, 405)


def test_without_a_yield_quota_no_allocation_route(tmp_path):
    services, _ = _services(tmp_path)
    paths = [getattr(route, "path", "") for router in services.routers for route in router.routes]
    assert not any("/sessions/allocation" in path for path in paths)


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
    answer = _allocation_of(quota, {EPISODE: P32}, clock=lambda: clock.now)("5Big")
    assert answer["lane"] == "main" and answer["groups"][EPISODE] > 4    # nothing asked yet: its share, not 1 group
    told = []

    async def allocation():
        told.append(answer["groups"][EPISODE])
        return answer["groups"][EPISODE]

    miner = loop(groups_in_flight=4, allocation=allocation)
    asyncio.run(miner._follow_allocation())
    assert miner._cap == 4 and told
    fresh = _allocation_of(quota, {EPISODE: P32})("5Fresh")
    assert fresh["lane"] == "newcomer" and fresh["groups"] == {EPISODE: 2}    # a newcomer's maximum


def test_the_groups_are_counted_in_each_envs_own_pool_and_served_from_a_snapshot():
    clock = SimpleNamespace(now=float(NOW))
    quota = YieldQuota(ledger=Ledger({}), pool_seeds=P32, group_size=P32 // 2, capacity=lambda: CAPACITY,
                       clock=lambda: clock.now)
    computed = []
    sessions_allowed = quota.sessions_allowed
    quota.sessions_allowed = lambda hotkey: computed.append(hotkey) or sessions_allowed(hotkey)
    answer_of = _allocation_of(quota, {"wide": 32, "narrow": 8}, clock=lambda: clock.now, ttl_s=5.0)
    answer = answer_of("5Fresh")                                       # a newcomer: 64 sessions
    assert answer["sessions"] == 64 and answer["groups"] == {"wide": 2, "narrow": 8}
    for _ in range(50):
        answer_of("5Fresh")
    assert computed == ["5Fresh"]                                      # one computation per snapshot
    clock.now += 5.0
    answer_of("5Fresh")
    assert computed == ["5Fresh", "5Fresh"]
    assert allocation_groups(answer, ("narrow",)) == 8 and allocation_groups(answer, ("wide", "narrow")) == 2


def test_an_unsized_fleet_tells_no_number():
    quota = YieldQuota(ledger=Ledger({"5Big": main_stats(300.0)}), pool_seeds=P32, group_size=P32 // 2,
                       capacity=lambda: 0, clock=lambda: float(NOW))
    assert _allocation_of(quota, {EPISODE: P32})("5Big")["groups"] is None


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


@pytest.mark.parametrize("reason", sorted(RL_OVERGENERATION_REFUSALS))
def test_a_group_with_m_usable_episodes_ends_its_opens_when_the_quota_has_no_room_for_more(reason):
    # Seeds beyond M are over-generation: the quota refuses them only when nothing has room for one more, and only
    # ever to a group that holds its M. Once M episodes ended usable, reopening them would hold the group's graded
    # sessions for room only its own submission frees (every group at M waiting on the others): the group plays from
    # the seeds it has, and its other seeds are not sent.
    m = POLICY.pool_seeds // 2

    def answer(seed, nth, total):
        if seed >= m and total <= 200:          # (past that the validator grants: a runner that kept reopening)
            raise refused(reason, 10.0)
        runner.note_admissible(seed, True)      # (its episode played at once and passed the group's check)
        return grant(seed)

    gate = Gate(answer)
    runner, _ = runner_for(gate)
    results = open_all(runner, range(2 * m))
    assert [r["session_id"] for r in results[:m]] == [f"s-{seed}" for seed in range(m)], results
    assert all(isinstance(r, SessionRefused) and r.reason == reason for r in results[m:]), results[m:]
    assert len([seed for seed in gate.sent if seed >= m]) <= 2 * RL_OPENS_SIDE_BY_SIDE    # not one per retry


def test_an_inadmissible_episode_is_not_counted_and_the_seeds_still_waiting_are_sent_again():
    # M episodes passed the group's check, one is then found inadmissible: below M again, the seeds the
    # over-generation refusal was holding back are asked for once more.
    m = POLICY.pool_seeds // 2
    runner, _ = runner_for(Gate(lambda seed, nth, total: grant(seed)))
    for seed in range(m):
        runner.note_admissible(seed, True)
    runner._overgeneration = refused("quota_live", 10.0)
    assert runner._task_refusal().reason == "quota_live"
    runner.note_admissible(3, False)
    assert runner._task_refusal() is None


@pytest.mark.parametrize("reason", sorted(RL_OVERGENERATION_REFUSALS))
def test_a_group_whose_sessions_still_play_reopens_a_seed_the_quota_refused(reason):
    # Below M usable episodes a session still playing may fail (its box never starts): the quota then admits the
    # group's next seed, so the refused seed is reopened, never given up.
    m = POLICY.pool_seeds // 2

    def answer(seed, nth, total):
        if seed >= m and nth <= 2:
            raise refused(reason, 10.0)
        return grant(seed)

    gate = Gate(answer)
    runner, _ = runner_for(gate)
    results = open_all(runner, range(2 * m))
    assert [r["session_id"] for r in results] == [f"s-{seed}" for seed in range(2 * m)], results


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


@pytest.mark.parametrize("reason", ["precommit_unknown", "precommit_stale", "session_too_long"])
def test_a_refusal_about_the_precommit_ends_every_seed_after_one_open(reason):
    def answer(seed, nth, total):
        raise refused(reason)

    gate = Gate(answer)
    runner, _ = runner_for(gate)
    results = open_all(runner, range(16))
    assert all(isinstance(r, SessionRefused) and r.reason == reason for r in results)
    assert len(gate.sent) == 1


def test_a_transport_failure_resends_the_same_open_and_counts_toward_the_stop():
    import json

    bodies = []

    def answer(seed, nth, total):
        if total <= 3:
            raise httpx.ConnectError("unreachable")
        return grant(seed)

    gate = Gate(answer)
    original = gate.open

    def recording(body):
        bodies.append(body)
        return original(body)

    gate.open = recording
    runner, now = runner_for(gate)
    (result,) = open_all(runner, [0])
    assert result == grant(0)
    assert len({body["request_id"] for body in bodies}) == 1 and len(bodies) == 4      # the SAME open, resent
    assert runner._unavailable == 0 and now[0] >= 1 + 2 + 4                           # after the back-off
    assert json.dumps(bodies[0]["engagement"]) == json.dumps(bodies[-1]["engagement"])


def test_a_validator_unreachable_for_good_stops_the_seed_and_keeps_the_open_unconfirmed():
    def answer(seed, nth, total):
        raise httpx.ConnectError("unreachable")

    gate = Gate(answer)
    runner, _ = runner_for(gate)
    (result,) = open_all(runner, [0])
    assert isinstance(result, SessionRefused) and result.reason == "validator_unavailable"
    assert len(gate.sent) == MAX_UNAVAILABLE and len(runner._unconfirmed) == 1        # a grant it got is closed later


def test_a_pause_that_lands_while_an_open_waits_for_a_side_by_side_slot_is_honoured():
    clock = []

    def answer(seed, nth, total):
        if seed == 0:
            return grant(seed)
        if seed == 1 and nth == 1:
            time.sleep(0.05)
            raise refused("quota_live", 10.0)                          # a pause while seed 5 waits for a slot
        time.sleep(0.2)
        return grant(seed)

    gate = Gate(answer)
    runner, now = runner_for(gate)
    body_of_open = runner._open_body

    def timed(index, request_id):                                     # when the runner decides to send (its loop)
        clock.append((index, now[0]))
        return body_of_open(index, request_id)

    runner._open_body = timed
    results = open_all(runner, range(8))
    assert all(isinstance(r, dict) for r in results)
    assert sorted(seed for seed, _ in clock[:5]) == [0, 1, 2, 3, 4] and all(at == 0.0 for _, at in clock[:5])
    later = [at for _, at in clock[5:]]                                # every send after seed 1's refusal at 0 s
    assert later and all(at >= 10.0 for at in later), clock


def test_a_grant_sent_before_a_throttle_does_not_reopen_side_by_side_sending():
    runner, _ = runner_for(Gate(lambda seed, nth, total: grant(seed)))
    runner._fanning = True
    before = runner._throttles
    runner._note_refusal(refused("quota_live", 10.0))                   # a throttle meanwhile
    runner._granted(before)                                            # the older open's grant lands
    assert runner._fanning is False
    runner._granted(runner._throttles)                                 # a probe granted after it: side by side again
    assert runner._fanning is True


def test_blocked_opens_of_many_groups_never_hold_the_threads_closes_need():
    from reliquary.miner.rl_episode_client import RL_OPENS_IN_FLIGHT

    release = threading.Event()

    def answer(seed, nth, total):
        if seed == 0:
            return grant(seed)
        release.wait(10)                                               # waiting in the validator's open queue
        return grant(seed)

    gates = [Gate(answer) for _ in range(8)]
    runners = [runner_for(gate)[0] for gate in gates]

    async def go():
        tasks = [asyncio.ensure_future(runner._open(seed, _Slot(runner._live)))
                 for runner in runners for seed in range(8)]
        await asyncio.sleep(0.5)                                       # every group started, its opens blocked
        blocked = sum(gate.live for gate in gates)
        closed = await asyncio.wait_for(runners[0]._close("s-0", "final", None), 5)    # a close still goes through
        release.set()
        results = await asyncio.gather(*tasks)
        return blocked, closed, results

    blocked, closed, results = asyncio.run(go())
    assert closed == {"state": "closed"}
    assert RL_OPENS_SIDE_BY_SIDE < blocked <= RL_OPENS_IN_FLIGHT        # all groups' opens, capped process-wide
    assert len(results) == 64 and all(isinstance(r, dict) for r in results)


def test_a_group_refused_its_envs_share_before_it_started_frees_its_slot():
    def answer(seed, nth, total):
        raise refused("quota_env", 10.0)

    gate = Gate(answer)
    runner, _ = runner_for(gate)
    results = open_all(runner, range(16))
    assert all(isinstance(r, SessionRefused) and r.reason == "quota_env" for r in results)
    assert len(gate.sent) == 1 and runner.start_refused.reason == "quota_env"


def test_a_started_group_reopens_its_seeds_after_quota_env():
    def answer(seed, nth, total):
        if seed >= 1 and nth == 1:
            raise refused("quota_env", 10.0)
        return grant(seed)

    gate = Gate(answer)
    runner, _ = runner_for(gate)
    results = open_all(runner, range(4))
    assert all(isinstance(r, dict) for r in results) and runner.start_refused is None


def test_the_group_miner_raises_a_group_that_could_not_start_and_the_loop_moves_to_another_env(tmp_path):
    from reliquary.constants import M_ROLLOUTS
    from reliquary.miner.episode_mining import ENV_PAUSE_S
    from tests.unit.test_episode_group_miner import Clock, alternating, world

    w = world(tmp_path, alternating(2 * M_ROLLOUTS), refused=set(range(2 * M_ROLLOUTS)), clock=Clock())
    w.runner.start_refused = SessionRefused("quota_env", {}, status=429)
    with pytest.raises(SessionRefused) as ended:
        w.mine(open_until=NOW + 3600)
    assert ended.value.reason == "quota_env" and w.submitted == []

    clock = SimpleNamespace(now=1000.0)

    async def read_state():
        return None

    async def build_stack(view):
        raise AssertionError("not reached")

    miner = EpisodeMiningLoop(environments=("swe", "tmax"), read_state=read_state, build_stack=build_stack,
                              sizes=lambda env: 1, clock=lambda: clock.now)

    async def fail():
        raise ended.value

    async def finish():
        task = asyncio.ensure_future(fail())
        miner._inflight[task] = (1, "swe", 3)
        task.add_done_callback(miner._finished)
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

    asyncio.run(finish())
    assert miner.open_environments() == ("tmax",)                      # the slot goes to the other env
    clock.now += ENV_PAUSE_S
    assert miner.open_environments() == ("swe", "tmax")


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
    told_of = _allocation_of(quota, {EPISODE: P32}, clock=lambda: clock.now, ttl_s=0.0)
    state = {h: {"run": ceiling, "groups": [], "made": 0} for h, (_w, ceiling) in keys.items()}
    held, run, made = {h: [] for h in keys}, {h: [] for h in keys}, [0]
    start = clock.now
    while clock.now < start + hours * 3_600:
        now = int(clock.now)
        for h, st in state.items():
            st["run"] = groups_to_run(keys[h][1], allocation_groups(told_of(h), (EPISODE,)), st["run"])
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
