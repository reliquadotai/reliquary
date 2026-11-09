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

from reliquary.miner.rl_episode_client import (  # noqa: E402
    RL_FINAL_REFUSALS, RL_OPENS_SIDE_BY_SIDE, RL_OVERGENERATION_REFUSALS, RL_TASK_FINAL_REFUSALS, RL_THROTTLES,
    RlSignedEpisodeRunner,
)
from reliquary.miner.signed_episode import MAX_UNAVAILABLE, _Slot  # noqa: E402
from reliquary.protocol.sandbox_session import SessionRefused  # noqa: E402
from reliquary.sandbox.rl_capacity import USEFUL_HORIZON_S, YieldStats  # noqa: E402
from reliquary.sandbox.rl_routes import PRECOMMIT_STATUS, build_allocation_router  # noqa: E402
from reliquary.sandbox.routes import REFUSAL_STATUS  # noqa: E402
from reliquary.protocol.signatures import verify_rl_allocation_signature  # noqa: E402
from reliquary.sandbox.sessions import LIVE, SandboxPolicy  # noqa: E402
from tests.unit.episode_v2_fixtures import EPISODE, episode_contract, episode_precommit  # noqa: E402
from tests.unit.sandbox_fixtures import NOW  # noqa: E402

MINER = bt.Keypair.create_from_uri("//Alice")
VALIDATOR = bt.Keypair.create_from_uri("//Bob")
CONTRACT = episode_contract()
POLICY = CONTRACT.episode_policy(EPISODE)
P32, CAPACITY = 32, 2_160


# --- the miner's number of groups ------------------------------------------------------------------------------

def sign(binding: bytes) -> str:
    return MINER.sign(binding).hex()


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


class Ledger:
    def __init__(self, table):
        self.table, self.version = table, 0

    def snapshot(self):
        return self.table


def main_stats(concurrent: float) -> YieldStats:
    return YieldStats(useful=180, decided=200, useful_24h=int(concurrent * 48),
                      useful_s_24h=concurrent * USEFUL_HORIZON_S)


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


# --- the precommit ---------------------------------------------------------------------------------------------

# --- the closed loop: the miner's rule against the real quota, production regime -------------------------------

