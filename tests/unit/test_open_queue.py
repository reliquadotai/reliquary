"""The fair open queue of RL sandbox sessions at saturation."""
import asyncio
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.sandbox.open_queue import OpenQueue  # noqa: E402
from reliquary.miner.rl_episode_client import RL_FINAL_REFUSALS, RL_THROTTLES  # noqa: E402
from reliquary.miner.signed_episode import SignedSweEpisodeRunner  # noqa: E402
from reliquary.protocol.sandbox_session import SessionRefused  # noqa: E402
from reliquary.sandbox.fleet import Placement  # noqa: E402
from reliquary.sandbox.routes import REFUSAL_STATUS  # noqa: E402
from reliquary.sandbox.sessions import Grant, _placement_class  # noqa: E402
from tests.unit.sandbox_fixtures import ADDRESS, MACHINE  # noqa: E402
from tests.unit.test_rl_engagements import HOTKEY, build, close, engagement, open_  # noqa: E402
from tests.unit.test_rl_episode_wiring import _services  # noqa: E402
from tests.unit.test_sandbox_fleet import make  # noqa: E402

A, B = ("env-a", 500, 1), ("env-b", 500, 1)          # placement classes (env, cpu_millis, memory_bytes)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def queue(scores=None, pooled=(), **kwargs):
    scores = scores or {}
    kwargs.setdefault("fair_every", 0)
    return OpenQueue(rank=lambda hotkey: (hotkey, scores.get(hotkey, 1.0), hotkey in pooled), **kwargs)


# --- the queue alone -------------------------------------------------------------------------------------------

async def test_the_first_waiter_gets_the_turn_at_once_and_the_best_score_is_served_next():
    q = queue({"low": 1.0, "high": 9.0, "mid": 5.0})
    first = q.join("low", cls=A)
    assert first.granted and q.busy
    mid, high = q.join("mid", cls=A), q.join("high", cls=A)
    q.done(first, placed=True)
    assert high.granted and not mid.granted
    q.done(high, placed=True)
    assert mid.granted


async def test_a_wake_up_offers_each_waiter_one_attempt():
    q = queue({"a": 3.0, "b": 2.0, "c": 1.0})
    holder = q.join("h", cls=A)
    a, b, c = q.join("a", cls=A), q.join("b", cls=B), q.join("c", cls=A)
    q.done(holder, placed=True)                     # the turn passes on: a tries
    assert a.granted
    q.missed(a)                                     # no room for a: b tries at once, in the same sweep
    assert b.granted and not a.granted
    q.missed(b)
    assert c.granted
    q.missed(c)                                     # everyone tried: nobody holds the turn
    assert not any(t.granted for t in (a, b, c)) and not q._holder
    q.notify()                                      # a new wake-up: a new sweep, best first
    assert a.granted


async def test_a_newcomer_of_a_class_that_missed_waits_for_the_next_wake_up():
    q = queue({"old": 1.0, "new": 9.0})
    old = q.join("old", cls=A)
    q.missed(old)                                   # its class found no room in this sweep
    new = q.join("new", cls=A)
    assert not new.granted                          # it would not fit either, and must not jump the older waiter
    q.notify()
    assert new.granted                              # the new sweep: the best score first


async def test_a_keys_waiting_time_is_kept_across_its_retries():
    clock = Clock()
    q = queue({"old": 1.0, "new": 3.0, "holder": 1.0}, aging_s=10.0, clock=clock)
    holder = q.join("holder", cls=A)
    first = q.join("old", cls=A)
    clock.now += 15.0
    q.done(first)                                   # its HTTP request timed out: the miner retries
    clock.now += 15.0
    retry = q.join("old", cls=A)                    # 30 s of waiting: 1 x (1 + 3) = 4 > 3 x 1
    new = q.join("new", cls=A)
    q.done(holder, placed=True)
    assert retry.granted and not new.granted


async def test_aging_is_forgotten_after_a_key_stopped_waiting():
    clock = Clock()
    q = queue({"old": 1.0, "new": 3.0, "holder": 1.0}, aging_s=10.0, aging_forget_s=20.0, clock=clock)
    holder = q.join("holder", cls=A)
    q.done(q.join("old", cls=A))
    clock.now += 100.0                              # long gone: its age is forgotten
    retry, new = q.join("old", cls=A), q.join("new", cls=A)
    q.done(holder, placed=True)
    assert new.granted and not retry.granted


async def test_a_key_that_placed_starts_aging_again():
    clock = Clock()
    q = queue({"old": 1.0, "new": 3.0, "holder": 1.0}, aging_s=10.0, clock=clock)
    holder = q.join("holder", cls=A)
    q.join("old", cls=A)
    clock.now += 30.0                               # old has aged to 4 > 3
    q.done(holder, placed=True)
    placed = q._holder
    assert placed.key == "old"
    second = q.join("old", cls=A)                   # old waits again, then places its first ticket
    new = q.join("new", cls=A)
    q.done(placed, placed=True)                     # old's age restarts now: 1 < 3
    assert new.granted and not second.granted


async def test_one_turn_in_n_goes_to_a_pooled_key_least_recently_served_first():
    q = queue({"big": 9.0, "new": 0.0}, pooled={"new"}, fair_every=2)
    q.done(q.join("big", cls=A), placed=True)       # grant 1: big served
    holder = q.join("big", cls=A)                   # grant 2 (fair, no pooled waiter): the only waiter
    q.join("big", cls=A)
    new = q.join("new", cls=A)
    q.done(holder, placed=True)                     # grant 3: by score -> big
    assert not new.granted
    q.done(q._holder, placed=True)                  # grant 4 (fair): the pooled ticket
    assert new.granted


async def test_the_fair_turn_goes_to_the_pooled_key_served_least_recently():
    clock = Clock()
    q = queue({"big": 9.0}, pooled={"p1", "p2"}, fair_every=1, clock=clock)
    q.done(q.join("p1", cls=A), placed=True)        # p1 served at 100
    clock.now += 1.0
    holder = q.join("big", cls=A)
    p1, p2 = q.join("p1", cls=A), q.join("p2", cls=A)
    q.done(holder, placed=True)                     # every turn is fair: p2 was never served
    assert p2.granted and not p1.granted


async def test_a_fair_turn_never_serves_a_main_key_out_of_its_weight():
    q = queue({"big": 9.0, "split-1": 1.0, "split-2": 1.0}, fair_every=2)
    holder = q.join("big", cls=A)
    q.join("big", cls=A), q.join("big", cls=A)
    splits = [q.join("split-1", cls=A), q.join("split-2", cls=A)]
    q.done(holder, placed=True)                     # grant 2 (fair): no pooled waiter -> by weight: big
    q.done(q._holder, placed=True)                  # grant 3: by weight: big
    assert not any(t.granted for t in splits)       # more hotkeys bought no fair turn


async def test_the_queue_is_bounded_globally_per_key_by_headroom_and_by_engagement():
    q = queue(max_waiting=3, max_per_key=2)
    q.join("a", cls=A, engagement="rl:1:x:0"), q.join("a", cls=A)           # one holds the turn, one waits
    assert q.join("a", cls=A).reason == "open_queue_key_full"
    assert q.join("b", cls=A, engagement="rl:1:x:0").reason == "open_queued"
    q.join("b", cls=A), q.join("c", cls=A)
    full = q.join("d", cls=A)
    assert full.reason == "open_queue_full" and full.retry_after >= 1
    tight = OpenQueue(rank=lambda hotkey: (hotkey, 1.0, False), headroom=lambda hotkey: 1)
    tight.join("a", cls=A)
    assert tight.join("a", cls=A).reason == "open_queue_key_full"           # the quota's headroom bounds it
    none = OpenQueue(rank=lambda hotkey: (hotkey, 1.0, False), headroom=lambda hotkey: 0)
    assert none.join("a", cls=A).reason == "open_queue_key_full"            # no free ticket at zero headroom


async def test_a_waiter_past_its_deadline_leaves_with_a_clear_answer():
    q = queue()
    holder = q.join("h", cls=A)
    waiter = q.join("w", cls=A)
    assert await q.turn(waiter, q.deadline() - 30.0) is False
    assert q.waiting == 0 and holder.granted


async def test_a_waiter_is_woken_when_the_turn_reaches_it_and_a_holder_that_leaves_passes_it_on():
    q = queue({"h": 1.0, "w": 1.0})
    holder = q.join("h", cls=A)
    waiter = q.join("w", cls=A)
    waiting = asyncio.ensure_future(q.turn(waiter, q.deadline()))
    await asyncio.sleep(0)
    assert not waiting.done()
    q.done(holder)                                  # left without placing (refused, cancelled): the turn passes on
    assert await asyncio.wait_for(waiting, 1.0) is True
    assert q._holder is waiter and q.waiting == 0
    q.missed(waiter)                                # no room: it waits again for the next wake-up
    again = asyncio.ensure_future(q.turn(waiter, q.deadline()))
    await asyncio.sleep(0)
    assert not again.done()
    q.notify()
    assert await asyncio.wait_for(again, 1.0) is True


async def test_done_is_idempotent_and_frees_the_engagement():
    q = queue()
    ticket = q.join("a", cls=A, engagement="rl:1:x:0")
    q.done(ticket)
    q.done(ticket)
    assert not q.busy and not q.waiting_in(A)
    assert q.join("a", cls=A, engagement="rl:1:x:0").granted


# --- the issuer with the queue ---------------------------------------------------------------------------------

def opening(env, seed, request_id):
    return asyncio.ensure_future(env.issuer.open(hotkey=HOTKEY, request_id=request_id,
                                                 engagement=engagement(env.precommit, seed)))


async def until(predicate, seconds=2.0):
    for _ in range(int(seconds / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never met")


class Refusals:
    def __init__(self):
        self.seen = []

    def opened(self, record):
        pass

    def closed(self, record, final):
        pass

    def refused(self, hotkey, engagement, refusal):
        self.seen.append(refusal.reason)


async def test_an_open_waits_outside_the_lock_for_room_then_gets_it(tmp_path):
    env = build(tmp_path)
    env.issuer._queue = q = queue(wait_s=5.0)
    env.issuer._fleet.placement = None
    task = opening(env, 0, "a" * 32)
    await until(lambda: q.busy and not env.issuer._lock.locked() and q._holder is None)
    env.issuer._fleet.placement = Placement(MACHINE, ADDRESS)
    q.notify()
    assert isinstance(await task, Grant)
    assert not q.busy


async def test_no_room_until_the_deadline_is_refused_saturated_and_told_to_the_listeners(tmp_path):
    env = build(tmp_path)
    env.issuer._queue = q = queue(wait_s=0.05)
    env.issuer._events = listener = Refusals()
    env.issuer._fleet.placement = None
    refused = await env.issuer.open(hotkey=HOTKEY, request_id="a" * 32, engagement=engagement(env.precommit, 0))
    assert refused.reason == "sandbox_saturated" and refused.retry_after
    assert listener.seen == ["sandbox_saturated"]                 # the ledger makes the precommit neutral
    assert not q.busy


async def test_an_open_no_machine_could_ever_take_is_refused_at_once_and_never_queued(tmp_path):
    env = build(tmp_path)
    env.issuer._queue = q = queue(wait_s=5.0)
    env.issuer._events = listener = Refusals()
    env.issuer._fleet.placement = None
    env.issuer._fleet.placeable = False
    refused = await env.issuer.open(hotkey=HOTKEY, request_id="a" * 32, engagement=engagement(env.precommit, 0))
    assert refused.reason == "task_unplaceable" and refused.detail["why"] == "no machine can serve it"
    assert not refused.retry_after and not q.busy and listener.seen == ["task_unplaceable"]


async def test_a_stale_directory_while_queued_is_retryable_never_unplaceable(tmp_path):
    env = build(tmp_path)
    env.issuer._queue = q = queue(wait_s=5.0)
    env.issuer._events = listener = Refusals()
    env.issuer._fleet.placement = None
    task = opening(env, 0, "a" * 32)
    await until(lambda: q.waiting == 1 and q._holder is None and not env.issuer._lock.locked())
    env.issuer._fleet.ready = False                  # our directory went stale while it waited
    env.issuer._fleet.placeable = False
    q.notify()
    refused = await task
    assert refused.reason == "directory_unavailable" and refused.retry_after
    assert listener.seen == [] and not q.busy


async def test_a_queue_full_refusal_is_a_capacity_refusal_but_a_second_ticket_is_not(tmp_path):
    env = build(tmp_path)
    env.issuer._events = listener = Refusals()
    env.issuer._fleet.placement = None
    env.issuer._queue = q = queue(wait_s=5.0, max_waiting=1)
    first = opening(env, 0, "a" * 32)
    await until(lambda: q.waiting == 1 and q._holder is None and not env.issuer._lock.locked())
    full = await env.issuer.open(hotkey=HOTKEY, request_id="b" * 32, engagement=engagement(env.precommit, 1))
    assert full.reason == "open_queue_full" and full.retry_after
    again = await env.issuer.open(hotkey=HOTKEY, request_id="c" * 32, engagement=engagement(env.precommit, 0))
    assert again.reason in {"open_queued", "open_queue_full"}
    assert listener.seen.count("open_queue_full") >= 1 and "open_queued" not in listener.seen
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not q.busy


async def test_an_unplaceable_high_score_waiter_never_blocks_a_placeable_open(tmp_path):
    env = build(tmp_path)
    env.issuer._queue = q = queue({"5Rich": 1000.0, HOTKEY: 1.0}, wait_s=5.0)
    rich = q.join("5Rich", cls=("reliquary-swe", 4000, 4))      # holds the turn, waits in another class
    assert rich.granted
    grant = await env.issuer.open(hotkey=HOTKEY, request_id="a" * 32, engagement=engagement(env.precommit, 0))
    assert isinstance(grant, Grant)                  # another class: placed at once, not behind the rich waiter


async def test_a_new_open_does_not_jump_a_waiter_of_its_own_class(tmp_path):
    env = build(tmp_path)
    env.issuer._queue = q = queue(wait_s=5.0)
    env.issuer._fleet.placement = None
    waiting = opening(env, 0, "a" * 32)              # no room: it tried, and waits in its class
    await until(lambda: q.waiting == 1 and q._holder is None and not env.issuer._lock.locked())
    picked = len(env.issuer._fleet.picked)
    env.issuer._fleet.placement = Placement(MACHINE, ADDRESS)
    late = opening(env, 1, "b" * 32)                 # room now, but a waiter of its class is ahead
    await until(lambda: q.waiting == 2)
    assert not late.done() and len(env.issuer._fleet.picked) == picked
    q.notify()                                       # the wake-up: the older waiter first, then the late one
    assert isinstance(await waiting, Grant) and isinstance(await late, Grant)


async def test_a_newcomer_with_another_image_queues_behind_a_better_waiter_of_its_class(tmp_path):
    env = build(tmp_path)
    env.issuer._queue = q = queue({"5Rich": 9.0, HOTKEY: 1.0}, wait_s=5.0)
    terms = await env.issuer._engagements["rl_precommit"].terms(HOTKEY, engagement(env.precommit, 1))
    cls = _placement_class(terms)
    assert terms.image not in cls                    # per-task images share their env's class
    rich = q.join("5Rich", cls=cls)                  # a better waiter of the class (another task's image)
    q.missed(rich)                                   # no room for its image now: it waits for the next wake-up
    late = opening(env, 1, "b" * 32)                 # room exists for this image, but the class has a waiter
    await until(lambda: q.waiting == 2)
    q.notify()
    assert rich.granted and not late.done()          # the better waiter tries first
    q.done(rich, placed=True)
    assert isinstance(await late, Grant)


async def test_a_cancelled_waiter_passes_its_turn(tmp_path):
    env = build(tmp_path)
    env.issuer._queue = q = queue(wait_s=5.0)
    env.issuer._fleet.placement = None
    gone = opening(env, 0, "a" * 32)
    await until(lambda: q.waiting == 1 and q._holder is None)
    stays = opening(env, 1, "b" * 32)
    await until(lambda: q.waiting == 2)
    gone.cancel()                                    # the miner's request was cut: its ticket must leave
    with pytest.raises(asyncio.CancelledError):
        await gone
    assert q.waiting == 1
    env.issuer._fleet.placement = Placement(MACHINE, ADDRESS)
    q.notify()
    assert isinstance(await asyncio.wait_for(stays, 2.0), Grant)
    assert not q.busy


async def test_a_waiter_cancelled_while_it_holds_the_turn_hands_it_on(tmp_path):
    env = build(tmp_path)
    env.issuer._queue = q = queue({"5Other": 0.5, HOTKEY: 1.0}, wait_s=5.0)
    env.issuer._fleet.placement = None
    gone = opening(env, 0, "a" * 32)
    await until(lambda: q.waiting == 1 and q._holder is None and not env.issuer._lock.locked())
    other = q.join("5Other", cls=_placement_class(
        await env.issuer._engagements["rl_precommit"].terms(HOTKEY, engagement(env.precommit, 1))))
    async with env.issuer._lock:                     # the holder is granted, then blocks on the issuer lock
        q.notify()
        assert q._holder is not None and q._holder.key == HOTKEY and not other.granted
        await asyncio.sleep(0.02)
        gone.cancel()
        with pytest.raises(asyncio.CancelledError):
            await gone
    assert other.granted                             # the turn passed on at once


def test_a_close_a_voided_machine_and_a_lapse_each_wake_the_queue(tmp_path):
    env = build(tmp_path)
    woken = []
    env.issuer._queue = q = queue()
    q.notify = lambda: woken.append(1)
    first, second, third = (open_(env, seed) for seed in range(3))   # room: placed directly, no wake-up
    assert all(isinstance(grant, Grant) for grant in (first, second, third)) and woken == []
    assert close(env, first, "aborted")["state"] == "aborted"         # a freed slot
    assert woken == [1]

    async def void():
        env.issuer.void_machine(MACHINE)
        await env.issuer.drain(2.0)
    asyncio.run(void())                                               # a drained machine's sessions
    assert woken == [1, 1]
    env.issuer._clock.now += 10 ** 6
    asyncio.run(env.issuer.maintain())                                # nothing live left to lapse: no wake-up
    assert woken == [1, 1]


def test_a_lapse_wakes_the_queue(tmp_path):
    env = build(tmp_path)
    woken = []
    env.issuer._queue = q = queue()
    q.notify = lambda: woken.append(1)
    assert isinstance(open_(env, 0), Grant)
    env.issuer._clock.now += 10 ** 6                                  # past its token's expiry
    asyncio.run(env.issuer.maintain())
    assert woken == [1]


def test_without_a_queue_no_room_is_refused_at_once_as_before(tmp_path):
    env = build(tmp_path)
    env.issuer._fleet.placement = None
    refused = asyncio.run(env.issuer.open(hotkey=HOTKEY, request_id="a" * 32,
                                          engagement=engagement(env.precommit, 0)))
    assert refused.reason == "sandbox_capacity" and "why" not in refused.detail


def test_the_queue_refusals_are_retried_by_the_miner():
    assert REFUSAL_STATUS["sandbox_saturated"] == REFUSAL_STATUS["open_queue_full"] == 503
    assert REFUSAL_STATUS["open_queue_key_full"] == REFUSAL_STATUS["open_queued"] == 429
    assert REFUSAL_STATUS["task_unplaceable"] == 409


def test_the_2c_miner_reopens_under_saturation_and_drops_an_unplaceable_seed():
    assert {"sandbox_saturated", "open_queue_full"} <= RL_THROTTLES
    assert "task_unplaceable" in RL_FINAL_REFUSALS
    runner = SimpleNamespace(validator_throttles=RL_THROTTLES, _backoff=0, _unavailable=0,
                             _hold_opens=lambda seconds: None)
    for _ in range(12):                              # more than MAX_UNAVAILABLE in a row
        answer = SignedSweEpisodeRunner._note_refusal(
            runner, SessionRefused("sandbox_saturated", {}, status=503, retry_after=10))
        assert answer.reason == "sandbox_saturated"  # never validator_unavailable


def test_the_wiring_builds_the_queue_with_the_quota_and_wakes_it_on_each_poll(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    assert isinstance(services.queue, OpenQueue) and services.issuer._queue is services.queue
    assert services.fleet.on_polled == services.queue.notify
    assert services.queue._headroom("5Nobody") >= 0            # headroom(h) -> quota.headroom(h, now)


def test_without_a_yield_quota_the_wiring_builds_no_queue(tmp_path):
    services, _ = _services(tmp_path)
    assert services.queue is None and services.issuer._queue is None and services.fleet.on_polled is None


def test_the_poll_listener_fires_on_an_early_return_too(tmp_path):
    fleet, _machine, _served, _clock = make(tmp_path)        # no machine answers: poll_once returns early
    calls = []
    fleet.on_polled = lambda: calls.append(1)
    asyncio.run(fleet.poll_once())
    assert calls == [1]
