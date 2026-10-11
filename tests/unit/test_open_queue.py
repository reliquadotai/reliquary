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


async def test_one_placement_in_n_goes_to_a_pooled_key():
    q = queue({"big": 9.0, "new": 0.0}, pooled={"new"}, fair_every=3)
    holder = q.join("big", cls=A)
    q.join("big", cls=A), q.join("big", cls=A)
    new = q.join("new", cls=A)
    q.done(holder, placed=True)                     # 1 MAIN placement: by score -> big
    assert not new.granted
    q.done(q._holder, placed=True)                  # 2 MAIN placements: the pooled share is due
    assert new.granted


async def test_the_pooled_share_counts_placements_not_grants():
    q = queue({"big": 9.0}, pooled={"p1", "p2", "p3"}, fair_every=4)
    for _ in range(40):
        q.join("big", cls=A)
    for key in ("p1", "p2", "p3"):
        for _ in range(10):
            q.join(key, cls=B)
    placed = []
    while len(placed) < 40:
        ticket = q._holder
        if ticket.key == "p1":                      # p1's image has no room: its misses use no share up
            q.missed(ticket)
            continue
        placed.append(ticket.pooled)
        q.done(ticket, placed=True)
    assert sum(placed) == 10                        # exactly one placement in four, misses aside
    assert all(sum(placed[i:i + 4]) == 1 for i in range(0, 40, 4))


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
    env.issuer._queue = q = queue(wait_s=5.0, max_waiting=3)
    first = opening(env, 0, "a" * 32)
    await until(lambda: q.waiting == 1 and q._holder is None and not env.issuer._lock.locked())
    again = await env.issuer.open(hotkey=HOTKEY, request_id="c" * 32, engagement=engagement(env.precommit, 0))
    assert again.reason == "open_queued"
    others = [q.join("5Other", cls=("elsewhere", None, None)) for _ in range(3)]   # another miner fills it
    assert q.keys() == [HOTKEY, "5Other", "5Other"]
    full = await env.issuer.open(hotkey=HOTKEY, request_id="b" * 32, engagement=engagement(env.precommit, 1))
    assert full.reason == "open_queue_full" and full.retry_after
    assert listener.seen == ["open_queue_full"]
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    for ticket in others:
        q.done(ticket)
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


# --- the fleet cannot always judge, coalesced sweeps, the head's hold, the room checks ----------------------

from dataclasses import fields  # noqa: E402

from reliquary.sandbox.fleet import HEARTBEAT_MAX_AGE_S  # noqa: E402
from reliquary.sandbox.sessions import SandboxPolicy, SessionBook, SessionIssuer, _tier  # noqa: E402
from tests.unit.sandbox_fixtures import capacity_report, machine_document, signer  # noqa: E402
from tests.unit.test_sandbox_fleet import PICK  # noqa: E402
from tests.unit.test_sandbox_sessions import Clock as IssuerClock  # noqa: E402


def test_the_fleet_answers_cannot_judge_for_our_outages_and_never_only_from_fresh_reports(tmp_path):
    fleet, machine, served, clock = make(tmp_path)
    now = clock.now
    assert fleet.could_place(**PICK, now=now) is None              # a start: no report accepted yet
    served[ADDRESS] = capacity_report(machine, at=now)
    asyncio.run(fleet.poll_once())
    assert fleet.could_place(**PICK, now=now) is True
    assert fleet.could_place(**{**PICK, "image": "other@sha256:" + "b" * 64}, now=now) is False   # never
    fleet.drained.add(MACHINE)                                       # its holder drained for a moment
    assert fleet.could_place(**PICK, now=now) is None
    fleet.drained.discard(MACHINE)
    assert fleet.could_place(**PICK, now=now + HEARTBEAT_MAX_AGE_S + 1) is None    # every report stale
    served.clear()                                                   # no machine answers: our egress
    asyncio.run(fleet.poll_once())
    assert fleet._egress_suspect and fleet.could_place(**PICK, now=now) is None


def test_the_fleet_never_and_unknown_follow_what_each_machine_last_reported(tmp_path):
    fleet, machine, served, clock = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=clock.now, images=())
    asyncio.run(fleet.poll_once())
    assert fleet.could_place(**PICK, now=clock.now) is False         # fresh, and it does not serve the image
    fleet.drained.add(MACHINE)
    assert fleet.could_place(**PICK, now=clock.now) is None          # no fresh report left: cannot judge


def test_a_drained_or_stale_image_holder_is_cannot_judge_beside_a_fresh_machine_without_it(tmp_path):
    one, two = signer(tmp_path, "a", "ka"), signer(tmp_path, "b", "kb")
    documents = [machine_document(one, machine_id="m-a", address="http://a:1"),
                 machine_document(two, machine_id="m-b", address="http://b:1")]
    fleet, _, served, clock = make(tmp_path, documents=documents)
    served["http://a:1"] = capacity_report(one, at=clock.now, machine_id="m-a", address="http://a:1")
    served["http://b:1"] = capacity_report(two, at=clock.now, machine_id="m-b", address="http://b:1", images=())
    asyncio.run(fleet.poll_once())
    assert fleet.could_place(**PICK, now=clock.now) is True
    fleet.drained.add("m-a")                         # the image's only holder drained; m-b fresh, without it
    assert fleet.could_place(**PICK, now=clock.now) is None
    fleet.drained.discard("m-a")
    fleet._reports["m-a"]["at"] = clock.now - HEARTBEAT_MAX_AGE_S - 5         # its report went stale
    assert fleet.could_place(**PICK, now=clock.now) is None


async def test_an_open_the_fleet_cannot_judge_is_retryable_untold_and_not_counted(tmp_path):
    env = build(tmp_path)
    env.issuer._queue = q = queue(wait_s=5.0)
    env.issuer._events = listener = Refusals()
    env.issuer._fleet.placement = None
    env.issuer._fleet.placeable = None
    refused = await env.issuer.open(hotkey=HOTKEY, request_id="a" * 32, engagement=engagement(env.precommit, 0))
    assert refused.reason == "directory_unavailable" and refused.retry_after
    assert listener.seen == [] and not q.busy and HOTKEY not in env.issuer._refused_opens


async def test_saturation_refusals_do_not_count_toward_the_refused_open_rate(tmp_path):
    env = build(tmp_path)
    env.issuer._queue = queue(wait_s=0.02)
    env.issuer._fleet.placement = None
    refused = await env.issuer.open(hotkey=HOTKEY, request_id="a" * 32, engagement=engagement(env.precommit, 0))
    assert refused.reason == "sandbox_saturated" and HOTKEY not in env.issuer._refused_opens
    env.issuer._fleet.placeable = False
    final = await env.issuer.open(hotkey=HOTKEY, request_id="b" * 32, engagement=engagement(env.precommit, 1))
    assert final.reason == "task_unplaceable" and len(env.issuer._refused_opens[HOTKEY]) == 1   # the miner's to stop


def test_without_a_queue_refused_opens_count_as_before(tmp_path):
    env = build(tmp_path)
    env.issuer._fleet.placement = None
    assert open_(env, 0).reason == "sandbox_capacity" and len(env.issuer._refused_opens[HOTKEY]) == 1


async def test_wake_ups_during_a_sweep_are_coalesced_into_one_follow_up_sweep():
    q = queue({"a": 3.0, "b": 2.0})
    a, b = q.join("a", cls=A), q.join("b", cls=A)
    sweep = q._sweep
    for _ in range(10):
        q.notify()                                  # a burst of closes while a holds the turn
    assert q._sweep == sweep and a.granted
    q.missed(a)                                     # the running sweep goes on: b
    assert b.granted and q._sweep == sweep
    q.missed(b)                                     # it ends: ONE follow-up sweep, best first
    assert q._sweep == sweep + 1 and a.granted
    q.missed(a)
    q.missed(b)
    assert q._holder is None and q._sweep == sweep + 1     # no more sweeps without a new wake-up


async def test_a_room_key_that_missed_is_passed_over_for_the_rest_of_the_sweep():
    q = queue()
    a = q.join("a", cls=A)
    q.missed(a, room="r1")
    assert q.room_missed("r1") and not q.room_missed("r2")
    q.notify()
    assert not q.room_missed("r1")                  # a new sweep: room may have been freed


async def test_the_head_waiter_that_keeps_missing_gets_a_machine_held_until_it_leaves():
    clock = Clock()
    q = queue({"big": 9.0, "small": 1.0}, hold_after_s=5.0, clock=clock)
    big = q.join("big", cls=A)
    small = q.join("small", cls=B)
    assert not q.should_hold(big)                   # it has not waited long enough
    clock.now += 6.0
    assert q.should_hold(big) and not q.should_hold(small)      # only the head waiter
    q.hold(big, "m-1")
    q.missed(big)
    assert q.held_for_others(small) == q.held_for_others(None) == {"m-1"} and not q.held_for_others(big)
    assert not q.should_hold(small)                 # one hold at a time
    q.done(big)                                     # it leaves (placed, refused, cancelled): the hold ends
    assert not q.held_for_others(small)
    no_hold = queue(hold_after_s=None, clock=clock)
    ticket = no_hold.join("big", cls=A)
    clock.now += 100.0
    assert not no_hold.should_hold(ticket)


def test_reservations_are_classed_by_tier():
    assert [_tier(v) for v in (1, 1000, 1024, 1500, 2048, 3000)] == [1, 1024, 1024, 2048, 2048, 4096]
    assert _tier(None) is None and _tier(0) == 0

    def cls(cpu, mem):
        return _placement_class(SimpleNamespace(env="e", image="i", reserve={"cpu_millis": cpu, "memory_bytes": mem}))
    assert cls(1500, 3 << 30) == cls(1800, (7 << 30) // 2)     # per-task reservations share a few classes
    assert cls(1500, 3 << 30) != cls(4000, 3 << 30)


# --- a fleet with CPUs: big and small reservations -------------------------------------------------------------

class Rooms:
    """Machines with CPUs; an open's need is its image's CPUs, taken when its session opens and freed when the
    test ends it."""

    def __init__(self, cpus, need):
        self.free, self.need, self.picks = dict(cpus), dict(need), 0
        self.ended = None

    def directory_ready(self, now=None):
        return True

    def pick(self, *, image, exclude=(), **kw):
        self.picks += 1
        fits = [m for m, free in self.free.items() if m not in exclude and free >= self.need[image]]
        return Placement(max(fits, key=lambda m: (self.free[m], m)), ADDRESS) if fits else None

    def could_place(self, **kw):
        return True

    def nearest(self, **kw):
        return max(self.free, key=lambda m: (self.free[m], m))

    def note_issued(self, machine_id, at, **kw):
        pass

    # the session listener: the open takes its CPUs
    def opened(self, record):
        image = record.engagement.split(":")[0]
        self.free[record.machine_id] -= self.need[image]
        if self.ended is not None:
            self.ended(record.machine_id, self.need[image])

    def closed(self, record, final):
        pass


class Opens:
    kind = "fake"

    def __init__(self, template, need):
        self.template, self.need = template, need

    async def terms(self, hotkey, engagement):
        terms = SimpleNamespace(**{f.name: getattr(self.template, f.name) for f in fields(self.template)})
        terms.image, terms.engagement = engagement["image"], f"{engagement['image']}:{engagement['n']}"
        terms.exclusive, terms.still_valid = False, None
        terms.reserve = {"cpu_millis": 1000 * self.need[terms.image], "memory_bytes": 1 << 30}
        return terms


async def rooms_issuer(tmp_path, rooms, q):
    env = build(tmp_path)
    template = await env.issuer._engagements["rl_precommit"].terms(HOTKEY, engagement(env.precommit, 0))
    policy = SandboxPolicy(max_live_per_hotkey=10 ** 6, max_opens_per_hour=10 ** 6,
                           max_refused_opens_per_minute=10 ** 6)
    validator = signer(tmp_path, "v2", "v2")
    return SessionIssuer(book=SessionBook(policy), store=env.issuer._store, fleet=rooms, signer=validator,
                         token_verifier=env.issuer._token_verifier, engagements={"fake": Opens(template, rooms.need)},
                         policy=policy, clock=IssuerClock(), events=rooms, queue=q)


def open_one(issuer, hotkey, image, n):
    return asyncio.ensure_future(issuer.open(hotkey=hotkey, request_id=f"{n:032x}",
                                             engagement={"kind": "fake", "image": image, "n": n}))


async def test_a_waiting_big_class_never_stops_a_small_class_that_has_room(tmp_path):
    rooms = Rooms({"m-1": 4, "m-2": 4}, {"big": 4, "small": 1})
    rooms.free = {"m-1": 3, "m-2": 3}
    q = queue({"5Big": 9.0}, wait_s=5.0, hold_after_s=None)
    issuer = await rooms_issuer(tmp_path, rooms, q)
    big = open_one(issuer, "5Big", "big", 0)
    await until(lambda: q.waiting == 1 and q._holder is None and not issuer._lock.locked())
    small = await open_one(issuer, "5Small", "small", 1)            # another class: placed at once
    assert isinstance(small, Grant) and not big.done()
    rooms.free["m-2"] = 4
    q.notify()
    assert isinstance(await big, Grant)


async def test_the_held_machine_is_kept_for_the_head_while_others_place_elsewhere(tmp_path):
    rooms = Rooms({"m-1": 4, "m-2": 4}, {"big": 4, "small": 1})
    rooms.free = {"m-1": 3, "m-2": 2}
    q = queue({"5Big": 9.0}, wait_s=5.0, hold_after_s=0.0)
    issuer = await rooms_issuer(tmp_path, rooms, q)
    big = open_one(issuer, "5Big", "big", 0)
    await until(lambda: q._hold is not None and q._holder is None and not issuer._lock.locked())
    assert q._hold_target == "m-1"                                  # the machine closest to fitting it
    for n in range(2):                                              # small opens go to the other machine only
        assert isinstance(await open_one(issuer, "5Small", "small", 1 + n), Grant)
    assert rooms.free == {"m-1": 3, "m-2": 0}
    rooms.free["m-1"] = 4                                           # a close on the held machine
    q.notify()
    assert isinstance(await big, Grant) and rooms.free["m-1"] == 0 and q._hold is None


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """An event loop on a virtual clock: when nothing is ready it jumps to its next timer instead of sleeping, so a
    timed simulation runs the same way whatever the machine's load (no wall clock anywhere)."""

    def __init__(self):
        super().__init__()
        self._now = 0.0
        real = self._selector.select

        def select(timeout=None):
            events = real(0)
            if not events and timeout:
                self._now += timeout
            return events
        self._selector.select = select

    def time(self):
        return self._now


def run_virtual(coroutine):
    loop = VirtualTimeLoop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


def _simulate(tmp_path, hold_after_s, seconds=3.0, rate=80.0):
    """The reviewer's scenario: one 4-CPU machine, big opens needing all 4 CPUs retried in a loop by one key, and
    small 1-CPU opens from fresh keys arriving at random (~80 % load), equal scores, every session 20-60 ms, a
    poll every 50 ms. On a virtual clock: deterministic."""
    import random

    rooms = Rooms({"m-1": 4}, {"big": 4, "small": 1})
    # The template's terms are read in a worker thread: on a real loop, before the virtual one starts.
    issuer = asyncio.run(rooms_issuer(tmp_path, rooms, None))

    async def main():
        rng = random.Random(1)
        q = OpenQueue(rank=lambda h: (h, 1.0, False), wait_s=0.3, fair_every=0,
                      hold_after_s=hold_after_s, clock=asyncio.get_running_loop().time)
        issuer._queue = q
        stats = {"big": 0, "big_sat": 0, "small": 0, "small_sat": 0}
        loop = asyncio.get_running_loop()

        def ended(machine, cpus):
            def free():
                rooms.free[machine] += cpus
                issuer._wake()                                      # the close wakes the queue
            loop.call_later(rng.uniform(0.02, 0.06), free)
        rooms.ended = ended
        counter = iter(range(10 ** 6))
        stop = loop.time() + seconds

        async def one(hotkey, image):
            answer = await issuer.open(hotkey=hotkey, request_id=f"{next(counter):032x}",
                                       engagement={"kind": "fake", "image": image, "n": next(counter)})
            kind = image if isinstance(answer, Grant) else image + "_sat"
            assert isinstance(answer, Grant) or answer.reason == "sandbox_saturated", answer
            stats[kind] += 1

        async def bigs():
            while loop.time() < stop:
                await one("5Big", "big")
                await asyncio.sleep(0.01)

        async def poller():
            while loop.time() < stop + 0.5:
                await asyncio.sleep(0.05)
                q.notify()

        tasks = [asyncio.ensure_future(bigs()), asyncio.ensure_future(poller())]
        smalls = []
        while loop.time() < stop:
            smalls.append(asyncio.ensure_future(one(f"5S{next(counter)}", "small")))
            await asyncio.sleep(rng.expovariate(rate))
        await asyncio.gather(*smalls, tasks[0])
        tasks[1].cancel()
        await asyncio.gather(tasks[1], return_exceptions=True)
        return stats

    return run_virtual(main())


def test_big_reservations_no_longer_starve_while_small_ones_keep_flowing(tmp_path):
    stats = _simulate(tmp_path, hold_after_s=0.05)                 # without the hold: about half time out
    assert stats == _simulate(tmp_path / "again", hold_after_s=0.05)    # deterministic (a virtual clock)
    assert stats["big"] >= 8 and stats["big_sat"] <= max(1, stats["big"] // 10)     # big opens place
    offered = stats["small"] + stats["small_sat"]
    assert stats["small"] >= 0.7 * offered                          # small opens keep flowing ...
    assert stats["small"] <= 4 * 3.5 / 0.02                         # ... at a bounded rate (the machine's)


def test_a_close_costs_a_bounded_number_of_locked_attempts_at_512_waiters(tmp_path):
    async def main():
        rooms = Rooms({"m-1": 0}, {"img-a": 1, "img-b": 1})
        q = OpenQueue(rank=lambda h: (h, 1.0, False), wait_s=30.0, fair_every=0, hold_after_s=None)
        issuer = await rooms_issuer(tmp_path, rooms, q)
        locked = []
        inner = issuer._issue_locked

        async def counting(*args, **kwargs):
            locked.append(kwargs.get("may_place"))
            return await inner(*args, **kwargs)
        issuer._issue_locked = counting
        waiters = [open_one(issuer, f"5K{n // 64}", "img-a" if n % 2 else "img-b", n) for n in range(513)]
        await until(lambda: q.waiting == 512 and q._holder is None and not issuer._lock.locked(), seconds=30.0)
        assert len(locked) <= 2 * 513
        for _ in range(3):                                          # three closes, one slot each
            locked.clear()
            picks = rooms.picks
            rooms.free["m-1"] += 1
            for _ in range(20):                                     # a burst of wake-ups (polls, other closes)
                issuer._wake()
            await until(lambda: q._holder is None and not issuer._lock.locked() and rooms.free["m-1"] == 0)
            await asyncio.sleep(0.05)
            assert len(locked) == 1                                 # only the open that placed took the lock
            assert rooms.picks - picks <= 8                         # one cheap look per room key and sweep
        answers = [w.result() for w in waiters if w.done()]
        assert sum(isinstance(a, Grant) for a in answers) == 3
        assert [a.reason for a in answers if not isinstance(a, Grant)] == ["open_queue_full"]    # the 513th
        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        assert not q.busy

    asyncio.run(main())
