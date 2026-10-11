"""The fair open queue of RL sandbox sessions at saturation."""
import asyncio

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.sandbox.open_queue import OpenQueue  # noqa: E402

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
