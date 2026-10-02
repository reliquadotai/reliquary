"""The arrival feed between the front and a judge process: ids posted in the
order accepted, nothing lost silently, and a judge that holds unaudited
passes until it can vouch nothing accepted is missing."""

from __future__ import annotations

import asyncio

from reliquary.validator.corpus_feed import ArrivalFeed, JudgeLink


class _Client:
    path = "judge.sock"

    def __init__(self):
        self.posts = []
        self.fail = 0

    async def post(self, path, body, **kw):
        if self.fail:
            self.fail -= 1
            raise ConnectionError("judge down")
        self.posts.append(body)
        return {"ok": True}


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_ids_are_posted_in_order_and_as_of_only_when_the_queue_empties(monkeypatch):
    from reliquary.validator import corpus_feed

    monkeypatch.setattr(corpus_feed, "FEED_POST_IDS", 3)
    client, clock = _Client(), _Clock()
    link = JudgeLink("judge.sock", ["math"], epoch="e1", clock=clock, client=client)
    for i in range(5):
        link.accepted("math", f"s{i}")
    assert asyncio.run(link.flush_once()) and asyncio.run(link.flush_once())
    first, second = client.posts
    assert first["ids"] == {"math": ["s0", "s1", "s2"]} and first["as_of"] is None
    assert second["ids"] == {"math": ["s3", "s4"]} and second["as_of"] == 1000.0
    assert len(link) == 0 and not first["dropped"]
    # Idle: a heartbeat vouching for "nothing new until now".
    clock.t = 1002.0
    assert asyncio.run(link.flush_once())
    assert client.posts[-1] == {"epoch": "e1", "dropped": False, "ids": {}, "as_of": 1002.0}


def test_a_failed_post_keeps_its_ids():
    client = _Client()
    client.fail = 1
    link = JudgeLink("judge.sock", ["math"], epoch="e1", client=client)
    link.accepted("math", "s0")
    assert not asyncio.run(link.flush_once())
    link.accepted("math", "s1")
    assert asyncio.run(link.flush_once())
    assert client.posts[-1]["ids"] == {"math": ["s0", "s1"]}


def test_an_overflow_is_announced_until_a_post_lands():
    client = _Client()
    link = JudgeLink("judge.sock", ["math"], epoch="e1", client=client, max_ids=2)
    for i in range(4):
        link.accepted("math", f"s{i}")
    client.fail = 1
    assert not asyncio.run(link.flush_once())
    assert asyncio.run(link.flush_once())
    assert client.posts[-1]["dropped"] and client.posts[-1]["ids"] == {"math": ["s2", "s3"]}
    assert asyncio.run(link.flush_once())
    assert client.posts[-1]["dropped"] is False


class _Auditor:
    def __init__(self, gate=None):
        self.enqueued = []
        self.listings = 0
        self.gate = gate
        self.fail = 0

    def enqueue(self, sid):
        self.enqueued.append(sid)

    async def rescan_store(self):
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            self.fail -= 1
            raise ConnectionError("list failed")
        self.listings += 1


def test_the_feed_is_complete_only_after_a_listing_and_while_fresh():
    async def scenario():
        clock = _Clock()
        gate = asyncio.Event()
        auditor = _Auditor(gate)
        feed = ArrivalFeed({"math": auditor}, clock=clock, retry_seconds=0.0)
        assert not feed.complete()
        feed.receive({"epoch": "e1", "dropped": False, "ids": {"math": ["a", "b"]},
                      "as_of": 1000.0})
        assert auditor.enqueued == ["a", "b"]
        await asyncio.sleep(0)
        assert not feed.complete()          # listing still running
        gate.set()
        await asyncio.sleep(0.01)
        assert auditor.listings == 1 and feed.complete() and feed.covered() == 1000.0
        clock.t = 1011.0                     # no post for 11 s: stale, still vouching
        assert not feed.complete() and feed.covered() == 1000.0
        feed.receive({"epoch": "e1", "dropped": False, "ids": {}, "as_of": 1011.0})
        assert feed.complete() and auditor.listings == 1
        # A new front: list again before trusting it.
        gate.clear()
        feed.receive({"epoch": "e2", "dropped": False, "ids": {}, "as_of": 1011.5})
        assert not feed.complete() and feed.covered() is None
        gate.set()
        await asyncio.sleep(0.01)
        assert feed.complete() and auditor.listings == 2
        # Dropped ids: list again.
        auditor.fail = 1
        feed.receive({"epoch": "e2", "dropped": True, "ids": {}, "as_of": 1011.6})
        assert not feed.complete() and feed.covered() is None
        await asyncio.sleep(0.05)
        assert feed.complete() and auditor.listings == 3

    asyncio.run(scenario())


def test_ids_of_a_job_judged_elsewhere_are_counted_not_lost_silently():
    async def scenario():
        feed = ArrivalFeed({"math": _Auditor()})
        feed.receive({"epoch": "e1", "dropped": False, "ids": {"code": ["x"]}, "as_of": None})
        return feed

    feed = asyncio.run(scenario())
    assert feed.unknown_ids == 1 and not feed.complete() and feed.covered() is None
