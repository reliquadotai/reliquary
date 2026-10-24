"""The arrival feed between the front and a judge process: ids posted in the
order accepted, nothing lost silently, and a judge that holds unaudited
passes until it can vouch nothing accepted is missing."""

from __future__ import annotations

import asyncio

import pytest

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


def test_coverage_waits_for_every_jobs_pending_body_before_announcing_the_cut():
    async def scenario():
        clock = _Clock()
        auditors = {job: _Auditor() for job in ("math", "code")}
        feed = ArrivalFeed(auditors, clock=clock)

        class Client(_Client):
            async def post(self, path, body, **kw):
                await super().post(path, body, **kw)
                feed.receive(body)
                return {"ok": True}

        client = Client()
        link = JudgeLink("judge.sock", auditors, epoch="e1", clock=clock, client=client)
        await link.flush_once()
        await asyncio.sleep(0.01)
        assert feed.covered() == 1000.0
        math_pending, code_pending = {"m": 1001.0}, {"c": 1002.0}
        link.pending_record_arrivals.update(math=math_pending, code=code_pending)
        clock.t = 1200.0
        await link.flush_once()
        assert client.posts[-1]["as_of"] is None and feed.covered() is None
        # One body is now canonical, but the other job still blocks this group's cut.
        math_pending.clear()
        link.accepted("math", "m")
        await link.flush_once()
        assert auditors["math"].enqueued == ["m"] and feed.covered() is None
        assert client.posts[-1]["as_of"] is None
        code_pending.clear()
        link.accepted("code", "c")
        await link.flush_once()
        assert auditors["code"].enqueued == ["c"] and feed.covered() == 1200.0

    asyncio.run(scenario())


def test_unpublished_old_body_holds_a_real_sampled_judge_after_clock_rollback():
    from reliquary.corpus.audit_policy import AuditParams, MinerState
    from reliquary.validator.corpus_auditor import CorpusAuditor
    from reliquary.validator.corpus_miner_states import MinerStates
    from tests.unit import test_corpus_judge_equivalence as equivalence
    from tests.unit.test_corpus_sibling_reuse_bound import RANDOMNESS, _ids, _round_at

    async def scenario():
        clock = _Clock(20_000.0)
        store = equivalence._Store()
        store.miners["5HkH"] = MinerState(audited_passed=50).to_dict()
        x, fresh1, fresh2 = _ids(False, 3)
        (pending_sid,) = _ids(True, 1, start=1000)

        def record(received):
            return {"hotkey": "5HkH", "received_at": received, "token_count": 10,
                    "completions": [{"tokens": [1]}]}

        store.submissions.update({x: record(0.0), fresh1: record(8990.0),
                                  fresh2: record(8995.0)})
        feed = ArrivalFeed(clock=clock)
        auditor = CorpusAuditor(
            job_id="math", records=store, model=None, tokenizer=None, proof=None,
            params=AuditParams(q=0.5, probation_submissions=5, hold_seconds=100.0),
            miner_states=MinerStates(store, "math", clock=clock),
            beacon=lambda r: RANDOMNESS, round_at=_round_at, clock=clock,
            accept_slack_seconds=5.0, prefetch_rounds=False, arrivals_covered=feed.covered)

        async def forward(records, *, local=False):
            return [dict(equivalence._OK) for _ in records]

        auditor._forward = forward
        feed.auditors["math"] = auditor

        class Client(_Client):
            async def post(self, path, body, **kw):
                await super().post(path, body, **kw)
                feed.receive(body)
                return {"ok": True}

        link = JudgeLink("judge.sock", ["math"], epoch="e1", clock=clock, client=Client())
        await link.flush_once()
        await asyncio.sleep(0.01)
        assert auditor._covers(feed.covered(), 0.0)
        # The committed body's original timestamp predates the prior watermark.
        pending = {pending_sid: 4.0}
        link.pending_record_arrivals["math"] = pending
        clock.t = 9000.0
        await link.flush_once()
        assert feed.covered() is None
        await auditor.judge_many([x])
        assert x not in store.verdicts
        # A restarted front and a listing cannot make that old watermark usable.
        link.epoch = "e2"
        await link.flush_once()
        await asyncio.sleep(0.01)
        assert feed.covered() is None
        await auditor.judge_many([x])
        assert x not in store.verdicts
        # Recovery publishes the actual body before announcing it and releasing coverage.
        store.submissions[pending_sid] = record(pending[pending_sid])
        pending.clear()
        link.accepted("math", pending_sid)
        await link.flush_once()
        assert feed.covered() == 9000.0
        await auditor.judge_many([x])
        assert store.verdicts[x]["passed"] is True and store.verdicts[x]["audited"] is False
        assert store.verdicts[pending_sid]["passed"] is True
        # Equal cuts are valid; a lower current cut replaces, rather than inherits, the old one.
        await link.flush_once()
        assert feed.covered() == 9000.0
        clock.t = 8999.0
        await link.flush_once()
        assert feed.covered() == 8999.0
        await feed.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("cut", [True, "10000", float("nan"), float("inf"),
                                 -float("inf"), -1.0, 10 ** 400])
def test_invalid_feed_cut_cannot_inherit_prior_coverage(cut):
    async def scenario():
        feed = ArrivalFeed({"math": _Auditor()})
        feed.receive({"epoch": "e1", "ids": {}, "as_of": 10000.0})
        await asyncio.sleep(0.01)
        assert feed.covered() == 10000.0
        feed.receive({"epoch": "e1", "ids": {}, "as_of": cut})
        assert feed.covered() is None

    asyncio.run(scenario())


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
