"""The settler learns new verdicts from the auditor's feed and lists the store
only at boot and every ``SETTLE_FULL_LIST_SECONDS``; listings never parse on
the event loop. (The period settler; the window one is gone.)"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure import corpus_record_store as records_mod
from reliquary.infrastructure.corpus_record_store import BucketRecordStore
from reliquary.validator.corpus_period_settlement import CorpusPeriodSettler
from reliquary.validator.corpus_periods import PERIOD_SECONDS
from reliquary.validator.corpus_settlement import SETTLE_FULL_LIST_SECONDS

EVERY = 1800.0
P = PERIOD_SECONDS
# Settling at CLOSED(p) closes period p (and every one before it).
CLOSE = 600.0


def CLOSED(period: int) -> float:
    return (period + 1) * P + CLOSE


def _id(n: int) -> str:
    return f"{n:064x}"


def _v(hk, n, ok=True, at=10.0):
    return {"hotkey": hk, "token_count": n, "passed": ok, "received_at": at}


class _Records:
    def __init__(self, verdicts=None):
        self.verdicts = dict(verdicts or {})
        self.voided: set[str] = set()
        self.state, self.etag = {}, None
        self.lists = 0
        self.fail_state_writes_after = None

    async def list_verdict_ids(self, job_id):
        self.lists += 1
        return sorted(self.verdicts)

    async def list_voided_ids(self, job_id):
        return sorted(self.voided)

    async def read_verdict(self, job_id, sid):
        return self.verdicts[sid]

    async def read_settlement(self, job_id):
        return json.loads(json.dumps(self.state)), self.etag

    async def write_settlement(self, job_id, state, etag):
        if self.fail_state_writes_after == 0:
            raise OSError("crash")
        if self.fail_state_writes_after is not None:
            self.fail_state_writes_after -= 1
        assert etag == self.etag
        self.state = json.loads(json.dumps(state))
        self.etag = f"e{len(json.dumps(state, sort_keys=True))}-{id(state)}"
        return self.etag


class _SlowVerdictReads(_Records):
    """Each verdict read takes a while; counts reads and how many overlap."""

    def __init__(self, verdicts=None):
        super().__init__(verdicts)
        self.reads, self.in_flight, self.peak = 0, 0, 0

    async def read_verdict(self, job_id, sid):
        self.reads += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.005)
            return await super().read_verdict(job_id, sid)
        finally:
            self.in_flight -= 1


class _Archives:
    """Period archives; ``written`` maps a work period to its archive's bytes."""

    def __init__(self):
        self.docs: dict[tuple[int, int], dict] = {}
        self.written: dict[int, bytes] = {}

    async def write(self, task_id, work, entry, document):
        assert (work, entry) not in self.docs
        self.docs[(work, entry)] = document
        self.written[work] = json.dumps(document, sort_keys=True).encode()

    async def read(self, task_id, work, entry):
        return self.docs.get((work, entry))

    async def list(self, task_id):
        return sorted(self.docs)


class _Clock:
    def __init__(self, now=CLOSED(0)):
        self.now = now

    def __call__(self):
        return self.now


def _settler(records, archives, clock, *, fed=True):
    return CorpusPeriodSettler(task_id="corpus-math", job_id="math-v1", cap=0.1,
                               records=records, archives=archives,
                               oldest_pending=lambda: None, genesis=lambda: 0.0, clock=clock,
                               full_list_every_seconds=EVERY if fed else None)


def test_the_default_safety_net_is_half_an_hour():
    assert SETTLE_FULL_LIST_SECONDS == 1800.0


def test_a_fed_settler_lists_at_boot_then_only_every_full_list_period():
    records, archives, clock = _Records({_id(1): _v("A", 10)}), _Archives(), _Clock()
    settler = _settler(records, archives, clock)
    assert asyncio.run(settler.settle_once()) == 0
    assert records.lists == 1
    for step in range(1, 10):
        clock.now = CLOSED(0) + step * 60.0
        records.verdicts[_id(100 + step)] = _v("B", 5, at=clock.now)
        settler.observe(_id(100 + step), records.verdicts[_id(100 + step)])
        assert asyncio.run(settler.settle_once()) is None  # period 1 is still open
    assert records.lists == 1
    clock.now = CLOSED(1)
    assert asyncio.run(settler.settle_once()) == 1
    assert records.lists == 2


def test_a_verdict_the_feed_missed_is_paid_at_the_next_full_listing():
    records, archives, clock = _Records({_id(1): _v("A", 10)}), _Archives(), _Clock()
    settler = _settler(records, archives, clock)
    asyncio.run(settler.settle_once())
    # Written by another path: never reported to this settler.
    records.verdicts[_id(2)] = _v("B", 30, at=P + 10)
    clock.now = CLOSED(1)
    settler._listed_at = clock.now  # listed just now: the next listing is EVERY away
    assert asyncio.run(settler.settle_once()) is None
    assert _id(2) not in records.state["settled"]
    clock.now += EVERY + 1
    assert asyncio.run(settler.settle_once()) == 1
    assert _id(2) in records.state["settled"]
    assert json.loads(archives.written[1])["rewards_by_hotkey"] == pytest.approx({"B": 0.1})


def test_a_verdict_written_before_boot_is_paid_by_the_first_settlement():
    records = _Records({_id(1): _v("A", 10), _id(2): _v("B", 10)})
    archives, clock = _Archives(), _Clock()
    assert asyncio.run(_settler(records, archives, clock).settle_once()) == 0
    assert records.state["settled"] == [_id(1), _id(2)]


def test_a_failed_full_listing_is_retried_on_the_next_call():
    records, archives, clock = _Records({_id(1): _v("A", 10)}), _Archives(), _Clock()
    real = records.list_verdict_ids
    calls = []

    async def flaky(job_id):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("listing failed")
        return await real(job_id)

    records.list_verdict_ids = flaky
    settler = _settler(records, archives, clock)
    with pytest.raises(ConnectionError):
        asyncio.run(settler.settle_once())
    clock.now += 60.0
    assert asyncio.run(settler.settle_once()) == 0


def test_a_fed_id_already_settled_or_fed_twice_is_never_paid_twice():
    records, archives, clock = _Records({_id(1): _v("A", 10)}), _Archives(), _Clock()
    settler = _settler(records, archives, clock)
    asyncio.run(settler.settle_once())
    # The auditor reports a standing verdict again (found, not written).
    settler.observe(_id(1), records.verdicts[_id(1)])
    settler.observe(_id(1), records.verdicts[_id(1)])
    clock.now = CLOSED(3)
    assert asyncio.run(settler.settle_once()) is None
    assert list(archives.written) == [0]
    # A restarted settler (fresh process) with the same feed: still nothing.
    again = _settler(records, archives, clock)
    again.observe(_id(1), records.verdicts[_id(1)])
    assert asyncio.run(again.settle_once()) is None
    assert list(archives.written) == [0]
    assert records.state["settled"] == [_id(1)]


def test_a_crash_after_the_archive_finishes_the_same_period_once():
    records, archives, clock = _Records({_id(1): _v("A", 10)}), _Archives(), _Clock()
    records.fail_state_writes_after = 1  # the pending write lands, the final one crashes
    settler = _settler(records, archives, clock)
    with pytest.raises(OSError):
        asyncio.run(settler.settle_once())
    first = archives.written[0]
    records.fail_state_writes_after = None
    clock.now += 60.0
    # The same instance, fed again, and a restarted one: one archive, same bytes.
    settler.observe(_id(1), records.verdicts[_id(1)])
    assert asyncio.run(settler.settle_once()) == 0
    assert asyncio.run(_settler(records, archives, clock).settle_once()) is None
    assert archives.written[0] == first and len(archives.docs) == 1
    assert records.state["settled"] == [_id(1)]


def _scenario(fed: bool):
    """One fixed sequence of verdicts, voids and settle calls over many periods;
    every verdict is reported the moment it is written: with itself when fed,
    by id only otherwise, so that settler reads every verdict back."""
    records, archives, clock = _SlowVerdictReads(), _Archives(), _Clock(0.0)
    settler = _settler(records, archives, clock, fed=fed)
    n = 0
    for step in range(40):
        clock.now = step * P / 3 + 30.0
        for k in range(step % 4):
            n += 1
            verdict = _v("ABCDE"[(n * 7) % 5], 10 + (n * 13) % 97, ok=(n % 3 != 0),
                         at=clock.now - k)
            records.verdicts[_id(n)] = verdict
            settler.observe(_id(n), dict(verdict) if fed else None)
        if step % 9 == 4 and n:
            records.voided.add(_id(n))
        asyncio.run(settler.settle_once())
    assert (records.reads == 0) == fed
    return archives.written, records.state


def test_a_fed_settler_writes_byte_identical_archives_and_state_to_a_listing_one():
    listed_archives, listed_state = _scenario(fed=False)
    fed_archives, fed_state = _scenario(fed=True)
    assert len(listed_archives) > 5
    assert fed_archives == listed_archives
    assert json.dumps(fed_state, sort_keys=True) == json.dumps(listed_state, sort_keys=True)


# --- listings never parse on the event loop ---


PAGES, PER_PAGE, PARSE_SECONDS = 10, 30_000, 0.15


def _burn(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


class _SlowListingR2:
    """A bucket whose listing pages cost CPU to parse, as aiobotocore's XML
    parsing does: 300k keys in 10 pages."""

    def __init__(self, prefix: str):
        self.prefix = prefix

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get_paginator(self, name):
        prefix = self.prefix

        class _Paginator:
            def paginate(self, Bucket, Prefix="", **kwargs):
                async def _pages():
                    for page in range(PAGES):
                        _burn(PARSE_SECONDS)
                        yield {"Contents": [{"Key": f"{prefix}{page * PER_PAGE + i:064x}.json"}
                                            for i in range(PER_PAGE)]}
                        await asyncio.sleep(0)

                return _pages()

        return _Paginator()


LOOP_GAP_LIMIT = 0.45


async def _max_gap_while(coro) -> tuple[float, object]:
    gaps = []
    done = asyncio.Event()

    async def heartbeat():
        last = time.perf_counter()
        while not done.is_set():
            await asyncio.sleep(0.002)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.01)
    try:
        result = await coro
    finally:
        done.set()
        await beat
    return max(gaps), result


def test_a_large_record_listing_does_not_block_the_event_loop(monkeypatch):
    prefix = "reliquary/corpus/jobs/math-v1/verdicts/"
    bucket = _SlowListingR2(prefix)
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kw: bucket)
    monkeypatch.setattr(records_mod, "get_s3_client", lambda **kw: bucket)
    store = BucketRecordStore()
    gap, ids = asyncio.run(_max_gap_while(store.list_verdict_ids("math-v1")))
    assert len(ids) == PAGES * PER_PAGE
    # Loose on purpose: a busy CI box stretches the thread hand-off, while the
    # same listing on the loop blocks it 600+ ms.
    assert gap < LOOP_GAP_LIMIT, f"event loop blocked {gap * 1000:.0f} ms"


def test_the_auditor_hook_feeds_the_settler_before_the_status_hook():
    from reliquary.validator.corpus_settlement import settler_fed

    records, archives, clock = _Records(), _Archives(), _Clock()
    settler = _settler(records, archives, clock)
    asyncio.run(settler.settle_once())

    def broken(submission_id, verdict):
        raise RuntimeError("status hook bug")

    report = settler_fed(settler, broken)
    records.verdicts[_id(1)] = _v("A", 10, at=P + 10)
    with pytest.raises(RuntimeError):
        report(_id(1), records.verdicts[_id(1)])
    clock.now = CLOSED(1)
    settler._listed_at = clock.now  # no listing: only the feed can tell it
    assert asyncio.run(settler.settle_once()) == 1


def test_a_fed_verdict_is_settled_without_reading_it_back():
    """Math's settlement read each new verdict one at a time and paid one
    window in three hours (2026-10-02): the auditor's report is the verdict."""
    records, archives, clock = _SlowVerdictReads(), _Archives(), _Clock()
    settler = _settler(records, archives, clock)
    asyncio.run(settler.settle_once())
    for n in range(50):
        records.verdicts[_id(n)] = _v("A" if n % 2 else "B", 10, at=P + 10)
        settler.observe(_id(n), records.verdicts[_id(n)])
    clock.now = CLOSED(1)
    assert asyncio.run(settler.settle_once()) == 1
    assert records.reads == 0
    assert json.loads(archives.written[1])["rewards_by_hotkey"] == pytest.approx(
        {"A": 0.05, "B": 0.05})


def test_unfed_verdicts_are_read_together():
    records = _SlowVerdictReads({_id(n): _v("A", 10) for n in range(40)})
    archives, clock = _Archives(), _Clock()
    assert asyncio.run(_settler(records, archives, clock).settle_once()) == 0
    assert records.reads == 40 and records.peak > 1
    assert records.state["settled"] == sorted(_id(n) for n in range(40))


def test_a_failed_verdict_read_raises_after_the_others_finish():
    class _OneFails(_SlowVerdictReads):
        async def read_verdict(self, job_id, sid):
            if sid == _id(3):
                raise OSError("reset")
            return await super().read_verdict(job_id, sid)

    records = _OneFails({_id(n): _v("A", 10) for n in range(20)})
    settler = _settler(records, _Archives(), _Clock())

    async def settle():
        with pytest.raises(OSError):
            await settler.settle_once()
        return records.in_flight

    assert asyncio.run(settle()) == 0
    assert records.state == {}
