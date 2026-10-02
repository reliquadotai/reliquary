"""The settler learns new verdicts from the auditor's feed and lists the store
only at boot and every ``SETTLE_FULL_LIST_SECONDS``; listings never parse on
the event loop; the other tasks' horizon is cached, never listed per settle."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.infrastructure import corpus_record_store as records_mod
from reliquary.infrastructure.corpus_record_store import BucketRecordStore
from reliquary.validator.corpus_settlement import (
    SETTLE_FULL_LIST_SECONDS,
    CorpusSettler,
    R2Archives,
)

STALL = 100.0
EVERY = 1800.0


def _id(n: int) -> str:
    return f"{n:064x}"


def _v(hk, n, ok=True):
    return {"hotkey": hk, "token_count": n, "passed": ok}


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


class _Archives:
    def __init__(self, other_max):
        self.other = other_max
        self.written: dict[int, bytes] = {}

    async def other_max(self, task_id):
        return self.other

    async def write(self, task_id, window, data):
        self.written[window] = json.dumps(data, sort_keys=True).encode()


class _Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def _settler(records, archives, clock, *, fed=True):
    return CorpusSettler(task_id="corpus-math", job_id="math-v1", cap=0.1,
                         records=records, archives=archives, stall_seconds=STALL,
                         clock=clock, full_list_every_seconds=EVERY if fed else None)


def test_the_default_safety_net_is_half_an_hour():
    assert SETTLE_FULL_LIST_SECONDS == 1800.0


def test_a_fed_settler_lists_at_boot_then_only_every_full_list_period():
    records, archives, clock = _Records({_id(1): _v("A", 10)}), _Archives(46000), _Clock()
    settler = _settler(records, archives, clock)
    assert asyncio.run(settler.settle_once()) == 46000
    assert records.lists == 1
    for step in range(1, 10):
        clock.now = step * 60.0
        archives.other = 46000 + step
        records.verdicts[_id(100 + step)] = _v("B", 5)
        settler.observe(_id(100 + step), records.verdicts[_id(100 + step)])
        assert asyncio.run(settler.settle_once()) == 46000 + step
    assert records.lists == 1
    clock.now = EVERY + 1
    asyncio.run(settler.settle_once())
    assert records.lists == 2


def test_a_verdict_the_feed_missed_is_paid_at_the_next_full_listing():
    records, archives, clock = _Records({_id(1): _v("A", 10)}), _Archives(46000), _Clock()
    settler = _settler(records, archives, clock)
    asyncio.run(settler.settle_once())
    # Written by another path: never reported to this settler.
    records.verdicts[_id(2)] = _v("B", 30)
    archives.other = 46001
    clock.now = 60.0
    assert asyncio.run(settler.settle_once()) is None
    assert _id(2) not in records.state["settled"]
    archives.other = 46002
    clock.now = EVERY + 1
    assert asyncio.run(settler.settle_once()) == 46002
    assert _id(2) in records.state["settled"]
    assert json.loads(archives.written[46002])["rewards_by_hotkey"] == pytest.approx({"B": 0.1})


def test_a_verdict_written_before_boot_is_paid_by_the_first_settlement():
    records = _Records({_id(1): _v("A", 10), _id(2): _v("B", 10)})
    archives, clock = _Archives(46000), _Clock()
    assert asyncio.run(_settler(records, archives, clock).settle_once()) == 46000
    assert records.state["settled"] == [_id(1), _id(2)]


def test_a_failed_full_listing_is_retried_on_the_next_call():
    records, archives, clock = _Records({_id(1): _v("A", 10)}), _Archives(46000), _Clock()
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
    clock.now = 60.0
    assert asyncio.run(settler.settle_once()) == 46000


def test_a_fed_id_already_settled_or_fed_twice_is_never_paid_twice():
    records, archives, clock = _Records({_id(1): _v("A", 10)}), _Archives(46000), _Clock()
    settler = _settler(records, archives, clock)
    asyncio.run(settler.settle_once())
    # The auditor reports a standing verdict again (found, not written).
    settler.observe(_id(1), records.verdicts[_id(1)])
    settler.observe(_id(1), records.verdicts[_id(1)])
    archives.other, clock.now = 46001, 60.0
    assert asyncio.run(settler.settle_once()) is None
    assert list(archives.written) == [46000]
    # A restarted settler (fresh process) with the same feed: still nothing.
    again = _settler(records, archives, clock)
    again.observe(_id(1), records.verdicts[_id(1)])
    assert asyncio.run(again.settle_once()) is None
    assert list(archives.written) == [46000]
    assert records.state["settled"] == [_id(1)]


def test_a_crash_after_the_archive_finishes_the_same_window_once():
    records, archives, clock = _Records({_id(1): _v("A", 10)}), _Archives(46000), _Clock()
    records.fail_state_writes_after = 1  # the pending write lands, the final one crashes
    settler = _settler(records, archives, clock)
    with pytest.raises(OSError):
        asyncio.run(settler.settle_once())
    first = archives.written[46000]
    records.fail_state_writes_after = None
    archives.other = 46005
    clock.now = 60.0
    # The same instance, fed again, and a restarted one: one archive, same bytes.
    settler.observe(_id(1), records.verdicts[_id(1)])
    assert asyncio.run(settler.settle_once()) == 46000
    assert asyncio.run(_settler(records, archives, clock).settle_once()) is None
    assert archives.written[46000] == first
    assert records.state["settled"] == [_id(1)]
    assert sum(1 for w in archives.written if json.loads(archives.written[w])["rewards_by_hotkey"]) == 1


def _scenario(fed: bool):
    """One fixed sequence of verdicts, voids, horizon moves and settle calls;
    every verdict is reported to the feed the moment it is written."""
    records, archives, clock = _Records(), _Archives(46000), _Clock()
    settler = _settler(records, archives, clock, fed=fed)
    n = 0
    for step in range(40):
        clock.now = step * 60.0
        for k in range(step % 4):
            n += 1
            verdict = _v("ABCDE"[(n * 7) % 5], 10 + (n * 13) % 97, ok=(n % 3 != 0))
            records.verdicts[_id(n)] = verdict
            settler.observe(_id(n), verdict)
        if step % 9 == 4 and n:
            records.voided.add(_id(n))
        if step % 5 == 0 or step > 30:
            archives.other = 46000 + step // 5
        asyncio.run(settler.settle_once())
    return archives.written, records.state


def test_a_fed_settler_writes_byte_identical_archives_and_state_to_a_listing_one():
    listed_archives, listed_state = _scenario(fed=False)
    fed_archives, fed_state = _scenario(fed=True)
    assert len(listed_archives) > 5
    assert fed_archives == listed_archives
    assert json.dumps(fed_state, sort_keys=True) == json.dumps(listed_state, sort_keys=True)


# --- the other tasks' horizon: cached, refreshed within a TTL ---


def _fake_bucket(windows: dict[str, list[int]]):
    calls = {"tasks": 0, "windows": 0}

    async def list_task_ids(*, strict=False, **kw):
        calls["tasks"] += 1
        return sorted(windows)

    async def list_all_window_keys(*, strict=False, task_id=None, **kw):
        calls["windows"] += 1
        return sorted(windows.get(task_id, []))

    return calls, list_task_ids, list_all_window_keys


def test_other_max_is_listed_once_per_ttl_and_a_fresh_window_is_seen_within_it():
    windows = {"default": [46000], "corpus-math": [3], "corpus-code": [7]}
    calls, tasks, keys = _fake_bucket(windows)
    clock = _Clock(0.0)
    archives = R2Archives(ttl_seconds=300.0, clock=clock)

    async def scenario():
        assert await archives.other_max("corpus-math") == 46000
        listed = dict(calls)
        for _ in range(5):
            assert await archives.other_max("corpus-math") == 46000
            assert await archives.other_max("corpus-code") == 46000
        assert calls == listed
        windows["default"].append(46001)
        clock.now = 299.0
        assert await archives.other_max("corpus-math") == 46000
        clock.now = 301.0
        assert await archives.other_max("corpus-math") == 46001
        assert calls["tasks"] == 2

    with patch("reliquary.infrastructure.storage.list_task_ids", AsyncMock(side_effect=tasks)), \
         patch("reliquary.infrastructure.storage.list_all_window_keys", AsyncMock(side_effect=keys)):
        asyncio.run(scenario())


def test_a_window_this_process_writes_is_seen_by_the_other_jobs_at_once(monkeypatch):
    windows = {"default": [10], "corpus-math": [3], "corpus-code": [7]}
    calls, tasks, keys = _fake_bucket(windows)
    monkeypatch.setenv("RELIQUARY_TASK_ID", "corpus-math,corpus-code")
    archives = R2Archives(ttl_seconds=300.0, clock=_Clock(0.0))

    async def scenario():
        assert await archives.other_max("corpus-code") == 10
        await archives.write("corpus-math", 11, {})
        assert await archives.other_max("corpus-code") == 11
        # Its own window is never its own horizon.
        assert await archives.other_max("corpus-math") == 10

    with patch("reliquary.infrastructure.storage.list_task_ids", AsyncMock(side_effect=tasks)), \
         patch("reliquary.infrastructure.storage.list_all_window_keys", AsyncMock(side_effect=keys)), \
         patch("reliquary.infrastructure.storage.upload_window_dataset", AsyncMock()):
        asyncio.run(scenario())


def test_a_failed_refresh_raises_and_is_retried_next_call():
    windows = {"default": [5]}
    calls, tasks, keys = _fake_bucket(windows)
    failing = AsyncMock(side_effect=ConnectionError("down"))
    archives = R2Archives(ttl_seconds=300.0, clock=_Clock(0.0))

    async def scenario():
        with patch("reliquary.infrastructure.storage.list_task_ids", failing):
            with pytest.raises(ConnectionError):
                await archives.other_max("corpus-math")
        with patch("reliquary.infrastructure.storage.list_task_ids", AsyncMock(side_effect=tasks)):
            assert await archives.other_max("corpus-math") == 5

    with patch("reliquary.infrastructure.storage.list_all_window_keys", AsyncMock(side_effect=keys)):
        asyncio.run(scenario())


def test_stall_detection_still_advances_alone_through_the_cache():
    # RL seals 46000 then stops; the corpus joins it, then advances alone only
    # once the stall is seen, through the cached R2Archives.
    windows = {"default": [46000]}
    calls, tasks, keys = _fake_bucket(windows)
    clock = _Clock(0.0)
    records = _Records({_id(1): _v("A", 10)})
    archives = R2Archives(ttl_seconds=300.0, clock=clock)
    written = {}

    async def write(task_id, window, data):
        written[window] = data

    archives.write = write
    settler = CorpusSettler(task_id="corpus-math", job_id="math-v1", cap=0.1, records=records,
                            archives=archives, stall_seconds=3 * 960.0,
                            advance_every_seconds=960.0, clock=clock,
                            full_list_every_seconds=EVERY)

    async def scenario():
        assert await settler.settle_once() == 46000
        n = 1
        results = []
        for t in range(60, 3 * 960 + 600, 60):
            clock.now = float(t)
            n += 1
            records.verdicts[_id(n)] = _v("A", 10)
            settler.observe(_id(n), records.verdicts[_id(n)])
            results.append((t, await settler.settle_once()))
        return results

    with patch("reliquary.infrastructure.storage.list_task_ids", AsyncMock(side_effect=tasks)), \
         patch("reliquary.infrastructure.storage.list_all_window_keys", AsyncMock(side_effect=keys)):
        results = asyncio.run(scenario())
    paid = [(t, w) for t, w in results if w is not None]
    assert paid and paid[0][1] == 46001
    assert paid[0][0] > 3 * 960  # never before the stall


def test_a_revived_other_task_is_seen_within_the_ttl_and_stops_lone_advances():
    windows = {"default": [46000]}
    calls, tasks, keys = _fake_bucket(windows)
    clock = _Clock(0.0)
    archives = R2Archives(ttl_seconds=300.0, clock=clock)

    async def scenario():
        assert await archives.other_max("corpus-math") == 46000
        windows["default"].append(46001)
        clock.now = 300.5
        assert await archives.other_max("corpus-math") == 46001

    with patch("reliquary.infrastructure.storage.list_task_ids", AsyncMock(side_effect=tasks)), \
         patch("reliquary.infrastructure.storage.list_all_window_keys", AsyncMock(side_effect=keys)):
        asyncio.run(scenario())


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


def test_the_other_tasks_listing_does_not_block_the_event_loop():
    async def slow_windows(*, strict=False, task_id=None, **kw):
        _burn(PARSE_SECONDS * 3)
        return list(range(47_000))

    async def tasks(*, strict=False, **kw):
        return ["corpus-math", "default"]

    archives = R2Archives(ttl_seconds=300.0, clock=_Clock(0.0))
    with patch("reliquary.infrastructure.storage.list_task_ids", AsyncMock(side_effect=tasks)), \
         patch("reliquary.infrastructure.storage.list_all_window_keys", AsyncMock(side_effect=slow_windows)):
        gap, best = asyncio.run(_max_gap_while(archives.other_max("corpus-math")))
    assert best == 46_999
    # Loose on purpose: a busy CI box stretches the thread hand-off, while the
    # same listing on the loop blocks it 600+ ms.
    assert gap < LOOP_GAP_LIMIT, f"event loop blocked {gap * 1000:.0f} ms"


def test_the_auditor_hook_feeds_the_settler_before_the_status_hook():
    from reliquary.validator.corpus_settlement import settler_fed

    records, archives, clock = _Records(), _Archives(46000), _Clock()
    settler = _settler(records, archives, clock)
    asyncio.run(settler.settle_once())

    def broken(submission_id, verdict):
        raise RuntimeError("status hook bug")

    report = settler_fed(settler, broken)
    records.verdicts[_id(1)] = _v("A", 10)
    with pytest.raises(RuntimeError):
        report(_id(1), records.verdicts[_id(1)])
    archives.other, clock.now = 46001, 60.0
    assert asyncio.run(settler.settle_once()) == 46001


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


def test_a_fed_verdict_is_settled_without_reading_it_back():
    """Math's settlement read each new verdict one at a time and paid one
    window in three hours (2026-10-02): the auditor's report is the verdict."""
    records, archives, clock = _SlowVerdictReads(), _Archives(46000), _Clock()
    settler = _settler(records, archives, clock)
    asyncio.run(settler.settle_once())
    for n in range(50):
        records.verdicts[_id(n)] = _v("A" if n % 2 else "B", 10)
        settler.observe(_id(n), records.verdicts[_id(n)])
    archives.other, clock.now = 46001, 60.0
    assert asyncio.run(settler.settle_once()) == 46001
    assert records.reads == 0
    assert json.loads(archives.written[46001])["rewards_by_hotkey"] == pytest.approx(
        {"A": 0.05, "B": 0.05})


def test_unfed_verdicts_are_read_together():
    records = _SlowVerdictReads({_id(n): _v("A", 10) for n in range(40)})
    archives, clock = _Archives(46000), _Clock()
    assert asyncio.run(_settler(records, archives, clock).settle_once()) == 46000
    assert records.reads == 40 and records.peak > 1
    assert records.state["settled"] == sorted(_id(n) for n in range(40))
