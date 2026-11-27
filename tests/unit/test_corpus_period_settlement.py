"""Settling a period-settled corpus task: dated by arrival, paid once a period closes."""

from __future__ import annotations

import asyncio

import pytest

from reliquary.infrastructure.corpus_period_store import parse_key, period_archive_key
from reliquary.validator import corpus_periods as cp
from reliquary.validator.corpus_period_settlement import CorpusPeriodSettler

GENESIS = 1_000_000.0
P = cp.PERIOD_SECONDS


def at(period, offset=10.0):
    return GENESIS + period * P + offset


class Records:
    def __init__(self):
        self.verdicts, self.voided, self.state, self.writes = {}, set(), {}, 0

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)

    async def list_voided_ids(self, job_id):
        return sorted(self.voided)

    async def read_settlement(self, job_id):
        return dict(self.state), "etag"

    async def write_settlement(self, job_id, state, etag):
        self.writes += 1
        self.state = dict(state)
        return "etag"


class Archives:
    def __init__(self, fail_once=False):
        self.docs, self.fail_once = {}, fail_once

    async def write(self, task_id, work, entry, document):
        if self.fail_once:
            self.fail_once = False
            raise ConnectionError("lost")
        assert (work, entry) not in self.docs, "an archive is never overwritten"
        self.docs[(work, entry)] = document

    async def read(self, task_id, work, entry):
        return self.docs.get((work, entry))

    async def list(self, task_id):
        return sorted(self.docs)


def verdict(sid, hotkey, tokens, received, passed=True):
    return {"submission_id": sid, "hotkey": hotkey, "token_count": tokens,
            "passed": passed, "received_at": received, "audited_at": received + 9000}


def settler(records, archives, now, oldest=None, cap=0.1):
    return CorpusPeriodSettler(task_id="eval-x", job_id="eval-x", cap=cap, records=records,
                               archives=archives, oldest_pending=lambda: oldest,
                               genesis=lambda: GENESIS, clock=lambda: now)


def test_observation_sees_committed_archive_even_if_its_local_write_fails():
    records, archives = Records(), Archives()
    records.verdicts["one"] = verdict("one", "miner", 12, at(1))
    run = settler(records, archives, at(4))
    seen = []

    def failed_disk(archive, ids, batch):
        assert list(archives.docs.values()) == [archive]
        assert records.state["settled"] == ["one"]
        seen.append((ids, batch[0]["token_count"]))
        raise OSError("local observation unavailable")

    run.on_observed_archive = failed_disk
    assert asyncio.run(run.settle_once()) == 1
    assert seen == [(["one"], 12)]
    assert asyncio.run(run.settle_once()) is None
    assert len(archives.docs) == 1


def test_each_closed_period_is_paid_its_own_cap_by_its_own_tokens():
    records, archives = Records(), Archives()
    records.verdicts = {
        "a1": verdict("a1", "a", 300, at(3)), "b1": verdict("b1", "b", 100, at(3)),
        "a2": verdict("a2", "a", 50, at(4)),
        "c5": verdict("c5", "c", 10, at(5)),  # period 5 is still open
    }
    # Now in period 5, nothing pending: periods up to 4 are closed.
    assert asyncio.run(settler(records, archives, at(5, 1000)).settle_once()) == 4
    # Both enter the next period: a backlog is paid back at once, up to
    # CATCHUP_ENTRIES archives a period.
    assert set(archives.docs) == {(3, 6), (4, 6)}
    assert archives.docs[(3, 6)]["rewards_by_hotkey"] == {"a": pytest.approx(0.075),
                                                          "b": pytest.approx(0.025)}
    assert archives.docs[(4, 6)]["rewards_by_hotkey"] == {"a": pytest.approx(0.1)}
    assert sorted(records.state["settled"]) == ["a1", "a2", "b1"]
    assert records.state["totals"]["verified_tokens"] == 450


def test_an_audit_backlog_does_not_merge_two_periods():
    """Verdicts of periods 3 and 4 written in the same instant: two archives,
    each paid its own cap, not one cap for both."""
    records, archives = Records(), Archives()
    records.verdicts = {"x": verdict("x", "a", 10, at(3)), "y": verdict("y", "b", 10, at(4))}
    asyncio.run(settler(records, archives, at(9)).settle_once())
    assert sum(sum(d["rewards_by_hotkey"].values()) for d in archives.docs.values()) == \
        pytest.approx(0.2)


def test_a_period_with_an_undecided_submission_waits():
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 10, at(2)), "b": verdict("b", "b", 10, at(3))}
    # Something received in period 3 is still being audited.
    asyncio.run(settler(records, archives, at(6), oldest=at(3, 500)).settle_once())
    assert set(archives.docs) == {(2, 7)}
    assert records.state["settled"] == ["a"]


def test_nothing_is_settled_while_the_auditor_cannot_tell():
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 10, at(2))}

    def unseeded():
        raise LookupError("not seeded")

    s = CorpusPeriodSettler(task_id="t", job_id="t", cap=0.1, records=records,
                            archives=archives, oldest_pending=unseeded,
                            genesis=lambda: GENESIS, clock=lambda: at(9))
    assert asyncio.run(s.settle_once()) is None and archives.docs == {}


def test_a_period_with_nothing_payable_burns_and_is_never_reconsidered():
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 10, at(2), passed=False)}
    records.voided = set()
    asyncio.run(settler(records, archives, at(9)).settle_once())
    assert archives.docs == {} and records.state["settled"] == ["a"]


def test_a_voided_pass_is_settled_unpaid():
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 10, at(2)), "b": verdict("b", "b", 30, at(2))}
    records.voided = {"b"}
    asyncio.run(settler(records, archives, at(9)).settle_once())
    assert archives.docs[(2, 10)]["rewards_by_hotkey"] == {"a": pytest.approx(0.1)}


def test_a_crash_after_choosing_pays_once():
    records, archives = Records(), Archives(fail_once=True)
    records.verdicts = {"a": verdict("a", "a", 10, at(2))}
    with pytest.raises(ConnectionError):
        asyncio.run(settler(records, archives, at(9)).settle_once())
    assert records.state["pending"]["work_period"] == 2
    # Restarted in period 15: the weight-sets of 10..15 never saw it, so it enters
    # at 16 instead, whole, rather than at 10 with most of its pay decayed away.
    assert asyncio.run(settler(records, archives, at(15)).settle_once()) == 2
    assert set(archives.docs) == {(2, 16)} and records.state["pending"] is None


def test_a_crash_after_the_write_does_not_write_again():
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 10, at(2))}
    s = settler(records, archives, at(9))

    async def crash_after_write():
        original = records.write_settlement
        calls = []

        async def flaky(job_id, state, etag):
            calls.append(state)
            if state.get("pending") is None and len(calls) > 1:
                raise ConnectionError("lost after the archive")
            return await original(job_id, state, etag)

        records.write_settlement = flaky
        with pytest.raises(ConnectionError):
            await s.settle_once()
        records.write_settlement = original

    asyncio.run(crash_after_write())
    assert set(archives.docs) == {(2, 10)} and records.state["pending"] is not None
    asyncio.run(settler(records, archives, at(14)).settle_once())
    assert set(archives.docs) == {(2, 10)} and records.state["pending"] is None


def test_a_backlog_enters_a_few_archives_a_period():
    records, archives = Records(), Archives()
    records.verdicts = {f"v{p}": verdict(f"v{p}", "a", 10, at(p)) for p in range(2, 8)}
    asyncio.run(settler(records, archives, at(9)).settle_once())
    k = cp.CATCHUP_ENTRIES
    assert sorted(entry for _, entry in archives.docs) == sorted(
        10 + i // k for i in range(6))


def test_a_new_period_enters_beside_a_queue_not_behind_it():
    """A backlog queued one per period by an older settler (entries 10..20)
    does not hold the next period's pay: it enters as soon as there is room."""
    records, archives = Records(), Archives()
    for work, entry in zip(range(0, 11), range(10, 21)):
        archives.docs[(work, entry)] = {"rewards_by_hotkey": {"old": 0.1}}
    records.verdicts = {"n": verdict("n", "a", 10, at(11))}
    asyncio.run(settler(records, archives, at(12, 1000)).settle_once())
    assert (11, 13) in archives.docs


def test_a_window_settled_job_is_not_settled_by_period():
    records, archives = Records(), Archives()
    records.state = {"last_window": 45000, "settled": [], "pending": None}
    records.verdicts = {"a": verdict("a", "a", 10, at(2))}
    assert asyncio.run(settler(records, archives, at(9)).settle_once()) is None
    assert archives.docs == {}


def test_late_verdicts_of_a_paid_period_are_paid_against_its_whole_count():
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 30, at(2))}
    asyncio.run(settler(records, archives, at(9)).settle_once())
    records.verdicts["b"] = verdict("b", "b", 10, at(2))
    asyncio.run(settler(records, archives, at(9, 3000)).settle_once())
    # Same settling period, its own entry: the first payees' archive is untouched.
    assert archives.docs[(2, 10)]["rewards_by_hotkey"] == {"a": pytest.approx(0.1)}
    assert archives.docs[(2, 11)]["rewards_by_hotkey"] == {"b": pytest.approx(0.1 * 10 / 40)}


def test_a_parked_records_late_verdict_is_paid_within_what_is_left_of_its_cap():
    """Owner, 2026-11-26: only an order's parked record, decided after its period closed without it, is paid within what
    is left of the cap (``capped``); any other late verdict keeps the whole-count pay above, in the same call too."""
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 30, at(2))}
    asyncio.run(settler(records, archives, at(9)).settle_once())
    records.verdicts["p"] = verdict("p", "p", 10, at(2))       # the parked record, decided late
    records.verdicts["b"] = verdict("b", "b", 10, at(2))       # an ordinary late verdict
    late = CorpusPeriodSettler(task_id="eval-x", job_id="eval-x", cap=0.1, records=records, archives=archives,
                               oldest_pending=lambda: None, genesis=lambda: GENESIS, clock=lambda: at(9, 3000),
                               capped=lambda verdicts, period_of: {sid for sid in verdicts if sid == "p"})
    asyncio.run(late.settle_once())
    assert archives.docs[(2, 11)]["rewards_by_hotkey"] == {"b": pytest.approx(0.1 * 10 / 40)}
    assert "p" in records.state["settled"] and set(archives.docs) == {(2, 10), (2, 11)}   # its period paid its cap
    # With room left (a cap raised since), its share of the whole count, never past the cap.
    records.verdicts["q"] = verdict("q", "q", 400, at(2))
    raised = CorpusPeriodSettler(task_id="eval-x", job_id="eval-x", cap=0.2, records=records, archives=archives,
                                 oldest_pending=lambda: None, genesis=lambda: GENESIS, clock=lambda: at(9, 3500),
                                 capped=lambda verdicts, period_of: set(verdicts))
    asyncio.run(raised.settle_once())
    # Its share (0.2 x 400/440) is past what is left of the cap (0.2 - 0.1 - 0.025): it gets that, never more.
    assert archives.docs[(2, 12)]["rewards_by_hotkey"] == {"q": pytest.approx(0.2 - 0.1 - 0.1 * 10 / 40)}
    assert sum(v for doc in archives.docs.values() for v in doc["rewards_by_hotkey"].values()) == pytest.approx(0.2)


def test_a_parked_record_of_a_period_settled_for_the_first_time_is_paid_with_it():
    """Capped only when its period was settled before without it: one decided before its period's first settlement
    shares its cap with the others, whatever kept the period open meanwhile."""
    records, archives = Records(), Archives()
    records.verdicts = {"o": verdict("o", "o", 10, at(2)), "p": verdict("p", "p", 10, at(2))}
    first = CorpusPeriodSettler(task_id="eval-x", job_id="eval-x", cap=0.1, records=records, archives=archives,
                                oldest_pending=lambda: None, genesis=lambda: GENESIS, clock=lambda: at(9),
                                capped=lambda verdicts, period_of: {"p"} & set(verdicts))
    asyncio.run(first.settle_once())
    assert archives.docs[(2, 10)]["rewards_by_hotkey"] == {"o": pytest.approx(0.05), "p": pytest.approx(0.05)}


def test_a_periods_ordinary_late_verdicts_are_settled_before_its_capped_ones():
    """With room left under the cap, the ordinary late verdict takes its whole-count share first; the parked record
    gets what is left of the cap after it."""
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 30, at(2))}
    asyncio.run(settler(records, archives, at(9)).settle_once())            # cap 0.1, paid in full
    records.verdicts["p"] = verdict("p", "p", 10, at(2))
    records.verdicts["b"] = verdict("b", "b", 10, at(2))
    late = CorpusPeriodSettler(task_id="eval-x", job_id="eval-x", cap=0.2, records=records, archives=archives,
                               oldest_pending=lambda: None, genesis=lambda: GENESIS, clock=lambda: at(9, 3000),
                               capped=lambda verdicts, period_of: {"p"} & set(verdicts))
    asyncio.run(late.settle_once())
    paid = {hk: v for doc in archives.docs.values() for hk, v in doc["rewards_by_hotkey"].items()}
    assert paid["b"] == pytest.approx(0.2 * 10 / 40)
    assert paid["p"] == pytest.approx(min(0.2 * 10 / 50, 0.2 - 0.1 - 0.2 * 10 / 40))


def test_parked_and_held_late_verdicts_stay_within_the_cap_and_ordinary_late_ones_keep_their_pay():
    """Owner, 2026-11-26: a parked record decided late and the verdicts its stake hold kept back are paid within what is
    left of the cap, so the period never pays past it; any other late verdict keeps the whole-count pay."""
    records, archives = Records(), Archives()
    records.verdicts = {"a": verdict("a", "a", 30, at(2))}
    asyncio.run(settler(records, archives, at(9)).settle_once())            # cap 0.1, paid in full

    def with_hooks(now):
        return CorpusPeriodSettler(task_id="eval-x", job_id="eval-x", cap=0.1, records=records, archives=archives,
                                   oldest_pending=lambda: None, genesis=lambda: GENESIS, clock=lambda: now,
                                   capped=lambda verdicts, period_of: {"p", "z"} & set(verdicts))

    records.verdicts["p"] = verdict("p", "p", 10, at(2))                    # the parked record
    records.verdicts["z"] = verdict("z", "p", 20, at(2))                    # what its hold kept back
    asyncio.run(with_hooks(at(9, 3000)).settle_once())
    assert {"p", "z"} <= set(records.state["settled"])
    assert sum(v for doc in archives.docs.values() for v in doc["rewards_by_hotkey"].values()) == pytest.approx(0.1)
    records.verdicts["b"] = verdict("b", "b", 10, at(2))                    # an ordinary late verdict
    asyncio.run(with_hooks(at(9, 3500)).settle_once())
    assert {hk: v for doc in archives.docs.values() for hk, v in doc["rewards_by_hotkey"].items() if hk == "b"} \
        == {"b": pytest.approx(0.1 * 10 / 40)}


def test_an_old_verdict_without_arrival_is_dated_by_its_audit():
    records, archives = Records(), Archives()
    old = verdict("a", "a", 10, at(2))
    del old["received_at"]
    old["audited_at"] = at(4)
    records.verdicts = {"a": old}
    asyncio.run(settler(records, archives, at(9)).settle_once())
    assert set(archives.docs) == {(4, 10)}


def test_archive_keys_sort_and_parse():
    key = period_archive_key("eval-x", 12, 15)
    assert key == "reliquary/corpus-periods/eval-x/0000000012-0000000015.json.gz"
    assert parse_key(key) == (12, 15)
    with pytest.raises(ValueError):
        period_archive_key("eval-x", 12, 11)
