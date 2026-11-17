"""A training task publishes what it paid miners, so the platform can bill the customer (reliquary/validator/rl_spend.py)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from reliquary.validator import corpus_periods as cp
from reliquary.validator.rl_ledger import LedgerConfig, RlPayLedger
from reliquary.validator.rl_spend import SpendPublisher, spend_key, spend_summary
from tests.unit.test_rl_auction import Clock, _limited
from tests.unit.test_rl_submit import ENV, NOW, service

EMISSION = 1.0                                       # ``auctions_for``: 1 alpha per period
PERIOD = cp.period_of(NOW)
AFTER = cp.period_end(PERIOD) + 3600.0               # the period closed and past its slack


def _rl(tmp_path, *, cap=0.5, **limits):
    book, auctions, clock = _limited(tmp_path, **limits)
    ledger = RlPayLedger(book, service(book, probation_groups=0), None, LedgerConfig(task_id="order-train-t1", cap=cap),
                         clock=clock)
    rl = SimpleNamespace(settings=SimpleNamespace(task_id="order-train-t1", run="order-train-t1"), book=book,
                         ledger=ledger)
    return rl, clock


def _accrue(rl, seq, *, price, tokens, amount=None, kind="trained", env=ENV, hotkey="A", status="accrued",
            at=NOW):
    amount = tokens * price if amount is None else amount
    rl.book.db.execute("INSERT INTO rl_accruals VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (seq, f"l{seq}", hotkey, env, kind, at, cp.period_of(at), tokens, price, amount, status))
    rl.book.db.commit()


def test_buckets_group_accruals_by_lease_price(tmp_path):
    rl, _ = _rl(tmp_path)
    _accrue(rl, 1, price=1e-6, tokens=1000)
    _accrue(rl, 2, price=1e-6, tokens=500, hotkey="B")
    _accrue(rl, 3, price=2e-6, tokens=100, amount=100 * 2e-6 - 20 * 2e-6)          # a debit of 20 tokens' worth
    _accrue(rl, 4, price=1e-6, tokens=0, amount=0.25 * 1e-6 * 800, kind="first_scan")
    _accrue(rl, 5, price=1e-6, tokens=4000, status="forfeited")                     # never paid: not billed
    _accrue(rl, 6, price=3e-6, tokens=10, env="other")
    summary = spend_summary(rl, now=NOW + 60)
    [period] = summary["periods"]
    assert period["period"] == PERIOD and period["settled"] is False
    buckets = period["buckets"]
    assert sorted(buckets) == [ENV, "other"]
    assert [p for p, _ in buckets[ENV]] == pytest.approx([1e-6 * EMISSION, 2e-6 * EMISSION])
    assert [t for _, t in buckets[ENV]] == pytest.approx([1000 + 500 + 200, 80])
    assert buckets["other"] == [[pytest.approx(3e-6), pytest.approx(10)]]
    assert period["accrued_alpha"] == pytest.approx((1500 + 200) * 1e-6 + 80 * 2e-6 + 30e-6)
    assert "A" not in json.dumps(summary) and "B" not in json.dumps(summary)         # no hotkeys
    assert summary["task_id"] == summary["run"] == "order-train-t1"
    assert summary["written_at"] == NOW + 60


def test_a_summary_counts_settled_pay_after_the_cap(tmp_path):
    rl, _ = _rl(tmp_path, cap=0.01)
    _accrue(rl, 1, price=1e-6, tokens=15_000)                                      # 0.015: over the cap
    _accrue(rl, 2, price=1e-6, tokens=5_000, hotkey="B")                           # 0.005
    assert len(rl.ledger.settle(now=AFTER)) == 1
    [period] = spend_summary(rl, now=AFTER)["periods"]
    assert period["settled"] is True
    assert period["accrued_alpha"] == pytest.approx(0.02 * EMISSION)
    assert period["paid_alpha"] == pytest.approx(0.01 * EMISSION)                 # the cap: half burned
    assert period["buckets"] == {ENV: [[pytest.approx(1e-6), pytest.approx(20_000)]]}


def test_an_open_period_is_not_settled(tmp_path):
    rl, _ = _rl(tmp_path)
    _accrue(rl, 1, price=1e-6, tokens=1000)
    assert rl.ledger.settle(now=AFTER) != []
    _accrue(rl, 2, price=1e-6, tokens=300, at=AFTER)                               # the next period, still open
    periods = spend_summary(rl, now=AFTER)["periods"]
    assert [(p["period"], p["settled"]) for p in periods] == [(PERIOD, True), (cp.period_of(AFTER), False)]
    assert periods[1]["paid_alpha"] is None and periods[1]["accrued_alpha"] == pytest.approx(300e-6)


def test_the_summary_carries_starvation_the_stop_and_the_checkpoint(tmp_path):
    rl, clock = _rl(tmp_path, deadline_at=NOW + 10)
    rl.book.db.execute("INSERT INTO rl_trainer_waits(start, end, polled) VALUES(?, ?, NULL)", (NOW - 1000, NOW - 100))
    rl.book.db.commit()
    rl.book.adopt_checkpoint(4)
    clock.t = NOW + 20
    summary = spend_summary(rl, now=clock())
    assert summary["starved"] == [[NOW - 700, NOW - 100]]
    assert summary["stopped"] == "deadline" and summary["latest_checkpoint"] == 4
    assert summary["leases_out"] == 0 and summary["periods"] == []


def test_the_summary_is_written_under_the_run_after_a_settle(tmp_path):
    rl, _ = _rl(tmp_path)
    written = []

    async def put(key, body, content_type, cache_control):
        written.append((key, json.loads(body)))

    clock = Clock(NOW)
    publisher = SpendPublisher(rl, put, clock=clock)
    asyncio.run(publisher.write(settled=False))
    asyncio.run(publisher.write(settled=False))                                    # nothing new: not rewritten
    clock.t += 60
    asyncio.run(publisher.write(settled=True))
    assert [key for key, _ in written] == [spend_key("order-train-t1")] * 2 == ["rl/order-train-t1/spend.json"] * 2
    assert written[-1][1]["written_at"] == NOW + 60


def test_the_ledger_loop_writes_the_summary_after_its_settle():
    from reliquary.validator.rl_async_service import rl_async_loops

    calls = []
    ledger = SimpleNamespace(ingest=lambda: calls.append("ingest") or 0,
                             settle=lambda: calls.append("settle") or [{"work_period": 1, "rewards_by_hotkey": {},
                                                                         "scale": 1.0}])

    class Spend:
        async def write(self, *, settled):
            calls.append(("spend", settled))

    rl = SimpleNamespace(ledger=ledger)

    async def run():
        loops = rl_async_loops(rl, archives=None, spend=Spend(), every={"ledger": 60})
        task = asyncio.ensure_future(loops[2])                                     # the ledger loop
        for _ in range(50):
            await asyncio.sleep(0.01)
            if ("spend", True) in calls:
                break
        task.cancel()
        for other in loops[:2] + loops[3:]:
            other.close()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert calls[:3] == ["ingest", "settle", ("spend", True)]
