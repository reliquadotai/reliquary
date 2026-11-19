"""A training task publishes what it paid miners, so the platform can bill the customer (reliquary/validator/rl_spend.py)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from reliquary.validator import corpus_periods as cp
from reliquary.validator.rl_leases import LeaseRequest
from reliquary.validator.rl_ledger import LedgerConfig, RlPayLedger
from reliquary.validator.rl_spend import SpendPublisher, spend_key, spend_summary
from tests.unit.test_rl_auction import Clock, _limited
from tests.unit.test_rl_submit import ENV, NOW, group, service

EMISSION = 1.0                                       # ``auctions_for``: 1 alpha per period
PERIOD = cp.period_of(NOW)
AFTER = cp.period_end(PERIOD) + 3600.0               # the period closed and past its slack


def _rl(tmp_path, *, cap=0.5, fill=False, **limits):
    book, auctions, clock = _limited(tmp_path, **limits)
    ledger = RlPayLedger(book, service(book, probation_groups=0), None,
                         LedgerConfig(task_id="order-train-t1", cap=cap, outage_fill=fill), clock=clock)
    rl = SimpleNamespace(settings=SimpleNamespace(task_id="order-train-t1", run="order-train-t1"), book=book,
                         ledger=ledger)
    return rl, clock


def _accrue(rl, seq, *, price, tokens, amount=None, kind="trained", env=ENV, hotkey="A", status="accrued",
            at=NOW):
    amount = tokens * price if amount is None else amount
    gross_tokens = tokens if kind == "trained" else amount / price
    rl.book.db.execute("INSERT INTO rl_accruals(seq, lease_id, hotkey, environment, kind, at, period, matched_tokens, "
                       "token_price, amount, status, budget_tokens) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (seq, f"l{seq}", hotkey, env, kind, at, cp.period_of(at), tokens, price, amount, status,
                        gross_tokens))
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
    assert summary["dead"] == []                                                    # never polled
    assert summary["stopped"] == "deadline" and summary["latest_checkpoint"] == 4
    assert summary["stopped_at"] == NOW + 10                                        # the deadline itself
    assert summary["pending"] == {"open_leases": 0, "buffered": 0, "audits": 0, "unledgered": 0}
    assert summary["periods"] == []


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

    async def publish(archives):
        calls.append("publish")

    ledger.publish = publish
    rl = SimpleNamespace(ledger=ledger)

    async def run():
        loops = rl_async_loops(rl, archives=object(), spend=Spend(), every={"ledger": 60})
        task = asyncio.ensure_future(loops[2])                                     # the ledger loop
        for _ in range(50):
            await asyncio.sleep(0.01)
            if "publish" in calls:
                break
        task.cancel()
        for other in loops[:2] + loops[3:]:
            other.close()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    # the summary first: an archive publish that fails does not hold it back
    assert calls[:4] == ["ingest", "settle", ("spend", True), "publish"]



def _deliver(rl, clock, prompt, *, train=True):
    lease = rl.book.grant([LeaseRequest("A", ENV, (prompt,), 0)])[0].leases[0]
    rl.ledger.submissions.submit(lease.lease_id, "A", group(8))
    if train:
        rl.ledger.submissions.report_training(lease.lease_id, rollout_match=[1.0] * 16, accepted=True)
    return lease


def test_a_lease_book_order_becomes_final_once_settled(tmp_path):
    from reliquary.admin.training import training_spend

    rl, clock = _rl(tmp_path, deadline_at=NOW + 10)
    rl.book.auctions.post_ask(ENV, "A", 0.5, 8)
    lease = rl.book.grant([LeaseRequest("A", ENV, (1,), 0)])[0].leases[0]
    entry = SimpleNamespace(status="retired", params={"training_order": {"usd_per_alpha": 4.0}})

    def final(now, retired_time=NOW + 20):
        rl.ledger.ingest()
        rl.ledger.settle(now=now)
        return training_spend(entry, spend_summary(rl, now=now), retired_time=retired_time)["final"]

    assert spend_summary(rl, now=NOW)["pending"]["open_leases"] == 1
    assert final(NOW + 1) is False                                               # a lease is out
    rl.ledger.submissions.submit(lease.lease_id, "A", group(8))
    assert spend_summary(rl, now=NOW + 2)["pending"]["buffered"] == 1             # the trainer may still train it
    rl.ledger.submissions.report_training(lease.lease_id, rollout_match=[1.0] * 16, accepted=True)
    assert spend_summary(rl, now=NOW + 3)["pending"]["unledgered"] == 1           # not yet in the ledger
    clock.t = NOW + 20                                                           # stopped by its deadline
    assert final(NOW + 20) is False                                              # its period is still open
    margin = spend_summary(rl, now=NOW + 20)["settle_margin_s"]
    assert final(cp.period_end(PERIOD) + margin - 1) is False                    # a summary from before the close
    assert final(cp.period_end(PERIOD) + margin + 1) is True
    summary = spend_summary(rl, now=cp.period_end(PERIOD) + margin + 1)
    [period] = summary["periods"]
    assert period["settled"] and period["paid_alpha"] > 0
    late = cp.period_end(PERIOD) + 60                                             # retired in the next period
    assert final(cp.period_end(PERIOD) + margin + 1, retired_time=late) is False
    assert final(cp.period_end(PERIOD + 1) + margin + 1, retired_time=late) is True


def test_groups_left_with_a_gone_trainer_cannot_pay(tmp_path):
    rl, clock = _rl(tmp_path, fill=True, deadline_at=NOW + 10)
    rl.book.auctions.post_ask(ENV, "A", 0.5, 8)
    _deliver(rl, clock, 1, train=False)
    rl.ledger.trainer_beat(now=NOW)
    clock.t = NOW + 20
    assert spend_summary(rl, now=NOW + 20)["pending"]["buffered"] == 1             # the trainer may still take it
    gone = NOW + 20 + rl.ledger.config.outage_grace_s
    clock.t = gone
    assert spend_summary(rl, now=gone)["pending"]["buffered"] == 0               # stopped, and the trainer is gone


def test_outage_fill_stops_at_the_orders_stop(tmp_path):
    book, auctions, clock = _limited(tmp_path, deadline_at=NOW + 10)
    ledger = RlPayLedger(book, service(book, probation_groups=0), None,
                         LedgerConfig(task_id="order-train-t1", cap=0.5, outage_fill=True), clock=clock)
    rl = SimpleNamespace(settings=SimpleNamespace(task_id="order-train-t1", run="order-train-t1"), book=book,
                         ledger=ledger)
    t = cp.period_end(PERIOD) - cp.PERIOD_SECONDS
    while t <= AFTER:                                                             # alive through its period
        ledger.trainer_beat(now=t)
        t += 500.0
    _accrue(rl, 1, price=1e-6, tokens=1000)
    clock.t = AFTER
    assert [d["work_period"] for d in ledger.settle(now=AFTER)] == [PERIOD]
    later = cp.period_end(PERIOD + 3) + 3600.0                                    # the trainer never came back
    clock.t = later
    assert ledger.settle(now=later) == []                                         # no fill after the stop


@pytest.mark.parametrize("fence", ["registry_paused", "registry_unavailable", "ledger_pending", "trainer_starting"])
def test_an_admission_fence_does_not_finalize_an_order_stop(tmp_path, fence):
    rl, clock = _rl(tmp_path, deadline_at=NOW + 100)
    auctions = rl.book.auctions
    auctions.admission = lambda: fence
    assert auctions.budget(ENV) == 0
    assert rl.ledger.stopped_at() is None
    assert rl.book.db.execute("SELECT at FROM rl_order_stop").fetchone() is None
    auctions.admission = None
    assert auctions.budget(ENV) > 0
    auctions.admission = lambda: fence
    clock.t = NOW + 110
    assert rl.ledger.stopped_at() == NOW + 100
    rl.book.close()


def test_a_retired_order_stops_leasing(tmp_path):
    rl, clock = _rl(tmp_path, deadline_at=NOW + 86_400)
    auctions = rl.book.auctions
    auctions.post_ask(ENV, "A", 0.5, 8)
    assert auctions.stopped() is None and auctions.budget(ENV) > 0
    auctions.retire()                                                            # e.g. the customer cancelled
    assert auctions.stopped() == "retired" and auctions.budget(ENV) == 0
    assert not rl.book.grant([LeaseRequest("A", ENV, (1,), 0)])[0].leases
    assert spend_summary(rl, now=NOW + 5)["stopped_at"] == NOW + 5


def test_the_controller_follows_its_tasks_retire():
    from reliquary.validator.rl_async_service import rl_async_loops

    retired = []
    statuses = iter(["active", "retired"])

    async def status():
        return next(statuses, "retired")

    rl = SimpleNamespace(book=SimpleNamespace(auctions=SimpleNamespace(retire=lambda: retired.append(True))))

    async def run():
        loops = rl_async_loops(rl, archives=None, registry_status=status, every={"registry": 0.01, "expire": 60,
                                                                                  "audit": 60, "ledger": 60,
                                                                                  "price": 60})
        task = asyncio.ensure_future(loops[-1])                                    # the registry follow
        for _ in range(100):
            await asyncio.sleep(0.01)
            if retired:
                break
        task.cancel()
        for other in loops[:-1]:
            other.close()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert retired


# --- final review ----------------------------------------------------------------------------------------------
def test_the_summary_names_the_last_period_its_settle_closed(tmp_path):
    rl, clock = _rl(tmp_path)
    assert spend_summary(rl, now=NOW)["settled_through"] is None                  # no settle yet
    rl.ledger.settle(now=AFTER)
    clock.t = AFTER + cp.PERIOD_SECONDS
    assert spend_summary(rl, now=AFTER + cp.PERIOD_SECONDS)["settled_through"] == PERIOD   # its settle's, not now's


def test_groups_left_when_no_trainer_ever_called_stop_blocking_after_the_grace(tmp_path):
    rl, clock = _rl(tmp_path, deadline_at=NOW + 10)
    rl.book.auctions.post_ask(ENV, "A", 0.5, 8)
    _deliver(rl, clock, 1, train=False)
    clock.t = NOW + 20
    summary = spend_summary(rl, now=NOW + 20)
    assert summary["stopped_at"] == NOW + 10 and summary["pending"]["buffered"] == 1
    gone = NOW + 10 + rl.ledger.config.outage_grace_s + 1
    clock.t = gone
    assert spend_summary(rl, now=gone)["pending"]["buffered"] == 0


def test_a_failing_spend_write_does_not_hold_the_archive_publish():
    from reliquary.validator.rl_async_service import rl_async_loops

    calls = []
    ledger = SimpleNamespace(ingest=lambda: 0, settle=lambda: [])

    class Spend:
        async def write(self, *, settled):
            calls.append("spend")
            raise RuntimeError("bucket down")

    async def publish(archives):
        calls.append("publish")

    ledger.publish = publish
    rl = SimpleNamespace(ledger=ledger)

    async def run():
        loops = rl_async_loops(rl, archives=object(), spend=Spend(), every={"ledger": 60})
        task = asyncio.ensure_future(loops[2])
        for _ in range(50):
            await asyncio.sleep(0.01)
            if "publish" in calls:
                break
        task.cancel()
        for other in loops[:2] + loops[3:]:
            other.close()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert calls[:2] == ["spend", "publish"]


def test_the_summary_lists_the_dead_pieces_and_caps_the_open_one_at_the_stop(tmp_path):
    rl, clock = _rl(tmp_path, deadline_at=NOW + 1000)
    rl.book.db.execute("INSERT INTO rl_trainer_dead(start, end) VALUES(?, ?)", (NOW - 900, NOW - 500))
    rl.book.db.execute("INSERT INTO rl_trainer_poll(id, at) VALUES(0, ?)", (NOW - 400,))
    rl.book.db.commit()
    clock.t = NOW + 600
    summary = spend_summary(rl, now=clock())
    assert summary["dead"] == [[NOW - 900, NOW - 500], [NOW - 220, NOW + 600]]      # open: last poll + 180 to now
    clock.t = NOW + 2000
    summary = spend_summary(rl, now=clock())
    assert summary["stopped_at"] == NOW + 1000
    assert summary["dead"][-1] == [NOW - 220, NOW + 1000]                           # capped at the stop
