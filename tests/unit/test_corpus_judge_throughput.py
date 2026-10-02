"""The judge keeps up with production traffic and drains a restart's backlog,
with the store as slow as measured on 2026-10-02 (reads 50-100 ms, create-only
verdict writes 1.2-1.8 s). Virtual time: hours of traffic in seconds.

The hold is shortened to 10 minutes so a run stays short; the rate is not."""

from __future__ import annotations

from reliquary.corpus.audit_policy import AuditParams
from reliquary.validator import corpus_auditor
from tests.unit import corpus_judge_sim as sim

PROD_RATE = 6900.0  # math submissions an hour on 2026-10-02
HOLD, SLACK = 600.0, 60.0
PARAMS = AuditParams(q=0.15, probation_submissions=100, hold_seconds=HOLD,
                     ban_after_failures=1000)
# A record is pending at least its hold and slack, plus a rescan and a pass.
FLOOR = HOLD + SLACK
MARGIN = 120.0


def _store(seed):
    return sim.Store(seed=seed, latency=(0.05, 0.10), write_latency=(1.2, 1.8),
                     in_flight_cap=64)


def _run(rate, hours, *, backlog=0, backlog_age=0.0, seed=0, drain_hours=0.0):
    return sim.simulate(
        corpus_auditor, rate_per_hour=rate, hours=hours, params=PARAMS, seed=seed,
        store=_store(seed), auditor_kwargs={"accept_slack_seconds": SLACK},
        start_backlog=backlog, backlog_age=backlog_age, sample_every=300.0,
        drain_hours=drain_hours)


def test_three_times_the_production_rate_is_judged_as_fast_as_the_hold_allows():
    rate = 3 * PROD_RATE
    result = _run(rate, hours=1.0)
    # Steady state from the second hold on: nothing waits much past its hold.
    late = [row for row in result.pending_series if row[0] >= 2 * FLOOR / 3600]
    assert late and all(oldest <= FLOOR + MARGIN for _, _, oldest, _ in late), late
    assert result.pending_end <= rate * (FLOOR + MARGIN) / 3600
    assert result.verdicts >= result.arrivals - rate * (FLOOR + MARGIN) / 3600


def test_a_restart_backlog_drains_while_traffic_continues():
    # 20,000 records pending up to two hours, as after the 2026-10-02 restart.
    result = _run(PROD_RATE, hours=1.5, backlog=20_000, backlog_age=7200.0)
    assert result.pending_series[0][1] >= 20_000
    final = result.pending_series[-1]
    assert final[2] <= FLOOR + MARGIN, result.pending_series
    assert result.pending_end <= PROD_RATE * (FLOOR + MARGIN) / 3600
