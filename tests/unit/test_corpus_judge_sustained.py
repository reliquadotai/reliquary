"""A math backlog of 150k records, all past their hold, rounds cached: the
judge writes at least 20k verdicts an hour (prod 2026-10-03 03:07-03:34:
~2.8k/h settled, ~5.9k/h judged, 6.9k/h arriving)."""

from __future__ import annotations

from reliquary.corpus.audit_policy import AuditParams
from reliquary.validator import corpus_auditor
from tests.unit import corpus_judge_sim as sim

HOLD = 4320.0


def sustained(hours=2.0, backlog=150_000):
    params = AuditParams(q=0.15, probation_submissions=100, hold_seconds=HOLD,
                         ban_after_failures=1000)
    population = sim.Population.prod_like(seed=0, probation=0)
    population.tokens = 4200
    result = sim.simulate(
        corpus_auditor, rate_per_hour=6900.0, hours=hours, params=params, population=population,
        store=sim.Store(seed=0, latency=(0.05, 0.10), write_latency=(1.2, 1.8),
                        in_flight_cap=64),
        gpu=sim.Gpu(tokens_per_second=8200.0), beacon=sim.Beacon(latency=0.0),
        start_backlog=backlog, backlog_age=15 * 3600.0, backlog_newest=HOLD + 600.0,
        sample_every=900.0)
    series = result.pending_series
    (h0, _, _, v0), (h1, _, _, v1) = series[2], series[-1]
    return (v1 - v0) / (h1 - h0), series


def test_a_150k_old_backlog_is_judged_at_20k_verdicts_an_hour():
    rate, series = sustained()
    print(f"\n150k old backlog: {rate:.0f} verdicts/h, series {series}")
    assert rate >= 20_000, (rate, series)
