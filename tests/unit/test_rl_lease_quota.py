"""The per-hotkey lease quota (rl-async-design.md §4.3): start small, double while the full quota is used and
delivered on time (slow start), halve on every expiry, settle at measured throughput x latency x headroom
(Little's law), out-of-zone submissions count as delivered work, and every number is explained."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from reliquary.validator.rl_lease_quota import QuotaConfig, QuotaController

T0 = 1_800_000_000.0


def lease(issued_at: float, expires_at: float | None = None):
    return SimpleNamespace(issued_at=issued_at, expires_at=expires_at or issued_at + 600)


def full_round(quota: QuotaController, hotkey: str, now: float, latency: float = 10.0) -> float:
    """Lease the whole quota, then deliver all of it on time."""
    n = quota.quota(hotkey)
    for _ in range(n):
        quota.record_lease(hotkey, now=now)
    for _ in range(n):
        quota.record_submission(hotkey, lease(now), in_zone=True, now=now + latency)
    return now + latency


def test_a_new_hotkey_starts_at_the_initial_quota_and_says_why():
    quota = QuotaController(QuotaConfig(initial=4))
    state = quota.state("new")
    assert state["quota"] == 4 and state["phase"] == "slow_start" and "new hotkey" in state["reason"]
    assert state["counters"] == {"leased": 0, "submitted": 0, "in_zone": 0, "out_of_zone": 0, "expired": 0,
                                 "rejected": 0}


def test_slow_start_doubles_after_a_full_quota_delivered_on_time():
    quota = QuotaController(QuotaConfig(initial=4, ceiling=64))
    now = T0
    seen = [quota.quota("A")]
    for _ in range(6):
        now = full_round(quota, "A", now)
        seen.append(quota.quota("A"))
    assert seen == [4, 8, 16, 32, 64, 64, 64]                    # capped at the ceiling
    assert "slow start" in quota.state("A")["reason"]


def test_no_doubling_without_using_the_full_quota():
    quota = QuotaController(QuotaConfig(initial=4))
    for i in range(3):                                           # never more than 3 in flight
        quota.record_lease("A", now=T0 + i)
    for i in range(3):
        quota.record_submission("A", lease(T0), in_zone=True, now=T0 + 10 + i)
    assert quota.quota("A") == 4


def test_every_expiry_halves_the_quota_down_to_the_floor():
    quota = QuotaController(QuotaConfig(initial=32, floor=2))
    for expected in (16, 8, 4, 2, 2):
        quota.record_expiry("A", lease(T0), now=T0 + 700)
        assert quota.quota("A") == expected
    state = quota.state("A")
    assert state["counters"]["expired"] == 5 and "expired" in state["reason"] and state["phase"] == "steady"


def test_an_expiry_inside_a_round_cancels_its_doubling():
    quota = QuotaController(QuotaConfig(initial=8))
    for _ in range(8):
        quota.record_lease("A", now=T0)
    for _ in range(7):
        quota.record_submission("A", lease(T0), in_zone=True, now=T0 + 10)
    quota.record_expiry("A", lease(T0), now=T0 + 601)
    assert quota.quota("A") == 4


def test_out_of_zone_submissions_count_as_delivered_work():
    quota = QuotaController(QuotaConfig(initial=4))
    for _ in range(4):
        quota.record_lease("A", now=T0)
    for _ in range(4):
        quota.record_submission("A", lease(T0), in_zone=False, now=T0 + 10)
    state = quota.state("A")
    assert state["quota"] == 8 and state["counters"]["out_of_zone"] == 4 and state["counters"]["in_zone"] == 0


def test_littles_law_steady_state_for_a_miner_of_fixed_capacity():
    """A miner that keeps c groups in flight, each taking L seconds, settles at about c x headroom."""
    c, L = 10, 60.0
    quota = QuotaController(QuotaConfig(initial=4, headroom=1.5, window_s=1800.0))
    in_flight: list[float] = []
    now = T0
    for _ in range(3000):
        while len(in_flight) < min(c, quota.quota("A")):
            in_flight.append(now)
            quota.record_lease("A", now=now)
        now += L / c                                          # one completes every L / c seconds
        issued = in_flight.pop(0)
        quota.record_submission("A", lease(issued), in_zone=True, now=now)
    state = quota.state("A")
    assert state["phase"] == "steady"
    assert state["throughput_per_min"] == pytest.approx(c * 60.0 / L, rel=0.1)
    assert state["latency_s"] == pytest.approx(L, rel=0.15)
    assert quota.quota("A") == pytest.approx(math.ceil(1.5 * c), abs=2)
    assert "throughput" in state["reason"]


def test_a_steady_miner_that_grows_its_capacity_regains_quota():
    L = 60.0
    quota = QuotaController(QuotaConfig(initial=4, headroom=1.5, window_s=600.0))
    in_flight: list[float] = []
    now = T0

    def run(capacity, steps):
        nonlocal now
        for _ in range(steps):
            while len(in_flight) < min(capacity, quota.quota("A")):
                in_flight.append(now)
                quota.record_lease("A", now=now)
            now += L / max(1, len(in_flight))
            quota.record_submission("A", lease(in_flight.pop(0)), in_zone=True, now=now)

    run(6, 1500)
    small = quota.quota("A")
    run(40, 4000)
    assert small <= 12 and quota.quota("A") >= 40


def test_floor_and_ceiling_are_configurable_and_checked():
    with pytest.raises(ValueError):
        QuotaConfig(initial=4, floor=8)
    with pytest.raises(ValueError):
        QuotaConfig(initial=4, ceiling=2)
    with pytest.raises(ValueError):
        QuotaConfig(headroom=0.5)
    quota = QuotaController(QuotaConfig(initial=2, floor=2, ceiling=3))
    full_round(quota, "A", T0)
    assert quota.quota("A") == 3
