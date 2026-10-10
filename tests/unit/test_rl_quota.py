"""The allocation of RL sandbox opens: weighted water-filling over entities.
Production regime: production constants, fleets of 216 / 2 160 / 21 600 slots, low and high MAIN demand."""
import time

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.sandbox.rl_capacity import (  # noqa: E402
    LOW_YIELD,
    MAIN,
    NEWCOMER,
    NEWCOMER_MAX_GROUPS,
    NEWCOMER_WEIGHT,
    PRESENT_WINDOW_S,
    USEFUL_HORIZON_S,
    YieldStats,
    entity_sizes,
    sessions_of,
)
from reliquary.sandbox.rl_quota import YieldQuota  # noqa: E402
from reliquary.sandbox.sessions import LIVE  # noqa: E402
from tests.unit.test_rl_episode_wiring import _record  # noqa: E402

P, M = 32, 16
C = 2_160                                              # N = 216, L = 64
NOW = int(time.time())
WASTEFUL = YieldStats(useful=0, decided=200)


def main_stats(concurrent: float, ratio: float = 0.9) -> YieldStats:
    """A MAIN hotkey that held ``concurrent`` useful sessions on average over the last 24 h."""
    return YieldStats(useful=int(200 * ratio), decided=200, useful_24h=int(concurrent * 48),
                      useful_s_24h=concurrent * USEFUL_HORIZON_S)


class Ledger:
    def __init__(self, stats=None):
        self.table, self.version = dict(stats or {}), 0

    def snapshot(self):
        return self.table


def quota(stats=None, capacity=C, **kwargs):
    kwargs.setdefault("refresh_s", 0.0)
    return YieldQuota(ledger=Ledger(stats), pool_seeds=P, group_size=M, capacity=lambda: capacity,
                      retry_after_s=7, clock=lambda: NOW, **kwargs)


def sha(n):
    return f"{n:064x}"


def frozen(q):
    """Hold the allocation still while a helper loops (production recomputes it at most once a second)."""
    class Freeze:
        def __enter__(self):
            self.kept, q._refresh_s = q._refresh_s, float("inf")

        def __exit__(self, *exc):
            q._refresh_s = self.kept
    return Freeze()


def want(q, hotkey, n, first=0, now=NOW):
    """``hotkey`` asks for ``n`` sessions now (P seeds per precommit): its current demand."""
    with frozen(q):
        for i in range(n):
            q.refusal(hotkey, now, precommit_sha256=sha(first + i // P), seed_index=i % P)
    q._alloc_key = None                                    # recomputed at the next read, with all of it


def hold(q, hotkey, precommit, seed, *, state=LIVE, status=None, issued_at=None):
    record = _record(f"{hotkey}-{precommit[-5:]}-{seed}", sha=precommit, hotkey=hotkey, state=state, seed=seed,
                     issued_at=issued_at or NOW - 30, expires_at=NOW + 3_600)
    record.closed_status = status
    q.opened(record)
    return record


def take(q, hotkey, first=0, *, groups=10_000, now=NOW):
    """Open sessions one at a time, P seeds per precommit, until the quota refuses; how many it holds."""
    held = 0
    with frozen(q):
        for n in range(first, first + groups):
            for seed in range(P):
                if q.admit_locked(hotkey, now, precommit_sha256=sha(n), seed_index=seed) is not None:
                    return held
                hold(q, hotkey, sha(n), seed)
                held += 1
    return held


@pytest.mark.parametrize("capacity", [216, 2_160, 21_600])
@pytest.mark.parametrize("load", ["low", "high"])
def test_one_hotkey_and_ten_hotkeys_with_a_tenth_of_its_volume_and_demand_get_the_same(capacity, load):
    demand = 64 if load == "low" else capacity
    other = {"5Other": main_stats(60.0)}
    one = quota({"5One": main_stats(30.0), **other}, capacity=capacity)
    ten = quota({**{f"5S{n}": main_stats(3.0) for n in range(10)}, **other}, capacity=capacity)
    for q in (one, ten):
        want(q, "5Other", capacity, first=900_000)
    want(one, "5One", 10 * demand)
    for n in range(10):
        want(ten, f"5S{n}", demand, first=10_000 * (n + 1))
    whole = one.allocation().main["5One"]
    parts = [ten.allocation().main[f"5S{n}"] for n in range(10)]
    assert sum(parts) == pytest.approx(whole)
    if capacity > 2_160:
        return                                             # admission is checked on the smaller fleets (time)
    taken = take(one, "5One", first=500_000)
    assert sessions_of(whole) - P <= taken <= sessions_of(whole)            # whole groups, never above
    assert sum(take(ten, f"5S{n}", first=500_000 + 1_000 * n) for n in range(10)) <= taken


@pytest.mark.parametrize("capacity", [216, 2_160, 21_600])
@pytest.mark.parametrize("load", ["low", "high"])
def test_a_hundred_fresh_hotkeys_get_no_more_than_the_newcomer_entitys_rule(capacity, load):
    newcomer_n, _ = entity_sizes(capacity, pool_seeds=P)
    for takes_spare in (True, False):
        q = quota({"5Main": main_stats(100.0)}, capacity=capacity, takes_spare=takes_spare)
        want(q, "5Main", capacity if load == "high" else 10, first=900_000)
        for n in range(100):
            want(q, f"5F{n:03d}", 2 * P, first=10 * n)
        rule = q.allocation().newcomer
        got = sum(take(q, f"5F{n:03d}", first=10 * n) for n in range(100))
        assert got <= sessions_of(rule)
        if not takes_spare or load == "high":
            assert rule <= newcomer_n + (100.0 / 132.0) * capacity * (load == "high" and takes_spare)
        if not takes_spare:
            assert got <= newcomer_n


@pytest.mark.parametrize("capacity", [216, 2_160, 21_600])
def test_ten_wasting_hotkeys_get_no_more_than_the_low_yield_cap(capacity):
    _, low_l = entity_sizes(capacity, pool_seeds=P)
    q = quota({f"5W{n}": WASTEFUL for n in range(10)}, capacity=capacity)
    for n in range(10):
        want(q, f"5W{n}", P, first=10 * n)
    got = sum(take(q, f"5W{n}", first=10 * n) for n in range(10))
    assert q.allocation().low_yield <= low_l and got <= low_l
    assert q.lane("5W0") == LOW_YIELD


def test_a_small_main_key_gets_its_weighted_share_under_contention_and_spare_when_alone():
    q = quota({"5Big": main_stats(500.0), "5Small": main_stats(5.0)})
    want(q, "5Big", C, first=900_000)
    want(q, "5Small", C)
    shares = q.allocation().main
    assert shares["5Small"] == pytest.approx(shares["5Big"] / 100)          # weights 5 : 500
    alone = quota({"5Small": main_stats(5.0)})
    want(alone, "5Small", 100)
    assert alone.allocation().main["5Small"] == 100                         # its demand, not the fleet


def test_a_dormant_main_key_holds_nothing_and_one_with_no_recent_useful_rejoins_the_newcomers():
    q = quota({"5Idle": main_stats(500.0), "5Busy": main_stats(5.0)})
    want(q, "5Busy", 100)
    alloc = q.allocation().main
    assert alloc.get("5Idle", 0.0) == 0.0 and alloc["5Busy"] == 100       # no demand, nothing; no history needed
    stale = quota({"5Back": YieldStats(useful=180, decided=200, useful_24h=0, useful_s_24h=0.0)})
    assert stale.lane("5Back") == NEWCOMER


def test_an_unpresented_token_stops_counting_after_its_present_window():
    listed = set()
    q = quota({"5Main": main_stats(100.0)}, reported=listed.__contains__)
    want(q, "5A", 2 * P, first=0)
    for n in range(NEWCOMER_MAX_GROUPS):
        hold(q, "5A", sha(n), 0, issued_at=NOW - PRESENT_WINDOW_S - 1)     # one token per group, never presented
    assert q.admit_locked("5A", NOW, precommit_sha256=sha(5), seed_index=0) is None   # its groups no longer count
    listed.update(f"5A-{sha(n)[-5:]}-0" for n in range(NEWCOMER_MAX_GROUPS))   # machines listed them: they count
    q._dirty = True
    assert q.admit_locked("5A", NOW, precommit_sha256=sha(6), seed_index=0).reason == "quota_live"


def test_first_opens_of_groups_go_in_fifo_order_by_group_not_by_hotkey():
    q = quota({"5Main": main_stats(100.0), "5W1": WASTEFUL, "5W2": WASTEFUL, "5W3": WASTEFUL}, capacity=216)
    want(q, "5W1", P)
    assert take(q, "5W1", first=0, groups=1) == P                           # L = 32 here: the entity is full
    want(q, "5W2", P, first=10, now=NOW + 1)                                # each asks its whole group at once
    want(q, "5W3", P, first=20, now=NOW + 2)
    assert q.admit_locked("5W2", NOW + 1, precommit_sha256=sha(10), seed_index=0).reason == "quota_pool"
    assert q.admit_locked("5W3", NOW + 2, precommit_sha256=sha(20), seed_index=0).reason == "quota_pool"
    for record in q._index["5W1"].values():
        record.state, record.closed_status = "closed", "expired"            # 5W1's group ended
    q._dirty = True
    assert q.admit_locked("5W3", NOW + 3, precommit_sha256=sha(20), seed_index=0).reason == "quota_pool"
    assert q.admit_locked("5W2", NOW + 4, precommit_sha256=sha(10), seed_index=0) is None


def test_the_allocation_follows_demand_within_its_refresh_and_the_ledger_at_once():
    ledger = Ledger({"5A": main_stats(10.0)})
    clock = [NOW]
    q = YieldQuota(ledger=ledger, pool_seeds=P, group_size=M, capacity=lambda: C, refresh_s=1.0,
                   clock=lambda: clock[0])
    for seed in range(50):
        q.refusal("5A", NOW, precommit_sha256=sha(seed // P), seed_index=seed % P)
    assert q.allocation(NOW + 1).main["5A"] == 50
    for seed in range(100):
        q.refusal("5A", NOW + 1, precommit_sha256=sha(100 + seed // P), seed_index=seed % P)
    assert q.allocation(NOW + 1).main["5A"] == 50                           # within the refresh: unchanged
    assert q.allocation(NOW + 2).main["5A"] == 150
    ledger.table, ledger.version = {"5A": WASTEFUL}, 1
    assert "5A" not in q.allocation(NOW + 2).main                           # a new snapshot: at once


def test_rank_headroom_groups_scale_and_formable():
    q = quota({"5Main": main_stats(100.0), "5W": WASTEFUL})
    want(q, "5Main", 320, first=900_000)
    assert q.rank("5Main") == ("5Main", pytest.approx(100.0), False)
    assert q.rank("5W") == ("5W", 0.0, True) and q.rank("5New") == ("5New", 0.0, True)
    assert q.headroom("5Main", NOW) == 320                                  # admission: its current demand
    assert q.sessions_allowed("5Main") == C                                 # told: its potential
    assert q.groups_allowed("5Main") == C // P + 1 and q.cap_scale("5Main") == C / P
    assert q.groups_allowed("5W") == 1 and q.groups_allowed("5New") == NEWCOMER_MAX_GROUPS and q.cap_scale("5W") == 1.0
    for seed in range(M):
        hold(q, "5New", sha(1), seed, state="closed_graded", status="graded")
    for seed in range(M - 1):
        hold(q, "5New", sha(2), seed, state="closed_graded", status="graded")
    assert q.groups_formable("5New", NOW) == 1


def test_a_restart_rebuilds_what_counts():
    q = quota({"5Main": main_stats(100.0)})
    records = [hold(q, "5Main", sha(1), seed) for seed in range(4)]
    again = quota({"5Main": main_stats(100.0)})
    again.rebuild(records)
    assert len(again._counted("5Main", NOW)) == 4


def test_no_fleet_seen_yet_is_a_retryable_refusal():
    q = quota({"5Main": main_stats(100.0)}, capacity=0)
    refused = q.admit_locked("5Main", NOW, precommit_sha256=sha(1), seed_index=0)
    assert refused.reason == "quota_live" and refused.detail["why"] == "no fleet capacity known yet"


def test_a_main_key_is_told_its_potential_before_it_asks_anything():
    q = quota({"5Main": main_stats(100.0)})
    assert q.sessions_allowed("5Main") == C                                 # an idle fleet: all of it, not 1 group
    busy = quota({"5Main": main_stats(100.0), "5Big": main_stats(300.0)})
    want(busy, "5Big", C, first=900_000)
    assert busy.sessions_allowed("5Main") == pytest.approx(C / 4, abs=1)  # its weighted share under contention


def test_a_graduate_keeps_its_newcomer_allocation_until_main_gives_as_much():
    ledger = Ledger({"5Big": main_stats(500.0), "5Grad": YieldStats(useful=50, decided=60)})
    q = YieldQuota(ledger=ledger, pool_seeds=P, group_size=M, capacity=lambda: C, refresh_s=0.0, clock=lambda: NOW)
    want(q, "5Big", C, first=900_000)
    want(q, "5Grad", 2 * P)
    assert q.lane("5Grad") == NEWCOMER
    ledger.table = {**ledger.table, "5Grad": main_stats(1.79)}           # graduates: tiny weight, 30-min sessions
    ledger.version += 1
    q._alloc_key = None
    assert q.lane("5Grad") == NEWCOMER                                    # MAIN would give it ~6: it stays
    assert take(q, "5Grad", first=50) == 2 * P                            # never below what it held
    ledger.table = {k: v for k, v in ledger.table.items() if k != "5Big"}  # the contention ends
    ledger.version += 1
    q._wants.pop("5Big", None)
    q._alloc_key = None
    assert q.lane("5Grad") == MAIN                                        # now MAIN gives it more: it graduates


def test_a_main_key_below_one_group_never_opens_a_session_that_cannot_form_a_group():
    q = quota({"5Big": main_stats(500.0), "5Small": main_stats(1.0)})
    want(q, "5Big", C, first=900_000)
    want(q, "5Small", P)
    assert 0 < q.allocation().main["5Small"] < M
    assert take(q, "5Small") == 0                                         # no group can form: nothing opened


def test_a_group_in_progress_is_never_cut_by_a_reallocation():
    ledger = Ledger({"5A": main_stats(100.0)})
    q = YieldQuota(ledger=ledger, pool_seeds=P, group_size=M, capacity=lambda: C, refresh_s=0.0, clock=lambda: NOW)
    want(q, "5A", P)
    for seed in range(5):
        assert q.admit_locked("5A", NOW, precommit_sha256=sha(0), seed_index=seed) is None
        hold(q, "5A", sha(0), seed)
    ledger.table = {"5A": main_stats(100.0), "5Big": main_stats(100_000.0)}   # its allocation collapses
    ledger.version += 1
    want(q, "5Big", C, first=900_000)
    assert sessions_of(q.allocation().main["5A"]) < M
    for seed in range(5, M):                                              # the group still forms
        assert q.admit_locked("5A", NOW, precommit_sha256=sha(0), seed_index=seed) is None
        hold(q, "5A", sha(0), seed)
    assert q.admit_locked("5A", NOW, precommit_sha256=sha(0), seed_index=M).reason == "quota_live"


def test_formable_groups_use_each_envs_m():
    q = quota({"5A": main_stats(10.0)}, group_sizes={"reliquary-swe": 4})
    for seed in range(4):
        hold(q, "5A", sha(1), seed, state="closed_graded", status="graded")
    assert q.groups_formable("5A", NOW) == 1


@pytest.mark.parametrize("parts", [10, 100])
def test_a_weak_main_key_split_into_fresh_hotkeys_is_bounded_by_the_newcomer_entity(parts):
    """Under contention a weak MAIN key gets less than a group (~7.7 sessions here), while fresh hotkeys reach the
    newcomer entity. Splitting into fresh hotkeys (and keeping a graduate in its grace) is bounded by that entity's
    rule, whatever the number of parts, and by the per-key group limit inside it."""
    weak = quota({"5Big": main_stats(500.0), "5Weak": main_stats(1.79)})
    want(weak, "5Big", C, first=900_000)
    want(weak, "5Weak", 2 * P)
    assert sessions_of(weak.allocation().main["5Weak"]) < M and take(weak, "5Weak") == 0

    ledger = Ledger({"5Big": main_stats(500.0), "5Grad": YieldStats(useful=50, decided=60)})
    q = YieldQuota(ledger=ledger, pool_seeds=P, group_size=M, capacity=lambda: C, refresh_s=0.0, clock=lambda: NOW)
    want(q, "5Big", C, first=900_000)
    want(q, "5Grad", 2 * P, first=800_000)
    assert q.lane("5Grad") == NEWCOMER
    ledger.table, ledger.version = {**ledger.table, "5Grad": main_stats(1.79)}, ledger.version + 1   # graduates
    for n in range(parts):
        want(q, f"5P{n:03d}", 10 * P, first=1_000 * (n + 1))
    assert q.lane("5Grad") == NEWCOMER                                    # the grace keeps it in the entity
    rule = q.allocation().newcomer
    newcomer_n, _ = entity_sizes(C, pool_seeds=P)
    spare = (C - newcomer_n) * NEWCOMER_WEIGHT / (NEWCOMER_WEIGHT + main_stats(500.0).useful_s_24h / USEFUL_HORIZON_S)
    assert rule <= newcomer_n + spare + 1e-6                              # the entity's rule, not one per part
    grad = take(q, "5Grad", first=800_000)
    got = []
    for n in range(parts):
        q._alloc_key = None                                               # recounted from the index between keys
        got.append(take(q, f"5P{n:03d}", first=1_000 * (n + 1)))
    assert grad <= NEWCOMER_MAX_GROUPS * P and max(got) <= NEWCOMER_MAX_GROUPS * P   # per-key groups
    assert grad + sum(got) <= sessions_of(rule)                           # graduate and parts share the entity
    assert q.allocation().main.get("5Grad", 0.0) == 0.0                   # never served from MAIN as well
