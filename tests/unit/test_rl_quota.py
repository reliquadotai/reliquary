"""The allocation of RL sandbox opens: weighted water-filling over entities.
Production regime: production constants, fleets of 216 / 2 160 / 21 600 slots, low and high MAIN demand."""
import asyncio
import time

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.sandbox.rl_capacity import (  # noqa: E402
    GRACE_SESSIONS,
    LOW_YIELD,
    MAIN,
    NEWCOMER,
    NEWCOMER_MAX_GROUPS,
    NEWCOMER_WEIGHT,
    PRESENT_WINDOW_S,
    USEFUL_HORIZON_S,
    YIELD_MIN_DECIDED,
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


def graduate(concurrent: float, decided: int = YIELD_MIN_DECIDED + 4, ratio: float = 0.9) -> YieldStats:
    """A hotkey that just left the newcomers: ``decided`` decided sessions, ``concurrent`` useful ones over 24 h."""
    return YieldStats(useful=int(decided * ratio), decided=decided, useful_24h=int(concurrent * 48),
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
            spare = (capacity - newcomer_n) * NEWCOMER_WEIGHT / (NEWCOMER_WEIGHT + 100.0)   # its weighted spare
            assert rule <= newcomer_n + spare * (load == "high" and takes_spare) + 1e-6
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
    want(alone, "5Small", 100)                                              # 3 groups of 32 seeds and 4 of a 4th
    assert alone.allocation().main["5Small"] == 4 * P                       # by group (an asked one: its pool)


def test_a_dormant_main_key_holds_nothing_and_one_with_no_recent_useful_rejoins_the_newcomers():
    q = quota({"5Idle": main_stats(500.0), "5Busy": main_stats(5.0)})
    want(q, "5Busy", 96)
    alloc = q.allocation().main
    assert alloc.get("5Idle", 0.0) == 0.0 and alloc["5Busy"] == 96        # no demand, nothing; no history needed
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
    assert q.allocation(NOW + 1).main["5A"] == 2 * P                        # two groups asked: two pools
    for seed in range(96):
        q.refusal("5A", NOW + 1, precommit_sha256=sha(100 + seed // P), seed_index=seed % P)
    assert q.allocation(NOW + 1).main["5A"] == 2 * P                        # within the refresh: unchanged
    assert q.allocation(NOW + 2).main["5A"] == 5 * P
    ledger.table, ledger.version = {"5A": WASTEFUL}, 1
    assert "5A" not in q.allocation(NOW + 2).main                           # a new snapshot: at once


def test_rank_headroom_groups_scale_and_formable():
    q = quota({"5Main": main_stats(100.0), "5W": WASTEFUL})
    want(q, "5Main", 320, first=900_000)
    assert q.rank("5Main") == ("5Main", pytest.approx(100.0), False)
    assert q.rank("5W") == ("5W", 0.0, True) and q.rank("5New") == ("5New", 0.0, True)
    assert q.headroom("5Main", NOW) == 320                                  # admission: its current demand
    assert q.sessions_allowed("5Main") == C                                 # told: its potential
    assert q.groups_allowed("5Main") == C // P + 1 and q.cap_scale("5Main") == C / (NEWCOMER_MAX_GROUPS * P)
    assert q.groups_allowed("5W") == 1 and q.groups_allowed("5New") == NEWCOMER_MAX_GROUPS and q.cap_scale("5W") == 1.0
    assert q.cap_scale("5New") == 1.0                                      # a newcomer: the policy's own budget
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
    ledger.table = {**ledger.table, "5Grad": graduate(1.79)}             # graduates: tiny weight, 30-min sessions
    ledger.version += 1
    q._alloc_key = None
    assert q.lane("5Grad") == NEWCOMER                                    # MAIN would give it ~6 < M: it stays
    assert take(q, "5Grad", first=50) == 2 * P                            # never below what it held
    ledger.table = {k: v for k, v in ledger.table.items() if k != "5Big"}  # the contention ends
    ledger.version += 1
    q._wants.pop("5Big", None)
    q._alloc_key = None
    assert q.lane("5Grad") == MAIN                                        # now MAIN gives it a group: it leaves


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
    ledger.table, ledger.version = {**ledger.table, "5Grad": graduate(1.79)}, ledger.version + 1     # graduates
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


def saturated(extra, *, background=1_880.0, capacity=C):
    """Ten incumbents of total weight ``background`` asking for the whole fleet, plus ``extra``."""
    ledger = Ledger({**{f"5Bg{i}": main_stats(background / 10) for i in range(10)}, **extra})
    q = YieldQuota(ledger=ledger, pool_seeds=P, group_size=M, capacity=lambda: capacity, refresh_s=0.0,
                   clock=lambda: NOW)
    for i in range(10):
        want(q, f"5Bg{i}", capacity, first=900_000 + 10_000 * i)
    return ledger, q


def test_a_graduate_leaves_the_newcomers_as_soon_as_main_gives_it_one_group():
    """Leaving at one group (M), not at the newcomer maximum: on a saturated fleet a graduate's weight, earned inside
    the entity, could never reach the newcomer maximum."""
    _, q = saturated({"5Grad": graduate(30.0)})
    want(q, "5Grad", 2 * P)
    assert M <= q.potential_sessions("5Grad") < NEWCOMER_MAX_GROUPS * P
    assert q.lane("5Grad") == MAIN
    assert take(q, "5Grad") >= M                                          # a group forms in MAIN


def test_the_grace_is_read_from_the_ledger_ends_after_its_sessions_and_survives_a_restart():
    for decided, lane in ((YIELD_MIN_DECIDED, NEWCOMER), (YIELD_MIN_DECIDED + GRACE_SESSIONS - 1, NEWCOMER),
                          (YIELD_MIN_DECIDED + GRACE_SESSIONS, MAIN)):
        ledger, q = saturated({"5Grad": graduate(1.79, decided=decided)})
        want(q, "5Grad", 2 * P)
        assert q.lane("5Grad") == lane
        again = YieldQuota(ledger=ledger, pool_seeds=P, group_size=M, capacity=lambda: C, refresh_s=0.0,
                           clock=lambda: NOW)                              # a restart: nothing in memory
        for i in range(10):
            want(again, f"5Bg{i}", C, first=900_000 + 10_000 * i)          # the miners ask again
        want(again, "5Grad", 2 * P)
        assert again.lane("5Grad") == lane
        assert take(q, "5Grad") == (2 * P if lane == NEWCOMER else 0)   # after the grace: MAIN's share, < M


def test_a_recovered_low_yield_key_leaves_its_pool_once_main_gives_it_a_group():
    _, q = saturated({"5Rec": main_stats(27.0, ratio=0.85), "5Mid": main_stats(3.0, ratio=0.7),
                      **{f"5W{j}": WASTEFUL for j in range(4)}})
    for j in range(4):
        want(q, f"5W{j}", P, first=1_000 * j)                              # the wasters ask first
    want(q, "5Rec", P, first=50_000, now=NOW + 1)
    want(q, "5Mid", P, first=60_000, now=NOW + 1)
    wasters = 0
    for j in range(4):
        q._alloc_key = None
        wasters += take(q, f"5W{j}", first=1_000 * j)
    assert wasters <= q.allocation().low_yield_cap
    assert q.lane("5Rec") == MAIN and take(q, "5Rec", first=50_000, now=NOW + 2) >= M
    assert q.lane("5Mid") == LOW_YIELD                                    # recovering, MAIN below a group: kept


def test_genuine_newcomers_go_before_graduates_in_grace():
    q = quota({"5Main": main_stats(100.0), "5N1": YieldStats(), "5N2": YieldStats(), "5G": graduate(1.0)},
              capacity=216, takes_spare=False)                             # N = 32: one group
    want(q, "5Main", 216, first=900_000)
    want(q, "5N1", P)
    assert take(q, "5N1", first=0, groups=1) == P                         # the entity is full
    want(q, "5G", P, first=10, now=NOW + 1)                               # the graduate asks first
    want(q, "5N2", P, first=20, now=NOW + 2)
    assert q.lane("5G") == NEWCOMER
    assert q.admit_locked("5G", NOW + 1, precommit_sha256=sha(10), seed_index=0).reason == "quota_pool"
    assert q.admit_locked("5N2", NOW + 2, precommit_sha256=sha(20), seed_index=0).reason == "quota_pool"
    for record in q._index["5N1"].values():
        record.state, record.closed_status = "closed", "expired"
    q._dirty = True
    refused = q.admit_locked("5G", NOW + 3, precommit_sha256=sha(10), seed_index=0)
    assert refused.reason == "quota_pool"                                 # the newcomer's group waits: it goes first
    assert q.admit_locked("5N2", NOW + 4, precommit_sha256=sha(20), seed_index=0) is None


def test_a_group_keeps_its_place_in_line_when_its_open_fails_after_the_check():
    q = quota({"5Main": main_stats(100.0), "5W1": WASTEFUL, "5W2": WASTEFUL, "5W3": WASTEFUL}, capacity=216)
    want(q, "5W1", P)
    assert take(q, "5W1", first=0, groups=1) == P
    want(q, "5W2", P, first=10, now=NOW + 1)
    want(q, "5W3", P, first=20, now=NOW + 2)
    q.admit_locked("5W2", NOW + 1, precommit_sha256=sha(10), seed_index=0)
    q.admit_locked("5W3", NOW + 2, precommit_sha256=sha(20), seed_index=0)
    for record in q._index["5W1"].values():
        record.state, record.closed_status = "closed", "expired"
    q._dirty = True
    assert q.admit_locked("5W2", NOW + 3, precommit_sha256=sha(10), seed_index=0) is None   # its open then fails
    assert q.admit_locked("5W3", NOW + 4, precommit_sha256=sha(20), seed_index=0).reason == "quota_pool"
    assert q.admit_locked("5W2", NOW + 5, precommit_sha256=sha(10), seed_index=0) is None   # still first
    hold(q, "5W2", sha(10), 0)
    assert sha(10) not in q._asks[LOW_YIELD]                               # it left the line once it opened


def test_a_new_group_needs_the_m_of_its_env():
    q = quota({"5A": main_stats(10.0)}, capacity=10, group_sizes={"small-env": 4, "big-env": 32})
    want(q, "5A", 10)                                                     # its allocation: the 10-session fleet
    assert q.allocation().main["5A"] == 10
    assert q.admit_locked("5A", NOW, precommit_sha256=sha(1), seed_index=0).reason == "quota_live"   # M = 16
    assert q.admit_locked("5A", NOW, precommit_sha256=sha(1), seed_index=0, environment="small-env") is None
    assert asyncio.run(q.admit_open("5A", environment="small-env", precommit_sha256=sha(2), seed_index=0)) is None
    roomy = quota({"5A": main_stats(10.0)}, capacity=20, group_sizes={"big-env": 32})
    want(roomy, "5A", 20)
    refused = asyncio.run(roomy.admit_open("5A", environment="big-env", precommit_sha256=sha(3), seed_index=0))
    assert refused.reason == "quota_live"                                 # 32 never fit in 20: nothing opened
    assert asyncio.run(roomy.admit_open("5A", environment="other", precommit_sha256=sha(4), seed_index=0)) is None


def test_the_index_keeps_only_held_sessions_and_the_opens_of_the_period():
    q = quota({"5A": main_stats(10.0)})
    records = [hold(q, "5A", sha(1), seed) for seed in range(4)]
    for record in records[:3]:
        record.state, record.closed_status, record.expires_at = "closed", "expired", NOW - 1_000
    q.closed(None, None)
    assert len(q._counted("5A", NOW)) == 1 and len(q._index["5A"]) == 1
    assert len(q._opens["5A"]) == 4                                       # the period's opens still count
    q.allocation(NOW + 3_600 + 60)
    assert "5A" not in q._opens                                           # past the period: forgotten


def test_a_graduate_in_grace_starts_a_group_only_in_the_room_newcomers_leave():
    q = quota({"5Main": main_stats(100.0), "5N": YieldStats(), "5G": graduate(1.0)}, capacity=216,
              takes_spare=False)                                           # N = 32: one group
    want(q, "5Main", 216, first=900_000)
    want(q, "5N", P, first=20)                                            # a newcomer asks, nobody refused yet
    want(q, "5G", P, first=10)
    q._asks[NEWCOMER].clear()                                              # no group waits in line
    assert q.lane("5G") == NEWCOMER
    assert q.admit_locked("5G", NOW, precommit_sha256=sha(10), seed_index=0).reason == "quota_pool"
    assert take(q, "5N", first=20, groups=1) == P                          # the newcomer's group takes the room
    alone = quota({"5Main": main_stats(100.0), "5G": graduate(1.0)}, capacity=216, takes_spare=False)
    want(alone, "5Main", 216, first=900_000)
    want(alone, "5G", P, first=10)
    assert take(alone, "5G", first=10, groups=1) == P                       # no newcomer waiting: the graduate runs


# --- the fleet bound and the demand memory: a turnover is no race --------------------------------------------------

def mutable_quota(stats, capacity, clock):
    return YieldQuota(ledger=Ledger(stats), pool_seeds=P, group_size=M, capacity=lambda: capacity["c"],
                      retry_after_s=7, clock=lambda: clock["now"], refresh_s=0.0)


def test_a_new_group_starts_only_within_the_fleets_room_and_a_group_under_m_is_never_cut():
    clock, capacity = {"now": NOW}, {"c": 128}
    q = mutable_quota({"5A": main_stats(100), "5B": main_stats(100)}, capacity, clock)
    for n in range(3):
        for seed in range(P):
            hold(q, "5A", sha(n), seed)                           # 96 sessions while the fleet had 128
    for seed in range(4):
        hold(q, "5A", sha(3), seed)                               # a group started, under M: it commits M
    capacity["c"] = 64                                            # machines lost: 112 committed on 64
    q._capacity_cache = None
    assert q.refusal("5A", NOW, precommit_sha256=sha(3), seed_index=4) is None       # never cut below M
    refused = q.refusal("5B", NOW, precommit_sha256=sha(50), seed_index=0)          # within its own allocation
    assert refused.reason == "quota_live" and refused.detail["fleet_committed"] == 112
    assert q.headroom("5B", NOW) == 0


def test_the_room_a_pooled_entity_is_allocated_is_kept_for_it():
    q = quota({"5A": main_stats(100), "5B": main_stats(100)})
    for n in range(65):
        for seed in range(P):
            hold(q, "5A", sha(n), seed)                           # 2 080 held while 5A was alone
    want(q, "5N", 2 * P, first=900)                               # a newcomer asks for its two groups
    want(q, "5B", 4 * P, first=500)
    for seed in range(M):
        hold(q, "5B", sha(500), seed)                             # 5B's first group: 2 096 committed
    q._alloc_key = None
    kept = q.refusal("5B", NOW, precommit_sha256=sha(501), seed_index=0)
    assert kept.reason == "quota_live" and kept.detail["kept_for_pools"] == 2 * P    # 2 096 + 16 + 64 > 2 160
    assert q.refusal("5N", NOW, precommit_sha256=sha(900), seed_index=0) is None     # the newcomer's room


def _turnover(keys, clock):
    q = mutable_quota({h: main_stats(w) for h, w in keys.items()}, {"c": C}, clock)
    held = []
    for k, hotkey in enumerate(keys):
        for n in range(int(32 * keys[hotkey] / 200)):
            for seed in range(P):
                held.append(hold(q, hotkey, sha(10_000 * (k + 1) + n), seed))
    want(q, "5B", P, first=90_000)
    before = q.potential_sessions("5B")
    for record in held:                                           # every group ends at once (the window's close)
        record.state, record.closed_status, record.closed_at = "closed", "expired", NOW
        q.closed(record, None)
    want(q, "5B", P, first=90_000)                                # 5B asks for one group first
    return q, before


@pytest.mark.parametrize("keys", [{"5A": 200.0, "5B": 200.0}, {"5A1": 100.0, "5A2": 100.0, "5B": 200.0}])
def test_after_a_turnover_no_key_is_told_the_idle_fleet_and_a_key_that_left_frees_its_share(keys):
    clock = {"now": NOW}
    q, before = _turnover(keys, clock)
    assert before == C - 1_024 < C                                # the others held 1 024 sessions
    assert q.potential_sessions("5B") == before                   # remembered: not the idle fleet
    clock["now"] = NOW + 601                                      # DEMAND_MEMORY_S later, the others never came back
    want(q, "5B", P, first=90_000, now=NOW + 601)
    assert q.potential_sessions("5B") >= C - P                    # their share is free again


def test_opens_between_two_recomputes_count_in_the_fleet_at_once():
    q = quota({"5A": main_stats(100), "5B": main_stats(100)}, capacity=64)
    want(q, "5A", P, first=0)
    want(q, "5B", P, first=100)
    q.allocation()
    with frozen(q):                                               # no recompute while they open (production: 1 s)
        for seed in range(P):
            assert q.refusal("5A", NOW, precommit_sha256=sha(0), seed_index=seed) is None
            hold(q, "5A", sha(0), seed)
        for seed in range(P):
            assert q.refusal("5B", NOW, precommit_sha256=sha(100), seed_index=seed) is None
            hold(q, "5B", sha(100), seed)
        assert q._fleet_committed == 2 * P                       # counted as they opened, before any recompute
        hold(q, "5B", sha(102), 0)                                # a new group commits its M at once
        assert q._fleet_committed == 2 * P + M


def test_at_a_full_fleet_a_started_groups_next_seed_still_joins_the_queue():
    """C = 64 held by two keys, each with a full group and one started under M: the seeds those groups owe are in
    the fleet's count already, so the queue's per-key bound (the quota's headroom) still takes them."""
    from reliquary.sandbox.open_queue import OpenQueue, Ticket

    q = quota({"5A": main_stats(100), "5B": main_stats(100)}, capacity=64)
    want(q, "5A", 64, first=0)
    want(q, "5B", 64, first=100)
    for hotkey, first in (("5A", 0), ("5B", 100)):
        for seed in range(M):
            hold(q, hotkey, sha(first), seed)
        hold(q, hotkey, sha(first + 1), 0)
    q._alloc_key = None
    q.allocation()
    assert q._fleet_committed == 64
    assert q.refusal("5A", NOW, precommit_sha256=sha(1), seed_index=1) is None      # never cut below M
    assert q.headroom("5A", NOW) == M - 1                                           # what its group still owes
    assert q.refusal("5A", NOW, precommit_sha256=sha(2), seed_index=0) is not None  # no new group: the fleet is full
    queue = OpenQueue(rank=q.rank, headroom=lambda hotkey: q.headroom(hotkey, NOW))
    assert isinstance(queue.join("5A", cls=("swe", 1, 1), engagement=f"rl:1:{sha(1)}:1"), Ticket)


def test_over_generating_one_group_never_lets_a_key_hold_more_than_its_allocation():
    """Groups started with one session each reserve their M: filling one to P while the others are short of M
    cannot carry the key past its limit, and the other key's groups within its own allocation stay possible."""
    q = quota({"5A": main_stats(100), "5B": main_stats(100)}, capacity=128)
    want(q, "5A", 128, first=0)
    want(q, "5B", 128, first=100)
    limit = sessions_of(q.allocation().main["5A"])
    assert limit == 64
    for seed in range(M):
        hold(q, "5A", sha(0), seed)
    for g in (1, 2):
        assert q.refusal("5A", NOW, precommit_sha256=sha(g), seed_index=0) is None
        hold(q, "5A", sha(g), 0)                                  # started: each commits M (48 committed)
    q._alloc_key = None
    seed = M
    while q.refusal("5A", NOW, precommit_sha256=sha(0), seed_index=seed) is None:
        hold(q, "5A", sha(0), seed)                               # over-generate G0
        seed += 1
        q._alloc_key = None
    for g in (1, 2):
        for s in range(1, M):
            assert q.refusal("5A", NOW, precommit_sha256=sha(g), seed_index=s) is None
            hold(q, "5A", sha(g), s)                              # never cut below M
    q._alloc_key = None
    assert len(q._counted("5A", NOW)) <= limit                    # was 94 when sessions beyond M counted held only
    for g in range(4):                                            # 5B: its four groups within its allocation of 64
        assert q.refusal("5B", NOW, precommit_sha256=sha(100 + g), seed_index=0) is None
        hold(q, "5B", sha(100 + g), 0)
        q._alloc_key = None


def test_a_burst_of_refused_asks_reserves_nothing_beyond_its_ttl():
    """A key holding nothing asks for 2 000 sessions while another holds the fleet: refused. Its asks are demand
    for DEMAND_TTL_S only, never remembered: 31 s later the other key's allocation is the whole fleet again."""
    clock, capacity = {"now": NOW}, {"c": 256}
    q = mutable_quota({"5A": main_stats(100), "5B": main_stats(100)}, capacity, clock)
    want(q, "5B", 256, first=100)
    for n in range(8):
        for seed in range(P):
            hold(q, "5B", sha(100 + n), seed)
    q._alloc_key = None
    want(q, "5A", 2_000, first=0)
    assert q.refusal("5A", NOW, precommit_sha256=sha(0), seed_index=0) is not None
    assert "5A" not in q._memory                                  # it held nothing: nothing to remember
    clock["now"] = NOW + 31
    want(q, "5B", 256, first=100, now=NOW + 31)
    allocation = q.allocation(NOW + 31)
    assert allocation.main.get("5A", 0.0) == 0.0 and allocation.main["5B"] == pytest.approx(256)


def test_a_pooled_group_under_m_is_kept_once():
    """A newcomer's group of one session commits M in the fleet's count: the pool's room kept from MAIN is its
    allocation less that M, not less the one session (else M - 1 would be kept twice)."""
    q = quota({"5A": main_stats(100)})
    want(q, "5N", 2 * P, first=900)
    hold(q, "5N", sha(900), 0)
    q._alloc_key = None
    alloc = q.allocation()
    assert q._lane_committed[NEWCOMER] == M and q._fleet_committed == M
    assert q._pool_room(alloc) == sessions_of(alloc.newcomer) - M


def test_a_clock_stepping_back_never_raises_what_is_remembered():
    clock, capacity = {"now": NOW}, {"c": C}
    q = mutable_quota({"5A": main_stats(100)}, capacity, clock)
    for seed in range(P):
        hold(q, "5A", sha(0), seed)
    q.allocation()
    assert q._memory["5A"][0] == P
    assert q._remembered("5A", 0, 0.0, NOW - 300) == P            # at most what it held, never more
