# tests/unit/test_service_exploration.py
import hashlib
import sqlite3

import pytest

from reliquary.constants import M_ROLLOUTS
from reliquary.services.exploration import (
    ExplorationLedger, apply_exploration_audit, audit_selected, exploration_cap, exploration_price,
    exploration_within_cap, finalize_exploration, record_exploration, training_group_price,
)
from reliquary.services.run_log import Observation, RunObservationLog

ORDER = "a" * 64
BEACON = "cd" * 32
DAY = 86400


def oid(i: int) -> str:
    return f"{i:064x}"


def test_price_is_15_percent_of_a_nominal_training_group():
    assert training_group_price(0.32, picks_target=7, batch_slots=16) == pytest.approx(0.32 / 112)
    assert exploration_price(0.32, picks_target=7, batch_slots=16, price_bps=1500) == pytest.approx(0.15 * 0.32 / 112)
    assert exploration_cap(0.32, cap_bps=1000) == pytest.approx(0.032)
    price = exploration_price(0.32, picks_target=6, batch_slots=8, price_bps=1500)
    cap = exploration_cap(0.32, cap_bps=1000)
    assert exploration_within_cap(32, price=price, cap=cap)       # 32 x 0.15 / 48 is exactly 10 %
    assert not exploration_within_cap(33, price=price, cap=cap)


def test_audit_draw_follows_the_beacon_and_the_id_and_a_forced_row_ignores_both():
    ids = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(4000)]

    def drawn(beacon):
        return {i for i in ids if audit_selected(beacon_randomness=beacon, observation_id=i, audit_bps=1500, forced=False)}

    first, other = drawn("ab" * 32), drawn("ba" * 32)
    assert first == drawn("ab" * 32)                      # deterministic
    assert 450 < len(first) < 750 and 450 < len(other) < 750
    assert first != other and len(first & other) < 250    # another beacon is another draw (independent: ~90 shared)
    assert 0 < len(first) < len(ids)                      # and it depends on the id
    # the exact rule, recomputed here: sha256(domain || beacon || id), top 64 bits against the rate
    for i in ids[:200]:
        digest = hashlib.sha256(b"reliquary-exploration-audit/v1" + bytes.fromhex("ab" * 32) + bytes.fromhex(i)).digest()
        assert (i in first) == (int.from_bytes(digest[:8], "big") / 2**64 < 0.15)
    # forced: selected whatever the beacon, the id or the rate (even a beacon that is not one)
    assert all(audit_selected(beacon_randomness=b, observation_id=i, audit_bps=0, forced=True)
               for i in ids[:50] for b in ("ab" * 32, "ba" * 32, ""))
    assert not any(audit_selected(beacon_randomness="ab" * 32, observation_id=i, audit_bps=0, forced=False) for i in ids[:50])
    assert all(audit_selected(beacon_randomness="ab" * 32, observation_id=i, audit_bps=10000, forced=False) for i in ids[:50])


@pytest.mark.parametrize("beacon", ["", "ab" * 31, "ab" * 33, "zz" * 32, "ab" * 31 + "a", None, b"\x00" * 32])
def test_audit_beacon_must_be_exactly_32_bytes(beacon):
    with pytest.raises(ValueError):
        audit_selected(beacon_randomness=beacon, observation_id=oid(1), audit_bps=1500, forced=False)


def ledger(tmp_path):
    return ExplorationLedger(sqlite3.connect(tmp_path / "l.sqlite3"), order_sha256=ORDER)


def reserve(book, i, *, hotkey="hk", env="math", cap=1.0, amount=0.1, window=1, groups=100, draw_round=None, now=None):
    with book.db:
        return book.reserve(window=window, environment=env, observation_id=oid(i), hotkey=hotkey,
                            prompt_idx=i, amount=amount, cap=cap, draw_round=50 + i if draw_round is None else draw_round,
                            new_hotkey_audit_groups=groups, now=now)


def draw(book, *, env="math", window=1, audit_bps=0, beacon=BEACON):
    with book.db:
        return book.resolve_draws(window, environment=env, beacon_for_round=lambda r: beacon, audit_bps=audit_bps)


def audit(book, i, passed, *, now=1000.0, ban=DAY):
    with book.db:
        return book.record_audit(oid(i), passed=passed, now=now, ban_seconds=ban)


def finalize(book, *, env="math", window=1):
    with book.db:
        return book.finalize_window(window, environment=env)


def states(book, *, env="math", window=1):
    return {int(r["observation_id"], 16): (r["audit"], r["status"]) for r in book.rows(window, environment=env)}


def test_cap_is_per_env_and_window(tmp_path):
    book = ledger(tmp_path)
    assert reserve(book, 1, cap=0.25) is not None
    assert reserve(book, 2, cap=0.25) is not None
    assert reserve(book, 3, cap=0.25) is None
    assert reserve(book, 4, cap=0.25, env="code") is not None
    assert reserve(book, 5, cap=0.25, window=2) is not None


def test_cap_counts_neither_forfeited_nor_unaudited_rows_nor_a_replay_twice(tmp_path):
    book = ledger(tmp_path)
    first = reserve(book, 1, cap=0.2)
    assert reserve(book, 1, cap=0.2) == first            # replay: same answer, not a second unit
    assert reserve(book, 2, cap=0.2) is not None         # still room for a second: the replay took none
    assert reserve(book, 3, hotkey="other", cap=0.2) is None
    assert len(book.rows(1, environment="math")) == 2
    draw(book)
    assert set(audit(book, 1, False)) == {oid(1), oid(2)}   # both forfeited: their units are free again
    assert reserve(book, 3, hotkey="other", cap=0.2) is not None
    assert reserve(book, 4, hotkey="other", cap=0.2) is not None
    assert reserve(book, 5, hotkey="third", cap=0.2) is None
    with book.db:
        book.mark_unaudited(oid(3))                      # an unaudited unit is free as well
    assert reserve(book, 5, hotkey="third", cap=0.2) is not None
    assert reserve(book, 6, hotkey="third", cap=0.2) is None


def test_zero_price_is_never_reserved_and_bad_numbers_raise(tmp_path):
    book = ledger(tmp_path)
    assert reserve(book, 1, amount=0.0, cap=0.0) is None
    assert reserve(book, 1, amount=0.0, cap=1.0) is None
    assert book.rows(1, environment="math") == []
    for amount, cap in ((float("nan"), 1.0), (-0.1, 1.0), (0.1, float("inf")), ("0.1", 1.0), (True, 1.0)):
        with pytest.raises(ValueError):
            reserve(book, 2, amount=amount, cap=cap)


def test_one_window_env_has_one_price(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, amount=0.1)
    with pytest.raises(ValueError, match="price"):
        reserve(book, 2, amount=0.2)
    assert reserve(book, 3, amount=0.2, window=2) is not None   # another window may have another pool


def test_probation_counts_passed_audits_only(tmp_path):
    book = ledger(tmp_path)
    # three reservations do not end the probation of 2: nothing has passed yet
    assert [reserve(book, i, groups=2)["forced"] for i in (1, 2, 3)] == [True, True, True]
    draw(book)
    audit(book, 1, True)
    assert reserve(book, 4, groups=2)["forced"] is True          # one pass < 2
    audit(book, 2, True)
    assert book.passed_audits("hk") == 2
    assert reserve(book, 5, groups=2)["forced"] is False         # two passes: probation over
    assert reserve(book, 6, hotkey="newcomer", groups=2)["forced"] is True


def test_probation_spans_windows_and_envs_and_survives_reopen(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, env="math", window=1, groups=2)
    reserve(book, 2, env="code", window=2, groups=2)
    draw(book, env="math", window=1)
    draw(book, env="code", window=2)
    audit(book, 1, True)
    finalize(book, env="math", window=1)
    audit(book, 2, True)
    book.db.close()
    reopened = ledger(tmp_path)
    assert reopened.passed_audits("hk") == 2
    assert reserve(reopened, 3, env="science", window=3, groups=2)["forced"] is False
    assert reserve(reopened, 4, env="science", window=3, hotkey="other", groups=2)["forced"] is True
    other_order = ExplorationLedger(reopened.db, order_sha256="b" * 64)
    assert other_order.passed_audits("hk") == 0                  # another order, another probation


def test_a_hotkey_whose_forced_groups_are_never_audited_stays_forced(tmp_path):
    book = ledger(tmp_path)
    for i in range(1, 6):
        assert reserve(book, i, groups=3)["forced"] is True
    assert len(draw(book)) == 5
    assert set(finalize(book)) == {oid(i) for i in range(1, 6)}  # drawn, never audited: unaudited
    assert book.passed_audits("hk") == 0
    for i in range(6, 12):                                       # window after window, still at 100 %
        assert reserve(book, i, window=2, groups=3)["forced"] is True


def test_forfeited_and_not_drawn_groups_do_not_count_for_probation(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=1)
    reserve(book, 2, groups=1)
    draw(book)
    audit(book, 1, True)
    assert book.passed_audits("hk") == 1
    audit(book, 2, False)                                        # forfeits the window, the passed one included
    assert book.passed_audits("hk") == 0
    assert reserve(book, 3, window=2, groups=1)["forced"] is True
    # a not-drawn group (possible only past probation) is paid but is not a passed audit
    reserve(book, 10, hotkey="old", window=5, groups=0)
    draw(book, window=5)
    finalize(book, window=5)
    assert book.payable(5, environment="math") == {"old": 1} and book.passed_audits("old") == 0


def test_failed_audit_forfeits_the_window_and_bans(tmp_path):
    book = ledger(tmp_path)
    for i in (1, 2, 3):
        reserve(book, i, groups=100)
    reserve(book, 4, hotkey="other", groups=100)
    selected = draw(book, audit_bps=1500)
    assert set(selected) == {oid(i) for i in (1, 2, 3, 4)}
    forfeited = audit(book, 2, False, now=1000.0)
    audit(book, 4, True, now=1000.0)
    assert set(forfeited) == {oid(i) for i in (1, 2, 3)}
    assert book.banned("hk", 1000.0 + 86399) and not book.banned("hk", 1000.0 + 86400)
    assert not book.banned("other", 1000.0)
    finalize(book)
    assert book.payable(1, environment="math") == {"other": 1}


def test_replaying_a_failed_verdict_returns_the_forfeited_ids_again_and_does_not_ban_again(tmp_path):
    book = ledger(tmp_path)
    for i in (1, 2, 3):
        reserve(book, i)
    reserve(book, 4, env="code")
    draw(book)
    first = audit(book, 2, False, now=1000.0, ban=100)
    assert set(first) == {oid(1), oid(2), oid(3), oid(4)}
    assert audit(book, 2, False, now=5000.0, ban=100) == first   # retry-safe release list
    assert not book.banned("hk", 1100.0)                         # the replay at t=5000 did not extend the ban
    finalize(book)
    assert audit(book, 2, False, now=9000.0, ban=100) == first   # still after the env is finalized
    assert not book.banned("hk", 9001.0)
    with pytest.raises(ValueError):
        audit(book, 99, False)


def test_unresolved_draws_are_unpaid_at_window_close(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)
    assert finalize(book) == [oid(1)]
    assert book.payable(1, environment="math") == {}


def test_draw_waits_for_its_beacon(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)
    assert book.pending_draw_rounds(1, environment="math") == [51]
    with book.db:
        assert book.resolve_draws(1, environment="math", beacon_for_round=lambda r: None, audit_bps=1500) == []
    assert book.rows(1, environment="math")[0]["audit"] == "pending_draw"


def test_ledger_survives_reopen_and_draws_are_identical_after_it(tmp_path):
    book = ledger(tmp_path)
    ids = list(range(1, 201))
    for i in ids:
        reserve(book, i, groups=0, cap=100.0)
    book.db.close()
    reopened = ledger(tmp_path)
    assert reopened.pending_draw_rounds(1, environment="math") == [50 + i for i in ids]
    assert reopened.rows(1, environment="math")[0]["status"] == "reserved"
    beacons = {50 + i: hashlib.sha256(str(i % 7).encode()).hexdigest() for i in ids}
    with reopened.db:
        selected = reopened.resolve_draws(1, environment="math", beacon_for_round=beacons.get, audit_bps=1500)
    # what a ledger that never closed, or any replayer, computes from the public inputs
    expected = [oid(i) for i in ids if audit_selected(beacon_randomness=beacons[50 + i], observation_id=oid(i),
                                                      audit_bps=1500, forced=False)]
    assert selected == expected and 10 < len(selected) < 60
    reopened.db.close()
    again = ledger(tmp_path)
    assert [r["observation_id"] for r in again.rows(1, environment="math") if r["audit"] == "queued"] == expected
    assert [r["observation_id"] for r in again.rows(1, environment="math") if r["drawn"]] == expected
    with again.db:
        assert again.resolve_draws(1, environment="math", beacon_for_round=beacons.get, audit_bps=10000) == []


def test_reserve_is_idempotent_and_late_audits_cannot_flip(tmp_path):
    book = ledger(tmp_path)
    first = reserve(book, 1, groups=100)
    assert reserve(book, 1, groups=100) == first
    assert len(book.rows(1, environment="math")) == 1
    with book.db:
        book.finalize_window(1, environment="math")
        assert book.record_audit(oid(1), passed=True, now=1.0, ban_seconds=10) == []
    assert not book.banned("hk", 2.0) and book.payable(1, environment="math") == {}
    assert book.passed_audits("hk") == 0                         # a late pass is not a passed audit


def test_a_finalized_window_env_accepts_no_money_and_no_draw(tmp_path):
    book = ledger(tmp_path)
    kept = reserve(book, 1, groups=0)
    draw(book)
    finalize(book)
    assert book.payable(1, environment="math") == {"hk": 1}
    assert reserve(book, 2, groups=0) is None                    # no new entitlement
    assert reserve(book, 1, groups=0) == kept                    # a replay only reads the frozen row
    assert reserve(book, 3, groups=0, env="code") is not None    # another env of the window is still open
    assert len(book.rows(1, environment="math")) == 1 and book.payable(1, environment="math") == {"hk": 1}
    # even a row that would be waiting for its draw (written behind the ledger's back) is not drawn
    with book.db:
        book.db.execute("INSERT INTO exploration_entitlements(observation_id, order_id, window, environment, hotkey, "
                        "prompt_idx, amount, draw_round, forced, audit, status) VALUES(?,?,1,'math','hk',9,0.1,60,1,"
                        "'pending_draw','reserved')", (oid(9), ORDER))
    assert draw(book, audit_bps=10000) == []
    assert states(book)[9] == ("pending_draw", "reserved")


def test_payable_is_refused_before_finalize(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)
    draw(book)
    with pytest.raises(ValueError, match="finalized"):
        book.payable(1, environment="math")


def test_unaudited_group_is_not_sanctioned(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)
    reserve(book, 2, groups=100)
    assert draw(book) == [oid(2)]
    assert finalize(book) == [oid(2)]
    assert states(book) == {1: ("not_drawn", "reserved"), 2: ("unaudited", "reserved")}
    # a failed audit at t would ban on [t, t + 86400): there is no such ban at any time around it
    assert not any(book.banned("hk", t) for t in (0.0, 1.0, 1000.0, 1000.0 + DAY / 2, 1000.0 + DAY - 1, 1e12))
    assert book.db.execute("SELECT COUNT(*) FROM exploration_bans").fetchone()[0] == 0
    assert book.payable(1, environment="math") == {"hk": 1}      # drawn at round 52: row 1 (round 51) is before the horizon


def test_window_methods_are_scoped_to_one_environment(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, env="math", groups=0)
    reserve(book, 2, env="code", groups=0)
    assert book.pending_draw_rounds(1, environment="code") == [52]
    assert finalize(book, env="math") == [oid(1)]
    assert [r["audit"] for r in book.rows(1, environment="math")] == ["unaudited"]
    assert [r["audit"] for r in book.rows(1, environment="code")] == ["pending_draw"]
    assert draw(book, env="code") == []
    assert finalize(book, env="code") == []
    assert book.payable(1, environment="code") == {"hk": 1}
    assert book.payable(1, environment="math") == {}


def test_finalized_set_survives_reopen(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)
    finalize(book)
    book.db.close()
    reopened = ledger(tmp_path)
    assert reopened.is_finalized(1, environment="math") and not reopened.is_finalized(1, environment="code")
    assert reserve(reopened, 2, groups=0) is None


def test_failed_audit_forfeits_every_unfinalized_env_but_never_a_finalized_one(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, env="math", groups=100)
    reserve(book, 2, env="code", groups=100)
    reserve(book, 3, env="science", groups=0)                    # past probation there: not drawn, PAID
    draw(book, env="math")
    draw(book, env="code")
    draw(book, env="science")
    finalize(book, env="science")
    science_before = book.rows(1, environment="science")
    assert book.payable(1, environment="science") == {"hk": 1}
    forfeited = audit(book, 1, False, now=10.0, ban=100)
    assert set(forfeited) == {oid(1), oid(2)}
    assert [r["status"] for r in book.rows(1, environment="code")] == ["forfeited"]
    assert book.rows(1, environment="science") == science_before
    assert book.payable(1, environment="science") == {"hk": 1}   # money already frozen is spared
    assert book.banned("hk", 50.0)
    finalize(book, env="code")
    assert book.payable(1, environment="code") == {}


def test_late_failed_audit_after_finalize_bans_once_and_changes_no_row(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=100)
    draw(book)
    finalize(book)
    before = book.rows(1, environment="math")
    assert audit(book, 1, False, now=10.0, ban=100) == []
    assert book.rows(1, environment="math") == before
    assert book.banned("hk", 50.0) and not book.banned("hk", 110.0)
    assert audit(book, 1, False, now=500.0, ban=100) == []       # the same late verdict again is not a second ban
    assert not book.banned("hk", 510.0)
    assert audit(book, 1, True, now=10.0, ban=100) == []
    assert book.rows(1, environment="math") == before


def test_only_a_drawn_group_can_lead_to_a_late_ban(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, hotkey="paid", groups=0, draw_round=10)     # not drawn, before the horizon: paid
    reserve(book, 2, hotkey="late", groups=0, draw_round=30)     # not drawn, after the horizon: unaudited
    reserve(book, 3, hotkey="drawn", groups=100, draw_round=20)  # drawn, never audited
    with book.db:
        book.resolve_draws(1, environment="math", beacon_for_round=lambda r: BEACON if r < 99 else None, audit_bps=0)
    reserve(book, 4, hotkey="waiting", groups=100, draw_round=99)  # still waiting for its draw at close
    with book.db:
        book.resolve_draws(1, environment="math", beacon_for_round=lambda r: BEACON if r < 99 else None, audit_bps=0)
    finalize(book)
    assert states(book) == {1: ("not_drawn", "reserved"), 2: ("unaudited", "reserved"),
                            3: ("unaudited", "reserved"), 4: ("unaudited", "reserved")}
    before = book.rows(1, environment="math")
    for i in (1, 2, 3, 4):
        assert audit(book, i, False, now=10.0, ban=100) == []
    assert [h for h in ("paid", "late", "drawn", "waiting") if book.banned(h, 50.0)] == ["drawn"]
    assert book.rows(1, environment="math") == before


def horizon_window(book):
    """old = past probation (drawn at 0 bps: never); new = in probation (always drawn)."""
    reserve(book, 1, hotkey="old", groups=0, draw_round=10)
    reserve(book, 2, hotkey="new", groups=100, draw_round=20)
    reserve(book, 3, hotkey="old", groups=0, draw_round=25)
    reserve(book, 4, hotkey="new", groups=100, draw_round=30)
    reserve(book, 5, hotkey="old", groups=0, draw_round=30)
    reserve(book, 6, hotkey="old2", groups=0, draw_round=40)
    assert draw(book) == [oid(2), oid(4)]


def test_audit_horizon_honest_case_every_not_drawn_row_is_paid(tmp_path):
    book = ledger(tmp_path)
    horizon_window(book)
    audit(book, 2, True)
    audit(book, 4, True)                                         # every drawn group was audited
    assert finalize(book) == []
    assert book.payable(1, environment="math") == {"new": 2, "old": 3, "old2": 1}
    assert not any(book.banned(h, 1000.0) for h in ("old", "old2", "new"))


def test_audit_horizon_a_drawn_group_left_unaudited_unpays_the_not_drawn_rows_from_its_round(tmp_path):
    book = ledger(tmp_path)
    horizon_window(book)
    audit(book, 2, True, now=1000.0)                             # round 20 audited; round 30 (row 4) left unaudited
    moved = finalize(book)
    assert moved == [oid(4), oid(5), oid(6)]                     # the drawn one, and not-drawn rows of round >= 30
    assert states(book) == {1: ("not_drawn", "reserved"), 2: ("passed", "reserved"), 3: ("not_drawn", "reserved"),
                            4: ("unaudited", "reserved"), 5: ("unaudited", "reserved"), 6: ("unaudited", "reserved")}
    assert book.payable(1, environment="math") == {"new": 1, "old": 2}   # rounds 10 and 25 < 30 stay paid
    assert not any(book.banned(h, t) for h in ("old", "old2", "new") for t in (1000.0, 1000.0 + DAY / 2))
    assert book.db.execute("SELECT COUNT(*) FROM exploration_bans").fetchone()[0] == 0
    assert finalize(book) == moved                               # again: same ids, nothing moves
    assert book.payable(1, environment="math") == {"new": 1, "old": 2}


def test_audit_horizon_is_the_smallest_unaudited_drawn_round_and_is_per_env(tmp_path):
    book = ledger(tmp_path)
    horizon_window(book)                                         # nothing audited: horizon = round 20
    reserve(book, 7, hotkey="old", env="code", groups=0, draw_round=90)
    draw(book, env="code")
    assert finalize(book) == [oid(2), oid(3), oid(4), oid(5), oid(6)]
    assert book.payable(1, environment="math") == {"old": 1}     # only round 10 < 20
    assert finalize(book, env="code") == []
    assert book.payable(1, environment="code") == {"old": 1}     # math's missing audits do not touch code


def test_pending_draws_at_close_do_not_set_the_horizon(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0, draw_round=10)
    reserve(book, 2, groups=0, draw_round=20)
    with book.db:
        book.resolve_draws(1, environment="math", beacon_for_round=lambda r: BEACON if r == 20 else None, audit_bps=0)
    assert finalize(book) == [oid(1)]                            # never drawn: unaudited as before
    assert book.payable(1, environment="math") == {"hk": 1}      # the not-drawn row of round 20 is paid


def test_constructor_refuses_an_open_transaction(tmp_path):
    db = sqlite3.connect(tmp_path / "t.sqlite3")
    db.execute("CREATE TABLE t(x)")
    db.execute("INSERT INTO t VALUES(1)")
    assert db.in_transaction
    with pytest.raises(ValueError, match="transaction"):
        ExplorationLedger(db, order_sha256=ORDER)
    assert db.in_transaction                                     # the caller's transaction is still open
    db.rollback()
    assert db.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
    ExplorationLedger(db, order_sha256=ORDER)


# ---- the single entry point: log and ledger on one connection ----

ENV = "reliquary_dapo_math_v1"
SEEDS = list(range(M_ROLLOUTS))
PRICE, CAP = 0.1, 0.25                                           # two entitlements fit


def obs(prompt, *, hotkey="hk", window=1, group=None, lane="exploration", env=ENV):
    return Observation(environment=env, dataset_id=f"{env}-train", prompt_idx=prompt, group_id=group or f"g{prompt}",
                       window=window, checkpoint_n=1, checkpoint_revision="c" * 40, observed_at=100.0,
                       rewards_bps=(0,) * M_ROLLOUTS, lane=lane, candidate={"pool_sha256": "p" * 64, "seeds": SEEDS},
                       hotkey=hotkey, token_count=10)


@pytest.fixture
def pair(tmp_path):
    db = sqlite3.connect(tmp_path / "run.sqlite3")
    return RunObservationLog(db, order_sha256=ORDER, sigma_min_bps=2400), ExplorationLedger(db, order_sha256=ORDER)


def admit(pair, o, *, amount=PRICE, cap=CAP, now=1000.0, groups=100, refuse=None, draw_round=60):
    log, book = pair
    return record_exploration(log, book, o, amount=amount, cap=cap, draw_round=draw_round,
                              new_hotkey_audit_groups=groups, now=now, refuse=refuse)


def last_event(log):
    return log.events()[-1][1]


def test_entry_point_records_and_reserves_together(pair, tmp_path):
    log, book = pair
    out = admit(pair, obs(1))
    assert out.inserted and out.first_scan and out.status == "exploration_pending" and out.reason is None
    assert out.entitlement == {"amount": PRICE, "forced": True, "draw_round": 60}
    assert not log.db.in_transaction                             # atomic on its own: committed
    other = sqlite3.connect(tmp_path / "run.sqlite3")            # ...and visible to another connection
    assert other.execute("SELECT COUNT(*) FROM exploration_entitlements").fetchone()[0] == 1
    assert other.execute("SELECT COUNT(*) FROM run_scans").fetchone()[0] == 1
    event = last_event(log)
    assert event["status"] == "exploration_pending" and event["proof"] == "pending" and "reason" not in event
    assert log.is_scanned(ENV, 1)
    assert [r["observation_id"] for r in book.rows(1, environment=ENV)] == [out.observation_id]


def refused_by_cap(pair):
    admit(pair, obs(1))
    admit(pair, obs(2))
    return {}


def refused_by_finalize(pair):
    with pair[1].db:
        pair[1].finalize_window(1, environment=ENV)
    return {}


def refused_by_ban(pair):
    first = admit(pair, obs(50), now=100.0)
    with pair[1].db:
        pair[1].resolve_draws(1, environment=ENV, beacon_for_round=lambda r: BEACON, audit_bps=0)
    apply_exploration_audit(*pair, first.observation_id, passed=False, now=100.0, ban_seconds=DAY)
    return {"now": 100.0 + DAY - 1}


@pytest.mark.parametrize("setup, kwargs, reason", [
    (refused_by_cap, {}, "cap"),
    (refused_by_finalize, {}, "finalized"),
    (refused_by_ban, {}, "banned"),
    (lambda pair: {}, {"amount": 0.0, "cap": 0.0}, "zero_price"),
    (lambda pair: {}, {"refuse": "env_not_explorable"}, "env_not_explorable"),
])
def test_entry_point_refusal_is_published_unpaid_and_never_burns_the_prompt(pair, setup, kwargs, reason):
    log, book = pair
    kwargs = {**kwargs, **setup(pair)}
    rows_before = len(book.rows(1, environment=ENV))
    out = admit(pair, obs(7), **kwargs)
    assert (out.status, out.reason, out.entitlement) == ("exploration_unpaid", reason, None)
    assert out.inserted and not out.first_scan
    event = last_event(log)
    assert event["id"] == out.observation_id and event["type"] == "observation"
    assert (event["status"], event["proof"], event["reason"]) == ("exploration_unpaid", "unproven", reason)
    assert "hotkey" not in event
    assert len(book.rows(1, environment=ENV)) == rows_before     # no entitlement
    assert not log.is_scanned(ENV, 7)                            # the first scan was given back
    assert log.window_observations(1)[-1]["first_scan"] is False
    # so the prompt is still payable: to another miner, in a window with room
    later = admit(pair, obs(7, hotkey="hk-2", window=2), now=1e9)
    assert later.status == "exploration_pending" and later.first_scan and log.is_scanned(ENV, 7)


def test_entry_point_ban_ends_with_the_ban(pair):
    kwargs = refused_by_ban(pair)
    assert admit(pair, obs(7), now=kwargs["now"]).reason == "banned"
    assert admit(pair, obs(8), now=kwargs["now"] + 1, cap=1.0).reason is None


def test_entry_point_second_scan_is_unpaid_and_keeps_the_first_scan_of_the_first(pair):
    log, book = pair
    a = admit(pair, obs(1, hotkey="hk-a"))
    b = admit(pair, obs(1, hotkey="hk-b"))
    assert a.reason is None and (b.status, b.reason, b.first_scan) == ("exploration_unpaid", "already_scanned", False)
    assert last_event(log)["reason"] == "already_scanned"
    assert log.is_scanned(ENV, 1) and len(book.rows(1, environment=ENV)) == 1
    assert log.db.execute("SELECT first_id FROM run_scans").fetchone()[0] == a.observation_id


def test_entry_point_replay_writes_nothing_and_reports_the_entitlement(pair):
    log, book = pair
    a = admit(pair, obs(1))
    again = admit(pair, obs(1), cap=0.0, now=5e9)                # whatever the state now: it is the same submission
    assert not again.inserted and again.observation_id == a.observation_id
    assert again.entitlement == a.entitlement and again.status == "exploration_pending" and again.first_scan
    assert len(log.events()) == 1 and len(book.rows(1, environment=ENV)) == 1
    refused = admit(pair, obs(2), refuse="too_long")
    replay = admit(pair, obs(2))
    assert not replay.inserted and (replay.status, replay.reason, replay.entitlement) == ("exploration_unpaid", "replay", None)
    assert replay.observation_id == refused.observation_id and len(log.events()) == 2
    assert not log.is_scanned(ENV, 2) and len(book.rows(1, environment=ENV)) == 1


def test_entry_point_is_all_or_nothing(pair, monkeypatch):
    log, book = pair

    def boom(**kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(book, "reserve", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        admit(pair, obs(1))
    assert log.events() == [] and not log.is_scanned(ENV, 1) and log.window_observations(1) == []
    assert not log.db.in_transaction
    monkeypatch.setattr(book, "reserve", lambda **kwargs: None)  # a reservation refused after the checks said yes
    with pytest.raises(RuntimeError, match="disagree"):
        admit(pair, obs(1))
    assert log.events() == [] and not log.is_scanned(ENV, 1)


def test_entry_point_joins_the_callers_transaction(pair):
    log, book = pair
    with pytest.raises(KeyError):
        with log.db:
            log.db.execute("CREATE TABLE IF NOT EXISTS side(x)")
            log.db.execute("INSERT INTO side VALUES(1)")
            admit(pair, obs(1))
            assert log.db.in_transaction                         # not committed behind the caller's back
            raise KeyError("caller fails after the entry point")
    assert log.events() == [] and not log.is_scanned(ENV, 1) and book.rows(1, environment=ENV) == []


def test_entry_point_refuses_misuse(pair, tmp_path):
    log, book = pair
    with pytest.raises(ValueError, match="exploration"):
        admit(pair, obs(1, lane="training"))
    elsewhere = ExplorationLedger(sqlite3.connect(tmp_path / "other.sqlite3"), order_sha256=ORDER)
    with pytest.raises(ValueError, match="share"):
        admit((log, elsewhere), obs(1))
    with pytest.raises(ValueError, match="share"):
        admit((log, ExplorationLedger(log.db, order_sha256="b" * 64)), obs(1))
    with pytest.raises(ValueError):
        admit(pair, obs(1), refuse="Not An Identifier")
    with pytest.raises(ValueError):
        admit(pair, obs(1), amount=float("nan"))
    assert log.events() == [] and not log.is_scanned(ENV, 1)


def test_failed_audit_releases_the_first_scans_it_forfeits(pair):
    log, book = pair
    a = admit(pair, obs(1))
    b = admit(pair, obs(2))
    other = admit(pair, obs(3, hotkey="honest"), cap=1.0)
    with book.db:
        book.resolve_draws(1, environment=ENV, beacon_for_round=lambda r: BEACON, audit_bps=0)
    forfeited = apply_exploration_audit(log, book, a.observation_id, passed=False, now=10.0, ban_seconds=DAY)
    assert set(forfeited) == {a.observation_id, b.observation_id}
    assert not log.is_scanned(ENV, 1) and not log.is_scanned(ENV, 2) and log.is_scanned(ENV, 3)
    assert apply_exploration_audit(log, book, other.observation_id, passed=True, now=10.0, ban_seconds=DAY) == []
    assert log.is_scanned(ENV, 3)
    # prompt 1 was re-scanned by someone else since: replaying the verdict must not release THAT scan
    again = admit(pair, obs(1, hotkey="honest", group="other-subset"), cap=1.0)
    assert again.first_scan
    assert set(apply_exploration_audit(log, book, a.observation_id, passed=False, now=10.0, ban_seconds=DAY)) == set(forfeited)
    assert log.is_scanned(ENV, 1)


def test_finalize_releases_the_first_scan_of_every_unpaid_row_and_only_those(pair):
    log, book = pair
    paid = admit(pair, obs(1, hotkey="old"), groups=0, draw_round=10, cap=1.0)
    drawn = admit(pair, obs(2, hotkey="new"), groups=100, draw_round=20, cap=1.0)
    behind = admit(pair, obs(3, hotkey="old"), groups=0, draw_round=30, cap=1.0)
    with book.db:
        book.resolve_draws(1, environment=ENV, beacon_for_round=lambda r: BEACON, audit_bps=0)
    waiting = admit(pair, obs(4, hotkey="old"), groups=0, draw_round=99, cap=1.0)
    unaudited = finalize_exploration(log, book, 1, environment=ENV)
    assert unaudited == [drawn.observation_id, behind.observation_id, waiting.observation_id]
    assert log.is_scanned(ENV, 1)                                # paid: the prompt stays scanned
    assert not any(log.is_scanned(ENV, p) for p in (2, 3, 4))    # unpaid (drawn-unaudited, horizon, no draw): reopened
    assert book.payable(1, environment=ENV) == {"old": 1}
    assert finalize_exploration(log, book, 1, environment=ENV) == unaudited
    assert log.is_scanned(ENV, 1) and admit(pair, obs(9), cap=1.0).reason == "finalized"
    assert paid.first_scan
