# tests/unit/test_service_exploration.py
import hashlib
import sqlite3

import pytest

from reliquary.constants import M_ROLLOUTS, PROBATION_PENDING_LIMIT
from reliquary.services.exploration import (
    ExplorationLedger, apply_exploration_verdict, audit_selected, exploration_cap, exploration_price,
    exploration_within_cap, finalize_exploration, record_exploration, training_group_price,
)
from reliquary.services.run_log import Observation, RunObservationLog

ORDER = "a" * 64
BEACON = "cd" * 32
DAY = 86400
SALT = b"\x07" * 32


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
        return {i for i in ids if audit_selected(beacon_randomness=beacon, observation_id=i, audit_bps=1500, forced=False, run_salt=SALT)}

    first, other = drawn("ab" * 32), drawn("ba" * 32)
    assert first == drawn("ab" * 32)                      # deterministic
    assert 450 < len(first) < 750 and 450 < len(other) < 750
    assert first != other and len(first & other) < 250    # another beacon is another draw (independent: ~90 shared)
    assert 0 < len(first) < len(ids)                      # and it depends on the id
    # the exact rule, recomputed here: sha256(domain || beacon || id || run salt), top 64 bits against the rate
    for i in ids[:200]:
        digest = hashlib.sha256(b"reliquary-exploration-audit/v2" + bytes.fromhex("ab" * 32) + bytes.fromhex(i) + SALT).digest()
        assert (i in first) == (int.from_bytes(digest[:8], "big") / 2**64 < 0.15)
    # forced: selected whatever the beacon, the id or the rate (even a beacon that is not one)
    assert all(audit_selected(beacon_randomness=b, observation_id=i, audit_bps=0, forced=True, run_salt=SALT)
               for i in ids[:50] for b in ("ab" * 32, "ba" * 32, ""))
    assert not any(audit_selected(beacon_randomness="ab" * 32, observation_id=i, audit_bps=0, forced=False, run_salt=SALT) for i in ids[:50])
    assert all(audit_selected(beacon_randomness="ab" * 32, observation_id=i, audit_bps=10000, forced=False, run_salt=SALT) for i in ids[:50])


@pytest.mark.parametrize("beacon", ["", "ab" * 31, "ab" * 33, "zz" * 32, "ab" * 31 + "a", None, b"\x00" * 32])
def test_audit_beacon_must_be_exactly_32_bytes(beacon):
    with pytest.raises(ValueError):
        audit_selected(beacon_randomness=beacon, observation_id=oid(1), audit_bps=1500, forced=False, run_salt=SALT)


def ledger(tmp_path):
    return ExplorationLedger(sqlite3.connect(tmp_path / "l.sqlite3"), order_sha256=ORDER)


def reserve(book, i, *, hotkey="hk", env="math", cap=1.0, amount=0.1, window=1, groups=100, draw_round=None, now=None):
    with book.db:
        return book.reserve(window=window, environment=env, observation_id=oid(i), hotkey=hotkey,
                            prompt_idx=i, amount=amount, cap=cap, draw_round=50 + i if draw_round is None else draw_round,
                            new_hotkey_audit_groups=groups, now=now)


def draw(book, *, env="math", window=1, audit_bps=0, beacon=BEACON):
    with book.db:
        return book.resolve_draws(window, environment=env, beacon_for_round=lambda r: beacon, audit_bps=audit_bps, run_salt=SALT)


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
    n = PROBATION_PENDING_LIMIT                                  # a probationer holds at most this many
    for i in range(1, n + 1):
        assert reserve(book, i, groups=n + 1)["forced"] is True
    assert len(draw(book)) == n
    assert set(finalize(book)) == {oid(i) for i in range(1, n + 1)}  # drawn, never audited: unaudited
    assert book.passed_audits("hk") == 0
    for i in range(n + 1, 2 * n + 1):                            # window after window, still at 100 %
        assert reserve(book, i, window=2, groups=n + 1)["forced"] is True


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
        assert book.resolve_draws(1, environment="math", beacon_for_round=lambda r: None, audit_bps=1500, run_salt=SALT) == []
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
        selected = reopened.resolve_draws(1, environment="math", beacon_for_round=beacons.get, audit_bps=1500, run_salt=SALT)
    # what a ledger that never closed, or any replayer, computes from the public inputs
    expected = [oid(i) for i in ids if audit_selected(beacon_randomness=beacons[50 + i], observation_id=oid(i),
                                                      audit_bps=1500, forced=False, run_salt=SALT)]
    assert selected == expected and 10 < len(selected) < 60
    reopened.db.close()
    again = ledger(tmp_path)
    assert [r["observation_id"] for r in again.rows(1, environment="math") if r["audit"] == "queued"] == expected
    assert [r["observation_id"] for r in again.rows(1, environment="math") if r["drawn"]] == expected
    with again.db:
        assert again.resolve_draws(1, environment="math", beacon_for_round=beacons.get, audit_bps=10000, run_salt=SALT) == []


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
    reserve(book, 1, hotkey="paid", groups=0, draw_round=10)     # not drawn, other hotkey: paid
    reserve(book, 2, hotkey="drawn", groups=0, draw_round=30)    # same hotkey as 3, after its horizon: unaudited
    reserve(book, 3, hotkey="drawn", groups=100, draw_round=20)  # drawn, never audited
    with book.db:
        book.resolve_draws(1, environment="math", beacon_for_round=lambda r: BEACON if r < 99 else None, audit_bps=0, run_salt=SALT)
    reserve(book, 4, hotkey="waiting", groups=100, draw_round=99)  # still waiting for its draw at close
    with book.db:
        book.resolve_draws(1, environment="math", beacon_for_round=lambda r: BEACON if r < 99 else None, audit_bps=0, run_salt=SALT)
    finalize(book)
    assert states(book) == {1: ("not_drawn", "reserved"), 2: ("unaudited", "reserved"),
                            3: ("unaudited", "reserved"), 4: ("unaudited", "reserved")}
    before = book.rows(1, environment="math")
    for i in (1, 2, 3, 4):
        assert audit(book, i, False, now=10.0, ban=100) == []
    assert [h for h in ("paid", "drawn", "waiting") if book.banned(h, 50.0)] == ["drawn"]
    assert book.rows(1, environment="math") == before


def mixed_hotkey(book, hotkey="A", *, first=1):
    """A hotkey past probation (one audit passed) with a drawn row left unaudited at round 20, a
    not-drawn row before it (15) and two after it (25, 30)."""
    reserve(book, first, hotkey=hotkey, groups=1, draw_round=5)
    assert draw(book) == [oid(first)]
    audit(book, first, True)
    reserve(book, first + 1, hotkey=hotkey, groups=1, draw_round=20)
    assert draw(book, audit_bps=10000) == [oid(first + 1)]       # drawn by the rate, never audited
    reserve(book, first + 2, hotkey=hotkey, groups=1, draw_round=15)
    reserve(book, first + 3, hotkey=hotkey, groups=1, draw_round=25)
    reserve(book, first + 4, hotkey=hotkey, groups=1, draw_round=30)
    assert draw(book) == []                                      # 0 bps: not drawn


def test_audit_horizon_honest_case_every_not_drawn_row_is_paid(tmp_path):
    book = ledger(tmp_path)
    mixed_hotkey(book)
    audit(book, 2, True)                                         # every drawn group was audited
    assert finalize(book) == []
    assert book.payable(1, environment="math") == {"A": 5}
    assert not book.banned("A", 1000.0)


def test_horizon_is_per_hotkey_a_drawn_group_left_unaudited_unpays_only_its_own_later_rows(tmp_path):
    book = ledger(tmp_path)
    mixed_hotkey(book, "A")                                      # rows 1..5, A's round 20 is left unaudited
    reserve(book, 6, hotkey="B", groups=0, draw_round=10)
    reserve(book, 7, hotkey="B", groups=0, draw_round=25)        # same (window, env), after A's horizon
    reserve(book, 8, hotkey="B", groups=0, draw_round=40)
    reserve(book, 9, hotkey="A", groups=1, draw_round=20)        # exactly the horizon round: unpaid too
    assert draw(book) == []
    moved = finalize(book)
    assert moved == [oid(2), oid(4), oid(5), oid(9)]             # A's drawn row and A's not-drawn rows >= 20
    assert states(book) == {1: ("passed", "reserved"), 2: ("unaudited", "reserved"), 3: ("not_drawn", "reserved"),
                            4: ("unaudited", "reserved"), 5: ("unaudited", "reserved"),
                            6: ("not_drawn", "reserved"), 7: ("not_drawn", "reserved"), 8: ("not_drawn", "reserved"),
                            9: ("unaudited", "reserved")}
    assert book.payable(1, environment="math") == {"A": 2, "B": 3}   # A: rows 1 (passed) and 3 (round 15 < 20)
    assert not any(book.banned(h, t) for h in ("A", "B") for t in (1000.0, 1000.0 + DAY / 2))
    assert book.db.execute("SELECT COUNT(*) FROM exploration_bans").fetchone()[0] == 0
    assert finalize(book) == moved                               # again: same ids, nothing moves
    assert book.payable(1, environment="math") == {"A": 2, "B": 3}


def test_a_forfeited_queued_row_sets_no_horizon_for_anyone(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, hotkey="A", groups=100, draw_round=10)
    reserve(book, 2, hotkey="A", groups=100, draw_round=40)
    reserve(book, 3, hotkey="A", groups=0, draw_round=50)        # not drawn
    reserve(book, 4, hotkey="B", groups=0, draw_round=60)
    assert draw(book) == [oid(1), oid(2)]
    assert audit(book, 1, False, now=10.0) == [oid(1), oid(2), oid(3)]   # row 2 is forfeited while still queued
    assert states(book)[2] == ("queued", "forfeited")
    assert finalize(book) == [oid(2)]                            # row 3 (round 50 >= 40) is NOT moved by a forfeited horizon
    assert states(book)[3] == ("not_drawn", "forfeited")
    assert book.payable(1, environment="math") == {"B": 1}


def test_audit_horizon_is_per_env(tmp_path):
    book = ledger(tmp_path)
    mixed_hotkey(book, "A")
    reserve(book, 7, hotkey="A", env="code", groups=0, draw_round=90)
    draw(book, env="code")
    assert finalize(book) == [oid(2), oid(4), oid(5)]
    assert book.payable(1, environment="math") == {"A": 2}
    assert finalize(book, env="code") == []
    assert book.payable(1, environment="code") == {"A": 1}       # math's missing audit does not touch code


def test_pending_draws_at_close_do_not_set_the_horizon(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0, draw_round=10)
    reserve(book, 2, groups=0, draw_round=20)
    with book.db:
        book.resolve_draws(1, environment="math", beacon_for_round=lambda r: BEACON if r == 20 else None, audit_bps=0, run_salt=SALT)
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
                       rewards_bps=(0,) * M_ROLLOUTS, lane=lane, candidate={"pool_sha256": "ab" * 32, "seeds": SEEDS},
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
        pair[1].resolve_draws(1, environment=ENV, beacon_for_round=lambda r: BEACON, audit_bps=0, run_salt=SALT)
    apply_exploration_verdict(*pair, first.observation_id, passed=False, now=100.0, ban_seconds=DAY)[1]
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
        book.resolve_draws(1, environment=ENV, beacon_for_round=lambda r: BEACON, audit_bps=0, run_salt=SALT)
    forfeited = apply_exploration_verdict(log, book, a.observation_id, passed=False, now=10.0, ban_seconds=DAY)[1]
    assert set(forfeited) == {a.observation_id, b.observation_id}
    assert not log.is_scanned(ENV, 1) and not log.is_scanned(ENV, 2) and log.is_scanned(ENV, 3)
    assert apply_exploration_verdict(log, book, other.observation_id, passed=True, now=10.0, ban_seconds=DAY)[1] == []
    assert log.is_scanned(ENV, 3)
    # prompt 1 was re-scanned by someone else since: replaying the verdict must not release THAT scan
    again = admit(pair, obs(1, hotkey="honest", group="other-subset"), cap=1.0)
    assert again.first_scan
    assert set(apply_exploration_verdict(log, book, a.observation_id, passed=False, now=10.0, ban_seconds=DAY)[1]) == set(forfeited)
    assert log.is_scanned(ENV, 1)


def test_finalize_releases_the_first_scan_of_every_unpaid_row_and_only_those(pair):
    log, book = pair
    paid = admit(pair, obs(1, hotkey="old"), groups=0, draw_round=10, cap=1.0)
    drawn = admit(pair, obs(2, hotkey="old"), groups=100, draw_round=20, cap=1.0)   # same hotkey: its horizon
    behind = admit(pair, obs(3, hotkey="old"), groups=0, draw_round=30, cap=1.0)
    with book.db:
        book.resolve_draws(1, environment=ENV, beacon_for_round=lambda r: BEACON, audit_bps=0, run_salt=SALT)
    waiting = admit(pair, obs(4, hotkey="old"), groups=0, draw_round=99, cap=1.0)
    unaudited = finalize_exploration(log, book, 1, environment=ENV)
    assert unaudited == [drawn.observation_id, behind.observation_id, waiting.observation_id]
    assert log.is_scanned(ENV, 1)                                # paid: the prompt stays scanned
    assert not any(log.is_scanned(ENV, p) for p in (2, 3, 4))    # unpaid (drawn-unaudited, horizon, no draw): reopened
    assert book.payable(1, environment=ENV) == {"old": 1}
    assert finalize_exploration(log, book, 1, environment=ENV) == unaudited
    assert log.is_scanned(ENV, 1) and admit(pair, obs(9), cap=1.0).reason == "finalized"
    assert paid.first_scan


# ---- round 4 ----

def mark(book, i):
    with book.db:
        book.mark_unaudited(oid(i))


def verdict(book, i, passed, *, now=1000.0, ban=DAY):
    with book.db:
        return book.apply_verdict(oid(i), passed=passed, now=now, ban_seconds=ban)


def test_n1_a_drawn_row_marked_unaudited_before_finalize_sets_the_horizon_like_a_queued_one(tmp_path):
    book = ledger(tmp_path)
    mixed_hotkey(book, "A", first=1)                             # drawn row 2 (round 20); not drawn 3 (15), 4 (25), 5 (30)
    mixed_hotkey(book, "B", first=11)
    audit(book, 12, True)                                        # B's drawn row passed: B has no horizon
    mark(book, 2)                                                # the runtime lost A's audit slot before the seal
    assert states(book)[2] == ("unaudited", "reserved")
    finalize(book)
    got = states(book)
    assert got[3] == ("not_drawn", "reserved")                   # before the horizon: paid
    assert got[4] == ("unaudited", "reserved") and got[5] == ("unaudited", "reserved")
    assert all(got[i] == ("not_drawn", "reserved") for i in (13, 14, 15))   # another hotkey: untouched
    assert book.payable(1, environment="math") == {"A": 2, "B": 5}          # A: passed + row 3


def test_n1_a_marked_row_that_was_forfeited_sets_no_horizon(tmp_path):
    book = ledger(tmp_path)
    mixed_hotkey(book, "A", first=1)
    mark(book, 2)
    with book.db:                                                # another env's failure forfeits A's rows of the window
        book.db.execute("UPDATE exploration_entitlements SET status='forfeited' WHERE observation_id=?", (oid(2),))
    finalize(book)
    assert states(book)[4] == ("not_drawn", "reserved") and states(book)[5] == ("not_drawn", "reserved")


def test_n1_a_verdict_on_a_row_marked_unaudited_never_raises(tmp_path):
    book = ledger(tmp_path)
    mixed_hotkey(book, "A", first=1)
    mark(book, 2)
    assert verdict(book, 2, True) == ("not_applied", [])         # a pass is ignored
    assert states(book)[2] == ("unaudited", "reserved") and not book.banned("A", 1.0)
    kind, forfeited = verdict(book, 2, False, now=10.0, ban=100)  # a failure still bans and forfeits
    assert kind == "failed" and set(forfeited) == {oid(i) for i in (2, 3, 4, 5)} | {oid(1)}
    assert book.banned("A", 50.0)
    assert verdict(book, 2, False, now=500.0, ban=100)[0] == "failed"      # replay: same, no second ban
    assert not book.banned("A", 510.0)


def test_n1_a_verdict_on_a_row_marked_unaudited_after_finalize_never_raises(tmp_path):
    book = ledger(tmp_path)
    mixed_hotkey(book, "A", first=1)
    mark(book, 2)
    finalize(book)
    before = book.rows(1, environment="math")
    assert verdict(book, 2, True) == ("not_applied", [])
    assert verdict(book, 2, False, now=10.0, ban=100)[0] == "failed"
    assert book.banned("A", 50.0) and book.rows(1, environment="math") == before


def test_n2_a_late_failure_forfeits_the_hotkeys_entitlements_in_the_envs_not_yet_finalized(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, env="math", groups=100)
    reserve(book, 2, env="code", groups=100)
    reserve(book, 3, env="code", hotkey="other", groups=0)       # another hotkey: spared
    draw(book, env="math")
    draw(book, env="code")
    finalize(book, env="math")                                   # row 1: drawn, never audited -> unaudited
    math_rows = book.rows(1, environment="math")
    kind, forfeited = verdict(book, 1, False, now=10.0, ban=100)
    assert (kind, forfeited) == ("failed", [oid(2)])             # the ids whose first scan the caller releases
    assert book.rows(1, environment="math") == math_rows         # finalized env: never changes
    assert states(book, env="code") == {2: ("queued", "forfeited"), 3: ("not_drawn", "reserved")}
    assert book.banned("hk", 50.0) and not book.banned("other", 50.0)
    assert verdict(book, 1, False, now=500.0, ban=100) == ("failed", [oid(2)])   # retry-safe
    finalize(book, env="code")
    assert book.payable(1, environment="code") == {"other": 1}


def test_n2_the_entry_point_releases_the_first_scans_a_late_failure_forfeits(pair):
    log, book = pair
    a = admit(pair, obs(1, env="math"), cap=1.0)
    b = admit(pair, obs(2, env="code"), cap=1.0)
    for env in ("math", "code"):
        with book.db:
            book.resolve_draws(1, environment=env, beacon_for_round=lambda r: BEACON, audit_bps=0, run_salt=SALT)
    finalize_exploration(log, book, 1, environment="math")
    assert not log.is_scanned("math", 1) and log.is_scanned("code", 2)
    forfeited = apply_exploration_verdict(log, book, a.observation_id, passed=False, now=10.0, ban_seconds=DAY)[1]
    assert forfeited == [b.observation_id] and not log.is_scanned("code", 2)


def test_n3_aborted_finalize_unpays_every_reserved_row_even_after_a_plain_finalize(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)                                   # not drawn: payable
    reserve(book, 2, groups=100)
    reserve(book, 3, hotkey="bad", groups=100)
    reserve(book, 4, hotkey="bad", groups=0)
    draw(book)
    audit(book, 2, True)                                         # passed: payable
    audit(book, 3, False, now=5.0)                               # forfeited (row 4 too)
    finalize(book)
    assert book.payable(1, environment="math") == {"hk": 2}
    with book.db:
        ids = book.finalize_window(1, environment="math", aborted=True)    # AFTER the plain finalize
    assert set(ids) == {oid(1), oid(2)}                          # forfeited rows were released when forfeited
    assert states(book) == {1: ("not_drawn", "unpaid"), 2: ("passed", "unpaid"), 3: ("failed", "forfeited"),
                            4: ("not_drawn", "forfeited")}        # a forfeited row keeps its label
    assert book.payable(1, environment="math") == {}
    with book.db:
        assert book.finalize_window(1, environment="math", aborted=True) == ids          # idempotent
    assert finalize(book) == ids and book.payable(1, environment="math") == {}
    assert book.passed_audits("hk") == 1                         # the audit that passed is still a fact


def test_n3_aborted_finalize_on_an_open_env_is_a_finalize_that_pays_nothing(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)
    draw(book)
    with book.db:
        assert book.finalize_window(1, environment="math", aborted=True) == [oid(1)]
    assert book.is_finalized(1, environment="math") and book.payable(1, environment="math") == {}
    assert reserve(book, 2, groups=0) is None                    # finalized: no new reservation


def test_n3_aborted_finalize_gives_every_first_scan_back_and_only_aborted_does(pair):
    log, book = pair
    paid = admit(pair, obs(1), groups=0, cap=1.0)
    other = admit(pair, obs(2, env="code"), groups=0, cap=1.0)
    for env in (ENV, "code"):
        with book.db:
            book.resolve_draws(1, environment=env, beacon_for_round=lambda r: BEACON, audit_bps=0, run_salt=SALT)
    assert finalize_exploration(log, book, 1, environment=ENV) == []
    assert log.is_scanned(ENV, 1) and book.payable(1, environment=ENV) == {"hk": 1}
    assert finalize_exploration(log, book, 1, environment=ENV, aborted=True) == [paid.observation_id]
    assert not log.is_scanned(ENV, 1) and log.is_scanned("code", 2)          # per (window, env)
    assert finalize_exploration(log, book, 1, environment=ENV, aborted=True) == [paid.observation_id]
    assert admit(pair, obs(1, window=2, hotkey="later"), cap=1.0).first_scan  # scannable again, later
    assert other.first_scan


def test_n5_racing_connections_admit_exactly_one_first_scan_and_nobody_raises(tmp_path):
    import threading
    path = tmp_path / "race.sqlite3"
    seed = sqlite3.connect(path)
    RunObservationLog(seed, order_sha256=ORDER, sigma_min_bps=2400)
    ExplorationLedger(seed, order_sha256=ORDER)
    seed.close()
    threads, start = 8, threading.Barrier(8)
    outcomes, errors = [], []

    def worker(index):
        db = sqlite3.connect(path, timeout=60)
        try:
            log = RunObservationLog(db, order_sha256=ORDER, sigma_min_bps=2400)
            book = ExplorationLedger(db, order_sha256=ORDER)
            start.wait()
            for prompt in range(3):                              # three never-scanned prompts, all contended
                outcomes.append((prompt, record_exploration(
                    log, book, obs(prompt, hotkey=f"hk{index}", group=f"g{index}-{prompt}"), amount=PRICE, cap=10.0,
                    draw_round=60, new_hotkey_audit_groups=100, now=1000.0)))
        except BaseException as exc:  # noqa: BLE001 - the point of the test
            errors.append(repr(exc))
        finally:
            db.close()

    pool = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join(120)
    assert errors == []
    assert len(outcomes) == threads * 3
    for prompt in range(3):
        mine = [o for p, o in outcomes if p == prompt]
        assert sum(o.first_scan for o in mine) == 1 and sum(o.status == "exploration_pending" for o in mine) == 1
        assert {o.reason for o in mine if not o.first_scan} == {"already_scanned"}


def test_n6_a_replay_reports_the_state_of_the_row_now(pair):
    log, book = pair
    a = admit(pair, obs(1), cap=1.0)
    b = admit(pair, obs(2), cap=1.0)
    c = admit(pair, obs(3, hotkey="honest"), cap=1.0)
    with book.db:
        book.resolve_draws(1, environment=ENV, beacon_for_round=lambda r: BEACON, audit_bps=0, run_salt=SALT)
    again = admit(pair, obs(1), cap=1.0)
    assert again.status == "exploration_pending" and again.entitlement == a.entitlement
    apply_exploration_verdict(log, book, a.observation_id, passed=False, now=10.0, ban_seconds=DAY)[1]
    for o, i in ((a, 1), (b, 2)):                                # forfeited by the failure
        r = admit(pair, obs(i), cap=1.0)
        assert (r.status, r.entitlement, r.inserted) == ("exploration_forfeited", None, False)
    finalize_exploration(log, book, 1, environment=ENV)          # c: drawn, never audited -> unpaid
    r = admit(pair, obs(3, hotkey="honest"), cap=1.0)
    assert (r.status, r.entitlement) == ("exploration_unpaid", None) and not r.first_scan


def test_n6_a_group_refused_for_cap_and_resubmitted_with_room_is_a_new_attempt(pair):
    log, book = pair
    first = admit(pair, obs(1, hotkey="a"))                      # CAP fits two
    second = admit(pair, obs(2, hotkey="b"))
    third = admit(pair, obs(3, hotkey="c"))
    assert (third.reason, third.first_scan) == ("cap", False) and not log.is_scanned(ENV, 3)
    again = admit(pair, obs(3, hotkey="c"))
    assert (again.reason, again.status) == ("cap", "exploration_unpaid")   # still full: the refusal, not "replay"
    with book.db:
        book.resolve_draws(1, environment=ENV, beacon_for_round=lambda r: BEACON, audit_bps=0, run_salt=SALT)
    apply_exploration_verdict(log, book, first.observation_id, passed=False, now=10.0, ban_seconds=DAY)[1]
    retry = admit(pair, obs(3, hotkey="c"), now=20.0)            # room again (a's row was forfeited)
    assert retry.observation_id == third.observation_id and not retry.inserted
    assert (retry.status, retry.reason, retry.first_scan) == ("exploration_pending", None, True)
    assert retry.entitlement is not None and log.is_scanned(ENV, 3)
    assert last_event(log)["status"] == "exploration_pending"
    assert admit(pair, obs(3, hotkey="c"), now=21.0).entitlement == retry.entitlement   # now a plain replay
    assert second.first_scan


def test_n6_a_cap_refused_retry_loses_to_someone_who_took_the_prompt_meanwhile(pair):
    log, book = pair
    admit(pair, obs(1, hotkey="a"))
    admit(pair, obs(2, hotkey="b"))
    late = admit(pair, obs(3, hotkey="c"))
    assert late.reason == "cap"
    admit(pair, obs(3, hotkey="d", group="elsewhere"), cap=1.0)  # another hotkey takes the prompt, cap raised
    retry = admit(pair, obs(3, hotkey="c"), cap=1.0)
    assert (retry.reason, retry.status, retry.first_scan) == ("already_scanned", "exploration_unpaid", False)


def test_n10_a_probationer_holds_at_most_the_limit_of_unpassed_rows_per_window_all_envs_together(tmp_path):
    book = ledger(tmp_path)
    half = PROBATION_PENDING_LIMIT // 2
    for i in range(1, half + 1):
        assert reserve(book, i, groups=100) is not None
    for i in range(half + 1, PROBATION_PENDING_LIMIT + 1):
        assert reserve(book, i, groups=100, env="code") is not None         # one probationer, 2 envs, 1 window
    for env in ("math", "code", "science"):                                 # 4 in the window, whatever the env
        assert reserve(book, 50, groups=100, env=env) is None
        assert book.refusal(window=1, environment=env, hotkey="hk", amount=0.1, cap=1.0, now=0.0,
                            new_hotkey_audit_groups=100) == "probation_limit"
    assert len(book.rows(1, environment="math")) + len(book.rows(1, environment="code")) == PROBATION_PENDING_LIMIT
    assert reserve(book, 52, groups=100, window=2) is not None              # per window
    assert reserve(book, 53, groups=100, hotkey="other") is not None        # and per hotkey
    draw(book)
    draw(book, env="code")
    audit(book, 1, True)                                         # a passed audit frees a slot, in any env
    assert reserve(book, 54, groups=100, env="code") is not None
    assert reserve(book, 55, groups=100) is None


def test_n10_a_hotkey_past_probation_is_not_limited(tmp_path):
    book = ledger(tmp_path)
    for i in range(1, 3 * PROBATION_PENDING_LIMIT - 2):          # 10 fit under the cap
        assert reserve(book, i, groups=0) is not None


def test_n10_an_unaudited_or_forfeited_row_does_not_hold_a_slot(tmp_path):
    book = ledger(tmp_path)
    for i in range(1, PROBATION_PENDING_LIMIT + 1):
        reserve(book, i, groups=100)
    mark(book, 1)
    assert reserve(book, 50, groups=100) is not None             # an unaudited row is unpaid: it frees its slot


def test_n10_the_entry_point_refuses_with_probation_limit_publishes_unpaid_and_releases_the_scan(pair):
    log, book = pair
    for p in range(1, PROBATION_PENDING_LIMIT + 1):
        assert admit(pair, obs(p), cap=10.0).first_scan
    over = admit(pair, obs(50), cap=10.0)
    assert (over.status, over.reason, over.first_scan, over.entitlement) == \
        ("exploration_unpaid", "probation_limit", False, None)
    assert not log.is_scanned(ENV, 50)
    assert last_event(log)["reason"] == "probation_limit" and last_event(log)["status"] == "exploration_unpaid"
    assert admit(pair, obs(60, hotkey="vet"), cap=10.0, groups=0).first_scan   # past probation: not limited
    for p in range(70, 80):
        assert admit(pair, obs(p, hotkey="vet"), cap=10.0, groups=0).first_scan


def test_m_a_a_probation_limit_refusal_is_retryable_like_a_cap_refusal(pair):
    """The same submission coming back after a slot freed is a NEW attempt, not a replay of the stale refusal."""
    log, book = pair
    held = [admit(pair, obs(p), cap=10.0) for p in range(1, PROBATION_PENDING_LIMIT + 1)]
    assert all(h.first_scan for h in held)
    over = obs(50)
    refused = admit(pair, over, cap=10.0)
    assert refused.reason == "probation_limit" and not log.is_scanned(ENV, 50)
    again = admit(pair, over, cap=10.0)                     # still full: the refusal stands (not "replay")
    assert (again.reason, again.status, again.entitlement) == ("probation_limit", "exploration_unpaid", None)
    with book.db:
        book.resolve_draws(1, environment=ENV, beacon_for_round=lambda r: BEACON, audit_bps=0, run_salt=SALT)
    apply_exploration_verdict(log, book, held[0].observation_id, passed=True, now=10.0, ban_seconds=DAY)[1]
    retry = admit(pair, over, cap=10.0, now=20.0)           # a passed audit freed a probation slot
    assert retry.observation_id == refused.observation_id and not retry.inserted
    assert (retry.status, retry.reason, retry.first_scan) == ("exploration_pending", None, True)
    assert retry.entitlement is not None and log.is_scanned(ENV, 50)
    assert admit(pair, over, cap=10.0, now=21.0).entitlement == retry.entitlement      # now a plain replay


# ---- Task 7 review: I1 + R17 ----

def trained(log, prompt, *, window=1, hotkey="trainer", env=ENV):
    half = (10000,) * (M_ROLLOUTS // 2) + (0,) * (M_ROLLOUTS - M_ROLLOUTS // 2)
    with log.db:
        return log.record(Observation(
            environment=env, dataset_id=f"{env}-train", prompt_idx=prompt, group_id=f"t{prompt}-{window}-{hotkey}",
            window=window, checkpoint_n=1, checkpoint_revision="c" * 40, observed_at=100.0, rewards_bps=half,
            lane="training", candidate={"pool_sha256": "ab" * 32, "seeds": SEEDS}, hotkey=hotkey, token_count=10),
            status="proven", proof="proven")


def scan_of(log, prompt, env=ENV):
    row = log.db.execute("SELECT first_id FROM run_scans WHERE environment=? AND prompt_idx=?", (env, prompt)).fetchone()
    return None if row is None else row[0]


def test_r17_ledger_voids_the_pay_of_reserved_rows_on_trained_prompts_and_never_their_audit(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, hotkey="old", groups=0, draw_round=10)
    reserve(book, 2, hotkey="old", groups=100, draw_round=20)       # forced: drawn, never audited
    reserve(book, 3, hotkey="old", groups=0, draw_round=30)         # behind it
    reserve(book, 4, hotkey="cheat", groups=100, draw_round=40)
    reserve(book, 5, hotkey="cheat", groups=100, draw_round=41)
    reserve(book, 6, hotkey="clean", groups=0, draw_round=5)
    reserve(book, 99, hotkey="old", env="code", groups=0)
    draw(book)
    audit(book, 4, False)                                           # 4 and 5 forfeited
    with book.db:
        unpaid = book.finalize_window(1, environment="math", trained_prompts={2, 5, 6, 99})
    # C2: training prompt 2 does not hide its lost audit: row 3 is behind the horizon row 2 sets
    assert unpaid == [oid(2), oid(3), oid(5), oid(6)]
    rows = {r["prompt_idx"]: (r["audit"], r["status"]) for r in book.rows(1, environment="math")}
    assert rows[2] == ("unaudited", "trained") and rows[3] == ("unaudited", "reserved")
    assert rows[6] == ("not_drawn", "trained")                      # a not-drawn row: pay voided, nothing else
    assert rows[5][1] == "forfeited"                                # a forfeited row keeps its label
    assert book.payable(1, environment="math") == {"old": 1}
    assert not book.banned("old", 10**9) and not book.banned("clean", 10**9)
    with book.db:                                                   # idempotent
        assert book.finalize_window(1, environment="math", trained_prompts={2, 5, 6, 99}) == unpaid
    # a training group recorded after the env was finalized: the second allowed transition
    with book.db:
        assert book.finalize_window(1, environment="math", trained_prompts={1, 2}) == [oid(i) for i in (1, 2, 3, 5, 6)]
    assert book.payable(1, environment="math") == {}
    assert [r["status"] for r in book.rows(1, environment="code")] == ["reserved"]   # the other env is untouched
    # the audit of the voided drawn row still bites, even late: ban, and the open env is forfeited
    assert audit(book, 2, False) == [oid(99)]
    assert book.banned("old", 1000.0 + DAY - 1)


def test_r18_the_audit_draw_is_keyed_with_the_run_salt(tmp_path):
    ids = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(2000)]

    def drawn(salt):
        return {i for i in ids if audit_selected(beacon_randomness=BEACON, observation_id=i, audit_bps=1500,
                                                 forced=False, run_salt=salt)}
    mine, other = drawn(SALT), drawn(b"\x08" * 32)
    assert mine == drawn(SALT) and 200 < len(mine) < 400 and 200 < len(other) < 400
    assert mine != other and len(mine & other) < 120             # independent draws (~45 shared), not the same one
    for bad in (None, b"", b"x" * 31, "s" * 32):
        with pytest.raises(ValueError, match="run salt"):
            audit_selected(beacon_randomness=BEACON, observation_id=ids[0], audit_bps=1500, forced=False, run_salt=bad)
    # the ledger draws with the salt it is given, and with nothing else
    book = ledger(tmp_path)
    for i in range(1, 41):
        reserve(book, i, hotkey=f"h{i}", groups=0, draw_round=50, cap=100.0)
    with book.db:
        selected = book.resolve_draws(1, environment="math", beacon_for_round=lambda r: BEACON, audit_bps=5000,
                                      run_salt=b"\x08" * 32)
    assert selected == [oid(i) for i in range(1, 41) if audit_selected(
        beacon_randomness=BEACON, observation_id=oid(i), audit_bps=5000, forced=False, run_salt=b"\x08" * 32)]
    assert selected != [oid(i) for i in range(1, 41) if audit_selected(
        beacon_randomness=BEACON, observation_id=oid(i), audit_bps=5000, forced=False, run_salt=SALT)]


def test_r17_entry_point_voids_and_reseats_the_scan_on_the_training_observation(pair):
    log, book = pair
    probe = admit(pair, obs(1, hotkey="x"), cap=1.0)
    other = admit(pair, obs(2, hotkey="x"), cap=1.0)
    elsewhere = trained(log, 2, window=2)                           # another window: not R17's business
    here = trained(log, 1)                                          # same window, after the exploration
    assert not here.first_scan and scan_of(log, 1) == probe.observation_id
    with book.db:
        book.resolve_draws(1, environment=ENV, beacon_for_round=lambda r: BEACON, audit_bps=0, run_salt=SALT)
    apply_exploration_verdict(log, book, other.observation_id, passed=True, now=10.0, ban_seconds=DAY)[1]
    unpaid = finalize_exploration(log, book, 1, environment=ENV)
    assert unpaid == [probe.observation_id]
    assert book.state(probe.observation_id)[1] == "trained" and book.payable(1, environment=ENV) == {"x": 1}
    assert scan_of(log, 1) == here.observation_id and log.is_scanned(ENV, 1)
    assert scan_of(log, 2) == other.observation_id                  # paid: it keeps its own first scan
    assert not book.banned("x", 10**9) and book.passed_audits("x") == 1
    assert admit(pair, obs(1, hotkey="late", window=2, group="other"), cap=1.0).reason == "already_scanned"
    assert elsewhere.inserted


@pytest.mark.parametrize("path", ["forfeit", "unaudited", "aborted"])
def test_i1_every_release_path_keeps_a_trained_prompt_scanned(pair, path):
    log, book = pair
    held = admit(pair, obs(1, hotkey="a"), cap=1.0)
    seat = trained(log, 1, window=2, hotkey="b")
    free = admit(pair, obs(2, hotkey="a"), cap=1.0)                 # nobody trains prompt 2
    if path == "forfeit":
        with book.db:
            book.resolve_draws(1, environment=ENV, beacon_for_round=lambda r: BEACON, audit_bps=0, run_salt=SALT)
        apply_exploration_verdict(log, book, held.observation_id, passed=False, now=10.0, ban_seconds=DAY)[1]
    else:
        finalize_exploration(log, book, 1, environment=ENV, aborted=path == "aborted")
    assert scan_of(log, 1) == seat.observation_id and log.is_scanned(ENV, 1)
    assert not log.is_scanned(ENV, 2) and free.first_scan
    later = admit(pair, obs(1, hotkey="c", window=3, group="again"), cap=1.0, now=10.0 + 2 * DAY)
    assert (later.status, later.reason, later.first_scan) == ("exploration_unpaid", "already_scanned", False)


def test_i1_an_admission_refusal_on_a_trained_prompt_leaves_the_training_scan_alone(pair):
    log, book = pair
    seat = trained(log, 1)
    refused = admit(pair, obs(1, hotkey="a"), refuse="token_limit")
    assert refused.reason == "already_scanned" and scan_of(log, 1) == seat.observation_id


# ---------------------------------------------------------------- R25: the unaudited reason

def sub(tmp_path, name):
    (tmp_path / name).mkdir()
    return ledger(tmp_path / name)


def finalize_as(book, reason, *, env="math", window=1):
    with book.db:
        return book.finalize_window(window, environment=env, unaudited_reason=reason)


def test_r25_validator_lost_rows_set_no_horizon_audits_could_run_rows_do(tmp_path):
    lost, horizon = sub(tmp_path, "a"), sub(tmp_path, "b")
    for book, reason in ((lost, "validator_lost"), (horizon, "unaudited")):
        mixed_hotkey(book, "A")                                  # A: drawn row 2 (round 20) never audited; 3 (15), 4 (25), 5 (30)
        finalize_as(book, reason)
    # audits could run: the R15 horizon voids A's later rows. The validator lost them: only the queued row is unpaid.
    assert states(horizon)[4] == ("unaudited", "reserved") and states(horizon)[5] == ("unaudited", "reserved")
    assert horizon.payable(1, environment="math") == {"A": 2}
    assert states(lost) == {1: ("passed", "reserved"), 2: ("unaudited", "reserved"), 3: ("not_drawn", "reserved"),
                            4: ("not_drawn", "reserved"), 5: ("not_drawn", "reserved")}
    assert lost.payable(1, environment="math") == {"A": 4}
    for book in (lost, horizon):
        assert book.db.execute("SELECT COUNT(*) FROM exploration_bans").fetchone()[0] == 0   # never sanctioned


def test_r25_the_ledger_records_the_reason_and_the_horizon_reason_is_the_default(tmp_path):
    book = ledger(tmp_path)
    mixed_hotkey(book, "A")
    finalize(book)                                               # default: audits could run
    assert [book.unaudited_reason(oid(i)) for i in (1, 2, 3, 4, 5)] == [None, "unaudited", None, "unaudited", "unaudited"]
    lost = sub(tmp_path, "x")
    mixed_hotkey(lost, "A")
    finalize_as(lost, "validator_lost")
    assert lost.unaudited_reason(oid(2)) == "validator_lost" and lost.unaudited_reason(oid(4)) is None
    with pytest.raises(ValueError):
        finalize_as(sub(tmp_path, "y"), "because")


def test_r25_a_row_marked_validator_lost_before_the_finalize_sets_no_horizon(tmp_path):
    book = ledger(tmp_path)
    mixed_hotkey(book, "A")
    with book.db:
        book.mark_unaudited(oid(2), "validator_lost")
    assert book.unaudited_reason(oid(2)) == "validator_lost"
    finalize(book)                                               # even a horizon finalize: this row set none
    assert states(book)[4] == ("not_drawn", "reserved") and states(book)[5] == ("not_drawn", "reserved")
    assert book.payable(1, environment="math") == {"A": 4}


def test_r25_a_late_failed_audit_on_a_validator_lost_drawn_row_still_bans(tmp_path):
    book = ledger(tmp_path)
    mixed_hotkey(book, "A")
    finalize_as(book, "validator_lost")
    kind, _ = verdict(book, 2, False, now=10.0, ban=100)
    assert kind == "failed" and book.banned("A", 50.0)


def test_r25_an_old_ledger_without_the_reason_column_is_migrated_and_reads_as_horizon(tmp_path):
    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.executescript("""CREATE TABLE exploration_entitlements(
        observation_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, window INTEGER NOT NULL,
        environment TEXT NOT NULL, hotkey TEXT NOT NULL, prompt_idx INTEGER NOT NULL,
        amount REAL NOT NULL, draw_round INTEGER NOT NULL, forced INTEGER NOT NULL,
        audit TEXT NOT NULL, status TEXT NOT NULL, drawn INTEGER NOT NULL DEFAULT 0);
        INSERT INTO exploration_entitlements VALUES('%s','%s',1,'math','hk',1,0.1,5,0,'unaudited','reserved',1);""" % (oid(1), ORDER))
    db.commit()
    book = ExplorationLedger(db, order_sha256=ORDER)
    assert book.unaudited_reason(oid(1)) == "unaudited"          # an old row set the horizon, as it did
    with book.db:
        book.mark_unaudited(oid(1), "validator_lost")            # the new column works on the migrated table


# ---------------------------------------------------------------- R31: graded audit sanction

from reliquary.constants import (  # noqa: E402
    SERVICE_REPROBATION_PASSES, SERVICE_STATISTICAL_FAIL_BAN_BPS, SERVICE_STATISTICAL_FAIL_WINDOW,
)
from reliquary.services.exploration import (  # noqa: E402
    AUDIT_FAILURE_CLASS_BY_STAGE, AUDIT_FAILURE_DETERMINISTIC as DET, AUDIT_FAILURE_STATISTICAL as STAT,
    audit_failure_class,
)


def _dir(path):
    path.mkdir()
    return path


def concluded(book, i, passed, *, klass=STAT, hotkey="hk", env="math", groups=0, now=1000.0):
    """One drawn row of ``hotkey`` in window ``i`` (alone in its window), then its verdict."""
    assert reserve(book, i, hotkey=hotkey, env=env, window=i, groups=groups) is not None
    draw(book, env=env, window=i, audit_bps=10000)
    with book.db:
        return book.apply_verdict(oid(i), passed=passed, now=now, ban_seconds=DAY, failure_class=klass)


def test_r31_the_constants_are_the_decided_ones():
    assert (SERVICE_REPROBATION_PASSES, SERVICE_STATISTICAL_FAIL_BAN_BPS, SERVICE_STATISTICAL_FAIL_WINDOW) == (20, 1000, 50)


def test_r31_a_deterministic_failure_bans_and_forfeits_and_starts_no_reprobation(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)
    reserve(book, 2, env="code", groups=0)
    draw(book, audit_bps=10000)
    draw(book, env="code", audit_bps=10000)
    with book.db:
        kind, ids = book.apply_verdict(oid(1), passed=False, now=1000.0, ban_seconds=DAY, failure_class=DET)
    assert kind == "failed" and set(ids) == {oid(1), oid(2)}            # every open env of the window
    assert book.banned("hk", 1000.0 + DAY - 1)
    assert book.audit_log("hk") == [{"observation_id": oid(1), "verdict": "failed", "failure_class": DET}]
    assert book.reprobation_passes("hk") is None and not book.in_probation("hk", 0)


def test_r31_a_statistical_failure_forfeits_the_same_rows_without_a_ban_and_reprobates(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)
    reserve(book, 2, env="code", groups=0)
    reserve(book, 3, hotkey="other", groups=0)
    draw(book, audit_bps=10000)
    draw(book, env="code", audit_bps=10000)
    assert not book.in_probation("hk", 0)
    with book.db:
        kind, ids = book.apply_verdict(oid(1), passed=False, now=1000.0, ban_seconds=DAY, failure_class=STAT)
    assert kind == "failed" and set(ids) == {oid(1), oid(2)}            # the forfeit is unchanged
    assert states(book)[1] == ("failed", "forfeited") and states(book, env="code")[2][1] == "forfeited"
    assert states(book)[3] == ("queued", "reserved")                      # another hotkey is untouched
    assert not book.banned("hk", 1000.0)                                 # no ban
    assert book.db.execute("SELECT failure_class FROM exploration_entitlements WHERE observation_id=?",
                           (oid(1),)).fetchone()[0] == STAT
    assert book.audit_log("hk") == [{"observation_id": oid(1), "verdict": "failed", "failure_class": STAT}]
    # re-probation: 100 % audit, and the probation pending limit binds again
    assert book.in_probation("hk", 0) and book.reprobation_passes("hk") == 0
    for i in range(10, 10 + PROBATION_PENDING_LIMIT):
        assert reserve(book, i, window=2, groups=0)["forced"] is True
    assert book.refusal(window=2, environment="math", hotkey="hk", amount=0.1, cap=1.0,
                        new_hotkey_audit_groups=0) == "probation_limit"
    assert reserve(book, 20, window=2, groups=0) is None
    assert reserve(book, 21, window=2, hotkey="other", groups=0)["forced"] is False


def test_r31_reprobation_lifts_after_20_passes_counted_from_the_failure_only(tmp_path):
    book = ledger(tmp_path)
    for i in range(1, 31):                                               # 30 passes BEFORE the failure
        assert concluded(book, i, True)[0] == "passed"
    assert concluded(book, 31, False)[0] == "failed"
    assert book.passed_audits("hk") == 30 and book.in_probation("hk", 0)   # old passes do not lift it
    for i in range(32, 32 + SERVICE_REPROBATION_PASSES - 1):
        concluded(book, i, True)
        assert book.in_probation("hk", 0)
        assert reserve(book, 1000 + i, window=i, groups=0)["forced"] is True   # 100 % audit throughout
    assert book.reprobation_passes("hk") == SERVICE_REPROBATION_PASSES - 1
    concluded(book, 200, True)                                           # the 20th new pass
    assert book.reprobation_passes("hk") == SERVICE_REPROBATION_PASSES
    assert not book.in_probation("hk", 0)
    assert reserve(book, 2000, window=500, groups=0)["forced"] is False
    # a replayed pass is not a new pass
    with book.db:
        assert book.apply_verdict(oid(200), passed=True, now=1000.0, ban_seconds=DAY) == ("passed", [])
    assert book.reprobation_passes("hk") == SERVICE_REPROBATION_PASSES
    # a deterministic failure does not restart the re-probation count (it bans instead)
    concluded(book, 201, False, klass=DET)
    assert not book.in_probation("hk", 0) and book.banned("hk", 1000.0)


def test_r31_a_new_hotkey_stays_in_probation_while_either_counter_holds_it(tmp_path):
    book = ledger(tmp_path)
    for i in range(1, 4):
        concluded(book, i, True, groups=100)
    concluded(book, 4, False, groups=100)
    for i in range(5, 5 + SERVICE_REPROBATION_PASSES):
        concluded(book, i, True, groups=100)
    assert book.reprobation_passes("hk") == SERVICE_REPROBATION_PASSES
    assert book.in_probation("hk", 100)                                  # 23 < 100: still a new hotkey
    assert not book.in_probation("hk", 23)


def test_r31_an_honest_single_statistical_failure_never_bans(tmp_path):
    book = ledger(tmp_path)
    assert concluded(book, 1, False, groups=100)[0] == "failed"          # its very first audit: 1 of 1
    assert not book.banned("hk", 1000.0)
    for i in range(2, 60):
        concluded(book, i, True)
    concluded(book, 60, False)                                           # one more, far apart
    assert not book.banned("hk", 1000.0)


def test_r31_recidivism_bans_above_the_threshold_and_not_at_it(tmp_path):
    allowed = SERVICE_STATISTICAL_FAIL_BAN_BPS * SERVICE_STATISTICAL_FAIL_WINDOW // 10000   # 5 of 50
    book = ledger(tmp_path)
    for i in range(1, allowed + 1):
        concluded(book, i, False)
    assert book.recent_statistical_failures("hk") == allowed
    assert not book.banned("hk", 1000.0)                                 # exactly 10 %: not more than it
    concluded(book, allowed + 1, False, now=2000.0)                      # 6 of the last 50: banned
    assert book.banned("hk", 2000.0 + DAY - 1) and not book.banned("hk", 2000.0 + DAY)
    # the window slides: a failure older than the last 50 audits no longer counts
    other = ledger(_dir(tmp_path / "slide"))
    concluded(other, 1, False, hotkey="hk")
    for i in range(2, 2 + SERVICE_STATISTICAL_FAIL_WINDOW - allowed):    # 45 passes
        concluded(other, i, True, hotkey="hk")
    for i in range(100, 100 + allowed):                                  # failures 2..6; the first slid out
        concluded(other, i, False, hotkey="hk")
    assert other.recent_statistical_failures("hk") == allowed
    assert not other.banned("hk", 1000.0)
    # deterministic failures are not statistical: they never count towards recidivism
    third = ledger(_dir(tmp_path / "det"))
    for i in range(1, allowed + 1):
        concluded(third, i, False, klass=DET, now=float(i))
    concluded(third, 50, False, now=10 * DAY)
    assert third.recent_statistical_failures("hk") == 1 and not third.banned("hk", 10 * DAY)


def test_r31_a_late_statistical_failure_forfeits_open_envs_without_ban_once(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=100)
    reserve(book, 2, env="code", groups=100)
    draw(book)
    draw(book, env="code")
    finalize(book)                                                       # math: drawn, never audited
    before = book.rows(1, environment="math")
    with book.db:
        kind, ids = book.apply_verdict(oid(1), passed=False, now=10.0, ban_seconds=100, failure_class=STAT)
    assert (kind, ids) == ("failed", [oid(2)])
    assert book.rows(1, environment="math") == before and not book.banned("hk", 50.0)
    with book.db:
        assert book.apply_verdict(oid(1), passed=False, now=20.0, ban_seconds=100, failure_class=STAT)[0] == "failed"
    assert len(book.audit_log("hk")) == 1                                # the replay is not a second failure
    assert book.in_probation("hk", 0)


def test_r31_the_failure_class_and_reprobation_survive_reopen(tmp_path):
    book = ledger(tmp_path)
    for i in range(1, 4):
        concluded(book, i, True)
    concluded(book, 4, False)
    for i in range(5, 12):
        concluded(book, i, True)
    log, passes = book.audit_log("hk"), book.reprobation_passes("hk")
    book.db.close()
    reopened = ledger(tmp_path)
    assert reopened.audit_log("hk") == log and reopened.reprobation_passes("hk") == passes == 7
    assert reopened.in_probation("hk", 0) and reserve(reopened, 99, window=99, groups=0)["forced"] is True
    for i in range(12, 12 + SERVICE_REPROBATION_PASSES - 7):
        concluded(reopened, i, True)
    assert not reopened.in_probation("hk", 0)


def test_r31_an_unknown_failure_class_is_refused(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1)
    draw(book)
    with pytest.raises(ValueError, match="failure class"):
        with book.db:
            book.apply_verdict(oid(1), passed=False, now=1.0, ban_seconds=DAY, failure_class="lenient")
    assert states(book)[1] == ("queued", "reserved")


def test_r31_stage_classification_table_and_unknown_stages_are_lenient_with_a_warning(caplog):
    import logging
    for stage in ("grail", "toploc", "termination", "forged_termination", "token_authenticity",
                  "all_token_authenticity", "force_span", "service_seed_coverage"):
        assert audit_failure_class(stage) == DET
    assert audit_failure_class("forced_seed", "cdf_hard_mismatch") == DET
    for stage, scope in (("forced_seed", "group"), ("forced_seed", "rollout"), ("forced_seed", None),
                         ("logprob", None), ("distribution", None), ("boxed_answer", None),
                         ("code_semantic_auth", None), ("episode_replay_binding", None)):
        assert audit_failure_class(stage, scope) == STAT
    assert set(AUDIT_FAILURE_CLASS_BY_STAGE.values()) == {DET, STAT}
    with caplog.at_level(logging.WARNING, logger="reliquary.services.exploration"):
        assert audit_failure_class("a_stage_added_later") == STAT
        assert audit_failure_class(None) == STAT
    assert sum("unclassified proof stage" in r.getMessage() for r in caplog.records) == 2
