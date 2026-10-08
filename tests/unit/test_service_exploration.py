# tests/unit/test_service_exploration.py
import hashlib
import sqlite3

import pytest

from reliquary.services.exploration import (
    ExplorationLedger, audit_selected, exploration_cap, exploration_price, training_group_price,
)

ORDER = "a" * 64


def test_price_is_15_percent_of_a_nominal_training_group():
    assert training_group_price(0.32, picks_target=7, batch_slots=16) == pytest.approx(0.32 / 112)
    assert exploration_price(0.32, picks_target=7, batch_slots=16, price_bps=1500) == pytest.approx(0.15 * 0.32 / 112)
    assert exploration_cap(0.32, cap_bps=1000) == pytest.approx(0.032)


def test_audit_draw_is_deterministic_and_respects_forcing():
    beacon = "ab" * 32
    ids = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(4000)]
    picked = sum(audit_selected(beacon_randomness=beacon, observation_id=i, audit_bps=1500, forced=False) for i in ids)
    assert 450 < picked < 750
    assert all(audit_selected(beacon_randomness=beacon, observation_id=i, audit_bps=0, forced=True) for i in ids[:10])
    assert not any(audit_selected(beacon_randomness=beacon, observation_id=i, audit_bps=0, forced=False) for i in ids[:10])


def ledger(tmp_path):
    return ExplorationLedger(sqlite3.connect(tmp_path / "l.sqlite3"), order_sha256=ORDER)


def reserve(book, i, *, hotkey="hk", env="math", cap=1.0, amount=0.1, window=1, groups=100):
    with book.db:
        return book.reserve(window=window, environment=env, observation_id=f"{i:064x}", hotkey=hotkey,
                            prompt_idx=i, amount=amount, cap=cap, draw_round=50 + i, new_hotkey_audit_groups=groups)


def test_cap_is_per_env_and_window(tmp_path):
    book = ledger(tmp_path)
    assert reserve(book, 1, cap=0.25) is not None
    assert reserve(book, 2, cap=0.25) is not None
    assert reserve(book, 3, cap=0.25) is None
    assert reserve(book, 4, cap=0.25, env="code") is not None


def test_first_hundred_groups_of_a_hotkey_are_forced(tmp_path):
    book = ledger(tmp_path)
    assert reserve(book, 1, groups=2)["forced"] is True
    assert reserve(book, 2, groups=2)["forced"] is True
    assert reserve(book, 3, groups=2)["forced"] is False


def test_failed_audit_forfeits_the_window_and_bans(tmp_path):
    book = ledger(tmp_path)
    for i in (1, 2, 3):
        reserve(book, i, groups=100)
    reserve(book, 4, hotkey="other", groups=100)
    with book.db:
        selected = book.resolve_draws(1, environment="math", beacon_for_round=lambda r: "cd" * 32, audit_bps=1500)
    assert set(selected) == {f"{i:064x}" for i in (1, 2, 3, 4)}
    with book.db:
        forfeited = book.record_audit(f"{2:064x}", passed=False, now=1000.0, ban_seconds=86400)
        book.record_audit(f"{4:064x}", passed=True, now=1000.0, ban_seconds=86400)
    assert set(forfeited) == {f"{i:064x}" for i in (1, 2, 3)}
    assert book.banned("hk", 1000.0 + 86399) and not book.banned("hk", 1000.0 + 86400)
    with book.db:
        book.finalize_window(1, environment="math")
    assert book.payable(1, environment="math") == {"other": pytest.approx(0.1)}


def test_unresolved_draws_are_unpaid_at_window_close(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)
    with book.db:
        moved = book.finalize_window(1, environment="math")
    assert moved == [f"{1:064x}"]
    assert book.payable(1, environment="math") == {}


def test_draw_waits_for_its_beacon(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)
    assert book.pending_draw_rounds(1, environment="math") == [51]
    with book.db:
        assert book.resolve_draws(1, environment="math", beacon_for_round=lambda r: None, audit_bps=1500) == []
    assert book.rows(1, environment="math")[0]["audit"] == "pending_draw"


def test_ledger_survives_reopen_with_pending_draws(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)
    book.db.close()
    reopened = ledger(tmp_path)
    assert reopened.pending_draw_rounds(1, environment="math") == [51]
    assert reopened.rows(1, environment="math")[0]["status"] == "reserved"


def test_reserve_is_idempotent_and_late_audits_cannot_flip(tmp_path):
    book = ledger(tmp_path)
    first = reserve(book, 1, groups=100)
    assert reserve(book, 1, groups=100) == first
    assert len(book.rows(1, environment="math")) == 1
    with book.db:
        book.finalize_window(1, environment="math")
        assert book.record_audit(f"{1:064x}", passed=True, now=1.0, ban_seconds=10) == []
    assert not book.banned("hk", 2.0) and book.payable(1, environment="math") == {}


def test_unaudited_group_is_not_sanctioned(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)
    reserve(book, 2, groups=100)
    with book.db:
        book.resolve_draws(1, environment="math", beacon_for_round=lambda r: "cd" * 32, audit_bps=0)
        book.finalize_window(1, environment="math")
    assert not book.banned("hk", 0.0)
    assert book.payable(1, environment="math") == {"hk": pytest.approx(0.1)}


def test_window_methods_are_scoped_to_one_environment(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, env="math", groups=0)
    reserve(book, 2, env="code", groups=0)
    assert book.pending_draw_rounds(1, environment="code") == [52]
    with book.db:
        assert book.finalize_window(1, environment="math") == [f"{1:064x}"]
    assert [r["audit"] for r in book.rows(1, environment="math")] == ["unaudited"]
    assert [r["audit"] for r in book.rows(1, environment="code")] == ["pending_draw"]
    with book.db:
        assert book.resolve_draws(1, environment="code", beacon_for_round=lambda r: "cd" * 32, audit_bps=0) == []
    assert book.payable(1, environment="code") == {"hk": pytest.approx(0.1)}


def test_finalized_set_survives_reopen(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=0)
    with book.db:
        book.finalize_window(1, environment="math")
    book.db.close()
    reopened = ledger(tmp_path)
    assert reopened.is_finalized(1, environment="math") and not reopened.is_finalized(1, environment="code")


def test_failed_audit_forfeits_every_unfinalized_env_but_never_a_finalized_one(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, env="math", groups=100)
    reserve(book, 2, env="code", groups=100)
    reserve(book, 3, env="science", groups=100)
    with book.db:
        book.resolve_draws(1, environment="math", beacon_for_round=lambda r: "cd" * 32, audit_bps=0)
        book.resolve_draws(1, environment="code", beacon_for_round=lambda r: "cd" * 32, audit_bps=0)
        book.finalize_window(1, environment="science")
    science_before = book.rows(1, environment="science")
    with book.db:
        forfeited = book.record_audit(f"{1:064x}", passed=False, now=10.0, ban_seconds=100)
    assert set(forfeited) == {f"{1:064x}", f"{2:064x}"}
    assert [r["status"] for r in book.rows(1, environment="code")] == ["forfeited"]
    assert book.rows(1, environment="science") == science_before
    assert book.banned("hk", 50.0)


def test_late_failed_audit_after_finalize_bans_and_changes_no_row(tmp_path):
    book = ledger(tmp_path)
    reserve(book, 1, groups=100)
    with book.db:
        book.resolve_draws(1, environment="math", beacon_for_round=lambda r: "cd" * 32, audit_bps=0)
    with book.db:
        book.finalize_window(1, environment="math")
    before = book.rows(1, environment="math")
    with book.db:
        assert book.record_audit(f"{1:064x}", passed=False, now=10.0, ban_seconds=100) == []
    assert book.rows(1, environment="math") == before
    assert book.banned("hk", 50.0)
    with book.db:
        assert book.record_audit(f"{1:064x}", passed=True, now=10.0, ban_seconds=100) == []
    assert book.rows(1, environment="math") == before
