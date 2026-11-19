"""A restart must keep each debt change even inside the normal snapshot throttle."""

import pytest

from reliquary.corpus import token_auction
from tests.unit.test_token_auction import _auction


def test_a_restart_keeps_new_debt_and_missed_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(token_auction.time, "monotonic", lambda: 100.0)
    path = tmp_path / "auction.json"
    auction, clock, _ = _auction(state_path=path)
    auction.post_ask("5A", 0.2, 8)  # A snapshot was just written: ordinary updates would be throttled.
    price = auction.params.share_of(0.2)
    auction.lease_ended("5A", price, 1.0)

    restored, _, _ = _auction(state_path=path, clock=clock)
    assert restored.debt_alpha("5A") == pytest.approx(auction.debt_alpha("5A"))
    assert restored.snapshot()["missed"] == auction.snapshot()["missed"]
    assert restored.debt_alpha("5A") > 0


def test_a_restart_does_not_charge_a_repaid_debt_again(tmp_path, monkeypatch):
    monotonic = [100.0]
    monkeypatch.setattr(token_auction.time, "monotonic", lambda: monotonic[0])
    path = tmp_path / "auction.json"
    auction, clock, _ = _auction(state_path=path)
    price = auction.params.share_of(0.2)
    auction.lease_ended("5A", price, 1.0)
    monotonic[0] += 1
    assert auction.take_debit("5A", 1.0) == pytest.approx(price * 1000)
    # No later mutation triggers a snapshot, however long the process stays idle.
    monotonic[0] += 60

    restored, _, _ = _auction(state_path=path, clock=clock)
    assert restored.debt_alpha("5A") == 0
    assert restored.take_debit("5A", 1.0) == 0


def test_a_restart_keeps_a_refunded_debit(tmp_path, monkeypatch):
    monkeypatch.setattr(token_auction.time, "monotonic", lambda: 100.0)
    path = tmp_path / "auction.json"
    auction, clock, _ = _auction(state_path=path)
    auction.post_ask("5A", 0.2, 8)
    refund = auction.params.share_of(0.2) * 1000
    auction.refund_debit("5A", refund)

    restored, _, _ = _auction(state_path=path, clock=clock)
    assert restored.take_debit("5A", 1.0) == pytest.approx(refund)
