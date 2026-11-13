"""Batch audit waves (design 2026-11-13 verification tiers, section 6): a model whose orders all audit in waves
wants no card while it waits, starts a wave when its backlog is worth a card, an order is nearly full or a
record waited too long, drains it, then waits again."""
from __future__ import annotations

import pytest

from reliquary.validator.eval_control import (
    WAVE_DRAINING, WAVE_WAITING, AuditWaves, ExecutorSizing, WaveSettings, order_audit_waves,
)


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _waves(clock=None, **kw):
    return AuditWaves(WaveSettings(**kw), clock=clock or _Clock())


def test_a_small_backlog_waits_and_one_worth_a_card_starts_a_wave():
    waves = _waves(min_seconds=1800.0)
    # 2 items/s: 3,000 items are 25 minutes of one card, under the 30 asked.
    assert waves.update("m", expected=3000, items_per_second=2.0, nearly_full=False,
                        oldest_wait=60.0)["phase"] == WAVE_WAITING
    started = waves.update("m", expected=4000, items_per_second=2.0, nearly_full=False, oldest_wait=60.0)
    assert (started["phase"], started["reason"]) == (WAVE_DRAINING, "backlog")
    # Draining until nothing is left, whatever the backlog meanwhile.
    assert waves.update("m", expected=10, items_per_second=2.0, nearly_full=False,
                        oldest_wait=1.0)["phase"] == WAVE_DRAINING
    done = waves.update("m", expected=0, items_per_second=2.0, nearly_full=False, oldest_wait=None)
    assert (done["phase"], done["reason"]) == (WAVE_WAITING, "drained")


def test_before_any_rate_the_item_count_decides():
    waves = _waves(min_items=2000.0)
    assert waves.update("m", expected=1999, items_per_second=None, nearly_full=False,
                        oldest_wait=0.0)["phase"] == WAVE_WAITING
    assert waves.update("m", expected=2000, items_per_second=None, nearly_full=False,
                        oldest_wait=0.0)["phase"] == WAVE_DRAINING


@pytest.mark.parametrize("nearly_full,oldest_wait,reason", [(True, 0.0, "nearly_full"),
                                                             (False, 10800.0, "max_wait")])
def test_an_order_nearly_full_or_a_record_waiting_too_long_starts_a_wave(nearly_full, oldest_wait, reason):
    waves = _waves()
    state = waves.update("m", expected=5, items_per_second=2.0, nearly_full=nearly_full, oldest_wait=oldest_wait)
    assert (state["phase"], state["reason"]) == (WAVE_DRAINING, reason)


def test_nothing_to_audit_starts_nothing_and_an_unknown_backlog_changes_nothing():
    waves = _waves()
    assert waves.update("m", expected=0, items_per_second=None, nearly_full=True,
                        oldest_wait=1e9)["phase"] == WAVE_WAITING
    waves.update("m", expected=5, items_per_second=None, nearly_full=True, oldest_wait=None)
    assert waves.update("m", expected=None, items_per_second=None, nearly_full=False,
                        oldest_wait=None)["phase"] == WAVE_DRAINING


def test_settings_come_from_the_environment():
    assert WaveSettings.from_env({}) == WaveSettings()
    tuned = WaveSettings.from_env({"RELIQUARY_ORDER_AUDIT_WAVE_MIN_SECONDS": "600",
                                   "RELIQUARY_ORDER_AUDIT_WAVE_FINISH_SHARE": "0.8"})
    assert (tuned.min_seconds, tuned.finish_share) == (600.0, 0.8)
    for bad in ({"RELIQUARY_ORDER_AUDIT_WAVE_FINISH_SHARE": "1.5"}, {"RELIQUARY_ORDER_AUDIT_WAVE_MIN_ITEMS": "0"}):
        with pytest.raises(ValueError):
            WaveSettings.from_env(bad)


def test_only_an_order_record_naming_waves_audits_in_waves():
    assert order_audit_waves({"audit_waves": True}) and not order_audit_waves({}) and not order_audit_waves(None)


def test_a_drained_wave_wants_no_card_at_once():
    clock = _Clock()
    sizing = ExecutorSizing(clock=clock)
    assert sizing.update("m", 2) == 2
    # Lower targets wait out the scale-down; a wave that drained does not.
    assert sizing.update("m", 1) == 2
    assert sizing.set("m", 0) == 0 and sizing.update("m", 0) == 0


def test_a_job_no_longer_open_starts_its_wave_whatever_its_backlog():
    # The control marks such a model nearly full: a closed order's audit must drain even under the size bar.
    waves = _waves(min_items=2000.0)
    state = waves.update("m", expected=3, items_per_second=None, nearly_full=True, oldest_wait=0.0)
    assert (state["phase"], state["reason"]) == (WAVE_DRAINING, "nearly_full")
