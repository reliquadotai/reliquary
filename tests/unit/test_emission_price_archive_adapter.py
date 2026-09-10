"""Reading window outcomes back out of the archive.

The shadow controller replays a history that is overwhelmingly made of archives
written before any of this existed, so the adapter's first job is to say "this
window carries no price signal" without guessing.

The distinction that matters most: a MISSING block of fields means the
validator did not instrument the window, while ``collect_ready_round: null``
inside an instrumented record means the window genuinely never gathered its
target. The first is silence; the second is a shortage that snaps the price up.
Conflating them would push the price up every time an old archive scrolled past.
"""

from __future__ import annotations

from reliquary.validator.emission_price import outcome_from_archive


def _instrumented(**overrides) -> dict:
    record = {
        "window_status": "completed",
        "window_open_round": 1000,
        "window_close_round": 1100,
        "collect_ready_round": 1010,
        "training_rounds": 40.0,
        "validation_rounds": 25.0,
    }
    record.update(overrides)
    return record


def test_an_instrumented_window_reads_back_whole():
    outcome = outcome_from_archive(_instrumented())

    assert outcome is not None
    assert outcome.elapsed_rounds == 100
    assert outcome.filled
    assert outcome.ratio == 0.25   # 10 rounds of collection against 40


def test_a_pre_instrumentation_archive_carries_no_signal():
    """Every archive written before this shipped must read as silence."""
    legacy = {"window_status": "completed", "rewards_by_hotkey": {"hk": 1.0}}

    assert outcome_from_archive(legacy) is None


def test_an_explicit_null_ready_round_is_a_shortage_not_silence():
    """The window was instrumented and it did not fill. That is real news."""
    outcome = outcome_from_archive(_instrumented(collect_ready_round=None))

    assert outcome is not None
    assert not outcome.filled


def test_an_aborted_window_carries_no_signal():
    """``_replay_ema`` already skips aborted windows; the price must agree.

    An aborted window's timings describe the abort, not the market.
    """
    assert outcome_from_archive(_instrumented(window_status="aborted")) is None
