"""Pulling the price signal out of the live batchers at seal.

Everything it needs already exists: ``window_open_drand_round`` is stamped from
the real beacon, ``_seal_trigger_round`` from the seal, and
``_submissions_per_prompt`` retains every admitted candidate for the window's
life -- it is appended to and never pruned. So readiness derives at seal with no
new bookkeeping in the admission path.

The mock guard is not defensive noise: the archive tests drive ``_archive_window``
with ``MagicMock`` batchers, whose every attribute answers with another Mock. A
signal built from those would be fiction written into a record that decides
money.
"""

from __future__ import annotations

from reliquary.validator.service import _window_price_signal


class _Pending:
    def __init__(self, drand_round: int) -> None:
        self.drand_round = drand_round


class _Batcher:
    def __init__(
        self,
        submissions_per_prompt: dict | None = None,
        *,
        open_round: int | None = None,
        seal_round: int | None = None,
    ) -> None:
        if submissions_per_prompt is not None:
            self._submissions_per_prompt = submissions_per_prompt
        self.window_open_drand_round = open_round
        self._seal_trigger_round = seal_round


def _target_for(_env_name, _batcher) -> int:
    return 2


def test_readiness_derives_from_the_retained_per_prompt_index():
    batcher = _Batcher(
        {7: [_Pending(1010), _Pending(1005)], 9: [_Pending(1020)]},
        open_round=1000,
        seal_round=1100,
    )

    signal = _window_price_signal(batcher, {"math": batcher}, _target_for)

    assert signal == {
        "window_open_round": 1000,
        "window_close_round": 1100,
        # prompt 7 arrived at 1005 (its earliest), prompt 9 at 1020: the
        # second distinct prompt landed at 1020.
        "collect_ready_round": 1020,
    }


def test_a_mock_batcher_produces_no_signal():
    """A MagicMock answers every attribute with another Mock.

    Reading a price signal off that would write invented numbers into the
    archive, so the absence of a real per-prompt index has to read as silence.
    """
    from unittest.mock import MagicMock

    mock = MagicMock()

    assert _window_price_signal(mock, {"math": mock}, _target_for) is None


def test_a_window_without_a_beacon_round_produces_no_signal():
    batcher = _Batcher({7: [_Pending(1010)]}, open_round=None, seal_round=1100)

    assert _window_price_signal(batcher, {"math": batcher}, _target_for) is None


def test_one_environment_short_still_reports_the_shortage():
    """Measured and it did not fill: that travels, unlike silence."""
    batcher = _Batcher({7: [_Pending(1010)]}, open_round=1000, seal_round=1100)

    signal = _window_price_signal(batcher, {"math": batcher}, _target_for)

    assert signal is not None
    assert signal["collect_ready_round"] is None
