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
        close_round: int | None = None,
        first_pick_round: int | None = None,
    ) -> None:
        if submissions_per_prompt is not None:
            self._submissions_per_prompt = submissions_per_prompt
        self.window_open_drand_round = open_round
        self._seal_trigger_round = seal_round
        self.window_close_drand_round = close_round
        self.window_first_pick_drand_round = first_pick_round


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
        "collect_ready_round_by_environment": {"math": 1020},
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


def test_a_fill_closed_window_reads_its_close_from_the_round_it_closed_at():
    """A v6 window never sets a seal trigger round; its close bound is the round it closed at."""
    batcher = _Batcher(
        {7: [_Pending(1010)], 9: [_Pending(1020)]},
        open_round=1000,
        close_round=1100,
    )

    signal = _window_price_signal(batcher, {"math": batcher}, _target_for)

    assert signal == {
        "window_open_round": 1000,
        "window_close_round": 1100,
        "collect_ready_round": 1020,
        "collect_ready_round_by_environment": {"math": 1020},
    }


def test_a_real_v6_window_carries_a_complete_signal_after_its_last_pick(monkeypatch):
    """Auction admission, prove-on-arrival and the Nth-pick close on one real batcher."""
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.validator import batcher as module
    from reliquary.validator.proof_scheduler import GlobalProofScheduler
    from tests.unit.test_grpo_window_batcher import (
        _execute_scheduler_payload,
        _make_batcher,
        _request,
    )
    from tests.unit.test_proof_scheduler import _wait_until

    for name, value in (
        ("FILL_CLOSED_ENABLED", True),
        ("FILL_CLOSED_BOUNDED_PROOFS", True),
        ("FILL_CLOSED_PROOF_DISPATCH_SECONDS", 60.0),
        ("FILL_CLOSED_MAX_SECONDS", 100.0),
        ("B_BATCH", 1),
    ):
        monkeypatch.setattr(module, name, value)
    chain = {"genesis_time": 1_000_000.0, "period": 3.0}
    now, wall = [10.0], [chain["genesis_time"] + 3000.0]
    scheduler = GlobalProofScheduler(
        devices=("gpu-0",),
        environments=("openmathinstruct", "opencodeinstruct", "reliquary_logic_v2"),
        proof_callable=_execute_scheduler_payload,
        checkpoint_revision="",
        clock=lambda: now[0],
    )
    try:
        batcher = _make_batcher(
            proof_scheduler=scheduler,
            time_fn=lambda: now[0],
            wall_clock_fn=lambda: wall[0],
            drand_chain_info=dict(chain),
        )
        batcher.fill_state = module.FillState(
            budgets={"openmathinstruct": 4}, picks_target=1
        )
        batcher._emit_training_batch_fn = lambda *_args: None
        batcher.mark_window_opened()
        open_round = batcher.window_open_drand_round
        assert isinstance(open_round, int)
        with scheduler._condition:
            for offset, prompt in ((2, 21), (5, 22)):
                request = _request(prompt_idx=prompt, hotkey=str(prompt)).model_copy(
                    update={"drand_round": open_round + offset}
                )
                assert batcher.accept_submission(request).accepted
        _wait_until(batcher._open_proof_plan_handle.done, timeout=5)
        assert batcher.can_pick() and batcher.pick_training_batch()
        wall[0] += 60.0
        assert batcher.poll_deadline() is True
        signal = _window_price_signal(
            batcher, {"openmathinstruct": batcher}, lambda _env, _batcher: 2
        )
    finally:
        assert scheduler.close()

    close_round = compute_current_drand_round(
        wall[0], chain["genesis_time"], chain["period"]
    )
    assert signal == {
        "window_open_round": open_round,
        "window_close_round": close_round,
        "collect_ready_round": open_round + 5,
        "collect_ready_round_by_environment": {"openmathinstruct": open_round + 5},
        # The only pick left 60 s before the close, and a round is 3 s.
        "training_rounds": 20,
    }


def test_the_training_span_runs_from_the_earliest_pick_to_the_close():
    """Environments pick in lockstep but not in the same instant, and the
    trainer has been busy since the first of them: the earliest pick is where
    the window's training span starts."""
    math = _Batcher(
        {7: [_Pending(1005)], 9: [_Pending(1020)]},
        open_round=1000,
        close_round=1100,
        first_pick_round=1040,
    )
    code = _Batcher(
        {7: [_Pending(1005)], 9: [_Pending(1020)]},
        open_round=1000,
        close_round=1100,
        first_pick_round=1030,
    )

    signal = _window_price_signal(math, {"math": math, "code": code}, _target_for)

    assert signal["training_rounds"] == 70


def test_a_window_no_environment_picked_reports_no_training_span():
    batcher = _Batcher(
        {7: [_Pending(1005)], 9: [_Pending(1020)]},
        open_round=1000,
        close_round=1100,
        first_pick_round=None,
    )

    signal = _window_price_signal(batcher, {"math": batcher}, _target_for)

    assert "training_rounds" not in signal
