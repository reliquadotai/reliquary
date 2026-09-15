"""The archive carries the window's price signal, or carries nothing.

The shadow controller replays these fields and nothing else, so this is where
the sensor either exists or does not. Both directions matter: a measurable
window must publish all three, and an unmeasurable one must publish none --
because window bounds without a readiness read downstream as a shortage.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.unit.test_archive_window_content import (
    _FakeEnv,
    _FakeWallet,
    _valid_submission,
)


class _StubQueue:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    def enqueue(self, window, archive) -> None:
        self._captured["archive"] = archive


class _Pending:
    def __init__(self, drand_round: int) -> None:
        self.drand_round = drand_round


async def _archive_with(batcher) -> dict:
    from reliquary.validator.service import ValidationService

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99
    service = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=tokenizer,
        env=_FakeEnv(), netuid=99,
    )
    submission = _valid_submission(prompt_idx=7)
    captured: dict = {}
    with patch(
        "reliquary.infrastructure.archive_queue.get_archive_queue",
        return_value=_StubQueue(captured),
    ):
        await service._archive_window(batcher, [submission])
    return captured["archive"]


@pytest.mark.asyncio
async def test_a_measurable_window_publishes_its_price_signal():
    submission = _valid_submission(prompt_idx=7)

    class _FakeBatcher:
        window_start = 500
        randomness = "abcd"
        window_opened_at = 0.0
        reject_counts: dict = {}
        rejected_submissions: list = []
        batch_target = 2
        window_open_drand_round = 1000
        _seal_trigger_round = 1100
        _submissions_per_prompt = {
            7: [_Pending(1010), _Pending(1005)],
            9: [_Pending(1020)],
        }

        def valid_submissions(self):
            return [submission]

    archive = await _archive_with(_FakeBatcher())

    assert archive["window_open_round"] == 1000
    assert archive["window_close_round"] == 1100
    # Prompt 7 first landed at 1005 and prompt 9 at 1020, so the second
    # distinct group -- the target -- was there at 1020.
    assert archive["collect_ready_round"] == 1020


@pytest.mark.asyncio
async def test_an_unmeasurable_window_publishes_nothing():
    """Silence, not a half-record that reads as shortage."""
    submission = _valid_submission(prompt_idx=7)

    class _FakeBatcher:
        window_start = 500
        randomness = "abcd"
        window_opened_at = 0.0
        reject_counts: dict = {}
        rejected_submissions: list = []

        def valid_submissions(self):
            return [submission]

    archive = await _archive_with(_FakeBatcher())

    assert "window_open_round" not in archive
    assert "window_close_round" not in archive
    assert "collect_ready_round" not in archive
