"""The shadow decision rides in the archive without touching the pool.

Phase 1 exists to answer a question we cannot answer any other way: is the
capturable gap 5x or 50x? It answers it by computing exactly what the armed
controller would have paid, publishing it, and paying none of it. Nothing here
may change what a miner earns.

These tests pin the disarmed path (``RELIQUARY_EMISSION_PRICE_ARMED=0``); the
armed pool is covered in test_price_arming.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.unit.test_archive_window_content import (
    _FakeEnv,
    _FakeWallet,
    _valid_submission,
)


@pytest.fixture(autouse=True)
def _disarmed(monkeypatch):
    monkeypatch.setattr("reliquary.constants.EMISSION_PRICE_ARMED", False)


class _StubQueue:
    def __init__(self, captured: list) -> None:
        self._captured = captured

    def enqueue(self, window, archive) -> None:
        self._captured.append(archive)


class _Pending:
    def __init__(self, drand_round: int) -> None:
        self.drand_round = drand_round


def _oversupplied_batcher(*, open_round: int):
    """A window whose collection finished 10 rounds into a 1000-round span."""
    submission = _valid_submission(prompt_idx=7)

    class _FakeBatcher:
        window_start = 500
        randomness = "abcd"
        window_opened_at = 0.0
        reject_counts: dict = {}
        rejected_submissions: list = []
        batch_target = 2
        window_open_drand_round = open_round
        _seal_trigger_round = open_round + 1000
        _submissions_per_prompt = {
            7: [_Pending(open_round + 5)],
            9: [_Pending(open_round + 10)],
        }

        def valid_submissions(self):
            return [submission]

    return _FakeBatcher(), submission


def _service():
    from reliquary.validator.service import ValidationService

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99
    return ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=tokenizer,
        env=_FakeEnv(), netuid=99,
    )


async def _archive_all(service, batchers) -> list:
    captured: list = []
    with patch(
        "reliquary.infrastructure.archive_queue.get_archive_queue",
        return_value=_StubQueue(captured),
    ):
        for batcher, submission in batchers:
            await service._archive_window(batcher, [submission])
    return captured


@pytest.mark.asyncio
async def test_an_oversupplied_window_publishes_a_lower_shadow_price():
    service = _service()

    archives = await _archive_all(service, [_oversupplied_batcher(open_round=1000)])

    shadow = archives[0]["emission_price_shadow"]
    assert shadow["regime"] == "descend"
    assert shadow["price"] < 1.0


@pytest.mark.asyncio
async def test_the_shadow_price_keeps_walking_across_windows():
    """The state carries forward, so the descent compounds rather than resets."""
    service = _service()

    archives = await _archive_all(
        service,
        [
            _oversupplied_batcher(open_round=1000),
            _oversupplied_batcher(open_round=3000),
        ],
    )

    assert (
        archives[1]["emission_price_shadow"]["price"]
        < archives[0]["emission_price_shadow"]["price"]
    )


@pytest.mark.asyncio
async def test_the_shadow_never_touches_what_the_window_pays():
    """Phase 1 is a measurement. Two windows priced differently pay the same.

    Comparing a walked-down service against a fresh one is the direct
    statement of the invariant: if pay tracked the shadow at all, the second
    window's rewards would differ from the first's.
    """
    fresh = await _archive_all(_service(), [_oversupplied_batcher(open_round=1000)])
    walked = await _archive_all(
        _service(),
        [
            _oversupplied_batcher(open_round=1000),
            _oversupplied_batcher(open_round=3000),
        ],
    )

    assert (
        walked[1]["emission_price_shadow"]["price"]
        < fresh[0]["emission_price_shadow"]["price"]
    )
    assert walked[1]["emission_price_shadow"]["applied"] is False
    assert walked[1]["rewards_by_hotkey"] == fresh[0]["rewards_by_hotkey"]


@pytest.mark.asyncio
async def test_an_unmeasurable_window_publishes_no_shadow():
    submission = _valid_submission(prompt_idx=7)

    class _FakeBatcher:
        window_start = 500
        randomness = "abcd"
        window_opened_at = 0.0
        reject_counts: dict = {}
        rejected_submissions: list = []

        def valid_submissions(self):
            return [submission]

    archives = await _archive_all(_service(), [(_FakeBatcher(), submission)])

    assert "emission_price_shadow" not in archives[0]


def test_the_shadow_publishes_a_price_per_environment():
    """Each environment's walk reaches the archive, not just the window's."""
    service = _service()          # follow this file's own existing helper
    signal = {
        "window_open_round": 0,
        "window_close_round": 1000,
        "collect_ready_round": 800,
        "collect_ready_round_by_environment": {"math": 800, "code": 200},
    }

    shadow = service._advance_price_shadow(signal)

    assert shadow["applied"] is False
    by_environment = shadow["by_environment"]
    assert set(by_environment) == {"math", "code"}
    for entry in by_environment.values():
        assert entry["applied"] is False
        assert set(entry) >= {"price", "last_good", "r", "r_smoothed", "regime", "applied"}


def test_an_environment_keeps_its_own_shadow_walk_across_windows():
    """Two windows, and the cheap environment's price must not track the dear one's."""
    service = _service()
    for _ in range(3):
        shadow = service._advance_price_shadow({
            "window_open_round": 0,
            "window_close_round": 1000,
            "collect_ready_round": 900,
            "collect_ready_round_by_environment": {"math": 900, "code": 100},
        })

    assert shadow["by_environment"]["math"]["price"] != (
        shadow["by_environment"]["code"]["price"]
    )


def test_a_record_without_the_map_still_produces_the_window_shadow():
    """Archives written before the map existed must keep working unchanged."""
    service = _service()

    shadow = service._advance_price_shadow({
        "window_open_round": 0,
        "window_close_round": 1000,
        "collect_ready_round": 800,
    })

    assert shadow["applied"] is False
    assert "by_environment" not in shadow


def test_a_timed_out_window_does_not_price_lagging_proofs_as_a_fast_fill():
    """A timed-out window's fast admission must not read as a cheap fill.

    ``outcomes_by_environment_from_archive`` zeroes the denominator for a
    timed-out record so a window that never finished proving reads as
    unmeasurable rather than fast. That branch only fires if the real
    ``window_status`` reaches it instead of a hardcoded "completed".
    """
    signal = {
        "window_open_round": 0,
        "window_close_round": 2400,
        "collect_ready_round": 110,
        "collect_ready_round_by_environment": {"math": 110, "code": 110},
    }

    completed = _service()._advance_price_shadow(signal, window_status="completed")
    timed_out = _service()._advance_price_shadow(signal, window_status="timed_out")

    assert completed["by_environment"]["math"]["regime"] == "descend"
    assert completed["by_environment"]["math"]["price"] < 1.0
    assert timed_out["by_environment"]["math"]["regime"] != "descend"
    assert timed_out["by_environment"]["math"]["price"] == 1.0


@pytest.mark.asyncio
async def test_every_priced_window_publishes_the_block_miners_read():
    """/tasks must show the decision the archive carries, not a separate one."""
    service = _service()

    archives = await _archive_all(
        service,
        [_oversupplied_batcher(open_round=1000), _oversupplied_batcher(open_round=3000)],
    )

    view = service.server._price_view
    shadow = archives[-1]["emission_price_shadow"]
    assert view["value"] == shadow["price"]
    assert view["regime"] == shadow["regime"]
    assert view["applied"] is False
