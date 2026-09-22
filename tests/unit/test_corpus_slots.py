"""V slots per prompt, consumed permanently. No timer: exhaustion is the end
of that prompt, and the job ends when every slot is gone."""

import pytest

from reliquary.corpus.slots import SlotExhausted, SlotLedger


def test_a_fresh_prompt_has_every_slot():
    ledger = SlotLedger(prompt_count=10, slots_per_prompt=4)
    assert ledger.remaining(3) == 4
    assert ledger.is_full(3) is False
    assert ledger.filled == 0
    assert ledger.total == 40
    assert ledger.is_complete is False


def test_consuming_returns_what_is_left_and_exhausts_permanently():
    ledger = SlotLedger(prompt_count=2, slots_per_prompt=2)
    assert ledger.consume(0) == 1
    assert ledger.consume(0) == 0
    assert ledger.is_full(0) is True
    with pytest.raises(SlotExhausted):
        ledger.consume(0)
    assert ledger.remaining(1) == 2


def test_the_job_completes_when_every_slot_is_gone():
    ledger = SlotLedger(prompt_count=2, slots_per_prompt=1)
    ledger.consume(0)
    assert ledger.is_complete is False
    ledger.consume(1)
    assert ledger.is_complete is True
    assert ledger.filled == 2


def test_an_index_outside_the_source_is_refused():
    ledger = SlotLedger(prompt_count=3, slots_per_prompt=1)
    for bad in (-1, 3, 4):
        with pytest.raises(IndexError):
            ledger.remaining(bad)
        with pytest.raises(IndexError):
            ledger.consume(bad)


def test_the_snapshot_is_sparse_and_round_trips():
    ledger = SlotLedger(prompt_count=1_000_000, slots_per_prompt=8)
    ledger.consume(7)
    ledger.consume(7)
    ledger.consume(999_999)
    snapshot = ledger.snapshot()
    assert snapshot == {7: 2, 999_999: 1}

    revived = SlotLedger.from_snapshot(1_000_000, 8, snapshot)
    assert revived.remaining(7) == 6
    assert revived.remaining(999_999) == 7
    assert revived.remaining(0) == 8
    assert revived.filled == 3


def test_a_generated_source_costs_no_memory():
    # reliquarylogic declares 1 << 31 prompts; the ledger must not allocate it.
    ledger = SlotLedger(prompt_count=1 << 31, slots_per_prompt=8)
    ledger.consume(2_000_000_000)
    assert ledger.remaining(2_000_000_000) == 7
    assert ledger.total == (1 << 31) * 8
    assert ledger.is_complete is False


@pytest.mark.parametrize("snapshot", [{-1: 1}, {0: 0}, {0: 9}, {0: -1}, {5: 1}])
def test_a_broken_snapshot_is_refused(snapshot):
    with pytest.raises(ValueError):
        SlotLedger.from_snapshot(5, 8, snapshot)
