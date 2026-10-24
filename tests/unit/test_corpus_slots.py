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


def test_a_ledger_with_a_start_keys_by_source_index():
    ledger = SlotLedger(prompt_count=3, slots_per_prompt=2, prompt_start=100)
    for bad in (0, 99, 103):
        with pytest.raises(IndexError):
            ledger.remaining(bad)
    assert ledger.consume(102) == 1
    assert ledger.snapshot() == {102: 1}
    assert ledger.total == 6
    revived = SlotLedger.from_snapshot(3, 2, {102: 1}, prompt_start=100)
    assert revived.remaining(102) == 1
    assert revived.filled == 1


def test_a_snapshot_outside_a_started_range_is_refused():
    with pytest.raises(ValueError):
        SlotLedger.from_snapshot(3, 2, {2: 1}, prompt_start=100)
    with pytest.raises(ValueError):
        SlotLedger.from_snapshot(3, 2, {103: 1}, prompt_start=100)


# -- the open map: which prompts still have a slot, one bit per source row --

def _bit(bitmap, offset):
    return bool(bitmap[offset >> 3] & (0x80 >> (offset & 7)))


def test_the_open_bitmap_of_a_fresh_ledger_is_every_prompt():
    bitmap, count = SlotLedger(prompt_count=10, slots_per_prompt=2).open_bitmap()
    # Ten rows, first row in the high bit, the last byte padded with zeros.
    assert (bitmap, count) == (bytes([0xFF, 0xC0]), 10)


def test_the_open_bitmap_drops_only_the_prompts_with_no_slot_left():
    ledger = SlotLedger(prompt_count=10, slots_per_prompt=2)
    ledger.consume(0)                      # one slot left: still open
    ledger.consume(3), ledger.consume(3)   # full
    ledger.consume(9), ledger.consume(9)   # full
    bitmap, count = ledger.open_bitmap()
    assert count == 8
    assert [_bit(bitmap, i) for i in range(10)] == [ledger.remaining(i) > 0 for i in range(10)]


def test_the_open_bitmap_is_keyed_from_the_jobs_first_row():
    ledger = SlotLedger(prompt_count=9, slots_per_prompt=1, prompt_start=500)
    ledger.consume(500), ledger.consume(508)
    bitmap, count = ledger.open_bitmap()
    assert count == 7 and len(bitmap) == 2
    assert [_bit(bitmap, i) for i in range(9)] == [ledger.remaining(500 + i) > 0 for i in range(9)]


def test_the_open_bitmap_of_a_complete_ledger_is_empty():
    ledger = SlotLedger(prompt_count=3, slots_per_prompt=1)
    for index in range(3):
        ledger.consume(index)
    assert ledger.is_complete and ledger.open_bitmap() == (bytes([0]), 0)


def test_the_open_bitmap_counts_a_slot_a_failed_submission_reopened():
    ledger = SlotLedger(prompt_count=2, slots_per_prompt=1)
    ledger.consume(0), ledger.consume(1)
    assert ledger.record_failure(1, "a" * 12) is True
    bitmap, count = ledger.open_bitmap()
    assert count == 1 and not _bit(bitmap, 0) and _bit(bitmap, 1)


def test_the_open_bitmap_agrees_with_remaining_on_a_random_ledger():
    import random

    rng = random.Random(7)
    ledger = SlotLedger(prompt_count=1003, slots_per_prompt=2, prompt_start=40)
    for _ in range(1500):
        index = 40 + rng.randrange(1003)
        if not ledger.is_full(index):
            ledger.consume(index)
    bitmap, count = ledger.open_bitmap()
    opened = [ledger.remaining(40 + i) > 0 for i in range(1003)]
    assert [_bit(bitmap, i) for i in range(1003)] == opened and count == sum(opened)
