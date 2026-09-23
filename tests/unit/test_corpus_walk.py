"""Each miner consumes the prompt source in its own deterministic order, so
collisions are random rather than correlated by a shared heuristic, and no
miner chooses which prompt it answers next. Skipping one is not made as
expensive as answering it; the audit tier closes that residue."""

import pytest

from reliquary.corpus.walk import CursorLedger, walk_index


def test_the_walk_is_deterministic():
    first = walk_index("math-v1", "5Gx", cursor=0, prompt_count=1000)
    assert first == walk_index("math-v1", "5Gx", cursor=0, prompt_count=1000)


def test_two_miners_walk_different_orders():
    a = [walk_index("math-v1", "5Gx", c, 100_000) for c in range(16)]
    b = [walk_index("math-v1", "5Hy", c, 100_000) for c in range(16)]
    assert a != b


def test_two_jobs_walk_different_orders():
    a = [walk_index("math-v1", "5Gx", c, 100_000) for c in range(16)]
    b = [walk_index("code-v1", "5Gx", c, 100_000) for c in range(16)]
    assert a != b


def test_the_walk_stays_inside_the_source():
    for cursor in range(200):
        assert 0 <= walk_index("math-v1", "5Gx", cursor, 7) < 7


def test_the_walk_spreads_over_the_source():
    # A shared heuristic would make every miner collide; a hash walk must not.
    seen = {walk_index("math-v1", "5Gx", c, 1000) for c in range(500)}
    assert len(seen) > 300


@pytest.mark.parametrize("bad", [(-1, 10), (0, 0), (0, -5)])
def test_impossible_arguments_are_refused(bad):
    cursor, prompt_count = bad
    with pytest.raises(ValueError):
        walk_index("math-v1", "5Gx", cursor, prompt_count)


def test_a_new_miner_starts_at_zero():
    ledger = CursorLedger()
    assert ledger.expected("5Gx") == 0


def test_the_cursor_advances_by_exactly_one():
    ledger = CursorLedger()
    assert ledger.advance("5Gx") == 1
    assert ledger.expected("5Gx") == 1
    assert ledger.advance("5Gx") == 2
    assert ledger.expected("5Hy") == 0


def test_the_cursor_round_trips_through_a_snapshot():
    ledger = CursorLedger()
    ledger.advance("5Gx")
    ledger.advance("5Gx")
    ledger.advance("5Hy")
    assert ledger.snapshot() == {"5Gx": 2, "5Hy": 1}

    revived = CursorLedger.from_snapshot({"5Gx": 2, "5Hy": 1})
    assert revived.expected("5Gx") == 2
    assert revived.expected("5Zz") == 0


@pytest.mark.parametrize("snapshot", [{"5Gx": -1}, {"": 1}])
def test_a_broken_cursor_snapshot_is_refused(snapshot):
    with pytest.raises(ValueError):
        CursorLedger.from_snapshot(snapshot)


@pytest.mark.parametrize("cursor", [1.5, 2.0, True, "2"])
def test_a_cursor_that_is_not_a_whole_number_is_refused_not_rounded(cursor):
    """`int()` would silently turn 1.5 into 1 and hand that miner a step it
    never took. A cursor is a position in a money ledger, so a snapshot this
    binary cannot read exactly is named rather than coerced."""
    with pytest.raises(ValueError):
        CursorLedger.from_snapshot({"5Gx": cursor})
