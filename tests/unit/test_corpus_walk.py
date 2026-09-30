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


def _job(prompt_start, prompt_count, job_id="math-v1"):
    from types import SimpleNamespace

    return SimpleNamespace(job_id=job_id, prompt_start=prompt_start, prompt_count=prompt_count)


def test_a_job_walk_is_the_plain_walk_shifted_by_its_start():
    from reliquary.corpus.walk import job_walk_index

    for start in (0, 1, 5000, 1 << 31):
        job = _job(start, 97)
        for cursor in range(2000):
            index = job_walk_index(job, "5Gx", cursor)
            assert index == start + walk_index("math-v1", "5Gx", cursor, 97)
            assert start <= index < start + 97


def test_a_job_walk_covers_its_range_and_nothing_else():
    from reliquary.corpus.walk import job_walk_index

    job = _job(300, 50)
    seen = {job_walk_index(job, "5Gx", cursor) for cursor in range(2000)}
    # Shifting is a bijection [0, N) -> [S, S+N): what the plain walk reaches,
    # the job walk reaches exactly once shifted, and it reaches all of it.
    assert seen == set(range(300, 350))


def test_a_job_walk_at_zero_start_is_unchanged():
    from reliquary.corpus.walk import job_walk_index

    job = _job(0, 1000)
    assert [job_walk_index(job, "5Gx", c) for c in range(64)] == [
        walk_index("math-v1", "5Gx", c, 1000) for c in range(64)
    ]
