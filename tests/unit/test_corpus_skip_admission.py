"""`skip`: step over a run of FULL walk positions, landing on `to_cursor`,
exactly as that many `prompt_full` refusals would -- and never over an open
prompt."""

import copy

import pytest

from reliquary.corpus.admission import MAX_SKIP_STEPS, admit, skip, skip_refusal, skip_target
from reliquary.corpus.slots import SlotLedger
from reliquary.corpus.walk import CursorLedger, job_walk_index
from tests.unit.test_corpus_admission import EOS, SHA, _job


def _fill(slots, index):
    while not slots.is_full(index):
        slots.consume(index)


def _state(job):
    return (SlotLedger(job.prompt_count, job.slots_per_prompt, prompt_start=job.prompt_start),
            CursorLedger())


def _fill_walk(job, slots, cursors, hotkey="5Gx", start=0, end=1):
    for c in range(start, end):
        _fill(slots, job_walk_index(job, hotkey, c))


def _skip(job, slots, cursors, *, cursor=0, to_cursor=1, prompt_index=None, hotkey="5Gx"):
    index = job_walk_index(job, hotkey, cursor) if prompt_index is None else prompt_index
    return skip(job, hotkey=hotkey, cursor=cursor, prompt_index=index, to_cursor=to_cursor,
                slots=slots, cursors=cursors)


def _open_run_job(run, **overrides):
    """A job big enough that `run` walk positions from 0 hit distinct rows."""
    return _job(prompt_count=100_000, **overrides)


def test_a_full_prompt_is_skipped_and_the_cursor_moves_one_step():
    job = _job()
    slots, cursors = _state(job)
    _fill_walk(job, slots, cursors)
    filled = slots.filled
    verdict = _skip(job, slots, cursors)
    assert (verdict.accepted, verdict.reason) == (True, "accepted")
    assert cursors.expected("5Gx") == 1
    assert slots.filled == filled


def test_a_run_of_full_positions_is_skipped_in_one_step():
    job = _open_run_job(5)
    slots, cursors = _state(job)
    _fill_walk(job, slots, cursors, end=5)
    verdict = _skip(job, slots, cursors, to_cursor=5)
    assert verdict.accepted is True
    assert cursors.expected("5Gx") == 5


def test_skip_leaves_exactly_the_state_that_many_prompt_full_refusals_leave():
    job = _open_run_job(4)
    slots, cursors = _state(job)
    cursors.advance("5Gx")
    cursors.advance("5Other")
    _fill_walk(job, slots, cursors, start=1, end=5)
    slots.consume(job_walk_index(job, "5Gx", 9))

    s_full, c_full = copy.deepcopy(slots), copy.deepcopy(cursors)
    for c in range(1, 5):
        refused = admit(job, hotkey="5Gx", cursor=c, prompt_index=job_walk_index(job, "5Gx", c),
                        checkpoint_sha256=SHA, token_counts=[10, 12], last_token_ids=[EOS, EOS],
                        digests=[f"d{c}a", f"d{c}b"], slots=s_full, cursors=c_full, seen=frozenset())
        assert refused.reason == "prompt_full"

    s_skip, c_skip = copy.deepcopy(slots), copy.deepcopy(cursors)
    assert _skip(job, s_skip, c_skip, cursor=1, to_cursor=5).accepted is True

    assert s_skip.snapshot() == s_full.snapshot()
    assert c_skip.snapshot() == c_full.snapshot()


@pytest.mark.parametrize("open_at", [0, 1, 3, 4])
def test_a_skip_can_never_jump_over_an_open_prompt(open_at):
    """Any open position in [cursor, to_cursor) refuses the whole skip."""
    job = _open_run_job(5)
    slots, cursors = _state(job)
    _fill_walk(job, slots, cursors, end=5)
    open_index = job_walk_index(job, "5Gx", open_at)
    slots = SlotLedger.from_snapshot(
        job.prompt_count, job.slots_per_prompt,
        {k: v for k, v in slots.snapshot().items() if k != open_index})
    before = (slots.snapshot(), cursors.snapshot())

    verdict = _skip(job, slots, cursors, to_cursor=5)

    assert (verdict.accepted, verdict.reason) == (False, "prompt_not_full")
    assert verdict.detail == {"cursor": open_at, "prompt_index": open_index,
                              "slots_remaining": job.slots_per_prompt}
    assert (slots.snapshot(), cursors.snapshot()) == before


def test_to_cursor_may_land_on_an_open_prompt_but_not_beyond_it():
    job = _open_run_job(3)
    slots, cursors = _state(job)
    _fill_walk(job, slots, cursors, end=2)
    assert _skip(job, copy.deepcopy(slots), CursorLedger(), to_cursor=3).reason == "prompt_not_full"
    assert _skip(job, slots, cursors, to_cursor=2).accepted is True
    assert cursors.expected("5Gx") == 2


@pytest.mark.parametrize("to_cursor", [0, -3, MAX_SKIP_STEPS + 1, 10**9])
def test_the_skip_length_is_bounded(to_cursor):
    job = _open_run_job(MAX_SKIP_STEPS + 1)
    slots, cursors = _state(job)
    _fill_walk(job, slots, cursors, end=MAX_SKIP_STEPS + 1)
    verdict = _skip(job, slots, cursors, to_cursor=to_cursor)
    assert (verdict.accepted, verdict.reason) == (False, "malformed_submission")
    assert verdict.detail == {"cursor": 0, "to_cursor": to_cursor, "max_skip_steps": MAX_SKIP_STEPS}
    assert cursors.snapshot() == {}


def test_the_longest_skip_is_allowed():
    job = _open_run_job(MAX_SKIP_STEPS)
    slots, cursors = _state(job)
    _fill_walk(job, slots, cursors, end=MAX_SKIP_STEPS)
    assert _skip(job, slots, cursors, to_cursor=MAX_SKIP_STEPS).accepted is True
    assert cursors.expected("5Gx") == MAX_SKIP_STEPS


def test_a_skip_at_the_wrong_cursor_is_bad_cursor():
    job = _job()
    slots, cursors = _state(job)
    _fill_walk(job, slots, cursors, start=1, end=2)
    verdict = _skip(job, slots, cursors, cursor=1, to_cursor=2)
    assert (verdict.accepted, verdict.reason) == (False, "bad_cursor")
    assert verdict.detail == {"expected": 0, "got": 1}
    assert cursors.snapshot() == {}


def test_a_skip_naming_another_prompt_is_prompt_mismatch():
    job = _job()
    slots, cursors = _state(job)
    index = job_walk_index(job, "5Gx", 0)
    other = (index + 1) % job.prompt_count
    _fill(slots, other)
    verdict = _skip(job, slots, cursors, prompt_index=other)
    assert (verdict.accepted, verdict.reason) == (False, "prompt_mismatch")
    assert verdict.detail == {"expected": index, "got": other}


def test_a_complete_job_skips_nothing():
    job = _job(prompt_count=2, slots_per_prompt=1)
    slots, cursors = _state(job)
    slots.consume(0)
    slots.consume(1)
    assert _skip(job, slots, cursors).reason == "job_complete"
    assert cursors.snapshot() == {}


def test_a_free_job_has_no_walk_to_skip_along():
    job = _job(prompt_order="free")
    slots, cursors = _state(job)
    _fill(slots, 5)
    verdict = skip(job, hotkey="5Gx", cursor=0, prompt_index=5, to_cursor=1,
                   slots=slots, cursors=cursors)
    assert (verdict.accepted, verdict.reason) == (False, "malformed_submission")
    assert verdict.detail == {"prompt_order": "free"}


def test_a_started_job_skips_along_its_shifted_walk():
    job = _job(prompt_start=5000)
    slots, cursors = _state(job)
    index = job_walk_index(job, "5Gx", 0)
    _fill(slots, index)
    assert _skip(job, slots, cursors, prompt_index=index - 5000).reason == "prompt_mismatch"
    assert _skip(job, slots, cursors).accepted is True


@pytest.mark.parametrize("hotkey", ["5Gx", "5Other"])
def test_a_skip_moves_only_its_own_hotkey(hotkey):
    job = _job()
    slots, cursors = _state(job)
    cursors.advance("5Third")
    _fill(slots, job_walk_index(job, hotkey, 0))
    _skip(job, slots, cursors, hotkey=hotkey)
    assert cursors.snapshot() == {"5Third": 1, hotkey: 1}


def test_skip_refusal_decides_without_moving_anything():
    job = _open_run_job(3)
    slots, cursors = _state(job)
    _fill_walk(job, slots, cursors, end=3)
    kwargs = dict(hotkey="5Gx", cursor=0, prompt_index=job_walk_index(job, "5Gx", 0),
                  to_cursor=3, slots=slots, cursors=cursors)
    assert skip_refusal(job, **kwargs) is None
    assert cursors.snapshot() == {}
    assert skip_refusal(job, **{**kwargs, "cursor": 1}).reason == "bad_cursor"


# --------------------------------------------------------------------------
# skip_target: where next tells the miner to skip to
# --------------------------------------------------------------------------


def test_the_target_is_the_first_open_position_after_the_cursor():
    job = _open_run_job(4)
    slots, cursors = _state(job)
    _fill_walk(job, slots, cursors, end=3)
    assert skip_target(job, "5Gx", 0, slots) == 3
    # The current position does not count, open or not.
    assert skip_target(job, "5Gx", 3, slots) == 4


def test_the_target_is_capped_at_the_skip_bound():
    job = _open_run_job(MAX_SKIP_STEPS + 2)
    slots, cursors = _state(job)
    _fill_walk(job, slots, cursors, end=MAX_SKIP_STEPS + 2)
    assert skip_target(job, "5Gx", 0, slots) == MAX_SKIP_STEPS
    # And a skip to it is accepted.
    assert _skip(job, slots, cursors, to_cursor=MAX_SKIP_STEPS).accepted is True


def test_a_skip_to_the_target_is_always_accepted_when_the_start_is_full():
    job = _job(prompt_count=40)
    slots, cursors = _state(job)
    for i in range(40):
        if i % 4:
            _fill(slots, i)
    for cursor in range(30):
        c = CursorLedger.from_snapshot({"5Gx": cursor} if cursor else {})
        if not slots.is_full(job_walk_index(job, "5Gx", cursor)):
            continue
        target = skip_target(job, "5Gx", cursor, slots)
        assert _skip(job, copy.deepcopy(slots), c, cursor=cursor, to_cursor=target).accepted
        assert not slots.is_full(job_walk_index(job, "5Gx", target))
