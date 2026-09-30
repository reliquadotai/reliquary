"""`skip`: step over a FULL prompt exactly the way a `prompt_full` refusal
does, and over nothing else."""

import copy

import pytest

from reliquary.corpus.admission import admit, skip
from reliquary.corpus.slots import SlotLedger
from reliquary.corpus.walk import CursorLedger, job_walk_index
from tests.unit.test_corpus_admission import EOS, SHA, _job


def _fill(job, slots, index):
    while not slots.is_full(index):
        slots.consume(index)


def _state(job):
    return (SlotLedger(job.prompt_count, job.slots_per_prompt, prompt_start=job.prompt_start),
            CursorLedger())


def test_a_full_prompt_is_skipped_and_the_cursor_moves_one_step():
    job = _job()
    slots, cursors = _state(job)
    index = job_walk_index(job, "5Gx", 0)
    _fill(job, slots, index)
    filled = slots.filled

    verdict = skip(job, hotkey="5Gx", cursor=0, prompt_index=index, slots=slots, cursors=cursors)

    assert verdict.accepted is True
    assert verdict.reason == "accepted"
    assert cursors.expected("5Gx") == 1
    assert slots.filled == filled


def test_skip_leaves_exactly_the_state_a_prompt_full_refusal_leaves():
    job = _job()
    slots, cursors = _state(job)
    cursors.advance("5Gx")
    cursors.advance("5Other")
    index = job_walk_index(job, "5Gx", 1)
    _fill(job, slots, index)
    slots.consume((index + 1) % job.prompt_count)

    s_full, c_full = copy.deepcopy(slots), copy.deepcopy(cursors)
    refused = admit(job, hotkey="5Gx", cursor=1, prompt_index=index, checkpoint_sha256=SHA,
                    token_counts=[10, 12], last_token_ids=[EOS, EOS], digests=["d0", "d1"],
                    slots=s_full, cursors=c_full, seen=frozenset())
    assert refused.reason == "prompt_full"

    s_skip, c_skip = copy.deepcopy(slots), copy.deepcopy(cursors)
    skipped = skip(job, hotkey="5Gx", cursor=1, prompt_index=index, slots=s_skip, cursors=c_skip)
    assert skipped.accepted is True

    assert s_skip.snapshot() == s_full.snapshot()
    assert c_skip.snapshot() == c_full.snapshot()


def test_a_prompt_with_a_slot_left_is_not_skipped_and_nothing_moves():
    job = _job()
    slots, cursors = _state(job)
    index = job_walk_index(job, "5Gx", 0)
    slots.consume(index)

    verdict = skip(job, hotkey="5Gx", cursor=0, prompt_index=index, slots=slots, cursors=cursors)

    assert verdict.accepted is False
    assert verdict.reason == "prompt_not_full"
    assert verdict.slots_remaining == 1
    assert verdict.detail == {"prompt_index": index, "slots_remaining": 1}
    assert cursors.snapshot() == {}
    assert slots.snapshot() == {index: 1}


def test_a_skip_at_the_wrong_cursor_is_bad_cursor():
    job = _job()
    slots, cursors = _state(job)
    index = job_walk_index(job, "5Gx", 1)
    _fill(job, slots, index)

    verdict = skip(job, hotkey="5Gx", cursor=1, prompt_index=index, slots=slots, cursors=cursors)

    assert (verdict.accepted, verdict.reason) == (False, "bad_cursor")
    assert verdict.detail == {"expected": 0, "got": 1}
    assert cursors.snapshot() == {}


def test_a_skip_naming_another_prompt_is_prompt_mismatch():
    """Even a full one: the walk names the step, not the miner."""
    job = _job()
    slots, cursors = _state(job)
    index = job_walk_index(job, "5Gx", 0)
    other = (index + 1) % job.prompt_count
    _fill(job, slots, other)

    verdict = skip(job, hotkey="5Gx", cursor=0, prompt_index=other, slots=slots, cursors=cursors)

    assert (verdict.accepted, verdict.reason) == (False, "prompt_mismatch")
    assert verdict.detail == {"expected": index, "got": other}
    assert cursors.snapshot() == {}


def test_a_complete_job_skips_nothing():
    job = _job(prompt_count=2, slots_per_prompt=1)
    slots, cursors = _state(job)
    slots.consume(0)
    slots.consume(1)
    index = job_walk_index(job, "5Gx", 0)

    verdict = skip(job, hotkey="5Gx", cursor=0, prompt_index=index, slots=slots, cursors=cursors)

    assert (verdict.accepted, verdict.reason) == (False, "job_complete")
    assert cursors.snapshot() == {}


def test_a_free_job_has_no_walk_to_skip_along():
    job = _job(prompt_order="free")
    slots, cursors = _state(job)
    _fill(job, slots, 5)

    verdict = skip(job, hotkey="5Gx", cursor=0, prompt_index=5, slots=slots, cursors=cursors)

    assert (verdict.accepted, verdict.reason) == (False, "malformed_submission")
    assert verdict.detail == {"prompt_order": "free"}
    assert cursors.snapshot() == {}


def test_a_started_job_skips_along_its_shifted_walk():
    job = _job(prompt_start=5000)
    slots, cursors = _state(job)
    index = job_walk_index(job, "5Gx", 0)
    assert index >= 5000
    _fill(job, slots, index)

    unshifted = skip(job, hotkey="5Gx", cursor=0, prompt_index=index - 5000,
                     slots=slots, cursors=cursors)
    assert unshifted.reason == "prompt_mismatch"
    assert skip(job, hotkey="5Gx", cursor=0, prompt_index=index,
                slots=slots, cursors=cursors).accepted is True


@pytest.mark.parametrize("hotkey", ["5Gx", "5Other"])
def test_a_skip_moves_only_its_own_hotkey(hotkey):
    job = _job()
    slots, cursors = _state(job)
    cursors.advance("5Third")
    index = job_walk_index(job, hotkey, 0)
    _fill(job, slots, index)
    skip(job, hotkey=hotkey, cursor=0, prompt_index=index, slots=slots, cursors=cursors)
    assert cursors.snapshot() == {"5Third": 1, hotkey: 1}
