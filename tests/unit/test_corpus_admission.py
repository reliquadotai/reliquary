"""One verdict per submission, from pure state. A weight-only node replays
this and must land on the same answer."""

import pytest

from reliquary.corpus.admission import admit
from reliquary.corpus.job import JOB_SCHEMA, PROMPT_ORDER_FREE, parse_job
from reliquary.corpus.slots import SlotLedger
from reliquary.corpus.walk import CursorLedger, walk_index

SHA = "a" * 64
EOS = 151645


def _job(**overrides):
    raw = {
        "schema": JOB_SCHEMA,
        "job_id": "math-v1",
        "checkpoint_repo": "ReliquaryForge/Reliquary-4B",
        "checkpoint_revision": "abc123",
        "checkpoint_sha256": SHA,
        "prompt_source": "openmathinstruct",
        "prompt_count": 1000,
        "renderer_id": "reliquary/render/v5",
        "eos_token_id": EOS,
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_new_tokens": 1,
            "max_new_tokens": 100,
            "n": 2,
        },
        "slots_per_prompt": 2,
        "filter": None,
        "prompt_order": "miner_walk",
        "deadline_round": None,
    }
    raw.update(overrides)
    return parse_job(raw)


def _state(job):
    return (
        SlotLedger(job.prompt_count, job.slots_per_prompt),
        CursorLedger(),
    )


def _call(job, slots, cursors, *, hotkey="5Gx", cursor=None, prompt_index=None, **kwargs):
    cursor = cursors.expected(hotkey) if cursor is None else cursor
    if prompt_index is None:
        prompt_index = walk_index(job.job_id, hotkey, cursor, job.prompt_count)
    payload = {
        "hotkey": hotkey,
        "cursor": cursor,
        "prompt_index": prompt_index,
        "checkpoint_sha256": SHA,
        "token_counts": [10, 12],
        "terminations": ["eos", "eos"],
        "last_token_ids": [EOS, EOS],
        "digests": ["d0", "d1"],
        "slots": slots,
        "cursors": cursors,
        "seen": frozenset(),
    }
    payload.update(kwargs)
    return admit(job, **payload)


def test_a_good_submission_is_accepted_and_moves_the_state():
    job = _job()
    slots, cursors = _state(job)
    verdict = _call(job, slots, cursors)
    assert verdict.accepted is True
    assert verdict.reason == "accepted"
    assert verdict.slots_remaining == 1
    assert cursors.expected("5Gx") == 1
    assert slots.filled == 1


def test_the_wrong_cursor_is_refused_and_moves_nothing():
    job = _job()
    slots, cursors = _state(job)
    verdict = _call(job, slots, cursors, cursor=5)
    assert verdict.accepted is False
    assert verdict.reason == "bad_cursor"
    assert verdict.detail == {"expected": 0, "got": 5}
    assert cursors.expected("5Gx") == 0
    assert slots.filled == 0


def test_a_prompt_outside_the_walk_is_refused_and_moves_nothing():
    job = _job()
    slots, cursors = _state(job)
    mine = walk_index(job.job_id, "5Gx", 0, job.prompt_count)
    verdict = _call(job, slots, cursors, prompt_index=(mine + 1) % job.prompt_count)
    assert verdict.accepted is False
    assert verdict.reason == "prompt_mismatch"
    assert cursors.expected("5Gx") == 0


def test_a_free_order_job_accepts_any_prompt():
    job = _job(prompt_order=PROMPT_ORDER_FREE)
    slots, cursors = _state(job)
    verdict = _call(job, slots, cursors, prompt_index=7)
    assert verdict.accepted is True
    assert slots.remaining(7) == 1


def test_a_full_prompt_is_refused_but_still_costs_the_cursor():
    job = _job()
    slots, cursors = _state(job)
    index = walk_index(job.job_id, "5Gx", 0, job.prompt_count)
    slots.consume(index)
    slots.consume(index)

    verdict = _call(job, slots, cursors)
    assert verdict.accepted is False
    assert verdict.reason == "prompt_full"
    assert verdict.slots_remaining == 0
    # A full prompt still costs the step, so a collision is not a free retry.
    assert cursors.expected("5Gx") == 1


def test_a_completed_job_refuses_everything():
    job = _job(prompt_count=1, slots_per_prompt=1)
    slots, cursors = _state(job)
    slots.consume(0)
    verdict = _call(job, slots, cursors, prompt_index=0)
    assert verdict.reason == "job_complete"
    assert cursors.expected("5Gx") == 0


def test_the_wrong_checkpoint_is_refused():
    job = _job()
    slots, cursors = _state(job)
    verdict = _call(job, slots, cursors, checkpoint_sha256="b" * 64)
    assert verdict.reason == "checkpoint_mismatch"
    assert slots.filled == 0


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        (
            {
                "token_counts": [10],
                "terminations": ["eos"],
                "digests": ["d0"],
                "last_token_ids": [EOS],
            },
            "bad_completion_count",
        ),
        ({"token_counts": [10, 200]}, "token_budget_exceeded"),
        ({"terminations": ["eos", "nope"]}, "bad_termination"),
        ({"seen": {"d1"}}, "hash_duplicate"),
    ],
)
def test_a_cheap_check_failure_is_reported_and_moves_nothing(kwargs, reason):
    job = _job()
    slots, cursors = _state(job)
    verdict = _call(job, slots, cursors, **kwargs)
    assert verdict.accepted is False
    assert verdict.reason == reason
    assert slots.filled == 0
    assert cursors.expected("5Gx") == 0


def test_a_short_completion_labelled_cap_is_refused():
    job = _job()
    slots, cursors = _state(job)
    verdict = _call(job, slots, cursors, terminations=["cap", "cap"])
    assert verdict.accepted is False
    assert verdict.reason == "bad_termination"
    assert verdict.detail == {
        "position": 0,
        "termination": "cap",
        "tokens": 10,
        "max_new_tokens": 100,
    }
    assert slots.filled == 0
    assert cursors.expected("5Gx") == 0


def test_an_eos_label_that_does_not_end_on_eos_is_refused():
    job = _job()
    slots, cursors = _state(job)
    verdict = _call(job, slots, cursors, last_token_ids=[EOS, 7])
    assert verdict.accepted is False
    assert verdict.reason == "bad_termination"
    assert verdict.detail == {
        "position": 1,
        "termination": "eos",
        "last_token_id": 7,
        "eos_token_id": EOS,
    }
    assert slots.filled == 0
    assert cursors.expected("5Gx") == 0


def test_sequences_that_disagree_in_length_are_refused():
    # Two token counts against one termination paid for a completion that no
    # check ever saw.
    job = _job()
    slots, cursors = _state(job)
    verdict = _call(
        job,
        slots,
        cursors,
        token_counts=[50, 50],
        terminations=["eos"],
        digests=["only-one"],
        last_token_ids=[EOS],
    )
    assert verdict.accepted is False
    assert verdict.reason == "malformed_submission"
    assert verdict.detail == {
        "token_counts": 2,
        "terminations": 1,
        "digests": 1,
        "last_token_ids": 1,
    }
    assert slots.filled == 0
    assert cursors.expected("5Gx") == 0


def test_a_one_token_probe_is_refused_under_a_floor():
    # The cheapest cursor step used to be n completions of one token each.
    job = _job(sampling={
        "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "min_new_tokens": 64, "max_new_tokens": 100, "n": 2,
    })
    slots, cursors = _state(job)
    verdict = _call(job, slots, cursors, token_counts=[1, 1])
    assert verdict.accepted is False
    assert verdict.reason == "token_budget_underrun"
    assert verdict.detail == {"position": 0, "tokens": 1, "min_new_tokens": 64}
    assert slots.filled == 0
    assert cursors.expected("5Gx") == 0


def test_two_miners_hold_independent_cursors():
    job = _job()
    slots, cursors = _state(job)
    _call(job, slots, cursors, hotkey="5Gx")
    assert cursors.expected("5Gx") == 1
    assert cursors.expected("5Hy") == 0
    verdict = _call(job, slots, cursors, hotkey="5Hy", digests=["e0", "e1"])
    assert verdict.accepted is True
