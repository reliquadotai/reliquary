"""The job manifest is the whole of the operator's intent, and it is refused
rather than defaulted when it is incomplete."""

import pytest

from reliquary.corpus.job import (
    JOB_SCHEMA,
    PROMPT_ORDER_FREE,
    PROMPT_ORDER_MINER_WALK,
    JobError,
    parse_job,
)


def _raw(**overrides):
    raw = {
        "schema": JOB_SCHEMA,
        "job_id": "math-v1",
        "checkpoint_repo": "ReliquaryForge/Reliquary-4B",
        "checkpoint_revision": "abc123",
        "checkpoint_sha256": "a" * 64,
        "prompt_source": "openmathinstruct",
        "prompt_count": 1000,
        "renderer_id": "reliquary/render/v5",
        "eos_token_id": 151645,
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_new_tokens": 2,
            "max_new_tokens": 4096,
            "n": 2,
        },
        "slots_per_prompt": 8,
        "filter": None,
        "prompt_order": PROMPT_ORDER_MINER_WALK,
        "deadline_round": None,
    }
    raw.update(overrides)
    return raw


def test_free_generation_job_has_no_filter():
    job = parse_job(_raw())
    assert job.filter is None
    assert job.rejection_sampling is False
    assert job.total_slots == 8000


def test_rejection_sampling_job_carries_a_grader():
    job = parse_job(_raw(filter={"grader_id": "openmathinstruct", "threshold": 1.0}))
    assert job.rejection_sampling is True
    assert job.filter.grader_id == "openmathinstruct"
    assert job.filter.threshold == 1.0


def test_contract_is_json_native_and_stable():
    contract = parse_job(_raw()).to_contract()
    assert contract["schema"] == JOB_SCHEMA
    assert contract["sampling"]["n"] == 2
    assert contract["sampling"]["min_new_tokens"] == 2
    assert contract["eos_token_id"] == 151645
    assert contract["filter"] is None
    assert contract == parse_job(_raw()).to_contract()


def test_a_floor_on_new_tokens_is_carried():
    # The floor prices a cursor step: a probe pays n * min_new_tokens tokens.
    job = parse_job(_raw(sampling={
        "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "min_new_tokens": 64, "max_new_tokens": 4096, "n": 2,
    }))
    assert job.sampling.min_new_tokens == 64
    assert job.to_contract()["sampling"]["min_new_tokens"] == 64


def test_a_floor_of_one_token_is_refused():
    """The terminator counts toward the budget, so a floor of 1 is a floor of
    nothing: `tokens=[eos]` with `text=""` clears the budget, the termination
    check and the text check, and is paid a slot for an empty corpus row.

    Refused in the parser rather than only in the CLI, because a floor is a
    money field and every caller that writes a manifest has to meet the rule.
    """
    from reliquary.corpus.job import MIN_NEW_TOKENS_FLOOR

    with pytest.raises(JobError) as excinfo:
        parse_job(_raw(sampling={
            "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "min_new_tokens": 1, "max_new_tokens": 4096, "n": 1,
        }))
    assert "min_new_tokens" in str(excinfo.value)
    # The lowest floor that leaves a token behind stays declarable, or the
    # refusal is just a bigger floor nobody chose.
    assert MIN_NEW_TOKENS_FLOOR == 2
    assert parse_job(_raw(sampling={
        "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "min_new_tokens": MIN_NEW_TOKENS_FLOOR, "max_new_tokens": 4096, "n": 1,
    })).sampling.min_new_tokens == MIN_NEW_TOKENS_FLOOR


def test_a_floor_above_the_ceiling_names_both_values():
    with pytest.raises(JobError) as excinfo:
        parse_job(_raw(sampling={
            "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "min_new_tokens": 9, "max_new_tokens": 8, "n": 1,
        }))
    assert "9" in str(excinfo.value) and "8" in str(excinfo.value)


def test_the_manifest_pins_the_eos_token():
    # The termination check compares against this, so the job must pin it.
    job = parse_job(_raw(eos_token_id=0))
    assert job.eos_token_id == 0
    assert job.to_contract()["eos_token_id"] == 0


def test_free_prompt_order_is_legal():
    assert parse_job(_raw(prompt_order=PROMPT_ORDER_FREE)).prompt_order == PROMPT_ORDER_FREE


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": "reliquary/corpus-job/v0"},
        {"job_id": "Math V1"},
        {"job_id": ""},
        {"checkpoint_sha256": "a" * 63},
        {"checkpoint_sha256": "z" * 64},
        {"prompt_count": 0},
        {"prompt_count": -1},
        {"slots_per_prompt": 0},
        {"renderer_id": ""},
        {"prompt_order": "whatever"},
        {"sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2, "max_new_tokens": 8, "n": 1}},
        {"sampling": {"temperature": 1.0, "top_p": 0.0, "top_k": 0, "min_new_tokens": 2, "max_new_tokens": 8, "n": 1}},
        {"sampling": {"temperature": 1.0, "top_p": 1.5, "top_k": 0, "min_new_tokens": 2, "max_new_tokens": 8, "n": 1}},
        {"sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_new_tokens": 2, "max_new_tokens": 8, "n": 1}},
        {"sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2, "max_new_tokens": 0, "n": 1}},
        {"sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2, "max_new_tokens": 8, "n": 0}},
        {"sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 0,
                      "max_new_tokens": 8, "n": 1}},
        {"sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 9,
                      "max_new_tokens": 8, "n": 1}},
        {"filter": {"grader_id": "", "threshold": 1.0}},
        {"filter": {"grader_id": "x"}},
        {"deadline_round": -1},
        {"eos_token_id": -1},
        {"eos_token_id": "151645"},
        {"eos_token_id": None},
    ],
)
def test_a_broken_manifest_is_refused(overrides):
    with pytest.raises(JobError):
        parse_job(_raw(**overrides))


def test_an_unknown_field_is_refused():
    with pytest.raises(JobError):
        parse_job(_raw(surprise=1))


@pytest.mark.parametrize("job_id", ["math-v1\n", "math-v1\nrm -rf", "\nmath-v1"])
def test_a_job_id_carrying_a_newline_is_not_a_job_id(job_id):
    """`$` matches before a trailing newline, and this id is interpolated into
    an object key by the job store, which has no `strip` of its own."""
    from reliquary.corpus.job import JOB_ID_RE

    assert JOB_ID_RE.match(job_id) is None


def test_a_checkpoint_digest_carrying_a_newline_is_not_a_digest():
    """Same anchor, same reason: a digest that compares equal with a newline
    attached is a digest two readers can disagree about."""
    from reliquary.corpus.job import _SHA256_RE

    assert _SHA256_RE.match("a" * 64 + "\n") is None


def test_a_job_asking_for_more_completions_than_the_wire_carries_is_refused():
    """`n` completions arrive in ONE request, so a job declaring more than the
    wire's list bound sells slots no submission can ever fill: a bare pydantic
    422, forever, with no reject reason to count."""
    from reliquary.corpus.job import MAX_COMPLETIONS_PER_SUBMISSION

    over = MAX_COMPLETIONS_PER_SUBMISSION + 1
    with pytest.raises(JobError) as excinfo:
        parse_job(_raw(sampling={
            "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "min_new_tokens": 2, "max_new_tokens": 8, "n": over,
        }))
    assert str(over) in str(excinfo.value)
    # The boundary itself stays declarable, or the refusal is only a smaller cap.
    assert parse_job(_raw(sampling={
        "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "min_new_tokens": 2, "max_new_tokens": 8,
        "n": MAX_COMPLETIONS_PER_SUBMISSION,
    })).sampling.n == MAX_COMPLETIONS_PER_SUBMISSION


def test_a_job_whose_token_cap_exceeds_the_wire_is_refused():
    """A completion at the job's own cap must be submissible. Above the wire's
    token bound it cannot be, so the job can only ever refuse."""
    from reliquary.corpus.job import MAX_COMPLETION_TOKENS

    over = MAX_COMPLETION_TOKENS + 1
    with pytest.raises(JobError) as excinfo:
        parse_job(_raw(sampling={
            "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "min_new_tokens": 2, "max_new_tokens": over, "n": 1,
        }))
    assert str(over) in str(excinfo.value)
    assert parse_job(_raw(sampling={
        "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "min_new_tokens": 2, "max_new_tokens": MAX_COMPLETION_TOKENS, "n": 1,
    })).sampling.max_new_tokens == MAX_COMPLETION_TOKENS
