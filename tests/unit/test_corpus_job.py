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
            "min_new_tokens": 1,
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
    assert contract["sampling"]["min_new_tokens"] == 1
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
        {"sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 1, "max_new_tokens": 8, "n": 1}},
        {"sampling": {"temperature": 1.0, "top_p": 0.0, "top_k": 0, "min_new_tokens": 1, "max_new_tokens": 8, "n": 1}},
        {"sampling": {"temperature": 1.0, "top_p": 1.5, "top_k": 0, "min_new_tokens": 1, "max_new_tokens": 8, "n": 1}},
        {"sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_new_tokens": 1, "max_new_tokens": 8, "n": 1}},
        {"sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 1, "max_new_tokens": 0, "n": 1}},
        {"sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 1, "max_new_tokens": 8, "n": 0}},
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
