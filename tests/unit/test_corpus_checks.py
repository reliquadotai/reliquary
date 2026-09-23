"""Everything a submission can be refused for without a GPU. These run on
every submission because they are free next to the audit."""

from reliquary.corpus.checks import (
    check_completion_count,
    check_duplicates,
    check_termination,
    check_token_budget,
    completion_digest,
)
from reliquary.corpus.job import Sampling

SAMPLING = Sampling(
    temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=1, max_new_tokens=100, n=2
)
EOS = 151645


def test_the_declared_number_of_completions_is_required():
    assert check_completion_count(2, SAMPLING).ok is True
    for wrong in (0, 1, 3):
        result = check_completion_count(wrong, SAMPLING)
        assert result.ok is False
        assert result.reason == "bad_completion_count"
        assert result.detail == {"expected": 2, "got": wrong}


def test_a_completion_over_the_budget_is_refused():
    assert check_token_budget([100, 4], SAMPLING).ok is True
    result = check_token_budget([4, 101], SAMPLING)
    assert result.ok is False
    assert result.reason == "token_budget_exceeded"
    assert result.detail == {"position": 1, "tokens": 101, "max_new_tokens": 100}


def test_a_completion_under_the_floor_is_refused():
    # The floor is what makes stepping the cursor cost tokens.
    floored = Sampling(
        temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=8, max_new_tokens=100, n=2
    )
    assert check_token_budget([8, 100], floored).ok is True
    result = check_token_budget([8, 7], floored)
    assert result.ok is False
    assert result.reason == "token_budget_underrun"
    assert result.detail == {"position": 1, "tokens": 7, "min_new_tokens": 8}


def test_an_empty_completion_is_an_underrun_not_a_cap_breach():
    # min_new_tokens is always at least 1, so zero tokens is short of the
    # floor. Labelling it "exceeded" reads as a cap breach in reject-reason
    # telemetry, which is the opposite of what happened.
    result = check_token_budget([4, 0], SAMPLING)
    assert result.ok is False
    assert result.reason == "token_budget_underrun"
    assert result.detail == {"position": 1, "tokens": 0, "min_new_tokens": 1}


def _termination(terminations, token_counts, last_token_ids):
    return check_termination(
        terminations, token_counts, last_token_ids, sampling=SAMPLING, eos_token_id=EOS
    )


def test_termination_must_be_declared_and_known():
    assert _termination(["eos", "cap"], [4, 100], [EOS, 9]).ok is True
    result = _termination(["eos", "truncated"], [4, 100], [EOS, 9])
    assert result.ok is False
    assert result.reason == "bad_termination"
    assert result.detail == {"position": 1, "termination": "truncated"}


def test_a_cap_label_must_have_reached_the_cap():
    # A label is not evidence: the token count has to agree with it.
    result = _termination(["cap"], [3], [9])
    assert result.ok is False
    assert result.reason == "bad_termination"
    assert result.detail == {
        "position": 0,
        "termination": "cap",
        "tokens": 3,
        "max_new_tokens": 100,
    }


def test_an_eos_label_must_end_on_the_eos_token():
    result = _termination(["eos"], [4], [9])
    assert result.ok is False
    assert result.reason == "bad_termination"
    assert result.detail == {
        "position": 0,
        "termination": "eos",
        "last_token_id": 9,
        "eos_token_id": EOS,
    }


def test_a_passing_check_never_shares_its_detail():
    # One OK singleton would hand every caller the same mutable dict, and this
    # package exists to be replayed.
    first = check_completion_count(2, SAMPLING)
    second = check_completion_count(2, SAMPLING)
    assert first.detail is not second.detail
    first.detail["poison"] = True
    assert second.detail == {}
    assert check_duplicates([], seen=frozenset()).detail == {}
    assert check_token_budget([4], SAMPLING).detail == {}
    assert _termination(["eos"], [4], [EOS]).detail == {}


def test_the_digest_binds_the_prompt_to_the_tokens():
    assert completion_digest(3, [1, 2, 3]) == completion_digest(3, [1, 2, 3])
    assert completion_digest(3, [1, 2, 3]) != completion_digest(4, [1, 2, 3])
    assert completion_digest(3, [1, 2, 3]) != completion_digest(3, [1, 2, 4])
    assert len(completion_digest(3, [1, 2, 3])) == 64


def test_a_completion_already_seen_is_refused():
    first = completion_digest(3, [1, 2, 3])
    second = completion_digest(3, [9, 9])
    assert check_duplicates([first, second], seen=frozenset()).ok is True

    result = check_duplicates([first, second], seen={second})
    assert result.ok is False
    assert result.reason == "hash_duplicate"
    assert result.detail == {"position": 1, "digest": second}


def test_a_submission_that_repeats_itself_is_refused():
    same = completion_digest(3, [1, 2, 3])
    result = check_duplicates([same, same], seen=frozenset())
    assert result.ok is False
    assert result.reason == "hash_duplicate"
    assert result.detail == {"position": 1, "digest": same}


from reliquary.corpus.checks import check_proof_shape


def test_one_proof_per_chunk_passes():
    assert check_proof_shape([32, 33, 70], [1, 2, 3], 32).ok


def test_a_missing_proof_is_bad_shape():
    result = check_proof_shape([33], [1], 32)
    assert (result.ok, result.reason) == (False, "bad_proof_shape")
    assert result.detail == {"completion": 0, "expected": 2, "got": 1}


def test_counts_must_describe_the_same_completions():
    assert check_proof_shape([10, 10], [1], 32).reason == "bad_proof_shape"
