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

SAMPLING = Sampling(temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=100, n=2)


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


def test_an_empty_completion_is_refused():
    result = check_token_budget([4, 0], SAMPLING)
    assert result.ok is False
    assert result.reason == "token_budget_exceeded"


def test_termination_must_be_declared_and_known():
    assert check_termination(["eos", "cap"]).ok is True
    result = check_termination(["eos", "truncated"])
    assert result.ok is False
    assert result.reason == "bad_termination"
    assert result.detail == {"position": 1, "termination": "truncated"}


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
