import pytest

from reliquary.orders.graders import REFERENCE_GRADERS, check_reference, grade_reference


def test_every_grader_reads_the_answer_after_the_reasoning_or_the_boxed_value():
    assert {g.final_answer_policy for g in REFERENCE_GRADERS.values()} <= {"boxed", "text"}
    assert {"numeric_v1", "choice_v1", "json_match_v1"} <= set(REFERENCE_GRADERS)


@pytest.mark.parametrize("reference, answer, score", [
    ("42", "The answer is 42.", 1.0),
    ("42", "first 41, then \\boxed{42}", 1.0),
    ("42", "\\boxed{42} but maybe 43", 1.0),
    ("42", "maybe 42, finally 43", 0.0),
    ("1/3", "0.3333334", 1.0),
    ("1/3", "0.333", 0.0),
    ("0.5", "\\boxed{\\frac{1}{2}}", 1.0),
    ("1234567", "about 1,234,567 people", 1.0),
    ("-2.5e3", "-2500", 1.0),
    ("100 ± 5", "104", 1.0),
    ("100 +/- 5", "106", 0.0),
    ("0", "1e-9", 1.0),
    ("0", "1e-5", 0.0),
    ("0", "0.0", 1.0),
    ("42", "no number here", 0.0),
    ("42", "\\boxed{x+1}", 0.0),
    ("1", "1e999999999", 0.0),
    ("1", "\\boxed{1e-999999999}", 0.0),
])
def test_numeric(reference, answer, score):
    assert grade_reference("numeric_v1", check_reference("numeric_v1", reference), answer) == score


@pytest.mark.parametrize("reference", ["abc", "1/0", "5 ± -1", "nan", "inf", ""])
def test_numeric_refuses_a_reference_that_is_not_a_number(reference):
    with pytest.raises(ValueError):
        check_reference("numeric_v1", reference)


@pytest.mark.parametrize("reference, answer, score", [
    ("B", "Thinking about A... The answer is (B).", 1.0),
    ("b", "Answer: B", 1.0),
    ("C", "\\boxed{C}", 1.0),
    ("C", "\\boxed{\\text{C}}", 1.0),
    ("D", "D", 1.0),
    ("D", "(D)", 1.0),
    ("A", "Answer: B", 0.0),
    ("A", "I cannot decide between A and B", 0.0),
    ("A", "the answer is A, no wait, the answer is B", 0.0),
    ("A", "the answer is a bit unclear", 0.0),
    ("A", "ANSWER IS A", 1.0),
])
def test_choice(reference, answer, score):
    assert grade_reference("choice_v1", check_reference("choice_v1", reference), answer) == score


@pytest.mark.parametrize("reference", ["AB", "1", "", "é"])
def test_choice_refuses_a_reference_that_is_not_one_letter(reference):
    with pytest.raises(ValueError):
        check_reference("choice_v1", reference)


@pytest.mark.parametrize("reference, answer, score", [
    ('{"a": 1, "b": [1, 2]}', '{"b": [1, 2], "a": 1}', 1.0),
    ('{"a": 1}', 'Here it is:\n```json\n{"a": 1.0}\n```', 1.0),
    ('{"a": 1}', '```json\n{"a": 2}\n```\nfixed:\n```json\n{"a": 1}\n```', 1.0),
    ('{"a": true}', '{"a": 1}', 0.0),
    ('{"a": 1}', '{"a": 1, "b": null}', 0.0),
    ('[1, 2]', '[2, 1]', 0.0),
    ('"x"', '"x"', 1.0),
    ('{"a": 1}', '{"a": 1} trailing', 0.0),
    ('{"a": 1}', '{"a": 1, "a": 1}', 0.0),
    ('{"a": 1}', '{"a": NaN}', 0.0),
])
def test_json_match(reference, answer, score):
    assert grade_reference("json_match_v1", check_reference("json_match_v1", reference), answer) == score


@pytest.mark.parametrize("reference", ["{", '{"a": NaN}', '{"a": 1, "a": 2}'])
def test_json_match_refuses_a_reference_that_is_not_json(reference):
    with pytest.raises(ValueError):
        check_reference("json_match_v1", reference)


def test_a_huge_answer_scores_zero_without_parsing():
    assert grade_reference("json_match_v1", '{"a": 1}', "[" * 200_000) == 0.0
    assert grade_reference("numeric_v1", "1", "1" * 200_000) == 0.0
