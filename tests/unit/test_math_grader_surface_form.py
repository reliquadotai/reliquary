"""Math grader must compare by value, not surface form: unit decorations and
structured answers (matrices/vectors/tuples) must not flip a correct reward.
"""
from reliquary.environment.openmathinstruct import _answers_equal, _normalize_answer


def _eq(a, b):
    return _answers_equal(_normalize_answer(a), _normalize_answer(b))


def test_degree_unit_is_stripped():
    assert _eq(r"119^\circ", "119")


def test_frac_matches_slash():
    assert _eq(r"\frac{1}{14}", "1/14")


def test_matrix_elementwise_value_match():
    cand = r"\begin{pmatrix}\frac{1}{14}\\\frac{3}{7}\end{pmatrix}"
    gt = r"\begin{pmatrix}1/14\\3/7\end{pmatrix}"
    assert _eq(cand, gt)


def test_matrix_shape_must_match():
    assert not _eq(
        r"\begin{pmatrix}1&2\end{pmatrix}",
        r"\begin{pmatrix}1\\2\end{pmatrix}",
    )
    assert not _eq(
        r"\begin{pmatrix}1&2\\3&4\end{pmatrix}",
        r"\begin{pmatrix}1&2&3&4\end{pmatrix}",
    )


def test_same_shape_matrix_value_match():
    assert _eq(
        r"\begin{pmatrix}\frac{1}{2}&0.25\end{pmatrix}",
        r"\begin{pmatrix}0.5&\frac{1}{4}\end{pmatrix}",
    )


def test_tuple_same_form_value_match():
    # Same container, decimal vs fraction per element.
    assert _eq("(0.5, 0.25)", "(1/2, 1/4)")


def test_open_and_closed_interval_differ():
    # Brackets carry meaning: an open interval must not match a closed one.
    assert not _eq("(1, 5)", "[1, 5]")


def test_percent_label_matches_bare_number():
    # "What percent" answers: the % is a unit label, "20%" and "20" are the same
    # answer (the dataset's GT carries % inconsistently across otherwise-equal rows).
    assert _eq(r"20\%", "20")


def test_wrong_scalar_stays_wrong():
    assert not _eq("16", "6")


def test_wrong_degree_stays_wrong():
    assert not _eq(r"120^\circ", "119")


def test_wrong_matrix_element_stays_wrong():
    assert not _eq(r"\begin{pmatrix}1/14\\3/7\end{pmatrix}", "(1/14, 9/7)")
