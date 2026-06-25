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


def test_tuple_vs_matrix_value_match():
    # Same vector, different container form, decimal vs fraction per element.
    assert _eq(r"\begin{pmatrix}0.5\\0.25\end{pmatrix}", "(1/2, 1/4)")


def test_wrong_scalar_stays_wrong():
    assert not _eq("16", "6")


def test_wrong_degree_stays_wrong():
    assert not _eq(r"120^\circ", "119")


def test_wrong_matrix_element_stays_wrong():
    assert not _eq(r"\begin{pmatrix}1/14\\3/7\end{pmatrix}", "(1/14, 9/7)")
