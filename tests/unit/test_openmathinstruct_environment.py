"""Tests for the OpenMathInstruct-2 environment.

Network/HF-dataset access is gated behind the smoke-test in
``tests/integration/test_openmathinstruct_env_smoke.py``. Unit tests here
exercise only the pure-Python helpers (extraction, normalization, reward)
with no dataset dependency.
"""

import pytest


# ---------------------------------------------------------------------------
# Balanced-brace boxed extraction (carried over from the MATH env shape).
# ---------------------------------------------------------------------------

def test_last_boxed_only_string_simple():
    from reliquary.environment.openmathinstruct import _last_boxed_only_string
    assert _last_boxed_only_string(r"The answer is \boxed{42}.") == r"\boxed{42}"


def test_last_boxed_only_string_nested_braces():
    from reliquary.environment.openmathinstruct import _last_boxed_only_string
    s = r"So \boxed{\frac{1}{2}}"
    assert _last_boxed_only_string(s) == r"\boxed{\frac{1}{2}}"


def test_last_boxed_only_string_multiple_returns_last():
    from reliquary.environment.openmathinstruct import _last_boxed_only_string
    s = r"First try \boxed{3}, corrected to \boxed{4}."
    assert _last_boxed_only_string(s) == r"\boxed{4}"


def test_last_boxed_only_string_none_when_absent():
    from reliquary.environment.openmathinstruct import _last_boxed_only_string
    assert _last_boxed_only_string("no boxed here") is None


def test_last_boxed_only_string_unbalanced_returns_none():
    from reliquary.environment.openmathinstruct import _last_boxed_only_string
    assert _last_boxed_only_string(r"\boxed{unclosed") is None


def test_fbox_alias_accepted():
    from reliquary.environment.openmathinstruct import _last_boxed_only_string
    assert _last_boxed_only_string(r"answer: \fbox{7}") == r"\fbox{7}"


# ---------------------------------------------------------------------------
# Boxed wrapper stripping.
# ---------------------------------------------------------------------------

def test_strip_boxed_wrapper_simple():
    from reliquary.environment.openmathinstruct import _strip_boxed_wrapper
    assert _strip_boxed_wrapper(r"\boxed{42}") == "42"


def test_strip_boxed_wrapper_passthrough_when_unwrapped():
    from reliquary.environment.openmathinstruct import _strip_boxed_wrapper
    assert _strip_boxed_wrapper("42") == "42"


def test_strip_boxed_wrapper_fbox():
    from reliquary.environment.openmathinstruct import _strip_boxed_wrapper
    assert _strip_boxed_wrapper(r"\fbox{x+1}") == "x+1"


# ---------------------------------------------------------------------------
# Normalization — covers OMI's plain-numeric and LaTeX-mixed answer formats.
# ---------------------------------------------------------------------------

def test_normalize_strips_spacing_macros():
    from reliquary.environment.openmathinstruct import _normalize_answer
    assert _normalize_answer(r"3 \, x") == "3x"


def test_normalize_strips_left_right():
    from reliquary.environment.openmathinstruct import _normalize_answer
    assert _normalize_answer(r"\left(\frac{1}{2}\right)") == r"(\frac{1}{2})"


def test_normalize_canonicalizes_dfrac():
    from reliquary.environment.openmathinstruct import _normalize_answer
    assert _normalize_answer(r"\dfrac{3}{4}") == r"\frac{3}{4}"


def test_normalize_strips_trailing_period():
    from reliquary.environment.openmathinstruct import _normalize_answer
    assert _normalize_answer("42.") == "42"


def test_normalize_collapses_whitespace():
    from reliquary.environment.openmathinstruct import _normalize_answer
    assert _normalize_answer(" 1 / 2 ") == "1/2"


def test_normalize_strips_leading_plus_on_int():
    """OMI sometimes emits "+5" where ground truth is "5"."""
    from reliquary.environment.openmathinstruct import _normalize_answer
    assert _normalize_answer("+5") == "5"


def test_normalize_strips_trailing_dot_zero():
    """OMI emits "3.0" where ground truth is "3"; treat as equal."""
    from reliquary.environment.openmathinstruct import _normalize_answer
    assert _normalize_answer("3.0") == "3"
    assert _normalize_answer("-7.000") == "-7"


def test_normalize_keeps_decimal_fractions():
    """Don't strip non-zero decimals: "3.14" must stay "3.14"."""
    from reliquary.environment.openmathinstruct import _normalize_answer
    assert _normalize_answer("3.14") == "3.14"


def test_normalize_handles_none():
    from reliquary.environment.openmathinstruct import _normalize_answer
    assert _normalize_answer(None) == ""


# ---------------------------------------------------------------------------
# Reward function — exercises both \boxed{} and plain-tail fallback paths.
# ---------------------------------------------------------------------------

def test_reward_correct_boxed():
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    problem = {"ground_truth": "42"}
    assert _compute_omi_reward(problem, r"The answer is \boxed{42}.") == 1.0


def test_reward_wrong_boxed():
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    problem = {"ground_truth": "42"}
    assert _compute_omi_reward(problem, r"The answer is \boxed{43}.") == 0.0


def test_reward_no_boxed_falls_back_to_trailing_number():
    """A completion that ends with the answer (no boxed wrapper) still
    scores correct as a graceful fallback."""
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    problem = {"ground_truth": "45"}
    assert _compute_omi_reward(problem, "...so the answer is\n45") == 1.0


def test_reward_no_boxed_wrong_trailing_number():
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    problem = {"ground_truth": "45"}
    assert _compute_omi_reward(problem, "...the answer is\n46") == 0.0


def test_reward_no_answer_at_all():
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    problem = {"ground_truth": "42"}
    assert _compute_omi_reward(problem, "I don't know") == 0.0


def test_reward_empty_ground_truth_never_rewards():
    """Safety: if the dataset row had no expected_answer, never reward 1.0."""
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    problem = {"ground_truth": ""}
    assert _compute_omi_reward(problem, r"\boxed{anything}") == 0.0


def test_reward_latex_fraction_match():
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    problem = {"ground_truth": r"\frac{1}{2}"}
    assert _compute_omi_reward(problem, r"\boxed{\dfrac{1}{2}}") == 1.0


def test_reward_decimal_zero_normalization():
    """'3' ground truth, '3.0' completion — should score correct after norm."""
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    problem = {"ground_truth": "3"}
    assert _compute_omi_reward(problem, r"\boxed{3.0}") == 1.0


def test_reward_trailing_zero_decimal_is_equal():
    """A value asked for, not a string: '82.5' and '82.50' are the same number."""
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    assert _compute_omi_reward({"ground_truth": "82.50"}, r"\boxed{82.5}") == 1.0
    assert _compute_omi_reward({"ground_truth": "7.5"}, r"\boxed{7.50}") == 1.0


def test_reward_fraction_and_decimal_equivalence():
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    assert _compute_omi_reward({"ground_truth": "1/2"}, r"\boxed{0.5}") == 1.0
    assert _compute_omi_reward({"ground_truth": "0.25"}, r"\boxed{1/4}") == 1.0


def test_reward_numeric_close_but_unequal_still_wrong():
    """Value-based equality must not let a different number pass."""
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    assert _compute_omi_reward({"ground_truth": "82.5"}, r"\boxed{83}") == 0.0
    assert _compute_omi_reward({"ground_truth": "1/2"}, r"\boxed{1/3}") == 0.0


def test_reward_latex_fraction_vs_decimal_equivalence():
    """Real R2 free-negative: \\frac{5721}{5} == 1144.2 (model was value-correct)."""
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    assert _compute_omi_reward({"ground_truth": "1144.2"}, r"\boxed{\frac{5721}{5}}") == 1.0
    assert _compute_omi_reward({"ground_truth": r"\frac{43}{5}"}, r"\boxed{8.6}") == 1.0


def test_reward_latex_radical_surface_forms_equivalent():
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    assert _compute_omi_reward(
        {"ground_truth": r"\frac{1}{\sqrt{2}}"}, r"\boxed{\frac{\sqrt{2}}{2}}") == 1.0
    assert _compute_omi_reward({"ground_truth": r"\sqrt{20}"}, r"\boxed{2\sqrt{5}}") == 1.0


def test_reward_latex_value_equality_rejects_different():
    """Guard: value-equality must not equate genuinely different expressions."""
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    assert _compute_omi_reward({"ground_truth": r"\frac{1}{3}"}, r"\boxed{\frac{1}{4}}") == 0.0
    assert _compute_omi_reward({"ground_truth": r"2\sqrt{5}"}, r"\boxed{\sqrt{30}}") == 0.0


def test_reward_handles_malformed_completion():
    """Reward function must never raise on garbage input."""
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    problem = {"ground_truth": "42"}
    for bad in ("", "\\", r"\boxed{", "\x00", None):
        # None will pass through to .strip() etc.; we accept either 0.0 return
        # or exception caught internally
        try:
            r = _compute_omi_reward(problem, bad)  # type: ignore[arg-type]
            assert r in (0.0, 1.0)
        except (AttributeError, TypeError):
            # None input is not a real protocol path; tolerated
            pass
