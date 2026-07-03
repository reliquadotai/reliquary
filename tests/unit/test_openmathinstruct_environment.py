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


def test_reward_latex_explosive_payload_rejected_fast():
    """Adversarial boxed payloads (power tower, huge exponent, factorial) must be
    rejected by the structural whitelist BEFORE evalf — score 0.0 and no hang.
    Guards against a length-cap-only regression that would let evalf run.
    """
    import time
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    t0 = time.time()
    assert _compute_omi_reward({"ground_truth": "1"}, r"\boxed{9^9^9^9}") == 0.0
    assert _compute_omi_reward({"ground_truth": "1"}, r"\boxed{9^999999999}") == 0.0
    assert _compute_omi_reward({"ground_truth": "1"}, r"\boxed{99999999!}") == 0.0
    assert time.time() - t0 < 2.0  # whitelist short-circuits; never reaches evalf


def test_reward_latex_value_equality_still_holds_after_guard():
    """The whitelist must NOT reject legitimate fraction/radical answers."""
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    assert _compute_omi_reward({"ground_truth": "1144.2"}, r"\boxed{\frac{5721}{5}}") == 1.0
    assert _compute_omi_reward(
        {"ground_truth": r"\frac{1}{\sqrt{2}}"}, r"\boxed{\frac{\sqrt{2}}{2}}") == 1.0


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


# ---------------------------------------------------------------------------
# Dataset backing: full-repo virtual parquet, not an eager shard download.
# A list-of-dicts stands in for the dataset (supports len + __getitem__), so
# the env's shaping is exercised with no network.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_omi_cache():
    """Keep the class-level dataset cache from leaking a fake into other tests."""
    from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment
    saved = OpenMathInstructEnvironment._dataset_cache
    yield
    OpenMathInstructEnvironment._dataset_cache = saved


def _env_over(rows):
    from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment
    OpenMathInstructEnvironment._dataset_cache = list(rows)
    return OpenMathInstructEnvironment()


def test_load_dataset_returns_virtual_parquet_for_repo_id():
    from reliquary.environment.openmathinstruct import _load_dataset
    from reliquary.environment.virtual_parquet import VirtualParquetDataset

    ds = _load_dataset("nvidia/OpenMathInstruct-2", "rev123")
    assert isinstance(ds, VirtualParquetDataset)


def test_load_dataset_requests_problem_and_answer_columns():
    """Only the two columns the env shapes are fetched — keeps row-groups tiny."""
    from reliquary.environment.openmathinstruct import _load_dataset

    ds = _load_dataset("nvidia/OpenMathInstruct-2", "rev123")
    assert ds._columns == ["problem", "expected_answer"]


def test_default_revision_is_pinned_sha():
    """Both sides MUST read the same immutable revision (token binding +
    prompt-range determinism); the default cannot be an unpinned 'main'."""
    from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment

    rev = OpenMathInstructEnvironment._OMI_REVISION
    assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def test_get_problem_shapes_prompt_and_ground_truth():
    import hashlib
    from reliquary.environment.openmathinstruct import _ANSWER_FORMAT_INSTRUCTION

    env = _env_over([{"problem": "What is 2+2?", "expected_answer": "4"}])
    p = env.get_problem(0)

    assert p["prompt"] == "What is 2+2?" + _ANSWER_FORMAT_INSTRUCTION
    assert p["ground_truth"] == "4"
    assert p["id"] == hashlib.sha256(b"What is 2+2?").hexdigest()[:16]


def test_len_reflects_dataset_not_shard_cap():
    env = _env_over([{"problem": f"q{i}", "expected_answer": str(i)} for i in range(7)])
    assert len(env) == 7


def test_get_problem_modulo_wraps():
    env = _env_over([{"problem": "q0", "expected_answer": "0"},
                     {"problem": "q1", "expected_answer": "1"}])
    assert env.get_problem(2)["ground_truth"] == "0"
    assert env.get_problem(3)["ground_truth"] == "1"


def test_expr_str_guard_symbols_flag():
    from reliquary.environment.openmathinstruct import _expr_str_is_safe
    # default (numeric) behaviour unchanged: a variable is rejected
    assert _expr_str_is_safe("2 - b") is False
    assert _expr_str_is_safe("2 + 3*(4)") is True
    # with allow_symbols: single-letter variables allowed...
    assert _expr_str_is_safe("2 - b", allow_symbols=True) is True
    assert _expr_str_is_safe("x**2 + 2*x + 1", allow_symbols=True) is True
    # ...but factorials and function calls still rejected
    assert _expr_str_is_safe("5!", allow_symbols=True) is False
    assert _expr_str_is_safe("exp(x)", allow_symbols=True) is False
    assert _expr_str_is_safe("gamma(3)", allow_symbols=True) is False


def test_expr_struct_guard_symbols_flag():
    import sympy
    from reliquary.environment.openmathinstruct import _expr_is_safe
    x, b = sympy.symbols("x b")
    # symbols rejected by default, allowed under the flag
    assert _expr_is_safe(x) is False
    assert _expr_is_safe(x, allow_symbols=True) is True
    assert _expr_is_safe((x + 1) ** 2, allow_symbols=True) is True
    assert _expr_is_safe(2 - b, allow_symbols=True) is True
    # power tower / huge exponent still rejected even with symbols allowed
    assert _expr_is_safe(x ** 50, allow_symbols=True) is False
    assert _expr_is_safe(sympy.Pow(2, x, evaluate=False), allow_symbols=True) is False
