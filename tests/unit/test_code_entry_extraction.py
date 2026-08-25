"""Extraction of the graded Python block by contract entry point (protocol v6).

Protocol v4/v5 grade ``matches[-1]`` — the last fenced block. That rule was
written when the environment ran a chat model that always closed with its final
implementation. Under the v5 reasoning prompt the model often closes with a
usage demo, an expected-output listing, or a test block instead, so the graded
span is not the implementation and the rollout scores zero despite being right.

v6 grades the last block that *defines the contract's entry function*, falling
back to ``matches[-1]`` when no block defines it. The change is wire-affecting
(miner and validator both recompute the reward and compare within 1e-6), so it
is gated on ``PROTOCOL_VERSION >= 6`` and only takes effect at the profile
cutover.
"""

import pytest


ENTRY = "is_balanced_parentheses"

IMPLEMENTATION = (
    "def is_balanced_parentheses(s):\n"
    "    counter = 0\n"
    "    for ch in s:\n"
    "        if ch == '(':\n"
    "            counter += 1\n"
    "        elif ch == ')':\n"
    "            counter -= 1\n"
    "            if counter < 0:\n"
    "                return False\n"
    "    return counter == 0"
)


def _fence(body: str) -> str:
    return f"```python\n{body}\n```"


@pytest.fixture
def v6(monkeypatch):
    """Activate the v6 extraction rule without switching the whole profile."""
    import reliquary.constants as constants

    monkeypatch.setattr(constants, "PROTOCOL_VERSION", 6, raising=False)


# ---------------------------------------------------------------------------
# _entry_function_name: the contract's graded entry point.
# ---------------------------------------------------------------------------

def test_entry_function_name_reads_the_function_contract():
    from reliquary.environment.opencodeinstruct import _entry_function_name

    cases = [{"entry": {"kind": "function", "name": ENTRY}, "args": ["()"]}]
    assert _entry_function_name(cases) == ENTRY


def test_entry_function_name_is_none_for_a_method_entry():
    """Only function entries pin a ``def <name>``; methods live inside a class."""
    from reliquary.environment.opencodeinstruct import _entry_function_name

    cases = [{"entry": {"kind": "method", "class_name": "Solution", "method": "run"}}]
    assert _entry_function_name(cases) is None


def test_entry_function_name_is_none_for_empty_cases():
    from reliquary.environment.opencodeinstruct import _entry_function_name

    assert _entry_function_name([]) is None


# ---------------------------------------------------------------------------
# _extract_python under v6.
# ---------------------------------------------------------------------------

def test_extract_python_skips_a_trailing_usage_demo(v6):
    """The observed production failure: implementation first, `print(...)` last."""
    from reliquary.environment.opencodeinstruct import _extract_python

    text = (
        "Here is my reasoning.\n\n"
        + _fence(IMPLEMENTATION)
        + "\n\nExample usage:\n\n"
        + _fence('print(is_balanced_parentheses("()"))')
    )
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_extract_python_skips_a_trailing_expected_output_listing(v6):
    """A fenced block holding the program's *output*, which defines nothing."""
    from reliquary.environment.opencodeinstruct import _extract_python

    text = _fence(IMPLEMENTATION) + "\n\nThis will output:\n\n" + _fence("True\nTrue\nFalse")
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_extract_python_skips_a_trailing_test_block(v6):
    from reliquary.environment.opencodeinstruct import _extract_python

    text = _fence(IMPLEMENTATION) + "\n\nTests:\n\n" + _fence(
        'assert is_balanced_parentheses("()") is True'
    )
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_extract_python_keeps_the_last_block_that_defines_the_entry(v6):
    """Self-correction must still win: a later rewrite supersedes an earlier draft."""
    from reliquary.environment.opencodeinstruct import _extract_python

    draft = "def is_balanced_parentheses(s):\n    return False"
    text = _fence(draft) + "\n\nActually, that is wrong. Corrected:\n\n" + _fence(IMPLEMENTATION)
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_extract_python_falls_back_to_last_block_when_none_defines_the_entry(v6):
    """No block defines the contract function — behave exactly as v5 did."""
    from reliquary.environment.opencodeinstruct import _extract_python

    text = _fence("helper = 1") + "\n\nand:\n\n" + _fence("other = 2")
    assert _extract_python(text, entry_name=ENTRY) == "other = 2"


def test_extract_python_single_block_is_unaffected_by_the_entry_name(v6):
    """83% of rollouts emit exactly one block; the rule must not touch them."""
    from reliquary.environment.opencodeinstruct import _extract_python

    text = "Reasoning first.\n\n" + _fence(IMPLEMENTATION)
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION
    assert _extract_python(text, entry_name=None) == IMPLEMENTATION


def test_extract_python_without_an_entry_name_keeps_last_block_wins(v6):
    """A method-entry problem pins no function name; the old rule still applies."""
    from reliquary.environment.opencodeinstruct import _extract_python

    text = _fence(IMPLEMENTATION) + "\n\n" + _fence("print('demo')")
    assert _extract_python(text, entry_name=None) == "print('demo')"


def test_extract_python_requires_a_definition_not_a_mention(v6):
    """A block merely *calling* the function must not be mistaken for the impl."""
    from reliquary.environment.opencodeinstruct import _extract_python

    text = _fence(IMPLEMENTATION) + "\n\n" + _fence(
        "result = is_balanced_parentheses('()')"
    )
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_extract_python_no_fence_still_returns_the_raw_completion(v6):
    from reliquary.environment.opencodeinstruct import _extract_python

    assert _extract_python(IMPLEMENTATION, entry_name=ENTRY) == IMPLEMENTATION


def test_extract_python_empty_completion_is_empty(v6):
    from reliquary.environment.opencodeinstruct import _extract_python

    assert _extract_python("", entry_name=ENTRY) == ""


# ---------------------------------------------------------------------------
# Wire compatibility: the rule must stay inert before the v6 cutover.
# ---------------------------------------------------------------------------

def test_extract_python_ignores_the_entry_name_below_protocol_v6(monkeypatch):
    """Miner and validator compare rewards within 1e-6. Changing the graded span
    before the coordinated cutover would reject honest miners as dishonest."""
    import reliquary.constants as constants
    from reliquary.environment.opencodeinstruct import _extract_python

    monkeypatch.setattr(constants, "PROTOCOL_VERSION", 5, raising=False)
    text = _fence(IMPLEMENTATION) + "\n\n" + _fence("True\nTrue\nFalse")
    assert _extract_python(text, entry_name=ENTRY) == "True\nTrue\nFalse"


# ---------------------------------------------------------------------------
# compute_reward wires the contract through to the extractor.
# ---------------------------------------------------------------------------

def test_compute_reward_grades_the_implementation_not_the_trailing_demo(v6, monkeypatch):
    from reliquary.environment import opencodeinstruct as mod

    graded: list[str] = []

    class _FakeGrader:
        def evaluate_cases(self, code, cases, timeout_s):
            graded.append(code)
            return 1.0

    env = mod.OpenCodeInstructEnvironment.__new__(mod.OpenCodeInstructEnvironment)
    cases = [{"entry": {"kind": "function", "name": ENTRY}, "args": ["()"], "expected": True}]
    env._cases_by_id = {"case-1": cases}
    env._grader = _FakeGrader()

    completion = _fence(IMPLEMENTATION) + "\n\nOutput:\n\n" + _fence("True")
    env.compute_reward({"ground_truth": "case-1"}, completion)

    assert graded == [IMPLEMENTATION]


# ---------------------------------------------------------------------------
# The admission path grades the same span as compute_reward.
# ---------------------------------------------------------------------------

def test_admission_grades_the_implementation_not_the_trailing_demo(v6, monkeypatch):
    """Admission recomputes the reward to check the miner's claim. If it graded a
    different span than compute_reward, honest miners would fail reward_mismatch."""
    from reliquary.validator import admission

    graded: list[str] = []

    class _FakeGrader:
        def evaluate_cases(self, code, cases, timeout_s):
            graded.append(code)
            return 1.0

    monkeypatch.setattr(admission, "GraderClient", lambda: _FakeGrader())

    cases = [{"entry": {"kind": "function", "name": ENTRY}, "args": ["()"], "expected": True}]
    completion = _fence(IMPLEMENTATION) + "\n\nOutput:\n\n" + _fence("True")

    admission._compute_code_rewards([completion], cases)

    assert graded == [IMPLEMENTATION]
