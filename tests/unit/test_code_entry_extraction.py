"""Extraction of the graded Python block by contract entry point.

Protocol v2-v4 grade ``matches[-1]`` — the last fenced block. That rule was
written when the environment ran a chat model that always closed with its final
implementation. Under the v5 reasoning prompt the model often closes with a
usage demo, an expected-output listing, or a test block instead, so the graded
span is not the implementation and the rollout scores zero despite being right.

From v5 on, the graded block is the last one that *defines the contract's entry
function*, falling back to ``matches[-1]`` when no block defines it.

The gate stops at v5 rather than applying everywhere so that v2-v4 stay
byte-exact as historical controls — their archived runs must stay reproducible.
That is the only thing it guards: Code rewards are validator-authoritative
(``validator_authoritative_reward = True``), so the validator overwrites the
miner's claim rather than comparing it, and no ``reward_mismatch`` can arise
from this change.
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
def v5(monkeypatch):
    """Pin the live protocol version; the entry rule applies from v5 on."""
    import reliquary.constants as constants

    monkeypatch.setattr(constants, "PROTOCOL_VERSION", 5, raising=False)


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

def test_extract_python_skips_a_trailing_usage_demo(v5):
    """The observed production failure: implementation first, `print(...)` last."""
    from reliquary.environment.opencodeinstruct import _extract_python

    text = (
        "Here is my reasoning.\n\n"
        + _fence(IMPLEMENTATION)
        + "\n\nExample usage:\n\n"
        + _fence('print(is_balanced_parentheses("()"))')
    )
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_extract_python_skips_a_trailing_expected_output_listing(v5):
    """A fenced block holding the program's *output*, which defines nothing."""
    from reliquary.environment.opencodeinstruct import _extract_python

    text = _fence(IMPLEMENTATION) + "\n\nThis will output:\n\n" + _fence("True\nTrue\nFalse")
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_extract_python_skips_a_trailing_test_block(v5):
    from reliquary.environment.opencodeinstruct import _extract_python

    text = _fence(IMPLEMENTATION) + "\n\nTests:\n\n" + _fence(
        'assert is_balanced_parentheses("()") is True'
    )
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_extract_python_keeps_the_last_block_that_defines_the_entry(v5):
    """Self-correction must still win: a later rewrite supersedes an earlier draft."""
    from reliquary.environment.opencodeinstruct import _extract_python

    draft = "def is_balanced_parentheses(s):\n    return False"
    text = _fence(draft) + "\n\nActually, that is wrong. Corrected:\n\n" + _fence(IMPLEMENTATION)
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_extract_python_falls_back_to_last_block_when_none_defines_the_entry(v5):
    """No block defines the contract function — behave exactly as v5 did."""
    from reliquary.environment.opencodeinstruct import _extract_python

    text = _fence("helper = 1") + "\n\nand:\n\n" + _fence("other = 2")
    assert _extract_python(text, entry_name=ENTRY) == "other = 2"


def test_extract_python_single_block_is_unaffected_by_the_entry_name(v5):
    """83% of rollouts emit exactly one block; the rule must not touch them."""
    from reliquary.environment.opencodeinstruct import _extract_python

    text = "Reasoning first.\n\n" + _fence(IMPLEMENTATION)
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION
    assert _extract_python(text, entry_name=None) == IMPLEMENTATION


def test_extract_python_without_an_entry_name_keeps_last_block_wins(v5):
    """A method-entry problem pins no function name; the old rule still applies."""
    from reliquary.environment.opencodeinstruct import _extract_python

    text = _fence(IMPLEMENTATION) + "\n\n" + _fence("print('demo')")
    assert _extract_python(text, entry_name=None) == "print('demo')"


def test_extract_python_requires_a_definition_not_a_mention(v5):
    """A block merely *calling* the function must not be mistaken for the impl."""
    from reliquary.environment.opencodeinstruct import _extract_python

    text = _fence(IMPLEMENTATION) + "\n\n" + _fence(
        "result = is_balanced_parentheses('()')"
    )
    assert _extract_python(text, entry_name=ENTRY) == IMPLEMENTATION


def test_extract_python_grades_nothing_when_the_completion_has_no_fence(v5):
    """From v5 on there is a single answer channel: what is between the fences, nothing else.

    The raw-completion fallback fired 762 times across 30 768 production
    rollouts and never once produced a positive reward — a rollout that contains
    code always fences it, so the fallback only ever ran `exec` on prose. Its
    zeros were deserved (no code was produced) and stay zeros; what goes away is
    executing model prose as Python.
    """
    from reliquary.environment.opencodeinstruct import _extract_python

    assert _extract_python("Step 1: initialise a counter.", entry_name=ENTRY) == ""
    assert _extract_python(IMPLEMENTATION, entry_name=ENTRY) == ""


def test_extract_python_empty_completion_is_empty(v5):
    from reliquary.environment.opencodeinstruct import _extract_python

    assert _extract_python("", entry_name=ENTRY) == ""


# ---------------------------------------------------------------------------
# v2-v4 stay byte-exact: they are the historical controls.
# ---------------------------------------------------------------------------

def test_extract_python_ignores_the_entry_name_below_protocol_v5(monkeypatch):
    """v4 is the immutable no-cue control; its rewards must stay reproducible."""
    import reliquary.constants as constants
    from reliquary.environment.opencodeinstruct import _extract_python

    monkeypatch.setattr(constants, "PROTOCOL_VERSION", 4, raising=False)
    text = _fence(IMPLEMENTATION) + "\n\n" + _fence("True\nTrue\nFalse")
    assert _extract_python(text, entry_name=ENTRY) == "True\nTrue\nFalse"


def test_extract_python_keeps_the_raw_fallback_below_protocol_v5(monkeypatch):
    """v2-v4 keep the raw fallback so their historical rewards stay reproducible."""
    import reliquary.constants as constants
    from reliquary.environment.opencodeinstruct import _extract_python

    monkeypatch.setattr(constants, "PROTOCOL_VERSION", 4, raising=False)
    assert _extract_python(IMPLEMENTATION, entry_name=ENTRY) == IMPLEMENTATION


# ---------------------------------------------------------------------------
# compute_reward wires the contract through to the extractor.
# ---------------------------------------------------------------------------

def test_compute_reward_grades_the_implementation_not_the_trailing_demo(v5, monkeypatch):
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

def test_admission_grades_the_implementation_not_the_trailing_demo(v5, monkeypatch):
    """Admission's reward is the one written into the batch. If it graded a
    different span than compute_reward, one rollout would be scored two ways
    inside a single window."""
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
