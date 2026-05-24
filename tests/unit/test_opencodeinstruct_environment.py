"""Tests for the OpenCodeInstruct environment.

Helpers (extraction, completion parsing) are pure-Python and tested
without the dataset or the grader. Grader-dependent tests use a fake
grader client. The HF dataset is exercised in the smoke test.
"""

import pytest


# ---------------------------------------------------------------------------
# _extract_python: pulls Python code out of model completions.
# ---------------------------------------------------------------------------

def test_extract_python_from_fenced_block():
    from reliquary.environment.opencodeinstruct import _extract_python
    text = "Sure, here is the code:\n```python\ndef f(x):\n    return x + 1\n```\nDone."
    assert _extract_python(text) == "def f(x):\n    return x + 1"


def test_extract_python_from_unmarked_fenced_block():
    """Fence without language tag still works."""
    from reliquary.environment.opencodeinstruct import _extract_python
    text = "```\ndef g():\n    return 42\n```"
    assert _extract_python(text) == "def g():\n    return 42"


def test_extract_python_last_block_wins():
    """If the model emits multiple code blocks, prefer the last (the final answer)."""
    from reliquary.environment.opencodeinstruct import _extract_python
    text = "```python\nfirst = 1\n```\nThen revised to:\n```python\nsecond = 2\n```"
    assert _extract_python(text) == "second = 2"


def test_extract_python_fallback_to_raw():
    """When no fence at all, return the full string — let exec decide."""
    from reliquary.environment.opencodeinstruct import _extract_python
    text = "def h():\n    return 'no fence'"
    assert _extract_python(text) == text


def test_extract_python_empty_string():
    from reliquary.environment.opencodeinstruct import _extract_python
    assert _extract_python("") == ""


def test_extract_python_handles_tilde_fences():
    """Some models emit ~~~ instead of ```. Accept both."""
    from reliquary.environment.opencodeinstruct import _extract_python
    text = "~~~python\nx = 1\n~~~"
    assert _extract_python(text) == "x = 1"


def test_extract_python_handles_python3_tag():
    """Models often emit ```python3 — should be accepted."""
    from reliquary.environment.opencodeinstruct import _extract_python
    text = "```python3\nprint(42)\n```"
    assert _extract_python(text) == "print(42)"


def test_extract_python_handles_py_short_tag():
    from reliquary.environment.opencodeinstruct import _extract_python
    text = "```py\nx = 1\n```"
    assert _extract_python(text) == "x = 1"


def test_extract_python_rejects_mismatched_fence_styles():
    """Opener ``` should not pair with closer ~~~ — falls back to raw."""
    from reliquary.environment.opencodeinstruct import _extract_python
    text = "```python\ncode_here\n~~~"
    # No valid fenced block extractable → return the full string
    assert _extract_python(text) == text


# ---------------------------------------------------------------------------
# OpenCodeInstructEnvironment — exercised with a stub dataset and a
# fake grader client. The real HF dataset and grader are covered by
# the integration smoke tests.
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Mimics the subset of HF datasets API the env touches."""
    def __init__(self, rows):
        self._rows = rows
    def __len__(self):
        return len(self._rows)
    def __getitem__(self, i):
        return self._rows[i]


class _FakeGraderClient:
    def __init__(self, response: float):
        self.response = response
        self.calls = []
    def evaluate(self, code, tests, timeout_s):
        self.calls.append((code, tests, timeout_s))
        return self.response


def _env_with(dataset_rows, grader_response=1.0):
    """Construct an env with fake parts, bypassing __init__ to avoid the
    HF dataset download. NOTE: if the class's _dataset / _grader attribute
    names change, these tests will silently pass with a broken env — keep
    this helper in sync with __init__."""
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
    env = OpenCodeInstructEnvironment.__new__(OpenCodeInstructEnvironment)
    env._dataset = _FakeDataset(dataset_rows)
    env._grader = _FakeGraderClient(grader_response)
    return env


def test_get_problem_shape():
    rows = [{
        "input": "Write a function add(a, b) returning their sum.",
        "unit_tests_parsed": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
    }]
    env = _env_with(rows)
    p = env.get_problem(0)
    assert p["prompt"] == "Write a function add(a, b) returning their sum."
    assert isinstance(p["ground_truth"], str)
    import json as _json
    assert _json.loads(p["ground_truth"]) == ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    assert len(p["id"]) == 16


def test_get_problem_id_is_deterministic():
    rows = [{"input": "Same prompt", "unit_tests_parsed": ["assert True"]}]
    env = _env_with(rows)
    assert env.get_problem(0)["id"] == env.get_problem(0)["id"]


def test_get_problem_modulo_wrap():
    rows = [
        {"input": "p0", "unit_tests_parsed": ["assert True"]},
        {"input": "p1", "unit_tests_parsed": ["assert True"]},
    ]
    env = _env_with(rows)
    assert env.get_problem(0)["prompt"] == "p0"
    assert env.get_problem(2)["prompt"] == "p0"  # wrap


def test_compute_reward_delegates_to_grader():
    rows = [{"input": "...", "unit_tests_parsed": ["assert f() == 1"]}]
    env = _env_with(rows, grader_response=0.6)
    p = env.get_problem(0)
    completion = "```python\ndef f(): return 1\n```"
    r = env.compute_reward(p, completion)
    assert r == 0.6
    assert env._grader.calls[0][0] == "def f(): return 1"
    assert env._grader.calls[0][1] == ["assert f() == 1"]


def test_compute_reward_never_raises_on_garbled_problem():
    rows = [{"input": "x", "unit_tests_parsed": ["assert True"]}]
    env = _env_with(rows, grader_response=0.0)
    r = env.compute_reward({"ground_truth": "not-json"}, "any completion")
    assert r == 0.0


def test_environment_name_constant():
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
    assert OpenCodeInstructEnvironment.name == "opencodeinstruct"


def test_compute_reward_returns_zero_for_valid_json_non_list_ground_truth():
    """Ground truth that JSON-parses to a non-list (e.g. number, dict)
    must score 0 without raising."""
    rows = [{"input": "x", "unit_tests_parsed": ["assert True"]}]
    env = _env_with(rows, grader_response=0.0)
    assert env.compute_reward({"ground_truth": "42"}, "any completion") == 0.0
    assert env.compute_reward({"ground_truth": '{"key": "val"}'}, "any completion") == 0.0


def test_compute_reward_handles_none_completion():
    """compute_reward accepts None as completion (`completion or ''` guard)."""
    rows = [{"input": "x", "unit_tests_parsed": ["assert True"]}]
    env = _env_with(rows, grader_response=0.0)
    p = env.get_problem(0)
    # Should not raise; returns whatever the grader returns for empty code.
    r = env.compute_reward(p, None)
    assert r == 0.0


def test_load_environment_factory_recognizes_opencodeinstruct(monkeypatch):
    """load_environment('opencodeinstruct') returns the class without
    actually downloading the dataset (we monkeypatch __init__)."""
    from reliquary.environment import load_environment
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment

    monkeypatch.setattr(OpenCodeInstructEnvironment, "__init__", lambda self: None)
    env = load_environment("opencodeinstruct")
    assert isinstance(env, OpenCodeInstructEnvironment)


def test_load_environment_unknown_still_raises():
    from reliquary.environment import load_environment
    with pytest.raises(ValueError, match="Unknown environment"):
        load_environment("doesnotexist")


def test_load_environments_returns_dict(monkeypatch):
    """load_environments(names) returns {name: Environment} for each name."""
    from reliquary.environment import load_environments
    from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment

    monkeypatch.setattr(OpenMathInstructEnvironment, "__init__", lambda self: None)
    monkeypatch.setattr(OpenCodeInstructEnvironment, "__init__", lambda self: None)

    envs = load_environments(["openmathinstruct", "opencodeinstruct"])
    assert set(envs.keys()) == {"openmathinstruct", "opencodeinstruct"}
    assert isinstance(envs["openmathinstruct"], OpenMathInstructEnvironment)
    assert isinstance(envs["opencodeinstruct"], OpenCodeInstructEnvironment)


def test_load_environments_unknown_name_raises():
    from reliquary.environment import load_environments
    with pytest.raises(ValueError, match="Unknown environment"):
        load_environments(["openmathinstruct", "nope"])
