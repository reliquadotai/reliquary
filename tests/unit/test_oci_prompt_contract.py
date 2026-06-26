"""The code grader calls a function by name and checks its return value, but the
raw OpenCodeInstruct prompts are stdin/stdout-framed and rarely name the function.
get_problem appends the exact contract (function name + "return, don't print")
derived from the structured cases. This changes prompt tokens, so the release that
ships it must reach miners too (GRAIL binds the prompt).
"""


class _FakeDataset:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]


def _env(rows):
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
    env = OpenCodeInstructEnvironment.__new__(OpenCodeInstructEnvironment)
    env._dataset = _FakeDataset(rows)
    env._cases_by_id = {}
    return env


def _row():
    return {
        "input": "Reverse the list.",
        "structured_cases": [
            {"entry": {"kind": "function", "name": "reverse_list"},
             "args": [[1, 2, 3]], "kwargs": {}, "expected": [3, 2, 1], "compare": "exact"}
        ],
    }


def test_contract_appended_names_function_and_return():
    p = _env([_row()]).get_problem(0)["prompt"]
    assert p.startswith("Reverse the list.")
    assert "`reverse_list`" in p          # the exact name the grader will call
    assert "return" in p.lower()
    assert "stdin" in p.lower()


def test_no_contract_for_prompt_only_row():
    # No structured cases -> nothing to pin -> prompt untouched.
    env = _env([{"input": "Do it."}])
    assert env.get_problem(0)["prompt"] == "Do it."


def test_no_contract_for_non_function_entry():
    # A method-kind entry (or missing name) leaves the prompt untouched.
    row = {"input": "Do it.", "structured_cases": [
        {"entry": {"kind": "method", "class_name": "S", "method": "run"}, "args": [1]}]}
    assert _env([row]).get_problem(0)["prompt"] == "Do it."
