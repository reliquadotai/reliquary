"""The packaged code environment's cases must reach the grader as it reads them.

The sandbox itself is proven in production — the live 4B run grades through it
every window. What is new here is the relay: a packaged environment hands its
cases over `admission_reward_cases`, the batcher fills the older `code_cases`
channel from the environment's resource class rather than its name, and the
grader reads them. A shape the grader refuses is silent: every completion
scores zero, the group is degenerate, and nothing in the metrics says why.
"""

import importlib.util
import json

import pytest

# Detected, never imported: the packaged environment is loaded by a verifying
# loader, and importing it here first would make every later submodule import
# look unverified to that loader. A distribution present without the runtime
# it declares is the same "not installed" for this file's purpose.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("reliquary_code") is None
    or importlib.util.find_spec("verifiers") is None,
    reason="the packaged code environment is not installed",
)

from reliquary.environment.grader.server import GraderServer  # noqa: E402
from reliquary.environment.registry import get_environment_spec  # noqa: E402

ENVIRONMENT = "reliquary_code_v1"
PROMPTS = (12345, 777_777, 2_000_000)


@pytest.fixture(scope="module")
def environment():
    return get_environment_spec(ENVIRONMENT).create()


def _cases(environment, index: int) -> list[dict]:
    spec = get_environment_spec(ENVIRONMENT)
    problem = environment.get_problem(index)
    return list(getattr(environment, spec.reward_materializer_method)(problem))


def test_the_environment_asks_for_the_sandbox_by_class_not_by_name():
    spec = get_environment_spec(ENVIRONMENT)
    assert spec.admission_resource_class == "sandbox"
    assert spec.reward_materializer_method == "admission_reward_cases"


def test_every_relayed_case_is_one_the_grader_accepts(environment):
    for index in PROMPTS:
        cases = _cases(environment, index)
        assert cases, f"prompt {index} relayed nothing"
        for case in cases:
            assert GraderServer._valid_case(case), (
                f"prompt {index} relayed a case the grader refuses: {case}"
            )


def test_a_relayed_case_carries_one_callable_entry(environment):
    for index in PROMPTS:
        entries = {
            (case["entry"].get("name"), case["entry"].get("method"))
            for case in _cases(environment, index)
        }
        assert len(entries) == 1, f"prompt {index} names several entries: {entries}"


def test_the_cases_are_json_safe_across_the_wire(environment):
    # They travel to the executor as JSON; a value that does not survive the
    # round trip grades a correct answer wrong.
    for index in PROMPTS:
        cases = _cases(environment, index)
        assert json.loads(json.dumps(cases)) == cases


def _answer_from_cases(cases: list[dict]) -> str:
    """A solution that is right by construction, whatever the problem asks.

    It answers each relayed case with that case's own expected value, so the
    only thing under test is the path: which span the graders read, and whether
    the cases execute as they were relayed. Written without imports, because
    the environment's runner only admits a short allowlist of roots.
    """
    name = cases[0]["entry"]["name"]
    table = ", ".join(
        f"({case['args']!r}, {case['kwargs']!r}, {case['expected']!r})"
        for case in cases
    )
    return (
        "Here is my reasoning about the problem.\n\n"
        "```python\n"
        f"def {name}(*args, **kwargs):\n"
        f"    table = [{table}]\n"
        "    for case_args, case_kwargs, expected in table:\n"
        "        if list(args) == case_args and kwargs == case_kwargs:\n"
        "            return expected\n"
        "    return None\n"
        "```\n"
    )


def _run_cases(source: str, cases: list[dict]) -> float:
    namespace: dict = {}
    exec(compile(source, "<relayed>", "exec"), namespace)  # noqa: S102 - our own text
    entry = namespace[cases[0]["entry"]["name"]]
    passed = sum(
        entry(*case["args"], **case["kwargs"]) == case["expected"] for case in cases
    )
    return passed / len(cases)


def test_the_validator_reads_the_span_the_environment_grades(environment):
    from reliquary.validator.admission import _entry_function_name, _extract_python

    for index in PROMPTS:
        problem = environment.get_problem(index)
        cases = _cases(environment, index)
        completion = _answer_from_cases(cases)

        # The environment's own verdict, through its packaged runner.
        assert environment.compute_reward(problem, completion) == 1.0

        # The span the validator hands the sandbox, run against the same cases.
        extracted = _extract_python(
            completion, entry_name=_entry_function_name(cases)
        )
        assert extracted, f"prompt {index}: the validator extracted nothing"
        assert _run_cases(extracted, cases) == 1.0


def test_a_wrong_answer_is_zero_on_both_sides(environment):
    from reliquary.validator.admission import _entry_function_name, _extract_python

    index = PROMPTS[0]
    problem = environment.get_problem(index)
    cases = _cases(environment, index)
    name = cases[0]["entry"]["name"]
    completion = f"```python\ndef {name}(*args, **kwargs):\n    return None\n```\n"

    assert environment.compute_reward(problem, completion) == 0.0
    extracted = _extract_python(completion, entry_name=_entry_function_name(cases))
    assert _run_cases(extracted, cases) == 0.0
