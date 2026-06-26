"""Tests for the sandbox worker's call-only eval logic."""


def test_call_function_returns_primitive_output():
    from reliquary.environment.grader.worker import evaluate_call

    output, status = evaluate_call(
        "def add(a, b): return a + b",
        {"kind": "function", "name": "add"},
        [1, 2],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 3


def test_call_method_entrypoint():
    from reliquary.environment.grader.worker import evaluate_call

    code = "class Solution:\n    def inc(self, x): return x + 1"
    output, status = evaluate_call(
        code,
        {"kind": "method", "class_name": "Solution", "method": "inc"},
        [4],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 5


def test_import_math_allowed():
    from reliquary.environment.grader.worker import evaluate_call

    code = "import math\ndef f(x): return math.sqrt(x)"
    output, status = evaluate_call(
        code,
        {"kind": "function", "name": "f"},
        [9],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 3.0


def test_forbidden_import_is_reported():
    from reliquary.environment.grader.worker import evaluate_call

    output, status = evaluate_call(
        "import os\ndef f(): return True",
        {"kind": "function", "name": "f"},
        [],
        {},
        timeout_s=5.0,
    )
    assert status == "forbidden_import"
    assert output is None


def test_custom_object_output_is_rejected():
    from reliquary.environment.grader.worker import evaluate_call

    code = """
class AlwaysEqual:
    def __eq__(self, other): return True
def f():
    return AlwaysEqual()
"""
    output, status = evaluate_call(
        code,
        {"kind": "function", "name": "f"},
        [],
        {},
        timeout_s=5.0,
    )
    assert status == "bad_output"
    assert output is None


def test_print_does_not_leak_to_stdout(capsys):
    from reliquary.environment.grader.worker import evaluate_call

    output, status = evaluate_call(
        'print("malicious noise")\ndef f(): return 7',
        {"kind": "function", "name": "f"},
        [],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 7
    captured = capsys.readouterr()
    assert "malicious noise" not in captured.out
    assert "malicious noise" not in captured.err


def test_runtime_exception_is_not_successful_none():
    from reliquary.environment.grader.worker import evaluate_call

    output, status = evaluate_call(
        "def f():\n    raise RuntimeError('boom')",
        {"kind": "function", "name": "f"},
        [],
        {},
        timeout_s=5.0,
    )
    assert status == "runtime_error"
    assert output is None


def test_call_function_resolves_when_name_differs_single_def():
    """Prompt asks for a behavior, not a name: a correct sole function under a
    different name must still be graded."""
    from reliquary.environment.grader.worker import evaluate_call

    output, status = evaluate_call(
        "def my_add(a, b): return a + b",
        {"kind": "function", "name": "add"},
        [2, 3],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 5


def test_call_method_resolves_when_class_name_differs():
    from reliquary.environment.grader.worker import evaluate_call

    code = "class Impl:\n    def run(self, x): return x * 2"
    output, status = evaluate_call(
        code,
        {"kind": "method", "class_name": "Solution", "method": "run"},
        [4],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 8


def test_call_disambiguates_multiple_functions_by_arity():
    """With a helper of different arity, the matching entry is still found."""
    from reliquary.environment.grader.worker import evaluate_call

    code = "def helper(x): return x + 1\ndef solve(a, b): return a * b"
    output, status = evaluate_call(
        code,
        {"kind": "function", "name": "main"},
        [3, 4],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 12


def test_call_picks_call_graph_root_when_arity_ties():
    """Helper + entry of the same arity: the one not called by the other wins."""
    from reliquary.environment.grader.worker import evaluate_call

    code = "def _twice(n): return n * 2\ndef compute(n): return _twice(n) + 1"
    output, status = evaluate_call(
        code,
        {"kind": "function", "name": "missing"},
        [5],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 11  # compute(5), not _twice(5)


def test_call_picks_last_defined_when_independent():
    """Truly independent same-arity functions: take the last (deterministic)."""
    from reliquary.environment.grader.worker import evaluate_call

    code = "def f(a, b): return a + b\ndef g(a, b): return a * b"
    output, status = evaluate_call(
        code,
        {"kind": "function", "name": "missing"},
        [2, 3],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 6  # g, the last-defined


def test_call_fails_when_no_function_matches_arity():
    """No defined function can accept the call: genuinely uncallable -> fail."""
    from reliquary.environment.grader.worker import evaluate_call

    code = "def f(a, b, c): return a + b + c"
    output, status = evaluate_call(
        code,
        {"kind": "function", "name": "missing"},
        [1, 2],
        {},
        timeout_s=5.0,
    )
    assert status != "ok"
    assert output is None


def test_call_skips_print_only_helper_defined_last():
    """A trailing print/format helper must not shadow the value-returning solution.

    When the requested entry name is absent and a print-only helper is defined
    after the real function, the old last-defined tie-break picked the helper
    and crashed on the case args. The value-returning function must win.
    """
    from reliquary.environment.grader.worker import evaluate_call

    code = (
        "def generate_spiral_matrix(n):\n"
        "    return [[1]] if n == 1 else [[0] * n for _ in range(n)]\n"
        "def print_spiral_matrix(grid):\n"
        "    for row in grid:\n"
        "        print(row)\n"
    )
    output, status = evaluate_call(
        code,
        {"kind": "function", "name": "generate_spiral"},
        [1],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == [[1]]  # the value-returning solution, not the printer


def test_call_ignores_nested_return_inside_print_only_wrapper():
    """A nested helper's return does not make the outer printer value-returning."""
    from reliquary.environment.grader.worker import evaluate_call

    code = (
        "def solve(n):\n"
        "    return n + 10\n"
        "def display(n):\n"
        "    def inner(x):\n"
        "        return x + 1\n"
        "    print(n)\n"
    )
    output, status = evaluate_call(
        code,
        {"kind": "function", "name": "missing"},
        [1],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 11


def test_compile_tamper_fails_without_passing():
    from reliquary.environment.grader.worker import evaluate_call

    code = """
import builtins
def f(): return 1
"""
    output, status = evaluate_call(
        code,
        {"kind": "function", "name": "f"},
        [],
        {},
        timeout_s=5.0,
    )
    assert status == "forbidden_import"
    assert output is None
