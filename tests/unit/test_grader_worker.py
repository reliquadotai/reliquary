"""Tests for the grader worker eval logic.

The worker can be tested in pure Python (no sandbox) by invoking its
eval function directly. The real worker reads (code, tests) from
stdin and writes the result to stdout; here we call the pure
function it wraps.
"""

import pytest


def test_eval_all_tests_pass():
    from reliquary.environment.grader.worker import evaluate_code

    code = "def add(a, b): return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]
    passed, total, status = evaluate_code(code, tests, timeout_s=5.0)
    assert passed == 3
    assert total == 3
    assert status == "ok"


def test_eval_some_tests_fail():
    from reliquary.environment.grader.worker import evaluate_code

    code = "def add(a, b): return a - b"  # wrong sign
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(5, 5) == 10"]
    passed, total, status = evaluate_code(code, tests, timeout_s=5.0)
    assert passed == 1  # only add(0, 0) == 0 happens to match
    assert total == 3
    assert status == "ok"


def test_eval_test_namespace_isolation():
    """A test that mutates the namespace must not corrupt subsequent tests."""
    from reliquary.environment.grader.worker import evaluate_code

    code = "x = 10"
    tests = [
        "x = 999",                       # mutates local copy
        "assert x == 10",                # should still see original
    ]
    passed, total, _ = evaluate_code(code, tests, timeout_s=5.0)
    assert passed == 2  # both pass — first one always passes (no assert), second sees x=10


def test_eval_syntax_error_in_one_test_does_not_break_others():
    from reliquary.environment.grader.worker import evaluate_code

    code = "def f(): return 1"
    tests = [
        "assert f() == 1",
        "this is not valid python",
        "assert f() == 1",
    ]
    passed, total, _ = evaluate_code(code, tests, timeout_s=5.0)
    assert passed == 2
    assert total == 3


def test_eval_code_with_syntax_error_returns_zero():
    from reliquary.environment.grader.worker import evaluate_code

    code = "def broken("  # syntax error
    tests = ["assert True"]
    passed, total, _ = evaluate_code(code, tests, timeout_s=5.0)
    assert passed == 0
    assert total == 1


def test_eval_empty_code_returns_zero():
    from reliquary.environment.grader.worker import evaluate_code

    passed, total, _ = evaluate_code("", ["assert True"], timeout_s=5.0)
    assert passed == 0
    assert total == 1


def test_eval_empty_tests_returns_zero_total():
    from reliquary.environment.grader.worker import evaluate_code

    passed, total, _ = evaluate_code("x = 1", [], timeout_s=5.0)
    assert passed == 0
    assert total == 0


def test_eval_runtime_error_in_code_returns_zero():
    """Code that raises at exec time → all tests fail (ns is empty)."""
    from reliquary.environment.grader.worker import evaluate_code

    code = "raise RuntimeError('boom')"
    tests = ["assert True", "assert 1 == 1"]
    passed, total, _ = evaluate_code(code, tests, timeout_s=5.0)
    assert passed == 0
    assert total == 2


def test_eval_print_does_not_leak_to_stdout(capsys):
    """Miner code's print() must not pollute the IPC pipe."""
    from reliquary.environment.grader.worker import evaluate_code

    code = 'print("malicious noise on stdout")\ndef f(): return 7'
    passed, total, _ = evaluate_code(code, ["assert f() == 7"], timeout_s=5.0)
    assert passed == 1
    assert total == 1
    # Verify nothing leaked to the real stdout/stderr — the redirect
    # captured "malicious noise on stdout" into the in-memory sink.
    captured = capsys.readouterr()
    assert "malicious noise" not in captured.out
    assert "malicious noise" not in captured.err


def test_eval_mutating_a_list_in_one_test_does_not_leak_to_next():
    """Mutable namespace values must be deep-copied per test."""
    from reliquary.environment.grader.worker import evaluate_code

    code = "shared = [1, 2, 3]"
    tests = [
        "shared.append(999)",         # mutates this test's copy
        "assert shared == [1, 2, 3]", # next test must see original
    ]
    passed, total, _ = evaluate_code(code, tests, timeout_s=5.0)
    assert passed == 2  # both pass — second sees pristine [1,2,3]
