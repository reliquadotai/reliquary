import json
import os
import subprocess
import sys

import pytest

from reliquary.validator.agentic_replay import TOOL_PROGRAM


def _run(tmp_path, tool, arguments):
    proc = subprocess.run([sys.executable, "-c", TOOL_PROGRAM],
                          input=json.dumps({"tool": tool, "arguments": arguments, "timeout": 30}),
                          capture_output=True, text=True, cwd=tmp_path)
    return proc.stdout


def test_bash_returns_stdout_then_stderr(tmp_path):
    assert _run(tmp_path, "bash", json.dumps({"command": "echo out; echo err >&2"})) == "out\nerr\n"


def test_edit_replaces_exactly_once(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\n")
    assert _run(tmp_path, "edit", json.dumps({"path": "f.py", "old_str": "1", "new_str": "2"})) == "Edited f.py"
    assert (tmp_path / "f.py").read_text() == "a = 2\n"


def test_edit_errors_match_verifiers(tmp_path):
    (tmp_path / "f.py").write_text("x x\n")
    assert _run(tmp_path, "edit", json.dumps({"path": "f.py", "old_str": "x", "new_str": "y"})) == \
        "error: old_str must appear exactly once in f.py (found 2)"


def test_invalid_json_arguments_reproduce_the_harness_message(tmp_path):
    out = _run(tmp_path, "bash", "{not json")
    assert out.startswith("error: invalid JSON in tool arguments (")


def test_non_object_arguments(tmp_path):
    assert _run(tmp_path, "bash", "[]") == \
        "error: tool arguments must be a JSON object, got list; resend as an object"


def test_unknown_tool(tmp_path):
    assert _run(tmp_path, "search", "{}") == "error: unknown tool 'search'"


def test_parity_with_pinned_verifiers(tmp_path):
    program = pytest.importorskip("verifiers.v1.harnesses.bash.program")
    (tmp_path / "g.py").write_text("k = 0\n")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        want = program.run_edit("g.py", "0", "1")
    finally:
        os.chdir(cwd)
    (tmp_path / "g.py").write_text("k = 0\n")
    assert _run(tmp_path, "edit", json.dumps({"path": "g.py", "old_str": "0", "new_str": "1"})) == want
