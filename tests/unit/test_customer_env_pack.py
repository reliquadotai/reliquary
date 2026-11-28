import subprocess

import pytest

from reliquary.customer_env.package import read_package
from reliquary.customer_env.pack import pack, pull_from_prime
from tests.unit.test_customer_env_qualify import EXAMPLES_OK

LOCKED = f"numpy==2.0.0 \\\n    --hash=sha256:{'a' * 64}\n"


def fake_run(calls, *, pulled=None):
    def run(argv):
        calls.append(list(argv))
        if argv[:3] == ["prime", "env", "pull"]:
            target = argv[argv.index("-t") + 1]
            for name, text in (pulled or {}).items():
                path = __import__("pathlib").Path(target, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
        return subprocess.CompletedProcess(argv, 0, stdout=LOCKED, stderr="")
    return run


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "wordle"
    (src / "wordle").mkdir(parents=True)
    (src / "pyproject.toml").write_text('[project]\nname = "wordle"\nversion = "0.1"\n')
    (src / "wordle" / "__init__.py").write_text("")
    (src / ".git").mkdir()
    (src / ".git" / "config").write_text("secret")
    examples = tmp_path / "examples.jsonl"
    examples.write_text(EXAMPLES_OK)
    return src, examples


def test_pack_adds_the_manifest_and_lock_and_is_reproducible(source):
    src, examples = source
    calls = []
    a = pack(src, examples, taskset="wordle", args={"split": "test"}, run=fake_run(calls))
    b = pack(src, examples, taskset="wordle", args={"split": "test"}, run=fake_run([]))
    assert a == b
    pkg = read_package(a)
    assert (pkg.taskset, pkg.args, pkg.root) == ("wordle", {"split": "test"}, "wordle")
    assert not any(".git" in f for f in pkg.files)
    (compile_argv,) = calls
    assert compile_argv[:3] == ["uv", "pip", "compile"] and "--generate-hashes" in compile_argv
    assert "verifiers" in compile_argv and "--no-emit-package" in compile_argv


def test_pack_needs_a_taskset_when_the_project_does_not_name_one(source):
    src, examples = source
    with pytest.raises(ValueError, match="--taskset"):
        pack(src, examples, run=fake_run([]))


def test_pull_from_prime_finds_the_pulled_project(tmp_path):
    calls = []
    root = pull_from_prime("will/wordle@0.1.0", tmp_path, run=fake_run(calls, pulled={
        "wordle/pyproject.toml": "[project]\nname='w'\n", "wordle/sub/pyproject.toml": "x"}))
    assert root == tmp_path / "wordle"
    assert calls == [["prime", "env", "pull", "will/wordle@0.1.0", "-t", str(tmp_path)]]
    with pytest.raises(ValueError):
        pull_from_prime("wordle", tmp_path, run=fake_run([]))
