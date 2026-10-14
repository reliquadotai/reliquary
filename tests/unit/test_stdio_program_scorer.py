"""Scoring a reliquary/stdio-program/v1 source: the wheel's tests, run here.

A fake wheel is installed the way the real one is (dist-info, RECORD, an
artifact pinning every file) with the real package's `extraction.py`,
`judge/compare.py` and `judge/guest.py` (tests/fixtures, pinned to the real
artifact by test_competitive_code_source). The program runs in the grading
service's workers; the comparison runs on this side with the wheel's own
`outputs_match`; the wheel's `grade` is never called.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from reliquary.environment.agentic.types import canonical_json
from reliquary.environment.registry import EnvironmentSpec

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "competitive_code"
NAME = "stdio_fake_v1"
PACKAGE = "stdio_fake_env"

_ENVIRONMENT = '''
class FakeEnvironment:
    name = "stdio_fake_v1"
    max_turns = 1
    validator_authoritative_reward = True

    TESTS = [("1 2\\n", "3\\n"), ("10 -4\\n", "6\\n"), ("0 0\\n", "0\\n")]

    def __init__(self, split="train"):
        self.split = split

    def __len__(self):
        return 3

    def task(self, index):
        use = "sft" if index < 2 else "rl"
        return {"id": f"t{index}", "prompt": f"Add two numbers ({index}).",
                "metadata": {"split": self.split, "use": use, "time_limit_s": 1.0}}

    def grade(self, index, completion):
        raise AssertionError("the wheel's grade ran model-written code on the host")

    def admission_reward_cases(self, index):
        return [{"stdin": i, "stdout": o, "time_limit_s": 1.0, "output_cap": 65536}
                for i, o in self.TESTS]
'''

SUM = "a, b = map(int, input().split())\nprint(a + b)\n"


def _install_fake_wheel(root: Path) -> EnvironmentSpec:
    files = {
        f"{PACKAGE}/__init__.py": "",
        f"{PACKAGE}/environment.py": _ENVIRONMENT,
        f"{PACKAGE}/extraction.py": (FIXTURES / "extraction.py").read_text(),
        f"{PACKAGE}/judge/__init__.py": "",
        f"{PACKAGE}/judge/compare.py": (FIXTURES / "compare.py").read_text(),
        f"{PACKAGE}/judge/guest.py": (FIXTURES / "guest.py").read_text(),
    }
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")
    artifact = {
        "schema": "reliquary/environment-artifact/v1",
        "environment": NAME,
        "contract": "reliquary/stdio-program/v1",
        "distribution": {"name": "reliquary-stdio-fake", "version": "1.0"},
        "entrypoints": {"taskset": f"{PACKAGE}:Taskset",
                        "replay": f"{PACKAGE}.environment:FakeEnvironment"},
        "source_manifest_sha256": "1" * 64,
        "files": {relative: hashlib.sha256(text.encode()).hexdigest()
                  for relative, text in files.items()},
    }
    (root / PACKAGE / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    dist_info = root / "reliquary_stdio_fake-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: reliquary-stdio-fake\nVersion: 1.0\n")
    (dist_info / "RECORD").write_text("".join(
        f"{path},,\n" for path in [*files, f"{PACKAGE}/artifact.json",
                                   f"{dist_info.name}/METADATA", f"{dist_info.name}/RECORD"]))
    return EnvironmentSpec(
        name=NAME,
        factory_path=f"{PACKAGE}.environment:FakeEnvironment",
        scorer_path="reliquary.environment.stdio_program:score_stdio_program",
        validator_authoritative_reward=True,
        admission_resource_class="sandbox",
        termination_policy="eos_or_cap",
        final_answer_policy="fenced_python",
        reward_lattice_policy="binary-v1",
        attainable_rewards=(0.0, 1.0),
        contract_version="reliquary/stdio-program/v1",
        reward_materializer_method="admission_reward_cases",
        environment_manifest_sha256=hashlib.sha256(
            canonical_json(artifact).encode()).hexdigest(),
        external_distribution="reliquary-stdio-fake",
        external_artifact_resource=f"{PACKAGE}/artifact.json",
    )


@pytest.fixture
def grader():
    from reliquary.environment.grader.server import GraderServer

    tmp = tempfile.TemporaryDirectory(prefix="g-", dir="/tmp")
    sock = os.path.join(tmp.name, "g.sock")
    server = GraderServer(
        socket_path=sock, pool_size=2,
        worker_argv=[sys.executable, "-m", "reliquary.environment.grader.worker"],
        eval_timeout_s=5.0, metrics_port=0,
        health_path=os.path.join(tmp.name, "health.json"),
    )
    server.start()
    deadline = time.time() + 5.0
    while not os.path.exists(sock) and time.time() < deadline:
        time.sleep(0.05)
    yield server
    server.stop()
    tmp.cleanup()


@pytest.fixture
def source(tmp_path, monkeypatch, grader):
    from reliquary.environment import registry, stdio_program
    from reliquary.environment.grader_client import GraderClient

    spec = _install_fake_wheel(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [n for n in sys.modules if n == PACKAGE or n.startswith(PACKAGE + ".")]:
        monkeypatch.delitem(sys.modules, name)
    importlib.invalidate_caches()
    monkeypatch.setattr(registry, "ENVIRONMENT_SPECS", {NAME: spec})
    monkeypatch.setattr(stdio_program, "_client",
                        lambda: GraderClient(socket_path=grader.socket_path))
    stdio_program._JUDGES.clear()
    return spec


def _score(spec, completion: str, index: int = 0) -> float:
    from reliquary.corpus.export import reward_scorer

    environment = spec.create()
    return reward_scorer(spec, environment)(environment.get_problem(index), completion)


def test_a_passing_program_scores_one_and_a_wrong_one_zero(source) -> None:
    assert _score(source, "Reasoning.\n```python\n" + SUM + "```") == 1.0
    assert _score(source, "```python\nprint(3)\n```") == 0.0  # passes test 1 only
    assert _score(source, "```python\nwhile True:\n    pass\n```") == 0.0
    assert _score(source, "```python\nimport os\n```") == 0.0


def test_the_wheels_own_extraction_and_comparison_decide(source) -> None:
    # Only the answer after </think> is searched, and "3.0" matches "3": the
    # package's rules, not this repository's.
    tolerant = "```python\na, b = map(int, input().split())\nprint(float(a + b))\n```"
    assert _score(source, "<think>```python\nprint(0)\n```</think>\n" + tolerant) == 1.0
    assert _score(source, "<think>```python\n" + SUM + "```</think>\nno code") == 0.0


def test_no_code_scores_zero_without_reaching_the_sandbox(source, monkeypatch) -> None:
    from reliquary.environment import stdio_program

    monkeypatch.setattr(stdio_program, "_client", lambda: pytest.fail("ran"))
    assert _score(source, "I would add them.") == 0.0


def test_harness_overload_is_never_a_reward(source, monkeypatch) -> None:
    from reliquary.environment import stdio_program
    from reliquary.environment.grader_client import GraderInfrastructureError

    class Overloaded:
        def run_stdio(self, **_):
            raise GraderInfrastructureError("harness_overload")

    monkeypatch.setattr(stdio_program, "_client", Overloaded)
    with pytest.raises(GraderInfrastructureError):
        _score(source, "```python\n" + SUM + "```")


@pytest.mark.parametrize("materials", [
    [], None, [{"stdin": "1 2\n"}],
    [{"stdin": "1", "stdout": "1", "time_limit_s": 0, "output_cap": 10}],
    [{"stdin": "1", "stdout": "1", "time_limit_s": 1.0, "output_cap": True}],
])
def test_malformed_materials_raise_rather_than_score(source, materials) -> None:
    from reliquary.environment.stdio_program import score_stdio_program

    problem = source.create().get_problem(0)
    with pytest.raises((TypeError, ValueError)):
        score_stdio_program(problem, ["```python\n" + SUM + "```"], materials)


def test_the_wheels_grade_is_refused_on_a_validator(source) -> None:
    environment = source.create()
    with pytest.raises(TypeError, match="sandbox"):
        environment.compute_reward(environment.get_problem(0), "```python\nprint(3)\n```")


def test_a_drifted_guest_is_refused_before_it_runs(source, tmp_path) -> None:
    from reliquary.environment import stdio_program

    guest = tmp_path / PACKAGE / "judge" / "guest.py"
    guest.write_text(guest.read_text() + "\n# drift\n")
    stdio_program._JUDGES.clear()
    with pytest.raises(ValueError, match="digest mismatch"):
        stdio_program.stdio_judge(source)
