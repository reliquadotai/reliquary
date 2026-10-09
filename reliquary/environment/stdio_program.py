"""Scoring for ``reliquary/stdio-program/v1``: a whole program on stdin tests.

The wheel supplies the tests (``admission_reward_cases``: stdin, expected
stdout, time limit, output cap) and three pieces of its judge, so a verdict
here is the package's verdict:

* ``extraction.extract_program`` picks the program out of the completion;
* ``judge/guest.py`` is the program's side of the sandbox. Its source is sent
  to the grading service, whose gVisor worker runs its ``run`` in a fork made
  for that one test;
* ``judge.compare.outputs_match`` compares, on this side: the sandbox never
  holds an expected output.

All or nothing, stopping at the first failure, as the package's ``judge``.
A starved host (``harness_overload``) or a failed service raises: never a 0.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

STDIO_PROGRAM_CONTRACT = "reliquary/stdio-program/v1"
# What the contract asks of the wheel's package besides its replay entrypoint.
_EXTRACTION = ("extraction", "extract_program")
_COMPARISON = ("judge.compare", "outputs_match")
_GUEST = "judge/guest.py"


@dataclass(frozen=True)
class StdioJudge:
    extract_program: Callable[[str], str | None]
    outputs_match: Callable[[str, str], bool]
    guest_source: str


@dataclass(frozen=True)
class StdioTest:
    stdin: str
    stdout: str
    time_limit_s: float
    output_cap: int


_JUDGES: dict[str, StdioJudge] = {}
_JUDGES_LOCK = threading.Lock()


def stdio_judge(spec: Any) -> StdioJudge:
    """The wheel's extraction, comparison and guest, from its verified bytes."""
    if spec.contract_version != STDIO_PROGRAM_CONTRACT:
        raise ValueError(f"{spec.name!r} is not a {STDIO_PROGRAM_CONTRACT} environment")
    digest = spec.environment_manifest_sha256
    with _JUDGES_LOCK:
        cached = _JUDGES.get(digest)
        if cached is not None:
            return cached
        import importlib.metadata

        from reliquary.environment.agentic.external import import_external_modules

        package = PurePosixPath(spec.external_artifact_resource).parts[0]
        artifact, (extraction, comparison) = import_external_modules(
            spec, [f"{package}.{_EXTRACTION[0]}", f"{package}.{_COMPARISON[0]}"])
        relative = f"{package}/{_GUEST}"
        pinned = artifact["files"].get(relative)
        if pinned is None:
            raise ValueError(f"{spec.name!r} pins no {relative}")
        located = Path(importlib.metadata.distribution(
            spec.external_distribution).locate_file(relative))
        body = located.read_bytes()
        if hashlib.sha256(body).hexdigest() != pinned:
            raise ValueError(f"external artifact file digest mismatch: {relative}")
        judge = StdioJudge(
            extract_program=getattr(extraction, _EXTRACTION[1]),
            outputs_match=getattr(comparison, _COMPARISON[1]),
            guest_source=body.decode("utf-8"),
        )
        _JUDGES[digest] = judge
        return judge


def _tests(materials: Any) -> list[StdioTest]:
    if not isinstance(materials, Sequence) or isinstance(materials, (str, bytes)):
        raise TypeError("stdio materials must be a list of tests")
    tests = []
    for case in materials:
        if not isinstance(case, dict):
            raise TypeError("each stdio test must be an object")
        stdin, stdout = case.get("stdin"), case.get("stdout")
        limit, cap = case.get("time_limit_s"), case.get("output_cap")
        if not isinstance(stdin, str) or not isinstance(stdout, str):
            raise TypeError("a stdio test's stdin and stdout must be strings")
        if (not isinstance(limit, (int, float)) or isinstance(limit, bool)
                or not math.isfinite(limit) or limit <= 0):
            raise ValueError("a stdio test's time limit must be positive")
        if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
            raise ValueError("a stdio test's output cap must be a positive integer")
        tests.append(StdioTest(stdin, stdout, float(limit), cap))
    if not tests:
        raise ValueError("a stdio problem needs at least one test")
    return tests


def _client():
    from reliquary.environment.grader_client import GraderClient

    return GraderClient()


def _passes(judge: StdioJudge, client: Callable[[], Any], tests: list[StdioTest],
            completion: str) -> bool:
    program = judge.extract_program(completion or "")
    if program is None:
        return False
    client = client()
    for test in tests:
        status, stdout = client.run_stdio(
            code=program, guest=judge.guest_source, stdin=test.stdin,
            output_cap=test.output_cap, time_limit_s=test.time_limit_s,
        )
        if status != "ok" or not judge.outputs_match(test.stdout, stdout):
            return False
    return True


def score_stdio_program(
    problem: dict[str, Any],
    completion_texts: list[str],
    reward_materials: Any = None,
) -> list[float]:
    """Module-level scorer: 1.0 when the program passes every test, else 0.0."""
    from reliquary.environment.registry import get_environment_spec

    spec = get_environment_spec(str(problem.get("environment", "")))
    tests = _tests(reward_materials)
    judge = stdio_judge(spec)
    return [1.0 if _passes(judge, _client, tests, text) else 0.0 for text in completion_texts]


__all__ = ["STDIO_PROGRAM_CONTRACT", "StdioJudge", "score_stdio_program", "stdio_judge"]
