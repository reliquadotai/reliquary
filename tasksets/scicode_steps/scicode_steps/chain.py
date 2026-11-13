"""The bookkeeping of a SciCode chain: one eval set per position, each built
from the answers to the positions before it, and the two published rates.

    previous = {}
    for step in range(1, max_position(split) + 1):
        set = build-set verifiers:scicode-steps {split, step, previous}
        answers = <the order's graded rows: step_number -> answer text, passed>
        previous = record(previous, answers)
    summary(split, passed)

The answer text is what the grader reads (after the reasoning block).
"""

from __future__ import annotations

from collections.abc import Mapping

from scicode_steps.taskset import (
    DATASET_REVISION, GIVEN_STEPS, PROBLEMS, SCORED_STEPS, extract_python_script, read_problems, scored_steps,
)

__all__ = ["max_position", "record", "summary"]


def max_position(split: str, revision: str = DATASET_REVISION) -> int:
    return max(len(p["sub_steps"]) for p in read_problems(split, revision))


def record(previous: Mapping[str, str], answers: Mapping[str, str]) -> dict[str, str]:
    """``previous`` with each answered step's code added (an unanswered step is
    passed in as "")."""
    out = dict(previous)
    for number, answer in answers.items():
        if number in GIVEN_STEPS:
            raise ValueError(f"step {number} is given as code, never asked")
        out[number] = extract_python_script(answer or "")
    return out


def summary(split: str, passed: Mapping[str, bool], revision: str = DATASET_REVISION,
            problems: list[dict] | None = None) -> dict:
    """Main-problem resolve rate (every scored step passed) and sub-step pass
    rate, over upstream's denominators; a step with no verdict counts as failed."""
    if problems is None:
        problems = read_problems(split, revision)
        counts = (len(problems), sum(len(scored_steps(p)) for p in problems))
        if counts != (PROBLEMS[split], SCORED_STEPS[split]):
            raise ValueError(f"{split} holds {counts}, not upstream's denominators")
    scored = {n for p in problems for n in scored_steps(p)}
    unknown = sorted(set(passed) - scored)
    if unknown:
        raise ValueError(f"steps outside {split}: {unknown[:5]}")
    solved = steps_passed = 0
    for problem in problems:
        ok = [bool(passed.get(number, False)) for number in scored_steps(problem)]
        steps_passed += sum(ok)
        solved += all(ok)
    return {"split": split, "problems": len(problems), "steps": len(scored),
            "problems_solved": solved, "steps_passed": steps_passed,
            "main_problem_resolve_rate": solved / len(problems),
            "subproblem_pass_rate": steps_passed / len(scored)}
