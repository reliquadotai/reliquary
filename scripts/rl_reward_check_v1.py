#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline reward checks for a JSON-answer task; no model or training is run.

Run with Python 3.10+: python3 scripts/rl_reward_check_v1.py
Task: remove duplicate integers while preserving their first occurrence.
Answers must be JSON arrays of integers. Invalid answers receive reward 0;
invalid task inputs or checker errors fail the check rather than becoming 0.
The hand-authored fixtures are examples, not a dataset or model benchmark.
"""

import json
from statistics import pvariance


def reward(task, response):
    """A task-specific 0/1 grader, not a drop-in training API adapter."""
    if not isinstance(task, list) or any(type(value) is not int for value in task):
        raise ValueError("Task input must be a JSON array of integers.")
    if not isinstance(response, str):
        raise TypeError("The candidate response must be text.")
    try:
        answer = json.loads(response)
    except json.JSONDecodeError:
        return 0
    if not isinstance(answer, list) or any(type(value) is not int for value in answer):
        return 0
    return int(answer == list(dict.fromkeys(task)))


CHECKS = [
    ("preserve-order", [3, 1, 3], "[3, 1]", 1),
    ("sorting-is-wrong", [3, 1, 3], "[1, 3]", 0),
    ("duplicates-remain", [3, 1, 3], "[3, 1, 3]", 0),
    ("missing-value", [3, 1, 3], "[3]", 0),
    ("empty-input", [], "[]", 1),
    ("invented-value", [], "[0]", 0),
    ("boolean-is-not-integer", [1], "[true]", 0),
    ("invalid-json", [3, 1, 3], "not-json", 0),
]

GROUPS = [
    ("all_fail", ["[1, 3]", "[3, 1, 3]", "[3]", "[]"]),
    ("mixed", ["[3, 1]", "[1, 3]", "[3, 1]", "[3]"]),
    ("all_pass", ["[3, 1]", "[3, 1]", "[3, 1]", "[3, 1]"]),
]


def run_checks(grader=reward):
    checks = []
    for name, task, response, expected in CHECKS:
        observed = grader(task, response)
        if type(observed) not in (int, float) or observed not in (0, 1):
            raise ValueError("This example requires binary numeric rewards.")
        checks.append({"case": name, "expected": expected, "observed": observed,
                       "passed": observed == expected})
    groups = []
    for name, responses in GROUPS:
        rewards = [grader([3, 1, 3], response) for response in responses]
        if any(type(value) not in (int, float) or value not in (0, 1) for value in rewards):
            raise ValueError("This example requires binary numeric rewards.")
        groups.append({"group": name, "rewards": rewards,
                       "population_variance": pvariance(rewards),
                       "reward_contrast": min(rewards) != max(rewards)})
    return {
        "version": 1,
        "scope": "Hand-authored offline fixtures; no model inference or training.",
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "groups": groups,
    }


def main(grader=reward):
    report = run_checks(grader)
    print(json.dumps(report, indent=2))
    return 0 if report["checks_passed"] == report["checks_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
