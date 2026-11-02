"""Graders for a customer's eval set: each line's ``reference`` against the model's answer.

A catalog environment grades its own rows by index; a customer's rows carry their reference instead, so they are
graded here, on the admin host, by a grader of this table. The answer is what follows the reasoning block
(``eval.grading.answer_text``), as for every other set.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

MAX_REFERENCE_CHARS = 4096


def _boxed_math(reference: str, answer: str) -> float:
    from reliquary.environment.openmathinstruct import _compute_omi_reward

    return float(_compute_omi_reward({"ground_truth": reference}, answer))


def _exact_match(reference: str, answer: str) -> float:
    return 1.0 if answer.strip() == reference.strip() else 0.0


@dataclass(frozen=True, slots=True)
class ReferenceGrader:
    name: str
    # What ``answer_text`` reads of a completion: "boxed" and "text" both take the part after the reasoning.
    final_answer_policy: str
    score: Callable[[str, str], float]
    description: str


REFERENCE_GRADERS: dict[str, ReferenceGrader] = {g.name: g for g in (
    ReferenceGrader("boxed_math_v1", "boxed", _boxed_math,
                    "the last \\boxed{} of the answer equals the reference (numbers, fractions, LaTeX values)"),
    ReferenceGrader("exact_match_v1", "text", _exact_match,
                    "the answer, stripped, equals the reference, stripped"),
)}


def check_reference(grader: str, reference) -> str:
    """The line's reference for ``grader``, or ValueError."""
    if grader not in REFERENCE_GRADERS:
        raise ValueError(f"grader_unknown: {grader!r}; graders: {sorted(REFERENCE_GRADERS)}")
    if not isinstance(reference, str) or not reference.strip() or len(reference) > MAX_REFERENCE_CHARS:
        raise ValueError(f"a reference is a non-empty string of at most {MAX_REFERENCE_CHARS} characters")
    return reference


def grade_reference(grader: str, reference: str, answer: str) -> float:
    return float(REFERENCE_GRADERS[grader].score(reference, answer))


__all__ = ["MAX_REFERENCE_CHARS", "REFERENCE_GRADERS", "ReferenceGrader", "check_reference", "grade_reference"]
