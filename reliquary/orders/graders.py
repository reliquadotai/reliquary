"""Graders for a customer's eval set: each line's ``reference`` against the model's answer.

A catalog environment grades its own rows by index; a customer's rows carry their reference instead, so they are
graded here, on the admin host, by a grader of this table. The answer is what follows the reasoning block
(``eval.grading.answer_text``), as for every other set.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

MAX_REFERENCE_CHARS = 4096
# An answer past this scores 0 unread: bounds what a grader parses on the admin host.
MAX_ANSWER_CHARS = 100_000
# numeric_v1 without an explicit tolerance: |answer - reference| <= this x max(1, |reference|).
NUMERIC_RELATIVE_TOLERANCE = Fraction(1, 10**6)
# A longer number, or a larger exponent, is not read: 1e999999999 would be computed exactly.
MAX_NUMBER_CHARS = 100
MAX_NUMBER_EXPONENT = 400


def _boxed_math(reference: str, answer: str) -> float:
    from reliquary.environment.openmathinstruct import _compute_omi_reward

    return float(_compute_omi_reward({"ground_truth": reference}, answer))


def _exact_match(reference: str, answer: str) -> float:
    return 1.0 if answer.strip() == reference.strip() else 0.0


_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:/\d+)?"
_NUMBER_RE = re.compile(rf"(?<![\w.]){_NUMBER}(?![\w])")
_WHOLE_NUMBER_RE = re.compile(rf"\A{_NUMBER}\Z")
_FRAC_RE = re.compile(r"\A([-+]?)\\d?frac\{([^{}]+)\}\{([^{}]+)\}\Z")
_TOLERANCE_RE = re.compile(r"\A(.+?)\s*(?:±|\+/-)\s*(.+)\Z")


def _number(text: str) -> Fraction | None:
    """A plain number (sign, thousands commas, decimals, exponent, a/b) or a \\frac, exactly; else None."""
    text = text.strip().strip("$").strip()
    frac = _FRAC_RE.match(text)
    if frac:
        num, den = _number(frac.group(2)), _number(frac.group(3))
        if num is None or not den:
            return None
        return -num / den if frac.group(1) == "-" else num / den
    if len(text) > MAX_NUMBER_CHARS or not _WHOLE_NUMBER_RE.match(text):
        return None
    exponent = re.search(r"[eE]([-+]?\d+)", text)
    if exponent and abs(int(exponent.group(1))) > MAX_NUMBER_EXPONENT:
        return None
    try:
        if "/" in text:
            num, den = text.split("/")
            return Fraction(num.replace(",", "")) / Fraction(den)
        return Fraction(text.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def _numeric_reference(reference: str) -> tuple[Fraction, Fraction | None]:
    match = _TOLERANCE_RE.match(reference.strip())
    value = _number(match.group(1) if match else reference)
    tolerance = _number(match.group(2)) if match else None
    if value is None or (match and (tolerance is None or tolerance < 0)):
        raise ValueError("a numeric_v1 reference is a number, optionally followed by ± and a tolerance")
    return value, tolerance


def _numeric_answer(answer: str) -> Fraction | None:
    """The last \\boxed{} value if there is one, else the last number of the answer."""
    from reliquary.environment.openmathinstruct import _last_boxed_only_string, _strip_boxed_wrapper

    boxed = _last_boxed_only_string(answer)
    if boxed is not None:
        return _number(_strip_boxed_wrapper(boxed))
    found = _NUMBER_RE.findall(answer)
    return _number(found[-1]) if found else None


def _numeric(reference: str, answer: str) -> float:
    value, tolerance = _numeric_reference(reference)
    got = _numeric_answer(answer)
    if got is None:
        return 0.0
    bound = tolerance if tolerance is not None else NUMERIC_RELATIVE_TOLERANCE * max(1, abs(value))
    return 1.0 if abs(got - value) <= bound else 0.0


_BOXED_CHOICE_RE = re.compile(r"\\boxed\{\s*(?:\\text\{\s*)?\(?([A-Za-z])\)?\s*\}?\s*\}")
# "answer" in any case, the letter upper case only: "the answer is a bit unclear" names no choice.
_SAID_CHOICE_RE = re.compile(r"\b(?i:answer)\s*(?:(?i:is)|:)\s*[:\-]?\s*\**\(?([A-Z])\)?(?![A-Za-z])")
_BARE_CHOICE_RE = re.compile(r"\A\s*\**\(?([A-Za-z])\)?\.?\**\s*\Z")


def _choice_reference(reference: str) -> str:
    letter = reference.strip()
    if len(letter) != 1 or not ("A" <= letter.upper() <= "Z"):
        raise ValueError("a choice_v1 reference is one letter, A to Z")
    return letter.upper()


def _choice(reference: str, answer: str) -> float:
    """The last \\boxed{X}; else the last "answer is X" / "Answer: X"; else an answer that is only X."""
    for pattern in (_BOXED_CHOICE_RE, _SAID_CHOICE_RE, _BARE_CHOICE_RE):
        found = pattern.findall(answer)
        if found:
            return 1.0 if found[-1].upper() == _choice_reference(reference) else 0.0
    return 0.0


_JSON_FENCE_RE = re.compile(r"```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```", re.IGNORECASE | re.DOTALL)


def _unique_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key")
    return dict(pairs)


def _refuse_constant(name):
    raise ValueError(f"{name} is not JSON")


def _json(text: str):
    return json.loads(text, object_pairs_hook=_unique_keys, parse_constant=_refuse_constant)


def _json_reference(reference: str):
    try:
        return _json(reference)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"a json_match_v1 reference is one JSON value: {exc}") from None


def _json_equal(a, b) -> bool:
    """Equal JSON values: numbers by value (1 == 1.0), but a boolean is never a number; key order is ignored."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _json_match(reference: str, answer: str) -> float:
    """The last ```json fence of the answer, else the whole answer, equals the reference as JSON."""
    fences = list(_JSON_FENCE_RE.finditer(answer))
    try:
        got = _json(fences[-1].group("body") if fences else answer.strip())
    except (ValueError, RecursionError):
        return 0.0
    return 1.0 if _json_equal(got, _json_reference(reference)) else 0.0


@dataclass(frozen=True, slots=True)
class ReferenceGrader:
    name: str
    # What ``answer_text`` reads of a completion: "boxed" and "text" both take the part after the reasoning.
    final_answer_policy: str
    score: Callable[[str, str], float]
    description: str
    # Raises ValueError on a reference this grader cannot read.
    check: Callable[[str], object] = lambda reference: None


class _GraderTable(dict):
    """The graders by name. ``get`` also answers for a customer env version id: a grader scoring on that
    version's reward host (``customer_env.scoring``). ``eval/grading.py`` reads graders only through this table, so
    a customer row grades without a line of it changing: its source is part of the grader fingerprint the sealed
    catalog sets carry (``eval.protocol.grader_sha256``)."""

    def get(self, key, default=None):
        found = super().get(key)
        if found is not None:
            return found
        from reliquary.customer_env import is_customer_env

        if is_customer_env(key):
            return ReferenceGrader(str(key), "text", lambda reference, answer: _score_customer(key, reference, answer),
                                   "a customer env version, scored on its reward host")
        return default


def _score_customer(env: str, reference: str, answer: str) -> float:
    """A customer row's score on its version's reward host; ValueError on a bad reference, RuntimeError with why
    when the host or the customer's grader gives no score (the row is then ungraded, never a zero)."""
    from reliquary.customer_env.scoring import default_scorer

    scorer = default_scorer()
    if scorer is None:
        raise RuntimeError("customer_env_unavailable: no reward host configured")
    value, detail = scorer.score(env, reference, answer)
    if value is None:
        raise RuntimeError(detail)
    return float(value)


REFERENCE_GRADERS: dict[str, ReferenceGrader] = _GraderTable({g.name: g for g in (
    ReferenceGrader("boxed_math_v1", "boxed", _boxed_math,
                    "the last \\boxed{} of the answer equals the reference (numbers, fractions, LaTeX values)"),
    ReferenceGrader("exact_match_v1", "text", _exact_match,
                    "the answer, stripped, equals the reference, stripped"),
    ReferenceGrader("numeric_v1", "text", _numeric,
                    "the answer's last \\boxed{} value, else its last number, equals the reference number "
                    "within the reference's '± tolerance', else within 1e-6 x max(1, |reference|)",
                    _numeric_reference),
    ReferenceGrader("choice_v1", "text", _choice,
                    "the answer's letter equals the reference letter: its last \\boxed{X}, else its last "
                    "'answer is X' / 'Answer: X', else an answer that is only X", _choice_reference),
    ReferenceGrader("json_match_v1", "text", _json_match,
                    "the answer's last ```json fence, else the whole answer, is JSON equal to the reference "
                    "(key order ignored, 1 equals 1.0, a boolean is not a number)", _json_reference),
)})


def is_known_grader(grader: object) -> bool:
    """A grader of this table, or a customer env version (scored on its reward host)."""
    from reliquary.customer_env import is_customer_env

    return grader in REFERENCE_GRADERS or is_customer_env(grader)


def check_reference(grader: str, reference) -> str:
    """The line's reference for ``grader``, or ValueError."""
    from reliquary.customer_env import is_customer_env

    if is_customer_env(grader):
        from reliquary.customer_env.scoring import check_reference as check_customer

        check_customer(grader, reference)
        return reference
    if grader not in REFERENCE_GRADERS:
        raise ValueError(f"grader_unknown: {grader!r}; graders: {sorted(REFERENCE_GRADERS)}")
    if not isinstance(reference, str) or not reference.strip() or len(reference) > MAX_REFERENCE_CHARS:
        raise ValueError(f"a reference is a non-empty string of at most {MAX_REFERENCE_CHARS} characters")
    REFERENCE_GRADERS[grader].check(reference)
    return reference


def grade_reference(grader: str, reference: str, answer: str) -> float:
    if len(answer) > MAX_ANSWER_CHARS:
        return 0.0
    found = REFERENCE_GRADERS.get(grader)
    if found is None:
        raise KeyError(grader)
    return float(found.score(reference, answer))


__all__ = ["MAX_ANSWER_CHARS", "MAX_REFERENCE_CHARS", "REFERENCE_GRADERS", "ReferenceGrader", "check_reference",
           "grade_reference", "is_known_grader"]
