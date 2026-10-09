"""Whether a program's output is the expected one.

The single source of truth for a verdict: the local runner and the core's
sandboxed grader both call this function on the captured output.

Token by token on whitespace, so line endings and spacing never decide a
verdict. Two relaxations, both what the original judges accept: `yes`/`no` in
any case, and a tolerance on decimals. An expected integer is compared
exactly: a decimal matches it only when its value is exactly that integer
("2.000000" for "2"), so a float that is merely close is wrong.
"""

from __future__ import annotations

import math

FLOAT_TOLERANCE = 1e-6
_CASELESS = frozenset({"yes", "no"})


def outputs_match(expected: str, actual: str) -> bool:
    want, got = expected.split(), actual.split()
    return len(want) == len(got) and all(map(_tokens_match, want, got))


def _tokens_match(want: str, got: str) -> bool:
    if want == got:
        return True
    if want.lower() in _CASELESS:
        return want.lower() == got.lower()
    if not _is_decimal(want):
        return _is_integer(want) and _is_decimal(got) and float(got) == int(want)
    a, b = _finite(want), _finite(got)
    return (
        a is not None
        and b is not None
        and math.isclose(a, b, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)
    )


def _is_decimal(token: str) -> bool:
    return any(c in token for c in ".eE") and _finite(token) is not None


def _is_integer(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def _finite(token: str) -> float | None:
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None
