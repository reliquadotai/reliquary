# Semantic-equality grader: exact symbolic equivalence for OMI math

**Date:** 2026-07-03
**Status:** Design approved, pending spec review
**Scope:** validator-side, `reliquary/environment/openmathinstruct.py` only

## Problem

The OpenMathInstruct grader marks a completion correct only if its `\boxed{}`
answer matches the ground truth by (a) exact numeric value, or (b) normalized
string. It already has a bounded, sympy-based **numeric** value-equality path
(`_latex_value_equal`), so `\frac{5}{3} == 5/3` and `82.50 == 82.5` pass.

It does **not** compare **algebraic** expressions by value. Any answer with a
free variable falls through to exact-string comparison, so a correct answer in a
reordered form is scored 0. GPU replay (2026-07-03, checkpoint-exact) confirmed
this is an actively-exploited "free negative" source: e.g. window 19002, ground
truth `-b+2`, a rollout boxing `2 - b` (algebraically identical) is graded 0.
Miners curate these wrongly-zeroed rollouts into their submission to guarantee a
mixed (in-zone) reward vector. Honest miners hit the same grader flaw and lose
reward they earned.

Measured on windows 19000–19180: ~4% of miner `5CGH8`'s reward-0 negatives are
exact-value-equal-to-ground-truth in a different algebraic form (0% for `5HL6`,
whose negatives are genuine wrong answers).

### Root cause

`_latex_value_equal` computes `complex(expr.evalf())`. On an expression with a
free symbol, `evalf()` stays symbolic and `complex(...)` raises, so the function
returns `False` via its `except`. Algebraic equivalence is never tested.

## Decision

Add **exact symbolic equivalence** only. Out of scope (deliberately):

- **Rounding / precision** (`60/7` vs `8.57`, `1.67` vs `5/3`): not exactly
  equal; accepting them needs a tolerance that would also accept genuinely-wrong
  close answers and opens a new exploit. This is a ground-truth-precision issue,
  handled separately if at all.
- **LLM-judge equivalence**: non-deterministic across validators (breaks
  consensus), heavier, and prompt-injectable via the boxed payload.

Rollout: **unconditional** — treated as a correctness bugfix, always on, no env
flag, no shadow counter (the change is a strict superset with zero false
positives, so there is nothing to pre-measure or gate).

## The change

One new single-purpose function plus one line appended to the existing cascade.

```python
def _latex_symbolic_equal(candidate: str, gt: str) -> bool:
    """True iff candidate and gt both parse to ALGEBRAIC expressions (each with
    at least one free symbol) whose expanded difference is identically zero.

    Closes algebraic reorderings the numeric path misses (2-b == -b+2,
    (x+1)^2 == x^2+2x+1). Purely-numeric answers have no free symbols and never
    enter here, so the numeric path's 1e-9 tolerance still governs numbers and no
    rounding false-positive can leak in. Never raises."""
    # same bounds as _latex_value_equal, EXTENDED to permit free symbols (the
    # existing guards reject variables) via an allow_symbols=True flag:
    #   1. len(candidate) <= 100 and len(gt) <= 100
    #   2. ec, eg = _latex_to_pyexpr(candidate), _latex_to_pyexpr(gt); None -> False
    #   3. _expr_str_is_safe(ec, allow_symbols=True) and same for eg
    #   4. parse_expr(..., evaluate=False) with the same transformations
    #   5. _expr_is_safe(xc, allow_symbols=True) and same for xg (power-tower guard)
    # then:
    #   6. require xc.free_symbols and xg.free_symbols        (both algebraic)
    #   7. return sympy.expand(xc - xg).is_zero is True       (deterministic)
    # any parse/eval error -> return False
```

`_answers_equal` gains one line at the end of its cascade:

```python
    if candidate == gt:
        return True
    if _latex_value_equal(candidate, gt):     # existing: numeric value-equality
        return True
    return _latex_symbolic_equal(candidate, gt)  # NEW: exact algebraic equivalence
```

### Why these method choices

- **`expand`, not `simplify` or `.equals()`.** The grader is a consensus
  mechanism: every validator must reach the same verdict. `expand` on an
  `evaluate=False` parse is deterministic across sympy versions; `simplify` is
  slow and version-sensitive, `.equals()` uses random numeric sampling
  (non-deterministic). `expand(a-b).is_zero` decides exactly the polynomial /
  rational reordering family that leaks, which is the whole target.
- **`.is_zero is True`** (not `== 0`): `.is_zero` returns `True` / `False` /
  `None`; treating only `True` as equal means an undecidable case (e.g. an
  exotic transcendental) safely reads as *not equal* rather than raising or
  guessing.
- **`free_symbols` gate on both sides** routes every purely-numeric answer to
  the numeric path, so this function can never introduce a rounding false
  positive.

## Safety / consensus guarantees

- **Strict superset**: reached only after the full existing cascade returns
  `False`; can only flip `0 → 1` for exactly-equivalent algebra, never `1 → 0`.
  No regression on currently-correct grading.
- **Determinism**: `expand` is deterministic; sympy is already a consensus
  dependency via the numeric path. No new consensus surface.
- **DoS-bounded**: same 100-char cap + `_expr_str_is_safe` (blocks factorials /
  function names `parse_expr` would eager-evaluate) + `_expr_is_safe` (blocks
  power towers / huge exponents), extended with an `allow_symbols=True` flag so
  single-letter variables pass while every other guard still fires. Bounded
  exponents (≤10) plus the char cap keep `expand` fast.
- **Never raises**: wrapped in `try/except -> False`, matching `_latex_value_equal`.

## Testing (TDD — tests before code)

Write failing tests first, in the OMI grader test module.

**Must now pass (0 → 1):**
- `2-b` ≡ `-b+2`, `a+b` ≡ `b+a`, `2(x+1)` ≡ `2x+2`, `(x+1)^2` ≡ `x^2+2x+1`
- LaTeX forms: `\frac{x}{2}+1` ≡ `\frac{x+2}{2}`

**Must stay rejected (0 → 0):**
- `2-b` vs `2+b`, `x^2` vs `x^3`, `a+b` vs `a+c` (different algebra)
- `8.57` vs `60/7`, `1.67` vs `5/3`, `3.1` vs `4` (numeric / rounding — no free
  symbols, never enters symbolic path)

**Must not regress (still 1):** existing numeric cases (`\frac{5}{3}`==`5/3`,
`82.50`==`82.5`), structured/matrix cases.

**Adversarial / DoS (→ `False`, no hang, no raise):**
- `9^9^9^9`, a 100+ char expression, unknown `\macro`, factorial `5!`.

**Replay regression:** the concrete boxed strings pulled from R2
(`2 - b` @ w19002 flips 0→1; `60/7` vs `8.57` @ w19003 stays 0).

## Impact

- Closes the exact-form free-negative family (~4% of `5CGH8`'s negatives) and,
  more importantly, stops zeroing honest miners' correctly-reordered answers.
- Does **not** address rounding-form negatives (out of scope) or **arrangement
  pinning** — the dominant cheat for both `5CGH8` (2.27 bit) and `5HL6`/UID205
  (1.46 bit), which curates genuine outcomes and is invisible to any grader.
  That remains a separate lever (arrangement-entropy detection).

## Out of scope

- Rounding/precision tolerance; LLM-judge grading; the code (OpenCode) grader;
  any change to reward-shape / σ-zone / arrangement-entropy detection.
