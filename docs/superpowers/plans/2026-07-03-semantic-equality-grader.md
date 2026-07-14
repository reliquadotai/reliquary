# Semantic-equality Grader (exact symbolic) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the OpenMathInstruct grader accept algebraically-equivalent answers (`2-b` == `-b+2`, `(x+1)^2` == `x^2+2x+1`) that its numeric-only value path currently scores 0.

**Architecture:** Extend the two existing DoS-guards with an `allow_symbols` flag (they currently reject free symbols), then add one `_latex_symbolic_equal` helper that expands the difference of two parsed expressions and checks it is identically zero. Wire it as the final fallback in `_answers_equal`, after the existing numeric value-equality path. Strict superset: can only flip a current 0 → 1, never 1 → 0.

**Tech Stack:** Python, sympy (already a grader dependency), pytest.

## Global Constraints

- Change is confined to `reliquary/environment/openmathinstruct.py` (validator-side, math env only). Do NOT touch the code/OpenCode grader.
- Consensus-critical: grading must be deterministic across validators. Use `sympy.expand(...).is_zero` — never `simplify()` or `.equals()` (non-deterministic / slow).
- DoS-bounded: every sympy call stays behind the existing 100-char cap + `_expr_str_is_safe` + `_expr_is_safe` guards (extended, not bypassed).
- Never raise: wrap parse/eval in `try/except -> return False`, matching `_latex_value_equal`.
- No env flag, no shadow: ship as an always-on correctness fix.
- Numbers must NOT enter the symbolic path (both sides must have free symbols) so no rounding false-positive can leak in; `60/7` vs `8.57` must stay rejected.
- Work on branch `fix/omi-symbolic-grader` (create off current `main` before Task 1). Never commit on `main`.

---

### Task 1: Extend DoS-guards to optionally allow free symbols

The existing `_expr_str_is_safe` (string token whitelist) and `_expr_is_safe` (post-parse structural whitelist) both reject variables, so an algebraic answer never reaches any sympy evaluation. Add a default-`False` `allow_symbols` flag to each so the numeric path is byte-for-byte unchanged while a new caller can opt in to single-letter variables.

**Files:**
- Modify: `reliquary/environment/openmathinstruct.py` (`_expr_is_safe` ~line 142, `_expr_str_is_safe` ~line 172)
- Test: `tests/unit/test_openmathinstruct_environment.py`

**Interfaces:**
- Produces: `_expr_str_is_safe(s: str, allow_symbols: bool = False) -> bool` and `_expr_is_safe(expr, _depth: int = 0, allow_symbols: bool = False) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_openmathinstruct_environment.py`:

```python
def test_expr_str_guard_symbols_flag():
    from reliquary.environment.openmathinstruct import _expr_str_is_safe
    # default (numeric) behaviour unchanged: a variable is rejected
    assert _expr_str_is_safe("2 - b") is False
    assert _expr_str_is_safe("2 + 3*(4)") is True
    # with allow_symbols: single-letter variables allowed...
    assert _expr_str_is_safe("2 - b", allow_symbols=True) is True
    assert _expr_str_is_safe("x**2 + 2*x + 1", allow_symbols=True) is True
    # ...but factorials and function calls still rejected
    assert _expr_str_is_safe("5!", allow_symbols=True) is False
    assert _expr_str_is_safe("exp(x)", allow_symbols=True) is False
    assert _expr_str_is_safe("gamma(3)", allow_symbols=True) is False


def test_expr_struct_guard_symbols_flag():
    import sympy
    from reliquary.environment.openmathinstruct import _expr_is_safe
    x, b = sympy.symbols("x b")
    # symbols rejected by default, allowed under the flag
    assert _expr_is_safe(x) is False
    assert _expr_is_safe(x, allow_symbols=True) is True
    assert _expr_is_safe((x + 1) ** 2, allow_symbols=True) is True
    assert _expr_is_safe(2 - b, allow_symbols=True) is True
    # power tower / huge exponent still rejected even with symbols allowed
    assert _expr_is_safe(x ** 50, allow_symbols=True) is False
    assert _expr_is_safe(sympy.Pow(2, x, evaluate=False), allow_symbols=True) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_openmathinstruct_environment.py::test_expr_str_guard_symbols_flag tests/unit/test_openmathinstruct_environment.py::test_expr_struct_guard_symbols_flag -v`
Expected: FAIL (TypeError: unexpected keyword argument `allow_symbols`).

- [ ] **Step 3: Implement the flag on both guards**

In `_expr_is_safe`, change the signature and add one symbol branch; thread the flag through the recursive calls:

```python
def _expr_is_safe(expr, _depth: int = 0, allow_symbols: bool = False) -> bool:
    import sympy

    if _depth > 40:
        return False
    if expr.is_Number or isinstance(expr, sympy.NumberSymbol):
        return True
    if allow_symbols and expr.is_Symbol:
        return True
    if isinstance(expr, (sympy.Add, sympy.Mul)):
        return all(_expr_is_safe(a, _depth + 1, allow_symbols) for a in expr.args)
    if isinstance(expr, sympy.Pow):
        base, exp = expr.as_base_exp()
        if not exp.is_Number:
            return False  # nested / symbolic exponent (e.g. a power tower)
        try:
            if abs(float(exp)) > 10:
                return False  # huge exponent
        except (TypeError, ValueError, OverflowError):
            return False
        return _expr_is_safe(base, _depth + 1, allow_symbols)
    return False  # factorial, other functions, disallowed symbols, ...
```

In `_expr_str_is_safe`, add the flag and strip isolated single-letter variables before the numeric-only fullmatch:

```python
def _expr_str_is_safe(s: str, allow_symbols: bool = False) -> bool:
    cleaned = re.sub(r"(?<![A-Za-z])(?:sqrt|pi)(?![A-Za-z])", "", s)
    if allow_symbols:
        # drop lone single-letter variables (not part of a longer identifier and
        # not a function call `f(`), leaving numbers / operators for the fullmatch.
        cleaned = re.sub(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z(])", "", cleaned)
    return re.fullmatch(r"[0-9.+\-*/() \t]*", cleaned) is not None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_openmathinstruct_environment.py -k "guard" -v`
Expected: PASS. Also run the full file to confirm no regression: `pytest tests/unit/test_openmathinstruct_environment.py -v`.

- [ ] **Step 5: Commit**

```bash
git add reliquary/environment/openmathinstruct.py tests/unit/test_openmathinstruct_environment.py
git commit -m "feat(grader): allow_symbols flag on OMI expression DoS-guards"
```

---

### Task 2: Symbolic equivalence in the OMI grader

Add `_latex_symbolic_equal` and make it the final fallback in `_answers_equal`, so an algebraic answer whose expanded difference with the ground truth is zero scores 1.0.

**Files:**
- Modify: `reliquary/environment/openmathinstruct.py` (add helper after `_latex_value_equal`; edit `_answers_equal` tail, ~line 269)
- Test: `tests/unit/test_openmathinstruct_environment.py`

**Interfaces:**
- Consumes: `_latex_to_pyexpr`, `_expr_str_is_safe(..., allow_symbols=True)`, `_expr_is_safe(..., allow_symbols=True)` from Task 1.
- Produces: `_latex_symbolic_equal(candidate: str, gt: str) -> bool`; `_answers_equal` now returns `True` for exact algebraic equivalence.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_openmathinstruct_environment.py`:

```python
def test_reward_algebraic_reorder_is_equal():
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    # the exact free-negative family seen in R2 replay (w19002)
    assert _compute_omi_reward({"ground_truth": "-b+2"}, r"\boxed{2 - b}") == 1.0
    assert _compute_omi_reward({"ground_truth": "a+b"}, r"\boxed{b + a}") == 1.0
    assert _compute_omi_reward({"ground_truth": "x^2+2x+1"}, r"\boxed{(x+1)^2}") == 1.0
    assert _compute_omi_reward({"ground_truth": "2x+2"}, r"\boxed{2(x+1)}") == 1.0


def test_reward_algebraic_nonequivalent_still_zero():
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    assert _compute_omi_reward({"ground_truth": "2+b"}, r"\boxed{2 - b}") == 0.0
    assert _compute_omi_reward({"ground_truth": "x^2"}, r"\boxed{x^3}") == 0.0
    assert _compute_omi_reward({"ground_truth": "a+c"}, r"\boxed{a + b}") == 0.0


def test_reward_rounding_stays_out_of_symbolic_path():
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    # numbers have no free symbols -> numeric path governs -> rounding rejected
    assert _compute_omi_reward({"ground_truth": "8.57"}, r"\boxed{60/7}") == 0.0
    assert _compute_omi_reward({"ground_truth": "\\frac{5}{3}"}, r"\boxed{1.67}") == 0.0
    assert _compute_omi_reward({"ground_truth": "4"}, r"\boxed{3.1}") == 0.0


def test_reward_symbolic_adversarial_no_hang():
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    # DoS-shaped payloads must return quickly as 0.0, never raise/hang
    assert _compute_omi_reward({"ground_truth": "x"}, r"\boxed{9^9^9^9}") == 0.0
    assert _compute_omi_reward({"ground_truth": "x"}, r"\boxed{x!}") == 0.0
    assert _compute_omi_reward({"ground_truth": "x"}, "\\boxed{" + "x+" * 60 + "x}") == 0.0


def test_reward_numeric_and_structured_not_regressed():
    from reliquary.environment.openmathinstruct import _compute_omi_reward
    assert _compute_omi_reward({"ground_truth": "1/2"}, r"\boxed{0.5}") == 1.0
    assert _compute_omi_reward({"ground_truth": "82.50"}, r"\boxed{82.5}") == 1.0
    assert _compute_omi_reward({"ground_truth": "43"}, r"\boxed{42}") == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_openmathinstruct_environment.py -k "algebraic or rounding_stays or symbolic_adversarial" -v`
Expected: FAIL (reorder cases return 0.0, not 1.0).

- [ ] **Step 3: Add `_latex_symbolic_equal`**

Insert directly after `_latex_value_equal` (before `_split_structure`):

```python
def _latex_symbolic_equal(candidate: str, gt: str) -> bool:
    """True iff candidate and gt both parse to ALGEBRAIC expressions (each with
    at least one free symbol) whose expanded difference is identically zero.

    Closes algebraic reorderings the numeric path misses (2-b == -b+2,
    (x+1)^2 == x^2+2x+1). Purely-numeric answers have no free symbols and never
    enter here, so the numeric 1e-9 path still governs numbers and no rounding
    false-positive can leak in. Bounded and deterministic; never raises.
    """
    if not candidate or not gt or len(candidate) > 100 or len(gt) > 100:
        return False
    ec, eg = _latex_to_pyexpr(candidate), _latex_to_pyexpr(gt)
    if ec is None or eg is None:
        return False
    if not (_expr_str_is_safe(ec, allow_symbols=True)
            and _expr_str_is_safe(eg, allow_symbols=True)):
        return False
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
        tr = standard_transformations + (implicit_multiplication_application,)
        xc = parse_expr(ec, transformations=tr, evaluate=False)
        xg = parse_expr(eg, transformations=tr, evaluate=False)
        if not (xc.free_symbols and xg.free_symbols):
            return False  # numbers stay on the numeric path
        if not (_expr_is_safe(xc, allow_symbols=True)
                and _expr_is_safe(xg, allow_symbols=True)):
            return False
        return sympy.expand(xc - xg).is_zero is True
    except Exception:
        return False
```

- [ ] **Step 4: Wire it into `_answers_equal`**

Replace the final line of `_answers_equal`:

```python
    if candidate == gt:
        return True
    return _latex_value_equal(candidate, gt)
```

with:

```python
    if candidate == gt:
        return True
    if _latex_value_equal(candidate, gt):
        return True
    return _latex_symbolic_equal(candidate, gt)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_openmathinstruct_environment.py -v`
Expected: PASS (all new tests + no regressions).

- [ ] **Step 6: Commit**

```bash
git add reliquary/environment/openmathinstruct.py tests/unit/test_openmathinstruct_environment.py
git commit -m "feat(grader): accept exact algebraic equivalence in OMI answers"
```

---

## Self-Review

**1. Spec coverage:**
- Symbolic exact equivalence (`expand(a-b).is_zero`) → Task 2 helper. ✓
- Restricted to expressions with free symbols (numbers untouched) → Task 2 Step 3 `free_symbols` gate + `test_reward_rounding_stays_out_of_symbolic_path`. ✓
- `expand` not `simplify`/`.equals()` (consensus determinism) → Task 2 Step 3. ✓
- Strict superset (only appended after existing cascade) → Task 2 Step 4. ✓
- DoS bounds reused/extended → Task 1 (`allow_symbols`) + `test_reward_symbolic_adversarial_no_hang`. ✓ (Spec said "reuse"; correctly refined to "extend with allow_symbols" because the existing guards reject symbols.)
- Always on, math env only → no flag added; change confined to `openmathinstruct.py`. ✓
- Replay regression (`2 - b` flips, `60/7` stays 0) → `test_reward_algebraic_reorder_is_equal` + `test_reward_rounding_stays_out_of_symbolic_path`. ✓

**2. Placeholder scan:** none — all steps carry runnable code/commands.

**3. Type consistency:** `_expr_str_is_safe(..., allow_symbols=True)` and `_expr_is_safe(..., allow_symbols=True)` defined in Task 1, consumed with those exact names/kwargs in Task 2. `_latex_symbolic_equal(str, str) -> bool` consistent between definition and `_answers_equal` call site. ✓
