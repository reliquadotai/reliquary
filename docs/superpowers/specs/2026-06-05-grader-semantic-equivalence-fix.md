# Reward graders must test what the prompt asks, not surface form

Date: 2026-06-05

## Problem

For both environments, the reward check is **stricter than the task we give the
model**. The prompt specifies a *behavior* (code) or a *value* (math), but the
grader requires an exact *surface form*. So semantically-correct completions are
scored `0.0`. These are false negatives: the reward signal punishes correct work
for cosmetic reasons the prompt never stated.

### Code (`opencodeinstruct`)
The grader resolves the entry point by exact name:

```python
# reliquary/environment/grader/worker.py  (evaluate_call)
fn = ns[entry["name"]]          # KeyError -> "runtime_error" -> 0.0
```

`entry["name"]` is whatever the reference solution happened to call its function.
But the prompt asks *"write a function that computes the LCM of two integers"* — it
does **not** state the required name, and carries no signature. A correct solution
named `compute_lcm` (or `levenshtein_distance` for an edit-distance task) fails with
`KeyError`, even though it is exactly what was requested. The model has no way to
know the expected name, so it picks a reasonable one and is penalized for it.

### Math (`openmathinstruct`)
The grader compares normalized strings:

```python
# reliquary/environment/openmathinstruct.py  (_compute_omi_reward)
return 1.0 if candidate == gt else 0.0
```

`_normalize_answer` only collapses a fully-zero decimal (`-?\d+\.0+`). So `82.5`
and `82.50` are treated as different, although they are the same number. The prompt
asks for the value; nothing tells the model how many trailing zeros to print.

### Why it matters
A grader that rejects correct answers injects noise into every downstream consumer
of the reward (selection, training signal, group shaping). The fix is to make the
**grade test the same thing the prompt asks**.

## Principle

Align the check with the request, from both ends:
1. **Make the contract explicit in the prompt** (tighten what we ask), and
2. **Judge by meaning, not surface** (loosen how we check).

Doing both removes the false negatives regardless of which side a given completion
diverges on, and it does so without ever accepting a *wrong* answer.

## Fix — math: compare by value

When both the ground truth and the extracted answer parse as numbers, compare them
numerically; otherwise keep the existing normalized-string comparison (LaTeX
expressions, set/tuple answers, etc.).

```python
from fractions import Fraction

def _as_number(s: str):
    s = s.strip().replace(" ", "").replace(",", "")
    try:
        return Fraction(s)            # handles "31", "82.50", "3/4", "-2"
    except (ValueError, ZeroDivisionError):
        return None

def _answers_equal(candidate: str, gt: str) -> bool:
    c, g = _as_number(candidate), _as_number(gt)
    if c is not None and g is not None:
        return c == g                 # exact rational equality; 82.5 == 82.50
    return _normalize_answer(candidate) == _normalize_answer(gt)
```

`Fraction` gives exact equality for decimals and simple fractions (no float
rounding). For irrational/expression answers (`\frac{8\pi\sqrt{6}}{3}`) it returns
`None` and we fall back to today's string path — no behavior change there.

This is **value-based**, so it cannot let a numerically-wrong answer pass.

## Fix — code

### Root: put the entry-point contract in the prompt
The required function name (and signature) is the contract the grader enforces, so
it must be part of what we ask the model — exactly as the math env already appends
its answer-format instruction (`_ANSWER_FORMAT_INSTRUCTION`). Surface the
entry-point in the prompt via `get_problem`, identical on the validator and the
miner-facing prompt set (required for GRAIL prompt consensus):

```
Implement your solution as a function named `lcm(a, b)`.
```

The name/signature is **public** (it is the contract); only the test cases stay
hidden. Once the model is told the name, the exact-match check is legitimate.

### Robustness: resolve the entry point by structure when the name is absent
The model cannot know the reference name, so the grader must not fail correct code
over it. When `entry["name"]` is absent, pick **one** entry deterministically and
run it (never run several and accept any pass — that would let a miner shotgun
functions against the limited cases). Among the submitted top-level functions
(imported callables excluded) that accept the call's arity:

1. exactly one arity match → use it (covers ~79% of solutions: a single function);
2. else the only **call-graph root** — a function no *other* top-level function
   calls (self-recursion excluded), i.e. the entry of a "main + helpers" solution;
3. else the **last-defined** arity match (a deterministic single pick).

A wrong pick simply fails the hidden cases, so being liberal here costs recall, not
correctness. We only fail outright when no defined function can accept the call.
(Measured: ~79% of completions define one function, ~16% define helpers; arity +
call-graph-root resolve most of the rest.)

Related robustness (optional): `_extract_python` takes the *last* fenced block; a
trailing "example usage" block can hide the solution. Prefer the block that defines
the entry point, or merge all blocks.

## Safety (no new false positives)
- Math: equality is on the numeric value, so wrong numbers still fail.
- Code: the chosen function is still run against the hidden cases; we relax *which*
  defined function we call (one deterministic pick), but never run several and
  accept any pass. Wrong logic still fails its cases.

## Rollout: shadow first
Gate behind a flag. In shadow, compute both the current grade and the new grade,
and log every disagreement (env, prompt id, old → new). Confirm on real traffic:
- the new grade flips only *previously-correct-but-mis-surfaced* completions, and
- no completion that the old grader passed now fails.

Then enable enforcement.

## Tests
- Math: `82.5`/`82.50`, `3`/`3.0`, `1/2`/`0.5` → equal; `82.5`/`83` → not equal;
  LaTeX expression unchanged vs today.
- Code: correct function under a non-reference name (sole def) → passes; correct
  reference name → still passes; genuinely wrong logic → still fails; two defined
  functions with the expected name absent → `entry_not_found` (unchanged fail).
- Prompt: validator and miner-facing `get_problem` produce byte-identical prompts
  including the entry-point line.

## Impact note
Grading correctness faithfully will increase the share of completions (and whole
groups) that pass. Confirm the downstream group-shape / zone handling behaves
sensibly when a group is fully or near-fully correct, so the fairer grade is not
silently discarded later in the pipeline.
