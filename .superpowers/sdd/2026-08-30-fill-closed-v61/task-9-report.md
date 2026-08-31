# Task 9 report — fill_window semantics (v6.1, R33/R35)

## Status: DONE, full suite green.

## Changes
- `reliquary/validator/fill_window.py`: `FillState` constructor now takes
  `budgets: Mapping[str, int]` + `picks_target: int` (was `targets`).
  `may_admit` gates on a new monotone `admitted` counter incremented only
  by `reserve()`, never decremented by `release()` (`release()` still
  frees `in_flight`). Added `record_pick()` (raises past `picks_target`)
  and `is_closed()` now reads `picks_emitted >= picks_target` instead of
  a per-env proven count. `snapshot()` exposes
  `budgets/admitted/proven/in_flight/picks_emitted/picks_target/closed`.
- `reliquary/constants.py`: added `FILL_CLOSED_ADMISSION_BUDGET_PER_ENV`
  (default 512, `RELIQUARY_FILL_CLOSED_ADMISSION_BUDGET_PER_ENV`),
  `FILL_CLOSED_TARGET_GROUPS_PER_ENV` left untouched (still used for the
  journal-key-range guard, unrelated to FillState now).
- `reliquary/validator/service.py`: `_build_window_batchers` now builds
  `FillState(budgets={env: FILL_CLOSED_ADMISSION_BUDGET_PER_ENV ...},
  picks_target=FILL_CLOSED_EMISSIONS_PER_WINDOW)`.
- `reliquary/validator/batcher.py`: one production read of
  `fill_state.snapshot()["targets"]` (proof-plan `required_passes`
  sizing in `_extend_proof_plan`) updated to `["budgets"]` — same value,
  renamed key only. No pick/emission logic touched (Task 11's seam).

## Test fallout (expected, not scope creep)
Renaming `targets=`→`budgets=`+`picks_target=` was mechanical across
`test_prove_on_arrival.py`, `test_rate_ordered_admission.py`,
`test_fill_close_and_emit.py`. Two classes of tests needed real rewrites
because their *behavior*, not just the kwarg, depended on the old rule:
1. Close-on-proven tests (3, in `test_fill_close_and_emit.py`) now call
   `record_pick()` directly to reach `is_closed()` — legitimate use of
   the new public API, no batcher.py change.
2. Five arrival-buffer tests relied on `release()` reopening admission
   room (`_drain_arrival_proof_buffer`'s "buffer while full, drain on
   release" path). Under the monotone budget this path is permanently
   dead once budget is spent — a real, intended consequence of R33
   (failed groups don't get free retries against the same budget).
   Rewrote these 5 to assert the new true behavior of the *unmodified*
   `_drain_arrival_proof_buffer` code (buffered work stays buffered
   after a release; multi-candidate sort-order races are now
   constructed by pre-populating the buffer directly, or by holding the
   auto-drain off during submission, rather than via release-then-drain).
   No coverage deleted; all rewritten tests document R33 in their
   docstring.

## Verification
- `tests/unit/test_fill_window.py` — 10/10 (TDD: watched all 10 fail on
  the old constructor signature before implementing).
- `test_fill_close_and_emit.py` (13), `test_prove_on_arrival.py` (16 incl.
  split test), `test_rate_ordered_admission.py` (1), `test_fill_closed_profile.py`
  (unaffected) — all green together (46 total).
- Full suite: `2321 passed, 1 deselected` (the pinned
  `test_admission_isolation.py::test_spawned_worker_deadline_is_terminal`
  deselect) — above the 2300 baseline.

## Concerns for Task 11
`_drain_arrival_proof_buffer`/`_extend_proof_plan` currently still
"grant a seat" (reserve into the proof plan) at drain time, on arrival —
exactly what amendment Component 1 says moves to pick time. Once budget
is spent for an environment, that buffer-drain path goes permanently
inert (documented, tested), which is consistent with "seats are granted
at picks, not on arrival" but means Task 11 likely needs to rework or
remove the reserve-at-drain step, not just add a pick step alongside it.
