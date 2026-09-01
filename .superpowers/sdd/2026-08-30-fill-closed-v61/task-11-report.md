# Task 11 report — pick-by-rate emission (v6.1, R32)

## Status: DONE. Commits `271fe5f`, `5784835`. Full suite 2334 passed
(baseline 2321 + 13 new pins), 1 deselected (the pinned
`test_admission_isolation.py::test_spawned_worker_deadline_is_terminal`).

## What changed

**A. Emission is a pick.** `_maybe_emit_batch`/`_emit_training_batch`
(watermark-order slice of the first `B_BATCH` proven groups) are gone,
replaced by `pick_training_batch()` (public) + `_claim_pick_chunk()`
(the one critical section). A pick takes the `B_BATCH` best by
`_pick_sort_key`: highest precommit rate first, unknown rate last, ties
by payload bytes DESCENDING, then receipt id — the comment says why in
as many words (length-neutral metric ⇒ ties are the common case ⇒ an
arrival tie-break would hand the seat back to the shortest answer).

`_proven_groups[env]` now holds `_ProvenGroup(value, rate,
payload_bytes, receipt_id, picked)` instead of a bare `ValidSubmission`.
The rate and payload size are read once at `_submit_arrival_proof`
(added `ThroughputAdmissionQueue.payload_bytes_of`, mirroring
`rate_of`'s None-on-miss contract; falls back to the request's own
`_payload_bytes`, else 0, which sorts last like an unknown rate),
carried through `_BufferedArrivalProof`, pinned to the candidate's
`job_id` in `_arrival_proof_meta` at drain time, and attached to the
proven record by the reconcile walk. The `picked` flag replaces
`_emitted_group_watermark` (deleted): a pick claims an arbitrary subset,
which no single integer can describe.

**B. Externally triggered.** `_reconcile_fill_state_decisions` no longer
calls emission — a completed proof only grows the pool. A pick never
emits a partial batch (returns False under `B_BATCH` unpicked, pool
untouched) and refuses once the window is closed or sealed rather than
letting `record_pick()` raise. Nothing calls `pick_training_batch()` in
production yet: that is T12. With `FILL_CLOSED_ENABLED` off, `fill_state`
is None and the method returns False immediately.

**C. Close burns the leftovers.** `_burn_unpicked_proven_groups()` runs
on BOTH v6 exits of `poll_deadline` (Nth-pick close and the
`FILL_CLOSED_MAX_SECONDS` backstop), after `_seal_v6_proof_plan()`.
Counts the unpicked pool and its `eos_tokens`, logs at INFO, exposes
`fill_closed_burned_groups` / `fill_closed_burned_eos_tokens` in
`upload_precommit_conservation()` — which the service already archives
per environment (`upload_precommit_conservation_by_environment`). Burned
groups never reach the assembler, so they are never paid (R32, stated in
the docstring).

**D. Drain seam.** `_drain_arrival_proof_buffer`'s docstring now states
the monotone-budget meaning (R33): draining spends real grading/proof
cost, `release()` frees `in_flight` only, admission never reopens, and
once the budget is spent the drain is permanently inert — which costs a
miner nothing it was promised, because budget buys a proof attempt and a
place in the pool, never a seat. Rewrote every comment still describing
the old close rule: `_seal_v6_proof_plan` (fill-close no longer implies
the plan self-finalised — it usually has real work on both paths now),
`_extend_proof_plan` (`required_passes` is the admission BUDGET, not the
close condition), the `fill_state`/lock-discipline constructor comments,
`fill_window.py`, `fill_closed_batch_assembler.py`, `service.py`.

## Tests (all watched red first)

New `tests/unit/test_pick_by_rate.py` (13): late full-rate beats early
low-rate; rate tie → larger payload, never earlier arrival; unknown rate
sorts last; no partial pick; a second pick never reuses the first's
groups; a completed proof (real `GlobalProofScheduler`) emits nothing on
its own; rate+payload travel from arrival to the proven record; one
window-wide pick across two environments; no pick after close; both
close paths burn (count + eos tokens + INFO log + telemetry); burn
counted once across polls; gate off ⇒ no picks, no burn.

Rewritten honestly, no coverage dropped:
`test_fill_close_and_emit.py` — the B_BATCH-emission test became "a pick
hands over one chunk of its own environment" (still pins the callback
signature and that proving alone emits nothing); the concurrency test
now races a writer thread against a PICKING thread (same lock-discipline
invariant, no drops/duplicates); the reconcile-hook test now pins that
the WALK grows the pool while `record_proven` alone does not.
`test_prove_on_arrival.py` (2 `_BufferedArrivalProof` constructions),
`test_v6_seal_seam.py` (cooldown covers picked AND burned records),
`test_validator_server.py` (health payload's exact dict).

## Decisions taken (flag for review)

1. **A window-wide pick counted once across environments.**
`FillState.picks_emitted` is window-wide (R35, service comment) but a
batcher only holds its own environment's pool, so N batchers each take
their env's share of the SAME pick k. `_claim_pick_chunk` therefore
advances `_batches_emitted` (this env's ordinal) every pick and calls
`record_pick()` only when that ordinal passes the window-wide count —
the first environment to reach pick k advances it, the sibling that
follows does not. Counting once per env would close a two-environment
window at half the batches R35 asks for. Pinned by
`test_a_window_wide_pick_is_counted_once_across_environments`. This is
the one point where the brief ("`fill_state.lock` for the selection +
`record_pick()`") did not say how many times per pick to record; T12
should confirm the pacing loop matches (one pick k = one call per env
batcher = one journal payload).

2. **Cooldown still covers burned groups.** `_seal_fill_closed_window`
records prompt/content cooldown and rollout hashes for every PROVEN
group, picked or burned. The sets are a replay defence, not a payment
record: a burned group was graded, proved and seen. Renamed the local
from `paid` to `recorded` so the code stops claiming otherwise.

3. **Post-seal stragglers are burned but uncounted.** A proof finalising
after the burn ran lands in the pool unpicked and unpayable (no pick can
take it — the window is sealed), but is not in the archived count. Said
so in the docstring rather than adding a second mechanism.

## Concerns

- Nothing calls `pick_training_batch()` yet, so on the v6.1 branch with
  the gate ON a window would currently emit nothing and burn everything
  at the backstop. T12 closes that.
- `_extend_proof_plan` sizes `required_passes` off the 512-per-env
  budget, so the plan is now essentially never self-finalising and every
  window relies on `_seal_v6_proof_plan`'s explicit seal. That path is
  tested but was previously the rare case, not the norm.

## Fix round 1 (R37)

Commit `8a85ff0`. All 11 new pins watched red first.

**Critical — per-environment pick ordinals.** `FillState` no longer keeps
one window-wide integer. `record_pick(environment)` advances that
environment's own ordinal; `snapshot()["picks_emitted"]` is the MIN over
environments (complete events only — which is exactly what
`_drive_fill_closed_picks` reads to name the event it is about to drive,
so T12's `next_pick` arithmetic stays correct mid-event); `is_closed()`
asks every environment; `picks_taken(env)` and
`snapshot()["picks_by_environment"]` are new. `_claim_pick_chunk` and
`can_pick()` gate on THIS environment's ordinal against `picks_target`,
never on the window-wide close — so the final event's sibling is never
refused, and `poll_deadline` cannot observe a closed window between the
two halves of one event.

**Important — divergence guard.** `record_pick` refuses (raises) when the
spread of ordinals would exceed one in-flight event, validated before it
mutates so a refused pick leaves the accounting untouched. The batcher
CATCHES that ValueError, logs at ERROR and returns None:
`_drive_fill_closed_picks` runs unguarded on the service's 0.5 s poll, so
a raise there would take the whole window wait down over a bookkeeping
fault. Loud and inert instead — and the sibling's next half is what puts
the ordinals back level.

**Minor (c) — total order.** `_ProvenGroup` carries a monotone `sequence`
(default factory, so no construction site can omit it) and
`_pick_sort_key` ends on it. Empty `receipt_id` + equal payload bytes
used to tie on every component and resolve by pool position — the
arrival tie-break the key's own docstring forswears.

**Minor (3) — callback raise.** `pick_training_batch` unflags the claimed
groups if `_emit_training_batch_fn` raises: nothing reached the
assembler, so leaving them flagged made them invisible to BOTH payment
and the close's burn count. They stay pickable; if no later pick takes
them the close burns them. The ordinal stays spent — the sibling's half
of that event is already out and rewinding would desynchronise the
window.

**Minor (f).** `test_an_unknown_rate_falls_back_to_lowest_priority_not_a_crash`
now drives the real `_submit_arrival_proof` (real `rate_of` /
`payload_bytes_of` lookups, only the auto-drain stubbed, like its sibling
in `test_rate_ordered_admission.py`) and asserts the size fallback
through a genuine queue miss.

### T12 tests that pinned the old single-counter semantics (flagged, not contorted)

- `test_pick_pacing.py::test_an_environment_that_refuses_after_a_sibling_picked_logs_error`
  asserted `picks_emitted == 1` after ONE environment picked. Under R37
  that event is incomplete, which is what the ERROR reports: the
  assertion is now `picks_by_environment == {math: 1, code: 0}` and
  `picks_emitted == 0`.
- `test_pick_pacing.py::test_can_pick_is_false_once_the_window_is_closed`
  closed the window with a single bare `record_pick()`; it now records
  both environments and additionally pins that either half alone already
  stops the environment that took it.
- `test_fill_close_and_emit.py`: three bare `record_pick()` calls became
  per-environment, and the two-batcher `is_closed` test now shows a
  leader one event ahead NOT closing the window.

### Verification

`test_fill_window.py` 14, `test_pick_by_rate.py` 20, plus
`test_pick_pacing.py`, `test_fill_close_and_emit.py`,
`test_prove_on_arrival.py`, `test_rate_ordered_admission.py`,
`test_v6_seal_seam.py`, `test_fill_closed_batch_assembler.py`,
`test_grpo_window_batcher.py`, `test_validator_server.py` — 335 together.
Full suite `2377 passed, 13 skipped, 1 deselected`.

### Concern: concurrent edits in the shared worktree

The R38/T10 fix round is editing `service.py`,
`training_payload_queue.py` and `test_pick_pacing.py` in this same tree
while this round ran. Their in-flight rename (`read_step_cursor` →
`fetch_step_cursor` in `_read_trainer_step_cursor`) makes T12's
`_CursorQueue` stub raise, failing every cursor-gated test in
`test_pick_pacing.py`, and their newer red tests reference a rotation
gate that does not exist yet. Those failures are theirs, not this
round's: the numbers above were measured with a local, non-repo pytest
plugin aliasing that one stub method, and every failure without it is an
`AttributeError` on their new names (`fetch_step_cursor`,
`FILL_CLOSED_ROTATION_POLL_SECONDS`, `_arm_fill_closed_rotation_gate`).
Re-run the suite once both rounds have landed.
