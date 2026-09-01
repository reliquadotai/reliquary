# Task 12 report — service pacing + the window-open gate (v6.1, R34/R35/R36)

## Status: DONE. Commits `9c01ef1`, `986a52b`, `5f552c6`, `a8d52a7`,
`226975c`. Full suite on HEAD: **2358 passed**, 13 skipped, 1 failed —
baseline 2334 + exactly the 24 new pins, and the single failure
(`test_admission_isolation.py::test_spawned_worker_deadline_is_terminal`)
is the pre-existing one, identical at the T11 baseline. Every new pin was
watched red first (24/24 red on the first run, with the right
`AttributeError`/`ImportError` reasons, not harness bugs).

## What changed

**Constants (`reliquary/constants.py`).**
`FILL_CLOSED_FIRST_PICK_SECONDS` (30.0) and
`FILL_CLOSED_PICK_PIPELINE_DEPTH` (2), both env-overridable and
range-checked. Neither encodes the trainer's step time (R31): they
describe the pipeline's SHAPE only. Depth is refused outside
`[1, FILL_CLOSED_EMISSIONS_PER_WINDOW]` — 0 would gate pick 1 on a batch
that pick 1 itself produces (a deadlock every window until the backstop),
and a depth past the window's emission count would put every pick back on
the time floor, i.e. the unpaced behaviour the amendment removes.

**`GrpoWindowBatcher.can_pick()`** (the only batcher change, and the
minimum the pick contract needs). True exactly when this environment
holds `B_BATCH` proven-and-unpicked groups and the window is neither
closed nor sealed. Counts to `B_BATCH` and stops rather than sizing the
pool — this runs per environment on the 0.5 s loop against a pool the
budget lets grow to 512. Check-then-pick is race-safe without holding the
lock across both: the proven pool only grows, the only subtraction (a
pick) is on the same poll thread, and `pick_training_batch` re-checks
under `fill_state.lock` regardless.

**A. The pick event loop.** `_drive_fill_closed_picks(batchers)` runs at
the top of `_wait_for_window_seal`'s existing `while True` — before the
per-batcher `poll_deadline`, so the 16th pick closes the window and the
seal lands on the SAME tick instead of one later. It fires at most one
event per tick, and only when:

1. every batcher with a `fill_state` reports `can_pick()` — R36, a pick
   is a WINDOW event; the ready environment is never picked alone;
2. the pacing gate for pick k = `picks_emitted + 1` is open;
3. `fill_state.is_closed()` is False.

Then it calls `pick_training_batch()` on every environment in one pass.
T11's ordinal mechanism advances the window-wide counter once (the first
environment to reach ordinal k moves it, the sibling does not) — pinned
directly by `test_a_window_wide_event_advances_the_count_exactly_once`.
A sibling refusing after a successful one is logged at ERROR naming the
environment and the event number; nothing is advanced twice by the
service (the counter lives in `_claim_pick_chunk`).

**The off-by-ones**, all in `_fill_closed_pick_gate_open`, derived and
pinned on both sides:

* pick k ≤ depth → `now − window_opened ≥ FILL_CLOSED_FIRST_PICK_SECONDS`,
  measured on the BATCHER's clock (`_window_open_age_seconds`), the same
  one `FILL_CLOSED_MAX_SECONDS` uses, so the floor and the backstop can
  never disagree about a window's age. Nothing has been emitted into the
  window's key range yet, so no cursor value could release these.
* pick k > depth → pick k emits batch `k − 1`, so the batch `depth` picks
  back is index `k − depth − 1`, and the gate is
  `cursor >= encoded_window_journal_key(window_start, k − depth − 1)`.
  With depth 2, pick 3 waits on batch 0 and batch 1 is still in the
  trainer's hands — R34's one-batch buffer.
* `>=` not `==`, because the trainer also advances its cursor over
  tombstones, quarantined batches and health skips (T10 generalised it to
  all five sites), and because the encoding is monotone across windows —
  a cursor left in the PREVIOUS window is simply too small, so no
  separate staleness rule is needed (`test_a_stale_cursor_from_an_older_
  window_does_not_release_a_pick`).

Readiness is checked BEFORE the gate, so a fleet that is not producing
costs no cursor I/O at all; the cursor is read at most once per tick and
only on the branch that needs it (both pinned).

**B. The next window opens on checkpoint detection (R35).**
`_wait_for_fill_closed_checkpoint()` sits at the loop's `"open"` stage,
before `_open_window()`. It is armed by
`_arm_fill_closed_checkpoint_gate()`, called right after the seal while
the closed window's assembler is still the current one, and ONLY when
that window actually emitted batches (`next_batch_index > 0`) — a window
that emitted nothing produces no checkpoint N+1, and waiting for one
would burn a whole backstop for nothing. The baseline is the revision the
closed window collected against.

Detection, not installation, releases the gate: a freshly polled
candidate manifest counts, and the multi-gigabyte staging download it
starts overlaps the next window's collection. The revision-detection path
DOES exist validator-side — `CheckpointIntake.poll()` on
`reliquary/training/candidate-manifest.json`, then `stage`, then
`_swap_staged_checkpoint` installs it into `_checkpoint_store` — so no
channel was invented. `_detached_checkpoint_tick`'s poll-and-stage half
was extracted to `_poll_and_stage_checkpoint_candidate()` so the gate
reuses it WITHOUT the swap; the swap keeps its own serial beat. Three
detection sources, cheapest first: the installed manifest, the intake
snapshot (staged / staging), the poll itself.

Bounded by `FILL_CLOSED_MAX_SECONDS` with `_wait_for_window_seal`'s
discipline (explicit poll interval, `await asyncio.sleep`, no busy loop).
The poll interval is a module-level `FILL_CLOSED_CHECKPOINT_POLL_SECONDS
= 2.0`, deliberately NOT the seal loop's 0.5 s: each ask is an R2 GET.
On expiry the next window opens on the same revision and the ERROR line
says exactly what follows (the trainer is dead, its cursor will not move
either, that window takes `depth` picks on the floor and ends on its own
backstop).

With `FILL_CLOSED_ENABLED` off, `_drive_fill_closed_picks` returns
immediately and `_wait_for_fill_closed_checkpoint` returns `"disabled"`
before touching anything — rotation is byte-identical to today, and the
existing suites confirm it.

## Tests

`tests/unit/test_pick_pacing.py`, 24 pins: constants defaults; pick 1
before/after the floor; pick depth+1 on both cursor boundary sides; pick
depth+2 waits for batch 1 (the off-by-one walks); one env short blocks
the whole event; one event = one window-wide count; N events close the
window and `poll_deadline` seals; the backstop still seals a stalled
window; absent cursor still lets the first `depth` picks fire and stops
the rest; a stale older-window cursor releases nothing; cursor read once
per tick and never while gated on the clock; a pool-short window reads
nothing; a refusing sibling logs ERROR; gate off fires nothing; legacy
`fill_state is None` batchers are skipped; `can_pick` true/false
transitions and false after close; and six rotation pins (waits then
releases on the third poll, returns at once when the swap already
happened, staged-but-unswapped counts, bounded by the backstop, not armed
without a baseline, disabled with the gate off, and the arming rule).

One test-harness note worth carrying: `encoded_window_journal_key` is
itself gated on `FILL_CLOSED_ENABLED`, and with it off it collapses every
batch of a window onto the bare window number. Any cursor test must
monkeypatch `training_payload_queue.FILL_CLOSED_ENABLED` to True (the
established pattern in `test_v6_emission.py` /
`test_fill_closed_batch_assembler.py`) or it silently compares 500 to
500. Done in `_two_env_window`.

## Concerns

1. **The cursor never reaches the validator.** `read_step_cursor()` reads
   the queue's LOCAL `step-cursor.json`; the trainer writes that file on
   the train worker and the drain UPLOADS it to
   `reliquary/training/step-cursor.json`. Nothing downloads it back —
   grepped: `step_cursor_key` has exactly one producer-side user and no
   consumer. So in production today this returns None forever, every v6
   window takes `depth` picks on the time floor and then sits until
   `FILL_CLOSED_MAX_SECONDS`. T12's seam is right and the behaviour is
   pinned (`test_an_absent_cursor_still_lets_the_first_picks_fire`), but
   the amendment does not actually pace until the missing half — one R2
   GET, a fetch-then-read on the queue — is built. Flagged in
   `_read_trainer_step_cursor`'s docstring as a visible GAP rather than
   invented here: cadence, cost and threading are transport decisions
   that belong with the rest of the cursor transport (T10's file, which
   T12 does not own). This is the one thing standing between the branch
   and a working v6.1.

2. **Pipelined mode + the open gate.** The R35 wait sits BEFORE
   `_open_window()`, so in `PIPELINED_WINDOWS` mode it runs before the
   stashed GPU half. That is safe — in v6 the training payloads are
   written mid-window by the assembler, not by the GPU half, so the
   checkpoint being waited on does not depend on the stashed half — but
   it does mean the stashed window's archive/payment is delayed by the
   wait. Serial mode (today's default, `PIPELINED_WINDOWS` is False) is
   unaffected. If pipelining is ever armed with v6, this ordering wants a
   second look.

3. **Emergency freeze.** `RELIQUARY_DISABLE_TRAIN` stops the validator
   adopting checkpoints, so during a freeze the R35 gate will wait a full
   `FILL_CLOSED_MAX_SECONDS` per window before opening. The backstop
   bounds it and the ERROR line explains it, but a freeze-aware skip (3
   lines in `_wait_for_fill_closed_checkpoint`) would turn a 30-minutes-
   per-window stall into a no-op. Left out to keep T12's scope to the
   brief; cheap to add with its own pin.

4. **A mid-window swap over-arms the gate.** If a checkpoint swap lands
   between a v6 window's build and its close, the baseline captured at
   close is the NEW revision, so the next window waits for yet another
   checkpoint. Conservative, bounded by the backstop, and in practice
   unreachable in serial mode (`_publication_due_next_half` routes
   exactly that case serially).

## Fix round 1 (R39)

Commit `8c4faf3`. All 6 findings addressed; 3 new pins (27 in
`test_pick_pacing.py`), the underfilled-window case watched red first.

**Important 1 — R39 replaces the revision-baseline gate.** The gate now
arms on the *journal key of the last batch the closed window emitted*
(`assembler.next_batch_index - 1`, read at seal before `close()` pads the
range) and releases when the trainer's cursor — remote since R38 —
reaches it. `_arm_fill_closed_checkpoint_gate` /
`_wait_for_fill_closed_checkpoint` / `_fill_closed_checkpoint_detected` /
`_fill_closed_checkpoint_baseline` are gone, replaced by
`_arm_fill_closed_rotation_gate` / `_wait_for_fill_closed_rotation` /
`_fill_closed_rotation_key`; `FILL_CLOSED_CHECKPOINT_POLL_SECONDS` →
`FILL_CLOSED_ROTATION_POLL_SECONDS`. Zero emitted disarms outright.
Neither old failure mode survives: an underfilled window waits only for
its own three batches, and a mid-window publish cannot arm anything
because publications are no longer read. The arming is *consumed* by the
wait (one close, one wait), so a stale key can never gate a later window;
if `window_start` is somehow unavailable the gate disarms rather than
guess a key nobody will consume.

**Important 2 — freeze contract.** The gate no longer polls or stages
anything, so no staging download can start under a freeze. Beyond that,
`RELIQUARY_DISABLE_TRAIN` now skips the gate entirely with a WARNING
naming the key nothing will consume — otherwise the freeze would cost a
full backstop of dead air per window during the incident it exists to
contain. `_poll_and_stage_checkpoint_candidate` is kept: it is still
`_detached_checkpoint_tick`'s own body, and checkpoint adoption stays on
that serial beat untouched.

**Minors.** (1) A *total* refusal after readiness passed now logs ERROR
like a partial one — same broken invariant — and the message no longer
claims an incomplete event advances the count, which R37 made false
(`picks_emitted` is the MIN over ordinals). (2)
`test_sixteen_events_close_the_window` no longer jumps the gates with
`cursor=1e9`: it is now
`test_every_event_walks_its_own_cursor_boundary_and_closes`, stepping
every pick past the depth over both sides of its own boundary, so the
window's LAST event (k = target → batch k−3) is pinned like the rest.
(3) The wait publishes a `fill_closed_rotation_wait` preparation stage,
so a held rotation is legible on `/state` instead of looking like a hung
validator. (4) The real guarantee is stated in
`_drive_fill_closed_picks`: single-threaded loop ownership of
readiness+pick+seal is what excludes the interleaving; pool-only-grows
merely makes a stale `False` harmless, never a `True` wrong.

**Concerns.**
- **Minor 4 is only half done.** The accurate statement went into
  `_drive_fill_closed_picks` (service.py, the caller where the guarantee
  actually lives). `can_pick`'s own docstring in `batcher.py` still leads
  with "the proven pool only ever GROWS" as the safety argument; that
  file was explicitly off-limits this round. One sentence for whoever
  owns `batcher.py` next.
- **The remainder batch.** Arming reads `next_batch_index` at seal, so
  the gate waits for the last *mid-window* batch. `close()` may afterwards
  force out one more real partial batch, which the gate does not wait for
  — at most one key, inside the depth−1 tolerance. Waiting for it would
  mean arming inside the GPU half, which the pipelined path defers.
