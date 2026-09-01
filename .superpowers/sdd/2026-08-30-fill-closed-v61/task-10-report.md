# Task 10 report — trainer step cursor (v6.1 trainer-paced picks)

## Status: DONE, committed (fix round 1 addresses the confirmed cli-wiring gap)

## Files touched (mine only)
- `reliquary/infrastructure/training_payload_queue.py`
- `reliquary/trainer/worker.py`
- `tests/unit/test_training_payload_queue.py`
- `tests/unit/test_trainer_worker.py`

No new module was needed — everything fit naturally in the queue and the worker.

## What changed

`TrainingPayloadQueue` gains `step_cursor_key()`, `write_step_cursor(journal_key)`,
`read_step_cursor()`. The cursor is a single object `step-cursor.json`
(`{"journal_key": int, "written_at": float}`), atomic temp+rename write via the
existing `_enqueue` helper — overwritten in place, never a growing log.
`read_step_cursor` swallows `OSError`/`ValueError`/`TypeError`/`KeyError` and
returns `None` for absent/torn/corrupt files (never raises). The cursor rides
the *same* drain/upload transport as payloads (`_pending`/`_key_for`/
`_try_upload`), but unlike a payload it is **not** deleted after a successful
upload — it stays local so the next step's write replaces it in place, and
re-uploads idempotently each drain cycle until it changes.

`TrainerWorker` gets a new optional constructor arg `cursor_writer: Callable[[int],
None] | None = None` (default `None` → no-op, so existing callers/tests are
unaffected). All 5 places the journal cursor previously did `self.cursor +=
self.stride` (trained payload, tombstone, quarantined-without-epoch, health-skip
exception, epoch-aborted) were unified into one `_advance_cursor()` helper that
increments and then calls `_write_cursor()`, which calls `cursor_writer` inside a
`try/except Exception` that logs and swallows — a telemetry failure never fails
the training step.

**Ruling on tombstones/quarantine (per task item 2):** they MUST bump the
cursor. Rationale documented in `_advance_cursor`'s docstring: the validator
paces picks on *consumption* of journal keys, not on optimizer steps
specifically — a tombstoned or BFT-quarantined key is never coming back to be
trained, so not bumping there would stall the pacer forever on the first
skipped/quarantined window. I generalized this to all 5 cursor-advance sites
(not just tombstone) for the same reason — a quarantined-without-epoch window
or a health-skip (`TrainingStepSkipped`) is exactly as "never coming back" as
a tombstone.

**Profile-agnostic (per task item 3):** `worker.py` never references
`FILL_CLOSED_ENABLED` or any v6-specific constant. With `cursor_writer=None`
(the default, unless a caller wires one in) it is a pure no-op — a cursor on
v5 is harmless, additive telemetry, exactly as specified.

## Not done here (explicitly out of scope)
- `reliquary/trainer/cli.py` wiring (constructing a `TrainingPayloadQueue` on
  the trainer host, passing `cursor_writer=queue.write_step_cursor`, running
  its `run_forever()` drain loop there) — not in my file list, left for
  integration/T12.
- The validator-side read of the cursor to gate picks — T12 (service.py /
  fill_window.py), explicitly another implementer's files.

## Tests (TDD, watched red first)
- `tests/unit/test_training_payload_queue.py`: 10 pre-existing + 8 new (round
  trip, overwrite-not-append, absent→None, corrupt JSON→None, missing
  field→None, key naming, same-transport upload without local deletion) — all
  new tests failed on import (`step_cursor_key` didn't exist) before the
  implementation, now pass.
- `tests/unit/test_trainer_worker.py`: 10 pre-existing + 5 new (writes after
  trained payload, after tombstone, after quarantined window, write failure
  doesn't fail the step, no-writer-configured no-op) — 4 of 5 failed red
  (`TypeError: unexpected keyword argument 'cursor_writer'`) before the
  implementation, now pass.
- `tests/unit/test_trainer_journal.py`: unaffected, still green (41/41 across
  the three files together).
- `tests/unit/test_v6_seal_seam.py`, `test_v6_emission.py`,
  `test_training_payload_writer.py`, `test_fill_closed_batch_assembler.py`,
  `tests/integration/test_detached_trainer_flow.py`: all green (58/58),
  including the integration flow test that drives `TrainerWorker` against a
  real `WindowJournal`.

Full-suite run (`pytest -q`) launched; result to be appended below.

## Full-suite result

`cd /tmp/claude-1000/-home-ubuntu-Catalyst/wt-v61 && TMPDIR=/tmp
/home/ubuntu/Catalyst/.venv/bin/python -m pytest -q`:

```
17 failed, 2300 passed, 13 skipped, 8 warnings in 222.69s
```

All 17 failures are in `tests/unit/test_admission_isolation.py`,
`tests/unit/test_fill_close_and_emit.py`, `tests/unit/test_prove_on_arrival.py`,
`tests/unit/test_rate_ordered_admission.py` — none of them mine. This
worktree is shared with the other implementer (T9), whose uncommitted,
in-flight changes to `fill_window.py`/`service.py`/`batcher.py` are visible
in `git status` alongside mine right now; those failures belong to that
still-in-progress work, not to anything I touched. **Zero failures in the
files I touched** (`training_payload_queue.py`, `worker.py`, and their two
test files) — 2300 passed baseline is intact modulo the other implementer's
work-in-progress additions/regressions.

## Commit
Committed with `git add` by filename (only the 5 files listed above under
"Files touched"), leaving the other implementer's uncommitted changes to
`constants.py`/`fill_window.py`/`service.py`/`batcher.py`/their test files
untouched in the working tree.

## Fix round 1 (cli wiring)

Reviewer confirmed the gap: `reliquary/trainer/cli.py::run_train_worker`
never constructed a `TrainingPayloadQueue` and its `TrainerWorker(...)` call
omitted `cursor_writer`, so the cursor was dead code in production.

Fix, in `reliquary/trainer/cli.py` only:
- Added `_build_cursor_queue(state_dir)`: a 5-line factory returning a
  `TrainingPayloadQueue` scoped at `state_dir / "cursor_queue"` (no such
  queue existed in cli.py's scope before, so this is a new instance, not a
  reused one — per the reviewer's "if one exists... else a new one"
  instruction). Named distinctly from the validator's own default
  (`pending_training_payloads`) so the two queues never glob each other's
  files even if `RELIQUARY_STATE_DIR`/`RELIQUARY_TRAINER_STATE_DIR` ever
  pointed at the same root.
- Added `_drain_cursor_queue_forever(queue)`: `asyncio.run(queue.run_forever())`.
  `run_train_worker`'s main loop is a plain synchronous `while True` (no
  persistent asyncio event loop — unlike the validator's async service,
  which schedules `run_forever` as a task on its already-running loop), so
  this runs on its own daemon thread instead.
- In `run_train_worker`: construct `cursor_queue = _build_cursor_queue(state_dir)`,
  start `_drain_cursor_queue_forever` on a daemon `threading.Thread` before
  constructing the worker, and pass `cursor_writer=cursor_queue.write_step_cursor`
  into the `TrainerWorker(...)` call. This makes the local `step-cursor.json`
  actually reach R2 in production.

`run_train_worker` itself still cannot be unit-tested (loads a real model on
`cuda:0`), so per the reviewer's instruction only the extracted factory is
covered: new `tests/unit/test_cli_cursor_queue.py` (4 tests — returns a
`TrainingPayloadQueue`, scoped under the given `state_dir`, subdir name
distinct from the validator's default, and a full write/read round trip
through it). Confirmed red-by-construction: `_build_cursor_queue` does not
exist in `git show HEAD:reliquary/trainer/cli.py` (grep count 0), so the
import in these tests would have failed before the fix; all 4 pass after.

`reliquary.trainer.cli` still imports cleanly and exposes both new
functions (`python -c "import reliquary.trainer.cli as m; m._build_cursor_queue;
m._drain_cursor_queue_forever"` — ok).

Full targeted run: `test_cli_cursor_queue.py` + `test_trainer_worker.py` +
`test_training_payload_queue.py` + `test_trainer_journal.py` +
`test_cli_train_worker.py` → **52 passed**. Did not re-run the full suite
this round (only `cli.py` + one new test file touched, no shared-file risk
with the other implementer's in-flight fill_window/service/batcher work).

## Fix round 2 (R38 cursor round-trip)

T12 found the real blocking hole: the drain uploads `step-cursor.json` to
R2 from the trainer host, but nothing validator-side ever downloads it —
`read_step_cursor()` only reads the LOCAL file, which on the validator
never exists (nothing writes it there). Ruling R38: the validator reads
the cursor via one bounded R2 GET, transport code living in
`training_payload_queue.py`.

Changes, `reliquary/infrastructure/training_payload_queue.py`:
- Extracted `_r2_client(config)`: the shared boto3-client-from-env-vars
  construction, factored out of `_default_delete` so a new direct R2 call
  doesn't re-derive the account/endpoint/credential resolution a third
  time ("reuse its client construction, do not build a second config
  path"). `_default_delete` now calls it.
- Added `_default_fetch_step_cursor() -> bytes | None`: sync boto3
  GetObject for `step_cursor_key()`, through `_r2_client`, but with a
  **short, non-retrying** `Config` (`connect_timeout=2, read_timeout=2,
  max_attempts=1`) — deliberately tighter than the upload/delete path's
  15s/30s/3-retries, since this backs a *synchronous* call on the
  validator's `_wait_for_window_seal` 0.5s poll cadence, not a background
  drain. Swallows everything to `None`, never raises.
- Added `_parse_step_cursor(raw)` — the corrupt/torn/wrong-schema →
  `None` parse, factored out and now shared by both `read_step_cursor`
  (local) and the new `fetch_step_cursor` (remote), same behavior as
  before for `read_step_cursor`.
- Added `TrainingPayloadQueue.fetch_step_cursor(fetch_fn=None)`: one
  bounded remote GET, defaulting to `_default_fetch_step_cursor`, same
  "never raises, absent/torn/wrong-schema/network-error/timeout → None"
  contract as `read_step_cursor`. Docstring states explicitly that
  throttling (at most once per gated poll tick) lives with the caller —
  this method neither caches nor rate-limits.

`reliquary/validator/service.py`: the one instructed line —
`_read_trainer_step_cursor` now calls `.fetch_step_cursor()` instead of
`.read_step_cursor()`. Nothing else in the file touched (confirmed via
`git diff` — single-line diff).

TDD: watched red (9 new tests failed with `AttributeError:
_default_fetch_step_cursor` before the change), then green. Covers: a
true two-instance round trip (writer drains to a fake remote-store dict,
a SEPARATE reader queue instance — different `queue_dir`, simulating the
validator/trainer host split — fetches it back, proving `read_step_cursor`
alone cannot see it); absent key, corrupt body, missing field, and a
raised network/timeout exception all → `None`; the default `fetch_fn`
wiring; and two tests pinning the actual production `_default_fetch_step_cursor`
(short/non-retrying Config, correct key, swallows a `get_object`
exception) plus one proving `_default_delete` and
`_default_fetch_step_cursor` share the same client-construction call
(same resolved endpoint from `_r2_client`). No real R2 credentials
needed — mirrors the existing drain tests' injectable-callable seam
(`upload_fn`/`delete_fn` style) for the two production-wiring tests,
`boto3.client` itself is monkeypatched (same technique already used
nowhere else in this file's tests, but it's the standard way to pin a
lazily-imported call without live credentials).

`tests/unit/test_training_payload_queue.py`: 25/25 pass (16 pre-existing
+ 9 new). Also re-ran `test_cli_cursor_queue.py`, `test_trainer_worker.py`,
`test_trainer_journal.py`, `test_pick_pacing.py`, `test_pick_by_rate.py`,
`test_fill_closed_batch_assembler.py` together: 26 failures, ALL in
`test_pick_pacing.py`/`test_pick_by_rate.py`
(`FillState.record_pick() missing 1 required positional argument:
'environment'`) — `batcher.py` calling the pre-R37 single-arg
`record_pick()` against `fill_window.py`'s in-flight (uncommitted) R37
signature change; both files are explicitly off-limits to me and mid-edit
by the other implementer, not something I touched. Zero failures in any
file I touched.

Full-suite result appended below once the background run finishes.

Additional fix in the same round: my one-line `service.py` change
(`read_step_cursor()` -> `fetch_step_cursor()`) broke T12's
`tests/unit/test_pick_pacing.py::_CursorQueue` test double, which only
implemented `read_step_cursor`. Renamed its method to
`fetch_step_cursor(self, fetch_fn=None)` (same body, one comment added) —
a direct, mechanical consequence of the instructed call-site change, not
a `fill_window.py`/`batcher.py` edit. `test_pick_pacing.py`: 24/24 pass
after. The remaining `record_pick()` TypeError failures seen mid-round
were the R37 fixer's WIP mid-commit; `8a85ff0` landed and resolved them
independently — confirmed by a second full run showing test_pick_pacing.py
fully green.

Committed without waiting for the final full-suite run (per coordinator:
the controller runs it at the final pass). Targeted runs green:
`test_training_payload_queue.py` 25/25, `test_pick_pacing.py` 24/24,
`test_cli_cursor_queue.py`/`test_trainer_worker.py`/`test_trainer_journal.py`
all green.

## Fix round 3 (final-review #1/#3/#4)

R40. Whole-amendment review found one Critical and two Importants, all in
my files.

**#1 Critical — cursor GET blocking the miner-serving event loop.** Fixed
all three parts, in `training_payload_queue.py`:
- (a) Memoised client: `_cached_step_cursor_client()`, a module-level
  singleton (`_STEP_CURSOR_CLIENT`) built once under a lock (double-checked
  locking), reused by every `_default_fetch_step_cursor()` call instead of
  a fresh `boto3.client(...)` (~50-200ms) each time.
- (b) Non-blocking call site: chose **fire-and-collect over
  `asyncio.to_thread`**. Reason: there are now TWO call sites through
  `_read_trainer_step_cursor` (the pick gate, and — new since R39 landed
  mid-amendment — the rotation-wait's `consumed()` closure at
  `service.py:~1547`), and both call it as a **plain synchronous method**.
  Threading `await`/`async def` through both chains (and their sync
  wrapper closures) would ripple across two independent call chains in a
  file I do not otherwise own this round. Fire-and-collect fixes both
  call sites with **zero changes to service.py's call shape**: `fetch_step_cursor`
  now returns the last COMPLETED value immediately and kicks a background
  `threading.Thread` (in-flight-guarded, so never more than one
  concurrent fetch) only when the cached value is stale. A one-tick-stale
  cursor is harmless — the pick gate's comparison is `>=`, so staleness
  only ever holds a pick back a little longer, never opens one early.
- (c) ~2s value cache: `_STEP_CURSOR_CACHE_TTL_SECONDS = 2.0` (matches
  `service.FILL_CLOSED_ROTATION_POLL_SECONDS` by value; importing it
  would be circular). Reversed `fetch_step_cursor`'s docstring, which
  used to explicitly refuse caching — throttling now lives in the queue,
  not the caller. On a fetch FAILURE the cache is left at its last good
  value (never regressed to `None`), so a transient R2 hiccup degrades to
  "slightly stale" rather than repeatedly stalling the gate; the TTL
  clock still resets on every completion (success or failure), giving a
  natural ~2s retry cadence during a real outage instead of hammering
  every poll tick.

**#3 Important — stale gap docstring.** `service._read_trainer_step_cursor`'s
docstring said "nothing downloads it back on this side" three lines above
the code that does. Rewrote it to describe the actual R38/R40 transport
(fire-and-collect, cached, safe on every poll tick). One-paragraph,
docstring-only diff — confirmed via `git diff` (no other line in the file
touched).

**#4 Important — unchanged-object upload forever, every profile.** Added
`self._step_cursor_last_uploaded_body` (per-queue-instance). `_try_upload`
now short-circuits (no network call, no counters moved) when
`path.name == step-cursor.json` and its freshly-read local body equals the
last body actually uploaded — an idle trainer between real steps no
longer PUTs an identical object every ~2s forever on v4/v5 too (~43k
Class A ops/day/trainer before this fix). A genuinely new step (different
`journal_key` or `written_at`) still uploads promptly.

TDD, watched red first (14 tests failed before the change: `AttributeError:
_STEP_CURSOR_CLIENT` and a same-body-twice assertion). Now green — new/
updated tests cover: client memoisation (two `_default_fetch_step_cursor()`
calls, one `boto3.client` construction); the value cache (repeated calls
within the TTL window → one fetch); the in-flight guard (two calls before
the first fetch resolves → one fetch, via a `threading.Event` gate, no
sleep-based flakiness); failure keeps the last good value instead of
regressing to `None`; and the changed-only upload (same body twice → one
PUT; a real rewrite → a second PUT). All existing `fetch_step_cursor`
tests from round 2 were updated for the new fire-and-collect contract
(kick + join + re-read, instead of expecting the fetched value on the
same call) rather than deleted.

`tests/unit/test_training_payload_queue.py`: 30/30. `test_pick_pacing.py`:
24/24 (exercises BOTH `_read_trainer_step_cursor` call sites — the pick
gate and the R39 rotation-wait — through the same `_CursorQueue` fake).
`test_cli_cursor_queue.py`: 4/4. Full-suite result appended below once the
background run finishes.
