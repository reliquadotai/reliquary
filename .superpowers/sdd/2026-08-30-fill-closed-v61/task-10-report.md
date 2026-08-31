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
