# Minimum incentive floor — implementation report

Worktree: `/home/ubuntu/catalyst-tasks`, branch `feat/multi-task`.
Commit: `68e4f084677aedccd7d5647aea1b11ceeaf1f2ff` —
"feat(weight_only): enforce a 2% minimum incentive floor, freed mass burns"

Round 2 commit: `1b3aeb4c5ef85a0b21028e4fa9711b2ef947c4fc` —
"fix(weight_only): move the min-incentive floor out of _replay_ema, into submit_once"

## What changed

1. **`reliquary/constants.py`** — added `MIN_INCENTIVE_SHARE` (default `0.02`,
   overridable via `RELIQUARY_MIN_INCENTIVE_SHARE`, `0` disables the floor)
   right after `EMA_ALPHA` in the `SCORING` section, with the same
   env-override / validation pattern used by neighboring constants in that
   file. Setting it to `0` is a pure runtime kill switch — no image rebuild.

2. **`reliquary/validator/weight_only.py`** — imported `MIN_INCENTIVE_SHARE`
   at module scope alongside `EMA_ALPHA`/`EPOCH_SUBMIT_LEAD_BLOCKS`/
   `POLL_INTERVAL_SECONDS`, and applied it as the very last step of
   `_replay_ema`, after the per-task cap clamp (`_clamp_to_cap`, applied per
   task before combining) and after the existing global `total > 1.0`
   rescale backstop:

   ```python
   if MIN_INCENTIVE_SHARE > 0.0:
       combined = {
           hk: v for hk, v in combined.items() if v >= MIN_INCENTIVE_SHARE
       }
   return combined
   ```

   Confirmed before relying on it: `combined`'s values are already fractions
   of the whole pool (each task's own EMA sums to at most that task's cap,
   see `_replay_ema`'s docstring and the per-task loop), so comparing them
   directly to `MIN_INCENTIVE_SHARE` is correct with no extra normalization
   needed. Downstream, `_submit_weights` computes
   `registered_total` from whatever survives in `combined` and sets
   `burn_weight = max(0.0, 1.0 - registered_total)` — dropping entries here
   (rather than renormalizing) is exactly what lets that existing burn
   mechanism absorb the freed mass instead of it flowing to survivors.

3. **`tests/unit/test_weight_reader_multi_task.py`** — added 4 new
   synchronous tests using the existing module-level `_archive(window, task,
   rewards)` helper and `WeightOnlyValidator._replay_ema`/`_merge_archives`
   static-helper style already used in that file, monkeypatching
   `reliquary.validator.weight_only.MIN_INCENTIVE_SHARE` (imported as
   `from reliquary.validator import weight_only`) the way the task specified:
   - `test_a_hotkey_above_the_floor_keeps_its_value_one_below_is_absent`
   - `test_a_hotkey_exactly_on_the_floor_is_kept` (pins the floor to the
     exact single-window EMA value to test the `>=` boundary)
   - `test_freed_mass_is_burned_not_redistributed` — the load-bearing test:
     asserts the survivor's value is byte-identical (`==`) between a
     floor-disabled run and a floor-enabled run of the *same* archives, and
     that the summed total drops. This fails loudly if anyone ever
     renormalizes survivors instead of dropping.
   - `test_floor_disabled_drops_nothing` (floor monkeypatched to `0.0`)

## Test-first sequence

1. Wrote the 4 new tests against the not-yet-existing `MIN_INCENTIVE_SHARE`
   import and confirmed they failed for the right reason
   (`AttributeError: ... has no attribute 'MIN_INCENTIVE_SHARE'` — the
   module didn't have the symbol yet, not a logic bug):

   ```
   $ python -m pytest tests/unit/test_weight_reader_multi_task.py -q
   ...
   FAILED tests/unit/test_weight_reader_multi_task.py::test_a_hotkey_above_the_floor_keeps_its_value_one_below_is_absent
   FAILED tests/unit/test_weight_reader_multi_task.py::test_a_hotkey_exactly_on_the_floor_is_kept
   FAILED tests/unit/test_weight_reader_multi_task.py::test_freed_mass_is_burned_not_redistributed
   FAILED tests/unit/test_weight_reader_multi_task.py::test_floor_disabled_drops_nothing
   4 failed, 12 passed in 9.18s
   ```

2. Implemented the two source changes above.

3. Re-ran: all 16 passed.

## Verification commands (run one at a time, sequential, no xdist/timeout)

```
$ python -m pytest tests/unit/test_weight_reader_multi_task.py -q
................                                                         [100%]
16 passed in 8.49s
```

```
$ python -m pytest tests/unit/test_weight_only_validator.py -q
................ [28 dots for passing tests interleaved] ...
6 failed, 28 passed in 9.85s

FAILED tests/unit/test_weight_only_validator.py::test_replay_ema_deterministic
FAILED tests/unit/test_weight_only_validator.py::test_replay_ema_reads_rewards_by_hotkey_field
FAILED tests/unit/test_weight_only_validator.py::test_replay_ema_k_way_split_applies_on_chain
FAILED tests/unit/test_weight_only_validator.py::test_replay_ema_boundary_fair_split_applies_on_chain
FAILED tests/unit/test_weight_only_validator.py::test_replay_ema_empty_rewards_decays_existing_ema
FAILED tests/unit/test_weight_only_validator.py::test_replay_ema_conservation_bound
```

  Root-cause confirmation — re-ran the same file with the floor disabled via
  env var (no code changes), which restores all 34 passes, proving these 6
  failures are caused solely by the new default 2% floor and are not
  pre-existing:

  ```
  $ RELIQUARY_MIN_INCENTIVE_SHARE=0 python -m pytest tests/unit/test_weight_only_validator.py -q
  ..................................                                       [100%]
  34 passed in 9.88s
  ```

```
$ python -m pytest tests/unit/test_v1_cutover.py -q
................                                                         [100%]
16 passed in 10.03s
```

```
$ python -m pytest tests/unit/test_archive_window_content.py -q
.......                                                                  [100%]
7 passed in 9.59s
```

```
$ python -m pytest tests/unit/test_task_archive_namespace.py -q
......................                                                   [100%]
22 passed in 10.08s
```

The full pytest suite was **not** run, per instructions (no swap on this
box; a full run has OOM-killed a production container here before).

## Assertions that moved — DELIBERATELY NOT EDITED, needs a ruling

All 6 in `tests/unit/test_weight_only_validator.py` are single-window (or
otherwise pre-steady-state) EMA replay tests. Each pins a raw
`EMA_ALPHA × reward` value from *one* window of history — far below what a
real, consistently-paid hotkey converges to over 72 windows — so every value
they assert lands under the new default 2% floor and the hotkey is now
dropped outright rather than tracked at that tiny fraction. I computed the
exact pre-floor values (via `RELIQUARY_MIN_INCENTIVE_SHARE=0`) to show old
vs. new precisely; I did not touch the test file.

1. **`test_replay_ema_deterministic`** (line 48)
   `assert "alice" in ema`
   - Old: `ema == {"bob": 0.006664233182611556, "alice": 0.006570406948796582, "carol": 0.003424657534246575}`
   - New: `ema == {}` (all three below 0.02) → `AssertionError: assert 'alice' in {}`

2. **`test_replay_ema_reads_rewards_by_hotkey_field`** (line 606)
   `assert abs(ema["alice"] - EMA_ALPHA * 0.05) < 1e-9`
   - Old: `ema["alice"] == 0.0013698630136986301` (== `EMA_ALPHA * 0.05`); `bob` same, `carol == 0.0017123287671232876`
   - New: all three dropped (all < 0.02) → `KeyError: 'alice'`

3. **`test_replay_ema_k_way_split_applies_on_chain`** (line 632)
   `assert abs(ema[f"sybil{i}"] - expected) < 1e-9` for `i in range(5)`,
   `expected = EMA_ALPHA * slot_share / 5 == 0.0006849315068493151`
   - Old: each `sybil0..4 == 0.0006849315068493151`
   - New: all 5 dropped → `KeyError: 'sybil0'`

4. **`test_replay_ema_boundary_fair_split_applies_on_chain`** (line 670)
   `assert abs(ema[f"r1_p{i}"] - EMA_ALPHA * slot_share) < 1e-9` for
   `i in range(B_BATCH - 2)` (also implicitly covers `boundary_0..3`)
   - Old: `r1_p* == 0.003424657534246575`, `boundary_* == 0.0017123287671232876`
   - New: all 14 `r1_p*` and all 4 `boundary_*` dropped → `KeyError: 'r1_p0'`

5. **`test_replay_ema_empty_rewards_decays_existing_ema`** (line 696)
   `assert abs(ema["alice"] - expected) < 1e-9`,
   `expected = (1 - EMA_ALPHA) * EMA_ALPHA * 0.5 == 0.013323325201726402`
   - Old: `ema["alice"] == 0.013323325201726402`
   - New: dropped (0.01332 < 0.02) → `KeyError: 'alice'`

6. **`test_replay_ema_conservation_bound`** (line 716)
   `assert abs(sum(ema.values()) - expected_total) < 1e-9`,
   `expected_total = EMA_ALPHA * 0.8 == 0.02191780821917808`
   (8 hotkeys, each individually `EMA_ALPHA * 0.1 == 0.002739726...`)
   - Old: `sum(ema.values()) == 0.02191780821917808`
   - New: every individual hotkey is below 0.02 so all 8 are dropped →
     `sum(ema.values()) == 0` (`ema == {}`) →
     `AssertionError: 0.02191780821917808 < 1e-09` fails
     (`abs(0 - 0.02191780821917808)`)

These six all test the raw single-window EMA arithmetic in isolation, not a
converged live payout, so whether they should be updated to reflect the new
floor (or reworked to replay enough windows to clear it, or something else)
is a judgment call for the operator, not something to silently paper over.

## Concerns / notes for the operator

- The floor is compared against `combined`'s values directly, which are pool
  fractions — confirmed against `_replay_ema`'s own docstring and the
  per-task clamp/combine logic before writing the filter, per the task's
  instruction to verify rather than assume.
- `tests/unit/test_weight_only_validator.py`'s 6 broken assertions above all
  come from single-window (non-steady-state) fixtures; no assertion in any
  of the other 4 required test files, nor in the new
  `test_weight_reader_multi_task.py` tests, was affected.
- Nothing was pushed or merged; the commit is local to
  `/home/ubuntu/catalyst-tasks` on `feat/multi-task`.

## Round 2 — moved the floor out of `_replay_ema`, into `submit_once`

Diagnosis accepted: `_replay_ema` is pure EMA arithmetic; the floor is a
payment-path policy and putting it inside the arithmetic function made
every raw single-window EMA unit test into a floor test.

### Other callers of `_replay_ema`?

Checked before changing anything:

```
$ grep -rn "_replay_ema" --include=*.py . | grep -v tests/
reliquary/validator/weight_only.py:241:        ema = self._replay_ema(archives, caps=self._caps_by_task(declared))
reliquary/validator/weight_only.py:304:    def _replay_ema(
reliquary/validator/emission_price.py:157:    replayable: ``_replay_ema`` reads a bounded slice of archives, so a price
reliquary/validator/emission_price.py:221:    Aborted windows are skipped for the same reason ``_replay_ema`` skips them:
```

Line 241 is inside `submit_once` — the only real call site.
`reliquary/validator/emission_price.py`'s two hits are comments/docstring
prose referencing `_replay_ema`'s behavior for context, not calls to it
(confirmed by reading both — no `import` of `WeightOnlyValidator` or
`_replay_ema` in that file). **There is exactly one payment path**
(`submit_once`), so the floor only needed to move to one place; no second
finding to report.

### What moved where

1. **`reliquary/validator/weight_only.py`** — `_replay_ema` reverted to be
   byte-for-byte identical to its pre-floor form. Verified directly:

   ```
   $ git diff HEAD~1 -- reliquary/validator/weight_only.py
   ```

   shows the only changes versus the commit before the floor existed are
   (a) the `MIN_INCENTIVE_SHARE` import staying in the `from reliquary.constants
   import (...)` block, and (b) the floor block now living in `submit_once`,
   applied to `miner_weights` right after `ema = self._replay_ema(...)` /
   `miner_weights = dict(ema)` and before `subtensor = await
   chain.get_subtensor()` (i.e. before `_submit_weights` is ever called):

   ```python
   ema = self._replay_ema(archives, caps=self._caps_by_task(declared))
   miner_weights = dict(ema)
   if MIN_INCENTIVE_SHARE > 0.0:
       miner_weights = {
           hk: v for hk, v in miner_weights.items()
           if v >= MIN_INCENTIVE_SHARE
       }
   ```

   `constants.py` (`MIN_INCENTIVE_SHARE`, including the `0` kill switch) was
   left exactly as written in round 1 — untouched this round.

2. **`tests/unit/test_weight_reader_multi_task.py`** — restored to its
   exact pre-round-1 content (confirmed byte-identical against
   `git show <pre-round-1 commit>:tests/unit/test_weight_reader_multi_task.py`).
   It only exercises `_replay_ema`/`_merge_archives` directly, so none of
   the floor tests belonged there any more.

3. **`tests/unit/test_weight_only_validator.py`** — added 4 tests, all
   driving `submit_once` through the file's existing
   `_patch_chain_and_storage` / `_restore` harness (no second harness
   built), with `storage.list_recent_datasets` overridden per test to
   control the exact rewards, and `MIN_INCENTIVE_SHARE` monkeypatched via
   `patch.object(wov_mod, "MIN_INCENTIVE_SHARE", ...)`:
   - `test_submit_once_drops_hotkey_below_floor_keeps_hotkey_above` —
     stubs `wov._submit_weights` to capture the `miner_weights` dict
     `submit_once` would have handed it; asserts the above-floor hotkey's
     value is untouched and the below-floor one is absent.
   - `test_submit_once_keeps_hotkey_exactly_on_the_floor` — same pattern,
     floor pinned to the exact single-window EMA value to test `>=`.
   - `test_submit_once_burns_freed_mass_without_redistributing` — **the
     test that matters most**, and stronger than round 1's version: it
     does *not* stub `_submit_weights`, it runs `submit_once` twice (floor
     off, floor on) against a fake two-hotkey metagraph with the real
     `_submit_weights`/`chain.get_metagraph`/`chain.set_weights` path, and
     asserts (a) the survivor's on-chain weight is byte-identical between
     the two runs, (b) the dropped hotkey is entirely absent, (c) the
     burn uid picks up exactly the dropped hotkey's freed share
     (`with_floor[0] == without_floor[0] + without_floor[20]`), and
     (d) both vectors still sum to 1.0 — i.e. conservation happens via the
     existing burn mechanism, not via rescaling survivors.
   - `test_submit_once_floor_disabled_drops_nothing` — floor monkeypatched
     to `0.0`.

### Verification (one command at a time, sequential, no xdist/timeout)

```
$ python -m pytest tests/unit/test_weight_only_validator.py -q
......................................                                   [100%]
38 passed in 10.06s
```

Explicit re-check that the six previously-failing assertions now pass
**with no edit to them** (confirmed both by running them in isolation and
by `git diff HEAD~1 -- tests/unit/test_weight_only_validator.py` showing no
lines touched inside those six test functions):

```
$ python -m pytest tests/unit/test_weight_only_validator.py -q -k "test_replay_ema_deterministic or test_replay_ema_reads_rewards_by_hotkey_field or test_replay_ema_k_way_split_applies_on_chain or test_replay_ema_boundary_fair_split_applies_on_chain or test_replay_ema_empty_rewards_decays_existing_ema or test_replay_ema_conservation_bound" -v
...
tests/unit/test_weight_only_validator.py ......                          [100%]
======================= 6 passed, 32 deselected in 8.21s =======================
```

```
$ python -m pytest tests/unit/test_weight_reader_multi_task.py -q
............                                                             [100%]
12 passed in 8.66s
```

(12 — exactly the original pre-floor count, confirming the file is back to
its untouched original content.)

```
$ python -m pytest tests/unit/test_v1_cutover.py -q
................                                                         [100%]
16 passed in 10.18s
```

```
$ python -m pytest tests/unit/test_archive_window_content.py -q
.......                                                                  [100%]
7 passed in 9.64s
```

```
$ python -m pytest tests/unit/test_task_archive_namespace.py -q
......................                                                   [100%]
22 passed in 10.01s
```

All five green. The full pytest suite was not run, per standing
instructions (no swap on this box; risk of OOM-killing a production
container).

### Concerns / notes for the operator (Round 2)

- `_replay_ema` is confirmed to have exactly one production caller
  (`submit_once`), so the floor now sits at the only place money is
  actually decided; there is no second payment path that needed the same
  fix.
- Nothing was pushed or merged; both commits are local to
  `/home/ubuntu/catalyst-tasks` on `feat/multi-task`.
