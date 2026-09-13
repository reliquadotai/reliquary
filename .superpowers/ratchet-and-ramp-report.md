# Rolling `last_good` + incentive ramp — implementation report

Worktree: `/home/ubuntu/catalyst-tasks`, branch `feat/multi-task`.
Starting HEAD: `1b3aeb4c`.

Commit A: `443aa6c9` — "fix(emission-price): make last_good a rolling minimum, not the last fill"
Commit B: `22aa04d5` — "fix(weight_only): ramp the 2% incentive floor instead of a cliff"

---

## CHANGE A — `last_good` as a rolling minimum

### Design

- `PriceParams` gains `last_good_fills: int` — how many of the most recent
  *filling* windows the rolling minimum is taken over. Named for what it
  counts (fills, not windows), matching `median_rounds` counting rounds.
  `PRODUCTION_PRICE_PARAMS.last_good_fills = 50`, commented as a shadow-phase
  starting point like every other number in that block.
- `PriceState` gains `recent_fill_prices: tuple[float, ...] = ()` — the
  bounded, fixed-size tuple of prices at which the last (up to)
  `last_good_fills` filling windows closed. `last_good` stays on the state as
  the reported scalar; inside `advance()` it is now **derived**:
  `min(recent_fill_prices) if recent_fill_prices else state.last_good`.
- On a fill: `recent_fill_prices = (*recent_fill_prices, price)[-last_good_fills:]`,
  then `last_good = min(recent_fill_prices)`. A fill at a raised price
  *joins* the window instead of overwriting `last_good`, so one expensive
  incident can no longer lift the floor the next snap escalates from — but
  once enough windows genuinely fill higher, the oldest cheap fill ages out
  and `last_good` does rise.
- On a non-fill (snap): the tuple is untouched, so `last_good` doesn't move.

### Backward compatibility — archives/state without the tuple

`recent_fill_prices` defaults to `()`. This is the one substantive design
decision for the "must tolerate absence" requirement:

- Every state constructed with only `price=`/`last_good=` (the exact
  round-trip shape from a pre-existing archive's two plain JSON scalars,
  see `test_state_survives_a_round_trip_through_plain_values`) gets an empty
  tuple, and `advance()`'s fallback (`state.last_good` when the tuple is
  empty) reproduces the old scalar-carrying behaviour exactly — no crash, no
  silent reset to "nothing ever filled."
- I did **not** change what `_advance_price_shadow` in `service.py` writes
  into the per-window archive (`price`, `last_good`, `r`, `r_smoothed`,
  `regime`, `applied`) — it still doesn't carry `recent_fill_prices`, and
  nothing currently reads a `PriceState` back out of an archive (per that
  method's own docstring, the walk is in-memory only and resets on restart;
  seeding it from the last archive is explicitly still-unimplemented future
  work). Its own `PriceState(price=price_params.start, last_good=price_params.start)`
  fallback construction needed no change — the default already makes it
  correct (genesis: empty tuple, scalar carries `start`).
- `replay()` re-folds the tuple from `params.start` (empty) on every call,
  identically to the incremental path — pinned by
  `test_replay_equals_the_incremental_fold_with_the_new_field`.

### Construction sites touched

`PriceParams` has no field with a default (matching every existing field,
and `task_registry.py`'s own stated policy: "a half-specified controller is
not a controller"), so `last_good_fills` had to join
`PRICE_PARAM_FIELDS` in `reliquary/shared/task_registry.py` (plus the
whole-number check alongside `rounds_per_step`/`median_rounds`). That in turn
required adding `"last_good_fills": 50` (or `1_000_000` in the emission-price
test helper, matching its "effectively unbounded" `median_rounds` pattern) to
every fixture `params` dict that flows through `validate_entry`/
`validate_registry`/`resolve_task_config`:

- `tests/unit/test_emission_price_controller.py` (`_params()` helper)
- `tests/unit/test_task_config.py`
- `tests/unit/test_task_pool_from_registry.py`
- `tests/unit/test_task_registry_rule.py`
- `tests/unit/test_task_registry_store.py`
- `tests/unit/test_weight_only_validator.py` and
  `tests/unit/test_weight_reader_multi_task.py` — these two never actually
  reach `validate_entry` (their `TaskEntry` objects are consumed only via
  `_caps_by_task`/`_replay_ema`, which read `params["cap"]` directly), so
  they weren't strictly required, but I added the field anyway for
  consistency/future-proofing since it's free.

`reliquary/cli/main.py`'s `build_task_entry` needed no change — it builds
`params` via `asdict(PRODUCTION_PRICE_PARAMS)`, so it picked up the new field
automatically.

### Tests added (`tests/unit/test_emission_price_controller.py`)

- `test_a_fill_at_a_raised_price_does_not_raise_last_good`
- `test_last_good_is_a_rolling_minimum_not_a_permanent_one`
- `test_replay_equals_the_incremental_fold_with_the_new_field`
- `test_a_state_without_the_recent_fills_tuple_still_loads`

All 4 written first, run, confirmed failing with
`TypeError: PriceParams.__init__() got an unexpected keyword argument 'last_good_fills'`
(the right reason), then the implementation was added, then all 19 tests in
the file (15 pre-existing + 4 new) pass.

---

## CHANGE B — ramp instead of a cliff

### Design

- `reliquary/constants.py`: added `MIN_INCENTIVE_RAMP_START` next to
  `MIN_INCENTIVE_SHARE`, default `0.01`, env-overridable via
  `RELIQUARY_MIN_INCENTIVE_RAMP_START`, validated
  `0.0 <= start <= MIN_INCENTIVE_SHARE` with a message naming both values.
  Extended (not replaced) the existing comment above `MIN_INCENTIVE_SHARE`'s
  block explaining the burn.
- `reliquary/validator/weight_only.py`: new static method
  `WeightOnlyValidator._ramped_incentive(value, *, start, threshold)`:
  ```python
  if value >= threshold:
      return value
  if start >= threshold or value < start:
      return 0.0
  return value * (value - start) / (threshold - start)
  ```
  `submit_once`'s filter block (was a `>= MIN_INCENTIVE_SHARE` dict-comprehension
  drop) now maps every hotkey's share through this and keeps it only if the
  payable amount is `> 0.0`. Values `>= MIN_INCENTIVE_SHARE` are returned by
  identity (byte-identical to no floor). `start == threshold` is guarded
  explicitly before the division, even though the range checks alone would
  never reach it — it collapses to exactly today's cliff.
- `MIN_INCENTIVE_SHARE == 0.0` still short-circuits the whole outer `if`,
  so the kill switch is unchanged.
- No change to `_submit_weights`: `burn_weight = max(0.0, 1 - registered_total)`
  already absorbs whatever a ramped-down (or fully dropped) hotkey no longer
  contributes; ramping down never rescales anyone else.

### Tests added (`tests/unit/test_weight_only_validator.py`)

Pure-function coverage of `_ramped_incentive` (exact, no EMA noise):
- `test_ramped_incentive_above_threshold_is_paid_in_full`
- `test_ramped_incentive_at_the_threshold_itself_is_paid_in_full`
- `test_ramped_incentive_at_the_midpoint_is_paid_half_its_share`
- `test_ramped_incentive_below_the_ramp_start_is_zero`
- `test_ramped_incentive_start_equals_threshold_reproduces_the_cliff`

`submit_once`-level coverage that the ramp is actually wired in:
- `test_submit_once_pays_a_ramped_share_between_start_and_threshold`
- `test_submit_once_kill_switch_ignores_a_nonzero_ramp_start`
- `test_submit_once_start_equals_threshold_reproduces_the_old_cliff_end_to_end`
- `test_submit_once_ramp_burns_the_freed_mass_without_redistributing` — mirrors
  the existing cliff burn test: survivor's value byte-identical across a
  floor-disabled and a ramped run, and the exact shortfall lands on the burn
  uid, never on the survivor.

All 9 written first, run against the unmodified module, confirmed failing
with `AttributeError: ... does not have the attribute 'MIN_INCENTIVE_RAMP_START'`
(the right reason — verified by temporarily reverting the implementation
edits via `git checkout --` on my own uncommitted changes from this session,
nothing else was uncommitted at the time, then reapplying from a saved
patch), then all 9 pass post-implementation.

### Flagged, not edited — two pre-existing assertions moved

`tests/unit/test_weight_only_validator.py` has two **pre-existing** tests
that patch only `MIN_INCENTIVE_SHARE` (never `MIN_INCENTIVE_RAMP_START`, so
it runs at the real production default, `0.01`) and use `hk_small` with
reward `0.5`, giving it a single-window EMA share of
`EMA_ALPHA * 0.5 ≈ 0.0136986301369863`. Under the old cliff (`0.02`) that
share was fully below the floor and dropped entirely. Under the new ramp
(`start=0.01`, `threshold=0.02`) that same share sits *inside* the ramp band
and is now paid a nonzero, partial amount:

```
fraction = (0.0136986301369863 - 0.01) / (0.02 - 0.01) = 0.369863...
paid     = 0.0136986301369863 * 0.369863... ≈ 0.00506661662600863
```

1. **`test_submit_once_drops_hotkey_below_floor_keeps_hotkey_above`**
   (line ~818) — `assert "hk_small" not in submitted_weights`
   - Old: `hk_small` absent.
   - New: `hk_small` present at `≈0.00506661662600863`.

2. **`test_submit_once_burns_freed_mass_without_redistributing`**
   (line ~875) — `assert 20 not in with_floor`
   - Old: uid `20` (`hk_small`) absent; `with_floor[0] == without_floor[0] + without_floor[20]`.
   - New: uid `20` present at `≈0.00506661662600863`; the burn only absorbs
     the *partial* shortfall (`without_floor[20] - with_ramp[20]`), not the
     whole share.

Both are a direct, correct consequence of the feature as specified (a share
between the new ramp start and the threshold is *supposed* to be paid
partially now, not dropped) — the ramp default (`start=0.01`) simply happens
to sit below this particular pre-existing fixture's hand-picked reward.
Per instructions, I did not edit either assertion or the fixture that feeds
it. Left both failing; ruling needed on:
- accept the new partial-payment behaviour and update these two assertions
  to the new numbers, or
- change the fixture's `hk_small` reward to something that lands below
  `MIN_INCENTIVE_RAMP_START` (e.g. reward `0.2`, giving a share
  `≈0.005479` — still below `0.01`) so the tests keep asserting "fully
  dropped," or
- something else.

No other file in the required verification list touches
`MIN_INCENTIVE_SHARE`/`MIN_INCENTIVE_RAMP_START`, so this is isolated to
these two tests.

---

## Verification — exact commands run, in order

```
python -m pytest tests/unit/test_emission_price_controller.py -q     # 19 passed
python -m pytest tests/unit/test_emission_price_archive_adapter.py -q  # 6 passed
python -m pytest tests/unit/test_archive_carries_price_shadow.py -q  # 4 passed
python -m pytest tests/unit/test_weight_only_validator.py -q         # 2 failed, 45 passed (see above)
python -m pytest tests/unit/test_weight_reader_multi_task.py -q      # 12 passed
python -m pytest tests/unit/test_v1_cutover.py -q                    # 16 passed
python -m pytest tests/unit/test_archive_window_content.py -q        # 7 passed
```

Also ran (not in the required list, but touched by Change A's construction-site
updates, to confirm no collateral damage): `test_task_config.py` (11 passed),
`test_task_pool_from_registry.py` (13 passed), `test_task_registry_rule.py`
(48 passed), `test_task_registry_store.py` (9 passed), `test_tasks_cli.py`
(13 passed).

No full-suite run was performed at any point.

---

## Round 2 — ruling applied: the two flagged tests updated

Ruling received: update both tests. The cliff they pinned is exactly what
change B deliberately removed, so they move with it — but the *numbers*
change because the policy changed, not the *invariant*. Commit:
`a45d4b7c` — "test(weight_only): re-point two floor tests from the cliff to
the ramp".

### `test_submit_once_drops_hotkey_below_floor_keeps_hotkey_above`

Subject unchanged ("below is dropped, above is kept"); fixture re-pointed so
it is genuinely below the ramp start, and a third hotkey added inside the
band so all three regimes are exercised. Name kept — still accurate.

| hotkey | reward | share (`EMA_ALPHA * reward`) | old assertion | new assertion |
|---|---|---|---|---|
| `hk_big` | 1.0 | `EMA_ALPHA ≈ 0.027397` | paid `EMA_ALPHA` (unchanged) | paid `EMA_ALPHA` (unchanged) |
| `hk_small` | was `0.5` → now `0.1` | was `≈0.013699` → now `≈0.002740` | absent (was below the old 2% cliff) | absent (now clearly below the 1% ramp start) |
| `hk_mid` *(new)* | `0.5` | `≈0.013699` | *(hotkey did not exist)* | paid `v·(v−0.01)/(0.02−0.01) ≈ 0.00506662` |

`hk_small`'s share had to move (`0.5`→`0.1` reward) because at the *old*
value (`≈0.0137`) it now sits inside the ramp band, not below it — that
share is exactly what `hk_mid` reuses to cover the partial-payment regime
instead.

### `test_submit_once_burns_freed_mass_without_redistributing`

Same two hotkeys, same rewards (`hk_big=1.0`, `hk_small=0.5`) — only the
floor parameters and the expected numbers change, because `hk_small`'s
share (`≈0.013699`) now falls inside the ramp band instead of below a 2%
cliff.

| quantity | old (cliff) | new (ramp) |
|---|---|---|
| `hk_small`'s share | `EMA_ALPHA·0.5 ≈ 0.0136986301369863` | same, `≈0.0136986301369863` |
| `hk_small` paid | `0` (dropped entirely) | `v·(v−0.01)/(0.02−0.01) ≈ 0.00506661662600863` |
| freed/burned from `hk_small` | `≈0.0136986301369863` (the whole share) | `≈0.00863201351097767` (only the shortfall) |
| `hk_big` (uid 10) | byte-identical to the no-floor run | byte-identical to the no-floor run (**unchanged assertion**) |
| burn uid (uid 0) | `without_floor[0] + without_floor[20]` | `without_floor[0] + (without_floor[20] − with_ramp[20])` |
| `sum(with_floor.values())` | `≈1.0` | `≈1.0` (**unchanged assertion**) |

The invariant kept intact: `hk_big`'s value is still asserted
byte-identical to the no-floor run (would fail loudly under any
renormalisation, since renormalising would raise it above that value), and
the burn-uid assertion still says "the exact shortfall, whatever its size,
lands only on the burn uid" — only the *size* of that shortfall changed,
from a whole share to a partial one.

### Verification (ruling's required sequence, run in order)

```
python -m pytest tests/unit/test_weight_only_validator.py -q     # 47 passed
python -m pytest tests/unit/test_weight_reader_multi_task.py -q  # 12 passed
python -m pytest tests/unit/test_v1_cutover.py -q                # 16 passed
python -m pytest tests/unit/test_archive_window_content.py -q    # 7 passed
```

All green. No full-suite run performed.

### Throwaway calculation: the ramp against a realistic live spread

Given: 29 hotkeys at (percent of pool) `4.74, 4.55, 4.47, 3.07, 2.87, 2.84,
2.83, 2.81, 2.80, 2.77, 2.73, 2.70, 2.68, 2.66, 2.51, 2.45, 2.42, 2.35,
2.26, 2.21, 2.16, 2.14, 2.10, 2.08, 1.96, 1.79, 1.78, 1.68, 1.57`, plus a
tail of 56 more hotkeys summing to ≈26% of the pool, none above 1.5%.
The visible 29 alone already sum to ≈75.98%, so 29+56=85 hotkeys and
≈75.98+26≈101.98% together — the ~2-point overshoot is just imprecision in
the illustrative figures, not a computed correction.

**Exact, for the 29 visible hotkeys** (threshold 2%, ramp start 1%):
- 24 paid in full (≥2%): summing 67.20% of the pool, all paid exactly their
  share (no change from a no-floor world).
- 5 inside the ramp band (1–2%): `1.96, 1.79, 1.78, 1.68, 1.57`, summing
  8.78% raw; paid 6.72% combined; **2.06% burned** from this group alone.
- 0 dropped outright among the visible 29 (all are ≥1.57%, above the 1%
  ramp start).

**Estimated, for the tail of 56** (their individual shares aren't given,
only the aggregate — this part is a modelled estimate, not exact): fit a
geometric decay to the observed slope at the end of the visible list
(ranks 25–29 shrink at a ratio of ≈0.945 per rank), starting the tail at
1.5% (the stated cap) and solving the ratio so the 56 terms sum to exactly
26% — giving values that trail off from 1.5% down to ≈0.066%. Under that
model:
- 0 tail hotkeys reach the 2% full-payment threshold.
- ≈8 tail hotkeys fall inside the ramp band (summing ≈9.92% raw, paid
  ≈2.58%, burning ≈7.33%).
- ≈48 tail hotkeys fall below the 1% ramp start and are dropped entirely
  (summing ≈16.08%, all burned).

A flat/uniform alternative for the tail (56 × 26/56 ≈ 0.46% each, all below
1%) would instead put all 56 in "dropped" and none in "partial" — the total
burn changes only modestly under that assumption (≈28.1% vs ≈25.5% below),
so the total-burn figure is fairly robust to the exact tail shape; the
full/partial/dropped *counts* for the tail are the less certain part.

**Combined (85 hotkeys, geometric-tail estimate):**

| regime | count | share of pool (raw) | paid |
|---|---|---|---|
| paid in full (≥2%) | 24 | 67.20% | 67.20% |
| paid partially (1–2%) | ≈13 (5 exact + ≈8 estimated) | ≈18.70% | ≈9.31% |
| dropped (<1%) | ≈48 (all in the estimated tail) | ≈16.08% | 0% |
| **total burned by the floor** | | | **≈25.5% of the pool** |

For comparison, the *old* cliff on this same spread would have burned
**≈34.8%** of the pool (every one of the 18 sub-2% visible+tail hotkeys in
the ramp band loses its whole share, not just the shortfall) — so the ramp
hands back roughly **9.3 points of pool share** to hotkeys that the cliff
would have paid nothing at all, while still keeping about a quarter of the
pool burned rather than distributed to anyone.
