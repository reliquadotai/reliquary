# Merge of `origin/main` into `feat/multi-task` — resolution report

Worktree: `/home/ubuntu/catalyst-tasks`  ·  branch `feat/multi-task`
Ours: `acd732cb` (pushed safe point)  ·  Theirs: `c4e13984` (`origin/main`)
Merge base: `625add42`

Three files were conflicted; every other path in the merge was auto-merged by git and
left untouched.

---

## 1. `reliquary/validator/fill_closed_recovery.py` — 5 zones

### Zone 1 — imports · **union, as ruled**
Kept ours (`TASK_ID`, still used to stamp the recovered archive) and took all of main's
new names:

```python
from reliquary.constants import (
    B_BATCH,
    FILL_CLOSED_EMISSIONS_PER_WINDOW,
    FILL_CLOSED_PICKS_PER_WINDOW,
    FILL_CLOSED_SELECTION_POLICY,
    LEGACY_FILL_CLOSED_SELECTION_POLICY,
    TASK_ID,
)
```

The payment imports main needs (`FIXED_GROUP_PAYMENT_POLICY`, `EOS_TOKEN_PAYMENT_POLICY`,
`split_fixed_environment_pool`, `split_environment_pool`) sit in the `token_rewards`
import block, which git auto-merged already — nothing to add there. `import math`
(main's, for the `window_pool` finiteness check) was likewise auto-merged.
All six constants verified present in the merged `reliquary/constants.py`.

### Zone 2 — `load()` key set · **main's side entirely, as ruled**
Took main's `base_keys = {...}` schema_version 1/2 scheme; dropped our inline
`set(value) - {"picks_target", "window_pool"}` widening. Main's v2 `expected_keys`
already contains `window_pool`, so ours was subsumed exactly as described.

Note (not a conflict, no ruling): the two lines

```python
        if "window_pool" in value and not _valid_window_pool(value["window_pool"]):
            raise ValueError("invalid active window pool")
```

are ours and sit **outside** the conflict hunk, so git auto-merged them into the tail of
`load()`. I left them. They tighten main's own check (main accepts any finite
`window_pool >= 0`; ours additionally caps it at 1.0, which is what an emission *share*
can be). Nothing in the merged tree writes a pool above 1.0. Say the word if you want
them gone and I'll drop them.

### Zone 3 — `begin()` · **both guards, as ruled**
Main's multi-line signature, main's new `B_BATCH` target check, and our
`_valid_window_pool(window_pool)` range check. Our pool check is placed first, so an
out-of-range pool still reports "invalid active window pool" rather than being masked by
the target check.

### Zone 4 — journal fields · **main's side entirely, as ruled**
`journal_slots`, `payment_policy`, `selection_policy` added; `window_pool` (ours, common
context) retained, `schema_version` is main's `2`.

### Zone 5 — payment computation · **main's side entirely, as ruled**

Verified identical before taking it, as instructed:

| | ours (`window_environment_pool(record)`) | main (inline) |
|---|---|---|
| numerator | `float(record.get("window_pool", 1.0))` | `window_pool = float(record.get("window_pool", 1.0))` |
| ÷ | `len(record["environments"])` | `len(environments)` — bound to `record["environments"]` |
| ÷ | `record.get("picks_target", FILL_CLOSED_EMISSIONS_PER_WINDOW)` | `picks_target = record.get("picks_target", FILL_CLOSED_EMISSIONS_PER_WINDOW)` |

Same formula, same source, same fallback. Main additionally switches to
`split_fixed_environment_pool(..., slots=record["batch_targets"][env])` when
`payment_policy == FIXED_GROUP_PAYMENT_POLICY`, and keeps `split_environment_pool` for
legacy (pre-policy) journals. Taken as-is.

### Consequence — `window_environment_pool` removed
It had exactly one production caller (the Zone 5 line we dropped) and three test callers.
Function deleted. The three tests in `tests/unit/test_task_pool_from_registry.py` were
re-pointed at main's recovery path — they now drive `FillClosedRecoveryStore.recover()`
and assert on the archive instead of calling the helper directly:

* `test_recovery_pays_from_the_journalled_pool` — `begin(window_pool=0.5)` → recovered
  rewards `== 2 * (0.5 / 2 / 16) / 16` (two environments, 16 picks, 16 fixed slots).
* `test_recovery_defaults_a_pre_upgrade_journal_to_the_whole_pool` — a hand-written
  `schema_version: 1` journal (which cannot legally carry `window_pool` under main's key
  set) recovers at the 1.0 default and pays `2 * (1.0 / 2 / 16)` under the legacy
  eos-token policy.
* `test_recovery_pays_nobody_for_a_journalled_zero_share` — `window_pool=0.0` → every
  recovered reward is 0.0.

No test was deleted.

---

## 2. `reliquary/validator/service.py` — 2 zones

### Zone 1 — duplicate `recovery.begin` · **main's (empty) side + the pool re-added, as ruled**
Our window-open `recovery.begin(target_window, ..., window_pool=self._emission_cap)` was
deleted; main's single call in the checkpoint-activation path now reads:

```python
                recovery.begin(
                    candidate_window,
                    checkpoint_n=checkpoint.checkpoint_n,
                    revision=revision,
                    targets=dict(self.env_mix),
                    window_pool=self._emission_cap,
                )
```

`grep -c "recovery.begin("` in the file is now **1**. (`self._emission_cap` is also still
handed to `FillClosedBatchAssembler(window_pool=...)` in the prepare path — that is a
different sink and was untouched.)

### Zone 2 — `_advance_price_shadow` · **DEVIATION FROM THE RULING — read this**
The ruling says "keep our side in full" and describes the zone as ours vs. nothing.
Reality: **main's side of this hunk is not empty.** It is a new method,

```python
    @staticmethod
    def _close_and_commit_fill_closed_paid_side_effects(batchers, assembler) -> ...
```

which exists on neither our branch nor the merge base, and which the merged
`service.py` **calls in three places** (the auto-merged bodies at lines 4669, 5344 and
5975 of the pre-resolution file) plus two test files
(`tests/unit/test_v6_seal_seam.py:113`, `tests/unit/test_v1_cutover.py:387`).
Taking ours alone would have deleted a method that main's own auto-merged call sites
depend on — an immediate `AttributeError` on the v6 seal path.

**I resolved it as a union**: our `_advance_price_shadow` in full, unmodified, followed by
main's new static method, also unmodified. Both are pure additions to the same class and
neither touches the other's state. Confirmed separately that
`reliquary/validator/emission_price.py` exists only on our branch, exactly as you said.

If you intended the literal reading (ours only), say so and I'll redo it — but the
v6 seal path will not run.

---

## 3. `tests/unit/test_weight_only_validator.py` — 1 zone

**Both sides kept, as ruled.** Main's side of the hunk was the rewritten tail of
`test_burn_fails_closed_when_owner_has_no_uid` (`assert captured == {}`, replacing our
now-stale `assert captured[0] == 0.6 / sum == 1.0`); our side was those two stale lines
plus 566 lines of new floor/ramp tests. Resolution = main's assertion, then our tests
appended verbatim.

Main did **not** change `_patch_chain_and_storage`, `_archive` or `_FakeWallet` (verified
by diffing base→main across the whole file), but it did change the production behaviour
two of our tests lean on: the burn target moved from "this validator's UID, else fall
back to UID 0" to "the subnet owner's UID, fail closed if absent"
(`weight_only._resolve_burn_uid`). Our two burn-conservation tests built
`MagicMock(hotkeys=["hk_big", "hk_small"], uids=[10, 20])` and relied on the removed
fallback; they failed with `RuntimeError: subnet owner hotkey unavailable in metagraph`.

Adapted to main's semantics, per your instruction to adapt rather than restore: both fake
metagraphs now declare `owner_hotkey="owner"` at uid 0. The burn still lands on uid 0, so
**every assertion in both tests is unchanged** — only the fixture and two comments moved.

---

## 4. Added regression test (one, deliberate)

Nothing in the suite covered the service→journal leg of `window_pool`, which is exactly
the leg the Zone 1 hazard would have silently cut. Added
`test_activating_a_window_journals_the_pool_it_opened_with` to
`tests/unit/test_task_pool_from_registry.py`: it drives the real
`ValidationService._activate_window` with `_emission_cap = 0.37` and a real
`FillClosedRecoveryStore`, then asserts `store.load(42)["window_pool"] == 0.37`.
Remove it if you consider it out of scope.

---

## 5. Proof that `window_pool` still reaches the journal and is read back

Static chain:
`service._activate_window` → `recovery.begin(..., window_pool=self._emission_cap)` →
`begin()` writes `"window_pool": float(window_pool)` → `load()` validates it →
`recover()` reads `window_pool = float(record.get("window_pool", 1.0))`, divides it per
environment/pick/slot, and stamps it back into the archive as both `window_pool` and
`task_emission_share` → `finish()` refuses any archive whose `window_pool` differs from
the journal's.

Dynamic proof — script at
`/tmp/claude-1000/-home-ubuntu-Catalyst/90a210f3-c672-4441-81e4-41b6d73db328/scratchpad/prove_window_pool.py`,
run against the worktree (no mocks of the code under test: real `_activate_window`, real
store, real on-disk journal, real `recover()`), including a negative control that shows
what taking main's Zone-1 side *alone* would have paid:

```
$ PYTHONPATH=/home/ubuntu/catalyst-tasks python .../prove_window_pool.py
code under test: /home/ubuntu/catalyst-tasks/reliquary/validator/fill_closed_recovery.py
[1] journal written by _activate_window: /tmp/pool-proof-mcwt8yrm/fill_active/window-42.json
    raw bytes on disk: {"archive":null,"batch_targets":{"code":16,"math":16},"environments":["math","code"],"identity":{"generation_contract_sha256":"d763f85b1366ac1b25d0de1251899d6a8aceee11f5235f18817176a039d15dd1","protocol_profile_id":"qwen35-2b-auction-v2","protocol_version":2,"training_run_id":"default"},"journal_slots":16,"parent_checkpoint_n":7,"parent_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","payment_policy":"fixed-selected-group/v1","picks_target":16,"schema_version":2,"selection_policy":"fifo-ingress/v1","window_pool":0.37,"window_start":42}
    OK journal['window_pool'] == _emission_cap == 0.37
[2] store.load() reads it back: 0.37
    recovered archive['window_pool'] = 0.37
    recovered archive['task_emission_share'] = 0.37
    recovered rewards_by_hotkey = {'alice': 0.0014453125}
    expected 2 * (pool/2 envs/16 picks)/16 slots = 0.0014453125
    OK recovery paid from the journalled pool
[3] negative control -- begin() WITHOUT window_pool (main's call as merged):
    journal['window_pool'] = 1.0 (the 1.0 default)
    rewards_by_hotkey = {'alice': 0.00390625}
    i.e. 2.7027x what the window actually opened with (1/0.37 = 2.7027)
PROOF OK
```

That last block is the failure mode you predicted, quantified: without the re-added
keyword a recovered window would have paid 2.70× this task's share.

---

## 6. Commands and output

### No conflict markers anywhere

```
$ grep -rn '^<<<<<<<\|^>>>>>>>' reliquary/ tests/
(no output, exit 1)
```

### Syntax / import / lint

```
$ python -c "ast.parse(...)" for the four touched files
parsed ok: reliquary/validator/fill_closed_recovery.py
parsed ok: reliquary/validator/service.py
parsed ok: tests/unit/test_weight_only_validator.py
parsed ok: tests/unit/test_task_pool_from_registry.py

$ python -c "import reliquary.validator.fill_closed_recovery, reliquary.validator.service"
imports ok

$ python -m pyflakes <the four files>
(no output, exit 0)
```

### The 12 required runs — sequential, no xdist, no --timeout

```
$ python -m pytest tests/unit/test_v1_cutover.py -q
.....................                                                    [100%]
21 passed in 10.35s

$ python -m pytest tests/unit/test_archive_window_content.py -q
.......                                                                  [100%]
7 passed in 10.19s

$ python -m pytest tests/unit/test_task_archive_namespace.py -q
......................                                                   [100%]
22 passed in 10.07s

$ python -m pytest tests/unit/test_weight_only_validator.py -q
...............................................                          [100%]
47 passed in 9.92s

$ python -m pytest tests/unit/test_weight_reader_multi_task.py -q
............                                                             [100%]
12 passed in 8.67s

$ python -m pytest tests/unit/test_task_registry_rule.py -q
................................................                         [100%]
48 passed in 9.30s

$ python -m pytest tests/unit/test_task_registry_store.py -q
.........                                                                [100%]
9 passed in 9.50s

$ python -m pytest tests/unit/test_task_config.py -q
...........                                                              [100%]
11 passed in 8.21s

$ python -m pytest tests/unit/test_task_pool_from_registry.py -q
..............                                                           [100%]
14 passed in 8.61s

$ python -m pytest tests/unit/test_tasks_cli.py -q
.............                                                            [100%]
13 passed in 9.77s

$ python -m pytest tests/unit/test_tasks_endpoint.py -q
............                                                             [100%]
12 passed in 12.21s

$ python -m pytest tests/unit/test_emission_price_controller.py -q
...................                                                      [100%]
19 passed in 8.87s
```

**235 passed, 0 failed.** No baseline comparison against `origin/main` was needed: nothing
in the required list failed.

### Two intermediate failures, both fixed before the final run

First pass of `test_weight_only_validator.py` (before the owner-burn adaptation in §3):

```
FAILED tests/unit/test_weight_only_validator.py::test_submit_once_burns_freed_mass_without_redistributing
FAILED tests/unit/test_weight_only_validator.py::test_submit_once_ramp_burns_the_freed_mass_without_redistributing
2 failed, 45 passed in 10.90s
RuntimeError: subnet owner hotkey unavailable in metagraph
  reliquary/validator/weight_only.py:459 in _resolve_burn_uid
```

Green after adapting both fixtures to main's owner-burn semantics (47 passed, above).

### Extra runs, not on your list — the rest of the payment/recovery surface the merge touches

Run because both conflicted production files sit on the paid path; all sequential.

```
$ python -m pytest tests/unit/test_fill_close_and_emit.py -q
..............                                                           [100%]
14 passed in 13.27s

$ python -m pytest tests/unit/test_v6_emission.py -q
..............................                                           [100%]
30 passed in 11.60s

$ python -m pytest tests/unit/test_v6_seal_seam.py -q
.........                                                                [100%]
9 passed in 9.63s

$ python -m pytest tests/unit/test_fill_closed_profile.py -q
..............                                                           [100%]
14 passed in 24.18s

$ python -m pytest tests/unit/test_state_machine.py -q
.........................................                                [100%]
41 passed in 12.42s

$ python -m pytest tests/unit/test_service_v2.py -q
..................                                                       [100%]
18 passed in 10.61s
```

**126 passed, 0 failed.** `test_v6_seal_seam.py` and `test_v1_cutover.py` are also the two
files that would have caught the Zone-2 union call, had it gone the other way.

The full suite was never run. `git stash` was never used. `git merge --abort` was never used.
Nothing was pushed.

---

## 7. Summary of deviations from the rulings

1. **`service.py` Zone 2 is a union, not ours-only** (§2). Main's side of that hunk is a new
   method with three production call sites, not the empty side the ruling assumed. This is
   the one place where reality did not match a ruling's shape.
2. **Two tests in `tests/unit/test_task_pool_from_registry.py` needed `B_BATCH` fixtures**
   (a knock-on of Zone 3, which you ordered): main's `begin()` now refuses any batch target
   that is not `B_BATCH`, so `targets={"math": 1}` raised. They monkeypatch
   `fill_closed_recovery.B_BATCH`, the same way main's own `test_v1_cutover._recovery_setup`
   does. `test_recovered_archive_carries_the_pool_the_window_actually_opened_with` needed
   the same one-line fixture.
3. **Two burn tests in `test_weight_only_validator.py` were adapted** to main's owner-burn
   behaviour (§3) — fixtures only, assertions untouched.
4. **One test added** (§4), covering the service→journal leg of `window_pool`.
5. Our `_valid_window_pool` check in `load()` survived as auto-merged context (§ Zone 2 note).
