# Emission Price and Task Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Repoussé après l'isolation des tâches (2026-09-11).** Ce plan rend les paramètres de *la*
> tâche qui tourne modifiables sans release miner. Ce n'est pas ce qui débloque la création
> d'une deuxième tâche : voir `2026-09-11-task-isolation.md`, à exécuter d'abord.

**Goal:** Finish the emission price (restore its state across restarts, arm it behind a switch) and make a task's generation parameters a contract the miner applies instead of a profile it must be recompiled for.

**Architecture:** The price controller already runs in shadow; it gains a startup restore from archives, a disarmed-by-default switch that feeds its price into the fill-closed window pool, and a recovery journal that remembers the pool a window opened with. Contracts reuse `ProtocolProfile.to_generation_contract()` byte for byte: the profile resolver can build `ACTIVE_PROTOCOL_PROFILE` from a contract file, so every import-time constant derives from it unchanged; a contract may retune only an allowlist of fields of its compiled base profile; a contract-driven miner stages a new contract and restarts onto it, exactly as it already does for checkpoint activation.

**Tech Stack:** Python 3.11, pytest, FastAPI (`TestClient`), Typer (`CliRunner`).

**Spec:** `docs/superpowers/specs/2026-09-10-task-scoped-emission-pricing-design.md` (§4-§7, §9, §15)

## Global Constraints

- Same version: no new `protocol_version`, no new profile id, no "v2" anywhere (labels, identifiers, branch names).
- A contract body is exactly `ProtocolProfile.to_generation_contract()`; its `profile_id` is the compiled base profile id (`constants.py` gates fill-closed on it).
- Tunable in this plan: `sampling.temperature`, `sampling.top_p`, `sampling.top_k`; per environment `max_new_tokens`, `prompt_template` (renderer `dollar-substitution-v1`), `bft`, `batch_target`. Everything else must equal the base profile, including the environment set.
- Contract digest: `canonical_sha256` of the parsed contract (`reliquary/protocol/release_contract.py`).
- Price parameters stay module constants in `emission_price.py`; the only new env var on the price path is the arm switch `RELIQUARY_EXPERIMENTAL_EMISSION_PRICE_ARMED`, default off, requiring fill-closed.
- Trust model in this plan: a miner trusts the contract its validator advertises, as it already trusts `checkpoint_revision`. Signed registry, `/tasks` publication scheduling and R2 mirror are the next plan.
- Local verification: targeted test files only, run sequentially. The full suite is CI's job (the VPS has no swap; a full run next to other work already killed a production container). Worktrees outside `/tmp`. Never `git stash`.
- Code comments: one or two lines; rationale goes in the commit message.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq
  ```

## File Map

| File | Responsibility | Tasks |
|---|---|---|
| `reliquary/validator/emission_price.py` | pure price logic; gains `restore_state` | 1 |
| `reliquary/validator/service.py` | startup restore, pool injection, archive fields, server push | 1, 2, 7 |
| `reliquary/constants.py` | `EMISSION_PRICE_ARMED` switch | 2 |
| `reliquary/validator/fill_closed_recovery.py` | journal the window pool; recovery pays with it | 2 |
| `reliquary/protocol/profiles.py` | contract ⇄ profile, tuning allowlist, resolver, diff | 3, 4, 5, 8 |
| `reliquary/miner/contract_activation.py` (new) | stage a servable contract for the next start | 6 |
| `reliquary/miner/engine.py` | restart onto a staged contract on mismatch | 6 |
| `reliquary/validator/server.py` | `/contracts/{sha256}`, `/tasks` | 7 |
| `reliquary/cli/main.py` | `reliquary task export`, `reliquary task validate` | 8 |
| `docs/generation-contracts.md` (new) | operator guide | 9 |

---

### Task 1: Restore the price walk from archives at startup

**Files:**
- Modify: `reliquary/validator/emission_price.py` (append)
- Modify: `reliquary/validator/service.py` (new method next to `_advance_price_shadow`; startup sequence after `await self._rebuild_hashes_from_history()`)
- Test: `tests/unit/test_emission_price_restore.py`

**Interfaces:**
- Consumes: `PriceParams`, `PriceState`, `WindowOutcome`, `advance`, `replay`, `outcome_from_archive`, `PRODUCTION_PRICE_PARAMS` (existing, `emission_price.py`); `ValidationService._load_archive_range(*, start_window, end_window, require_all) -> list[dict]`; `_PRICE_SHADOW_HISTORY_WINDOWS` (existing, `service.py`).
- Produces: `restore_state(records: Sequence[Mapping[str, Any]], params: PriceParams, *, history: int) -> tuple[PriceState, list[WindowOutcome]]`; `ValidationService._restore_price_shadow_from_history() -> None` (async), which sets `self._price_shadow_state: PriceState` and `self._price_shadow_outcomes: collections.deque`.

- [ ] **Step 0: Create the worktree and rebase onto main**

```bash
git -C /home/ubuntu/catalyst-main worktree add -b feat/emission-price-and-task-contracts /home/ubuntu/catalyst-contracts design/task-scoped-emission-pricing
cd /home/ubuntu/catalyst-contracts
git fetch origin
git rebase origin/main
python -m pytest tests/unit/test_emission_price_controller.py tests/unit/test_emission_price_archive_adapter.py tests/unit/test_emission_price_ready_round.py tests/unit/test_emission_price_signal_fields.py tests/unit/test_window_price_signal_extraction.py tests/unit/test_archive_carries_price_signal.py tests/unit/test_archive_carries_price_shadow.py tests/unit/test_archive_window_content.py -q
```

Expected: rebase completes without conflicts (a dry `git merge-tree` against `origin/main` was clean); all listed tests pass. The shared branch `design/task-scoped-emission-pricing` is not modified.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_emission_price_restore.py`:

```python
"""Restarting must resume the price walk, not reset it to ``start``.

Once armed, a reset would hand miners the full pool on every restart.
"""

from __future__ import annotations

import pytest

from reliquary.validator.emission_price import (
    PRODUCTION_PRICE_PARAMS,
    PriceState,
    WindowOutcome,
    advance,
    replay,
    restore_state,
)


def _window(open_round: int, span: int, *, ready_offset: int) -> WindowOutcome:
    # incompressible = span, which is what outcome_from_archive reads back
    return WindowOutcome(
        open_round=open_round,
        close_round=open_round + span,
        collect_ready_round=open_round + ready_offset,
        incompressible_rounds=span,
    )


def _decided_archives(outcomes, params):
    """Archives as the validator writes them: signal plus the decision taken after it."""
    records, state = [], PriceState(price=params.start, last_good=params.start)
    for index, outcome in enumerate(outcomes, start=1):
        decision = advance(state, outcomes[:index], params)
        state = decision.state
        records.append({
            "window_start": index,
            "window_status": "completed",
            "window_open_round": outcome.open_round,
            "window_close_round": outcome.close_round,
            "collect_ready_round": outcome.collect_ready_round,
            "emission_price_shadow": {
                "price": decision.price,
                "last_good": decision.last_good,
                "r": decision.r,
                "r_smoothed": decision.r_smoothed,
                "regime": decision.regime,
                "applied": False,
            },
        })
    return records


def _shadow(price, last_good):
    return {"price": price, "last_good": last_good, "regime": "descend", "applied": False}


def test_no_history_restores_the_starting_price():
    state, outcomes = restore_state([], PRODUCTION_PRICE_PARAMS, history=64)

    assert state == PriceState(price=1.0, last_good=1.0)
    assert outcomes == []


def test_a_restart_resumes_the_walk_exactly_where_the_archive_left_it():
    params = PRODUCTION_PRICE_PARAMS
    outcomes = [_window(1000 * i, 1000, ready_offset=50) for i in range(1, 7)]

    uninterrupted = replay(outcomes, params)
    state, restored = restore_state(_decided_archives(outcomes[:-1], params), params, history=64)
    resumed = advance(state, restored + [outcomes[-1]], params)

    assert resumed.price == pytest.approx(uninterrupted.price)
    assert resumed.last_good == pytest.approx(uninterrupted.last_good)


def test_the_last_recorded_decision_wins_even_if_newer_archives_lack_one():
    records = [
        {"window_start": 1, "window_status": "completed", "emission_price_shadow": _shadow(0.8, 0.8)},
        {"window_start": 2, "window_status": "completed"},
    ]

    state, _ = restore_state(records, PRODUCTION_PRICE_PARAMS, history=64)

    assert state == PriceState(price=0.8, last_good=0.8)


def test_an_aborted_window_does_not_move_the_restored_price():
    records = [
        {"window_start": 1, "window_status": "completed", "emission_price_shadow": _shadow(0.8, 0.8)},
        {"window_start": 2, "window_status": "aborted", "emission_price_shadow": _shadow(0.1, 0.1)},
    ]

    state, _ = restore_state(records, PRODUCTION_PRICE_PARAMS, history=64)

    assert state.price == 0.8


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "0.5", True])
def test_a_malformed_decision_is_ignored(bad):
    records = [{"window_start": 1, "window_status": "completed", "emission_price_shadow": _shadow(bad, 0.5)}]

    state, _ = restore_state(records, PRODUCTION_PRICE_PARAMS, history=64)

    assert state == PriceState(price=1.0, last_good=1.0)


def test_outcomes_come_back_in_window_order_and_bounded():
    windows = [_window(1000 * i, 1000, ready_offset=50) for i in range(1, 6)]
    records = list(reversed(_decided_archives(windows, PRODUCTION_PRICE_PARAMS)))

    _, outcomes = restore_state(records, PRODUCTION_PRICE_PARAMS, history=2)

    assert outcomes == windows[-2:]


@pytest.mark.asyncio
async def test_the_service_restores_its_walk_before_the_next_window():
    from tests.unit.test_archive_carries_price_shadow import (
        _archive_all,
        _oversupplied_batcher,
        _service,
    )

    service = _service()
    service._window_n = 3

    async def _archives(*, start_window, end_window, require_all):
        assert require_all is False
        return [{"window_start": 3, "window_status": "completed", "emission_price_shadow": _shadow(0.5, 0.5)}]

    service._load_archive_range = _archives
    await service._restore_price_shadow_from_history()

    archives = await _archive_all(service, [_oversupplied_batcher(open_round=10_000)])

    # one 1000-round step of decay from the restored 0.5
    assert archives[0]["emission_price_shadow"]["price"] == pytest.approx(0.495)


@pytest.mark.asyncio
async def test_an_unreadable_history_falls_back_to_the_full_pool():
    from tests.unit.test_archive_carries_price_shadow import _service

    service = _service()
    service._window_n = 3

    async def _broken(**_kwargs):
        raise RuntimeError("archive lookup failed")

    service._load_archive_range = _broken
    await service._restore_price_shadow_from_history()

    assert getattr(service, "_price_shadow_state", None) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_emission_price_restore.py -q`
Expected: collection error `ImportError: cannot import name 'restore_state'`.

- [ ] **Step 3: Implement `restore_state`**

Append to `reliquary/validator/emission_price.py`:

```python
def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def restore_state(
    records: Sequence[Mapping[str, Any]],
    params: PriceParams,
    *,
    history: int,
) -> tuple[PriceState, list[WindowOutcome]]:
    """The walk as the last archived decision left it, plus the trailing outcomes."""
    state = PriceState(price=params.start, last_good=params.start)
    outcomes: list[WindowOutcome] = []
    ordered = sorted(
        (r for r in records if r.get("window_status", "completed") != "aborted"),
        key=lambda r: int(r.get("window_start", 0)),
    )
    for record in ordered:
        outcome = outcome_from_archive(record)
        if outcome is not None:
            outcomes.append(outcome)
        shadow = record.get("emission_price_shadow")
        if (
            isinstance(shadow, Mapping)
            and _finite_number(shadow.get("price"))
            and _finite_number(shadow.get("last_good"))
        ):
            state = PriceState(
                price=float(shadow["price"]),
                last_good=float(shadow["last_good"]),
            )
    return state, outcomes[-history:] if history > 0 else []
```

- [ ] **Step 4: Implement the service restore and call it at startup**

In `reliquary/validator/service.py`, add right before `def _advance_price_shadow(`:

```python
    async def _restore_price_shadow_from_history(self) -> None:
        """Resume the price walk from archives; on any failure keep the full pool."""
        from reliquary.validator.emission_price import (
            PRODUCTION_PRICE_PARAMS,
            restore_state,
        )

        current_window = self._window_n
        if current_window <= 0:
            return
        try:
            archives = await self._load_archive_range(
                start_window=max(1, current_window + 1 - _PRICE_SHADOW_HISTORY_WINDOWS),
                end_window=current_window,
                require_all=False,
            )
        except Exception:
            logger.warning(
                "price shadow restore failed; the walk restarts at start",
                exc_info=True,
            )
            return
        state, outcomes = restore_state(
            archives, PRODUCTION_PRICE_PARAMS, history=_PRICE_SHADOW_HISTORY_WINDOWS
        )
        self._price_shadow_state = state
        self._price_shadow_outcomes = collections.deque(
            outcomes, maxlen=_PRICE_SHADOW_HISTORY_WINDOWS
        )
        logger.info(
            "Restored price shadow from %d archives: price=%.4f last_good=%.4f",
            len(archives), state.price, state.last_good,
        )
```

In the startup sequence, replace:

```python
        await self._rebuild_hashes_from_history()
        self._log_startup_config_banner()
```

with:

```python
        await self._rebuild_hashes_from_history()
        await self._restore_price_shadow_from_history()
        self._log_startup_config_banner()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_emission_price_restore.py tests/unit/test_archive_carries_price_shadow.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/emission_price.py reliquary/validator/service.py tests/unit/test_emission_price_restore.py
git commit -m "feat(emission): resume the price walk from archives at startup

Once the price is armed, a walk that restarts at start would hand miners
the full pool on every validator restart. The walk now resumes from the
last archived decision, and its smoothing window from the trailing
outcomes. An unreadable history keeps the full pool: liveness over savings.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---
### Task 2: Arm the price behind a switch, and pay recovered windows with their own pool

**Files:**
- Modify: `reliquary/constants.py` (after the fill-closed validation block ending with `"experimental fill-closed capability"`)
- Modify: `reliquary/validator/service.py` (constants import list; `_build_window_batchers`; `_advance_price_shadow`; `_archive_window`)
- Modify: `reliquary/validator/fill_closed_recovery.py` (`begin`, `load`, `recover`)
- Test: `tests/unit/test_emission_price_arming.py`

**Interfaces:**
- Consumes: `self._price_shadow_state: PriceState | None` (Task 1); `FillClosedRecoveryStore` (existing).
- Produces: `EMISSION_PRICE_ARMED: bool` (`constants.py`); `ValidationService._window_pool_for_new_window() -> float`; `self._window_pool_by_window: dict[int, float]`; archive field `emission_window_pool: float`; shadow field `applied: bool` now equals the switch; `FillClosedRecoveryStore.begin(window, *, checkpoint_n, revision, targets, window_pool: float = 1.0)`; `window_environment_pool(record: dict) -> float` (`fill_closed_recovery.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_emission_price_arming.py`:

```python
"""Arming is the only step that changes what a miner earns.

It stays off by default, applies the price posted when a window opens, and a
crash-recovered window pays with the pool it opened with, never a fresh 1.0.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from reliquary.validator.emission_price import PriceState
from reliquary.validator.fill_closed_recovery import (
    FillClosedRecoveryStore,
    window_environment_pool,
)
from tests.unit.test_archive_carries_price_shadow import (
    _archive_all,
    _oversupplied_batcher,
    _service,
)


def test_unarmed_pays_the_full_pool_whatever_the_walk_says(monkeypatch):
    monkeypatch.setattr("reliquary.validator.service.EMISSION_PRICE_ARMED", False)
    service = _service()
    service._price_shadow_state = PriceState(price=0.5, last_good=0.5)

    assert service._window_pool_for_new_window() == 1.0


def test_armed_pays_the_walked_price(monkeypatch):
    monkeypatch.setattr("reliquary.validator.service.EMISSION_PRICE_ARMED", True)
    service = _service()
    service._price_shadow_state = PriceState(price=0.5, last_good=0.5)

    assert service._window_pool_for_new_window() == 0.5


def test_armed_without_any_walk_pays_the_full_pool(monkeypatch):
    monkeypatch.setattr("reliquary.validator.service.EMISSION_PRICE_ARMED", True)

    assert _service()._window_pool_for_new_window() == 1.0


@pytest.mark.asyncio
async def test_the_shadow_says_whether_it_is_applied(monkeypatch):
    monkeypatch.setattr("reliquary.validator.service.EMISSION_PRICE_ARMED", True)

    archives = await _archive_all(_service(), [_oversupplied_batcher(open_round=1000)])

    assert archives[0]["emission_price_shadow"]["applied"] is True


@pytest.mark.asyncio
async def test_the_archive_records_the_pool_the_window_opened_with():
    service = _service()
    batcher, submission = _oversupplied_batcher(open_round=1000)
    service._window_pool_by_window = {batcher.window_start: 0.5}

    archives = await _archive_all(service, [(batcher, submission)])

    assert archives[0]["emission_window_pool"] == 0.5
    assert batcher.window_start not in service._window_pool_by_window


def test_arming_without_fill_closed_refuses_to_import():
    env = {k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")}
    env["RELIQUARY_EXPERIMENTAL_EMISSION_PRICE_ARMED"] = "1"

    completed = subprocess.run(
        [sys.executable, "-c", "import reliquary.constants"],
        capture_output=True, text=True, env=env,
    )

    assert completed.returncode != 0
    assert "RELIQUARY_EXPERIMENTAL_EMISSION_PRICE_ARMED requires" in completed.stderr


def test_a_recovered_window_pays_with_its_journaled_pool():
    record = {"environments": ["openmathinstruct", "opencodeinstruct"], "picks_target": 4, "window_pool": 0.5}

    assert window_environment_pool(record) == pytest.approx(0.5 / 2 / 4)


def test_a_journal_written_before_arming_pays_the_full_pool():
    record = {"environments": ["openmathinstruct"], "picks_target": 4}

    assert window_environment_pool(record) == pytest.approx(1.0 / 1 / 4)


def test_the_journal_round_trips_the_window_pool(tmp_path):
    store = FillClosedRecoveryStore(tmp_path)
    store.begin(7, checkpoint_n=1, revision="a" * 40, targets={"openmathinstruct": 16}, window_pool=0.5)

    assert store.load(7)["window_pool"] == 0.5


@pytest.mark.parametrize("bad", [0.0, 1.5, float("nan"), True])
def test_the_journal_refuses_an_impossible_pool(tmp_path, bad):
    store = FillClosedRecoveryStore(tmp_path)

    with pytest.raises(ValueError):
        store.begin(7, checkpoint_n=1, revision="a" * 40, targets={"openmathinstruct": 16}, window_pool=bad)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_emission_price_arming.py -q`
Expected: collection error `ImportError: cannot import name 'window_environment_pool'`.

- [ ] **Step 3: Add the switch to `constants.py`**

Immediately after this existing block:

```python
if PROTOCOL_PROFILE_ID in _FILL_CLOSED_PROFILE_IDS and not FILL_CLOSED_ENABLED:
    raise ValueError(
        f"the {PROTOCOL_PROFILE_ID!r} profile requires its explicit "
        "experimental fill-closed capability"
    )
```

insert:

```python

# Feeds the emission price into the window pool. Off: the price is only published.
EMISSION_PRICE_ARMED = _os.environ.get(
    "RELIQUARY_EXPERIMENTAL_EMISSION_PRICE_ARMED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
if EMISSION_PRICE_ARMED and not FILL_CLOSED_ENABLED:
    raise ValueError(
        "RELIQUARY_EXPERIMENTAL_EMISSION_PRICE_ARMED requires the fill-closed "
        "window, whose assembler is the only pool injection point"
    )
```

- [ ] **Step 4: Journal the pool in `fill_closed_recovery.py`**

Add after the imports:

```python
def _valid_window_pool(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 < value <= 1.0
    )


def window_environment_pool(record: dict) -> float:
    """One environment's share of one batch, from the pool the window opened with."""
    return (
        float(record.get("window_pool", 1.0))
        / len(record["environments"])
        / record.get("picks_target", FILL_CLOSED_EMISSIONS_PER_WINDOW)
    )
```

Replace `begin`:

```python
    def begin(self, window: int, *, checkpoint_n: int, revision: str, targets: dict,
              window_pool: float = 1.0) -> None:
        if not _valid_window_pool(window_pool):
            raise ValueError("invalid active window pool")
        if self._path(window).exists():
            raise RuntimeError("active window requires recovery before reuse")
        write_json(self._path(window), {
            "schema_version": 1, "window_start": window,
            "identity": active_training_identity(), "parent_checkpoint_n": checkpoint_n,
            "parent_revision": revision, "environments": list(targets),
            "batch_targets": targets, "archive": None,
            "picks_target": FILL_CLOSED_PICKS_PER_WINDOW,
            "window_pool": float(window_pool),
        })
        self.load(window)
```

In `load`, replace `set(value) - {"picks_target"} != {` with `set(value) - {"picks_target", "window_pool"} != {`, and after the existing `picks` check add:

```python
        if "window_pool" in value and not _valid_window_pool(value["window_pool"]):
            raise ValueError("invalid active window pool")
```

In `recover`, replace:

```python
                ], pool=1.0 / len(environments) / record.get("picks_target", FILL_CLOSED_EMISSIONS_PER_WINDOW))
```

with:

```python
                ], pool=window_environment_pool(record))
```

- [ ] **Step 5: Wire the service**

In the `from reliquary.constants import (` list of `service.py`, replace:

```python
    DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR,
    ENVIRONMENT_MIX,
```

with:

```python
    DIFFICULTY_AUCTION_SHADOW_MAX_SLOTS_PER_OPERATOR,
    EMISSION_PRICE_ARMED,
    ENVIRONMENT_MIX,
```

Add right before `def _advance_price_shadow(`:

```python
    def _window_pool_for_new_window(self) -> float:
        """The pool a window opening now pays: the walked price once armed."""
        state = getattr(self, "_price_shadow_state", None)
        if not EMISSION_PRICE_ARMED or state is None:
            return 1.0
        return float(state.price)
```

In `_advance_price_shadow`, replace `"applied": False,` with `"applied": bool(EMISSION_PRICE_ARMED),`.

In `_build_window_batchers`, replace:

```python
        recovery = getattr(self, "_fill_closed_recovery_store", None)
        if FILL_CLOSED_ENABLED and recovery is not None:
            recovery.begin(target_window, checkpoint_n=cp.checkpoint_n,
                           revision=cp_hash, targets=dict(self.env_mix))
```

with:

```python
        recovery = getattr(self, "_fill_closed_recovery_store", None)
        window_pool = self._window_pool_for_new_window()
        if FILL_CLOSED_ENABLED:
            pools = getattr(self, "_window_pool_by_window", None)
            if pools is None:
                pools = self._window_pool_by_window = {}
            pools[target_window] = window_pool
        if FILL_CLOSED_ENABLED and recovery is not None:
            recovery.begin(target_window, checkpoint_n=cp.checkpoint_n,
                           revision=cp_hash, targets=dict(self.env_mix),
                           window_pool=window_pool)
```

and in the `FillClosedBatchAssembler(` call replace `window_pool=1.0,` with `window_pool=window_pool,`.

In `_archive_window`, replace:

```python
        price_shadow = self._advance_price_shadow(price_signal)
```

with:

```python
        price_shadow = self._advance_price_shadow(price_signal)
        applied_pool = (getattr(self, "_window_pool_by_window", None) or {}).pop(
            first_batcher.window_start, None
        )
```

and replace:

```python
            **({"emission_price_shadow": price_shadow} if price_shadow else {}),
```

with:

```python
            **({"emission_price_shadow": price_shadow} if price_shadow else {}),
            **({"emission_window_pool": applied_pool} if applied_pool is not None else {}),
```

- [ ] **Step 6: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_emission_price_arming.py tests/unit/test_emission_price_restore.py tests/unit/test_archive_carries_price_shadow.py tests/unit/test_archive_window_content.py -q
python -m pytest tests/unit/test_v1_cutover.py -q
```

Expected: all pass. `test_v1_cutover.py` is the existing recovery-journal suite; it must stay green because old journals carry no `window_pool`.

- [ ] **Step 7: Commit**

```bash
git add reliquary/constants.py reliquary/validator/service.py reliquary/validator/fill_closed_recovery.py tests/unit/test_emission_price_arming.py
git commit -m "feat(emission): arm the price behind a switch that defaults off

With RELIQUARY_EXPERIMENTAL_EMISSION_PRICE_ARMED the fill-closed window
pool becomes the walked price posted when the window opens; without it
nothing changes and the shadow still reports applied=false.

Crash recovery used to recompute a window's pay with a literal 1.0. The
active-window journal now records the pool the window opened with, and
recovery pays with it. Journals written before this default to 1.0, which
is exactly what those windows paid.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---
### Task 3: Rebuild a profile from its generation contract

**Files:**
- Modify: `reliquary/protocol/profiles.py` (insert immediately before the line `DEFAULT_PROFILE_ID = "qwen35-2b-auction-v2"`)
- Test: `tests/unit/test_generation_contract_roundtrip.py`

**Interfaces:**
- Consumes: `ProtocolProfile`, `SamplingProfile`, `EnvironmentProfile`, `BFTProfile`, `EpisodeProfile`, `PromptTemplateProfile`, `ThroughputTiebreakProfile`, `PROFILES` (existing, `profiles.py`); `canonical_sha256` (existing, `release_contract.py`).
- Produces: `generation_contract_sha256(contract: Mapping[str, Any]) -> str`; `profile_from_contract(contract: Mapping[str, Any]) -> ProtocolProfile` (raises `ValueError` on a template whose sha256 or renderer does not match).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_generation_contract_roundtrip.py`:

```python
"""A generation contract is a profile's data, and must rebuild that profile exactly.

The contract body stays byte-identical to ``to_generation_contract()``: miners
that still compare contracts would otherwise break on the first deploy, before
any parameter changed.
"""

from __future__ import annotations

import copy

import pytest

from reliquary.protocol import profiles


@pytest.mark.parametrize("profile_id", sorted(profiles.PROFILES))
def test_every_compiled_profile_round_trips_through_its_contract(profile_id):
    contract = profiles.PROFILES[profile_id].to_generation_contract()

    rebuilt = profiles.profile_from_contract(contract)

    assert rebuilt.to_generation_contract() == contract
    assert profiles.generation_contract_sha256(
        rebuilt.to_generation_contract()
    ) == profiles.generation_contract_sha256(contract)


def test_the_digest_ignores_key_order():
    contract = profiles.PROFILES["qwen3-4b-base-dapo-reliquary-v1"].to_generation_contract()
    reordered = dict(reversed(list(contract.items())))

    assert profiles.generation_contract_sha256(reordered) == profiles.generation_contract_sha256(contract)


def test_a_template_whose_text_does_not_match_its_sha256_is_refused():
    contract = copy.deepcopy(
        profiles.PROFILES["qwen3-4b-base-dapo-reliquary-v1"].to_generation_contract()
    )
    contract["environments"]["openmathinstruct"]["prompt_template"]["template"] += " tampered"

    with pytest.raises(ValueError, match="sha256"):
        profiles.profile_from_contract(contract)


def test_an_unknown_prompt_renderer_is_refused():
    contract = copy.deepcopy(
        profiles.PROFILES["qwen3-4b-base-dapo-reliquary-v1"].to_generation_contract()
    )
    contract["environments"]["openmathinstruct"]["prompt_template"]["renderer"] = "jinja"

    with pytest.raises(ValueError, match="renderer"):
        profiles.profile_from_contract(contract)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_generation_contract_roundtrip.py -q`
Expected: FAIL with `AttributeError: module 'reliquary.protocol.profiles' has no attribute 'profile_from_contract'`.

- [ ] **Step 3: Implement**

In `reliquary/protocol/profiles.py`, insert immediately before `DEFAULT_PROFILE_ID = "qwen35-2b-auction-v2"`:

```python
def generation_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Identity of a generation contract: sha256 of its canonical JSON."""
    from reliquary.protocol.release_contract import canonical_sha256

    return canonical_sha256(contract)


def profile_from_contract(contract: Mapping[str, Any]) -> ProtocolProfile:
    """Rebuild the profile a contract describes; the inverse of ``to_generation_contract``."""
    environments: dict[str, EnvironmentProfile] = {}
    for name, environment in contract["environments"].items():
        template = environment.get("prompt_template")
        prompt_template = None
        if template is not None:
            if template.get("renderer") != "dollar-substitution-v1":
                raise ValueError(f"environment {name!r} uses an unknown prompt renderer")
            prompt_template = PromptTemplateProfile(template["id"], template["template"])
            if prompt_template.sha256 != template.get("sha256"):
                raise ValueError(
                    f"environment {name!r} prompt template sha256 does not match its text"
                )
        bft = environment["bft"]
        episode = environment.get("episode")
        environments[name] = EnvironmentProfile(
            max_new_tokens=environment["max_new_tokens"],
            bft=None if bft is None else BFTProfile(
                thinking_budget=bft["thinking_budget"],
                answer_budget=bft["answer_budget"],
                force_answer=bft["force_answer"],
            ),
            answer_format=environment.get("answer_format"),
            prompt_template=prompt_template,
            batch_target=environment.get("batch_target"),
            environment_contract_id=environment.get("environment_contract_id"),
            environment_manifest_sha256=environment.get("environment_manifest_sha256"),
            episode=None if episode is None else EpisodeProfile(**episode),
        )
    sampling = contract["sampling"]
    tiebreak = contract["throughput_tiebreak"]
    return ProtocolProfile(
        profile_id=contract["profile_id"],
        model_id=contract["model_id"],
        model_revision=contract["model_revision"],
        protocol_version=contract["protocol_version"],
        collection_seconds=contract["collection_seconds"],
        upload_grace_seconds=contract["upload_grace_seconds"],
        prompt_encoding=contract["prompt_encoding"],
        sampling=SamplingProfile(
            rollouts=sampling["rollouts"],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            do_sample=sampling["do_sample"],
        ),
        environments=environments,
        throughput_tiebreak=None if tiebreak is None else ThroughputTiebreakProfile(
            token_cap=tiebreak["token_cap"],
            bucket_tokens_per_round=tiebreak["bucket_tokens_per_round"],
        ),
    )
```

Values are passed through without coercion on purpose: coercing `1` to `1.0` would change the canonical JSON and therefore the digest.

- [ ] **Step 4: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_generation_contract_roundtrip.py -q
python -m pytest tests/unit/test_protocol_profiles.py -q
```

Expected: all pass (the second file is the existing profile suite and must be untouched by this addition).

- [ ] **Step 5: Commit**

```bash
git add reliquary/protocol/profiles.py tests/unit/test_generation_contract_roundtrip.py
git commit -m "feat(contracts): rebuild a protocol profile from its generation contract

profile_from_contract is the exact inverse of to_generation_contract,
checked on every compiled profile, so a contract can become the source of
a profile without changing a byte miners already compare. Values are not
coerced: turning 1 into 1.0 would change the canonical digest.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---
### Task 4: Let a contract retune only what its base profile allows

**Files:**
- Modify: `reliquary/protocol/profiles.py` (insert immediately before the line `DEFAULT_PROFILE_ID = "qwen35-2b-auction-v2"`, i.e. after Task 3's functions)
- Test: `tests/unit/test_generation_contract_tuning.py`

**Interfaces:**
- Consumes: `profile_from_contract`, `PROFILES` (Task 3, existing).
- Produces: `validated_contract_profile(contract: Mapping[str, Any]) -> ProtocolProfile` — raises `ValueError` naming the first non-tunable or invalid field.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_generation_contract_tuning.py`:

```python
"""What a contract may change without a release, and what it may not.

Everything outside the allowlist is bound elsewhere in the code: the profile id
gates fill-closed, upload grace is asserted at import, rollouts and collection
time size proof capacity, the model revision binds checkpoint identity, and
answer formats and environment identities are graded by shipped code.
"""

from __future__ import annotations

import copy

import pytest

from reliquary.protocol import profiles
from reliquary.protocol.profiles import PromptTemplateProfile

LIVE = "qwen3-4b-base-dapo-reliquary-v1"


def _live() -> dict:
    return copy.deepcopy(profiles.PROFILES[LIVE].to_generation_contract())


def test_an_untouched_contract_is_its_base_profile():
    assert profiles.validated_contract_profile(_live()) == profiles.PROFILES[LIVE]


def test_sampling_can_be_retuned():
    contract = _live()
    contract["sampling"].update(temperature=0.8, top_p=0.95, top_k=20)

    profile = profiles.validated_contract_profile(contract)

    assert (profile.sampling.temperature, profile.sampling.top_p, profile.sampling.top_k) == (0.8, 0.95, 20)


def test_an_environment_budget_template_and_bft_can_be_retuned():
    contract = _live()
    math = contract["environments"]["openmathinstruct"]
    math["max_new_tokens"] = 4096
    math["prompt_template"] = PromptTemplateProfile("math-short-v1", "Solve: $problem").to_generation_contract()
    math["bft"] = {"thinking_budget": 2048, "answer_budget": 512, "force_answer": True}
    contract["environments"]["reliquary_logic_v2"]["batch_target"] = 8

    profile = profiles.validated_contract_profile(contract)

    assert profile.environments["openmathinstruct"].max_new_tokens == 4096
    assert profile.environments["openmathinstruct"].prompt_template.template == "Solve: $problem"
    assert profile.environments["openmathinstruct"].bft.thinking_budget == 2048
    assert profile.environments["reliquary_logic_v2"].batch_target == 8


def _set(path, value):
    def mutate(contract):
        target = contract
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


def _delete(path):
    def mutate(contract):
        target = contract
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        _set(("profile_id",), "not-a-profile"),
        _set(("model_revision",), "0" * 40),
        _set(("protocol_version",), 7),
        _set(("prompt_encoding",), "chat_template"),
        _set(("collection_seconds",), 50),
        _set(("upload_grace_seconds",), 10),
        _set(("sampling", "rollouts"), 8),
        _set(("sampling", "do_sample"), True),
        _set(("environments", "openmathinstruct", "answer_format"), "last_json_object_v1"),
        _set(("environments", "extra_env"), {"max_new_tokens": 10, "answer_format": None, "bft": None}),
        _delete(("environments", "opencodeinstruct")),
        _set(("unexpected",), 1),
    ],
)
def test_a_non_tunable_change_is_refused(mutate):
    contract = _live()
    mutate(contract)

    with pytest.raises(ValueError):
        profiles.validated_contract_profile(contract)


@pytest.mark.parametrize(
    "mutate",
    [
        _set(("sampling", "temperature"), 0),
        _set(("sampling", "temperature"), True),
        _set(("sampling", "top_p"), 1.5),
        _set(("sampling", "top_k"), -1),
        _set(("sampling", "top_k"), True),
        _set(("environments", "openmathinstruct", "max_new_tokens"), 0),
        _set(("environments", "openmathinstruct", "bft"), {"thinking_budget": -1, "answer_budget": 512, "force_answer": True}),
        _set(("environments", "reliquary_logic_v2", "batch_target"), 0),
    ],
)
def test_an_impossible_value_is_refused(mutate):
    contract = _live()
    mutate(contract)

    with pytest.raises(ValueError):
        profiles.validated_contract_profile(contract)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_generation_contract_tuning.py -q`
Expected: FAIL with `AttributeError: module 'reliquary.protocol.profiles' has no attribute 'validated_contract_profile'`.

- [ ] **Step 3: Implement**

In `reliquary/protocol/profiles.py`, insert immediately before `DEFAULT_PROFILE_ID = "qwen35-2b-auction-v2"`:

```python
_TUNABLE_SAMPLING_FIELDS = frozenset({"temperature", "top_p", "top_k"})
_TUNABLE_ENVIRONMENT_FIELDS = frozenset({"max_new_tokens", "prompt_template", "bft", "batch_target"})


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validated_contract_profile(contract: Mapping[str, Any]) -> ProtocolProfile:
    """Rebuild a contract's profile, refusing anything its base profile does not let it retune."""
    import math

    base = PROFILES.get(contract.get("profile_id"))
    if base is None:
        raise ValueError(f"contract names unknown base profile {contract.get('profile_id')!r}")
    expected = base.to_generation_contract()
    if set(contract) != set(expected):
        raise ValueError("contract top-level fields differ from its base profile")
    for key, value in expected.items():
        if key not in ("sampling", "environments") and contract[key] != value:
            raise ValueError(f"contract changes {key!r}, which is not tunable")

    sampling = contract["sampling"]
    if not isinstance(sampling, Mapping) or set(sampling) != set(expected["sampling"]):
        raise ValueError("contract sampling fields differ from its base profile")
    for key, value in expected["sampling"].items():
        if key not in _TUNABLE_SAMPLING_FIELDS and sampling[key] != value:
            raise ValueError(f"contract changes sampling.{key}, which is not tunable")
    temperature, top_p, top_k = sampling["temperature"], sampling["top_p"], sampling["top_k"]
    if not (isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
            and math.isfinite(temperature) and temperature > 0):
        raise ValueError("sampling.temperature must be a positive finite number")
    if not (isinstance(top_p, (int, float)) and not isinstance(top_p, bool) and 0 < top_p <= 1):
        raise ValueError("sampling.top_p must be in (0, 1]")
    if not (isinstance(top_k, int) and not isinstance(top_k, bool) and top_k >= 0):
        raise ValueError("sampling.top_k must be a non-negative integer")

    environments = contract["environments"]
    if not isinstance(environments, Mapping) or set(environments) != set(expected["environments"]):
        raise ValueError("contract environments differ from its base profile")
    for name, environment in environments.items():
        if not isinstance(environment, Mapping):
            raise ValueError(f"environments.{name} must be an object")
        base_environment = expected["environments"][name]
        for key in set(environment) | set(base_environment):
            if key not in _TUNABLE_ENVIRONMENT_FIELDS and environment.get(key) != base_environment.get(key):
                raise ValueError(f"contract changes environments.{name}.{key}, which is not tunable")
        if not _positive_int(environment.get("max_new_tokens")):
            raise ValueError(f"environments.{name}.max_new_tokens must be a positive integer")
        bft = environment.get("bft")
        if bft is not None and not (
            isinstance(bft, Mapping)
            and set(bft) == {"thinking_budget", "answer_budget", "force_answer"}
            and _positive_int(bft["thinking_budget"])
            and _positive_int(bft["answer_budget"])
            and isinstance(bft["force_answer"], bool)
        ):
            raise ValueError(f"environments.{name}.bft is malformed")
        batch_target = environment.get("batch_target")
        if batch_target is not None and not _positive_int(batch_target):
            raise ValueError(f"environments.{name}.batch_target must be a positive integer")

    try:
        return profile_from_contract(contract)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"contract is malformed: {exc}") from exc
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_generation_contract_tuning.py tests/unit/test_generation_contract_roundtrip.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add reliquary/protocol/profiles.py tests/unit/test_generation_contract_tuning.py
git commit -m "feat(contracts): allow a contract to retune only its safe fields

A contract names a compiled base profile and may change sampling
temperature/top_p/top_k and, per environment, the token budget, prompt
template, BFT and batch target. Everything else is bound elsewhere in the
code: the profile id gates fill-closed, upload grace is asserted at import,
rollouts and collection time size proof capacity, the model revision binds
checkpoint identity, and answer formats are graded by shipped code.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---
### Task 5: Resolve the active profile from a contract file

**Files:**
- Modify: `reliquary/protocol/profiles.py` (new functions before `DEFAULT_PROFILE_ID = "qwen35-2b-auction-v2"`; `resolve_protocol_profile`)
- Test: `tests/unit/test_generation_contract_resolution.py`

**Interfaces:**
- Consumes: `validated_contract_profile`, `generation_contract_sha256` (Tasks 3-4).
- Produces: `GENERATION_CONTRACT_PATH_ENV_VAR = "RELIQUARY_GENERATION_CONTRACT_PATH"`; `GENERATION_CONTRACT_SHA256_ENV_VAR = "RELIQUARY_GENERATION_CONTRACT_SHA256"`; `load_contract_profile(path: str, expected_sha256: str | None) -> ProtocolProfile`; `contract_profile_from_environment() -> ProtocolProfile | None`. `ACTIVE_PROTOCOL_PROFILE` is built from the contract when the path env var points at a file, so every constant in `constants.py` follows it with no further change.

Resolution rules:
- path unset → compiled profile, exactly as today;
- path set, file present → validated contract profile; the optional sha256 pin must match; `RELIQUARY_PROTOCOL_PROFILE`, if set, must equal the contract's base profile id;
- path set, file absent, no pin → compiled profile (a contract-driven miner before its first contract);
- path set, file absent, pin set → error (a validator must never silently fall back).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_generation_contract_resolution.py`:

```python
"""Where the active profile comes from once contracts exist."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest

from reliquary.protocol import profiles

LIVE = "qwen3-4b-base-dapo-reliquary-v1"


def _retuned(temperature=0.8) -> dict:
    contract = copy.deepcopy(profiles.PROFILES[LIVE].to_generation_contract())
    contract["sampling"]["temperature"] = temperature
    return contract


@pytest.fixture
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("RELIQUARY_"):
            monkeypatch.delenv(key)
    return monkeypatch


def test_without_a_contract_path_nothing_changes(clean_env):
    assert profiles.contract_profile_from_environment() is None


def test_a_pinned_contract_file_becomes_the_profile(clean_env, tmp_path):
    contract = _retuned()
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract, indent=2))
    clean_env.setenv(profiles.GENERATION_CONTRACT_PATH_ENV_VAR, str(path))
    clean_env.setenv(profiles.GENERATION_CONTRACT_SHA256_ENV_VAR, profiles.generation_contract_sha256(contract))

    profile = profiles.contract_profile_from_environment()

    assert profile.profile_id == LIVE
    assert profile.sampling.temperature == 0.8


def test_a_pin_that_does_not_match_refuses(clean_env, tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(_retuned()))
    clean_env.setenv(profiles.GENERATION_CONTRACT_PATH_ENV_VAR, str(path))
    clean_env.setenv(profiles.GENERATION_CONTRACT_SHA256_ENV_VAR, "0" * 64)

    with pytest.raises(ValueError, match="does not match"):
        profiles.contract_profile_from_environment()


def test_a_missing_unpinned_file_falls_back_to_the_compiled_profile(clean_env, tmp_path):
    clean_env.setenv(profiles.GENERATION_CONTRACT_PATH_ENV_VAR, str(tmp_path / "absent.json"))

    assert profiles.contract_profile_from_environment() is None


def test_a_missing_pinned_file_refuses(clean_env, tmp_path):
    clean_env.setenv(profiles.GENERATION_CONTRACT_PATH_ENV_VAR, str(tmp_path / "absent.json"))
    clean_env.setenv(profiles.GENERATION_CONTRACT_SHA256_ENV_VAR, "0" * 64)

    with pytest.raises(ValueError, match="does not exist"):
        profiles.contract_profile_from_environment()


def test_a_conflicting_profile_selection_refuses(clean_env, tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(_retuned()))
    clean_env.setenv(profiles.GENERATION_CONTRACT_PATH_ENV_VAR, str(path))
    clean_env.setenv("RELIQUARY_PROTOCOL_PROFILE", "qwen3-4b-base-dapo-v4")

    with pytest.raises(ValueError, match="conflicts"):
        profiles.contract_profile_from_environment()


def test_duplicate_keys_are_refused(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text('{"profile_id": "a", "profile_id": "b"}')

    with pytest.raises(ValueError, match="duplicate"):
        profiles.load_contract_profile(str(path), None)


def test_the_contract_is_what_the_process_imports(tmp_path):
    contract = _retuned(temperature=0.7)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    env = {k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")}
    env[profiles.GENERATION_CONTRACT_PATH_ENV_VAR] = str(path)
    script = (
        "import json; from reliquary.protocol import profiles as p; "
        "a = p.ACTIVE_PROTOCOL_PROFILE; "
        "print(json.dumps([a.profile_id, a.sampling.temperature]))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True, env=env,
    )

    assert json.loads(completed.stdout) == [LIVE, 0.7]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_generation_contract_resolution.py -q`
Expected: FAIL with `AttributeError: module 'reliquary.protocol.profiles' has no attribute 'contract_profile_from_environment'`.

- [ ] **Step 3: Implement the loaders**

In `reliquary/protocol/profiles.py`, insert immediately before `DEFAULT_PROFILE_ID = "qwen35-2b-auction-v2"`:

```python
GENERATION_CONTRACT_PATH_ENV_VAR = "RELIQUARY_GENERATION_CONTRACT_PATH"
GENERATION_CONTRACT_SHA256_ENV_VAR = "RELIQUARY_GENERATION_CONTRACT_SHA256"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("generation contract has duplicate keys")
    return dict(pairs)


def load_contract_profile(path: str, expected_sha256: str | None) -> ProtocolProfile:
    """Read, pin-check and validate a contract file."""
    import json
    from pathlib import Path

    contract = json.loads(Path(path).read_bytes(), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(contract, dict):
        raise ValueError("generation contract must be a JSON object")
    digest = generation_contract_sha256(contract)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"generation contract sha256 {digest} does not match the pinned {expected_sha256}"
        )
    return validated_contract_profile(contract)


def contract_profile_from_environment() -> ProtocolProfile | None:
    """The contract-selected profile, or None to keep the compiled one."""
    path = os.environ.get(GENERATION_CONTRACT_PATH_ENV_VAR, "").strip()
    if not path:
        return None
    pinned = os.environ.get(GENERATION_CONTRACT_SHA256_ENV_VAR, "").strip() or None
    if not os.path.exists(path):
        if pinned is not None:
            raise ValueError(f"pinned generation contract {path!r} does not exist")
        return None
    profile = load_contract_profile(path, pinned)
    selected = os.environ.get(_PROFILE_ENV_VAR, "").strip()
    if selected and selected != profile.profile_id:
        raise ValueError(
            f"{_PROFILE_ENV_VAR}={selected!r} conflicts with the contract's "
            f"base profile {profile.profile_id!r}"
        )
    return profile
```

- [ ] **Step 4: Use it in the resolver**

In `resolve_protocol_profile`, replace:

```python
    selected_id = (
        os.environ.get(_PROFILE_ENV_VAR, DEFAULT_PROFILE_ID)
        if profile_id is None
        else profile_id
    )
```

with:

```python
    if profile_id is None:
        contract_profile = contract_profile_from_environment()
        if contract_profile is not None:
            return contract_profile
    selected_id = (
        os.environ.get(_PROFILE_ENV_VAR, DEFAULT_PROFILE_ID)
        if profile_id is None
        else profile_id
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_generation_contract_resolution.py tests/unit/test_generation_contract_tuning.py tests/unit/test_generation_contract_roundtrip.py -q
python -m pytest tests/unit/test_protocol_profiles.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add reliquary/protocol/profiles.py tests/unit/test_generation_contract_resolution.py
git commit -m "feat(contracts): resolve the active profile from a contract file

With RELIQUARY_GENERATION_CONTRACT_PATH the process builds
ACTIVE_PROTOCOL_PROFILE from a validated contract, so every constant that
derives from the profile at import follows it unchanged, on the validator
and the miner alike. A validator pins the digest and never falls back; a
miner may start before its first contract exists.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---
### Task 6: The miner restarts onto a new contract it can serve

**Files:**
- Create: `reliquary/miner/contract_activation.py`
- Modify: `reliquary/miner/engine.py` (imports; new function after `_state_matches_active_protocol`; the mismatch branch of the mining loop)
- Test: `tests/unit/test_miner_contract_activation.py`

**Interfaces:**
- Consumes: `validated_contract_profile`, `GENERATION_CONTRACT_PATH_ENV_VAR` (Tasks 4-5); `canonical_json_bytes` (existing, `release_contract.py`).
- Produces: `ContractActivationRestartRequired(RuntimeError)`; `stage_contract_activation(remote_contract: Mapping[str, Any] | None, active_contract: Mapping[str, Any], contract_path: str | None) -> bool`; `engine._stage_or_report_contract_mismatch(state) -> None` (raises `ContractActivationRestartRequired` when a contract was staged).

The restart is handled like `CheckpointActivationRestartRequired` today: the exception leaves `mine_window`, the CLI re-raises, the process exits, the operator's restart policy starts it again, and `resolve_protocol_profile` reads the staged file (Task 5). A miner without `RELIQUARY_GENERATION_CONTRACT_PATH` behaves exactly as before.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_miner_contract_activation.py`:

```python
"""A contract-driven miner follows its validator's retune by restarting onto it."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from reliquary.miner.contract_activation import (
    ContractActivationRestartRequired,
    stage_contract_activation,
)
from reliquary.protocol import profiles
from reliquary.protocol.release_contract import canonical_json_bytes

LIVE = "qwen3-4b-base-dapo-reliquary-v1"


def _live() -> dict:
    return copy.deepcopy(profiles.PROFILES[LIVE].to_generation_contract())


def test_an_identical_contract_stages_nothing(tmp_path):
    path = tmp_path / "contract.json"

    assert stage_contract_activation(_live(), _live(), str(path)) is False
    assert not path.exists()


def test_a_miner_without_a_contract_path_is_unchanged(tmp_path):
    remote = _live()
    remote["sampling"]["temperature"] = 0.8

    assert stage_contract_activation(remote, _live(), None) is False


def test_a_validator_without_a_contract_stages_nothing(tmp_path):
    assert stage_contract_activation(None, _live(), str(tmp_path / "c.json")) is False


def test_a_servable_retune_is_staged_atomically(tmp_path):
    path = tmp_path / "contract.json"
    remote = _live()
    remote["sampling"]["temperature"] = 0.8

    assert stage_contract_activation(remote, _live(), str(path)) is True
    assert path.read_bytes() == canonical_json_bytes(remote)
    assert [p.name for p in tmp_path.iterdir()] == ["contract.json"]


def test_a_retune_this_build_cannot_serve_is_refused(tmp_path):
    path = tmp_path / "contract.json"
    remote = _live()
    remote["sampling"]["rollouts"] = 8

    with pytest.raises(ValueError):
        stage_contract_activation(remote, _live(), str(path))
    assert not path.exists()


def test_a_different_base_profile_is_a_release_not_a_retune(tmp_path):
    remote = copy.deepcopy(profiles.PROFILES["qwen3-4b-base-dapo-v4"].to_generation_contract())

    with pytest.raises(ValueError, match="different base profile"):
        stage_contract_activation(remote, _live(), str(tmp_path / "c.json"))


def _mismatched_state(engine, temperature):
    remote = copy.deepcopy(engine.to_generation_contract(engine.ACTIVE_PROTOCOL_PROFILE))
    remote["sampling"]["temperature"] = temperature
    return SimpleNamespace(
        generation_contract=remote,
        generation_profile_id=engine.ACTIVE_PROTOCOL_PROFILE.profile_id,
        protocol_version=engine.ACTIVE_PROTOCOL_PROFILE.protocol_version,
    )


def test_the_engine_restarts_onto_a_staged_contract(tmp_path, monkeypatch):
    from reliquary.miner import engine

    path = tmp_path / "contract.json"
    monkeypatch.setenv(profiles.GENERATION_CONTRACT_PATH_ENV_VAR, str(path))
    state = _mismatched_state(engine, temperature=0.5)

    with pytest.raises(ContractActivationRestartRequired):
        engine._stage_or_report_contract_mismatch(state)
    assert json.loads(path.read_bytes()) == state.generation_contract


def test_the_engine_only_reports_when_not_contract_driven(monkeypatch):
    from reliquary.miner import engine

    monkeypatch.delenv(profiles.GENERATION_CONTRACT_PATH_ENV_VAR, raising=False)

    engine._stage_or_report_contract_mismatch(_mismatched_state(engine, temperature=0.5))


def test_the_engine_keeps_waiting_on_a_contract_it_cannot_serve(tmp_path, monkeypatch):
    from reliquary.miner import engine

    path = tmp_path / "contract.json"
    monkeypatch.setenv(profiles.GENERATION_CONTRACT_PATH_ENV_VAR, str(path))
    state = _mismatched_state(engine, temperature=0.5)
    state.generation_contract["sampling"]["rollouts"] += 1

    engine._stage_or_report_contract_mismatch(state)

    assert not path.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_miner_contract_activation.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'reliquary.miner.contract_activation'`.

- [ ] **Step 3: Create `reliquary/miner/contract_activation.py`**

```python
"""Stage a validator's new generation contract for this miner's next start."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reliquary.protocol.profiles import validated_contract_profile
from reliquary.protocol.release_contract import canonical_json_bytes


class ContractActivationRestartRequired(RuntimeError):
    """A new generation contract is staged; the process must restart to apply it."""


def stage_contract_activation(
    remote_contract: Mapping[str, Any] | None,
    active_contract: Mapping[str, Any],
    contract_path: str | None,
) -> bool:
    """Write ``remote_contract`` where the next start reads it; True when a restart applies it."""
    if not contract_path or remote_contract is None or dict(remote_contract) == dict(active_contract):
        return False
    if remote_contract.get("profile_id") != active_contract.get("profile_id"):
        raise ValueError(
            "validator contract names a different base profile; that is a release, not a retune"
        )
    validated_contract_profile(remote_contract)
    path = Path(contract_path)
    staging = path.with_name(f"{path.name}.staging")
    staging.write_bytes(canonical_json_bytes(dict(remote_contract)))
    os.replace(staging, path)
    return True
```

- [ ] **Step 4: Wire the engine**

In `reliquary/miner/engine.py`, replace:

```python
from reliquary.protocol.profiles import (
    ACTIVE_PROTOCOL_PROFILE,
    to_generation_contract,
)
```

with:

```python
from reliquary.protocol.profiles import (
    ACTIVE_PROTOCOL_PROFILE,
    GENERATION_CONTRACT_PATH_ENV_VAR,
    to_generation_contract,
)
from reliquary.miner.contract_activation import (
    ContractActivationRestartRequired,
    stage_contract_activation,
)
```

Immediately after the end of `_state_matches_active_protocol` (the block ending with `== to_generation_contract(ACTIVE_PROTOCOL_PROFILE)` and `)`), add:

```python


def _stage_or_report_contract_mismatch(state) -> None:
    """Restart onto a new contract this build can serve; otherwise log and keep waiting."""
    contract_path = os.environ.get(GENERATION_CONTRACT_PATH_ENV_VAR, "").strip() or None
    try:
        staged = stage_contract_activation(
            state.generation_contract,
            to_generation_contract(ACTIVE_PROTOCOL_PROFILE),
            contract_path,
        )
    except ValueError as exc:
        logger.error("validator generation contract cannot be served by this build: %s", exc)
        staged = False
    if staged:
        raise ContractActivationRestartRequired(
            "validator advertised a new generation contract; restarting to apply it"
        )
    logger.error(
        "validator generation contract mismatch: local=%s/v%d remote=%s/v%s",
        ACTIVE_PROTOCOL_PROFILE.profile_id,
        ACTIVE_PROTOCOL_PROFILE.protocol_version,
        state.generation_profile_id,
        state.protocol_version,
    )
```

In the mining loop, replace:

```python
                if not _state_matches_active_protocol(state):
                    logger.error(
                        "validator generation contract mismatch: local=%s/v%d "
                        "remote=%s/v%s",
                        ACTIVE_PROTOCOL_PROFILE.profile_id,
                        ACTIVE_PROTOCOL_PROFILE.protocol_version,
                        state.generation_profile_id,
                        state.protocol_version,
                    )
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue
```

with:

```python
                if not _state_matches_active_protocol(state):
                    _stage_or_report_contract_mismatch(state)
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue
```

- [ ] **Step 5: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_miner_contract_activation.py -q
python -m pytest $(git grep -l "_state_matches_active_protocol\|protocol_mismatch" -- tests/unit) -q
```

Expected: all pass. The second command reruns every existing test that exercises the miner's contract check.

- [ ] **Step 6: Commit**

```bash
git add reliquary/miner/contract_activation.py reliquary/miner/engine.py tests/unit/test_miner_contract_activation.py
git commit -m "feat(miner): restart onto a validator contract this build can serve

A miner started with RELIQUARY_GENERATION_CONTRACT_PATH no longer stops
at a generation contract mismatch: if the validator's contract retunes the
same base profile within the allowlist, it is staged atomically and the
process restarts onto it, the way checkpoint activation already restarts.
A contract it cannot serve, or one naming another base profile, is logged
and waited out as before. Miners without the path are unchanged.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---
### Task 7: Serve the contract and the running task

**Files:**
- Modify: `reliquary/validator/server.py` (`ValidatorServer.__init__`; `_build_app`, right after the `/checkpoint` route)
- Modify: `reliquary/validator/service.py` (`_archive_window`, after the Task 2 `applied_pool` lines)
- Test: `tests/unit/test_validator_contract_endpoints.py`

**Interfaces:**
- Consumes: `generation_contract_sha256` (Task 3); `PROTOCOL_GENERATION_CONTRACT`, `PROTOCOL_PROFILE_ID` (existing, already imported in `server.py`); the shadow dict returned by `_advance_price_shadow` (Tasks 1-2).
- Produces: `GET /contracts/{contract_sha256}` → the active contract, or 404 `unknown_contract`; `GET /tasks` → `{"tasks": [{"task_id", "contract_sha256", "contract_url", "price", "window": {"window_n", "state"}}]}`; `ValidatorServer.latest_price_shadow: dict | None`.

Both routes are new: no field is added to an existing response, so `extra="forbid"` miners are unaffected. With one running task, `task_id` is the base profile id.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_validator_contract_endpoints.py`:

```python
"""Everything a miner needs to know about the running task, from two new routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reliquary.constants import PROTOCOL_GENERATION_CONTRACT, PROTOCOL_PROFILE_ID
from reliquary.protocol.profiles import generation_contract_sha256
from reliquary.validator.server import ValidatorServer


def _digest() -> str:
    return generation_contract_sha256(PROTOCOL_GENERATION_CONTRACT)


def test_the_active_contract_is_served_by_its_digest():
    response = TestClient(ValidatorServer().app).get(f"/contracts/{_digest()}")

    assert response.status_code == 200
    assert response.json() == dict(PROTOCOL_GENERATION_CONTRACT)
    assert generation_contract_sha256(response.json()) == _digest()


def test_an_unknown_contract_is_not_found():
    response = TestClient(ValidatorServer().app).get(f"/contracts/{'0' * 64}")

    assert response.status_code == 404


def test_tasks_lists_the_running_task_and_its_contract():
    tasks = TestClient(ValidatorServer().app).get("/tasks").json()["tasks"]

    assert len(tasks) == 1
    assert tasks[0]["task_id"] == PROTOCOL_PROFILE_ID
    assert tasks[0]["contract_sha256"] == _digest()
    assert tasks[0]["contract_url"] == f"/contracts/{_digest()}"
    assert tasks[0]["price"] is None
    assert tasks[0]["window"]["window_n"] is None


def test_tasks_shows_the_latest_price_decision():
    server = ValidatorServer()
    server.latest_price_shadow = {"price": 0.73, "applied": False, "regime": "descend"}

    task = TestClient(server.app).get("/tasks").json()["tasks"][0]

    assert task["price"] == {"price": 0.73, "applied": False, "regime": "descend"}


@pytest.mark.asyncio
async def test_the_service_publishes_each_decision_to_the_server():
    from tests.unit.test_archive_carries_price_shadow import (
        _archive_all,
        _oversupplied_batcher,
        _service,
    )

    service = _service()
    await _archive_all(service, [_oversupplied_batcher(open_round=1000)])

    assert service.server.latest_price_shadow["regime"] == "descend"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_validator_contract_endpoints.py -q`
Expected: FAIL — `/contracts/...` and `/tasks` return 404, and `ValidatorServer` has no `latest_price_shadow`.

- [ ] **Step 3: Add the server state and routes**

In `ValidatorServer.__init__`, replace:

```python
        self.active_batcher: GrpoWindowBatcher | None = None
```

with:

```python
        self.active_batcher: GrpoWindowBatcher | None = None
        # Latest shadow price decision, pushed by the service at each archive.
        self.latest_price_shadow: dict | None = None
```

In `_build_app`, immediately after this existing route:

```python
        @app.get("/checkpoint")
        async def checkpoint():
            cp = self._current_checkpoint
            if cp is None:
                raise HTTPException(status_code=404, detail="no_checkpoint")
            return {
                "checkpoint_n": cp.checkpoint_n,
                "repo_id": cp.repo_id,
                "revision": cp.revision,
                "signature": cp.signature,
            }
```

add:

```python

        @app.get("/contracts/{contract_sha256}")
        async def get_generation_contract(contract_sha256: str):
            from reliquary.protocol.profiles import generation_contract_sha256

            contract = dict(PROTOCOL_GENERATION_CONTRACT)
            if contract_sha256 != generation_contract_sha256(contract):
                raise HTTPException(status_code=404, detail="unknown_contract")
            return contract

        @app.get("/tasks")
        async def get_tasks():
            from reliquary.protocol.profiles import generation_contract_sha256

            digest = generation_contract_sha256(PROTOCOL_GENERATION_CONTRACT)
            batcher = self.active_batcher
            state = getattr(self, "_current_state", None)
            return {
                "tasks": [{
                    "task_id": PROTOCOL_PROFILE_ID,
                    "contract_sha256": digest,
                    "contract_url": f"/contracts/{digest}",
                    "price": (
                        dict(self.latest_price_shadow)
                        if self.latest_price_shadow is not None
                        else None
                    ),
                    "window": {
                        "window_n": batcher.window_start if batcher is not None else None,
                        "state": getattr(state, "value", None if state is None else str(state)),
                    },
                }],
            }
```

- [ ] **Step 4: Push each decision from the service**

In `service.py` `_archive_window`, replace:

```python
        price_shadow = self._advance_price_shadow(price_signal)
        applied_pool = (getattr(self, "_window_pool_by_window", None) or {}).pop(
            first_batcher.window_start, None
        )
```

with:

```python
        price_shadow = self._advance_price_shadow(price_signal)
        applied_pool = (getattr(self, "_window_pool_by_window", None) or {}).pop(
            first_batcher.window_start, None
        )
        if price_shadow is not None:
            self.server.latest_price_shadow = dict(price_shadow)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_validator_contract_endpoints.py tests/unit/test_archive_carries_price_shadow.py -q
python -m pytest tests/unit/test_validator_server.py -q --deselect tests/unit/test_validator_server.py::test_submission_protocol_stamps_wire_ingress --deselect tests/unit/test_validator_server.py::test_submission_protocol_closes_stalled_fresh_connection
```

Expected: all pass. The two deselected tests already fail on `origin/main` without this work.

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/server.py reliquary/validator/service.py tests/unit/test_validator_contract_endpoints.py
git commit -m "feat(validator): serve the running task and its contract

GET /contracts/{sha256} returns the active generation contract, so anyone
can check the digest a miner will adopt. GET /tasks lists the running task
with that digest, the latest price decision and the window state. Both are
new routes: no existing response gains a field, so strict miners are
unaffected.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---
### Task 8: `reliquary task export` and `reliquary task validate`

**Files:**
- Modify: `reliquary/protocol/profiles.py` (insert `contract_changes` before `DEFAULT_PROFILE_ID = "qwen35-2b-auction-v2"`)
- Modify: `reliquary/cli/main.py` (right after `app = typer.Typer(name="reliquary", help="Reliquary — Verifiable Inference Subnet")`)
- Test: `tests/unit/test_cli_task_contract.py`

**Interfaces:**
- Consumes: `PROFILES`, `generation_contract_sha256`, `load_contract_profile` (Tasks 3-5).
- Produces: `contract_changes(base: Mapping[str, Any], contract: Mapping[str, Any], prefix: str = "") -> list[str]` (lines `path: old -> new`, sorted by path); CLI `reliquary task export PROFILE_ID OUTPUT` and `reliquary task validate PATH` (exit 1 with the reason on an invalid contract).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_cli_task_contract.py`:

```python
"""The operator loop: export the running contract, edit it, validate it."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from reliquary.cli.main import app
from reliquary.protocol import profiles

LIVE = "qwen3-4b-base-dapo-reliquary-v1"
runner = CliRunner()


def test_changes_are_listed_by_dotted_path():
    base = profiles.PROFILES[LIVE].to_generation_contract()
    edited = json.loads(json.dumps(base))
    edited["sampling"]["temperature"] = 0.8

    assert profiles.contract_changes(base, edited) == ["sampling.temperature: 1.0 -> 0.8"]


def test_an_exported_contract_validates_unchanged(tmp_path):
    path = tmp_path / "contract.json"

    exported = runner.invoke(app, ["task", "export", LIVE, str(path)])
    validated = runner.invoke(app, ["task", "validate", str(path)])

    digest = profiles.generation_contract_sha256(profiles.PROFILES[LIVE].to_generation_contract())
    assert exported.exit_code == 0
    assert validated.exit_code == 0
    assert f"sha256: {digest}" in validated.output
    assert "changes: none" in validated.output


def test_a_retune_validates_and_shows_what_changed(tmp_path):
    path = tmp_path / "contract.json"
    runner.invoke(app, ["task", "export", LIVE, str(path)])
    contract = json.loads(path.read_text())
    contract["sampling"]["temperature"] = 0.8
    path.write_text(json.dumps(contract, indent=2))

    result = runner.invoke(app, ["task", "validate", str(path)])

    assert result.exit_code == 0
    assert "sampling.temperature: 1.0 -> 0.8" in result.output
    assert f"sha256: {profiles.generation_contract_sha256(contract)}" in result.output


def test_a_non_tunable_change_fails_with_the_reason(tmp_path):
    path = tmp_path / "contract.json"
    runner.invoke(app, ["task", "export", LIVE, str(path)])
    contract = json.loads(path.read_text())
    contract["sampling"]["rollouts"] = 8
    path.write_text(json.dumps(contract))

    result = runner.invoke(app, ["task", "validate", str(path)])

    assert result.exit_code == 1
    assert "not tunable" in result.output


def test_exporting_an_unknown_profile_fails(tmp_path):
    result = runner.invoke(app, ["task", "export", "not-a-profile", str(tmp_path / "c.json")])

    assert result.exit_code == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_cli_task_contract.py -q`
Expected: FAIL — `AttributeError: ... no attribute 'contract_changes'` and `No such command 'task'`.

- [ ] **Step 3: Implement `contract_changes`**

In `reliquary/protocol/profiles.py`, insert immediately before `DEFAULT_PROFILE_ID = "qwen35-2b-auction-v2"`:

```python
def contract_changes(
    base: Mapping[str, Any], contract: Mapping[str, Any], prefix: str = ""
) -> list[str]:
    """Where ``contract`` differs from ``base``, as ``dotted.path: old -> new``."""
    changes: list[str] = []
    for key in sorted(set(base) | set(contract)):
        path = f"{prefix}{key}"
        old, new = base.get(key), contract.get(key)
        if isinstance(old, Mapping) and isinstance(new, Mapping):
            changes.extend(contract_changes(old, new, f"{path}."))
        elif old != new:
            changes.append(f"{path}: {old!r} -> {new!r}")
    return changes
```

- [ ] **Step 4: Add the CLI commands**

In `reliquary/cli/main.py`, immediately after:

```python
app = typer.Typer(name="reliquary", help="Reliquary — Verifiable Inference Subnet")
```

add:

```python
task_app = typer.Typer(name="task", help="Export and validate generation contracts")
app.add_typer(task_app, name="task")


@task_app.command("export")
def export_task_contract(
    profile_id: str = typer.Argument(..., help="Compiled base profile to start from"),
    output: str = typer.Argument(..., help="Where to write the contract JSON"),
) -> None:
    import json
    from pathlib import Path

    from reliquary.protocol.profiles import PROFILES, generation_contract_sha256

    profile = PROFILES.get(profile_id)
    if profile is None:
        typer.echo(f"unknown profile {profile_id!r}", err=True)
        raise typer.Exit(code=1)
    contract = profile.to_generation_contract()
    Path(output).write_text(
        json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"sha256: {generation_contract_sha256(contract)}")


@task_app.command("validate")
def validate_task_contract(
    path: str = typer.Argument(..., help="Contract JSON to check before deploying it"),
) -> None:
    from reliquary.protocol.profiles import (
        PROFILES,
        contract_changes,
        generation_contract_sha256,
        load_contract_profile,
    )

    try:
        profile = load_contract_profile(path, None)
    except (OSError, ValueError) as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(code=1)
    contract = profile.to_generation_contract()
    changes = contract_changes(PROFILES[profile.profile_id].to_generation_contract(), contract)
    typer.echo(f"valid: base profile {profile.profile_id}")
    typer.echo(f"sha256: {generation_contract_sha256(contract)}")
    if not changes:
        typer.echo("changes: none")
        return
    typer.echo("changes:")
    for change in changes:
        typer.echo(f"  {change}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_cli_task_contract.py tests/unit/test_generation_contract_resolution.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add reliquary/protocol/profiles.py reliquary/cli/main.py tests/unit/test_cli_task_contract.py
git commit -m "feat(cli): export and validate generation contracts

reliquary task export writes a base profile's contract as an editable
starting point; reliquary task validate checks an edited one against the
tuning allowlist and prints its digest and every changed field. The digest
is what a validator pins with RELIQUARY_GENERATION_CONTRACT_SHA256.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---
### Task 9: Operator guide, and bring the design doc in line

**Files:**
- Create: `docs/generation-contracts.md`
- Modify: `docs/superpowers/specs/2026-09-10-task-scoped-emission-pricing-design.md`

**Interfaces:**
- Consumes: the env vars and commands produced by Tasks 2, 5, 6 and 8.
- Produces: documentation only.

- [ ] **Step 1: Write the operator guide**

Create `docs/generation-contracts.md`:

````markdown
# Generation contracts

A generation contract is everything a miner needs to produce valid work for the
running task. Its body is exactly `ProtocolProfile.to_generation_contract()`, it
names a compiled base profile (`profile_id`), and its identity is the sha256 of
its canonical JSON.

## What can change without a release

| Field | Tunable |
|---|---|
| `sampling.temperature`, `sampling.top_p`, `sampling.top_k` | yes |
| per environment: `max_new_tokens`, `prompt_template`, `bft`, `batch_target` | yes |
| profile id, model, protocol version, prompt encoding, rollouts, `do_sample`, collection and upload timing, answer formats, environment set and identities | no — these need a release |

## Retuning the running task (validator)

```bash
reliquary task export qwen3-4b-base-dapo-reliquary-v1 contract.json
# edit contract.json
reliquary task validate contract.json      # prints sha256 and every change
```

Deploy the task's validator with:

```
RELIQUARY_GENERATION_CONTRACT_PATH=/path/to/contract.json
RELIQUARY_GENERATION_CONTRACT_SHA256=<sha256 printed by validate>
```

and restart it. A pinned contract that is missing or does not match refuses to
start; it never falls back to the compiled profile. `GET /tasks` and
`GET /contracts/<sha256>` show what is being served.

## Following retunes (miner)

Set `RELIQUARY_GENERATION_CONTRACT_PATH` to a writable file and run the miner
under a restart policy (already required for checkpoint activation). When the
validator advertises a contract this build can serve, the miner writes it there
and exits; on restart it runs under it. Without the variable, the miner behaves
as before and stops at the first retune until it is updated.

## Caveats

- Changing a prompt template changes prompt text, hence prompt content hashes:
  problems already consumed under the old template are no longer recognised by
  the content cooldown.
- Miners that are not contract-driven stop mining at the first retune.
- There is no signature or scheduled activation yet: a miner trusts its
  validator's contract as it already trusts its checkpoint. The signed task
  registry is the next step.

## Emission price

The price is computed and published in every archive (`emission_price_shadow`)
and on `GET /tasks`. It is applied to the window pool only with
`RELIQUARY_EXPERIMENTAL_EMISSION_PRICE_ARMED=1`, which requires fill-closed.
Before arming, watch the shadow `r`: if it oscillates instead of drifting,
miners are not reading the posted price.
````

- [ ] **Step 2: Bring the design doc in line with what was built**

Run from the worktree root:

```bash
python3 - <<'PYEOF'
import pathlib, re
p = pathlib.Path("docs/superpowers/specs/2026-09-10-task-scoped-emission-pricing-design.md")
s = p.read_text(encoding="utf-8")

def swap(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)

swap(
    "*Changer le sens d'un champ existant est une décision à confirmer à l'implémentation.*",
    "*Décision prise à l'implémentation : `generation_profile_id` garde son sens (le profil de base, "
    "dont `constants.py` dépend). Le contrat exact est lié des deux côtés — le validateur le publie "
    "dans `MinerState.generation_contract` et sur `/contracts/{sha256}`, et le miner ne soumet rien "
    "tant que son contrat actif n'y est pas identique.*",
)
swap(
    "1. vérifie la signature du registre et le digest du contrat ;",
    "1. vérifie le digest du contrat (la signature arrive avec le registre ; d'ici là le miner fait "
    "confiance au contrat de son validateur, comme au checkpoint) ;",
)
swap(
    "**Trou connu, à combler avant d'armer.** En shadow, la marche du prix vit en mémoire : un "
    "redémarrage la remet à `start`. Sans conséquence tant que rien n'est appliqué ; une fois armé, "
    "**chaque redémarrage rendrait aux miners le pool entier**. L'état doit être amorcé depuis la "
    "dernière archive. `_load_archive_range`, qui fusionne déjà archives distantes et file locale, "
    "est le point d'accroche.",
    "**Reprise après redémarrage.** Au démarrage, la marche reprend depuis la dernière décision "
    "archivée et le lissage depuis les dernières mesures. Si l'historique est illisible, le pool "
    "reste entier : la liveness passe avant l'économie. Une fenêtre reprise après crash paie avec "
    "le pool journalisé à son ouverture.",
)

# Same version: no V1/V2/V3 labels.
for old, new in [
    (r"\bde la V([123])\b", r"de l'étape \1"),
    (r"\bà la V([123])\b", r"à l'étape \1"),
    (r"\bLa V([123])\b", r"L'étape \1"),
    (r"\bla V([123])\b", r"l'étape \1"),
    (r"\bEn V([123])\b", r"À l'étape \1"),
    (r"\bV([123])\b", r"étape \1"),
]:
    s = re.sub(old, new, s)
assert not re.search(r"\bV[123]\b", s)
p.write_text(s, encoding="utf-8")
print("synced")
PYEOF
grep -n "étape [123]" docs/superpowers/specs/2026-09-10-task-scoped-emission-pricing-design.md | head -20
```

Expected: `synced`, then the renamed headings and sentences. Read the listed lines and fix any sentence the mechanical rename left ungrammatical, by hand.

Then, in §15 of the same file, set the state column: phase 2 → `implémenté (désarmé par défaut)`; étape 2 rows **a** → `en partie : digest et validation ; publication signée et préavis à faire`, **b** and **c** → `implémenté`, **d** → `en partie : /tasks, /contracts, export, validate ; publish à faire`. Update the `**Statut**` line at the top to match.

- [ ] **Step 3: Commit**

```bash
git add docs/generation-contracts.md docs/superpowers/specs/2026-09-10-task-scoped-emission-pricing-design.md
git commit -m "docs(contracts): operator guide, and align the design with what was built

The guide covers the retune loop (export, validate, pin, restart), how a
miner follows retunes, and the price switch. The design doc records the
decisions taken while building: the profile id keeps its meaning, a miner
trusts its validator's contract until the signed registry exists, restarts
resume the price walk, and the work is one version rather than V1/V2/V3.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---

## Out of this plan (next plan, same version)

- Signed task registry: signer operation, sequence, `effective_from_round`, hash chain.
- `reliquary task publish`, R2 mirror of registry and contracts, `upcoming` on `/tasks`.
- Hot contract switch at window open instead of a validator restart.
- Several tasks: front routing `/t/<task_id>/…`, multi-prefix archive reader, time-anchored EMA with per-round payment, `EMA_ALPHA` fix.
