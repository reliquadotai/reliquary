# Per-Environment Pricing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each environment its own discovered price, so one starving environment stops taxing every other one.

**Architecture:** The per-environment ready rounds already exist — `window_ready_round` computes them and keeps only the maximum. This plan stops discarding them: the sensor becomes a per-environment map, the controller keeps one price per environment against a shared denominator, the registry declares an `env_split`, and the assembler receives a pool per environment instead of one scalar it divides evenly. A timeout becomes its own window status that pays what was accrued without training.

**Tech Stack:** Python 3.11, pytest, Cloudflare R2, drand (quicknet, 3 s rounds).

**Spec:** `docs/superpowers/specs/2026-09-13-per-environment-pricing-design.md`

## Global Constraints

- **`default` must pay and archive byte-for-byte as today.** Declaring `env_split` in equal proportions must reproduce today's payouts exactly. The witness suites are `tests/unit/test_v1_cutover.py`, `tests/unit/test_archive_window_content.py`, `tests/unit/test_task_archive_namespace.py`, and **no assertion in them may be edited, relaxed or deleted** to make a change pass. If one fails, the change is wrong, not the test.
- **Writer-side only.** A weight-only validator replays `rewards_by_hotkey` verbatim; nothing here may require a reader to change.
- **Archives already written must still replay.** They carry `collect_ready_round` as a scalar; the new map must not break them.
- **NEVER run the full pytest suite.** This box has no swap and runs a production container; a full run OOMs it and has killed a production container here. Targeted files only, one command at a time, sequential, no `xdist`, no `--timeout` (pytest-timeout is not installed).
- Roughly 25 tests already fail on `origin/main`; verify there before attributing a failure.
- **NEVER use `git stash`**, in any form — checkouts are shared between sessions here and a stash has destroyed uncommitted work.
- Commit locally only. **Never push, never merge.**
- End every commit message with:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4`
- Comments run one or two lines; the reasoning belongs in the commit message.

---

### Task 1: Stop discarding the per-environment ready rounds

**Files:**
- Modify: `reliquary/validator/emission_price.py` (`window_ready_round` ~:310, `price_signal_fields` :328)
- Test: `tests/unit/test_emission_price_signal_fields.py`

**Interfaces:**
- Consumes: `ready_round(arrival_rounds, target)`, `distinct_prompt_arrival_rounds(arrivals_by_prompt)` — both already exist and are unchanged.
- Produces: `ready_rounds_by_environment(arrivals_by_environment, targets_by_environment) -> dict[str, int | None]`; `price_signal_fields` now emits `collect_ready_round_by_environment` **alongside** the existing scalar `collect_ready_round`.

Both fields travel together. The scalar stays because every archive already written carries it and the replay must keep reading them; the map is what later tasks consume. Emitting both is what makes this task deployable on its own.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_emission_price_signal_fields.py`:

```python
def test_ready_rounds_are_reported_per_environment():
    from reliquary.validator.emission_price import ready_rounds_by_environment

    rounds = ready_rounds_by_environment(
        {"math": {1: [10], 2: [12]}, "code": {3: [20], 4: [99]}},
        {"math": 2, "code": 2},
    )

    assert rounds == {"math": 12, "code": 99}


def test_an_environment_that_never_reached_its_target_reports_none():
    from reliquary.validator.emission_price import ready_rounds_by_environment

    rounds = ready_rounds_by_environment(
        {"math": {1: [10], 2: [12]}, "code": {3: [20]}},
        {"math": 2, "code": 2},
    )

    assert rounds == {"math": 12, "code": None}


def test_the_signal_carries_both_the_scalar_and_the_map():
    from reliquary.validator.emission_price import price_signal_fields

    fields = price_signal_fields(
        open_round=100,
        close_round=200,
        arrivals_by_environment={"math": {1: [110]}, "code": {2: [130]}},
        targets_by_environment={"math": 1, "code": 1},
    )

    # The scalar is the slowest environment, as before; the map is per env.
    assert fields["collect_ready_round"] == 130
    assert fields["collect_ready_round_by_environment"] == {"math": 110, "code": 130}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_emission_price_signal_fields.py -q`
Expected: FAIL — `ImportError: cannot import name 'ready_rounds_by_environment'`.

- [ ] **Step 3: Implement**

In `reliquary/validator/emission_price.py`, add above `window_ready_round`:

```python
def ready_rounds_by_environment(
    arrivals_by_environment: Mapping[str, Mapping[int, Sequence[int]]],
    targets_by_environment: Mapping[str, int],
) -> dict[str, int | None]:
    """When each environment reached its own target, or None for one that did not.

    ``window_ready_round`` computes exactly this and keeps only the maximum;
    a per-environment price needs the values it throws away.
    """
    return {
        environment: ready_round(
            distinct_prompt_arrival_rounds(
                arrivals_by_environment.get(environment, {})
            ),
            target,
        )
        for environment, target in targets_by_environment.items()
    }
```

Then rewrite `window_ready_round` to use it, so the two cannot drift:

```python
def window_ready_round(
    arrivals_by_environment: Mapping[str, Mapping[int, Sequence[int]]],
    targets_by_environment: Mapping[str, int],
) -> int | None:
    """The round the SLOWEST environment reached its target.

    A fill-closed window is not ready until every environment holds its own
    target, so averaging across them would report a readiness neither one had.
    """
    rounds = ready_rounds_by_environment(
        arrivals_by_environment, targets_by_environment
    )
    if any(reached is None for reached in rounds.values()):
        return None
    return max(rounds.values(), default=None)
```

Read the current body first: if it already calls `distinct_prompt_arrival_rounds` itself, keep that call inside the new helper exactly as it was, so the arrival counting is untouched.

Finally, in `price_signal_fields`, add the map beside the scalar:

```python
        "collect_ready_round_by_environment": ready_rounds_by_environment(
            arrivals_by_environment, targets_by_environment
        ),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_emission_price_signal_fields.py -q
python -m pytest tests/unit/test_emission_price_controller.py -q
python -m pytest tests/unit/test_archive_carries_price_signal.py -q
```

Expected: all pass. The third proves the archive still carries what it carried.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/emission_price.py tests/unit/test_emission_price_signal_fields.py
git commit -m "feat(price): report the ready round of every environment, not just the slowest

window_ready_round already computed each environment's ready round and kept
only the maximum, so a window whose math env starved reported the same
readiness as one where every env starved. The per-environment values it
discarded are exactly what a per-environment price needs.

The scalar stays beside the new map: every archive already written carries
it, and the replay has to keep reading them.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4"
```

---

### Task 2: One price per environment

**Files:**
- Modify: `reliquary/validator/emission_price.py` (`WindowOutcome`, `advance`, `replay`, `outcome_from_archive`)
- Test: `tests/unit/test_emission_price_controller.py`

**Interfaces:**
- Consumes: `ready_rounds_by_environment` (Task 1); `PriceParams`, `PriceState`, `advance`, `replay` as they stand.
- Produces: `EnvironmentOutcome` (frozen dataclass: `environment`, `open_round`, `close_round`, `ready_round: int | None`, `incompressible_rounds`); `advance_by_environment(states, recent_by_environment, params) -> dict[str, PriceDecision]`; `outcomes_by_environment_from_archive(record) -> dict[str, EnvironmentOutcome] | None`.

`advance` itself is not changed — one environment's price is decided by exactly the rule that decided the window's, so the new function maps the existing one over the environments. That is what keeps this task small and keeps one controller to calibrate.

- [ ] **Step 1: Write the failing test — this is the test the whole plan exists for**

Add to `tests/unit/test_emission_price_controller.py`:

```python
def test_a_starving_environment_does_not_raise_the_other_ones_price():
    """The defect this work exists to fix: today one starving env snaps the
    price for every env, because the window's readiness is the slowest env's."""
    from reliquary.validator.emission_price import (
        PRODUCTION_PRICE_PARAMS,
        EnvironmentOutcome,
        PriceState,
        advance_by_environment,
    )

    states = {
        "math": PriceState(price=0.5, last_good=0.5, recent_fill_prices=(0.5,)),
        "code": PriceState(price=0.5, last_good=0.5, recent_fill_prices=(0.5,)),
    }
    recent = {
        # math never reached its target; code reached it comfortably early.
        "math": [EnvironmentOutcome("math", 0, 1000, None, 500.0)],
        "code": [EnvironmentOutcome("code", 0, 1000, 100, 500.0)],
    }

    decisions = advance_by_environment(states, recent, PRODUCTION_PRICE_PARAMS)

    assert decisions["math"].regime == "snap"
    assert decisions["math"].price > 0.5
    assert decisions["code"].regime != "snap"
    assert decisions["code"].price <= 0.5


def test_each_environment_keeps_its_own_walk():
    from reliquary.validator.emission_price import (
        PRODUCTION_PRICE_PARAMS,
        EnvironmentOutcome,
        PriceState,
        advance_by_environment,
    )

    states = {
        "math": PriceState(price=0.8, last_good=0.8, recent_fill_prices=(0.8,)),
        "code": PriceState(price=0.2, last_good=0.2, recent_fill_prices=(0.2,)),
    }
    recent = {
        "math": [EnvironmentOutcome("math", 0, 1000, 900, 500.0)],
        "code": [EnvironmentOutcome("code", 0, 1000, 900, 500.0)],
    }

    decisions = advance_by_environment(states, recent, PRODUCTION_PRICE_PARAMS)

    # Same outcome shape, different starting prices: the walks do not merge.
    assert decisions["math"].price != decisions["code"].price
```

If `PriceState`'s constructor does not take `recent_fill_prices`, read the dataclass and use its real field names — it gained a rolling-minimum field in an earlier change.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_emission_price_controller.py -q`
Expected: FAIL — `ImportError: cannot import name 'EnvironmentOutcome'`.

- [ ] **Step 3: Implement**

Add to `reliquary/validator/emission_price.py`:

```python
@dataclass(frozen=True, slots=True)
class EnvironmentOutcome:
    """One environment's contribution to one window's price signal.

    The numerator is this environment's own readiness; the denominator is the
    window's, because collecting one environment faster than the shared
    training step can consume buys nothing.
    """

    environment: str
    open_round: int
    close_round: int
    ready_round: int | None
    incompressible_rounds: float

    @property
    def elapsed_rounds(self) -> int:
        return int(self.close_round) - int(self.open_round)

    @property
    def filled(self) -> bool:
        return self.ready_round is not None

    @property
    def ratio(self) -> float | None:
        if self.ready_round is None or self.incompressible_rounds <= 0:
            return None
        return (
            int(self.ready_round) - int(self.open_round)
        ) / self.incompressible_rounds


def advance_by_environment(
    states: Mapping[str, PriceState],
    recent_by_environment: Mapping[str, Sequence[EnvironmentOutcome]],
    params: PriceParams,
) -> dict[str, PriceDecision]:
    """Decide every environment's price independently, on one shared rule.

    ``advance`` is reused unchanged: one environment's price is decided by
    exactly the rule that decided the window's, so there is still one
    controller to calibrate, not one per environment.
    """
    return {
        environment: advance(
            states.get(
                environment,
                PriceState(
                    price=params.start,
                    last_good=params.start,
                    recent_fill_prices=(),
                ),
            ),
            recent,
            params,
        )
        for environment, recent in recent_by_environment.items()
        if recent
    }
```

`advance` takes a sequence whose last element is the window being decided and reads `outcome.filled`, `outcome.ratio` and `outcome.elapsed_rounds`. `EnvironmentOutcome` provides all three with the same names, so `advance` needs no change — confirm that by reading it rather than assuming, and if it touches any other attribute, add it to `EnvironmentOutcome` rather than special-casing `advance`.

Then add the archive adapter beside `outcome_from_archive`:

```python
def outcomes_by_environment_from_archive(
    record: Mapping[str, Any],
) -> dict[str, EnvironmentOutcome] | None:
    """Per-environment outcomes, or None when the record carries no map.

    Archives written before the map existed carry only the scalar; they keep
    replaying through ``outcome_from_archive`` and contribute nothing here.
    """
    if record.get("window_status", "completed") == "aborted":
        return None
    by_environment = record.get("collect_ready_round_by_environment")
    if not isinstance(by_environment, dict):
        return None
    scalar = outcome_from_archive(record)
    if scalar is None:
        return None
    return {
        environment: EnvironmentOutcome(
            environment=environment,
            open_round=scalar.open_round,
            close_round=scalar.close_round,
            ready_round=None if ready is None else int(ready),
            incompressible_rounds=scalar.incompressible_rounds,
        )
        for environment, ready in by_environment.items()
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_emission_price_controller.py -q
python -m pytest tests/unit/test_emission_price_archive_adapter.py -q
python -m pytest tests/unit/test_archive_carries_price_shadow.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/emission_price.py tests/unit/test_emission_price_controller.py
git commit -m "feat(price): give every environment its own walk on one shared rule

A window-level ratio is the slowest environment's, so a starving env snapped
the price for every env while an abundant one's surplus was never measured.
Each environment now carries its own state and its own ratio -- numerator per
env, denominator shared, because collecting faster than the common training
step can consume buys nothing.

advance is reused unchanged: one controller to calibrate, N prices.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4"
```

---

### Task 3: `env_split` in the registry

**Files:**
- Modify: `reliquary/shared/task_registry.py` (`TaskEntry`, `validate_entry`, `parse_registry`, `render_registry`)
- Modify: `reliquary/validator/task_config.py:70-75`
- Modify: `reliquary/cli/main.py` (`build_task_entry`, `tasks_create`)
- Test: `tests/unit/test_task_registry_rule.py`, `tests/unit/test_task_config.py`

**Interfaces:**
- Consumes: `TaskEntry`, `validate_entry`, `RegistryError`, `_number` (all in `task_registry.py`).
- Produces: `TaskEntry.env_split: Mapping[str, float] | None`; `TaskConfig.env_caps: dict[str, float]`.

`cap_e = cap * env_split_e`. The task cap says how much the task may spend; the split says how that divides. The two move independently, which is the point.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_task_registry_rule.py`:

```python
def test_an_env_split_must_sum_to_one():
    entry = replace(_entry("a", 0.5), env_split={"math": 0.6, "code": 0.5})

    with pytest.raises(RegistryError, match="env_split"):
        add_task({}, entry)


def test_an_env_split_that_sums_to_one_is_accepted():
    entry = replace(_entry("a", 0.5), env_split={"math": 0.6, "code": 0.4})

    assert set(add_task({}, entry)) == {"a"}


def test_a_negative_env_share_is_refused():
    entry = replace(_entry("a", 0.5), env_split={"math": 1.2, "code": -0.2})

    with pytest.raises(RegistryError, match="env_split"):
        add_task({}, entry)


def test_an_env_split_round_trips():
    entries = {"a": replace(_entry("a", 0.5), env_split={"math": 0.6, "code": 0.4})}

    assert parse_registry(render_registry(entries)) == entries
```

And to `tests/unit/test_task_config.py`:

```python
def test_env_caps_are_the_cap_times_the_split():
    config = _resolve(
        {"default": _entry(env_split={"math": 0.6, "code": 0.4})}
    )

    assert config.env_caps == {"math": 0.3, "code": 0.2}


def test_no_env_split_spreads_the_cap_evenly_over_the_profile():
    config = _resolve({"default": _entry()})

    assert sum(config.env_caps.values()) == pytest.approx(config.emission_cap)
```

Adapt `_entry` in each file to accept the new keyword; read the existing helper first. In `test_task_config.py`, `_resolve` passes a profile — use its environment names so the even spread has something to spread over.

- [ ] **Step 2: Run the tests to verify they fail**

Run, one at a time:

```bash
python -m pytest tests/unit/test_task_registry_rule.py -q
python -m pytest tests/unit/test_task_config.py -q
```

Expected: FAIL — `TypeError` on the unexpected `env_split` keyword.

- [ ] **Step 3: Implement in the registry**

Add `env_split: Mapping[str, float] | None` to `TaskEntry` (it is a frozen, slotted dataclass — add the field, do not subclass). In `validate_entry`, after the price-parameter checks:

```python
    if entry.env_split is not None:
        if not isinstance(entry.env_split, Mapping) or not entry.env_split:
            raise RegistryError("env_split must be a non-empty object")
        total = 0.0
        for environment, share in entry.env_split.items():
            value = _number(share, f"env_split[{environment}]")
            if not 0.0 <= value <= 1.0:
                raise RegistryError(
                    f"env_split[{environment}] must be between 0.0 and 1.0"
                )
            total += value
        if abs(total - 1.0) > _SUM_TOLERANCE:
            raise RegistryError(
                f"env_split shares total {total:.4f}, which is not 1.0"
            )
```

`_number` already rejects booleans, non-numbers and values too large to convert, so the share checks inherit those guarantees. In `parse_registry`, read `env_split` from the entry body with `body.get("env_split")` and pass it through; in `render_registry`, emit it beside `status`.

- [ ] **Step 4: Implement in `task_config.py` and the CLI**

In `task_config.py`, `TaskConfig` gains `env_caps: dict[str, float]`, and `resolve_task_config` fills it after `emission_cap` (currently line 75):

```python
    cap = float(entry.params["cap"])
    environments = list(generation_contract.get("environments") or ())
    if entry.env_split is not None:
        env_caps = {
            environment: cap * float(share)
            for environment, share in entry.env_split.items()
        }
    elif environments:
        env_caps = {environment: cap / len(environments) for environment in environments}
    else:
        env_caps = {}
```

Read `to_generation_contract()`'s actual shape before relying on `environments` — if the contract nests them differently, take the names from the resolved `ProtocolProfile` instead and say so in your report.

In `cli/main.py`, `build_task_entry` gains an `env_split` argument passed straight through, and `tasks_create` gains a `--env-split` option taking `math=0.6,code=0.4`, parsed into a dict. Refuse a split naming an environment the profile does not declare, with a message listing both sets — a silent mismatch would put a real budget decision on a fallback.

- [ ] **Step 5: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_task_registry_rule.py -q
python -m pytest tests/unit/test_task_registry_store.py -q
python -m pytest tests/unit/test_task_config.py -q
python -m pytest tests/unit/test_tasks_cli.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add reliquary/shared/task_registry.py reliquary/validator/task_config.py reliquary/cli/main.py tests/unit/
git commit -m "feat(tasks): declare how a task's budget divides between its environments

The task cap says how much a task may spend; env_split says how that divides.
Keeping them separate is the point: the total budget and the relative worth of
an environment are decisions that move at different times and for different
reasons. Absent a split, the cap spreads evenly, which is what the assembler
already did -- so an undeclared split is exactly today's behaviour.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4"
```

---

### Task 4: A pool per environment in the assembler

**Files:**
- Modify: `reliquary/validator/fill_closed_batch_assembler.py:77` (`window_pool`), `:88`, `:447-448` (`batch_pool_per_env`), `:496` (the `window_pool` property)
- Modify: `reliquary/validator/service.py` (the `FillClosedBatchAssembler(...)` construction)
- Test: `tests/unit/test_fill_closed_batch_assembler.py`

**Interfaces:**
- Consumes: `TaskConfig.env_caps` (Task 3).
- Produces: `FillClosedBatchAssembler(window_pool: float | Mapping[str, float])`.

Accepting either shape is what keeps the witness suites untouched: they construct the assembler with a scalar and must keep working, and a scalar must behave exactly as it does today.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_fill_closed_batch_assembler.py`:

```python
def test_a_scalar_pool_still_divides_evenly():
    """The identity case: today's behaviour, byte for byte."""
    assembler = _assembler(env_order=["math", "code"], window_pool=1.0)

    assert assembler.pool_for("math") == assembler.pool_for("code")
    assert assembler.pool_for("math") + assembler.pool_for("code") == pytest.approx(1.0)


def test_a_per_environment_pool_is_used_as_given():
    assembler = _assembler(
        env_order=["math", "code"], window_pool={"math": 0.6, "code": 0.4}
    )

    assert assembler.pool_for("math") == pytest.approx(0.6)
    assert assembler.pool_for("code") == pytest.approx(0.4)


def test_an_environment_missing_from_the_map_is_refused():
    with pytest.raises(ValueError, match="code"):
        _assembler(env_order=["math", "code"], window_pool={"math": 1.0})
```

Write `_assembler` to match how the existing tests in that file build one — read them first and reuse their fixture rather than inventing a second way.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_fill_closed_batch_assembler.py -q`
Expected: FAIL — `AttributeError: 'FillClosedBatchAssembler' object has no attribute 'pool_for'`.

- [ ] **Step 3: Implement**

In `__init__`, replace `self._window_pool = float(window_pool)` with:

```python
        # A scalar keeps today's even split; a map gives each environment its
        # own pool. Both shapes are accepted so a scalar caller is unchanged.
        if isinstance(window_pool, Mapping):
            missing = [e for e in self._env_order if e not in window_pool]
            if missing:
                raise ValueError(
                    f"window_pool is missing environments: {', '.join(missing)}"
                )
            self._pool_by_env = {e: float(window_pool[e]) for e in self._env_order}
        else:
            total = float(window_pool)
            share = total / len(self._env_order) if self._env_order else 0.0
            self._pool_by_env = {e: share for e in self._env_order}
        self._window_pool = sum(self._pool_by_env.values())
```

Add the accessor:

```python
    def pool_for(self, environment: str) -> float:
        return self._pool_by_env[environment]
```

Then at `:447`, `batch_pool_per_env` stops dividing the total by the environment count and reads `self.pool_for(environment)` divided by whatever per-batch factor the current expression uses. **Read that expression before changing it** — it currently divides by the environment count and by an emissions figure, and only the environment-count division is being replaced. Keep the `window_pool` property returning the total, so existing readers are unaffected.

In `service.py`, pass `window_pool=self._env_caps or self._emission_cap` at the `FillClosedBatchAssembler(...)` construction, where `self._env_caps` is the task config's `env_caps` stored alongside `self._emission_cap`.

- [ ] **Step 4: Run the tests to verify they pass**

Run, one command at a time — the last three are the witnesses and must pass untouched:

```bash
python -m pytest tests/unit/test_fill_closed_batch_assembler.py -q
python -m pytest tests/unit/test_v1_cutover.py -q
python -m pytest tests/unit/test_archive_window_content.py -q
python -m pytest tests/unit/test_task_archive_namespace.py -q
```

Expected: all pass, with the three witness files unmodified.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/fill_closed_batch_assembler.py reliquary/validator/service.py tests/unit/test_fill_closed_batch_assembler.py
git commit -m "feat(fill-closed): give each environment its own pool, not an even slice

The assembler divided one scalar evenly between environments, which is the
fixed split the task registry abolished one level up. It now accepts a map and
uses each environment's own pool; a scalar still divides evenly, so every
existing caller and every witness suite is unchanged.

How a pool divides WITHIN an environment is untouched -- that is the fixed
per-group policy, and this composes with it.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4"
```

---

### Task 5: A timeout is not a crash

**Files:**
- Modify: `reliquary/validator/service.py:433` (the aborted skip), `:4309`, `:4926`, `:5414` (the `window_status` writers), `:1878-1886` (the deadline)
- Modify: `reliquary/validator/emission_price.py` (`outcome_from_archive`, `outcomes_by_environment_from_archive`)
- Test: `tests/unit/test_window_timeout_status.py` (create)

**Interfaces:**
- Consumes: `EnvironmentOutcome`, `outcomes_by_environment_from_archive` (Task 2).
- Produces: archives carrying `window_status: "timed_out"`.

A crash leaves partially validated payloads and must pay nothing; a timeout leaves fully validated work that was simply not enough, and its timings are the shortage signal the controller needs. Collapsing them throws away the most informative window there is.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_window_timeout_status.py`:

```python
"""A timeout describes the market; a crash describes us."""

from __future__ import annotations

from reliquary.validator.emission_price import (
    outcome_from_archive,
    outcomes_by_environment_from_archive,
)


def _record(status: str) -> dict:
    return {
        "window_status": status,
        "window_open_round": 0,
        "window_close_round": 1000,
        "collect_ready_round": None,
        "collect_ready_round_by_environment": {"math": None, "code": 400},
        "training_rounds": 0.0,
        "validation_rounds": 500.0,
    }


def test_an_aborted_window_carries_no_price_signal():
    assert outcome_from_archive(_record("aborted")) is None
    assert outcomes_by_environment_from_archive(_record("aborted")) is None


def test_a_timed_out_window_carries_its_signal():
    outcomes = outcomes_by_environment_from_archive(_record("timed_out"))

    assert outcomes is not None
    assert outcomes["math"].filled is False
    assert outcomes["code"].filled is True


def test_a_timed_out_window_yields_no_ratio_for_anyone():
    """No training ran, so the denominator is undefined. Shortage is carried
    by `filled`, and inventing a denominator would inflate the ratio exactly
    on the worst windows."""
    outcomes = outcomes_by_environment_from_archive(_record("timed_out"))

    assert outcomes["math"].ratio is None
    assert outcomes["code"].ratio is None


def test_a_completed_window_does_yield_a_ratio():
    outcomes = outcomes_by_environment_from_archive(_record("completed"))

    assert outcomes["code"].ratio is not None


def test_a_completed_window_still_carries_its_signal():
    assert outcomes_by_environment_from_archive(_record("completed")) is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_window_timeout_status.py -q`
Expected: FAIL — `timed_out` is currently treated as a completed window by the string comparison, or the map adapter rejects it; read the failure before fixing.

- [ ] **Step 3: Implement the status**

In `outcomes_by_environment_from_archive`, a `timed_out` record must yield no
ratio for any environment: no training ran, so the denominator is undefined,
and `filled` already carries the shortage. Build those outcomes with
`incompressible_rounds=0.0`, which makes `EnvironmentOutcome.ratio` return
`None` by its own existing rule rather than through a second special case:

```python
    timed_out = record.get("window_status") == "timed_out"
    incompressible = 0.0 if timed_out else scalar.incompressible_rounds
```

and pass `incompressible` where the outcomes are built.

`outcome_from_archive` and `outcomes_by_environment_from_archive` both skip on `record.get("window_status", "completed") == "aborted"`. That comparison already admits `timed_out`, so confirm by running the test; if a different guard rejects it, widen that guard rather than the comparison.

In `service.py`, the window loop's deadline at `:1878` decides a timeout. Where that path currently routes to `_enqueue_aborted_window`, route it instead to a seal that writes `"window_status": "timed_out"` and carries the rewards the assembler has already accrued, with **no training payload**. Read `_enqueue_aborted_window` and the completed-seal path at `:4309` and build the timeout seal from the two: the rewards and archive fields of the first, the tombstone's refusal to write training data.

**Pay only fully assembled batches.** `close(allow_partial=True)` can force a
final incomplete batch; the timeout seal must not pay it. The pool divides per
assembled batch, so a half-empty batch would give each of its groups a larger
share than a full one — an incentive to arrive in the last carriage. Take the
rewards the assembler had already accrued for complete batches and stop there.
Add a test that a group present only in the partial remainder receives nothing,
and that the groups of the complete batches are paid exactly what they would
have been paid had the window completed.

Leave every other caller of `_enqueue_aborted_window` alone — a crash keeps paying nothing.

- [ ] **Step 4: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_window_timeout_status.py -q
python -m pytest tests/unit/test_emission_price_archive_adapter.py -q
python -m pytest tests/unit/test_archive_window_content.py -q
python -m pytest tests/unit/test_v1_cutover.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/service.py reliquary/validator/emission_price.py tests/unit/test_window_timeout_status.py
git commit -m "feat(window): a timeout pays what was earned and is not a crash

outcome_from_archive skipped aborted windows because their timings describe
the abort rather than the market. A timeout describes the market exactly --
it is the shortage signal -- so collapsing the two threw away the most
informative window there is, and paid nothing to miners whose work had been
admitted, proven and graded.

A timed-out window now seals with the rewards already accrued and no training
payload: the work was done, the batch is partial, and a partial batch biases
the update. A crash is unchanged.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4"
```

---

### Task 6: The circuit breaker

**Files:**
- Modify: `reliquary/validator/emission_price.py` (`PriceParams`, `advance_by_environment`)
- Test: `tests/unit/test_emission_price_controller.py`

**Interfaces:**
- Consumes: `EnvironmentOutcome`, `advance_by_environment` (Task 2).
- Produces: `PriceParams.breaker_timeouts: int`; `PriceDecision.regime == "frozen"`.

Because every environment must fill, a structurally dry environment no longer degrades the gradient — it halts the subnet. Without a freeze its price rises forever while nothing unblocks, so the breaker is mandatory, not optional. It freezes the price and raises an alarm; it does **not** remove the environment from the training mix, because that would hand the market the gradient's composition through the back door.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_emission_price_controller.py`:

```python
def test_a_chronically_dry_environment_stops_escalating():
    from reliquary.validator.emission_price import (
        PRODUCTION_PRICE_PARAMS,
        EnvironmentOutcome,
        PriceState,
        advance_by_environment,
    )

    dry = [EnvironmentOutcome("math", 0, 1000, None, 500.0)] * 3
    states = {"math": PriceState(price=0.5, last_good=0.5, recent_fill_prices=(0.5,))}

    decisions = advance_by_environment(states, {"math": dry}, PRODUCTION_PRICE_PARAMS)

    assert decisions["math"].regime == "frozen"
    assert decisions["math"].price == 0.5


def test_two_dry_windows_still_snap():
    from reliquary.validator.emission_price import (
        PRODUCTION_PRICE_PARAMS,
        EnvironmentOutcome,
        PriceState,
        advance_by_environment,
    )

    dry = [EnvironmentOutcome("math", 0, 1000, None, 500.0)] * 2
    states = {"math": PriceState(price=0.5, last_good=0.5, recent_fill_prices=(0.5,))}

    decisions = advance_by_environment(states, {"math": dry}, PRODUCTION_PRICE_PARAMS)

    assert decisions["math"].regime == "snap"
    assert decisions["math"].price > 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_emission_price_controller.py -q`
Expected: FAIL — `AttributeError` on `breaker_timeouts`, or the first test sees `snap`.

- [ ] **Step 3: Implement**

Add to `PriceParams`:

```python
    # How many consecutive unfilled windows before an environment's price stops
    # escalating: past this, the likelier explanation is that the environment is
    # broken on our side, not that the market is short.
    breaker_timeouts: int
```

and to `PRODUCTION_PRICE_PARAMS`:

```python
    # A starting point, calibrated in shadow like every other number here.
    breaker_timeouts=3,
```

In `advance_by_environment`, before delegating to `advance`, freeze an environment whose trailing run is all unfilled and long enough:

```python
        trailing = list(recent)[-params.breaker_timeouts:]
        if (
            len(trailing) >= params.breaker_timeouts
            and all(not outcome.filled for outcome in trailing)
        ):
            logger.critical(
                "environment %s has not filled for %d consecutive windows; "
                "freezing its price at %.4f -- treat this as broken on our side "
                "until shown otherwise",
                environment, params.breaker_timeouts, state.price,
            )
            decisions[environment] = PriceDecision(
                price=state.price,
                last_good=state.last_good,
                recent_fill_prices=state.recent_fill_prices,
                r=None,
                r_smoothed=None,
                regime="frozen",
            )
            continue
```

Rewrite the comprehension as an explicit loop to hold that branch. Read `PriceDecision`'s real fields first and construct it with all of them; it gained a rolling-minimum field earlier and the names must match exactly. Add `logger = logging.getLogger(__name__)` at module scope if the file does not already have one.

- [ ] **Step 4: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_emission_price_controller.py -q
python -m pytest tests/unit/test_emission_price_archive_adapter.py -q
python -m pytest tests/unit/test_archive_carries_price_shadow.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/emission_price.py tests/unit/test_emission_price_controller.py
git commit -m "feat(price): stop escalating an environment that never fills

Since every environment must fill for a window to close, a structurally dry
one no longer degrades the gradient -- it halts the subnet. Its price would
otherwise climb forever while nothing unblocks, paying every miner more
because one environment is broken on our side.

The breaker freezes the price and says so loudly. It deliberately does NOT
drop the environment from the training mix: that would change the gradient's
composition automatically and hand the market a decision that is ours.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4"
```

---

## Out of this plan

- **Arming the price.** The controller stays in shadow; `window_pool` is still the declared cap, not the discovered price. Arming still needs the walk seeded from the last archive, or a restart hands miners the full pool.
- **Per-environment controller parameters.** Shared parameters, independent prices.
- **Independent per-environment sealing.** Rejected: it would give the market the gradient's composition.
- **Removing an environment from the mix** when the breaker trips — a human decides that.
- **Stage timing instrumentation** (`training_rounds` / `validation_rounds`). Separable, and it is what uncaps `r` and makes a proportional descent implementable.
- **Raising `FILL_CLOSED_MAX_SECONDS`.** An operator decision, not a code change.
