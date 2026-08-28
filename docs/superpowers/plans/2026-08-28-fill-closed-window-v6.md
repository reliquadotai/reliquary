# Fill-Closed Window (protocol v6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the fill-closed window as a coexisting `v6` protocol profile — admission ordered by production rate, GRAIL on arrival, the window closing when every environment holds its target of proven groups, a training batch emitted every 32 proven groups, and emission split by EOS-terminated completion tokens under a per-operator token-share cap.

**Architecture:** A new `ProtocolProfile` (`qwen3-4b-base-dapo-fill-closed-v6`) plus one capability gate select the new path. The auction path stays live and byte-identical for `v4`/`v5`. Three new pure modules carry the decisions (rate queue, fill accounting, token split); the batcher and service consult them only when the capability is active.

**Tech Stack:** Python 3.13, pytest, existing `GlobalProofScheduler` open-ended plans (`open_ended` / `extend` / `seal`, already on this branch).

**Spec:** `docs/superpowers/specs/2026-08-28-fill-closed-window-design.md`

**Base:** branch `design/fill-closed-v6`, which is
`checkpoint-epoch-scheduling-prototype` (PR #198) merged with the
fill-closed work. **The checkpoint-epoch capability is inherited, not
rewritten** — its commits stand as they are, gated by
`RELIQUARY_EXPERIMENTAL_CHECKPOINT_EPOCH_ENABLED`. v6 is a third regime
beside it and beside the production auction, each independently gated and
all three off by default. Nothing in this plan modifies epoch code.

## Global Constraints

- **Never change `v4` or `v5` behaviour.** `tests/unit/test_protocol_profiles.py` pins their canonical generation-contract bytes; those tests must pass untouched.
- **`RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED` defaults to `0`.** With it off, every new code path is unreachable and `/state`, `/health` and the archive are byte-identical to today.
- **`B_BATCH = 16`** groups per environment per optimizer step; 2 environments; **16 steps per window**; therefore **`target = 256` proven groups per environment**, 512 per window.
- **The pi_old contract does not change.** 16 optimizer steps share one published checkpoint. `ppo_ratio_outside_clip_ratio` must not move.
- **Admission gates cheaply before expensively.** Duplicate payload hash and environment-full are refused before any payload moves, grading runs, or GRAIL runs.
- **Two controls change status, not existence.** `robust_uncertain_reward_utility` becomes a rejection rule (utility `0.0` → refuse); content dedup becomes an economic control.
- **`proof_scheduler.py` is in `PROOF_PATH_FILES`.** This branch already invalidates the proof capacity qualification; a re-benchmark is required before any deploy.
- Run the suite as `TMPDIR=/tmp .venv/bin/python -m pytest -q --ignore=tests/gpu --ignore=tests/integration/test_grader_e2e.py --deselect tests/unit/test_admission_isolation.py::test_spawned_worker_deadline_is_terminal`. That deselected test fails 3/3 on pristine main on this hardware (6.45 s spawn+import against a 5.0 s budget) and is unrelated.

## File Structure

| File | Responsibility |
|---|---|
| `reliquary/validator/admission_priority.py` | **exists** — rate-ordered queue of precommits |
| `reliquary/validator/fill_window.py` | **new** — per-environment proven / in-flight / target accounting and the close decision |
| `reliquary/validator/token_rewards.py` | **new** — per-environment pool split by EOS-terminated tokens, with the operator share cap |
| `reliquary/protocol/profiles.py` | add the `v6` profile |
| `reliquary/constants.py` | v6-derived constants and the capability gate |
| `reliquary/validator/batcher.py` | arrival-time validation, fill accounting, progressive emission |
| `reliquary/validator/service.py` | the v6 window loop |

Tasks 1-4 are pure or additive and testable in isolation. Tasks 5-8 wire them in.

---

### Task 1: The v6 profile and the capability gate

**Files:**
- Modify: `reliquary/protocol/profiles.py` (append to `_PROFILE_VALUES`)
- Modify: `reliquary/constants.py` (after `EXPERIMENTAL_CHECKPOINT_EPOCH_*`, ~line 574)
- Test: `tests/unit/test_fill_closed_profile.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PROFILE_ID = "qwen3-4b-base-dapo-fill-closed-v6"`, `constants.FILL_CLOSED_ENABLED: bool`, `constants.FILL_CLOSED_TARGET_GROUPS_PER_ENV: int`, `constants.FILL_CLOSED_MAX_SECONDS: float`, `constants.FILL_CLOSED_MAX_OPERATOR_TOKEN_SHARE: float`.

- [ ] **Step 1: Write the failing test**

```python
"""v6 coexists: it is selectable, and it leaves v4/v5 untouched."""
from reliquary.protocol.profiles import resolve_protocol_profile


def test_v6_is_selectable_and_carries_the_v5_generation_contract():
    v5 = resolve_protocol_profile("qwen3-4b-base-dapo-reasoning-v5")
    v6 = resolve_protocol_profile("qwen3-4b-base-dapo-fill-closed-v6")

    assert v6.protocol_version == 6
    # v6 changes the WINDOW, not what a miner generates. Everything a miner
    # samples from must be byte-identical or the change is not what it claims.
    assert v6.sampling == v5.sampling
    assert v6.model_id == v5.model_id
    assert v6.model_revision == v5.model_revision
    assert {name: env.max_new_tokens for name, env in v6.environments.items()} == \
           {name: env.max_new_tokens for name, env in v5.environments.items()}


def test_v6_has_no_throughput_tiebreak():
    """There is no ranking in v6, so there is nothing to break ties in."""
    v6 = resolve_protocol_profile("qwen3-4b-base-dapo-fill-closed-v6")
    assert v6.throughput_tiebreak is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_fill_closed_profile.py -v`
Expected: FAIL — `resolve_protocol_profile` raises on the unknown id.

- [ ] **Step 3: Add the profile**

In `reliquary/protocol/profiles.py`, append inside `_PROFILE_VALUES` after the v5 entry. Copy v5's fields verbatim except `profile_id`, `protocol_version` and `throughput_tiebreak`:

```python
    ProtocolProfile(
        # Same model, same sampling, same budgets, same prompts as v5. v6
        # changes when a window ends and who gets admitted, never what a
        # miner generates -- so the generation contract is v5's, field for
        # field, and a diff of the two profiles shows only the window.
        profile_id="qwen3-4b-base-dapo-fill-closed-v6",
        model_id="Qwen/Qwen3-4B-Base",
        model_revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
        protocol_version=6,
        collection_seconds=100,
        upload_grace_seconds=33,
        prompt_encoding="raw",
        sampling=_SAMPLING_DAPO,
        environments={
            "openmathinstruct": EnvironmentProfile(
                max_new_tokens=8192,
                bft=None,
                answer_format="boxed",
                prompt_template=_MATH_REASONING_PROMPT,
            ),
            "opencodeinstruct": EnvironmentProfile(
                max_new_tokens=8192,
                bft=None,
                prompt_template=_CODE_REASONING_PROMPT,
            ),
        },
        # No ranking in v6, so no tie-break to rank with.
        throughput_tiebreak=None,
    ),
```

- [ ] **Step 4: Run the test and the profile pins**

Run: `.venv/bin/python -m pytest tests/unit/test_fill_closed_profile.py tests/unit/test_protocol_profiles.py -v`
Expected: PASS, and the v4/v5 contract-hash pins still pass.

- [ ] **Step 5: Add the capability gate and its constants**

In `reliquary/constants.py`, after the `EXPERIMENTAL_CHECKPOINT_EPOCH_*` block:

```python
# ────────────────  FILL-CLOSED WINDOW (v6)  ────────────────

# The v6 window ends when every environment holds its target of PROVEN
# groups rather than when a clock expires. Gated so the auction path stays
# live and byte-identical for v4/v5: a validator reaches this code only by
# selecting the v6 profile AND arming the capability.
FILL_CLOSED_ENABLED = _os.environ.get(
    "RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Proven groups that close one environment. 16 optimizer steps x B_BATCH.
FILL_CLOSED_TARGET_GROUPS_PER_ENV = int(_os.environ.get(
    "RELIQUARY_FILL_CLOSED_TARGET_GROUPS_PER_ENV",
    str(CHECKPOINT_PUBLISH_INTERVAL_WINDOWS * B_BATCH),
))
if FILL_CLOSED_TARGET_GROUPS_PER_ENV <= 0:
    raise ValueError(
        "RELIQUARY_FILL_CLOSED_TARGET_GROUPS_PER_ENV must be positive"
    )

# Backstop only. A window normally ends on its fill; this stops a stalled
# fleet holding one open forever, and seals whatever is proven.
FILL_CLOSED_MAX_SECONDS = float(_os.environ.get(
    "RELIQUARY_FILL_CLOSED_MAX_SECONDS", "1800"
))
if not _math.isfinite(FILL_CLOSED_MAX_SECONDS) or FILL_CLOSED_MAX_SECONDS <= 0:
    raise ValueError("RELIQUARY_FILL_CLOSED_MAX_SECONDS must be positive")

# Ceiling on one operator's share of an environment's accepted tokens.
# Counted in TOKENS, not groups: under per-token payment a group count
# bounds nothing, since an operator can take few very long groups. 0.34
# denies any single operator a third of an environment while leaving room
# for a fleet of three to fill it. Move it from measured concentration.
FILL_CLOSED_MAX_OPERATOR_TOKEN_SHARE = float(_os.environ.get(
    "RELIQUARY_FILL_CLOSED_MAX_OPERATOR_TOKEN_SHARE", "0.34"
))
if not 0.0 < FILL_CLOSED_MAX_OPERATOR_TOKEN_SHARE <= 1.0:
    raise ValueError(
        "RELIQUARY_FILL_CLOSED_MAX_OPERATOR_TOKEN_SHARE must be in (0, 1]"
    )
```

- [ ] **Step 6: Verify the gate is off by default**

Run: `.venv/bin/python -c "from reliquary import constants; assert not constants.FILL_CLOSED_ENABLED; assert constants.FILL_CLOSED_TARGET_GROUPS_PER_ENV == 256; print('gate off, target 256')"`
Expected: `gate off, target 256`

- [ ] **Step 7: Commit**

```bash
git add reliquary/protocol/profiles.py reliquary/constants.py tests/unit/test_fill_closed_profile.py
git commit -m "feat(protocol): add the v6 fill-closed profile, gated off"
```

---

### Task 2: Fill accounting and the close decision

**Files:**
- Create: `reliquary/validator/fill_window.py`
- Test: `tests/unit/test_fill_window.py`

**Interfaces:**
- Consumes: `constants.FILL_CLOSED_TARGET_GROUPS_PER_ENV`.
- Produces: `FillState(targets: Mapping[str, int])` with `may_admit(env) -> bool`, `reserve(env) -> None`, `release(env) -> None`, `record_proven(env) -> None`, `is_closed() -> bool`, `snapshot() -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
"""Admission stops on proven + in-flight; the close fires on proven alone."""
import pytest

from reliquary.validator.fill_window import FillState

MATH, CODE = "openmathinstruct", "opencodeinstruct"


def _state(target=2):
    return FillState(targets={MATH: target, CODE: target})


def test_admission_counts_in_flight_work():
    """Gating admission on proven alone would over-admit by the whole proof
    pipeline depth: every reservation still in flight would look like room."""
    state = _state(target=2)
    state.reserve(MATH)
    state.reserve(MATH)

    assert state.may_admit(MATH) is False


def test_the_close_ignores_in_flight_work():
    """Closing on proven + in-flight would close on work that may still fail
    GRAIL, and a failed proof is not a group."""
    state = _state(target=2)
    state.reserve(MATH)
    state.reserve(MATH)
    state.record_proven(MATH)
    state.record_proven(CODE)
    state.record_proven(CODE)

    assert state.is_closed() is False


def test_a_released_reservation_reopens_capacity():
    """A failed grade or proof must not consume a slot forever, or a run of
    failures would stall the window below its target."""
    state = _state(target=1)
    state.reserve(MATH)
    assert state.may_admit(MATH) is False

    state.release(MATH)

    assert state.may_admit(MATH) is True


def test_the_window_closes_only_when_every_environment_is_full():
    state = _state(target=1)
    state.record_proven(MATH)
    assert state.is_closed() is False

    state.record_proven(CODE)

    assert state.is_closed() is True


def test_releasing_what_was_never_reserved_is_a_bug_not_a_no_op():
    """Silent tolerance here would hide a double-release, which would let the
    environment admit past its target."""
    state = _state()
    with pytest.raises(ValueError):
        state.release(MATH)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_fill_window.py -v`
Expected: FAIL — `No module named 'reliquary.validator.fill_window'`

- [ ] **Step 3: Write the implementation**

```python
"""Per-environment fill accounting for the v6 window.

Pure and dependency-free: it counts, it decides, it touches no submission,
no model and no GPU. The window's whole control loop is these two rules —

    admit while   proven + in_flight < target
    close when    proven >= target, for every environment

which are deliberately asymmetric. Gating admission on ``proven`` alone
would over-admit by the entire proof pipeline depth, because every
reservation still in flight would look like room. Closing on
``proven + in_flight`` would close on work that may still fail GRAIL, and a
group that fails its proof is not a group.
"""

from __future__ import annotations

from typing import Mapping


class FillState:
    def __init__(self, targets: Mapping[str, int]) -> None:
        if not targets:
            raise ValueError("fill state requires at least one environment")
        if any(int(target) <= 0 for target in targets.values()):
            raise ValueError("fill targets must be positive")
        self._targets = {str(name): int(target) for name, target in targets.items()}
        self._proven = {name: 0 for name in self._targets}
        self._in_flight = {name: 0 for name in self._targets}

    def _known(self, environment: str) -> str:
        if environment not in self._targets:
            raise ValueError(f"unknown environment {environment!r}")
        return environment

    def may_admit(self, environment: str) -> bool:
        name = self._known(environment)
        committed = self._proven[name] + self._in_flight[name]
        return committed < self._targets[name]

    def reserve(self, environment: str) -> None:
        self._in_flight[self._known(environment)] += 1

    def release(self, environment: str) -> None:
        name = self._known(environment)
        if self._in_flight[name] <= 0:
            raise ValueError(f"no reservation to release for {name!r}")
        self._in_flight[name] -= 1

    def record_proven(self, environment: str) -> None:
        name = self._known(environment)
        if self._in_flight[name] > 0:
            self._in_flight[name] -= 1
        self._proven[name] += 1

    def is_closed(self) -> bool:
        return all(
            self._proven[name] >= target
            for name, target in self._targets.items()
        )

    def snapshot(self) -> dict:
        return {
            "targets": dict(self._targets),
            "proven": dict(self._proven),
            "in_flight": dict(self._in_flight),
            "closed": self.is_closed(),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_fill_window.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/fill_window.py tests/unit/test_fill_window.py
git commit -m "feat(window): per-environment fill accounting and the close rule"
```

---

### Task 3: Per-token reward split with the operator cap

**Files:**
- Create: `reliquary/validator/token_rewards.py`
- Test: `tests/unit/test_token_rewards.py`

**Interfaces:**
- Consumes: `constants.FILL_CLOSED_MAX_OPERATOR_TOKEN_SHARE`.
- Produces: `AcceptedGroup(hotkey: str, operator_id: str, eos_tokens: int)` and `split_environment_pool(groups: Sequence[AcceptedGroup], *, pool: float, max_operator_share: float) -> dict[str, float]` keyed by hotkey.

- [ ] **Step 1: Write the failing tests**

```python
"""Emission is divided by EOS-terminated completion tokens, not by slot."""
from reliquary.validator.token_rewards import AcceptedGroup, split_environment_pool


def _g(hotkey, tokens, operator=None):
    return AcceptedGroup(
        hotkey=hotkey, operator_id=operator or hotkey, eos_tokens=tokens
    )


def test_the_pool_is_divided_in_proportion_to_tokens():
    """Under a flat per-slot share, revenue per GPU-second is proportional to
    1/L: halving response length doubles income. Dividing by tokens makes it
    independent of length, so the policy decides how long to reason."""
    rewards = split_environment_pool(
        [_g("short", 1_000), _g("long", 9_000)],
        pool=1.0,
        max_operator_share=1.0,
    )

    assert rewards["short"] == 0.1
    assert rewards["long"] == 0.9


def test_only_eos_terminated_tokens_count():
    """Callers pass EOS-terminated tokens only. A group with none earns
    nothing, which is what keeps padding a strictly negative margin."""
    rewards = split_environment_pool(
        [_g("terminated", 1_000), _g("padded", 0)],
        pool=1.0,
        max_operator_share=1.0,
    )

    assert rewards["terminated"] == 1.0
    assert rewards.get("padded", 0.0) == 0.0


def test_an_operator_over_the_cap_is_clipped_and_the_rest_reflows():
    """Bounded in TOKENS, not groups: under per-token payment a group count
    bounds nothing, since an operator can take few very long groups."""
    rewards = split_environment_pool(
        [
            _g("whale-a", 9_000, operator="whale"),
            _g("whale-b", 9_000, operator="whale"),
            _g("small", 2_000, operator="small"),
        ],
        pool=1.0,
        max_operator_share=0.5,
    )

    whale = rewards["whale-a"] + rewards["whale-b"]
    assert abs(whale - 0.5) < 1e-9
    assert abs(rewards["small"] - 0.5) < 1e-9
    assert abs(rewards["whale-a"] - rewards["whale-b"]) < 1e-9


def test_an_empty_environment_pays_nothing_rather_than_dividing_by_zero():
    assert split_environment_pool([], pool=1.0, max_operator_share=1.0) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_token_rewards.py -v`
Expected: FAIL — `No module named 'reliquary.validator.token_rewards'`

- [ ] **Step 3: Write the implementation**

```python
"""Divide an environment's pool by EOS-terminated completion tokens.

Replaces ``slot_share = pool / B_BATCH``. Under the flat share a group costs
``16L/r`` rounds at rate ``r`` and length ``L`` and pays the same whatever
``L`` is, so revenue per GPU-second is proportional to ``1/L`` and halving
response length doubles income. Dividing by tokens removes that.

Only EOS-terminated rollouts contribute. The caller does that filtering; this
module simply never invents value for a token it was not given. That
restriction is load-bearing rather than cosmetic: the flat share is currently
one of four barriers against EOS suppression, and per-token payment removes
it. Paying only terminated tokens restores a strictly negative margin on
padding.

Pure and deterministic: two validators replaying the same archive must reach
the same numbers bit for bit.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AcceptedGroup:
    hotkey: str
    operator_id: str
    eos_tokens: int


def split_environment_pool(
    groups: Sequence[AcceptedGroup],
    *,
    pool: float,
    max_operator_share: float,
) -> dict[str, float]:
    """Return ``{hotkey: share}`` summing to ``pool`` over paying groups."""
    paying = [group for group in groups if group.eos_tokens > 0]
    if not paying:
        return {}

    by_operator: dict[str, int] = defaultdict(int)
    for group in paying:
        by_operator[group.operator_id] += group.eos_tokens
    total = sum(by_operator.values())

    # Clip operators over the cap, then reflow what they gave up across the
    # operators still under it, repeating until the split is stable. One pass
    # would leave the reflow itself pushing a second operator over the cap.
    ceiling = pool * max_operator_share
    weights = {
        operator: pool * tokens / total for operator, tokens in by_operator.items()
    }
    for _ in range(len(weights)):
        over = {op: w for op, w in weights.items() if w > ceiling + 1e-12}
        if not over:
            break
        spare = sum(w - ceiling for w in over.values())
        under = {op: w for op, w in weights.items() if op not in over}
        under_total = sum(under.values())
        for operator in over:
            weights[operator] = ceiling
        if under_total <= 0:
            break
        for operator, weight in under.items():
            weights[operator] = weight + spare * weight / under_total

    rewards: dict[str, float] = {}
    for group in paying:
        operator_tokens = by_operator[group.operator_id]
        rewards[group.hotkey] = rewards.get(group.hotkey, 0.0) + (
            weights[group.operator_id] * group.eos_tokens / operator_tokens
        )
    return rewards
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_token_rewards.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/token_rewards.py tests/unit/test_token_rewards.py
git commit -m "feat(rewards): split an environment pool by EOS-terminated tokens"
```

---

### Task 4: The robust utility becomes a rejection rule

**Files:**
- Modify: `reliquary/validator/admission.py` (the reward-admission path, near the existing sigma gate)
- Test: `tests/unit/test_robust_utility_rejection.py`

**Interfaces:**
- Consumes: `reliquary.validator.difficulty_auction.robust_uncertain_reward_utility`.
- Produces: `reliquary.validator.admission.robust_utility_admits(rewards, *, sigma_min, truncated_indices, attainable_rewards) -> bool`.

**Why this task exists:** with no auction, "utility 0" prices nothing. Today the manufactured zero — suppressing EOS on a correct rollout to push an all-correct group into the sigma zone — is defeated by *pricing* the group under its least favourable interpretation. Removing the valuation without retranslating this reopens that path at exactly the moment per-token payment makes it profitable.

- [ ] **Step 1: Write the failing test**

```python
"""A group whose least-favourable reading leaves the zone is refused."""
from reliquary.validator.admission import robust_utility_admits


def test_a_manufactured_zero_is_refused_not_priced():
    """All 16 rollouts correct is out of zone and worthless. Break one by
    suppressing EOS and the observed vector looks in-zone -- but the truncated
    rollout may have been correct, and that reading is out of zone again."""
    rewards = [1.0] * 15 + [0.0]

    assert robust_utility_admits(
        rewards,
        sigma_min=0.24,
        truncated_indices=(15,),
        attainable_rewards=(0.0, 1.0),
    ) is False


def test_an_honest_in_zone_group_is_admitted():
    rewards = [1.0] * 8 + [0.0] * 8

    assert robust_utility_admits(
        rewards,
        sigma_min=0.24,
        truncated_indices=(),
        attainable_rewards=(0.0, 1.0),
    ) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_robust_utility_rejection.py -v`
Expected: FAIL — `cannot import name 'robust_utility_admits'`

- [ ] **Step 3: Write the implementation**

Append to `reliquary/validator/admission.py`:

```python
def robust_utility_admits(
    rewards: Sequence[float],
    *,
    sigma_min: float,
    truncated_indices: Iterable[int] = (),
    attainable_rewards: Iterable[float] = (0.0, 1.0),
) -> bool:
    """Whether a group survives its least favourable interpretation.

    The auction defended against the manufactured zero by PRICING it: the
    gated utility is minimised over every joint assignment of the truncated
    rollouts' reward lattice, and the gate returns 0.0 below SIGMA_MIN, so a
    manipulated group could never score above its honest value.

    With no auction there is no price, so the same computation has to become
    an admission decision. Utility 0.0 means some true reading of this group
    is out of zone; refuse it.
    """
    from reliquary.validator.difficulty_auction import (
        robust_uncertain_reward_utility,
    )

    return robust_uncertain_reward_utility(
        rewards,
        sigma_min=sigma_min,
        uncertain_indices=truncated_indices,
        attainable_rewards=attainable_rewards,
    ) > 0.0
```

Add `Iterable` and `Sequence` to the `typing` import at the top of the file if absent.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_robust_utility_rejection.py -v`
Expected: 2 passed

- [ ] **Step 5: Pin content dedup as an economic control**

Under per-slot payment a resubmitted group won a duplicate slot at worst.
Under per-token payment it collects the same tokens twice, so the existing
check stops being about data quality and starts being about money. It must be
airtight rather than best-effort, and the precommit gives it for free — at the
hash, before any payload moves.

```python
def test_a_duplicate_payload_hash_is_refused_at_precommit(monkeypatch):
    """Same group twice is the same tokens paid twice."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _make_batcher()
    batcher.mark_window_opened()
    digest = "a" * 64

    assert batcher._register_payload_digest(digest) is True
    assert batcher._register_payload_digest(digest) is False
```

`_register_payload_digest` records into a per-window set and returns whether
the digest was new. Call it from the precommit path before any capacity is
reserved.

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/admission.py reliquary/validator/batcher.py tests/unit/test_robust_utility_rejection.py
git commit -m "feat(admission): robust utility rejects, and dedup guards payment"
```

---

### Task 5: Feed admission from the rate-ordered queue

**Files:**
- Modify: `reliquary/validator/batcher.py` — the precommit registration path (~line 1243) and the reveal path
- Test: `tests/unit/test_rate_ordered_admission.py`

**Interfaces:**
- Consumes: `ThroughputAdmissionQueue` (already on this branch, `reliquary/validator/admission_priority.py`).
- Produces: `GrpoWindowBatcher.admission_queue: ThroughputAdmissionQueue | None`, and `_next_admission(environment) -> QueuedPrecommit | None`.

**Why this task exists:** closing on a fill hands the slots still open near the close to whoever finishes first — systematically whoever produced the shortest rollouts. Per-token payment does not fix that: a long group has to get *in* before it can be paid. Ordering the queue by production rate is what stops length deciding who is admitted, and it must sit in front of the proof path from the start or the bias ships with the first window.

- [ ] **Step 1: Write the failing test**

```python
"""A precommit that lands later is served first if it was produced faster."""
from tests.unit.test_grpo_window_batcher import _make_batcher


def test_a_faster_later_precommit_is_admitted_before_a_slower_earlier_one(
    monkeypatch,
):
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _make_batcher()
    batcher.mark_window_opened()
    batcher.admission_queue = batcher_module.ThroughputAdmissionQueue(
        window_opened_at=batcher.window_opened_at
    )
    env = "openmathinstruct"

    # slow: 1000 bytes over 50 s. fast: 9000 bytes over 60 s, arriving LAST.
    batcher.admission_queue.offer(
        receipt_id="slow", hotkey="a", environment=env,
        payload_bytes=1_000, arrived_at=batcher.window_opened_at + 50.0,
    )
    batcher.admission_queue.offer(
        receipt_id="fast", hotkey="b", environment=env,
        payload_bytes=9_000, arrived_at=batcher.window_opened_at + 60.0,
    )

    assert batcher._next_admission(env).receipt_id == "fast"
    assert batcher._next_admission(env).receipt_id == "slow"
    assert batcher._next_admission(env) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_rate_ordered_admission.py -v`
Expected: FAIL — `GrpoWindowBatcher` has no attribute `admission_queue`

- [ ] **Step 3: Implement**

In `GrpoWindowBatcher.__init__`, beside the other v6 state:

```python
        # v6 only. Precommits wait here until the validator has budget to
        # validate one, ordered by production rate rather than arrival.
        self.admission_queue: ThroughputAdmissionQueue | None = None
```

Set it in the v6 window-open path, `ThroughputAdmissionQueue(window_opened_at=self.window_opened_at)`. Register every accepted precommit into it at the point where `_claim_upload_precommit` succeeds, passing the signed `payload_bytes` and the validator-observed arrival. Then:

```python
    def _next_admission(self, environment: str):
        """The highest-rate precommit waiting for this environment."""
        if self.admission_queue is None:
            return None
        return self.admission_queue.take_best(environment)
```

**Both terms of the rate are outside the miner's control.** `payload_bytes` comes from the signed precommit and is enforced against the upload by `_precommit_matches_submission`; elapsed comes from validator-observed arrivals, measured from that hotkey's previous arrival rather than from window open — measured from open, a miner's Nth precommit shows elapsed `N × generation_time` and only its first submission would ever compete.

- [ ] **Step 4: Run the test and the precommit suites**

Run: `.venv/bin/python -m pytest tests/unit/test_rate_ordered_admission.py tests/unit/test_admission_priority.py tests/unit/test_validator_server.py -v`
Expected: PASS, and the auction precommit path unchanged.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_rate_ordered_admission.py
git commit -m "feat(v6): feed admission from the rate-ordered precommit queue"
```

---

### Task 6: Prove on arrival

**Files:**
- Modify: `reliquary/validator/batcher.py` — the admission path around `_pending.append` (~line 2565 and ~3121), and `_seal_batch_inner` (~line 5680)
- Test: `tests/unit/test_prove_on_arrival.py`

**Interfaces:**
- Consumes: `FillState` (Task 2), `robust_utility_admits` (Task 4), `_next_admission` (Task 5), `GlobalProofScheduler.extend` / `seal` (already on this branch).
- Produces: `GrpoWindowBatcher.fill_state: FillState | None`, and `_submit_arrival_proof(pending) -> None` which reserves, extends the open-ended plan, and records proven or releases.

**Why this task exists:** the close rule counts proven groups. GRAIL runs at seal today (`batcher.py`: admission *"never runs GRAIL"*), so a proven count is identically zero until the window is already over. Proving on arrival is what makes the close decidable and what lets proof work overlap collection.

- [ ] **Step 1: Write the failing test**

```python
"""Under v6 a submission is proven when it arrives, not at seal."""
from tests.unit.test_grpo_window_batcher import _make_batcher


def test_an_arriving_submission_is_extended_onto_the_open_plan(monkeypatch):
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    extended = []
    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 4, "opencodeinstruct": 4}
    )
    batcher._extend_proof_plan = lambda candidates: extended.extend(candidates)

    batcher._submit_arrival_proof(_pending_stub(prompt_idx=1))

    assert len(extended) == 1
    assert batcher.fill_state.snapshot()["in_flight"]["openmathinstruct"] == 1
```

Define `_pending_stub` in the test file by copying the `PendingSubmission` construction from `tests/unit/test_batch_fill_offset.py::_pending`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_prove_on_arrival.py -v`
Expected: FAIL — `GrpoWindowBatcher` has no attribute `fill_state`

- [ ] **Step 3: Implement the arrival proof path**

In `GrpoWindowBatcher.__init__`, beside the other v6 state:

```python
        # v6 only. None on the auction path, which proves at seal.
        self.fill_state: FillState | None = None
        self._open_proof_plan_id: str | None = None
```

Add the method, and call it from the admission path immediately after the
`_pending.append(pending)` sites, guarded by `if FILL_CLOSED_ENABLED and self.fill_state is not None:`

```python
    def _submit_arrival_proof(self, pending: PendingSubmission) -> None:
        """Reserve capacity and hand the group to the open-ended plan.

        The two cheap refusals (duplicate hash, environment full) have already
        run; this is the expensive half. A reservation that fails its grade or
        its proof is released, which immediately reopens capacity for the next
        arrival -- without that, a run of failures would stall the window below
        its target.
        """
        environment = str(getattr(self.env, "name", ""))
        if not self.fill_state.may_admit(environment):
            return
        self.fill_state.reserve(environment)
        try:
            self._extend_proof_plan([self._ranked_proof_for(pending)])
        except Exception:
            self.fill_state.release(environment)
            raise
```

`_ranked_proof_for(pending)` builds a `RankedProof` exactly as
`_prove_ranked_scheduled` does today (job id, monotonically increasing rank
from an arrival counter, `prompt_key`, payload, resources). Reuse that code
rather than rewriting it — extract it into a helper and call it from both.

**The rank must increase monotonically in dispatch order.** `extend` refuses a
rank behind the plan's existing work, because decisions are applied in rank
order through `next_apply_index` and a candidate landing behind it would be
applied never or twice. This is why the rate queue sits in FRONT of the
scheduler rather than inside it: the queue reorders, and `extend` is called
with the current best when capacity frees, so ranks stay monotone in dispatch
order.

- [ ] **Step 3b: Restate the per-window proof budgets as rates**

Proving on arrival changes the shape of the work, so the seal-time envelopes
stop describing it. Under the v6 gate:

- `MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW` is superseded by the open-ended
  plan's `required_passes` (the environment target) — the plan stops
  dispatching at `target_reached` on its own.
- `MAX_PROOF_WALL_SECONDS = 240` bounded one seal burst. Under v6 the
  equivalent bound is `FILL_CLOSED_MAX_SECONDS` on the window as a whole,
  which the backstop in Task 7 enforces; the per-seal wall is not consulted.
- The fail-closed capacity qualification still applies and still gates
  startup. `proof_scheduler.py` is in `PROOF_PATH_FILES`, so this branch has
  already invalidated the manifest and a re-benchmark is required regardless.

Assert the supersession rather than leaving it implicit:

```python
def test_v6_does_not_consult_the_seal_time_proof_wall(monkeypatch):
    """The wall bounded ONE seal burst. v6 has no burst -- it proves
    continuously -- so the window backstop is the only time bound."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(batcher_module, "MAX_PROOF_WALL_SECONDS", 0.0)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 4, "opencodeinstruct": 4}
    )
    batcher.mark_window_opened()

    # A zero wall would abort the auction path instantly; v6 must ignore it.
    assert batcher.poll_deadline() is False
```

- [ ] **Step 4: Run the test and the whole batcher suite**

Run: `.venv/bin/python -m pytest tests/unit/test_prove_on_arrival.py tests/unit/test_grpo_window_batcher.py tests/unit/test_deferred_proof.py -v`
Expected: PASS, and every auction-path test unchanged — with `FILL_CLOSED_ENABLED` off, none of this code is reached.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_prove_on_arrival.py
git commit -m "feat(v6): prove each submission when it arrives"
```

---

### Task 7: The window closes on its fill, and emits as it goes

**Files:**
- Modify: `reliquary/validator/batcher.py` — `poll_deadline` (~line 1650)
- Modify: `reliquary/validator/service.py` — the window loop, beside `_train_and_publish`
- Test: `tests/unit/test_fill_close_and_emit.py`

**Interfaces:**
- Consumes: `FillState` (Task 2), the arrival proof path (Task 6).
- Produces: the batcher seals when `fill_state.is_closed()`; a training payload is emitted every `2 * B_BATCH` proven groups.

- [ ] **Step 1: Write the failing tests**

```python
"""The window ends on its fill, and batches leave while it is still open."""
from reliquary.constants import B_BATCH
from tests.unit.test_grpo_window_batcher import _make_batcher


def test_the_window_seals_when_every_environment_is_full(monkeypatch):
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 1, "opencodeinstruct": 1}
    )
    batcher.mark_window_opened()

    batcher.fill_state.record_proven("openmathinstruct")
    assert batcher.poll_deadline() is False

    batcher.fill_state.record_proven("opencodeinstruct")

    assert batcher.poll_deadline() is True


def test_a_batch_is_emitted_every_b_batch_proven_groups(monkeypatch):
    """16 x 32 = 512 is arithmetic, not a schedule the miner can see."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    emitted = []
    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 64, "opencodeinstruct": 64}
    )
    batcher._emit_training_batch = lambda: emitted.append(1)

    for _ in range(B_BATCH):
        batcher.fill_state.record_proven("openmathinstruct")
        batcher.fill_state.record_proven("opencodeinstruct")
        batcher._maybe_emit_batch()

    assert len(emitted) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_fill_close_and_emit.py -v`
Expected: FAIL — no `_maybe_emit_batch`

- [ ] **Step 3: Implement the close and the emission**

In `poll_deadline`, immediately after `now = self._time_fn()`:

```python
        if FILL_CLOSED_ENABLED and self.fill_state is not None:
            if self.fill_state.is_closed():
                self._seal_flag.set()
                return True
            if now - self.window_opened_at >= FILL_CLOSED_MAX_SECONDS:
                # Backstop: a stalled fleet must not hold a window open.
                self._seal_flag.set()
                return True
            return False
```

Add the emission counter and hook:

```python
    def _maybe_emit_batch(self) -> None:
        """Emit one training batch per B_BATCH proven groups in every env.

        A DAPO step needs B_BATCH groups from EACH environment, so the trigger
        is the slowest environment. BalancedTrainingAccumulator already carries
        a sparse environment's deficit forward and is the right home for the
        remainder.
        """
        proven = self.fill_state.snapshot()["proven"]
        ready = min(proven.values()) // B_BATCH
        while self._batches_emitted < ready:
            self._batches_emitted += 1
            self._emit_training_batch()
```

Initialise `self._batches_emitted = 0` in `__init__`, and call `_maybe_emit_batch()` from wherever `record_proven` is invoked in Task 5's completion path.

**Emission must be ordered.** Proofs finish out of order and `TrainerWorker`
consumes the journal strictly by cursor, so batch 5 must never be written
before batch 4. The `while` loop above is the ordering barrier: it emits
consecutively from a monotonically increasing counter, never from proof
completion order.

- [ ] **Step 3b: Advertise per-environment fill on /state**

Each environment closes independently, and the reference miner already
re-reads `/state` every loop iteration and samples its environment by the mix
weights — so a fleet that honours a closed environment rebalances toward the
scarce one with no client change. Expose it, or that rebalancing cannot
happen:

```python
def test_state_advertises_which_environments_are_still_admitting(monkeypatch):
    """Math has historically under-filled, so Code will close first and
    Math will set the window duration. Code miners need to see that."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _make_batcher()
    batcher.fill_state = batcher_module.FillState(
        targets={"openmathinstruct": 1, "opencodeinstruct": 1}
    )
    batcher.fill_state.record_proven("opencodeinstruct")

    fill = batcher.upload_precommit_conservation()["fill_state"]

    assert fill["proven"]["opencodeinstruct"] == 1
    assert fill["closed"] is False
```

Carry `self.fill_state.snapshot()` under a `"fill_state"` key in
`upload_precommit_conservation()` — the channel that already carries
`early_close` and the graded fill offsets, and that already reaches both the
R2 archive and `/health` per environment. `None` on the auction path.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_fill_close_and_emit.py tests/unit/test_collection_deadline.py tests/unit/test_auction_early_close.py tests/unit/test_validator_server.py -v`
Expected: PASS, and the auction deadline and early-close tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher.py reliquary/validator/service.py tests/unit/test_fill_close_and_emit.py
git commit -m "feat(v6): close the window on its fill and emit batches as it fills"
```

---

### Task 8: Pay per token, and carry the tokens into the archive

**Files:**
- Modify: `reliquary/validator/batch_selection.py` — `select_batch_and_distribute` (~line 110)
- Modify: `reliquary/validator/service.py` — `_archive_window`, per-submission fields (~line 3300)
- Test: `tests/unit/test_v6_emission.py`

**Interfaces:**
- Consumes: `split_environment_pool` and `AcceptedGroup` (Task 3).
- Produces: an `eos_tokens` integer on every accepted submission in the archive; `select_batch_and_distribute` returns token-proportional rewards when the v6 gate is on.

**Why the archive changes:** weight-only validators replay the EMA from R2 archives and must converge bit for bit. If payment depends on tokens, the token count per accepted group has to be in the archive or the replay cannot reproduce it.

- [ ] **Step 1: Write the failing tests**

```python
"""v6 pays by token; the archive must carry what the payment divides."""
from reliquary.validator.batch_selection import select_batch_and_distribute


from reliquary.validator.cooldown import CooldownMap

from tests.unit.test_archive_window_content import _valid_submission


def test_v6_rewards_are_proportional_to_eos_tokens(monkeypatch):
    """Two accepted groups, nine times the tokens, nine times the share."""
    import reliquary.validator.batch_selection as selection
    monkeypatch.setattr(selection, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(
        selection, "eos_completion_tokens",
        lambda submission: {"short": 1_000, "long": 9_000}[submission.hotkey],
    )

    batch, rewards = select_batch_and_distribute(
        [
            _valid_submission(prompt_idx=1, hotkey="short", eos_first=True),
            _valid_submission(prompt_idx=2, hotkey="long", eos_first=True),
        ],
        b=2,
        cooldown_map=CooldownMap(),
        current_window=42,
        pool=1.0,
    )

    assert len(batch) == 2
    assert abs(rewards["short"] - 0.1) < 1e-9
    assert abs(rewards["long"] - 0.9) < 1e-9


def test_the_auction_path_is_untouched_when_the_gate_is_off():
    """v4 and v5 keep the flat slot share, byte for byte."""
    _batch, rewards = select_batch_and_distribute(
        [
            _valid_submission(prompt_idx=1, hotkey="short", eos_first=True),
            _valid_submission(prompt_idx=2, hotkey="long", eos_first=True),
        ],
        b=2,
        cooldown_map=CooldownMap(),
        current_window=42,
        pool=1.0,
    )

    assert rewards["short"] == rewards["long"]


def test_the_archive_records_eos_tokens_per_accepted_group():
    """The weight-only replay divides by tokens, so the archive must carry
    them or two validators cannot converge on the same weights."""
    archive = _archive_one_v6_window()

    for entry in archive["batch"]:
        assert isinstance(entry["eos_tokens"], int)
```

`_archive_one_v6_window()` reuses the window construction already in
`tests/unit/test_archive_window_content.py` — build the batcher and batch the
same way that file does, arm `FILL_CLOSED_ENABLED`, and return the archive
dict it produces.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_v6_emission.py -v`
Expected: FAIL

- [ ] **Step 3: Branch the distribution**

In `select_batch_and_distribute`, where `slot_share = pool / B_BATCH` is computed, add the v6 branch. Leave the auction arithmetic untouched on the else path:

```python
    if FILL_CLOSED_ENABLED:
        from reliquary.validator.token_rewards import (
            AcceptedGroup,
            split_environment_pool,
        )

        return batch, split_environment_pool(
            [
                AcceptedGroup(
                    hotkey=submission.hotkey,
                    operator_id=operator_by_hotkey.get(
                        submission.hotkey, submission.hotkey
                    ),
                    eos_tokens=eos_completion_tokens(submission),
                )
                for submission in batch
            ],
            pool=pool,
            max_operator_share=FILL_CLOSED_MAX_OPERATOR_TOKEN_SHARE,
        )
```

`eos_completion_tokens(submission)` sums completion token counts over rollouts
that terminated with EOS, using the existing `is_cap_truncation` helper in
`reliquary/validator/verifier.py` to exclude the rest. A rollout that hit the
cap without EOS contributes zero.

- [ ] **Step 4: Add the archive field**

In `_archive_window`, alongside the existing per-submission fields, add
`"eos_tokens": eos_completion_tokens(submission)`. It is additive: older
consumers ignore it, and `ROLLING_WINDOWS_HISTORY` needs re-dimensioning
separately once v6 window durations are measured.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_v6_emission.py tests/unit/test_archive_window_content.py tests/unit/test_batch_selection.py -v`
Expected: PASS, with the auction distribution tests unchanged.

- [ ] **Step 6: Run the full suite**

Run: `TMPDIR=/tmp .venv/bin/python -m pytest -q --ignore=tests/gpu --ignore=tests/integration/test_grader_e2e.py --deselect tests/unit/test_admission_isolation.py::test_spawned_worker_deadline_is_terminal`
Expected: all pass. Every v4/v5 test must be untouched — the gate is off by default.

- [ ] **Step 7: Commit**

```bash
git add reliquary/validator/batch_selection.py reliquary/validator/service.py tests/unit/test_v6_emission.py
git commit -m "feat(v6): pay per EOS-terminated token and archive the token counts"
```

---

## After the plan

Two things are deliberately **not** in it, and both need their own decision:

- **Auditability.** The checkpoint-epoch prototype publishes a signed
  pre-beacon intent and a signed frozen commitment set, so a third party can
  reproduce its selection. v6 has no frozen set — precommits stream — so the
  equivalent property needs signed arrival observations
  (`receipt_id`, `payload_bytes`, observed arrival) rather than a port of that
  code. Same goal, different mechanism, worth designing rather than grafting.
- **Restart and staleness.** The epoch prototype's create-only persistence,
  live-state revalidation and durable quarantine are orthogonal to the control
  loop and adoptable as they stand.
