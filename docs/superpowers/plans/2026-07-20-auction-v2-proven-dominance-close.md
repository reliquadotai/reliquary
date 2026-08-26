# Auction v2 Proven-Dominance Early Close — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Seal an auction window as soon as `B_BATCH` distinct prompts are covered by *GPU-proven* candidates at the theoretical maximum difficulty value, instead of always waiting the full 300 s collection deadline.

**Architecture:** A per-window background prover thread proves top-ranked `V_MAX` candidates while admission continues (proofs of `V_MAX` candidates are never wasted — their rank can only improve). Results land in a proof cache that `_prove_ranked` consults at seal, sharing one window-wide attempt/wall budget. When every member of the paying `V_MAX` tiers is resolved and coverage ≥ `B_BATCH`, the thread seals via `force_seal("proven_dominance_close")` under `_upload_precommit_lock` (excluding precommit races); the existing drain + `_prove_ranked` re-walk at seal is the safety net for queue stragglers.

**Tech Stack:** Python 3, `threading`, pytest (`tests/unit/`), existing helpers `tests/unit/test_grpo_window_batcher.py::{_make_batcher,_request,_always_false_grail}`.

**Spec:** `docs/superpowers/specs/2026-07-20-auction-v2-proven-dominance-close-design.md`

## Global Constraints

- Branch: `feat/auction-v2-proven-dominance-close` off `main`. Commit per task; NEVER push, NEVER commit on main.
- Kill switch `RELIQUARY_AUCTION_EARLY_CLOSE_ENFORCE`, durable default ON in `constants.py` (never rely on `.env` for the default — drand-tolerance regression lesson).
- Kill switch OFF ⇒ byte-identical behavior to today (no thread, no tracking).
- Budgets span the window: mid-window + post-seal proof attempts share `MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW`; mid-window GPU seconds count into the `MAX_PROOF_WALL_SECONDS` check as accumulated seconds (not wall-since-arm, which would starve the deadline path).
- `_verify_expensive` stays strictly serial: one prover thread, joined in `seal_batch` before `_prove_ranked` runs.
- No wire/schema changes; post-close rejects reuse the existing sealed-window `BATCH_FILLED` path (already exercised by `force_seal`).
- All repo-bound text in English. Inline comments 1–2 sentences.
- Deviation from spec §"Dominance tracking" (documented, deliberate): the prover does not wait for accepted coverage ≥ B_BATCH to start proving — it proves any unproven member of the *currently paying* `V_MAX` tiers from the moment one exists. Identical safety argument (such a candidate is proven by `_prove_ranked` at every possible seal), and the proofs overlap the fill phase, closing sooner. Update the spec paragraph in Task 8.

---

### Task 1: `max_difficulty_value` helper

**Files:**
- Modify: `reliquary/validator/difficulty_auction.py` (after `difficulty_score`, ~line 93)
- Test: `tests/unit/test_difficulty_auction.py` (append)

**Interfaces:**
- Produces: `max_difficulty_value(reward_count: int, *, delta: float = 1.0) -> float` — cached, the exact float `difficulty_score` returns for the best achievable reward profile of that size.

- [ ] **Step 1: Branch**

```bash
git -C /home/ubuntu/Catalyst checkout -b feat/auction-v2-proven-dominance-close
```

- [ ] **Step 2: Write the failing tests**

```python
# append to tests/unit/test_difficulty_auction.py
from itertools import product

from reliquary.validator.difficulty_auction import (
    difficulty_score,
    max_difficulty_value,
)


def test_max_difficulty_value_is_the_binary_k2_peak_for_8_rollouts():
    """delta=1: v(p)=sqrt(p(1-p))*(1-p) peaks at p=1/4 -> k=2 of 8; the
    constant must be the exact float difficulty_score emits for that profile
    (ranking compares with ==, no epsilon)."""
    expected = difficulty_score([1.0, 1.0] + [0.0] * 6, delta=1.0).value
    assert max_difficulty_value(8, delta=1.0) == expected
    assert all(
        difficulty_score([1.0] * k + [0.0] * (8 - k), delta=1.0).value
        <= max_difficulty_value(8, delta=1.0)
        for k in range(9)
    )


def test_no_fractional_profile_exceeds_the_binary_maximum():
    """For fixed mean, std is maximized only by extremal (0/1) rewards, so no
    in-[0,1] profile can beat the binary max. Grid-check 4-rollout profiles
    exhaustively on a 0/0.25/0.5/0.75/1 lattice."""
    cap = max_difficulty_value(4, delta=1.0)
    lattice = (0.0, 0.25, 0.5, 0.75, 1.0)
    for profile in product(lattice, repeat=4):
        assert difficulty_score(list(profile), delta=1.0).value <= cap


def test_max_difficulty_value_zero_and_one_rollout_degenerate_to_zero():
    assert max_difficulty_value(0, delta=1.0) == 0.0
    assert max_difficulty_value(1, delta=1.0) == 0.0
```

- [ ] **Step 3: Run to verify failure**

Run: `cd /home/ubuntu/Catalyst && python -m pytest tests/unit/test_difficulty_auction.py -k max_difficulty -x -q`
Expected: FAIL — `ImportError: cannot import name 'max_difficulty_value'`

- [ ] **Step 4: Implement**

```python
# reliquary/validator/difficulty_auction.py, after difficulty_score
@functools.lru_cache(maxsize=None)
def max_difficulty_value(reward_count: int, *, delta: float = 1.0) -> float:
    """Exact float ceiling of ``difficulty_score`` over achievable rewards.

    For a fixed mean, std is maximized only by extremal (all 0/1) profiles,
    so the global maximum is attained on a binary profile; enumerating k is
    exhaustive. Computed through ``difficulty_score`` itself so the value
    compares bit-for-bit (==) with candidate scores.
    """
    if reward_count <= 0:
        return 0.0
    return max(
        difficulty_score(
            [1.0] * k + [0.0] * (reward_count - k), delta=delta
        ).value
        for k in range(reward_count + 1)
    )
```

Add `import functools` to the module imports.

- [ ] **Step 5: Run tests, then commit**

Run: `python -m pytest tests/unit/test_difficulty_auction.py -q`
Expected: PASS (all, including pre-existing).

```bash
git add reliquary/validator/difficulty_auction.py tests/unit/test_difficulty_auction.py
git commit -m "feat(auction): exact ceiling of difficulty_score for early-close dominance checks"
```

---

### Task 2: Kill switch + poll constant

**Files:**
- Modify: `reliquary/constants.py` (next to `DIFFICULTY_AUCTION_ENFORCE`, ~line 524)
- Test: `tests/unit/test_constants.py` if it exists, else fold assertions into Task 7's kill-switch test.

**Interfaces:**
- Produces: `AUCTION_EARLY_CLOSE_ENFORCE: bool` (env `RELIQUARY_AUCTION_EARLY_CLOSE_ENFORCE`, default ON), `AUCTION_EARLY_CLOSE_POLL_SECONDS: float = 1.0`.

- [ ] **Step 1: Implement (no dedicated test file — consumed by Task 7 tests)**

```python
# reliquary/constants.py, after the DIFFICULTY_AUCTION_ENVIRONMENTS block
# Proven-dominance early close: seal an auction window before the collection
# deadline once B_BATCH distinct prompts are covered by GPU-PROVEN candidates
# at the theoretical difficulty ceiling (no future arrival can change the
# outcome). OFF restores the deadline-only seal bit-for-bit.
AUCTION_EARLY_CLOSE_ENFORCE = _os.environ.get(
    "RELIQUARY_AUCTION_EARLY_CLOSE_ENFORCE", "1"
).strip().lower() not in ("0", "false", "no", "off", "")
AUCTION_EARLY_CLOSE_POLL_SECONDS = 1.0
```

- [ ] **Step 2: Sanity + commit**

Run: `python -c "from reliquary.constants import AUCTION_EARLY_CLOSE_ENFORCE, AUCTION_EARLY_CLOSE_POLL_SECONDS; print(AUCTION_EARLY_CLOSE_ENFORCE, AUCTION_EARLY_CLOSE_POLL_SECONDS)"`
Expected: `True 1.0`

```bash
git add reliquary/constants.py
git commit -m "feat(auction): early-close kill switch and poll constant"
```

---

### Task 3: `_arrival_round_of` extraction (pure refactor)

**Files:**
- Modify: `reliquary/validator/batcher.py` — new method near `_prove_ranked` (~line 3380); `_prove_ranked` uses it (replaces lines 3436–3449 loop body building `arrival_by_id`/`arrival_source_by_id`).

**Interfaces:**
- Produces: `GrpoWindowBatcher._arrival_round_of(p: PendingSubmission) -> tuple[int, str]` returning `(round, source)` where source ∈ {"arrival", "submitted_fallback"} — semantics identical to the inline code today.

- [ ] **Step 1: Implement**

```python
# batcher.py, above _prove_ranked
def _arrival_round_of(self, pending: PendingSubmission) -> tuple[int, str]:
    """Validator-observed arrival round; miner-submitted round is the
    mock-mode fallback (production stamps every admitted request)."""
    telemetry = getattr(pending, "telemetry", None)
    arrival = getattr(telemetry, "arrival_drand_round", None)
    if arrival is not None:
        return int(arrival), "arrival"
    return int(pending.drand_round), "submitted_fallback"
```

In `_prove_ranked`, replace the body of the arrival-resolution loop:

```python
for pending_submission, _score in scored:
    operator = self._operator_by_hotkey.get(pending_submission.hotkey)
    if operator is None and not self._operator_mapping_enforced:
        operator = pending_submission.hotkey
    operator_by_id[id(pending_submission)] = operator
    arrival, source = self._arrival_round_of(pending_submission)
    arrival_by_id[id(pending_submission)] = arrival
    arrival_source_by_id[id(pending_submission)] = source
```

- [ ] **Step 2: Run the existing auction/deferred tests, commit**

Run: `python -m pytest tests/unit/test_deferred_proof.py tests/unit/test_auction_resource_guards.py tests/unit/test_grpo_window_batcher.py -q`
Expected: PASS, zero behavior change.

```bash
git add reliquary/validator/batcher.py
git commit -m "refactor(auction): extract _arrival_round_of for reuse by the early-close prover"
```

---

### Task 4: Proof cache consumed by `_prove_ranked`, shared budgets, `proof_phase` rows

**Files:**
- Modify: `reliquary/validator/batcher.py` — `__init__` state; `_prove_ranked` (cache branch + budget init).
- Test: `tests/unit/test_early_close.py` (new file)

**Interfaces:**
- Produces (read by Tasks 5–8):
  - `self._early_proof_results: dict[int, ValidSubmission | None]` — id(pending) → proven sub / None (fail). Missing key = never attempted mid-window.
  - `self.early_close_proof_attempts: int`, `self.early_close_proof_wall_seconds: float`, `self.early_close_proof_failures: int` — mid-window budget/telemetry counters.
  - `_prove_ranked` candidate rows gain `"proof_phase": "midwindow" | "post_seal" | None`.
- Consumes: Task 3 `_arrival_round_of` (already merged into `_prove_ranked`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_early_close.py
"""Proven-dominance early close: mid-window proofs cached and reused at seal,
budgets spanning the window, and the close trigger itself."""
import threading

from reliquary.constants import B_BATCH, DIFFICULTY_AUCTION_DELTA
from tests.unit.test_grpo_window_batcher import (
    _always_false_grail,
    _make_batcher,
    _request,
)


def _auction_batcher(**overrides):
    kwargs = dict(env_name="openmathinstruct")
    kwargs.update(overrides)
    b = _make_batcher(**kwargs)
    assert b.difficulty_auction_enabled
    return b


def _accept(b, prompt_idx, hotkey, k=2, drand_round=10):
    resp = b.accept_submission(
        _request(prompt_idx=prompt_idx, hotkey=hotkey, k=k,
                 drand_round=drand_round)
    )
    assert resp.accepted, resp.reason
    return b.pending_submissions()[-1]


def test_prove_ranked_reuses_cached_pass_without_touching_the_gpu():
    calls = []

    def _grail(commit, model, randomness):
        calls.append(1)
        return _always_false_grail(commit, model, randomness)

    b = _auction_batcher(verify_commitment_proofs_fn=_grail)
    p = _accept(b, prompt_idx=1, hotkey="hk1")
    proven = b._verify_expensive(p)          # simulate the mid-window prover
    assert proven is None and calls          # our fake grail always fails
    calls.clear()
    b._early_proof_results[id(p)] = None
    b.early_close_proof_attempts = 1
    b.early_close_proof_failures = 1

    b.force_seal("test")
    b.seal_batch(pool=1.0)

    assert calls == []                       # cache hit: GPU untouched at seal
    row = next(r for r in b.auction_candidates if r["hotkey"] == "hk1")
    assert row["proof_phase"] == "midwindow"
    assert row["proof_passed"] is False
    # attempts telemetry includes the mid-window attempt exactly once
    assert b.proof_attempts == 1


def test_cached_pass_is_selected_and_debt_gates_do_not_reevaluate_it():
    """Sequential semantics: a candidate proven mid-window (before its operator
    hit the failure cap) stays selected at seal even if later mid-window
    failures pushed the operator over the cap."""
    b = _auction_batcher()
    p = _accept(b, prompt_idx=1, hotkey="hk1")
    sub = b._verify_expensive(p)             # default grail passes
    assert sub is not None
    b._early_proof_results[id(p)] = sub
    b.early_close_proof_attempts = 1
    # operator over the cap AFTER the proof happened
    b._expensive_proof_failures_by_operator[
        b._operator_by_hotkey.get("hk1", "hk1")
    ] = 10_000

    b.force_seal("test")
    b.seal_batch(pool=1.0)

    row = next(r for r in b.auction_candidates if r["hotkey"] == "hk1")
    assert row["status"] == "selected"
    assert row["proof_phase"] == "midwindow"


def test_midwindow_wall_seconds_count_into_the_seal_wall_budget():
    from reliquary.constants import MAX_PROOF_WALL_SECONDS

    b = _auction_batcher()
    _accept(b, prompt_idx=1, hotkey="hk1")
    b.early_close_proof_wall_seconds = MAX_PROOF_WALL_SECONDS  # spent it all
    b.force_seal("test")
    b.seal_batch(pool=1.0)

    assert b.proof_wall_exhausted is True
    assert b.valid_submissions() == []
```

Note: `_make_batcher(env_name=...)` — check the helper's actual way to get an auction-enabled batcher (`FakeEnv` name / `difficulty_auction_enabled` override) and adapt `_auction_batcher` accordingly before running; the intent is a batcher with `difficulty_auction_enabled=True` and the default always-pass GRAIL. `_request(k=...)` likewise: use the helper's existing knob for reward pattern (2 positive / 6 zero rewards).

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_early_close.py -x -q`
Expected: FAIL — `AttributeError: ... has no attribute '_early_proof_results'`

- [ ] **Step 3: Implement**

`__init__` (near the other auction state, ~line 594):

```python
# Mid-window proof results (id(pending) -> ValidSubmission|None). Written
# by the early-close prover thread under _lock, consumed by _prove_ranked;
# a missing key means the candidate was never attempted mid-window.
self._early_proof_results: dict[int, "ValidSubmission | None"] = {}
self.early_close_proof_attempts = 0
self.early_close_proof_failures = 0
self.early_close_proof_wall_seconds = 0.0
self.early_close_armed_round: int | None = None
self.early_close_sealed_round: int | None = None
```

`_prove_ranked` changes:

1. Row template gains `"proof_phase": None`.
2. Budget init: `attempts = self.early_close_proof_attempts` (replaces `attempts = 0`).
3. Wall check becomes:

```python
elapsed = (
    self._time_fn() - self._proof_wall_started_at
    + self.early_close_proof_wall_seconds
)
```

4. Cache branch, inserted after the cooldown check and before the operator gates (cache hits bypass debt/budget gates — those were evaluated when the proof actually ran, preserving sequential semantics):

```python
cached_hit = id(p) in self._early_proof_results
if cached_hit:
    sub = self._early_proof_results[id(p)]
    row["proof_attempted"] = True
    row["proof_phase"] = "midwindow"
    attempted_ids.add(id(p))
    if sub is None:
        # Debt was charged when the mid-window proof failed; never twice.
        row["proof_passed"] = False
        row["status"] = "proof_failed"
        continue
    row["proof_passed"] = True
    row["selected"] = True
    row["status"] = "selected"
    proven.append(sub)
    claimed.add(p.prompt_idx)
    self._auction_tier_by_id[id(sub)] = tier
    self.difficulty_auction_metadata_by_id[id(sub)] = row
    continue
```

5. The live-proof branch sets `row["proof_phase"] = "post_seal"` next to `row["proof_attempted"] = True`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/unit/test_early_close.py tests/unit/test_deferred_proof.py tests/unit/test_auction_resource_guards.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_early_close.py
git commit -m "feat(auction): mid-window proof cache consumed by _prove_ranked with window-spanning budgets"
```

---

### Task 5: `_early_close_next_action_locked` — the ranking walk

**Files:**
- Modify: `reliquary/validator/batcher.py`
- Test: `tests/unit/test_early_close.py` (append)

**Interfaces:**
- Produces: `GrpoWindowBatcher._early_close_next_action_locked() -> tuple[str, PendingSubmission | None, int | None]` — `("prove", p, None)` next candidate to prove; `("close", None, boundary_round)` every paying V_MAX member resolved and proven coverage ≥ B_BATCH; `("wait", None, None)` nothing provable yet; `("exhausted", None, None)` window proof budget spent. MUST be called under `self._lock`.
- Consumes: Task 1 `max_difficulty_value`, Task 3 `_arrival_round_of`, Task 4 cache/counters.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_early_close.py
def test_walk_waits_without_vmax_candidates():
    b = _auction_batcher()
    _accept(b, prompt_idx=1, hotkey="hk1", k=6)   # below the k=2 peak
    with b._lock:
        assert b._early_close_next_action_locked() == ("wait", None, None)


def test_walk_proves_vmax_candidates_in_arrival_order():
    b = _auction_batcher()
    late = _accept(b, prompt_idx=1, hotkey="late", drand_round=20)
    early = _accept(b, prompt_idx=2, hotkey="early", drand_round=10)
    with b._lock:
        action, target, _ = b._early_close_next_action_locked()
    assert (action, target) == ("prove", early)


def test_walk_skips_cached_and_stops_at_the_boundary():
    """8 distinct proven V_MAX prompts -> close with the boundary round; a 9th
    distinct prompt in a later tier is never offered for proving."""
    b = _auction_batcher()
    subs = [
        _accept(b, prompt_idx=i, hotkey=f"hk{i}", drand_round=10 + i)
        for i in range(B_BATCH + 1)
    ]
    for p in subs[:B_BATCH]:
        b._early_proof_results[id(p)] = b._verify_expensive(p)
    with b._lock:
        action, target, boundary = b._early_close_next_action_locked()
    assert action == "close"
    assert target is None
    assert boundary == 10 + B_BATCH - 1     # arrival round of the 8th tier


def test_walk_offers_failed_slots_replacement():
    """A mid-window proof failure reopens its slot: the next V_MAX arrival on a
    new prompt is offered for proving instead of closing."""
    b = _auction_batcher()
    subs = [
        _accept(b, prompt_idx=i, hotkey=f"hk{i}", drand_round=10 + i)
        for i in range(B_BATCH)
    ]
    for p in subs[:-1]:
        b._early_proof_results[id(p)] = b._verify_expensive(p)
    b._early_proof_results[id(subs[-1])] = None       # failed mid-window
    replacement = _accept(b, prompt_idx=99, hotkey="fresh", drand_round=40)
    with b._lock:
        action, target, _ = b._early_close_next_action_locked()
    assert (action, target) == ("prove", replacement)


def test_walk_reports_exhausted_budget():
    from reliquary.constants import MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW

    b = _auction_batcher()
    _accept(b, prompt_idx=1, hotkey="hk1")
    b.early_close_proof_attempts = MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW
    with b._lock:
        assert b._early_close_next_action_locked() == ("exhausted", None, None)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_early_close.py -k walk -x -q`
Expected: FAIL — no attribute `_early_close_next_action_locked`.

- [ ] **Step 3: Implement**

```python
# batcher.py — mirrors _prove_ranked's tier walk on the V_MAX prefix of the
# ranking; kept separate because _prove_ranked interleaves row bookkeeping
# and budget mutation that have no mid-window equivalent. The equivalence is
# pinned by tests (a cache-complete walk and a seal walk agree).
def _early_close_next_action_locked(
    self,
) -> tuple[str, "PendingSubmission | None", int | None]:
    if self.early_close_proof_attempts >= MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW:
        return "exhausted", None, None
    if self.early_close_proof_wall_seconds >= MAX_PROOF_WALL_SECONDS:
        return "exhausted", None, None
    pool = [
        p for p in self._pending
        if p.value == max_difficulty_value(
            len(p.rewards), delta=DIFFICULTY_AUCTION_DELTA
        )
    ]
    if not pool:
        return "wait", None, None
    pool.sort(key=lambda p: (self._arrival_round_of(p)[0], _within_slot_key(p)))
    claimed: set[int] = set()
    boundary_round: int | None = None
    tier_round: int | None = None
    claimed_before_tier: set[int] = set()
    for p in pool:
        arrival = self._arrival_round_of(p)[0]
        if arrival != tier_round:
            if len(claimed) >= B_BATCH:
                break                     # boundary crossed: later tiers lose
            tier_round = arrival
            claimed_before_tier = set(claimed)
        boundary_round = tier_round
        if p.prompt_idx in claimed_before_tier:
            continue                      # same_prompt_superseded
        if self._cooldown.is_in_cooldown(p.prompt_idx, self.window_start):
            continue
        if id(p) in self._early_proof_results:
            if self._early_proof_results[id(p)] is not None:
                claimed.add(p.prompt_idx)
            continue
        operator = self._operator_by_hotkey.get(p.hotkey)
        if operator is None and not self._operator_mapping_enforced:
            operator = p.hotkey
        if operator is None:
            continue                      # operator_unmapped
        if (
            self.operator_proof_failure_debt(operator)
            >= MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW
        ):
            continue
        if (
            self.proof_failure_debt(p.hotkey)
            >= MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW
        ):
            continue
        return "prove", p, None
    if len(claimed) >= B_BATCH:
        return "close", None, boundary_round
    return "wait", None, None
```

Note on the boundary round: `boundary_round` tracks the round of the last tier *visited before* the break, i.e. the boundary tier — exactly the round condition 3 must beat. Careful review point: it must be updated only for tiers at or below B-coverage (the loop above sets it before the skip checks, after the break — verify with the boundary test).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/unit/test_early_close.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_early_close.py
git commit -m "feat(auction): early-close ranking walk over paying V_MAX tiers"
```

---

### Task 6: `_try_early_close` + `_current_round`

**Files:**
- Modify: `reliquary/validator/batcher.py` — constructor param `current_round_fn`, `_current_round()`, `_try_early_close()`.
- Test: `tests/unit/test_early_close.py` (append)

**Interfaces:**
- Produces:
  - constructor kwarg `current_round_fn: Callable[[], int | None] | None = None` (tests inject; production default derives the round from the drand chain info exactly like `observe_drand_round`).
  - `GrpoWindowBatcher._current_round() -> int | None` — None when drand is unavailable (mock mode) ⇒ close never fires, deadline seals.
  - `GrpoWindowBatcher._try_early_close() -> bool` — True once sealed (by us or anyone).
- Consumes: Task 5 walk.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_early_close.py
def _saturate(b):
    subs = [
        _accept(b, prompt_idx=i, hotkey=f"hk{i}", drand_round=10 + i)
        for i in range(B_BATCH)
    ]
    for p in subs:
        b._early_proof_results[id(p)] = b._verify_expensive(p)
    return subs


def test_try_early_close_seals_with_reason_and_round():
    b = _auction_batcher(current_round_fn=lambda: 100)
    _saturate(b)
    assert b._try_early_close() is True
    assert b.is_sealed() is True
    assert b.force_seal_reason == "proven_dominance_close"
    assert b.early_close_sealed_round == 100


def test_try_early_close_blocked_by_same_round_arrival_window():
    """3 s round granularity: while the current round equals the boundary
    tier's round, an equal-key arrival could still join the fair-split."""
    b = _auction_batcher(current_round_fn=lambda: 10 + B_BATCH - 1)
    _saturate(b)                                     # boundary round = 17
    assert b._try_early_close() is False
    assert b.is_sealed() is False


def test_try_early_close_blocked_by_pending_upload_precommit():
    b = _auction_batcher(current_round_fn=lambda: 100)
    _saturate(b)
    accepted, reason, _ = b.try_register_upload_precommit(
        "receipt-1", "uploader", t_arrival_wall=0.0, payload_bytes=10,
    )
    assert accepted, reason
    assert b._try_early_close() is False
    assert b.is_sealed() is False


def test_try_early_close_noop_without_drand():
    b = _auction_batcher()                           # no current_round_fn
    _saturate(b)
    assert b._try_early_close() is False
    assert b.is_sealed() is False
```

Note: `t_arrival_wall=0.0` must be before the collection close; check `_make_batcher`'s `window_opened_wall_ts` and pass a compatible wall timestamp.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_early_close.py -k try_early_close -x -q`
Expected: FAIL — unexpected kwarg / missing attribute.

- [ ] **Step 3: Implement**

Constructor: accept `current_round_fn=None`, store `self._current_round_fn = current_round_fn`.

```python
def _current_round(self) -> int | None:
    """Wall-clock drand round, for the same-round close guard. None (mock /
    drand outage) fails safe: the close never fires, the deadline seals."""
    if self._current_round_fn is not None:
        try:
            return self._current_round_fn()
        except Exception:
            return None
    try:
        if self._drand_chain_info is None:
            from reliquary.infrastructure.drand import get_current_chain
            self._drand_chain_info = get_current_chain()
        from reliquary.infrastructure.chain import compute_current_drand_round
        ci = self._drand_chain_info
        return int(compute_current_drand_round(
            self._wall_clock(), ci["genesis_time"], ci["period"],
        ))
    except Exception:
        return None

def _try_early_close(self) -> bool:
    """Seal now iff the auction outcome is proven frozen.

    force_seal runs under _upload_precommit_lock so a racing precommit
    either lands before the emptiness check (blocking the close) or
    observes the seal flag and is rejected — no reservation can straddle
    the close.
    """
    with self._lock:
        action, _target, boundary_round = (
            self._early_close_next_action_locked()
        )
    if action != "close":
        return self._seal_flag.is_set()
    current = self._current_round()
    if current is None or boundary_round is None or current <= boundary_round:
        return self._seal_flag.is_set()
    now = self._time_fn()
    with self._upload_precommit_lock:
        if self._seal_flag.is_set():
            return True
        self._prune_upload_precommits_locked(now)
        if self._upload_precommits:
            return False
        self.early_close_sealed_round = current
        self.force_seal("proven_dominance_close")
    return True
```

- [ ] **Step 4: Run tests, commit**

Run: `python -m pytest tests/unit/test_early_close.py -q`
Expected: PASS.

```bash
git add reliquary/validator/batcher.py tests/unit/test_early_close.py
git commit -m "feat(auction): atomic proven-dominance close with same-round and precommit guards"
```

---

### Task 7: Prover thread + lifecycle

**Files:**
- Modify: `reliquary/validator/batcher.py` — `_early_close_prove`, `_early_close_worker`, thread start in `mark_window_opened`, join in `seal_batch`.
- Test: `tests/unit/test_early_close.py` (append)

**Interfaces:**
- Produces: background thread `self._early_close_thread` (daemon). Starts in `mark_window_opened` iff `difficulty_auction_enabled and AUCTION_EARLY_CLOSE_ENFORCE and current-round available or injected`. Exits when `self._seal_flag` is set, on `exhausted`, or on any exception (safety valve: revert to deadline behavior). `seal_batch` joins it (timeout 60 s) before `_seal_batch_inner`.
- Consumes: Tasks 4–6.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/unit/test_early_close.py
import time as _time


def _run_worker_inline(b):
    """Drive the worker loop synchronously (no thread) for determinism."""
    b._early_close_worker()


def test_worker_proves_saturation_and_seals_end_to_end():
    b = _auction_batcher(current_round_fn=lambda: 1000)
    for i in range(B_BATCH):
        _accept(b, prompt_idx=i, hotkey=f"hk{i}", drand_round=10 + i)
    _run_worker_inline(b)
    assert b.is_sealed()
    assert b.force_seal_reason == "proven_dominance_close"
    assert b.early_close_proof_attempts == B_BATCH
    batch, rewards = b.seal_batch(pool=1.0)
    assert len(batch) == B_BATCH            # all proven from cache, no re-proof
    assert sum(rewards.values()) > 0


def test_worker_failure_then_refill_then_close():
    fail = {"hk_bait"}

    def _grail_for(commit, model, randomness):
        return _always_false_grail(commit, model, randomness)

    b = _auction_batcher(current_round_fn=lambda: 1000)
    # bait fails proof; make _verify_expensive fail only for the bait hotkey
    real_verify = b._verify_expensive

    def _verify(p):
        if p.hotkey in fail:
            b._reject_bait(p)  # replaced below — see implementation note
        return None if p.hotkey in fail else real_verify(p)

    b._verify_expensive = _verify
    for i in range(B_BATCH - 1):
        _accept(b, prompt_idx=i, hotkey=f"hk{i}", drand_round=10 + i)
    _accept(b, prompt_idx=50, hotkey="hk_bait", drand_round=9)  # ranks first
    _run_worker_inline(b)                    # bait fails, coverage stuck at 7
    assert not b.is_sealed()
    _accept(b, prompt_idx=60, hotkey="hk_fresh", drand_round=30)
    _run_worker_inline(b)
    assert b.is_sealed()
    assert b.force_seal_reason == "proven_dominance_close"
    assert b.early_close_proof_failures == 1


def test_worker_thread_lifecycle_and_kill_switch(monkeypatch):
    import reliquary.validator.batcher as batcher_mod

    b = _auction_batcher(current_round_fn=lambda: 1000)
    monkeypatch.setattr(batcher_mod, "AUCTION_EARLY_CLOSE_ENFORCE", False)
    b.mark_window_opened()
    assert b._early_close_thread is None      # kill switch: no thread

    b2 = _auction_batcher(current_round_fn=lambda: 1000)
    monkeypatch.setattr(batcher_mod, "AUCTION_EARLY_CLOSE_ENFORCE", True)
    b2.mark_window_opened()
    assert b2._early_close_thread is not None
    b2.force_seal("test")                     # any seal stops the worker
    b2._early_close_thread.join(timeout=10)
    assert not b2._early_close_thread.is_alive()


def test_worker_exception_is_a_safety_valve_not_a_crash():
    b = _auction_batcher(current_round_fn=lambda: 1000)
    _accept(b, prompt_idx=1, hotkey="hk1")

    def _boom(p):
        raise RuntimeError("gpu on fire")

    b._verify_expensive = _boom
    _run_worker_inline(b)                     # returns, no raise
    assert not b.is_sealed()                  # deadline path still owns the seal
```

Implementation note for the second test: don't reference `_reject_bait` — the fake `_verify` returning None is enough; mid-window operator debt is charged by `_early_close_prove` (the production `_verify_expensive` charges hotkey debt internally via `_reject`; the fake bypasses it, which is fine for this test's coverage assertions). Drop the `_grail_for` and `b._reject_bait` lines when writing the real test.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_early_close.py -k worker -x -q`
Expected: FAIL — no `_early_close_worker`.

- [ ] **Step 3: Implement**

```python
def _early_close_prove(self, pending: "PendingSubmission") -> None:
    """One serial mid-window proof; budgets and debt mirror _prove_ranked."""
    started = self._time_fn()
    sub = self._verify_expensive(pending)     # may raise -> worker safety valve
    elapsed = max(0.0, self._time_fn() - started)
    operator = self._operator_by_hotkey.get(pending.hotkey)
    if operator is None and not self._operator_mapping_enforced:
        operator = pending.hotkey
    with self._lock:
        self._early_proof_results[id(pending)] = sub
        self.early_close_proof_attempts += 1
        self.early_close_proof_wall_seconds += elapsed
        if sub is None:
            self.early_close_proof_failures += 1
    if sub is None and operator is not None:
        with self._proof_admission_lock:
            self._expensive_proof_failures_by_operator[operator] = (
                self._expensive_proof_failures_by_operator.get(operator, 0) + 1
            )

def _early_close_worker(self) -> None:
    """Prove-and-close loop; any exception disables early close for the
    window (the deadline seal is always the fallback)."""
    try:
        while not self._seal_flag.is_set():
            with self._lock:
                action, target, _boundary = (
                    self._early_close_next_action_locked()
                )
            if action == "prove":
                self._early_close_prove(target)
                continue
            if action == "exhausted":
                return
            if action == "close" and self._try_early_close():
                return
            if self._seal_flag.is_set():
                return
            time.sleep(AUCTION_EARLY_CLOSE_POLL_SECONDS)
    except Exception:
        logger.exception(
            "early-close prover disabled for window %s", self.window_start,
        )
```

`mark_window_opened` addition (after existing body):

```python
if (
    self.difficulty_auction_enabled
    and AUCTION_EARLY_CLOSE_ENFORCE
    and self._early_close_thread is None
):
    self._early_close_thread = threading.Thread(
        target=self._early_close_worker,
        name=f"early-close-{getattr(self.env, 'name', 'env')}",
        daemon=True,
    )
    self._early_close_thread.start()
```

`__init__`: `self._early_close_thread: threading.Thread | None = None`.

`seal_batch`, before `_seal_batch_inner` (the worker exits on the seal flag; the join only waits out an in-flight GPU proof, which then lands in the cache):

```python
thread = self._early_close_thread
if thread is not None and thread.is_alive():
    thread.join(timeout=60.0)
```

Also set `self.early_close_armed_round` in `_try_early_close` on the first `close` decision if still None (telemetry: the round dominance was first observed proven): `if self.early_close_armed_round is None: self.early_close_armed_round = current`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/unit/test_early_close.py -q`
Expected: PASS. The end-to-end test also exercises `seal_batch` → cache-only, verifying Task 4 integration.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_early_close.py
git commit -m "feat(auction): mid-window prover thread with proven-dominance close"
```

---

### Task 8: Telemetry, spec touch-up, full suite

**Files:**
- Modify: `reliquary/validator/batcher.py` (`_seal_batch_inner` shadow dict)
- Modify: `docs/superpowers/specs/2026-07-20-auction-v2-proven-dominance-close-design.md` (prover-scope paragraph)
- Test: `tests/unit/test_archive_window_content.py` (repair if it pins shadow keys), `tests/unit/test_early_close.py` (append)

**Interfaces:**
- Produces: shadow dict keys `early_close` (bool), `early_close_armed_round`, `early_close_sealed_round`, `midwindow_proof_attempts`, `midwindow_proof_failures`, `midwindow_proof_wall_seconds`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_early_close.py
def test_shadow_dict_reports_early_close_telemetry():
    b = _auction_batcher(current_round_fn=lambda: 1000)
    for i in range(B_BATCH):
        _accept(b, prompt_idx=i, hotkey=f"hk{i}", drand_round=10 + i)
    _run_worker_inline(b)
    b.seal_batch(pool=1.0)
    shadow = b.difficulty_auction_shadow
    assert shadow["early_close"] is True
    assert shadow["early_close_sealed_round"] == 1000
    assert shadow["midwindow_proof_attempts"] == B_BATCH
    assert shadow["midwindow_proof_failures"] == 0
    assert shadow["midwindow_proof_wall_seconds"] >= 0.0
```

- [ ] **Step 2: Implement**

In `_seal_batch_inner`'s shadow dict:

```python
"early_close": self.force_seal_reason == "proven_dominance_close",
"early_close_armed_round": self.early_close_armed_round,
"early_close_sealed_round": self.early_close_sealed_round,
"midwindow_proof_attempts": self.early_close_proof_attempts,
"midwindow_proof_failures": self.early_close_proof_failures,
"midwindow_proof_wall_seconds": self.early_close_proof_wall_seconds,
```

- [ ] **Step 3: Spec touch-up**

In the spec's "Mid-window prover" section, replace the arming sentence with: the prover proves any unproven member of the currently paying `V_MAX` tiers as soon as one exists (before full saturation); dominance-armed tracking survives only as telemetry (`early_close_armed_round`).

- [ ] **Step 4: Full suite**

Run: `python -m pytest tests/unit -q` (and `tests/integration` if it runs without GPU).
Expected: PASS; repair any archive/shadow-schema pins that enumerate keys.

- [ ] **Step 5: Commit**

```bash
git add -A reliquary tests docs/superpowers/specs/2026-07-20-auction-v2-proven-dominance-close-design.md
git commit -m "feat(auction): early-close telemetry in the shadow schema"
```

---

## Self-Review Notes

- **Spec coverage:** V_MAX (T1), kill switch (T2), cache+budgets (T4), walk incl. never-prove-a-loser and boundary (T5), close guards: same-round, precommit, drand-outage (T6), thread lifecycle + forger-bait + safety valve (T7), telemetry + spec deviation note (T8). Spec test list items 1→T7, 2→T7, 3→T7(partial: bait covered; deadline-fallback in T7 exception/exhausted tests), 4→T6 same-round, 5→covered structurally by drain reuse (no new code; asserted indirectly by T4 cache-at-seal), 6→T4 wall-budget + T6 noop, 7→T5 boundary, 8→T4/T5 budgets, 9→T7 kill switch, 10→T1, 11→untouched paths (existing suites in T3/T8 runs).
- **Types:** walk returns 3-tuple everywhere; cache dict `int -> ValidSubmission|None`; counters ints/float — consistent across T4–T8.
- **Known judgement calls for the implementer:** exact `_make_batcher` knob for auction mode and `_request` reward shape must be read from `tests/unit/test_grpo_window_batcher.py` before writing `_auction_batcher`; `time` import already present in batcher.py (verify); imports of `max_difficulty_value`, `AUCTION_EARLY_CLOSE_ENFORCE`, `AUCTION_EARLY_CLOSE_POLL_SECONDS` into batcher.py are part of T4/T5/T7 steps.
