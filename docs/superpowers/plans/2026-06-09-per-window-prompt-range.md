# Per-Window Prompt Range Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confine each window's accepted prompts to a contiguous slice of the prompt index space, derived on both sides from the existing per-window `randomness`, so a static/shared bank of pre-curated prompts only lands in-range a small fraction of windows.

**Architecture:** A pure shared function `window_prompt_range()` maps `(randomness, env_name, universe_n, size)` → `[lo, hi)`. The validator caches the slice per window (gated on a baked-in cutover window) and rejects out-of-range submissions in both the cheap arrival path and `_accept_locked`. The reference miner restricts its prompt sampling to the same slice. Zero protocol/wire change — `randomness` already crosses the wire via `/state`.

**Tech Stack:** Python 3.13, pytest, FastAPI TestClient, hashlib (stdlib).

**Conventions:**
- Work on a feature branch; never commit on `main`, never push unless asked.
- Inline comments 1–2 sentences; rationale goes in the commit message.
- All commit messages end with the repo's `Co-Authored-By` trailer.
- The spec and this plan live under `docs/superpowers/` and stay **untracked** — never `git add` them.

**Reference spec:** `docs/superpowers/specs/2026-06-09-per-window-prompt-range-design.md`

---

## File Structure

- **Create** `reliquary/shared/prompt_range.py` — pure range function (single source of truth, imported by both miner and validator).
- **Create** `tests/unit/test_prompt_range.py` — unit tests for the function + constants/enum wiring.
- **Modify** `reliquary/constants.py` — add `PROMPT_RANGE_SIZE`, `PROMPT_RANGE_ENFORCE_FROM_WINDOW`.
- **Modify** `reliquary/protocol/submission.py` — add `RejectReason.PROMPT_OUT_OF_RANGE`.
- **Modify** `reliquary/validator/batcher.py` — `prompt_range` attribute, `set_prompt_range()`, enforce in `_accept_locked`.
- **Modify** `reliquary/validator/service.py` — call `set_prompt_range()` after randomness is set.
- **Modify** `reliquary/validator/server.py` — enforce in the cheap arrival path.
- **Modify** `reliquary/miner/engine.py` — restrict `pick_prompt_idx` / `pick_env_and_prompt` to the slice.
- **Modify** `tests/unit/test_grpo_window_batcher.py`, `tests/unit/test_cheap_rejects_pre_queue.py`, `tests/unit/test_miner_engine_v2.py` — enforcement tests.

---

## Branch Setup (do once, before Task 1)

- [ ] **Create the feature branch**

```bash
cd /home/ubuntu/Catalyst
git checkout -b feat/per-window-prompt-range
git status
```
Expected: on branch `feat/per-window-prompt-range`, the untracked `docs/superpowers/...` files listed but nothing staged.

---

## Task 1: Shared `window_prompt_range` function

**Files:**
- Create: `reliquary/shared/prompt_range.py`
- Test: `tests/unit/test_prompt_range.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_prompt_range.py`:

```python
"""Per-window prompt range: deterministic, per-env, half-open, tiny-env no-op."""

from reliquary.shared.prompt_range import window_prompt_range


def test_deterministic_same_inputs():
    a = window_prompt_range("deadbeef", "openmathinstruct", 880_000, 5000)
    b = window_prompt_range("deadbeef", "openmathinstruct", 880_000, 5000)
    assert a == b


def test_window_is_size_wide_and_in_bounds():
    lo, hi = window_prompt_range("deadbeef", "openmathinstruct", 880_000, 5000)
    assert hi - lo == 5000
    assert 0 <= lo
    assert hi <= 880_000


def test_per_env_diverges():
    # Same seeds, different env names must produce different windows.
    math_los = [
        window_prompt_range(f"s{i}", "openmathinstruct", 880_000, 5000)[0]
        for i in range(50)
    ]
    code_los = [
        window_prompt_range(f"s{i}", "opencode", 880_000, 5000)[0]
        for i in range(50)
    ]
    assert math_los != code_los


def test_randomness_spreads_window():
    los = {
        window_prompt_range(f"seed{i}", "openmathinstruct", 880_000, 5000)[0]
        for i in range(200)
    }
    assert len(los) > 150  # 200 distinct seeds -> mostly distinct windows


def test_tiny_env_is_no_op():
    # universe_n <= size -> whole space eligible (covers test envs)
    assert window_prompt_range("deadbeef", "test", 100, 5000) == (0, 100)
    assert window_prompt_range("deadbeef", "test", 5000, 5000) == (0, 5000)


def test_membership_is_half_open():
    lo, hi = window_prompt_range("deadbeef", "openmathinstruct", 880_000, 5000)
    assert lo in range(lo, hi)
    assert hi not in range(lo, hi)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_prompt_range.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reliquary.shared.prompt_range'`.

- [ ] **Step 3: Write the implementation**

Create `reliquary/shared/prompt_range.py`:

```python
"""Deterministic per-window prompt range.

Validator and miner each derive the same contiguous ``[lo, hi)`` slice of an
environment's prompt index space from the shared per-window ``randomness``
seed, so a static/shared bank of pre-curated prompts only lands in-range a
small fraction of windows. Pure and dependency-free: both sides import this
single source of truth — any divergence would reject honest miners.
"""

from __future__ import annotations

import hashlib


def window_prompt_range(
    randomness: str,
    env_name: str,
    universe_n: int,
    size: int,
) -> tuple[int, int]:
    """Return the ``[lo, hi)`` prompt-index slice eligible this window.

    ``randomness`` is the per-window seed both sides already agree on (block
    hash + drand round). ``env_name`` domain-separates math vs code so their
    slices are independent. ``universe_n`` is the prompt index space size
    (``len(env)``); both sides MUST pass the same value, which holds whenever
    they load identical shards (already required by token binding). When
    ``universe_n <= size`` the whole space is eligible (no restriction).
    """
    if universe_n <= size:
        return (0, universe_n)
    seed = hashlib.sha256(
        b"prompt-range/v1|" + env_name.encode() + b"|" + randomness.encode()
    ).digest()
    lo = int.from_bytes(seed[:8], "big") % (universe_n - size)
    return (lo, lo + size)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_prompt_range.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add reliquary/shared/prompt_range.py tests/unit/test_prompt_range.py
git commit -m "feat(prompt-range): add deterministic per-window prompt range function"
```

---

## Task 2: Constants and reject reason

**Files:**
- Modify: `reliquary/constants.py` (after the `COOLDOWN_REBUILD_LOOKBACK` block, ~line 313)
- Modify: `reliquary/protocol/submission.py` (after `PROMPT_FULL`, line 55)
- Test: `tests/unit/test_prompt_range.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_prompt_range.py`:

```python
def test_prompt_range_constants_exist():
    from reliquary.constants import (
        PROMPT_RANGE_SIZE,
        PROMPT_RANGE_ENFORCE_FROM_WINDOW,
    )
    assert isinstance(PROMPT_RANGE_SIZE, int) and PROMPT_RANGE_SIZE > 0
    # Default is a "never enforce" sentinel so the code ships disabled.
    assert PROMPT_RANGE_ENFORCE_FROM_WINDOW >= 2 ** 62


def test_reject_reason_out_of_range_value():
    from reliquary.protocol.submission import RejectReason
    assert RejectReason.PROMPT_OUT_OF_RANGE.value == "prompt_out_of_range"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_prompt_range.py::test_prompt_range_constants_exist tests/unit/test_prompt_range.py::test_reject_reason_out_of_range_value -v`
Expected: FAIL with `ImportError` / `AttributeError: PROMPT_OUT_OF_RANGE`.

- [ ] **Step 3: Add the constants**

In `reliquary/constants.py`, immediately after the `COOLDOWN_REBUILD_LOOKBACK = int(...)` block (the closing `)` near line 312), add:

```python

# Per-window prompt range (anti pre-curation). Each window, miner and
# validator derive the same contiguous slice of the prompt index space from
# the shared per-window randomness; once enforcement is armed only
# submissions whose prompt_idx falls in the slice are accepted. A static or
# shared bank of pre-curated prompts then lands in-range only
# ~PROMPT_RANGE_SIZE/len(env) of windows. See reliquary/shared/prompt_range.py.
PROMPT_RANGE_SIZE = int(_os.environ.get("PROMPT_RANGE_SIZE", "5000"))

# Window number from which the validator hard-enforces the prompt range.
# Below this window the slice is NOT enforced (current behavior, no rejects),
# so the upgraded miner client can ship ahead of the cutover. The default is
# a "never enforce" sentinel: set PROMPT_RANGE_ENFORCE_FROM_WINDOW=N* to the
# agreed cutover window AFTER the gated client is released and announced,
# otherwise un-upgraded miners are rejected ~every window.
PROMPT_RANGE_ENFORCE_FROM_WINDOW = int(
    _os.environ.get("PROMPT_RANGE_ENFORCE_FROM_WINDOW", str(2 ** 63 - 1))
)
```

- [ ] **Step 4: Add the reject reason**

In `reliquary/protocol/submission.py`, immediately after `PROMPT_FULL = "prompt_full"` (line 55), add:

```python
    PROMPT_OUT_OF_RANGE = "prompt_out_of_range"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_prompt_range.py -v`
Expected: PASS (8 passed).

- [ ] **Step 6: Commit**

```bash
git add reliquary/constants.py reliquary/protocol/submission.py tests/unit/test_prompt_range.py
git commit -m "feat(prompt-range): add range size, cutover-window constants and reject reason"
```

---

## Task 3: Validator batcher — cache and enforce

**Files:**
- Modify: `reliquary/validator/batcher.py` (constants import line 17; `__init__` after line 284; new method; `_accept_locked` after line 706)
- Test: `tests/unit/test_grpo_window_batcher.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_grpo_window_batcher.py`:

```python
from reliquary.validator import batcher as batcher_mod


def test_set_prompt_range_none_before_cutover(monkeypatch):
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_ENFORCE_FROM_WINDOW", 10_000)
    b = _make_batcher(window_start=500)
    b.randomness = "deadbeef"
    b.set_prompt_range()
    assert b.prompt_range is None  # window 500 < cutover 10000 -> not armed


def test_set_prompt_range_none_without_randomness(monkeypatch):
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_ENFORCE_FROM_WINDOW", 0)
    b = _make_batcher(window_start=500)
    b.randomness = ""
    b.set_prompt_range()
    assert b.prompt_range is None  # no randomness yet -> no restriction


def test_set_prompt_range_armed(monkeypatch):
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_ENFORCE_FROM_WINDOW", 0)
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_SIZE", 100)
    b = _make_batcher(window_start=500)
    b.randomness = "deadbeef"
    b.set_prompt_range()
    lo, hi = b.prompt_range
    assert hi - lo == 100
    assert 0 <= lo and hi <= 1000  # FakeEnv len is 1000


def test_accept_rejects_out_of_range(monkeypatch):
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_ENFORCE_FROM_WINDOW", 0)
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_SIZE", 100)
    b = _make_batcher(window_start=500)
    b.randomness = "deadbeef"
    b.set_prompt_range()
    lo, hi = b.prompt_range
    out = (hi + 1) % 1000
    if lo <= out < hi:
        out = (lo - 1) % 1000
    resp = b.accept_submission(_request(prompt_idx=out, window_start=500))
    assert resp.accepted is False
    assert resp.reason == RejectReason.PROMPT_OUT_OF_RANGE


def test_accept_in_range_passes_range_gate(monkeypatch):
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_ENFORCE_FROM_WINDOW", 0)
    monkeypatch.setattr(batcher_mod, "PROMPT_RANGE_SIZE", 100)
    b = _make_batcher(window_start=500)
    b.randomness = "deadbeef"
    b.set_prompt_range()
    lo, hi = b.prompt_range
    resp = b.accept_submission(_request(prompt_idx=lo, window_start=500))
    # Passes the range gate; may still hit a later gate, but never this one.
    assert resp.reason != RejectReason.PROMPT_OUT_OF_RANGE
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_grpo_window_batcher.py -k "prompt_range or out_of_range or in_range" -v`
Expected: FAIL — `AttributeError: 'GrpoWindowBatcher' object has no attribute 'set_prompt_range'`.

- [ ] **Step 3a: Add imports to `reliquary/validator/batcher.py`**

In the `from reliquary.constants import (` block at line 17, add these two names (keep alphabetical/grouping consistent with the existing block):

```python
    PROMPT_RANGE_SIZE,
    PROMPT_RANGE_ENFORCE_FROM_WINDOW,
```

After that import block (with the other top-level `from reliquary...` imports), add:

```python
from reliquary.shared.prompt_range import window_prompt_range
```

- [ ] **Step 3b: Add the `prompt_range` attribute**

In `GrpoWindowBatcher.__init__`, immediately after `self.randomness: str = ""` (line 284), add:

```python
        # Per-window eligible prompt slice [lo, hi). None = no restriction
        # (randomness not yet known, or window is before the enforcement
        # cutover). Set by set_prompt_range() once randomness is assigned.
        self.prompt_range: tuple[int, int] | None = None
```

- [ ] **Step 3c: Add the `set_prompt_range` method**

Add this method to `GrpoWindowBatcher` (place it just after `__init__`):

```python
    def set_prompt_range(self) -> None:
        """Compute and cache this window's eligible prompt slice.

        Leaves ``prompt_range`` None (accept any prompt_idx, current behavior)
        until randomness is known AND ``window_start`` has reached
        ``PROMPT_RANGE_ENFORCE_FROM_WINDOW``. Call after assigning randomness.
        """
        if (
            not self.randomness
            or self.window_start < PROMPT_RANGE_ENFORCE_FROM_WINDOW
        ):
            self.prompt_range = None
            return
        self.prompt_range = window_prompt_range(
            self.randomness,
            getattr(self.env, "name", ""),
            len(self.env),
            PROMPT_RANGE_SIZE,
        )
```

- [ ] **Step 3d: Enforce in `_accept_locked`**

In `_accept_locked`, immediately after the bounds check (line 705-706):

```python
        if request.prompt_idx >= len(self.env):
            return reject(RejectReason.BAD_PROMPT_IDX, "prompt")
```

insert:

```python
        if self.prompt_range is not None:
            lo, hi = self.prompt_range
            if not (lo <= request.prompt_idx < hi):
                return reject(RejectReason.PROMPT_OUT_OF_RANGE, "prompt_range")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_grpo_window_batcher.py -v`
Expected: PASS (all, including the 5 new tests; existing tests unaffected because `prompt_range` defaults to None).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_grpo_window_batcher.py
git commit -m "feat(prompt-range): cache and enforce window slice in batcher accept path"
```

---

## Task 4: Validator service wiring + cheap-path enforcement

**Files:**
- Modify: `reliquary/validator/service.py` (`_set_window_randomness`, ~line 721-722)
- Modify: `reliquary/validator/server.py` (cheap path, after line 988)
- Test: `tests/unit/test_cheap_rejects_pre_queue.py` (modify `_setup`, append tests)

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_cheap_rejects_pre_queue.py`, modify the `_setup` signature and body to add a `prompt_range` knob. Change the signature line:

```python
def _setup(*,
           current_checkpoint_hash: str = "sha256:current",
           cooldown_prompts: list[int] | None = None,
           env_len: int = 1000,
           drand_round_check_enabled: bool = False,
           validate_round_returns: RejectReason | None = None,
           prompt_count: int = 0,
           prompt_range: tuple[int, int] | None = None) -> tuple[ValidatorServer, MagicMock]:
```

and, right after `batcher._seal_trigger_round = None` (line 106), add:

```python
    # MagicMock would auto-create a truthy prompt_range; pin it so the range
    # gate only fires when a test sets it explicitly.
    batcher.prompt_range = prompt_range
```

Then append these tests:

```python
def test_out_of_range_rejected_pre_queue():
    s, _ = _setup(prompt_range=(100, 200))
    payload = _submission(prompt_idx=42)  # 42 not in [100, 200)
    _assert_pre_queue_reject(s, payload, RejectReason.PROMPT_OUT_OF_RANGE)


def test_in_range_passes_pre_queue():
    s, _ = _setup(prompt_range=(0, 100))
    payload = _submission(prompt_idx=42)  # in [0, 100)
    with TestClient(s.app) as client:
        r = client.post("/submit", json=payload)
    assert r.json()["reason"] == RejectReason.ACCEPTED.value


def test_no_range_skips_gate_pre_queue():
    s, _ = _setup(prompt_range=None)  # enforcement off (pre-cutover)
    payload = _submission(prompt_idx=42)
    with TestClient(s.app) as client:
        r = client.post("/submit", json=payload)
    assert r.json()["reason"] == RejectReason.ACCEPTED.value
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_cheap_rejects_pre_queue.py -k "range" -v`
Expected: FAIL — `test_out_of_range_rejected_pre_queue` gets `ACCEPTED` instead of `PROMPT_OUT_OF_RANGE` (gate not implemented yet).

- [ ] **Step 3a: Enforce in the cheap arrival path**

In `reliquary/validator/server.py`, immediately after the bad-prompt-idx check (line 987-988):

```python
            if request.prompt_idx >= len(batcher.env):
                return _cheap_reject(RejectReason.BAD_PROMPT_IDX, reject_stage="prompt")
```

insert (use `getattr` to duck-type the batcher — the cheap path already does this for `try_reserve_proof_admission`, and several test-double batchers don't define `prompt_range`):

```python
            prompt_range = getattr(batcher, "prompt_range", None)
            if prompt_range is not None:
                lo, hi = prompt_range
                if not (lo <= request.prompt_idx < hi):
                    return _cheap_reject(
                        RejectReason.PROMPT_OUT_OF_RANGE,
                        reject_stage="prompt_range",
                    )
```

- [ ] **Step 3b: Wire `set_prompt_range()` into the service**

In `reliquary/validator/service.py`, in `_set_window_randomness`, change the assignment loop (line 721-722):

```python
                for batcher in self._active_batchers.values():
                    batcher.randomness = randomness
```

to:

```python
                for batcher in self._active_batchers.values():
                    batcher.randomness = randomness
                    batcher.set_prompt_range()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_cheap_rejects_pre_queue.py -v`
Expected: PASS (all, including the 3 new range tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/service.py reliquary/validator/server.py tests/unit/test_cheap_rejects_pre_queue.py
git commit -m "feat(prompt-range): enforce window slice on cheap path and wire service"
```

---

## Task 5: Miner reference client — restrict sampling

**Files:**
- Modify: `reliquary/miner/engine.py` (imports line 17; `pick_prompt_idx` 81-108; `pick_env_and_prompt` 111-144; loop call site line 335)
- Test: `tests/unit/test_miner_engine_v2.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_miner_engine_v2.py`:

```python
def test_pick_prompt_respects_explicit_range():
    env = FakeEnv()  # len 100
    rng = random.Random(1)
    for _ in range(50):
        idx = pick_prompt_idx(
            env, cooldown_prompts=set(), rng=rng, prompt_range=(20, 40),
        )
        assert 20 <= idx < 40


def test_pick_prompt_range_skips_cooldown():
    env = FakeEnv()
    rng = random.Random(1)
    cooldown = set(range(20, 38))  # leaves only 38, 39 free in [20, 40)
    for _ in range(20):
        idx = pick_prompt_idx(
            env, cooldown_prompts=cooldown, rng=rng, prompt_range=(20, 40),
        )
        assert idx in (38, 39)


def test_pick_prompt_full_range_unchanged():
    env = FakeEnv()
    rng = random.Random(42)
    for _ in range(50):
        idx = pick_prompt_idx(env, cooldown_prompts=set(), rng=rng)
        assert 0 <= idx < 100


def test_pick_env_and_prompt_confines_to_window():
    from reliquary.miner.engine import pick_env_and_prompt
    from reliquary.shared.prompt_range import window_prompt_range
    from reliquary.constants import PROMPT_RANGE_SIZE

    class BigEnv:
        name = "openmathinstruct"
        def __len__(self):
            return 20_000

    envs = {"openmathinstruct": BigEnv()}
    mix = [("openmathinstruct", 8)]
    cooldown = {"openmathinstruct": set()}
    rng = random.Random(7)
    rand = "deadbeefcafe"
    lo, hi = window_prompt_range(rand, "openmathinstruct", 20_000, PROMPT_RANGE_SIZE)
    for _ in range(50):
        name, idx = pick_env_and_prompt(envs, mix, cooldown, rng=rng, randomness=rand)
        assert name == "openmathinstruct"
        assert lo <= idx < hi
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_miner_engine_v2.py -k "range or confines" -v`
Expected: FAIL — `pick_prompt_idx() got an unexpected keyword argument 'prompt_range'`.

- [ ] **Step 3a: Add imports to `reliquary/miner/engine.py`**

In the `from reliquary.constants import (` block at line 17, add `PROMPT_RANGE_SIZE,`. After the constants block, add:

```python
from reliquary.shared.prompt_range import window_prompt_range
```

- [ ] **Step 3b: Restrict `pick_prompt_idx`**

Replace the body of `pick_prompt_idx` (lines 81-108) with:

```python
def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
    prompt_range: tuple[int, int] | None = None,
) -> int:
    """Pick a random prompt index that isn't currently in cooldown.

    When ``prompt_range`` is given, sampling is confined to ``[lo, hi)`` —
    the per-window slice the validator enforces. The reference miner uses
    uniform-random selection with rejection sampling against the cooldown
    set; more sophisticated strategies are left to miner operators.

    Raises ``RuntimeError`` if no eligible prompt can be found.
    """
    rng = rng or _random
    n = len(env)
    lo, hi = (0, n) if prompt_range is None else prompt_range
    lo = max(0, lo)
    hi = min(n, hi)
    span = hi - lo
    if span <= 0:
        raise RuntimeError("no eligible prompt — empty range")
    cd_in_span = sum(1 for c in cooldown_prompts if lo <= c < hi)
    if cd_in_span < span / 2:
        for _ in range(max_attempts):
            idx = lo + rng.randrange(span)
            if idx not in cooldown_prompts:
                return idx
        raise RuntimeError("no eligible prompt found after max attempts")
    eligible = [i for i in range(lo, hi) if i not in cooldown_prompts]
    if not eligible:
        raise RuntimeError("no eligible prompt — range fully in cooldown")
    return rng.choice(eligible)
```

- [ ] **Step 3c: Compute the slice in `pick_env_and_prompt`**

Replace the body of `pick_env_and_prompt` (lines 111-144) with:

```python
def pick_env_and_prompt(
    envs: dict,
    mix: list[tuple[str, int]],
    cooldown_per_env: dict[str, set[int]],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
    randomness: str | None = None,
) -> tuple[str, int]:
    """Sample env per `mix` weights, then a prompt within that env.

    When ``randomness`` is given, each env's prompt is drawn only from that
    window's slice (``window_prompt_range``), matching the validator. Falls
    through to the next env (re-sampling with the chosen env masked) if the
    chosen env's slice is fully in cooldown.
    """
    rng = rng or _random
    names = [n for n, _ in mix]
    weights = [w for _, w in mix]
    if not names:
        raise RuntimeError("pick_env_and_prompt: empty mix")

    available = list(names)
    while available:
        avail_weights = [weights[names.index(n)] for n in available]
        env_name = rng.choices(available, weights=avail_weights)[0]
        env = envs[env_name]
        prompt_range = None
        if randomness:
            env_label = getattr(env, "name", env_name)
            prompt_range = window_prompt_range(
                randomness, env_label, len(env), PROMPT_RANGE_SIZE,
            )
        try:
            idx = pick_prompt_idx(
                env, cooldown_per_env.get(env_name, set()),
                rng=rng, max_attempts=max_attempts, prompt_range=prompt_range,
            )
            return env_name, idx
        except RuntimeError:
            available.remove(env_name)

    raise RuntimeError("pick_env_and_prompt: all envs fully in cooldown")
```

- [ ] **Step 3d: Pass `randomness` from the mining loop**

In `reliquary/miner/engine.py`, at the call site (line 335-337), change:

```python
                    env_name, prompt_idx = pick_env_and_prompt(
                        self.envs, self.mix, self._cooldown_per_env, rng=rng,
                    )
```

to:

```python
                    env_name, prompt_idx = pick_env_and_prompt(
                        self.envs, self.mix, self._cooldown_per_env, rng=rng,
                        randomness=randomness,
                    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_miner_engine_v2.py -v`
Expected: PASS (all, including the 4 new tests; the pre-existing `test_pick_prompt_*` tests still pass because the default `prompt_range=None` reproduces the prior sampling sequence).

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/engine.py tests/unit/test_miner_engine_v2.py
git commit -m "feat(prompt-range): confine reference miner sampling to the window slice"
```

---

## Final Verification

- [ ] **Run the full affected test set**

Run:
```bash
python -m pytest tests/unit/test_prompt_range.py tests/unit/test_grpo_window_batcher.py tests/unit/test_cheap_rejects_pre_queue.py tests/unit/test_miner_engine_v2.py -v
```
Expected: all PASS.

- [ ] **Confirm the gate is shipped OFF**

Run: `python -c "from reliquary.constants import PROMPT_RANGE_ENFORCE_FROM_WINDOW as W; print(W)"`
Expected: a value `>= 2**62` (never-enforce sentinel) — the code is inert until armed.

---

## Deployment (operational — not code; do after merge)

The range is computed everywhere but enforced nowhere until `PROMPT_RANGE_ENFORCE_FROM_WINDOW` is set to a real window `N*`. Sequence:

1. Merge all five tasks (gate still off).
2. **Release the upgraded miner client first** and announce `N*` to miners.
3. Set `PROMPT_RANGE_ENFORCE_FROM_WINDOW=N*` on the validator and deploy. It does nothing different until window `N*`.
4. At `N*` the validator hard-enforces. Un-upgraded miners are then in-range ~`PROMPT_RANGE_SIZE/len(env)` (~0.57% at 5000/880k) → effectively rejected. This is the accepted, pre-announced consequence; the client-first + announcement is the safety rail.

**Hard requirement:** the gated client MUST be live and adopted before `N*`, or honest miners are zeroed too.

**Out of scope (see spec "Future"):** scattered slice for diversity, roaming the full 14M, held-out ground truth (closes the solo difficulty-map residual), per-miner ranges, and pinning `universe_n` to a constant if operators may load a superset of shards (v1 uses `len(env)`, which agrees as long as both sides load identical shards — already required by token binding).
