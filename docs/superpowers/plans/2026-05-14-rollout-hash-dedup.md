# Rollout Hash Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject submissions containing rollouts whose content matches any rollout already entered in a sealed batch within the cooldown horizon. Couple with bumping `BATCH_PROMPT_COOLDOWN_WINDOWS` from 72 → 200.

**Architecture:** New `RolloutHashSet` class mirrors `CooldownMap` (sliding-window set, rebuild from R2 archives at startup). Per-rollout SHA256 over `commit["tokens"]` is computed once at accept-time in `GrpoWindowBatcher._accept_locked` (placed between `SUPERSEDED` and `REWARD_MISMATCH`), stored on `ValidSubmission`, reused at `seal_batch` to populate the persistent set and at `_archive_window` to embed the hex hash in the R2 payload. Compat path in `rebuild_from_history` recomputes from archived `tokens` when the new `hash` field is absent, so the dedup is effective from the first window post-deploy.

**Tech Stack:** Python 3.11+, pytest + pytest-asyncio, hashlib SHA256, existing pydantic models in `reliquary.protocol.submission`, existing dataclasses in `reliquary.validator.batcher`.

**Reference spec:** `docs/superpowers/specs/2026-05-14-rollout-hash-dedup-design.md`

---

## File Structure

**Create:**
- `reliquary/validator/dedup.py` — `RolloutHashSet` class + `compute_rollout_hash` helper. Single responsibility: in-memory hash set with retention horizon and archive-rebuild. Mirrors `reliquary/validator/cooldown.py`.
- `tests/unit/test_dedup.py` — unit tests for the above.

**Modify:**
- `reliquary/protocol/submission.py` — add `RejectReason.HASH_DUPLICATE`.
- `reliquary/constants.py` — bump `BATCH_PROMPT_COOLDOWN_WINDOWS` 72 → 200.
- `reliquary/validator/batcher.py` — add `rollout_hashes` field to `ValidSubmission`, accept `hash_set` parameter, run check in `_accept_locked`, populate set in `seal_batch`.
- `reliquary/validator/service.py` — instantiate `RolloutHashSet`, wire through `open_grpo_window`, add `_rebuild_hashes_from_history`, embed hash field in archive payload.
- `tests/unit/test_grpo_window_batcher.py` — new tests for hash dedup behaviour in the batcher.
- `tests/unit/test_constants.py` — update the stale `BATCH_PROMPT_COOLDOWN_WINDOWS == 50` assertion to match new value.
- `tests/unit/test_archive_window_content.py` — new test for the per-rollout `hash` field in the R2 archive.
- `tests/unit/test_service_v2.py` — new test for `_rebuild_hashes_from_history`.

---

## Task 1: Add `HASH_DUPLICATE` to `RejectReason`

**Files:**
- Modify: `reliquary/protocol/submission.py:22-56`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_dedup.py` (create the file):

```python
"""Unit tests for hash-dedup primitives."""

from reliquary.protocol.submission import RejectReason


def test_hash_duplicate_reject_reason_exists():
    assert RejectReason.HASH_DUPLICATE.value == "hash_duplicate"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_dedup.py::test_hash_duplicate_reject_reason_exists -v`

Expected: FAIL with `AttributeError: HASH_DUPLICATE`.

- [ ] **Step 3: Add the enum value**

In `reliquary/protocol/submission.py`, inside the `RejectReason` enum class, add the new member between `GRAIL_FAIL` and `LOGPROB_MISMATCH` to keep alphabetical ordering:

```python
    GRAIL_FAIL = "grail_fail"
    HASH_DUPLICATE = "hash_duplicate"
    LOGPROB_MISMATCH = "logprob_mismatch"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_dedup.py::test_hash_duplicate_reject_reason_exists -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/protocol/submission.py tests/unit/test_dedup.py
git commit -m "$(cat <<'EOF'
feat(protocol): add HASH_DUPLICATE reject reason

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Bump `BATCH_PROMPT_COOLDOWN_WINDOWS` 72 → 200

**Files:**
- Modify: `reliquary/constants.py:195`
- Modify: `tests/unit/test_constants.py:21-23`

- [ ] **Step 1: Update the failing test to match the target value**

In `tests/unit/test_constants.py`, replace the body of `test_v2_cooldown_values`:

```python
def test_v2_cooldown_values():
    assert C.BATCH_PROMPT_COOLDOWN_WINDOWS == 200
    assert C.BOOTSTRAP_WINDOWS == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_constants.py::test_v2_cooldown_values -v`

Expected: FAIL because constants currently has 72.

- [ ] **Step 3: Update the constant**

In `reliquary/constants.py`, change line 195:

```python
BATCH_PROMPT_COOLDOWN_WINDOWS = 200
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_constants.py::test_v2_cooldown_values -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/constants.py tests/unit/test_constants.py
git commit -m "$(cat <<'EOF'
chore(constants): bump BATCH_PROMPT_COOLDOWN_WINDOWS 72 -> 200

Couples with the rollout-hash dedup mechanism: a wider cooldown
horizon gives the distribution filter more training drift to bite
stale replays that escape the byte-equal hash check via perturbation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement `compute_rollout_hash`

**Files:**
- Create: `reliquary/validator/dedup.py`
- Modify: `tests/unit/test_dedup.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_dedup.py`:

```python
def test_compute_rollout_hash_returns_32_bytes():
    from reliquary.validator.dedup import compute_rollout_hash
    h = compute_rollout_hash([1, 2, 3, 4])
    assert isinstance(h, bytes)
    assert len(h) == 32


def test_compute_rollout_hash_deterministic():
    from reliquary.validator.dedup import compute_rollout_hash
    h1 = compute_rollout_hash([100, 200, 300, 400, 500])
    h2 = compute_rollout_hash([100, 200, 300, 400, 500])
    assert h1 == h2


def test_compute_rollout_hash_differs_on_single_token_change():
    from reliquary.validator.dedup import compute_rollout_hash
    a = compute_rollout_hash([10, 20, 30, 40, 50])
    b = compute_rollout_hash([10, 20, 31, 40, 50])  # single token diff
    assert a != b


def test_compute_rollout_hash_rejects_negative_tokens():
    import pytest
    from reliquary.validator.dedup import compute_rollout_hash
    with pytest.raises(ValueError):
        compute_rollout_hash([1, -2, 3])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_dedup.py -v -k compute_rollout_hash`

Expected: FAIL (4 tests) with `ModuleNotFoundError: No module named 'reliquary.validator.dedup'`.

- [ ] **Step 3: Create the module with the function**

Create `reliquary/validator/dedup.py`:

```python
"""RolloutHashSet — per-rollout content dedup across a cooldown horizon.

A miner that re-submits a rollout whose token content matches one already
entered in a sealed batch within the retention window is rejected with
``RejectReason.HASH_DUPLICATE``. Mirrors the lifecycle of
``reliquary.validator.cooldown.CooldownMap``: in-memory set, rebuilt at
validator startup from the recent R2 archive payloads.
"""

from __future__ import annotations

import hashlib
from typing import Iterable


def compute_rollout_hash(tokens: Iterable[int]) -> bytes:
    """Return SHA256 digest of *tokens* packed as big-endian uint32.

    Deterministic over Python implementations: each int is serialised as a
    fixed 4-byte big-endian unsigned integer and concatenated before
    hashing. Rejects negative values (vocab token ids are always
    non-negative; a negative slipping in here means upstream corruption).
    """
    h = hashlib.sha256()
    for t in tokens:
        if t < 0:
            raise ValueError(f"compute_rollout_hash: negative token id {t}")
        h.update(int(t).to_bytes(4, "big", signed=False))
    return h.digest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_dedup.py -v -k compute_rollout_hash`

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/dedup.py tests/unit/test_dedup.py
git commit -m "$(cat <<'EOF'
feat(validator): compute_rollout_hash helper

Deterministic SHA256 over tokens packed as big-endian uint32. Used by
the upcoming RolloutHashSet for per-rollout dedup.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement `RolloutHashSet.add` / `__contains__` / `__len__`

**Files:**
- Modify: `reliquary/validator/dedup.py`
- Modify: `tests/unit/test_dedup.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_dedup.py`:

```python
def test_hashset_empty_does_not_contain():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    assert compute_rollout_hash([1, 2, 3]) not in s
    assert len(s) == 0


def test_hashset_add_then_contains():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    h = compute_rollout_hash([10, 20, 30])
    s.add(h, window=100)
    assert h in s
    assert len(s) == 1


def test_hashset_add_duplicate_is_idempotent():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    h = compute_rollout_hash([10, 20, 30])
    s.add(h, window=100)
    s.add(h, window=110)
    assert len(s) == 1  # same hash → one entry, latest window kept


def test_hashset_negative_window_rejected():
    import pytest
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    h = compute_rollout_hash([1, 2, 3])
    with pytest.raises(ValueError):
        s.add(h, window=-1)


def test_hashset_negative_retention_rejected():
    import pytest
    from reliquary.validator.dedup import RolloutHashSet
    with pytest.raises(ValueError):
        RolloutHashSet(retention_windows=-1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_dedup.py -v -k hashset`

Expected: FAIL (5 tests) with `ImportError: cannot import name 'RolloutHashSet'`.

- [ ] **Step 3: Implement the class skeleton**

In `reliquary/validator/dedup.py`, append below `compute_rollout_hash`:

```python
class RolloutHashSet:
    """Per-rollout content set with a sliding retention horizon.

    Membership tested via ``__contains__``. Entries older than
    ``retention_windows`` are dropped via ``prune``.
    """

    def __init__(self, retention_windows: int) -> None:
        if retention_windows < 0:
            raise ValueError("retention_windows must be non-negative")
        self._retention_windows = retention_windows
        self._entries: dict[bytes, int] = {}

    def add(self, h: bytes, window: int) -> None:
        if window < 0:
            raise ValueError("window must be non-negative")
        # Keep the most recent window for any given hash.
        prev = self._entries.get(h, -1)
        if window > prev:
            self._entries[h] = window

    def __contains__(self, h: bytes) -> bool:
        return h in self._entries

    def __len__(self) -> int:
        return len(self._entries)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_dedup.py -v -k hashset`

Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/dedup.py tests/unit/test_dedup.py
git commit -m "$(cat <<'EOF'
feat(validator): RolloutHashSet add/contains/len

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement `RolloutHashSet.prune`

**Files:**
- Modify: `reliquary/validator/dedup.py`
- Modify: `tests/unit/test_dedup.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_dedup.py`:

```python
def test_hashset_prune_drops_expired_entries():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    old = compute_rollout_hash([1, 1, 1])
    recent = compute_rollout_hash([2, 2, 2])
    s.add(old, window=100)
    s.add(recent, window=145)
    # At window 151: window 100 is 51 away (>= 50) → drop
    s.prune(current_window=151)
    assert old not in s
    assert recent in s


def test_hashset_prune_keeps_boundary_at_minus_one():
    """An entry at window=100 with retention=50 must stay until current=150."""
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    h = compute_rollout_hash([3, 3, 3])
    s.add(h, window=100)
    s.prune(current_window=149)
    assert h in s
    s.prune(current_window=150)
    assert h not in s


def test_hashset_prune_zero_retention_drops_everything():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=0)
    h = compute_rollout_hash([4, 4, 4])
    s.add(h, window=100)
    s.prune(current_window=100)
    assert h not in s
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_dedup.py -v -k prune`

Expected: FAIL with `AttributeError: 'RolloutHashSet' object has no attribute 'prune'`.

- [ ] **Step 3: Implement `prune`**

In `reliquary/validator/dedup.py`, append inside the `RolloutHashSet` class (after `__len__`):

```python
    def prune(self, current_window: int) -> None:
        """Drop entries whose window is older than the retention horizon.

        An entry at ``window=W`` survives while
        ``current_window - W < retention_windows``. At equality the entry
        is dropped — same half-open interval semantics as ``CooldownMap``.
        """
        if self._retention_windows == 0:
            self._entries.clear()
            return
        horizon = current_window - self._retention_windows
        self._entries = {
            h: w for h, w in self._entries.items() if w > horizon
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_dedup.py -v -k prune`

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/dedup.py tests/unit/test_dedup.py
git commit -m "$(cat <<'EOF'
feat(validator): RolloutHashSet.prune with half-open horizon

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Implement `RolloutHashSet.rebuild_from_history`

**Files:**
- Modify: `reliquary/validator/dedup.py`
- Modify: `tests/unit/test_dedup.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_dedup.py`:

```python
def test_rebuild_from_history_indexes_hash_field():
    """When archives carry an explicit `hash` field per rollout, use it."""
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    h_a = compute_rollout_hash([1, 2, 3]).hex()
    h_b = compute_rollout_hash([4, 5, 6]).hex()
    archives = [
        {
            "window_start": 100,
            "batch": [
                {
                    "prompt_idx": 42,
                    "rollouts": [
                        {"tokens": [1, 2, 3], "hash": h_a, "reward": 1.0},
                        {"tokens": [4, 5, 6], "hash": h_b, "reward": 0.0},
                    ],
                }
            ],
        }
    ]
    s.rebuild_from_history(archives, current_window=110)
    assert bytes.fromhex(h_a) in s
    assert bytes.fromhex(h_b) in s
    assert len(s) == 2


def test_rebuild_from_history_recomputes_when_hash_missing():
    """Backwards-compat: pre-feature archives have only `tokens`, no `hash`."""
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    archives = [
        {
            "window_start": 100,
            "batch": [
                {
                    "prompt_idx": 42,
                    "rollouts": [
                        {"tokens": [7, 8, 9], "reward": 1.0},  # no hash key
                    ],
                }
            ],
        }
    ]
    s.rebuild_from_history(archives, current_window=110)
    assert compute_rollout_hash([7, 8, 9]) in s


def test_rebuild_from_history_skips_expired_windows():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    archives = [
        {
            "window_start": 40,  # expired at current=100 (50 horizon)
            "batch": [{"prompt_idx": 1, "rollouts": [{"tokens": [9, 9]}]}],
        },
        {
            "window_start": 90,
            "batch": [{"prompt_idx": 2, "rollouts": [{"tokens": [8, 8]}]}],
        },
    ]
    s.rebuild_from_history(archives, current_window=100)
    assert compute_rollout_hash([9, 9]) not in s
    assert compute_rollout_hash([8, 8]) in s


def test_rebuild_from_history_clears_previous_state():
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
    s = RolloutHashSet(retention_windows=50)
    stale = compute_rollout_hash([1])
    s.add(stale, window=100)
    s.rebuild_from_history([], current_window=110)
    assert stale not in s
    assert len(s) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_dedup.py -v -k rebuild`

Expected: FAIL with `AttributeError: 'RolloutHashSet' object has no attribute 'rebuild_from_history'`.

- [ ] **Step 3: Implement `rebuild_from_history`**

In `reliquary/validator/dedup.py`, append inside the `RolloutHashSet` class (after `prune`):

```python
    def rebuild_from_history(
        self, archives: list[dict], current_window: int,
    ) -> None:
        """Replace state from a list of archived window payloads.

        Each archive must carry ``window_start`` (int) and ``batch`` (list
        of submissions). Each submission must carry ``rollouts`` (list of
        dicts). Each rollout either has an explicit ``hash`` (hex string)
        — used directly — or only ``tokens`` (list[int]), in which case
        the hash is recomputed via :func:`compute_rollout_hash`.

        Archives whose ``window_start`` is older than the retention horizon
        relative to ``current_window`` are skipped — same semantics as
        :meth:`prune`.
        """
        self._entries.clear()
        horizon = current_window - self._retention_windows
        for archive in archives:
            w = int(archive["window_start"])
            if w <= horizon:
                continue
            for sub in archive.get("batch", []):
                for rollout in sub.get("rollouts", []):
                    h_hex = rollout.get("hash")
                    if h_hex is not None:
                        h = bytes.fromhex(h_hex)
                    else:
                        h = compute_rollout_hash(rollout["tokens"])
                    self.add(h, w)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_dedup.py -v -k rebuild`

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/dedup.py tests/unit/test_dedup.py
git commit -m "$(cat <<'EOF'
feat(validator): RolloutHashSet.rebuild_from_history with compat path

Reads recent R2 archives at validator startup. Uses the new `hash`
field when present, recomputes from `tokens` otherwise — so the dedup
is operational from the first window post-deploy without waiting for
new archives to accumulate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Extend `ValidSubmission` with `rollout_hashes`

**Files:**
- Modify: `reliquary/validator/batcher.py:54-77` (`ValidSubmission` dataclass)
- Modify: `tests/unit/test_grpo_window_batcher.py` (add new test)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_grpo_window_batcher.py`:

```python
def test_valid_submission_has_rollout_hashes_field():
    """ValidSubmission exposes a per-rollout hash list (default empty)."""
    from reliquary.validator.batcher import ValidSubmission
    s = ValidSubmission(
        hotkey="hk", prompt_idx=42,
        merkle_root_bytes=b"\x00" * 32,
    )
    assert s.rollout_hashes == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_grpo_window_batcher.py::test_valid_submission_has_rollout_hashes_field -v`

Expected: FAIL with `AttributeError: 'ValidSubmission' object has no attribute 'rollout_hashes'`.

- [ ] **Step 3: Add the field**

In `reliquary/validator/batcher.py`, modify the `ValidSubmission` dataclass (around lines 54-77). Add `rollout_hashes` after `claimed_checkpoint_hash`:

```python
@dataclass
class ValidSubmission:
    """A submission that passed all v2 verification checks."""

    hotkey: str
    prompt_idx: int
    merkle_root_bytes: bytes
    merkle_root: bytes = field(init=False)  # alias for select_batch Protocol
    sigma: float = 0.0
    rollouts: list[RolloutSubmission] = field(default_factory=list)
    completion_texts: list[str] = field(default_factory=list)
    arrived_at: float = 0.0
    sketch_diff_max: int | None = None
    lp_dev_max: float | None = None
    dist_q10_min: float | None = None
    claimed_checkpoint_hash: str = ""
    rollout_hashes: list[bytes] = field(default_factory=list)

    def __post_init__(self):
        self.merkle_root = self.merkle_root_bytes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_grpo_window_batcher.py::test_valid_submission_has_rollout_hashes_field -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_grpo_window_batcher.py
git commit -m "$(cat <<'EOF'
feat(validator): add rollout_hashes field to ValidSubmission

Holds the per-rollout SHA256 computed at accept time so seal_batch
and _archive_window can reuse them without re-hashing commit tokens.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Wire `hash_set` into `GrpoWindowBatcher` (constructor + accept check)

**Files:**
- Modify: `reliquary/validator/batcher.py:106-209` (constructor + `_accept_locked`)
- Modify: `tests/unit/test_grpo_window_batcher.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_grpo_window_batcher.py`:

```python
def test_hash_dup_rejects_replay_from_persistent_set():
    """A rollout whose tokens are already in the shared hash_set is rejected."""
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash

    hs = RolloutHashSet(retention_windows=50)
    # Seed the set with the hash of the rollout the test will resubmit.
    req = _request(prompt_idx=42, rewards=[1.0] * 4 + [0.0] * 4)
    h = compute_rollout_hash(req.rollouts[0].commit["tokens"])
    hs.add(h, window=499)

    b = _make_batcher(hash_set=hs)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.HASH_DUPLICATE


def test_hash_dup_intra_submission_collision_rejects():
    """Two rollouts in the same submission with identical tokens → reject."""
    from reliquary.validator.dedup import RolloutHashSet

    hs = RolloutHashSet(retention_windows=50)
    # Build a request whose 8 rollouts all share identical commit["tokens"].
    rollouts = []
    for i in range(M_ROLLOUTS):
        commit = _make_commit(success=(i < 4), total_reward=(1.0 if i < 4 else 0.0))
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"], reward=(1.0 if i < 4 else 0.0),
                commit=commit,
            )
        )
    req = BatchSubmissionRequest(
        miner_hotkey="hk", prompt_idx=42, window_start=500,
        merkle_root="00" * 32, rollouts=rollouts, checkpoint_hash="sha256:test",
    )

    b = _make_batcher(hash_set=hs)
    resp = b.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason == RejectReason.HASH_DUPLICATE


def test_hash_dup_none_set_disables_check():
    """Passing hash_set=None disables the check (back-compat for tests)."""
    b = _make_batcher(hash_set=None)
    req = _request(prompt_idx=42, rewards=[1.0] * 4 + [0.0] * 4)
    resp = b.accept_submission(req)
    assert resp.accepted is True


def test_hash_dup_accept_when_not_in_set():
    """Fresh content with no prior hash entry passes."""
    from reliquary.validator.dedup import RolloutHashSet

    hs = RolloutHashSet(retention_windows=50)
    b = _make_batcher(hash_set=hs)
    req = _request(prompt_idx=42, rewards=[1.0] * 4 + [0.0] * 4)
    resp = b.accept_submission(req)
    assert resp.accepted is True
    # rollout_hashes populated on the stored ValidSubmission
    stored = b.valid_submissions()[0]
    assert len(stored.rollout_hashes) == M_ROLLOUTS
    assert all(isinstance(h, bytes) and len(h) == 32 for h in stored.rollout_hashes)
```

Note: the intra-submission test relies on `_make_commit` returning a default fixed token list (no `tokens=` override → all M rollouts share the same default tokens). Verify by inspecting the helper in the same file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_grpo_window_batcher.py -v -k hash_dup`

Expected: FAIL — `_make_batcher` doesn't accept `hash_set`, the new `RejectReason.HASH_DUPLICATE` is never produced.

- [ ] **Step 3: Update `_make_batcher` helper in the test file**

In `tests/unit/test_grpo_window_batcher.py`, modify `_make_batcher` (around line 104) to thread `hash_set` through:

```python
def _make_batcher(**overrides) -> GrpoWindowBatcher:
    class _DefaultFakeTokenizer:
        eos_token_id = 99

    class _DefaultModelStub:
        class config:
            vocab_size = 10000
            max_position_embeddings = 4096

    kwargs = dict(
        window_start=500,
        env=FakeEnv(),
        model=_DefaultModelStub(),
        tokenizer=_DefaultFakeTokenizer(),
        verify_commitment_proofs_fn=_always_true_grail,
        verify_signature_fn=_always_true_sig,
        completion_text_fn=lambda rollout: (
            "CORRECT" if rollout.reward > 0.5 else "wrong"
        ),
        hash_set=None,
    )
    kwargs.update(overrides)
    return GrpoWindowBatcher(**kwargs)
```

- [ ] **Step 4: Add `hash_set` parameter to `GrpoWindowBatcher.__init__`**

In `reliquary/validator/batcher.py`, first add the import at the top of the module (next to `from reliquary.validator.cooldown import CooldownMap`):

```python
from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash
```

Then modify `GrpoWindowBatcher.__init__` signature:

```python
    def __init__(
        self,
        window_start: int,
        env: Environment,
        model: Any,
        *,
        tokenizer: Any = None,
        cooldown_map: CooldownMap | None = None,
        hash_set: RolloutHashSet | None = None,
        bootstrap: bool = False,
        completion_text_fn: Callable[[RolloutSubmission], str],
        canonical_prompt_tokens_fn: Callable[[int], list[int]] | None = None,
        verify_commitment_proofs_fn: Callable[..., Any] | None = None,
        verify_signature_fn: Callable[[dict, str], bool] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
```

In the `__init__` body, store the reference next to the existing `_cooldown` line:

```python
        self._cooldown = (
            cooldown_map if cooldown_map is not None
            else CooldownMap(cooldown_windows=BATCH_PROMPT_COOLDOWN_WINDOWS)
        )
        self._hash_set: RolloutHashSet | None = hash_set
```

- [ ] **Step 5: Insert the HASH_DUPLICATE check in `_accept_locked`**

In `reliquary/validator/batcher.py`, in `_accept_locked` (around lines 220-245), add the new check immediately after the `SUPERSEDED` check and before `get_problem(request.prompt_idx)`:

```python
        if request.prompt_idx in self._claimed_prompts:
            return self._reject(RejectReason.SUPERSEDED, hotkey=hk, prompt_idx=pi)

        # Per-rollout hash dedup against the persistent set + within this
        # submission. Computed once here, reused at seal_batch and archive.
        rollout_hashes: list[bytes] = []
        local_seen: set[bytes] = set()
        for rollout in request.rollouts:
            h = compute_rollout_hash(rollout.commit["tokens"])
            if h in local_seen or (
                self._hash_set is not None and h in self._hash_set
            ):
                return self._reject(
                    RejectReason.HASH_DUPLICATE, hotkey=hk, prompt_idx=pi,
                )
            local_seen.add(h)
            rollout_hashes.append(h)

        problem = self.env.get_problem(request.prompt_idx)
```

Then, at the bottom of `_accept_locked` (around lines 408-422), include `rollout_hashes` when building the `ValidSubmission`:

```python
        self._valid.append(
            ValidSubmission(
                hotkey=request.miner_hotkey,
                prompt_idx=request.prompt_idx,
                merkle_root_bytes=bytes.fromhex(request.merkle_root),
                sigma=sigma,
                rollouts=list(request.rollouts),
                completion_texts=completion_texts,
                arrived_at=self._time_fn(),
                sketch_diff_max=sketch_diff_max,
                lp_dev_max=lp_dev_max,
                dist_q10_min=dist_q10_min,
                claimed_checkpoint_hash=request.checkpoint_hash,
                rollout_hashes=rollout_hashes,
            )
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_grpo_window_batcher.py -v -k hash_dup`

Expected: PASS (4 tests).

Also re-run the full batcher test file to confirm no regressions:

Run: `pytest tests/unit/test_grpo_window_batcher.py -v`

Expected: every existing test still passes.

- [ ] **Step 7: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_grpo_window_batcher.py
git commit -m "$(cat <<'EOF'
feat(validator): batcher HASH_DUPLICATE check + ValidSubmission hashes

Per-rollout SHA256 computed once at accept-time, checked against (a)
the persistent RolloutHashSet and (b) a per-submission local set. The
computed hashes are stored on the resulting ValidSubmission so seal
and archive paths reuse them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Populate hash set + prune at `seal_batch`

**Files:**
- Modify: `reliquary/validator/batcher.py:480-490` (`seal_batch`)
- Modify: `tests/unit/test_grpo_window_batcher.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_grpo_window_batcher.py`:

```python
def test_seal_batch_populates_hash_set():
    """After seal_batch, every batched rollout's hash is in the shared set."""
    from reliquary.validator.dedup import RolloutHashSet

    hs = RolloutHashSet(retention_windows=50)
    b = _make_batcher(hash_set=hs)
    req = _request(prompt_idx=42, rewards=[1.0] * 4 + [0.0] * 4)
    resp = b.accept_submission(req)
    assert resp.accepted is True

    batch = b.seal_batch()
    assert len(batch) == 1
    for sub in batch:
        assert len(sub.rollout_hashes) == M_ROLLOUTS
        for h in sub.rollout_hashes:
            assert h in hs


def test_seal_batch_prunes_expired_hashes():
    """seal_batch calls prune so the set stays bounded across windows."""
    from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash

    hs = RolloutHashSet(retention_windows=50)
    # Seed a stale hash from a window way past retention.
    stale = compute_rollout_hash([1234, 5678])
    hs.add(stale, window=100)

    b = _make_batcher(hash_set=hs)
    # window_start defaults to 500 — stale (w=100) is 400 windows old, well
    # past retention=50.
    req = _request(prompt_idx=42, rewards=[1.0] * 4 + [0.0] * 4)
    b.accept_submission(req)
    b.seal_batch()
    assert stale not in hs


def test_seal_batch_with_none_hash_set_is_noop():
    """seal_batch must not crash when hash_set=None (test fixture path)."""
    b = _make_batcher(hash_set=None)
    req = _request(prompt_idx=42, rewards=[1.0] * 4 + [0.0] * 4)
    b.accept_submission(req)
    batch = b.seal_batch()
    assert len(batch) == 1  # behaviour unchanged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_grpo_window_batcher.py -v -k seal_batch_populates or seal_batch_prunes`

Expected: FAIL — seal_batch doesn't call `add`/`prune`.

- [ ] **Step 3: Update `seal_batch`**

In `reliquary/validator/batcher.py`, replace `seal_batch` (around lines 480-490) with:

```python
    def seal_batch(self) -> list[ValidSubmission]:
        with self._lock:
            batch = select_batch(
                self._valid,
                b=B_BATCH,
                current_window=self.window_start,
                cooldown_map=self._cooldown,
            )
            for sub in batch:
                self._cooldown.record_batched(sub.prompt_idx, self.window_start)
                if self._hash_set is not None:
                    for h in sub.rollout_hashes:
                        self._hash_set.add(h, self.window_start)
            if self._hash_set is not None:
                self._hash_set.prune(self.window_start)
            return batch
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_grpo_window_batcher.py -v -k seal_batch`

Expected: PASS — both new tests and pre-existing `seal_batch_*` tests still green.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_grpo_window_batcher.py
git commit -m "$(cat <<'EOF'
feat(validator): seal_batch populates and prunes RolloutHashSet

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Wire `RolloutHashSet` into `ValidationService`

**Files:**
- Modify: `reliquary/validator/service.py:93-125` (`open_grpo_window`)
- Modify: `reliquary/validator/service.py:142-242` (`ValidationService.__init__`)
- Modify: `reliquary/validator/service.py:325-350` (`_open_window`)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_service_v2.py`:

```python
def test_service_constructs_hash_set_with_cooldown_retention():
    """ValidationService owns a RolloutHashSet sized to BATCH_PROMPT_COOLDOWN_WINDOWS."""
    from unittest.mock import MagicMock
    from reliquary.constants import BATCH_PROMPT_COOLDOWN_WINDOWS
    from reliquary.validator.dedup import RolloutHashSet
    from reliquary.validator.service import ValidationService

    class _FakeEnv:
        name = "fake"
        def __len__(self): return 100
        def get_problem(self, i): return {"prompt": "p", "ground_truth": "a"}
        def compute_reward(self, p, c): return 0.0

    class _FakeWallet:
        class _Hk:
            ss58_address = "5FHk"
            @staticmethod
            def sign(d): return b"sig"
        hotkey = _Hk()

    fake_tok = MagicMock()
    fake_tok.eos_token_id = 99
    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=fake_tok,
        env=_FakeEnv(), netuid=99,
    )
    assert isinstance(svc._hash_set, RolloutHashSet)
    # Retention horizon equals the cooldown horizon (we reuse the constant).
    assert svc._hash_set._retention_windows == BATCH_PROMPT_COOLDOWN_WINDOWS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_service_v2.py::test_service_constructs_hash_set_with_cooldown_retention -v`

Expected: FAIL with `AttributeError: 'ValidationService' object has no attribute '_hash_set'`.

- [ ] **Step 3: Add `_hash_set` in `__init__`**

In `reliquary/validator/service.py`, add the import near the top alongside other validator imports:

```python
from reliquary.validator.dedup import RolloutHashSet
```

In `ValidationService.__init__`, immediately after the line that creates `self._cooldown_map` (around line 220), add:

```python
        self._hash_set = RolloutHashSet(
            retention_windows=BATCH_PROMPT_COOLDOWN_WINDOWS,
        )
```

- [ ] **Step 4: Thread `hash_set` through `open_grpo_window`**

In `reliquary/validator/service.py`, modify `open_grpo_window` (around lines 93-125):

```python
def open_grpo_window(
    window_start: int,
    env,
    model,
    *,
    cooldown_map: CooldownMap,
    hash_set,
    tokenizer,
    bootstrap: bool = False,
) -> GrpoWindowBatcher:
    """Instantiate a GrpoWindowBatcher for this window."""

    def _completion_text(rollout: RolloutSubmission) -> str:
        prompt_len = rollout.commit.get("rollout", {}).get("prompt_length", 0)
        return tokenizer.decode(rollout.tokens[prompt_len:])

    def _canonical_prompt_tokens(prompt_idx: int) -> list[int]:
        problem = env.get_problem(prompt_idx)
        return list(tokenizer.encode(problem["prompt"], add_special_tokens=False))

    return GrpoWindowBatcher(
        window_start=window_start,
        env=env,
        model=model,
        tokenizer=tokenizer,
        cooldown_map=cooldown_map,
        hash_set=hash_set,
        bootstrap=bootstrap,
        completion_text_fn=_completion_text,
        canonical_prompt_tokens_fn=_canonical_prompt_tokens,
    )
```

- [ ] **Step 5: Pass `self._hash_set` from `_open_window`**

In `reliquary/validator/service.py`, modify `_open_window` (around lines 341-346) to pass the set:

```python
        self._active_batcher = open_grpo_window(
            window_start=self._window_n,
            env=self.env, model=self.verify_model,
            cooldown_map=self._cooldown_map,
            hash_set=self._hash_set,
            tokenizer=self.tokenizer,
            bootstrap=bootstrap,
        )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/unit/test_service_v2.py::test_service_constructs_hash_set_with_cooldown_retention -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add reliquary/validator/service.py tests/unit/test_service_v2.py
git commit -m "$(cat <<'EOF'
feat(validator): wire RolloutHashSet into ValidationService

Service owns a single long-lived RolloutHashSet, sized to the cooldown
horizon, threaded into every GrpoWindowBatcher via open_grpo_window.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Implement `_rebuild_hashes_from_history`

**Files:**
- Modify: `reliquary/validator/service.py:625-700` (`run` + new method near `_rebuild_cooldown_from_history`)
- Modify: `tests/unit/test_service_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_service_v2.py`:

```python
@pytest.mark.asyncio
async def test_rebuild_hashes_from_history_populates_set():
    """_rebuild_hashes_from_history reads R2 archives and seeds the hash set."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from reliquary.validator.dedup import compute_rollout_hash
    from reliquary.validator.service import ValidationService

    class _FakeEnv:
        name = "fake"
        def __len__(self): return 100
        def get_problem(self, i): return {"prompt": "p", "ground_truth": "a"}
        def compute_reward(self, p, c): return 0.0

    class _FakeWallet:
        class _Hk:
            ss58_address = "5FHk"
            @staticmethod
            def sign(d): return b"sig"
        hotkey = _Hk()

    fake_tok = MagicMock()
    fake_tok.eos_token_id = 99
    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=fake_tok,
        env=_FakeEnv(), netuid=99,
    )
    svc._window_n = 110

    # Two archive entries, one with explicit hash, one compat (tokens only).
    h_explicit = compute_rollout_hash([10, 20, 30]).hex()
    archives = [
        {
            "window_start": 100,
            "batch": [
                {
                    "prompt_idx": 7,
                    "rollouts": [
                        {"tokens": [10, 20, 30], "hash": h_explicit},
                        {"tokens": [40, 50, 60]},  # compat: no hash key
                    ],
                }
            ],
        }
    ]
    with patch(
        "reliquary.infrastructure.storage.list_recent_datasets",
        new=AsyncMock(return_value=archives),
    ):
        await svc._rebuild_hashes_from_history()

    assert bytes.fromhex(h_explicit) in svc._hash_set
    assert compute_rollout_hash([40, 50, 60]) in svc._hash_set
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_service_v2.py::test_rebuild_hashes_from_history_populates_set -v`

Expected: FAIL with `AttributeError: 'ValidationService' object has no attribute '_rebuild_hashes_from_history'`.

- [ ] **Step 3: Add `_rebuild_hashes_from_history`**

In `reliquary/validator/service.py`, immediately after `_rebuild_cooldown_from_history` (around line 781-809), add:

```python
    async def _rebuild_hashes_from_history(self) -> None:
        """Rebuild ``self._hash_set`` from the last cooldown-horizon archives.

        Mirror of ``_rebuild_cooldown_from_history`` — same archives, same
        horizon. The dedup is operational from the first window post-
        deploy because the compat path in ``RolloutHashSet.rebuild_from_history``
        recomputes hashes from archived ``tokens`` when the new ``hash``
        field is absent.
        """
        try:
            current_window = self._window_n
            archives = await storage.list_recent_datasets(
                current_window=current_window + 1,
                n=BATCH_PROMPT_COOLDOWN_WINDOWS,
            )
            self._hash_set.rebuild_from_history(
                archives, current_window=current_window,
            )
            logger.info(
                "Rebuilt hash set from %d archive windows (current=%d, size=%d)",
                len(archives), current_window, len(self._hash_set),
            )
        except Exception:
            logger.exception(
                "Failed to rebuild hash set from history; starting empty"
            )
```

- [ ] **Step 4: Call it from `run()` after the cooldown rebuild**

In `reliquary/validator/service.py`, in `run()` (around line 625-630), add the call right after `_rebuild_cooldown_from_history`:

```python
        await self._bootstrap_state_from_external()
        await self._rebuild_cooldown_from_history()
        await self._rebuild_hashes_from_history()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_service_v2.py::test_rebuild_hashes_from_history_populates_set -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/service.py tests/unit/test_service_v2.py
git commit -m "$(cat <<'EOF'
feat(validator): rebuild hash set from R2 archives at startup

Same lifecycle as the cooldown rebuild — reads the last
BATCH_PROMPT_COOLDOWN_WINDOWS archives and seeds the in-memory hash
set. Compat path covers pre-feature archives (no `hash` field) by
recomputing from `tokens`.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Embed per-rollout `hash` in the R2 archive

**Files:**
- Modify: `reliquary/validator/service.py:519-624` (`_archive_window` + its `_rollout_payload` helper)
- Modify: `tests/unit/test_archive_window_content.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_archive_window_content.py`:

```python
@pytest.mark.asyncio
async def test_archive_includes_per_rollout_hash():
    """Each rollout in the archive's batch entry carries a hex SHA256 hash."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from reliquary.validator.service import ValidationService

    fake_tok = MagicMock()
    fake_tok.eos_token_id = 99
    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=fake_tok,
        env=_FakeEnv(), netuid=99,
    )

    # Two rollouts with distinct tokens to verify per-rollout hashing.
    r0_tokens = [1, 2, 3, 4]
    r1_tokens = [5, 6, 7, 8]
    valid_sub = _valid_submission(prompt_idx=42)
    valid_sub.rollouts = [
        RolloutSubmission(
            tokens=r0_tokens, reward=1.0,
            commit={"tokens": r0_tokens, "proof_version": "v5",
                    "rollout": {"prompt_length": 2, "completion_length": 2,
                                "token_logprobs": []}},
        ),
        RolloutSubmission(
            tokens=r1_tokens, reward=0.0,
            commit={"tokens": r1_tokens, "proof_version": "v5",
                    "rollout": {"prompt_length": 2, "completion_length": 2,
                                "token_logprobs": []}},
        ),
    ]
    valid_sub.completion_texts = ["a", "b"]

    from reliquary.validator.dedup import compute_rollout_hash
    valid_sub.rollout_hashes = [
        compute_rollout_hash(r0_tokens),
        compute_rollout_hash(r1_tokens),
    ]

    class _FakeBatcher:
        window_start = 500
        randomness = "abcd"
        window_opened_at = 0.0
        reject_counts: dict = {}
        rejected_submissions: list = []
        def valid_submissions(self): return [valid_sub]

    captured = {}

    def _capture_enqueue(window, archive):
        captured["archive"] = archive

    class _StubQueue:
        def enqueue(self, w, a):
            _capture_enqueue(w, a)

    with patch(
        "reliquary.infrastructure.archive_queue.get_archive_queue",
        return_value=_StubQueue(),
    ):
        await svc._archive_window(_FakeBatcher(), [valid_sub])

    archive = captured["archive"]
    entry = archive["batch"][0]
    assert len(entry["rollouts"]) == 2
    assert entry["rollouts"][0]["hash"] == compute_rollout_hash(r0_tokens).hex()
    assert entry["rollouts"][1]["hash"] == compute_rollout_hash(r1_tokens).hex()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_archive_window_content.py::test_archive_includes_per_rollout_hash -v`

Expected: FAIL — the archive payload doesn't have a `hash` field yet.

- [ ] **Step 3: Update `_rollout_payload` inside `_archive_window`**

In `reliquary/validator/service.py`, in `_archive_window` (around line 531-553), change `_rollout_payload` to thread the per-rollout hash through. Replace it with:

```python
        def _rollout_payload(s, with_text: bool):
            out = []
            texts = s.completion_texts if with_text else [None] * len(s.rollouts)
            # rollout_hashes is populated at accept-time; for legacy paths
            # (e.g. test fixtures bypassing _accept_locked) it may be empty,
            # in which case we omit the `hash` field rather than guessing.
            hashes = s.rollout_hashes if s.rollout_hashes else [None] * len(s.rollouts)
            for r, text, h in zip(s.rollouts, texts, hashes):
                tokens = list(r.tokens)
                rollout_dict = (r.commit or {}).get("rollout", {}) or {}
                prompt_length = int(rollout_dict.get("prompt_length", 0))
                completion_length = int(rollout_dict.get(
                    "completion_length", max(0, len(tokens) - prompt_length),
                ))
                eos_terminated = (
                    bool(tokens) and eos_id is not None and tokens[-1] == eos_id
                )
                entry = {
                    "tokens": tokens,
                    "reward": r.reward,
                    "completion_length": completion_length,
                    "eos_terminated": eos_terminated,
                }
                if h is not None:
                    entry["hash"] = h.hex()
                if with_text:
                    entry["completion_text"] = text
                out.append(entry)
            return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_archive_window_content.py::test_archive_includes_per_rollout_hash -v`

Expected: PASS.

Also re-run the pre-existing archive test to confirm the older shape is preserved when `rollout_hashes` is empty:

Run: `pytest tests/unit/test_archive_window_content.py::test_archive_includes_prompt_and_rollout_content -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/service.py tests/unit/test_archive_window_content.py
git commit -m "$(cat <<'EOF'
feat(validator): include per-rollout hash in R2 archive payload

Each batched rollout's hex-encoded SHA256 lands next to the existing
`tokens`/`reward` fields, sourced from the ValidSubmission populated
at accept time. Older test fixtures that bypass _accept_locked omit
the field cleanly (no defaulting).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Full-suite regression sweep

**Files:**
- None — verification only.

- [ ] **Step 1: Run the full unit test suite**

Run: `pytest tests/unit -v --maxfail=5`

Expected: all unit tests pass. If any pre-existing test fails because of the `rollout_hashes` field on `ValidSubmission` (e.g. it positional-constructs the dataclass), update the offending call sites to use keyword arguments and commit separately as a follow-up — but do NOT change behaviour.

- [ ] **Step 2: Run the integration tests that exercise the v2.1 window loop**

Run: `pytest tests/integration/test_v21_window_loop.py tests/integration/test_grpo_market_smoke.py -v`

Expected: green. These tests touch the full service path; they're the highest-confidence check that the hash wiring doesn't break the runtime loop.

- [ ] **Step 3: Static check that all new imports resolve**

Run: `python -c "from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash; from reliquary.validator.batcher import GrpoWindowBatcher; from reliquary.validator.service import ValidationService; print('ok')"`

Expected: `ok` printed, no `ImportError`.

- [ ] **Step 4: Branch ship-readiness summary**

Print a summary of the diff:

Run: `git log --oneline origin/main..HEAD`

Expected output: 13 commits in this order (the spec commit landed before the plan started, the remaining 12 come from the plan tasks):

1. docs(spec): rollout hash deduplication
2. feat(protocol): add HASH_DUPLICATE reject reason
3. chore(constants): bump BATCH_PROMPT_COOLDOWN_WINDOWS 72 -> 200
4. feat(validator): compute_rollout_hash helper
5. feat(validator): RolloutHashSet add/contains/len
6. feat(validator): RolloutHashSet.prune with half-open horizon
7. feat(validator): RolloutHashSet.rebuild_from_history with compat path
8. feat(validator): add rollout_hashes field to ValidSubmission
9. feat(validator): batcher HASH_DUPLICATE check + ValidSubmission hashes
10. feat(validator): seal_batch populates and prunes RolloutHashSet
11. feat(validator): wire RolloutHashSet into ValidationService
12. feat(validator): rebuild hash set from R2 archives at startup
13. feat(validator): include per-rollout hash in R2 archive payload

If the order or count differs, investigate before pushing.

- [ ] **Step 5: Push the branch and open a PR**

Run:

```bash
git push -u origin feat/rollout-hash-dedup
```

Then open a PR via `gh pr create` with the title `feat(validator): per-rollout hash dedup + cooldown bump` and a body that references `docs/superpowers/specs/2026-05-14-rollout-hash-dedup-design.md` and summarises the changes. **Do not push or open the PR without explicit user confirmation** — pushing is a shared-state action.

---

## Notes on test data shape

Two batcher helpers in `tests/unit/test_grpo_window_batcher.py` materially affect the new tests:

1. `_make_commit(tokens=None, ...)` defaults `tokens = list(range(CHALLENGE_K + prompt_length))`. When called repeatedly without an explicit `tokens=` override, every rollout shares the same token list — which is exactly what the intra-submission duplicate test exploits.
2. `_request(prompt_idx=..., rewards=...)` constructs M rollouts, each via `_make_commit`. By default these rollouts all hash to the same value (point 1), so distinct submissions for distinct prompts also hash-collide unless the test explicitly varies the tokens. The persistent-set test seeds the hash set with the first rollout's hash and relies on this default identity.

If a future test needs distinct rollout tokens, pass `tokens=` explicitly per `_make_commit` call.

## Notes on import direction

`reliquary.validator.dedup` is a leaf module (no imports back into `batcher`/`service`), so the unconditional `from reliquary.validator.dedup import RolloutHashSet, compute_rollout_hash` at the top of `batcher.py` is safe — no circular-import risk.
