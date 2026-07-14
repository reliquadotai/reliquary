# Forced-Seed Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each rollout's sampling draw a deterministic function of public per-window randomness, and add a validator gate that detects rollouts NOT generated from that forced draw (killing reward-shape curation).

**Architecture:** A shared module defines the exact warp + inverse-CDF sampler and the per-position public uniform `u_{i,t}`. The miner samples via that sampler instead of a local RNG. The validator, inside the forward pass it already runs for GRAIL, checks per stochastic position that the submitted token equals the inverse-CDF pick under `u_{i,t}`, aggregates a per-group consistency score, and rejects below a floor — enforced only from an announced cutover window (shadow-logged before).

**Tech Stack:** Python 3.12, PyTorch, pydantic v2, pytest. HuggingFace `transformers` on both sides.

## Global Constraints

- All repo-bound text (code, comments, commit messages) in **English**.
- Sampler params are protocol-fixed: `T_PROTO = 0.6`, `TOP_P_PROTO = 0.95`, `TOP_K_PROTO = 20`. The miner and validator MUST use the **identical** warp + inverse-CDF algorithm (same tie-break, same top-p "include the crossing token" rule, canonical order = ascending token id).
- Forced-seed changes the sampling **draw**, not the distribution. The recorded `token_logprobs` (π_old) stay the true model logprobs at chosen tokens — do NOT add a logits processor that alters the distribution (constants.py:326-328).
- New submission fields must have defaults (`BatchSubmissionRequest` is pydantic v2 `extra="forbid"`; old miners omit them).
- Aggregation is **per-group** (all 8 rollouts of a submission), decided after the per-rollout verify loop.
- Enforcement is window-gated: shadow-log before `FORCED_SEED_ENFORCE_FROM_WINDOW`, hard-reject at/after.
- Keep comments to 1-2 sentences; rationale goes in commit messages.

---

### Task 1: Shared forced-sampling module

**Files:**
- Create: `reliquary/environment/forced_sampling.py`
- Test: `tests/unit/test_forced_sampling.py`

**Interfaces:**
- Produces:
  - `warp(logits: torch.Tensor, t: float, top_k: int, top_p: float) -> torch.Tensor` — 1-D `[vocab]` warped probability distribution (canonical order = ascending token id).
  - `pick(probs: torch.Tensor, u: float) -> int` — inverse-CDF token id.
  - `u_at(randomness: str, hotkey: str, prompt_idx: int, checkpoint_hash: str, rollout_index: int, t: int) -> float` — public uniform in `[0, 1)`.
  - `seed_consistency(logits: torch.Tensor, token_ids: list[int], u_values: list[float], *, t: float, top_k: int, top_p: float, stochastic_threshold: float) -> tuple[int, int]` — returns `(n_stochastic, n_match)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_forced_sampling.py
import hashlib
import torch
from reliquary.environment import forced_sampling as fs


def test_pick_inverse_cdf_boundaries():
    probs = torch.tensor([0.5, 0.5])          # token 0 -> [0,0.5), token 1 -> [0.5,1)
    assert fs.pick(probs, 0.0) == 0
    assert fs.pick(probs, 0.49) == 0
    assert fs.pick(probs, 0.5) == 1
    assert fs.pick(probs, 0.999) == 1


def test_warp_topk_topp_masks():
    logits = torch.tensor([10.0, 9.0, 1.0, 1.0])
    probs = fs.warp(logits, t=0.6, top_k=2, top_p=1.0)
    assert probs[2] == 0.0 and probs[3] == 0.0          # top_k=2 masks tail
    assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_u_at_deterministic_and_field_sensitive():
    a = fs.u_at("cd" * 16, "hk1", 7, "sha:abc", 0, 3)
    b = fs.u_at("cd" * 16, "hk1", 7, "sha:abc", 0, 3)
    assert a == b and 0.0 <= a < 1.0
    assert fs.u_at("cd" * 16, "hk1", 7, "sha:abc", 0, 4) != a   # position changes it
    assert fs.u_at("cd" * 16, "hk2", 7, "sha:abc", 0, 3) != a   # hotkey changes it


def test_seed_consistency_perfect_when_tokens_follow_u():
    # two peaked positions (argmax ~1 -> not stochastic) + two flat positions (stochastic)
    logits = torch.tensor([[10.0, 0.0, 0.0],      # argmax token 0
                           [0.2, 0.1, 0.0],       # flat -> stochastic
                           [10.0, 0.0, 0.0],       # argmax token 0
                           [0.1, 0.2, 0.15]])      # flat -> stochastic
    u = [fs.u_at("r", "h", 0, "c", 0, t) for t in range(4)]
    # tokens = what the forced u actually picks (honest miner)
    tokens = [fs.pick(fs.warp(logits[i], t=0.6, top_k=20, top_p=0.95), u[i]) for i in range(4)]
    n_stoch, n_match = fs.seed_consistency(
        logits, tokens, u, t=0.6, top_k=20, top_p=0.95, stochastic_threshold=0.99)
    assert n_stoch >= 1
    assert n_match == n_stoch                       # honest -> every stochastic pos matches


def test_seed_consistency_low_when_tokens_ignore_u():
    logits = torch.tensor([[0.2, 0.1, 0.0], [0.1, 0.2, 0.15], [0.0, 0.1, 0.2]])
    u = [fs.u_at("r", "h", 0, "c", 0, t) for t in range(3)]
    wrong = [fs.u_at("OTHER", "h", 0, "c", 0, t) for t in range(3)]
    tokens = [fs.pick(fs.warp(logits[i], t=0.6, top_k=20, top_p=0.95), wrong[i]) for i in range(3)]
    n_stoch, n_match = fs.seed_consistency(
        logits, tokens, u, t=0.6, top_k=20, top_p=0.95, stochastic_threshold=0.99)
    assert n_stoch >= 1
    assert n_match < n_stoch                         # ignoring u -> mismatches appear
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_forced_sampling.py -v`
Expected: FAIL — `ModuleNotFoundError: reliquary.environment.forced_sampling`.

- [ ] **Step 3: Write minimal implementation**

```python
# reliquary/environment/forced_sampling.py
"""Protocol-fixed sampler shared by miner (generation) and validator (verification).

The per-position draw is a public deterministic function of window randomness, so
there is exactly one legal generation per (miner, prompt, rollout, window). A rollout
not generated from this draw is detectable by teacher-forced consistency.
"""
from __future__ import annotations

import hashlib

import torch

from reliquary.constants import FORCED_SEED_DOMAIN


def warp(logits: torch.Tensor, t: float, top_k: int, top_p: float) -> torch.Tensor:
    """Temperature -> top-k -> top-p, returned in canonical (token-id ascending) order."""
    lg = logits.float() / float(t)
    if top_k and top_k > 0:
        kth = torch.topk(lg, top_k).values[-1]
        lg = torch.where(lg < kth, torch.full_like(lg, float("-inf")), lg)
    probs = torch.softmax(lg, dim=-1)
    if top_p and top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp = torch.where((cum - sp) < top_p, sp, torch.zeros_like(sp))  # include crossing token
        probs = torch.zeros_like(probs).scatter(-1, si, sp)
    return probs / probs.sum()


def pick(probs: torch.Tensor, u: float) -> int:
    """First token id whose cumulative probability exceeds u (inverse-CDF)."""
    cdf = torch.cumsum(probs, dim=-1)
    idx = int(torch.searchsorted(cdf, torch.tensor(float(u), dtype=cdf.dtype), right=True))
    return min(idx, probs.numel() - 1)


def _lp(b: bytes) -> bytes:
    return len(b).to_bytes(2, "big") + b


def u_at(randomness: str, hotkey: str, prompt_idx: int, checkpoint_hash: str,
         rollout_index: int, t: int) -> float:
    """Public uniform in [0, 1) for rollout `rollout_index`, completion position `t`."""
    msg = (FORCED_SEED_DOMAIN.encode()
           + _lp(randomness.encode()) + _lp(hotkey.encode())
           + int(prompt_idx).to_bytes(8, "big")
           + _lp(checkpoint_hash.encode())
           + int(rollout_index).to_bytes(4, "big")
           + int(t).to_bytes(4, "big"))
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big") / 2.0**64


def seed_consistency(logits: torch.Tensor, token_ids: list[int], u_values: list[float], *,
                     t: float, top_k: int, top_p: float,
                     stochastic_threshold: float) -> tuple[int, int]:
    """Teacher-forced check. logits is [n, vocab] predicting token_ids[i] at u_values[i].
    Counts stochastic positions (max_prob < threshold) and how many match the forced pick."""
    n_stoch = n_match = 0
    n = min(len(token_ids), len(u_values), logits.shape[0])
    for i in range(n):
        probs = warp(logits[i], t=t, top_k=top_k, top_p=top_p)
        if float(probs.max()) < stochastic_threshold:
            n_stoch += 1
            if pick(probs, u_values[i]) == int(token_ids[i]):
                n_match += 1
    return n_stoch, n_match
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_forced_sampling.py -v`
Expected: PASS (5 tests). (Task 2 adds `FORCED_SEED_DOMAIN`; if run before Task 2, import fails — do Task 2 first or add the constant now.)

- [ ] **Step 5: Commit**

```bash
git add reliquary/environment/forced_sampling.py tests/unit/test_forced_sampling.py
git commit -m "feat(forced-seed): shared warp + inverse-CDF sampler and seed derivation"
```

---

### Task 2: Constants

**Files:**
- Modify: `reliquary/constants.py` (add near the `T_PROTO`/`TOKEN_AUTH_*` block, ~line 318-666)
- Test: `tests/unit/test_forced_seed_constants.py`

**Interfaces:**
- Produces module constants: `FORCED_SEED_DOMAIN`, `FORCED_SEED_STOCHASTIC_MAXPROB`, `FORCED_SEED_CONSISTENCY_FLOOR`, `FORCED_SEED_MIN_STOCH_POSITIONS`, `FORCED_SEED_ENFORCE_FROM_WINDOW`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_forced_seed_constants.py
import reliquary.constants as c


def test_forced_seed_constants_defaults():
    assert c.FORCED_SEED_DOMAIN == "reliquary-forced-seed-v1"
    assert c.FORCED_SEED_STOCHASTIC_MAXPROB == 0.99
    assert c.FORCED_SEED_CONSISTENCY_FLOOR == 0.80
    assert c.FORCED_SEED_MIN_STOCH_POSITIONS == 30
    assert c.FORCED_SEED_ENFORCE_FROM_WINDOW == 2 ** 63 - 1   # sentinel: never, until armed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_forced_seed_constants.py -v`
Expected: FAIL — `AttributeError: module 'reliquary.constants' has no attribute 'FORCED_SEED_DOMAIN'`.

- [ ] **Step 3: Write minimal implementation**

Add to `reliquary/constants.py` (after the `TOKEN_AUTH_*` block; `_os` is already imported at line 52):

```python
# ──────────────── FORCED-SEED SAMPLING ────────────────
# Domain separation for the per-position public uniform u_{i,t}.
FORCED_SEED_DOMAIN = "reliquary-forced-seed-v1"
# A position counts toward the seed-consistency check only if its warped max
# probability is below this (i.e. the forced draw actually chooses the token).
FORCED_SEED_STOCHASTIC_MAXPROB = 0.99
# Reject a group whose stochastic-position match rate is below this floor
# (measured: honest ~0.92-0.96, non-forced ~0.60).
FORCED_SEED_CONSISTENCY_FLOOR = 0.80
# Below this many stochastic positions in a group, the gate abstains (accepts)
# rather than risk a false reject on thin signal.
FORCED_SEED_MIN_STOCH_POSITIONS = 30
# Enforce from this window onward; before it the gate is shadow-only. Default
# sentinel = never, until the operator announces + arms the cutover window.
FORCED_SEED_ENFORCE_FROM_WINDOW = int(
    _os.environ.get("FORCED_SEED_ENFORCE_FROM_WINDOW", str(2 ** 63 - 1))
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_forced_seed_constants.py tests/unit/test_forced_sampling.py -v`
Expected: PASS (Task 1 now imports `FORCED_SEED_DOMAIN` cleanly too).

- [ ] **Step 5: Commit**

```bash
git add reliquary/constants.py tests/unit/test_forced_seed_constants.py
git commit -m "feat(forced-seed): constants (thresholds + announced-cutover window)"
```

---

### Task 3: RejectReason member + `protocol_version` schema field

**Files:**
- Modify: `reliquary/protocol/submission.py:22` (enum) and `:113-159` (`BatchSubmissionRequest`)
- Test: `tests/unit/test_forced_seed_schema.py`

**Interfaces:**
- Produces: `RejectReason.SEED_MISMATCH` (value `"seed_mismatch"`); `BatchSubmissionRequest.protocol_version: int` (default 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_forced_seed_schema.py
from reliquary.protocol.submission import RejectReason, BatchSubmissionRequest


def test_seed_mismatch_reason_exists():
    assert RejectReason.SEED_MISMATCH.value == "seed_mismatch"


def test_protocol_version_defaults_zero_and_accepts_int():
    fields = BatchSubmissionRequest.model_fields
    assert "protocol_version" in fields
    assert fields["protocol_version"].default == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_forced_seed_schema.py -v`
Expected: FAIL — `AttributeError: SEED_MISMATCH`.

- [ ] **Step 3: Write minimal implementation**

In `reliquary/protocol/submission.py`, add to the `RejectReason` enum (near `TOKEN_TAMPERED`):

```python
    SEED_MISMATCH = "seed_mismatch"
```

And add to `BatchSubmissionRequest` (mirror the `drand_round` sentinel style, next to it):

```python
    protocol_version: int = Field(default=0, ge=0)  # 0 = pre-forced-seed client
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_forced_seed_schema.py tests/unit/test_batch_submission_schema.py -v`
Expected: PASS (schema tests unaffected — new field has a default).

- [ ] **Step 5: Commit**

```bash
git add reliquary/protocol/submission.py tests/unit/test_forced_seed_schema.py
git commit -m "feat(forced-seed): SEED_MISMATCH reason + protocol_version field"
```

---

### Task 4: ProofResult seed counters + verifier plumbing

**Files:**
- Modify: `reliquary/validator/verifier.py` — `ProofResult` (~:79-83), `verify_commitment_proofs` (:281), `_gpu_completion_token_stats` (:451-482)
- Test: `tests/unit/test_forced_seed_verifier.py`

**Interfaces:**
- Consumes: `forced_sampling.seed_consistency` (Task 1); constants (Task 2).
- Produces: `ProofResult.seed_n_stochastic: int`, `ProofResult.seed_n_match: int` (default 0). `verify_commitment_proofs(..., seed_u_values: list[float] | None = None)` fills them when `seed_u_values` is provided.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_forced_seed_verifier.py
import torch
from reliquary.validator import verifier
from reliquary.environment import forced_sampling as fs


def test_gpu_completion_seed_counts_perfect_for_forced_tokens():
    # 3 completion positions, flat distributions -> all stochastic
    logits = torch.tensor([[0.2, 0.1, 0.0], [0.1, 0.2, 0.15], [0.0, 0.1, 0.2]])
    u = [fs.u_at("r", "h", 0, "c", 0, t) for t in range(3)]
    tokens = [fs.pick(fs.warp(logits[i], t=0.6, top_k=20, top_p=0.95), u[i]) for i in range(3)]
    n_stoch, n_match = verifier._gpu_seed_consistency(logits, tokens, u)
    assert n_stoch == 3 and n_match == 3


def test_proof_result_has_seed_fields():
    p = verifier.ProofResult(all_passed=True, passed=0, checked=0, has_sparse_outputs=False)
    assert p.seed_n_stochastic == 0 and p.seed_n_match == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_forced_seed_verifier.py -v`
Expected: FAIL — `AttributeError: module 'reliquary.validator.verifier' has no attribute '_gpu_seed_consistency'`.

- [ ] **Step 3: Write minimal implementation**

Add fields to `ProofResult` (the dataclass around verifier.py:79):

```python
    seed_n_stochastic: int = 0
    seed_n_match: int = 0
```

Add a helper next to `_gpu_completion_token_stats` in `reliquary/validator/verifier.py`:

```python
def _gpu_seed_consistency(logits_slice, token_ids, u_values):
    """Seed-consistency counts over completion positions. `logits_slice` is
    [n, vocab] (position i predicts token_ids[i]); returns (n_stochastic, n_match)."""
    from reliquary.environment.forced_sampling import seed_consistency
    from reliquary.constants import (
        T_PROTO, TOP_K_PROTO, TOP_P_PROTO, FORCED_SEED_STOCHASTIC_MAXPROB,
    )
    return seed_consistency(
        logits_slice.float().cpu(), list(token_ids), list(u_values),
        t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO,
        stochastic_threshold=FORCED_SEED_STOCHASTIC_MAXPROB)
```

In `verify_commitment_proofs(commit, model, window_randomness, *, tokenizer=None, seed_u_values=None)`
add the `seed_u_values` kwarg. After `logits_gpu` is available and completion bounds
(`t_start`, `t_end`) are computed (near verifier.py:451-482), when `seed_u_values` is not None,
gather the completion-position logits and the completion token ids aligned to `seed_u_values`
(index `j` = completion offset), call `_gpu_seed_consistency`, and set
`result.seed_n_stochastic`, `result.seed_n_match`. Align by completion offset `j = t - prompt_length`
so `seed_u_values[j]` matches position `t`; skip offsets `>= len(seed_u_values)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_forced_seed_verifier.py tests/unit/test_behavioural_validators.py -v`
Expected: PASS (existing verifier tests unaffected — `seed_u_values` defaults to None → no-op).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/verifier.py tests/unit/test_forced_seed_verifier.py
git commit -m "feat(forced-seed): verifier computes per-rollout seed-consistency counts"
```

---

### Task 5: Batcher group gate + shadow forensics

**Files:**
- Modify: `reliquary/validator/batcher.py` — the `_verify_commitment` call (:1051-1056), the per-rollout loop, and post-loop in `_accept_locked`
- Modify: `reliquary/validator/auth_forensics.py` — add a shadow-record helper
- Test: `tests/unit/test_forced_seed_gate.py`

**Interfaces:**
- Consumes: `forced_sampling.u_at` (Task 1); constants (Task 2); `ProofResult.seed_n_*` (Task 4); `RejectReason.SEED_MISMATCH` (Task 3).
- Produces: group-level reject on `SEED_MISMATCH`; shadow forensics record.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_forced_seed_gate.py
from reliquary.validator.batcher import _forced_seed_verdict
from reliquary.protocol.submission import RejectReason


def test_gate_rejects_below_floor_when_enforcing():
    # 100 stochastic positions, 60 matches (0.60) -> below 0.80 floor
    reject = _forced_seed_verdict(n_stoch=100, n_match=60, window=200,
                                  enforce_from=100)
    assert reject is True


def test_gate_accepts_above_floor():
    reject = _forced_seed_verdict(n_stoch=100, n_match=95, window=200, enforce_from=100)
    assert reject is False


def test_gate_abstains_below_min_positions():
    reject = _forced_seed_verdict(n_stoch=10, n_match=0, window=200, enforce_from=100)
    assert reject is False       # too few positions -> abstain, never false-reject


def test_gate_shadow_before_cutover():
    reject = _forced_seed_verdict(n_stoch=100, n_match=0, window=50, enforce_from=100)
    assert reject is False       # before window -> shadow only
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_forced_seed_gate.py -v`
Expected: FAIL — `ImportError: cannot import name '_forced_seed_verdict'`.

- [ ] **Step 3: Write minimal implementation**

Add a module-level pure function to `reliquary/validator/batcher.py` (near the other helpers):

```python
def _forced_seed_verdict(n_stoch: int, n_match: int, window: int, enforce_from: int) -> bool:
    """True => reject the group for seed mismatch. Abstains on thin signal; shadow
    before the cutover window."""
    from reliquary.constants import (
        FORCED_SEED_CONSISTENCY_FLOOR, FORCED_SEED_MIN_STOCH_POSITIONS,
    )
    if window < enforce_from:
        return False
    if n_stoch < FORCED_SEED_MIN_STOCH_POSITIONS:
        return False
    return (n_match / n_stoch) < FORCED_SEED_CONSISTENCY_FLOOR
```

Wire it into `_accept_locked`:

1. Before the `for rollout in request.rollouts` loop, init `grp_stoch = grp_match = 0`.
2. When calling `self._verify_commitment(...)` (batcher.py:1051), build and pass the per-rollout u-stream:

```python
from reliquary.environment.forced_sampling import u_at
seed_u = [u_at(self.randomness, request.miner_hotkey, request.prompt_idx,
               request.checkpoint_hash, rollout_index, j)
          for j in range(completion_len)]
proof = self._verify_commitment(rollout.commit, self.model, self.randomness,
                                tokenizer=self.tokenizer, seed_u_values=seed_u)
```

where `rollout_index` is the loop index (use `enumerate(request.rollouts)`).

3. After each proof: `grp_stoch += proof.seed_n_stochastic; grp_match += proof.seed_n_match`.
4. After the loop (all rollouts passed the other gates), before final accept:

```python
from reliquary.constants import FORCED_SEED_ENFORCE_FROM_WINDOW
if _forced_seed_verdict(grp_stoch, grp_match, self._window_n, FORCED_SEED_ENFORCE_FROM_WINDOW):
    logger.info("seed_mismatch hotkey=%s stoch=%d match=%d", hk, grp_stoch, grp_match)
    return reject(RejectReason.SEED_MISMATCH, "forced_seed")
record_forced_seed_shadow(hk, request.prompt_idx, grp_stoch, grp_match)
```

Add `record_forced_seed_shadow(hotkey, prompt_idx, n_stoch, n_match)` to
`reliquary/validator/auth_forensics.py` mirroring the existing private-forensics writers
(log the score `n_match / max(1, n_stoch)` and counts).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_forced_seed_gate.py tests/unit/test_grpo_window_batcher.py -v`
Expected: PASS (existing batcher tests: `FORCED_SEED_ENFORCE_FROM_WINDOW` defaults to sentinel → shadow → no new rejects).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher.py reliquary/validator/auth_forensics.py tests/unit/test_forced_seed_gate.py
git commit -m "feat(forced-seed): per-group validator gate (shadow before cutover)"
```

---

### Task 6: Miner forced-seed sampler + protocol_version

**Files:**
- Modify: `reliquary/miner/engine.py` — `_generate_m_rollouts` (:596), the `model.generate` calls (:659, :266, phase-2 kwargs :664-672)
- Modify: `reliquary/miner/submitter.py` — set `protocol_version`
- Test: `tests/unit/test_forced_seed_miner.py`

**Interfaces:**
- Consumes: `forced_sampling.warp/pick/u_at` (Task 1); constants (Task 2).
- Produces: rollouts whose stochastic-position tokens equal the forced-`u` picks (so the Task 4/5 validator scores ~1.0); `protocol_version=1` on submissions.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_forced_seed_miner.py
import torch
from reliquary.miner.engine import _forced_decode
from reliquary.environment import forced_sampling as fs


class _FixedLogitsModel:
    """Returns constant per-step logits so the test is deterministic."""
    def __init__(self, logits_row):
        self._row = torch.tensor(logits_row)
    def __call__(self, input_ids, **kw):
        b, s = input_ids.shape
        out = self._row.expand(b, s, self._row.numel()).clone()
        class _O: pass
        o = _O(); o.logits = out; o.past_key_values = None
        return o


def test_forced_decode_follows_u_stream():
    model = _FixedLogitsModel([0.2, 0.1, 0.0])          # flat -> stochastic
    prompt = [1, 2, 3]
    toks = _forced_decode(model, prompt, u_stream=lambda t: fs.u_at("r", "h", 0, "c", 0, t),
                          max_new_tokens=4, eos_ids=set(), t=0.6, top_k=20, top_p=0.95)
    expected = [fs.pick(fs.warp(model._row, t=0.6, top_k=20, top_p=0.95),
                        fs.u_at("r", "h", 0, "c", 0, i)) for i in range(len(toks))]
    assert toks == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_forced_seed_miner.py -v`
Expected: FAIL — `ImportError: cannot import name '_forced_decode'`.

- [ ] **Step 3: Write minimal implementation**

Add a single-sequence forced decode loop to `reliquary/miner/engine.py`:

```python
def _forced_decode(model, prompt_tokens, *, u_stream, max_new_tokens, eos_ids,
                   t, top_k, top_p):
    """Autoregressive decode picking each token via inverse-CDF on the public u stream."""
    import torch
    from reliquary.environment.forced_sampling import warp, pick
    dev = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
    ids = list(prompt_tokens)
    out = model(torch.tensor([ids], device=dev), use_cache=True)
    past, last = out.past_key_values, out.logits[0, -1]
    gen = []
    for step in range(max_new_tokens):
        tokid = pick(warp(last, t=t, top_k=top_k, top_p=top_p), u_stream(step))
        gen.append(tokid)
        if tokid in eos_ids:
            break
        out = model(torch.tensor([[tokid]], device=dev), past_key_values=past, use_cache=True)
        past, last = out.past_key_values, out.logits[0, -1]
    return gen
```

Then in `_generate_m_rollouts` (engine.py:596), replace the batched `self.vllm_model.generate(...)`
call (engine.py:659) with a per-rollout loop over `range(M_ROLLOUTS)` calling `_forced_decode`,
building each rollout's `u_stream` as
`lambda t, i=i: u_at(randomness, self.hotkey_ss58, problem_idx, self.checkpoint_hash, i, t)`
(source `self.hotkey_ss58`, `problem_idx`, `self.checkpoint_hash` from the fields the engine already
holds — the miner hotkey the wallet signs with, the prompt index from `problem`, and the loaded
checkpoint revision; confirm exact attribute names at the call site). Apply the same to the phase-2
BFT answer generate (engine.py:266, kwargs :664-672).

In `reliquary/miner/submitter.py`, set `protocol_version=1` when constructing
`BatchSubmissionRequest`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_forced_seed_miner.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/engine.py reliquary/miner/submitter.py tests/unit/test_forced_seed_miner.py
git commit -m "feat(forced-seed): miner samples via forced-u inverse-CDF; protocol_version=1"
```

---

## Post-implementation (operator, not code)

1. Deploy validator (gate in shadow) + reference miner client.
2. Announce the cutover window to miners.
3. Watch `record_forced_seed_shadow` scores climb toward ~0.92-0.96 as adoption proceeds; recalibrate `FORCED_SEED_CONSISTENCY_FLOOR` / `FORCED_SEED_MIN_STOCH_POSITIONS` against the live honest floor.
4. Set `FORCED_SEED_ENFORCE_FROM_WINDOW` to the announced window. Enforcement arms.

## Self-review notes

- Spec §3 seed derivation → Task 1 `u_at`. §4.1 sampler → Task 1 `warp`/`pick`. §4.3 gate → Tasks 4+5. §4.4 per-group → Task 5 (`_forced_seed_verdict` aggregates, called post-loop). §4.5 cutover → Task 2 window + Task 5 `window < enforce_from` shadow branch. §4.6 params → Task 2. §4.7 abstention → Task 5 `n_stoch < MIN` branch (test `test_gate_abstains_below_min_positions`). §2 evidence informs thresholds. BFT exclusion (§4.3 step 3): the validator already passes `exempt_positions`/force-span-carved completion bounds; align `seed_u_values` to the same completion offsets so injected force-span tokens are naturally outside the checked range — verify during Task 4 wiring.
- Type consistency: `seed_consistency`/`_gpu_seed_consistency` return `(n_stochastic, n_match)` everywhere; `ProofResult.seed_n_stochastic/seed_n_match` names used identically in Tasks 4 and 5.
- Open follow-up (not in this plan): vectorize `seed_consistency` on-GPU if per-window verify latency regresses; per-rollout hardening variant.
