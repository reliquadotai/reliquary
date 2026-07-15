# Difficulty Auction - Protocol Implementation Plan

> **Audit status (2026-07-15): DO NOT EXECUTE.** This is the original plan, not
> an approved rollout checklist. The active branch has been reconciled with
> current `main`; deferred proof, the 300 second collection deadline, and
> `PROMPT_CLAIMED` are intentionally absent. Follow
> `docs/superpowers/specs/2026-07-15-difficulty-auction-audit.md` instead.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop proving submissions that cannot win, then replace the 8-distinct seal trigger with a fixed 300 s collection deadline — so the validator trains on the hardest prompts the fleet can find instead of the fastest ones to arrive.

**Architecture:** `_accept_locked` currently runs the GRAIL GPU proof inline (batcher.py:1366), under `self._lock`, for every admitted submission. We split it: a **cheap admission phase** (schema → dedup → grade → score) that never touches the GPU, and an **expensive verification phase** (proof + every proof-dependent gate) that runs at seal time, top-down over the score ranking, and stops once `B_BATCH` submissions have passed. Stage 1 delivers this and is shippable on its own (a pure cadence win — proofs drop from ~19/window to ≤16). Stage 2 then swaps the seal trigger for the deadline, which is only affordable *because* Stage 1 landed.

**Tech Stack:** Python 3.11, pytest, asyncio. No new dependencies.

## Global Constraints

- **Score, selection and shadow already exist and are tested.** `verifier.submission_value()`, `batch_auction.select_batch_auction()`, `ValidSubmission.value`, `GrpoWindowBatcher.shadow_auction`. Do not reimplement them; wire to them.
- **Proving IS selecting — there is no economically neutral middle state.** An earlier draft of this plan claimed arming was out of scope and that only *when we prove* would change. That was wrong: you cannot choose whom to prove without choosing whom to pay, because an unproven submission can never be paid. Ranked proving therefore arms the auction, by construction. Do not try to "keep Stage 1 neutral" — it is not a thing.
- **The free-negative guard is structural, not a pre-flight measurement.** Under the auction a correct answer graded wrong lowers k toward the score peak, so a false negative is worth the maximum payout — and the grader has false negatives *by construction* (`openmathinstruct.py:312` returns "not equal" when an expression is too costly to expand; `except Exception: return False` does the same on any parse failure). Those bounds exist because the FAST path must grade ~69 submissions in-window. The auction selects only 8 — so we can afford, on exactly those 8, the unbounded oracle the fast path cannot. See Task 9.
- **The proof is held under `self._lock`** (batcher.py:923 → 1366, the whole per-rollout loop 1320-1854, 5-25 s/submission). Moving it out is the point; do not reintroduce GPU work under that lock.
- **Determinism is consensus-critical.** Validators must converge on identical weights. Never order anything by wall-clock arrival, local dict order, or `id()`. Use `drand_round` and the canonical hash.
- All repo-bound text (code, comments, commits) in **English**.
- Run tests with: `python3 -m pytest tests/unit -q --ignore=tests/unit/test_envelope_priming_bypass.py --ignore=tests/unit/test_envelope_signature.py --ignore=tests/unit/test_validator_server.py`
  Baseline on this branch: **1093 pass**, 2 pre-existing failures in `test_security.py` (`test_rejects_v4`, `test_rejects_unknown_version`) — unrelated, do not try to fix.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `reliquary/constants.py` | protocol constants | add `MAX_PROOF_ATTEMPTS_PER_WINDOW`, `FORENSIC_SAMPLE_PER_WINDOW`, `WINDOW_COLLECTION_SECONDS` |
| `reliquary/validator/batcher.py` | admission + seal | split `_accept_locked`; add `PendingSubmission`, `_verify_expensive`, `_prove_ranked` |
| `reliquary/validator/service.py` | window lifecycle | swap seal trigger for the deadline (Stage 2) |
| `reliquary/validator/server.py` | HTTP admission | drop the proof-budget reservation dance (Stage 2) |
| `tests/unit/test_deferred_proof.py` | **new** | Stage 1 behaviour |
| `tests/unit/test_collection_deadline.py` | **new** | Stage 2 behaviour |

---

# STAGE 1 — Prove only what can win

Independently shippable. After this stage the window still seals on 8 distinct
prompts; only the *proving* moves.

---

### Task 1: `PendingSubmission` — a graded, scored, UNPROVEN submission

**Files:**
- Modify: `reliquary/validator/batcher.py` (near `ValidSubmission`, ~line 256)
- Test: `tests/unit/test_deferred_proof.py` (create)

**Interfaces:**
- Produces: `PendingSubmission` dataclass with fields `hotkey: str`, `prompt_idx: int`,
  `request: Any`, `rewards: list[float]`, `value: float`, `drand_round: int`,
  `merkle_root: bytes`, `selection_digest: bytes`, `arrived_at: float`, plus
  `telemetry: Any = None`. `value` is computed in `__post_init__` via
  `submission_value(self.rewards)`.
  It must satisfy the same duck-type `select_batch_auction` consumes
  (`hotkey`, `prompt_idx`, `drand_round`, `value`, `merkle_root`, `selection_digest`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_deferred_proof.py
"""Proof deferral: a submission is graded and scored at admission, but not proven
until it is ranked high enough to win. See
docs/superpowers/specs/2026-07-14-difficulty-auction-design.md §7
"""
from reliquary.validator.batcher import PendingSubmission


def _pending(hotkey="a", prompt_idx=1, k=2, m=8, drand_round=1):
    return PendingSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        request=None,
        rewards=[1.0] * k + [0.0] * (m - k),
        drand_round=drand_round,
        merkle_root=hotkey.encode().ljust(32, b"\x00"),
        selection_digest=hotkey.encode().ljust(32, b"\x00"),
    )


def test_pending_submission_is_scored_at_admission():
    """Scoring is cheap (it only needs the rewards), so it happens before the GPU
    ever sees the submission — that is what lets us rank before proving."""
    assert _pending(k=2).value > _pending(k=6).value


def test_pending_submission_ranks_in_the_auction():
    """It must satisfy the duck-type select_batch_auction consumes, so the same
    ranking code works on unproven candidates."""
    from reliquary.validator.batch_auction import select_batch_auction
    from reliquary.validator.cooldown import CooldownMap

    hard = _pending(hotkey="hard", prompt_idx=1, k=2)
    easy = _pending(hotkey="easy", prompt_idx=2, k=6)

    batch, _ = select_batch_auction(
        [easy, hard], b=1,
        cooldown_map=CooldownMap(cooldown_windows=0), current_window=1, pool=1.0,
    )

    assert [s.hotkey for s in batch] == ["hard"]
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python3 -m pytest tests/unit/test_deferred_proof.py -q`
Expected: `ImportError: cannot import name 'PendingSubmission'`

- [ ] **Step 3: Implement**

In `reliquary/validator/batcher.py`, immediately before `class ValidSubmission`:

```python
@dataclass
class PendingSubmission:
    """A submission that passed every CHEAP check and has been graded, but has
    NOT been proven on the GPU yet.

    The proof (5-25 s of GPU, under ``_lock``) is the most expensive thing the
    validator does, and today we spend it on every admitted submission — ~19 per
    window, to keep 8. Since the score depends only on the rewards, we can rank
    first and prove only the candidates that can actually win. A submission that
    cannot reach the top B is never proven.

    Fabricated groups DO rank at the top (a miner who never runs the model can
    hand-write a k=2 reward vector). That is safe: the proof still runs before
    anyone is paid, so fabricating earns zero. See the spec §7.
    """

    hotkey: str
    prompt_idx: int
    request: Any
    rewards: list[float]
    drand_round: int
    merkle_root: bytes
    selection_digest: bytes
    arrived_at: float = 0.0
    telemetry: Any = None
    value: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.value = submission_value(self.rewards)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_deferred_proof.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_deferred_proof.py
git commit -m "feat(auction): PendingSubmission — graded and scored, not yet proven"
```

---

### Task 2: Extract the expensive phase into `_verify_expensive`

This is the core refactor. `_accept_locked` (batcher.py:1016-2040) currently does
cheap checks, then grading (1169), then the GPU proof (1366) and every
proof-dependent gate (1400-1918), then accepts. Cut it at the proof.

**Files:**
- Modify: `reliquary/validator/batcher.py:1366-1935` (the post-`u`-stream region)
- Test: `tests/unit/test_deferred_proof.py`

**Interfaces:**
- Produces: `GrpoWindowBatcher._verify_expensive(pending: PendingSubmission) -> ValidSubmission | None`
  Returns the `ValidSubmission` on success, `None` on any proof-gate rejection
  (having already called `self._reject(...)` with the correct stage, so the
  existing per-hotkey debt accounting and archive entries still happen).

**What moves (all of it needs the GPU forward — confirmed against the code):**
`GRAIL_FAIL` (1460), `is_cap_truncation` (1499), `verify_termination`'s natural-EOS
path (1492, needs `proof.p_stop`), `verify_logprobs_claim` (1608),
`evaluate_token_distribution` (1630), `evaluate_boxed_answer_probability` (1654),
`evaluate_token_authenticity` (1674), `evaluate_all_token_auth_shadow` (1698),
`evaluate_code_semantic_token_authenticity` (1772), and the forced-seed gate
(1856-1918, which reads only proof-derived `seed_n_*` tallies).

**What does NOT need the proof but may stay in the expensive phase for simplicity:**
`has_eos_padding` (1486) and `validate_force_span` (1595) take no `proof` argument.
Leave them where they are — moving them buys nothing and risks changing behaviour.

- [ ] **Step 1: Write the failing test**

```python
def test_accept_does_not_touch_the_gpu():
    """Admission must be proof-free. If the GPU is called during accept, the
    whole design collapses — we would be proving ~69 submissions per window."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    calls = []

    def _exploding_proof(*a, **kw):
        calls.append(1)
        raise AssertionError("GRAIL must not run during admission")

    b = _make_batcher(verify_commitment_proofs_fn=_exploding_proof)
    resp = b.accept_submission(_request(prompt_idx=7, hotkey="miner"))

    assert resp.accepted is True
    assert calls == []
    assert len(b.pending_submissions()) == 1
    assert b.pending_submissions()[0].value > 0.0   # graded + scored


def test_verify_expensive_runs_the_proof_and_returns_a_valid_submission():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    b.accept_submission(_request(prompt_idx=7, hotkey="miner"))
    pending = b.pending_submissions()[0]

    proven = b._verify_expensive(pending)

    assert proven is not None
    assert proven.hotkey == "miner"
    assert proven.value == pending.value
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python3 -m pytest tests/unit/test_deferred_proof.py -q -k "gpu or expensive"`
Expected: FAIL — `AssertionError: GRAIL must not run during admission` (the proof
still runs inline), and `AttributeError: 'GrpoWindowBatcher' object has no attribute 'pending_submissions'`.

- [ ] **Step 3: Implement**

1. Add to `__init__` (next to `self._valid`): `self._pending: list[PendingSubmission] = []`
   and `self.pending_count = 0` (lock-free read for `/state`, mirroring `valid_count`).

2. Add the accessor next to `valid_submissions()`:

```python
    def pending_submissions(self) -> list[PendingSubmission]:
        """Graded, scored, not yet proven. The auction ranks these."""
        with self._lock:
            return list(self._pending)
```

3. In `_accept_locked`, **cut immediately after the `u`-stream build (line ~1364)**.
   Everything from the `self._verify_commitment(...)` call (1366) through the
   forced-seed gate (1918) moves verbatim into a new method `_verify_expensive`.
   In `_accept_locked`, replace that whole region with:

```python
        pending = PendingSubmission(
            hotkey=request.miner_hotkey,
            prompt_idx=request.prompt_idx,
            request=request,
            rewards=list(rewards),
            drand_round=request.drand_round,
            merkle_root=merkle_root_bytes,
            selection_digest=selection_digest_bytes or merkle_root_bytes,
            arrived_at=self._time_fn(),
            telemetry=telemetry,
        )
        self._pending.append(pending)
        self._submissions_per_prompt.setdefault(request.prompt_idx, []).append(pending)
        self.pending_count = len(self._pending)
        self.last_valid_submission_at = self._time_fn()
        self.last_valid_submission_wall_ts = self._wall_clock()
        self._maybe_trigger_seal(request, telemetry)
        return AcceptResponse(accepted=True)
```

4. `_verify_expensive(self, pending)` re-derives its locals from
   `pending.request` (the same `request` object `_accept_locked` had), runs the
   moved code unchanged, and on success ends with the existing
   `ValidSubmission(...)` construction (currently at 1937-1982) — but instead of
   appending to `self._valid`, it **returns** it. Every `reject(...)` inside the
   moved region becomes `return None` after calling `self._reject(...)` exactly as
   before, so debt accounting and archive entries are unchanged.

5. Move the `distinct_valid_prompt_count() >= B_BATCH` seal trigger (1993-2010)
   into `_maybe_trigger_seal`, and make it count **distinct PENDING prompts**, not
   valid ones. This is load-bearing: `_valid` no longer fills during the window,
   so a trigger that reads it would never fire.

```python
    def distinct_pending_prompt_count(self) -> int:
        """Distinct non-cooldown prompts among graded (unproven) submissions."""
        seen = {
            p.prompt_idx for p in self._pending
            if not self._cooldown.is_in_cooldown(p.prompt_idx, self.window_start)
        }
        return len(seen)
```

- [ ] **Step 4: Run the full suite**

Run: `python3 -m pytest tests/unit -q --ignore=tests/unit/test_envelope_priming_bypass.py --ignore=tests/unit/test_envelope_signature.py --ignore=tests/unit/test_validator_server.py`
Expected: the 2 new tests pass. **Many existing `test_grpo_window_batcher.py` tests
will now fail** — they assert on `b._valid` after `accept_submission`, which is now
empty until seal. That is correct and expected; fix them in Task 3, not by
weakening the new behaviour.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_deferred_proof.py
git commit -m "refactor(batcher): split _accept_locked — cheap admission vs expensive proof

The GRAIL proof (5-25s of GPU, held under _lock) ran on every admitted
submission: ~19 per window, to keep 8. Extract it and every proof-dependent gate
into _verify_expensive, callable at seal time on the ranked candidates only."
```

---

### Task 3: Prove top-down at seal, bounded

**Files:**
- Modify: `reliquary/constants.py`, `reliquary/validator/batcher.py` (`seal_batch`, ~2229)
- Test: `tests/unit/test_deferred_proof.py`

**Interfaces:**
- Consumes: `_verify_expensive`, `select_batch_auction`, `PendingSubmission`
- Produces: `GrpoWindowBatcher._prove_ranked(pool: float) -> list[ValidSubmission]` —
  fills `self._valid` with up to `B_BATCH` proven submissions, spending at most
  `MAX_PROOF_ATTEMPTS_PER_WINDOW` proofs.

- [ ] **Step 1: Write the failing test**

```python
def test_proving_stops_once_b_submissions_pass():
    """The GPU saving. We must not prove candidate 9 when 8 have already passed."""
    from tests.unit.test_grpo_window_batcher import (
        _always_true_grail, _make_batcher, _request,
    )

    proofs = []

    def _counting_proof(commit, model, randomness):
        proofs.append(1)
        return _always_true_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_counting_proof)
    for i in range(12):
        b.accept_submission(_request(prompt_idx=i, hotkey=f"m{i}"))

    b.seal_batch()

    assert len(b.valid_submissions()) == 8
    assert len(proofs) == 8          # NOT 12


def test_failed_proof_promotes_the_next_ranked():
    """Promote-on-failure: a fabricated group tops the ranking (it names its own
    score), fails the proof, and the honest submission behind it takes the slot."""
    from tests.unit.test_grpo_window_batcher import (
        _always_false_grail, _always_true_grail, _make_batcher, _request,
    )

    def _fail_only_the_faker(commit, model, randomness):
        # the faker's group is the one whose rollouts carry its hotkey; the test
        # helper stamps the hotkey into the commit, so key off that.
        if commit.get("hotkey") == "faker":
            return _always_false_grail(commit, model, randomness)
        return _always_true_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_fail_only_the_faker)
    b.accept_submission(_request(prompt_idx=1, hotkey="faker"))
    b.accept_submission(_request(prompt_idx=2, hotkey="honest"))

    b.seal_batch()

    assert [s.hotkey for s in b.valid_submissions()] == ["honest"]


def test_proof_attempts_are_capped():
    """A griefer fabricates groups that rank at the top and always fail the proof.
    He costs us at most MAX_PROOF_ATTEMPTS_PER_WINDOW, never more."""
    from reliquary.constants import MAX_PROOF_ATTEMPTS_PER_WINDOW
    from tests.unit.test_grpo_window_batcher import (
        _always_false_grail, _make_batcher, _request,
    )

    proofs = []

    def _counting_false_grail(commit, model, randomness):
        proofs.append(1)
        return _always_false_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_counting_false_grail)
    for i in range(40):
        b.accept_submission(_request(prompt_idx=i, hotkey=f"grief{i}"))

    b.seal_batch()

    assert b.valid_submissions() == []
    assert len(proofs) == MAX_PROOF_ATTEMPTS_PER_WINDOW
```

- [ ] **Step 2: Run and watch fail**

Run: `python3 -m pytest tests/unit/test_deferred_proof.py -q -k "prove or promote or capped"`
Expected: `ImportError: cannot import name 'MAX_PROOF_ATTEMPTS_PER_WINDOW'`

- [ ] **Step 3: Implement**

In `reliquary/constants.py`, after `MAX_SLOTS_PER_COLDKEY_PER_WINDOW`:

```python
# Ceiling on GPU proofs spent in one window under the difficulty auction. We prove
# the ranked candidates top-down and stop as soon as B_BATCH have passed, so the
# honest cost is exactly B_BATCH. This bounds the dishonest one.
#
# A fabricated group ranks at the TOP by construction: the score is computed from
# the reward vector, and a miner who never runs the model can hand-write a k=2
# vector, which is the exact peak of v(k). Fabricating cannot earn anything (the
# proof runs before payment, so he fails and is paid zero), but it can make us
# spend a proof. This caps that at 16 — and since a hotkey is locked out after
# MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW=2 failures, burning all 16
# requires 8 registered hotkeys, which registration cost already taxes.
MAX_PROOF_ATTEMPTS_PER_WINDOW = 16

# Random non-winners proven each window purely for forensics. Deferring the proof
# means the authenticity gates (token-auth, distribution, forced-seed) only ever
# see the WINNERS — enforcement stays complete (nobody unproven is paid) but fleet
# VISIBILITY is lost, and those gates are how we caught the pre-generating miner
# and 1088 seed_mismatch rejects in 855 windows. Sampled by drand so a miner
# cannot predict whether he will be looked at.
FORENSIC_SAMPLE_PER_WINDOW = 2
```

In `batcher.py`, add `_prove_ranked` and call it at the top of `seal_batch`
(before `explain_batch_selection` / `select_batch_and_distribute`, which read
`self._valid`):

```python
    def _prove_ranked(self, pool: float) -> list[ValidSubmission]:
        """Prove candidates in score order until B_BATCH pass. Never prove a loser.

        Fabricated groups rank first (they name their own score), so the failures
        we pay for cluster at the top. That is fine — they earn nothing — but it
        is why the attempt budget exists.
        """
        ranked = sorted(self._pending, key=_rank_key)
        attempts = 0
        proven: list[ValidSubmission] = []
        claimed: set[int] = set()

        for pending in ranked:
            if len(proven) >= B_BATCH:
                break
            if attempts >= MAX_PROOF_ATTEMPTS_PER_WINDOW:
                logger.warning(
                    "proof_budget_exhausted window=%s proven=%d attempts=%d",
                    self.window_start, len(proven), attempts,
                )
                break
            if pending.prompt_idx in claimed:
                continue
            if self._cooldown.is_in_cooldown(pending.prompt_idx, self.window_start):
                continue
            attempts += 1
            sub = self._verify_expensive(pending)
            if sub is None:
                continue          # rejected; promote the next-ranked
            proven.append(sub)
            claimed.add(pending.prompt_idx)

        self._valid = proven
        self.valid_count = len(proven)
        self.proof_attempts = attempts
        return proven
```

Import `_rank_key` from `reliquary.validator.batch_auction` (export it there).

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_deferred_proof.py -q`
Expected: all pass.

- [ ] **Step 5: Repair the existing batcher tests**

`test_grpo_window_batcher.py` has tests asserting `b._valid` right after
`accept_submission`. Those now need a `b.seal_batch()` first (or should assert on
`b.pending_submissions()`). Update them — the new behaviour is correct; do not
weaken it. Then run the full suite and confirm you are back to **1093 pass + the
2 known `test_security` failures**.

- [ ] **Step 6: Commit**

```bash
git add reliquary/ tests/
git commit -m "feat(auction): prove ranked candidates top-down, bounded at 16 attempts"
```

---

### Task 4: Forensic sample of non-winners

**Files:** `reliquary/validator/batcher.py`, `tests/unit/test_deferred_proof.py`

- [ ] **Step 1: Write the failing test**

```python
def test_forensic_sample_proves_some_losers():
    """We stop paying losers, but we must not stop LOOKING at them: the auth gates
    only run on proven submissions, and they are how tampering gets caught."""
    from reliquary.constants import FORENSIC_SAMPLE_PER_WINDOW
    from tests.unit.test_grpo_window_batcher import (
        _always_true_grail, _make_batcher, _request,
    )

    proofs = []

    def _counting(commit, model, randomness):
        proofs.append(1)
        return _always_true_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_counting)
    for i in range(20):
        b.accept_submission(_request(prompt_idx=i, hotkey=f"m{i}"))

    b.seal_batch()

    assert len(b.valid_submissions()) == 8
    assert len(proofs) == 8 + FORENSIC_SAMPLE_PER_WINDOW
    assert len(b.forensic_sample) == FORENSIC_SAMPLE_PER_WINDOW
```

- [ ] **Step 2: Run and watch fail** — `AttributeError: forensic_sample`

- [ ] **Step 3: Implement.** After `_prove_ranked` fills `self._valid`, pick
`FORENSIC_SAMPLE_PER_WINDOW` submissions from the *unproven remainder*, chosen
deterministically from the window randomness (`sha256(randomness ‖ merkle_root)`,
lowest digests win — never `random.sample`, which would break validator
consensus). Run `_verify_expensive` on them, discard the result, keep the reject
verdicts for the archive. Store the outcome on `self.forensic_sample`.
These proofs are outside the `MAX_PROOF_ATTEMPTS_PER_WINDOW` budget.

- [ ] **Step 4: Run tests.** Expected: pass.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(auction): prove a drand-chosen sample of non-winners for forensics"
```

---

### Task 5: Retire the dead proof-admission budgets

Stage 1 makes several counters meaningless — nothing reserves a proof at HTTP
admission any more, because admission no longer proves.

**Files:** `reliquary/constants.py`, `reliquary/validator/batcher.py`,
`reliquary/validator/server.py`, `reliquary/validator/service.py`

- [ ] **Step 1:** Delete `try_reserve_proof_admission` / `start_proof_admission` /
`cancel_proof_admission` / `finish_proof_admission` (batcher.py:767-886) and their
call sites (server.py:1847-1849, 1884, 1959, 2097, 2142, 2177, 2292).
Delete `MAX_PROOF_CANDIDATES_PER_WINDOW` (already dead), `MAX_POST_TRIGGER_PROOF_CANDIDATES`,
and `service._proof_admission_exhausted_and_drained` (service.py:676-677).

**Keep** `MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW = 96` — it now bounds *grading*,
which is the real admission cost, and it is the DoS bound on the submit queue.
**Keep** `_expensive_proof_failures_by_hotkey` and
`MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW = 2` — under deferred proving
this is the per-hotkey half of the griefer bound (Task 3), so it becomes *more*
important, not less. It is now charged inside `_verify_expensive`, at seal.

- [ ] **Step 2:** Fix the pre-existing bug the code map surfaced: the
`distribution` reject stage is in `_PROOF_FAILURE_DEBT_STAGES` (batcher.py:236-249)
but is *also* emitted by the CPU-only opposite-reward-clone check (batcher.py:1247).
A submission that never reached the GPU therefore charges GPU-failure debt. Split
the stage name: emit `distribution_clones` from the cheap check and leave
`distribution` for the proof-derived one, with only the latter in the debt set.

- [ ] **Step 3:** Run the full suite. Expected: 1093 pass + the 2 known failures.

- [ ] **Step 4: Commit**

```bash
git commit -am "chore(batcher): drop the proof-admission reservation machinery

Nothing reserves a GPU proof at HTTP admission any more — admission is proof-free.
Also stops the CPU-only opposite-reward-clone reject from charging per-hotkey
GPU-failure debt (it emits stage 'distribution', which is in the debt set)."
```

---

### 🚦 Stage 1 gate — verify the cadence win before continuing

- [ ] Deploy Stage 1 alone and watch for 24h. The claim to verify:
  **proofs per window drop from ~19 to ≤ 8 + 2, and the processing phase
  (last-arrival → last-decision, median 173 s today) roughly halves.**
- [ ] Re-run `.r2_analysis/window_cadence.py` against fresh archives. If
  processing has not dropped, **stop** — Stage 2's deadline is only affordable
  because of this, and shipping it without the win costs 18% of throughput.

---

# STAGE 2 — The collection deadline

Only start once the Stage 1 gate passes.

---

### Task 6: `WINDOW_COLLECTION_SECONDS` — seal on time, not on count

**Files:** `reliquary/constants.py`, `reliquary/validator/service.py`,
`reliquary/validator/batcher.py`, `tests/unit/test_collection_deadline.py` (create)

**Interfaces:**
- Produces: the window seals exactly `WINDOW_COLLECTION_SECONDS` after it opened,
  regardless of how many submissions arrived.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_collection_deadline.py
"""The window is time-boxed. Its MINIMUM duration is the deadline itself — an
early seal is exactly the speed race we are removing: whoever triggered it would
cut off the slow-but-hard submissions still generating.
"""

def test_eighth_distinct_prompt_does_not_seal_the_window():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for i in range(12):
        b.accept_submission(_request(prompt_idx=i, hotkey=f"m{i}"))

    assert b.is_sealed() is False       # 12 > B_BATCH, and still open


def test_window_seals_when_the_deadline_expires():
    from reliquary.constants import WINDOW_COLLECTION_SECONDS
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    now = [1000.0]
    b = _make_batcher(time_fn=lambda: now[0])
    b.accept_submission(_request(prompt_idx=1, hotkey="m"))

    assert b.is_sealed() is False
    now[0] += WINDOW_COLLECTION_SECONDS + 1
    b.poll_deadline()

    assert b.is_sealed() is True


def test_late_submission_is_accepted_while_the_window_is_open():
    """No more BATCH_FILLED. A miner who took 250s on a hard prompt still gets in
    — that is the entire point of the deadline."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for i in range(20):
        assert b.accept_submission(_request(prompt_idx=i, hotkey=f"m{i}")).accepted
```

- [ ] **Step 2: Run and watch fail** — the 8-distinct trigger seals at 8.

- [ ] **Step 3: Implement**

`constants.py`:

```python
# Fixed collection window for the difficulty auction. The window no longer seals
# on the 8th distinct prompt; it stays open for this long and accepts everything.
#
# This is also the MINIMUM window duration, by construction. An early seal would
# be the speed race we are removing: whoever triggered it would cut off the
# slow-but-hard submissions still generating. Sized from live traffic — math
# generation is 176s at the median and 267s at p75, and windows already run 277s
# of collection today, so 300s captures ~89% of submissions at near-zero cadence
# cost ONCE proofs are deferred (spec §2.5).
WINDOW_COLLECTION_SECONDS = 300.0
```

In `batcher.py`: delete `_maybe_trigger_seal` (Task 2), `_seal_trigger_round`,
`_delayed_seal_at_drand_boundary`, and the `BATCH_FILLED` reject (1044-1052).
Record `self.opened_at = self._time_fn()` in `__init__` and add:

```python
    def poll_deadline(self) -> bool:
        """Seal iff the collection deadline has expired. Idempotent."""
        if self._seal_flag.is_set():
            return True
        if self._time_fn() - self.opened_at >= WINDOW_COLLECTION_SECONDS:
            self._seal_flag.set()
            return True
        return False
```

In `service.py`, the window loop waits on the deadline instead of the seal event
from the 8th prompt. Remove the sparse-window breakers (`SPARSE_VALID_*`,
service.py:596-629) and `WINDOW_TIMEOUT_SECONDS` — a fixed deadline answers
"when do we stop waiting?" unconditionally, which is the only question they exist
to answer.

- [ ] **Step 4: Run tests.** Expected: pass.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(auction): time-box the window on a 300s collection deadline

Replaces the 8-distinct seal trigger, the drand-boundary seal extension, the
sparse-window liveness breakers and WINDOW_TIMEOUT_SECONDS — all of which exist
only to answer 'when do we stop waiting?', which a fixed deadline answers
unconditionally. BATCH_FILLED is gone: 96% of rejects were that, 45593 valid
submissions thrown away in 855 windows for arriving late."
```

---

### Task 7: One prompt, one slot — dedup at admission

**Files:** `reliquary/constants.py`, `reliquary/validator/batcher.py:1082-1084`,
`tests/unit/test_collection_deadline.py`

- [ ] **Step 1: Write the failing test**

```python
def test_second_submission_for_a_claimed_prompt_is_rejected():
    """One prompt = one slot, decided at ADMISSION. This also kills the
    variance-farming sybil: the forced-seed seed contains the hotkey, so N hotkeys
    on one prompt would otherwise buy N independent draws of k and the operator
    would submit whichever landed nearest k=2."""
    from reliquary.validator.batcher import RejectReason
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    assert b.accept_submission(_request(prompt_idx=7, hotkey="first")).accepted

    resp = b.accept_submission(_request(prompt_idx=7, hotkey="second"))

    assert resp.accepted is False
    assert resp.reason == RejectReason.PROMPT_CLAIMED
```

- [ ] **Step 2: Run and watch fail** — `MAX_SUBMISSIONS_PER_PROMPT = 10` allows it.

- [ ] **Step 3: Implement.** Add `RejectReason.PROMPT_CLAIMED`. Change the
`PROMPT_FULL` check (batcher.py:1082-1084) to reject at the *first* existing
submission for the prompt. Delete `MAX_SUBMISSIONS_PER_PROMPT`. Delete the
same-prompt K-way emission split in `batch_selection.py` — it can no longer occur.

- [ ] **Step 4: Run tests.** Expected: pass.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(auction): one prompt, one slot — reject duplicates at admission"
```

---

### Task 8: Code grader throughput — NOT NEEDED (scope note)

Dropped after review. The original worry was that the deadline triples the graded
pool (~19 → ~69) and code grading (gVisor, 5s, 8 workers) could exceed the window.
It does not, for two reasons confirmed in the code:

1. **Code is not on the live path.** `DEFAULT_ENVIRONMENTS = "openmathinstruct"`;
   the sandboxed code grader does not run in production.
2. **Grading is at admission, not at seal.** `env.compute_reward` runs in
   `_accept_locked` (batcher.py:1108) as each submission arrives, so it is spread
   across the full 300s collection window — never a seal-time spike. Eight workers
   over 300s handle far more than the pool, and `MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW`
   + `MAX_PENDING_PROOF_QUEUE_DEPTH` already bound bursts.

If `opencodeinstruct` is ever enabled under the deadline, re-measure code-grading
wall time per submission and confirm `p95 × pool / GRADER_POOL_SIZE` fits inside
`WINDOW_COLLECTION_SECONDS`; raise `GRADER_POOL_SIZE` if not. No change today.

### 🚦 Stage 2 gate

- [ ] Re-run `.r2_analysis/auction_replay.py` on post-deadline archives. The
  candidate pool should now be ~69/window, not ~19 — and mean k of the shadow
  auction batch should fall **below** the 3.49 the pre-deadline replay produced,
  because there is finally enough hard supply to fill 8 slots.
- [ ] Confirm cadence held: `.r2_analysis/window_cadence.py` should show a cycle
  near 300 s + (proving 8-10) + train, i.e. **≥ 9 windows/h**.

---

# STAGE 3 — Arming (BLOCKED)

Do not start. Swapping `select_batch_and_distribute` for `select_batch_auction` in
`seal_batch` is a two-line change, and it is gated on the free-negative
measurement (spec §9): under the auction a **false negative is worth the maximum
payout**, because a correct answer graded wrong lowers k toward the peak. The
grader has a class of false negatives *by construction* — `openmathinstruct.py:312`
returns "not equal" when an expression is too expensive to expand, and
`except Exception: return False` does the same on any parse failure.

Measure first (spec §9), then arm.
