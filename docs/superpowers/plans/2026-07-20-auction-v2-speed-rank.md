# Auction v2 Speed-Rank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the drand-hash tiebreak in the production difficulty auction with speed (validator-observed arrival drand round), and pay exact `(score, arrival)` draws with the v1 fair-split machinery.

**Architecture:** `_prove_ranked` ranks by `(-value, arrival_round)`, groups exact ties into tiers, proves every member of paying tiers (top tiers + the boundary tier), and records a per-winner tier ordinal. `select_batch_and_distribute` / `explain_batch_selection` gain a `slot_round_of` callable so the auction can feed the tier ordinal into the battle-tested v1 slot/split/burn machinery unchanged. The observation-only `difficulty_auction._rank_key` is aligned.

**Tech Stack:** Python, pytest. Repo: `/home/ubuntu/Catalyst`, branch `feat/auction-v2-speed-rank`.

**Spec:** `docs/superpowers/specs/2026-07-20-auction-v2-speed-rank-design.md`

## Global Constraints

- All repo-bound text (comments, commit messages) in English; inline comments 1-2 sentences max.
- Never push; commit on `feat/auction-v2-speed-rank` only.
- Run tests with `python -m pytest tests/unit/<file> -x -q` from repo root.
- Tie = EXACT equality of `(score.value, arrival_round)`.
- `arrival_round` = `pending.telemetry.arrival_drand_round` when present, else `pending.drand_round` (miner-submitted; mock/test-mode fallback only).
- Do not change: admission filters, forensic sampling, budgets/caps, burn rule, legacy (non-auction) mode defaults.

---

### Task 1: `slot_round_of` parameter in batch_selection

**Files:**
- Modify: `reliquary/validator/batch_selection.py` (functions at :68 and :210)
- Test: `tests/unit/test_batch_selection.py` (append)

**Interfaces:**
- Produces: `select_batch_and_distribute(..., slot_round_of: Callable[[Any], int] | None = None)` and `explain_batch_selection(..., slot_round_of=None)`. `None` keeps today's behavior (`sub.drand_round`).

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_batch_selection.py`; reuse the module's existing submission/cooldown helpers if equivalent ones exist — read the file first):

```python
from dataclasses import dataclass, field


class _NoCooldown:
    def is_in_cooldown(self, prompt_idx, window):
        return False


@dataclass
class _RoundSub:
    hotkey: str
    prompt_idx: int
    drand_round: int
    merkle_root: bytes = field(default=b"\x00" * 32)
    selection_digest: bytes = field(default=b"\x00" * 32)


def test_slot_round_of_overrides_drand_round_grouping():
    """The auction passes a tier ordinal instead of the miner-attached round;
    the split machinery must group by that override."""
    from reliquary.validator.batch_selection import select_batch_and_distribute

    # Same miner-attached drand_round, but the override puts them in
    # different chronological slots -> two slots, one full share each.
    a = _RoundSub(hotkey="a", prompt_idx=1, drand_round=100)
    b = _RoundSub(hotkey="b", prompt_idx=2, drand_round=100)
    tiers = {id(a): 0, id(b): 1}

    batch, rewards = select_batch_and_distribute(
        [a, b], b=2, cooldown_map=_NoCooldown(), current_window=1,
        pool=1.0, slot_round_of=lambda s: tiers[id(s)],
    )
    assert rewards == {"a": 0.5, "b": 0.5}
    assert [s.hotkey for s in batch] == ["a", "b"]


def test_slot_round_of_default_is_drand_round():
    from reliquary.validator.batch_selection import select_batch_and_distribute

    a = _RoundSub(hotkey="a", prompt_idx=1, drand_round=101)
    b = _RoundSub(hotkey="b", prompt_idx=2, drand_round=100)
    batch, _ = select_batch_and_distribute(
        [a, b], b=1, cooldown_map=_NoCooldown(), current_window=1, pool=1.0,
    )
    assert [s.hotkey for s in batch] == ["b"]   # earlier round wins the slot
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_batch_selection.py -x -q -k slot_round_of`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'slot_round_of'`

- [ ] **Step 3: Implement.** In `reliquary/validator/batch_selection.py`:

Add to the signature of `select_batch_and_distribute` (after `pool: float = 1.0`):

```python
    slot_round_of: Callable[[Any], int] | None = None,
```

(import `Callable` from `typing`). Add one docstring line: `slot_round_of: optional override of the chronological slot key; the difficulty auction passes its rank-tier ordinal so the v1 fair-split applies to (score, arrival) tiers.`

At the top of the body:

```python
    round_of = slot_round_of if slot_round_of is not None else (
        lambda sub: sub.drand_round
    )
```

Replace the grouping line `prompts = by_round.setdefault(sub.drand_round, {})` with `prompts = by_round.setdefault(round_of(sub), {})`.

Make the identical change (signature + `round_of` + grouping line at :243) in `explain_batch_selection`.

- [ ] **Step 4: Run the new tests AND the whole file**

Run: `python -m pytest tests/unit/test_batch_selection.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batch_selection.py tests/unit/test_batch_selection.py
git commit -m "feat(auction-v2): parameterize the slot key of the v1 fair-split machinery"
```

---

### Task 2: `_prove_ranked` — rank by (score, arrival), prove full paying tiers

**Files:**
- Modify: `reliquary/validator/batcher.py:3384-3594` (`_prove_ranked`), `batcher.py` `__init__` (near line 593, where `_attempted_pending_ids` is documented), import block at :76
- Test: `tests/unit/test_deferred_proof.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `self._auction_tier_by_id: dict[int, int]` mapping `id(ValidSubmission) -> tier ordinal` for every proven winner (reset each `_prove_ranked` call; `{}` in `__init__`). Candidate rows lose `operator_tiebreak` / `rank_entropy_source` and gain `arrival_drand_round: int`, `arrival_round_source: "arrival"|"submitted_fallback"`, `tier: int`, `tier_size: int`.

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_deferred_proof.py`):

```python
import pytest

from reliquary.constants import B_BATCH


def _telemetry(prompt_idx, hotkey, arrival_round):
    from reliquary.validator.observability import SubmitTelemetry
    return SubmitTelemetry(
        window_n=500, prompt_idx=prompt_idx, hotkey=hotkey,
        merkle_root="00" * 32, protocol_version=2,
        arrival_drand_round=arrival_round,
    )


def _accept(b, req, arrival_round):
    resp = b.accept_submission(
        req, telemetry=_telemetry(req.prompt_idx, req.miner_hotkey, arrival_round)
    )
    assert resp.accepted is True


def _shift_tokens(req, offset):
    """Mark a request's rollouts so a test proof fn can target it."""
    for rollout in req.rollouts:
        tokens = [t + offset for t in rollout.tokens]
        rollout.tokens = tokens
        rollout.commit["tokens"] = tokens
    return req


def test_equal_score_earlier_arrival_ranks_first():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    _accept(b, _request(prompt_idx=1, hotkey="slow"), arrival_round=105)
    _accept(b, _request(prompt_idx=2, hotkey="fast"), arrival_round=103)
    b.seal_batch()

    rows = {r["hotkey"]: r for r in b.auction_candidates}
    assert rows["fast"]["rank"] < rows["slow"]["rank"]
    assert rows["fast"]["tier"] < rows["slow"]["tier"]
    assert rows["fast"]["arrival_round_source"] == "arrival"


def test_score_dominates_speed():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    _accept(b, _request(prompt_idx=1, hotkey="fast-easy",
                        rewards=[1.0] * 6 + [0.0] * 2), arrival_round=100)
    _accept(b, _request(prompt_idx=2, hotkey="slow-hard",
                        rewards=[1.0] * 2 + [0.0] * 6), arrival_round=199)
    b.seal_batch()

    rows = {r["hotkey"]: r for r in b.auction_candidates}
    assert rows["slow-hard"]["rank"] == 1


def test_speed_decides_the_last_slot_between_equal_scores():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for i in range(B_BATCH):
        _accept(b, _request(prompt_idx=i, hotkey=f"fast{i}"), arrival_round=103)
    _accept(b, _request(prompt_idx=99, hotkey="slow"), arrival_round=104)
    _batch, rewards = b.seal_batch()

    assert "slow" not in rewards
    assert b.proof_attempts == B_BATCH        # the losing tier is never proven
    rows = {r["hotkey"]: r for r in b.auction_candidates}
    assert rows["slow"]["status"] == "not_needed"
    assert rows["slow"]["proof_attempted"] is False


def test_no_telemetry_falls_back_to_submitted_round():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    slow = _request(prompt_idx=1, hotkey="slow")
    slow.drand_round = 105
    fast = _request(prompt_idx=2, hotkey="fast")
    fast.drand_round = 103
    assert b.accept_submission(slow).accepted
    assert b.accept_submission(fast).accepted
    b.seal_batch()

    rows = {r["hotkey"]: r for r in b.auction_candidates}
    assert rows["fast"]["rank"] < rows["slow"]["rank"]
    assert rows["fast"]["arrival_round_source"] == "submitted_fallback"


def test_boundary_tier_is_fully_proven_and_marked():
    """2 slots left, boundary tier holds 3 prompts: all 3 proven (they all
    earn), tiers beyond the boundary never proven."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for i in range(B_BATCH - 2):
        _accept(b, _request(prompt_idx=i, hotkey=f"a{i}"), arrival_round=100)
    for j in range(3):
        _accept(b, _request(prompt_idx=10 + j, hotkey=f"b{j}"), arrival_round=101)
    _accept(b, _request(prompt_idx=20, hotkey="late"), arrival_round=102)
    b.seal_batch()

    assert b.proof_attempts == (B_BATCH - 2) + 3
    rows = {r["hotkey"]: r for r in b.auction_candidates}
    assert rows["late"]["proof_attempted"] is False
    for j in range(3):
        assert rows[f"b{j}"]["proof_passed"] is True


def test_prompt_falls_to_next_tier_when_winning_tier_fails():
    from tests.unit.test_grpo_window_batcher import (
        _always_false_grail, _always_true_grail, _make_batcher, _request,
    )

    def _fail_marked(commit, model, randomness):
        if commit["tokens"][0] >= 1000:
            return _always_false_grail(commit, model, randomness)
        return _always_true_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_fail_marked)
    _accept(b, _shift_tokens(_request(prompt_idx=5, hotkey="faker"), 1000),
            arrival_round=100)
    _accept(b, _request(prompt_idx=5, hotkey="honest"), arrival_round=101)
    _batch, rewards = b.seal_batch()

    assert [s.hotkey for s in b.valid_submissions()] == ["honest"]
    assert rewards == {"honest": pytest.approx(1.0 / B_BATCH)}
    assert b.proof_failure_debt("faker") == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_deferred_proof.py -q -k "arrival or dominates or last_slot or falls_to_next or boundary_tier"`
Expected: FAIL — rows have no `"tier"` / `"arrival_round_source"` keys, and `test_prompt_falls_to_next_tier...` fails because today the honest same-prompt candidate is `same_prompt_superseded` only after a pass, but ranking order differs (hash), so the assertion set may partially pass — the row-key assertions are the hard failures.

- [ ] **Step 3: Implement.** In `reliquary/validator/batcher.py`:

3a. Add `_within_slot_key` to the import at :76:

```python
from reliquary.validator.batch_selection import (
    _within_slot_key,
    explain_batch_selection,
    select_batch_and_distribute,
)
```

3b. In `__init__`, next to the `_attempted_pending_ids` initialization (near :593), add:

```python
        # Tier ordinal of each proven winner (id(ValidSubmission) -> int),
        # consumed by _seal_batch_inner as the fair-split slot key.
        self._auction_tier_by_id: dict[int, int] = {}
```

3c. Replace the ranking section of `_prove_ranked` (docstring through the `ranked = sorted(...)` statement, currently :3385-3452) with:

```python
        """Prove candidates in (score, arrival) order until ``B_BATCH`` distinct
        prompts pass. Never prove a candidate that cannot earn.

        Ranking is by ``-value``, then the validator-observed arrival drand
        round: the score gates, submission speed breaks equal-score ties.
        The arrival round is stamped by the validator at HTTP arrival, so a
        miner cannot antedate it; the miner-submitted round is only the
        mock-mode fallback. Candidates exactly equal on BOTH form a tier.
        Every member of a paying tier is proven, because the v1 fair-split
        pays all of them (same prompt: k-way split; boundary tier: equal
        split across its prompts — see ``select_batch_and_distribute``).
        Tiers past the boundary are never proven. The canonical hash only
        orders rows inside a tier for display; it has no economic weight.

        Same-prompt resolution (spec §2.2): a prompt claimed by a PASSING
        submission in an earlier tier supersedes later tiers; a fabricated
        squatter fails the proof and never locks a prompt. Inside one tier,
        same-prompt candidates (necessarily distinct operators) are all
        proven and split the prompt's payout.

        Bounds (spec §2.3) are unchanged: per-hotkey and per-operator
        failure skips, the global attempt ceiling
        (``MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW``) and the proof wall. On
        exhaustion we log, stop, and advance with the shortfall.

        Runs OUTSIDE ``_lock``: ``_verify_expensive`` is 5-25 s of GPU per
        candidate, mutates reject state, and is not thread-safe, so the loop
        is strictly serial.
        """
        with self._lock:
            pending = list(self._pending)
        scored = [
            (p, difficulty_score(p.rewards, delta=DIFFICULTY_AUCTION_DELTA))
            for p in pending
        ]
        operator_by_id: dict[int, str | None] = {}
        arrival_by_id: dict[int, int] = {}
        arrival_source_by_id: dict[int, str] = {}
        for pending_submission, _score in scored:
            operator = self._operator_by_hotkey.get(pending_submission.hotkey)
            if operator is None and not self._operator_mapping_enforced:
                operator = pending_submission.hotkey
            operator_by_id[id(pending_submission)] = operator
            telemetry = getattr(pending_submission, "telemetry", None)
            arrival = getattr(telemetry, "arrival_drand_round", None)
            if arrival is not None:
                arrival_by_id[id(pending_submission)] = int(arrival)
                arrival_source_by_id[id(pending_submission)] = "arrival"
            else:
                # Mock / no-drand mode only: production stamps the arrival
                # round on every admitted request.
                arrival_by_id[id(pending_submission)] = int(
                    pending_submission.drand_round
                )
                arrival_source_by_id[id(pending_submission)] = (
                    "submitted_fallback"
                )
        ranked = sorted(
            scored,
            key=lambda item: (
                -item[1].value,
                arrival_by_id[id(item[0])],
                _within_slot_key(item[0]),
            ),
        )
        # Tier = maximal run of exactly-equal (value, arrival_round).
        tier_by_id: dict[int, int] = {}
        tier_sizes: list[int] = []
        last_tier_key: tuple[float, int] | None = None
        for pending_submission, score in ranked:
            tier_key = (score.value, arrival_by_id[id(pending_submission)])
            if tier_key != last_tier_key:
                tier_sizes.append(0)
                last_tier_key = tier_key
            tier_by_id[id(pending_submission)] = len(tier_sizes) - 1
            tier_sizes[-1] += 1
```

3d. In the row dict (currently :3464-3488), replace the two retired fields

```python
                "operator_tiebreak": tiebreak_by_id[
                    id(pending_submission)
                ].hex(),
                "rank_entropy_source": rank_entropy_source,
```

with:

```python
                "arrival_drand_round": arrival_by_id[id(pending_submission)],
                "arrival_round_source": arrival_source_by_id[
                    id(pending_submission)
                ],
                "tier": tier_by_id[id(pending_submission)],
                "tier_size": tier_sizes[
                    tier_by_id[id(pending_submission)]
                ],
```

3e. Replace the proof loop (currently `for (p, _score), row in zip(ranked, candidate_rows):` through the `proven.append(...)` block, :3496-3573) with a tier-aware loop. Full replacement:

```python
        self._auction_tier_by_id = {}
        current_tier = -1
        claimed_before_tier: set[int] = set()
        for (p, _score), row in zip(ranked, candidate_rows):
            tier = tier_by_id[id(p)]
            if tier != current_tier:
                # Tier boundary: stop BEFORE a tier that cannot earn; the
                # tier that crosses B_BATCH is proven in full (fair-split
                # pays every one of its prompts).
                if len(claimed) >= B_BATCH:
                    stop_reason = "batch_filled"
                    break
                current_tier = tier
                claimed_before_tier = set(claimed)
            if p.prompt_idx in claimed_before_tier:
                row["status"] = "same_prompt_superseded"
                continue          # an earlier tier already won this prompt
            if self._cooldown.is_in_cooldown(p.prompt_idx, self.window_start):
                row["status"] = "cooldown"
                continue
            operator = row["operator_id"]
            if operator is None:
                row["status"] = "operator_unmapped"
                self.auction_operator_unmapped_skips += 1
                continue
            if (
                self.operator_proof_failure_debt(operator)
                >= MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW
            ):
                row["status"] = "operator_proof_debt"
                self.auction_operator_proof_debt_skips += 1
                continue
            # Global proof budget: proving cannot exceed the graded-pool ceiling
            # (v2 §2.3). This bounds a multi-hotkey fabricated flood that the
            # per-hotkey skip below cannot, since each fake hotkey pays only one
            # registration. On exhaustion we stop and advance short.
            if attempts >= MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW:
                logger.warning(
                    "proof budget exhausted window=%d attempts=%d proven=%d "
                    "pending=%d — advancing with shortfall",
                    self.window_start, attempts, len(proven), len(pending),
                )
                stop_reason = "attempt_budget"
                break
            elapsed = self._time_fn() - self._proof_wall_started_at
            if elapsed >= MAX_PROOF_WALL_SECONDS:
                self.proof_wall_exhausted = True
                stop_reason = "wall_budget"
                logger.warning(
                    "proof wall budget exhausted window=%d elapsed_s=%.2f "
                    "attempts=%d proven=%d pending=%d — advancing with shortfall",
                    self.window_start,
                    elapsed,
                    attempts,
                    len(proven),
                    len(pending),
                )
                break
            # Per-hotkey griefer bound. A fabricated group ranks at the top by
            # construction and fails the proof; each hotkey is skipped after its
            # failure cap so honest fill below the fakes always proceeds.
            if (
                self.proof_failure_debt(p.hotkey)
                >= MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW
            ):
                row["status"] = "hotkey_proof_debt"
                continue
            attempts += 1
            attempted_ids.add(id(p))
            row["proof_attempted"] = True
            row["status"] = "proof_started"
            sub = self._verify_expensive(p)
            if sub is None:
                self._expensive_proof_failures_by_operator[operator] = (
                    self._expensive_proof_failures_by_operator.get(
                        operator, 0
                    )
                    + 1
                )
                row["proof_passed"] = False
                row["status"] = "proof_failed"
                continue          # rejected; promote the next-ranked for prompt
            row["proof_passed"] = True
            row["selected"] = True
            row["status"] = "selected"
            proven.append(sub)
            claimed.add(p.prompt_idx)
            self._auction_tier_by_id[id(sub)] = tier
            self.difficulty_auction_metadata_by_id[id(sub)] = row
```

Note the two behavior deltas from the old loop, both intentional: the `len(proven) >= B_BATCH` per-candidate break becomes a `len(claimed) >= B_BATCH` check at tier boundaries only, and `same_prompt_superseded` keys off `claimed_before_tier` (earlier-tier claims), not `claimed`, so same-tier same-prompt candidates are all proven.

Everything after the loop (`with self._lock: self._valid = proven ...` through `return proven`) is unchanged.

- [ ] **Step 4: Run the new tests, then the file**

Run: `python -m pytest tests/unit/test_deferred_proof.py -q`
Expected: new tests PASS. `test_proving_stops_once_b_submissions_pass` now FAILS (12 equal-score no-telemetry candidates form ONE tier → all 12 proven). Fix it in Step 5.

- [ ] **Step 5: Repair `test_proving_stops_once_b_submissions_pass`** — stagger arrivals so the GPU-saving claim is still tested, and document the tier rule:

```python
def test_proving_stops_once_b_submissions_pass():
    """The GPU saving. Distinct arrival tiers: we must not prove candidate 9
    when 8 earlier-tier candidates have already passed. (Candidates in ONE
    tier are all proven — the fair-split pays them all — so this test gives
    each candidate its own arrival round.)"""
    from reliquary.constants import B_BATCH, M_ROLLOUTS
    from reliquary.validator.observability import SubmitTelemetry
    from tests.unit.test_grpo_window_batcher import (
        _always_true_grail, _make_batcher, _request,
    )

    proofs = []

    def _counting_proof(commit, model, randomness):
        proofs.append(1)
        return _always_true_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_counting_proof)
    for i in range(12):
        b.accept_submission(
            _request(prompt_idx=i, hotkey=f"m{i}"),
            telemetry=SubmitTelemetry(
                window_n=500, prompt_idx=i, hotkey=f"m{i}",
                merkle_root="00" * 32, protocol_version=2,
                arrival_drand_round=100 + i,
            ),
        )

    b.seal_batch()

    assert len(b.valid_submissions()) == B_BATCH
    assert b.proof_attempts == B_BATCH                     # 8 tiers, NOT 12
    assert len(proofs) == B_BATCH * M_ROLLOUTS
```

- [ ] **Step 6: Run the file again**

Run: `python -m pytest tests/unit/test_deferred_proof.py -q`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_deferred_proof.py
git commit -m "feat(auction-v2): rank by score then arrival round; prove full paying tiers"
```

---

### Task 3: fair-split payout wiring in `_seal_batch_inner`

**Files:**
- Modify: `reliquary/validator/batcher.py` `_seal_batch_inner` (:3884-3910, the two calls to `explain_batch_selection` / `select_batch_and_distribute`)
- Test: `tests/unit/test_deferred_proof.py` (append)

**Interfaces:**
- Consumes: `self._auction_tier_by_id` (Task 2), `slot_round_of` (Task 1).

- [ ] **Step 1: Write the failing tests** (append; `_telemetry`, `_accept`, `_shift_tokens` come from Task 2):

```python
def test_same_prompt_same_tier_splits_the_prompt_share():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for hk in ("op-a", "op-b"):
        _accept(b, _request(prompt_idx=7, hotkey=hk), arrival_round=100)
    batch, rewards = b.seal_batch()

    share = 1.0 / B_BATCH
    assert rewards["op-a"] == pytest.approx(share / 2)
    assert rewards["op-b"] == pytest.approx(share / 2)
    assert len(b.valid_submissions()) == 2   # both proven, both paid
    assert len(batch) == 1                   # one training representative


def test_same_prompt_split_survivor_takes_full_share():
    from tests.unit.test_grpo_window_batcher import (
        _always_false_grail, _always_true_grail, _make_batcher, _request,
    )

    def _fail_marked(commit, model, randomness):
        if commit["tokens"][0] >= 1000:
            return _always_false_grail(commit, model, randomness)
        return _always_true_grail(commit, model, randomness)

    b = _make_batcher(verify_commitment_proofs_fn=_fail_marked)
    _accept(b, _request(prompt_idx=7, hotkey="honest"), arrival_round=100)
    _accept(b, _shift_tokens(_request(prompt_idx=7, hotkey="cheat"), 1000),
            arrival_round=100)
    _batch, rewards = b.seal_batch()

    assert rewards == {"honest": pytest.approx(1.0 / B_BATCH)}


def test_boundary_tier_fair_split_payout():
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    b = _make_batcher()
    for i in range(B_BATCH - 2):
        _accept(b, _request(prompt_idx=i, hotkey=f"a{i}"), arrival_round=100)
    for j in range(3):
        _accept(b, _request(prompt_idx=10 + j, hotkey=f"b{j}"), arrival_round=101)
    batch, rewards = b.seal_batch()

    share = 1.0 / B_BATCH
    for i in range(B_BATCH - 2):
        assert rewards[f"a{i}"] == pytest.approx(share)
    for j in range(3):
        assert rewards[f"b{j}"] == pytest.approx(2 * share / 3)
    assert sum(rewards.values()) == pytest.approx(1.0)
    assert len(batch) == B_BATCH
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_deferred_proof.py -q -k "splits_the_prompt or survivor or fair_split_payout"`
Expected: `splits_the_prompt` FAILS (today the two same-tier submissions carry miner `drand_round` 0 for both → grouped in one v1 slot only if rounds match; the hard failure is `boundary_tier_fair_split_payout`, where miner rounds are all 0 → ONE v1 round → 9-way boundary split instead of 6 full + 3 boundary shares).

- [ ] **Step 3: Implement.** In `_seal_batch_inner`, before the `explain_batch_selection` call, add:

```python
            # Auction mode replaces the chronological slot key with the rank
            # tier: the v1 machinery then pays full tiers a slot each and
            # fair-splits the boundary tier. `.get(..., 0)` collapses to one
            # tier for test doubles that bypass _prove_ranked.
            slot_round_of = None
            if self.difficulty_auction_enabled:
                tier_of = self._auction_tier_by_id
                slot_round_of = lambda sub: tier_of.get(id(sub), 0)
```

and pass `slot_round_of=slot_round_of` to BOTH `explain_batch_selection(...)` and `select_batch_and_distribute(...)`.

- [ ] **Step 4: Run**

Run: `python -m pytest tests/unit/test_deferred_proof.py tests/unit/test_grpo_window_batcher.py -q`
Expected: all PASS (the legacy `_prove_all_pending` seal tests keep working through the `.get(..., 0)` single-tier collapse).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/batcher.py tests/unit/test_deferred_proof.py
git commit -m "feat(auction-v2): pay (score, arrival) tiers through the v1 fair-split"
```

---

### Task 4: align the observation-only `_rank_key`

**Files:**
- Modify: `reliquary/validator/difficulty_auction.py:19-31` (`ShadowSubmission`), `:102-110` (`_rank_key`)
- Test: `tests/unit/test_difficulty_auction.py` (append)

**Interfaces:**
- Produces: `ShadowSubmission.arrival_drand_round: int | None = None`; `_rank_key` returns `(-value, arrival_or_submitted_round, canonical_hash)`.

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_difficulty_auction.py`, reusing that file's existing `ShadowSubmission` construction helper if present — read the file first):

```python
def test_rank_key_breaks_score_ties_by_arrival_round():
    from reliquary.validator.difficulty_auction import (
        ShadowSubmission, difficulty_score, _rank_key,
    )

    def _sub(source_id, hotkey, arrival):
        return ShadowSubmission(
            source_id=source_id, hotkey=hotkey, prompt_idx=source_id,
            drand_round=999, merkle_root=b"\x00" * 32,
            selection_digest=hotkey.encode().ljust(32, b"\x00"),
            rewards=(1.0, 1.0) + (0.0,) * 6,
            arrival_drand_round=arrival,
        )

    slow = _sub(1, "slow", 105)
    fast = _sub(2, "fast", 103)
    ranked = sorted(
        ((s, difficulty_score(s.rewards, delta=1.0)) for s in (slow, fast)),
        key=_rank_key,
    )
    assert [s.hotkey for s, _ in ranked] == ["fast", "slow"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_difficulty_auction.py -q -k arrival`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'arrival_drand_round'`

- [ ] **Step 3: Implement.** In `difficulty_auction.py`, add to `ShadowSubmission` (after `in_cooldown`):

```python
    arrival_drand_round: int | None = None
```

Replace `_rank_key` with:

```python
def _rank_key(
    item: tuple[ShadowSubmission, DifficultyScore],
) -> tuple[float, int, bytes]:
    """Mirror of the production ranking: score gates, validator-observed
    arrival breaks ties (miner-submitted round is the fallback), canonical
    hash orders within a tier for display only."""
    submission, score = item
    arrival = getattr(submission, "arrival_drand_round", None)
    chronological = (
        arrival if arrival is not None else submission.drand_round
    )
    return (
        -score.value,
        int(chronological),
        _within_slot_key(submission),
    )
```

(The `getattr` keeps `test_pending_submission_ranks_in_the_auction` working: `PendingSubmission` has no `arrival_drand_round` attribute.)

- [ ] **Step 4: Run**

Run: `python -m pytest tests/unit/test_difficulty_auction.py tests/unit/test_difficulty_auction_shadow.py tests/unit/test_difficulty_auction_report.py tests/unit/test_deferred_proof.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/difficulty_auction.py tests/unit/test_difficulty_auction.py
git commit -m "feat(auction-v2): align the observation _rank_key with production speed ranking"
```

---

### Task 5: archive/telemetry surfaces and consumer repairs

**Files:**
- Modify: `reliquary/validator/service.py:2173-2175` (archive mapping), `service.py:1637-1641` (seal-randomness comment)
- Modify: `tests/unit/test_archive_window_content.py:160,255-284`
- Modify: `tests/unit/test_auction_resource_guards.py:223-250`

**Interfaces:**
- Produces: archive entries expose `difficulty_auction_arrival_drand_round`, `difficulty_auction_arrival_round_source`, `difficulty_auction_tier`, `difficulty_auction_tier_size`; `difficulty_auction_rank_entropy_source` is removed.

- [ ] **Step 1: Update the archive mapping.** In `service.py`, replace:

```python
                "difficulty_auction_rank_entropy_source": difficulty_meta.get(
                    "rank_entropy_source"
                ),
```

with:

```python
                "difficulty_auction_arrival_drand_round": difficulty_meta.get(
                    "arrival_drand_round"
                ),
                "difficulty_auction_arrival_round_source": difficulty_meta.get(
                    "arrival_round_source"
                ),
                "difficulty_auction_tier": difficulty_meta.get("tier"),
                "difficulty_auction_tier_size": difficulty_meta.get(
                    "tier_size"
                ),
```

- [ ] **Step 2: Update the seal-randomness comment** at `service.py:1637-1641` to reflect that the beacon now keys ONLY the forensic sample:

```python
        # Fetch a fresh drand beacon now — AFTER the collection deadline — to key
        # each batcher's forensic sample. Its randomness did not exist when miners
        # submitted, so the sample cannot be ground in advance. Ranking no longer
        # consumes it: equal scores are ordered by validator-observed arrival.
        # If the fetch fails, the forensic sample is disabled for the window.
```

- [ ] **Step 3: Repair `test_archive_window_content.py`.** At :160 replace the fixture row fields `"rank_entropy_source": "seal_drand",` with `"arrival_drand_round": 103, "arrival_round_source": "arrival", "tier": 0, "tier_size": 1,`. At :261 replace

```python
    assert entry0["difficulty_auction_rank_entropy_source"] == "seal_drand"
```

with:

```python
    assert entry0["difficulty_auction_arrival_drand_round"] == 103
    assert entry0["difficulty_auction_arrival_round_source"] == "arrival"
    assert entry0["difficulty_auction_tier"] == 0
    assert entry0["difficulty_auction_tier_size"] == 1
```

(Adapt surrounding fixture expectations as the file's structure requires — read the test before editing.)

- [ ] **Step 4: Replace `test_operator_tiebreak_does_not_change_when_hotkey_changes`** in `test_auction_resource_guards.py` with the invariant that survives the redesign — payouts are operator-stable under hotkey renames (both same-prompt candidates are now proven and split):

```python
def test_same_prompt_tie_payout_is_stable_under_hotkey_rename():
    """Equal-score, equal-arrival candidates on one prompt split its share;
    renaming an operator's hotkey must not change any operator's payout."""
    def seal_with(operator_a_hotkey):
        mapping = {
            operator_a_hotkey: "operator-a",
            "operator-b-hotkey": "operator-b",
        }
        batcher = _batcher(operator_by_hotkey=mapping)
        for hotkey in mapping:
            assert batcher.accept_submission(
                _request(prompt_idx=7, hotkey=hotkey)
            ).accepted
        _batch, rewards = batcher.seal_batch()
        return {mapping[hk]: amount for hk, amount in rewards.items()}

    first = seal_with("operator-a-hotkey-1")
    second = seal_with("operator-a-hotkey-999")

    assert first == second
    assert first["operator-a"] == first["operator-b"]
```

- [ ] **Step 5: Run the touched files plus a grep for stragglers**

Run: `python -m pytest tests/unit/test_archive_window_content.py tests/unit/test_auction_resource_guards.py tests/unit/test_verdicts_endpoint.py tests/unit/test_state_machine.py -q`
Expected: all PASS
Run: `grep -rn "rank_entropy_source\|operator_tiebreak\|seal_drand" reliquary/ tests/ --include=*.py`
Expected: no hits left outside `.r2_analysis/` (fix any found).

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/service.py tests/unit/test_archive_window_content.py tests/unit/test_auction_resource_guards.py
git commit -m "feat(auction-v2): surface arrival/tier in archives; retire rank entropy fields"
```

---

### Task 6: full-suite verification

- [ ] **Step 1: Run the entire unit suite**

Run: `python -m pytest tests/unit -q`
Expected: all PASS. Triage any failure to its owning task's code (most likely candidates: other tests asserting `proof_attempts`, `same_prompt_superseded`, or auction candidate row keys).

- [ ] **Step 2: Run any integration suite present**

Run: `ls tests/ && python -m pytest tests -q --ignore=tests/unit -x` (skip if only `unit/` exists or if it requires GPU/network — note what was skipped).

- [ ] **Step 3: Final commit if repairs were needed**

```bash
git add -A tests/ reliquary/
git commit -m "test(auction-v2): repair suite for speed-ranked fair-split auction"
```
