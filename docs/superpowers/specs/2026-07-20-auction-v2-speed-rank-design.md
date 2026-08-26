# Auction v2 — score-ranked, speed-tiebroken, fair-split (design)

Date: 2026-07-20
Branch: `feat/auction-v2-speed-rank` (off `main` @ `21cb2b8`)
Status: approved by owner (conversation 2026-07-20)

## Problem

The shipped auction ranks candidates by `(-difficulty_score, drand-salted operator
hash)`. Equal-score candidates — the common case, since scores are discrete
(k/8) and k=2 is the value peak — are decided by a post-deadline drand lottery.
The owner's intended v2 design is: **score first, speed second**. Submission
speed (earliest validator-observed drand arrival round) must decide equal-score
ties, and exact `(score, arrival_round)` draws must **fair-split** the prompt's
emission (the v1 `(round, prompt)` slot-split semantics), not be raffled.

## Mechanism (approved)

Window flow is unchanged: 300 s of cheap admission (grade + filters, GPU proof
deferred), then seal.

At seal, in each auction environment:

1. **Rank** all pending candidates by the key `(-value, arrival_round)` where
   - `value` = `difficulty_score(rewards, delta)` (unchanged), and
   - `arrival_round` = `telemetry.arrival_drand_round`, the drand round the
     validator computed from its own wall clock at HTTP arrival. Miners cannot
     antedate it. Fallback when absent (mock / no-drand test mode only; in
     production the server always stamps it): the miner-submitted
     `drand_round`.
2. **Tiers**: candidates with exactly equal `(value, arrival_round)` form one
   tier. Tiers are processed best-first.
3. **Slots** (`B_BATCH` per env): filled tier by tier with distinct prompts.
   - A tier that fits entirely: each of its prompts takes a full slot
     (`slot_share = pool / B_BATCH`).
   - The **boundary tier** (the tier that reaches or crosses `B_BATCH`
     cumulative distinct prompts): the remaining `r` slots' emission
     (`r × slot_share`) is split **equally across all N prompts of that tier**
     (v1 boundary fair-split). Which `r` of those prompts enter the training
     batch is picked by the canonical unsalted hash — an implementation detail
     with no economic weight, since payout is equal.
   - Tiers below the boundary earn nothing and are **never proven**.
4. **Same-prompt draws**: if several candidates (necessarily distinct
   operators — per-(operator, prompt) logical dedup is already enforced at
   admission) share the same prompt inside the prompt's winning tier, every
   proven one splits that prompt's payout equally (v1 within-slot split). One
   representative (canonical hash) enters the training batch.
5. **Deferred proof**: top-down as today, with the same budgets
   (`MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW`, `MAX_PROOF_WALL_SECONDS`,
   per-hotkey and per-operator failure debt). Differences:
   - every member of a **paying** tier must be proven (they all earn), bounded
     by `MAX_SUBMISSIONS_PER_PROMPT` per prompt and the global budgets;
   - a candidate whose proof fails drops out of its split (its share
     redistributes among the prompt's surviving proven members: `k` = proven
     count, exactly v1's `slot_share / k`);
   - if **all** candidates for a prompt's winning tier fail, the prompt falls
     through to its next tier (promotion, as today);
   - on budget exhaustion we advance with shortfall; unproven candidates are
     not paid; unfilled slots burn (unchanged).
6. **Randomness**: `seal_randomness` (post-deadline drand beacon) is no longer
   a ranking input. It remains the key of the forensic sample. The salted
   `operator_tiebreak` hash is removed from ranking and from the archive rows.

## Implementation shape

- `batcher.py::_prove_ranked` — rank by `(-value, arrival_round)`; group into
  tiers; prove every member of paying tiers (same-prompt members included)
  instead of skipping `same_prompt_superseded` within a paying tier; stop
  after the boundary tier. Store a per-proven-submission tier ordinal
  (`self._auction_tier_by_id: dict[int, int]`).
- `batch_selection.py::select_batch_and_distribute` and
  `explain_batch_selection` — add an optional `slot_round_of:
  Callable[[sub], int]` (default: `sub.drand_round`, current behavior). In
  auction mode the batcher passes the **tier ordinal**. This reuses the
  battle-tested v1 split machinery verbatim: full-round slots, k-way
  within-slot split, boundary fair-split, canonical-hash representative,
  burn-not-redistribute.
- `batcher.py::_seal_batch_inner` — pass `slot_round_of` when
  `difficulty_auction_enabled`.
- Archive candidate rows: drop `operator_tiebreak` and `rank_entropy_source`;
  add `arrival_drand_round`, `arrival_round_source` (`"arrival"` |
  `"submitted_fallback"`), `tier` (ordinal), `tier_size`. `rank` stays (1-based
  position in the ranked order; ties ordered by canonical hash for display
  stability only).
- `difficulty_auction.py::_rank_key` (observation-only module) — align to
  `(-value, arrival_round, canonical hash)` so the shadow model matches
  production. `ShadowSubmission` gains `arrival_drand_round: int | None`.

## Explicitly accepted trade-offs

- **Latency race returns** between 3 s drand rounds: being one round earlier
  now deterministically beats an equal-score rival. Owner's explicit choice —
  speed is meant to be the second-order incentive.
- **Single-validator assumption**: `arrival_drand_round` is local to the
  observing validator. Fine today (one validator); a multi-validator future
  needs a consensus arrival measure. Documented, out of scope.
- **More proofs per window** than strict top-8: paying ties are proven in
  full. Bounded by existing budgets; shortfall semantics unchanged.

## Non-goals

- No change to admission filters, grading, cooldown, forensic sampling,
  emission pool math (`pool / len(env_mix)`), or the burn rule.
- No change to legacy (non-auction) mode.
- No multi-validator arrival consensus.

## Tests

New (TDD, written first):
1. Equal score, different arrival rounds → earlier round wins the slot.
2. Score dominates: slower higher-score candidate beats faster lower-score.
3. Same prompt, same tier, two operators, both prove → payout split 50/50,
   one training representative.
4. Same prompt, same tier, one of two fails proof → survivor takes the full
   prompt share.
5. Boundary tier: 2 slots left, tier holds 3 prompts → each prompt paid
   `2 × slot_share / 3`; 2 canonical-hash representatives train; total window
   payout equals `filled_slots / B × pool`.
6. Tier below boundary is never proven (`never prove a loser` preserved).
7. All winning-tier members for a prompt fail proof → next-tier candidate for
   the prompt is promoted.
8. Fallback: telemetry without `arrival_drand_round` ranks by submitted
   `drand_round`.

Repairs: `test_deferred_proof.py` (rank-key shape), `test_auction_resource_guards.py`
(tiebreak-hash tests replaced by arrival-order tests), `test_archive_window_content.py`
(`rank_entropy_source` assertions → new fields), `test_difficulty_auction*.py`
(shadow `_rank_key`).
