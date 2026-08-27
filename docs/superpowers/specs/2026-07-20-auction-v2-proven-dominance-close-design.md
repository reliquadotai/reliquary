# Auction v2 — proven-dominance early close (design)

Date: 2026-07-20
Status: approved direction (conversation 2026-07-20); spec pending owner review
Builds on: `2026-07-20-auction-v2-speed-rank-design.md` (merged, PR #156)

## Problem

Auction windows collect for the full `WINDOW_COLLECTION_SECONDS` (300 s) even
when the outcome is already decided. In practice every window saturates at the
theoretical maximum of `difficulty_score` well before the deadline: once
`B_BATCH` (8) distinct prompts are covered by candidates at that maximum, no
future submission can win anything — any later arrival has `value ≤ v_max`
and a strictly later `arrival_round`, hence a strictly worse rank key. The
remaining admission tail is dead time for the validator (window cadence) and
for miners (GPU burned on a decided window).

A naive early seal at saturation was rejected because admission is cheap (no
GPU proof): a forger could trigger it with pre-generated rollouts that grade
at `v_max` but would fail GRAIL, shortening the window for honest miners.
The owner's design instead **proves the winners while the window is still
open** and closes only when dominance is *proven*.

## Mechanism

### v_max

`difficulty_score(rewards, delta) = std(rewards) · (1 − mean(rewards))^delta`
over the per-rollout rewards in `[0, 1]`. For a fixed mean `p`, `std` is
maximised only by extremal (all-0/1) reward vectors, where `std = √(p(1−p))`;
therefore the global maximum over the reward domain is attained only on
binary profiles. Define, at batcher construction:

```
V_MAX(n) = max over k in 0..n of
           difficulty_score(k ones + (n−k) zeros, delta=DIFFICULTY_AUCTION_DELTA).value
```

with `n` = the environment's rollouts-per-submission (8 today → peak at
k=2, value ≈ 0.32476). `V_MAX` is compared with exact float equality
against `score.value`, computed through the same code path
(`difficulty_score`), so no epsilon is involved — identical inputs give
identical floats, and tier formation already relies on exact value equality.

### Dominance tracking

The prover derives its work from the pending pool on each poll (the pool is
small and candidates carry a precomputed `value`, so no admission-hot-path
tracking is needed): the candidates whose `value == V_MAX` form the v_max
pool, ranked by `(arrival_round, canonical hash)`.

Implementation note (deviation from the first draft of this spec, deliberate):
the prover does **not** wait for accepted coverage ≥ `B_BATCH` before proving —
it proves any unproven member of the *currently paying* `V_MAX` tiers as soon
as one exists. The safety argument is identical (such a candidate is proven by
`_prove_ranked` at every possible seal, so the proof is never wasted), and the
proofs overlap the fill phase, so the close fires sooner. The dominance-armed
notion survives only as telemetry (`early_close_armed_round`, the round at
which the close condition first held).

### Mid-window prover

A single background thread per armed batcher proves candidates from the
v_max pool in rank order (`arrival_round`, then canonical hash), respecting
exactly the existing `_prove_ranked` skip/budget rules (per-hotkey and
per-operator failure debt, `MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW`,
`MAX_PROOF_WALL_SECONDS`, `MAX_SUBMISSIONS_PER_PROMPT`) against the same
window-scoped counters — mid-window attempts and post-seal attempts draw
from one budget.

Scope rule ("never prove a loser", adapted): only members of v_max tiers
that are inside the current winning coverage are proven — i.e. the tiers
that would pay if the window sealed now. A v_max candidate beyond the
boundary (e.g. the 9th distinct prompt) is not proven unless a failure
promotes its tier into coverage.

Proving a covered v_max candidate is never wasted work: its rank can only
improve (rivals failing proof), never degrade (any future arrival ranks
strictly below it). If the window later seals on the deadline instead, the
proof is already cached and `_prove_ranked` reuses it.

Results land in a per-window **proof cache**: submission id → terminal
status (passed / failed / skipped-debt). `_prove_ranked` consults the cache
before attempting any proof, in both the early-close and deadline paths. A
failed proof applies failure debt exactly as today, removes the candidate,
and shrinks the v_max pool — possibly disarming the prover until new v_max
arrivals (or promoted tiers) restore coverage; then it resumes and proves
only the new members. No candidate is ever proven twice.

Concurrency: `_verify_expensive` stays strictly serial (one prover thread;
the post-seal `_prove_ranked` loop reuses the same discipline — the thread
is joined before `seal_batch` runs). State mutations (debt counters, reject
state, cache) happen under `_lock`; the GPU call runs outside it. The GPU
is idle during collection (the loop is sequential: collect → seal → train),
so mid-window proving contends with nothing.

### Close condition

The batcher requests an early close when ALL of:

1. distinct prompts covered by **proven** v_max candidates ≥ `B_BATCH`;
2. every member of the winning v_max tiers has a terminal proof status
   (the boundary tier's fair-split composition is fully resolved);
3. the current drand round is strictly greater than the maximum
   `arrival_round` across those winning tiers (a same-round arrival could
   still join the boundary tier — 3 s round granularity).

Condition 3 is nearly always implied (each proof takes 5–25 s ≫ 3 s) but is
kept as an explicit guard for the cached-promotion edge case.

### Close sequence — reuse the deadline path verbatim

On close request the batcher seals with reason `proven_dominance_close`
(same effect as `poll_deadline` firing: `is_sealed()` flips, HTTP admission
cheap-rejects new arrivals with `BATCH_FILLED` — a reject miners already
handle). The service's `_wait_for_window_seal` loop observes the seal on
its next poll; everything downstream is unchanged:

- `_freeze_auction_populations` drains the submit queue, in-flight workers
  and pending upload precommits (drain deadline 390 s, receipt conservation
  check, abort semantics) — untouched;
- `seal_batch` → `_prove_ranked` re-ranks the **full drained population**
  and proves any not-yet-proven member of paying tiers via the cache.

This drain-then-reprove structure is what makes the close safe against
in-flight stragglers: a submission that arrived over HTTP (or reserved an
upload precommit, whose drand observation is stamped at *precommit*
arrival) before the close carries an early `arrival_round` and may still
join a winning tier — it is admitted during the drain and proven post-seal
like today. Only arrivals *after* the close are rejected, and those are
provable losers: their round is strictly later than the proven coverage
(condition 3). Expected steady state: zero post-seal proofs, seal completes
in cache-lookup time.

The deadline stays as the unchanged upper bound: if proven dominance is
never reached, `poll_deadline` seals at 300 s and the window proceeds
exactly as today (with whatever proofs the mid-window prover banked).

### What this does NOT change

- Ranking, tiers, fair-split, promotion, burn-not-redistribute, forensic
  sampling, cooldown, emission pool math: untouched. `seal_batch` does not
  know it was called early.
- No reopening, ever: a proof failure after close follows today's
  promotion-then-burn semantics. (In practice the early-close path cannot
  hit it — it closes only with every paying member already proven; the
  deadline path keeps its existing exposure.)
- Wire compatibility: no new response fields; post-close rejects reuse
  `BATCH_FILLED`. Window duration becomes variable, which miner clients
  already handle (any env can be force-sealed today).
- Multi-env: the feature is per-environment. The window advances when all
  active envs seal (`_wait_for_window_seal`), so the wall-clock gain is
  `max` over envs of their close times. Both auction envs (math and code)
  get the feature; a saturating env never cuts a slower env short.

## Incentive analysis

- **No new speed race.** The close only fires when no live or future
  candidate can win anything; every arrival denied by the close is a
  proven loser under the unchanged rank key.
- **Forgery cannot shorten the window.** Closing requires *proofs passed*;
  pre-generated `v_max` bait fails GRAIL mid-window, eats failure debt, and
  the window simply continues — the auction re-fills and the prover
  resumes. This is strictly better than a grace-period seal, which a
  forger could still trigger.
- **Slow-but-good miners keep their protection.** The window never closes
  while a higher-value submission could still change the outcome — that is
  the definition of the close condition.
- **Post-saturation submissions lose their promotion-lottery value** only
  in the early-close path, where promotion can no longer occur (everything
  paying is proven). No economic loss to honest miners.

## Config

- `RELIQUARY_AUCTION_EARLY_CLOSE_ENFORCE` — kill switch, code default ON
  (matching `DIFFICULTY_AUCTION_ENFORCE` precedent; the durable default
  lives in `constants.py`, not in a `.env` override — see the
  drand-tolerance regression for why). OFF restores today's behaviour
  bit-for-bit: no tracking, no prover thread, deadline-only seal.
- No new tunable beyond the switch: no grace period (the proof duration
  plays that role structurally), thresholds are `B_BATCH` and `V_MAX`,
  both derived.

## Telemetry / archive

- `force_seal_reason` / seal reason: `proven_dominance_close`.
- Auction shadow dict gains: `early_close_armed_round`
  (`dominance_observed_round`), `early_close_sealed_round`,
  `midwindow_proof_attempts`, `midwindow_proof_passes`,
  `midwindow_proof_failures`, `early_close` (bool).
- Candidate rows: `proof_phase` (`"midwindow"` | `"post_seal"`) on
  attempted proofs.
- These make the forgery-bait pattern (arm → mass mid-window failures →
  no close) a trivial archive query.

## Implementation shape

- `constants.py`: `AUCTION_EARLY_CLOSE_ENFORCE` (+ env var), no other knobs.
- `difficulty_auction.py`: `max_difficulty_value(n_rollouts, delta)` helper
  (+ property test pinning it to `difficulty_score` over all binary
  profiles and asserting no fractional profile exceeds it).
- `batcher.py`:
  - v_max pool tracking in `_accept_locked` (and removal paths);
  - prover thread lifecycle (arm/disarm/join), proof cache, shared budget
    counters; `_prove_ranked` consults the cache;
  - close-condition check + `force_seal("proven_dominance_close")`;
  - `poll_deadline` untouched.
- `service.py`: no structural change (`_wait_for_window_seal` already
  polls `is_sealed()` generically); only telemetry plumbing.

## Tests (TDD, written first)

1. Saturation with 8 distinct proven v_max prompts → window seals early;
   seal output identical to the deadline seal of the same population.
2. One of the 8 fails mid-window proof → no close; a later v_max arrival
   on a new prompt restores coverage → only the new member is proven →
   close. Failure debt identical to the post-seal path.
3. Forger bait: 8 pre-gen v_max candidates all fail proof → window stays
   open to the deadline; honest candidates admitted before/after the bait
   win under normal deadline seal; debt applied.
4. Same-round guard: an arrival in `dominance_observed_round` at v_max on
   a new prompt joins the boundary tier; close waits for the next round
   and includes it (condition 3).
5. Straggler in drain: a pre-close HTTP/precommit submission with an early
   arrival round is admitted during drain, joins a winning tier, is proven
   post-seal from the shared budget; totals conserved.
6. Deadline fallback: coverage never proven → seal at 300 s; mid-window
   proof cache reused by `_prove_ranked` (no double proof).
7. Beyond-boundary v_max candidate (9th distinct prompt) is not proven
   unless promoted ("never prove a loser").
8. Budgets span the window: mid-window attempts + post-seal attempts ≤
   `MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW`; wall-clock budget likewise.
9. Kill switch OFF → byte-identical behaviour to today (no thread, no
   tracking, deadline seal).
10. `V_MAX` property test (see above).
11. Non-auction/legacy mode: completely unaffected.

## Non-goals

- No change to the legacy (non-auction) seal-extension path.
- No multi-validator arrival consensus (unchanged single-validator
  assumption).
- No redesign of the discrete score scale (the fact that every window
  saturates at `V_MAX` means the score no longer discriminates at the top
  and the effective contest is arrival speed among `V_MAX` submissions —
  a real design smell, but explicitly out of scope here).
