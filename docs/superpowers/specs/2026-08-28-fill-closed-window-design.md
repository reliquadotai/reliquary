# Fill-closed window: validate on arrival, pay per token — Design

Date: 2026-08-28
Status: draft for review
Scope: validator window lifecycle, admission, proof scheduling, emission,
and the reward split. No miner protocol break: the precommit endpoint and
the miner loop already support everything this needs.
Baseline: main @ c0b01d1.
Supersedes: `2026-08-24-macro-window-dapo-batch-design.md` (same goal, but
it kept the auction, a time-based close, and per-slot payment).

## Problem

One window == one DAPO optimizer step, and the window is floored by a
clock. 16 steps — one pi_old interval — cost ~1632 s, so the run does
~740 optimizer steps/day against the ~5000 of a reference DAPO run. The
measured lever on this run is updates/day.

The clock is not what the fleet needs. PR #212 measured it on the live
validator (2026-08-26): **both environments charge their 64 productive
slots at +19 s to +45 s**, after which the window spends a median 79 s of
its 102 s cycle rejecting everything with `batch_filled`. Three quarters
of every cycle serve nobody, and the fleet's true offer rate is higher
still — the cap binds, not the fleet.

So the fleet supplies at least **~4.3 groups/s** while the batch consumes
0.31/s. The waste is not idle hardware; it is graded work thrown away
because a clock says the window is not over.

Two further costs follow from the same clock:

- A rollout that takes longer than the deadline is not merely late, it is
  unpaid and untrained — a direct disincentive against the response-length
  growth a DAPO run exists to produce.
- GRAIL is deferred to seal (`batcher.py`: admission "never runs GRAIL"),
  so the proof plane works in bursts and sits at 47% occupancy.

## Goal

Replace the clock with a fill condition, and validate on arrival instead
of at seal.

    window opens      one drand beacon -> generation seed + prompt slices
    admission         precommit, then payload, graded AND proven on arrival
    window closes     when every environment has its target of PROVEN groups
    emission          one training batch per 32 proven groups, continuously
    payment           per completion token, within each environment's share

One window is one pi_old interval: 16 optimizer steps, 512 groups
(`B_BATCH` = 16 prompts per env per step, 2 envs, 16 steps), one published
checkpoint.

Target steady state:

    fill            ~120 s at the measured offer rate
    proofs          280-610 s for 512 groups, depending on proof slots
    trainer         16 x ~37 s = ~592 s
    cycle           ~600-650 s, bounded by proofs or trainer, never by fill
    steps/day       ~740 -> ~2200

## Non-goals (hard constraints)

1. **NO miner protocol break.** The precommit endpoint
   (`server.py: async def precommit`, `_claim_upload_precommit`) is already
   in production, and the miner loop already polls `/state` and submits in
   a tight loop. Nothing here requires a coordinated cutover.
2. **NO change to the pi_old contract.** 16 optimizer steps per published
   checkpoint, all 512 groups generated from that one checkpoint.
   `ppo_ratio_outside_clip_ratio` must not move.
3. **NO speculative proving.** A group is proven only after its payload is
   revealed and matched against its precommit.
4. **NO weakening of the anti-pregeneration property.** The per-window
   drand beacon stays, because it is what forces the work to be done
   inside the window (see Component 1).
5. **NO ranking.** With a flat valuation every in-zone group is worth the
   same; this design takes that to its conclusion rather than breaking
   ties on something the valuation already said we are not buying.

## Architecture

```
 drand beacon -> window randomness -> generation seed + 16 prompt slices
       |
       v
  [ WINDOW OPEN ]  no announced deadline
       |
  miner: generate -> precommit(hash) -> upload payload
       |
  validator, per submission, ON ARRIVAL:
       dedup hash -> grade (CPU) -> sigma gate -> GRAIL (GPU) -> accept
       |
       +--> every 32 proven groups: emit one training batch to the journal
       |
  env target reached -> that env stops admitting
  both envs reached  -> WINDOW CLOSES
       |
  trainer (detached): 16 steps -> publish checkpoint -> next window
```

## Component 1 — The window and its close rule

**One drand beacon at open.** Verified, bound to the window, and expanded
into the generation randomness and the prompt slices exactly as today.
This is the only remaining use of drand in the hot path.

Keeping it is not optional. Every sampled token is a deterministic
function of it —

```python
u = u_at(randomness, prompt_idx, checkpoint_hash, rollout_index, t)
```

— and `forced_sampling.py` states the property it buys: *"Anti-pregeneration
still holds: `randomness` is unknown until [window open]."* Remove it and a
miner can generate before the window opens; the batch then fills in
seconds from pre-generated stock, and the winner is whoever has the lowest
network latency and the largest pre-generated inventory. The fill-closed
window depends on the seed to force the work to happen inside it.

The beacon is also what stops the validator grinding a seed that favours a
colluding miner — it can choose neither the prompt slices nor the
forced-seed stream.

**Everything else drand did is deleted.** `arrival_drand_round`, the seal
randomness used for the auction tie-break, `window_open_round` in
`throughput_rank`, `DRAND_ROUND_BACKWARD_TOLERANCE`, and the arrival-round
stamping at ingress all existed to *order* candidates. With no ranking
there is nothing to order. Drand calls drop from one per ~102 s window to
one per ~600 s window.

**Close rule.** The window closes when every environment has reached
`WINDOW_TARGET_GROUPS_PER_ENV` **proven** groups. Not admitted, not graded
— proven, because a group that fails GRAIL is not a group.

    admit while:  proven[e] + in_flight[e] < target[e]
    close when:   for every env e, proven[e] >= target[e]
    backstop:     WINDOW_MAX_SECONDS elapsed -> seal partial

Admission stops on `proven + in_flight`, the close fires on `proven` alone.
Gating admission on `proven` would over-admit by the whole proof pipeline
depth; gating the close on `proven + in_flight` would close on work that
may still fail GRAIL. A released reservation — a failed grade or proof —
immediately reopens capacity for the next arrival, which is what keeps a
run of failures from stalling the window.

Three properties follow:

- **The window is elastic with no tuning.** Its duration is
  `target / fleet rate`. As the policy's responses lengthen, the fleet
  produces fewer groups per second and the window stretches by itself.
  This is the property a length-growing DAPO run needs, and it is free.
- **Nothing is thrown away.** Redundancy goes from ~4:1 to 1:1. The 79 s
  per cycle currently spent rejecting `batch_filled` disappears.
- **Long generations are not cut off.** There is no deadline to miss. A
  rollout that takes 400 s lands, as long as the window has not filled —
  and per-token payment (Component 5) pays it for what it cost.

PR #212's proven-dominance early close is the direct precedent: it already
closes a window when the outcome is provably fixed. This generalises it
from "capacity charged" to "training target proven".

## Component 2 — Admission: precommit, then validate on arrival

**The flow, per submission:**

1. The miner generates a group, then sends the existing signed precommit —
   payload hash, size, prompt, environment, checkpoint, window.
2. The validator refuses a hash it has already seen this window, refuses it
   if the environment has no remaining capacity, and otherwise **reserves a
   slot in arrival order**. Both checks are cheap and both come before any
   payload moves.
3. The miner uploads the payload immediately. No reveal phase, no cohort,
   no waiting.
4. The validator grades it (CPU), applies the sigma gate and the rejection
   rules of Component 3, then proves it (GRAIL, GPU) — **all on arrival**.
5. On success the group is accepted; on any failure the reservation is
   released and the next arrival takes it.

The order matters: the two cheap refusals gate the two expensive stages, so
the validator never grades or proves a group it already knows it cannot
use.

**Why the precommit earns its place.** It makes arrival order the order in
which miners *finished generating*, not the order in which they finished
*uploading*. Without it a fat uplink beats a fast GPU on the last slots.
With it, and with the drand seed forcing generation to happen inside the
window, "first arrived" means "first to finish computing" — which is
exactly what the subnet wants to buy.

**Validating on arrival is the load-bearing change.** Today
`_prove_ranked` is called only from `_seal_batch_inner`, at seal, because
proving is expensive and only the ranked prefix is worth proving. With no
ranking there is no prefix: the only groups proven are those already
reserved against remaining capacity, so the proof plane spends its budget
on groups that will be trained on unless they fail — and it runs at a
steady rate instead of in a burst at seal. It is also what makes "close when
enough are proven" a decidable condition.

## Component 3 — What replaces the auction

Deleted from the production path: ranking, the value function, the
throughput tie-break, `slot_share`, the forensic sample ordering. Most of
`difficulty_auction.py` and `batch_selection.py` become dead code.

What remains, and must be re-expressed as **admission rules** rather than
prices:

| Control | Today | Here |
|---|---|---|
| `SIGMA_MIN` zone filter | eligibility | unchanged — still an absolute gate |
| `MAX_TRUNCATED_PER_SUBMISSION` | 1 math / 3 code | unchanged |
| `robust_uncertain_reward_utility` | **prices** a truncated group under its least favourable interpretation | **rejects** it: utility 0 -> refuse |
| content dedup | data quality | **economic control** (see below) |

### The robust-utility translation is mandatory

This is the one place where removing the auction silently removes a
defense. Today a manufactured zero — suppressing EOS on a correct rollout
to push an all-correct group into the sigma zone — is defeated by pricing:
`robust_uncertain_reward_utility` minimises the *gated* utility over every
joint assignment of the truncated rollouts' reward lattice, and the gate
returns `0.0` below `SIGMA_MIN`. A manipulated group can therefore never
score above its honest value.

With no valuation, "utility 0" prices nothing. It must become a rejection:
**if the robust utility is 0, refuse the submission.** Miss this and the
manufactured-loser path reopens at exactly the moment per-token payment
makes it profitable.

### Content dedup becomes monetary

Under per-slot payment, resubmitting an identical group won a duplicate
slot at worst. Under per-token payment it collects the same tokens twice.
`compute_rollout_hash` and the precommit's payload hash both already
exist; the requirement is that a duplicate is refused **at precommit**,
before any upload, and that the check is airtight rather than best-effort.

## Component 4 — Progressive emission

**Every 32 proven groups (16 per environment), emit one training payload**
to the trainer journal and keep collecting. Sixteen emissions fill the
window; `16 x 32 = 512` is arithmetic, not a schedule the miner can see.

Two requirements:

- **Ordering barrier at emission, not at proof.** Proofs finish out of
  order (`ProofScheduler` "applies results in rank order even when proofs
  finish out of order"), and `TrainerWorker` consumes the journal strictly
  by cursor. Emitting batch 5 before batch 4 would transpose two DAPO
  steps. Serialise publication only; proving stays concurrent.
- **Environment balance per batch.** A batch needs `B_BATCH` groups from
  each environment. `BalancedTrainingAccumulator` already carries a sparse
  environment's deficit forward and is the right home for this.

Trainer side is unchanged: `publish_every = 16`, one checkpoint per
window, the existing intake and swap path.

## Component 5 — Payment per completion token

Replaces `slot_share = pool / B_BATCH`.

```
pool_env      = pool * w_env                      # w_env = 1/2, 1/2
share(group)  = pool_env * tokens(group) / sum(tokens over accepted in env)
tokens(group) = completion tokens of EOS-TERMINATED rollouts only
```

**Why.** Under flat per-slot payment a group costs `16L/r` rounds and pays
the same regardless of `L`, so revenue per GPU-second is proportional to
`1/L`: halving response length doubles income. That is a standing bounty
on short answers, and it fights the training objective. Per-token payment
makes revenue per GPU-second independent of length, so the policy decides
how long to reason, not the miner's wallet.

**The EOS restriction is load-bearing, not a refinement.** Flat per-slot
payment is currently the simplest of four barriers against EOS
suppression: padding a rollout to the cap costs ~1.85x the group's GPU for
identical pay, a strictly negative margin. Per-token payment removes that
barrier and makes padding exactly *neutral* — and at neutral, any noise
tips it positive. Paying only EOS-terminated tokens restores a strictly
negative margin and aligns payment with what the loss already does (the
soft overlong punishment zeroes forced and truncated rollouts).

The three other barriers are unaffected: `MAX_TRUNCATED_PER_SUBMISSION`,
the robust-utility rejection of Component 3, and the fact that a padded
rollout grades 0 and can push its group out of the sigma zone.

**Per-operator cap, expressed as a token share.** Without a cap,
first-arrived is winner-take-all for the fastest operator. A cap counted
in *groups* does not bound the payout under per-token payment — an
operator can take few, very long groups — so the cap must bound an
operator's share of an environment's accepted tokens.

**Archive and weights.** Weight-only validators replay the EMA from R2
archives and must converge bit-for-bit, so the accepted token count per
group becomes part of the archive schema. `ROLLING_WINDOWS_HISTORY = 72`
bounds the replay depth in windows and must be re-dimensioned to preserve
the same wall-clock horizon as windows shorten.

## Component 6 — Proof throughput becomes a rate, not a budget

Proving on arrival makes the proof plane the admission rate limiter. At
the measured offer rate (>=4.3 groups/s) and ~1.2 s of proof per group,
demand exceeds one slot by a wide margin — but only until each
environment fills, and the fill target is what bounds total work per
window, not the offer rate.

Two consequences:

- **Every per-window proof budget must be restated as a rate.**
  `MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW`, `MAX_PROOF_WALL_SECONDS = 240`,
  and the fail-closed capacity qualification are all seal-time envelopes
  that no longer describe the shape of the work.
- **A bounded queue with an explicit backpressure policy.** When arrivals
  outrun proof capacity, the validator either rejects with a retry hint or
  drops the overflow. Rejecting is preferable: it is visible to the miner
  and keeps arrival order meaningful. The queue depth and the reject
  reason are part of the contract, not an implementation detail.

PR #207 (several proof slots per GPU, measured 12.5 s at one slot against
5.7 s at four with MPS) is what makes this affordable; re-measure at
cutover rather than trusting the ratio.

## Component 7 — Per-environment fill

Each environment closes independently when it reaches its target, and
`/state` advertises that per environment. The window closes when both
have closed.

The reference miner already samples its environment by the mix weights
and re-reads `/state` every iteration, so a fleet whose miners honour a
closed environment will rebalance toward the scarce one without a client
change.

**Expect Math to be the gating environment.** It has historically
under-filled — the receipt-starvation work of 2026-08-24 was Math running
at 14.51/16 — so Code will close first and Math will set the window
duration. That is the right pressure (it pulls the fleet toward the
scarce environment) but it means Code miners lose earning time once Code
closes, and the per-environment fill rate is the telemetry to watch.

## Configuration

| Variable | Meaning | Value |
|---|---|---|
| `RELIQUARY_WINDOW_TARGET_GROUPS_PER_ENV` | proven groups that close an environment | `256` (= 16 steps x `B_BATCH`) |
| `RELIQUARY_WINDOW_MAX_SECONDS` | backstop; seal partial past it | `1800` |
| `RELIQUARY_ENV_POOL_WEIGHTS` | share of the pool per environment | equal |
| `RELIQUARY_MAX_OPERATOR_TOKEN_SHARE` | cap on one operator's share of an env's accepted tokens | to be chosen deliberately |
| `RELIQUARY_PROOF_QUEUE_DEPTH` | bounded arrival queue before backpressure | to be sized from measurement |

`CHECKPOINT_PUBLISH_INTERVAL_WINDOWS` is deleted: one window is one pi_old
interval by construction. The emission cadence (32 proven groups) is
derived from `B_BATCH` and the environment mix, not separately tunable —
the same reasoning as `GRAD_ACCUM_STEPS = len(ENVIRONMENT_MIX)`.

No design-level enablement flag. Today's behaviour is not a point in this
parameter space, so an on/off flag would buy a second, unexecuted code
path — and this repository has paid for that once already, when the
proof-isolation work sat inert behind an OFF flag until a review found
nine findings, two critical, in code nobody had run.

The close rule does get an `AUCTION_EARLY_CLOSE_MODE`-style
`off | shadow | enforce`, which is a different thing: shadow mode *runs*
the rule on live traffic and records what it would have decided. It is a
measurement, not a dormant branch, and it is retired once the rule is
armed.

## Testing strategy

**Stage 1 — measure the fill, which the close rule cannot yet see.**

The close rule counts *proven* groups, and today GRAIL runs at seal
(`batcher.py`: admission "never runs GRAIL"), so a proven-count condition
is identically zero until the window is already over. Shadowing the real
rule is therefore impossible before Stage 2 — an earlier draft of this
document said otherwise and was wrong.

What is measurable today is the graded fill, and it **brackets** the answer
rather than pinning it. Nothing in `_pending` is proven, and a group that
fails its proof is not a group, so:

    floor    B_BATCH distinct graded prompts        every proof passes
    ceiling  MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW   half fail, the budgeted
             distinct graded prompts                worst case

The ceiling is not an invented number: the ranked prefix is `2 * B_BATCH`
precisely because it is "B_BATCH winners plus B_BATCH possible failed
candidates". The proven fill sits between the two, and recording only the
floor would size the design on its most optimistic case.

PR #212 measured the adjacent quantity (productive *capacity* charges at
+19 s to +45 s), but capacity is 64 receipts while a batch needs distinct
prompts.

It is read by polling, not by hooking admission: the resolution needed is
seconds, and the admission locks have a documented convoy history. It is
recorded once per environment per window and carried in
`upload_precommit_conservation`, which already reaches the R2 archive and
`/health` per environment.

Acceptance: both offsets are populated on healthy windows and the width of
the bracket answers open question 1. Nothing else changes.

The bracket collapses to a point in Stage 2: once proving runs on arrival,
the proven count is observable during the window and the close rule can be
shadowed directly.

**Stage 2 — validate on arrival, proof path unchanged.** Move GRAIL from
seal to arrival with the auction still in place. Acceptance criterion is
the one used for proof process isolation: **`proof_path_hash` unchanged**
and training payloads byte-identical on R2 replay.

**Stage 3 — remove the auction, arm the fill close.**

**Stage 4 — per-token payment.** Emission is the widest blast radius in
the system; it ships last and alone, so any regression is attributable.

**Unit coverage required:**

- close rule: fires only when every env has its target of *proven* groups;
  never on admitted-but-unproven; backstop seals partial
- robust utility as a rejection: a group whose least-favourable
  interpretation leaves the sigma zone is refused, not priced
- duplicate payload hash refused at precommit, before upload
- emission ordering: out-of-order proof completion still yields in-order
  journal entries
- per-token split: sums to the environment pool; truncated and non-EOS
  rollouts contribute zero; operator share cap binds
- backpressure: arrivals beyond the queue depth are rejected with a
  distinct reason, and arrival order is preserved for what is admitted

**Production counters. Must stay flat:**

- `ppo_ratio_outside_clip_ratio` — the pi_old interval is unchanged, so
  drift must not move. Strongest invariant in the design.
- `batch_filled` rejections — should fall to near zero; they are the 79 s
  of dead cycle this design exists to remove.

**Must move:** proof plane occupancy 47% -> steady; steps/day ~740 ->
~2200; redundancy 4:1 -> 1:1.

**The real risk to watch: `k_mean` and in-zone yield.** Accepting
everything removes the last soft filter above the sigma gate, the robust
utility, the proof, and the dedup. If `k_mean` leaves the gradient zone
(~6.7/16 today) or in-zone yield drifts, the cause is upstream of this
design — curation — and the response is to tighten admission, not to
restore ranking.

## Rejected alternatives (do not re-litigate)

- **A time-based close, with or without a minimum.** A minimum only helps
  long generations if selection is *not* by arrival; with a fill close the
  window is already elastic and the minimum has nothing to add.
- **Lottery selection among commitments.** Removes the compute race the
  subnet wants to pay for, and makes revenue depend on ticket count rather
  than work delivered.
- **Keeping the throughput tie-break.** It exists to stop pure-arrival
  ordering penalising long generation. With no ordering and per-token
  payment, it has no job left.
- **Sixteen simultaneous prompt-slice lanes** (PR #198). Lanes all finish
  at the same instant — the end of collection — so nothing can be
  pipelined, and the per-lane submission caps stop being a rate.
- **One optimizer step over the whole batch** (`aggregate_one_step`). With
  a single update pi_theta == pi_old, the ratio is identically 1, the clip
  never fires, and `eps_high = 0.28` becomes a no-op: DAPO degrades to
  vanilla policy gradient at 1/16 the update rate.
- **Dropping the drand beacon.** It is the anti-pregeneration property and
  the anti-grinding property; the precommit is complementary to it, not a
  substitute.

## Open questions

1. **Where does the fleet's offer rate actually saturate?** PR #212 gives
   a lower bound of ~4.3 groups/s because the 64-receipt cap binds. The
   true ceiling sets how fast a window can fill and therefore how much
   headroom the trainer floor really has. Stage 1 instruments the
   batch-shaped half of this; the receipt-shaped half needs the cap raised
   before it can be seen at all.
2. **Does removing all selection move the trained distribution?** With a
   flat valuation the auction was not selecting on quality, so the
   expectation is no — but `k_mean` and in-zone yield are the measurement,
   and they should be read before Stage 3 is armed, not after.
3. **The operator token-share cap.** It is now the only fairness lever
   against concentration under first-arrived admission, so its value
   deserves a deliberate choice rather than inheriting `B_BATCH`.
