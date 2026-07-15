# Difficulty Auction Audit and Rollout Decision

**Date:** 2026-07-15

**Branch:** `design/difficulty-auction`

**Original tip reviewed:** `d1374d8456c41f37d0d22812e9d69738182b0ade`

**Production baseline reviewed:** `c33560d4cb810e394f7224419570bda87f32ed9e`

## Executive Decision

The difficulty objective is promising enough to keep measuring, but the
original active protocol implementation is not safe to deploy.

This branch is now a merge-safe, observation-only experiment:

- Production admission, proofs, sealing, batch selection, training, cooldown,
  and emission are unchanged from current `main`.
- The fully validated Math candidate pool is copied into immutable shadow
  values only after production selection is complete.
- The counterfactual ranks by
  `std(reward) * (1 - mean(reward)) ** delta`.
- A two-slot operator cap is simulated only when every eligible candidate maps
  to a real metagraph coldkey. Incomplete ownership data disables the cap and is
  reported explicitly.
- Operator ownership is archived with the window so historical replay does not
  silently use a later chain state.
- Code remains out of scope. Its reward and grader semantics are not equivalent
  to the binary Math experiment.

No miner update is required for this branch. It does not change the wire
protocol or a miner's production verdict, reward, or selection probability.

## Production Evidence

The live validator was healthy at revision
`c33560d4cb810e394f7224419570bda87f32ed9e`. The audit sampled R2 windows
23172 through 23177, all with exact HTTP arrival ages.

| Metric | Observed |
|---|---:|
| Windows | 6 |
| Fully validated Math candidates | 34 |
| Candidate hotkeys | 9 |
| Production-selected groups | 26 |
| Production mean reward | 0.600962 |
| Archived `batch_filled` rejects outside the scored population | 358 |
| Shadow compute time in the initial four-window sample | 0.31-0.75 ms/window |
| Shadow errors | 0 |

The sample is enough to reject unsafe assumptions, but not enough to choose a
production incentive curve.

### Difficulty replay

| Delta | Selected | Mean reward | Mean production Jaccard | Top hotkey share |
|---:|---:|---:|---:|---:|
| 0.0-0.5 | 26 | 0.591346 | 0.896825 | 34.62% |
| 0.75-2.0 | 26 | 0.586538 | 0.813492 | 34.62% |

This confirms that the score changes the training distribution. It does not yet
show that the changed distribution improves the checkpoint.

### Operator-cap replay

The nine candidate hotkeys were mapped completely from the SN81 metagraph.
With a two-slot cap:

| Metric | Production | Capped shadow |
|---|---:|---:|
| Selected groups | 26 | 24 |
| Mean reward at delta 0-0.5 | 0.600962 | 0.578125 |
| Top operator share | 30.77% | 29.17% |

The cap made only a small concentration improvement in this sample and burned
two otherwise usable training slots. A hard cap is therefore not approved. It
must be compared with a conditional cap, deterministic relaxation/backfill, and
operator-level diminishing returns before activation.

### Deadline replay

| Deadline | Mean candidates | Mean distinct prompts | Windows reaching 8 prompts |
|---:|---:|---:|---:|
| 120 s | 1.17 | 0.83 | 0/6 |
| 180 s | 2.17 | 1.50 | 0/6 |
| 300 s | 2.83 | 2.00 | 0/6 |
| 360 s | 2.83 | 2.00 | 0/6 |

The proposed fixed 300 second deadline would have underfilled every observed
window. The current state-driven sealing and sparse-window liveness controls
must remain in place.

## Rejected Original Behaviors

### 1. Deferred proof and top-down proving

The proposed path ranked unproven groups, then spent GPU verification only on
the apparent winners. An attacker can fabricate high-ranked groups cheaply and
force expensive failures before honest candidates are reached. The branch had
only a per-hotkey failure cap, which is bypassed with multiple registered
hotkeys, and no global proof-attempt or wall-time budget.

This is a validator GPU denial-of-service vector. Deferred proof remains
rejected unless a future design has all of the following:

- a global proof attempt and wall-time budget,
- a deterministic exhaustion policy,
- Sybil-resistant accounting stronger than one counter per hotkey,
- evidence that honest fill probability remains acceptable under adversarial
  ordering,
- unchanged final-verdict semantics for miners.

### 2. First unproven claim owns a prompt

`PROMPT_CLAIMED` let the first cheap submission reserve a prompt before GRAIL
validation. A fabricated submission could therefore block an honest group even
when the fabricated proof later failed. It also changed established logical
dedup verdict precedence and caused a regression in the current test suite.

Prompt uniqueness must be resolved after validation and during deterministic
selection. An unproven payload must never acquire economic ownership.

### 3. Fixed 300 second collection

The fixed deadline is contradicted directly by exact production arrival data.
It also expands the spam, grading, and sandbox workload while removing the
existing liveness controls. No fixed deadline should replace the state machine
without a much larger arrival study and an explicit resource budget.

### 4. Identity fallback for operator caps

Treating an unmapped hotkey as its own operator makes a Sybil guard appear
active while allowing the exact multi-hotkey behavior it is supposed to
measure. The current shadow implementation applies a cap only with complete
ownership for all eligible candidates. Missing data is visible and fails safe.

### 5. Active selection under a shadow label

The original branch filtered the live candidate pool before the production
selector and only then computed a field named `shadow_auction`. That changed
training and emission despite the shadow label. The current implementation
copies detached values after production metadata exists, contains all shadow
exceptions, and has randomized enabled-versus-disabled equivalence tests.

## Conceptual Risks Still Open

The score is not automatically a better learning objective.

1. **False-negative amplification.** Lowering a correct group's measured mean
   can increase its difficulty value. Any symbolic-grader false negative can
   therefore become economically attractive.
2. **Lucky-success amplification.** A `k=1` or `k=2` group may contain one lucky
   final answer with poor reasoning. High normalized advantage can reinforce
   that trajectory strongly.
3. **Curriculum shock.** Replacing the current distribution with mostly
   low-success groups may reduce short-term benchmark quality or increase
   unstable language even if the groups are authentic.
4. **Coarse ties.** With eight binary rewards there are few distinct scores, so
   speed and the canonical tie-break still determine many winners.
5. **Coldkey is not beneficial ownership.** A participant can register through
   several coldkeys. The cap raises Sybil cost but does not prove independent
   operators.
6. **Selection is not reward curvature.** Choosing harder groups does not by
   itself produce the desired top-heavy operator reward distribution. Selection
   quality, operator aggregation, and emission curvature need separate tests.
7. **Observed-population bias.** Current archives contain fully validated batch
   entries and runners-up, not the difficulty of pre-validation or
   `batch_filled` payloads. The shadow answers "what if we reranked what
   production fully validated," not "what would a longer open auction receive."

## Activation Gates

An active protocol PR must not be opened until all gates below pass.

### Operational gate

- At least 200-500 fully validated Math candidates.
- At least five hotkeys and five mapped operators, with concentration reported
  both ways.
- At least one checkpoint transition.
- Zero shadow exceptions and no measurable seal/archive regression.
- Complete eligible-candidate ownership in every window used for cap analysis.

### Mechanism gate

- Compare deltas 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, and 2.
- Compare no cap, cap 2, cap 3, and deterministic relax-to-fill variants.
- Report selected count, burn/underfill, reward mean, reward histogram,
  production overlap, hotkey concentration, and operator concentration.
- Segment results by checkpoint and by reward bucket. Pooled averages alone are
  insufficient.
- Reject any configuration whose diversity gain is mostly purchased by empty
  training slots.

### Learning gate

- Train matched short branches from the same checkpoint and data budget.
- Compare current selection against each finalist on held-out Math quality,
  termination/rambling, KL drift, entropy, length, and grader disagreement.
- Manually inspect low-k winners for lucky answers and false-negative grading.
- Require a repeat run before attributing a checkpoint gain to the mechanism.

### Security gate

- Adversarially simulate multi-hotkey and multi-coldkey operators.
- Test prompt collision, fabricated high-score groups, replay, malformed reward
  vectors, incomplete metagraph data, and proof-budget exhaustion.
- Preserve quota neutrality and existing reject precedence.
- Prove that shadow-disabled and shadow-enabled runs produce identical
  production batches, rewards, cooldown state, and dedup state.

## Recommended Direction

1. Merge and deploy only the observation-only branch.
2. Smoke-check two or three windows for health, archive schema, mapping
   completeness, and `production_changed=false`.
3. Continue collection until the operational gate is met; do not impose an
   automatic 24 hour delay if the sample arrives sooner.
4. Use archived operator snapshots for historical replay, with current-chain
   lookup only as fallback for older windows.
5. Choose finalists from evidence, then run matched training experiments.
6. Put any active selection or emission change in a new coordinated protocol
   PR with an explicit activation flag and rollback condition.

The present branch is suitable for measurement. It is not evidence that the
difficulty auction, delta 1, a hard operator cap, or a new collection cadence is
ready for production.
