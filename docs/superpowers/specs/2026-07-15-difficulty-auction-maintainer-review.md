# Difficulty Auction Maintainer Review

**Date:** 2026-07-15  
**Reviewed branch:** `design/difficulty-auction` at
`d1374d8456c41f37d0d22812e9d69738182b0ade`  
**Current main:** `d95e4254f12a9dc4c92692e80c251bccd3934024`  
**Decision:** Do not merge or deploy the reviewed branch. Merge only the
observation-only shadow instrumentation built from current `main`.

## 1. Executive verdict

The design is directionally right about one important problem: Reliquary should
spend more optimizer budget on informative, difficult groups instead of letting
arrival order determine all training data. The proposed score

```text
v = std(rewards) * (1 - mean(rewards)) ** delta
```

is a reasonable research hypothesis. It combines a proxy for non-zero GRPO
signal with a harder-question tilt.

The implementation is not production-ready. Its nominal shadow mode changes the
live candidate population before computing the shadow result, the coldkey cap is
not wired into the validator, the fixed deadline is unsupported by the available
population data, and deferred proof creates new prompt-squatting, queue-memory,
and multi-hotkey variance attacks. The branch also misses the current main
`force_span` trust-boundary fix.

The correct next move is measurement, not activation:

1. Merge inert math-only shadow scoring and exact arrival telemetry.
2. Run a full-pool, non-weight-setting canary with bounded resource controls.
3. Compare optimizer-side difficulty balancing against the market auction on a
   fixed validated dataset.
4. Close operator-level variance farming and grader false-negative risk before
   any payout or training-selection change.

## 2. Review scope and evidence

- Reconstructed the full diff from merge-base `10706b0`: 45 files, 8,371
  insertions, 1,616 deletions, including 24 unrelated historical documents.
- Refetched GitHub before concluding. The reviewed branch and `main` revisions
  above were still current.
- Ran the reviewed branch CPU suite: 1 failed, 1,191 passed, 5 skipped. The
  failure expected `hash_duplicate` but received `prompt_claimed`, showing the
  new first-claim rule preempts an existing dedup contract.
- Inspected the live validator at `209.20.157.231`. It runs current `main`, not
  the auction branch, and its OCI revision is `d95e4254...`.
- Replayed 250 and 1,100 public R2 windows with a deterministic script.
- Replayed the 250-window sample with a fresh SN81 hotkey-to-coldkey map.
- Ran the clean shadow branch suite: 1,179 passed, 5 skipped. Focused auction,
  archive, and replay tests: 19 passed.
- Reviewed current RLVR methods including
  [DAPO](https://arxiv.org/abs/2503.14476),
  [Dr. GRPO](https://arxiv.org/abs/2503.20783),
  [DisCO](https://arxiv.org/abs/2505.12366), and ICLR 2026
  [MathForge/DGPO](https://arxiv.org/abs/2601.20614).

## 3. Blocking implementation findings

### P0: The branch's shadow mode already changes production

`_prove_ranked()` sorts the pending pool by auction score and replaces
`self._valid` with only the proven top-ranked survivors. The existing production
selector then runs over that prefiltered set. `_compute_shadow_auction()` runs
afterward over the same survivors.

Therefore the archived shadow result is nearly tautological and live training,
cooldown, and emission eligibility have already been auction-filtered. The code
comments saying production remains drand-FCFS are false.

Evidence:
[batcher.py lines 2143-2201](https://github.com/reliquadotai/reliquary/blob/d1374d8456c41f37d0d22812e9d69738182b0ade/reliquary/validator/batcher.py#L2143-L2201)
and
[batcher.py lines 2243-2275](https://github.com/reliquadotai/reliquary/blob/d1374d8456c41f37d0d22812e9d69738182b0ade/reliquary/validator/batcher.py#L2243-L2275).

### P0: The coldkey cap is not active

The auction accepts a `coldkey_of` callback, but `ValidationService._open_window`
does not pass one. The fallback treats each hotkey as a different operator, which
cannot constrain a multi-hotkey coldkey. Calling that fallback safe is incorrect.

Evidence:
[service.py lines 531-567](https://github.com/reliquadotai/reliquary/blob/d1374d8456c41f37d0d22812e9d69738182b0ade/reliquary/validator/service.py#L531-L567)
and
[batch_auction.py lines 59-124](https://github.com/reliquadotai/reliquary/blob/d1374d8456c41f37d0d22812e9d69738182b0ade/reliquary/validator/batch_auction.py#L59-L124).

An enforceable cap needs a block-pinned metagraph snapshot and must fail closed
when any eligible hotkey lacks an operator mapping. It must never silently
substitute hotkey identity.

### P0: Multi-hotkey variance farming remains profitable

The forced seed includes the hotkey, so one coldkey controlling `N` hotkeys gets
`N` independent legal groups for the same prompt. It can generate all groups
inside the window and submit only the highest-scoring result. First-claim at the
validator cannot observe the discarded groups.

Exact binomial simulation for a prompt with true success probability `q=0.50`:

| Hotkeys | E[max v] | P(any k=2) | P(any k=2 or 3) |
|---:|---:|---:|---:|
| 1 | 0.2320 | 10.9% | 32.8% |
| 4 | 0.3002 | 37.1% | 79.6% |
| 8 | 0.3143 | 60.4% | 95.8% |
| 16 | 0.3212 | 84.3% | 99.8% |

The theoretical maximum is about `0.3248`. Eight hotkeys can therefore make a
truly medium prompt look almost auction-optimal. A per-window coldkey slot cap
limits payout count but does not remove this score inflation.

Before activation, bind the forced seed to a block-pinned operator identity for
`(window, coldkey, prompt, rollout, token)`, or replace one-group observed
difficulty with a persistent/shrunk estimate that cannot be improved by hidden
retries. This is a coordinated miner protocol change.

### P0: First claim enables prompt squatting

A fabricated in-zone group can claim a prompt after cheap grading and before
GRAIL. The honest group is then rejected as `PROMPT_CLAIMED`; the fabricated
group fails only at seal. This costs the attacker no generation and denies the
prompt for the entire window.

The design's 14-million-prompt argument is not the live contract. The live
validator reports `PROMPT_RANGE_SIZE=5000` and enforcement from window 13715.
With per-hotkey quota 8, coordinated hotkeys can reserve a meaningful fraction
of a public 5,000-prompt slice.

Resolve same-prompt competition at final ranking after proof. Bound candidates
per `(coldkey, prompt)` and promote a valid fallback when a higher-ranked claim
fails. Do not add the new wire-level `PROMPT_CLAIMED` enum; strict older miners
can fail response deserialization on an unknown enum value.

### P0: Deferred proof has an unbounded wall-clock attack

The design document proposes 16 proof attempts, but the implementation explicitly
removes that bound. It can grade/admit 96 candidates and prove serially until
eight pass, with only a two-failure limit per hotkey. A multi-hotkey attacker can
therefore force tens of 5-25 second GPU proofs before honest candidates are
reached. This can add minutes and stall checkpoint cadence.

Evidence:
[constants.py lines 152-169](https://github.com/reliquadotai/reliquary/blob/d1374d8456c41f37d0d22812e9d69738182b0ade/reliquary/constants.py#L152-L169)
and
[constants.py lines 312-320](https://github.com/reliquadotai/reliquary/blob/d1374d8456c41f37d0d22812e9d69738182b0ade/reliquary/constants.py#L312-L320).

A canary needs both a global proof-attempt limit and a global proof wall-clock
budget. Exhaustion must use an explicit fallback policy, archive the shortfall,
and burn unpaid slots rather than run indefinitely.

### P1: Queue memory is reserved too late

HTTP checks `proof_grading_attempts`, but the worker increments it only when
grading starts. Many concurrent requests can all pass the check and enqueue up
to 256 large rollout payloads. This is a memory/backpressure regression and lets
requests admitted before the deadline get silently dropped after seal.

Reserve grading count and payload bytes atomically before queue insertion.
Release both on every cancel/drop path. Bound count, serialized bytes, per-hotkey
bytes, and per-environment sandbox work.

### P1: The forensic sample is predictable

`sha256(current_window_randomness || miner_controlled_merkle_root)` is computable
before submission. A miner can grind its root or choose which candidate to send
after seeing the window randomness. Use future drand entropy revealed only after
the commitment deadline, then prove the sample before publication.

### P1: The branch applies the redesign to code prematurely

The fixed deadline and deferred grading path are shared by both environments,
while the design says math first. Live production enables both math and code.
Worst-case code grading is sandbox-bound and can exceed the proposed deadline.
Keep `opencodeinstruct` entirely out of scope until a separate throughput and
failure-isolation canary passes.

### P1: It conflicts with current main

The branch diverged before PR #128 and does not contain the validated BFT
`force_span` trust-boundary fix. It also carries broad historical changes and
cannot be safely merged by resolving conflicts mechanically. Any retained work
must be rebuilt from current `main`, as the shadow branch is.

## 4. Scoring analysis

For eight binary rewards with `p=k/8`, current GRPO uses
`A=(r-p)/sqrt(p(1-p))`. The total absolute group advantage is

```text
2 * G * sqrt(p * (1-p))
```

and the aggregate positive and negative magnitudes are equal. The design's claim
that `k=2` has a dominant positive update while `k=6` has a dominant negative
update looks only at the largest per-rollout coefficient; it does not account
for how many positive and negative rollouts there are. The two groups are
symmetric under aggregate GRPO magnitude before sequence lengths and clipping.

The proposed score can still be interpreted as:

```text
current GRPO group magnitude proxy * explicit difficulty tilt
```

but that interpretation makes it a hypothesis, not a derivation of optimal
training value. It also uses one noisy eight-draw outcome as if it were intrinsic
prompt difficulty, creating winner's curse and hidden-retry incentives.

The stronger current research pattern is "balance, then reweight":

- DAPO removes all-correct and all-wrong groups so batches retain non-zero
  gradients. It does not establish `k=2` as a universal optimum.
- DGPO first uses mean-absolute-deviation normalization to equalize question
  update magnitude, then applies a temperature-controlled question difficulty
  weight inside the optimizer.
- Dr. GRPO and DisCO are further warnings that std normalization and implicit
  difficulty weighting can create optimization biases.

For Reliquary, optimizer-side balancing is lower risk than making emission an
all-or-nothing tournament on observed `k`. It should be the first H200 ablation.

## 5. Production replay results

### 250-window replay, no coldkey cap

Population: 246 windows with candidates, 1,596 fully validated submissions.
The archive also reports 10,867 `batch_filled` rejects whose reward vectors are
not available. They cannot be called valid or assigned counterfactual scores.
Coldkey shares use the fresh current metagraph because historical block-pinned
maps were not archived; treat them as a recent attribution, not consensus data.

| Selector | Selected | Mean reward | Mean Jaccard vs production | Top hotkey | Top coldkey |
|---|---:|---:|---:|---:|---:|
| Production | 1,432 | 0.6155 | 1.0000 | 18.09% | 33.94% |
| delta=0 | 1,432 | 0.6128 | 0.9506 | 18.23% | 34.22% |
| delta=0.5 | 1,432 | 0.6095 | 0.9408 | 18.37% | 34.08% |
| delta=1 | 1,432 | 0.6084 | 0.9387 | 18.23% | 34.15% |

The score tilts the observed pool harder, but only modestly and without a
concentration improvement. `delta=1`, `1.5`, and `2` select the same candidates
under the current sigma-gated discrete support, so this replay cannot identify
`delta=1` as optimal.

### 250-window replay, coldkey cap 2

All 25 eligible hotkeys mapped to coldkeys. At `delta=1`, selected slots fell
from 1,432 to 1,211, mean reward fell to 0.5873, top coldkey share fell from
33.94% to 29.81%, and top hotkey share rose to 19.16%. The cap modestly reduces
operator concentration but underfills about 15.4% of otherwise filled slots.

### 1,100-window replay, no cap

Population: 1,096 windows, 10,104 validated candidates, and 63,370 unobserved
`batch_filled` rejects. Production selected 8,221 groups with mean reward
0.5854. `delta=1` lowers mean reward to 0.5625, has Jaccard 0.8418, and leaves
top-hotkey concentration almost unchanged (10.53% to 10.49%).

### Deadline evidence

Historical archives do not contain exact HTTP arrival age. `response_time` is an
upper-bound proxy that includes queueing and validator work because `arrived_at`
was set after validation. On the 250-window accepted population:

| Deadline | Mean distinct validated prompts seen | Windows reaching 8 |
|---:|---:|---:|
| 120 s | 1.65 | 0.0% |
| 180 s | 2.86 | 1.2% |
| 300 s | 3.91 | 6.1% |
| 360 s | 4.33 | 9.3% |

This does not prove 300 seconds is bad; the rejected population may contain more
valid candidates. It proves the current archive cannot support the claim that a
300-second full auction preserves fill and cadence. Exact arrival telemetry is
included in the safe shadow branch for future windows.

## 6. Safe implementation built from current main

Branch `codex/difficulty-auction-shadow` contains three isolated commits:

1. `389ebad` - pure, side-effect-free shadow scoring and selection.
2. `2896dd9` - archive the real counterfactual over production's fully validated
   pool.
3. `f96a05f` - reproducible R2 replay plus exact arrival-age telemetry.

Properties:

- Math only.
- No miner wire change.
- No new reject reason.
- No admission, proof, queue, seal, cooldown, payout, weight, or training change.
- Shadow failures are caught and cannot fail production sealing.
- The operator cap is reported as active only with a complete mapping; no
  hotkey-identity fallback.
- Every archive states its population limitations.
- Per-candidate mean, std, count, eligibility, rank, and selected state make any
  future delta replayable without another deployment.

Reproduce historical analysis with:

```bash
python scripts/report_difficulty_auction.py \
  --from-r2 --current-window 23102 --n 250 \
  --environment openmathinstruct --operator-map-from-chain \
  --max-slots-per-operator 2
```

## 7. Activation plan and stop rules

There is no scientifically meaningful fixed wait such as 24 hours. Progress is
gated by denominators and checkpoint coverage.

### Stage A: deploy passive telemetry

Merge the safe shadow branch. Collect at least:

- 200 math windows;
- 1,500 fully validated groups with exact HTTP arrival ages;
- at least three checkpoint transitions;
- candidate, selected, rewarded, and reject denominators split by checkpoint;
- score/k histogram, selection overlap, operator concentration, underfill, and
  queue/verification latency.

Stop if production selection, rewards, archive publication, or window cadence
changes. That would be a shadow-integrity bug.

### Stage B: full-pool H200 canary, no weights

Run a separate validator instance that cannot publish weights, checkpoints, or
training updates. Math only. Duplicate a bounded miner cohort into it.

Required controls:

- atomic grading-count and payload-byte reservation before enqueue;
- per-hotkey and global byte caps;
- one or two candidates per `(coldkey, prompt)`, resolved after proof;
- global proof-attempt and wall-clock budgets with explicit burn fallback;
- future-drand forensic sampling;
- block-pinned hotkey-to-coldkey map, fail closed;
- archive attempted, signed, queued, graded, eligible, proven, selected, paid,
  rejected, dropped, and unobserved counts separately.

Stop on OOM, event-loop starvation, code-environment work, unknown identity,
more than 10% canary underfill relative to production, or proof-budget
exhaustion in more than 1% of windows.

### Stage C: independent grader audit

Blindly double-adjudicate low-k math negatives using a different symbolic path
plus human review for disagreements. Re-running the same grader is not an
independent measurement. Report false-negative rate with a confidence interval,
split by prompt source, answer format, checkpoint, miner, and k.

Do not admit or pay `k=1` until the upper confidence bound is below an agreed
operational threshold. Low-k candidates need stricter re-adjudication because a
broken label is both maximally paid and poisonous training data.

### Stage D: equal-token optimizer ablation on H200

Use the same frozen validated dataset, checkpoint, token budget, and evaluation
suite for at least three seeds per arm:

1. Current GRPO baseline.
2. MAD-balanced advantages only.
3. MAD-balanced advantages plus difficulty weighting, sweeping temperature.
4. Current GRPO trained on the auction-selected subset.

Track held-out math accuracy, checkpoint-to-checkpoint regression, KL, entropy,
completion length, bad termination, reward/k histogram, gradient norm per group,
and trainable token count. Reject any arm that improves training reward without
held-out gains, increases gibberish/bad termination materially, or regresses easy
and medium problems beyond the agreed tolerance.

### Stage E: protocol design before any economic activation

- Bind forced sampling to coldkey or another operator-stable identity.
- Remove first-claim ownership; promote proven fallbacks at selection.
- Keep the sigma gate until low-k label quality is independently cleared.
- Use a complete block-pinned coldkey map.
- Keep code out.
- Version any miner-visible contract and coordinate rollout.
- Run shadow, then a small capped canary payout, then staged activation with a
  one-flag rollback.

## 8. Final maintainer position

Do not merge `design/difficulty-auction`. Its central training intuition deserves
continued work, but the implementation currently changes production under a
shadow label and creates exploitable economic and liveness surfaces.

Merge the observation-only branch, collect the missing population evidence, and
test optimizer-side DGPO-style balancing first. If the auction still adds
held-out model quality after those tests, rebuild the active mechanism from
current `main` with operator-bound sampling, post-proof prompt resolution,
bounded resources, and math-only staged rollout.
