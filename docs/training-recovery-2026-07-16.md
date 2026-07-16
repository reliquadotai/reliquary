# Training Recovery Runbook

Date: 2026-07-16

## Operator Decision

Keep validation, scoring, archive publication, and weight setting online while
optimizer steps remain frozen with `RELIQUARY_DISABLE_TRAIN=1`. Checkpoint 33
must not be used as the recovery starting point.

The recovery must be append-only:

1. Select an immutable historical source from the holdout evidence.
2. Republish those weights as checkpoint 34 so miners see a monotonic update.
3. Deploy the recovery image while training is still frozen.
4. Unfreeze only after the validator reports the complete pinned contract.
5. Stop automatically after checkpoint 35 and evaluate before continuing.

This is a validator training change. It does not change miner generation,
submission, scoring, or wire contracts.

## Incident Evidence

The production validator remained OPEN and continued filling balanced batches,
but one training step at 2026-07-16 07:57 UTC reported:

- approximate KL: `19.6033`
- pre-clip gradient norm: `10432`
- optimizer step: applied after clipping
- later publication: checkpoint 33, revision
  `606434551d06f37098f30fe46edf93a0c41320c2`

Gradient clipping limits update magnitude. It does not make a pathological
direction trustworthy. A paired protocol-parity screen subsequently measured
checkpoint 33 below upstream base on math reward, so the checkpoint is rejected
as a recovery source.

Production is frozen with `RELIQUARY_DISABLE_TRAIN=1`. The flag does not disable
serving or rewards. A ready balanced batch remains bounded in memory while the
optimizer and checkpoint publisher are bypassed.

## Correct Policy Contracts

The fixed KL design is useful only when three policies retain separate roles:

- `verify_model` is the published behavior policy used by miners. It supplies
  validator-recomputed `pi_old` for PPO.
- `base_ref_model` is the immutable upstream base checkpoint. It supplies only
  the KL anchor that constrains long-run drift.
- `train_model` is the mutable policy receiving the optimizer update.

Fixed-reference mode now refuses startup unless both
`RELIQUARY_KL_BETA` and
`RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY=true` are explicit. Miner-reported
log-probabilities remain useful telemetry, but no longer control PPO ratios in
this mode.

## Recovery Controls

The recovery image exposes and reports these independent controls:

1. `RELIQUARY_KL_BASE_MODEL`: immutable `repo@40-character-sha` KL anchor.
2. `RELIQUARY_KL_BETA`: explicit fixed-reference penalty strength.
3. `RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY=true`: trusted PPO behavior policy.
4. `RELIQUARY_LEARNING_RATE`: calibrated optimizer rate.
5. `RELIQUARY_GRAD_NORM_SKIP_THRESHOLD`: rejects a bad update before
   `optimizer.step()` and checkpoint cadence advancement.
6. `RELIQUARY_SHAPE_PENALTY`: validator-only auxiliary objective; zero disables
   it.
7. `RELIQUARY_TRAIN_UNTIL_CHECKPOINT_N`: restart-persistent canary ceiling.

Rejected steps report loss, KL, PPO tails, shaping exposure, and gradient
telemetry, then discard the failed batch without publishing a checkpoint.

## Exact Replay Evidence

All replay arms used the production runtime, immutable model revisions, the
same archived balanced windows, fixed seeds, validator-recomputed `pi_old`, and
the upstream Qwen base revision
`15852e8c16360a2fea060d615a32b45270f8a8fc` as the KL reference.

### Checkpoint 32 stability grid

Eleven consecutive production batches were replayed from checkpoint 32.

| Learning rate | KL beta | Result | Max grad norm | Final PPO outside clip |
|---:|---:|---|---:|---:|
| `5e-6` | `0` | 11/11 applied | `1.375` | `1.276%` |
| `5e-6` | `0.04` | 11/11 applied | `13.188` | not selected |
| `5e-6` | `0.10` | rejected at batch 11 | `171` | rejected |
| `3e-6` | `0` | 11/11 applied | `1.391` | `0.738%` |
| `3e-6` | `0.001` | 11/11 applied | `1.383` | `0.668%` |
| `3e-6` | `0.004` | 11/11 applied | stable | similar to `0.001` |

The least aggressive configuration retaining a real fixed-base constraint is
therefore `learning_rate=3e-6`, `KL beta=0.001`. The production circuit breaker
is set to `50`, well above the healthy replay range and far below the incident
value of `10432`.

### Shaping ablation

Ten exact checkpoint-10 windows, 22746 through 22755, were replayed from the
same starting weights with either `shape=0` or `shape=0.5`.

| Metric | Shape `0` | Shape `0.5` |
|---|---:|---:|
| Applied steps | `10/10` | `10/10` |
| Max grad norm | `1.813` | `2.422` |
| Final PPO outside clip | `0.309%` | `6.917%` |
| Mean PPO outside clip | `0.092%` | `1.074%` |
| Final max absolute PPO log-ratio | `0.843` | `1.112` |
| Rollout advantages changed, mean | `0%` | `38.203%` |
| Mean absolute advantage delta | `0` | `0.2691` |

`shape=0.5` is a dominant auxiliary objective, not a small regularizer. It is
provisionally disabled. It may be restored only if its paired held-out candidate
shows a material quality benefit that justifies the much larger policy-ratio
tail.

## Checkpoint Lineage

The first 24 pinned OpenMathInstruct holdout prompts were screened with one
protocol-forced sample per prompt and production generation budgets.

| Model | Pass | Forced close | Rambling proxy | Termination |
|---|---:|---:|---:|---:|
| Upstream base | `54.17%` | `95.83%` | `20.83%` | `45.83%` |
| Checkpoint 10 | `50.00%` | `83.33%` | `4.17%` | `45.83%` |
| Checkpoint 15 | `50.00%` | `70.83%` | `4.17%` | `41.67%` |
| Checkpoint 20 | `33.33%` | `66.67%` | `20.83%` | `62.50%` |
| Checkpoint 25 | `29.17%` | `41.67%` | `4.17%` | `87.50%` |
| Checkpoint 30 | `33.33%` | `70.83%` | `8.33%` | `37.50%` |

Checkpoint 20 is the first screened checkpoint with a statistically clear
capability regression versus base: `-20.83` percentage points with paired 95%
CI `[-37.50, -4.17]`. Checkpoint 15 remains statistically tied on reward while
significantly reducing forced-close and rambling behavior. A second disjoint
24-prompt holdout is running before the final reset-source decision.

The archive identity was checked from every accepted group, not inferred from
timestamps. Windows 22796 through 22805 all claim checkpoint-15 revision
`2f3d4a0b9224abdf1e5707d385f0620ab43e47c9` and contain exactly eight math
plus eight code groups.

## Remaining Pre-Deployment Gates

The following gates must complete before unfreezing production:

1. Combine the disjoint 24-prompt holdouts and run paired bootstrap comparisons
   for base, checkpoint 10, and checkpoint 15.
2. Compare the held-out checkpoint-10 candidates trained with shape `0` and
   shape `0.5`.
3. Replay the selected controls across the exact ten checkpoint-15 windows if
   checkpoint 15 is selected.
4. Run pinned gVisor code screens for base and the selected reset candidate.
5. Repeat a small selected-model math screen in a fresh solo process and compare
   exact completion hashes as a numerical-runtime check.
6. Run the complete test suite and verify the final diff.

## Expected Production Contract

Subject to the remaining checkpoint and shaping screens, the recovery contract
is expected to be:

```dotenv
RELIQUARY_RESUME_FROM=sha:<checkpoint-34-commit>
RELIQUARY_KL_BASE_MODEL=Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc
RELIQUARY_KL_BETA=0.001
RELIQUARY_LEARNING_RATE=0.000003
RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY=true
RELIQUARY_GRAD_NORM_SKIP_THRESHOLD=50
RELIQUARY_SHAPE_PENALTY=0
RELIQUARY_SHAPE_LEN_FRAC=0.5
RELIQUARY_TRAINING_RUN_ID=qwen35-2b-recovery-20260716
RELIQUARY_WANDB_VERSION=training-recovery-20260716
RELIQUARY_TRAIN_UNTIL_CHECKPOINT_N=35
RELIQUARY_DISABLE_TRAIN=1
```

Checkpoint 34 is an append-only reset publication, not a newly trained policy.
After the frozen deployment reports every identity above, remove only
`RELIQUARY_DISABLE_TRAIN`. Ten successful balanced optimizer steps publish
checkpoint 35; the persistent ceiling then pauses further training while
serving, scoring, archiving, and weight setting continue.

## Abort Conditions

Re-freeze training immediately if any of the following occurs:

- `train/step_skipped_nonfinite == 1`
- `train/step_skipped_grad_spike == 1`
- fixed KL identity, behavior policy, beta, learning rate, or shaping differs
  from the approved contract
- large unexplained movement in bad termination, forced-close, or rambling
- sustained growth in PPO clipping, absolute log-ratio, KL tails, or gradients
- checkpoint-35 evaluation regresses beyond paired holdout uncertainty
- archive publication, checkpoint publication, or validator health degrades

An aborted optimizer step must never advance trained-window cadence or publish
a checkpoint. Validation and reward accounting remain online during a freeze.
