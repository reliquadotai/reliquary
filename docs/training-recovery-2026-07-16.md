# Training Recovery Runbook

Date: 2026-07-16

## Decision

Keep validation, scoring, archive publication, and weight setting online, but
freeze optimizer steps until a known-good checkpoint and fixed-reference
configuration pass the H200 calibration below. Do not continue training from
checkpoint 33 merely because the process is mechanically healthy.

The fixed KL design is directionally correct, but it is not sufficient alone.
PPO's behavior policy and the KL reference have different roles:

- `verify_model` is the published policy that generated accepted rollouts. It
  supplies validator-recomputed `pi_old`.
- `base_ref_model` is an immutable base checkpoint. It supplies only the KL
  anchor used to constrain long-run drift.
- `train_model` is the mutable policy receiving the optimizer update.

## Incident Evidence

The live validator remained OPEN and continued filling balanced batches, but a
training step at 2026-07-16 07:57 UTC reported:

- approximate KL: `19.6033`
- pre-clip gradient norm: `10432`
- optimizer step: applied after clipping
- later publication: checkpoint 33, revision
  `606434551d06f37098f30fe46edf93a0c41320c2`

Gradient clipping limits update magnitude; it does not make a pathological
direction trustworthy. This event is sufficient to reject checkpoint 33 as an
automatic recovery starting point pending quality evaluation.

Production was therefore frozen with `RELIQUARY_DISABLE_TRAIN=1`. This flag
does not disable serving or rewards. The balanced accumulator is bounded at its
configured target and retains one ready batch without publishing a checkpoint.

## Recovery Controls

The recovery patch adds four independent controls:

1. `RELIQUARY_KL_BASE_MODEL`: fail-closed immutable `repo@40-char-sha` anchor.
2. `RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY=true`: removes miner-reported
   token log-probabilities from the PPO objective.
3. `RELIQUARY_LEARNING_RATE`: allows calibration without source edits.
4. `RELIQUARY_GRAD_NORM_SKIP_THRESHOLD`: rejects non-finite or pathological
   pre-clip gradients before `optimizer.step()` and checkpoint publication.

The default skip threshold is `100`. It is deliberately much higher than the
previously observed healthy range (approximately `1-8`) and much lower than the
production failure (`10432`). It is a circuit breaker, not a target.

## H200 Calibration

Use one immutable holdout, one archived balanced training batch, fixed seeds,
and the exact production runtime. Never compare candidates on different data.

### Phase A: choose the restart checkpoint

Evaluate at minimum:

- upstream base revision
  `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`
- the best available checkpoint before the suspected degradation interval
- checkpoints 20, 30, and 33

Measure per environment: strict reward/pass rate, bad termination, rambling or
repetition, completion length, BFT-forced rate, and seed/CDF verification. The
restart checkpoint must improve over base on the primary reward metric without
a statistically meaningful regression in termination quality.

### Phase B: one-step stability grid

From the selected checkpoint, replay the same balanced batch across:

- learning rate: `1e-6`, `2e-6`, `5e-6`
- KL beta: `0.001`, `0.004`, `0.01`, `0.04`
- validator `pi_old` recomputation: always enabled
- gradient skip threshold: `100`

Record pre-clip gradient norm, PPO loss, weighted KL objective, KL/PPO ratio,
clip-active fraction, non-finite token ratio, peak VRAM, and wall time. Reject
any candidate that trips the health gate or has non-finite values.

### Phase C: short closed loop

Run the strongest two or three Phase B candidates for at least ten sequential
on-policy updates, regenerating rollouts from the newly updated policy between
steps. Re-evaluate the frozen holdout after each publish-equivalent interval.
Do not select a configuration from one-step loss alone.

Select the least aggressive candidate whose reward trend is positive and whose
termination, repetition, KL, gradient, and PPO clipping metrics remain stable.

## Production Restart

Only unfreeze on an image containing the recovery controls. Pin every identity:

```dotenv
RELIQUARY_RESUME_FROM=sha:<selected-known-good-checkpoint>
RELIQUARY_KL_BASE_MODEL=Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc
RELIQUARY_KL_BETA=<selected-beta>
RELIQUARY_LEARNING_RATE=<selected-lr>
RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY=true
RELIQUARY_GRAD_NORM_SKIP_THRESHOLD=100
RELIQUARY_WANDB_VERSION=training-recovery-20260716
```

Remove `RELIQUARY_DISABLE_TRAIN` only after startup health reports the selected
resume revision, fixed KL revision, explicit beta, learning rate, behavior
source `verify_model`, and threshold `100`.

Watch the first optimizer step synchronously. It must be finite, below the
health threshold, archived as `trained=true`, and followed by publication only
on the configured successful-step cadence. Keep the previous production env
file and image digest as the immediate rollback point.

## Abort Conditions

Re-freeze training immediately if any of the following occurs:

- `train/step_skipped_nonfinite == 1`
- `train/step_skipped_grad_spike == 1`
- a large unexplained step change in bad termination or rambling
- sustained KL or clipping growth across sequential updates
- evaluation falls below the selected restart checkpoint beyond the holdout's
  expected sampling uncertainty
- health reports a mutable or unexpected KL/reference revision

An aborted training step must never increment the trained-window cadence or
publish a checkpoint. Validation and reward accounting should remain online.
