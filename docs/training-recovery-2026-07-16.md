# Training Recovery Decision and Runbook

Date: 2026-07-16

## Final Decision

Restore training through one bounded, append-only canary:

1. Keep validation, scoring, weights, and archive publication online while
   `RELIQUARY_DISABLE_TRAIN=1` remains set.
2. Republish immutable checkpoint 15 as monotonically newer checkpoint 34.
3. Resume with a fixed upstream-base KL reference, validator-recomputed
   `pi_old`, `learning_rate=3e-6`, `KL beta=0.01`, and no length shaping.
4. Reject a step before `optimizer.step()` when the gradient is nonfinite or
   above `50`, or when more than `10%` of completion tokens lie outside PPO's
   `[0.8, 1.2]` ratio band.
5. Publish checkpoint 35 after ten successful balanced steps, then stop
   automatically and evaluate before allowing another step.

This release changes validator training and recovery operations only. It does
not change miner generation, submission schemas, scoring, BFT budgets, or wire
protocols. The difficulty-auction v2 protocol remains a separate release.

## Why Recovery Was Required

The production validator remained available, but a training step at
2026-07-16 07:57 UTC reported approximate KL `19.6033` and pre-clip gradient
norm `10432`. Gradient clipping limited update magnitude but could not make the
direction trustworthy. Those weights later became checkpoint 33 at revision
`606434551d06f37098f30fe46edf93a0c41320c2`.

A protocol-parity screen found checkpoint 33 below upstream base on math
reward. Checkpoint 33 is therefore rejected as a recovery source. Production
training was frozen before further optimizer steps while validation and archive
publication continued.

During calibration, an independent Subtensor runtime codec change caused every
miner to be rejected as `registration_unavailable`. PR #136 upgraded the chain
stack and exposed registration-cache health. The immutable production image was
verified with 256 registered hotkeys, accepted math and code submissions, and a
successful R2 upload for window 23332. Training remained frozen throughout.

## Policy Contracts

The fixed-KL design is correct only when three policies remain distinct:

- `verify_model` is the last published behavior policy used by miners. It
  supplies trusted, validator-recomputed `pi_old` for PPO.
- `base_ref_model` is immutable upstream Qwen3.5-2B. It supplies only the KL
  anchor that constrains long-run drift.
- `train_model` is the mutable candidate receiving optimizer updates.

Using the immutable base as both KL reference and `pi_old` would make PPO
off-policy as soon as the published model differs from upstream. Trusting miner
log-probability claims instead would leave the importance ratio miner-writable.
Fixed-reference mode therefore fails closed unless the base revision, beta, and
`RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY=true` are explicit.

## Reconciliation With The Initial Design

Rom's core diagnosis was right: the 2B model needed a stable reference to slow
policy drift. That fixed-base KL mechanism is retained. The experiments changed
the surrounding assumptions:

| Initial choice | Recovery choice | Reason |
|---|---|---|
| Fixed upstream-base KL | Retained | It adds a real long-run drift constraint. |
| `KL beta=0.04` | `0.01` | `0.04` pulled toward the upstream base's worse forced-close/rambling profile and lost held-out reward. |
| `learning_rate=5e-6` | `3e-6` | The lower rate produced stable ten-step trajectories with useful gate margin. |
| Two-sided shaping `0.5` | Disabled (`0`) | It changed `38.2%` of advantages, greatly enlarged PPO tails, and showed no held-out benefit. |
| Miner-claimed `pi_old` | Validator recomputation | Fixed KL and PPO behavior policy are separate contracts; claims cannot control the PPO ratio. |
| Open-ended continuation | One ten-step canary | Earlier training improved transiently and then collapsed; checkpoint publication is now an evaluation boundary. |

The miner-only presence penalty was not restored. A presence or repetition
processor changes the sampled policy. Unless the validator applies the exact
same stateful transform when recomputing old log-probabilities and forced-seed
decisions, PPO trains against the wrong distribution. BFT remains the auditable
math termination mechanism; CODE is still single-phase and has a separate
long-tail termination problem.

## Reset Source Selection

Base, checkpoint 10, and checkpoint 15 were screened on the same 48 pinned
OpenMathInstruct prompts with one protocol-forced sample per prompt.

| Model | Pass | Forced close | Rambling proxy | Termination |
|---|---:|---:|---:|---:|
| Upstream base | `56.25%` | `95.83%` | `18.75%` | `50.00%` |
| Checkpoint 10 | `58.33%` | `68.75%` | `2.08%` | `54.17%` |
| Checkpoint 15 | `56.25%` | `62.50%` | `2.08%` | `52.08%` |

Checkpoint 15 revision
`2f3d4a0b9224abdf1e5707d385f0620ab43e47c9` was selected. It is tied with
base on reward while substantially improving forced-close and rambling rates,
and it preserves five more historical training steps than checkpoint 10.

Windows 22796 through 22805 each contain exactly eight math and eight code
groups, and every accepted group claims that checkpoint-15 revision. They form
the exact ten-step replay corpus.

## Shaping Ablation

Ten exact checkpoint-10 windows were replayed with shape `0` and `0.5`.

| Metric | Shape `0` | Shape `0.5` |
|---|---:|---:|
| Applied steps | `10/10` | `10/10` |
| Max gradient norm | `1.813` | `2.422` |
| Mean PPO outside clip | `0.092%` | `1.074%` |
| Final PPO outside clip | `0.309%` | `6.917%` |
| Advantages changed | `0%` | `38.203%` |
| Mean absolute advantage delta | `0` | `0.2691` |

Shape `0.5` is a dominant auxiliary objective, not a small regularizer. It is
disabled for recovery. Shaping may return only through a new paired experiment
that demonstrates quality benefit, not merely shorter output.

## KL Calibration

All arms started from checkpoint 15, replayed the same ten balanced windows,
used `LR=3e-6`, shape `0`, validator-recomputed `pi_old`, and immutable upstream
Qwen revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.

| KL beta | Applied | Max grad | Mean PPO outside | Peak | Final | Final KL |
|---:|---:|---:|---:|---:|---:|---:|
| `0.001` | `10/10` | `6.281` | `2.951%` | `9.244%` | `8.397%` | `0.1736` |
| `0.01` | `10/10` | `6.156` | `2.817%` | `8.812%` | `8.151%` | `0.1707` |
| `0.04` | `10/10` | `5.656` | `2.365%` | `7.470%` | `6.849%` | `0.1601` |

The 16-prompt, two-sample paired math screen then measured:

| Model | Pass avg | Pass@1 | Pass@2 | Forced | Ramble | Termination |
|---|---:|---:|---:|---:|---:|---:|
| Checkpoint 15 | `43.75%` | `50.00%` | `50.00%` | `62.50%` | `3.13%` | `43.75%` |
| Beta `0.001` | `46.88%` | `43.75%` | `50.00%` | `62.50%` | `12.50%` | `43.75%` |
| Beta `0.01` | `43.75%` | `37.50%` | `50.00%` | `62.50%` | `0%` | `46.88%` |
| Beta `0.04` | `37.50%` | `37.50%` | `43.75%` | `65.63%` | `6.25%` | `37.50%` |

Beta `0.001` sits closest to the PPO gate and increased the rambling proxy.
Beta `0.04` lost aggregate reward and moved forced-close, rambling, and
termination in the wrong direction. Beta `0.01` preserved aggregate reward,
removed observed rambling in this screen, and retained more drift margin. It is
the selected compromise.

## Reproducibility And Numerical Contract

The selected replay was repeated in fresh processes under the production
runtime. The final provenance-correct run reported:

- source revision: `f401a637e82cb4ef0ce2752c9c5783cd90a2b5b4`
- runtime: H100 PCIe, torch `2.7.0+cu128`, transformers `5.9.0`,
  flash-linear-attention `0.5.0`, no causal-conv1d
- applied steps: `10/10`
- max gradient norm: `6.125`
- mean / peak / final PPO outside clip: `2.810% / 8.715% / 8.094%`
- final KL: `0.17089`
- nonfinite or rejected steps: `0`

Three nominally identical runs produced different model byte hashes. The two
gated repeats nevertheless differed by at most `0.1574` percentage points in
per-step PPO outside-clip ratio, `0.0625` in gradient norm, and `0.000504` in
per-step KL. Bit-identical weights are therefore not the release contract. The
contract is pinned runtime identity plus bounded policy-health and held-out
quality. This matches the known cached/full-forward kernel sensitivity.

## CODE Sentinel

CODE was screened with its actual single-phase generation path, full `32768`
cap, pinned curated dataset revision, gVisor grader, common prompts, and common
forced draws. No math BFT was applied.

| Prompt set | Source reward | Candidate reward | Reward transitions | Source/candidate ramble | Termination |
|---|---:|---:|---:|---:|---:|
| First 8 | `87.50%` | `83.93%` | `+0 / -1 / =7` | `25.0% / 12.5%` | `100% / 100%` |
| Disjoint 8 | `65.63%` | `84.38%` | `+2 / -0 / =6` | `25.0% / 12.5%` | `87.5% / 87.5%` |
| Combined 16 | `76.56%` | `84.15%` | `+2 / -1 / =13` | `25.0% / 12.5%` | `93.75% / 93.75%` |

One rollout hit the `32768` cap in both source and candidate. The recovery
update does not regress CODE and improves this small sentinel overall, but it
does not solve CODE's single-phase cap-running tail. That remains a separate
protocol design question.

## Safety Controls Added

- Pre-step nonfinite, gradient-norm, and PPO-ratio circuit breakers.
- Per-environment reward, token-mass, and policy-health telemetry.
- BFT forced/natural path exposure and validated force-span masking metrics.
- Runtime, source, archive, model, and candidate byte provenance.
- Append-only reset publication with source and artifact hashes plus an HF
  parent-commit race guard.
- Publication retry that preserves the exact in-memory candidate and applies no
  extra optimizer step while HF is unavailable.
- Restart-persistent checkpoint ceiling.
- Health fields for publication-pending and registration-cache state.
- Unique grader runsc identities and metrics-socket shutdown cleanup.

## Production Contract

```dotenv
RELIQUARY_RESUME_FROM=sha:<checkpoint-34-commit>
RELIQUARY_KL_BASE_MODEL=Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc
RELIQUARY_KL_BETA=0.01
RELIQUARY_LEARNING_RATE=0.000003
RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY=true
RELIQUARY_GRAD_NORM_SKIP_THRESHOLD=50
RELIQUARY_PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD=0.1
RELIQUARY_SHAPE_PENALTY=0
RELIQUARY_SHAPE_LEN_FRAC=0.5
RELIQUARY_TRAINING_RUN_ID=qwen35-2b-recovery-20260716
RELIQUARY_WANDB_VERSION=training-recovery-20260716
RELIQUARY_TRAIN_UNTIL_CHECKPOINT_N=35
RELIQUARY_DISABLE_TRAIN=1
```

## Deployment Sequence

1. Merge the recovery PR and wait for its immutable validator image.
2. Install the complete contract above while training remains frozen.
3. Start the recovery image on checkpoint 33 and verify all health identities.
4. Dry-run the reset publisher against immutable checkpoint 15.
5. Publish those exact source weights as append-only checkpoint 34.
6. Set `RELIQUARY_RESUME_FROM=sha:<checkpoint-34-commit>` and restart frozen.
7. Verify checkpoint number, source manifest, artifact hashes, fixed-base
   identity, beta, LR, shaping, PPO gate, ceiling, registration, and R2 health.
8. Remove only `RELIQUARY_DISABLE_TRAIN`.
9. Monitor successful balanced steps until checkpoint 35 is published. The
   ceiling must then retain data and refuse further optimizer steps.
10. Run paired math and CODE screens on the actual checkpoint-35 revision.
    Continue only through a new explicit decision.

## Abort Conditions

Re-freeze immediately if any of the following occurs:

- nonfinite gradient or policy ratio
- gradient norm above `50`
- PPO outside-clip ratio above `0.10`
- fixed base, behavior policy, beta, LR, shape, or source identity mismatch
- checkpoint publication remains pending or retries a different candidate
- large unexplained movement in reward, BFT force rate, bad termination,
  rambling, or completion length
- registration cache becomes unusable, archive uploads fail, or validator
  health degrades
- checkpoint 35 regresses outside paired holdout uncertainty

An aborted training step must not advance optimizer, scheduler, publication
cadence, or checkpoint number. Validation, rewards, archives, and weight setting
remain online during any freeze.

## Verification Record

- Full CPU-contract suite: `1290 passed, 8 skipped`.
- CUDA-visible affected training suite: `38 passed`.
- Dedicated GPU proof suite: `7 passed, 1 known failure`.
- Lint: all changed Python files pass apart from the repository's pre-existing
  `constants.py` import-placement exceptions.
- Compile and `git diff --check`: clean.

The one GPU proof failure is the previously documented GRAIL-v7 wrong-model
sketch weakness: the broad `5000` tolerance can accept random-model sketches.
It is not introduced by this release and cannot be repaired validator-only
without breaking miners. It remains a coordinated GRAIL-v8 item; forced-seed,
token-distribution, termination, and reward verification continue to provide
independent checks in the current protocol.
