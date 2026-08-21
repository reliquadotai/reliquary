# Detached Trainer — Operator Runbook

Design: `docs/superpowers/specs/2026-08-21-detached-trainer-r2-design.md`.
Plan: `docs/superpowers/plans/2026-08-21-detached-trainer-r2.md`.

The validator writes one training payload (or tombstone) per sealed
window to R2 under `reliquary/training/`; a standalone
`reliquary train-worker` process consumes them strictly in order, runs
`train_step`, and publishes checkpoints to HF (miners' source) plus an
R2 mirror (validator's fast path). The validator polls the candidate
manifest, downloads the mirrored snapshot in the background, and swaps
its verify plane on the serial publication beat.

## Flags

| env | component | effect |
|---|---|---|
| `RELIQUARY_WRITE_TRAINING_PAYLOADS=1` | validator | write payloads + tombstones and start the upload worker. Independent of the cutover — required for shadow mode. |
| `RELIQUARY_DETACHED_TRAINER=1` | validator | skip in-process train/publish; poll + stage + swap trainer checkpoints. Requires the writer flag too. |
| `RELIQUARY_TRAINER_BOOTSTRAP_CURSOR=<window_n>` | trainer | first-run journal start (refused to guess). Only read when no candidate manifest exists yet. |
| `RELIQUARY_TRAINER_STATE_DIR` | trainer | staging + resume directory (default `/root/reliquary/trainer`). |
| `RELIQUARY_TRAINER_WINDOW_STRIDE` | trainer | journal stride (default 1, matching live window numbering). |
| `RELIQUARY_HF_REPO_ID` + R2 env (`R2_*`) | trainer | checkpoint repo and payload bucket — same names as the validator. |

Both flags default OFF; flag-off is byte-identical to main.

## Topology

Same image everywhere; placement is deployment-only:

| setup | validator | trainer |
|---|---|---|
| 1× H100 (fallback) | flags off — today's in-process path | — |
| 1 box, 2 GPU | flags on, proofs on `cuda:0` | own container, `CUDA_VISIBLE_DEVICES=1` |
| 2 boxes | flags on | any box with a GPU + R2/HF access |

Checkpoint transfers use multipart parallel R2
(`TransferConfig` 32 MiB × 16) — measured 147 MB/s down / 120 MB/s up on
the prod box (single-stream is ~20 MB/s: per-connection window, do not
"simplify" this away). 8 GB ≈ one minute.

## Cutover procedure (shadow first)

1. Deploy with `RELIQUARY_WRITE_TRAINING_PAYLOADS=1` only. Verify
   `reliquary/training/window-*.npz` objects appear each window and the
   queue stays shallow (`/state` publish block).
2. Start the trainer on its box in shadow:
   `reliquary train-worker --shadow` with
   `RELIQUARY_TRAINER_BOOTSTRAP_CURSOR=<current window>`. It trains but
   never publishes.
3. Compare wandb telemetry (loss, grad-norm, `train/ppo_ratio_*`)
   between the in-process run and the shadow trainer over ≥16 windows.
   They consume the same payloads; curves must track.
4. Cutover, in one deploy: validator gets `RELIQUARY_DETACHED_TRAINER=1`;
   trainer restarts WITHOUT `--shadow`. The trainer's first publish
   writes the candidate manifest; the validator stages and swaps on the
   next serial beat.
5. Watch: `windows_since_checkpoint_swap` in `/state` (staleness — the
   headline degradation metric; alert at 3× the publish interval),
   `train/ppo_ratio_outside_clip_ratio` (drift breaker backstop at 0.5),
   payload queue depth, trainer backlog (cursor vs live window).

## Recovery

* **Trainer crash/restart**: it resumes from the candidate manifest —
  loads that revision (R2 mirror, HF fallback), reads
  `trained_window_cursor` + `lr_schedule_step` from the checkpoint
  PROFILE (authoritative), and replays the ≤15 unpublished windows from
  R2. Nothing is lost except optimizer moments.
* **Trainer box lost**: rent any box, same env, same command. Payload
  retention must cover the outage (lifecycle rule: keep ≥7 days AND
  everything above the last published cursor).
* **Dual trainer**: the illegitimate instance exits with code 3
  (`TrainerLockLost`) when the HF repo HEAD is not its own last publish.
  Never run two non-shadow trainers.
* **Validator restart under the flag**: it must start FROM the currently
  published revision (the intake refuses to relabel stale train-model
  weights — a divergence is a `FatalProofPlaneError` restart, never a
  wrong-weights proof).

## Rollback

Set both flags to 0 and restart the validator: the in-process path is
untouched code. Stop the trainer. Payloads already in R2 are inert.
