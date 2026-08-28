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
| `RELIQUARY_TRAINER_BOOTSTRAP_REVISION=<hf_sha>` | trainer | mid-run bootstrap weights: start from this published checkpoint instead of the base model. REQUIRED for shadow starts and for the cutover (set it to the validator's last published revision). |
| `RELIQUARY_TRAINER_CHECKPOINT_N=<n>` | trainer | bootstrap-only checkpoint number (cutover: set to the validator's current `checkpoint_n` so numbering never regresses). After the first publish the manifest carries it. |
| `RELIQUARY_DISABLE_TRAIN=1` | both | emergency freeze works in the detached path too: the trainer stops consuming/publishing ("frozen"), the validator stops polling/swapping checkpoints. |
| `RELIQUARY_TRAINER_STATE_DIR` | trainer | staging + resume directory (default `/root/reliquary/trainer`). |
| `RELIQUARY_TRAINER_WINDOW_STRIDE` | trainer | journal stride (default 1, matching live window numbering). |
| `RELIQUARY_HF_REPO_ID` + R2 env (`R2_*`) | trainer | checkpoint repo and payload bucket — same names as the validator. |

When the disabled checkpoint-epoch capability is explicitly enabled, every
journal payload also carries its epoch ID, manifest hash, training-run identity,
lane offset, horizon, and selected training mode. The validator uploads a
terminal epoch marker only after all lane payloads/tombstones are durable in
R2. The trainer waits for that marker before consuming lane zero, skips an
aborted epoch atomically, and then either performs the normal sequential steps
or one horizon-wide aggregate step according to the manifest. Ordinary
production payloads retain their existing schema and behavior.

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

This procedure is for a same-profile infrastructure move that continues the
current weights. A reasoning-prompt v5 reset uses the dedicated procedure below
instead of bootstrapping from the last v4-trained revision.

1. Deploy with `RELIQUARY_WRITE_TRAINING_PAYLOADS=1` only. Verify
   `reliquary/training/window-*.npz` objects appear each window and the
   queue stays shallow (`/state` publish block).
2. Start the trainer on its box in shadow:
   `reliquary train-worker --shadow` with
   `RELIQUARY_TRAINER_BOOTSTRAP_CURSOR=<window of the last publish>` and
   `RELIQUARY_TRAINER_BOOTSTRAP_REVISION=<last published HF sha>` (from
   `/state`). Same starting weights as prod's train model, same
   payloads, same order — the curves are comparable. It trains but
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

## Reasoning-prompt v5 fresh reset

V5 changes the canonical miner prompt and starts a new training run. Do not use
the generic mid-run bootstrap above with the last v4-trained checkpoint.

1. Finish and archive the final v4 window. Record that window as the v5 trainer
   bootstrap cursor; no payload at or below it may enter the v5 run.
2. Publish the pinned Qwen3-4B-Base weights as the next append-only checkpoint
   under `qwen3-4b-base-dapo-reasoning-v5`. Record its HF revision and checkpoint
   number.
3. Set the same `RELIQUARY_PROTOCOL_PROFILE` and new
   `RELIQUARY_TRAINING_RUN_ID` on validator and train-worker. On the worker set
   the explicit bootstrap revision to the v5 base reset, the cursor to the final
   v4 window, and the checkpoint number to the base reset number.
4. Restart the validator from that same v5 base-reset revision with payload
   writing enabled. Start the H100 worker with `--shadow` first. Protocol-v5
   payload identity prevents it from consuming legacy v4 journal objects.
5. Compare the shadow step, memory, loss, ratio, and termination telemetry. Then
   restart the worker without `--shadow` and enable detached intake on the
   validator. The first v5 publish replaces the old candidate manifest.

Candidate manifests and payloads now carry profile ID, protocol version,
training-run ID, and generation-contract hash. A stale v4 manifest is ignored
in favor of the explicit v5 bootstrap; a mismatched payload fails closed rather
than advancing the journal cursor.

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
