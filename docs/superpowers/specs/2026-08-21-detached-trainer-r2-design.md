# Detached Trainer via R2 — Design

Date: 2026-08-21
Status: draft for review
Scope: validator + new trainer service; zero wire/protocol change; zero
miner-visible rule change (one timing improvement: shorter cycles).
Baseline: main @ 80c112f (PR #188). Stacks on PR #185 (pipelined window
collection) but degrades gracefully without it.

## Problem

Even with pipelined collection (PR #185), the GPU half is serial:
proofs (~100 s) then train (~115 s) on the same device. Cycle floor
≈ 215 s. One window = one DAPO update, so updates/day is capped at
~395. The measured v4 lever is updates/day (155 real updates vs ~5000
for a reference DAPO run), so the train segment itself is the target.

## Goal

Move `train_step` off the validator's GPU and out of its window loop
entirely, onto a second H100 that may live in the same pod or in a
different datacenter. Target steady-state cycle:

    validator: max(collection 100 s, proofs ~100 s)  ≈ 105-110 s
    trainer:   ~115 s/update, concurrent, on its own card
    effective: ~115 s/window → ~750 updates/day (~2.3× today)

Amortized cost of the publish serial beat (~1 serial window per 16):
~5 s/window, included in the numbers above.

## Non-goals (hard constraints)

1. NO speculative proving. Proofs only ever run on a SEALED window.
   Unchanged from PR #185 non-goal #1.
2. NO change to collection deadlines, auction, tie-break, admission,
   payment, archive schema, or publish cadence semantics.
3. NO miner-visible checkpoint announcement before the validator's own
   verify plane holds the announced weights (download → load → announce,
   never announce-then-download; a failed download must cost staleness,
   never a burned window).
4. ONE trainer instance, structurally enforced (see single-writer guard).
5. NO gaps in the update sequence. A skipped window is always an
   explicit, counted event (tombstone), never a timeout race.
6. ONE numeric path for π_old regardless of topology. π_old is shipped
   in the payload; the trainer never recomputes it. (No topology flag —
   a flag would make the PPO ratio denominator depend on deployment and
   break R2 replay comparability.)

## Architecture

```
VALIDATOR (proof plane, box A)              TRAINER (policy plane, box B or A)
──────────────────────────────              ──────────────────────────────────
loop: collect(N+1) ‖ prove(N)               loop:
at seal(N):                                   payload = R2.get(window cursor+1)
  write training payload → R2                 absent    → sleep, retry
  (or tombstone if aborted)                   tombstone → cursor++, count
pay / archive as today                        else      → train_step → cursor++
                                              every 16 updates → publish ckpt
poll for candidate checkpoint  ◄────────────  (HF upload + candidate manifest)
download → load → swap on serial beat
→ sign manifest → announce in /state
```

The two planes never speak to each other. R2 is the only channel:
training payloads flow right, checkpoints flow back left (via HF, with
an R2 mirror option). Killing either side leaves the other fully
functional: miners are paid and proven with a stale-but-valid
checkpoint; the trainer catches up from the durable payload backlog.

Why concurrent training is sound: π_old is anchored to the PUBLISHED
checkpoint (frozen for 16 windows, `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS`),
not to the live policy. Training window N on card B while card A proves
N+1 against the published revision cannot corrupt either side. This is
the invariant the whole design rests on.

## Component 1 — Training payload

One R2 object per sealed window, written at seal time (after
`commit_seal_side_effects`, alongside the archive enqueue):

    reliquary/training/window-{window_start}.npz        (payload)
    reliquary/training/window-{window_start}.tombstone  (aborted window)

The `training/` prefix is deliberately DISJOINT from the archive prefix
the dashboard consumes (`reliquary/dataset/window-*.json.gz`). Dashboard
listings and pulls are untouched by this design. This is also why the
payload is a new object rather than extra fields on the archive: the
archive is the dashboard's data source, and per-token float arrays
appended there would bloat every dashboard pull for data it never
reads. The archive keeps its schema, byte-identical.

Content (per env, groups in `select_batch` order; per rollout):

| field | dtype | source | consumer in train_step |
|---|---|---|---|
| tokens | int32, ragged | commit["tokens"] | forward input |
| prompt_length | int32 | commit.rollout | completion slicing |
| completion_length | int32 | commit.rollout | keep-mask |
| pi_old_logprobs | **fp32, log-space** | `_validated_completion_logprobs` (proof forward, PR #184) | PPO denominator |
| token_logprobs | fp32 | miner claim (commit.rollout) | overlong length + `RECOMPUTE=0` incident fallback |
| reward | fp32 | graded reward | advantages |
| forced / truncated | bool | commit.rollout | `_training_rewards` zeroing + masks |
| env_name | str | batch structure | per-env `w_e/N_e` normalization |

Window-level header: `window_start`, `checkpoint_revision` (behavior
checkpoint the window was generated against), `env_order`,
`window_quarantine` (the seal-time `assess_training_batch` verdict —
computed validator-side because it needs reject counts), and
`schema_version`.

Rules:

* **fp32, log-space, non-negotiable for pi_old_logprobs.** The ratio is
  `exp(logπ_θ − logπ_old)`; fp16 truncation of the denominator shifts
  every ratio silently.
* The T=1.0 gate from `_verify_logprobs_for_training` (batcher.py:279)
  is checked at WRITE time: if `T_PROTO != 1.0` or coverage is partial,
  the field is omitted and the trainer falls back to the miner-claim
  `token_logprobs` path (same fallback ladder as today).
* Size: ~1.2 MB/window median (512 traj × ~600 tokens × fp32), ~17 MB
  worst case (all at 8192 cap). Retention 7 days AND above the last
  published cursor, whichever keeps more. Steady state ≈ 8 GB ≈
  $0.12/month. The forensic archive (which stores full completion
  text, untrimmed) remains the bigger object.
* Tombstones: every sealed-or-aborted window produces exactly one of
  payload/tombstone. `_enqueue_aborted_window` call sites gain the
  tombstone write. The trainer NEVER advances on timeout — "no object
  yet" always means wait. This is what makes a lost update impossible
  to miss.

## Component 2 — Trainer service

New CLI entrypoint `reliquary train-worker` (own process, own
container; `CUDA_VISIBLE_DEVICES` selects its card). It owns everything
that today lives between "sealed batches exist" and "checkpoint
published" in `_train_and_publish` (service.py:2086):

* `TrainingAccumulator` (add_window / ready / training_batches / reset)
* window + accumulated quarantine handling (`assess_training_batch` on
  the accumulated batch; the window-level verdict arrives precomputed
  in the payload header)
* `train_step` with its health gates (`TrainingStepSkipped`,
  grad-norm skip, `policy_ratio_drift` → adaptive early publication)
* LR schedule ownership (`current_lr_schedule_step`,
  `_lr_global_step_hint` equivalent)
* checkpoint publication via `CheckpointStore.publish` mechanics
  (HF upload + `write_checkpoint_profile`)
* wandb telemetry (same run naming)

Main loop (pseudocode):

    cursor = restore_cursor()          # see Recovery
    loop:
        obj = r2_get(cursor + 1)
        if absent:    sleep(5); continue
        if tombstone: record_skip(cursor+1); cursor += 1; continue
        batches = decode(obj)          # includes checkpoint_revision check
        accumulator.add_window(...)    # quarantine gates as today
        if accumulator.ready and not publication_pending:
            train_step(...)            # health gates as today
        cursor += 1
        if trained_since_publish >= 16 or adaptive_pending:
            publish()

`checkpoint_revision` consistency: the payload header carries the
behavior revision; the trainer rejects (and loudly reports) a payload
whose revision does not match the checkpoint lineage it is training —
the same guard `_train_and_publish` applies across envs today, extended
across the wire.

### Publication and the candidate manifest

The trainer's `publish()` = save HF format + `write_checkpoint_profile`
(with `lr_schedule_step` AND the new `trained_window_cursor`) + HF
upload — reusing `CheckpointStore`'s save/upload path. It then writes a
small **candidate manifest** object to R2:
`reliquary/training/candidate-manifest.json` =
`{checkpoint_n, repo_id, revision, trained_window_cursor}`.

Signing stays with the validator. Today `ManifestEntry.signature` is
the validator wallet's ed25519 over `(checkpoint_n || revision)` — an
attestation of "this is MY current checkpoint". The trainer does not
hold the wallet and must not. The validator signs at swap time, after
it has downloaded, profile-validated (`validate_checkpoint_profile`)
and loaded the weights — which is exactly when the attestation becomes
true. Trainer→validator authenticity rides on R2 bucket credentials
plus the profile lineage check; a stronger trainer-key signature on the
candidate manifest is a v2 option, noted in Open Questions.

### Single-writer guard (anti dual-trainer)

Before every publish, the trainer verifies that the HF repo HEAD equals
the revision IT last published (or the revision it resumed from).
Mismatch → halt loudly (process exit, alert), never overwrite. This
converts the "old box comes back to life" scenario from silent
divergence into a visible crash of the illegitimate instance.

### Recovery

Commit point = published checkpoint (every 16 windows). Journal = R2
payloads. On start:

1. Load the last published checkpoint (local staging copy if present,
   else HF) + its profile → `lr_schedule_step`, `trained_window_cursor`.
2. Restore optimizer state from the local snapshot saved at last
   publish, if the disk survived; else fresh moments (same behavior as
   a validator crash today — tolerated by the run; exact-state upload
   to R2 is a v2 option).
3. Resume the loop at `cursor+1`. The 0-15 unpublished updates are
   REPLAYED from the same payloads starting from the same published
   weights — nothing is lost.

Total box loss = rent any box, point the image at R2+HF, same three
steps. The validator never notices beyond checkpoint staleness.

## Component 3 — Checkpoint return path (validator side)

The validator loses `train_step` and gains a small poller + a swap
condition on the existing serial beat:

1. **Poll** (background, cheap): read `candidate-manifest.json` each
   window. New revision → start background download (HF; R2 mirror of
   the snapshot is an option to avoid HF rate limits — Open Questions).
2. **Stage**: download to disk, `validate_checkpoint_profile`, load
   into RAM/GPU staging. All off the window loop's critical path.
3. **Swap on the serial beat**: the existing publish-serial-beat
   machinery (`_publication_due_next_half` forecast + in-half deferral)
   is repointed: "publication due" becomes "staged candidate ready".
   On a serial iteration: install weights into `verify_model`
   (`_refresh_verify_model_from_train` becomes install-from-staged),
   `_synchronize_proof_models` (drain → refresh replicas → resume),
   sign `(checkpoint_n || revision)`, install `ManifestEntry`,
   `server.set_current_checkpoint`. Miners see the flip exactly as
   today: between collections, never mid-window.
4. **Degraded**: download slow/failed → candidate stays staged/absent,
   validator keeps proving and announcing the current revision. Cost is
   staleness only. Staleness (windows since last swap) is exported as a
   first-class metric with an alert threshold (default: 3× the publish
   interval), because the drift budget (`eps_high=0.28`, drift breaker
   at 0.5) was calibrated for ~16 windows of π_old lag.

Nominal-case timing note: with the trainer publishing right after its
16th update and download running in the background, the swap lands on
the same serial beat it would in-process at ≥~300 Mbps. The gating only
bites on degraded links — by design.

## Validator loop changes

* `_train_and_publish` keeps: beacon check, seal, rewards, verdicts,
  telemetry, archive, READY transitions. Loses: accumulator,
  `train_step`, `CheckpointStore.publish`, verify-refresh-from-train.
* Adds: payload/tombstone write at seal (async, via the existing
  `archive_queue` worker pattern — R2 outage buffers on disk exactly
  like archives do).
* The `train_model`/`verify_model` duality collapses: the validator
  holds ONE frozen model per proof device (the published checkpoint).
  On a 1-GPU box this frees the ~8 GB train-model residency and the
  optimizer/activation peaks — extra allocator headroom on the proof
  card (the ~88 % allocator cliff, memory 2026-08-10).
* Feature flag `RELIQUARY_DETACHED_TRAINER=1`. OFF = today's in-process
  path, byte-identical (same discipline as `RELIQUARY_PIPELINED_WINDOWS`;
  the existing `RELIQUARY_DISABLE_TRAIN` escape hatch already proves
  the loop runs correctly with train removed).

## Topology

Decided entirely by deployment env, zero code difference:

| setup | validator | trainer | checkpoint hop |
|---|---|---|---|
| 1× H100 (fallback) | flag OFF | in-process | in-memory copy |
| 1 box, 2 GPU (pod) | cuda:0 proofs | own process, cuda:1 | local disk |
| 2 boxes (target) | box A | box B | HF (+ optional R2 mirror) |

The 2-box link budget: payloads ~1-17 MB/window outbound (trivial);
checkpoint 8 GB per publish interval (~30 min).

**Measured 2026-08-21 on the live validator box (209.20.157.231),
after enabling gzip on /state (see below):**

| path | measured | verdict |
|---|---|---|
| HF download | 1.7-7.9 MB/s single, 9.3 MB/s over 4 streams | **HF↔DC peering is bad; unusable for the 8 GB hop** |
| R2 / OVH, single stream | ~21-31 MB/s | per-CONNECTION limit, not the pipe |
| **R2, 8 parallel ranged GETs** | **245.8 MB/s aggregate** | the pipe is ≥2 Gbps |
| **R2, boto3 `TransferConfig` ×16 (the implementation API)** | **147 MB/s down / 120 MB/s up** | **8 GB ≈ 55-65 s** |

Root cause of the single-stream ceiling: per-connection receive
window (`net.core.rmem_max` = 208 KB on the box) × 11 ms RTT to R2
≈ 19 MB/s per flow. App-level parallelism sidesteps it entirely —
no host sysctl tuning required (an optional `rmem_max`/`wmem_max`
bump would lift single-flow rates but is not needed).

Consequences:

* **The R2 checkpoint mirror is REQUIRED, not optional** (resolves
  former Open Question 1): the trainer uploads each checkpoint to
  BOTH HF (miners' source, unchanged) and R2; the validator pulls
  from R2. All checkpoint R2 transfers MUST use multipart parallel
  transfer — `TransferConfig(multipart_chunksize=32 MiB,
  max_concurrency=16)` measured 147/120 MB/s on the live box.
  **8 GB ≈ one minute, fully inside one window's background time →
  swap defers ≤1 window; effective refresh stays ~16-17 windows.
  Nominal.** (HF stays unusable even parallelized — 9.3 MB/s
  aggregate — the peering is the bottleneck there.)
* Related live finding: nginx served /state (105 KB × ~330 req/s ≈
  285 Mbps, uncompressed — `gzip_proxied`/`gzip_types` were commented
  out). **Fixed in prod 2026-08-21** (backup:
  `/etc/nginx/nginx.conf.bak-2026-08-21`): /state now 44 KB gzipped,
  egress ~285 → ~125 Mbps. Miners' HTTP clients negotiate gzip
  transparently; watch per-hotkey submission rates post-flip for any
  hand-rolled client that mishandles it.
* Today's IN-PROCESS publish uploads 8 GB to HF through the same bad
  HF path, inside the window loop — the next live publish's wall time
  should be observed (a median-cycle metric hides a 1-in-16 stall).
  Detaching moves that upload to the trainer's box entirely.
* Still worth asking the provider about the link provision on box A;
  every MB/s directly shortens the swap deferral.
* Trainer-box bandwidth (upload 2×8 GB per interval: HF + R2) must be
  measured before cutover — same protocol as above.

## Failure matrix

| failure | behavior | data loss |
|---|---|---|
| trainer crash | validator unaffected; backlog accumulates; replay on restart | none (replayed) |
| trainer box lost | same + re-rent + resume from HF profile | optimizer moments only |
| R2 outage | payload writes buffer on validator disk (archive_queue pattern); trainer idles | none |
| HF outage at publish | trainer retries; validator keeps current revision | staleness |
| download failed/slow | swap deferred; staleness metric rises | none |
| dual trainer | illegitimate instance halts on HEAD mismatch | none |
| validator crash | unchanged from today; trainer idles on missing payloads | unchanged |
| payload gap | impossible by construction (tombstones); trainer waits forever on a truly missing object → staleness alert fires | none |

## Observability

* Trainer: cursor, backlog depth (windows behind), updates/day,
  publish latency, health-gate skips, replay count on restart —
  wandb + structured logs as today.
* Validator: payload write success/buffer depth, candidate-manifest
  age, download state, **checkpoint staleness in windows** (the
  headline degradation metric), swap deferrals.
* /state: `training_accumulator` block is replaced by a `trainer`
  block (last payload written, last swap, staleness). Wire-compat per
  the /state extra=forbid convention (new response fields are fine;
  no new required query params).

## Testing strategy

1. Unit: payload encode/decode **round-trip equality** — a decoded
   payload must reconstruct group/rollout structures that drive
   `train_step` to bit-identical loss vs. the in-process path on the
   same fixture (this is the guard against the silent-field-loss
   failure mode). Tombstone sequencing. Cursor restore from profile.
   Single-writer guard. T=1.0 write-gate.
2. Integration: full loop with a stub R2 (filesystem) — seal → payload
   → trainer step → candidate manifest → staged swap on serial beat.
   Crash-replay: kill trainer mid-interval, restart, assert final
   weights equal the uninterrupted run (modulo optimizer moments).
3. Shadow (production gate): run the detached trainer AGAINST live
   payloads while the in-process path still trains. Compare per-window
   loss/grad-norm/ratio telemetry for N windows. Cut over only on
   agreement; the flag makes rollback a restart.

## Rejected alternatives (do not re-litigate)

* **Push RPC validator→trainer**: new authenticated service in the
  learning path, hand-rolled backpressure, no durable buffer, no
  replay. R2 gives all four for free.
* **Recompute π_old on the trainer**: pays 10-15 s/update (~10 % of
  the target cycle — the exact measured cost PR #184 removed) plus a
  frozen 8 GB replica and per-window pinned-revision bookkeeping, to
  avoid shipping ~1.2 MB. Also considered as a topology flag
  (same-box reuse / cross-box recompute): rejected because it makes
  the PPO denominator deployment-dependent and forks the numeric path.
* **Micro-batch streaming (train each group as its proof passes)**:
  zero throughput gain (pipeline throughput = slowest stage regardless
  of intra-stage granularity), and the per-token loss scale
  `s_b = w_e/N_e` (training.py:699) needs the full batch's token count
  before any backward — working around it means per-env grad buffers
  (~16 GB) and a `train_step` rewrite for a latency win that is noise
  against the 16-window publish interval. It would also re-couple the
  planes in streaming fashion, destroying the R2 decoupling.
* **Mega-windows (16× collection, then batch verify+train)**: same
  throughput (collection is miner generation time; per-window overhead
  is already hidden by pipelining), 16× incident blast radius at ~3
  proof-plane restarts/day, 16× miner feedback latency, and a
  miner-visible protocol redesign of the per-window economics.
* **Fusing 16 windows into 1 big update**: divides updates/day by 16 —
  the opposite of the v4 lever.

## Scaling note (after this ships)

The proof plane already scales by device (`GlobalProofScheduler`, one
worker per device; `RELIQUARY_PROOF_DEVICES`). With the trainer
detached, the remaining ladder is: 2nd proof device → proofs ~50 s;
DDP on the trainer → train ~65 s; floor = collection 100 s, a protocol
constant. Beyond 3 GPUs nothing helps without a protocol change.
Caveat for heterogeneous/proof-on-cuda:1 setups:
`collect_runtime_fingerprint` reads device index 0 unconditionally
(runtime_fingerprint.py:114-117) — must be parameterized by proof
device before any such deployment.

## Open questions

1. ~~R2 mirror of the checkpoint snapshot~~ — RESOLVED 2026-08-21:
   required (see Topology measurements; the HF path to box A is
   unusable for the 8 GB hop).
2. Trainer-key signature on the candidate manifest (defense beyond
   bucket credentials). v2.
3. Optimizer-state upload to R2 at publish (~16 GB) for exact recovery
   after total box loss. v1 ships local-disk snapshot only.
4. Payload compression (zstd on the npz): ~2-3× smaller, trivial CPU.
   Decide at implementation.
5. ~~Lossless checkpoint DELTA (XOR+zstd) for the validator hop~~ —
   MOOT 2026-08-21: multipart parallel R2 transfer already moves 8 GB
   in ~1 min (see Topology). Kept as a note only in case the link
   degrades. Lossy compression stays permanently excluded: the verify
   plane must be byte-identical to what miners load from HF or every
   proof fails.
