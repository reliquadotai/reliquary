# Running a Reliquary Validator

Operational guide for running a validator on subnet 81. Both modes deploy via Docker.

## Two modes — pick one

| Mode | Who | Hardware | Auto-update |
|---|---|---|---|
| **Weight-only** | recommended for almost every operator | CPU box, 4 GB RAM, no GPU | Watchtower polls GHCR every 5 min |
| **Trainer** | the Reliquary core team | A100 40 GB+ GPU, 64 GB RAM | manual (sensitive — never restart mid-step) |

While the network is bootstrapping there is exactly **one** trainer and the
core team runs it. Every other operator runs the weight-only mode, which
mirrors the on-chain weight signal from the trainer and earns validator
emission without any of the GPU cost or coordination overhead.

---

## Weight-only quickstart (5 minutes)

You need:

- A Linux host with Docker 24+ and the Compose plugin.
- A Bittensor wallet registered on netuid 81 (only the hotkey reaches this box — coldkey stays offline).
- R2 read credentials (the trainer publishes window archives to R2; you read them).

```bash
git clone https://github.com/reliquadotai/reliquary.git
cd reliquary/docker
cp .env.example.weight-only .env
# Edit .env with your values (see "What goes in .env" below)
export BT_WALLETS_DIR=/path/to/validator-signing-wallets
docker compose -f docker-compose.weight-only.yml up -d
```

That's it. Watchtower will pull and restart your container automatically every time a new image is published.

### What goes in `.env`

The example file is annotated. The required keys:

```bash
BT_NETWORK=finney
BT_NETUID=81
BT_WALLET_NAME=<your-wallet-name>
BT_HOTKEY=<your-hotkey-name>

RELIQUARY_TRAIN=0                          # weight-only mode — DO NOT change

R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_ID=reliquary
R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
```

`RELIQUARY_TRAIN=0` is what makes this a weight-only deployment — the entrypoint reads it and starts in the right mode. **Don't change it to `1` unless you are the trainer.**

Existing deployments do not need a new wallet-path variable. `BT_WALLETS_DIR`
is still the host-side credential source. `BT_WALLET_PATH` is optional and
only changes the in-container mount target for custom deployments.

### Verify it's running

```bash
# Validator container is up and submitting weights
docker logs -f reliquary-weight-only

# Watchtower is polling GHCR
docker logs watchtower | tail -20
# Expect periodic "Checking containers for updated images" lines
```

---

## Trainer quickstart


You need:

- A GPU host with NVIDIA driver, CUDA 12.8+, and the NVIDIA Container Toolkit.
- 1× A100 40 GB minimum, 64 GB RAM, 150 GB disk.
- A public IP and an open inbound TCP port (default 8080) — miners must reach you.
- HF Hub token with **write** access to your checkpoint repo.
- R2 **write** credentials.

```bash
git clone https://github.com/reliquadotai/reliquary.git
cd reliquary/docker
cp .env.example.trainer .env
# Edit .env (see below)
export BT_WALLETS_DIR=/path/to/validator-signing-wallets
docker compose -f docker-compose.trainer.yml up -d
docker logs -f reliquary-trainer
```

Trainer-specific `.env` keys (full list in `.env.example.trainer`):

```bash
RELIQUARY_TRAIN=1
RELIQUARY_HF_REPO_ID=your-org/reliquary-sn   # HF repo to push checkpoints to
HF_TOKEN=hf_xxx                              # write access to that repo
RELIQUARY_EXTERNAL_IP=<your-public-ip>       # advertised on-chain
RELIQUARY_EXTERNAL_PORT=8080
# Optional — resume after a restart so miners don't reset to base:
# RELIQUARY_RESUME_FROM=sha:<40-hex-hf-commit>
```

The CLI compatibility default remains `openmathinstruct`, but the production
auction contract is mixed Math+Code. Configure the trainer explicitly:

```bash
RELIQUARY_ENVIRONMENTS=openmathinstruct,opencodeinstruct
```

Both validator and miner load the same public curated dataset
(`R0mAI/opencodeinstruct-curated`, pinned by default) lazily — the
`structured_cases` ship with it, and the validator runs the grader and
recomputes the code reward authoritatively. Auction, deferred proof, resource
caps, and operator/prompt dedup apply independently to both environments. Do not start
the mixed trainer until the image contains the grader rootfs, `runsc` starts
successfully, and the loopback grader canaries pass.

Training recovery also requires the complete pinned policy contract:

```bash
RELIQUARY_KL_BASE_MODEL=Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc
RELIQUARY_KL_BETA=0.01
RELIQUARY_LEARNING_RATE=0.000003
RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY=true
RELIQUARY_GRAD_NORM_SKIP_THRESHOLD=50
RELIQUARY_PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD=0.1
RELIQUARY_SHAPE_PENALTY=0
```

### Cooldown on training restart

The prompt cooldown is restored at startup from a run-keyed snapshot on R2, so
the full cooldown survives a restart. Key it with `RELIQUARY_TRAINING_RUN_ID`
(default `default`): keep it stable while a training run continues, and **bump
it to a fresh value when you start a new training from scratch** so the cooldown
resets to zero — a fresh model must be allowed to re-see every prompt.


## Sanity checks (both modes)

```bash
# Health
curl http://localhost:8080/health
# → {"status":"ok","active_window":42}

# State (trainer only — weight-only doesn't expose HTTP)
curl http://localhost:8080/state

# Real-time per-submission verdicts for a given miner hotkey (trainer only).
# Use to confirm a specific miner is being accepted (or what reject reason
# they're hitting) without waiting for the post-window R2 archive upload.
curl 'http://localhost:8080/verdicts/<miner_hotkey_ss58>?since=0'
# → {"verdicts":[{"merkle_root":"...","window_n":N,"accepted":true,"reason":"accepted","ts":...}, ...]}
```

For the weight-only mode, the only signal that things are working is the log line `Submitting weights: N miners …` once per subnet epoch (~30 minutes on netuid 81).

### `/verdicts/{hotkey}` — what to expect

The trainer exposes the last `VERDICT_CAP_PER_HOTKEY = 200` lifecycle verdicts per miner hotkey via a small in-memory ring buffer:

- HTTP-level early rejects (`rate_limited`, `window_not_active`, `batch_filled`)
- Worker admission outcomes after bounded checks and reward grading
- Auction-seal outcomes with final rank, deferred-proof result, selection, and reward flags
- Worker drains on window swap (`worker_dropped`)
- Inline accepts under TestClient (`accepted`)

An admission `accepted` is not a win. The final auction record is the one with
non-null `selected_for_batch` and `rewarded`. Public read is intentional and
uses the same trust model as the R2 archive.

For submit lifecycle fields, drand timing interpretation, `batch_filled`
reasons, and final selected vs rewarded semantics, see
[Validator Observability Notes](validator_observability.md).

---

## Troubleshooting

| Symptom | What to check |
|---|---|
| `BT_WALLET_NAME is required` at startup | `.env` not loaded or variable empty. Confirm `env_file: .env` resolves and the file is in the same dir as the compose file. |
| Container restarts in a loop | `docker logs <container>` — usually invalid R2 credentials, missing HF token (trainer), or wallet mount path wrong. |
| Weight-only: no weight submissions logged | Check `validator_permit` in the metagraph. Without it, `set_weights` is a no-op. |
| Trainer: miners not submitting | Confirm `RELIQUARY_EXTERNAL_IP` matches your real public IP and the host firewall allows inbound on `RELIQUARY_HTTP_PORT`. |
| Trainer: high `WRONG_CHECKPOINT` rate sustained | Miners are not polling `/state` often enough. Brief spikes after each publish are normal. |
| Watchtower never updates | Check the `com.centurylinklabs.watchtower.enable: "true"` label survived your edits to the compose file, and that watchtower itself is running (`docker ps`). |
| HF publish failing (trainer) | Verify `HF_TOKEN` has write access: `huggingface-cli whoami` and try a manual `huggingface-cli upload`. |

For deeper protocol-level issues (high `GRAIL_FAIL`, batches not sealing, EMA drift), see [concepts.md](concepts.md) for the verification pipeline and reject reason reference.

---

## What the validator actually enforces

These are the live thresholds the trainer applies on every submission. The same constants are explained from the miner's perspective in [mining.md](mining.md#rejection-reasons).

| Constant | Value | Effect |
|---|---|---|
| `B_BATCH` | 8 | Maximum proven winners and uniform reward slots per active environment |
| `M_ROLLOUTS` | 8 | Required rollout count per submission |
| `T_PROTO` | 0.9 | Protocol-fixed sampling temperature (validator's recompute uses this) |
| `FORCED_SEED_PROTOCOL_VERSION` | 2 | Mandatory hotkey-free forced stream while enforcement is active |
| `WINDOW_COLLECTION_SECONDS` | 100 | Fixed collection interval for both Math and Code auction populations |
| `MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW` | 96 | Started grading/proof ceiling per environment/window |
| `MAX_PROOF_WALL_SECONDS` | 240 | Seal-time proof wall-clock ceiling per environment |
| `MAX_EXPENSIVE_PROOF_FAILURES_PER_OPERATOR_PER_WINDOW` | 4 | Operator-wide seal GPU debt limit per environment |
| `MAX_SUBMISSION_PAYLOAD_BYTES` | 64 MiB | Per-request parsed JSON payload limit |
| `MAX_PENDING_SUBMISSION_BYTES_PER_HOTKEY` | 128 MiB | Retained pending payload cap per hotkey/environment |
| `MAX_PENDING_SUBMISSION_BYTES_PER_ENV` | 512 MiB | Retained pending payload cap per environment |
| `SIGMA_MIN` (steady) | 0.43 | Zone filter: groups below this are rejected `OUT_OF_ZONE` (binary equivalent: k ∈ [2, 6] for M=8) |
| `BOOTSTRAP_SIGMA_MIN` | 0.33 | Relaxed zone filter during first `BOOTSTRAP_WINDOWS = 100` windows (k ∈ [1, 7]) |
| `BATCH_PROMPT_COOLDOWN_WINDOWS` | 1,000,000 | A winning prompt is effectively one-shot in the OpenMath phase |
| `COOLDOWN_REBUILD_LOOKBACK` | 300 | R2 windows replayed at startup to rebuild cooldown without scanning the whole one-shot horizon |
| `PROOF_SKETCH_TOLERANCE_BASE` | 5000 | GRAIL sketch tolerance — actual threshold = `5000 + 5 × √position` |
| `PROOF_SKETCH_TOLERANCE_GROWTH` | 5.0 | Per-position sqrt growth |
| `LOGPROB_IS_EPS` | 0.10 | Per-token log-prob deviation max — exceeding triggers `LOGPROB_MISMATCH` |
| `MIN_EOS_PROBABILITY` | 0.01 | Required EOS token probability for proper termination |
| `MAX_TRUNCATED_PER_SUBMISSION` | 1 | Steady-state cap/non-EOS truncation allowance; accepted cap hits still pass GRAIL/logprob/distribution/boxed checks |
| `BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION` | 1 | Bootstrap truncation allowance |
| `TRAINING_QUARANTINE_ENABLED` | true | Suspicious selected windows skip GRPO/publish but remain archived/credited |
| `TRAINING_QUARANTINE_MAX_SINGLE_COMPLETION_LENGTH` | 7000 | Rollout length that counts as extreme-length telemetry |
| `TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_ROLLOUTS` | 4 | Minimum long/cap rollouts before length alone can quarantine a window |
| `TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_GROUPS` | 3 | Minimum groups with long/cap rollouts before length alone can quarantine a window |
| `MAX_SEAL_QUEUE_DRAIN_SECONDS` | 60 | Deadline work-drain bound before the auction population freezes |
| `SPARSE_VALID_*` / `WINDOW_TIMEOUT_SECONDS` | legacy fallback | Used when the auction kill switch restores count/idle-based selection |
| `EMA_ALPHA` | ≈0.0274 | Weight-update smoothing (`2/(72+1)` — ~25-window half-life) |
| `REJECTED_LIST_CAP_PER_HOTKEY` | 5 | Max rejected samples retained per hotkey per window archive |

Source of truth: `reliquary/constants.py`. If any of these change, this table and `concepts.md` need a sync.

### Balanced training accumulation

Sparse seals no longer discard otherwise valid gradient signal. The validator
retains at most the configured target for each active environment and trains
only when all targets are present. Pending groups are bound to one checkpoint
revision and are cleared on revision drift, accumulated-batch quarantine, or a
completed or failed training attempt. A process restart also clears this
in-memory buffer; window archives and miner rewards are independent and remain
durable.

Operators can inspect `training_accumulator_checkpoint_revision`,
`training_accumulator_targets`, `training_accumulator_counts`, and
`training_accumulator_ready` in `/health`. Every archive also includes a
`training_accumulator` record with per-window additions, overflow, source
windows, reset reason, and whether a step was attempted.

### Submission pipeline

Every `/submit` flows through this sequence on the validator. The first rejection short-circuits the rest.

Upgraded miners first send a small signed `/submit/precommit` containing the
final body's SHA-256, byte count, routing fields, nonce, checkpoint, protocol,
and current drand round. A receipt accepted before the collection cutoff allows
only that exact body to finish within `SUBMISSION_UPLOAD_GRACE_SECONDS = 33`.
It consumes normal hotkey quota but no prompt or auction slot, so abandoned
precommits cannot squat economic capacity. Direct `/submit` remains valid
before cutoff for compatibility; after cutoff a matching receipt is required.

```
HTTP/pre-queue                 environment worker
--------------                 ------------------
window/checkpoint/protocol     prompt/token/randomness/signature checks
envelope/registration          validator-authoritative reward grading
operator logical claim         zone and cheap authenticity guards
rate/queue/payload bounds      -> pending auction pool
-> reason="submitted"          -> first /verdicts lifecycle record

100 s deadline
-> stop new admission and drain pre-deadline work (max 60 s)
-> freeze Math and Code populations independently
-> fetch post-deadline drand salt
-> rank by difficulty, validator arrival round, sealed operator/prompt tie hash
-> prove top-down under attempt/wall/operator-debt bounds
-> at most 8 distinct prompts; no operator winner cap
-> pay exactly the selected training groups; no boundary split
-> final /verdicts lifecycle records
-> R2 archive + rewards + balanced training accumulator
```

Code grader candidate failures produce legitimate zero rewards. Grader
infrastructure failures are counted separately: retryable outages return
`WORKER_DROPPED` and refund quota, while ambiguous worker crashes fail closed as
`REWARD_MISMATCH` and consume the logical claim.

R2's canonical mechanism payload is `difficulty_auction`; the historical
`difficulty_auction_shadow` field is retained as an identical compatibility
alias. In active mode its `mode` is `production`, not a counterfactual shadow.

The wire-v1 root check is validator-only and defaults to shadow mode
(`RELIQUARY_LEGACY_MERKLE_ROOT_ENFORCE=false`). It recomputes the exact root
current miners already sign, logs a `legacy_merkle_checked` lifecycle stage,
and carries the status into later verdicts. It does not reject until explicitly
enabled. Summarize a captured validator log with:

```bash
python scripts/report_legacy_merkle_shadow.py validator.log \
  --required-env openmathinstruct --required-env opencodeinstruct
```

Do not enable enforcement until the report has at least 500 authenticated
checks, five hotkeys, 24 windows, both active environments, zero compute
errors, and zero unexplained mismatches. `/health` exposes the cumulative
counts and the active enforcement flag.

`/health` also reports the auction policy, per-environment queue/proof state,
operator mapping, forced-seed ratio/CDF policy, Code grader failures, and the
persistent archive queue. A nonzero `archive_queue_depth` is safe during a
transient R2 failure, but growing depth or old
`archive_queue_oldest_age_seconds` requires attention.
`archive_last_uploaded_window` confirms that a recent archive left the retry
queue.

Prompt Parquet range reads prefer exact full files already present in the
persistent Hugging Face cache. If the range backend fails, the validator may
download the same revision-pinned blob once and continue locally. Prewarm both
active sources before a restart with:

```bash
python -m scripts.prewarm_prompt_sources
python -m scripts.prewarm_prompt_sources --verify-only
```

`/health.prompt_sources` reports each source revision, manifest readiness,
range failures, local full-file hits, and fallback downloads. A source that
cannot serve from either path changes health to `degraded`; `/submit` returns a
retryable HTTP 503 and refunds the request's rate-limit reservation. Prompt
source failures are operator outages, not miner protocol verdicts.

Forced-seed CDF enforcement also defaults off. Private schema-v3 calibration
rows bind each observation to its window, environment, and checkpoint and
count CDF misses above 0.01, 0.05, and 0.10. Run:

```bash
python scripts/report_forced_seed_cdf.py
```

Any unexplained hard mismatch produces an immediate
`HOLD_AND_REVIEW_CDF_HARD_MISMATCHES`; the 24-hour, 1,000-group, five-hotkey
threshold is only a minimum for becoming eligible to canary enforcement. Do
not raise the boundary epsilon merely to make the report pass: first separate
environment, checkpoint, forced-span, and numerical-kernel effects using the
schema-v3 fields.

Termination keeps its exact current gate, but interesting low-probability EOS,
natural-close, and cap-truncation decisions are written privately to
`auth_forensics/termination-shadow.jsonl`. The rows include the distance from
the public uniform to the submitted stop token's CDF interval. Summarize them
with:

```bash
python scripts/report_termination_shadow.py
```

`REVIEW_BOUNDARY_CANDIDATES_KEEP_GATE_UNCHANGED` means reproduce those rows on
the matching checkpoint and generation stack. It does not authorize a wider
acceptance interval: a miner can also search for near-boundary injected stops,
so adversarial controls are required before any termination rule changes.

Before `train_step`, the validator runs the training-quarantine gate. If the
selected batch has high-confidence poison signals, the archive still publishes
and emissions remain replayable from `rewards_by_hotkey`, but GRPO is skipped
for that window. Checkpoint publish cadence is counted by successful trained
windows, so a quarantined modulo-boundary window does not by itself freeze the
public checkpoint. The archive field is:

```text
training_quarantine = {quarantined, reasons, metrics}
```

Submissions that get HTTP-accepted but reach the worker after population freeze
are dropped as `WORKER_DROPPED`. They receive a `/verdicts` record, and aggregate
per-hotkey/reason late-drop counts are persisted in the window archive.

---

## Security notes on signing credentials

The compose files mount the validator signing credential directory
**read-only**. Even if the container were compromised, it could not write
to that mount.

What goes there:

- `coldkeypub.txt` — public coldkey, fine to expose.
- `hotkeys/<your-hotkey>` — private signing key. Required.

What must NOT be there:

- The `coldkey` private file. Keep it on a separate machine entirely.

A safe layout is a dedicated validator-signing directory containing only
the public coldkey file and the required hotkey file. Keep any coldkey
private material outside this directory and off the validator host.
