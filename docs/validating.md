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

The trainer exposes the last `VERDICT_CAP_PER_HOTKEY = 200` per-submission verdicts per miner hotkey via a small in-memory ring buffer. Every code path that decides accept/reject records to it:

- HTTP-level early rejects (`rate_limited`, `window_not_active`, `batch_filled`)
- Worker-level rejects after GRAIL (`grail_fail`, `wrong_randomness`, `logprob_mismatch`, `out_of_zone`, `hash_duplicate`, `bad_termination`, etc.)
- Worker drains on window swap (`worker_dropped`)
- Inline accepts under TestClient (`accepted`)

This is the cheapest way for operators to debug "why is miner X not making the batch" without grep'ing the validator's own logs or pulling R2 archives. Public read by design — same trust model as the R2 archive. Memory cost is ~2.5 MB for a 50-hotkey subnet.

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
| `B_BATCH` | 8 | Number of valid distinct-prompt submissions that seal a window |
| `M_ROLLOUTS` | 8 | Required rollout count per submission |
| `T_PROTO` | 0.9 | Protocol-fixed sampling temperature (validator's recompute uses this) |
| `SIGMA_MIN` (steady) | 0.43 | Zone filter: groups below this are rejected `OUT_OF_ZONE` (binary equivalent: k ∈ [2, 6] for M=8) |
| `BOOTSTRAP_SIGMA_MIN` | 0.33 | Relaxed zone filter during first `BOOTSTRAP_WINDOWS = 100` windows (k ∈ [1, 7]) |
| `BATCH_PROMPT_COOLDOWN_WINDOWS` | 1,000,000 | A winning prompt is effectively one-shot in the OpenMath phase |
| `COOLDOWN_REBUILD_LOOKBACK` | 300 | R2 windows replayed at startup to rebuild cooldown without scanning the whole one-shot horizon |
| `PROOF_SKETCH_TOLERANCE_BASE` | 5000 | GRAIL sketch tolerance — actual threshold = `5000 + 5 × √position` |
| `PROOF_SKETCH_TOLERANCE_GROWTH` | 5.0 | Per-position sqrt growth |
| `LOGPROB_IS_EPS` | 0.10 | Per-token log-prob deviation max — exceeding triggers `LOGPROB_MISMATCH` |
| `MIN_EOS_PROBABILITY` | 0.01 | Required EOS token probability for proper termination |
| `MAX_TRUNCATED_PER_SUBMISSION` | 5 | Steady-state cap/non-EOS truncation allowance; accepted cap hits still pass GRAIL/logprob/distribution/boxed checks |
| `BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION` | 5 | Bootstrap truncation allowance |
| `TRAINING_QUARANTINE_ENABLED` | true | Suspicious selected windows skip GRPO/publish but remain archived/credited |
| `TRAINING_QUARANTINE_MAX_SINGLE_COMPLETION_LENGTH` | 7000 | Rollout length that counts as extreme-length telemetry |
| `TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_ROLLOUTS` | 4 | Minimum long/cap rollouts before length alone can quarantine a window |
| `TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_GROUPS` | 3 | Minimum groups with long/cap rollouts before length alone can quarantine a window |
| `WINDOW_TIMEOUT_SECONDS` | 7200 | Safety-net auto-seal if fewer than B submissions arrive in 2 h |
| `EMA_ALPHA` | ≈0.0274 | Weight-update smoothing (`2/(72+1)` — ~25-window half-life) |
| `REJECTED_LIST_CAP_PER_HOTKEY` | 5 | Max rejected samples retained per hotkey per window archive |

Source of truth: `reliquary/constants.py`. If any of these change, this table and `concepts.md` need a sync.

### Submission pipeline

Every `/submit` flows through this sequence on the validator. The first rejection short-circuits the rest.

```
HTTP enqueue          worker dequeue → verify
─────────────         ─────────────────────────
WINDOW_NOT_ACTIVE? → reject     →    WINDOW_MISMATCH? → reject
rate/envelope/drand checks           WRONG_CHECKPOINT? → reject
queue submission                     BAD_PROMPT_IDX / PROMPT_IN_COOLDOWN? → reject
return reason="submitted"            PROMPT_FULL? → reject
                                     BAD_SCHEMA / TOKENS_MISMATCH / BAD_TOKENS? → reject
                                     PROMPT_MISMATCH / HASH_DUPLICATE? → reject
                                     validator verifies claimed rollout rewards
                                     REWARD_MISMATCH / OUT_OF_ZONE? → reject
                                     BAD_SIGNATURE / WRONG_RANDOMNESS? → reject
                                     GRAIL_FAIL? → reject
                                     BAD_TERMINATION / LOGPROB_MISMATCH? → reject
                                     DISTRIBUTION_SUSPICIOUS? → reject
                                     ─────────────────
                                     → batch[] (selected representative rows)
                                     → runners_up[] (valid pool, not selected)

window seals → R2 archive published at reliquary/dataset/window-<N>.json.gz
             → /set_weights at next epoch boundary
```

Before `train_step`, the validator runs the training-quarantine gate. If the
selected batch has high-confidence poison signals, the archive still publishes
and emissions remain replayable from `rewards_by_hotkey`, but GRPO is skipped
for that window. Checkpoint publish cadence is counted by successful trained
windows, so a quarantined modulo-boundary window does not by itself freeze the
public checkpoint. The archive field is:

```text
training_quarantine = {quarantined, reasons, metrics}
```

Submissions that get HTTP-accepted but reach the worker after the window seals are **dropped late**. They appear in container logs (`INFO | dropping late submission prompt=N hotkey=...`) but not in any R2-archive bucket. The public dashboard surfaces aggregate queue pressure (batch saturation %) as a proxy; per-hotkey late-drop counts are intentionally not exposed publicly.

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
