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
- A capacity-qualified GPU fleet for the active profile, 64 GB RAM, and 150 GB disk. Protocol v6 qualification must cover 16-rollout, near-8192-token proofs for the 512-attempt admission budget per environment; normal dispatch stops once the 256 trainer-required passes are reached.
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
# A pair: startup refuses the profile without the capability, and the
# capability without exactly this profile. Every validator-side process that
# imports the constants needs both.
RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-base-dapo-reliquary-v1
RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED=1
RELIQUARY_CHECKPOINT=Qwen/Qwen3-4B-Base
RELIQUARY_HF_REPO_ID=your-org/reliquary-sn   # HF repo to push checkpoints to
HF_TOKEN=hf_xxx                              # write access to that repo
RELIQUARY_EXTERNAL_IP=<your-public-ip>       # advertised on-chain
RELIQUARY_EXTERNAL_PORT=8080
# Preserve the immutable checkpoint currently advertised for this V1 run:
RELIQUARY_RESUME_FROM=sha:<40-hex-hf-commit>
RELIQUARY_PROOF_DEVICES=<qualified-canonical-device-list>
RELIQUARY_PROOF_CAPACITY_MANIFEST=/root/reliquary/state/proof-capacity.json
RELIQUARY_PROOF_CAPACITY_MANIFEST_SHA256=<64-lowercase-hex>
```

### Proof slots (several proof processes per GPU)

`RELIQUARY_PROOF_SLOTS_PER_DEVICE` (default `1`) runs more than one proof
process on each configured GPU. One proof costs roughly
`60 ms + 0.0145 ms/token`, so at these rollout lengths it is ~87% fixed dispatch
and a single process leaves the card at 39% utilisation. Measured on an H100
PCIe over 192 archived rollouts:

| slots | no MPS | with MPS |
|---|---|---|
| 1 | 12.5 s | 12.1 s |
| 2 | 9.1 s | 6.3 s |
| 4 | 8.3 s | **5.7 s** |
| 8 | – | 5.8 s (plateau) |

Two rules follow from that table:

- **Start the CUDA MPS daemon** (`scripts/setup_cuda_mps.sh`, on the host),
  or most of the gain stays on the table — without it the CUDA contexts
  time-slice instead of overlapping. MPS changes no verdict: on/off at 1 and 4
  slots returned 192/192 identical proof results.
- **Budget 10.2 GB of VRAM per slot.** That figure is flat in rollout length
  (512 → 8959 tokens moves peak allocation by 0.20 GB), so size the fleet
  against free VRAM, not against how long completions may grow. Stay clear of
  ~88% card occupancy, where the allocator cliff starts.

Two costs scale with the slot count, both measured and both small: boot loads
the replicas serially (26 s for 4 slots against ~7 s for 1), and a checkpoint
swap reloads each slot in turn (1.5 s each, so 5.9 s for 4 against 1.5 s
today). The swap lands on the serial publication beat, which already runs
longer than a normal window.

Slots require `RELIQUARY_PROOF_PROCESS_ISOLATION=1` — in-process they would
share this interpreter's GIL, which is exactly what isolation exists to avoid,
and startup refuses the combination. Capacity qualification is unaffected: it
is a claim about physical cards and keeps counting `RELIQUARY_PROOF_DEVICES`,
never slots.

With an isolated plane the validator's own train/verify pair no longer needs a
GPU — it neither trains (the detached trainer owns that) nor proves (each
worker holds its own replica) — so it loads on the CPU and the card is left to
the proof workers. Budget **~24 GB of host RAM**: the pair is ~16 GB steady,
but `RELIQUARY_RESUME_FROM` rebinds `train_model` only after the replacement
has loaded, so three replicas are alive for a moment at every boot (a fourth,
+8 GB, if `RELIQUARY_KL_BASE_MODEL` is set). That is a permanent floor on host
RSS, so check free host RAM before enabling — this validator has a known RSS
leak and has OOM-restarted before, and a fixed floor shortens time-to-OOM
proportionally.

#### Setting this up on a fresh box

Nothing here is discoverable from a running validator, so follow the list
rather than memory. Every value below must be identical in all three places it
appears (host daemon, `.env`, compose bind).

1. **Host** — start the MPS daemon. Idempotent, safe to re-run:
   ```bash
   bash scripts/setup_cuda_mps.sh
   ```
   It refuses rather than continuing if `nvidia-cuda-mps-control` is missing:
   on a host that cannot run it, stay at one slot. It also installs a systemd
   unit, because the container restarts after a host reboot and the daemon
   would not — the box would come back at ~1.45× with nothing but a boot
   warning to say so. `--no-service` skips that.
2. **`docker/.env`** — the four keys that move together, all documented in
   `docker/.env.example.trainer`:
   ```bash
   RELIQUARY_PROOF_PROCESS_ISOLATION=1
   RELIQUARY_DETACHED_TRAINER=1          # required by isolation
   RELIQUARY_PROOF_SLOTS_PER_DEVICE=2    # raise once you have watched a day
   CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
   RELIQUARY_IPC_MODE=host               # slots only; see step 3
   ```
3. **Compose** — `docker-compose.trainer.yml` binds the pipe directory
   unconditionally and takes its IPC mode from `RELIQUARY_IPC_MODE`, which
   defaults to `private`. Both the bind and the host IPC namespace are needed:
   MPS reaches its server over shared memory as well as the pipe. The IPC
   namespace is opt-in on purpose — it hands the host's SysV IPC and
   `/dev/shm` to a container that runs untrusted submissions (gVisor, not the
   IPC namespace, is the boundary for miner code), and a one-slot deployment
   gains nothing from it.
4. **Confirm** — start the validator and read the boot log. It warns when the
   control pipe is not reachable:
   ```
   4 proof slots per GPU, but no CUDA MPS control pipe at /tmp/nvidia-mps/control
   ```
   No warning means the pipe is there. It does **not** mean the contexts
   actually overlap — a broken IPC namespace fails silently, and CUDA reports
   nothing either way. The only real confirmation is the clock: time a window
   at one slot against N.

The runtime default is `openmathinstruct`. OpenCode remains an explicit
operator choice until its sandbox and miner rollout are qualified:

```bash
RELIQUARY_ENVIRONMENTS=openmathinstruct,opencodeinstruct
```

Both validator and miner load the same public curated dataset
(`R0mAI/opencodeinstruct-curated`, pinned by default) lazily — the
`structured_cases` ship with it, and the validator runs the grader and
recomputes the code reward authoritatively. FIFO proof, fixed-slot payment,
resource caps, and operator/prompt dedup apply independently to all environments. Do not start
the mixed trainer until the image contains the grader rootfs, `runsc` starts
successfully, and the loopback grader canaries pass.

Protocol v6 inherits these pinned training defaults from v4:

```bash
RELIQUARY_KL_BETA=0
RELIQUARY_LEARNING_RATE=0.000001
RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY=true
RELIQUARY_GRAD_NORM_SKIP_THRESHOLD=100
RELIQUARY_PPO_RATIO_OUTSIDE_CLIP_SKIP_THRESHOLD=0.5
RELIQUARY_CHECKPOINT_PUBLISH_INTERVAL_WINDOWS=16
RELIQUARY_SHAPE_PENALTY=0
```

Do not assemble v6 from independent environment overrides. The trainer refuses
to start unless the selected checkpoint matches the profile, the activation
checkpoint carries the v6 lineage stamp, and the exact proof fleet/runtime has
a release-bound capacity manifest. Re-run qualification whenever the proof
path, runtime fingerprint, checkpoint, or hardware identity changes.

The V5→V6 metadata transition preserves the exact weights, training-run ID,
LR step, and last fully consumed archive boundary. Run
`scripts/prepare_v5_fill_checkpoint.py` against the immutable V5 HEAD, review
its create-only commit plan, and publish only after the final V5 window is
drained. V6 keeps the model, templates, and sampling and changes the window,
selection, proof, and payment regime.

The 16-step checkpoint cadence limits behavior-policy staleness. If the ratio
gate still trips before cadence, the rejected update is excluded and the
validator publishes only the previously accepted in-memory steps before
resuming against the refreshed behavior policy.

### Fill-closed FIFO collection

V6 has no adaptive 60–100-second auction. Each environment continuously admits
and proves eligible groups in monotone FIFO order. Every `B_BATCH=16` proven
groups form one pick; `FILL_CLOSED_PICKS_PER_WINDOW` closes the window.
The default is 16 picks and 256 groups per environment.
`FILL_CLOSED_MAX_SECONDS=1800` is only a stalled-window backstop.

The global proof plan sets `required_passes` to the exact derived group target
and stops dispatching the speculative tail as soon as that demand is met.
The default 256-group shape permits 512 proof attempts and 1024 cheap grading
starts. The default service mode is `strict`. Any smaller operational shape or
bounded service is valid only with matching proof-capacity qualification; none
of these values is an economic score input.

The first pick cannot occur before 30 seconds. With the default pipeline depth
of one, later picks are paced by the durable trainer cursor. Upload rate,
payload bytes, completion tokens, reward difficulty, and submitted drand are
recorded for audit but cannot reorder FIFO or change the fixed group payment.

### Cooldown on training restart

The prompt cooldown is restored at startup from a run-keyed snapshot on R2, so
the full cooldown survives a restart. Key it with `RELIQUARY_TRAINING_RUN_ID`
(default `default`): keep it stable while a training run continues, and **bump
it to a fresh value when you start a new training from scratch** so the cooldown
resets to zero — a fresh model must be allowed to re-see every prompt.

The validator also maintains a canonical rendered-prompt cooldown under the
same run id. Its local gzip snapshot is startup-critical: windows remain closed
until the map is complete and restart-safe. The R2 copy may lag during an
outage, but `/health.content_cooldown` must show `complete=true` and a current
local or mirrored snapshot before serving miners.


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
- Continuous-proof outcomes with selection and reward flags
- Worker drains on window swap (`worker_dropped`)
- Inline accepts under TestClient (`accepted`)

An admission `accepted` is not a win. The final lifecycle record is the one with
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

These are the protocol-v6 release-candidate values. They are one atomic
generation profile; do not assemble them from independent overrides. The same
current constants are explained from the miner's perspective in
[mining.md](mining.md#rejection-reasons).

| Constant | Value | Effect |
|---|---|---|
| `PROTOCOL_PROFILE_ID` | `qwen3-4b-base-dapo-reliquary-v1` | Signed three-environment protocol-v6 profile required from miners and validators |
| `PROTOCOL_MODEL_ID` | `Qwen/Qwen3-4B-Base` | Base model; revision `906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| `B_BATCH` | 16 | Proven groups and uniform reward slots per pick/environment |
| `M_ROLLOUTS` | 16 | Required rollout count per submission |
| `prompt_encoding` | `raw` | Tokenize the canonical prompt directly; applying a chat template is a mismatch |
| Math / Code `prompt_template` | signed step-by-step templates | Exact template ID, renderer, text, and SHA-256 are advertised in `/state.generation_contract` |
| Logic contract | `reliquary/answer-json/v1` | Pinned external task manifest with validator-authoritative JSON answer verification |
| `T_PROTO` / `TOP_P_PROTO` / `TOP_K_PROTO` | `1.0` / `1.0` / `0` | Full-support profile sampling reproduced by the validator |
| Math `answer_format` | `boxed` | Only a valid final `\boxed{...}` or `\fbox{...}` can earn positive Math reward |
| Code `answer_format` | `null` | Code grading is validator-authoritative and has no boxed-answer contract |
| Math / Code / Logic `max_new_tokens` | `8192` | Per-rollout generation cap for every active environment |
| Math / Code / Logic `bft` | `null` | Budget-forced termination is disabled in v6 |
| `FORCED_SEED_PROTOCOL_VERSION` | 6 | Mandatory hotkey-free forced stream while enforcement is active |
| `FILL_CLOSED_EMISSIONS_PER_WINDOW` | 16 | Immutable journal slots/stride reserved per window |
| `FILL_CLOSED_PICKS_PER_WINDOW` | 16 default | Operational pick target; a smaller value requires matching qualification |
| `FILL_CLOSED_TARGET_GROUPS_PER_ENV` | 256 default | Derived as picks per window × `B_BATCH` |
| `FILL_CLOSED_MAX_SECONDS` | 1800 | Backstop for a window that never fills — not a collection target |
| `FILL_CLOSED_PROOF_SERVICE_MODE` | `strict` | Dispatch until trainer demand is met; bounded mode is an explicit qualified override |
| `FILL_CLOSED_ADMISSION_BUDGET_PER_ENV` | 512 | Maximum proof attempts per environment, including failed proofs |
| `FILL_CLOSED_GRADING_START_BUDGET_PER_ENV` | 1024 | Cheap grading-start anti-DoS ceiling per environment |
| `FILL_CLOSED_FIRST_PICK_SECONDS` | 30 | Earliest first pick after window open |
| `FILL_CLOSED_PICK_PIPELINE_DEPTH` | 1 | Later picks are paced by the durable trainer cursor |
| `FILL_CLOSED_SELECTION_POLICY` | `fifo-ingress/v1` | Monotone eligible ingress; rate and payload measurements are telemetry only |
| `MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW` | 512 | Per-environment hotkey quota for the full fill-closed window |
| `SUBMISSION_UPLOAD_GRACE_SECONDS` | 33 | Reveal grace for an exact body precommitted before the cutoff |
| `MAX_SUBMISSION_PAYLOAD_BYTES` | 64 MiB | Per-request parsed JSON payload limit |
| `MAX_PENDING_SUBMISSION_BYTES_PER_HOTKEY` | 128 MiB | Retained pending payload cap per hotkey/environment |
| `MAX_PENDING_SUBMISSION_BYTES_PER_ENV` | 512 MiB | Retained pending payload cap per environment |
| `SIGMA_MIN` (steady) | 0.24 | Zone filter: groups below this are rejected `OUT_OF_ZONE` (binary equivalent: k ∈ [1, 15] for M=16) |
| `BOOTSTRAP_SIGMA_MIN` | 0.22 | Relaxed zone filter during first `BOOTSTRAP_WINDOWS = 100` windows; binary Math still admits k ∈ [1, 15] |
| `BATCH_PROMPT_COOLDOWN_WINDOWS` | 1,000,000 | A winning prompt is effectively one-shot in the OpenMath phase |
| `COOLDOWN_REBUILD_LOOKBACK` | 2000 | Bounded R2 gap replay for legacy/fallback prompt-index cooldown recovery; canonical content uses its run-keyed snapshot |
| `PROOF_SKETCH_TOLERANCE_BASE` | 5000 | GRAIL sketch tolerance — actual threshold = `5000 + 5 × √position` |
| `PROOF_SKETCH_TOLERANCE_GROWTH` | 5.0 | Per-position sqrt growth |
| `LOGPROB_IS_EPS` | 0.10 | Per-token log-prob deviation max — exceeding triggers `LOGPROB_MISMATCH` |
| exact forced terminal pick | required | A natural EOS must be the public forced inverse-CDF token; probability alone cannot authorize it |
| `MAX_TRUNCATED_PER_SUBMISSION` | 1 | Steady-state cap/non-EOS truncation allowance; accepted cap hits still pass GRAIL/logprob/distribution/boxed checks |
| `BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION` | 1 | Bootstrap truncation allowance |
| `TRAINING_QUARANTINE_ENABLED` | true | Suspicious selected windows skip GRPO/publish but remain archived/credited |
| `TRAINING_QUARANTINE_MAX_SINGLE_COMPLETION_LENGTH` | 32768 | Rollout length that counts as extreme-length telemetry |
| `TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_ROLLOUTS` | 4 | Minimum long/cap rollouts before length alone can quarantine a window |
| `TRAINING_QUARANTINE_EXTREME_LENGTH_MIN_GROUPS` | 3 | Minimum groups with long/cap rollouts before length alone can quarantine a window |
| `EMA_ALPHA` | ≈0.0274 | Weight-update smoothing (`2/(72+1)` — ~25-window half-life) |
| `REJECTED_LIST_CAP_PER_HOTKEY` | 5 | Max rejected samples retained per hotkey per window archive |

Source of truth: `reliquary/constants.py`. If any of these change, this table and `concepts.md` need a sync.

### Fill-closed training journal

Each pick writes one immutable checkpoint-bound payload containing up to 16
fixed slots per active environment. The detached trainer consumes the 16
consecutive journal keys in order. Create-only receipts, the window recovery
record, and the trainer cursor make replay idempotent across controller or
trainer restarts; payloads from different checkpoint revisions never mix.

### Submission pipeline

Every `/submit` flows through this sequence on the validator. The first rejection short-circuits the rest.

Upgraded miners first send a small signed `/submit/precommit` containing the
final body's SHA-256, byte count, routing fields, nonce, checkpoint, protocol,
and current drand round. A receipt accepted before the collection cutoff allows
only that exact body to finish within `SUBMISSION_UPLOAD_GRACE_SECONDS = 33`.
It consumes normal hotkey quota but no prompt or proof slot, so abandoned
precommits cannot squat economic capacity. Direct `/submit` remains valid
before cutoff for compatibility; after cutoff a matching receipt is required.

```
HTTP/pre-queue                 environment worker
--------------                 ------------------
window/checkpoint/protocol     prompt/token/randomness/signature checks
envelope/registration          validator-authoritative reward grading
operator logical claim         zone and cheap authenticity guards
rate/queue/payload bounds      -> monotone FIFO proof queue
-> reason="submitted"          -> first /verdicts lifecycle record

continuous fill-closed proof
-> dispatch eligible groups FIFO; rate/bytes/tokens never reorder them
-> stop speculative dispatch at 256 successful proofs per environment
-> proof failures consume the independent 512-attempt budget
-> assemble one pick from each 16 proven groups
-> pay one fixed share per selected group; missing slots burn
-> close after 16 picks or the 1800 s backstop
-> final /verdicts lifecycle records
-> immutable training journal + R2 archive + rewards
```

Code grader candidate failures produce legitimate zero rewards. Grader
infrastructure failures are counted separately: retryable outages return
`WORKER_DROPPED` and refund quota, while ambiguous worker crashes fail closed as
`REWARD_MISMATCH` and consume the logical claim.

R2 retains the historical `difficulty_auction` and
`difficulty_auction_shadow` field names for schema compatibility. Under v6
their policy metadata records FIFO/fixed-slot behavior; the fields do not mean
that a seal-time auction ran.

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

`/health` also reports the selection policy, per-environment queue/proof state,
operator mapping, forced-seed ratio/CDF policy, Code grader failures, and the
persistent archive queue. A nonzero `archive_queue_depth` is safe during a
transient R2 failure, but growing depth or old
`archive_queue_oldest_age_seconds` requires attention.
`archive_last_uploaded_window` confirms that a recent archive left the retry
queue.

Protocol v6 additionally reports the global proof scheduler state, queue and
active work by device, checkpoint readiness, per-environment proof latency,
capacity qualification, and capacity abort totals. A required scheduler in any
state other than `running` makes health degraded and prevents a new window from
opening.

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

All-token forced-seed CDF enforcement defaults off. This is distinct from the
v6 terminal rule: a natural EOS is already required to be the exact public
forced inverse-CDF pick. Private schema-v3 calibration
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

Termination keeps that exact forced-terminal gate, but interesting low-probability EOS,
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

Before `train_step`, the validator runs the training-quarantine gate. If a
selected payload has high-confidence poison signals, the archive still
publishes and emissions remain replayable from `rewards_by_hotkey`, but that
payload is excluded from GRPO. Checkpoint publish cadence is counted by
successful trained steps. The archive field is:

```text
training_quarantine = {quarantined, reasons, metrics}
```

Submissions that get HTTP-accepted but reach the worker after the fill/backstop
has closed are dropped as `WORKER_DROPPED`. They receive a `/verdicts` record,
and aggregate per-hotkey/reason late-drop counts are persisted in the archive.

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
