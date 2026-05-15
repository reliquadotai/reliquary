# Running a Reliquary Miner

Operational guide for running a miner on Bittensor subnet 81. For conceptual background see [docs/concepts.md](concepts.md).

## Boot sequence

1. Miner starts with `reliquary mine --wallet-name ... --hotkey ...`
2. Discovers the validator's HTTP URL via the Bittensor metagraph (or uses `--validator-url` override).
3. Calls `GET /state` to read `checkpoint_repo_id` and `checkpoint_revision`.
4. If the validator has a published checkpoint, downloads it from Hugging Face and loads those weights.
5. Falls back to `--checkpoint` (default: `Qwen/Qwen3-4B-Instruct-2507`) if no checkpoint is published yet.
6. Enters the main loop in `MiningEngine.mine_window()`:
   - Poll `/state` every tick.
   - If `state.checkpoint_n > local_n`, download the new HF revision and reload both model copies.
   - If `state.state == OPEN`, pick a prompt, generate rollouts, and submit.

The boot query ensures a miner joining an already-running subnet lands directly on the current model, skipping an initial reject cycle.

## What a miner does (v2.2)

Windows are event-driven, not time-based. A window seals the instant `B_BATCH = 8` valid distinct-prompt submissions land; there is no fixed 60-second cadence. A safety-net timeout (`WINDOW_TIMEOUT_SECONDS = 7200`) auto-seals the window if fewer than 8 valid submissions land in that window.

Every miner runs a continuous poll-submit loop:

1. **Polls `/state`.** The response (`GrpoBatchState`) carries `state`, `window_n`, `checkpoint_n`, `checkpoint_repo_id`, `checkpoint_revision`, and `cooldown_prompts`.
   - If `state != "open"`, the validator is in `TRAINING` or `PUBLISHING`. Sleep briefly (1 s) and re-poll. Do not submit while the window is not open.
   - If `checkpoint_n` advanced since the last poll, download the new HF revision and reload weights.

2. **Picks a prompt.** Selects a `prompt_idx` from the Hendrycks MATH environment (`qwedsacf/competition_math` mirror, ~12 500 problems, 5 difficulty levels, 7 subjects) that is not in `cooldown_prompts`. The reference engine uses uniform-random sampling with rejection against the cooldown set.

3. **Generates M=8 rollouts.** Runs exactly 8 completions at the protocol-fixed `T_PROTO = 0.9`, `top_p = 1.0`, `top_k = 0`. No cherry-picking — all eight go in the submission.

4. **Computes local rewards.** Calls `env.compute_reward` on each completion. The validator recomputes rewards independently; claims that do not match are rejected with `REWARD_MISMATCH`.

5. **Builds GRAIL sketches.** Runs the bit-identical HuggingFace forward pass on the proof GPU to construct sketch commitments that bind the completions to the model.

6. **Submits.** POSTs a `BatchSubmissionRequest` to `/submit` containing: `miner_hotkey`, `prompt_idx`, `window_start` (from the last `/state`), 8 rollouts, local rewards, GRAIL commits, `merkle_root`, and `checkpoint_hash` (the HF revision from the last `/state` response). Submission ordering is by validator-side TCP arrival — there is no miner-supplied round or timestamp field.

The validator processes submissions in real time. Only the first `B_BATCH = 8` accepted submissions with distinct `prompt_idx` values that pass all checks form the training batch.

## Submission lifecycle — where your rollout actually ends up

The most common miner question is *"the validator returned `accepted=True`, but the dashboard says my submission was rejected — what's going on?"* There are **two distinct accept events** in the pipeline, ~seconds apart.

```
miner                    validator HTTP                 validator worker
─────                    ──────────────                 ────────────────
generate 8 rollouts ──▶  enqueue submission              dequeue submission
                         ◀── reason="submitted"          run GRAIL + zone + reward checks
                            (HTTP-accepted)              ──▶ batch[]       if all checks pass
                                                         ──▶ runners_up[]  if valid but B already filled
                                                         ──▶ rejected[]    if any check fails
                                                         ──▶ dropped       if window sealed before pickup
```

1. **HTTP enqueue.** The validator's HTTP layer accepts your POST and returns `accepted=True reason="submitted"`. This is what `submitted ... accepted=True` in your miner log means. **It does NOT mean the validator accepted your work.** It means the submission is in the worker queue.

2. **Worker verification.** A background worker dequeues each submission and runs the full validation pipeline. One of four things happens:
   - **`batch[]`** — first 8 valid distinct-prompt submissions go here. You earn weight share on these.
   - **`runners_up[]`** — valid but lost the FIFO race to fill the batch. No emission, but the validator records you.
   - **`rejected[]`** — failed at least one check. Reason + the failing GRAIL diagnostic value are published.
   - **dropped late** — the batch sealed before the worker reached your queued submission. Not surfaced in the archive; only visible in validator logs.

The R2 archive (`reliquary/dataset/window-<N>.json.gz`) contains the first three buckets. The public dashboard reads it.

### How to look up your specific submission

Per submission you have `(window_n, prompt_idx)`. Two lookup paths:

- **Dashboard drawer.** Click your hotkey row on `https://reliqua.ai/dashboard`. The drawer's "last 5w" table shows `sub / acc / soft / hard` counts per window for your hotkey, and when `hard > 0` it lists every rejection with its `prompt_idx`, reason, and the actual GRAIL diagnostic values (`sketch_diff`, `lp_dev`, `dist_q10`) that pushed it over threshold.
- **Raw archive.** `GET https://reliqua.ai/api/r2/window/<N>` returns the full window archive for any cached window. Search `batch[]`, `runners_up[]`, `rejected[]` for your `prompt_idx`. If it's in none of them, the submission was dropped late.

### Prompt selection strategy

The reference strategy (`pick_prompt_idx` in `reliquary/miner/engine.py`) is uniform-random sampling with rejection against the cooldown set:

```
GET /state  →  GrpoBatchState
```

- Read `cooldown_prompts` and pick any `prompt_idx` not in that set.
- Read `checkpoint_revision` and include it verbatim as `checkpoint_hash` in your submission.
- Read `window_n` and use it as the authoritative window identifier.

**This is a baseline, not a ceiling.** The protocol enforces no further constraint on `prompt_idx`, but the economics strongly reward miners who can predict which prompts will pass the zone filter (`σ ≥ 0.43`) for the current checkpoint:

- An `OUT_OF_ZONE` rejection wastes the full rollout group (eight generations plus their GRAIL proofs). The retry arrives later, so miners who guessed right on the first attempt have already claimed the slot for that `prompt_idx` (`SUPERSEDED`) or filled the batch ahead of you.
- A good picker → first submission passes → earlier validator-side arrival → batch slot won.

Techniques miners are expected to develop (non-exhaustive):

- A per-prompt success-rate estimate, updated online and reset (or decayed) whenever `checkpoint_n` advances.
- Clustering problems by difficulty or feature signature and sampling preferentially at the policy's current frontier.
- A cheap proxy (a smaller model, draft decoding, a few low-temperature samples) used only to predict σ — *never* to pre-generate the actual submission, which must be exactly eight rollouts at `T_PROTO = 0.9`.

The goal is to locate the *learning frontier* — prompts where the current policy succeeds on some attempts and fails on others. Every high-σ pick feeds the GRPO step a gradient-rich group instead of a wasted slot: miner optimization and training efficiency are aligned.

### Zone filter

The validator computes the population standard deviation σ of your 8 rollout rewards. `σ ≥ 0.43` passes; `σ < 0.43` is rejected with `OUT_OF_ZONE`. During bootstrap (first `BOOTSTRAP_WINDOWS = 100` windows) the threshold is `σ ≥ 0.33`.

For MATH's binary `{0, 1}` rewards (answer extracted from `\boxed{...}` and compared after conservative LaTeX normalization) this is mathematically equivalent to having between 2 and 6 correct out of 8. The σ formulation works for continuous reward environments too, without any validator changes. See [docs/concepts.md](concepts.md#zone-filter-σ--043----only-train-on-learnable-prompts) for the full explanation.

You cannot cherry-pick an easy prompt (8/8 correct → σ ≈ 0) or fail on a hard prompt (0/8 correct → σ ≈ 0). Both extremes are worthless for GRPO training.

### Payment model

Earning is EMA-based, not flat per-submission. After each window, the validator updates each miner's score:

```
score_new = α × (slots_won / 8) + (1 − α) × score_old
```

where `α ≈ 0.027` (`EMA_ALPHA = 2 / (72 + 1)`). Once per subnet epoch (~360 blocks), the validator calls `set_weights` on-chain with these EMA values. Your emission for the epoch is proportional to your EMA score relative to other miners.

A miner that consistently wins 2 of 8 batch slots per window converges to roughly 25% of the filled-slot emission budget. Unused slots burn to `UID_BURN = 0`.

See [docs/concepts.md](concepts.md#economic-model) for the full economic model.

### Rejection reasons

The validator emits one of the following reasons on every failed submission. Each is published per-submission in the window archive's `rejected[]` array (capped at 5 entries per hotkey per window). Definitions live in `reliquary/protocol/submission.py::RejectReason`.

**Rejected synchronously at HTTP enqueue (the `/submit` response carries the reason directly):**

| Reason | Meaning | Action |
|---|---|---|
| `WINDOW_NOT_ACTIVE` | Window is in `TRAINING`, `PUBLISHING`, or `READY` — not accepting submissions | Sleep and re-poll `/state` until `state == "open"` |
| `RATE_LIMITED` | You exceeded `MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = 8` submissions in this window | Throttle locally; the counter resets at every window boundary |
| `BATCH_FILLED` | The batcher already accepted `B_BATCH = 8` distinct non-cooldown valid submissions — your submission can never displace one, so it's rejected before GRAIL runs (PR #22) | Fire earlier next window, or accept that this window is closed |
| `WINDOW_MISMATCH` | `window_start` in your request doesn't match the active batcher | Refresh `/state` and retry with the current `window_n` |

**Rejected asynchronously by the worker (look up via `GET /verdicts/{hotkey}` or the R2 archive):**

| Reason | Meaning | Action |
|---|---|---|
| `WRONG_CHECKPOINT` | `checkpoint_hash` does not match the active HF revision | Re-poll `/state`, update revision, retry. Most common transient reject — happens briefly after every new checkpoint publish. |
| `WRONG_RANDOMNESS` | `commit.beacon.randomness` doesn't match the validator's per-window seed derived from `block_hash(window_n) + drand(round_for_window)` (PR #23). Almost always caused by reusing a sketch built for an earlier window after the window advanced. | Derive randomness per-window from chain + drand; tag each sketch with the window it was built for and discard before firing if the window has advanced. |
| `BAD_PROMPT_IDX` | `prompt_idx` out of range for the active environment | Use the env's prompt-index space (`0..N-1`) |
| `PROMPT_IN_COOLDOWN` | `prompt_idx` was in the active cooldown set (a prompt that entered a batch is cooled for `BATCH_PROMPT_COOLDOWN_WINDOWS = 72` windows) | Read `cooldown_prompts[]` from `/state` **before each pick** and skip anything in the list. This is the #1 cause of persistent self-inflicted rejection. |
| `SUPERSEDED` / `HASH_DUPLICATE` | Another miner already won this `prompt_idx` for the current window, or your group is bit-identical to one already accepted in the last `HASH_DEDUP_RETENTION_WINDOWS = 10000` | Pick a different `prompt_idx` or arrive earlier |
| `OUT_OF_ZONE` | σ of your 8 rewards is below threshold (`SIGMA_MIN = 0.43` steady, `0.33` during the first `BOOTSTRAP_WINDOWS = 100` windows) | Pick a prompt where your model gets 2–6 / 8 correct — not 0/8 or 8/8 |
| `REWARD_MISMATCH` | Your claimed rewards don't match the validator's recompute under `env.compute_reward` | Confirm env + model + tokenizer match the validator |
| `GRAIL_FAIL` | Sketch differs from the validator's forward pass by more than `PROOF_SKETCH_TOLERANCE_BASE + PROOF_SKETCH_TOLERANCE_GROWTH × √position` (currently `5000 + 5 × √P`) | Same checkpoint + `attn_implementation=flash_attention_2` + matching CUDA/torch + same GPU class as validator (H200 today) |
| `LOGPROB_MISMATCH` | Per-token log-prob deviation from validator's recompute exceeds `LOGPROB_IS_EPS = 0.10` | Same root cause as `GRAIL_FAIL` — quantization, attention kernel, or precision drift |
| `BAD_TERMINATION` | A rollout did not end on EOS or hit `MIN_EOS_PROBABILITY = 0.01` correctly | Confirm generation config matches protocol (max-tokens, EOS handling) |
| `DISTRIBUTION_SUSPICIOUS` | Reward distribution heuristics flagged the group as low-entropy / cheater-like | Submit all 8 rollouts honestly at `T_PROTO = 0.9` — no pre-screening |
| `WRONG_ROLLOUT_COUNT` | Group has fewer or more than `M_ROLLOUTS = 8` rollouts | Always submit exactly 8 |
| `BAD_SCHEMA` / `BAD_TOKENS` | Submission payload malformed | Validate against the protocol schema |
| `PROMPT_MISMATCH` | Canonical prompt tokens for `prompt_idx` don't match the request | Re-derive prompt tokens from the env's deterministic mapping |
| `BAD_SIGNATURE` | GRAIL commit signature failed | Check wallet hotkey and signing code |
| `WORKER_DROPPED` | Your submission was queued before the active batcher swapped (e.g. window advanced while sitting in the worker queue). The submission was dropped without running GRAIL because re-archiving into a sealed window is impossible. | Fire sooner inside the window; under sustained `worker_dropped` the validator is back-pressured — back off briefly. |

`PROMPT_IN_COOLDOWN` is the most common **persistent** rejection caused by miner code: if your picker doesn't read `cooldown_prompts[]` before each pick, you will repeatedly submit prompts the validator has already cooled. Read the field — it's small and refreshes every `/state` call. The dashboard surfaces this directly on the miner drawer.

### Real-time verdict feedback (`/verdicts/{hotkey}`)

Under the production worker path the `/submit` response carries only a provisional sentinel — `accepted=True reason="submitted"` — that means "queued for verification", **not** "passed verification". The real verdict (`ACCEPTED` / `GRAIL_FAIL` / `WRONG_RANDOMNESS` / etc.) is only known after the worker drains the submission and runs the full pipeline (~5–25 s of GRAIL per item). If your miner logs every `/submit` response as `ACCEPTED`, it is lying — those logs include submissions that the worker silently rejected.

The validator exposes the real per-submission verdicts via:

```
GET http://<validator-host>:8080/verdicts/{your_hotkey}?since=<unix_ts>
```

Response (`VerdictsResponse` in `reliquary/protocol/submission.py`):

```json
{
  "verdicts": [
    {"merkle_root": "ab12...64hex", "window_n": 1858, "accepted": true,  "reason": "accepted",         "ts": 1747353600.5},
    {"merkle_root": "ef56...64hex", "window_n": 1858, "accepted": false, "reason": "grail_fail",       "ts": 1747353601.1},
    {"merkle_root": "gh90...64hex", "window_n": 1858, "accepted": false, "reason": "wrong_randomness", "ts": 1747353601.4}
  ]
}
```

Properties:

- **Per-hotkey ring buffer** of the last `VERDICT_CAP_PER_HOTKEY = 200` verdicts. Older entries roll off silently.
- **Ordered by `ts` ascending.** Pass the highest `ts` you've seen as `?since=<ts>` to get only newer entries — strict `>` filter, so the same `ts` is excluded.
- **Empty list for unseen hotkeys** (200, not 404).
- **Public read.** Same trust model as the R2 archive; anyone can query any hotkey's verdicts.
- **Lock-free.** Doesn't compete with the submit worker for the batcher lock.

Recommended miner integration (~20 lines):

```python
last_seen_ts = 0.0

async def poll_verdicts(client, hotkey, validator_url):
    global last_seen_ts
    while True:
        try:
            r = await client.get(
                f"{validator_url}/verdicts/{hotkey}",
                params={"since": last_seen_ts},
                timeout=5.0,
            )
            for v in r.json()["verdicts"]:
                if v["accepted"]:
                    logger.info(
                        "verdict ACCEPTED win=%d mr=%s",
                        v["window_n"], v["merkle_root"][:12],
                    )
                else:
                    logger.warning(
                        "verdict REJECTED win=%d mr=%s reason=%s",
                        v["window_n"], v["merkle_root"][:12], v["reason"],
                    )
                last_seen_ts = max(last_seen_ts, v["ts"])
        except Exception:
            pass
        await asyncio.sleep(5)
```

Then change your fire-time log from `ACCEPTED ...` to something honest like `SUBMITTED window=N mr=<short_merkle>`. The real verdict will land 5–15 s later via the poller. Without this, you cannot tell `submitted to queue` apart from `passed GRAIL`, which makes debugging "why is my slot share dropping" much harder than it needs to be.

This endpoint is purely additive — existing miners that don't poll it keep working exactly as before; they just continue to mislabel their logs.

---

## Requirements

| Item | Requirement |
|---|---|
| OS | Linux (tested on Ubuntu 22.04 / 24.04) |
| Python | 3.11 or newer |
| GPU | 1× or 2× NVIDIA GPU, ≥ 24 GB VRAM each. Reference config: 2× GPU (generation on GPU 0, proof on GPU 1). Single GPU works if it holds two model copies. **Test phase: use an NVIDIA H200** — see "Hardware homogeneity" note below. |
| CUDA | 12.x with `flash-attn`-compatible drivers |
| RAM | 32 GB minimum |
| Disk | 50 GB (model weights and HF cache) |
| Network | Stable outbound HTTPS to HF Hub and the active validator |
| Bittensor wallet | Created and registered on netuid 81 |

No R2 or S3 credentials are needed on the miner — only the validator uploads the window dataset.

### Hardware homogeneity (test phase)

The subnet is still in its test phase. We ran a 10-run, 30-step cheater-curve
sweep (`scripts/cheater_curve_threshold.py`) and tightened
`PROOF_SKETCH_TOLERANCE_BASE` to **1000** based on observed signal-vs-noise:
this catches an off-policy miner (one running a checkpoint older than the
validator's) starting from the very first weight update, with 0 % false
positives **on identical hardware**. Cross-GPU honest noise has not yet been
measured.

Until that calibration is published, miners are recommended to run the same
card as the validator — currently an **NVIDIA H200**. Running a different
card (H100, A100, etc.) may produce sketch divergence above the tolerance
even on a perfectly honest checkpoint, leading to `GRAIL_FAIL` rejections
through no fault of the miner. We will widen the tolerance or publish a
per-GPU calibration once we have honest cross-GPU data.

## Install

```bash
git clone <repo-url> reliquary
cd reliquary
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Verify:

```bash
reliquary --help
```

You should see `mine` and `validate` subcommands.

## Register your hotkey on the subnet

```bash
btcli wallet new-coldkey --wallet.name my_miner
btcli wallet new-hotkey  --wallet.name my_miner --wallet.hotkey default
btcli subnet register    --wallet.name my_miner --wallet.hotkey default --netuid 81
```

Confirm your hotkey appears in `btcli subnet metagraph --netuid 81` with a valid UID.

## Launch

> **Subnet-launch phase — `--validator-url` is required.**
> For the first weeks after subnet go-live, the subnet owner's validator will not yet hold enough stake to earn `validator_permit`, so the metagraph auto-discovery path (`discover_validator_url`) will raise `no validator with permit and routable axon`. Until the owner's hotkey gains the permit, you **must** pin the validator manually with `--validator-url`.
>
> The official subnet-owner validator hotkey is:
>
> ```
> 5CXzFHfeiJ4Xkiirq4ej1MrRVCd789wEJXhpf2ZKRW6MNFJF
> ```
>
> Cross-check the axon IP advertised on-chain for this hotkey in `btcli subnet metagraph --netuid 81` before passing it to `--validator-url` — that confirms you are connecting to the real owner validator and not a look-alike.

```bash
reliquary mine \
    --network finney \
    --netuid 81 \
    --wallet-name my_miner \
    --hotkey default \
    --checkpoint Qwen/Qwen3-4B-Instruct-2507 \
    --validator-url http://<owner-validator-ip>:8888 \
    --log-level INFO
```

Once the owner validator earns `validator_permit`, you can drop `--validator-url` and the miner will auto-discover it from the metagraph.

The miner queries the validator at boot and downloads the current HF checkpoint automatically. You do not need to find or pin the checkpoint hash manually.

Additional flags:

| Flag | Default | When to use it |
|---|---|---|
| `--environment` | `math` | Pinned by the protocol; do not change unless the subnet announces a migration. |
| `--use-drand` / `--no-use-drand` | `--use-drand` | Turn off only for offline testing. Mainnet always uses drand. |
| `--validator-url` | *(auto-discovered)* | **Required during the subnet-launch phase** (see note above) and for local testing, e.g. `http://127.0.0.1:8888`. Once the owner validator (`5CXzFHfeiJ4Xkiirq4ej1MrRVCd789wEJXhpf2ZKRW6MNFJF`) holds `validator_permit`, leave empty and the miner will discover it from the metagraph. |

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `DRAND_CHAIN` | `quicknet` | Override only if drand announces a chain rotation. |
| `GRAIL_ATTN_IMPL` | `flash_attention_2` | Override to `eager` or `sdpa` in test envs without flash-attn. Do not override on mainnet. |

## What you should see

On a healthy startup:

```
... | Starting Reliquary miner (network=finney, netuid=81, env=math)
... | Validator at http://x.x.x.x:8888 is on checkpoint 7 (your-org/reliquary-sn@abc123def...)
... | Downloading to seed the miner model.
... | Loading models from /home/.../.cache/huggingface/...
... | Miner ready. Entering main loop.
... | submitted window=42 prompt=4821 accepted=True reason=accepted
```

If submissions are rejected, the `reason` field tells you why (see the rejection table above).

## Monitoring and stopping

The miner loop runs until killed. Between windows (when `/state` returns `state != "open"`) it sleeps 1 s and re-polls. On network errors it backs off for up to 12 s. No per-window state is kept locally, so restarting is safe.

```bash
# GPU utilization during generation and proof construction.
nvidia-smi

# Submission results.
grep -E "submitted|rejected|accepted" ~/miner.log | tail -50
```

## Troubleshooting

- **`no validator with permit and routable axon`**: no active validator has published an HTTP endpoint on the metagraph. During the subnet-launch phase this is expected — the owner validator (`5CXzFHfeiJ4Xkiirq4ej1MrRVCd789wEJXhpf2ZKRW6MNFJF`) does not yet hold `validator_permit`. Pass `--validator-url http://<owner-validator-ip>:8888` to pin it explicitly (see [Launch](#launch)). After launch, wait for a validator to come back online or point at a known one.
- **CUDA out of memory**: two copies of Qwen3-4B-Instruct require ~16 GB bfloat16 total. If you have a single GPU with less than 24 GB you may hit OOM. Use a GPU with more VRAM or reduce precision.
- **`GRAIL_FAIL` / `REWARD_MISMATCH`**: your local compute diverged from the validator's. Most often caused by a different `attn_implementation` build, CUDA/torch version mismatch, or wrong checkpoint. Re-install on a clean environment and confirm you are on the same HF revision as the validator (check `/state`).
- **All submissions land `OUT_OF_ZONE`**: the prompts you are selecting are too easy (σ ≈ 0) or too hard (σ ≈ 0) for the current checkpoint. The reference engine samples uniformly; if you have overridden prompt selection, broaden the range.
- **Persistent `WRONG_CHECKPOINT`**: the miner is not picking up the latest revision from `/state`. Ensure the poll loop reads `checkpoint_revision` before each submission.
