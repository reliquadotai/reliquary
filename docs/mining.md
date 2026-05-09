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

## What a miner does (v2.1)

Windows are event-driven, not time-based. A window seals the instant `B_BATCH = 16` valid distinct-prompt submissions land; there is no fixed 60-second cadence.

Every miner runs a continuous poll-submit loop:

1. **Polls `/state`.** The response (`GrpoBatchState`) carries `state`, `window_n`, `checkpoint_n`, `checkpoint_repo_id`, `checkpoint_revision`, and `cooldown_prompts`.
   - If `state != "open"`, the validator is in `TRAINING` or `PUBLISHING`. Sleep briefly (1 s) and re-poll. Do not submit while the window is not open.
   - If `checkpoint_n` advanced since the last poll, download the new HF revision and reload weights.

2. **Picks a prompt.** Selects a `prompt_idx` from the Hendrycks MATH environment (`qwedsacf/competition_math` mirror, ~12 500 problems, 5 difficulty levels, 7 subjects) that is not in `cooldown_prompts`. The reference engine uses uniform-random sampling with rejection against the cooldown set.

3. **Generates M=8 rollouts.** Runs exactly 8 completions at the protocol-fixed `T_PROTO = 0.9`, `top_p = 1.0`, `top_k = 0`. No cherry-picking — all eight go in the submission.

4. **Computes local rewards.** Calls `env.compute_reward` on each completion. The validator recomputes rewards independently; claims that do not match are rejected with `REWARD_MISMATCH`.

5. **Builds GRAIL sketches.** Runs the bit-identical HuggingFace forward pass on the proof GPU to construct sketch commitments that bind the completions to the model.

6. **Submits.** POSTs a `BatchSubmissionRequest` to `/submit` containing: `miner_hotkey`, `prompt_idx`, `window_start` (from the last `/state`), 8 rollouts, local rewards, GRAIL commits, `merkle_root`, and `checkpoint_hash` (the HF revision from the last `/state` response). Submission ordering is by validator-side TCP arrival — there is no miner-supplied round or timestamp field.

The validator processes submissions in real time. Only the first `B_BATCH = 16` accepted submissions with distinct `prompt_idx` values that pass all checks form the training batch.

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

| Reason | Meaning | Action |
|---|---|---|
| `WINDOW_NOT_ACTIVE` | Window is in `TRAINING` or `PUBLISHING` | Wait and re-poll `/state` |
| `WINDOW_MISMATCH` | `window_start` in request is stale | Refresh `/state` and retry |
| `WRONG_CHECKPOINT` | `checkpoint_hash` does not match the active revision | Re-poll `/state`, update revision, retry |
| `PROMPT_IN_COOLDOWN` | `prompt_idx` is in the active cooldown set | Retry with a different `prompt_idx` |
| `SUPERSEDED` | Another miner already claimed this `prompt_idx` for the current window | Submit faster next window, or pick a different `prompt_idx` |
| `OUT_OF_ZONE` | σ below threshold (0.43 steady / 0.33 bootstrap) | Pick a different prompt |
| `REWARD_MISMATCH` | Claimed reward does not match validator's `env.compute_reward` | Check env and model version alignment |
| `GRAIL_FAIL` | Sketch does not match validator's forward pass | Check checkpoint, `attn_implementation`, and CUDA version |
| `BAD_SIGNATURE` | GRAIL commit signature failed | Check wallet hotkey and signing code |

`WRONG_CHECKPOINT` is the most common transient rejection — it happens briefly after the validator publishes a new checkpoint. The miner's next `/state` poll will carry the updated revision, and submissions will succeed again.

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
