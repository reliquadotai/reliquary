# Running a Reliquary Miner

Operational guide for running a miner on Bittensor subnet 81. For conceptual background see [docs/concepts.md](concepts.md).

## Boot sequence

1. Miner starts with `reliquary mine --wallet-name ... --hotkey ...`
2. Discovers the validator's HTTP URL via the Bittensor metagraph (or uses `--validator-url` override).
3. Calls `GET /state` to read `checkpoint_repo_id` and `checkpoint_revision`.
4. If the validator has a published checkpoint, downloads it from Hugging Face and loads those weights.
5. Falls back to `--checkpoint` (default: `Qwen/Qwen3.5-4B`) if no checkpoint is published yet.
6. Enters the main loop in `MiningEngine.mine_window()`:
   - Poll `/state` every tick.
   - If `state.checkpoint_n > local_n`, download the new HF revision and reload both model copies.
   - If `state.state == OPEN`, pick a prompt, generate rollouts, and submit.

The boot query ensures a miner joining an already-running subnet lands directly on the current model, skipping an initial reject cycle.

## What a miner does (v2.4)

Windows are event-driven, not time-based. A normal window seals when the active
environment targets have enough valid distinct-prompt submissions. Each
environment target is `B_BATCH = 8`; in the current mixed OpenMath/OpenCode
rollout, a fully trained window is 8 OpenMath groups plus 8 OpenCode groups.
If an environment remains sparse, the validator may force-seal partial once the
queue and proof work are drained:

- `SPARSE_VALID_IDLE_SEAL_SECONDS = 180` after at least 4 distinct valid prompts;
- `SPARSE_VALID_MAX_WINDOW_SECONDS = 600` for sparse or zero-valid windows;
- `WINDOW_TIMEOUT_SECONDS = 7200` remains the outer safety-net timeout.

Partial windows are archived and credited, but they skip GRPO/publish unless
every active environment reaches its target.

> **v2.3 (May 2026)**: TCP-arrival FIFO is gone. Ordering is now anchored to the drand quicknet round each submission carries (`drand_round` field), and the per-prompt single-winner short-circuit (`SUPERSEDED`) is replaced by a per-prompt cap with **emission split among all submitters** for that prompt. Co-location with the validator no longer wins the race. Full design in [docs/superpowers/specs/2026-05-15-drand-ordering-and-prompt-split-design.md](superpowers/specs/2026-05-15-drand-ordering-and-prompt-split-design.md), implementation in [PR #28](https://github.com/reliquadotai/reliquary/pull/28).

Every miner runs a continuous poll-submit loop:

1. **Polls `/state`.** The response (`GrpoBatchState`) carries `state`, `window_n`, `checkpoint_n`, `checkpoint_repo_id`, `checkpoint_revision`, `cooldown_prompts`, and (new in v2.3) **`randomness`** — the validator's per-window seed sourced from drand-quicknet + drand-round. Use it directly as the GRAIL r_vec seed; do **not** recompute it locally from `block_hash + drand` like v2.2 miners did. (`block_hash` was dropped from the v2.3 seed entirely — see the design spec for the reasoning).
   - If `state != "open"`, the validator is in `TRAINING` or `PUBLISHING`. Sleep briefly (1 s) and re-poll. Do not submit while the window is not open.
   - If `checkpoint_n` advanced since the last poll, download the new HF revision and reload weights.

2. **Picks a prompt.** Selects a `prompt_idx` from one active environment. OpenMath uses **OpenMathInstruct-2** ([`nvidia/OpenMathInstruct-2`](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2), ~14 million problems, math-reasoning style) and local reward computation. OpenCode uses the public curated dataset (`R0mAI/opencodeinstruct-curated`) with validator-authoritative grading. In both cases, skip prompts in `cooldown_prompts`. The reference engine uses uniform-random sampling with rejection against the cooldown set. (v2.3 switched OpenMath from Hendrycks MATH because the 12 500-prompt env exhausted under one-shot cooldown — see "One-shot prompts" below.)

3. **Generates M=8 rollouts.** Runs exactly 8 completions at the protocol-fixed `T_PROTO = 0.9`, `top_p = 1.0`, `top_k = 0`. Generate cleanly, terminate at the first EOS, and do not pad or force max length.

4. **Provides rollout rewards.** OpenMath miners compute `env.compute_reward(problem, completion_text)` locally and send that value as `rollout.reward`; the validator recomputes it and rejects mismatches. OpenCode is validator-authoritative: miners send placeholder rewards if the client shape requires them, and the validator recomputes the real code reward and overwrites local claims before the zone filter. Miners never run the grader.

5. **Builds GRAIL sketches.** Runs the bit-identical HuggingFace forward pass on the proof GPU to construct sketch commitments that bind the completions to the model. The r_vec seed **must** come from `state.randomness` exactly — local re-derivation will diverge from the validator's seed and the binding check rejects with `WRONG_RANDOMNESS`.

6. **Submits.** POSTs a `BatchSubmissionRequest` to `/submit` containing: `miner_hotkey`, `prompt_idx`, `window_start` (from the last `/state`), 8 rollouts, claimed rewards, GRAIL commits, `merkle_root`, `checkpoint_hash`, and **(new in v2.3) `drand_round`** — the drand quicknet round currently in progress at the wall-clock instant of the POST. Compute it as `1 + (time.time() - drand_genesis_time) // drand_period`. Quicknet's `period = 3 s` since launch.
   - The validator gates this field with **zero tolerance**: too old → `STALE_ROUND`, too new → `FUTURE_ROUND`. Network jitter that pushes your POST across a round boundary now costs the submission.
   - Pre-baking `drand_round` at sketch-build time is **wrong** — gen takes 50-100 s, so by fire time the round is 1-2 buckets stale → guaranteed `STALE_ROUND`. Compute the round just before the POST.

The validator processes submissions in real time. **Ordering and emission distribution happen at seal time, anchored to the drand round of each submission and a post-window drand seed for prompt rotation — TCP arrival timing is irrelevant in v2.3.**

### Multi-miner-per-prompt and emission split

v2.3 drops the `SUPERSEDED` short-circuit. Up to `MAX_SUBMISSIONS_PER_PROMPT = 10` distinct hotkeys can submit on the same `prompt_idx` within a window; submission #11+ for the same prompt rejects with `PROMPT_FULL` *before* the heavy GRAIL verify, bounding worst-case validator GPU cost.

At seal time, **each filled slot pays `pool / B_BATCH` (= `pool/8`)**. Within a slot, the `K_p` miners who submitted for that `(round, prompt)` split that share equally:

```
reward(miner) = sum over winning_prompts p where miner submitted on p of:
    (pool / B_BATCH) / K_p
```

Consequences:

- **Same-prompt sybil is strictly neutral.** N sybils on prompt P share `pool/8/N` each — total `pool/8`, identical to one hotkey on P alone. Plus N − 1 registration burns. So sybiling the same prompt is strictly wasteful.
- **Different-prompt sybil is additive minus reg-burn.** Each new hotkey covers up to `MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = 8` more distinct prompts per window. The arbitrage is "expected emission gain > registration cost".
- **Unfilled slots burn.** If only M < 8 distinct winning prompts exist, the validator pays out `M × (pool/8)` and the rest goes to `UID_BURN = 0`. No redistribution.

### One-shot prompts

`BATCH_PROMPT_COOLDOWN_WINDOWS = 1_000_000` makes every prompt effectively single-use within any realistic training run. Once a prompt enters `winning_prompts`, it never returns. The 14M-prompt OpenMathInstruct-2 env supplies enough fresh material for ~1.75 million windows at the current B = 8 cadence, which is well beyond any practical training horizon.

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
   - **`batch[]`** — selected representative rows for winning prompts. These rows are the training candidates unless the window is quarantined.
   - **`runners_up[]`** — valid submissions that entered the pool but were not selected as the representative training row. They may still receive emission if their prompt landed in the final winning set.
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

**This is a baseline, not a ceiling.** The protocol enforces no further constraint on `prompt_idx`, but the economics strongly reward miners who can predict which prompts will pass the validator's frontier checks for the current checkpoint:

- An `OUT_OF_ZONE` rejection wastes the full rollout group (eight generations plus their GRAIL proofs). v2.3: it no longer wastes a queue position (drand-round ordering, not FIFO), but it still wastes the ~60-100 s of gen + GRAIL cost on a submission that earns zero.
- A good picker → more of your 8 per-window submissions land in `winning_prompts` → more `pool/8/K_p` shares earned. With multi-miner-per-prompt, picking prompts other miners are NOT targeting (low `K_p`) raises your per-prompt yield.

Techniques miners are expected to develop (non-exhaustive):

- A per-prompt success-rate estimate, updated online and reset (or decayed) whenever `checkpoint_n` advances.
- Clustering problems by difficulty or feature signature and sampling preferentially at the policy's current frontier.
- A cheap proxy (a smaller model, draft decoding, a few low-temperature samples) used only to predict frontier likelihood. Do not build a miner around brittle label/reward oracle tricks; current reward claims are verifier-checked, and future tasks may be private/generated.

The goal is to locate the *learning frontier* — prompts where the current policy succeeds on some attempts and fails on others. Every high-σ pick feeds the GRPO step a gradient-rich group instead of a wasted slot: miner optimization and training efficiency are aligned.

### Zone filter

The validator computes the population standard deviation σ of the verifier-checked rewards for your 8 rollouts. `σ ≥ 0.43` passes; `σ < 0.43` is rejected with `OUT_OF_ZONE`. During bootstrap (first `BOOTSTRAP_WINDOWS = 100` windows) the threshold is `σ ≥ 0.33`.

For OpenMath's binary `{0, 1}` rewards, this admits k=2..6 correct out of 8 in steady state (k=1..7 during bootstrap). You cannot cherry-pick an easy prompt (8/8 correct → σ ≈ 0) or fail on a hard prompt (0/8 correct → σ ≈ 0). Both extremes are worthless for GRPO training.

### Payment model

Earning is EMA-based, not flat per-submission. After each window the validator computes a per-hotkey reward share for the window, then updates each miner's score:

```
# v2.3: per-window reward share = sum over (your submissions on prompts in winning_prompts) of:
#         (pool / B_BATCH) / K_p
# where K_p is the number of miners who submitted on prompt p.
share_this_window = sum(pool/B / K_p for p in winning_prompts if you submitted on p)
score_new = α × share_this_window + (1 − α) × score_old
```

where `α ≈ 0.027` (`EMA_ALPHA = 2 / (72 + 1)`). Once per subnet epoch (~360 blocks), the validator calls `set_weights` on-chain with these EMA values. Your emission for the epoch is proportional to your EMA score relative to other miners.

A miner that consistently lands on 2 of 8 winning prompts per window (each with `K_p = 1`, i.e. no other miner on the same prompt) converges to roughly **25 % of the filled-slot emission budget** (`2 × (pool/8) / pool = 0.25`). Sharing a prompt with `K_p = 2` other miners halves that prompt's contribution: same prompt becomes worth `pool/8/2 = pool/16` to you. Unused slots (M winning prompts < 8) burn to `UID_BURN = 0` — no redistribution.

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
| `STALE_ROUND` | (v2.3) Your `drand_round` field is older than the validator's current drand round at receipt. Zero tolerance — even one round of staleness rejects. | Compute the drand round **immediately before** the POST, never at sketch-build time. `current_drand_round = 1 + (time.time() - genesis_time) // period` with quicknet `period = 3 s`. |
| `FUTURE_ROUND` | (v2.3) Your `drand_round` field is newer than the validator's current round. Implies clock skew. | Ensure miner host is NTP-synced. Drand quicknet rounds advance on a fixed wall-clock schedule; sending a future round means your clock is ahead of UTC. |
| `PROMPT_FULL` | (v2.3) `MAX_SUBMISSIONS_PER_PROMPT = 10` miners already submitted for this `prompt_idx` in this window | Pick a different prompt. With 14M prompts in OpenMathInstruct-2 the probability of collision is low unless many miners coordinate on the same idx. |

**Rejected asynchronously by the worker (look up via `GET /verdicts/{hotkey}` or the R2 archive):**

| Reason | Meaning | Action |
|---|---|---|
| `WRONG_CHECKPOINT` | `checkpoint_hash` does not match the active HF revision | Re-poll `/state`, update revision, retry. Most common transient reject — happens briefly after every new checkpoint publish. |
| `WRONG_RANDOMNESS` | `commit.beacon.randomness` doesn't match the validator's per-window seed (`state.randomness` on v2.3+; locally-derived `H(block_hash + drand)` on v2.2). Almost always caused by reusing a sketch built for an earlier window. | (v2.3) Read `state.randomness` from `/state` directly; do not re-derive locally. (v2.2) Derive per-window from chain + drand. In both cases: tag each sketch with the window it was built for and discard before firing if the window has advanced. |
| `BAD_PROMPT_IDX` | `prompt_idx` out of range for the active environment | Use the env's prompt-index space (`0..N-1`). v2.3 / OpenMathInstruct-2: `N ≈ 14_000_000`. |
| `PROMPT_IN_COOLDOWN` | `prompt_idx` was in the active cooldown set | v2.3: `BATCH_PROMPT_COOLDOWN_WINDOWS = 1_000_000` makes prompts effectively single-use. Read `cooldown_prompts[]` from `/state` **before each pick** and skip anything in the list. |
| `SUPERSEDED` / `HASH_DUPLICATE` | (deprecated v2.3+) `SUPERSEDED` no longer emitted — multiple miners may submit on the same prompt. `HASH_DUPLICATE` still active: your rollout group is bit-identical to one already accepted in the recent hash retention window. | Generate fresh — don't replay tokens. |
| `OUT_OF_ZONE` | σ of your 8 rewards is below threshold (`SIGMA_MIN = 0.43` steady, `0.33` during the first `BOOTSTRAP_WINDOWS = 100` windows) | Pick a prompt where your model gets 2–6 / 8 correct — not 0/8 or 8/8 |
| `REWARD_MISMATCH` | OpenMath local reward claim disagreed with validator recompute, or validator reward computation failed. OpenCode rewards are validator-authoritative and local placeholders are ignored. | For OpenMath, recheck completion decoding, answer parsing, prompt indexing, and env version. For OpenCode, debug generated code validity and hidden-case score through returned verdict/archive data. |
| `GRAIL_FAIL` | Sketch differs from the validator's forward pass by more than `PROOF_SKETCH_TOLERANCE_BASE + PROOF_SKETCH_TOLERANCE_GROWTH × √position` (currently `5000 + 5 × √P`) | Same checkpoint + `attn_implementation=flash_attention_2` + matching CUDA/torch + same GPU class as validator (H200 today) |
| `LOGPROB_MISMATCH` | Per-token log-prob deviation from validator's recompute exceeds `LOGPROB_IS_EPS = 0.10` | Same root cause as `GRAIL_FAIL` — quantization, attention kernel, or precision drift |
| `BAD_TERMINATION` | A rollout did not terminate naturally, hit the cap without EOS, or contains EOS padding/repeated stop-token tails | Confirm generation config matches protocol. Do not force `min_new_tokens`, suppress EOS, ride the 8192 cap, or append tokens after first EOS |
| `DISTRIBUTION_SUSPICIOUS` | Token probability distribution heuristics flagged low-entropy / cheater-like generation | Submit natural rollouts at `T_PROTO = 0.9`; avoid forced prefixes or constrained fillers |
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
    --checkpoint Qwen/Qwen3.5-4B \
    --environments openmathinstruct,opencodeinstruct \
    --validator-url http://<owner-validator-ip>:8888 \
    --log-level INFO
```

Once the owner validator earns `validator_permit`, you can drop `--validator-url` and the miner will auto-discover it from the metagraph.

The miner queries the validator at boot and downloads the current HF checkpoint automatically. You do not need to find or pin the checkpoint hash manually.

### Qwen3.5 model reset

The live model family is `Qwen/Qwen3.5-4B`. This is a hard tokenizer/model reset from earlier Qwen3 checkpoints:

- Always use the shared chat-template prompt encoding; do not tokenize raw prompt text directly.
- The model is a thinking model, so canonical prompts enter the assistant turn with `<think>`.
- Treat both `<|endoftext|>` and `<|im_end|>` as valid stop tokens; current tooling passes the full EOS set into generation and trims at the first stop token.
- Checkpoint downloads are sharded safetensors. Custom miners must download `model*.safetensors`, `model.safetensors.index.json`, `config.json`, tokenizer files, `vocab.json`, `merges.txt`, and `chat_template.jinja`.
- Expect longer completions than the old non-thinking checkpoint and recalibrate local EOS/truncation filters from validator verdicts.

### OpenCode mode

The live mixed rollout enables `opencodeinstruct` next to `openmathinstruct`.
OpenCode rewards are **validator-authoritative**: the validator owns the grader
and recomputes the code reward, so the miner only generates rollouts — it never
runs the grader. A miner that includes OpenCode just sets:

```bash
export RELIQUARY_ENVIRONMENTS=openmathinstruct,opencodeinstruct
```

Both miner and validator load the same public curated dataset
(`R0mAI/opencodeinstruct-curated`, pinned by default) **lazily** — only the
row-groups a window touches are fetched, so there is no bulk dataset download on
top of the model. The structured test cases are visible (the reward grades
genuine model output, not secrecy); the miner does **not** launch or require a
local OpenCode grader. Generate clean Python solutions and keep the same
GRAIL/logprob/termination rules as OpenMath. If your custom miner is not
code-ready yet, keep:

```bash
export RELIQUARY_ENVIRONMENTS=openmathinstruct
```

Additional flags:

| Flag | Default | When to use it |
|---|---|---|
| `--environments` | `openmathinstruct` | Comma-separated active miner environments. Use `openmathinstruct,opencodeinstruct` for mixed mining. |
| `--use-drand` / `--no-use-drand` | `--use-drand` | Turn off only for offline testing. Mainnet always uses drand. |
| `--validator-url` | *(auto-discovered)* | **Required during the subnet-launch phase** (see note above) and for local testing, e.g. `http://127.0.0.1:8888`. Once the owner validator (`5CXzFHfeiJ4Xkiirq4ej1MrRVCd789wEJXhpf2ZKRW6MNFJF`) holds `validator_permit`, leave empty and the miner will discover it from the metagraph. |

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `RELIQUARY_ENVIRONMENTS` | `openmathinstruct` | Comma-separated environment list. Set to `openmathinstruct,opencodeinstruct` for mixed mining. |
| `RELIQUARY_OCI_REPO` | `R0mAI/opencodeinstruct-curated` | OpenCode dataset repo (public, curated). Override only to pin a fork. |
| `RELIQUARY_OCI_REVISION` | pinned commit | OpenCode dataset revision. Override only to pin a different snapshot. |
| `DRAND_CHAIN` | `quicknet` | Override only if drand announces a chain rotation. |
| `GRAIL_ATTN_IMPL` | `flash_attention_2` | Override to `eager` or `sdpa` in test envs without flash-attn. Do not override on mainnet. |

## What you should see

On a healthy startup:

```
... | Starting Reliquary miner (network=finney, netuid=81, envs=['openmathinstruct', 'opencodeinstruct'])
... | OpenCode miner: reward is validator-authoritative; skipping local grader launch.
... | Validator at http://x.x.x.x:8080 is on checkpoint 7 (your-org/reliquary-sn@abc123def...)
... | Downloading to seed the miner model.
... | Loading models from /home/.../.cache/huggingface/...
... | Miner ready. Entering main loop.
... | submitted window=42 prompt=4821 accepted=True reason=submitted
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
- **CUDA out of memory**: two copies of Qwen3.5-4B require roughly 18-20 GB bfloat16 before activations and KV/cache overhead. If you have a single GPU with less than 48 GB you may hit OOM under proofs plus generation. Use a larger GPU or split generation/proofs across devices.
- **`GRAIL_FAIL` / `LOGPROB_MISMATCH`**: your local proof compute diverged from the validator's. Most often caused by a different `attn_implementation` build, CUDA/torch version mismatch, or wrong checkpoint. Re-install on a clean environment and confirm you are on the same HF revision as the validator (check `/state`).
- **`REWARD_MISMATCH`**: for OpenMath, validator-side reward computation disagreed with the miner's claimed `rollout.reward`, or reward computation failed. Recheck completion decoding, answer parsing, prompt indexing, and env version. For OpenCode, local reward claims are placeholders; focus on code validity and validator-scored hidden-case results.
- **All submissions land `OUT_OF_ZONE`**: the prompts you are selecting are too easy (σ ≈ 0) or too hard (σ ≈ 0) for the current checkpoint. On OpenCode, this often means your code lane is producing all-zero or all-pass hidden-case vectors. Split metrics by environment before changing global filters.
- **Persistent `WRONG_CHECKPOINT`**: the miner is not picking up the latest revision from `/state`. Ensure the poll loop reads `checkpoint_revision` before each submission.
