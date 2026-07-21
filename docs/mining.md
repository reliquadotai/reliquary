# Running a Reliquary Miner

Operational guide for running a miner on Bittensor subnet 81. For conceptual background see [docs/concepts.md](concepts.md).

## Boot sequence

1. Miner starts with `reliquary mine --wallet-name ... --hotkey ...`
2. Discovers the validator's HTTP URL via the Bittensor metagraph (or uses `--validator-url` override).
3. Calls `GET /state` to read `checkpoint_repo_id` and `checkpoint_revision`.
4. If the validator has a published checkpoint, downloads it from Hugging Face and loads those weights.
5. Falls back to `--checkpoint` (default: `Qwen/Qwen3.5-2B`) if no checkpoint is published yet.
6. Enters the main loop in `MiningEngine.mine_window()`:
   - Poll `/state` every tick.
   - If `state.checkpoint_n > local_n`, download the new HF revision and reload both model copies.
   - If `state.state == OPEN`, pick a prompt, generate rollouts, and submit.

The boot query ensures a miner joining an already-running subnet lands directly on the current model, skipping an initial reject cycle.

## What a miner does (auction v2)

Math and Code each collect submissions for a fixed 100-second interval. The
validator does not close early when eight groups arrive. At the deadline it
drains pre-deadline work, freezes each environment's pending population, ranks
groups by difficulty, and runs deferred proof top-down until at most eight
distinct-prompt winners pass. There is no per-operator winner cap. Unfilled
slots burn.

Clean selected groups are retained in a bounded accumulator tied to the public
checkpoint. GRPO waits until the accumulator has one target batch from every
active environment; extra groups from one environment do not overweight it.

> **Hard cutover:** auction and deferred proof intentionally apply to both
> `openmathinstruct` and `opencodeinstruct`. Forced-seed protocol v2 is mandatory.
> BFT remains Math-only and should not be added to the Code generation path.

Every miner runs a continuous poll-submit loop:

1. **Polls `/state`.** The response (`GrpoBatchState`) carries `state`, `window_n`, `checkpoint_n`, `checkpoint_repo_id`, `checkpoint_revision`, `cooldown_prompts`, and (new in v2.3) **`randomness`** — the validator's per-window seed sourced from drand-quicknet + drand-round. Use it directly as the GRAIL r_vec seed; do **not** recompute it locally from `block_hash + drand` like v2.2 miners did. (`block_hash` was dropped from the v2.3 seed entirely — see the design spec for the reasoning).
   - If `state != "open"`, the validator is in `TRAINING` or `PUBLISHING`. Sleep briefly (1 s) and re-poll. Do not submit while the window is not open.
   - If `checkpoint_n` advanced since the last poll, download the new HF revision and reload weights.

2. **Picks a prompt.** Selects a `prompt_idx` from one active environment. OpenMath uses **OpenMathInstruct-2** ([`nvidia/OpenMathInstruct-2`](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2), ~14 million problems, math-reasoning style) and local reward computation. OpenCode uses the public curated dataset (`R0mAI/opencodeinstruct-curated`) with validator-authoritative grading. In both cases, skip prompts in `cooldown_prompts`. The reference engine uses uniform-random sampling with rejection against the cooldown set. (v2.3 switched OpenMath from Hendrycks MATH because the 12 500-prompt env exhausted under one-shot cooldown — see "One-shot prompts" below.)

3. **Generates M=8 rollouts.** Runs exactly 8 completions with the repository's forced-seed v2 sampler. The deterministic stream excludes hotkey identity and is derived from window randomness, prompt, checkpoint, rollout index, and token position. Set `protocol_version=2`, terminate at the first EOS, and do not add a presence/repetition processor that the validator does not reproduce.

4. **Provides rollout rewards.** OpenMath miners compute `env.compute_reward(problem, completion_text)` locally and send that value as `rollout.reward`; the validator recomputes it and rejects mismatches. OpenCode is validator-authoritative: miners send placeholder rewards if the client shape requires them, and the validator recomputes the real code reward and overwrites local claims before the zone filter. Miners never run the grader.

5. **Builds GRAIL sketches.** Runs the bit-identical HuggingFace forward pass on the proof GPU to construct sketch commitments that bind the completions to the model. The r_vec seed **must** come from `state.randomness` exactly — local re-derivation will diverge from the validator's seed and the binding check rejects with `WRONG_RANDOMNESS`.

6. **Commits, then uploads.** Finalize the signed `BatchSubmissionRequest`,
   serialize it once, and compute its byte length and SHA-256. POST the small
   signed metadata to `/submit/precommit`, then POST those exact bytes to
   `/submit` with the returned `X-Reliquary-Precommit` receipt. A precommit
   received before the 100-second cutoff grants at most 33 seconds for that
   exact reveal; it does not extend generation or reserve an auction slot.
   - Compute and sign the current quicknet round immediately before
     serialization. The validator applies zero backward tolerance and records
     drand at precommit arrival. Pre-baking the round at sketch-build time is
     wrong.
   - Do not rebuild, reformat, or re-sign the body after precommit. The receipt
     binds its SHA-256, byte count, routing fields, nonce, checkpoint, and
     protocol version. A same-sized substitution is rejected.
   - The reference submitter falls back to deadline-sensitive direct `/submit`
     only when an older validator returns 404 for `/submit/precommit`.

The validator grades submissions during collection, but expensive GRAIL and
auth proof run at seal only for candidates that can still win. Difficulty is
the primary ranking key. Equal scores prefer the earlier validator-observed
precommit drand round; candidates still tied inside that three-second bucket
use an operator/prompt hash salted by post-deadline drand. Submitted drand is a
freshness check, not an economic ordering key. TCP milliseconds do not matter
while seal drand is available, and hotkey count, Merkle-root grinding, or
harmless payload variation cannot mint extra tickets for one operator/prompt.

### Prompt competition and payment

Up to `MAX_SUBMISSIONS_PER_PROMPT = 10` bounded pending groups may exist for one
prompt, but an operator can reserve only one logical claim for that prompt. At
seal, the first ranked candidate for a prompt that passes proof wins. If it
fails, the next candidate is promoted. Runners-up do not split emission.

Each selected group earns `pool / B_BATCH` for its environment. An operator can
win any number of distinct prompt slots on merit. Missing winners are not
redistributed; their shares burn to `UID_BURN = 0`.

### One-shot prompts

`BATCH_PROMPT_COOLDOWN_WINDOWS = 1_000_000` makes every prompt effectively single-use within any realistic training run. Once a prompt enters `winning_prompts`, it never returns. The 14M-prompt OpenMathInstruct-2 env supplies enough fresh material for ~1.75 million windows at the current B = 8 cadence, which is well beyond any practical training horizon.

## Submission lifecycle — where your rollout actually ends up

The most common miner question is *"the validator returned `accepted=True`, but I earned no slot — what's going on?"* Auction mode has three lifecycle stages.

```
miner                 HTTP/worker admission             100 s seal
-----                 ---------------------             ----------
POST precommit   ->   signed upload receipt             freeze pending pool
POST exact body  ->   reason="submitted"                rank by difficulty
                      cheap checks + grading            rank by difficulty
                      first ACCEPTED verdict            prove top-down
                      pending pool only                 final verdict + reward
```

1. **HTTP enqueue.** `accepted=True reason="submitted"` means only that the request entered the worker queue.

2. **Pool admission.** The worker runs bounded schema, identity, reward, zone, and authenticity-independent checks. Its `ACCEPTED` verdict means your group is in the pending auction pool. Code grader infrastructure failures are not converted into zero rewards.

3. **Seal result.** The validator publishes a second verdict after ranking and deferred proof. A paid winner has `selected_for_batch=true` and `rewarded=true`. An honest non-winner remains `accepted=true` with both fields false. A candidate that reaches proof and fails gets its actual rejection with `reject_stage="auction_seal"`.

The R2 archive (`reliquary/dataset/window-<N>.json.gz`) contains selected rows,
rejections, and the full auction candidate metadata under `difficulty_auction`.
The historical `difficulty_auction_shadow` key is an identical compatibility
alias.

### How to look up your specific submission

Per submission you have `(window_n, prompt_idx)`. Two lookup paths:

- **Dashboard drawer.** Click your hotkey row on `https://reliqua.ai/dashboard`. The drawer's "last 5w" table shows `sub / acc / soft / hard` counts per window for your hotkey, and when `hard > 0` it lists every rejection with its `prompt_idx`, reason, and the actual GRAIL diagnostic values (`sketch_diff`, `lp_dev`, `dist_q10`) that pushed it over threshold.
- **Raw archive.** `GET https://reliqua.ai/api/r2/window/<N>` returns the full window archive for any cached window. Search `batch[]`, `rejected[]`, and `difficulty_auction.<environment>.candidates[]` for your prompt and hotkey. Ingress evidence includes payload/body timing, precommit status, queue wait, reward grading, and admission commit time.

### Prompt selection strategy

The reference strategy (`pick_prompt_idx` in `reliquary/miner/engine.py`) is uniform-random sampling with rejection against the cooldown set:

```
GET /state  →  GrpoBatchState
```

- Read `cooldown_prompts` and pick any `prompt_idx` not in that set.
- Read `checkpoint_revision` and include it verbatim as `checkpoint_hash` in your submission.
- Read `window_n` and use it as the authoritative window identifier.

**This is a baseline, not a ceiling.** The protocol enforces no further constraint on `prompt_idx`, but the economics strongly reward miners who can predict which prompts will pass the validator's frontier checks for the current checkpoint:

- An `OUT_OF_ZONE` rejection wastes the eight generations, although deferred proof prevents it from consuming seal-time GRAIL.
- A good picker puts more groups near the `k=2` score peak and high enough in the frozen ranking to justify proof. Coverage matters because only one proven winner can occupy each prompt, but there is no operator winner cap.

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
# One uniform slot per proven distinct-prompt winner.
share_this_window = winning_slots * (pool / B_BATCH)
score_new = α × share_this_window + (1 − α) × score_old
```

where `α ≈ 0.027` (`EMA_ALPHA = 2 / (72 + 1)`). Once per subnet epoch (~360 blocks), the validator calls `set_weights` on-chain with these EMA values. Your emission for the epoch is proportional to your EMA score relative to other miners.

A miner may win multiple distinct prompt slots in one environment. Unused slots burn; there is no boundary-tier payment, runner-up split, or redistribution. In auction mode `rewarded=true` if and only if `selected_for_batch=true`.

See [docs/concepts.md](concepts.md#economic-model) for the full economic model.

### Rejection reasons

The validator emits one of the following reasons on every failed submission. Each is published per-submission in the window archive's `rejected[]` array (capped at 5 entries per hotkey per window). Definitions live in `reliquary/protocol/submission.py::RejectReason`.

**Rejected synchronously at HTTP enqueue (the `/submit` response carries the reason directly):**

| Reason | Meaning | Action |
|---|---|---|
| `WINDOW_NOT_ACTIVE` | Window is in `TRAINING`, `PUBLISHING`, or `READY` — not accepting submissions | Sleep and re-poll `/state` until `state == "open"` |
| `PRECOMMIT_REQUIRED` | Collection closed and the body has no valid predeadline upload receipt | Upgrade the submitter; precommit the final serialized body before cutoff |
| `PRECOMMIT_INVALID` | Receipt, body hash/size, nonce, routing fields, or signature do not match | Reuse the exact serialized bytes associated with the receipt; never rebuild the body after precommit |
| `PRECOMMIT_EXPIRED` | The exact body did not finish inside the bounded reveal grace | Start finalization earlier or improve the upload path; do not increase generation after precommit |
| `MERKLE_ROOT_MISMATCH` | After the validator operator enables the calibrated gate, the signed wire-v1 root does not equal its byte-compatible recomputation | Use the repository's existing `_compute_merkle_root` output without altering its serialization |
| `RATE_LIMITED` | You exceeded `MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = 8` submissions in this window | Throttle locally; the counter resets at every window boundary |
| `BATCH_FILLED` | The collection population, queue, or resource reservation is closed/full; auction mode does not emit this merely because eight candidates arrived | Re-poll `/state`; if still open, back off and inspect validator capacity telemetry |
| `WINDOW_MISMATCH` | `window_start` in your request doesn't match the active batcher | Refresh `/state` and retry with the current `window_n` |
| `STALE_ROUND` | Your signed `drand_round` is older than the validator round at precommit/direct-body arrival. Backward tolerance is zero. | Compute the drand round immediately before final serialization and precommit, never at sketch-build time. |
| `FUTURE_ROUND` | (v2.3) Your `drand_round` field is newer than the validator's current round. Implies clock skew. | Ensure miner host is NTP-synced. Drand quicknet rounds advance on a fixed wall-clock schedule; sending a future round means your clock is ahead of UTC. |
| `PROMPT_FULL` | `MAX_SUBMISSIONS_PER_PROMPT = 10` pending groups already occupy this prompt | Pick a different prompt |
| `HASH_DUPLICATE` | Your operator already reserved this prompt or your tokens duplicate retained/recent content | Do not rotate hotkeys or replay a forced group; choose another prompt |
| `SEED_MISMATCH` | The client does not advertise forced-seed protocol v2 | Pull the current miner, rebuild, and confirm `protocol_version=2` |

**Rejected asynchronously by the worker (look up via `GET /verdicts/{hotkey}` or the R2 archive):**

| Reason | Meaning | Action |
|---|---|---|
| `WRONG_CHECKPOINT` | `checkpoint_hash` does not match the active HF revision | Re-poll `/state`, update revision, retry. Most common transient reject — happens briefly after every new checkpoint publish. |
| `WRONG_RANDOMNESS` | `commit.beacon.randomness` doesn't match the validator's per-window seed (`state.randomness` on v2.3+; locally-derived `H(block_hash + drand)` on v2.2). Almost always caused by reusing a sketch built for an earlier window. | (v2.3) Read `state.randomness` from `/state` directly; do not re-derive locally. (v2.2) Derive per-window from chain + drand. In both cases: tag each sketch with the window it was built for and discard before firing if the window has advanced. |
| `BAD_PROMPT_IDX` | `prompt_idx` out of range for the active environment | Use the env's prompt-index space (`0..N-1`). v2.3 / OpenMathInstruct-2: `N ≈ 14_000_000`. |
| `PROMPT_IN_COOLDOWN` | `prompt_idx` was in the active cooldown set | v2.3: `BATCH_PROMPT_COOLDOWN_WINDOWS = 1_000_000` makes prompts effectively single-use. Read `cooldown_prompts[]` from `/state` **before each pick** and skip anything in the list. |
| `SUPERSEDED` | Historical only; current same-prompt competition resolves at auction seal | Upgrade parsers that still expect the old runner-up flow |
| `OUT_OF_ZONE` | σ of your 8 rewards is below threshold (`SIGMA_MIN = 0.43` steady, `0.33` during the first `BOOTSTRAP_WINDOWS = 100` windows) | Pick a prompt where your model gets 2–6 / 8 correct — not 0/8 or 8/8 |
| `REWARD_MISMATCH` | OpenMath reward claim disagreed with recomputation, or a Code grader worker crashed ambiguously while handling the candidate | Recheck Math parsing; for Code, report repeatable crash-triggering output rather than retrying indefinitely |
| `GRAIL_FAIL` | At seal, a ranked or forensic-sampled sketch differs from the validator forward pass beyond tolerance | Match checkpoint, tokenizer, attention/runtime stack, and proof construction exactly |
| `LOGPROB_MISMATCH` | Per-token log-prob deviation from validator's recompute exceeds `LOGPROB_IS_EPS = 0.10` | Same root cause as `GRAIL_FAIL` — quantization, attention kernel, or precision drift |
| `BAD_TERMINATION` | A rollout did not terminate naturally, hit the cap without EOS, or contains EOS padding/repeated stop-token tails | Confirm generation config matches protocol. Do not force `min_new_tokens`, suppress EOS, ride the 8192 cap, or append tokens after first EOS |
| `DISTRIBUTION_SUSPICIOUS` | Token probability distribution heuristics flagged low-entropy / cheater-like generation | Submit natural rollouts at `T_PROTO = 0.9`; avoid forced prefixes or constrained fillers |
| `WRONG_ROLLOUT_COUNT` | Group has fewer or more than `M_ROLLOUTS = 8` rollouts | Always submit exactly 8 |
| `BAD_SCHEMA` / `BAD_TOKENS` | Submission payload malformed | Validate against the protocol schema |
| `PROMPT_MISMATCH` | Canonical prompt tokens for `prompt_idx` don't match the request | Re-derive prompt tokens from the env's deterministic mapping |
| `BAD_SIGNATURE` | GRAIL commit signature failed | Check wallet hotkey and signing code |
| `WORKER_DROPPED` | The batcher swapped before dequeue, or the Code grader had a retryable infrastructure outage. Grader-outage quota is refunded. | Re-poll and retry later; sustained events indicate validator backpressure or grader health problems |

`PROMPT_IN_COOLDOWN` is the most common **persistent** rejection caused by miner code: if your picker doesn't read `cooldown_prompts[]` before each pick, you will repeatedly submit prompts the validator has already cooled. Read the field — it's small and refreshes every `/state` call. The dashboard surfaces this directly on the miner drawer.

### Real-time verdict feedback (`/verdicts/{hotkey}`)

Under the production worker path `/submit` returns only `accepted=True reason="submitted"`. The first `/verdicts` result reports pool admission. The final result arrives after the collection deadline and seal-time proof. Identify it by non-null `selected_for_batch` and `rewarded`; do not treat the first `ACCEPTED` as a win.

The validator exposes the real per-submission verdicts via:

```
GET http://<validator-host>:<validator-port>/verdicts/{your_hotkey}?since=<unix_ts>
```

Response (`VerdictsResponse` in `reliquary/protocol/submission.py`):

```json
{
  "verdicts": [
    {"merkle_root": "ab12...64hex", "window_n": 1858, "accepted": true, "reason": "accepted", "ts": 1747353600.5},
    {"merkle_root": "ab12...64hex", "window_n": 1858, "accepted": true, "reason": "accepted", "selected_for_batch": true, "rewarded": true, "canonical_rank": 2, "ts": 1747353901.1},
    {"merkle_root": "ef56...64hex", "window_n": 1858, "accepted": false, "reason": "grail_fail", "accepted_into_pool": true, "selected_for_batch": false, "rewarded": false, "reject_stage": "auction_seal", "ts": 1747353902.0}
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
                if v.get("selected_for_batch") is True:
                    logger.info(
                        "verdict WON win=%d rank=%s mr=%s",
                        v["window_n"], v.get("canonical_rank"),
                        v["merkle_root"][:12],
                    )
                elif v.get("selected_for_batch") is False and v["accepted"]:
                    logger.info(
                        "verdict NOT_SELECTED win=%d rank=%s mr=%s",
                        v["window_n"], v.get("canonical_rank"),
                        v["merkle_root"][:12],
                    )
                elif v["accepted"]:
                    logger.info(
                        "verdict POOL_ACCEPTED win=%d mr=%s",
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

Log the fire-time response as `SUBMITTED`, the worker result as `POOL_ACCEPTED`, and only the seal-time selected result as `WON`. A non-winner is ordinary auction competition, not a rejection or a reason to quarantine the model.

Polling is optional for protocol validity, but it is the authoritative live feedback path for auction outcome.

---

## Requirements

| Item | Requirement |
|---|---|
| OS | Linux (tested on Ubuntu 22.04 / 24.04) |
| Python | 3.11 or newer |
| GPU | 1x or 2x NVIDIA GPU, at least 24 GB VRAM each. Reference config: generation and proof on separate devices; one larger device also works. |
| CUDA | 12.x with `flash-attn`-compatible drivers |
| RAM | 32 GB minimum |
| Disk | 50 GB (model weights and HF cache) |
| Network | Stable outbound HTTPS to HF Hub and the active validator |
| Bittensor wallet | Created and registered on netuid 81 |

No R2 or S3 credentials are needed on the miner — only the validator uploads the window dataset.

### Inference runtime parity

Hardware speed is not the protocol contract; generation/proof numerics are.
Match the validator's pinned model, tokenizer, Torch, Transformers, attention
implementation, dtype, and optional-kernel set. The current validated stack is
Torch `2.7.0+cu128`, Transformers `5.9.0`, flash-linear-attention `0.5.0`, and
no `causal-conv1d`. Do not install a different fast-path kernel on miners alone:
that can increase miner-validator drift even when it improves throughput.

Different GPU models may still shift logits and GRAIL sketches. Use the runtime
fingerprint and final verdict telemetry to canary a new GPU type before scaling
it. Exact-CDF enforcement remains off because cached generation and full
teacher forcing are not bit-identical on every supported stack.

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
    --checkpoint Qwen/Qwen3.5-2B \
    --environments openmathinstruct,opencodeinstruct \
    --validator-url http://<owner-validator-ip>:8888 \
    --log-level INFO
```

Once the owner validator earns `validator_permit`, you can drop `--validator-url` and the miner will auto-discover it from the metagraph.

The miner queries the validator at boot and downloads the current HF checkpoint automatically. You do not need to find or pin the checkpoint hash manually.

### Qwen3.5 model reset

The live model family is `Qwen/Qwen3.5-2B`. This is a hard tokenizer/model reset from earlier Qwen3 checkpoints:

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
- **CUDA out of memory**: two copies of Qwen3.5-2B require roughly 9-10 GB bfloat16 before activations and KV/cache overhead. The KV cache at the 32k context cap dominates under long thinking rollouts, so headroom above the weights is what matters. A single 24 GB GPU is comfortable; below that you may hit OOM under proofs plus generation. Use a larger GPU or split generation/proofs across devices.
- **`GRAIL_FAIL` / `LOGPROB_MISMATCH`**: your local proof compute diverged from the validator's. Most often caused by a different `attn_implementation` build, CUDA/torch version mismatch, or wrong checkpoint. Re-install on a clean environment and confirm you are on the same HF revision as the validator (check `/state`).
- **`REWARD_MISMATCH`**: for OpenMath, validator-side reward computation disagreed with the miner's claimed `rollout.reward`. For OpenCode it may also report an ambiguous grader worker crash. Recheck Math parsing or inspect repeatable crash-triggering Code output.
- **All submissions land `OUT_OF_ZONE`**: the prompts you are selecting are too easy (`sigma ~= 0`) or too hard (`sigma ~= 0`) for the current checkpoint. On OpenCode, this often means all-zero or all-pass structured-case vectors. Split metrics by environment before changing global filters.
- **Persistent `WRONG_CHECKPOINT`**: the miner is not picking up the latest revision from `/state`. Ensure the poll loop reads `checkpoint_revision` before each submission.
