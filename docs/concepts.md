# Concepts

How Reliquary works, why it is built this way, and what each mechanism defends against.

## The thesis

DAPO ([ByteDance Seed + Tsinghua, March 2025](https://arxiv.org/abs/2503.14476)) reached state-of-the-art on AIME 2024 (50 points, from a 30-point naive-GRPO baseline) in **50% of the training steps** used by DeepSeek-R1-Zero-Qwen-32B. The paper stacks four techniques; the published ablation (Table 1) credits **Dynamic Sampling** — discarding rollout groups where all answers have the same reward — with the single largest contribution of the stack: **+8 AIME 2024 points** on top of all four other techniques combined (42 → 50). It is a data-selection change, not a loss change. The core finding is that at this scale, *which prompts you train on* is the single largest lever anyone has published.

The catch: DAPO's filter is reactive. The system generates a rollout group, measures its reward variance, discards it if the variance is zero. As the policy strengthens, intermediate-difficulty prompts become rarer, the rejection rate rises, and more compute is spent generating groups the trainer will throw away. The paper flags this cost explicitly.

Reliquary turns this filter problem into a **prediction market**. Every training window, independent GPU miners bet their own compute on which prompt sits at the policy's learning frontier (high σ). The generate-then-discard cost is pushed outside the validator — miners who pick poorly burn their own rollouts; miners who pick well earn batch slots and emission. As the policy matures and the frontier narrows, the market becomes *more* valuable, not less — exactly the regime where DAPO's centralized filter pays the highest tax.

**Expected outcome.** Match or exceed DAPO's 50%-training-step efficiency, with a widening edge as training progresses. The structural argument is that an ex-ante predictor — miners committing compute only to prompts they believe are in-zone — dominates a reactive discard-on-measurement filter on compute per gradient-rich group. The claim is directional, not benchmarked.

Two structural guarantees come with the market:

- **Forced curriculum diversity.** A prompt that enters a training batch is locked out for `BATCH_PROMPT_COOLDOWN_WINDOWS = 50` windows. The validator sees roughly 400 distinct prompts before any one recurs — no collapse onto a handful of high-variance outliers, a failure mode that a single-node pipeline has no automatic defense against.
- **Cryptographic training-data provenance.** Every rollout carries a GRAIL sketch that binds the generation to the model weights that produced it. The validator re-verifies with its own forward pass. Fabricated data earns zero.

The zone filter (`σ ≥ 0.43`, see below) is the mechanical realization of DAPO's Dynamic Sampling, reformulated to be reward-scale-agnostic. The miner-side incentive to predict σ *before* generating is what turns DAPO's post-hoc filter into an ex-ante market.

---

## The core loop

One full training window, step by step.

**1. Miners read `/state`.**
Miners poll `GET /state` continuously. The response (`GrpoBatchState`) carries `state`, `window_n`, `checkpoint_n`, `checkpoint_repo_id`, `checkpoint_revision`, and `cooldown_prompts`. If `checkpoint_n` has advanced since the last poll, the miner downloads the new HF revision before doing anything else.

**2. Miner picks a prompt.**
The miner selects a `prompt_idx` from the environment (Hendrycks MATH, `qwedsacf/competition_math` mirror, ~12 500 problems, 5 difficulty levels, 7 subjects) that is not in `cooldown_prompts`. The reference engine uses uniform-random sampling with rejection against the cooldown set. This is a baseline: smarter miner-side selection — predicting which prompts will pass the zone filter for the current checkpoint — is expected and directly rewarded by FIFO (fewer `OUT_OF_ZONE` rejects → earlier successful `/submit` → batch slot won). See [mining.md §Prompt selection strategy](mining.md#prompt-selection-strategy).

**3. Miner generates M=8 rollouts.**
The miner runs exactly `M_ROLLOUTS = 8` completions at the protocol-fixed temperature `T_PROTO = 0.9`, `top_p = 1.0`, `top_k = 0`. No cherry-picking — all eight go in the submission regardless of their rewards.

**4. Miner computes local rewards and builds GRAIL sketches.**
For each rollout the miner calls `env.compute_reward`, then runs a bit-identical HuggingFace forward pass on the proof GPU to construct a GRAIL sketch commitment. The sketch binds the completion to the model's hidden-state activations. The miner signs the commit and packages everything into a `BatchSubmissionRequest` that includes `checkpoint_hash` (the HF revision from the last `/state` response).

**5. Miner submits.**
`POST /submit` sends the request to the validator. The validator runs the full verification pipeline (see below). On success the submission is appended to the open window's valid pool. The response is immediate (`accepted=True`) — the heavy GRAIL verification runs in a background worker.

**6. Validator verifies, filters, selects batch.**
The validator checks: window match → checkpoint hash → prompt index bounds → cooldown → per-prompt FIFO short-circuit (`SUPERSEDED` if another miner already claimed this `prompt_idx` this window) → reward match → zone filter (`σ ≥ 0.43`) → GRAIL sketch. Any failure returns a `RejectReason` immediately. Valid submissions accumulate. Once `B_BATCH = 16` submissions with distinct `prompt_idx` values pass, `seal_event` fires.

**7. Validator runs a GRPO step.**
State transitions to `TRAINING`. `train_step()` computes group-relative advantages from each group's rewards, runs a PPO-clipped surrogate loss + KL penalty against the frozen reference model, and applies one AdamW step. The EMA scores are updated for all miners seen this window.

**8. Validator publishes a new checkpoint.**
State transitions to `PUBLISHING`. Every `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS = 10` windows the model is saved locally, pushed to HF Hub, and signed: `ed25519(checkpoint_n || revision)`. The signed manifest is installed in `/checkpoint`. Between publishes the miners stay on the last-published revision (enforced by the checkpoint hash gate). The window dataset is archived to R2.

**9. State → READY → OPEN.**
The next window opens immediately. Batched prompts enter a 50-window cooldown. Once per subnet epoch the validator calls `set_weights` on-chain with the current EMA snapshot.

**Safety net.** If fewer than `B_BATCH` valid submissions arrive within `WINDOW_TIMEOUT_SECONDS = 600` seconds, the window seals on whatever arrived. Unused batch slots contribute to the burn weight for `UID_BURN`.

---

## Why each mechanism exists

### GRAIL proofs — anti-fabrication

A GRAIL sketch is a compact linear commitment over a sampled subset of the model's last hidden-state activations for a given completion. The validator recomputes the forward pass on the same tokens with the same model, draws the same random challenge positions (seeded from the window's randomness), and checks that the two sketches agree within a position-dependent tolerance (`base = 6000`, growth `= 5.0 × sqrt(position)`). The tolerance is calibrated empirically to cover cross-GPU floating-point drift — legitimate proofs pass even on different hardware, while fabricated activations diverge by orders of magnitude.

Because each rollout's sketch is bound to the specific token sequence and the model's weights, a miner cannot fabricate completions, copy another miner's rollouts, or replay proofs from a different model revision without failing the sketch check.

### Zone filter (σ ≥ 0.43) — only train on learnable prompts

`σ` is the population standard deviation of the eight rollout rewards in a group. A group with `σ < 0.43` carries essentially no gradient signal for GRPO: either every rollout succeeds (all advantages ≈ 0) or every rollout fails (same). Dropping these groups saves compute without losing learning.

Binary equivalence note: MATH rewards are binary `{0, 1}` (the validator extracts the final `\boxed{...}` answer and compares after conservative LaTeX normalization — `\dfrac` → `\frac`, strip `\left`/`\right`, drop `\text{}`, collapse whitespace). With binary rewards, `σ = sqrt(p(1−p))` where `p = k/8`. `σ(p=2/8) ≈ 0.433`, so the `σ ≥ 0.43` gate is equivalent to "between 2 and 6 correct out of 8". The σ formulation is preferred because it is reward-scale-agnostic — the same threshold works for continuous partial-credit environments without any validator changes.

Bootstrap phase (`BOOTSTRAP_WINDOWS = 100` windows from `SUBNET_START_BLOCK`): threshold relaxes to `σ ≥ 0.33` (binary equivalent: k ∈ [1, 7]) to keep batches filling while miner population and env coverage are thin.

### Cooldown (50 windows) — curriculum rotation

Once a `prompt_idx` enters the training batch it is ineligible for the next `BATCH_PROMPT_COOLDOWN_WINDOWS = 50` training steps. This prevents the policy from overfitting to a small set of high-signal prompts. With 50 windows of cooldown and `B_BATCH = 16` distinct prompts per window, roughly 800 distinct prompts are trained before any one prompt can recur.

The cooldown map is rebuilt from R2 archives at validator startup — up to 50 recent windows are downloaded and replayed — so the curriculum state survives restarts without needing a local state file for cooldowns specifically.

### FIFO by TCP arrival — speed matters, not cherry-picking

Submissions are ranked by `arrived_at` — the validator-side timestamp captured under the batcher lock the moment a submission clears the full validation pipeline. The first submission to claim a given `prompt_idx` for the current window wins that slot; every subsequent submission for the same prompt is rejected `SUPERSEDED` before any heavy work (reward + GRAIL forward pass) runs. Across distinct prompts, batch members are ordered by `arrived_at` with a deterministic hotkey/prompt/merkle-root tiebreak for the (rare) case of identical timestamps.

Cherry-picking "easy" prompts requires extra inference compute → takes longer → arrives later → another miner has already claimed the slot. The flat payment per batch slot means there is no reward multiplier for a higher-reward prompt either. The optimal strategy is to submit as fast as possible on whatever prompt passes the zone filter.

Why arrival time and not a miner-claimed value? In the v2.1 design, FIFO ranked submissions by a miner-supplied `signed_round` (a drand round) bounded by a `STALE_ROUND_LAG_MAX` lag. Because the value was claimed by the miner, a sophisticated miner could always pin `signed_round` at the floor of the accepted range and win FIFO regardless of true arrival order. v2.2 removes the field entirely and orders by validator-set `arrived_at` instead. Miners cannot manipulate it.

### EMA scoring — one payment per training step, not per submission

Before EMA, weights were submitted as "fraction of batch slots won over the interval, counted from scratch each epoch". This lost intra-epoch data because Bittensor records only the last `set_weights` call of an epoch for emissions.

The EMA fixes this: after each window, every hotkey's score is updated as:

```
score_new = α × (slots_won / B_BATCH) + (1 − α) × score_old
```

where `α = EMA_ALPHA = 2 / (72 + 1) ≈ 0.027`. With a 72-window history, this gives a ~25-window half-life. A miner that stops contributing loses half its score in ~25 windows. The EMA is replayed from R2 archives at startup (no local state file) — loss of local disk does not lose scoring history.

At each `set_weights` call the validator submits the current EMA values directly. The sum of all EMA scores is the smoothed fill rate; `burn = max(0, 1 − sum)` goes to `UID_BURN = 0`.

### Checkpoint hash gate — miners always run the current model

Every `BatchSubmissionRequest` includes `checkpoint_hash` — the HF commit revision the miner loaded. The validator compares this to `current_checkpoint_hash` (the revision of the most recently published HF snapshot). A mismatch returns `WRONG_CHECKPOINT` immediately, before any GRAIL verification, saving both parties compute.

This guarantees that training data always reflects the currently-published policy. Without it, a stale miner could produce rollouts from an old model, creating a training distribution mismatch.

### Publish-every-N (10 windows) — HF cannot keep up with per-step pushes

The base model is Qwen3-4B-Instruct (~4 billion parameters, ~8 GB in bfloat16). Pushing a new safetensors file to HF Hub on every window (roughly every 60 seconds under load) is infeasible due to Git LFS latency and HF rate limits. The validator trains every window in-memory but publishes to HF every `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS = 10` windows. Between publishes, miners stay on the last-published revision — the hash gate keeps them there. `checkpoint_n` only increments on a successful publish, so the gate remains stable across the publish gap.

---

## Economic model

### How a miner earns

1. Submit a valid in-zone group on a non-cooldown prompt when the window is `OPEN`.
2. Be among the first `B_BATCH = 16` submissions with distinct `prompt_idx` values (FIFO by validator-side TCP arrival; first to claim each `prompt_idx` wins that slot).
3. Each batch slot you win contributes `1/B_BATCH` to your EMA update for that window.
4. Once per subnet epoch (~360 blocks), the validator calls `set_weights` on-chain with the current EMA values. All validators submit inside a shared ~20-block window before the epoch boundary so they converge on identical weights. Your emission for the epoch is proportional to your EMA score.

### Rough expected earnings

Suppose the network emits `E` TAO per epoch. You win an average of `s` batch slots per window. The EMA converges to approximately `s / B_BATCH = s / 16` of the total filled-slot budget. Your share of emissions per epoch is approximately:

```
(s / 16) / (sum of all miners' EMA scores)
```

A miner consistently winning 2 slots per window gets roughly `2/8 = 25%` of the epoch's filled-slot emissions.

### What disqualifies a submission

| Reject reason | Cause | Remediation |
|---|---|---|
| `WINDOW_NOT_ACTIVE` | Window is in `TRAINING` or `PUBLISHING` | Wait and re-poll `/state` |
| `WINDOW_MISMATCH` | `window_start` in request does not match current window | Refresh `/state` and retry |
| `WRONG_CHECKPOINT` | `checkpoint_hash` is stale | Re-poll `/state`, update revision, retry |
| `BAD_PROMPT_IDX` | `prompt_idx >= len(env)` | Use a valid index from the environment |
| `PROMPT_MISMATCH` | `tokens[:prompt_length]` does not match the canonical tokenization of `env.get_problem(prompt_idx).prompt` (CoT prefix, alternate chat template, custom system prompt, etc.) | Use the env's exact prompt string and the pinned tokenizer; do not modify the prompt before generation |
| `PROMPT_IN_COOLDOWN` | Prompt is in the active 50-window cooldown set | Pick a different `prompt_idx` |
| `SUPERSEDED` | Another submission for this `prompt_idx` already passed validation in this window — the slot is claimed | Submit faster next window, or pick a different `prompt_idx` |
| `REWARD_MISMATCH` | Claimed reward does not match validator's `env.compute_reward` | Check env and model version alignment |
| `OUT_OF_ZONE` | `σ < 0.43` (or `σ < 0.33` during bootstrap) | Pick a different prompt |
| `WRONG_ROLLOUT_COUNT` | Submission does not have exactly `M_ROLLOUTS = 8` rollouts | Always submit exactly 8 |
| `BAD_SIGNATURE` | GRAIL commit signature verification failed | Check wallet hotkey and signing code |
| `GRAIL_FAIL` | Sketch does not match validator's forward pass | Check checkpoint, `attn_implementation`, and CUDA version |

---

## Anti-cheat properties

| Attack | Mitigation | Realistic outcome |
|---|---|---|
| Fabricate completions | GRAIL sketch fails | 0 earnings |
| Resubmit old completions | `WRONG_CHECKPOINT` (rotation invalidates stale rollouts) | 0 earnings |
| Cherry-pick only easy prompts | σ ≈ 0 → `OUT_OF_ZONE` | 0 earnings |
| Spam the same prompt every window | Cooldown blocks re-entry for 50 windows | 0 earnings after first batch inclusion |
| Generate extra rollouts to select the most favorable 8 | Extra compute → later TCP arrival → another miner has already claimed the `prompt_idx` (`SUPERSEDED`) or the batch is full | 0 earnings |
| Submit extremely fast | Rewarded by FIFO selection | Expected, intended behavior |
| Run a stale model | `WRONG_CHECKPOINT` rejects before GRAIL | 0 earnings |

---

## Known limitations (v2.1)

- **Single trainer.** v2.1 assumes a single trainer writing to R2. Multiple trainers in the same bucket would collide on archive keys (`reliquary/dataset/window-<N>.json.gz`). Multi-trainer consensus is v2.2 work.
- **Optimizer and scheduler state not persisted.** A validator restart resets AdamW momentum and the LR scheduler step count to zero. Training regresses for `LR_WARMUP_WINDOWS = 10` windows before stabilizing. Minimize restarts.
- **No automatic HF checkpoint garbage collection.** Every publish creates a new HF commit. Old revisions accumulate. Plan manual or cron-based cleanup.
- **No automatic R2 retention.** Every window archives ~1 MB compressed. Add a bucket lifecycle rule for archives older than your retention window.
- **HF bootstrap auth.** `_bootstrap_state_from_external` calls `HfApi().list_repo_commits` to count published checkpoints. For private repos, `HF_TOKEN` must be set at startup. Public repos are readable without authentication but the call still hits the HF API rate limit (~500 req/hour for unauthenticated). Set `HF_TOKEN` anyway to avoid rate-limit failures on restart.

---

## Further reading

- [docs/mining.md](mining.md) — operator guide for miners
- [docs/validating.md](validating.md) — operator guide for validators
