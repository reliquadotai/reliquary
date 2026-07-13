# Concepts

How Reliquary works, why it is built this way, and what each mechanism defends against.

## The thesis

DAPO ([ByteDance Seed + Tsinghua, March 2025](https://arxiv.org/abs/2503.14476)) reached state-of-the-art on AIME 2024 (50 points, from a 30-point naive-GRPO baseline) in **50% of the training steps** used by DeepSeek-R1-Zero-Qwen-32B. The paper stacks four techniques; the published ablation (Table 1) credits **Dynamic Sampling** — discarding rollout groups where all answers have the same reward — with the single largest contribution of the stack: **+8 AIME 2024 points** on top of all four other techniques combined (42 → 50). It is a data-selection change, not a loss change. The core finding is that at this scale, *which prompts you train on* is the single largest lever anyone has published.

The catch: DAPO's filter is reactive. The system generates a rollout group, measures its reward variance, discards it if the variance is zero. As the policy strengthens, intermediate-difficulty prompts become rarer, the rejection rate rises, and more compute is spent generating groups the trainer will throw away. The paper flags this cost explicitly.

Reliquary turns this filter problem into a **prediction market**. Every training window, independent GPU miners bet their own compute on which prompt sits at the policy's learning frontier (high σ). The generate-then-discard cost is pushed outside the validator — miners who pick poorly burn their own rollouts; miners who pick well earn batch slots and emission. As the policy matures and the frontier narrows, the market becomes *more* valuable, not less — exactly the regime where DAPO's centralized filter pays the highest tax.

**Expected outcome.** Match or exceed DAPO's 50%-training-step efficiency, with a widening edge as training progresses. The structural argument is that an ex-ante predictor — miners committing compute only to prompts they believe are in-zone — dominates a reactive discard-on-measurement filter on compute per gradient-rich group. The claim is directional, not benchmarked.

Three structural guarantees come with the market:

- **Forced curriculum diversity.** A prompt that enters a winning batch is locked out for `BATCH_PROMPT_COOLDOWN_WINDOWS = 1_000_000` windows in the current OpenMath phase. In practice, prompts are one-shot. This prevents collapse onto a handful of high-variance outliers; R2 cooldown rebuild uses a bounded recent-window lookback so startup stays cheap.
- **Cryptographic training-data provenance.** Every rollout carries a GRAIL sketch that binds the generation to the model weights that produced it. The validator re-verifies with its own forward pass. Fabricated data earns zero.
- **Validator-side reward authority and training quarantine.** For public-reward environments such as OpenMath, miners submit local reward claims and the validator independently recomputes them before accepting the group. For private-reward environments such as OpenCode, miners see prompts only; the validator owns hidden cases and computes rewards itself. Windows with high-confidence poison signatures are archived and credited but skipped for GRPO/publish.

The zone filter (`σ ≥ 0.43`, see below) is the mechanical realization of DAPO's Dynamic Sampling, reformulated to be reward-scale-agnostic. The miner-side incentive to predict σ *before* generating is what turns DAPO's post-hoc filter into an ex-ante market.

---

## The core loop

One full training window, step by step.

**1. Miners read `/state`.**
Miners poll `GET /state` continuously. The response (`GrpoBatchState`) carries `state`, `window_n`, `checkpoint_n`, `checkpoint_repo_id`, `checkpoint_revision`, and `cooldown_prompts`. If `checkpoint_n` has advanced since the last poll, the miner downloads the new HF revision before doing anything else.

**2. Miner picks a prompt.**
The miner selects a `prompt_idx` from one active environment that is not in that environment's cooldown set. OpenMath uses OpenMathInstruct-2 (`nvidia/OpenMathInstruct-2`) with public labels and validator-recomputed local reward claims. OpenCode uses a public prompt-only mirror while the validator keeps hidden structured cases private and computes rewards itself. The reference engine uses uniform-random sampling with rejection against the cooldown set. This is a baseline: smarter miner-side selection — predicting which prompts will pass the zone filter for the current checkpoint — is expected. See [mining.md §Prompt selection strategy](mining.md#prompt-selection-strategy).

**3. Miner generates M=8 rollouts.**
The miner runs exactly `M_ROLLOUTS = 8` completions at the protocol-fixed temperature `T_PROTO = 0.9`, `top_p = 1.0`, `top_k = 0`. The validator cannot prove a miner did not generate extra candidates before submission, so the protocol combines reward-claim verification, binary reward-distribution guards, and training quarantine to reduce the value and blast radius of reward-vector shaping.

**4. Miner builds GRAIL sketches.**
For each rollout the miner runs a bit-identical HuggingFace forward pass on the proof GPU to construct a GRAIL sketch commitment. The sketch binds the completion to the model's hidden-state activations. The miner signs the commit and packages everything into a `BatchSubmissionRequest` that includes `checkpoint_hash` (the HF revision from the last `/state` response). In OpenMath, `rollout.reward` must match the miner's local `env.compute_reward` value; the validator recomputes it and rejects mismatches. In OpenCode, local reward fields are placeholders and the validator computes hidden-case rewards before applying the zone filter.

**5. Miner submits.**
`POST /submit` sends the request to the validator. In production, the HTTP response is provisional (`accepted=True, reason="submitted"`) once the request is queued. A background worker later runs the full verification pipeline (see below) and appends the submission to the open window's valid pool only if it passes.

**6. Validator verifies, filters, selects batch.**
The validator checks: window match → checkpoint hash → prompt index bounds → cooldown → per-prompt capacity → schema/token invariants → prompt-token binding → rollout-hash dedup → reward verification or validator-authoritative reward scoring → zone filter (`σ ≥ 0.43`) → signatures/randomness → GRAIL sketch → termination/logprob/token-distribution checks. Any failure returns a `RejectReason`. Valid submissions accumulate per active environment. Once the active environment targets have enough eligible distinct prompts, `seal_event` fires after the drand-round drain.

**7. Validator accumulates clean signal and runs a balanced GRPO step.**
State transitions to `TRAINING`. Before retention, the validator assesses the selected groups and current reject profile. Quarantined windows remain archived and credited but do not enter training. Clean partial batches are retained across windows under the exact public checkpoint revision, capped at one target batch per environment. Once every active environment is full, the validator assesses the balanced retained batch again and runs `train_step()`. A checkpoint change discards pending samples before any new-revision samples are retained, so one optimizer step never mixes generation policies.

**8. Validator publishes a new checkpoint.**
State transitions to `PUBLISHING`. Every `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS = 10` trained windows the model is saved locally, pushed to HF Hub, and signed: `ed25519(checkpoint_n || revision)`. The signed manifest is installed in `/checkpoint`. Between publishes the miners stay on the last-published revision (enforced by the checkpoint hash gate). The window dataset is archived to R2, including quarantine metadata when present.

**9. State → READY → OPEN.**
The next window opens immediately. Winning prompts enter one-shot cooldown. Once per subnet epoch the validator calls `set_weights` on-chain with the current EMA snapshot.

**Safety net.** Normal windows seal by valid-prompt targets. Sparse windows force-seal partial when the validator has drained queued/proof work and either has at least `SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS = 4` distinct valid prompts with no new valid prompt for `SPARSE_VALID_IDLE_SEAL_SECONDS = 180`, or reaches `SPARSE_VALID_MAX_WINDOW_SECONDS = 600` with fewer than the target. The older `WINDOW_TIMEOUT_SECONDS = 7200` remains the outer safety net. Partial windows are archived and credited immediately; clean selected groups can complete a later checkpoint-consistent balanced training batch. Unused reward slots still contribute to the burn weight for `UID_BURN`.

---

## Why each mechanism exists

### GRAIL proofs — anti-fabrication

A GRAIL sketch is a compact linear commitment over a sampled subset of the model's last hidden-state activations for a given completion. The validator recomputes the forward pass on the same tokens with the same model, draws the same random challenge positions (seeded from the window's randomness), and checks that the two sketches agree within a position-dependent tolerance (`PROOF_SKETCH_TOLERANCE_BASE = 5000`, growth `= 5.0 × sqrt(position)`). The tolerance is calibrated empirically to cover cross-GPU floating-point drift — legitimate proofs pass even on different hardware, while fabricated activations diverge by orders of magnitude.

Because each rollout's sketch is bound to the specific token sequence and the model's weights, a miner cannot fabricate completions, copy another miner's rollouts, or replay proofs from a different model revision without failing the sketch check.

### Zone filter — only train on useful frontier prompts

`σ` is the population standard deviation of the eight rollout rewards in a group. A group with `σ < 0.43` carries essentially no gradient signal for GRPO: either every rollout succeeds (all advantages ≈ 0) or every rollout fails (same). Dropping these groups saves compute without losing learning.

Binary equivalence note: OpenMath rewards are binary `{0, 1}` (the validator extracts the final `\boxed{...}` answer and compares after conservative normalization). With binary rewards, `σ = sqrt(p(1−p))` where `p = k/8`. `σ(p=2/8) ≈ σ(p=6/8) ≈ 0.433`, so `σ ≥ 0.43` admits k=2..6.

Bootstrap phase (`BOOTSTRAP_WINDOWS = 100` windows from `SUBNET_START_BLOCK`): threshold relaxes to `σ ≥ 0.33` (binary equivalent: k ∈ [1, 7]) to keep batches filling while miner population and env coverage are thin.

### Cooldown — one-shot prompt rotation

Once a `prompt_idx` enters the winning batch it is ineligible for `BATCH_PROMPT_COOLDOWN_WINDOWS = 1_000_000` windows. With OpenMathInstruct-2's large prompt pool, this makes prompts effectively one-shot across any realistic run.

The cooldown map is rebuilt from recent R2 archives using `COOLDOWN_REBUILD_LOOKBACK`, not the full one-shot horizon. This preserves recent curriculum state without forcing startup to download a million archives.

### Training quarantine — protect model health during exploit discovery

Training quarantine is a model-health gate, not an emission slash. When a
selected batch has high-confidence poison signatures, the validator archives
the window and credits emissions, but skips GRPO for that window. Current
hard signals include dense cap-length/extreme-length completion patterns and
high-risk reject spikes such as
`reward_distribution`, `bad_termination`, `tokens_mismatch`, or
`distribution_suspicious`. Hotkey concentration and reward-vector dominance
are archived as metrics, but do not quarantine by themselves: a single honest
miner may be the only one printing useful work in a sparse window, and binary
frontier mining naturally clusters around a small set of reward vectors.

The archive carries:

```text
training_quarantine = {quarantined, reasons, metrics}
```

This is the blast-radius control: if a new exploit appears, the network can
observe and account for the window without immediately teaching the model that
pattern.

### v2.3: Drand-round chronological ordering + multi-miner-per-prompt + emission split

> **Major design shift (May 2026).** v2.2's TCP-arrival FIFO and single-winner-per-prompt rules were replaced by drand-round chronological ordering with multi-miner-per-prompt and emission split. Full spec: [docs/superpowers/specs/2026-05-15-drand-ordering-and-prompt-split-design.md](superpowers/specs/2026-05-15-drand-ordering-and-prompt-split-design.md). Implementation in [PR #28](https://github.com/reliquadotai/reliquary/pull/28).

**Three coupled changes:**

1. **Per-window randomness is drand-only.** `block_hash` is dropped from the seed; randomness is `H(drand_randomness || drand_round)`. Window OPEN no longer waits on `chain.get_block_hash` — substrate flakiness can't stall window scheduling. The randomness is exposed on `/state.randomness` so miners use it directly instead of recomputing locally (avoids the entire class of `WRONG_RANDOMNESS` rejects from local-derivation bugs).

2. **Submissions carry a `drand_round` field.** Each `BatchSubmissionRequest` includes the drand quicknet round currently in progress at the wall-clock instant of POST. The validator gates with **zero tolerance** — too old → `STALE_ROUND`, too new → `FUTURE_ROUND`. Selection at seal time orders submissions by `drand_round` (chronological 3 s buckets), ties broken by `H(hotkey ‖ prompt_idx ‖ selection_digest)`. The validator derives `selection_digest` from the ordered token streams and environment; it deliberately does not reuse the full payload Merkle root, because miners may legally vary validator-overwritten metadata and must not gain a representative-selection grinding surface. **Co-location with the validator no longer wins the race** — geography and millisecond TCP advantages are irrelevant; being in the same drand round as the rest of the network is what matters.

3. **Multi-miner-per-prompt with emission split.** `SUPERSEDED` is deprecated. Up to `MAX_SUBMISSIONS_PER_PROMPT = 10` distinct hotkeys may submit on the same `prompt_idx`. At seal time each filled slot pays `pool / B_BATCH`, and within a slot the `K_p` miners who submitted for that prompt split equally. **Same-prompt sybil is strictly neutral** (N sybils on one prompt total `pool/B` — same as a single hotkey, minus N − 1 registration burns). Distinct-prompt sybil is additive but bounded by registration cost.

### Why this design (briefly)

v2.2 concentrated emission on whoever won the FIFO race for each prompt. Geography decided who that was — a miner in the validator's datacenter beat any non-co-located miner regardless of inference quality. v2.1's original `signed_round` ordering tried to fix this but was grindable: miners could pin the round at the floor of the accepted range and game ordering. v2.3 chains the round to drand (unforgeable per-window) and replaces the per-prompt single-winner rule with multi-winner emission-split — so an honest GRAIL-validated rollout earns something even if you arrived second. The full reasoning (grinding attack analysis, sybil-neutrality proof, DoS bounds) is in the design spec.

### Historical (v2.2): FIFO by TCP arrival — deprecated

Pre-v2.3 the validator ranked submissions by `arrived_at` (validator-side timestamp at accept). The first submission to claim a given `prompt_idx` for the current window won that slot; subsequent submissions on the same prompt were rejected `SUPERSEDED` before any heavy work ran. v2.3 retains `arrived_at` in `ValidSubmission` for forensics but no longer drives selection.

### EMA scoring — one payment per window, not per submission

Before EMA, weights were submitted as "fraction of batch slots won over the interval, counted from scratch each epoch". This lost intra-epoch data because Bittensor records only the last `set_weights` call of an epoch for emissions.

The EMA fixes this: after each window, every hotkey's score is updated as:

```
score_new = α × share_this_window + (1 − α) × score_old
```

where `share_this_window` is the final per-hotkey emission share from the selected prompt set, including same-prompt splits. `α = EMA_ALPHA = 2 / (72 + 1) ≈ 0.027`. With a 72-window history, this gives a ~25-window half-life. A miner that stops contributing loses half its score in ~25 windows. The EMA is replayed from R2 archives at startup (no local state file) — loss of local disk does not lose scoring history.

At each `set_weights` call the validator submits the current EMA values directly. The sum of all EMA scores is the smoothed fill rate; `burn = max(0, 1 − sum)` goes to `UID_BURN = 0`.

### Checkpoint hash gate — miners always run the current model

Every `BatchSubmissionRequest` includes `checkpoint_hash` — the HF commit revision the miner loaded. The validator compares this to `current_checkpoint_hash` (the revision of the most recently published HF snapshot). A mismatch returns `WRONG_CHECKPOINT` immediately, before any GRAIL verification, saving both parties compute.

This guarantees that training data always reflects the currently-published policy. Without it, a stale miner could produce rollouts from an old model, creating a training distribution mismatch.

### Publish every N trained windows — HF cannot keep up with per-step pushes

The base model is Qwen3.5-2B (~2 billion parameters, sharded safetensors, thinking chat template). Pushing a new safetensors snapshot to HF Hub on every window (roughly every 60 seconds under load) is infeasible due to Git LFS latency and HF rate limits. The validator publishes to HF every `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS = 10` successful balanced optimizer steps. A partial window may contribute retained samples but does not increment the cadence by itself. Quarantined windows are archived/credited but excluded from the accumulator. Between publishes, miners stay on the last-published revision — the hash gate keeps them there. `checkpoint_n` only increments on a successful publish, so the gate remains stable across the publish gap.

---

## Economic model

### How a miner earns

1. Submit a valid in-zone group on a non-cooldown prompt when the window is `OPEN`.
2. Land on a winning prompt after drand/canonical selection. Multiple miners may submit on the same prompt up to `MAX_SUBMISSIONS_PER_PROMPT`; they split that prompt's slot emission.
3. Each winning prompt contributes `(pool / B_BATCH) / K_p` to each miner that submitted on it, where `K_p` is the number of miners sharing that prompt.
4. Once per subnet epoch (~360 blocks), the validator calls `set_weights` on-chain with the current EMA values. All validators submit inside a shared ~20-block window before the epoch boundary so they converge on identical weights. Your emission for the epoch is proportional to your EMA score.

### Rough expected earnings

Suppose the network emits `E` TAO per epoch. You land on an average of `s` unshared winning prompts per window. The EMA converges to approximately `s / B_BATCH = s / 8` of the total filled-slot budget. Your share of emissions per epoch is approximately:

```
(s / 8) / (sum of all miners' EMA scores)
```

A miner consistently landing on 2 unshared winning prompts per window gets roughly `2/8 = 25%` of the filled-slot emission budget. Shared prompts divide that prompt's share by `K_p`.

### What disqualifies a submission

| Reject reason | Cause | Remediation |
|---|---|---|
| `WINDOW_NOT_ACTIVE` | Window is in `TRAINING` or `PUBLISHING` | Wait and re-poll `/state` |
| `WINDOW_MISMATCH` | `window_start` in request does not match current window | Refresh `/state` and retry |
| `WRONG_CHECKPOINT` | `checkpoint_hash` is stale | Re-poll `/state`, update revision, retry |
| `BAD_PROMPT_IDX` | `prompt_idx >= len(env)` | Use a valid index from the environment |
| `PROMPT_MISMATCH` | `tokens[:prompt_length]` does not match the canonical tokenization of `env.get_problem(prompt_idx).prompt` (CoT prefix, alternate chat template, custom system prompt, etc.) | Use the env's exact prompt string and the pinned tokenizer; do not modify the prompt before generation |
| `PROMPT_IN_COOLDOWN` | Prompt is in the active one-shot cooldown set | Pick a different `prompt_idx` |
| `PROMPT_FULL` | `MAX_SUBMISSIONS_PER_PROMPT` validated submissions already exist for this prompt | Pick a less crowded prompt |
| `HASH_DUPLICATE` | Rollout tokens duplicate a recently accepted rollout hash | Generate fresh tokens; do not replay |
| `REWARD_MISMATCH` | Validator reward computation failed or produced a non-finite value | Treat as malformed output/env failure; miner rewards are not trusted |
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
| Spam the same prompt every window | One-shot cooldown blocks re-entry after the prompt wins | 0 earnings after first winning inclusion |
| Generate extra rollouts to select favorable reward vectors | Monitoring and training quarantine reduce blast radius; long-term private tasks / commit-first sampling are the durable fix | Some shaping value remains until durable mitigations land |
| Submit extremely fast | Drand-round ordering and prompt emission split reduce pure TCP/colo advantage | Timing still matters for being in-window, not millisecond FIFO |
| Run a stale model | `WRONG_CHECKPOINT` rejects before GRAIL | 0 earnings |

---

## Known limitations

- **Public OpenMath labels.** The validator verifies miner reward claims, but OpenMath labels are public/reconstructable. OpenCode now uses private validator-side hidden cases, but the long-term moat still improves with private/generated tasks, archive redaction/delay, property tests, and possibly commit-first sampling.
- **Single trainer.** The current deployment assumes a single trainer writing to R2. Multiple trainers in the same bucket would collide on archive keys (`reliquary/dataset/window-<N>.json.gz`). Multi-trainer consensus is future work.
- **Optimizer and scheduler state not persisted.** A validator restart resets AdamW momentum and the LR scheduler step count to zero. Training regresses for `LR_WARMUP_WINDOWS = 10` windows before stabilizing. Minimize restarts.
- **No automatic HF checkpoint garbage collection.** Every publish creates a new HF commit. Old revisions accumulate. Plan manual or cron-based cleanup.
- **No automatic R2 retention.** Every window archives ~1 MB compressed. Add a bucket lifecycle rule for archives older than your retention window.
- **HF bootstrap auth.** `_bootstrap_state_from_external` calls `HfApi().list_repo_commits` to count published checkpoints. For private repos, `HF_TOKEN` must be set at startup. Public repos are readable without authentication but the call still hits the HF API rate limit (~500 req/hour for unauthenticated). Set `HF_TOKEN` anyway to avoid rate-limit failures on restart.

---

## Further reading

- [docs/mining.md](mining.md) — operator guide for miners
- [docs/validating.md](validating.md) — operator guide for validators
