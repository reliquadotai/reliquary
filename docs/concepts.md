# Concepts

How Reliquary works, why it is built this way, and what each mechanism defends against.

## The thesis

DAPO ([ByteDance Seed + Tsinghua, March 2025](https://arxiv.org/abs/2503.14476)) reached state-of-the-art on AIME 2024 (50 points, from a 30-point naive-GRPO baseline) in **50% of the training steps** used by DeepSeek-R1-Zero-Qwen-32B. The paper stacks four techniques; the published ablation (Table 1) credits **Dynamic Sampling** — discarding rollout groups where all answers have the same reward — with the single largest contribution of the stack: **+8 AIME 2024 points** on top of all four other techniques combined (42 → 50). It is a data-selection change, not a loss change. The core finding is that at this scale, *which prompts you train on* is the single largest lever anyone has published.

The catch: DAPO's filter is reactive. The system generates a rollout group, measures its reward variance, discards it if the variance is zero. As the policy strengthens, intermediate-difficulty prompts become rarer, the rejection rate rises, and more compute is spent generating groups the trainer will throw away. The paper flags this cost explicitly.

Reliquary turns this filter problem into a **prediction market**. Every training window, independent GPU miners bet their own compute on which prompt sits at the policy's learning frontier (high σ). The generate-then-discard cost is pushed outside the validator — miners who pick poorly burn their own rollouts; miners who pick well earn batch slots and emission. As the policy matures and the frontier narrows, the market becomes *more* valuable, not less — exactly the regime where DAPO's centralized filter pays the highest tax.

**Expected outcome.** Match or exceed DAPO's 50%-training-step efficiency, with a widening edge as training progresses. The structural argument is that an ex-ante predictor — miners committing compute only to prompts they believe are in-zone — dominates a reactive discard-on-measurement filter on compute per gradient-rich group. The claim is directional, not benchmarked.

Three structural guarantees come with the market:

- **Forced curriculum diversity.** A prompt that enters a winning batch is locked out for `BATCH_PROMPT_COOLDOWN_WINDOWS = 1_000_000` windows in the current OpenMath phase. In practice, prompts are one-shot. The validator enforces this by both dataset index and canonical rendered-prompt digest, so duplicate content at another index cannot win again. The run-keyed snapshots survive restarts.
- **Cryptographic training-data provenance.** Every rollout carries a GRAIL sketch that binds the generation to the model weights that produced it. The validator re-verifies with its own forward pass. Fabricated data earns zero.
- **Validator-side reward authority and training quarantine.** For OpenMath, miners submit local reward claims and the validator independently recomputes them. For OpenCode, the curated structured cases are public but the validator alone executes them in its trusted sandbox and overwrites miner reward placeholders. Windows with high-confidence poison signatures are archived and credited but skipped for GRPO/publish.

The zone filter (`σ ≥ 0.43`, see below) is the mechanical realization of DAPO's Dynamic Sampling, reformulated to be reward-scale-agnostic. The miner-side incentive to predict σ *before* generating is what turns DAPO's post-hoc filter into an ex-ante market.

---

## The core loop

One full training window, step by step.

**1. Miners read `/state`.**
Miners poll `GET /state` continuously. The response (`GrpoBatchState`) carries `state`, `window_n`, `checkpoint_n`, `checkpoint_repo_id`, `checkpoint_revision`, and `cooldown_prompts`. If `checkpoint_n` has advanced since the last poll, the miner downloads the new HF revision before doing anything else.

**2. Miner picks a prompt.**
The miner selects a `prompt_idx` from one active environment that is not in that environment's cooldown set. OpenMath uses OpenMathInstruct-2 (`nvidia/OpenMathInstruct-2`) with public labels and validator-recomputed local reward claims. OpenCode uses the pinned public curated prompt/case dataset, while the validator remains authoritative by executing cases in its sandbox. The reference engine uses uniform-random sampling with rejection against the cooldown set. This is a baseline: smarter miner-side selection — predicting which prompts will pass the zone filter for the current checkpoint — is expected. See [mining.md §Prompt selection strategy](mining.md#prompt-selection-strategy).

**3. Miner generates M=8 rollouts.**
The miner runs exactly `M_ROLLOUTS = 8` completions with the protocol-v2 forced sampling stream. The stream is derived from window randomness, prompt, checkpoint, rollout index, and token position; it deliberately excludes hotkey identity. The validator recomputes the same stream. Multiple hotkeys therefore cannot obtain different legal draws for one prompt, and clients advertising any protocol version other than `2` are rejected before grading while enforcement is active.

**4. Miner builds GRAIL sketches.**
For each rollout the miner runs a bit-identical HuggingFace forward pass on the proof GPU to construct a GRAIL sketch commitment. The sketch binds the completion to the model's hidden-state activations. The miner signs the commit and packages everything into a `BatchSubmissionRequest` that includes `checkpoint_hash` (the HF revision from the last `/state` response). In OpenMath, `rollout.reward` must match the miner's local `env.compute_reward` value; the validator recomputes it and rejects mismatches. In OpenCode, local reward fields are placeholders and the validator computes sandboxed structured-case rewards before applying the zone filter.

**5. Miner submits.**
The miner serializes and hashes its final signed body, obtains a signed upload receipt from `POST /submit/precommit`, then sends the exact bytes to `POST /submit`. A predeadline receipt grants bounded upload grace without extending generation. In production, the body response is provisional (`accepted=True, reason="submitted"`) once queued. A background worker runs bounded admission and reward grading. Its later `ACCEPTED` verdict means the group entered the pending auction pool; it is not yet a GRAIL pass or a paid slot.

**6. Validator admits, ranks, proves, and selects.**
Math and Code each collect an independent pending population for 100 seconds. Admission checks the window, checkpoint, protocol, registration/operator mapping, prompt, payload bounds, signatures, randomness, dedup, validator-authoritative rewards, and zone filter (`sigma >= 0.43`) without running the expensive model proof. At the deadline the validator drains pre-deadline work, freezes both populations, and ranks them by `std(rewards) * (1 - mean(rewards))`, validator-observed precommit drand round, then a post-deadline drand tie-break. It proves candidates top-down until it has at most eight distinct-prompt winners, under strict proof-attempt/wall-time bounds and with no operator winner cap. A failed high-ranked proof promotes the next candidate; an unselected candidate is never paid.

**7. Validator accumulates clean signal and runs a balanced GRPO step.**
State transitions to `TRAINING`. Before retention, the validator assesses the selected groups and current reject profile. Quarantined windows remain archived and credited but do not enter training. Clean partial batches are retained across windows under the exact public checkpoint revision, capped at one target batch per environment. Once every active environment is full, the validator assesses the balanced retained batch again and runs `train_step()`. A checkpoint change discards pending samples before any new-revision samples are retained, so one optimizer step never mixes generation policies.

**8. Validator publishes a new checkpoint.**
State transitions to `PUBLISHING`. Every `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS = 10` trained windows the model is saved locally, pushed to HF Hub, and signed: `ed25519(checkpoint_n || revision)`. The signed manifest is installed in `/checkpoint`. Between publishes the miners stay on the last-published revision (enforced by the checkpoint hash gate). The window dataset is archived to R2, including quarantine metadata when present.

**9. State → READY → OPEN.**
The next window opens immediately. Winning prompts enter one-shot cooldown. Once per subnet epoch the validator calls `set_weights` on-chain with the current EMA snapshot.

**Safety net.** Auction windows seal on their fixed 100-second collection deadline, not on candidate count. Queue drain and ranked proof work are bounded independently, and incomplete batches advance with unpaid slots burned. The legacy sparse-window breakers remain relevant only when the auction kill switch restores the old selector. Clean partial winners may complete a later checkpoint-consistent balanced training batch.

---

## Why each mechanism exists

### GRAIL proofs — anti-fabrication

A GRAIL sketch is a compact linear commitment over a sampled subset of the model's last hidden-state activations for a given completion. The validator recomputes the forward pass on the same tokens with the same model, draws the same random challenge positions (seeded from the window's randomness), and checks that the two sketches agree within a position-dependent tolerance (`PROOF_SKETCH_TOLERANCE_BASE = 5000`, growth `= 5.0 * sqrt(position)`). The tolerance absorbs calibrated numerical drift, but miners must still match the pinned inference stack; arbitrary kernel or hardware changes are not guaranteed compatible. Fabricated activations diverge far beyond honest numerical noise.

Because each rollout's sketch is bound to the specific token sequence and the model's weights, a miner cannot fabricate completions, copy another miner's rollouts, or replay proofs from a different model revision without failing the sketch check.

### Zone filter — only train on useful frontier prompts

`σ` is the population standard deviation of the eight rollout rewards in a group. A group with `σ < 0.43` carries essentially no gradient signal for GRPO: either every rollout succeeds (all advantages ≈ 0) or every rollout fails (same). Dropping these groups saves compute without losing learning.

Binary equivalence note: OpenMath rewards are binary `{0, 1}` (the validator extracts the final `\boxed{...}` answer and compares after conservative normalization). With binary rewards, `σ = sqrt(p(1−p))` where `p = k/8`. `σ(p=2/8) ≈ σ(p=6/8) ≈ 0.433`, so `σ ≥ 0.43` admits k=2..6.

Bootstrap phase (`BOOTSTRAP_WINDOWS = 100` windows from `SUBNET_START_BLOCK`): threshold relaxes to `σ ≥ 0.33` (binary equivalent: k ∈ [1, 7]) to keep batches filling while miner population and env coverage are thin.

### Cooldown — one-shot prompt rotation

Once a `prompt_idx` enters the winning batch it is ineligible for `BATCH_PROMPT_COOLDOWN_WINDOWS = 1_000_000` windows. With OpenMathInstruct-2's large prompt pool, this makes prompts effectively one-shot across any realistic run.

The prompt-index cooldown is restored from its complete run-keyed snapshot and may replay a bounded R2 gap. A separate full-SHA256 canonical-content snapshot closes dataset-alias bypasses. On its first deployment, the validator resolves every selected index in the complete prompt snapshot and refuses to open a window until the derived content map is durable locally.

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

### Difficulty auction v2: fixed collection and top-down deferred proof

> **Current production design (July 2026).** The auction supersedes the v2.3 same-prompt runner-up split. Full contract: [difficulty-auction-v2-design.md](superpowers/specs/2026-07-15-difficulty-auction-v2-design.md).

Per-window randomness remains drand-derived and exposed by `/state`. Submissions carry the current drand round, with stale/future rounds rejected at signed precommit arrival. Submitted drand is not a ranking key. Candidates rank by difficulty, then validator-observed precommit drand round. Exact ties inside that three-second bucket use a post-deadline drand salt bound to checkpoint, window, environment, operator, and prompt, never hotkey or miner-controlled payload metadata. A bounded seal-beacon outage falls back to exact validator-observed precommit arrival, not known window randomness.

Multiple distinct operators may enter the same prompt pool, bounded at ten groups, but only the first ranked candidate for that prompt that passes deferred proof can win. One operator may reserve only one logical claim per prompt; there is no per-operator winner cap. The active selector does not split a prompt slot among runners-up.

Prompt uniqueness is canonical-content based, not index-only. The observation-only foundation for a future validator-authoritative utility tie-break is documented in [Auction v3 Utility Foundation](auction-v3-utility-foundation.md). It does not alter the auction-v2 order or payout.

This removes the old hotkey-count dilution surface: extra hotkeys neither produce different forced draws, reserve additional operator/prompt claims, nor create additional equal-score tie tickets.

### EMA scoring — one payment per window, not per submission

Before EMA, weights were submitted as "fraction of batch slots won over the interval, counted from scratch each epoch". This lost intra-epoch data because Bittensor records only the last `set_weights` call of an epoch for emissions.

The EMA fixes this: after each window, every hotkey's score is updated as:

```
score_new = α × share_this_window + (1 − α) × score_old
```

where `share_this_window` is the final per-hotkey share from proven auction slots. Each selected group earns one uniform slot; there is no active same-prompt split. `alpha = EMA_ALPHA = 2 / (72 + 1) ~= 0.027`. With a 72-window history, this gives a roughly 25-window half-life. A miner that stops contributing loses half its score in about 25 windows. The EMA is replayed from R2 archives at startup, so loss of local disk does not lose scoring history.

At each `set_weights` call the validator submits the current EMA values directly. The sum of all EMA scores is the smoothed fill rate; `burn = max(0, 1 − sum)` goes to `UID_BURN = 0`.

### Checkpoint hash gate — miners always run the current model

Every `BatchSubmissionRequest` includes `checkpoint_hash` — the HF commit revision the miner loaded. The validator compares this to `current_checkpoint_hash` (the revision of the most recently published HF snapshot). A mismatch returns `WRONG_CHECKPOINT` immediately, before any GRAIL verification, saving both parties compute.

This guarantees that training data always reflects the currently-published policy. Without it, a stale miner could produce rollouts from an old model, creating a training distribution mismatch.

### Publish every N trained windows — HF cannot keep up with per-step pushes

The base model is Qwen3.5-2B (~2 billion parameters, sharded safetensors, thinking chat template). Pushing a new safetensors snapshot to HF Hub on every window (roughly every 60 seconds under load) is infeasible due to Git LFS latency and HF rate limits. The validator publishes to HF every `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS = 10` successful balanced optimizer steps. A partial window may contribute retained samples but does not increment the cadence by itself. Quarantined windows are archived/credited but excluded from the accumulator. Between publishes, miners stay on the last-published revision — the hash gate keeps them there. `checkpoint_n` only increments on a successful publish, so the gate remains stable across the publish gap.

---

## Economic model

### How a miner earns

1. Submit a protocol-v2, valid, in-zone group on a non-cooldown prompt during the 100-second collection interval.
2. Rank highly enough by difficulty and pass the validator's deferred proof.
3. Be the first proven candidate for that prompt. Each selected group earns one `pool / B_BATCH` environment slot.
4. Once per subnet epoch (~360 blocks), the validator calls `set_weights` on-chain with the current EMA values. All validators submit inside a shared ~20-block window before the epoch boundary so they converge on identical weights. Your emission for the epoch is proportional to your EMA score.

### Rough expected earnings

Suppose the network emits `E` TAO per epoch. You land on an average of `s` unshared winning prompts per window. The EMA converges to approximately `s / B_BATCH = s / 8` of the total filled-slot budget. Your share of emissions per epoch is approximately:

```
(s / 8) / (sum of all miners' EMA scores)
```

A miner consistently landing two winning prompts in one environment reaches its protocol cap there and earns roughly `2/8 = 25%` of that environment's filled-slot budget before cross-window EMA normalization.

### What disqualifies a submission

| Reject reason | Cause | Remediation |
|---|---|---|
| `WINDOW_NOT_ACTIVE` | Window is in `TRAINING` or `PUBLISHING` | Wait and re-poll `/state` |
| `WINDOW_MISMATCH` | `window_start` in request does not match current window | Refresh `/state` and retry |
| `WRONG_CHECKPOINT` | `checkpoint_hash` is stale | Re-poll `/state`, update revision, retry |
| `BAD_PROMPT_IDX` | `prompt_idx >= len(env)` | Use a valid index from the environment |
| `PROMPT_MISMATCH` | `tokens[:prompt_length]` does not match the canonical tokenization of `env.get_problem(prompt_idx).prompt` (CoT prefix, alternate chat template, custom system prompt, etc.) | Use the env's exact prompt string and the pinned tokenizer; do not modify the prompt before generation |
| `PROMPT_IN_COOLDOWN` | Prompt is in the active one-shot cooldown set | Pick a different `prompt_idx` |
| `PROMPT_FULL` | The prompt's bounded pending population is full | Pick a less crowded prompt |
| `SEED_MISMATCH` | Client does not advertise forced-seed protocol v2 or its sampled stream disagrees | Upgrade the miner and rebuild against the current protocol |
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
| Submit extremely fast | Fixed 100-second collection prevents early count-based seal | Arrival round breaks exact score ties; sub-round milliseconds matter only during a seal-beacon outage |
| Register many hotkeys | Hotkey-free seed, operator/prompt dedup, and operator-bound equal-score ties | No extra legal draw or tie ticket for the same operator/prompt |
| Run a stale model | `WRONG_CHECKPOINT` rejects before GRAIL | 0 earnings |

---

## Known limitations

- **Public task oracles.** OpenMath labels and OpenCode structured cases are public/reconstructable. Validator authority prevents miners from writing their own rewards, but secrecy is not the moat. Private/generated tasks, delayed/redacted archives, property tests, and commit-first sampling remain future hardening directions.
- **Single trainer.** The current deployment assumes a single trainer writing to R2. Multiple trainers in the same bucket would collide on archive keys (`reliquary/dataset/window-<N>.json.gz`). Multi-trainer consensus is future work.
- **Optimizer and scheduler state not persisted.** A validator restart resets AdamW momentum and the LR scheduler step count to zero. Training regresses for `LR_WARMUP_WINDOWS = 10` windows before stabilizing. Minimize restarts.
- **No automatic HF checkpoint garbage collection.** Every publish creates a new HF commit. Old revisions accumulate. Plan manual or cron-based cleanup.
- **No automatic R2 retention.** Every window archives ~1 MB compressed. Add a bucket lifecycle rule for archives older than your retention window.
- **HF bootstrap auth.** `_bootstrap_state_from_external` calls `HfApi().list_repo_commits` to count published checkpoints. For private repos, `HF_TOKEN` must be set at startup. Public repos are readable without authentication but the call still hits the HF API rate limit (~500 req/hour for unauthenticated). Set `HF_TOKEN` anyway to avoid rate-limit failures on restart.

---

## Further reading

- [docs/mining.md](mining.md) — operator guide for miners
- [docs/validating.md](validating.md) — operator guide for validators
