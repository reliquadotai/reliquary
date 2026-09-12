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

The zone filter (`σ ≥ 0.24`, see below) is the mechanical realization of DAPO's Dynamic Sampling, reformulated to be reward-scale-agnostic. The miner-side incentive to predict σ *before* generating is what turns DAPO's post-hoc filter into an ex-ante market.

---

## The core loop

One full training window, step by step.

**1. Miners read `/state`.**
Miners poll `GET /state` continuously. The response (`GrpoBatchState`) carries `state`, `window_n`, `checkpoint_n`, `checkpoint_repo_id`, `checkpoint_revision`, and `cooldown_prompts`. If `checkpoint_n` has advanced since the last poll, the miner downloads the new HF revision before doing anything else.

**2. Miner picks a prompt.**
The miner selects a `prompt_idx` from one active environment that is not in that environment's cooldown set. OpenMath uses OpenMathInstruct-2 (`nvidia/OpenMathInstruct-2`) with public labels and validator-recomputed local reward claims. OpenCode uses the pinned public curated prompt/case dataset, while the validator remains authoritative by executing cases in its sandbox. The reference engine uses uniform-random sampling with rejection against the cooldown set. This is a baseline: smarter miner-side selection — predicting which prompts will pass the zone filter for the current checkpoint — is expected. See [mining.md §Prompt selection strategy](mining.md#prompt-selection-strategy).

**3. Miner generates M=16 rollouts.**
The miner runs exactly `M_ROLLOUTS = 16` completions with the protocol-v6 forced sampling stream. The stream is derived from window randomness, prompt, checkpoint, rollout index, and token position; it deliberately excludes hotkey identity. The validator recomputes the same stream. Multiple hotkeys therefore cannot obtain different legal draws for one prompt, and clients advertising any protocol version or generation profile other than the active v6 contract are rejected before grading. The contract uses signed step-by-step Math and Code prompts plus the pinned Logic environment contract, raw prompt encoding, full-support sampling (`T=1`, `top_p=1`, `top_k=0`), an 8192-token cap, and no BFT. A natural EOS is valid only when it is the exact public forced inverse-CDF pick at that position; an EOS with merely plausible probability is not enough. A cap hit without EOS counts against the bounded truncation allowance.

**4. Miner builds GRAIL sketches.**
For each rollout the miner runs a bit-identical HuggingFace forward pass on the proof GPU to construct a GRAIL sketch commitment. The sketch binds the completion to the model's hidden-state activations. The miner signs the commit and packages everything into a `BatchSubmissionRequest` that includes `checkpoint_hash` (the HF revision from the last `/state` response). In OpenMath, `rollout.reward` must match the miner's local `env.compute_reward` value; the validator recomputes it and rejects mismatches. In OpenCode, local reward fields are placeholders and the validator computes sandboxed structured-case rewards before applying the zone filter.

**5. Miner submits.**
The miner serializes and hashes its final signed body, obtains a signed upload receipt from `POST /submit/precommit`, then sends the exact bytes to `POST /submit`. A predeadline receipt grants bounded upload grace without extending generation. In production, the body response is provisional (`accepted=True, reason="submitted"`) once queued. A background worker runs bounded admission and reward grading. Its later `ACCEPTED` verdict means the group entered the proof queue; it is not yet a GRAIL pass or a paid slot.

**6. Validator admits, proves FIFO, and fills batches.**
Each active environment admits independently. Cheap admission checks the window, checkpoint, protocol, registration/operator mapping, prompt, payload bounds, signatures, randomness, dedup, validator-authoritative rewards, termination shape, and zone filter (`sigma >= 0.24`) before GPU proof; proof then enforces GRAIL and the exact forced terminal pick. Eligible groups enter one monotone FIFO proof plan: observed upload rate, payload bytes, token count, difficulty, and submitted drand cannot improve order or payment. The scheduler asks for exactly `FILL_CLOSED_TARGET_GROUPS_PER_ENV = FILL_CLOSED_PICKS_PER_WINDOW × B_BATCH` successful groups per environment and stops speculative dispatch once that demand is met. Failed proofs consume the separate bounded admission budget. Every `B_BATCH=16` proven groups form one pick; the configured pick target closes the window. There is no seal-time auction or forensic proof tail.

**7. Validator emits clean training batches.**
Each completed pick emits one immutable training payload containing up to 16 selected groups per active environment. The detached trainer consumes those payloads in journal order and runs GRPO without mixing checkpoint revisions. Before training, quarantine checks may archive and credit a suspicious payload while excluding it from optimizer and publish work. If the 1800-second backstop closes a short window, only complete or explicitly sealed partial picks are paid; missing fixed slots burn rather than being redistributed.

**8. Validator publishes a new checkpoint.**
Every `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS = 16` successful V6 payloads the model is saved locally, pushed to HF Hub, and signed: `ed25519(checkpoint_n || revision)`. If the PPO ratio gate detects behavior-policy drift sooner, the rejected step is excluded and the previously accepted in-memory updates are published immediately. The signed manifest is installed in `/checkpoint`. Between publishes the miners stay on the last-published revision (enforced by the checkpoint hash gate). The window dataset is archived to R2, including quarantine metadata when present.

**9. State → READY → OPEN.**
The next window opens immediately. Winning prompts enter one-shot cooldown. Once per subnet epoch the validator calls `set_weights` on-chain with the current EMA snapshot.

**Safety net.** A v6 window normally closes after its configured pick target and has an unconditional 1800-second backstop. The default shape is 16 picks, 256 required passes, 512 productive admissions, and 1024 grading starts per environment; production overrides are valid only with matching capacity qualification. Incomplete fixed slots are unpaid and burn. The older 60–100-second adaptive auction constants apply only to legacy profiles.

---

## Why each mechanism exists

### GRAIL proofs — anti-fabrication

A GRAIL sketch is a compact linear commitment over a sampled subset of the model's last hidden-state activations for a given completion. The validator recomputes the forward pass on the same tokens with the same model, draws the same random challenge positions (seeded from the window's randomness), and checks that the two sketches agree within a position-dependent tolerance (`PROOF_SKETCH_TOLERANCE_BASE = 5000`, growth `= 5.0 * sqrt(position)`). The tolerance absorbs calibrated numerical drift, but miners must still match the pinned inference stack; arbitrary kernel or hardware changes are not guaranteed compatible. Fabricated activations diverge far beyond honest numerical noise.

Because each rollout's sketch is bound to the specific token sequence and the model's weights, a miner cannot fabricate completions, copy another miner's rollouts, or replay proofs from a different model revision without failing the sketch check.

### Zone filter — only train on useful frontier prompts

`σ` is the population standard deviation of the 16 rollout rewards in a group. A group with `σ < 0.24` carries too little usable signal under the active reward-scale-agnostic gate; all-equal groups have zero GRPO advantage. Dropping them saves compute without losing learning.

Binary equivalence note: OpenMath rewards are binary `{0, 1}` (the validator extracts the final `\boxed{...}`/`\fbox{...}` answer and compares after conservative normalization). With binary rewards, `σ = sqrt(p(1−p))` where `p = k/16`. The extreme non-degenerate groups have `σ(k=1 or 15) ≈ 0.242`, so `σ ≥ 0.24` admits k=1..15 while still rejecting k=0 and k=16.

Bootstrap phase (`BOOTSTRAP_WINDOWS = 100` windows from `SUBNET_START_BLOCK`): threshold relaxes to `σ ≥ 0.22` to keep continuous-reward groups filling while miner population and env coverage are thin. For binary Math rewards, both v6 thresholds admit k=1..15.

The v6 Math prompt explicitly asks for step-by-step reasoning and requires a boxed final answer; only that final channel can earn positive reward. Plain trailing numbers and `Answer:` lines score zero. A missing box is not treated as a trustworthy negative for admission: the validator evaluates the group under both attainable binary outcomes and requires every interpretation to remain in-zone. Deleting a naturally generated box therefore cannot manufacture eligibility. Empty, special-token, or unclosed final boxes are rejected separately as malformed.

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

### Fill-closed FIFO under protocol v6

Protocol v6 replaces the timed auction with a fill-closed stream. Per-window randomness remains drand-derived and exposed by `/state`; the submitted drand round is checked for freshness at precommit arrival but is never a ranking key. After admission, proof dispatch and selection use monotone FIFO ingress. Rate and payload measurements remain observability fields only.

Multiple distinct operators may enter the same prompt pool, bounded at ten groups, while one operator may reserve only one logical claim per prompt. There is no per-operator winner cap and no runner-up split. Prompt uniqueness is canonical-content based, not index-only.

Every selected group occupies one fixed slot and receives the same slot share regardless of completion length, byte size, arrival rate, reward vector, or difficulty. A missing group leaves that share unpaid; it is never redistributed among winners. Extra hotkeys do not create different forced draws or additional operator/prompt claims, although FIFO still gives earlier valid arrivals a residual advantage until the window fills.

### EMA scoring — one payment per window, not per submission

Before EMA, weights were submitted as "fraction of batch slots won over the interval, counted from scratch each epoch". This lost intra-epoch data because Bittensor records only the last `set_weights` call of an epoch for emissions.

The EMA fixes this: after each window, every hotkey's score is updated as:

```
score_new = α × share_this_window + (1 − α) × score_old
```

where `share_this_window` is the final per-hotkey share from fixed FIFO slots. Each selected group earns one uniform slot; there is no active same-prompt split. `alpha = EMA_ALPHA = 2 / (72 + 1) ~= 0.027`. With a 72-window history, this gives a roughly 25-window half-life. A miner that stops contributing loses half its score in about 25 windows. The EMA is replayed from R2 archives at startup, so loss of local disk does not lose scoring history.

At each `set_weights` call the validator submits the current EMA values directly. The sum of all EMA scores is the smoothed fill rate; `burn = max(0, 1 − sum)` goes to the current subnet owner's UID, unless `RELIQUARY_UID_BURN` explicitly overrides it.

### Checkpoint hash gate — miners always run the current model

Every `BatchSubmissionRequest` includes `checkpoint_hash` — the HF commit revision the miner loaded. The validator compares this to `current_checkpoint_hash` (the revision of the most recently published HF snapshot). A mismatch returns `WRONG_CHECKPOINT` immediately, before any GRAIL verification, saving both parties compute.

This guarantees that training data always reflects the currently-published policy. Without it, a stale miner could produce rollouts from an old model, creating a training distribution mismatch.

### Publish every 16 successful payloads — HF cannot keep up with per-step pushes

The base model is Qwen3-4B-Base, used with raw prompt text rather than a chat template. Pushing a new safetensors snapshot to HF Hub after every payload is infeasible due to Git LFS latency and HF rate limits. The trainer therefore publishes after 16 successful optimizer steps by default; the legacy constant name remains `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS`. Quarantined or ratio-rejected payloads do not increment the successful-step cadence. If `policy_ratio_drift` rejects a later step, that rejected step never reaches the optimizer and the trainer publishes the preceding safe updates, refreshes the serving behavior policy, and retries a failed upload without training again. Between publishes, miners stay on the last-published revision — the hash gate keeps them there. `checkpoint_n` only increments on a successful publish, so the gate remains stable across the publish gap.

---

## Economic model

### How a miner earns

1. Submit a protocol-v6, valid, in-zone group on a non-cooldown prompt before the fill/backstop closes.
2. Pass the validator's continuous proof early enough to enter a FIFO pick.
3. Each selected group earns one fixed `window_pool / (environment_count × picks_target × B_BATCH)` slot.
4. Once per subnet epoch (~360 blocks), the validator calls `set_weights` on-chain with the current EMA values. All validators submit inside a shared ~20-block window before the epoch boundary so they converge on identical weights. Your emission for the epoch is proportional to your EMA score.

### Rough expected earnings

Suppose the network emits `E` TAO per epoch. You land on an average of `s` selected groups per window, across `n_env` environments and `picks_target` picks. The EMA converges to this share of the total window budget:

```
s / (n_env * picks_target * B_BATCH)
```

The final emission share is that fixed-slot value divided by the sum of all miners' EMA scores. Completion length never changes it.

### What disqualifies a submission

| Reject reason | Cause | Remediation |
|---|---|---|
| `WINDOW_NOT_ACTIVE` | Window is in `TRAINING` or `PUBLISHING` | Wait and re-poll `/state` |
| `WINDOW_MISMATCH` | `window_start` in request does not match current window | Refresh `/state` and retry |
| `WRONG_CHECKPOINT` | `checkpoint_hash` is stale | Re-poll `/state`, update revision, retry |
| `BAD_PROMPT_IDX` | `prompt_idx >= len(env)` | Use a valid index from the environment |
| `PROMPT_MISMATCH` | `tokens[:prompt_length]` does not match the canonical raw tokenization of `env.get_problem(prompt_idx).prompt` (chat template, altered reasoning cue, custom system prompt, etc.) | Render the v6 contract's exact environment template and use the pinned tokenizer; do not apply a chat template |
| `PROMPT_IN_COOLDOWN` | Prompt is in the active one-shot cooldown set | Pick a different `prompt_idx` |
| `PROMPT_FULL` | The prompt's bounded pending population is full | Pick a less crowded prompt |
| `SEED_MISMATCH` / `PROTOCOL_MISMATCH` | Client does not advertise protocol v6 or its forced sampled stream disagrees | Upgrade the miner and rebuild against the current generation contract |
| `HASH_DUPLICATE` | Rollout tokens duplicate a recently accepted rollout hash | Generate fresh tokens; do not replay |
| `REWARD_MISMATCH` | Validator reward computation failed or produced a non-finite value | Treat as malformed output/env failure; miner rewards are not trusted |
| `OUT_OF_ZONE` | `σ < 0.24` (or `σ < 0.22` during bootstrap), including the conservative interpretation of unboxed Math outcomes | Pick a different prompt and always produce a valid boxed final Math answer |
| `MALFORMED_FINAL_ANSWER` | Final zero-reward answer box is empty, unclosed, or contains a special token | Preserve genuine generation and emit a well-formed final answer box |
| `WRONG_ROLLOUT_COUNT` | Submission does not have exactly `M_ROLLOUTS = 16` rollouts | Always submit exactly 16 |
| `BAD_SIGNATURE` | GRAIL commit signature verification failed | Check wallet hotkey and signing code |
| `GRAIL_FAIL` | Sketch does not match validator's forward pass | Check checkpoint, `attn_implementation`, and CUDA version |
| `BAD_TERMINATION` | EOS is not the exact forced inverse-CDF pick, EOS appears before the final token, or the cap-truncation allowance is exceeded | Preserve the forced stream exactly and stop at its first configured EOS |

---

## Anti-cheat properties

| Attack | Mitigation | Realistic outcome |
|---|---|---|
| Fabricate completions | GRAIL sketch fails | 0 earnings |
| Resubmit old completions | `WRONG_CHECKPOINT` (rotation invalidates stale rollouts) | 0 earnings |
| Cherry-pick only easy prompts | σ ≈ 0 → `OUT_OF_ZONE` | 0 earnings |
| Delete or omit answer boxes to manufacture zero rewards | Boxless output scores zero, then conservative uncertain-outcome utility removes any manufactured eligibility/value | No economic advantage; the group may be out of zone |
| Spam the same prompt every window | One-shot cooldown blocks re-entry after the prompt wins | 0 earnings after first winning inclusion |
| Generate extra rollouts to select favorable reward vectors | Monitoring and training quarantine reduce blast radius; long-term private tasks / commit-first sampling are the durable fix | Some shaping value remains until durable mitigations land |
| Inflate rate, payload bytes, or completion length | Those fields are telemetry only; FIFO order and fixed slot payment ignore them | No direct scoring or payment gain |
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
