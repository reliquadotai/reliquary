# RL Evaluation & Metrics — Design Spec

**Target run:** `qwen3-4b-base-dapo-v4` (`Qwen/Qwen3-4B-Base` @ `906bfd4b4dc7f14ee4320094d8b41684abff8539`, DAPO recipe)
**Branch:** `feat/qwen3-base-dapo-v4-profile` (currently `f0eee16`, not pushed)
**Date:** 2026-08-12
**Status:** design approved, implementation not started

---

## 0. What this document is

A specification for two things, plus the dashboard that displays them:

- **Part A** — an **offline checkpoint evaluation harness**: given a published checkpoint, produce a complete, reproducible measurement of model quality (pass@k, per-environment scores, in-zone rate, termination health) — and, from those, a **termination criterion** answering "is this run done?" (§3.8).
- **Part B** — a **catalogue of in-training metrics**: what the validator already emits to wandb, and what is missing.
- **Part C** — the **charts** that turn A and B into a readable story of the run.

Part A is the priority. It is what produces the headline claim we want to be able to make: *pass@1 climbing toward pass@16 as training progresses.*

Nothing here changes production behaviour. The harness is read-only with respect to the validator; Part B adds telemetry keys only.

---

## 1. What we are trying to prove, and the traps

The v3 run failed to show progress, and we know precisely why. Any evaluation design that cannot detect these failures is not worth building.

**The headroom finding (2026-08-03, re-confirmed at ckpt138).** On Qwen3.5-4B, post-training had already consumed the dispersion RL exists to convert: pass@8 was **identical (0.611)** for the upstream model, ckpt88, and ckpt138. 138 checkpoints of RL moved nothing. The whole thesis of v4 is that starting from a *true base* model restores a pass@1↔pass@16 gap that RL can actually close. **So the gap is not a side metric — it is the primary outcome variable.**

**The termination collapse (ckpt84, 2026-08-02).** On hard prompts, EOS rate fell 84% → 50% and at-cap rate rose 16% → 50%. The model learned to ruminate. This was invisible in `rewards/mean`.

**The 8k evaluation mistake (2026-08-05).** The ckpt138 evaluation capped generation at 8k while the overlong ramp targets the 16k cap. Every rollout that would have terminated between 8k and 16k was counted as non-terminating. The conclusion "no termination recovery" was therefore not established. **Evaluation length budget must equal the production cap.**

**The grader drift (three fixes: surface-form, `Answer:` lines, inline LaTeX).** Each fix silently invalidated every previously measured number. False negatives were 28% at one point, 0.9% after the fix. Any harness that embeds its own grader will produce numbers that cannot be compared to training rewards or to each other across time.

**Selection bias in live data.** Rollouts arriving at the validator are on prompts *chosen by miners*, who hunt the zone (σ ≥ `SIGMA_MIN`). The prompt distribution is adversarially concentrated on k in the middle of the range. **A pass@1 curve computed from live submissions is meaningless.** This is the single reason Part A must exist separately from Part B.

---

## 2. The core abstraction: the k-histogram

Everything in Part A reduces to one measurement.

> For each prompt in a frozen suite, sample **G = 16** completions, grade each with the **production** grader, and record **k** = the number of correct completions (0 ≤ k ≤ 16).

The per-suite **k-histogram** `h[0..16]`, where `h[j]` is the fraction of prompts with exactly `j` correct rollouts, is the single source of truth. Every metric below is a functional of `h` (or, where rewards are not binary, of the raw reward vectors that produce it).

### 2.1 Identities

With `N` prompts and `G = 16`:

```
pass@1        = E[k]/G                       = Σ_j h[j] · j/G
pass@G        = P(k ≥ 1)                     = 1 − h[0]
all_correct   = P(k = G)                     = h[G]
solve_rate_gap = pass@G − pass@1             ("headroom")
nondegenerate = P(1 ≤ k ≤ G−1)               (DAPO's dynamic-sampling filter)
in_zone       = P(k ∈ K_zone)                (production σ-gate — see 2.3)
```

**Unbiased pass@k** (Chen et al. 2021), valid for any `k ≤ G`:

```
pass@k = E_prompts[ 1 − C(G − k_correct, k) / C(G, k) ]
```

with the convention `C(a, b) = 0` when `a < b`. Note `pass@1` from this estimator equals `E[k]/G` exactly — a useful internal consistency check.

Implementation note: compute the ratio as a product of `(1 − k/(G−i))` for `i` in `0..k_correct−1` to avoid overflow; do not build large binomials.

### 2.2 Answering the original question: can in-zone be derived from pass@1 and pass@8?

**No — not from those two numbers alone.** `pass@1` and `pass@G` are two moments of the distribution of per-prompt solve probability `p`. The in-zone rate depends on the *whole shape* of that distribution, not on two of its moments. Two suites can share a pass@1 and a pass@G and have very different in-zone rates.

**But two weaker statements are true and useful:**

1. **The gap and the in-zone rate vanish together.** `pass@G − pass@1 = 0` ⟺ `p ∈ {0,1}` almost surely ⟺ `in_zone = 0`. So the gap is a valid *collapse detector*: if it goes to zero, there is no learnable signal left in the suite, and the σ-gate will find nothing. This is exactly the v3 pathology.
2. **The in-zone rate is exactly derivable from the k-histogram**, which costs no more to collect than pass@k does — it is the same generation run. So there is no reason to approximate it.

**Design consequence:** collect `h`, derive everything. Report the gap as a headline number *and* the in-zone rate as a separate, exact one. Do not present the gap as a proxy for in-zone in the dashboard.

### 2.3 The σ-gate at G=16 — a confound that must be reported

Production defines in-zone as `reward_std ≥ SIGMA_MIN`, where `reward_std` is the **population** standard deviation of the group's rewards (`difficulty_auction.py:90-92`, division by `count`), computed on the **raw graded reward** — *not* the shaped one. `_training_rewards` states this explicitly: the overlong penalty and the BFT zeroing live only in advantage space; `rollout.reward`, which drives the σ-gate, the auction and emission, is never modified.

For **binary** rewards, `σ = √(p̂(1−p̂))` with `p̂ = k/G`, so the gate is exactly a k-range:

| G | `SIGMA_MIN = 0.43` | `BOOTSTRAP_SIGMA_MIN = 0.33` |
|---|---|---|
| 8 | k ∈ {2,…,6} (5 of 7 non-degenerate classes) | k ∈ {1,…,7} (all 7) |
| **16** | **k ∈ {4,…,12}** (9 of 15) | k ∈ {2,…,14} (13 of 15) |

σ per k at G=16: `k=1 → 0.2421`, `k=2 → 0.3307`, `k=3 → 0.3903`, `k=4 → 0.4330`, `k=8 → 0.5000` (symmetric).

**Two consequences the dashboard must encode:**

- **The v3→v4 cutover mechanically lowers the in-zone rate.** At G=8 the gate rejected 2 of 7 non-degenerate classes; at G=16 it rejects 6 of 15 — and specifically k ∈ {1,2,3,13,14,15}, which are exactly the prompts DAPO's dynamic sampling is designed to keep. An in-zone drop across the cutover is **not** evidence of learning. Any in-zone chart spanning the cutover needs a vertical marker and a G annotation.
- **Report `nondegenerate` (0 < k < G) alongside `in_zone`.** They answer different questions: `nondegenerate` = "does this prompt produce gradient", `in_zone` = "does this prompt earn a miner a slot". Their divergence at G=16 is itself a finding worth watching (`SIGMA_MIN` was calibrated for G=8; the comment at `constants.py:532-535` still describes the M=8 regime).

### 2.4 Non-binary rewards: the code environment

`openmathinstruct.compute_reward` returns exactly `1.0` or `0.0`. **`opencodeinstruct.compute_reward` does not** — it returns `passed/total ∈ [0,1]` from `evaluate_cases`.

Therefore, for the code environment:

- **in-zone must be computed from the real reward vector**, using the same population-std formula as production. The k↔σ table does not apply.
- **pass@k needs an explicit binarization rule.** Use `correct ⟺ reward == 1.0` (all cases pass). State it in every chart legend. Also report the mean fractional reward separately — it is the more sensitive progress signal early in training, when nothing passes all cases yet.
- The k-histogram for code is therefore a histogram of *fully-correct* counts, and it is strictly less informative than the math one. Complement it with the distribution of mean fractional reward.

### 2.5 Grader infrastructure failures are not zeros

`GraderInfrastructureError` (sandbox down, queue full) must **never** be graded as 0.0 — that would silently depress code scores. The harness records such rollouts with `reward = null` and excludes them from all denominators, reporting `grader_error_ratio` per suite. A suite with `grader_error_ratio > 1%` is marked invalid and must not be plotted. (Precedent: the runsc cgroup leak that produced a run of spurious 0.0 code rewards.)

---

## 3. Part A — Offline checkpoint evaluation harness

### 3.1 Architecture: two stages, one artifact between them

```
  ┌──────────────────────────┐        ┌───────────────────────┐        ┌──────────────────┐
  │  Stage 1: GENERATE       │        │  completions.jsonl    │        │ Stage 2: GRADE   │
  │  GPU pod (vLLM)          │ ─────► │  raw text, no grades  │ ─────► │ + METRICS        │
  │  checkpoint = repo@rev   │        │  + run manifest       │        │ CPU, sandbox     │
  └──────────────────────────┘        └───────────────────────┘        └──────────────────┘
                                                                                │
                                                                                ▼
                                                                       metrics.json + report
```

**Why the split** — three concrete payoffs:

1. **Re-grading without re-generating.** The grader has changed three times. Each change can be replayed over every historical artifact for a few CPU-minutes, restoring comparability across the whole run instead of invalidating it.
2. **The code sandbox stays off the GPU pod.** Grading `opencodeinstruct` needs the gVisor grader service. Installing it on a rented GPU pod is avoidable work; stage 2 runs anywhere.
3. **Stage 2 is unit-testable without a GPU.** All the metric logic — the part most likely to be subtly wrong — is tested from fixtures on any machine.

**Where it runs.** Stage 1 runs on a **separate on-demand GPU pod**, never on the validator's H100. That box sits at ~96.6% memory occupancy with a measured allocator cliff at ~88%; adding a vLLM instance re-opens the root cause of the `FatalProofPlaneError` restarts. Stage 1's only input is `repo@revision`, so it is fully decoupled and can be replayed for any past checkpoint.

### 3.2 Stage 1 — generation contract

Input:

| Field | Value |
|---|---|
| `--checkpoint` | `hf-repo@revision`, or a local path |
| `--suite` | path to a frozen suite file (§3.4) |
| `--out` | output JSONL path |
| `--samples` | `G`, default **16** |

Sampling **must** be read from the protocol profile, not hardcoded — this is pinned by a test (§6.5). For `qwen3-4b-base-dapo-v4`:

| Parameter | Value | Source |
|---|---|---|
| `temperature` | 1.0 | `_SAMPLING_DAPO` |
| `top_p` | 1.0 | `_SAMPLING_DAPO` |
| `top_k` | 0 (disabled) | `_SAMPLING_DAPO` — `0`, never `None` |
| `max_new_tokens` | **16384** | `max_new_tokens_for_environment(env)` |
| BFT | **none** | v4 `EnvironmentProfile.bft = None` |
| prompt format | **raw completion** | `RAW_COMPLETION_PROMPTS = PROTOCOL_VERSION >= 4`; no chat template, no `<think>` |

At T=1.0 with top-k/top-p disabled, `warp()` is the identity — the eval distribution equals the training distribution equals the sampling distribution. This is the property that makes the v4 numbers meaningful at all, and the reason not to "improve" eval sampling with a lower temperature.

**Not in v1: a greedy track.** A single `temperature=0` completion per prompt would cost 1/16th of a run and carry no sampling variance, but it measures a different quantity from pass@1 at T=1.0 and would need its own axis in every chart. Revisit once the main curves are reading cleanly (§8).

**Determinism.** The seed for sample `i` of prompt `q` is `seed = H(suite_id, q.prompt_content_sha256, i)`, a pure function. Re-running stage 1 on the same checkpoint reproduces the artifact up to kernel non-determinism. Record the resolved seed on every row.

### 3.3 The artifact — one JSON object per rollout

```jsonc
{
  "schema_version": 1,
  "run_id": "eval-2026-08-12T14:03:11Z-a1b2c3",
  "checkpoint": {"repo": "org/repo", "revision": "…", "training_window": 28412},
  "suite_id": "protocol-math-v1",
  "suite_sha256": "…",              // content hash of the frozen prompt list
  "env": "openmathinstruct",
  "prompt_id": "…",
  "prompt_content_sha256": "…",     // same field production uses for cooldown
  "sample_index": 3,
  "seed": 918273645,
  "prompt_text": "…",               // exactly what was fed to the model
  "completion_text": "…",           // raw, ungraded, untruncated
  "completion_token_count": 4127,
  "finish_reason": "eos" | "length" | "stop",
  "generation": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 16384},
  "profile_id": "qwen3-4b-base-dapo-v4"
}
```

Plus a sibling `manifest.json` holding the checkpoint identity, suite hashes, sampling profile, model revision, harness git SHA, and wall-clock timings. **A metrics file that cannot name the harness SHA and the grader SHA that produced it is not admissible in the dashboard.**

`completion_text` is stored uncompressed-but-gzipped-at-rest; at ~12k rollouts × up to 16k tokens the artifact is a few hundred MB per checkpoint. Keep them: the ability to re-grade is worth the storage.

### 3.4 Stage 2 — grading and metrics

**Non-negotiable: stage 2 imports the production graders.** `reliquary.environment.openmathinstruct` and `reliquary.environment.opencodeinstruct`, the real classes, via the real `compute_reward`. Not a reimplementation, not the string-equality normalizer from the `.r2_analysis` scripts — that divergence is exactly what made the old numbers non-commensurable with training rewards. A test asserts this (§6.4).

Output `metrics.json`, keyed by `(suite_id, env)`:

```jsonc
{
  "n_prompts": 256, "G": 16, "grader_error_ratio": 0.000,
  "k_histogram": [0.449, 0.041, ..., 0.094],   // 17 entries, sums to 1
                                                // h[0] = 1 − pass@16 ; h[16] = all_correct
  "pass_at": {"1": 0.212, "2": 0.288, "4": 0.371, "8": 0.463, "16": 0.551},
  "all_correct": 0.094,
  "headroom_gap": 0.339,                        // pass@16 − pass@1
  "in_zone": 0.221,                             // σ ≥ SIGMA_MIN, prod definition
  "nondegenerate": 0.457,                       // 0 < k < G  = 1 − h[0] − h[16]
  "mean_reward": 0.212,                         // == pass@1 for binary envs
  "termination": {
    "eos_ratio": 0.61, "at_cap_ratio": 0.28,
    "overlong_zone_ratio": 0.33,                // ran into the last 4096 tokens
    "len_mean": 1840, "len_p50": 612, "len_p90": 8104, "len_max": 16384
  },
  "format": {
    "parseable_ratio": 0.97,
    "extraction_path": {"boxed": 0.30, "answer_line": 0.65, "tail_number": 0.02, "none": 0.03}
  },
  "degeneracy": {"repetition_ratio": 0.07, "distinct_4gram_mean": 0.71},
  "contamination": {"suite_prompts_seen_in_training": 12, "ratio": 0.047}
}
```

Notes on the less obvious entries:

- **`overlong_zone_ratio`** — fraction of rollouts landing in `[cap − OVERLONG_PENALTY_CACHE_TOKENS, cap]` = `[12288, 16384]`, i.e. the rollouts actually paying the Eq. 13 ramp. This is the metric that answers "is the soft overlong punishment doing anything", which the 8k evaluation could not.
- **`extraction_path`** — which branch of `_compute_omi_reward` produced the answer. The v4 DAPO prompt makes ~2/3 of outputs land on an `Answer:` line, a path that did not exist in the grader until `c54d716`. If `none` starts rising, the model is drifting out of a gradable format, and reward will fall for reasons unrelated to reasoning.
- **`repetition_ratio`** — fraction of completions containing a token n-gram block repeated beyond a threshold. This is the looping detector; 77% of non-finishers were measured to loop. Compute on token ids, not text.
- **`contamination`** — see §3.5.

### 3.5 The suites

Two suites per environment. Both are **frozen files** committed to the repo (prompt ids + content hashes + ground truth reference), so every checkpoint is measured on identical inputs and cross-checkpoint comparison is **paired**.

**Protocol suite** — `protocol-math-v1` (256 prompts), `protocol-code-v1` (128 prompts).
Drawn from the real corpus exactly the way the manifest draws it (OMI canonical `train-*` shards; curated opencode set), with a fixed seed. This is the only suite on which `in_zone` means anything: it is the distribution miners actually draw from, so its in-zone rate predicts subnet supply.

> **Contamination is not preventable, so it is measured.** The manifest layer is footer-only by design; row-level exclusion needs a precomputed index artifact and is deferred. Miners can and will submit prompts that are in the protocol suite. Rather than pretend otherwise, stage 2 counts how many suite prompts appear in accepted submissions (join on `prompt_content_sha256`) and reports it. If contamination stays small the protocol-suite numbers stand; if it grows, the capacity suite carries the claim.

**Capacity suite** — `capacity-math-v1` (256 prompts), `capacity-code-v1` (128 prompts).
`capacity-math-v1` is a fixed 256-prompt subsample of **MATH-500** — genuinely outside the training corpus, and a number a third party can read. `capacity-code-v1` is a held-out slice of the curated opencode set, excluded from the training manifest build. A true public code benchmark (HumanEval+/MBPP+) would need its own adapter for the structured-cases grader; **deferred**, and named as such rather than silently skipped.

**Sizing rationale.** With `N = 256`, `G = 16`, absolute standard error on pass@1 is ≈ 2.9 pts and on in-zone ≈ 2.0 pts. That sounds loose, but the suite is frozen, so checkpoint-to-checkpoint comparison is **paired**: the between-prompt variance cancels and the standard error on Δpass@1 is ≈ 0.9 pt. A 1.5-point move between two checkpoints is detectable; a 3-point absolute claim is not. **Report deltas against the run's own step-0 baseline, with the absolute number as context.**

### 3.6 Baselines and cadence

Always evaluated, plotted as horizontal reference lines:

- **step 0** — `Qwen/Qwen3-4B-Base` @ the pinned revision, evaluated with the identical harness. This is the number every later checkpoint is a delta against.
- **v3 terminal checkpoint** — for context only, dashed, and explicitly annotated as a different profile (different G, different sampling, different prompt format). Not comparable point-to-point.

**Cadence.** v4 publishes a checkpoint every 16 windows. Evaluating every publish is affordable early (see §3.7) and wasteful late. Recommendation: evaluate **every 4th publish** (= every 64 windows) on both suites, plus every publish on `protocol-math-v1` alone if the budget allows. Milestone checkpoints get the full four-suite run.

### 3.7 Cost model

Per checkpoint, all four suites: `(256 + 128) × 2 suites × 16 samples = 12,288 completions`.

Cost is driven by mean completion length, which is expected to *grow* during the run:

| Regime | Mean length | Tokens | vLLM on 1×H100 @ ~6k tok/s |
|---|---|---|---|
| Early (base model, raw DAPO prompt) | ~500 | 6.1 M | **~17 min** |
| Mid | ~2,000 | 25 M | ~1.1 h |
| Late (DAPO Fig. 7a regime) | ~4,000 | 49 M | ~2.3 h |

The ~500-token figure is the measured v4 baseline (real OMI prompts all terminated at ≤1392 tokens during threshold calibration), against v3's 16,220. Code grading adds `128 × 16 × 2 = 4,096` sandbox calls, roughly 15–25 min at parallelism 8, on CPU.

**Watch mean completion length as the cost driver** — it is also the metric that tells you whether the overlong punishment is working, so it is instrumented anyway.

### 3.8 Termination criterion — when is the run done?

`H(t) = pass@G(t) − pass@1(t)` is the fuel gauge: it is exactly "how many points of pass@1 remain available without acquiring new capability". So "the gap stops shrinking" is the right instinct. It is also, alone, **ambiguous** — `H → 0` has three causes with opposite meanings:

| `H → 0` because | pass@1 | pass@G | Meaning |
|---|---|---|---|
| **Mastery** | rises to meet pass@G | flat or rising | Done: headroom converted |
| **Collapse** | flat | **falls** to meet pass@1 | Failed: exploration destroyed |
| **Death** | ≈ 0 | ≈ 0 | Never started, or diverged |

All three produce the same `H` curve. The criterion is therefore a conjunction, not a scalar.

#### Headline scalar: headroom conversion

```
C(t) = 1 − H(t)/H(0)          H(t) = pass@G(t) − pass@1(t)
```

`C(0) = 0`. `C = 1` means pass@1 has climbed to the base model's *initial* pass@G — the entire measured headroom has been converted into reliable single-shot performance. `C > 1` means RL found capability the base model did not have (the best possible outcome, and the thing v3 never did). This is the run's progress bar, and it is the direct executable form of the headroom thesis. Always plot it with the guard below; on its own it rises during a collapse too.

#### D — the three "done" conditions

Evaluated over a window of `W ≥ 3` consecutive evaluations, on the frozen suite so every comparison is paired:

- **D1 — outcome stalled.** A paired bootstrap over the suite's prompts gives a CI on `Δpass@1(t, t−W)` that contains 0, with half-width below the MDE (≈ 1.8 pts at `N = 256`). Same for `ΔH`. *The half-width condition matters as much as the containment: a wide CI around zero is ignorance, not convergence.*
- **D2 — not a collapse.** `pass@G(t) ≥ pass@G(t−W) − 2·SE_paired`. **If D1 holds and D2 fails, the run has collapsed, not converged** — roll back rather than stop.
- **D3 — dynamics frozen.** `prompt_churn(t) = E_prompts[ |k_t − k_{t−W}| ] / G` below threshold. This is the condition that distinguishes *converged* from *wandering*: an aggregate that sits still while individual prompts trade places means the policy is still moving and D1 is a coincidence. Free to compute — per-prompt `k` is already in the artifact, and the suite is frozen, so prompts are directly comparable across checkpoints.

**Precondition on all three:** no circuit-breaker storm in the window. A flat curve while `train/step_skipped_*` is firing is a stalled optimizer, not a converged policy.

#### M — the mechanism, which decides what to do next

D tells you it stopped. M tells you why, and the actions are completely different:

- **M1 — fuel exhausted.** `nondegenerate` rate on the *protocol* suite falls below ~5%: almost every prompt in the corpus is now all-right or all-wrong, so advantages are zero and no number of further steps can help. → change the corpus, add harder prompts, or implement dynamic sampling.
  > **This one is existential, not academic.** In-zone prompts are what miners get paid for. Fuel exhaustion is simultaneously "the model finished learning" and "the subnet stops paying". The same number governs both, which is why it belongs on the main dashboard and not in an appendix.
- **M2 — exploration floor.** Entropy flat and low — in particular entropy *conditioned on unsolved (`k = 0`) prompts* — while the non-degenerate rate is still healthy. The model has fuel and cannot use it. → more steps will not help; restart with different regularization. With `KL_BETA = 0` in v4 there is no anchor at all, so this is the failure mode to watch.
- **M3 — neither.** Fuel present, entropy healthy, nothing moves. The bottleneck is upstream: learning rate, π_old fidelity, grader, or admission. → do not declare convergence; debug.

#### `dead_prompt_ratio`

Fraction of prompts with `k = 0` **and** near-zero completion diversity across the G samples (low mean distinct-4-gram across samples, or a shared long prefix). These are permanently out of reach: the model does not merely fail them, it fails them the same way every time. Then

```
frontier(t) = 1 − pass@G(t) − dead_prompt_ratio(t)
```

is the set of prompts that are unsolved but still being explored — the honest remaining opportunity. `frontier → 0` is the sharpest single statement that the run is over.

#### Reporting

Emit a status block, never a single boolean:

```jsonc
"termination": {
  "conversion": 0.62,                 // C(t)
  "D1_outcome_stalled": false, "D2_no_collapse": true, "D3_churn_frozen": false,
  "prompt_churn": 0.081, "frontier": 0.221, "dead_prompt_ratio": 0.228,
  "mechanism": null,                  // "fuel_exhausted" | "exploration_floor" | "upstream" | null
  "breaker_storm_in_window": false,
  "verdict": "training"               // "training" | "converged" | "collapsed" | "stalled_upstream"
}
```

**Thresholds must be calibrated from this run's own first ~10 evaluations, not inherited from v3** — v3 ran at G=8 with a different profile, and its `H` never moved at all, so it offers no scale for what a normal `ΔH` looks like. Until calibrated, report the raw quantities and leave `verdict` null rather than emitting a criterion nobody can defend.

**"Done" is corpus-relative.** Converged on the protocol suite means converged on *this* corpus. The capacity suite can still be moving, and usually will be — the two are allowed to disagree, and the disagreement is informative.

---

## 4. Part B — In-training telemetry

Emitted from `training.py` to wandb, indexed by `window_index` as the wandb step. This is a *health and supply* view; because of selection bias (§1) it is **not** a capability view, and its `pass@1`-like quantities must never be plotted on the same axis as Part A's.

### 4.1 What already exists

Substantial. Do not rebuild any of this.

| Family | Keys |
|---|---|
| Optimizer | `train/lr`, `train/lr_applied`, `train/lr_next`, `train/grad_norm`, `train/grad_clip_ratio`, `train/grad_was_clipped` |
| PPO | `train/ppo_loss`, `train/ppo_objective_component`, `train/ppo_clip_active_ratio`, `train/ppo_ratio_above_clip_ratio`, `train/ppo_ratio_below_clip_ratio`, `train/ppo_ratio_outside_clip_ratio`, `train/ppo_log_ratio_abs_max`, `train/ppo_log_ratio_abs_gt_{1,2,5}_ratio`, `train/ppo_ratio_nonfinite_ratio` |
| KL | `train/kl`, `train/kl_beta`, `train/kl_objective_component`, `train/kl_penalty_objective`, `train/kl_to_ppo_abs_ratio`, `train/kl_token_{count,max,nonfinite_ratio}`, `train/kl_token_gt_{0_1,1,10}_ratio` |
| π_old fidelity | `train/pi_old_recomputed`, `train/pi_old_claim_abs_error_{mean,max}`, `train/pi_old_claim_gt_1e_3_ratio`, `train/pi_old_claim_token_count` |
| Circuit breakers | `train/step_skipped_{nonfinite,nonfinite_policy_ratio,grad_spike,policy_ratio_drift}`, `train/ppo_ratio_outside_clip_skip_threshold` |
| Overlong (Eq. 13) | `train/overlong_{penalty_factor,rollout_count,forced_zeroed_ratio,changed_ratio,reward_delta_mean}` |
| Rewards | `rewards/{mean,std,min,max}` |
| Batch | `batch/{n_groups,n_degenerate_groups,degenerate_ratio}` |
| Throughput | `train/rollouts_{processed,total}`, `train/valid_rollout_ratio` |
| **Per environment** | `train/env/<env>/{groups,rollouts,reward_mean,reward_std,reward_min,reward_max,reward_nonzero_ratio,raw_completion_tokens,trainable_completion_tokens,plan_groups,plan_rollouts,abs_adv_weighted_tokens}` |

(`train/shaping_*` is the v3 shaping path and is inert under v4; keep it off the v4 dashboard.)

### 4.2 What is missing

Ordered by value. All are cheap — the tensors are already in hand at the point of logging.

| # | Proposed key | What it is | Why it matters |
|---|---|---|---|
| 1 | `train/env/<env>/k_histogram` (wandb Histogram) + `.../k_mean` | Per-group count of correct rollouts | The in-training echo of §2. Makes the gradient supply visible per env. |
| 2 | `train/env/<env>/in_zone_ratio`, `.../nondegenerate_ratio` | Same definitions as §2.3, on trained groups | `batch/degenerate_ratio` exists but is global; per-env is where the divergence lives. |
| 3 | `train/env/<env>/eos_ratio`, `.../at_cap_ratio` | `finish_reason` breakdown | **The ckpt84 collapse detector.** Currently invisible. |
| 4 | `train/env/<env>/overlong_zone_ratio` | Fraction in the last 4096 tokens before cap | Whether Eq. 13 is engaging at all. |
| 5 | `train/env/<env>/completion_len_{p50,p90,max}` | Length percentiles | Only the *sum* (`raw_completion_tokens`) exists; a mean hides rumination. Also the cost driver (§3.7). |
| 6 | `train/entropy_mean`, `train/entropy_p10` | Token-level policy entropy over trained tokens | **The best single early-warning for mode collapse.** Logits are already computed; this is a `logsumexp` away. |
| 7 | `train/advantage_{abs_mean,std}`, `train/zero_advantage_token_ratio` | Advantage distribution | Distinguishes "no gradient because no variance" from "no gradient because clipped". |
| 8 | `train/env/<env>/repetition_ratio` | Looping detector on token ids | 77% of non-finishers loop; nothing measures it. |
| 9 | `train/env/<env>/parseable_ratio` + extraction-path breakdown | Grader extraction branch taken | Catches format drift before it looks like a reasoning regression. |
| 10 | `train/prompt_unique_ratio`, `train/prompt_repeat_rate` | Prompt diversity within the step | Corpus has 36.4% duplicate rows; curated prompts are drawn 2–4× too often. |

**Entropy (#6) is the highest-value single addition.** With `KL_BETA = 0` in v4 there is no anchor at all, and entropy collapse is the failure mode clip-higher exists to prevent. Without it, the first evidence of collapse arrives from Part A, one eval cadence late.

### 4.3 The bias caveat, restated for the dashboard

Every Part B quantity is conditioned on "a miner chose to submit this prompt and it passed the σ-gate". `train/env/<env>/reward_mean` rising can mean the model improved, or that miners shifted toward easier prompts. **Label the Part B panel "supply & health (miner-selected prompts)" in the UI.** It is not a mislabelling risk worth taking.

---

## 5. Part C — The dashboard

Eight charts. The first four are the story; the rest are the diagnosis.

**1. Headline — the closing gap.** Per env: `pass@1` and `pass@16` versus training window, from the capacity suite. Base-model values as dashed horizontals. *The claim is the two lines converging while `pass@16` holds or rises.*

**2. Headroom.** `pass@16 − pass@1`, single line, monotone-down is the win condition. Annotate the v3 reference value (which stayed flat over 138 checkpoints) as a dashed horizontal.

**3. k-histogram over time.** 17-band stacked area, protocol suite, math. Mass visibly migrating from `k=0` toward `k=16` is the most legible single picture of RL working. Band 0 and band 16 in neutral greys, the interior in a sequential ramp.

**4. In-zone vs non-degenerate.** Two lines: `in_zone` (σ ≥ 0.43, prod definition) and `nondegenerate` (0 < k < G). Vertical marker at the v3→v4 cutover with a "G: 8 → 16" annotation, because the gate's k-range changes there (§2.3). *Expected story: in-zone falls because mass reaches k=16 — which is only credible when read against chart 3.*

**5. Termination health.** `eos_ratio` and `at_cap_ratio` (stacked to 1 with `stop`), plus `overlong_zone_ratio`. Second axis: `len_p50` and `len_p90`.

**6. Per-env small multiples.** Chart 1, repeated small for `openmathinstruct` and `opencodeinstruct`, plus the code environment's mean fractional reward (§2.4).

**7. Training loop panel** (Part B): entropy, `ppo_ratio_above_clip_ratio` vs `below`, `grad_norm`, `batch/degenerate_ratio`, with skipped steps as rug marks.

**8. Supply panel** (Part B): per-env in-zone ratio of accepted groups, k distribution of accepted groups, window fill. Labelled as miner-selected.

**9. Am I done? (§3.8).** Headroom conversion `C(t)` as a progress bar toward 1.0, with `prompt_churn` as a second line underneath and `frontier` as a shaded area. Three status lamps for D1/D2/D3 and the mechanism label. *This is the chart someone glances at to answer "is it still worth running".*

### 5.1 Reading the charts — diagnostic table

| Observation | Interpretation | Action |
|---|---|---|
| `pass@1` ↑, `pass@16` flat, gap ↓ | **Healthy.** RL is converting existing headroom. | Continue. |
| `pass@1` ↑, `pass@16` ↓ | Mode collapse — exploration traded for exploitation. | Check entropy; widen `PPO_CLIP_EPSILON_HIGH`. |
| Both flat, gap flat | No learning — the v3 failure mode. | Check `degenerate_ratio`, advantage stats, π_old fidelity. |
| in-zone ↓ **and** k-mass → 16 | **Success**, with a supply problem: prompts are being mastered. | Harder corpus, or dynamic sampling to replace the σ-gate. |
| in-zone ↓ **and** k-mass → 0 | Regression or degeneration. | Check length, EOS, repetition, format. |
| `at_cap_ratio` ↑, `eos_ratio` ↓ | Termination collapse (ckpt84 pattern). | Check the Eq. 13 ramp is engaging (`overlong_zone_ratio`). |
| `len_p90` ↑ with reward flat | Rumination. | As above. |
| `extraction_path.none` ↑ | Format drift, not a reasoning change. | Grader/prompt issue — do not tune the recipe. |
| in-zone ↓ exactly at the cutover | **Artefact of `SIGMA_MIN` at G=16.** | Not a finding. Compare `nondegenerate` instead. |
| gap flat, churn ≈ 0, non-degenerate < 5% | **Converged — fuel exhausted (M1).** | Stop. Change corpus, or add dynamic sampling. |
| gap flat, **churn high** | Wandering, not converged — the policy is still moving, the aggregate just cancels out. | Keep running; D1 alone would have lied. |
| gap ↓, `pass@G` ↓ (D2 fails) | **Collapse dressed as convergence.** | Roll back. Do not stop and call it done. |

---

## 6. Testing strategy

The metric layer is pure, deterministic, and small — so it should be tested to a standard the rest of the analysis code has never met. Every test below runs on CPU in under a second unless marked.

### 6.1 pass@k estimator — exact known values

Table-driven, `G = 16`, using values that are exact rationals:

| `k_correct` | `k` | Expected | Derivation |
|---|---|---|---|
| 0 | any | `0.0` | no correct sample |
| 16 | any | `1.0` | all correct |
| 1 | 8 | `0.5` exactly | `1 − C(15,8)/C(16,8) = 1 − 6435/12870` |
| 8 | 1 | `0.5` exactly | `1 − C(8,1)/C(16,1) = 1 − 8/16` |
| 4 | 2 | `0.45` exactly | `1 − C(12,2)/C(16,2) = 1 − 66/120` |
| c | 1 | `c/16` | estimator must reduce to the mean |

Plus properties: monotone non-decreasing in `k`; bounded in `[0,1]`; `pass@G == 1` iff `k_correct ≥ 1`; no overflow at `G = 1024` (guards a naive factorial implementation).

### 6.2 In-zone must equal production, not a re-derivation

The highest-value test in the suite. For `G ∈ {8, 16}` and every `k ∈ [0, G]`:

1. build the binary reward vector with `k` ones,
2. compute σ the way the harness does,
3. assert the harness's in-zone verdict **equals `reliquary.validator.verifier.is_in_zone(σ)`**, imported from the repo,
4. assert the resulting admitted k-set matches the table in §2.3 (`{2..6}` at G=8, `{4..12}` at G=16).

Step 3 is what keeps the harness pinned to production when `SIGMA_MIN` moves. Step 4 is what makes the *change* visible when it does.

Further cases:

- `test_in_zone_uses_raw_reward_not_shaped` — construct a group where the Eq. 13 overlong penalty would change σ enough to flip the verdict; assert the harness uses the raw graded reward (matching `_training_rewards`' contract).
- `test_in_zone_continuous_reward_for_code` — fractional rewards `[1.0, 0.5, 0.0, …]`; assert σ is computed on the vector, and that the code path never routes through the k↔σ table.
- `test_sigma_uses_population_not_sample_std` — a vector whose verdict differs under `n` vs `n−1`; assert `n` (matching `difficulty_auction.py`).
- `test_nondegenerate_and_in_zone_disagree_at_g16` — `k=2, G=16` is non-degenerate but **not** in-zone. This test is the executable form of the §2.3 confound.

### 6.3 Histogram → metrics

- `test_all_metrics_derive_from_histogram` — from a hand-built histogram, assert every field of `metrics.json`.
- `test_pass_at_1_equals_mean_reward_for_binary_env` — internal consistency; must hold on real artifacts too.
- `test_histogram_sums_to_one` and `test_histogram_has_g_plus_one_bins`.
- `test_empty_suite_raises` — an empty or all-errored suite must fail loudly, never emit zeros.
- `test_grader_errors_excluded_from_denominator` — one `null` reward in a group of 16 must yield `G_effective = 15`, not a zero.

**Termination criterion (§3.8)** — these are logic tests over synthetic checkpoint pairs, and the most important ones are the negatives:

- `test_collapse_is_not_reported_as_converged` — `pass@1` flat, `pass@G` falling, so `H` shrinks and D1 passes; assert `verdict == "collapsed"`, never `"converged"`. This is the criterion's whole reason for existing.
- `test_wandering_is_not_reported_as_converged` — identical aggregate `pass@1` at `t` and `t−W`, but per-prompt `k` shuffled; assert `prompt_churn` is large and D3 fails.
- `test_churn_is_zero_for_identical_per_prompt_k` and `test_churn_is_max_for_inverted_k`.
- `test_churn_requires_matching_suite_hash` — churn across different suites is meaningless and must raise.
- `test_breaker_storm_blocks_verdict` — with `step_skipped_*` firing in the window, `verdict` stays null regardless of D1–D3.
- `test_conversion_exceeds_one_when_pass_at_g_rises` — `C > 1` is legal and must not be clamped.
- `test_verdict_is_null_before_thresholds_calibrated` — the default configuration emits raw quantities and no verdict.
- `test_dead_prompt_ratio_requires_both_conditions` — `k = 0` with high sample diversity is *not* dead; assert it counts toward `frontier`.

### 6.4 Grading is production grading

- `test_harness_uses_production_grader` — assert identity of the imported callable, not behaviour. If someone copies the grader, this fails.
- **Golden fixtures** — a small corpus of real completions with expected verdicts, covering every extraction path: `\boxed{}`, `Answer:` line, inline LaTeX `\(…\)`, tail-number fallback, no parseable answer, and a known surface-form case (the reformatted-ground-truth pattern from the 5CX7 analysis).
- `test_regrade_is_pure` — grading the same artifact twice yields byte-identical metrics for math (no RNG, no network).
- `test_code_infrastructure_error_is_not_zero` — a stubbed `GraderInfrastructureError` produces `reward = null`, and raises `grader_error_ratio`, and never produces `0.0`.

### 6.5 The harness cannot drift from the protocol

- `test_eval_sampling_matches_active_profile` — assert the generation config equals `ACTIVE_PROTOCOL_PROFILE.sampling` field by field. Analogue of `test_v4_sampling_makes_ppo_ratio_space_match_sampling_space`.
- `test_eval_max_tokens_equals_env_cap` — `max_new_tokens == max_new_tokens_for_environment(env)` for every env in the suite. **This is the test that would have prevented the 8k evaluation mistake.**
- `test_top_k_zero_not_none` — `top_k=0` (the disable sentinel); `None` crashes `ForcedSeedLogitsProcessor.__init__`'s `int()` coercion.
- `test_v4_uses_raw_completion_prompt` — no chat template, no `<think>` injection, under `RAW_COMPLETION_PROMPTS`.
- `test_no_bft_under_v4` — the v4 profile has `bft = None`; the harness must not force-close anything.

### 6.6 Artifact contract

- Schema round-trip; unknown fields rejected; missing required fields fail loudly rather than defaulting.
- `test_suite_hash_pins_prompts` — comparing two runs with different `suite_sha256` raises. Silent comparison of two different prompt sets is the worst failure this system can have, because the result looks plausible.
- `test_seed_is_deterministic_function_of_prompt_and_index`.
- `test_manifest_records_harness_and_grader_sha`.

### 6.7 GPU smoke (`tests/gpu/`, not in CI)

- Generate 2 prompts × 2 samples on a tiny model; assert schema validity, non-empty completions, `finish_reason` present, and that stage 2 consumes the output end to end.
- One long-generation case that actually hits the 16384 cap, asserting `finish_reason == "length"` and that `at_cap_ratio` counts it.

### 6.8 Suggested layout

```
reliquary/eval/
  __init__.py
  histogram.py     # k-histogram, pass@k, in-zone, all derived metrics  ← the tested core
  termination.py   # §3.8: conversion, churn, frontier, D1-D3, mechanism, verdict
  suites.py        # frozen suite loading, hashing, contamination join
  artifact.py      # JSONL schema, read/write, manifest
  grade.py         # thin adapter over the production graders
  report.py        # metrics.json → markdown/HTML
scripts/
  eval_generate.py # stage 1 (GPU)
  eval_report.py   # stage 2 (CPU)
tests/unit/
  test_eval_histogram.py     # §6.1, §6.3
  test_eval_termination.py   # §6.3 termination block
  test_eval_in_zone.py       # §6.2  ← the important one
  test_eval_grading.py       # §6.4
  test_eval_protocol_pin.py  # §6.5
  test_eval_artifact.py      # §6.6
tests/gpu/
  test_eval_generate_smoke.py
```

`reliquary/eval/` is importable by the validator process, so Part B can reuse `histogram.py` for its in-training k-histogram rather than reimplementing the definitions. That shared definition is the reason not to put this in `scripts/` only.

---

## 7. Non-goals and deferred

- **Dynamic sampling** (DAPO §3.1, resample until `0 < k < G`). Consensus-affecting; belongs to the post-cutover auction workstream. This spec only *measures* the gap between the σ-gate and DAPO's filter.
- **Relaxing `SIGMA_MIN` for G=16.** §2.3 shows the gate is stricter than it was at G=8. That is a real decision, but it is an auction decision, not an evaluation one.
- **Public code benchmark adapter** (HumanEval+/MBPP+) for the capacity suite. Needs a translation layer to the structured-cases grader.
- **Row-level corpus filtering** to build a truly held-out protocol suite. Blocked on the footer-only manifest; contamination is measured instead (§3.5).
- **Automatic triggering** of stage 1 on checkpoint publication. Manual invocation first; automate once the cadence is settled.
- **The dashboard implementation itself.** Part C specifies the charts and their semantics, not the stack.

## 8. Open questions

1. **Which GPU pod, and is it standing or on-demand?** Affects whether the cadence in §3.6 is realistic. On-demand implies batching several checkpoints per session.
2. **Where do artifacts live?** R2 alongside the existing window archives is the obvious answer; needs a prefix and a retention decision (a few hundred MB per checkpoint).
3. **Does the capacity suite use full MATH-500 or the 256-prompt subsample?** Full is more quotable externally; 256 halves the cost and matches the protocol suite's power. Currently specified as 256.
4. **Greedy secondary track: yes or no?** Cheap (1/16th of a run) and zero-variance, but it is a third number to explain in every chart.
5. **Window `W` for the termination criterion (§3.8).** At the §3.6 cadence of one full evaluation per 64 windows, `W = 3` spans ~192 training windows — long enough that a stall is real, short enough to notice within a day. Confirm once the real cadence is set; `W` and the cadence must be decided together.
