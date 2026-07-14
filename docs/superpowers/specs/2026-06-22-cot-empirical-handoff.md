# Reliquary CoT — Empirical Session Handoff

**Date:** 2026-06-22 · **From:** Rom's session (H200 empirical run) · **To:** strategy/SOTA author
**Purpose:** the empirical layer under your three docs (Signal-Preservation 06-18, Findings 06-19,
SOTA Next-Steps 06-22). This is essentially the **P2 "offline bounded-CoT A/B"** you called for —
run on a fresh H200 against a prod-faithful env. Honesty labels: **[measured]** = ran it,
**[code]** = verified in repo, **[est]** = extrapolated.

---

## 0. TL;DR

- **Re-enabling CoT on a small Qwen3.5 (0.8B/2B) is not viable as the subnet is designed today** —
  not because of memory or the token cap, but because the model **doesn't reliably terminate its CoT**,
  and the **truncation gate then makes the resulting signal un-submittable**.
- **Your "termination collapse" (Doc 19) and "signal collapse" (Doc 18) are the same root we hit
  empirically.** Memory is a non-issue; the binding constraints are **termination + the truncation
  gate + verifier weakness.**
- **The safe path to CoT is a *canonical forced reasoning budget*, not relaxing the truncation gate.**
  Relaxing the gate re-opens an injection class the current detectors can't catch.

---

## 1. Setup (reproducible)

- **HW:** 1× H200 (143 GiB). Two venvs: prod-faithful HF (torch 2.7.0+cu128, transformers==5.9.0,
  flash-attn 2.8.3, **flash-linear-attention 0.5.0**, bitsandbytes) and a separate **vLLM 0.23.0**
  venv (torch 2.11/cu130). **vLLM needs nvcc** (CUDA-13 toolkit) for the Qwen GatedDeltaNet JIT
  kernel — without it engine init fails.
- **Models:** `Qwen/Qwen3.5-0.8B` (hidden 1024, 24L=18 linear+6 full), `-2B` (hidden 2048, 24L),
  `-4B` (hidden 2560, 32L). All: vocab **248320**, ctx 262144, tied embeddings, packaged as
  `Qwen3_5ForConditionalGeneration` (the **vision tower loads into VRAM** too). All are post-trained
  (base_model: `-Base`).
- **Harness:** `scripts/cot_vllm.py` (vLLM, n=M rollouts/prompt, native presence_penalty, streaming
  or batch), `cot_analysis.py`/`cot_dump.py` (HF), `cot_aggregate.py`, `analyze_inzone.py`,
  `smoke_train_vram.py`, `profile_verify.py`, `bench_train_microbatch.py`. Math reward = in-process
  `_compute_omi_reward`; in-zone = `verifier.rewards_std`/`is_in_zone` (SIGMA_MIN=0.43).

## 2. Memory & throughput — NOT the constraint [measured, 4B]

- **train_step peak VRAM:** 30.2 GiB @8k → 32.8 @16k → 39.2 @32k (resident: 1 model 8.55 / 2 models
  17.1 GiB; +~3 GiB per +8k). Micro-batch peak = the single longest rollout, not B_BATCH. **No OOM
  until ~200k tok**; the RoPE position limit (262k) bites before VRAM. H200 makes memory a non-issue.
- **Micro-batch budget doesn't speed training:** ~80 s/step regardless of 8k/16k/32k/64k budget (and
  per-rollout) — compute-bound, not overhead-bound. Bigger logical batch is also supply-limited by
  verify throughput, not memory.
- **Verify (GRAIL, batch=1 serial):** 168 ms@4k / 268@8k / 556@16k / 1378@32k. **OOMs at ~50-64k**
  in `verifier.py:424` `_gpu_completion_token_stats` — it casts the full `[N_completion × vocab]`
  logits to fp32 (≈60 GiB). **This is the real historical "validator OOM," and it's fixable** by
  row-chunking like the training path already does (`training.py:267`).
- **vLLM gen throughput (2B):** 9,672 tok/s @4k ctx (128 seqs) → **~340 tok/s @24k** (KV-cache
  preemption). Long-context CoT is KV-bound even on vLLM. (HF `model.generate` batch-8 ≈ 360 tok/s,
  head-of-line-blocked.)

## 3. CoT termination — the core problem [measured]

- **0.8B:** repetition loops; even with anti-repetition it doesn't cleanly terminate. Qwen's own card
  documents the 0.8B "is more prone to entering thinking loops" and ships **no generation_config.json**.
  Excluded.
- **2B:** terminates **stochastically** — proven it *can* (clean `<think>…</think>` + `\boxed{45}`
  correct + EOS @ **2124 tok**), but only on prompts within its competence.
- **Sampling matters and we used the right config.** The subnet's canonical sampling
  (`T_PROTO=0.9, top_p=1.0, top_k=0, no penalty`) is pathological for thinking (loops at any size).
  Qwen's **published benchmark thinking config is `T=0.6, top_p=0.95, top_k=20, presence_penalty=1.5`**
  (their headline "recommended" T=1.0 is *not* what produced their scores). **All our headline
  numbers below use T=0.6** (the benchmark config); T=1.0 runs were the unstable ones (3/3 ramble).
- **It is NOT a token-cap artifact [measured, no-max test]:** with `max_tokens = native 262144`,
  hard prompts ran **~30 min / ~60-130k tokens with zero termination**, while an easy control
  terminated in 28 s (~2-4k tok). Genuine non-convergence (coherent endless rumination), not a cut-off.
- **Finish-rate vs cap (48-prompt sample, T=0.6):** 34% @8k → **52% @16k** → ~55% plateau (64k/262k).
  Three families: ~⅓ easy converge <4k, ~⅕ medium converge 4k-16k (**cap recovers these**), **~45%
  hard never converge**. So a cap helps up to ~16k, then diminishing returns.

## 4. In-zone yield — looks OK, is mostly a truncation artifact [measured]

- **Raw (24 prompts × M=8, T=0.6, cap 8k):** in-zone **20.8% (5/24)**, pass@1 34%, EOS-term 34%,
  all-fail(k=0) **46%**, all-pass(k=8) 21%.
- **Validity check [measured]:** 95% of *correct* answers come from EOS-terminated rollouts (the model
  genuinely solves). **BUT the in-zone σ is manufactured by the truncated ramblers** acting as the
  negative class. Recomputing σ on **EOS-only** rollouts: **only 1 of 5 in-zone groups survives**
  (the rest collapse to all-pass). → **Real in-zone ≈ 4-8%, not 21%.** The model bifurcates:
  easy→saturate (σ0), hard→starve (σ0); the "trainable middle" is largely a cap artifact.

## 5. The killer: the truncation gate makes it un-trainable [code]

`batcher.py:960-967`: a submission with **>1 truncated/non-EOS rollout is rejected outright**
(`BAD_TERMINATION`). Comment `batcher.py:881`: it exists to stop miners manufacturing weak loser
slots via forced truncation (anti-exploit). Consequence chain:

1. CoT-2B truncates ~48% of rollouts.
2. The in-zone σ on medium prompts is **made by** those truncated ramblers (§4).
3. A group with >1 truncated is **rejected** → the variance-bearing groups are **un-submittable**.
4. The submittable groups (≤1 truncated) are terminated-mostly-correct → all-pass (σ0) → not trained.
5. On hard prompts the model truncates ~all 8 → can't even form a submittable group.

→ **Submittable in-zone ≈ 0-4%. The model effectively cannot be trained on CoT in this design.** The
gate (which assumes truncation is a rare accident) is fundamentally incompatible with a CoT model that
truncates ~half the time. This sharpens your Doc-19 point: it's not "truncated trains as wrong," it's
"truncated → the whole group is **rejected**, never seen by training."

## 6. Will GRPO fix termination? Partially, and not alone

- **For:** within an in-zone group GRPO reinforces terminated-correct over rambled-fail; your own
  prior run cut truncation 80.5%→41.2% and doubled pass@1 — so GRPO *does* reduce rambling when the
  model already has the capability.
- **Against (3 limits):** (a) the ~45% all-fail groups have σ0 → **zero gradient on exactly the prompts
  that ramble most** (your signal-collapse, confirmed); (b) cold-start is thin (real in-zone ~4-8%);
  (c) the reward rewards *correctness within cap*, not termination — and unguarded GRPO can **make
  termination worse** (your Doc-19 4B code regression: grad-norm 35.6M, KL ratchet → enshrined
  "won't stop"). **GRPO renforces termination if the model already has it (→ distill/SFT warm-start);
  it won't create it from scratch reliably.**

## 7. Security — relaxing the gate re-opens injection [code]

If we relax the truncation gate (overlong shaping) to make CoT trainable, mind this:
- **token-authenticity** flags only `chosen<1e-10 & argmax≥0.99` (`constants.py:618-619`).
- **token-distribution** assumes tampered ≈ 1/vocab ≈ 1e-5 (`median>0.30, q10>0.025`,
  `constants.py:588`).
- **Both are evaded by substituting the rank-2 token** (prob 0.01-0.4 ≫ both thresholds) — and
  **fundamentally so**: `T=0.9, top_k=0` sampling emits low-rank tokens *naturally*, so the validator
  cannot distinguish an honest rank-2 sample from an injected one. With **public ground truths**, the
  high-value injection is the answer/`</think>` region.
- Remaining structural defenses: `verify_termination` (p(EOS)≥`MIN_EOS_PROBABILITY=0.01` at the last
  token — but **cap-truncated rollouts BYPASS it**, `verifier.py:32`), `BOXED_ANSWER_MIN_PROB=0.001`,
  and GRAIL (self-consistent → blind to substitution).

**→ Safer design for CoT: a *canonical* forced reasoning budget** (deterministic `</think>` injection
at thinking-token B, applied identically by miner and validator → verified as a *protocol rule*, not
*detected* as tampering) **beats relaxing the gate + trusting detection.** This matches your
"harden verifier/trainer BEFORE expanding CoT."

## 8. Merged recommendation (your plan + our evidence)

1. **Trainer/verifier hardening first** (your P1) — and note the verifier weakness above (rank-2
   evasion) when designing the logprob/old-lp recompute.
2. **Fix the verify fp32-cast OOM** (`verifier.py:424`, chunk it) — unlocks long-seq verify cheaply.
3. **CoT only via a canonical forced reasoning budget** (~bounded 8-16k thinking) — trainable
   (kills truncation-starvation) AND non-tamperable. NOT gate-relaxation-by-detection.
4. **Overlong soft-shaping** as the reward complement (grade truncated, keep `verify_termination`).
5. **Model ≥ 4B or a distilled/SFT warm-start** — the decisive lever; GRPO refines termination, it
   doesn't create it. The raw 2B bifurcates.
6. **Signal-preservation** (DAPO dynamic sampling / RL-ZVP / bigger M — note M=8 undersamples) and
   **procedural difficulty-tunable envs** to stop feeding the ~45% hard prompts that give σ0.

## 9. Open question I'd want your read on

How to make the **forced reasoning budget canonical + GRAIL-verifiable**: the injected `</think>`
at position B is a token the model didn't sample, so it must be (a) deterministic across miner/
validator and (b) excluded from `verify_termination`/token-auth as a known protocol insertion rather
than flagged. That's the one design piece that makes CoT both trainable and safe.

## Artifacts (this machine)
- `cot_analysis_results.xlsx` — 9 sheets: configs, VRAM, verify, micro-batch, CoT-termination,
  vLLM throughput, **in-zone per-group (24 rows + totals)**, in-zone validity.
- `scripts/cot_vllm.py`, `cot_dump.py`, `cot_analysis.py`, `analyze_inzone.py`, `build_cot_excel.py`.
- Raw: `/root/inzone16k.jsonl`, `/root/vllm_inzone_2b.jsonl` (per-rollout: len, reward, eos, boxed).

*Honesty caveats: in-zone samples are 24/48 prompts (small); Doc-19 regression numbers (0.78→0.54)
not independently re-verified here; the 8k→16k finish-rate delta mixes cap + prompt-sample (direction
is solid). All sampling at the Qwen benchmark config T=0.6/top_p0.95/top_k20/presence1.5.*
