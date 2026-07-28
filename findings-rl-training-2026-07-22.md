# What GRPO/RL training actually changed in our model — findings

**Date:** 2026-07-22
**Subject:** SN81 model, run `base-fresh-v3` — base **Qwen3.5-2B** vs the live checkpoint (**ckpt ~48–49**, ≈ 500 optimizer steps; checkpoints are published every ~10–12 steps).
**Method:** GPU evals on **held-out** prompts (0 overlap with training), production sampling (T=0.6 / top-p 0.95 / top-k 20), thinking mode on, graders validated (correct reference solution → 1.0, wrong → 0.0). Numbers are base → checkpoint.

---

## TL;DR

- **The model learned to TERMINATE — to conclude its output instead of rambling — not to reason or code better.**
- **Code is mature / plateaued** under this training (termination acquired ~100%, code *quality* capped). **Math still has termination headroom** (self-terminates ~63%, was 11%).
- **Root cause:** outcome-reward RL **elicits latent capability, it does not create it.** A 2B base has little latent capability to elicit. DeepSeek found the same — for small models they used **distillation, not RL**.
- **The real lever is distillation** (inject capability from a stronger model), *then* RL to sharpen — plus cleaning the reward (retire BFT gradually, fix the OCI grader).

---

## 1. What clearly improved: **termination**

The single dominant, robust effect of training:

| metric | base | checkpoint |
|---|---|---|
| Math — closes `</think>` on its own | **11%** | **63%** |
| Math — needs the BFT force | 89% | 37% |
| Math — natural (un-forced) pass@1 | 0.183 | 0.565 |
| Code (OCI) — EOS rate | 62% | 74% |

**Decomposition of the code gain (OCI, held-out).** Base fails mostly by *rambling and never emitting code* ("nocode"). Categorizing 20 base vs checkpoint generations at budget 4096:

- base: PASS 8, **nocode 7**, wrote-code-but-wrong 4, string-test-fail 1
- ckpt: PASS 14, **nocode 1**, wrote-code-but-wrong 2, string-test-fail 2

→ The **+6 passes come almost entirely from the −6 nocode.** The code the model writes is *not* better; it just now writes code instead of rambling. **Same mechanism as math: it learned to conclude.**

---

## 2. What did NOT improve: **capability**

- **Code (OCI, held-out, 80 prompts):** pass@1 0.353 → 0.539 — but per §1 this is termination, not coding skill.
- **Math (OMI, held-out, 120 prompts, BFT = production regime):** pass@1 **0.703 → 0.647** — flat / slightly down (within noise at n=120). The model's raw solving ability did not improve.
- **General code (out-of-distribution):** HumanEval 0.604 → 0.463, MBPP 0.556 → 0.529 — i.e. training on our distribution slightly *hurt* general coding (overfitting). *(Internal note — this is what proves the OCI gain doesn't generalize; keep it for the team, out of any public framing that isn't scoped to our env.)*

---

## 3. Response length did **not** grow — the contrast with DeepSeek-R1

Code length: median 1514 → 1588 (flat); **mean 2309 → 2088 (down)** — the base's high mean was rambling-to-cap, which the checkpoint does less of. **The model does not generate longer/richer reasoning.**

R1's signature was the *opposite*: its chain-of-thought **lengthened on its own** as capability grew. The absence of that here is additional evidence we are not getting R1-style capability emergence — only termination.

---

## 4. Why R1 grew and we don't — RL elicits, it doesn't create

R1 never rewarded length. Reward is outcome (correct/wrong). On a **capable base**, *reasoning longer ⇒ more likely correct*, so within a group the correct rollouts **are** the long ones → the GRPO advantage pulls toward length. Length is rewarded *indirectly, via the correlation length↔correct.*

- **R1 / DeepSeek-V3-Base (671B):** long → more correct → CoT grows.
- **Our Qwen 2B:** long → **rambling** (not more correct) → no gradient toward length → flat.

We verified this **in our own data**: correct rollouts are **not** longer than wrong ones —
- Math: correct median 2064 vs wrong 2067 (equal).
- Code: correct 1020 vs wrong 1051 (correct even slightly *shorter*).

The correlation "long ⇒ correct" exists only if the base can use long reasoning productively — a 671B can, a 2B can't. **This is capacity, not method.**

**DeepSeek confirmed this directly:** their 1.5B–32B models (`R1-Distill-*`) are **SFT distillations**, not RL. They ran GRPO directly on Qwen-32B and distillation beat it. Their conclusion: small models via large-scale RL don't even reach the performance of distillation. **We are running pure RL (R1-Zero style, no cold-start) on a 2B — exactly the case they showed does not work.**

---

## 5. Where the training signal (within-group variance) comes from

If not length, what makes some rollouts in a group correct and others wrong? (recent production data, 1240 groups)

**Math (with BFT):**
- Length: equal (2064 vs 2069).
- Termination: correct EOS 84% vs wrong 72%; correct forced 70% vs wrong 78%.
- **~75% of all rollouts are BFT-forced** (the model rambles on these hard prompts) → most of the reward variance is *"did the forced guess land right"* = a **coin-flip** on k≈2 problems. Not learnable.
- Wrong-answer diversity 0.62 distinct/count → a mix of sampling luck + some systematic error.

→ Math variance is dominated by **termination (learnable → learned) + forced-guess luck (not learnable).** The genuine reasoning signal is thin and buried.

**Code (no BFT):**
- **100% of rollouts terminate and write a code block** (both correct and wrong). Among wrong: **0% nocode, 100% wrote-code-but-fails.**
- → Code variance is **purely code correctness** — a **clean, uncontaminated signal.**

**The decisive point:** code has a *perfect* signal (everyone terminates; the variance is only "does the code pass") and capability **still plateaus.** So the problem is **not signal quality** — it is **capacity + learnability**: each prompt is seen once, there is no transferable "coding skill" pattern to accumulate across different problems, and the 2B is capacity-limited. Cleaning the math signal (removing BFT) will therefore **not** unlock capability either.

---

## 6. BFT (budget-forced termination) — what tuning it does

BFT applies to **math only** (forces `</think>\boxed{}` at 2048 tokens if the model hasn't concluded).

**Raising the thinking budget** (2048 → 4096), checkpoint:
- Termination up: `</think>` 63% → 74%, forced 37% → 26%.
- Solving up: pass@1 0.647 → 0.680 — **+0.03 ≈ 1 SE at n=120 → not significant (flat).**

→ A bigger budget **increases the chance it finishes** (BFT@2048 was cutting some legitimately-longer thinkers mid-thought), but **does not increase solving.** Budget is not the capability lever.

**Removing BFT:** makes the reward honest (today ~2/3 of the math reward is the forced guess). But **premature now** — at 37% still forced, dropping BFT turns those rollouts into reward-0 → k collapses toward 0/1 → the σ-gate starves the training signal (the old "math starvation" failure). Do it **gradually**: first add a small penalty on *forced* rollouts (currently exempt in `_shape_advantages`) to push forced% down, then remove BFT when forced% < ~15%. **It won't create capability** (§5) — it's signal hygiene.

---

## 7. Environment-implementation caveats (OCI grader)

Independent of the model, the OCI code env has real grading flaws that depress absolute scores and add noise (they hit both models, so they don't drive the base→checkpoint delta, but they matter for a clean reward):
- **21% of tests are exact-string match**, some **impossible** (e.g. `chatbot_response` — the exact expected string isn't derivable from the prompt) or **fragile** (e.g. `serialize_to_json` — exact JSON spacing).
- **100% of prompts are stdin/stdout-framed** while graded as function-return (the "contract" instruction mitigates it, so it is *not* the main failure mode, but it is friction).

→ Worth cleaning (drop impossible/fragile string-exact tests) for a cleaner reward and cleaner eval.

---

## 8. Implications / recommendations

1. **RL alone will not build capability on a 2B.** It has picked the easy fruit (termination) and refines surface/in-distribution patterns; capability is at the ceiling of what this training can extract.
2. **The proven recipe for small models (DeepSeek): distillation / cold-start SFT first** — inject high-quality reasoning traces from a stronger model so there is a transferable pattern to learn — **then** RL to sharpen.
3. **Clean the reward:** retire BFT gradually (soft penalty on forced first), fix the OCI grader (remove impossible/fragile string-exact tests).
4. **Manage expectations:** even well-distilled, a 2B has a ceiling; a materially stronger model needs a bigger base.

**One line:** *the training taught the model to conclude its answers (real, useful), not to reason better; on a 2B, RL elicits the little latent capability there is and then plateaus — capability has to be injected by distillation, RL only polishes it.*

---

## Appendix — headline numbers

| eval (held-out) | base | checkpoint | n |
|---|---|---|---|
| OCI code — pass@1 | 0.353 | 0.539 | 80×8 |
| OMI math — pass@1 (BFT) | 0.703 | 0.647 | 120×8 |
| OMI math — pass@1 (natural) | 0.183 | 0.565 | 120×8 |
| Math — `</think>` self-close | 11% | 63% | 120×8 |
| HumanEval (OOD) — pass@1 | 0.604 | 0.463 | 164 |
| MBPP (OOD) — pass@1 | 0.556 | 0.529 | 257 |

Within-group signal (production, 1240 groups): length correct≈wrong (math 2064/2067, code 1020/1051); math ~75% BFT-forced (variance ≈ forced-guess luck); code 100% terminate (variance = pure code correctness, yet capability flat).
