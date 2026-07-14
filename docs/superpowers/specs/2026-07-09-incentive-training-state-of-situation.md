# Reliquary SN81 — State of the Situation: Incentive & Training-Quality Gaps

**Date:** 2026-07-09
**Scope:** The open *structural* problems in the incentive mechanism that degrade
training quality and cap model progress. This is deliberately distinct from the
defensive/anti-cheat layer, which is largely mature (see §1). The point of this
document is to name *why the model plateaus*, not to catalogue more exploits.

---

## 1. What is already solid (do not re-solve)

The defensive layer has reached diminishing returns. The following are shipped or
in-flight and cover the exploit classes we spent months on:

- **Semantic-equality grader (math + code).** Value/unit/structured/algebraic
  equivalence for OMI; entry-name injection + modality contract for OpenCode.
  The "GT-reformatted free-negative" family is closed.
- **DAPO overlong handling.** Cap-truncated rollouts penalized (`SHAPE_PENALTY`),
  BFT force-span masked out of the loss.
- **Token-level per-env loss** (DAPO normalization).
- **Token-authenticity gates** (numeric + all-token consistency).
- **Truncation guard** (`MAX_TRUNCATED_PER_SUBMISSION=1`).
- **drand-round ordering** — killed the millisecond TCP / co-location race.

The remaining problems below are **not** more anti-cheat patches. They are
structural misalignments between what the market pays for and what the trainer
needs.

---

## 2. Root cause (unifying)

**Emission is decoupled from pedagogical value.** A miner is paid for *landing an
in-zone group cheaply*, regardless of whether that group is a genuine
learning-frontier prompt or cheaply-manufactured easy variance. Both problems
below are symptoms of the same thing: the reward signal cares only that a group
passes the σ-gate, not what the group is worth to training. Miners therefore
optimize for "the cheapest way to pass the σ-gate," which is (P1) curate the
sample and (P2) generate short on easy prompts.

---

## 3. Problem 1 — Reward-shape curation

**Mechanism.** The protocol asks for 8 rollouts, but the validator cannot prove a
miner did not generate N ≫ 8 and cherry-pick the 8 that produce a favorable reward
vector (e.g. exactly k=4 → σ ≈ 0.5 → in-zone). The variance is not *discovered*,
it is *manufactured by post-hoc selection*.

**Why it is harmful.**
- It defeats the core thesis. The subnet is supposed to pay for *predicting* which
  prompts sit at the learning frontier. Curation lets miners *fabricate* an
  in-zone group on almost any prompt, so the thing being rewarded is
  brute-force generation + selection, not frontier prediction.
- It biases training. The two harmful channels are free-negatives (a correct
  answer graded wrong → label noise that pushes the policy the wrong way) and, to
  a lesser extent, arrangement/order gaming (cosmetic against detectors; note the
  GRPO advantage itself is order-invariant, so ordering does not corrupt the
  gradient directly).

**Structural enabler.** Public math labels — the miner can compute the reward
locally and instantly, so any "commit-then-reveal" timing defense is useless for
math. (Code is already validator-private.)

**Fix direction.** Forced-seed / commit-first sampling: the per-position sampling
draw is forced to a public value `u = H(drand ‖ hotkey ‖ prompt ‖ i ‖ t)`, so
there is only one legal group per (miner, prompt, window). Verification is a
teacher-forced inverse-CDF consistency check that piggybacks the forward pass the
validator already runs for GRAIL, deployable as a statistical gate in the existing
auth-forensics framework (shadow first, then enforce). Pair with private/generated
math to move it out of the public-label regime. **Cost: a coordinated miner-client
upgrade with an adoption window** — unlike the grader/DAPO fixes this is not
validator-only.

**Status.** Not built. Needs an offline feasibility measurement first (does the
honest cross-GPU consistency floor separate cleanly from cherry-picked
submissions?).

---

## 4. Problem 2 — Speed race → short generation → easy-skew

**The race still exists.** drand ordering removed the millisecond/TCP advantage,
but coarse-grained speed still wins:
- `drand_round` is stamped at POST time. Generating shorter → finishing sooner →
  POSTing sooner → an **earlier drand round**, and selection at seal orders by
  earliest round → short generation gets selection priority.
- The window seals on a distinct-prompt target, so a slow miner may not make the
  pool at all.
- Shorter generation → more prompts scanned per second → more in-zone groups found.

There is **no generation-length floor** — only a 32k-token ceiling. So generating
short-and-tight is structurally rewarded.

**Consequence.** Miners minimize tokens subject to "in-zone." The cheapest way to
produce variance is on **easy problems**: on a truly easy prompt the model is
8/8 (σ ≈ 0, rejected); on a hard prompt with a tight budget everything fails
(σ ≈ 0, rejected); but on an *easy-ish* prompt a tight budget makes some rollouts
land and some miss → mixed k → in-zone. The variance is fabricated by the *budget*,
not by genuine difficulty. This is consistent with the observed **k=6 easy-lean**
mode and the 7–19% hard-task signal share — the model keeps re-learning the easy
region and never explores the hard frontier. Pass@1 plateaus.

**Amplifier (the important part).** The anti-truncation guard
(`MAX_TRUNCATED_PER_SUBMISSION=1` + penalty) means a group full of
max-token-hitting rollouts is rejected. But **hard prompts are exactly the ones
where the model rambles to max tokens.** So the guard that stops truncation-gaming
*also structurally excludes the hardest prompts* — the high-variance, high-value
frontier prompts that carry the biggest learning signal never make it into a valid
group. We are, in effect, filtering out our best training data as a side effect of
a defensive rule.

**Fix direction.** (a) A **generation-length floor** — mandatory thinking budget
(BFT used as a *floor*, not only a ceiling) so short-and-easy generation is invalid
and variance can only come from genuine difficulty; (b) **speed-neutral selection**
— draw the batch slots randomly (drand-seeded) among all valid in-zone submissions
instead of "earliest round wins," so speed only decides whether you enter the
window, not whether you are selected. Tension: both slow the window cadence, which
is already bottlenecked by GRAIL verify.

**Status.** Hypothesis strongly consistent with evidence, but confirm before
building — see M1/M3 in §6.

---

## 5. Additional observations (added beyond the two problems above)

**5a. The σ-gate is difficulty-blind.** It admits k=2..6 uniformly, treating a k=2
group (hard: model right only 2/8, true frontier) and a k=6 group (easy) as equally
valuable and equally paid. In curriculum terms they are not equal — the low-k group
is far richer signal. A **difficulty-aware reward** (weight low-k in-zone groups
higher, or shift the band) directly counters easy-skew and is a cheap,
validator-side lever that touches both problems.

**5b. The two problems compound.** Curation (P1) lets a miner manufacture in-zone
variance *even on prompts that are not genuinely variable*, which makes the
easy-skew (P2) worse: you do not even need a real easy-ish prompt, you can curate
one. Fixing P1 (forced seed) also removes the ability to fake variance, which
partially defends P2.

**5c. Operational fragility is a multiplier, not a footnote.** The GRAIL verify
throughput bottleneck tightens the seal (fewer slots processed per window) → more
pressure to be fast → *worse* speed race. Beyond that: single-trainer SPOF; the
grader cgroup leak took all code rewards to zero for days; a validator restart
resets AdamW momentum and regresses training. These directly shape how hard the
incentive problems bite.

**5d. The thesis is still unproven.** There is no benchmark showing the
decentralized market matches or beats centralized DAPO sample-efficiency. The
problems above are a plausible reason *why* it might not yet (easy-skew caps
pass@1). A proper benchmark is both a credibility artifact and the only objective
way to tell whether any of these fixes actually move the model.

---

## 6. Measurements to run before building (verify-first)

- **M1 — P2 smoking gun.** In in-zone math groups, is `completion_length` of the
  *incorrect* rollouts systematically shorter than the *correct* ones? Short →
  "wrong-because-short" → P2 confirmed. Same/longer → wrong-because-hard → P2
  refuted. Validator-side, on R2 archives, runnable now.
- **M2 — Are miners capping short?** Distribution of accepted-rollout
  `completion_length`. Clustered low → confirms tight-budget behavior.
- **M3 — Does the truncation filter exclude hard prompts?** Rejection rate by
  reason (truncation/out-of-zone) as a function of prompt difficulty. Confirms the
  §4 amplifier.
- **M4 — P1 feasibility.** Offline, on existing honest rollouts + a simulated `u`
  stream across different GPUs: does the inverse-CDF consistency score separate
  honest-follows-seed from cherry-picked? This de-risks the entire forced-seed
  design and is validator-only.
- **M5 — Did the grader fix land?** Post-fix free-negative rate on live math
  negatives (was ~26% GT-reformatted). Confirms §1.

---

## 7. Priority read (recommendation)

1. **Confirm P2 (M1/M2/M3)** — cheap, validator-only, now.
2. **P2 fixes (length floor + speed-neutral selection)** are likely the #1
   *model-quality* lever and are protocol/validator-side — cheaper than P1.
3. **σ-gate difficulty-weighting (5a)** — cheap complementary lever.
4. **P1 / commit-first** — the incentive-*integrity* fix; higher cost (client
   upgrade). Gate it behind the M4 feasibility measurement.
5. **Prove the thesis (5d)** — run the DAPO sample-efficiency benchmark; it also
   becomes the yardstick for everything above.
