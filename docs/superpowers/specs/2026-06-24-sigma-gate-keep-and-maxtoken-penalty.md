# Keep the σ-gate strict; fill the band with a max-tokens penalty, not by widening the gate

**Date:** 2026-06-24 · **Classification:** INTERNAL · **Re:** the σ-Zone Reward Redesign proposal

## TL;DR

Your diagnosis of the *symptom* (the in-zone band collapses as the model improves → training starves) is real. But the proposed cure — **remove the filter / accept all groups with a global baseline** — throws away the one thing that makes this subnet worth running, and it does so by misreading why the filter exists.

Two distinct claims, both of which we reject:

1. **"Remove the σ-gate entirely" (your change A).** No. The σ-gate *is* DAPO's Dynamic Sampling, and it's the measured source of our 2× per-step edge.
2. **"Relax σ_min toward DAPO's k∈{0,8}-only boundary" (my earlier counter-suggestion).** Also no — and this is the subtler point. **DAPO's boundary is calibrated for a trusted trainer; ours has to be stricter because the band edge doubles as an anti-cherry-pick barrier.**

The right fix for band-collapse is to **fill the band honestly** with a **max-tokens (truncation) penalty** — DAPO's 4th brick, which we don't yet have — not to widen the gate.

---

## 1. The filter is the product, not a bug

DAPO = **D**ecoupled Clip and **D**ynamic s**A**mpling **P**olicy **O**ptimization. Four bricks:

1. **Dynamic Sampling** — drop groups with acc=1 or acc=0 (zero variance → zero advantage → zero gradient), oversample to keep the batch full of prompts that actually carry gradient.
2. **Clip-Higher** — decouple ε_low / ε_high, raise ε_high to prevent entropy collapse.
3. **Token-level loss** — long sequences contribute proportionally.
4. **Overlong Reward Shaping** — soft penalty on truncated/overlong completions to cut reward noise.

Our σ-gate (`σ(rewards) ≥ 0.43`) **is brick #1.** And it's not theoretical: at 300 matched train_steps we measured **Reliquary pass@1 0.61 vs vanilla GRPO 0.47** (base 0.33) — vanilla GRPO trains on *all* groups, zero-variance included. **The 2× per-step efficiency is the filter concentrating the gradient.** "Accept all groups with a global baseline" is, mechanically, a return to vanilla GRPO → we regress toward ~0.47 → we delete the subnet's reason to exist. Plus a global baseline introduces a cross-prompt coupling a miner can game by flooding the batch with saturated groups to shift everyone's advantage. Change A is self-sabotage.

## 2. Why ours is *stricter* than DAPO — and must be

The numbers (8 binary rollouts, k correct, σ = √((k/8)(1−k/8))):

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| σ | 0 | 0.33 | **0.43** | 0.48 | 0.50 | 0.48 | **0.43** | 0.33 | 0 |

- **DAPO** filters only k∈{0,8}. It *keeps* k=1 and k=7.
- **We** require σ≥0.43 ⇒ accept k∈{2..6}, reject k∈{0,1,7,8}.

I previously argued we should relax to DAPO's boundary on gradient-efficiency grounds. **That argument is wrong for an adversarial setting, and here's the mechanism:**

> The σ-gate is also a **proof-of-difficulty requirement on the curator.** To land in k∈{2..6} from a pool, a cherry-picker must assemble a group containing **at least 2 genuine minority outcomes on each side** — at least 2 real failures *and* 2 real successes. That is costly to source and harder to fake.
>
> Relax to k∈{1,7} and the curator only needs **one** minority outcome. A near-saturated group (k=7, "almost everyone got it") is **submitted directly, zero manufacturing** — it's weak evidence of genuine difficulty, cheaply faked, and it pays. **Widening the band collapses the marginal cost of curation toward zero exactly at the saturated edge.**

So the extra strictness isn't over-filtering by accident. It's load-bearing: **k∈{2..6} forces the miner to demonstrate balanced genuine difficulty; k∈{1,7} lets them cash in on barely-perturbed saturation.** In a trusted trainer (DAPO) that distinction is irrelevant — there's no adversary assembling groups. For us it's the whole game. **Keep σ_min = 0.43.**

## 3. Band-collapse is real — but the cure is "fill honestly," not "widen the gate"

Your underlying worry stands: as the policy masters the data, genuine groups cluster at k=7,8 → the gate rejects more → batches don't fill → starvation. DAPO's *own* answer to this is **not** "stop filtering" — it's **oversample + keep the data hard.** Two clean levers:

- **(C from your doc — keep it):** difficulty-tuned / procedural prompt selection to keep feeding frontier prompts where variance is *natural*. This is the only permanent cure as the model approaches the dataset ceiling.
- **(new — below):** a max-tokens penalty that turns ramble-saturated prompts into trainable, in-zone groups **without touching the gate.**

Both fill the band by producing *real* variance. Neither weakens the anti-curation barrier.

---

## 4. The fix we're adding: a max-tokens (truncation) penalty

This is DAPO brick #4 (Overlong Reward Shaping), which the subnet doesn't have yet. Penalize rollouts that hit `max_tokens` (ran off the budget without terminating). Two payoffs, one mechanism:

### 4a. It creates *directed* variance → fills the band without widening it
Today, a ramble-prone prompt produces a would-be-k=8 where some rollouts finish (correct) and some hit the cap. With the penalty, that group now spans `{finished-correct = high, truncated = low}` → real σ → **in-zone, trainable** — and the gate stays strict. Crucially the variance comes from **termination behaviour, not from manufactured correctness noise**, so it's legitimate signal, not the fake σ the cap currently manufactures from truncated ramblers.

Note this pairs with the branch's existing `MAX_TRUNCATED 1→7` relaxation: we already **admit** truncated rollouts instead of rejecting them — penalizing them is the natural completion (admit + grade-down = directed signal, vs the old admit-and-zero or reject-and-discard).

### 4b. It teaches the model to finish — the CoT termination signal
This is exactly the signal CoT needs: learn to think *within budget* and terminate (EOS / `\boxed`). The saturated/ramble prompts stop being waste and become the termination-training set.

### Design rules so it doesn't backfire
- **Penalize running off the budget, NOT thinking long.** The penalty triggers on `hit_max_tokens`, not on length per se. A rollout that uses 90% of the budget and terminates correctly scores full reward; only the one that gets cut off is penalized. This is what keeps it from teaching *under-thinking* (premature `\boxed` on hard prompts) — the failure mode that generic brevity-shaping (your change B) risks.
- **Soft ramp + generous budget**, not a cliff at the cap. (DAPO uses a buffer zone where the penalty ramps in.) A steep penalty + tight budget = the model learns to rush. Keep the budget generous enough that finishing is achievable on hard prompts.
- **Hard floor, never partial-positive.** Truncation is strictly *worst*, not a consolation reward. If truncated-on-track paid something positive, miners would farm it by submitting truncated garbage. Floor it (e.g. truncation → 0 or a small negative, below finished-wrong is optional but keep it ≤ finished-wrong).
- **It's transient by design.** As the model learns to finish, those prompts stop truncating → they saturate at k=8 → they correctly leave the band. That's not a failure; it means the skill was acquired. Sustained band-fill is (C)'s job, not the penalty's. Don't oversell the penalty as the permanent starvation fix.

### One interaction to watch (security)
We have a known exploit family: a curator suppresses EOS/`\boxed` to **manufacture** reward-0 negatives on easy prompts (the distribution filter is blind to it). A truncation penalty makes those manufactured negatives score *lower*, which **widens the σ the curator manufactures** — i.e. it can make that specific exploit marginally *easier* to land in-zone. The penalty doesn't create the hole, but it amplifies it. **Mitigation:** gate the penalty's contribution behind a truncation-authenticity check (is the truncation organic rambling, or suppressed generation?), the same way we gate token-authenticity. Ship the penalty and the authenticity check together, not the penalty alone.

---

## 5. Verdict — what survives, what dies

| Proposal | Decision | Reason |
|---|---|---|
| **A.** Remove filter / global baseline | **Kill** | Deletes the 2× edge; reopens batch-composition gaming |
| Relax σ_min to DAPO's k∈{0,8} | **Kill** (my own earlier idea) | Makes near-saturated cherry-pick free; strict band = proof-of-difficulty barrier |
| **B.** Generic conciseness/brevity shaping | **Replace** | Too close to under-thinking; superseded by the safer truncation-only penalty |
| **C.** Difficulty-tuned / procedural selection | **Keep** | DAPO-consistent answer to band-collapse; the only permanent cure |
| **NEW.** Max-tokens penalty (DAPO brick #4) | **Add** | Directed variance fills the band *without* widening the gate; teaches termination |

**Net:** keep the gate strict (σ_min=0.43) → add the max-tokens penalty (+ truncation-authenticity) → keep difficulty-selection. That makes us **DAPO-complete** (we'd finally have all four bricks) **and** adversarially hardened — instead of trading the edge away to remove a filter that was never the real problem.

## 6. How to settle it without risking prod

Offline A/B on the H200 (reuse `cot_grpo_smoke.py`): fixed prompt set, compare **(a) strict gate, no penalty** vs **(b) strict gate + max-tokens penalty**. Metrics: pass@1-over-steps, fraction of groups that land in-zone (band-fill), and median completion length / truncation-rate over steps (does it actually learn to finish?). If (b) fills the band and cuts truncation without hurting pass@1, it ships behind the testnet canary.
