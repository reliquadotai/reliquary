# SOTA Handoff — σ-gate decision + DAPO alignment audit

**Date:** 2026-06-24 · **Classification:** INTERNAL · **Branch context:** `feat/cot-2b`
**Companion (buddy-facing reply):** `2026-06-24-sigma-gate-keep-and-maxtoken-penalty.md`

This is the state-of-thread after evaluating the **σ-Zone Reward Redesign** proposal against DAPO and against the live code. Read the BLUF, then the arguments, then the open items.

---

## BLUF — decisions reached

1. **KEEP the strict σ-gate (`SIGMA_MIN = 0.43`, admits k∈{2..6}).** Reject the proposal's "remove the filter / accept-all + global baseline" (its change **A**). Also reject the intermediate idea of relaxing to DAPO's k∈{0,8}-only boundary (`σ≥0.33`). The strictness is load-bearing in an adversarial setting.
2. **ADD a max-tokens (truncation) penalty** — DAPO's 4th brick (Overlong Reward Shaping), which the subnet does not yet have. This is the correct way to address band-collapse/starvation: it fills the in-zone band with *real, directed* variance instead of widening the gate, and it supplies the CoT termination signal.
3. **KEEP** the proposal's change **C** (difficulty-tuned / procedural prompt selection) — DAPO-consistent answer to band-collapse. **DROP/REPLACE** change **B** (generic conciseness shaping) — superseded by the safer truncation-only penalty.
4. **DAPO alignment audit (new, code-verified):** the subnet does **not** follow DAPO's Clip-Higher, and **keeps** the KL penalty DAPO removes. It is a conservative DeepSeek-GRPO config that borrowed only DAPO's learning rate. This is in tension with the CoT goal (which needs exploration / drift from base). Flagged for a second A/B; no change committed.

---

## Background — the thread

A collaborator proposed replacing the σ-zone accept gate with accept-all + a global/LOO (RLOO/RL-ZVP) baseline, plus conciseness shaping (B) and difficulty selection (C), arguing the σ-gate starves training as the model masters data (measured: 2B saturates ~58% of MATH-500 → k=8 → rejected) and creates a perverse "drop correct answers to manufacture variance" curation incentive.

The diagnosis of the *symptom* is real; the proposed *cure* (A) was rejected because it deletes the subnet's core advantage. The thread then drilled into DAPO mechanics and audited the live code.

---

## Decision 1 — keep the strict σ-gate

### Why not remove it (vs change A)
The σ-gate **is DAPO's Dynamic Sampling** (drop zero-variance groups; keep gradient-carrying ones). It is the measured source of the subnet's edge: at 300 matched train_steps, **Reliquary pass@1 0.61 vs vanilla GRPO 0.47** (base 0.33); vanilla GRPO trains on all groups including zero-variance. "Accept-all + global baseline" ≈ vanilla GRPO → regress toward ~0.47 → delete the value prop. A global baseline also adds a cross-prompt coupling a miner games by flooding saturated groups to shift everyone's advantage.

### Why not even relax to DAPO's boundary
DAPO filters only k∈{0,8} and keeps k=1,7. The subnet is stricter (σ≥0.43 ⇒ k∈{2..6}, rejecting k=1,7). **The extra strictness is an anti-cherry-pick barrier, not over-filtering:**

> The gate is a **proof-of-difficulty requirement on the curator.** Landing in k∈{2..6} forces assembling a group with **≥2 genuine minority outcomes on each side** (≥2 real failures *and* ≥2 real successes) — costly to source, hard to fake. Relax to k∈{1,7} and the curator needs only **one** minority outcome → a near-saturated group is **submitted directly, zero manufacturing**, weak evidence of difficulty, cheaply faked. Widening the band collapses the marginal cost of curation toward zero exactly at the saturated edge.

DAPO's k∈{0,8} boundary is correct for a **trusted** trainer (no adversary assembling groups). The subnet is **adversarial**, so the band edge must double as an economic barrier. This is the decisive point and it reverses the earlier "relax to DAPO" suggestion.

**Code confirms the design intent** — `constants.py:254`: *"Bernoulli rewards this admits k ∈ [2, 6] for M=8 (σ at k=2/6 ≈ 0.433)."*

---

## Decision 2 — add a max-tokens (truncation) penalty

Penalize rollouts that hit `max_tokens` (ran off the budget without terminating). DAPO brick #4. Two payoffs, one mechanism:

- **Fills the band without touching the gate.** A ramble-prone would-be-k=8 splits into `{finished-correct = high, truncated = low}` → real σ → in-zone, trainable, gate stays strict. The variance comes from **termination behaviour**, not manufactured correctness noise.
- **Teaches the model to finish** — the exact CoT termination signal (think within budget, emit EOS/`\boxed`).

### Design rules (so it doesn't backfire)
- **Penalize running off the budget, NOT thinking long** (trigger on `hit_max_tokens`, not length) → avoids teaching under-thinking / premature `\boxed`. This is why it's safer than change B's generic brevity shaping.
- **Soft ramp + generous budget**, not a cliff at the cap (DAPO buffer zone).
- **Hard floor, never partial-positive** for truncation → not farmable by truncated garbage.
- **Transient by design:** as the model learns to finish, those prompts saturate at k=8 and correctly leave the band. Sustained band-fill is change C's job, not the penalty's.

### Security interaction (already has a partial home in code)
A known exploit manufactures reward-0 negatives by suppressing EOS/`\boxed` (e.g. `\boxed{<|im_end|>`) to fake a k=4/σ=0.5 vector that passes the gate — see `reliquary/validator/boxed_integrity.py`. A truncation penalty makes those manufactured negatives score *lower* → **widens the σ the curator manufactures** → marginally easier exploit. **Ship the penalty gated behind the truncation/boxed-authenticity check (extend `boxed_integrity.py`), not alone.**

---

## DAPO alignment audit (code-verified)

| DAPO brick | Subnet status | Locus |
|---|---|---|
| Dynamic Sampling (drop zero-variance) | **Yes** (this is the σ-gate) | `verifier.is_in_zone` / `batcher.py:829` |
| Clip-Higher (ε_high > ε_low) | **No — symmetric ε=0.2** | `constants.py:521`, `training.py:397` & `:509` |
| Token-level loss | Yes | `training.py:507-516` |
| Overlong Reward Shaping | **No — to be added (Decision 2)** | n/a |
| (DAPO removes KL) | **Opposite — KL kept, β=0.04** | `constants.py:525`, `training.py:516` |

**Key findings:**
- **Clip is symmetric:** `torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON)` with a single `PPO_CLIP_EPSILON = 0.2` → `[0.8, 1.2]`. **Not** clip-higher. Comment self-describes as *"Standard in GRPO/RLHF literature."*
- **KL penalty kept:** `KL_BETA = 0.04`, loss `= scale·(ppo + KL_BETA·kl)` against the frozen reference (base). DAPO sets β=0 to let long-CoT policies drift from base.
- **LR borrowed from DAPO only:** `LEARNING_RATE = 5e-6`, comment: *"Matched DAPO / R1-Zero-scale literature."* So they took DAPO's LR but **not** its two signature algorithmic changes.
- **Net:** the config is conservative/anchored on two axes at once (symmetric clip + KL-to-base) — the **opposite philosophy to DAPO** (explore-and-drift). For the CoT goal (long reasoning must diverge from base, needs entropy), both settings work *against* emergence. Candidate for the second A/B below. **No change made.**
- **Bonus:** `BOOTSTRAP_SIGMA_MIN = 0.33` (`constants.py:258`) — the subnet already relaxes to ~DAPO's boundary during bootstrap windows only; confirms 0.33 is a deliberate, separate regime, not the steady-state gate.

---

## Conceptual clarifications (for whoever picks this up)

- **The clip ratio is `π_new / π_old`, where π_old is the policy that *generated* the rollouts (a recent snapshot), NOT the base model.** The base model is the reference only for the KL term. In the subnet there's genuine off-policy distance (miner generates with one checkpoint, validator trains later) so the clip actually bites.
- **π_old must be recomputed validator-side** (never trusted from the miner) or a miner manufactures a fake ratio → fake advantage. This guardrail must stay on for any reward/advantage change.
- **Reward variance (σ-gate) and epsilon (clip) are different pipeline stages.** σ decides *if* a group trains (gate, stage 2) and scales the advantage `A=(r−μ)/σ` (stage 3); ε caps *how much* each token moves at update (stage 4). **Reducing ε is the wrong tool for curation/variance** — it's downstream of the gate, doesn't change what's accepted, and shrinking it symmetrically risks entropy collapse (the opposite of clip-higher).

---

## Open items / next steps

1. **A/B #1 (the deciding test for Decision 2)** — offline on the provisioned H200, reuse `cot_grpo_smoke.py`: **(a)** strict gate, no penalty vs **(b)** strict gate + max-tokens penalty. Metrics: pass@1-over-steps, in-zone fill-rate (band-fill), median completion length / truncation-rate over steps. Ship behind the testnet-462 canary if (b) fills the band and cuts truncation without hurting pass@1.
2. **A/B #2 (the DAPO-direction question)** — current (symmetric 0.2 + KL β=0.04) vs DAPO-direction (clip-higher 0.2/0.28 + reduced/zero KL). Metrics: policy entropy + completion length over steps. This is the exploration/CoT-emergence question, independent of Decisions 1–2.
3. **Implementation loci for the penalty:** the reward path in `reliquary/validator/verifier.py` (where correctness reward is assigned, before σ is computed), plus the truncation/boxed-authenticity gate in `reliquary/validator/boxed_integrity.py`. The σ-gate (`batcher.py:829`, `verifier.is_in_zone`) stays untouched.

---

## Pointers

- **Code:** σ-gate `constants.py:257` (`SIGMA_MIN`), comment `:254` (k∈[2,6]); `verifier.py:456/472` (`rewards_std`/`is_in_zone`); `batcher.py:829-836` (accept/OUT_OF_ZONE); clip `constants.py:521` + `training.py:397/509`; KL `constants.py:525` + `training.py:516`; advantage `training.py:120`; LR `constants.py:518`; manufacture exploit `boxed_integrity.py`.
- **Docs:** companion reply `2026-06-24-sigma-gate-keep-and-maxtoken-penalty.md`; original proposal (the σ-Zone Reward Redesign); CoT context `2026-06-23-cot-truncation-gate-decision.md`.
- **Memory:** `project_cot_reenable_qwen35_2026_06_22` (the redesign was previously framed as the fix — **this handoff refines that:** keep the gate, add the penalty); `project_offline_baseline_2026_05_23` (the 0.61 vs 0.47 / 2× edge).
