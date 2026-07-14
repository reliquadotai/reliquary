# CoT truncation handling: allow max-token hits, or not?

**Date:** 2026-06-23
**Branch:** `feat/cot-2b` (Qwen3.5-2B + thinking-on)
**Status:** decision pending. The branch currently ships the gate **strict (=1)**
to start (CoT therefore does not train yet — deferred on purpose); this doc frames
the choice and the measured evidence for picking A / B / C later.

## The decision in one line

When we re-enable Chain-of-Thought, a large share of rollouts run past the token
cap without finishing (no `</think>`, no `\boxed{}`, no EOS). We must decide
whether the protocol **allows** those max-token (truncated) rollouts to be
submitted and trained on, or **forbids** them. This single choice drives whether
CoT can train at all, how the model learns, and which cheating vectors open.

There are two ends of the spectrum and one middle:

- **A — Allow (relax the gate).** Raise `MAX_TRUNCATED_PER_SUBMISSION` (1 → 7) so
  groups with many truncated rollouts are accepted.
- **B — Forbid + force termination (canonical reasoning budget).** Keep the gate
  strict (=1) but make the model deterministically emit `</think>` + answer
  before the cap, so truncated rollouts essentially stop existing.
- **C — Middle.** Relax to a lower threshold (e.g. 3–4) to keep CoT trainable
  while making truncated-negative harvesting less profitable.

## Background: why the gate exists and why CoT fights it

A GRPO group is 8 rollouts on one prompt. Each rollout's reward is ~binary:
boxed-correct = 1 (a *positive*), wrong/no-answer = 0 (a *negative*). The σ-zone
gate only trains (and pays) a group that has **variance** — k ∈ {2..6} correct
of 8. All-correct (k=8) or all-wrong (k=0) → σ=0 → out of zone → no training,
no emission.

`MAX_TRUNCATED_PER_SUBMISSION = 1` rejects any submission with more than one
truncated (cap-hit, no natural EOS) rollout. Its original purpose: stop miners
from manufacturing "loser slots" by force-capping rollouts to fabricate a reward
vector.

With thinking ON the 2B truncates a large fraction of rollouts (it reasons long;
hard/medium prompts ramble to the cap). So the **variance-bearing in-zone groups
are exactly the ones with several truncated rollouts** → under the strict gate
they are rejected → **CoT cannot train (starvation).**

## Measured evidence (Qwen3.5-2B, vLLM, branch sampling config: T=0.6 / top_p=0.95 / top_k=20 / presence=1.5)

### Finish behaviour (the model bifurcates)
- Easy prompts → short, terminate, high-k (saturated, out of zone).
- Hard prompts → ramble to the cap, never terminate, k=0 (out of zone).
- Finish-rate vs cap: ~34% @ 8k → ~52% @ 16k → ~55% plateau (≈45% never finish,
  proven by an unbounded 262k-token run). 16k recovers most slow convergers; beyond
  has steep diminishing returns. The branch ships the cap at **32768** (commit
  fefc5c4) — the max safe under the ~50-64k verify OOM (verifier.py:424); 64k would
  need that fp32 cast chunked first. Note: verify time per rollout ~2× at 32k vs 16k.

### Over-generation truncation axis (the question: can a miner manufacture variance on an all-1s prompt?)
On 6 saturated/near-saturated prompts, M=48 rollouts each, cap 16384, counting
rollouts that **naturally hit max-tokens AND are reward-0** (harvestable negatives):

| prompt | k/48 | EOS/48 | truncated | trunc & reward-0 | harvest axis? |
|--------|------|--------|-----------|------------------|---------------|
| 9  | 48 | 48 | 0  | 0 | no |
| 13 | 48 | 48 | 0  | 0 | no |
| 24 | 48 | 48 | 0  | 0 | no |
| 3  | 47 | 47 | 1  | 1 | no (<2) |
| 21 | 45 | 46 | 2  | 1 | no (<2) |
| 26 | 45 | 36 | 12 | 3 | **yes (can build k≤6)** |

**Conclusion:** the axis is **real but prompt-dependent**. Genuinely saturated
prompts (the model terminates 100%, 0% ramble) yield **nothing** — there is no
negative to harvest. Prompts with a natural ramble-tail (e.g. #26, ~25% ramble)
yield reward-0 truncations a miner can harvest. It **scales with over-generation
effort**: ~6%/rollout (#26) is immediate; ~2% (#3, #21) needs more generation but
eventually pays; 0% prompts are immune. Relaxing the gate is what makes these
harvested negatives *submittable in quantity* (k=5 = 5 positives + 3 negatives).

### Filters do not stop the negatives
- Distribution filter (median chosen-prob ≥ 0.30, q10 ≥ 0.025): truncated/rambling
  rollouts **pass 100%** (median ~0.97) — the model rambles with near-maximal
  confidence. A "forcing off-distribution craters the probs" defense does **not**
  materialise.
- Authenticity filter (chosen-prob < 1e-10): blind to genuine sampled tokens; only
  catches injected/edited tokens.
- A second, *terminated* negative vector also exists (suppress the box token in the
  miner's own sampler → reward-0 rollout that ends on EOS) which bypasses the
  truncation gate entirely. It is the same manufacture/curation family that already
  dominates the subnet; the real fix is a σ-zone reward redesign, not any
  truncation setting. *(Note: this vector is disputed pending a head-to-head test
  against a specific filter; the over-generation axis above stands on its own.)*

## Option A — Allow (relax the gate to 7)

**How:** `MAX_TRUNCATED_PER_SUBMISSION` 1 → 7 (require ≥1 natural EOS per group, to
bound all-rambling compute grief). No generation changes beyond sampling.

**Pros**
- CoT becomes trainable immediately: the variance-bearing in-zone groups are
  accepted instead of rejected.
- Minimal code (one constant), no new generation mechanism, no verifier changes.
- **Training signal pushes toward termination "for free":** the reward-0 truncated
  rollouts are the negative class, so GRPO pushes the policy *away* from rambling
  and *toward* terminating-and-boxing. The model can learn to finish from the
  reward itself.

**Cons**
- Opens the over-generation truncation-harvest axis (measured: works on prompts
  with a natural ramble-tail; immune only on truly-saturated prompts).
- The model is never *forced* to terminate, so ~45% of hard-prompt rollouts still
  ramble to the cap → wasted generation + verify compute (verify cost grows with
  length; 16k is well under the ~50–64k verify OOM, so no crash, but it is slow).
- Does not close the curation/manufacture problem (which is open regardless).

## Option B — Forbid + force termination (canonical reasoning budget)

**How:** keep the gate strict (=1); the generation harness deterministically emits
`</think>` and forces the answer before the cap (a protocol rule, verified, not
flagged as tampering). Truncated rollouts essentially cease to exist.

**Pros**
- Closes the over-generation truncation axis: borderline prompts now terminate, so
  there is nothing to harvest (verified mechanism — no truncated rollouts → the
  gate at 1 never even triggers, and harvesting has no material).
- Bounds compute: every rollout finishes near or below budget → no 16k ramble waste
  → faster, cheaper verify.
- Cleanest behaviour for honest miners; the gate stays as designed.

**Cons**
- **Most work:** a generation mechanism (miner + reference) plus a verifier
  carve-out to recognise the forced-budget termination as valid.
- **Does not teach conciseness:** termination is *forced* at decode time, not
  *learned*. Remove the budget later and the model reverts; teaching real
  conciseness still needs an overlong/length-shaping reward on top.
- **Does not close the manufacture family either:** the box-suppressed *terminated*
  negative vector survives. So this buys exploit-safety on the truncation axis only,
  not on manufacture as a whole.
- Changes the rollout length distribution (all terminate) → in-zone variance leans
  more on answer-correctness and less on terminate-vs-ramble.

## Option C — Middle (relax to 3–4)

Keep CoT trainable (most in-zone groups have ≤3–4 truncated) while making the
harvest axis less profitable (a miner needs the natural truncations to fit under a
tighter ceiling, and cannot pad a group with many fabricated truncated negatives).
Cheap (one constant). Still does not force termination or close manufacture; a
heuristic middle, not a principled fix.

## Training impact summary

| | Strict gate (=1), no budget | A: relax (7) | B: canonical budget | C: relax (3–4) |
|---|---|---|---|---|
| CoT trainable | **no** (starvation) | yes | yes | yes |
| Model learns to terminate | n/a | yes, via reward | no (forced, not learned) | yes, via reward |
| Compute waste (16k rambles) | n/a | high | low | high |
| Over-gen truncation axis | closed (but CoT dead) | **open** | **closed** | reduced |
| Box-suppression manufacture | open | open | open | open |
| Code cost | none | ~1 line | high | ~1 line |

## Recommendation

For an **experiment branch** whose goal is to see whether CoT lets the subnet move
model performance fast: **Option A (relax to 7)** is the pragmatic choice. The
truncation axis it opens is weak and prompt-dependent, the broader manufacture
problem is open regardless of this setting (so the gate buys little real security),
and — importantly — the natural reward already pushes the policy toward terminating.
It is one constant and CoT trains today.

**Option B (canonical budget)** is the right **production-hardening** step once CoT
is shown to train: it closes the truncation axis and bounds compute, at the cost of
real implementation work, and should be paired with the σ-zone reward redesign that
actually closes the manufacture family.

**Option C** is a reasonable hedge if the truncation axis feels too exposed but the
canonical-budget work is not yet warranted.

The choice does **not** affect security against the dominant curation/manufacture
exploit — that requires the σ-zone redesign + semantic grader, tracked separately.
