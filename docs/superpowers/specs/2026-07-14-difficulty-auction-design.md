# Difficulty Auction — Design

**Date:** 2026-07-14
**Status:** Design approved, ARMING BLOCKED on one measurement (see §9)
**Supersedes the "Problem 2" fix direction in** `2026-07-09-incentive-training-state-of-situation.md`
(whose stated mechanism is refuted below — see §2.3)

---

## 1. Problem

The model plateaus because it trains on prompts it already solves.

Today a miner earns a slot by landing an in-zone group among the **8 first
submissions to arrive** (ordered by attached drand round). Emission is flat per
slot. Difficulty is not paid. The observed accepted distribution is easy-leaning:

| k (correct of 8) | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| share of accepted | 7.8% | 9.4% | 34.7% | 19.5% | **28.5%** |

Mean k = **4.52** — the model is right more often than not on the prompts it
trains on. Measured over 855 live windows (w22000–22965, `.r2_analysis/`).

---

## 2. Evidence

All figures from the live run, 855 windows, 13 414 accepted submissions.

### 2.1 The arrival race dominates everything

`batch_filled` is **96.1% of all rejects**: 45 593 valid submissions turned away
against 13 414 accepted. **Roughly three of every four valid submissions are
discarded for arriving late**, before anyone looks at what they contain.

### 2.2 The σ-gate is what actually causes the easy-skew

Only **51 `out_of_zone` rejects in 47 000**. Miners essentially never submit an
out-of-zone group — for math they compute the reward locally and self-filter.

This is the root cause. `SIGMA_MIN = 0.43` admits only k ∈ [2,6]. A genuinely
hard prompt — where the model succeeds 0 or 1 times in 8 — is **rejected and
pays zero**. Attempting hard prompts is economically irrational, so miners never
attempt them. The easy-skew is not caused by the speed race; it is caused by
**hard prompts being worthless**.

### 2.3 REFUTED: "miners fabricate variance by generating short"

The state-of-situation doc (§4) hypothesised that miners manufacture in-zone
variance with a tight token budget on easy prompts, making wrong rollouts short.
Measured directly, inside accepted groups:

| | mean len CORRECT | mean len INCORRECT | ratio |
|---|---|---|---|
| math | 1094 tok | 1117 tok | **1.01** |
| code | 755 tok | 763 tok | **1.01** |

Wrong rollouts are **not** shorter (they are shorter in only 46.5% of groups —
a coin flip). Current group variance is **genuine**: the model is wrong because
the problem is hard, not because it was cut off. **The generation-length floor
proposed in that doc is therefore unnecessary and is dropped from this design.**

### 2.4 Timing — the deadline is nearly free

| | median | p75 | p90 |
|---|---|---|---|
| generation time, math | **176 s** | 267 s | 364 s |
| generation time, code | 68 s | 117 s | 180 s |
| **actual window duration today** | **277 s** | 392 s | 490 s |

Windows already stay open ~4.6 min. A **300 s** deadline captures ~89% of
current submissions at **near-zero cadence cost**. (A 120 s deadline would be
*shorter* than today's median window and would cut off most math.)

---

## 3. Why k=2 and not k=6 — the GRPO advantage flips sign

GRPO advantage is `A = (r − μ)/σ`. With 8 binary rollouts:

**k=6** (μ=0.75): the 6 correct rollouts get `+0.58`; the 2 wrong get **`−1.73`**.
The dominant per-sample signal is **negative** — "stop making these mistakes."
On a prompt the model already solves, those mistakes are noise. We polish.

**k=2** (μ=0.25): the 2 correct rollouts get **`+1.73`**; the 6 wrong get `−0.58`.
The dominant per-sample signal is **positive** — "that rare solution you found,
do it again." The model locks in a path it currently misses 3 times out of 4.
**That is where capability is acquired.**

> k > 4 → suppress errors. k < 4 → amplify discoveries. Identical σ, opposite
> pedagogical value. `SIGMA_MIN` is blind to this because `√(p(1−p))` is
> **symmetric** — it cannot tell k=2 from k=6. That symmetry is the core bug.

---

## 4. Mechanism

Replace *"the 8 fastest in-zone submissions win"* with
*"the 8 hardest submissions win; speed breaks ties."*

1. **Fixed collection deadline** (`WINDOW_COLLECTION_SECONDS = 300`). The window
   no longer seals on the 8th distinct prompt. It stays open for the full
   deadline and accepts everything valid.
2. **Grade every admitted submission** (cheap: reward computation, no GPU proof).
3. **Score** each submission with `v(k)` (§5).
4. **Rank** and select the top 8 over **distinct prompts** (§6).
5. **GRAIL-prove only the selected**, promoting the next-ranked on failure (§7).
6. **Pay the 8 selected slots flat**, reusing the existing emission math (§8).

Speed keeps exactly the job it had — pressure on training throughput — but it is
now a **tie-break inside a difficulty class**, not the primary filter.

**The window becomes time-boxed, which retires a whole class of seal logic.**
The 8-distinct seal trigger, the drand-boundary seal extension, and the
sparse-window liveness breakers (`SPARSE_VALID_IDLE_SEAL_SECONDS`,
`SPARSE_VALID_MAX_WINDOW_SECONDS`, and the `WINDOW_TIMEOUT_SECONDS` backstop) all
exist to answer "when do we stop waiting?". A fixed deadline answers that
unconditionally, so they become dead weight and should be removed rather than
left to interact with the new path. Under-filled windows behave as today:
fewer than 8 selected slots means the unused share **burns** (`UID_BURN`), with
no redistribution.

---

## 5. The score

Let `k` = number of correct rollouts in the group, `p = k / M_ROLLOUTS`.

```
v(k) = sqrt(p * (1 - p)) * (1 - p) ** DIFFICULTY_DELTA        # DIFFICULTY_DELTA = 1.0
```

Two factors:

- `sqrt(p(1−p))` — **is there anything to learn?** The group's disagreement.
  Zero when all 8 rollouts agree; GRPO extracts nothing from a unanimous group
  (all advantages cancel). This is exactly today's σ.
- `(1 − p)^δ` — **must the model explore?** The failure rate. This is the term
  that breaks the symmetry and separates k=2 from k=6.

At `δ = 1.0`:

| k | 0 | 1 | **2** | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| v(k) | **0** | .29 | **.32** | .30 | .25 | .18 | .11 | .04 | **0** |
| % of peak | 0 | 89% | **100%** | 93% | 77% | 56% | 33% | 13% | 0 |

**`v` replaces `SIGMA_MIN` entirely.** `v(0) = v(8) = 0` already excludes the
zero-signal groups the σ-gate was there to reject, and the band widens on the
hard side to admit k=1 (today rejected outright).

**Why the peak is at k=2 and not k=1.** With a single success in 8, that success
may be a **lucky guess** — a `\boxed{}` that lands right on wrong reasoning.
Reinforcing it teaches bad reasoning. Two or three independent successes are far
more likely to be a genuinely discoverable strategy. `δ = 1.0` places the peak
exactly at k=2. k=1 is still paid (89% of peak) but is not the optimum.

**Why the curve stays smooth rather than becoming a hard band.** A hard band
(e.g. reject k ≥ 5) would starve windows in which the fleet finds nothing hard —
the exact failure mode already hit once (forced-seed → math starvation →
13 train_steps/h collapsing to 1). With a smooth curve an easy group can still
fill a slot when nothing better exists; it simply pays into a lower rank.
**Liveness is preserved by construction.**

---

## 6. Selection

Deterministic across validators (they must converge to identical weights).

Sort key, descending priority:

1. **`v(k)` descending** — difficulty first.
2. **`drand_round` ascending** — the earlier 3-second bucket wins. This is the
   user-visible "speed breaks ties" rule, and it is already a validated,
   consensus-safe field.
3. **`_within_slot_key` ascending** — the existing canonical hash
   (`sha256(hotkey ‖ prompt_idx ‖ selection_digest)`). Deterministic, and bound
   to the validator-computed `selection_digest` so it cannot be ground.

Constraints applied while filling the 8 slots:

- **Distinct prompts.** One slot per `prompt_idx` (as today). The highest-scoring
  submission for a prompt takes it. This makes the "two miners picked the same
  prompt" case a *consequence* of the general rule, not a special case.
- **Cooldown.** Unchanged.
- **Per-coldkey slot cap** (`MAX_SLOTS_PER_COLDKEY_PER_WINDOW = 2`). See §8.

**`v(k)` is coarse — expect ties to be the norm, not the exception.** With only
7 distinct score values and ~69 submissions per window, many submissions tie at
the top. In practice the mechanism therefore behaves as **a speed race
restricted to the hardest prompts** — which is the intended outcome, but it
should be understood as the design's actual steady state rather than an edge
case.

---

## 7. GRAIL ordering (feasibility — this is load-bearing)

Today every admitted submission goes through the GRAIL GPU proof (the candidate
budget was removed in PR #107). At ~69 submissions per window and ~4 s per proof,
that is ~276 s of serial GPU work and **does not fit** a 300 s window.

**Grade first, prove last:**

```
admit  →  grade (CPU/sandbox, cheap)  →  score v(k)  →  rank
       →  GRAIL-prove the top 8 only
       →  on proof failure, drop and promote the next-ranked, repeat
```

This **reduces** GPU load relative to today (8–16 proofs per window instead of
~30–60), and it is what makes the 300 s deadline affordable.

Consequence to size before shipping: the grading path now sees every submission
rather than only the pre-seal ones. `MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW = 96`
becomes the *effective admission bound* — and because it is first-come, **it
would reintroduce a latency race at the admission layer if it binds**. It must be
sized comfortably above the expected per-window submission count, or made
non-first-come. Code grading (gVisor sandbox, 5 s timeout, `GRADER_POOL_SIZE = 8`)
is the throughput risk here, not math.

---

## 8. Emission and Sybil

**Payout is flat across the 8 selected slots** — `slot_share = pool / B_BATCH`,
exactly as today. The score decides **who gets in**, not what a slot is worth.

Rejected alternative: paying proportionally to `v(k)`. It adds a continuous pull
toward ever-lower k, i.e. toward k=1 — the lucky-guess and broken-label region
(§9). A tournament with a flat prize already ratchets difficulty upward
endogenously: to earn anything you must beat the 8th-best submission, so the cut
tightens as the fleet supplies harder prompts. Flat is simpler, safer, and reuses
the existing, audited emission math.

**New Sybil vector introduced by paying for difficulty.** The forced-seed seed is
`H(drand ‖ hotkey ‖ prompt ‖ i ‖ t)` — it contains the **hotkey**. Two hotkeys on
the same prompt therefore produce **different groups, hence different k**. An
operator with N hotkeys gets **N independent draws of k on one prompt** and
submits whichever lands nearest k=2. An honest single miner gets one draw and
must take it. This is **variance farming**, and paying for difficulty is what
creates it.

Two defences:

- **`MAX_SUBMISSIONS_PER_COLDKEY_PER_PROMPT = 1`.** Searching many *prompts* is
  the work we want to pay for (frontier prediction). Replaying many *seeds* on
  one prompt is pure fabrication. This separates the two surgically.
- **`MAX_SLOTS_PER_COLDKEY_PER_WINDOW = 2`.** The missing per-miner cap — today's
  8-distinct cap is per *prompt*, not per *operator*, which is what let coldkey
  `5CQ6…` take 13.1% of emission. It also bounds the centralisation pressure that
  the speed tie-break creates (fastest hardware would otherwise win every tie).

---

## 9. The load-bearing risk: grader correctness

**A false negative is now worth the maximum payout.**

Fabricating a low k by degrading generation is already impossible — verified
against current `main`:

| Attack | Blocked by |
|---|---|
| Force BFT early (300 instead of 2048 tokens) | `validate_force_span` pins the force span to **exactly** `BFT_THINKING_BUDGET` and rejects any `</think>` before it (`verifier.py:1096`) |
| Lower `max_new_tokens` to cut rollouts | no EOS → `MAX_TRUNCATED_PER_SUBMISSION = 1` — only **one** truncated rollout allowed per group |
| Degrade sampling (temperature / top-p) | `LogprobValidator`, `DistributionValidator`, and above all **forced-seed** (merged, PR #106, `FORCED_SEED_ENFORCE=true`) which pins every draw |
| Swap in a weaker model | GRAIL sketch + logprob checks |

So the only way to reach k=2 by generation is to **find a genuinely hard prompt**.
The anti-fabrication substrate is already built.

**But that displaces the attack onto the label.** A prompt whose ground truth is
mis-formatted, ambiguous, or defeats the grader produces a **correct** model
answer graded **wrong** → k collapses → maximum score. Hunting broken prompts is
far cheaper than hunting hard ones, and under this design it **pays more**. Worse,
the resulting training signal has **inverted labels**: we would actively push the
model away from correct answers. This is the one scenario in which this redesign
makes the model *worse* than today.

This is not hypothetical. The #1 math earner stopped editing tokens after PR #92
and pivoted to exactly this: **≥26% of its negatives were reformatted ground
truth** (model right, grader wrong).

**BLOCKING GATE — measure before arming.** The semantic-equality grader (value /
unit / structured / algebraic equivalence) has since shipped and should have
largely closed this. That is an assumption, not a measurement.

> **M5: residual free-negative rate on live math negatives, post semantic-grader.**
> Take recent accepted groups, isolate rollouts graded incorrect, and count how
> many are in fact correct (answer equivalent to ground truth but rejected).
>
> - **~0%** → arm the auction.
> - **still a few %** → a guard is required first: low-k groups (the ones that
>   now pay the most) must be re-verified more strictly than high-k groups
>   before they can be selected.

---

## 10. Scope and rollout

**In scope.** `openmathinstruct` first — it is `DEFAULT_ENVIRONMENTS` and the
live path. `opencodeinstruct` follows once grading throughput (§7) is sized.

**Prerequisites.**
- forced-seed armed (**already merged and on by default**)
- M5 measurement passed (§9)
- grading-throughput sizing (§7)

**Rollout.** Ship the score and the ranking in **shadow** first: compute `v(k)`
and the would-be auction batch every window, archive it alongside the real
drand-ordered batch, and compare — how different is the selected set, what is the
mean k of the auction batch, does any coldkey dominate the ranking. Arm only once
the shadow batch looks the way this document predicts.

**Constants introduced.**

| name | value |
|---|---|
| `WINDOW_COLLECTION_SECONDS` | 300 |
| `DIFFICULTY_DELTA` | 1.0 |
| `MAX_SLOTS_PER_COLDKEY_PER_WINDOW` | 2 |
| `MAX_SUBMISSIONS_PER_COLDKEY_PER_PROMPT` | 1 |

**Constants retired.** `SIGMA_MIN` / `BOOTSTRAP_SIGMA_MIN` — subsumed by `v(k)`,
which is zero exactly where the σ-gate rejected for zero signal (k=0, k=8) and
non-zero across the whole informative band.

---

## 11. What this design does NOT claim

It does **not** reduce wasted miner compute. There are still 8 winners out of
~69 submissions; the losers still earn nothing. What changes is **why** a
submission loses: today "you arrived late" (arbitrary, rewards latency and
hardware), under this design "your prompt was not hard enough" (aligned with what
the model needs). The waste is unchanged; the competition becomes useful.

It also does not, by itself, prove the subnet beats centralised DAPO on
sample-efficiency. That benchmark remains the open thesis question
(state-of-situation §5d).
