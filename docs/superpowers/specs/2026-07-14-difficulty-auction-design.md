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

### 2.5 Half the window cycle is GPU proofs

Measured end-to-end over the same 855 windows:

| phase | min | median | p90 |
|---|---|---|---|
| collection (open → last arrival) | 58 s | 176 s | 355 s |
| **processing (GRAIL + seal)** | 2 s | **173 s** | 396 s |
| **full cycle (open → next open)** | 36 s | **388 s** | 710 s |

**9.3 windows/h today**, and **45% of the cycle is proving submissions** — ~19
proofs per window, to keep 8. That is the budget the deadline gets paid out of:

| | cadence | vs today |
|---|---|---|
| today | 9.3 /h | — |
| 300 s deadline, proving at admission | 7.6 /h | **−18%** |
| 300 s deadline, proving only the top 8 | ~9.7 /h | **≈ neutral** |

The deadline and the proof reordering are therefore **one change, not two**.
Shipping the deadline alone costs 18% of training throughput; shipping it with
the reordering is free. (Note the min cycle today is 36 s — under a fixed
deadline the *minimum* window duration becomes the deadline itself, by
construction. An early seal is exactly the speed race being removed.)

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
   deadline and accepts everything valid. `BATCH_FILLED` disappears.
2. **Admit cheaply**: schema, signature, prompt match, dedup — **and no GPU
   proof at all**. A prompt already claimed this window rejects the second
   submission outright (`PROMPT_CLAIMED`), so a prompt is worth exactly one slot
   and is taken by whoever submitted for it first.
3. **Grade** every admitted submission (reward computation — CPU / sandbox).
4. **Score** each with `v(k)` (§5).
5. At the deadline, **rank**, then **prove top-down until 8 have passed** (§7).
   A submission that cannot reach the top 8 is **never proven** — that is where
   the GPU saving comes from.
6. **The 8 survivors split the env's pool**, flat (§8).

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
   "speed breaks ties" rule, and it is already a validated, consensus-safe field.
3. **`_within_slot_key` ascending** — the existing canonical hash
   (`sha256(hotkey ‖ prompt_idx ‖ selection_digest)`). Deterministic, and bound
   to the validator-computed `selection_digest` so it cannot be ground.

**`v(k)` is coarse — ties are the norm, not the exception.** With 7 distinct
score values and ~69 submissions per window, submissions tie at the top
constantly. The mechanism's real steady state is therefore **a speed race
restricted to the hardest prompts**. That is the intended outcome, not an edge
case, and it should be understood as such.

### One prompt, one slot — enforced at ADMISSION

A prompt already claimed this window rejects any further submission for it
(`PROMPT_CLAIMED`), rather than letting several miners compete for the same slot
and resolving it at selection. The first submitter for a prompt owns it.

This is simpler than resolving the collision at ranking time, and it is strictly
better on three counts:

- It saves the grading (and sandbox) cost of submissions that could never win.
- It removes the same-prompt K-way emission split entirely.
- **It kills the variance-farming sybil outright.** The forced-seed seed is
  `H(drand ‖ hotkey ‖ prompt ‖ i ‖ t)` — it contains the **hotkey**, so two
  hotkeys on one prompt produce *different groups, hence different k*. An
  operator with N hotkeys would otherwise get **N independent draws of k on one
  prompt** and submit whichever landed nearest k=2, while an honest miner gets a
  single draw and must take it. Admitting only the first submission per prompt
  means there is only ever one draw to have.

Residual, accepted: two miners who independently find the same hard prompt race
for it, and the loser wasted 8 rollouts. With a 14M-prompt corpus and
miner-chosen prompts, collisions are rare enough not to matter. Prompt-squatting
(submitting junk to deny a prompt) is likewise not worth defending against — a
squatter can block at most `MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = 8` prompts
out of 14M, and cannot know which prompts a competitor wants.

---

## 7. Prove only what can win

Today every admitted submission goes through the GRAIL GPU proof. That is ~19
proofs per window and it is **45% of the window cycle** (§2.5). Under a 300 s
deadline the admitted pool triples, so proving at admission would mean ~69 proofs
— which does not fit in the window at all.

**Grade first, prove last, and never prove a loser:**

```
admit (cheap: schema, sig, prompt dedup — NO proof)
  → grade  → score v(k)  → [deadline]  → rank
  → prove top-down until 8 have PASSED
  → a submission that cannot reach the top 8 is never proven
```

### Why deferring the proof does not create a cheating incentive

The score is computed from the reward vector, and a miner who never runs the
model can fabricate one: hand-write 2 correct answers and 6 wrong ones, and the
validator's own grader scores it k=2 — the exact peak. So fabricated groups
**rank at the top by construction**.

This is safe anyway, because **GRAIL still runs before anyone is paid.** The
fabricator reaches the top of the ranking, gets proven, fails, and earns zero. If
he is *not* in the top 8, he is neither proven nor paid. There is no path to
profit, so there is no incentive to fabricate, and at equilibrium the top of the
ranking is honest.

### The residual is griefing, not cheating — and it is bounded

The threat that survives is an attacker who does not want to *earn*, only to cost
us GPU. Fabricating costs him nothing (he never runs the model) and costs us a
proof. That attack **already exists today** — a fake burns a proof at admission
right now. What changes is only the **order in which the GPU budget is spent**:
today proofs go in *arrival* order; under the auction they go in *score* order,
and a fabricated group names its own score. We would be sorting our own GPU queue
to serve the griefer first.

Two bounds, one of which already exists:

- **`MAX_PROOF_ATTEMPTS_PER_WINDOW = 16`** (new). Prove top-down until 8 pass, or
  until 16 attempts are spent. A griefer costs at most 16 proofs, ever.
- **`MAX_EXPENSIVE_PROOF_FAILURES_PER_HOTKEY_PER_WINDOW = 2`** (exists). After
  two failed proofs a hotkey is out for the window, so burning all 16 attempts
  needs **8 registered hotkeys** — and registration is the tax that bounds it.

Promote-on-failure does the rest: every fake that drops out promotes the
next-ranked submission, so honest miners still fill the batch.

### The real cost: we stop verifying the losers

Deferring the proof means the forensic gates (token authenticity, distribution,
logprob, forced-seed) **only ever run on the winners**. Enforcement stays
complete — nobody unproven is ever paid — but *visibility* into the rest of the
fleet is lost. Those gates are how we caught the pre-generating miner, the
token-tamperers, and 1 088 `seed_mismatch` rejects in 855 windows.

Mitigation: prove a small **random sample of non-winners** each window
(`FORENSIC_SAMPLE_PER_WINDOW = 2`, drand-chosen so it cannot be predicted) purely
for telemetry. Cheap, and it keeps the detectors fed.

### Throughput risk to size before shipping

Grading now sees every submission (~69) instead of only the pre-seal ones (~19).
Math grading is cheap (sympy). **Code grading is the risk**: gVisor sandbox, 5 s
timeout, `GRADER_POOL_SIZE = 8`, 8 rollouts per submission → a worst case of
69 × 8 × 5 s / 8 workers ≈ 345 s, which would *exceed* the 300 s window. Math
ships first for exactly this reason (§10); code needs the grader pool sized up.

---

## 8. Emission

**Payout is flat across the 8 winners** — `slot_share = pool / B_BATCH`. The
score decides **who gets in**, not what a slot is worth. Because one prompt now
yields exactly one slot (§6), there is no same-prompt split left to compute.

Rejected alternative: paying proportionally to `v(k)`. It adds a continuous pull
toward ever-lower k, i.e. toward k=1 — the lucky-guess and broken-label region
(§9). A flat-prize tournament already ratchets difficulty upward endogenously: to
earn anything you must beat the 8th-best submission, so the cut tightens on its
own as the fleet supplies harder prompts.

**`MAX_SLOTS_PER_COLDKEY_PER_WINDOW = 2`.** The per-operator cap that today's
rule is missing — the 8-distinct cap is per *prompt*, not per *operator*, which
is how coldkey `5CQ6…` took 13.1% of emission by flooding distinct prompts across
hotkeys. It also bounds the centralisation the speed tie-break creates: with a
coarse 7-valued score, ties are constant, so the fastest hardware would otherwise
win every one of them. Replaying the auction on live windows *without* this cap,
the top hotkey's share of slots rose from 7.1% to 8.4% — the cap is not
theoretical.

Under-filled windows burn the unused share (`UID_BURN`), unchanged.

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

| name | value | role |
|---|---|---|
| `WINDOW_COLLECTION_SECONDS` | 300 | fixed deadline; also the MIN window duration |
| `DIFFICULTY_DELTA` | 1.0 | difficulty dial; peaks `v(k)` at k=2 |
| `MAX_SLOTS_PER_COLDKEY_PER_WINDOW` | 2 | per-operator slot cap |
| `MAX_PROOF_ATTEMPTS_PER_WINDOW` | 16 | bounds the griefer (§7) |
| `FORENSIC_SAMPLE_PER_WINDOW` | 2 | random non-winners proven for telemetry (§7) |

**Constants retired.** `SIGMA_MIN` / `BOOTSTRAP_SIGMA_MIN` (subsumed by `v(k)`),
`BATCH_FILLED` and the drand-boundary seal extension, `MAX_POST_TRIGGER_PROOF_CANDIDATES`,
and the sparse-window liveness breakers (`SPARSE_VALID_IDLE_SEAL_SECONDS`,
`SPARSE_VALID_IDLE_MIN_DISTINCT_PROMPTS`, `SPARSE_VALID_MAX_WINDOW_SECONDS`,
`WINDOW_TIMEOUT_SECONDS`) — all of which exist only to answer "when do we stop
waiting?", which a fixed deadline answers unconditionally.

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
