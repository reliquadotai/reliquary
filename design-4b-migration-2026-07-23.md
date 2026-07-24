# Design: migrate to Qwen3.5-4B — 16k budget cap + throughput-based draw tie-break

**Date:** 2026-07-23
**Status:** proposal (backed by GPU behavior study, this doc's appendix)
**Author:** training/incentives

## Summary

Three coupled changes, all flowing from a single fact: **the 4B model reasons much longer than the 2B.**

1. **Model:** `Qwen/Qwen3.5-2B` → `Qwen/Qwen3.5-4B`. The 2B has hit a capability ceiling — RL taught it only to terminate, not to reason, because its errors are *random* (nothing to learn from). The 4B's errors are *systematic* → RL finally has a correctable signal.
2. **Budget cap / BFT:** `2048` → **`16000`, as a clean cap (no forced answer)**. BFT@2048 was calibrated for a 2B that rambled; it would cut 45% of the 4B's *productive* reasoning.
3. **Draw tie-break:** pure arrival speed → **throughput (tokens/second)**. A pure-speed tie-break penalizes long generation, which directly fights the deep reasoning we now want.

Changes 2 and 3 are not optional add-ons — a longer-reasoning model **requires** re-basing every "punish slow / punish long" mechanism (all tuned for the short-rambling 2B) onto a high cap and a throughput metric.

---

## Part 1 — Model: Qwen3.5-2B → Qwen3.5-4B

### Why the 2B is done
Extensive GPU testing established that RL on the 2B improves **termination** (the model learns to conclude its output) but **not capability** (math solving flat, code gains are termination). The mechanistic reason: in the 2B's rollout groups, correct answers are **not** distinguished from wrong ones by anything learnable — wrong-answer diversity ≈ **0.62** (near-random guessing). There is no systematic error for GRPO to correct, so the gradient can only sharpen surface/termination behavior.

### Why the 4B is a real step up (behavior study, held-out OMI/OCI)
- **Math pass@1 (natural, no forcing): 0.68** vs 2B 0.57 (2B base 0.18).
- **Errors are systematic: wrong-answer diversity 0.27** (vs 2B 0.62). When the 4B is wrong, it is *consistently* wrong — a correctable pattern. **This is the signal the 2B never had.**
- **Code is at the model's frontier:** spread score distribution (~55% of prompts land in the trainable k∈{2..6} band) → rich, non-degenerate RL signal.

→ On the 2B, RL had nothing to teach about *solving*. On the 4B, the systematic errors mean RL can actually move capability. The bottleneck shifts **from the model to our curriculum and grading**.

### Cost
The 4B needs the budget cap and tie-break retuned (Parts 2–3) and a curriculum that keeps it at its frontier (math is bimodal on easy prompts — either k=8 or k=0; the auction must surface harder-but-solvable problems).

---

## Part 2 — Budget cap / BFT: 2048 → 16000, clean cap (no forced answer)

### Background
Today, math rollouts use **BFT** (budget-forced termination): if the model has not closed `</think>` by `BFT_THINKING_BUDGET = 2048` tokens, we inject `</think>\n\nFinal Answer: \boxed{` and let it emit a forced answer. This was built for the 2B, which rarely self-terminated and rambled to the cap.

### The tests

**Behavior run** (4B, held-out math, `max_tokens=16384`):
- Thinking length until `</think>`: **median 3766, p90 11297**.
- 66% close `</think>` on their own within 16384; **34% do not finish**.

The 4B's natural reasoning is *far* past 2048. BFT@2048 would guillotine legitimate mid-reasoning.

**Budget probe** (regenerate math at `max_tokens=32768`, isolate the rollouts that did NOT finish at 16384):

| of the non-finishers at 16384 (35% of rollouts) | share | detail |
|---|---|---|
| just needed more budget (concluded 16k–32k) | **23%** | **62% of them CORRECT**, median length 21933 |
| **never concluded even at 32768** | **77%** | 43% are detectable repetition loops |

**Verdict: 23% budget-limited vs 77% never/loop.**

### The decision: cap at 16000

- **2048 (current) → reject.** Cuts 45% of productive reasoning. Catastrophic for a model whose median thinking is 3766.
- **No cap / unbounded → reject.** 77% of non-finishers (≈27% of *all* rollouts) never terminate — they burn compute forever and never produce a trainable answer.
- **16000 → adopt.** Captures ~95% of productive reasoning (thinking p90 = 11297) while bounding looper waste. Raising to 24k would recover the ~5% extra-long correct solvers, but at the price of letting the 27% loopers each burn 24k — not worth it.

### Clean cap, NOT a forced answer

On the 2B, BFT *forced* a boxed answer out of non-concluders. We proved this **pollutes the signal**: ~75% of production rollouts were forced, so most of the reward variance was a **coin-flip guess**, not the model's reasoning — RL cannot learn from luck.

The 4B is different: at pass@1 0.68 it produces plenty of genuine correct answers, so **within-group variance already exists** (real correct vs looper-scored-0) without any forcing. Therefore:

> Non-concluders at the 16k cap are graded as **`bad_termination` (reward 0)** — an honest failure — rather than rescued with a forced coin-flip answer. Cleaner signal, no luck pollution.

### Dynamics (the cap is not static)
- Training improves termination (fewer loopers over time), exactly as on the 2B.
- Start the cap at **16k** (20k if you want headroom early).
- Add a **small penalty on non-concluders** to accelerate termination learning.
- **Lower the cap** as the thinking-length p90 falls.

### Config
- `BFT_THINKING_BUDGET`: 2048 → 16000.
- New flag to **disable the forced-answer** and treat cap-hits as `bad_termination` (keep the old forced behavior behind a kill-switch for clean revert).

---

## Part 3 — Draw tie-break: pure speed → throughput (tokens/second)

### The problem — and why it is coupled to the 4B
The current auction breaks **draws** (equal-value submissions) by **arrival speed** (earlier drand round wins). With a model that reasons long, this backfires: a miner producing **16k tokens of correct reasoning arrives after** a miner who answers in 500 tokens, and **loses the draw** — despite doing exactly what we now want. A pure-speed tie-break is a standing incentive to **generate short**, which directly cancels the point of moving to the 4B.

### The fix
Break draws by **throughput**, not latency:

```
draw_score = min(tokens, CAP) / max(elapsed, 1)     # higher wins
    tokens  = verified completion length (GRAIL)
    elapsed = arrival_round − window_open_round        (drand rounds, 3s each)
    CAP     = 16000                                    (the Part-2 generation cap)
```

### Why throughput is the right ratio
- **Length-neutral.** At equal hardware, 16k-in-32s and 500-in-1s give the *same* tok/s. Long reasoning is **no longer penalized**.
- **Rewards serving efficiency.** Faster hardware / better serving → higher tok/s → wins. This is a legitimate thing to reward.
- **No padding incentive.** Throughput is a *rate*: padding adds tokens **and** time proportionally → score unchanged. And `min(tokens, CAP)` gives **zero** benefit to generating past the useful cap.
- **Incentive shift:** from *"answer fast (therefore short)"* to *"serve efficiently (high tok/s)"* — miners invest in infrastructure, not in brevity.

### Why not just reward length?
Rewarding raw length would re-introduce the rambling/padding failure mode we spent months fighting. Throughput is deliberately **neutral** to length — it *unblocks* long reasoning without *rewarding* token count. (If we later want to gently favor depth, do it only as length **conditioned on correctness**, never raw length — but neutral throughput is the safe default.)

### Robustness
- **Floor the denominator:** `max(elapsed, 1)` (no divide-by-near-zero for instant arrivals).
- **Bucket the score** (e.g. round to 50 tok/s), then **fall back to arrival round** for exact ties → deterministic ordering, no token-level gaming.
- **Value first, always.** The tie-break only orders submissions of **equal value** (same correctness). Throughput never overrides the value/correctness ranking.
- **Trust of `elapsed`:** it is derived from validator-observed arrival vs window-open; forced-seed already prevents pre-generation, so `elapsed` ≈ genuine in-window generation time.

---

## How the three pieces connect

All three follow from one root: **the 4B reasons longer than the 2B.**

- Longer reasoning ⇒ **the 2048 BFT must move up to 16k** (else 45% of thought is cut).
- Longer reasoning ⇒ **the speed tie-break must become throughput** (else long-but-correct miners lose draws and the incentive pushes back toward short).
- Both mechanisms were **calibrated for a short-rambling 2B**; a longer-reasoning model requires re-basing every "penalize slow/long" rule onto **a high cap and a rate metric**, not latency and a low cap.

Ship them together: swapping the model without moving the cap and tie-break would leave the 4B strangled (BFT) and disincentivized (speed) from doing the deep reasoning that justified the swap.

---

## Rollout & kill-switches

1. **Model:** `RELIQUARY_CHECKPOINT = Qwen/Qwen3.5-4B` (fresh HF repo for a true base reset), bump `TRAINING_RUN_ID` (resets cooldown), clear `RESUME_FROM`.
2. **Budget cap:** `BFT_THINKING_BUDGET = 16000` + forced-answer-disable flag → cap-hit = `bad_termination`. Old behavior behind a kill-switch.
3. **Tie-break:** throughput ranking behind a flag (default off until validated on one validator), clean revert to arrival-round ordering.
4. Add the small **termination penalty** on non-concluders after the cap change lands, so termination keeps improving and the cap can be lowered later.

---

## Implementation (branch `feat/4b-migration-bft16k-throughput`)

All changes ship with defaults that preserve current behavior — nothing activates until the flags flip. `412 passed` on the affected unit suites.

| Change | File(s) | Flag / knob (default) | Tests |
|---|---|---|---|
| Budget cap 2048→16000 | `constants.py` `BFT_THINKING_BUDGET` | wire constant `16000` | existing suites use the constant → consistent |
| Clean cap (no forced answer) | `constants.py` `BFT_FORCE_ANSWER`, `miner/engine.py` | `BFT_FORCE_ANSWER = True` (legacy force kept; flip to `False` for clean-cap) | `test_miner_engine_v2`, `test_cheap_rejects_pre_queue` green |
| Throughput draw tie-break | `batch_selection.py` `make_throughput_slot_key`, wired in `batcher.py` | env `RELIQUARY_THROUGHPUT_TIEBREAK` (default off); `THROUGHPUT_TOKEN_CAP=16000`, `THROUGHPUT_BUCKET_TOKENS_PER_ROUND=50` | `test_throughput_tiebreak.py` (6 cases) |
| Model swap | deploy `.env` (`RELIQUARY_CHECKPOINT=Qwen/Qwen3.5-4B`, fresh repo + run id) | ops | — |

**Staging (each row is a separate, revertible step):**
1. Bump `BFT_THINKING_BUDGET` to 16000 with `BFT_FORCE_ANSWER=True` — coordinated miner+validator deploy, contract stays consistent (both import the constant). Fixes the 45%-cut immediately.
2. Turn on `RELIQUARY_THROUGHPUT_TIEBREAK` on one validator (validator-only, no miner change) → validate → roll out.
3. Flip `BFT_FORCE_ANSWER=False` (clean cap) once miners ship it — coordinated + client version bump.
4. Model swap (fresh training run).

**Wiring audit — the throughput feature was a prod no-op on first pass; fixed:**
- ✅ *Difficulty auction owned the slot key* (`DIFFICULTY_AUCTION_ENFORCE` defaults on for openmathinstruct, so the old `elif` never fired for math). Now composed: `slot_round_of = (value_tier, −throughput_bucket, arrival)` — throughput orders draws *within* a value tier; value still dominates.
- ✅ *`ValidSubmission` had no `completion_length`* → added as a property = sum of the group's rollout token counts (the work numerator; `THROUGHPUT_TOKEN_CAP` is now group-scale, `M_ROLLOUTS × 16000`).
- ✅ *`window_open_drand_round` was never populated* → set in `mark_window_opened` from the drand chain (best-effort; None ⇒ throughput cleanly disables, arrival ordering holds).
- Tests: composition (value dominates; throughput breaks within-tier draws) + summed-length property.

**Still open for review / before deploy:**
- **Verify cost (BFT 16k):** a 16k budget makes forced/long rollouts up to ~16.5k tokens, so the GRAIL verify forward runs on ~6.5× longer sequences. Load-test verify latency/memory before deploy (the verifier warns about memory ceilings) — this can gate window cadence or OOM.
- **Clean-cap path untested:** `BFT_FORCE_ANSWER=False` has no test yet, and its premise (an unterminated rollout stays a reward-0 member of the group, not dropped) must be confirmed before flipping the flag.
- **Consensus determinism (minor):** the throughput key uses float division; IEEE-754 is deterministic but consider integer arithmetic if a multi-validator set is ever assumed (currently single-validator).
- Coordinate the BFT wire changes with 0xgrizz (active on the validator — PR #160).

## Appendix — the numbers

**4B behavior (held-out, this study):**

| | MATH (120×8) | CODE (80×8) |
|---|---|---|
| pass@1 (natural) | 0.68 | 0.46 |
| score dist | bimodal (k0 24% / k8 46%) | spread (~55% in k2–6) |
| thinking median / p90 | 3766 / 11297 | — |
| EOS (self-terminate) | 66% | 90% |
| truncated @16384 | 34% | 10% |
| wrong-answer diversity | 0.27 (systematic) | — |
| length correct / wrong | 4428 / 16384(cap) | 965 / 4269 |

**Budget probe (math @32768):** non-finishers 35% of rollouts → 23% budget-limited (62% correct, median 21933) vs **77% never (43% loops)**.

**Model comparison (natural math pass@1 / error type):** 2B 0.57 / random(0.62) · **4B 0.68 / systematic(0.27)** · 32B ~0.70 / systematic(0.26).

*Caveats: 4B numbers are the untrained base model, single held-out sample; code 0.46 is base (the 2B's 0.54 was trained on OCI). Directional, not production-final.*
