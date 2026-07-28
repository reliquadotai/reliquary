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

### Token budget — one uniform ~16k ceiling (final)

| tier | value | role |
|---|---|---|
| BFT thinking budget | `BFT_THINKING_BUDGET = 15616` | math thinking cap (BFT forces here) |
| answer budget | `BFT_ANSWER_BUDGET = 512` | the forced answer phase |
| **global cap** | `MAX_NEW_TOKENS_PROTOCOL_CAP = 16384` | generation ceiling, **both envs** |

Sized so a forced completion (thinking + force-span + answer = 16128 + template) fits under 16384 with 256 of margin — asserted in `constants.py`.

**Per-env truncation allowance** (`MAX_TRUNCATED_PER_SUBMISSION_BY_ENV`): math **1**, code **3**.
- *Math stays 1* because BFT terminates every rollout through an accepted path (EOS / forced cap / natural cap), so a truncated math rollout means a non-compliant client. It is **not** set to 0: a zero-tolerance gate has burned honest miners here before (`drand_tolerance=0` rejecting honest `stale_round`), and the "one free truncation" is no longer an exploit now that manufacturing is priced out — so the buffer is free.
- *Code gets 3* because it has no BFT and the 4B loops on ~10% of code rollouts: at 8 rollouts, ~19% of **honest** code groups carry 2+ truncations and were being rejected wholesale. They are now admitted and discounted conservatively instead.

Lowering the global cap from 32768 only became safe once truncated rollouts stopped being rejected outright — the earlier attempt was reverted for exactly that reason.

**Why BFT is kept for math (and not removed):** without BFT the 4B truncates ~35% of math rollouts, so an honest math group carries ~3 truncations and conservative valuation would discount **every** group by ~64% (measured), plus 83% of honest math groups would exceed any sane truncation allowance. BFT converts "unknown" into "known" (a scored answer), which is precisely what the conservative rule penalises — so the two mechanisms are **complementary**: BFT removes the unknown where it can be forced (math), conservative valuation prices the unknown where it cannot (code).

### (superseded) Earlier reasoning: the two envs are deliberately DIFFERENT
It is tempting to shrink `MAX_NEW_TOKENS_PROTOCOL_CAP` (32768) toward the BFT budget so the number "looks aligned". **Don't** — it strands honest code miners for no math benefit:

- **Math** is already capped at 16000 by BFT (`min(max_new_tokens, BFT_THINKING_BUDGET)`), independent of the global cap. So lowering the global cap does nothing for math.
- **Code has no BFT** to force termination. A code rollout that hasn't emitted EOS by the global cap counts as *truncated*, and `MAX_TRUNCATED_PER_SUBMISSION = 1` **rejects the whole submission when more than one rollout is truncated** (`admission.py::_termination_reject`). Lowering the cap makes more code rollouts truncate → more honest submissions rejected (at the 4B's ~10% code-truncation rate, ≥2-of-8 lands ~19% of groups in the reject bucket).

So the caps are legitimately split: **math 16k (BFT), code up to the 32768 global cap.** Keep the global cap high so honest code has room to terminate. `MAX_NEW_TOKENS_PROTOCOL_CAP > BFT_THINKING_BUDGET + BFT_ANSWER_BUDGET` is asserted in `constants.py` (a forced completion is thinking + force_span + answer ≈ 16.5k, so the cap must exceed thinking+answer).

**Separate, real problem this surfaced (needs a decision):** even at 32768, a 4B that loops on code (~8% never terminate) will land ~1-in-8 groups with ≥2 truncated rollouts → rejected by `MAX_TRUNCATED_PER_SUBMISSION = 1`. That check was tuned against manufactured losers on the 2B; it is strict for an honestly-rambling 4B. Options: raise the per-submission truncated allowance for code, or a check that separates honest loops from manufactured ones. Do NOT just raise it blindly — it guards the reward-shape exploit.

---

## Part 2b — Forced rollouts: mask from the loss (DAPO overlong filtering, adapted)

**The bug the first run exposed.** BFT taught the 2B to *shorten* its math reasoning. Mechanism: hitting the budget forces a (usually wrong) answer → negative group-relative advantage → the policy gradient learns *"generating long leads to a penalty, so don't"*. That is a length bias with nothing to do with correctness.

**DAPO's fix, verified.** DAPO ([arXiv 2503.14476](https://arxiv.org/abs/2503.14476), *Overlong Reward Shaping*) hits the analogous case with truncated samples and **"masks the loss of truncated samples"** — they get no policy gradient — while the advantage `Â = (R − mean{R})/std{R}` (Eq. 9) is still computed **over the full group**, i.e. the masked sample stays in the mean/std. So DAPO masks the *loss*, not the *baseline*.

**Why we can't copy it verbatim.** DAPO has one layer (training). We have two: training **and** an economic layer (σ-gate / auction / emission). If we simply let rollouts truncate and drop them (the naive port), a miner can **suppress EOS to route a would-be-wrong rollout into truncation and dodge the reward-0**, and the σ-gate/emission math breaks. So we keep BFT (a forced answer → the economic layer scores every rollout, un-gameable) and apply DAPO's masking to the **forced** rollouts:

> `BFT_MASK_FORCED_FROM_LOSS` (default off, validator-only training control): a forced rollout's advantage is zeroed — **no policy gradient** — while it stays in `_compute_advantages`'s group mean/std (baseline unchanged, exactly DAPO Eq. 9) **and** in the σ-gate/auction/emission (economic layer still scores it). A masked rollout is also **skipped from the training forward** (`_build_microbatch_items`) and from the `N_e` denominator (`_plan_from_batches`) — it contributes no PPO or KL gradient, so materialising its (up to 16k) sequence is pure waste. This completes the DAPO filtering (whole loss masked, incl. KL) and avoids materialising a forward that owes no gradient (commit `1412a8b`). *Measured 2026-07-28:* the skip cuts rollouts forwarded 16 → 10, i.e. it saves compute — but **not** peak memory, which is set by the longest single sequence in a micro-batch. The real `train_step` at the 16k cap peaks at **~40 GB of 80** in every scenario tested, so there was never a memory blocker; see `overnight-4b-test-verdict-2026-07-28.md`.

This settles the internal debate (does masking change the mean?): **no — the baseline is over the full group; only the gradient is masked.** It also removes the forced-guess *pollution* we measured earlier (~75% of production reward variance was the forced coin-flip) from the gradient, while keeping it in the scoring. Implemented in `training.py::_shape_advantages`; the existing `test_rollout_loss_zero_advantage_gives_zero_ppo_loss` confirms advantage 0 ⇒ zero PPO loss. Pairs with the 16k budget (Part 2): the model reasons up to 16k, forced beyond that, and the forced tail no longer biases length.

---

## Part 2c — Manufactured zeros: price truncation instead of policing it

**A pre-existing hole, found while testing the 16k change.** The auction pays more for hard prompts (value peaks at low k), so a miner can inflate a prompt's value by **breaking one of its own correct rollouts**: suppress EOS → it runs to `max_tokens` → grades 0 → the prompt looks harder. On the live value function `std(r)·(1−mean(r))`, **one manufactured zero at k=6 is worth +68%** (0.1083 → 0.1816). This exists today, independent of the 4B/16k work, and `MAX_TRUNCATED_PER_SUBMISSION = 1` already permits exactly one.

**Detection was tried and rejected (GPU-measured).** Simulating a cheating miner (ban EOS, run to the cap) through every content filter:
- forced-seed consistency: **12/12 cheaters pass** (~97% agreement vs a 0.75 floor — the manipulation's 2-4% footprint is an order of magnitude under the HW tolerance);
- token distribution: 12/12 pass;
- token-auth (`chosen<1e-5 & argmax≥0.99`): catches 9/12 — misses *soft* terminations;
- a purpose-built **EOS-suppression filter** (count positions where the forced-seed pick is EOS but the token isn't) and an **EOS-survival test** (`S = Σ p_eos`, aggregate and HW-robust, 0% false positives): catch 78% of real fake attempts, miss the rest.

Decisive objection: **any detector is farmable** — a miner retries prompts until one evades, so residual detection failure converges to "always evades". Detection cannot be the primary defence here, and the underlying cause (forced-seed's wide HW tolerance) is not fixable without a HW-invariant fingerprint.

**The fix — conservative valuation.** A truncated rollout has no gradeable answer, and the validator cannot know whether it would have been correct. So score the group under the interpretation **least favourable to the miner**:

```
value = min over j∈[0..t] of  difficulty_score(rewards with j of the t truncated raised to 1.0)
```

Because the group's true outcome is always one of the interpretations and we take the minimum, **a manipulated group can never score above the honest group it came from — the gain is exactly zero, by construction**, for any value function, with no threshold to tune and no detection. Verified exhaustively over every `(k, truncated-correct, truncated-wrong)` combination: **0 profitable cases out of 156**.

*Why not simply exclude truncated rollouts from the ratio (`c/n_terminated`)?* Because that shifts the denominator: at k=1, truncating a **wrong** rollout moves 1/8 → 1/7 and gains +3.6%, and an honest rollout that rambles is punished far harder (−54% before any discount) since removing a genuine failure makes the prompt look easier. Keep the denominator at M and take the min.

**Cost.** An honestly-rambling group is also valued down — worst at high k (k=6: −62%, but those easy prompts were already worth ~3× less than the peak and don't win auctions) and mildest exactly where the auction competes (k=2: **−7%**, k=1: **0%**). The 4B rambles most on prompts it cannot solve, i.e. the low-k region where the penalty is smallest. And it is **strictly better than today for the worst-hit honest miners**: a group with 2+ truncations is currently rejected outright (**earns zero**) — under conservative valuation it is admissible and earns a real value, which is what lets `MAX_TRUNCATED_PER_SUBMISSION` be relaxed for code (blocker 1).

**Implementation.** `conservative_difficulty_score()` / `auction_value()` in `difficulty_auction.py`; `count_truncated_rollouts()` in `admission.py` (sharing `_classify_termination` with the admission gate, so both read one predicate); `truncated_count` threaded `PreparedSubmission → PendingSubmission →` the auction value and `_prove_ranked` ranking. Behind `RELIQUARY_CONSERVATIVE_TRUNCATION_VALUE` (**default off**). Auction/emission only — training keeps the real reward vector. 11 tests incl. the exhaustive proof; 466 passed on affected suites.

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
- **`MAX_TRUNCATED_PER_SUBMISSION = 1` vs an honestly-rambling 4B (code):** a code group with ≥2 rollouts that hit the cap without EOS is rejected wholesale. The 4B loops on ~8-10% of code, so ~1-in-8 honest code groups get rejected even at the 32768 cap. The check guards the manufactured-loser exploit, so don't just raise it — decide between a higher code allowance or a smarter honest-loop-vs-manufactured check. (Kept the global cap at 32768 to not make this worse.)
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
