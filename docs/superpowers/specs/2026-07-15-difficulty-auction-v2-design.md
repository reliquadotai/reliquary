# Difficulty Auction v2 — Design

**Date:** 2026-07-15
**Branch:** `design/difficulty-auction-v2` (off `design/difficulty-auction` @ c1ce152,
which carries current `main` hardening incl. the `force_span` fix, the
observation-only shadow module `difficulty_auction.py`, and operator attribution)
**Builds on:** `2026-07-14-difficulty-auction-design.md` (the mechanism) and the
protocol plan `2026-07-14-difficulty-auction-protocol.md` (deferred proof + deadline)
**Answers:** `2026-07-15-difficulty-auction-maintainer-review.md` (the P0/P1 blockers)

---

## 0. Why a v2

The v1 protocol branch (deferred proof + 300s deadline + prompt dedup) was
correct in shape but the maintainer review blocked it for deployment with real
P0s: multi-hotkey variance farming, prompt squatting, an unbounded proof
wall-clock, a predictable forensic sample, a wire-breaking reject enum, and code
included prematurely. v2 keeps the mechanism (pay for hard groups, time-boxed
collection, prove only what can win) and closes each blocker. The mechanism math
(the score, the ranking) is **unchanged** and is reused from the merged shadow
module.

**Scope stays math-only** (`openmathinstruct`). `opencodeinstruct` is out until a
separate grader-throughput canary passes.

---

## 1. What is carried over unchanged (from v1 + shadow)

- **Score** `v(k) = std(rewards) · (1 − mean)^δ`, `δ = 1.0`, peak at k=2. Reuse
  `reliquary/validator/difficulty_auction.difficulty_score` (the shadow module
  already on the branch) — do NOT fork a second copy.
- **Ranking** `(-value, drand_round, _within_slot_key)` — reuse the module's
  `_rank_key`.
- **Time-boxed window** at `WINDOW_COLLECTION_SECONDS = 300`, seal on the
  deadline (not on an 8-distinct count). Accept everything valid during the
  window. Sized from live data (math gen median 176s, p75 267s; windows already
  ran ~277s collection).
- **Deferred proof**: cheap admission (schema/sig/prompt/grade/score, no GPU),
  then prove the ranked candidates at seal. A submission that cannot win is never
  proven.
- **Operator attribution** (from the friend's commits) — the hotkey→operator map
  used by the emission cap. Kept.

---

## 2. The v2 changes (one per maintainer P0/P1)

### 2.1 Variance farming → remove the hotkey from the forced seed (P0 #3)

**The fix chosen over the maintainer's coldkey seed-binding.** Today
`u_at(randomness, hotkey, prompt_idx, checkpoint_hash, rollout, t)` keys the
forced-sampling stream on the **hotkey**. That is exactly what lets one operator
with N hotkeys draw N *different* legal groups on one prompt and submit the best
(8 hotkeys → 60% chance of hitting k=2, per the review's binomial table).

Drop the hotkey from the hash:

```
u_at(randomness, prompt_idx, checkpoint_hash, rollout, t)   # no hotkey
```

Now the forced group for a prompt is identical for everyone in the window, so N
hotkeys buy N copies of the *same* draw — farming is dead.

**Why this over coldkey-binding:** coldkey-binding needs a block-pinned metagraph
snapshot synchronized across validators, fail-closed on missing mappings — a real
consensus hazard the review itself flags. Removing the hotkey has **no identity
in the seed at all**, so it needs no metagraph sync. Simpler and consensus-safer.

**What it costs, and why it is acceptable:**
- The group becomes a **public deterministic function** of (window randomness,
  checkpoint, prompt). Anti-pregeneration still holds — randomness is unknown
  until the window opens. To submit you still must run the model (GRAIL checks
  real forward-pass tokens at 32 challenge positions), and mid-window nobody's
  submission is visible, so there is no free-riding/copying.
- Two honest miners on one prompt now produce **byte-identical tokens** →
  `compute_rollout_hash(tokens)` collides → the content dedup makes it
  one-submission-per-prompt automatically, resolved by who submits first. This
  interacts with §2.2 and must be handled there (see the dedup note).
- The competition becomes "cover the most prompts, submit the high-k ones fast"
  rather than "curate a group" — a compute/coverage race. It re-centralizes
  toward GPU scale (as the current race already does), NOT toward hotkey count.

**Coordinated change:** both sides must hash identically —
`reliquary/miner/forced_seed_sampler.py:85` and
`reliquary/validator/batcher.py:1400`, plus the `u_at` signature in
`reliquary/environment/forced_sampling.py:60`. This is a miner-client protocol
change with an adoption window (bump the forced-seed protocol version).

### 2.2 Prompt squatting → resolve same-prompt at seal, after proof (P0 #4)

Do **not** claim a prompt at admission, and do **not** add a wire-level
`PROMPT_CLAIMED` enum (older miners fail deserialization on an unknown value —
review P0). Instead:

- Admit and grade every submission (bounded, §2.3). Multiple submissions for one
  prompt may enter the pending pool.
- At seal, rank; when filling slots, the **first submission per prompt that
  PASSES the proof** takes the slot; the rest for that prompt are dropped
  (promote-on-failure). A fabricated group fails the proof, so it can never lock
  a prompt.

**Dedup note (interaction with §2.1):** under the hotkey-free seed, two *honest*
submissions for a prompt are byte-identical, so the content dedup would reject
the second at admission as `HASH_DUPLICATE` — which already yields
one-honest-submission-per-prompt without a new enum. A *fabricated* squatting
group has different (garbage) tokens, so it does NOT collide and still enters the
pool; it is removed at seal when it fails the proof. So squatting resolution =
existing content dedup (for honest twins) + seal-time proof filter (for fakes).
No new reject reason on the wire.

### 2.3 Unbounded proof wall-clock → global budget + wall-clock (P0 #5)

v1 removed the fixed proof-attempt cap (which had starved honest fill). v2 keeps
honest fill safe AND bounds GPU: prove ranked top-down until `B_BATCH` pass, with
**both** a global attempt ceiling (tie it to `MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW`,
the pool bound) **and** a global proof wall-clock budget. On exhaustion: explicit
fallback — archive the shortfall, burn unpaid slots, advance. The per-hotkey
failure skip stays. (The earlier starvation came from a cap SMALLER than the fake
population; here the ceiling is the pool itself, so fakes cannot exhaust it before
honest candidates are reached, and the wall-clock stops a pathological flood.)

### 2.4 Queue memory reserved too late (P1 #6)

Reserve grading count AND payload bytes atomically **before** queue insertion,
release on every cancel/drop path. Bound count, serialized bytes, per-hotkey
bytes, per-env sandbox work. Closes the 256-payload backpressure hole.

### 2.5 Predictable forensic sample (P1 #7)

v1 sampled non-winners by `sha256(window_randomness ‖ miner_merkle_root)` — the
miner controls the root and sees the randomness, so it is grindable. Use **future
drand entropy revealed only after the collection deadline** to choose the sample,
then prove it before publication. Miner cannot predict whether it is watched.

### 2.6 Selection changes production — state it plainly (P0 #1)

There is no "neutral shadow" once proving is ranked: proving only the top-ranked
fills `_valid` with auction-selected winners, which is the point. v2 does not
label this shadow. It is the armed selection. Emission still uses the existing
distributor over `_valid`; paying *by score* is a later, separate switch, gated
on the free-negative guard (§3).

---

## 3. Still required before ANY payout/training-selection activation

Carried from the original design §9 and the review:

- **Grader false-negative guard.** A correct answer graded wrong lowers k toward
  the score peak → a false negative is worth the maximum payout. Re-grade the
  selected top-8 with the unbounded oracle (the fast path's DoS bounds are for 69
  submissions; the auction keeps 8, so it can afford it). Measure the residual
  free-negative rate first (M5).
- **Operator emission cap.** `MAX_SLOTS_PER_COLDKEY_PER_WINDOW`, wired to the
  operator attribution, bounds centralization (a farmer cartel filling all 8
  slots). The seed change (§2.1) removes the score-inflation incentive; the cap
  bounds residual concentration.

---

## 4. Rollout

1. Land v2 mechanics behind the existing enforcement flag OFF (compute + archive,
   change nothing), rebased on current main.
2. Run a **full-pool, non-weight-setting canary** with the bounded resource
   controls above — this is the measurement the pure shadow cannot do, because it
   collects the real 300s population instead of the speed-race survivors.
3. Compare optimizer-side difficulty balancing vs the market auction on a fixed
   validated dataset (review step 3).
4. Close M5 + the seed protocol adoption window, then arm.

**Reconcile with the maintainer before merge** — this branch answers their review
rather than bypassing it, so it should go back through the same review.

---

## 5. Open question for the maintainer

The hotkey-free seed (§2.1) makes the per-prompt group a public deterministic
artifact. This is a deliberate trade: it kills variance farming without a
metagraph-synchronized operator map, at the cost of a compute/coverage race and
losing per-hotkey token uniqueness. If per-operator draw diversity is considered
valuable (each operator exploring a different draw), coldkey-binding is the
alternative — heavier consensus, but preserves per-operator uniqueness. v2 picks
the simpler, consensus-safer option; flag if the trade is wrong.
