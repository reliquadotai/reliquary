# Forced-Seed Sampling + Teacher-Forced Verification

**Date:** 2026-07-09
**Status:** Design approved, ready for implementation plan
**Related:** [[2026-07-09-incentive-training-state-of-situation]] (Problem 1: reward-shape curation)

## 1. Motivation & threat model

The dominant emission-capture mechanism today is **reward-shape curation**: a miner
generates N ≫ 8 rollouts, then cherry-picks the 8 that produce a favorable reward
vector (e.g. exactly k=4 correct → σ ≈ 0.5 → in-zone → paid). The variance is not
*discovered*, it is *manufactured by post-hoc selection*. The validator recomputes
per-rollout reward but cannot see the discarded rollouts, so it cannot tell a
curated group from an honest one.

**Fix.** Force the per-position sampling draw to be a deterministic function of
public, per-window randomness. There is then exactly **one legal generation** per
(miner, prompt, rollout index, window). A rollout that was *not* generated from the
forced draw stream is detectable, so curation collapses: to pass, the miner must
use the forced stream, and if he uses it he cannot cherry-pick.

Detection does **not** require regenerating the rollout (impossible cross-GPU/stack
— see §3). It is a **teacher-forced consistency check** that piggybacks the forward
pass the validator already runs for GRAIL.

This gate composes with the existing token-authenticity gate: token-auth covers the
near-argmax positions (a low-prob token injected at a confident position); the
forced-seed gate covers the **stochastic** positions (did you sample from the forced
stream?).

## 2. Empirical basis (measured on A100, 2026-07-09)

Seven probes on Qwen3.5-2B established feasibility. Key numbers (base model, hard
math prompts, protocol sampling T=0.6/top-k=20/top-p=0.95):

- **Verification needs no reproducibility.** Teacher-forcing conditions each
  position on the miner's submitted prefix → no autoregressive cascade. Same-stack
  decode-vs-decode match is 100%; prefill-vs-decode ~95%.
- **Signal lives in stochastic positions.** ~75-82% of tokens are effectively argmax
  (the forced draw doesn't change them). Restricting to `max_prob < 0.99` positions
  (~26% of tokens on hard prompts):
  - honest (used the forced stream): **~96%** match
  - non-forced ("cheater", own RNG): **~60%** match (chance collisions)
- **Cross-stack does not collapse the honest floor.** Miner=vLLM → validator=HF:
  honest floor **95.9%** on stochastic positions — identical to HF's own
  prefill/decode floor (95.9%). vLLM and HF compute close-enough logits; the engine
  swap adds essentially nothing beyond the unavoidable prefill/decode noise.
- **Per-rollout separation is overwhelming** on base: ~264 stochastic positions per
  1024-token rollout, honest ~96% vs cheater ~60% ⇒ several σ per single rollout,
  crushing per group of 8. Tolerates the honest floor dropping to ~65% before a
  single rollout becomes marginal.
- **Known limitation — entropy collapse.** RL sharpens the policy; a *healthy*
  trained policy has fewer stochastic positions than base (base 25.8%; the current
  broken checkpoint measured 2.5% on in-distribution and is discarded as an
  artifact; a healthy operating point is unknown, between the two). The gate is
  strongest early in training and on hard prompts; the `MIN_STOCH_POSITIONS`
  safety valve (§4.7) makes it *abstain* rather than false-reject when signal is
  thin.

## 3. Seed derivation (miner and validator agree)

Nothing is distributed to miners. The per-position uniform is **derived** from
values the miner already has:

```
fields = [ drand_randomness,          # /state.randomness, per-window, unforgeable
           hotkey_ss58,               # the miner's own address
           prompt_idx,                # the chosen prompt
           checkpoint_hash,           # current published revision (binds to model)
           rollout_index i,           # 0..7
           token_position t ]

msg = b"reliquary-forced-seed-v1" ++ length_prefixed_concat(fields)
u_{i,t} = int.from_bytes(SHA256(msg)[:8], "big") / 2**64      # in [0, 1)
```

Fields are **length-prefixed** (2-byte big-endian length before each variable field)
to avoid delimiter ambiguity. Because `drand_randomness` is only revealed at window
open, the miner cannot pre-compute or grind seeds.

## 4. Design

### 4.1 Shared sampler module (single source of truth)

A new module (e.g. `reliquary/environment/forced_sampling.py`) exports the exact
warp + inverse-CDF used by **both** the miner reference client and the validator.
This guarantees they agree at the algorithm level. Fixed algorithm:

```
def warp(logits):                       # logits fp32
    lg = logits / T_PROTO               # 0.6
    keep top TOP_K_PROTO (20) by logit; others -> -inf
    probs = softmax(lg)
    top-p (0.95): sort desc, keep while (cumsum - p) < TOP_P_PROTO   # include crossing token
                  zero the rest, renormalize
    return probs                        # canonical order = ascending token id

def pick(probs, u):                     # inverse-CDF
    cdf = cumsum(probs)                 # over ascending token id
    return first token id with cdf > u
```

### 4.2 Miner reference sampler

At each decode step for rollout `i`, position `t`: compute `warp(logits)`, then
`pick(probs, u_{i,t})` instead of drawing from a local RNG. Ship in the reference
miner client. Add `protocol_version` to `BatchSubmissionRequest` (miners on the new
client advertise forced-seed support — used for forensics/observability).

### 4.3 Validator consistency gate

For a submission (8 rollouts):

1. **Teacher-force** each rollout's tokens through the forward pass already run for
   GRAIL — obtain per-position logits (the token-auth path already recomputes these).
2. `warp` each position → distribution.
3. **Stochastic positions** = those with `max_prob < STOCHASTIC_MAXPROB_THRESHOLD`.
   Exclude BFT-injected `force_span` tokens (already masked from the loss — reuse
   `_completion_keep_list`).
4. For each stochastic position, check `pick(dist, u_{i,t}) == submitted_token`.
5. **Aggregate over the whole group** (all 8 rollouts): `score = matches / n_stoch`.
6. `reject(RejectReason.FORCED_SEED_MISMATCH)` iff
   `score < CONSISTENCY_FLOOR` **and** `n_stoch >= MIN_STOCH_POSITIONS`.

Runs after GRAIL/token-auth in the verification pipeline (reuses their forward
pass). Env-gated to the active environments.

### 4.4 Aggregation: per-group (chosen)

Score aggregated across all 8 rollouts of the submission. Rationale: the target
attack (curation) makes **all** submitted rollouts non-forced, so per-group
aggregation maximizes statistical power and minimizes false positives (one decision
per submission, not eight). *Alternative considered:* per-rollout strict (reject if
any single rollout fails) — catches partial cheating (7 honest + 1 swapped) but has
8× the false-positive surface. Deferred to a hardening pass if partial cheating is
observed.

### 4.5 Enforcement & cutover (announced hard flag-day)

- `FORCED_SEED_ENFORCE_FROM_WINDOW` constant, default sentinel `2**63 - 1` (never),
  same pattern as `PROMPT_RANGE_ENFORCE_FROM_WINDOW`.
- **Before** the window: **shadow** — compute `score` and write it to
  `auth_forensics`, never reject. Lets the operator watch the honest floor climb as
  miners adopt the new client, and pick the flag-day once adoption looks complete.
- **At/after** the window: **enforce** (hard reject). The operator announces the
  window to miners in advance, then sets the constant. This is the approved
  "pre-announced hard cutover".

### 4.6 Parameters

| param | default | basis |
|---|---|---|
| `STOCHASTIC_MAXPROB_THRESHOLD` | 0.99 | probes: signal lives below this |
| `CONSISTENCY_FLOOR` | 0.80 | between honest ~92-96% and cheater ~60% |
| `MIN_STOCH_POSITIONS` (per group) | 30 | false-positive safety valve |
| `FORCED_SEED_ENFORCE_FROM_WINDOW` | 2**63−1 | announced flag-day |
| env gating | both | — |

### 4.7 Safety valve & honest limitation

If a group has `n_stoch < MIN_STOCH_POSITIONS` (very peaked model / over-trained
policy / very short rollouts), the gate **abstains** (accepts) rather than
false-rejecting on thin signal. Consequence, stated honestly: on an ultra-peaked
policy the gate **detects less** (a cheater can slip through) but **never breaks an
honest miner**. This is the direct translation of the entropy-collapse finding in
§2. The gate is strongest early in training and on hard prompts; token-auth still
covers the argmax positions independently.

## 5. Files touched

- `reliquary/constants.py` — params + `FORCED_SEED_ENFORCE_FROM_WINDOW` + seed domain string.
- `reliquary/environment/forced_sampling.py` — **new**, shared warp + inverse-CDF + `u_at` seed derivation.
- `reliquary/protocol/submission.py` — `protocol_version` field.
- miner reference client — use the shared sampler in the decode loop.
- `reliquary/validator/batcher.py` — call the gate; `RejectReason.FORCED_SEED_MISMATCH`.
- `reliquary/validator/auth_forensics.py` — shadow log of the consistency score.
- tests — see §6.

## 6. Testing

- **Seed derivation determinism**: `u_at` is stable across processes; changes with
  each field; length-prefixing prevents field-boundary collisions.
- **Sampler ↔ verifier agreement**: a sequence generated with the shared sampler,
  teacher-forced back through the same warp, matches 100% same-stack (no drift).
- **Honest vs cheater separation** on fixtures: forced-stream tokens score ~1.0 on
  stochastic positions; a different-seed stream scores near the collision baseline;
  the gate accepts the former and rejects the latter given `n_stoch >= MIN_STOCH`.
- **Abstention**: a group with `n_stoch < MIN_STOCH` is accepted regardless of score.
- **BFT exclusion**: force_span positions are not counted.
- **Shadow vs enforce**: below the window, a failing group is accepted and logged;
  at/after, it is rejected.
- GPU probes (`seed_probe*.py`, `gen_hf.py`, `vllm_dist.py`) stand as integration
  evidence and are archived on the experiment box.

## 7. Alternatives considered

- **Spot-check by re-generation** — rejected: relies on reproducibility (broken
  cross-GPU/stack), and expensive.
- **Private/generated math + commit-reveal** — a complementary Problem-1 fix that
  needs no sampling entropy, but heavier data infrastructure; forced-seed chosen
  first.
- **Clip-higher / raising T** to increase entropy — rejected: `T_PROTO` is fixed for
  reward-variance reasons.

## 8. Rollout checklist

1. Land shared sampler + gate in **shadow** (`FORCED_SEED_ENFORCE_FROM_WINDOW` =
   sentinel), ship reference client with the sampler + `protocol_version`.
2. Announce the flag-day window to miners.
3. Watch the shadow honest-floor in forensics climb as adoption proceeds.
4. When adoption is complete, set `FORCED_SEED_ENFORCE_FROM_WINDOW` to the announced
   window. Enforcement flips on.

## 9. Open questions / future work

- Calibrate `CONSISTENCY_FLOOR` / `MIN_STOCH_POSITIONS` on the live shadow floor
  before the flag-day (the ~0.80 / 30 defaults are from base-model probes).
- Confirm the cross-GPU (different architecture) honest floor once a second GPU tier
  is available — expected fine given the wide margin, untested.
- Per-rollout hardening if partial cheating (mostly-honest groups) appears.
