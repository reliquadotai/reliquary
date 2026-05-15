# Drand-anchored ordering + multi-miner-per-prompt emission split

## Problem

Two coupled issues in the current free-prompt GRPO market:

1. **Latency exploit.** `batch_selection.py:53` sorts valid submissions by
   `(arrived_at, tiebreak_hash)` — pure TCP-arrival FIFO. A miner co-located
   with the validator (same datacenter, low RTT) wins every prompt race.
   Geography decides the batch, not inference quality. Reverted to FIFO in
   v2.2 after the v1 `signed_round` (drand-anchored) ordering was found
   grindable: the miner embedded the round into the submission and could
   choose an already-published round, then grind their `merkle_root`
   against `H(σ_R || ...)` until favorable.

2. **Substrate dependency stalls window OPEN.** `service.py:347` (commit
   `db40280`) defers the window OPEN until `_set_window_randomness`
   succeeds, which calls `chain.get_block_hash` over WebSocket. When the
   substrate endpoint hiccups (finney WebSocket 503, transient network),
   `_set_window_randomness` retries (commit `ebeec31`) and the window
   stays closed for tens of seconds. Miners can't submit during this gap.

A separate but related observation: the current "single winner per
prompt, single batch slot per winner" model means emission concentrates
on whoever wins the FIFO race. A miner who submitted a perfectly valid,
honest, GRAIL-passing rollout but arrived second receives nothing.

## Goal

1. Make submission arrival time irrelevant for both batch selection AND
   emission.
2. Eliminate the per-window block_hash fetch from the critical path so
   substrate flakiness no longer stalls window OPEN.
3. Distribute emission across all miners who produced a GRAIL-validated
   submission in the window, with sybil-resistance: spawning multiple
   hotkeys to attack the same prompt must not increase total payout.

## Non-goals

- Keep the GRPO training math unchanged (still B=8 prompts per training
  step, still picked from validated submissions).
- Keep the GRAIL primitive and verification pipeline unchanged.
- No change to the wire protocol fields visible to miners (drand round
  selection happens entirely on the validator side; miners submit as
  before).

## Approach

Three independent changes, designed to ship together:

### A. Drop block_hash from the per-window randomness

`chain.compute_window_randomness(block_hash, drand_randomness, drand_round)`
currently builds the seed as `H(block_hash || drand_randomness || drand_round)`.
Drop `block_hash`. The seed becomes `H(drand_randomness || drand_round)`.

Rationale for the security tradeoff:
- Drand quicknet (3 s, BLS12-381 threshold signing, ~15 League of Entropy
  nodes) is the strong cryptographic source. Threshold compromise is
  effectively a state-actor attack; this is the unpredictability source.
- `block_hash` was protecting against substrate validator grinding of the
  block hash and providing canonical chain anchoring across validators.
  Today the subnet has a single validator in prod (`86.38.238.30`), so
  the multi-validator anchoring property is unused. The grinding attack
  exists only against block_hash; drand alone is not exploitable that way.
- When the subnet eventually adds validators, re-introduce the mix or
  derive the seed from a substrate-anchored but non-grinding source
  (e.g., a deterministic round counter `window_n * K` for an offset K).
  Out of scope here; tracked as a follow-up.

Effect: window OPEN no longer depends on `chain.get_block_hash` returning
in time. Substrate flakiness affects weight submission and window
scheduling but not the per-window randomness used by GRAIL.

### B. Drand-anchored ordering computed post-window-close

Introduce a second drand-derived seed, distinct from the GRAIL one:

- **GRAIL seed** (existing) — round R₁ with `round_time(R₁) ≤ window_open`.
  Public to miners during the window (`/state` exposes it). Used inside
  `verify_commitment_proofs` and `verify_logprobs_claim`.

- **Ordering seed** (new) — round R₂ with `round_time(R₂) > window_close + δ`,
  where δ ≈ 1 s margin. Fetched after the window closes. Not exposed to
  miners during the window. Used to order winning prompts.

R₂ is chosen deterministically from `window_close_ts` so two validators
agree (when the subnet eventually has more than one). Concretely:

```python
ordering_round = compute_drand_round_for_window(
    window_close_ts + 1,  # δ = 1 s past close
    chain_info["genesis_time"],
    chain_info["period"],
)
```

The ordering seed is `H(drand_randomness_R₂ || R₂)`. Verified locally with
the cached drand pubkey + bittensor_drand cross-check (`drand.py:115`),
same path the GRAIL seed already uses.

### C. Multi-miner-per-prompt with split emission

Drop the SUPERSEDED short-circuit at `batcher.py:247`. Accept multiple
submissions for the same `prompt_idx` within a window, subject to a hard
cap.

**New constant:** `MAX_SUBMISSIONS_PER_PROMPT = 10` in `constants.py`.

**New reject reason:** `RejectReason.PROMPT_FULL` returned when a prompt
has already received `MAX_SUBMISSIONS_PER_PROMPT` accepted submissions.
The cap is enforced BEFORE the heavy GRAIL verify, so a saturated prompt
can't be used to DoS the validator's GPU.

**Per-hotkey cap unchanged.** `_per_window_counts` (#18) keeps capping at
B_BATCH = 8 submissions per hotkey per window. The new per-prompt cap is
orthogonal.

**Batch selection at seal time:**

```
1. Group all GRAIL-validated submissions by prompt_idx
   → M distinct prompts, each with K_p submissions (1 ≤ K_p ≤ MAX_SUBMISSIONS_PER_PROMPT)

2. Order the M prompts by H(ordering_seed || prompt_idx) ascending
   → drand-determined prompt ordering, no miner influence

3. winning_prompts = top min(M, 8) prompts

4. For each winning prompt p, pick ONE submission for the training step:
   - Sort submissions_for_p by (hotkey, merkle_root) to canonicalize order
     across validators (otherwise random.choice depends on accept order,
     which is network-dependent).
   - training_pick(p) = random.Random(
         H(ordering_seed || prompt_idx || "train")
     ).choice(sorted_submissions_for_p)
   - Drand-seeded, canonical, no miner influence.

5. Training batch = [training_pick(p) for p in winning_prompts]   (size 8 or less if M < 8)
```

### D. Emission formula

For each miner `m` with a GRAIL-validated submission on prompt `p`:

```
if p in winning_prompts:
    reward(m, p) = pool / |winning_prompts| / K_p
else:
    reward(m, p) = 0

reward(m) = sum over all (m, p) pairs where p in winning_prompts
```

This satisfies the design invariants:

- **Same-prompt sybil neutral.** Attacker with N sybils all on prompt p
  splits `pool/|winning_prompts|/N` per sybil, totaling `pool/|winning_prompts|`
  — identical to a single hotkey winning that prompt alone, minus N-1
  registration costs.
- **Different-prompt sybil tax.** Sybil on different prompts gains
  per-prompt payouts, but bounded by Bittensor registration burn.
- **Latency neutral.** All terms in the reward formula depend on
  `ordering_seed` (unknown at submission time) and on K_p (set
  post-window), neither of which arrival timing can influence.
- **Bounded validator compute.** Training step always runs B=8 forward
  passes (or fewer if M < 8). The increase in accepted submissions only
  affects ingress + GRAIL verify cost during the window, capped by
  `MAX_SUBMISSIONS_PER_PROMPT` × len(env) and by `_per_window_counts`.

## Changes

### `reliquary/constants.py`

Add `MAX_SUBMISSIONS_PER_PROMPT = 10`.

### `reliquary/protocol/submission.py`

Add `RejectReason.PROMPT_FULL = "prompt_full"`.

### `reliquary/infrastructure/chain.py`

- `compute_window_randomness` — make `block_hash` parameter optional;
  when None, derive seed from `drand_randomness || drand_round` only.
- Add helper `compute_drand_round_for_ordering(window_close_ts, ...)` or
  reuse `compute_drand_round_for_window` with `window_close_ts + 1`.

### `reliquary/validator/service.py`

- `_derive_randomness` — drop the `block_hash` call (drand-only path).
  Existing `use_drand` flag becomes effectively always-true; keep the flag
  for tests.
- `_set_window_randomness` — unchanged externally (same field set on
  batcher), just no substrate dependency.
- New `_derive_ordering_seed(window_close_ts)` — same pattern, different
  round.
- Window close flow — before calling `seal_batch`, fetch ordering seed
  and pass it through.

### `reliquary/validator/batcher.py`

- `_accept_locked` — remove the `prompt_idx in self._claimed_prompts →
  SUPERSEDED` block. Replace with `len(self._submissions_per_prompt[pi])
  ≥ MAX_SUBMISSIONS_PER_PROMPT → PROMPT_FULL`.
- Track `_submissions_per_prompt: dict[int, list[ValidSubmission]]`
  alongside the existing flat `_valid` list.
- Drop `_claimed_prompts` set.
- `seal_batch` — accept an `ordering_seed: bytes` argument. Group by
  prompt, drand-order the prompts, pick top 8, choose training submission
  per prompt with drand-seeded random. Return both the training batch
  AND the full reward distribution (a `dict[hotkey, float]` or similar).

### `reliquary/validator/batch_selection.py`

Replace `select_batch` entirely. New signature:

```python
def select_batch_and_distribute(
    submissions_by_prompt: dict[int, list[ValidSubmission]],
    *,
    b: int,
    ordering_seed: bytes,
    pool: float,
    cooldown_map: CooldownMap,
    current_window: int,
) -> tuple[list[ValidSubmission], dict[str, float]]:
    """Return (training_batch, rewards_by_hotkey)."""
```

Module rename: rename file to `selection_and_rewards.py` or keep
`batch_selection.py` — bike-shedding; pick one in implementation.

### `reliquary/validator/weight_only.py` / scoring

The scoring layer must consume the `rewards_by_hotkey` dict returned by
`seal_batch` and propagate to weight calculation. Implementation detail
depends on current archive → weight pipeline; defer to plan phase.

### Tests

The following will break and need rewriting:

- `tests/unit/test_batch_selection.py` — all FIFO `arrived_at`
  assertions invalid.
- `tests/unit/test_late_drop_server.py` — depends on `_per_window_counts`
  but should survive unless it asserts SUPERSEDED.
- `tests/unit/test_*_batcher*.py` — any test asserting SUPERSEDED on
  duplicate prompt submission needs to flip to PROMPT_FULL semantics.
- `tests/integration/*` — anywhere the integration test injects two
  submissions for the same prompt expecting the second to fail SUPERSEDED.

New tests required:
- `test_select_batch_and_distribute_drand_ordering` — given a fixed
  seed, ordering is deterministic and latency-independent.
- `test_select_batch_and_distribute_emission_split` — formula
  correctness across M, K_p combinations including M < 8 and M > 8.
- `test_select_batch_and_distribute_sybil_neutral` — N sybils on the
  same prompt produce the same attacker-total as a single submission.
- `test_prompt_full_cap` — MAX_SUBMISSIONS_PER_PROMPT + 1 rejected
  before heavy verify.

## Risks and open questions

- **Per-prompt training pick is random (drand-seeded).** Chosen for
  simplicity. If empirically the selected submission is often lower
  quality than others on the same prompt, switch to reward-max in a
  follow-up. Low risk: with K_p capped at 10, picking randomly costs
  at most a small drop in training signal quality.
- **DoS surface widens.** Without SUPERSEDED, the validator runs GRAIL
  verify on K_p submissions per prompt instead of 1. Bounded by
  `MAX_SUBMISSIONS_PER_PROMPT × |env|` per window and by the existing
  `_per_window_counts` per-hotkey cap. Watch validator GPU saturation in
  prod-burn-in; if pressure, lower `MAX_SUBMISSIONS_PER_PROMPT` to 5.
- **Cooldown semantics under multi-miner-per-prompt.** Today
  `record_batched(prompt_idx, window)` puts a prompt in cooldown when
  it enters the batch. Question: does it cool down when ANY submission
  on it is in `winning_prompts`, or only when one is picked for
  training? Answer: cool down on entering `winning_prompts` — the
  prompt has had its emission round, miners should rotate. The training
  pick is internal validator detail and shouldn't affect cooldown.
- **Multi-validator deployment.** Drand-only randomness depends on
  validators agreeing on which round to use. Currently single-validator
  so non-issue. When adding validators, the deterministic round-from-
  window-timestamp derivation must use a substrate-anchored
  `window_close_ts` (e.g., the timestamp of the closing block) to
  ensure consensus.
- **Transition.** No backward compatibility needed: this is a server
  protocol change with a single producing validator. Coordinate with a
  brief miner downtime; v30 miners can stay running but their
  spam-at-window-open behavior becomes useless. Reverting requires a
  validator container redeploy.

## Implementation lift

Code: ~150 LOC of additions + ~50 LOC removed. Tests: ~250 LOC rewritten,
~150 LOC new. Total: half a day of coding, one day of prod-burn-in. The
drand infrastructure already exists (`infrastructure/drand.py`, quicknet
default) so no new external dependency or trust assumption.
