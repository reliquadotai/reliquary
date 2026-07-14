# Per-Window Prompt Range — Design

- **Date:** 2026-06-09
- **Status:** Approved design, pending implementation plan
- **Scope:** v1 — anti pre-curation lever for the validator/miner prompt-selection path

## Problem & Threat Model

Today a miner may submit a rollout for **any** `prompt_idx` in the loaded dataset. The
only gates are a bounds check (`prompt_idx < len(env)`), the 200-window per-prompt
cooldown, and per-prompt capacity (`batcher.py:705-716`, `cooldown.py:35-42`). The
loaded universe is the same fixed ~880k slice (2 shards of OpenMathInstruct-2) every
window (`openmathinstruct.py:185-203`), so it is maximally pre-curatable.

**Targeted threat: whole-dataset pre-curation.** A miner (or a collusion ring) maintains
a static/shared "bank" of known-good prompts across the whole loaded universe and submits
its favorites every window. This is the pattern behind the previously observed 24-hotkey
ring with a shared rollout bank.

**Explicit non-goal (residual, out of scope here):** the *solo* attacker who precomputes a
full per-prompt difficulty map over the loaded universe is **not** stopped by this design,
because the dataset and its ground truths are public. That residual is addressed separately
(held-out / private ground truth), not by ranging. We do not claim otherwise.

Note: *literal* pregeneration (generate offline against a stale checkpoint, replay later) is
already blocked — GRAIL binds every rollout to the current checkpoint hash
(`batcher.py:693`). What survives is pre-*selection*, which is what this design narrows.

## Goals

1. Force the eligible prompt set to rotate unpredictably each window, so a static/shared
   bank's fixed favorites fall in-range only ~`size/universe` of windows (~0.57% at
   size=5000) → static coordination stops paying.
2. Zero protocol/wire change (honor `GrpoBatchState extra=forbid`).
3. Deterministic, pre-announced cutover (no silent breakage of un-upgraded miners).
4. Likely side benefit (to measure, not promised): concentrating submissions onto a smaller
   set may help the chronic 8-distinct seal starvation.

## Current State (confirmed in code)

- No per-window range exists. Reject path: `batcher.py:705-716`, cheap path
  `server.py:980-1000`.
- Per-window `randomness` already exists and is derived **identically** by validator and
  miner from the same block hash + drand round (`service.py:695-701`); the miner reads it
  from `/state` as `state.randomness` (`engine.py:313`) **before** it picks the prompt
  (`engine.py:335`). It is unpredictable before window open (drand future round).
- Index→prompt-text resolution must already agree between miner and validator, enforced by
  token binding to the canonical prompt (`batcher.py:738-745`); a miner on mismatched shard
  text is already rejected today.

## Design Overview

Derive a contiguous per-window, per-env prompt range `[lo, hi)` from the existing
`randomness` seed, on both sides, from a shared pure function. The miner samples only inside
the range; the validator rejects out-of-range submissions once a graved cutover window is
reached.

### Component 1 — Shared range function

New module `reliquary/shared/prompt_range.py`, pure and unit-testable in isolation:

```
def window_prompt_range(randomness: str, env_name: str, universe_n: int,
                        size: int) -> tuple[int, int]:
    # seed is domain-separated so it cannot collide with GRAIL's use of randomness
    seed = sha256(b"prompt-range/v1|" + env_name.encode() + b"|" + randomness.encode())
    if universe_n <= size:          # tiny env (tests) -> no restriction
        return (0, universe_n)
    lo = int.from_bytes(seed[:8], "big") % (universe_n - size)
    return (lo, lo + size)
```

- **Contiguous block** (not scattered): the miner samples `lo + rng.randrange(size)` —
  trivial, no per-window enumeration/hashing of the whole universe; the validator tests
  `lo <= idx < hi`. Scattered selection is a possible v2 refinement (diversity), rejected
  for v1 on cost grounds.
- **Per-env** (`env_name` in the seed): math and code each get their own range.
- **`universe_n` agreement:** both sides MUST use the same universe size. Because shard
  config already must agree for token binding, the reference path uses `universe_n =
  len(env)` with a startup assertion that logs `env_name`, `len(env)`, and the configured
  shard count. Hardening note for the plan: if operators may load a *superset* of shards
  (making `len(env)` diverge while in-range text still matches), pin a per-env
  `PROMPT_RANGE_UNIVERSE` constant instead of `len(env)` so the range is independent of
  local shard count. Decide this in the implementation plan.

### Component 2 — Validator: cache + enforce

- When `randomness` is set (`_set_window_randomness`, `service.py:695`), compute and cache
  `(lo, hi)` per active env on the batcher, alongside the existing cooldown snapshot.
- Enforce in **both** reject sites, mirroring the cooldown check:
  - cheap arrival path `server.py:980-1000`
  - `_accept_locked` at `batcher.py:705-708` (insert right after the bounds check, before
    cooldown)
- New `RejectReason.PROMPT_OUT_OF_RANGE`.
- **Scheduled cutover:** new constant `PROMPT_RANGE_ENFORCE_FROM_WINDOW = N*`. For windows
  `< N*` the range is **not** applied (current behavior, no rejects). From `N*` onward, hard
  enforce. No log-only/measurement mode. `N*` is published and announced before the gated
  miner client ships.

### Component 3 — Miner reference client

- `pick_env_and_prompt` / `pick_prompt_idx` (`engine.py:81-144`) take `randomness`, compute
  `(lo, hi)` per env via the shared function, and restrict uniform sampling to `[lo, hi)`
  (still rejecting cooldown indices within the block). ~10 lines.
- `randomness` is already available at the call site (`engine.py:313` precedes `:335`).
- The reference client respects the range immediately on release; enforcement only begins at
  `N*`, so the client can ship ahead of the validator cutover safely.

### Component 4 — Constants

In `reliquary/constants.py`:
- `PROMPT_RANGE_SIZE = 5000` (env-overridable, tunable).
- `PROMPT_RANGE_ENFORCE_FROM_WINDOW = N*` (set at cutover-planning time).

## Data Flow

1. Window opens → `_set_window_randomness` derives `randomness` (existing).
2. Validator computes & caches `(lo, hi)` per env from `randomness`.
3. Miner reads `state.randomness`, computes the same `(lo, hi)`, samples a prompt inside it.
4. Submission arrives → cheap path checks bounds → **range (if window ≥ N*)** → cooldown →
   capacity → drand → … → GRAIL.
5. `_accept_locked` re-checks range (defense in depth) before the heavy verify.

## Edge Cases

- **Tiny env** (`universe_n <= size`, e.g. test envs): range = whole env → no restriction.
- **Range ∩ cooldown:** an in-range prompt that is also in cooldown is still rejected by the
  existing cooldown check. With size=5000 and ~2 prompts in cooldown per block, ~4998 remain
  eligible → ample headroom for the 8-distinct seal.
- **Randomness re-derived on retry** (`service.py:716` retry loop): same `randomness` → same
  range; stable for the whole window.
- **Per-prompt capacity:** concentration onto 5000 prompts may hit `MAX_SUBMISSIONS_PER_PROMPT`
  sooner; excess submissions get the existing `PROMPT_FULL` reject and miners move to another
  in-range prompt. 5000 choices is ample.

## Rollout Plan (enforce-direct via baked-in cutover window)

1. Implement and merge all three components; `PROMPT_RANGE_ENFORCE_FROM_WINDOW = N*` chosen a
   safe margin in the future.
2. **Release the upgraded miner client first** and announce `N*` to miners.
3. Deploy the validator (it does nothing different until `N*`).
4. At `N*` the validator hard-enforces. Any miner still on an old client is now in-range
   ~0.57% of windows → effectively rejected. This is the accepted, pre-announced consequence
   of enforce-direct; the announcement + client-first sequencing is the safety rail.

**Hard requirement:** the gated client MUST be live and adopted before `N*`, or honest miners
are zeroed too.

## Testing Strategy

- Unit: `window_prompt_range` determinism (same inputs → same output), per-env divergence,
  uniform `lo` distribution, tiny-env no-op, boundary `[lo, hi)` half-open.
- Validator: out-of-range rejected with `PROMPT_OUT_OF_RANGE` at both sites only when
  `window ≥ N*`; in-range accepted; cooldown still wins when both apply.
- Miner: sampled indices always within `[lo, hi)` and never in cooldown.
- Agreement: given one `randomness`, miner-computed and validator-computed ranges are
  bit-identical for the same `universe_n`.

## Future / Out of Scope

- Scattered/striped in-range set for batch diversity (v2).
- Roaming the full 14M (needs full/streaming dataset access on validator + miners).
- Held-out / private ground truth to close the solo difficulty-map residual (separate work).
- Per-miner ranges (keyed by hotkey) to break ring work-sharing further (v2 hardening).
