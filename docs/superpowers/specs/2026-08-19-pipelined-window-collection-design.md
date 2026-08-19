# Pipelined Window Collection — Design

Date: 2026-08-19
Status: v2 — revised after the 2026-08-19 high review of the v1 implementation (10 verified findings; serial-path identity held)
Scope: validator only; zero wire/protocol change; zero miner-visible rule change.

## Problem

The window loop is strictly serial: collect (CPU/network, 100s) → seal →
GRAIL proofs (GPU, ~100s) → train (GPU, ~115s) → open next window. Measured
cycle ≈ 315-330s. The GPU idles during collection and the network idles
during GPU work. Micro-optimisations of the train segment are exhausted
(pi_old-from-verify shipped; residual gain ~10-15s).

## Goal

Hide collection entirely under GPU work: cycle → max(GPU, collection)
≈ P + T ≈ 215s (~+50% steps/hour), without changing any economic,
security, or miner-facing behaviour.

## Non-goals (hard constraints)

1. NO speculative proving: proofs only ever run on a SEALED window (ranking
   frozen, seal randomness drawn). The throughput tie-break makes leadership
   undecidable before the deadline (see ThroughputTiebreakProfile comment);
   this design opens early, never closes or proves early.
2. NO tie-break change. NO early close.
3. NO concurrent GPU work: proofs and train never overlap on the device;
   peak memory must remain max(proof peak, train peak), never the sum.
4. NO change to per-window economics: deadline length, admission rules,
   rate limits (per-window), payment, archive, publish cadence unchanged.

## Design

### Window lifecycle (state machine)

COLLECTING → SEALED (ranking + seal randomness frozen, waiting for GPU)
→ PROVING → TRAINING → PAID/ARCHIVED

Each window traverses every state exactly as today. New: up to two windows
in flight, with these invariants: at most ONE in COLLECTING; at most ONE on
the GPU (PROVING or TRAINING); at most ONE in SEALED (waiting for the GPU).
COLLECTING and SEALED are mutually exclusive for the same window but N may be
TRAINING while N+1 sits SEALED (~15s in the measured regime).

### GPU queue

A single serial GPU executor processes, in order:
proofs(N) → train(N) → proofs(N+1) → train(N+1) → …
The seal of N+1 is a CPU event (stop accepting, freeze ranking, fetch drand);
its GPU phase enqueues and waits. Constraint 3 is enforced structurally: the
executor runs one job at a time.

MAX_PROOF_WALL_SECONDS already anchors at the start of the expensive proof
phase (`batcher.py` sets `_proof_wall_started_at` inside the proof routine,
verified 2026-08-19), so queued waiting does not burn the wall. No change
needed.

### Pacing rule

Collection of window N+1 opens when train(N) STARTS (not at seal(N)).
Rationale: train (~115s) ≥ collection (100s), so seal(N+1) lands roughly when
the GPU frees. This bounds in-flight windows to two and prevents an unbounded
sealed-but-unproven queue. Degradation is graceful in both directions:
- train longer than collection → N+1 waits SEALED (frozen state, harmless);
- train shorter than collection → GPU idles briefly (no queue growth).

### /submit routing

Two live batchers per environment: the COLLECTING one and the downstream one.
Submissions already declare window_n; route to the collecting batcher for
that window. Submissions for a sealed window are rejected as today (late).
/state advertises the collecting window (and its pinned checkpoint), as today.

### Shared cross-window structures

hash-dedup and prompt/content cooldowns must span in-flight windows, not just
the current one (otherwise a miner replays the same submission into N+1 while
N is proving). These stores become shared between the two live batchers.
This is the most delicate part of the refactor; the existing
HASH_DEDUP_RETENTION_WINDOWS machinery already spans windows in TIME — the
change is making the live-batcher lookups hit the shared store.

### Checkpoint publication (SERIAL BEAT — v2, replaces the deferred swap)

v1's deferred verify-swap was refuted by review: (a) _ensure_proof_scheduler_
ready re-syncs the verify plane at iteration top, defeating the deferral;
(b) refreshing verify_model FROM train_model after train_step installs
revision+delta weights labeled as the published revision; (c) publishing
mid-collection flips /state under miners pinned to the old hash.

v2 rule: A PUBLISH NEVER PIPELINES. When the stashed window's train will
trigger a publication (cadence counter reaching threshold, or the adaptive
flag already set — both knowable before opening the next window), that
iteration runs in SERIAL order: GPU half first, publish (upload + verify
refresh + replica sync, exactly the serial-path sequence, with
train_model == published weights), THEN open the next window. Consequences:
no publish lands mid-collection; no old-revision window is ever in flight
when the verify plane swaps; the deferred-swap machinery is DELETED, not
fixed. Cost: one un-overlapped beat per publish (~100s / 16 windows, ~3% of
the gain). If a publish fires in a pipelined iteration anyway (unexpected
path), fail loudly and fall back to serial for that iteration.

### (v1 section kept for history — superseded)


A window is pinned to the checkpoint it was opened under (checkpoint_hash is
a forced-seed derivation input; proofs must verify against the generation
checkpoint). At a publish boundary (every 16 trained windows):

1. train(N) completes → K+1 UPLOAD starts in the background (async, as today).
2. verify_model stays at K until the last window generated under K (N+1,
   already collecting when the publish decision landed) is fully PROVEN.
3. Swap verify_model → K+1 at the queue boundary after proofs(N+1).
4. Window N+2 opens pinned to K+1.

No pause, no lost window, no extra resident model (the swap is deferred, not
duplicated). Fallback v1 if sequencing proves hairy: skip one overlap beat at
the publish boundary (window opens post-swap, serial) — costs ~100s per 16
windows (~3% of the gain), trivial to implement.

Note: pi_old-from-verify values attached at proof time are computed on the
window's generation checkpoint (correct by construction under the deferred
swap — proofs(N+1) still run on K).

## v2 phase plan (from the v1 review findings)

1. Seal the GPU half hermetically: guard _set_state with owns_routing
   (finding 1 — fatal: TRAINING state was rejecting the collecting window's
   submissions); capture the beacon verify task in the stash and await the
   stashed window's own task (5); snapshot late-drops at seal (10);
   tombstones carry explicit window metadata and allow the pipelined stages
   (4).
2. Seal side effects (7) — RESOLVED by analysis, lighter than planned. The
   commit DOES depend on winners (rewarded_submissions exist only after
   seal_batch = ranking + proofs, which runs in the GPU half), so it cannot
   move to stash time. But no enforcement gap actually exists: cooldown and
   content-cooldown are re-checked at seal-time RANKING, and GPU halves are
   strictly serialized, so window N's commit always lands before window
   N+1's ranking. Rollout-hash replay across windows dies at ARRIVAL
   (claimed randomness must equal the window randomness) and at forced-seed
   proof. Residual, ACCEPTED costs of one-window-late commits (v2 review):
   (a) advisory staleness — a prompt rewarded in N is not advertised as
   cooled during N+1's collection, so a miner re-picking it wastes one
   generation (bounded: ≤B_BATCH prompts/env, one window). A projection of
   the stashed pool into N+1's snapshot was BUILT AND REVERTED: since the
   snapshot also drives arrival rejects, unioning the whole admitted pool
   turned a losing bid into a cheap one-window prompt-denial primitive
   against competitors (review MAJOR) and changed admission rules (design
   non-goal). (b) the arrival hash-dedup set misses N's rewarded hashes
   during N+1's collection — defense-in-depth only; every paying path is
   proof-gated and randomness-bound. (c) server-side _recent_reject_counts
   are snapshotted into the stash so N's archive keeps N's counters; the
   upload-precommit conservation shift at activation is accepted
   (telemetry-only).
3. Serial beat at publish (above) — deletes the deferred swap (2, 3, 8, 9).
4. Failure paths (6) — shipped: a stashed-half failure tombstones ONLY the
   stashed window and the collecting window continues; an open-phase
   failure salvages the sealed backlog by running its GPU half serially
   before resetting, so it is paid+trained, or tombstoned as
   PipelinedSalvageFailure if the salvage itself fails. Fatal proof-plane
   errors always tombstone the backlog before terminating (no silent
   archive gap, no same-number replay) and are never downgraded by the
   salvage path. ACCEPTED structural cost: any hard process death (SIGKILL,
   OOM) while a backlog is stashed loses that sealed window unpaid — one
   more window per crash than serial mode; acceptable at the current
   crash rate, revisit if fatal restarts climb again.

5. Miner-visible timing (v2 review FATAL, fixed): poll_deadline is only
   driven by _wait_for_window_seal, so running the GPU half first would
   seal the collecting window ~a GPU half late and pin /state to OPEN for
   the entire cycle — out-of-phase miners would never re-sync. Fix: the
   collecting window's seal-wait runs as a CONCURRENT task
   (_seal_wait_and_close) alongside the stashed GPU half; it seals on the
   deadline and flips the FSM to READY at that moment, restoring the
   serial-mode OPEN -> not-OPEN edge. The stash also releases routing and
   FSM state exactly like the serial end-of-window path.

## Edge cases

- Restart: a SEALED-not-PROVEN window must resume or be aborted exactly once
  (the historical "window replayed under the same number" complaint makes
  this the top resume invariant). Persisted seal snapshots already exist;
  the resume path must reconstruct the queue position.
- Empty/starved window: traverses states with no GPU work; next opens on
  schedule.
- TrainingStepSkipped (drift breaker): frees the GPU slot early; queue
  advances; adaptive publication logic unchanged.
- force_seal safety valve: still valid per window; it seals, the GPU queue
  picks the window up in order.
- drand: N+1's randomness needs are available at its open/seal, independent
  of N's GPU work (already true today).

## Rollout

- Feature flag RELIQUARY_PIPELINED_WINDOWS (default off at merge; flipped in
  the compose overlay after a soak on the serial path with the new code).
- Deploy = image swap, same procedure as today; no manifest interaction
  expected (no PROOF_PATH_FILES change — the GPU queue wraps callers of the
  proof path, it does not modify it). Verify with the proof-path hash check
  against origin/main before merge.
- Observability: window state timestamps (opened/sealed/proof_start/
  proof_end/train_start/train_end) exported per window; alert if
  sealed→proof_start wait exceeds ~60s sustained (pacing broken) or if more
  than 2 windows are in flight (invariant broken).

## Test plan

- State-machine unit tests: legal transitions only; at most one COLLECTING
  and one downstream window.
- Pacing: simulated clocks — train longer/shorter than collection; assert
  bounded in-flight count in both regimes.
- Routing: submissions to collecting vs sealed windows; late rejection.
- Shared dedup/cooldown: replay across in-flight windows is rejected.
- Publish boundary: proofs of the K-generation window verify against K after
  K+1 upload started; N+2 pins K+1; fallback beat-skip path.
- Restart: seal-snapshot resume reconstructs queue position; no double-pay,
  no window replayed twice.
- Proof wall: starts at PROVING, not at seal (queued wait excluded).

## Expected result

Cycle ~215s (from ~320s), ~16.7 steps/hour (from ~11.2), GPU utilisation
near-continuous. Next floor beyond this (not in scope): a dedicated proof
GPU (RELIQUARY_PROOF_DEVICES already abstracts devices) → cycle ≈ max(C, T).
