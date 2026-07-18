# Difficulty Auction v2 - Final Production Contract

- **Initial design:** 2026-07-15
- **Finalized:** 2026-07-17
- **Implementation branch:** `design/difficulty-auction-v2`
- **Status:** production mechanism, enabled by default
**Environment scope:** `openmathinstruct` and `opencodeinstruct`

This document supersedes the rollout assumptions in the earlier auction plans
and reviews. In particular, the mechanism is intentionally active for both Math
and Code. BFT remains Math-only; that is a generation/termination distinction,
not an auction distinction.

## 1. Objective

The old event-driven selector rewarded the first eight valid prompt groups and
split a prompt's slot among same-prompt runners-up. That made arrival coverage
and extra identities economically important and diluted top miners even when a
later group sat closer to the learning frontier.

Auction v2 instead:

1. collects a bounded population for a fixed interval;
2. ranks in-zone groups by expected training value;
3. proves only candidates that can still win;
4. pays at most one proven group per prompt;
5. allows any operator to win multiple distinct prompts on merit; and
6. burns unfilled slots rather than paying unproven or low-ranked work.

The score selects training examples. It does not scale the value of a selected
slot: every winner receives one uniform `pool_per_env / B_BATCH` slot.

## 2. Scope And Timing

- Production environments: Math and Code.
- Each environment has an independent admission queue, grading budget, retained
  payload budget, pending population, and proof accounting.
- The collection deadline is `WINDOW_COLLECTION_SECONDS = 300` seconds after
  the environment is actually activated. Preparation time cannot consume the
  miner's collection interval.
- At the deadline, new admission stops. Work received before the deadline is
  given at most `MAX_SEAL_QUEUE_DRAIN_SECONDS = 60` seconds to quiesce before the
  pending population is frozen.
- There is no count-triggered early seal in auction mode. A fast miner cannot
  close the window while slower hard-prompt generations are still running.

## 3. Admission Contract

Before queueing, the validator checks the signed envelope, active window,
checkpoint, environment, registration, operator mapping, protocol version,
rate limits, logical claim, queue capacity, and serialized payload bounds.

An upgraded miner commits the exact finalized body before upload. The signed
`/submit/precommit` binds its SHA-256 and byte count alongside routing,
checkpoint, nonce, protocol, randomness, and drand. A predeadline receipt gives
that exact reveal at most 33 seconds to complete. It reserves normal hotkey
quota but no prompt or auction slot. The receipt's precommit-time drand
observation is authoritative for the reveal; submitted drand never affects
rank. Direct bodies remain compatible before the collection cutoff.

The worker then performs the bounded non-GPU path:

- canonical prompt and token binding;
- signature and window-randomness binding;
- rollout/hash invariants and recent content dedup;
- validator-authoritative reward computation;
- reward-shape and termination-independent cheap guards; and
- the zone threshold (`sigma >= 0.43`, or bootstrap threshold when active).

Passing this path means only **admitted to the pending auction pool**. It does
not mean the group passed GRAIL or won a slot.

Code grades its eight rollouts concurrently in isolated grader workers. A
candidate-caused outcome such as bad output, forbidden import, runtime error,
tampering, or timeout is a legitimate zero reward. Infrastructure outcomes such
as an unreachable grader, malformed grader response, or grader service error
never become a zero reward:

- retryable service failures return `WORKER_DROPPED`, cancel the logical claim,
  and refund submission quota;
- an ambiguous worker crash returns `REWARD_MISMATCH` and consumes the claim,
  preventing a crash-triggering candidate from obtaining free retries.

## 4. Difficulty Score And Eligibility

For validator-derived rollout rewards `r`:

```text
value(r) = std(r) * (1 - mean(r))^delta
delta = 1.0
```

For eight binary rewards the zone gate admits `k = 2..6`, and the score peaks at
`k = 2`. This favors hard groups that still contain positive signal instead of
rewarding already-solved groups. Reward vectors outside the zone never enter the
auction.

## 5. Identity And Anti-Sybil Rules

### Forced seed v2

The forced sampling stream is:

```text
u_at(window_randomness, prompt_idx, checkpoint_revision, rollout_idx, token_idx)
```

It deliberately excludes hotkey identity. Multiple hotkeys cannot buy different
legal draws for the same prompt. `BatchSubmissionRequest.protocol_version = 2`
is mandatory while forced-seed enforcement and a checkpoint are active. Older
clients fail before quota, grading, or proof admission with `SEED_MISMATCH`.

Exact per-token CDF enforcement remains disabled. The tolerant consistency gate
stays active because cached generation and validator teacher forcing are not a
bit-identical numerical contract across all supported runtimes.

### One logical claim per operator and prompt

Auction dedup reserves one `(operator, prompt_idx)` claim per window regardless
of hotkey or harmless payload variation. Missing or ambiguous operator ownership
fails closed. The historical hotkey fallback is not used in production auction
mode.

### Ranking tie-break

Candidates are ordered by:

```text
(-difficulty_value, operator_prompt_tiebreak)
```

`submitted_drand_round` is checked for freshness but never affects economic
ordering. The tie-break hashes only the operator, prompt, and post-deadline
drand salt. It excludes hotkey, Merkle root, and miner-controlled metadata, so
an operator cannot mint extra equal-score lottery tickets. If seal-time drand
is temporarily unavailable, window randomness is the deterministic liveness
fallback.

## 6. Deferred Proof And Selection

At seal, the validator walks the frozen ranking top-down:

1. skip a prompt already won by a higher-ranked proven candidate;
2. skip cooldown prompts;
3. fail closed on missing operator identity;
4. skip identities whose proof-failure debt is exhausted;
5. run the expensive GRAIL/auth/termination/distribution proof; and
6. on pass, claim the prompt; on failure, promote the next
   candidate.

A fabricated high-scoring group therefore cannot squat a prompt. It must pass
the proof before it can claim or earn anything.

The proof loop is bounded independently per environment:

- at most 96 grading/proof attempts;
- at most 240 seconds of seal-time proof wall clock;
- at most 2 expensive failures per hotkey;
- at most 4 expensive failures per operator; and
- at most 8 proven winners.

Two post-deadline-drand-selected non-winners are additionally proven for auth
forensics when budget remains. They never enter training or rewards.

## 7. Resource Bounds

- maximum parsed submission payload: 64 MiB;
- maximum retained pending payload per hotkey: 128 MiB;
- maximum retained pending payload per environment: 512 MiB;
- global active/draining queue depth backstop: 256;
- maximum retained groups per prompt: 10; and
- maximum started grading attempts per environment/window: 96.

Reservations are atomic before queue insertion and released on every reject,
drop, outage, seal, and cancellation path. Math and Code queues are isolated so
a pathological Code submission cannot head-of-line block Math admission.

## 8. Reward And Training Semantics

Each environment receives half of the window emission pool in the canonical
two-environment deployment. Every selected winner receives one of eight equal
slots from its environment. There is no active same-prompt runner-up split.
Unfilled slots remain unpaid and contribute to burn.

Selected prompts enter cooldown and selected rollout hashes enter replay dedup.
Only proven selected groups can enter the balanced Math+Code training
accumulator. Quarantine may still credit rewards while preventing suspicious
groups from changing model weights.

Training recovery remains a separate but coupled safety contract:

- immutable fixed KL reference
  `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`;
- explicit `RELIQUARY_KL_BETA=0.01`;
- validator-recomputed behavior-policy `pi_old`;
- `RELIQUARY_LEARNING_RATE=0.000003`;
- no length shaping; and
- pre-step gradient and PPO-ratio circuit breakers.

The fixed base is the KL anchor only. The last published checkpoint remains the
behavior policy miners used and therefore the source of `pi_old`.

## 9. Verdict And Archive Contract

The production lifecycle has three observable stages:

1. `/submit`: `SUBMITTED` means queued only.
2. First `/verdicts` record: `ACCEPTED` means admitted to the pending pool.
3. Seal-time `/verdicts` record: non-null `selected_for_batch` and `rewarded`
   report the final outcome; a deferred-proof failure carries its real reject
   reason and `reject_stage="auction_seal"`.

An honest non-winner remains `accepted=true`, with
`selected_for_batch=false` and `rewarded=false`.

R2 publishes the canonical per-environment payload under
`difficulty_auction`. `difficulty_auction_shadow` remains an identical alias for
older dashboards. Candidate rows include rank, score components, operator,
proof status, selection status, proof budgets, wall time, cap skips, and entropy
source. Code grader infrastructure failures are archived separately from
candidate reward outcomes.

## 10. Rollout And Rollback

This is a coordinated miner/validator hard cutover because forced seed v2 is a
wire-level generation contract. Deploy miners before or at validator activation.
No legacy grace period is implied once enforcement is on.

Production defaults to:

```dotenv
RELIQUARY_DIFFICULTY_AUCTION_ENFORCE=1
FORCED_SEED_ENFORCE=true
FORCED_SEED_CDF_ENFORCE=false
```

The emergency selection rollback is
`RELIQUARY_DIFFICULTY_AUCTION_ENFORCE=0`. It restores the legacy selector but
does not revert forced-seed protocol v2. Image rollback remains the stronger
option when wire/runtime parity itself is in doubt.

## 11. Release Gates

The release is acceptable only when:

- the complete unit/integration suite passes in the pinned validator runtime;
- both environment queues and grader health are green;
- the startup banner reports auction enabled for both environments;
- operator mapping is complete;
- one full live 300-second window seals without queue/proof-wall exhaustion;
- both environment archives land in R2 under the same window;
- final verdict records appear for winners, non-winners, and proof failures;
- rewards sum to no more than the window pool; and
- the balanced accumulator and training safety gates remain healthy.

Any proof-wall exhaustion, grader infrastructure failure, incomplete ownership
mapping, archive failure, reward over-allocation, or policy-health gate trip is a
stop/rollback signal, not something to reinterpret as miner underperformance.
