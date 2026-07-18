# Auction Ingress And Deadline Fairness Fix

**Date:** 2026-07-18

**Branch:** `fix/auction-ingress-admission`

**Scope:** Math and Code difficulty-auction admission

**Decision:** replace stale-round transport tolerance with a signed, exact-body
precommit/reveal contract and remove submitted drand from economic ordering.

## Incident Summary

Large groups, especially eight long Code rollouts, could finish generation on
time but cross a three-second quicknet boundary while their body was serialized
or uploaded. Production temporarily set `DRAND_ROUND_BACKWARD_TOLERANCE=1`.
That reduced `STALE_ROUND`, but it did not prove that generation had completed
before cutoff and did not address validator admission backlog.

The audit also found that submitted drand was part of the auction rank. A
positive tolerance therefore gave an older claimed round an ordering advantage.
Post-change production telemetry showed that every sampled window exhausted the
60-second queue-drain budget, with substantial predeadline work dropped after
seal. The archived `force_seal_reason` remained null, masking the failure mode.

## Final Protocol

1. The miner finishes all rollouts and GRAIL commitments.
2. Immediately before send, it reads current quicknet drand, creates a fresh
   nonce, signs the envelope, and serializes the final body exactly once.
3. It computes `payload_bytes` and `SHA256(payload)`.
4. It signs and sends `/submit/precommit`, binding hotkey, window, prompt,
   Merkle root, checkpoint, environment, exact size and digest, drand,
   randomness, protocol version, and nonce.
5. A validator receipt accepted before the 300-second collection cutoff grants
   up to 33 seconds for only that exact reveal.
6. The body is sent with `X-Reliquary-Precommit`. Its actual ASGI-stream digest
   and byte count must match. The receipt's precommit-time drand observation is
   authoritative even if the body completes after a quicknet boundary.
7. The precommit consumes one normal hotkey attempt but reserves neither a
   prompt nor an auction slot. Abandoning it cannot squat economic capacity.
8. Exact retries are idempotent and replay the completed body outcome.

Direct `/submit` remains compatible before the cutoff. After cutoff, the
validator accepts only a valid predeadline receipt. An old validator's 404 on
`/submit/precommit` triggers the reference miner's direct-submit fallback.

## Economic Ordering

Candidates now rank by:

```text
(-difficulty_value, operator_prompt_tiebreak)
```

The tie-break uses operator, prompt, and post-deadline drand entropy. It excludes
hotkey, Merkle root, miner metadata, and submitted drand. Submitted drand remains
a zero-tolerance freshness gate and telemetry field only. There is no
per-operator winner cap; one operator may win multiple distinct prompts.

## Admission Throughput

Reward grading moved outside the batcher's mutation lock. Admission uses a
bounded pool of four Math group workers and two Code group workers. Code groups
still fan out over the fixed grader sandbox pool; the group-worker bound avoids
an unbounded thread explosion. Stateful checks and pending-pool insertion remain
serialized under the batcher lock.

Auction admission performs no model forward, so parallel auction workers no
longer call `torch.cuda.empty_cache()`. Legacy inline-GRAIL admission retains
allocator cleanup, and CUDA/OOM exceptions still trigger emergency cleanup.

## Security Properties

- A miner cannot precommit a placeholder and replace it with a same-sized body.
- A miner cannot extend generation into the upload grace: the exact body digest
  must already exist before the precommit is signed.
- A bearer receipt cannot authorize another window, prompt, checkpoint, nonce,
  identity, environment, protocol, body size, or body digest.
- A precommit does not reserve prompt ownership or ranking position.
- Future drand is always rejected; backward tolerance returns to zero.
- Extra hotkeys do not change forced draws or equal-score operator/prompt ties.
- GRAIL failures continue to scrub all non-identity diagnostics.

## Telemetry Contract

Structured lifecycle logs, live verdicts, and R2 candidate rows now expose:

- `payload_bytes`, `content_length_bytes`, `payload_sha256_lead`;
- `body_read_ms`, `ingress_ms`, `upload_precommit_status`;
- `precommit_arrival_ts`, drand status and delta;
- queue depth at enqueue/dequeue and `queue_wait_ms`;
- `reward_grading_ms`, `admission_commit_ms`, and total decision time.

R2 also carries `force_seal_reason_by_environment`. A recorded
`auction_queue_drain_timeout` means the deadline was reached normally but the
bounded admission drain did not quiesce before population freeze.

## Deployment Order

1. Merge and build one immutable validator/miner image revision.
2. Deploy the validator first. It remains compatible with legacy direct bodies
   that finish before cutoff.
3. Set `DRAND_ROUND_BACKWARD_TOLERANCE=0`, preserve existing volumes, and
   recreate only `reliquary-trainer`.
4. Verify `/health`, `/state`, checkpoint continuity, both environment queues,
   operator mapping, grader pool, archive publisher, and active miner traffic.
5. Upgrade miners to the exact-body precommit submitter. Long bodies then gain
   upload grace without a scoring or generation-time advantage.

## Acceptance Criteria

- Same-size body substitution is rejected.
- A precommit accepted in round R can reveal after R changes without stale-round
  rejection, provided the exact body completes within grace.
- Direct late bodies without a receipt are rejected.
- Receipt retries do not consume quota twice and return the original outcome.
- Swapping submitted drand values cannot change the selected winner set.
- Reward grading overlaps across bounded workers without concurrent state
  mutation.
- Queue-drain timeouts are visible per environment in R2.
- Full local test suite and production smoke checks pass before unpinning the
  previous image digest.

## Verification Result

The complete repository suite passed on the pinned local runtime:

```text
1369 passed, 13 skipped, 2 third-party deprecation warnings
```

Focused tests additionally cover exact-body substitution, cross-drand reveal,
receipt outcome replay, queue-worker overlap, CUDA-cache isolation, archive
timeout persistence, and GRAIL diagnostic scrubbing.

## Rollback

Rollback is the previous immutable validator image plus its compose backup.
Restore the prior image only if health, checkpoint continuity, or submission
admission regresses. Do not re-enable positive drand tolerance as the long-term
transport fix; it does not provide an exact predeadline completion proof.
