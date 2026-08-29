# Adaptive auction fairness cutover

**Date:** 2026-08-29
**Scope:** validator collection, admission capacity, equal-difficulty ranking,
and observability
**Deployment:** direct production image on merge; no staging activation gate

## Production evidence

The decision uses 220 complete R2 archives (windows 35,964–36,183) plus live
health observations through window 36,196. Archive bodies refused after the
old seal are not retained, so every late-demand measurement is a lower bound.

| Finding | Math | Code |
|---|---:|---:|
| Windows mathematically early-sealed | 100% | 70.5% |
| Median close offset | +36.5 s | +53.3 s |
| Median 64th accepted offset | +34.5 s | +49.2 s |
| Selection rate, arrival quartiles Q1/Q2/Q3/Q4 | 61.2/28.6/8.2/2.0% | 74.7/27.3/1.1/0% |
| Rejected body median size vs accepted | 643 KiB vs 505 KiB | later bodies observed, same directional bias |
| Rejected bodies larger than same-window accepted median | 72.8% | lower-bound sample |

Payload size is a strong completion-length proxy (Spearman 0.979 Math, 0.921
Code), so the late-body skew is evidence that longer answers were being
excluded, not merely that slow network requests lost. At the same time, ranked
GPU proof attempts had median 16, p95 18, and maximum 19; no observed window
reached the existing v5 limit of 32, and all observed windows produced 16
winners. The proof-attempt limit is therefore not the collection bottleneck.

The last 64-window accepted-plus-full-reject population had p95/max 79/89 for
Math and 74/76 for Code. A 96-candidate productive pool covers every observed
window. Projected retained payload at 96 is about 55 MiB Math and 59 MiB Code,
well below the existing 512 MiB per-environment ceiling.

## Implemented policy

1. Keep `WINDOW_COLLECTION_SECONDS=100` as the hard ceiling.
2. Never adaptively close before 60 seconds.
3. Require the primary 64-candidate population and at least `B_BATCH` distinct
   trainable prompts.
4. Keep productive admission open to 96 candidates by default, leaving 32
   challenger positions. Operators may set
   `RELIQUARY_MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW=128` without a miner
   protocol cutover.
5. Require the previous pipelined window's GPU half to be finished. Collection
   time hidden under useful proof/train/archive work is not removed.
6. Require no pending upload receipts, queued admission, or in-flight grading,
   plus one actual drand period with no newly accepted candidate.
7. Freeze first, fetch post-seal randomness, then rank and prove. No
   mid-window proof or cached leader is introduced.
8. Keep `MAX_RANKED_PROOF_ATTEMPTS_PER_WINDOW=32` on v5 and keep the proof wall
   unchanged.

The v5 equal-difficulty key is now:

```text
(-difficulty_value, throughput_bucket, post_seal_operator_prompt_ticket)
```

Validator-observed arrival drand remains in the throughput denominator:

```text
capped_generated_tokens / max(arrival_round - window_open_round, 1)
```

It is removed only as a second standalone key. Applying arrival after it has
already formed the synchronized elapsed-time denominator double-penalized later
long answers inside the same throughput bucket. Historical profiles without a
throughput contract retain their original arrival ordering.

## Observability

`/health` publishes the productive limit, primary target, challenger capacity,
ranked proof limit, adaptive mode, and minimum/ceiling timing. Each environment's
`upload_precommit_conservation.early_close` block reports:

- GPU readiness and the offset at which it became ready;
- current close blocker;
- configured primary/challenger capacity;
- quiet-period duration;
- hypothetical eligibility offset; and
- actual early-seal offset.

Startup telemetry emits the same configuration, so a window archive and its
running image can be correlated without reading host environment files.

## Direct cutover and risk

Merging a code change to `main` triggers the validator image workflow, publishes
`ghcr.io/.../reliquary-validator:latest`, and the live watchtower restarts onto
that image. The current production environment already sets
`RELIQUARY_AUCTION_EARLY_CLOSE_MODE=enforce`, so the behavior is active on that
restart. Opening or updating the draft PR does not deploy anything.

There is no data migration, checkpoint migration, miner protocol/version
change, proof-fleet qualification change, or staged feature flip. The direct
cutover risks are:

- **Intentional economic change:** later and longer candidates get more access,
  and exact throughput-bucket ties use sealed randomness rather than raw
  arrival. Winner composition can change immediately.
- **Admission CPU increase:** up to 50% more productive reward-grading work and
  retained candidates (64 to 96). The never-refunded grading-start backstop
  stays fixed at 256, and measured payload headroom is large.
- **Residual early-close tradeoff:** adaptive close is not a proof that no
  candidate could arrive between the observed seal and 100 seconds. The
  60-second floor, 32 challenger positions, prior-GPU gate, drand quiet period,
  and drain gates reduce that risk; the 100-second ceiling remains the fallback.
- **One restart boundary:** watchtower replacement can forfeit the active
  in-memory window, as with any production image merge. Persistent cooldown and
  checkpoint state are unaffected.

Emergency rollback is operational: set
`RELIQUARY_AUCTION_EARLY_CLOSE_MODE=off` to restore the fixed 100-second ceiling.
Set `RELIQUARY_MAX_PROOF_GRADING_ATTEMPTS_PER_WINDOW=64` as well to restore the
old productive population bound. Both require only a validator restart, not a
miner cutover or a code revert.
