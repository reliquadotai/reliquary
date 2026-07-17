# Validator Observability Notes

Validator submit logs use the `validator_submit_lifecycle` event with a
`stage` field. The important stages are:

- `submit_received`: FastAPI has parsed the request and the middleware
  arrival timestamp is attached. This is the HTTP arrival clock.
- `drand_validated`: the miner's `submitted_drand_round` has been compared
  to the validator's `arrival_drand_round`. `drand_delta =
  submitted_drand_round - arrival_drand_round`; positive is future, zero is
  current, negative is stale. `drand_tolerance` is the configured backward
  grace in rounds.
- `proof_started` / `proof_finished`: historical stage names for async worker
  admission and grading. In auction mode they do not imply GRAIL ran.
  `queue_wait_ms` is time from enqueue to worker start and `total_ms` is the
  admission decision latency.
- `candidate_accepted`: the submission passed bounded admission and entered
  the pending auction pool. It is not yet proven, selected, or rewarded.
- `candidate_rejected`: the validator made a final reject decision. Read
  `reject_stage` and `reject_reason` together; for `batch_filled`, also read
  `batch_filled_reason`, `current_valid_count`, and `trigger_round`.
- `seal_triggered`: legacy-selector stage only. Auction environments seal on
  the 300-second collection deadline and bounded queue drain.
- `auction_finalized`: seal-time result for every pending candidate. Read
  `canonical_rank`, `auction_status`, `selected_for_batch`, and `rewarded`.
- `final_batch_selected`: final drand/canonical ordering has run. A submission
  can be `accepted_into_pool=true` but `selected_for_batch=false`.
- `reward_assigned`: the submission earned emission in the final distribution.

Interpretation guide:

- Delayed admission logs are normal: production `/submit` returns
  `reason=submitted` after queueing, while `candidate_accepted` appears after
  async grading. Final auction outcome arrives only at seal.
- `submitted_drand_round` is what the miner attached. `arrival_drand_round`
  is the validator's drand round at HTTP arrival. `drand_delta=0` means
  current; `<0` means stale; `>0` means future.
- `seal_trigger_round` applies to legacy mode. In auction mode use the
  collection deadline, population-freeze state, and rank entropy source.
- `batch_filled` does not always mean the same thing. Check
  `batch_filled_reason`: common values include
  `submitted_round_gt_seal_trigger_round`, `batch_already_sealed`,
  `batch_already_sealed_or_draining`, and `batch_already_draining`.
- Accepted into pool, final selected, and rewarded are separate outcomes:
  `accepted_into_pool` means bounded admission passed; `selected_for_batch`
  means deferred proof passed and the candidate won; `rewarded` means it
  received one uniform auction slot. A non-winner is accepted but unselected,
  not rejected.
- Training and emission are also separate. A window can be rewarded/archived
  but have `training_quarantine.quarantined=true`; in that case the validator
  skipped GRPO and checkpoint publishing for model-health reasons.

The `/health` endpoint exposes non-secret operator state such as image
revision, app start time, checkpoint revision, current window, drand round,
per-environment pending/queue/proof state, operator mapping, grader failures,
archive queue, and recent reject counts. It must not include access keys,
tokens, wallet material, or private keys.

## Inference runtime and BFT telemetry

`GET /runtime-contract` advertises optional runtime telemetry without changing
the strict legacy `/state` response. Upgraded miners attach a self-reported
runtime fingerprint only when this endpoint exists. The profile hash is bound
to the signed envelope nonce, but it is observability data rather than remote
attestation.

Private `forced-seed-shadow.jsonl` schema v4 adds validator-derived termination
paths, first CDF mismatch offsets, token repetition metrics, and the miner
runtime profile. Summarize it with:

```bash
python scripts/report_forced_seed_cdf.py \
  /root/reliquary/state/auth_forensics/forced-seed-shadow.jsonl --json
```

Training telemetry under the `bft/` prefix reports forced-rollout share,
masked injected-token share, trainable-token share, and absolute-advantage
weighted exposure by validated termination path. These are review signals and
do not alter acceptance, reward, or gradient computation.
