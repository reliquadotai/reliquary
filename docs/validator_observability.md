# Validator Observability Notes

Validator submit logs use the `validator_submit_lifecycle` event with a
`stage` field. The important stages are:

- `submit_received`: FastAPI has parsed the request and the middleware
  arrival timestamp is attached. `payload_bytes`, `body_read_ms`, `ingress_ms`,
  and `upload_precommit_status` distinguish transport time from queue time.
- `drand_validated`: the miner's `submitted_drand_round` has been compared
  to the validator's `arrival_drand_round`. `drand_delta =
  submitted_drand_round - arrival_drand_round`; positive is future, zero is
  current, negative is stale. `drand_tolerance` is the configured backward
  grace in rounds.
- `proof_started` / `proof_finished`: historical stage names for async worker
  admission and grading. In auction mode they do not imply GRAIL ran.
  `queue_wait_ms` is time from enqueue to dequeue, `reward_grading_ms` measures
  reward work outside the state lock, `admission_commit_ms` measures the
  bounded state mutation, and `total_ms` is the end-to-end decision latency.
- `candidate_accepted`: the submission passed bounded admission and entered
  the pending auction pool. It is not yet proven, selected, or rewarded.
- `candidate_rejected`: the validator made a final reject decision. Read
  `reject_stage` and `reject_reason` together; for `batch_filled`, also read
  `batch_filled_reason`, `current_valid_count`, and `trigger_round`.
- `seal_triggered`: legacy-selector stage only. Auction environments seal on
  the 100-second collection deadline and bounded queue drain.
- `auction_finalized`: seal-time result for every pending candidate. Read
  `canonical_rank`, `auction_status`, `selected_for_batch`, and `rewarded`.
- `final_batch_selected`: final auction ordering has run. A submission
  can be `accepted_into_pool=true` but `selected_for_batch=false`.
- `reward_assigned`: the submission earned emission in the final distribution.

Interpretation guide:

- Delayed admission logs are normal: production `/submit` returns
  `reason=submitted` after queueing, while `candidate_accepted` appears after
  async grading. Final auction outcome arrives only at seal.
- `submitted_drand_round` is what the miner attached. For a valid signed upload
  precommit, `arrival_drand_round` is the validator round at precommit arrival;
  otherwise it is the body request's arrival round. `drand_delta=0` means
  current; `<0` means stale; `>0` means future. Submitted drand does not affect
  auction rank.
- `upload_precommit_status=valid` means a signed commitment to the exact body
  hash and byte count arrived before collection cutoff. The reveal may complete
  during the bounded 33-second upload grace. `present`, `invalid`, `expired`,
  and `replay` distinguish the other receipt paths.
- `seal_trigger_round` applies to legacy mode. In auction mode use the
  collection deadline, population-freeze state, and rank entropy source.
- `batch_filled` does not always mean the same thing. Check
  `batch_filled_reason`: common values include
  `submitted_round_gt_seal_trigger_round`, `batch_already_sealed`,
  `batch_already_sealed_or_draining`, and `batch_already_draining`.
- Accepted into pool, final selected, and rewarded are separate outcomes:
  `accepted_into_pool` means bounded admission passed; `selected_for_batch`
  means deferred proof passed and the candidate won; `rewarded` means it
  received one uniform auction slot. In auction mode the final two fields must
  be identical. A non-winner is accepted but unselected, not rejected.
- Training and emission are also separate. A window can be rewarded/archived
  but have `training_quarantine.quarantined=true`; in that case the validator
  skipped GRPO and checkpoint publishing for model-health reasons.

The `/health` endpoint exposes non-secret operator state such as image
revision, app start time, checkpoint revision, current window, drand round,
per-environment pending/queue/proof state, operator mapping, grader failures,
archive queue, and recent reject counts. It must not include access keys,
tokens, wallet material, or private keys.

R2 archives persist `force_seal_reason_by_environment` and
`reward_alignment_by_environment`. Completed auction windows require zero
paid-unselected and selected-unrewarded groups. A non-null
`auction_queue_drain_timeout` means the fixed collection closed correctly but
some predeadline admission work exceeded the bounded drain period. Candidate
rows carry the same ingress and admission timings as live verdicts.

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
