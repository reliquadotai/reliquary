# Async RL: implementation map

Status: study (2026-10-15), read-only. Target: `docs/rl-async-design.md` (DRAFT).

Baseline: `origin/feat/next-rl-run` @ `b9afcb13`. The local worktree is 6 commits behind, at `fa0f3a76`.
- All `file:line` references are relative to the repo root at `b9afcb13`.
- The 6 newer commits shift line numbers slightly in these files: `constants.py` (+9 after l.823), `service.py` (+3 after l.4048), `training.py` (+5 after l.2054), `admission.py` (+2 after l.637), `cli/main.py` (+5 after l.2899).
- `feat/forced-seed-runner-fix` (= `feat/gumbel-forced-draw`, both @ `a033b07c`) branches from `fa0f3a76`, not `b9afcb13`. It touches `miner/vllm_generation.py`, `miner/corpus_generate_server.py` and their tests. No Gumbel code exists yet in either branch: `grep -i gumbel reliquary/` is empty. The reference implementation is `~/offpolicy-lab/lab/gumbel.py` (59 lines) and `runner_forced.py` (148 lines).

Sizes below are rough, in lines of code.

## 1. Component map

### 1.1 Forced draw (protocol, miner, validator)

**Today: inverse-CDF on a public uniform**

- Legacy stream: `u_at(randomness, prompt_idx, checkpoint_hash, rollout_index, t)` in `environment/forced_sampling.py:66-82`.
  - Its domain is `FORCED_SEED_DOMAIN`, which embeds `PROTOCOL_VERSION` (`constants.py:1925`).
  - The hotkey is excluded on purpose (`:70-74`).
- Pick: `pick`/`warp` at `forced_sampling.py:38-60`.
- Validator statistics: `seed_consistency_diagnostics` at `forced_sampling.py:177-329`. It returns exact, boundary and hard-mismatch counts, with `exact` computed at `:280`.
- Public pool: `SeedPool.uniform` at `protocol/seed_pool.py:258-269`, defined as `sha256(DRAW_DOMAIN‖pool digest‖seed‖pos)`.

**Miner**

- `ForcedSeedVLLMProcessor` at `miner/vllm_generation.py:40-115`:
  - It uses its own per-slot counter (`:72`, `:113-114`).
  - The processor itself does the CDF and `searchsorted` (`:94-115`).
  - The extra args are built at `:122-146`.
- HF fallback: `ForcedSeedLogitsProcessor` at `miner/forced_seed_sampler.py:449-506`, which positions by `input_ids` length (`:495-498`).
- BFT phase 2 uses the same processor (`miner/engine.py:457-548`).

**Validator**

- `verify_commitment_proofs` at `validator/verifier.py:772-1079`; the seed block is `:899-974`.
- The uniforms come from `seed_uniforms` (`validator/batcher.py:5151-5161`).
- Verdicts: `_forced_seed_verdict` and `_forced_seed_rollout_reject` (`batcher.py:477-506`), applied at `batcher.py:6100-6171`. The hard CDF mismatch is always enforced on the service path.
- Floors (`constants.py`):

| Constant | Line | Value |
|---|---|---|
| Group floor | 1942 | 0.80 |
| Rollout floor | 1958 | 0.70 |
| `FORCED_SEED_STOCHASTIC_MAXPROB` | 1928 | 0.99 |
| `FORCED_SEED_MIN_STOCH_POSITIONS` | 1945 | 30 |
| `FORCED_SEED_ROLLOUT_MIN_STOCH` | 1962 | 20 |

**Reuse**

- The validator's stochastic-position definition (max p < 0.99).
- The per-rollout and per-group tally plumbing (`batcher.py:5121-5140`, `:5336-5398`).
- `policy_token_positions` (`verifier.py:520-542`), which already counts model tokens across episode turns.
- The proof transport (`validator/proof_worker.py:82-110`, `:633-678`; `remote_proof*.py`). `validator/proof_scheduler.py` (1,703 lines) is draw-agnostic.

**Change**

- New module, e.g. `protocol/gumbel_draw.py` (~150): splitmix64 noise, bit-identical across GPUs; `nonce = H(S_k, prompt, k)`; per-position seed `(nonce, rollout_index, position)`; `pick`; validator stats (v*, match, score gap).
- Port the runner-keyed processor (P0 branch, +253/−41 in `vllm_generation.py`) to produce a Gumbel pick instead of searchsorted (~100).
- HF fallback processor and BFT phase 2 (~60).
- Verifier seed block: compute v* and the score gap per position, and return `vstar` in `ProofResult` (`verifier.py:50-146`, `:1044-1057`) (~150).
- Remote-proof schema: replace `seed_u_values` (`remote_proof_protocol.py:176`, `:233-236`, `:283-304`) with a nonce plus a per-rollout seed spec, and return v* (~80).
- Floors and verdicts become the Gumbel per-rollout floor (initial candidate 0.95) plus pooled per-hotkey statistics (~80).
- Optional fp32 LM head on both sides: the miner's vLLM model config and `verifier.py:870-879` (~40).

**Delete**

- The CDF-interval path in `forced_sampling.py`: about 250 of 329 lines.
- `ForcedSeedVLLMProcessor`, which the design keeps only for tests.
- `FORCED_SEED_CDF_*` constants (`constants.py:1968-1969`).
- The "hard CDF mismatch" class in `services/exploration.py:112-136`.

**Size:** about 600 new or changed lines and about 400 deleted.

### 1.2 Leases (new)

**Closest code today**

- Single-turn upload precommit:
  - `POST /submit/precommit` (`validator/server.py:4619-5018`).
  - Binding: `build_precommit_binding` and its verify (`protocol/signatures.py:574`, `:643`).
  - Registration: `try_register_upload_precommit` (`batcher.py:2598-2707`).
  - Receipts: `server.py:1911-2343`.
  - No-reveal breaker: `validator/no_reveal_circuit.py` (114).
- Episode precommit:
  - `POST /rl/episodes/precommit` (`sandbox/rl_routes.py:90-222`), recorded by `ServiceRuntime.record_episode_precommit` (`services/runtime.py:1093`).
  - One live precommit per (hotkey, env, task, window).
  - One-use seed engagement `rl:{window}:{precommit}:{seed}` (`validator/rl_engagements.py:1-28`). This is already a "no re-roll" mechanism.
- Generic lease prior art: `LeaseExecutor` (`validator/lease_executor.py:40`) and `_Lease` (`corpus_grade_remote.py:256`).
- Concurrency cap analog: `MAX_PENDING_UPLOAD_PRECOMMITS_PER_HOTKEY/_PER_OPERATOR = B_BATCH` (`constants.py:1242-1243`).

**Reuse**

- The per-hotkey minute limiter, pre-auth semaphore and replay guard in `rl_routes.py:47-62`, `:119-187`, `:266-283`.
- The signature helpers' style in `signatures.py`.
- The SQLite plus `_txn` pattern of `services/runtime.py` for persistence: open leases must survive a restart (design §8).

**Change or new**

- `validator/rl_leases.py` (~500). It holds:
  - the lease table (lease_id, hotkey, env, prompt, k, issued, expires, state);
  - one open lease per (prompt, k), with prompts closed for k once submitted;
  - 1 s interval batching with random conflict resolution;
  - expiry per env from the contract;
  - restart safety.
- `S_k` commit and reveal (~150). Generate S_k at checkpoint publish/adopt, publish its commitment next to the checkpoint identity, and reveal it once k is retired (max lag S). There is no existing secret-per-checkpoint mechanism. The closest is `_bind_window_activation_randomness`, which mixes `os.urandom(32)` (`service.py:150-166`).
- Routes and models (~200).
- Cooldown and prompt-range checks move to lease time (see §1.4).

**Delete**

- `/submit/precommit` and the receipt plumbing (~830 in `server.py`).
- `try_register_upload_precommit` and the upload-reservation block (`batcher.py:2538-2994`, ~456).
- The no-reveal breaker, which is replaced by expiry halving.
- The episode precommit router, if episodes move to leases.

### 1.3 Quota (new controller)

**Closest code today**

- `sandbox/rl_quota.py` `YieldQuota` (`:139-747`).
- `sandbox/rl_capacity.py`: `water_fill` `:148-166`, `allocate` `:185-200`, `standing`/`weight`/`multiplier` `:124-145`, `potential` `:203-212`.
- `sandbox/rl_yield.py` `SessionYieldLedger` (useful/wasted/fault verdicts).
- Signed `POST /rl/sessions/allocation` (`rl_routes.py:225-300`).
- Miner side: `episode_mining.py:69-75`, `:909-956`.

All of these share a known fleet capacity C. None of them does slow start, halving on expiry, or a Little's-law target.

**Reuse**

- The route template.
- The yield-ledger idea, with useful = "lease submitted on time" and wasted = "expired".
- The demand-memory decay pattern.
- `water_fill`, if a global cap on concurrent leases is wanted.

**New**

- An AIMD controller per hotkey (~250):
  - start at 4;
  - double while the full quota is used and delivered on time;
  - halve on each expiry;
  - target = throughput × latency × 1.5 over a sliding window;
  - out-of-zone submissions count as delivered.
- Its persistence.
- `GET /rl/quota` with reasons (~100).
- The miner reader (~60).

**Episodes:** an episode env keeps the sandbox `YieldQuota`, because fleet capacity binds. The lease quota and the sandbox allocation then compose as a minimum. The design's §4.10 ("apply unchanged") does not say this.

### 1.4 Submission and grading (`POST /rl/submit`)

**Today**

- `POST /submit` (`server.py:5020-~6009`): auction reveal (`:4139`), then `_process_auction_submission` (`:6713`), then `prepare_submission` (`validator/admission.py:1502`), then `reserve_prepared_identity` (`batcher.py:4004`), then `accept_prepared_submission`.
- Model: `BatchSubmissionRequest` (`protocol/submission.py:212`, exactly `M_ROLLOUTS`, `:282`).
- Grading: `score_and_finalize_submission` (`admission.py:942-1401`) calls `EnvironmentSpec.score_many` (`environment/registry.py:279`).
- Reward authority:
  - Code is authoritative; math (`registry.py:445`) requires the miner's claim to equal the validator's grade, or the group is rejected with `REWARD_MISMATCH` (`admission.py:1153-1170`).
  - The validator already grades everything; the miner's reward is only a claim.
- Lane: `service_lane` (`services/admission_policy.py:201-228`) returns training, unproven or exploration. It uses `classify_signal` (`services/scoring.py:40-62`) and `robust_utility_admits` (`admission.py:1563`).
- Zone: `_in_zone` (`admission.py:878-888`, sigma ≥ 0.24, `constants.py:743`).
- Termination: `_termination_reject` (`:618-646`); the truncation cap is skipped on the service path (`:639-641`, commit `437e6347`).
- Dedup: within a group, `admission.py:808-811`; across windows, `RolloutHashSet` (`validator/dedup.py:94`).

**Reuse as is**

`parse_and_validate_submission` (`:647-870`), the scoring adapters and termination (`:494-646`, `:870-1413`), `service_lane`, `classify_signal`, `robust_utility_admits`, `exploration_pay_entitlement` (`admission_policy.py:140-155`), the grader service (`environment/grader/*`), and the process-pool path (`server.py:6797`).

**Change**

- `score_and_finalize_submission` returns early on OUT_OF_ZONE (`admission.py:1284-1294`) and skips the later checks: malformed box (`:1297-1315`) and opposite-reward clones (`:1322-1332`). It must instead always return rewards plus lane, and reject only on integrity failures (~80).
- Request model: drop `window_start`, `drand_round`, randomness, `pool_selection`, merkle/commitments and `token_logprobs`; add `lease_id`, `k` and the generating checkpoint. Payload: compressed tokens only (~150 across `submission.py` and `signatures.py`).
- Group integrity: all 16 rollout indices, each bound to the lease nonce.
- The `unproven` lane (in zone but not robust) has no counterpart in the design. Decide: train, store, or treat as out of zone.

**Delete**

- The auction-reveal queue (`server.py:4139-4528`).
- The legacy direct `/submit` (`server.py:5033-6009`, ~975).
- `MAX_SUBMISSIONS_PER_PROMPT` (`constants.py:1372`).
- `window_prompt_range` per window (`shared/prompt_range.py:15`). The service path already uses `rl_prompt_range` (`protocol/rl_prompt_ranges.py`, new in `599c194b`), and that is kept.
- The per-window quota counters (`server.py:1563`, `:4838-4844`, `:5544`).

### 1.5 In-zone verification

**Today:** `_verify_expensive` (`batcher.py:4938-6312`, 1,375 lines). It runs GRAIL/TOPLOC, the logprob claim, the forced-seed tally and `verify_termination`, then dispatches through `proof_scheduler`.

pi_old comes from `completion_chosen_probs` (`verifier.py:870-879`), goes through `_verify_logprobs_for_training` (`batcher.py:380-408`), and is stored at `batcher.py:5333`.

**Reuse:** `proof_scheduler.py`, `proof_worker.py`, the remote proof transport, the termination and stop-pick checks (`verifier.py:576-606`, `:1113-1220`), and the pi_old extraction (still useful for telemetry and the "current" loss mode).

**Change**

- The seed source is the lease nonce and checkpoint k. The verify model is always the latest checkpoint; there is no per-window `self.randomness`.
- Store v* per policy position, sentinel -1 for force-span positions (`verifier.py:920-923`).
- Extract `_verify_expensive` out of the window batcher into a per-group verifier service (~600 moved, ~300 changed).

**Delete for RL only**

- `protocol/grail_verifier.py` (246) and `protocol/crypto.py` (198), which only GRAIL and challenge indices use.
- `validator/toploc_check.py` (95).
- The verifier sketch, challenge and TOPLOC blocks (`verifier.py:608`, `:978-987`, `:1020-1040`, `:1237`).
- In the batcher: `_with_toploc_spec`/`_proof_commit` (`:514-546`), the TOPLOC part of `signed_episode_proof_refusal` (`:575-616`, keep the stop-pick part), `:5221-5229` and `:5413-5440`.
- The miner logprob gate: `verify_logprobs_claim` (`verifier.py:1571-1642`) and `_verify_short_logprob_claim` (`batcher.py:411-474`).
- `GRAIL_*_PROOF_VERSION` (`constants.py:20-23`).

**Keep for corpus:** `protocol/toploc.py`, `toploc_proof.py` build/verify, `toploc_wire.py`, `corpus_audit.py`, `profiles.py` `ProofProfile` (`:230-318`), and the span proofs in `corpus_generate_server.py:411-426`.

**Coupling:** every RL commit signature hashes `commitments` (`signatures.py:54-85`, verify `:264-345`). Removing GRAIL therefore changes every RL signature format.

### 1.6 Out-of-zone sampled audits

**Today:** `services/exploration.py` (764 lines) has:
- the audit draw `audit_selected` (`:179-200`): drand beacon of arrival + 2 rounds, observation id and secret run salt, against `audit_bps`;
- probation `in_probation` (`:284-290`): `new_hotkey_audit_groups`, plus 20 re-probation passes (`SERVICE_REPROBATION_PASSES`);
- graded sanctions `_sanction` (`:316-327`) and forfeit (`:431-440`);
- verdicts `apply_verdict` (`:442-497`).

Dispatch is in the batcher: `_audit_candidates`, `_submit_audits`, `_execute_exploration_audit` and `finalize_service_exploration` (`batcher.py:6860-7480`). Priority is `SERVICE_EXPLORATION_AUDIT_PRIORITY=5`, with budget constants at `constants.py:880-923`. Ledger tables are at `exploration.py:222-254`.

**Reuse:** `audit_selected`, the probation and ban tables, the class table (`:119-136`), and the priority plans.

**Change**

- Audit rows exist today only for paid first scans: refusals `zero_price`/`cap` (`exploration.py:352-371`) and `already_scanned` (`:703-704`) never create a row. Every out-of-zone group must become audit-eligible without money attached.
- Sanctions forfeit exploration money. Under the design they become quota penalties or bans (§4.4).
- Audits are window-scoped: queued audits drain at seal (`service.py:4783-4890`), and an undrawn audit is unpaid and unsanctioned. They must become time-bounded per group. Under "verify against latest", an audit that runs late sees a newer checkpoint and a lower match. Audit promptly, or apply the lag-aware floor.

**Size:** ~300 changed in `exploration.py` and ~400 moved out of `batcher.py`.

### 1.7 Observation log and publication

**Today**

- `services/run_log.py` (381): `run_observations` indexed on `(order_id, environment, prompt_idx)`, `run_scans` keyed the same way, `run_events` (`:121-138`). The dataset id is in the payload. `observation_id` hashes the window (`:75-82`).
- `services/publication.py` (447):
  - `observations/run-<id>/seg-NNNNNN.jsonl.gz`, `index.json`, index pages (`:79-88`);
  - sr25519-signed (`:116-119`);
  - flush every 60 s (`constants.py:883-884`);
  - started at `service.py:8636-8656`.
- Miner side: `miner/observation_client.py` (570), including `preferred_seeds` (`:379-395`).
- Training cooldown: `validator/cooldown.py` (311), recorded at seal (`batcher.py:8759-8780`). `BATCH_PROMPT_COOLDOWN_WINDOWS=1_000_000` (`constants.py:1111`).
- Cooldown advisor: `services/cooldown_advisor.py:172-222`.

**Reuse:** almost everything.

**Change**

- The `window` attribute becomes `k` plus `lease_id`.
- Drop the `seeds` field, since the 2×M pool goes.
- Status vocabulary: trained / stored / audited, and no "exploration paid" if C1 goes that way.
- Cooldown keyed by checkpoints or seconds, and checked at lease time.

### 1.8 Buffer (new)

**Today:** a strict window journal.

Write side:
- `FillClosedBatchAssembler._prepare_payload_locked` (`validator/fill_closed_batch_assembler.py:321-390`), with keys from `encoded_window_journal_key` (`infrastructure/training_payload_queue.py:74-105`).
- Payloads are committed create-only, with receipts (`:504-645`) and tombstones (`service.py:7342-7429`).

Read side:
- `WindowJournal.next_entry` (`trainer/journal.py:115-165`).
- `BalancedTrainingAccumulator` (`validator/training_accumulator.py:65-109`) resets on a checkpoint change (`:78-83`). This blocks mixed-checkpoint batches.

**Reuse**

- The queue upload and drain transport (`training_payload_queue.py:131-148`, `:315+`, `:780-940`).
- The payload codec (`shared/training_payload.py`).
- The accumulator core as the env-mix filler.
- `quarantine.assess_training_batch` (`validator/quarantine.py:81`).

**New:** a buffer of accepted in-zone groups tagged (env, k, arrival), taken in arrival order across envs until the batch is full, with a max-lag drop (~300; revised 2026-10-16).

Placement is open:
- validator-side, emitting sequential batch payloads to R2 (keeps the detached-trainer topology);
- trainer-side, reading per-group objects.

**Delete:** the fill-closed modules:

| Module | Lines |
|---|---|
| `fill_closed_batch_assembler.py` | 728 |
| `fill_closed_recovery.py` | 380 |
| `fill_closed_rotation.py` | 182 |
| `validator/fill_window.py` | 139 |

Also delete the encoded keys, the committed-journal receipts (`training_payload_queue.py:348-470`, `:504-645`), and the accumulator reset.

### 1.9 Trainer loop

**Today:** `TrainerWorker.run_once` (`trainer/worker.py:164-246`). It advances a cursor, writes `step-cursor.json` (`:95-106`, `training_payload_queue.py:647-674`), and publishes every `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS=16` (`constants.py:703-706`, `worker.py:108-123`). It also publishes early ("adaptive") on ratio drift (`worker.py:231-232`).

Other pieces:
- `TrainRunner` (`trainer/train_runner.py:150-169`).
- `TrainerPublisher` (`trainer/publisher.py:316-431`): HF commit, R2 mirror `reliquary/checkpoints/<rev>/`, `reliquary/training/candidate-manifest.json`.
- Validator intake: `CheckpointIntake` (`validator/checkpoint_intake.py:93-412`).
- FSDP2: `trainer/distributed.py`, with `check_same_plan` added in `0290297b`.

**Reuse:** `training.py` (2,345 lines; nearly all of it), `publisher.py` (publish at any time), `distributed.py`, `checkpoint_intake.py`, and `resume.py` (with new cursor semantics).

**Change**

- Loop becomes "fill batch, step, publish every n steps". The design says checkpoints are "published continuously"; n is not fixed (see C7).
- Drop `trained_window_cursor` and `journal_key_space` (`publisher.py:367-395`).
- Retarget the drift gate (`training.py:2223-2226`) and adaptive publish to the new drift metric.

**Delete**

- `trainer/journal.py` (178).
- The cursor, stride and drain logic in `worker.py` (~150).
- The step cursor (`training_payload_queue.py:108-117`, `:200-278`, `:647-780`).
- Cursor migration in `trainer/cli.py:167-197`, `:335-350`, `:489-501`.
- On the validator: pick pacing (`service.py:4601-4782`: `_fill_closed_pick_gate_open`, `_trainer_consumed_through`, the `candidate_shortfall` reason at `:4746`, `_drive_fill_closed_picks`), the rotation gate (`service.py:2046-2342`), and pick claims (`batcher.py:2173-2538`). `fifo-ingress/v1` lives at `_pick_sort_key` (`batcher.py:1010-1015`) and `constants.py:1069`.

### 1.10 Loss (substitution)

**Today**

- Labels are the miner tokens: `targets = input_ids[b_idx, p_idx+1]` (`training.py:1544`, in `_batched_completion_logprobs` `:1514-1555`).
- Main loss: `_microbatch_grad` (`:1783-1958`); ratio at `:1905-1907`, loss at `:1920`.
- Advantages: `:436-450`. Overlong: `:554-611`. Env weights: `:815-838`. Degenerate drop: `:955-959`. Clip: `:1282-1307`.
- pi_old: `_validator_completion_logprobs` (`:1310-1343`).
- Miner-logprob fallback (`:1426-1446`) and miner-logprob lengths used for accounting (`_completion_token_logprobs`, `:876-889`, used at `:606`, `:968`, `:2016-2022`).

**Change**

- Add v* as an item element in `_build_microbatch_items` (`:1999-2036`).
- `target_override` in `_batched_completion_logprobs`, defaulting to None for `_padding_pass` (`:2039-2046`); the FSDP2 head (`distributed.py:95-103`) gets the same targets.
- A loss switch (substitution / mask / current) around `:1905-1920`.
- Telemetry in `_record_kl_stats` (`:1591-1780`).
- An equivalent of `_filter_missing_pi_old` (`train_runner.py:65-105`) for v*.
- Lengths from tokens, not miner logprobs.

**Size:** ~250.

v* must come from the proof. The trainer cannot recompute it: it has no pi_old model, and the payload carries no nonce.

### 1.11 Payment

**Today:** per-group, per-window.

- `fixed-selected-group/v1` (`validator/token_rewards.py:9-57`).
- `_accrue_payment_locked` (`fill_closed_batch_assembler.py:456-514`).
- Service money: `services/settlement.py` (`_compute` `:151-175`, `_count_rows` `:99-114` counts rows, `settle_window` `:178-252`), called via `runtime.reconcile_archive` (`runtime.py:1640-1732`).
- Weights: `weight_only.py` replays per-window archives (`:216-225`, `ROLLING_WINDOWS_HISTORY=72` `:31`). `SERVICE_ROW_FIELDS` has no tokens (`:41`). EMA at `:627-730`.
- `eos_tokens` is already measured (`admission.py:582-620`), and `split_environment_pool` already splits by `eos_tokens` (`token_rewards.py:21-38`).

**Change or new**

- A pay-per-matched-token policy with a per-rollout floor (~200).
- A token field in archive rows and in `SERVICE_ROW_FIELDS`.
- A settlement period that is not a window, e.g. a fixed wall-clock period (~300 across `settlement.py`, `weight_only.py` and the archive writer).

`weight_only.py` runs on every validator. Its replay must accept the new archive before the owner emits it (§3).

**Delete:** the window parts of `_archive_window` (`service.py:6484`, 796 lines). The 15 % first-scan exploration money depends on C1.

### 1.12 Price controller

**Today:** `validator/emission_price.py` (793).
- Regimes snap, rise, descend and hold in `advance` (`:205-264`).
- `r` = training rounds over incompressible rounds (`:427-437`), advanced once per window.
- Per env: `:307-383`.
- Trainer wait accrued per env in 5 s ticks (`service.py:4669-4699`, commit `86bce6e3`).
- Applied as a pool factor (`service.py:6238-6276`). Armed by default (`constants.py:1577-1579`).

**Reuse:** the per-env walk, clamp, breaker and restore (`service.py:6322-6360`), and the trainer-wait accounting (its signal survives as "trainer waited on env").

**Change:** a per-token price, not a pool factor. Advance on time or steps, not windows. Signal (revised 2026-10-16): each env's queued tokens per second against its target share of the trainer's reported speed. Remove the snap regime (no window fill). Size ~300.

### 1.13 `/rl/pay` and `/rl/quota`

**Today:** `GET /rl/pay` (`server.py:6328-6341`), backed by `validator/rl_pay.py` (373):
- `window_stat` (`:68-141`);
- paid fraction low/high (`:282-286`, commit `5bcf6aba`);
- GPU-hour estimate (`:336-354`).

**Change:** feed from per-period trained-token pay instead of window stats. Add `GET /rl/quota` (§1.3). Size ~150.

### 1.14 Reference miner

**Today**

- `mine` (`cli/main.py:3265-3482`).
- `MiningEngine.mine_window` (`miner/engine.py:708-1188`) is a sequential window follower: one prompt at a time, re-checking state and discarding work on change (`:87-152`, `:1102-1165`).
- Generation is blocked unless the local checkpoint equals the advertised one (`:968-987`).
- Checkpoint download happens in the foreground, and vLLM is rebuilt on each checkpoint (`vllm_generation.py:186-195`).
- The default generation backend is `transformers` (`constants.py:1635-1640`).
- GRAIL and TOPLOC commits are built on the miner (`engine.py:1722-1858`), with a second HF model for proofs.
- Env choice by advertised pay (`miner/rl_pay_client.py`, `engine.py:303-364`) is kept.
- Episode miner: `miner/episode_mining.py` (1,014). It has an allocation-following scheduler (`:69-75`, `:813-838`) and exits with 75 on a new revision (`:780-792`).

**Reuse:** `rl_pay_client.py`, `pick_env_and_prompt`, the submit transport (`submitter.py:92-226`, `:639-709`), the `episode_mining` scheduler skeleton as the template for a quota-aware single-turn scheduler, and `checkpoint_identity.py`.

**New or changed:** a lease client, background checkpoint download (and possibly a vLLM weight swap, otherwise a rebuild per checkpoint), and a Gumbel processor. Drop the proof GPU model, GRAIL, TOPLOC and logprobs.

**Delete:** ~1,000 lines. This covers window following, drand freshness, M-of-2M selection (`engine.py:434-454`, `:1536-1596`), precommit and receipt (`submitter.py:396-513`), and `/state` polling.

### 1.15 Episodes (multi-turn)

**Today**

- Per-turn draw by model-token offset: `miner/forced_draw.py:23-31` and `corpus_generate_server.py:191-205`. The runner fix `a033b07c` covers this.
- Plays 2M seeds and keeps an in-zone M (`miner/episode_group_miner.py:135-151`, `:243-290`, `:372-386`).
- Seed-to-precommit binding: `validator/episode_admission.py:113-182`.
- Intake: `validator/episode_intake.py`. Sandbox quota stack: `sandbox/*`.
- TOPLOC span proofs: `miner/episode_commit.py:182-222`.

**Change:** lease-then-open instead of precommit-then-open. Re-key `DrawBinding` by nonce. Submit exactly 16 episodes, with no M-of-2M. Drop TOPLOC. Size ~400. Built in §9 "Episodes and batched generation" (service wiring still open).

**Keep:** signed sandboxes, sessions, yield ledger and fleet placement.

### 1.16 Profiles, contracts and protocol versioning

**Today**

- `protocol/profiles.py`: `protocol_version` per profile (`:897-1250`); the generation contract digest (`:408-442`). `SamplingProfile` (`:22-27`) has no draw-scheme field.
- `protocol/service_contract.py` v2:
  - sampling kinds `public-seed-pool/v3`/`legacy/v1` (`:157-160`);
  - reward policy `exploration-first-scan/v1` with `price_bps, cap_bps, audit_bps, new_hotkey_audit_groups, ban_seconds` (`:149-151`, `:377-383`);
  - `cooldown_windows` (`:324`, `:350`);
  - `trainer-driven/v1` checkpoints (`:371-374`);
  - advice in windows (`:152-154`).
- Capabilities: `:161-164`.
- `ServicePolicyAnnouncement` with `pool_epoch`/`pool_randomness` (`submission.py:420-460`).
- Signature domains: `signatures.py:23-41`. Release contract: `protocol/release_contract.py:22-39`.
- Remote proof protocol: `remote_proof_protocol.py:20`.
- Payload schema version: `shared/training_payload.py:23-38`.
- Fill-closed profile gates (`constants.py:823-869`): the Teutonic suite was added as optional in `8684da01`.

**Change:** see §3.

## 2. Conflicts with DECIDED items

**C1. Exploration pay 15 %** (decision B, ruling 5) against the design's "Out-of-zone groups: no pay".

- The decision says: "Price: 15 % of a training group. Paid: the first miner to send a prompt never scanned before in the run." (`next-rl-run-decisions.md:67-69`).
- The design says: "Out-of-zone groups: no pay (they count for quota)." (`rl-async-design.md:166`, also `:26-27`).
- In code:
  - contract `exploration-first-scan/v1` (`service_contract.py:149-151`, `:381-383`);
  - `exploration.py:156-176` (price and cap);
  - `settlement.py:151-175` (proportional scaling, ruling 5);
  - `admission_policy.py:140-155`;
  - the NOTE in `rl_pay.py:48-60`.
- Options:
  - (a) Drop exploration pay. The in-zone price absorbs the failure cost through the controller (design §4.8). First-scan stats stay for the advisor and publication.
  - (b) Keep a first-scan bonus, paid per matched token at 15 %, only after an audit or probation.
  - (c) Pay out-of-zone groups in quota credit only (this is what the design says).
- With mandatory submission, out-of-zone work is no longer optional for a miner. That strengthens the case for (a) or (c), but it moves exploration risk onto the in-zone price.

**C2. Seeds 2×M with miner cherry-pick** (decision F, KEEP 10-08) against the design's nonce, no re-roll, and all 16 indices.

- The decision says:
  - "A miner builds its group from ANY M distinct seeds of the pool. Cherry-picking is intended: miners over-generate and keep the subset they like" (`decisions:159-161`).
  - "Public observation events list the chosen seeds next to the per-rollout rewards" (`:166-167`; also D `:120`).
- The design says:
  - "Same prompt + same checkpoint = same draw for every lease: no re-roll" (`design:100-102`).
  - "all 16 rollout indices required; the draw is bound to the rollout index" (`:129`).
  - "miners cannot select on the realised outcome" (`:104-105`).
- **Does the code allow choosing among realised draws? Yes, by design.**
  - The pool digest is fixed by (order, env, prompt, checkpoint, pool epoch = window, beacon) (`seed_pool.py:7`, `:182-231`).
  - The beacon is the window randomness: public in `/state` (`service.py:5084`), and asserted equal to it at `service.py:3792`.
  - A seed's draw is `sha256(DRAW_DOMAIN‖digest‖seed‖pos)` with no hotkey and no rank (`seed_pool.py:258-269`).
  - Any strictly increasing M-subset is valid (`:66-84`, `:237-246`, `validate_rollout_selection` `:272-297`, enforced at `admission_policy.py:72-75`).
- **The reference miners exercise this.**
  - `choose_public_seed_group` (`engine.py:1549-1578`) generates, scores and keeps a subset.
  - `ObservationClient.preferred_seeds` (`observation_client.py:379-395`) picks seeds from past seed rewards.
  - `EpisodeGroupMiner` plays all 2M and chooses an in-zone subset (`episode_group_miner.py:135-151`, `:372-386`).
- **What it allows:**
  - Selection on outcome: up to C(32,16) subsets of realised completions. The trained group is not an i.i.d. sample of π for the prompt, so the advantages are computed on a reward-selected subset.
  - Pre-generation by anyone, as soon as the window opens, because the pool is public and identical for all miners.
  - Duplicate work across miners on the same pool.
- Options:
  - (a) Drop F (the design).
  - (b) Keep over-generation inside a lease: the nonce gives 2M streams, all 2M are submitted, and the validator picks M by a rule blind to reward. This only adds cost.
  - (c) Keep F with a secret per-lease nonce. Pre-generation is gone, but the reward-selection bias remains.
- If F goes, also remove the `seeds` field from D and `preferred_seeds`, and retire `seed_pool.py` (297) and the `public-seed-pool/v3` capability.

**C3. Observation-log key** (decision A) against leases on (prompt, k).

- The decision says "keyed by `(order, dataset, prompt)` ... keeps `checkpoint`, `window` and a timestamp as attributes" (`decisions:54-56`).
- The code keys on `(order_id, environment, prompt_idx)` (`run_log.py:121-138`) and keeps the dataset in the payload; `observation_id` hashes the window (`:75-82`).
- These are compatible. The lease key (prompt, k) is a separate exclusivity key, and k is already an attribute. To change: `window` becomes `k`/`lease_id` in the attributes and in `observation_id`. "Order" needs a new meaning if the run is no longer an order with windows.

**C4. Audits** (decision C and ruling 2) against design §4.4.

- The decision says:
  - "sampled TOPLOC audit ... q = 15 % base ... 100 % audit for the first 100 exploration groups of a new hotkey ... Audit draw by drand **after** submission" (`decisions:85-89`).
  - Graded sanctions: deterministic forgery means a 24 h ban, statistical means re-probation (`:92-102`).
  - Ruling 2: "an exploration group is audited inside its own window" (`:228-230`).
- The design says "A random fraction (initially 10 %) gets the full forced-draw check; a failed audit is a heavy quota penalty or ban" (`design:126-127`).
- Differences:
  - The check is TOPLOC versus the forced draw.
  - q is 15 % versus 10 %.
  - The sanction is forfeiting exploration money versus a quota penalty.
  - Under Gumbel-only verification there is no TOPLOC mismatch and no hard CDF mismatch (`exploration.py:112-136`). Almost every failure becomes statistical, and only prompt fidelity, token authenticity and forged termination stay deterministic.
  - Audit timing: "verify against latest" means a late audit runs on a newer checkpoint.
- Options:
  - Keep the probation count (100 groups at 100 %) and the graded classes, redefined for Gumbel.
  - Use q = 10 % or 15 %.
  - Run audits within the max lag, or verify against the declared k (the design's §4.5 option).

**C5. Public observation segments** (decision D) against design §4.9 "Optional public difficulty map (open decision)".

- Decision D is DECIDED and implemented: `publication.py`, `observation_client.py`, and per-group rewards published within 60 s (`decisions:118-135`).
- The design lists publication as open (`design:179`, `:200`).
- Options:
  - (a) Keep D: rewrite the status values and drop the seed indices.
  - (b) Reopen it. With leases and mandatory submission, a public log lets miners avoid 0/16 and 16/16 prompts, which is the D rationale. It does not leak open leases, since only submitted groups are published.

**C6. Training cooldown** (decision E, ruling 6) against the design, which has no cooldown beyond "closed for k".

- The decision says "the training cooldown of in-zone prompts stays" (`decisions:62`), set manually per env and changeable live, with the advisor `N × p / Q × margin`.
- In code:
  - contract `cooldown_windows` (`service_contract.py:324`, `:350`), advice in windows (`:152-154`);
  - `CooldownMap` (`cooldown.py:74-121`);
  - `BATCH_PROMPT_COOLDOWN_WINDOWS=1_000_000` (`constants.py:1111`);
  - the episode precommit `task_in_cooldown` (`runtime.py:1093+`).
- The design only says "A prompt submitted at k is closed for k" (`design:94-95`). Without a cooldown, an in-zone prompt can be leased again at k+1 and trained every checkpoint.
- Options: express the cooldown in checkpoints or seconds, check it at lease time, and keep the advisor with Q equal to trained groups per unit time.

**C7. Steps per checkpoint and env quotas** (decisions G, N, ruling 3) against the design's "one optimizer step per batch" and "checkpoints published continuously".

- Decision N says "4 optimizer steps per checkpoint, not 16 ... about 8,000 rollouts per env per checkpoint ... LR 2e-6" (`decisions:268-281`).
- Decision G and ruling 3 say "per-env quota per pick, one optimizer step per pick" and "16 groups per env per pick stays a protocol constant" (`:172-173`, `:231-233`).
- The design says "One optimizer step per batch (decided)" (`design:149`) and "With one step per batch, the PPO ratio is 1" (`:158`). That holds only if a checkpoint is published after every step and v* is computed on it. With N's 4 steps per checkpoint, steps 2-4 train at pi_θ ≠ pi_verify.
- Things to decide:
  - publish cadence (every step versus every n steps);
  - batch size (N's 8,000 per env per checkpoint versus a buffer batch);
  - the meaning of "env quota" (the buffer target mix versus the emission share).

**C8. Truncation rule** (decision H, commit `437e6347`) against design §4.4 "at most one truncated rollout".

- The code skips the count limit on the service path and applies the robust rule to every assignment of capped rollouts (`admission.py:639-641`, `:1243-1247`). The `unproven` lane (`admission_policy.py:201-228`) has no design counterpart.
- Option: state in the design that "current env rules" means the robust rule, and decide what happens to unproven groups.

**C9. Forced-seed floors** (decision I) against the design's 0.95 Gumbel candidate. Decision I keeps floors 0.80/0.70 (`decisions:196-199`, `constants.py:1942`, `:1958`). The design supersedes this by intent, but it needs a recorded decision.

**C10. Async phase 3** (decision M): "only the current checkpoint is accepted. First measure the fraction of tokens outside the clip at a lag of 1/4/8/16 steps" (`decisions:313-317`). It is superseded by the design and by `docs/rl-off-policy-tests.md`. The Gumbel staleness measurement is still open (design §4.5).

**C11. Multi-turn TOPLOC and sandbox quota** (decision L and the phase-2 capacity rules).

- Decision L says "TOPLOC on the model spans" (`decisions:310`); the design says no TOPLOC on RL.
- The sandbox quota is a share of fleet capacity (water-filling, newcomer 10 %, low-yield 3 %), while the lease quota is per-hotkey AIMD. They must compose; design §4.10 says "apply unchanged".
- The "play from M" rules (`decisions:448-455`) assume 2M seeds (C2).

**C12. Exclusive leases** against the redesign notes (not DECIDED, but argued).

- The notes say "Rejected: option B, exclusive reservation. Squatting with many keys, and the validator's nearest miner wins every race on a popular prompt." (`rl-windows-redesign-notes.md:154-155`), and recommend non-exclusive option A (`:142-153`).
- The design picks exclusivity with 1 s interval fairness, mandatory submission, a small newcomer quota and the burn (`design:93-118`).
- This is a reversal, and the design should answer the many-keys squatting argument explicitly.
- Open point: with a deterministic nonce, a lease that expires leaves the same draw for the next lessee at the same k. The first holder knows that outcome, so expiry acts as a paid "release" whose only cost is quota halving, which falls on throwaway keys.

**C13. Window-scoped settlement and weights.**

- Ruling 5 (proportional split per window) and `weight_only` replay of per-window archives (`weight_only.py:216-225`) presuppose windows.
- Decision J (`decisions:203-204`) says a bad service archive makes the whole subnet abstain; this still applies to the new period archive.

**C14. Prompt rows reserved for corpus** (`599c194b`, `rl_prompt_ranges.py`). This is compatible: leases must refuse rows outside `rl_prompt_range`.

## 3. Wire and compatibility

### Pydantic models (`protocol/submission.py` and others)

**New**

- `RlLeaseRequest{env, prompt_ids[], k, hotkey, signature}`.
- `RlLeaseResponse{leases:[{lease_id, prompt_idx, nonce, k, expires_at}] | refused:"taken"}`.
- `RlQuotaResponse{hotkey, quota, in_use, reasons, lag_limit}`.
- `RlSubmitRequest{lease_id, k_generated, rollouts[16]{tokens (compressed), termination}, signature}`. It has no commitments, proofs or logprobs.
- `RlSubmitResponse{verdict: in_zone_accepted | stored | rejected, reason}`.

**Retired:** `SubmissionPrecommitRequest/Response` (`:335-380`), `FillClosedWindowState` (`:382-418`), `BatchSubmissionRequest` (`:212`) for RL, `ServicePolicyAnnouncement.pool_epoch/pool_randomness` (`:420-460`), the window fields of `GrpoBatchState` (`:462`) and `MinerState` (`:559-588`), and `EpisodePrecommitRequest` (`rl_routes.py:70`) if episodes lease.

**Changed:** `RolloutSubmission` drops the `token_logprobs` requirement (`:761`, `:941`, `:1014-1043`). The remote proof `ProofInput`/`ProofValues` replace `seed_u_values` with a nonce spec, add `vstar`, and drop `toploc_*`/`commitments` (`remote_proof_protocol.py:169-304`).

### Versions to bump

- A new protocol profile, with `protocol_version` bumped. `FORCED_SEED_DOMAIN` follows `PROTOCOL_VERSION` (`constants.py:1925`).
- The generation contract digest gains the draw scheme and the fp32-head flag (`profiles.py:408-442`).
- `service-contract/v3`, or new v2 kinds:
  - sampling `gumbel-lease/v1` replaces `public-seed-pool/v3`;
  - a reward policy per matched token (floor, no exploration or C1's choice);
  - a checkpoint policy with max lag S;
  - cooldown in checkpoints or seconds;
  - lease expiry per env;
  - quota parameters.
- `SUPPORTED_V2_CAPABILITIES` (`service_contract.py:161-164`).
- New signature domains for lease and submit (`signatures.py:23-41`), plus a release-contract capability (`release_contract.py:22-39`).
- `PROOF_PROTOCOL` v2 (`remote_proof_protocol.py:20`).
- `PAYLOAD_SCHEMA_VERSION` (`training_payload.py:23`): adds v* per position, per-group k and arrival, and allows several k per payload.
- New archive and settlement schema (`archive_schema_version: 2` at `service.py:7083+`).

### Deployment order

1. **`weight_only` replay first.** A merge to main is a weight-only deploy on every validator (`decisions:3-5`). The replay must read the new period archive before the owner validator writes one. A bad archive makes the subnet abstain (decision J).
2. **Proof workers** speaking `PROOF_PROTOCOL` v2 (keep v1 while the old run lives).
3. **Trainer** reading the new payload schema (v*, mixed k).
4. **Lease validator.** The design's P9 calls for validators that "accept both protocols". In practice this is a new run, a new order and a new profile, because windows and leases share no admission state. Dual-stack is realistic only as "corpus plus legacy windows" next to "async RL" on separate tasks.
5. **Miners.** The episode miner currently restarts on each revision (`episode_mining.py:780-792`), which is incompatible with continuous checkpoints.

Main runs the current production RL (fill-closed v6). The window code cannot be deleted from main while that run is live, so deletion is a final phase after cutover. Corpus tasks have their own settler periods (`corpus_validator.py:386-396`) and are unaffected.

### R2 layout

| What | Today | After |
|---|---|---|
| Training payload | `reliquary/[tasks/<t>/]training/window-<K>.npz`, tombstones, `step-cursor.json` (`training_payload_queue.py:50-117`) | Sequential `.../training/batch-<seq>.npz` with no tombstones or cursor, or per-group objects if the buffer is trainer-side |
| Window archive | `reliquary/[tasks/<t>/]dataset/window-<N>.json.gz` (`storage.py:44-53`) | A per-period settlement archive (new key), plus a group store for every group in and out of zone (new, e.g. `.../rl/groups/k-<k>/...` or a database) |
| Observation segments | `observations/run-<id>/...` | Unchanged; the `window` and `seeds` fields change |
| `S_k` commitment and reveal | none | New, e.g. alongside `reliquary/training/candidate-manifest.json` or `.../rl/nonce/<k>.json` |
| Checkpoints | `reliquary/checkpoints/<rev>/` and the manifest | Unchanged |

## 4. Phases (refining design §7)

| Phase | Content | Depends on | Size |
|---|---|---|---|
| D0 | Settle C1-C8, C12 (exploration pay, F, cooldown units, publish cadence, audit class, truncation, unproven lane, exclusivity) | — | decisions |
| P0 | Rebase `feat/forced-seed-runner-fix` on `b9afcb13` and merge into `feat/next-rl-run` | user go | done, 607 |
| P1a | `gumbel_draw` protocol module; Gumbel runner processor (port of `lab/gumbel.py` noise); HF fallback and BFT phase 2; verifier stats with v* and score gap; remote proof v2; episode per-turn draw re-keyed; fp32 head option | P0 | ~1,000 |
| P1b | GPU: staleness under Gumbel (0/1/2/4 steps); per-rollout floor; max lag S | P1a | measurement |
| P2 | Lease service (SQLite, restart-safe), interval fairness, exclusivity, expiry, `S_k` commit/reveal, AIMD quota, `/rl/lease`, `/rl/quota`, signatures; cooldown at lease time | D0 | ~1,200 |
| P3 | `/rl/submit`: admission refactor (no early OUT_OF_ZONE return), lane classification, in-zone proof through `proof_scheduler`, out-of-zone audit (exploration ledger decoupled from pay), group store, run log and publication rewired | P1a, P2 | ~1,500 (plus ~1,000 moved out of `batcher.py`) |
| P4 | Buffer (arrival order, full batch, max-lag drop), payload schema bump, trainer loop without journal or cursor, accumulator without checkpoint reset, publish cadence, continuous `CheckpointIntake` swap | P3 | ~800 |
| P5 | Substitution, mask and current loss switch; v* threading; telemetry; drift gate | P1a, P4 (payload) | ~250 |
| P6 | Pay per matched token with floor; settlement period; `weight_only` replay of the new archive; async price controller; `/rl/pay` | P3, P4 | ~900 |
| P7 | Reference miner: lease client, quota scheduler (from the `episode_mining` skeleton), background download and vLLM rebuild or weight swap, Gumbel generation, GRAIL/TOPLOC/proof model dropped; episode miner leases and stops restarting on revision | P1a, P2 | ~1,200 new, ~1,000 deleted |
| P8 | Qualification on GPU testbed: honest, fp8, int4, stale, Sybil, squat, expiry-release | all | — |
| P9 | Deploy: `weight_only` replay via main, then proof workers, then trainer, then the lease validator on a new run or order, then miners | P8 | — |
| P10 | Delete the window path after the current production run ends (see list below) | P9 | ~−12,000 |

P2 and P1a can run in parallel. P5 can start once P1a fixes the v* format.

**P10 deletion list**

- `fill_closed_*`, `fill_window`, `journal`.
- Pick pacing, rotation gate and the `candidate_shortfall` reason.
- `fifo-ingress/v1`.
- Precommit and receipt routes, auction reveal, legacy `/submit`.
- The `/state` and `/miner-state` window fields.
- `seed_pool.py`, GRAIL, `crypto.py`, RL TOPLOC.
- `ForcedSeedVLLMProcessor`.
- The window price snap.
- The constants at `constants.py:815-1069`, `:1751-1763`.

## 5. P2 as built (branch `feat/rl-leases`)

- `reliquary/validator/rl_leases.py`: `LeaseBook` (SQLite, WAL, `synchronous=FULL`, file `0600`; partial unique
  index = one live lease per (env, prompt, k)), `LeaseCollector` (interval batching, `random.SystemRandom` shuffle
  of all (request, prompt) claims), nonce `H(NONCE_DOMAIN, S_k, env, prompt, k)`, commitment
  `H(COMMIT_DOMAIN, S_k, k)`, reveal when `k < latest - max_lag` (`SECRET_REVEAL_LAG = 16`); leases only on a k
  whose S_k is unrevealed. Refusals: taken / cooling / blacklisted / quota / stale. A lease due while the validator
  was down is extended to `now + restart_grace_s` at boot.
- Prompt lifecycle (design §4.5), in the same file: `classify_group(rewards, truncated)` (population sigma >=
  `SIGMA_MIN_V4` = 0.24, at most `MAX_TRUNCATED_PER_SUBMISSION` truncated, then mean >= 0.5 too easy, else too
  hard); `LeaseBook.record_outcome(env, prompt, k, rewards, verdict)` (P3 calls it after grading) appends to
  `rl_prompt_scans` (with `first_scan` for P5) and sets `rl_prompt_state`: too easy = blacklisted for the run, every
  other verdict = cooldown until `latest + cooldown` checkpoints (`LeaseBookConfig.cooldown_of`, default the
  profile's per-env `prompt_cooldown_windows_for_environment` counted in checkpoints, read at each outcome so a live
  operator change applies to the next cooldown, decision E; `CooldownMap`'s half-open rule; C6). `prompt_state(since)` is the incremental miner view. The former
  `checkpoint_cooldown(maps)` predicate is gone: the cooldown now lives in the lease DB.
- `reliquary/validator/rl_lease_quota.py`: `QuotaController` (AIMD cap with slow start, halving on expiry,
  Little's-law target `headroom x throughput x latency` once steady, persisted in the lease book's file).
- `reliquary/validator/rl_lease_routes.py`: `POST /rl/lease` (signed, `build_rl_lease_binding` in
  `protocol/signatures.py`), `GET /rl/quota?hotkey=`, `GET /rl/lease/commitments`, `GET /rl/prompts/state?since=`
  (public, no hotkey); `ValidatorServer.mount_rl_lease_router`. Not mounted by the CLI/service yet (P3);
  `LeaseBook.submit` and `LeaseBook.record_outcome` are the hooks P3 calls.

## 6. P3 as built (branch `feat/rl-submit`)

New code next to the window path (nothing of it changed; `materialize_admission_problem` is now a thin wrapper
over the module-level `batcher.admission_problem_materials`, shared with P3).

- Wire (`protocol/rl_submit.py`, `signatures.build_rl_submit_binding`): `POST /rl/submit {miner_hotkey, at,
  lease_id, rollouts[16]{index, tokens, force_span}, signature}`; `tokens` = base64(zlib(uint32 LE)) of the
  completion only (the validator renders the prompt, so prompt fidelity holds by construction); the signature
  covers `rollouts_digest` of the rollouts as sent, so nothing is inflated before it verifies; bounded inflate.
  No `k_generated`: the lease's k is the generation checkpoint (the nonce binds it). Route in `rl_lease_routes.py`
  (mounted with `submissions=`), own 600/min bound, grading off the loop (4 at once).
- `validator/rl_submit.py`: `GroupGrader` (single-turn only; termination by `admission._classify_termination`,
  force spans by `_force_span_valid`, rewards by `EnvironmentSpec.score_many`, length by the env cap) and
  `RlSubmissions`: `check_submission` -> grade -> one transaction (lease closed, `record_outcome`, `rl_groups`,
  buffer or audit draw). Store: `rl_groups` (private, hotkey, verdict, grades, first scan, audit/training state)
  next to `rl_prompt_scans` (public log). TODO: the decision-D R2 segment writer reads `rl_prompt_scans` by `seq`.
- `validator/rl_buffer.py`: `TrainingBuffer` (SQLite, same file): `take(n)` any env in arrival order, `release`,
  `depth`, `drain_arrivals` (queued completion tokens per env: the prices'
  supply); taken groups requeue at reopen. `SubmittedGroup` carries env, k, nonce, arrival, hotkey,
  prompt tokens, completions, force spans, rewards.
- Audit (§C, design §4.4): draw after submission with `SystemRandom`; probation = < 100 passed out-of-zone audits,
  or < 20 passes since the last failure; deterministic -> `LeaseBook.ban` 24 h (no lease: refusal `banned`, no
  submission); statistical -> re-probation, ban when > 3 of the last 20 non-forced verdicts failed. Queue order:
  past-probation hotkeys first, oldest first. `ProofForward(SubmittedGroup) -> AuditResult(passed | None,
  failure_class, rollout_match)` is the Gumbel verifier's slot; `run_audits` drains.
- Hooks for P4/P5: `report_training(lease_id, rollout_match, accepted, failure_class=None)` (accepted -> `trained`
  event with per-rollout match and token counts; rejected with a class -> sanctioned like an audit; without a class,
  e.g. staleness -> unpaid only). `rl_events` (`first_scan` 25 % payable: at once when not drawn, at pass when
  drawn; `first_scan_forfeited`; `trained`; `untrained`; `ban`) is the P5 ledger's input (`events(since)`).
- Quota: in and out of zone are delivered (`record_submission`); a group refused for integrity (bad termination,
  tampered span, too long, out of vocab, unscorable) closes the lease, records no scan, and halves the quota
  (`record_expiry(cause="rejected")`). A schema error (indices, decode) is refused before the lease is touched; a
  grader outage answers 503 with the lease still open.
- Resolved ambiguities: probation counts PASSED audits (decisions-doc rule in force), not submitted groups; training
  verdicts count in the recidivism window, never as probation passes; an inconclusive audit forfeits the first-scan
  pay without sanction; a lease that expires while its group is being graded is refused (the arrival time is used,
  but a concurrent sweep can still expire it: grading is short next to the expiry).

## 7. P5 as built (branch `feat/rl-pay`): pay, async prices, `/rl/pay`

New code next to the window path (`emission_price`, `settlement`, `rl_pay`, `weight_only` unchanged).

- `validator/rl_ledger.py`: `RlPayLedger` reads `RlSubmissions.events(since)` (cursor persisted) into `rl_accruals`
  (same SQLite file). `trained`: flat pay, matched tokens of every rollout with match >= `floor` (0.95; re-checked
  even though the trainer masks them) x the env's price per token at ingestion. Matched tokens = the trainer's
  exact counts (`report_training(..., matched_tokens=[...])`, new optional argument, bounded by the completion
  length), else `floor(match x completion tokens)`. `first_scan`: `fraction` (0.25) x **an in-zone group's pay**,
  defined as the env's price per token x the mean matched tokens of the env's last 200 trained groups in the ledger
  (16,384 before any); never the out-of-zone group's own length (a 0/16 group is cheap to make long). P3 already
  emits it at once when not drawn, at pass when drawn; failed or inconclusive audits emit `first_scan_forfeited`
  (no money). `ban`: unsettled accruals of the hotkey forfeited, and any payable event dated before the ban's end;
  settled periods are not clawed back.
- Epochs = the drand periods of `corpus_periods` (4,320 s = one Bittensor epoch). `settle(now)`: a period closes
  `slack_s` (120 s) after its end; per hotkey sum, normalised to the task cap (`scale = min(1, cap / total)`, the
  rest burns); one archive per period (`reliquary/rl-async-period/v1`: `work_period`, `entry_period` by
  `cp.entry_for`, `rewards_by_hotkey`, plus the nominal accruals, scale and tokens). An accrual landing after its
  period settled goes to the next open period. `publish(archives)` writes them through an `R2PeriodArchives`-like
  `write` (an entry period already begun is moved forward first, as the corpus settler does). On chain: the
  unchanged `weight_only._period_weights` pays these archives (per-period EMA, per-task cap, `CATCHUP_ENTRIES`)
  once the async RL task is declared with `params.settlement = "period-ema-v1"`; no window archive is needed.
- `validator/rl_price.py`: `AsyncPriceController` keeps the window controller's `PriceParams` (base =
  production values with `cap = 4.0`), `PriceState`, `PriceDecision` and walk; ticks on the validator's timer
  (`tick_s` 300 s, `due`). Trainer speed (`note_speed` per step: completion tokens / seconds worked, sums decayed
  with a one-tick half-life, `rl_trainer_speed`); per env r = queued tokens per second (`drain_arrivals`) /
  (target share x speed): r < `low` (0.95) rises x1.10, r > `high` (1.15) or refused by the global cap
  (`LeaseBook.drain_full`) descends (`decay` scaled to the tick), else holds; no speed for `stale_ticks` (3): hold.
  Clamped to [0.05, 4.0]. Per-token price = multiplier x cap x target share / `planned_tokens_per_period` (at 1, the planned
  consumption at target mix pays exactly the cap). Snap and breaker dropped (no window to fill). State in
  `rl_prices` (restart-safe).
- `GET /rl/pay?hotkey=` (`build_rl_lease_router(pay=ledger)`): per-env multiplier, token price, per-Mtok price,
  target, regime and last signal; per-hotkey accrued / paid / forfeited / banned_until; the last settled periods.
  `ValidatorServer.mount_rl_lease_router` now replaces an app route at the same path and method, so mounting it
  supersedes the window board's `/rl/pay`.
- Wired since: the service loops and the registry entry (§9 "Service wiring as built").

## 8. Reference miner as built (branch `feat/rl-miner-client`)

- `reliquary/miner/rl_async.py`, CLI `reliquary mine-async` (the window miner `mine` is untouched).
  - `PromptPlanner`: per env, fed by `GET /rl/prompts/state?since=` (cursor, `more` pages); skips blacklisted
    prompts, cooldowns until `latest >= until_k`, and prompts held, submitted or `taken` until their scan arrives
    (or one lease lifetime of the env, `lease_expiry_s`; a held lease: its own expiry). Default order: never scanned, then by share of in-zone scans; `priority` overrides it.
  - `AsyncMiner`: open leases <= `min(quota, --max-open)`, waiting leases <= `--queue-groups`, requests of
    <= `--lease-batch` at the held k; refusals: `taken` -> wait for the scan, `cooling`/`blacklisted` -> table,
    `quota` -> quota := open, `stale` -> no lease until newer weights. Earliest expiry generated first; every generated
    group is submitted with retries until its expiry (a resend gets the original verdict); open leases journaled
    for restarts and recovered from `GET /rl/lease/mine` at start.
  - Checkpoints: `CheckpointSync` on a CPU copy in a thread; the copy is handed to the generator between two batches
    and not patched again before (`swap_if_ready`). Submissions carry `k_generated`.
  - `VllmGroupGenerator`: calls `VLLMRolloutGenerator.generate` per lease (`draw_context` =
    `protocol.rl_draw.lease_draw_context`: nonce as randomness, `k<k>` as checkpoint hash); weights pushed with `LLM.collective_rpc` + the model's `load_weights` (needs
    `VLLM_ALLOW_INSECURE_SERIALIZATION=1`, untested on GPU), fallback `--weight-swap reload` (HF dir + engine rebuild).
- Aligned at integration (§9): the submit wire and signature, the draw keying, lease recovery. Still open: a
  multi-prompt vLLM call.

## 9. Integration (branch `feat/rl-async`, from `origin/main` @ `fde75c0c`)

One branch carries the whole async RL work on top of current main.

### What landed

- Forced draw, rebased from `fa0f3a76` by cherry-pick (no conflict): `feat/forced-seed-runner-fix` (runner-keyed
  vLLM position, P0), `feat/gumbel-forced-draw` (Gumbel-max draw, `sampling.draw` / `sampling.lm_head_fp32`, v10
  profile, protocol 10, validator check, miner draw, scripts), `feat/substitution-loss` (substitution loss, v* from
  the trainer's logits; the batcher carries the verifier's uniforms as `draw_u_flat` / `draw_u_off` float64 arrays).
- Merged from `b9afcb13` (already in main): `feat/rl-pay` (P2 leases + lifecycle, P3 submission / buffer / audits,
  P5 ledger / async prices / `/rl/pay`), then `feat/rl-miner-client` (checkpoint deltas, HF backup, reference miner).
  The shared `feat/rl-leases` commits come in once.
- Conflicts: only this file (sections 6-7 of `feat/rl-pay` and 6 of `feat/rl-miner-client` kept, the miner
  renumbered 8) and `docs/rl-async-design.md` (add/add: the latest draft kept).

### Seams fixed

- Merge regression: main made the window price rise opt-in (`RELIQUARY_EMISSION_PRICE_RISE`, default 1.0), which
  silently removed the async controller's rise. `rl_price.ASYNC_PRICE_RISE = 1.10` pins it.
- Merge drift: main added two envs to v9, so the v10 Gumbel profile (v9 + Gumbel draw) compiles to a new
  contract; its golden digests are re-pinned.
- Submission wire: the miner sends the validator's body (`protocol/rl_submit.py`) signed with
  `build_rl_submit_binding` over `rollouts_digest`; the draft `u32le-zlib/1` encoding and binding are deleted.
  Generators may return `(tokens, force_span)` per rollout. `k_generated` is optional, unsigned and ignored.
- Idempotent submit: `RlSubmissions.submit` answers a resend of a graded lease with the stored verdict (200,
  `resent: true`), also when a concurrent resend closed the lease first; another hotkey still gets 403.
- Draw keying: `protocol/rl_draw.py` (`lease_draw_context`, `lease_uniforms`): randomness = nonce hex, prompt,
  `checkpoint_hash = "k<k>"`, t = completion position. The miner's `draw_context` and
  `SubmittedGroup.draw_uniforms(i)` both call it; `tests/unit/test_rl_lease_draw.py` checks miner (through
  `forced_seed_extra_args` and the processor's `_uniform`) == validator == trainer for one granted lease, and the
  Gumbel picks. `SubmittedGroup` stores no uniforms: nonce, prompt, k, rollout index and completion length are the
  seed, so the trainer recomputes them (to set `rollout._forced_draw_uniforms`).
- `GET /rl/lease/mine?hotkey=&at=&signature=` (`build_rl_lease_mine_binding`, skew-bounded, signature answered once,
  30/min): the caller's open leases. The miner calls it at start (`AsyncMiner.recover_leases`).
- Lease lifetime: lease answers carry `lease_expiry_s` (the env's), `/rl/quota` and `/rl/lease/mine` a map of every
  env's; the planner waits that long on a `taken` prompt (30 min only while unknown).
- Model files: `DeltaPublisher(model_dir=)` uploads config / tokenizer / chat template with each anchor
  (`anchors[].model_files`, presigned, pruned with the anchor); `CheckpointSync(model_dir=)` keeps the newest set
  (SHA-256 checked); `mine-async` uses `--base-model`, else the initial checkpoint's dir, else the synced files.
- Audit `ProofForward` (`validator/rl_audit.py`): per rollout, `verifier.proof_forward_rows` (the proof path's
  batch-1 forward and lazy fp32 head, now shared with `verify_commitment_proofs`) and the verifier's
  `_gpu_seed_consistency_diagnostics` with `draw_uniforms(i)`, force span excluded. Match = exact / stochastic.
  Pooled match under `RL_ROLLOUT_MATCH_FLOOR` (0.95 placeholder, also the ledger's floor) -> `statistical`; a
  rollout on the draw (match >= floor) with more than `RL_AUDIT_FORGERY_MISMATCH_TOLERANCE` (1) hard mismatches at
  near-deterministic positions -> `deterministic`; no model -> `passed=None`. Hook: `run_rl_audits(model,
  submissions, limit=8)`, to schedule from the service with its verify model.

### Remaining

Everything below needs a GPU testbed (P8 qualification) unless marked otherwise.

- The v10 Gumbel profile and fp32 head end to end: vLLM miner picks vs the validator's proof forward (audits) and vs
  the trainer's forward (FSDP2, packed, fp32 head) on the 9B; calibrate `RL_ROLLOUT_MATCH_FLOOR`,
  `ASYNC_GROUP_FLOOR` and `RL_AUDIT_FORGERY_MISMATCH_TOLERANCE` on honest lag-0/2 rollouts. No separate
  terminal-pick check yet (the last token is judged like any position).
- The miner's in-place weight push (`collective_rpc`, `VLLM_ALLOW_INSECURE_SERIALIZATION`) and its sync time on
  the 9B; the batched call's throughput (one `generate_batch` for `batch_groups` x 16 requests, see below).
- `train-async` on the 9B: single card and FSDP2 under torchrun (rank-0 gather per step as `state_fn`, the
  followers of `train-worker` reused), snapshot + delta size and time per step, anchor interval, HF backup upload
  and rotation under `ReliquaryForge/`.
- The validator's audit model, local (`RELIQUARY_RL_AUDIT_DEVICE`) or on the proof worker (V1, see "V1 operation"
  below): following the deltas on the card next to the proof replicas (memory: one more bf16 copy plus the CPU
  state) and the audit throughput (probation audits 100 % of a newcomer's out-of-zone groups; one worker forward
  at a time, batch 1). Without either every audit is inconclusive (first scans unpaid).
- Lease expiry (`rl_lease_reference_tokens_per_s`, margin; for episode envs the wall budget and `EPISODE_SESSION_SLACK_S`, or `rl_lease_expiry_s` overrides) and starting quota numbers.
- Multi-turn envs: done (episodes and their service / miner wiring, below). Done since: the scan log on R2,
  discovery and the trainer listener (`feat/rl-async-publish`, below); the remote signer's narrow operations for the
  checkpoint index and the scan log's head and pages ("V1 operation" below).
- Not GPU, still open: the decision-D R2 segments of `rl_prompt_scans` (the `ObservationPublisher` reads a service
  runtime's window events; the scans are served by `/rl/prompts/state` meanwhile); the remote signer has no index
  operation (the async validator signs `/rl/checkpoint` with the local wallet and refuses `RELIQUARY_SIGNER_URL`);
  axon publication of the async validator (miners use `--validator-url`).

### Episodes and batched generation as built (branch `feat/rl-async-multiturn`)

Design §4.10: leases, mandatory submission (16 episodes, every index, no M-of-2M), quota and lifecycle unchanged;
signed sandboxes and the per-turn forced draw kept, the draw keyed by the lease; no TOPLOC.

- Wire (`protocol/rl_submit.py`): an episode rollout adds `episode = {spans, stop, transcript}`; `tokens` is the
  whole completion after the validator's prompt render (model turns + observations), `spans` the model turns
  (completion-relative, read strictly by `episode_spans`: ordered, the completion starts and ends on a model
  turn). `rollouts_digest` adds `0x02 || episode_digest` (canonical JSON) per episode; single-turn digests unchanged.
  The route (`RlSubmitRollout.episode`) refuses a group mixing episodes and single-turn rollouts.
- Draw (`protocol/rl_draw.py`): `t` counts model tokens across turns; `lease_extra_args(..., base_offset)` is the
  turn request's args, `model_positions(len, spans)` the completion positions in draw order.
  `forced_draw.LeaseDrawBinding` replaces the window seed binding in the forced generate engine (unchanged: it
  passes the session's model tokens so far as `base_offset`). `SubmittedGroup.spans` (packed, None for
  single-turn) makes `draw_uniforms(i)` one per model token; miner == validator == trainer is tested per turn.
- Engagement: `rl-lease:{lease_id}:{index}` (`service_episode.rl_lease_engagement`); sandbox engagement book
  `sandbox/rl_engagements.RlLeaseEngagements` (kind `rl_lease`): lease open, the caller's, unbanned
  (`LeaseBook.check_submission`, again under the issuer lock), index < 16, one session per (lease, index)
  (`exclusive`), checkpoint `k<k>`, task = leased prompt, budgets as the window book.
- Validator (`validator/rl_episode.LeaseEpisodeChecker`, through `GroupGrader(episodes={env: checker})` and
  `grade_episodes`): per rollout `episode_admission.verified_transcript` (factored out of the window checker:
  signatures, chain, graded final, deadline, record 0) against the lease binding, then
  `episode_shape_refusal` (signed parse: every observation is its signed record's rendering; turn shape, budgets,
  termination, limit stops) after prompt fidelity; one session twice refused; reward = final record. Any refusal
  rejects the group (integrity: no observation, quota halved); a stale machine directory raises
  `GraderInfrastructureError` (lease left open). `rl_groups.completion_tokens`, `report_training`'s bound and the
  ledger's `completion_tokens` count model tokens only.
- Trainer (`training_group`): an episode is the window's Episode v1 shape (`rollout.episode.assistant_spans`,
  `_validated_assistant_spans` absolute): observations get no loss, no substitution and no match; uniforms and the
  `token_logprobs` placeholder are one per model token. Audit (`rl_audit.rollout_draw_diagnostics`): one forward
  over the whole sequence, only model positions judged, each against its model-token draw.
- Miner: `rl_async.EpisodeRollout` encodes into the wire; `miner/rl_async_episodes.EpisodeLeaseGenerator`
  (`agenerate`, awaited by `AsyncMiner.generate_once`) plays a lease's 16 episodes through the window machinery:
  `rl_episode_client.LeaseEpisodeRunner` (open body `rl_lease`, lease-bound transcript precheck
  `lease_transcript_refusal`, opens stop at the lease's expiry), `episode_group_miner.play_bound` (factored out of
  `_play_seed`) with `LeaseDrawBinding`, `episode_commit.episode_admissible`; a machine-signed abort is replayed
  once; one unusable episode fails the whole lease (graded sessions withdrawn; never a choice among episodes).
- Batched generation (task 2): `VLLMRolloutGenerator.generate_batch([(prompt, per-request extra_args, cap)])`
  sends every request of a batch in ONE `LLM.generate`; `VllmGroupGenerator.generate` builds `batch_groups` x 16
  requests, each with its own lease's `lease_extra_args`, read by request id in the runner-keyed processor (test:
  rows shuffled, each request still draws its lease's stream).
- Tests: `tests/unit/test_rl_async_episodes.py` (15; 6 need `reliquary_sandbox`).

Episode wiring as built (branch `feat/rl-async`, after the multi-turn merge):

- Policy source (`validator/rl_async_episodes.py`). The window takes an env's episode policy from the order
  contract; the async path has no order, so the task's registry entry carries the same blocks:
  `params.rl_episode_envs = {env: {episode, catalogue?, version?}}` (`EPISODE_ENVS_PARAM`), validated with the
  contract's own rules (`service_contract._validate_episode`, `validate_catalogue`; a sha256 `version`), every env
  in the entry's `env_split`. The policy's seed pool is 2 x 16 (it only sizes the session policy's caps). A bridged
  env (a `catalogue`) is built as the window builds it (`BridgedEpisodeSource`; the installed identity must be the
  block's `env_package`, its manifest the `version` when given) and registered as a catalogue spec; an env without
  a catalogue must be a registered signed-episode env. Declared with `reliquary tasks create --rl-async
  --rl-episode-envs <file.json> [--lease-expiry env=s,...]` (`rl_async_params(episode_envs=, lease_expiry_s=)`).
  Miners read the blocks back from `GET /rl/episodes` (the blocks + each env's lease lifetime; 404 without
  episode envs): the validator serves exactly what it grades with.
- Validator (`rl_async_service`): `run_rl_async_validator` loads the episode envs off the loop
  (`load_episode_environments`), needs `RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE` / `_KEY_ID` (refuses to start
  without), and `wire_episode_sandbox` builds `rl_async_episodes.build_rl_lease_sandbox`: the window's pieces minus
  windows: the token key, the machine directory and fleet (resources required in production), a `SessionBook` and
  `SessionIssuer` whose only engagement book is `RlLeaseEngagements` (kind `rl_lease`, over the lease book),
  sessions stored under `sandbox_store.RL_LEASE_SESSION_PREFIX` (`reliquary/sandbox/rl-lease-sessions/`, never
  restored by window mode or the corpus), the session open / close routes under `/rl` (no precommit route), one
  `LeaseEpisodeChecker` per env added to the grader (`GroupGrader.add_episodes`). The session policy lifts the
  per-key live and hourly-open caps (`NO_PER_KEY_CAP`: the lease quota bounds a hotkey, 16 sessions per open
  lease, `exclusive`), keeps the daily abort cap scaled for 16-session groups. Started before serving (directory
  read, sessions restored with retries, fail closed); background: the fleet poll and `maintain_forever`;
  drained at shutdown. `SandboxEngagement` (the open body's model) now admits `kind: rl_lease` with its
  `lease: {lease_id, rollout_index}` (the route refused it before: the multi-turn branch never went through HTTP).
- Not ported (documented in `rl_async_episodes`): the window's session settlement (sessions marked submitted /
  handed back on a verdict) and the yield ledger / yield quota. The engagement makes a transcript single-use
  (one session per (lease, rollout), the lease closes at its first submission) and the lease quota decides how
  many sessions a hotkey holds; settlement and yield are the window's capacity accounting, which the lease quota
  replaces. Sessions end by their close (final record) or expiry. Also not ported: the window's boot check that
  every renderer stop is in the proof's stop set (there is no TOPLOC proof per span on the async path; the
  trainer's forward and the audits judge every model position) and the degraded-env served watch.
- `/rl/submit` body: `MAX_SUBMIT_BODY_BYTES` (8 MiB) for a single-turn group, always (also gzipped, also on a
  validator serving episodes); `MAX_EPISODE_SUBMIT_BODY_BYTES` (64 MiB inflated) for an episode group on a
  validator serving episode envs (`max_episode_body_bytes`, `rl_async_service.episode_body_cap`), body timeout
  120 s there. `Content-Encoding: gzip` is read with a bounded inflate (`protocol/rl_submit.inflate_body`: one
  stream, no bomb); another encoding is 422. Each transcript within `MAX_TRANSCRIPT_BYTES`. The miner gzips any
  episode group (`HttpRlTransport.submit`; the signature is over the rollouts, never the encoding); single-turn
  bodies are sent as before.
- Lease lifetime: an episode env's default is `max_episode_tokens / rl_lease_reference_tokens_per_s x margin`
  (the single-turn rule on the episode's token budget) `+ budgets.wall_s` (tools may use the whole session) `+
  EPISODE_SESSION_SLACK_S` (300 s: a queued open, image prepare, transcript upload); one session's budget, not
  16 (the episodes run side by side). `params.rl_lease_expiry_s = {env: s}` (`LEASE_EXPIRY_PARAM`) overrides
  any env's lifetime, single-turn ones included; an override outside the split is refused. Example: 4096 tokens at
  20 tok/s x 2 + 3600 s + 300 s = 4309.6 s.
- `mine-async` (`miner/rl_async.miner_generator`): the miner asks `GET /rl/episodes`; its `--environments` the
  validator serves as episode envs go to `rl_async_episodes.build_episode_generator` (the window
  `build_episode_stack` minus precommits and the TOPLOC prover: one forced engine for all of them, caps checked
  equal, the task prompts from the validator's own source, `LeaseEpisodeRunner` per lease over `HttpRlEpisodes`,
  `--generate-port`, `--episode-max-num-seqs`, `--episode-gpu-memory-utilization`, `--harness-env[-file]`), the
  others keep `VllmGroupGenerator`; both under `MixedGenerator` when one process mines both (they share the GPU).
  An episode env's prompt range is its whole taskset.
- Checkpoint swaps for the episode engine: restart, not in place (`EpisodeEngineHost`, `EpisodeWeights`).
  `VllmTurnCore`'s `LLM` is stepped by the `GenerateEngine` thread with hidden-state capture; a `collective_rpc`
  push while that thread owns the engine is not supported by vLLM. `AsyncMiner` swaps between batches only (no
  episode in flight), so `load_weights` writes the state as an HF dir, stops the engine (joins its thread),
  releases the core and builds a new core + engine on the dir; the loopback generate app is built once over the
  host and reaches the engine of the moment. Cost: one full model load per checkpoint (`--batch-groups` amortizes
  it; measure at GPU qualification). The core still computes TOPLOC span proofs (deployed defaults) the async
  path discards.
- Tests: `tests/unit/test_rl_async_episode_service.py` (11; 3 need `reliquary_sandbox`): registry blocks and
  lease lifetimes, the CLI declaration, gzip bounds, the per-kind body cap (single-turn > 8 MiB gzipped: 413; bad
  gzip and other encodings: 422; no episode env: 413 and no `/rl/episodes`), an end-to-end CPU run through the
  async validator app (`AsyncMiner` over `HttpRlTransport` -> `/rl/episodes` and the lifetime -> lease -> 16 real
  opens on `/rl/sandbox/sessions` (issuer, `RlLeaseEngagements`, fleet over a fake directory and capacity report)
  -> transcripts signed by a fake machine for the issued tokens -> real closes -> gzipped submit ->
  `LeaseEpisodeChecker` -> in zone, buffered with its spans, model tokens counted -> a reopen refused), refusals
  (second open of a rollout, rollout 16, another hotkey's lease), the engine restart per checkpoint, the caps
  check, and the episode / single-turn routing.

Remaining (episodes), GPU only: the forced episode engine on the 9B (vLLM `VllmTurnCore` with the lease draw,
restart time per checkpoint), a real gateway and harness under `mine-async` against the async validator (the
CPU e2e fakes the machine and the harness), the episode lease lifetime and slack calibrated on real sessions,
and the audit / trainer forward on episode groups (model positions only) on the 9B.

### Service wiring as built (branch `feat/rl-async`)

One switch: `RELIQUARY_RL_MODE=async` (default `window`: `validate` runs the window service exactly as before; a
task declared for period settlement refuses to start in window mode).

- Registry: `reliquary tasks create --task-id <t> --profile-id <p> --cap <c> --env-split math=0.6,code=0.4
  --rl-async --planned-tokens-per-period <N> [--lease-tokens-per-s <s>]` (`cli.with_rl_async_params`): mechanism
  `rl-discovered-price`, `params.settlement = "period-ema-v1"` (the unchanged weight setter pays the period
  archives), `env_split` = the trainer's target mix, `rl_planned_tokens_per_period`, `rl_lease_reference_tokens_per_s`,
  `rl_lease_expiry_margin`, optional `rl_price_tick_s` / `rl_max_pending_groups` (the lease book's global cap, default 256).
- Validator (`validator/rl_async_service.py`, `run_rl_async_validator`, branched in `validate` after the corpus
  branch): one process like the corpus validator; `build_rl_async` puts the lease book, quota, collector,
  submissions + buffer, price controller and ledger on `<state dir>/rl-async.sqlite3`; `RlCheckpointBoard` over
  `rl/<run>/index.json` in `reliquary-public` (run = `RELIQUARY_RL_RUN`, default the task id); `mount_rl_async`
  mounts the lease router (with `/rl/submit`, `/rl/pay`, `/rl/lease/mine`, `/rl/prompts/state`) plus
  `/rl/checkpoint` and `/rl/observations`, through `ValidatorServer.mount_rl_lease_router` (with
  `RELIQUARY_RL_TRAINER_TOKEN`, the trainer link on its own listener); registration gate as the corpus validator; the weight setter thread as
  in window mode. Loops (`rl_async_loops`): expiry sweep (5 s), audits (2 s, 8 per pass, `AuditModel` at the latest
  checkpoint or inconclusive without a device), ledger ingest / settle / publish through `R2PeriodArchives` (30 s),
  price tick when due (the trainer ticks per step). `RlSubmissions.run_audits` now skips a raising forward and
  concludes it inconclusive after `AUDIT_MAX_ATTEMPTS` (3).
- Trainer link (`/rl/trainer/*`, bearer token; client `trainer/rl_link.TrainerLink`, one object for the trainer's
  `buffer`, `submissions`, `book` and `prices`): config, latest, adopt, take, take_older, release, requeue, depth,
  report, wait, tick, checkpoint (the signed `/rl/checkpoint` answer, for the trainer's resume). Served on its own
  listener (`build_rl_trainer_app`), never on the public port: `RELIQUARY_RL_TRAINER_BIND=host:port` (default
  `127.0.0.1:8889`; a private interface such as a WireGuard/Tailscale address, or an SSH tunnel from the GPU box). Matches today's topology: the buffer stays in the controller's SQLite file, the trainer on its
  GPU box. Calls retry transport errors / 429 / 5xx for 10 min; a resent report the validator no longer holds
  counts as done; idle, the trainer requeues taken groups (a take whose answer was lost).
- Trainer (`reliquary train-async`, `trainer/async_cli.py`): `RELIQUARY_RL_TRAINER_URL`, `_TOKEN`,
  `RELIQUARY_RL_VALIDATOR_HOTKEY`, `R2_PUBLIC_*`; mix and run from `/rl/trainer/config`; fresh book = base model,
  published and adopted as 0; else resume = the latest index version rebuilt through the signed `/rl/checkpoint`
  (`CheckpointSync`, an unadopted published version adopted first), written as an HF dir and loaded (optimizer
  fresh, `global_step_hint = k x steps_per_checkpoint`); `DeltaPublisher(model_dir=)` into `reliquary-public`;
  `BackgroundHfBackup` (`RELIQUARY_RL_HF_BACKUP_PREFIX` / `_EVERY` / `_KEEP` / `_MILESTONE_EVERY`, repos under
  `ReliquaryForge/`); FSDP2 = `broadcasting_train_step` + rank-0 `gather_full_state_dict` as `state_fn`, followers
  of `train-worker`. A failed step is fatal (restart resumes from the last adopted checkpoint).
- Smoke test `tests/unit/test_rl_async_service.py::test_end_to_end_lease_submit_train_pay_and_sync` (CPU, < 1 s):
  uvicorn on a socket, `mine-async`'s `AsyncMiner` with `HttpRlTransport` and a fake generator, checkpoint 0 synced
  from `/rl/checkpoint`, leases -> submissions (in zone and 0/16) -> buffer and probation audits (inconclusive),
  one `AsyncTrainer` step through the link with a fake step -> reports -> delta 1 to a memory bucket, adopted ->
  ledger accrual -> `/rl/pay` -> miner and audit model sync 1 -> period settled and published.

Publication and discovery (branch `feat/rl-async-publish`):

- Scan log (decisions §A/§D, design §4.5; `validator/rl_observations.py`): every `rl_prompt_scans` row is one
  event `{type: scan, id: seq, env, prompt_idx, checkpoint, rewards, verdict, first_scan, state, until_k, ts}` (no
  hotkey, no token; `state` / `until_k` are what the scan moved the prompt to, new columns of `rl_prompt_scans`,
  added to an older book at open). The window-era `services/publication.ObservationPublisher` is reused unchanged in
  format (gzip JSONL segments with immutable cache and conditional PUT, head `index.json` with `max-age=15` signed
  by the validator hotkey, closed pages of 200 segments) with a `root` (`rl/<run>/observations/` in
  `reliquary-public`, `ckpt_index.observations_root`) and its own projection; `ScanJournal` is its runtime over the
  lease book's file (segment plans and closed pages persisted before upload: a restart rebuilds the same bytes). A
  flush every 60 s (`rl_async_loops(publisher=)`), signed like the checkpoint index: the remote signer's
  observation operation with `RELIQUARY_SIGNER_URL` ("V1 operation" below), else the local wallet.
- `GET /rl/observations?after=<n>` (`ObservationBoard`): the bucket is private, so the validator serves the head it
  last wrote and presigned GET URLs (15 min) for its pages and for segments `n+1 ..` (512 per answer); 503 before
  the first flush. Miner reader `protocol/rl_scan_log.read_scan_log` (head, pages and segments verified with the
  publication verifiers); `AsyncMiner.bootstrap_state` folds the log into the planners at start, then
  `/rl/prompts/state` continues from the last seq (any failure falls back to the route from 0).
- Discovery: `run_rl_async_validator` serves its axon at start with `--external-ip` / `--external-port`
  (`validator/axon.serve_axon_on_chain`, shared with the window validator). `mine-async` without `--validator-url`
  takes the first permitted validator axon and its hotkey from the metagraph (`submitter.discover_validator`);
  `--validator-url` (+ `--validator-hotkey`) overrides.

Start (async mode):

```
# validator (controller)
RELIQUARY_RL_MODE=async RELIQUARY_TASK_ID=<t> RELIQUARY_RL_TRAINER_TOKEN=<secret> \
  R2_PUBLIC_ENDPOINT=... R2_PUBLIC_ACCESS_KEY_ID=... R2_PUBLIC_SECRET_ACCESS_KEY=... [RELIQUARY_RL_AUDIT_DEVICE=cuda:0] \
  [RELIQUARY_RL_TRAINER_BIND=<private ip>:8889] \
  reliquary validate --checkpoint <base model> --wallet-name ... --hotkey ... --external-ip <public ip>
# trainer (GPU box; torchrun --nproc-per-node N with RELIQUARY_TRAINER_BACKEND=fsdp2)
RELIQUARY_RL_TRAINER_URL=http://<controller private ip, or localhost via ssh -L>:8889 RELIQUARY_RL_TRAINER_TOKEN=<secret> \
  RELIQUARY_RL_VALIDATOR_HOTKEY=<ss58> R2_PUBLIC_...=... [RELIQUARY_RL_HF_BACKUP_PREFIX=<p> RELIQUARY_RL_HF_BACKUP_EVERY=200] \
  reliquary train-async
# miner
reliquary mine-async --environments math     # discovered; or --validator-url http://<controller>:<port> --validator-hotkey <ss58>
```

### V1 operation (branch `feat/rl-async-v1ops`)

V1's controller holds no key and no GPU; both gaps are closed (the checkpoint index, the scan log and the axon
are signed by signer-01, the audits run on the proof worker).

- Remote signer, index operation: `POST /v1/rl/index/sign` (`signer/protocol.RlIndexSignRequest`): the index is
  validated field by field (`RlCheckpointIndex`: strict, no extra field, `format` fixed, every key under
  `rl/<run>/`, anchors <= latest, deltas `from < to <= latest`, bounded lists), the operation id binds netuid and
  the SHA-256 of `index_signing_payload(index)`, and signer-01 rebuilds that payload (domain
  `reliquary-ckpt-index/1\n` + canonical JSON) itself: no bytes are taken from the client. The run is fixed by
  `RELIQUARY_SIGNER_RL_RUN` on signer-01 (unset: 403 `rl_index_operation_disabled`); journal cursor
  `rl_index_version` may repeat (pruned index, new cumulative deltas) but never goes back (409); a signature lost
  to a restart may be retried, like checkpoint signatures. Client: `RemoteSignerClient.sign_rl_index` (verifies
  the signature against the expected hotkey). The async validator signs through it when `RELIQUARY_SIGNER_URL` is
  set (`rl_async_service.index_signer_for`); `RlCheckpointBoard` now loads and signs off the event loop.
- Remote signer, scan log operation: `POST /v1/rl/observations/sign` (`signer/protocol.RlObservationSignRequest`):
  a head (`RlObservationHead`) or a closed page (`RlObservationPage`) of the scan log, strict field by field (no
  extra field, so no `signature`; `schema` fixed; bounded lists, a page exactly 200 segments on a page boundary),
  then the publication's own structural rules (`services/publication.check_index_body`, shared with the readers'
  `verify_index`): contiguous segments and pages, every key exactly `rl/<run>/observations/...` (so never a
  window-era `observations/run-<id>/` document, which shares the observation domain). The run is the same
  `RELIQUARY_SIGNER_RL_RUN` (unset: 403 `rl_observation_operation_disabled`); the document must name the signer's
  own hotkey (403 `validator_denied`); a head may not be dated more than 300 s ahead (403 `head_from_the_future`).
  Journal cursors `rl_observation_head` (`last_number`) and `rl_observation_page` (`first_number`) may repeat (a
  restart rewrites the head; a lost page signature) but never go back (409). signer-01 rebuilds the payload
  (`document_signing_payload`: `reliquary/observation-index/v1\n` + canonical JSON) itself. Client:
  `RemoteSignerClient.sign_rl_observation` (verified against the expected hotkey); the validator's publisher signs
  through it when `RELIQUARY_SIGNER_URL` is set (`rl_observations.scan_publisher(signer_client=)`,
  `ObservationPublisher(signer=)`), with the signer's public identity as `wallet`. The async validator's axon is
  served through the signer's serve-axon operation in that case.
- Remote audits (`validator/remote_rl_audit.py`), on the proof worker's mTLS listener: `POST /v1/rl/checkpoint`
  takes the board's answer (signed index + presigned URLs); the worker checks the signature against
  `RELIQUARY_PROOF_RL_VALIDATOR_HOTKEY` and rebuilds the latest version with `CheckpointSync` into a dedicated
  verify model on `RELIQUARY_PROOF_RL_AUDIT_DEVICE` (base `DEFAULT_BASE_MODEL`, overwritten at the first anchor).
  `POST /v1/rl/audit` takes a group's tokens, force spans and lease draw params (nonce, prompt, k; never uniforms:
  the worker recomputes them through `protocol/rl_draw`) plus the expected version, runs `rl_audit.audit_forward`
  and answers the `AuditResult` bound to the request digest; 409 when it holds another version (a restart), after
  which the controller re-adopts once and retries. `RemoteAuditModel` is the controller's `AuditModel` over it
  (`follow_once` / `audit`). The service picks the audit forward with `audit_source`: proof executor remote ->
  worker; else `RELIQUARY_RL_AUDIT_DEVICE` -> local; else inconclusive; both -> refuses to start.
  `remote_rl_audit.py` is part of the proof transport hash.

Config to set on V1:

```
# signer-01
RELIQUARY_SIGNER_RL_RUN=<run>                      # = the controller's RELIQUARY_RL_RUN (default: the task id)
RELIQUARY_SIGNER_AXON_IP=<public ip> RELIQUARY_SIGNER_AXON_PORT=<port>   # = the controller's --external-ip/port
# proof worker
RELIQUARY_PROOF_RL_AUDIT_DEVICE=cuda:0 RELIQUARY_PROOF_RL_VALIDATOR_HOTKEY=<validator ss58> \
  [RELIQUARY_PROOF_RL_AUDIT_DIR=<scratch dir for anchor/delta downloads>]
# controller (unchanged proof/signer client config; no RELIQUARY_RL_AUDIT_DEVICE)
RELIQUARY_RL_MODE=async RELIQUARY_SIGNER_URL=... RELIQUARY_SIGNER_{CA,CERT,KEY,EXPECTED_HOTKEY}=... \
  RELIQUARY_PROOF_EXECUTOR_MODE=remote RELIQUARY_PROOF_EXECUTOR_URL=https://<worker>:8445 \
  RELIQUARY_PROOF_TLS_{CA,CERT,KEY}=... RELIQUARY_PROOF_EXPECTED_WORKER_ID=<id> \
  RELIQUARY_RL_TRAINER_TOKEN=<secret> RELIQUARY_RL_TRAINER_BIND=<private ip>:8889 \
  R2_PUBLIC_{ENDPOINT,ACCESS_KEY_ID,SECRET_ACCESS_KEY}=... \
  reliquary validate ... --external-ip <public ip> --external-port <port>
```

The worker must reach the `reliquary-public` presigned URLs (HTTPS only). Deploy signer-01 and the worker before
the controller (both are additive; the transport hash changes with this release on both sides).

### P4 as built (branch `feat/rl-async-trainer`)

- `reliquary/trainer/async_trainer.py` (`AsyncTrainer`, `AsyncTrainerConfig`, `DrawGate`), next to the window
  trainer (unchanged). `run_once`: drop groups older than `max_lag` (8) checkpoints unpaid without a forward
  (`TrainingBuffer.take_older`); take `batch_groups` of any env in arrival order, polling until the batch is full
  (the groups of a take whose answer was lost are requeued before each step); one batch, global token mean;
  `train_step(..., gate=DrawGate)`; `report_training` for every group; a checkpoint every
  `steps_per_checkpoint` (N, default 1): CPU snapshot on the step path, `DeltaPublisher.publish` +
  `LeaseBook.adopt_checkpoint(k+1)` in a one-deep publish thread (an error stops the loop at the next
  publication); `BackgroundHfBackup.submit` every `hf_backup_every` (M) steps; `prices.note_speed(completion tokens, seconds
  worked)` per step. Health skip: groups reported untrained, unsanctioned. Crash: groups released.
- Verification in the training forward (`training.py`, `gate=None` keeps the window path as it was): per
  micro-batch the substitution forward gives v* and per-rollout tallies (stochastic / drawn, matches); pi_old :=
  the policy (ratio 1); rollouts under `RL_ROLLOUT_MATCH_FLOOR` (constants, 0.95 placeholder; the ledger's floor)
  leave the loss before the backward. Rollout match = rate on stochastic positions when >= 20 of them, else on
  every drawn position. After all micro-batches `DrawGate.judge` pools each group (FSDP2: tallies all-reduced) at
  `ASYNC_GROUP_FLOOR` 0.90 (placeholder: one fully bad rollout of 16 survives, two do not); a failed group's
  admitted rollouts are backed out by a second pass with negated advantages (exact up to bf16 noise; only failed
  groups pay it). Accepted: paid matched tokens of admitted rollouts; rejected: `statistical` when lag <=
  `sanction_max_lag` (2), else unsanctioned. Telemetry: train/* plus `async/` checkpoint, lag, buffer depth,
  waits, batch groups per env, forwarded / below-floor rollouts, accepted / rejected / undone groups.
- Known approximations: masked rollouts stay in their env's planned token denominator; with N > 1 the forward
  verifies against the trainer's in-between weights; the publisher copies the snapshot once more on CPU.
- GPU qualification still owed: trainer-forward picks vs the proof path on the 9B (FSDP2, fp32 head), the floors,
  snapshot + delta time per step.
