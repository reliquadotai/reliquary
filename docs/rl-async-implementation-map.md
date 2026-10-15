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

**New:** a buffer of accepted in-zone groups tagged (env, k, arrival), filled freshest-first per env, at the target mix, without blocking, with a max-lag drop (~300).

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

**Change:** a per-token price, not a pool factor. Advance on time or steps, not windows. New signals: env share of trained tokens below target, and buffer overflow causes a descend. Remove the snap regime (no window fill). Size ~300.

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

**Change:** lease-then-open instead of precommit-then-open. Re-key `DrawBinding` by nonce. Submit exactly 16 episodes, with no M-of-2M. Drop TOPLOC. Size ~400.

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
| P4 | Buffer (env mix, freshest-first, max-lag drop), payload schema bump, trainer loop without journal or cursor, accumulator without checkpoint reset, publish cadence, continuous `CheckpointIntake` swap | P3 | ~800 |
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

## 6. Reference miner as built (branch `feat/rl-miner-client`)

- `reliquary/miner/rl_async.py`, CLI `reliquary mine-async` (the window miner `mine` is untouched).
  - `PromptPlanner`: per env, fed by `GET /rl/prompts/state?since=` (cursor, `more` pages); skips blacklisted
    prompts, cooldowns until `latest >= until_k`, and prompts held, submitted or `taken` until their scan arrives
    (or 30 min). Default order: never scanned, then by share of in-zone scans; `priority` overrides it.
  - `AsyncMiner`: open leases <= `min(quota, --max-open)`, waiting leases <= `--queue-groups`, requests of
    <= `--lease-batch` at the held k; refusals: `taken` -> wait for the scan, `cooling`/`blacklisted` -> table,
    `quota` -> quota := open, `stale` -> no lease until newer weights. Earliest expiry generated first; every generated
    group is submitted with retries until its expiry (`already_submitted` / `lease_submitted` = delivered); open
    leases journaled for restarts.
  - Checkpoints: `CheckpointSync` on a CPU copy in a thread; the copy is handed to the generator between two batches
    and not patched again before (`swap_if_ready`). Submissions carry `k_generated`.
  - `VllmGroupGenerator`: calls `VLLMRolloutGenerator.generate` per lease (`draw_context`: nonce as randomness,
    `k<k>` as checkpoint hash); weights pushed with `LLM.collective_rpc` + the model's `load_weights` (needs
    `VLLM_ALLOW_INSECURE_SERIALIZATION=1`, untested on GPU), fallback `--weight-swap reload` (HF dir + engine rebuild).
- To align: `POST /rl/submit` body and signature (`encode_submission`, `rl_submit_binding`: DRAFT, `u32le-zlib/1`
  tokens), the Gumbel profile's nonce keying (`draw_context`), and a multi-prompt vLLM call.
