# Verification tiers, audit waves and precision for orders

Status: decided with the user 2026-11-13; implemented on `feat/verification-tiers-precision` (Catalyst) and
`feat/verification-tiers-precision` (platform, on top of `feat/customer-batch-price`). Not deployed.

Applies to order jobs (Batch, Evals, Affine evaluations) and, for the audit knobs only, to the RL async path.

## 1. Decisions

1. **Default verification for Batch and Eval orders is a sampled audit at a low share** (`audit_q` 0.05,
   configurable), not 100 %. Affine evaluations keep 100 %. Miners are paid the same whatever the grade, so an
   eval gives them no more reason to cheat than a Batch order.
2. **Safety comes from the penalties**, not from auditing everything: probation (a hotkey's first N records
   audited), suspicion/ban with forfeit of held pay, and **retroactive invalidation**: a caught hotkey's
   unaudited rows in the order are discarded and their slots reopened, so a delivery never holds a caught
   cheater's rows. The delivery report states the audited share, the hotkeys caught and the rows redone.
3. **"Full verification"** (`audit_level: full`, audit 100 %) is an order option, priced with a platform
   multiplier (placeholder 1.25x) and delivered with per-row evidence.
4. **RL** gets the same two knobs (audit share, probation length) as operator settings; defaults unchanged.
5. **Batch audits in waves**: a card is rented when the audit backlog is worth a card (or the order is nearly
   done, or records waited too long), drains it, and is returned. Cards are shared per model.
6. **Precision** (`bf16` | `fp8`) is an order field recorded in the job contract and the report. Only `bf16` is
   enabled: `fp8` stays behind a flag until the measurement in section 7 sets its tolerance.

## 2. What exists (evidence from code, origin/main 6bde6d1b)

| Piece | Where | State before this change |
|---|---|---|
| Sampled audit | `corpus/audit_policy.py` `decision()`, `drawn()` (sha256 of drand randomness and submission id < q) | exists; `audit_q` per job |
| Batch default share | `admin/service.py` `GEN_AUDIT_PARAMS` (`audit_q` 1.0, probation 20, hold 600 s, suspect 24 h, ban after 3 failures in 7 d, ban 7 d) | the platform sends `DATASET_AUDIT_Q` (0.2 when unset) |
| Eval share | `admin/service.py` (eval/pinned args), `eval_control.pinned_order_refusal`, `eval/episode_jobs.eval_job_source` | **forced to 1.0**: three places refuse anything else; platform `createEvalJob` hard-codes `audit_q: 1.0` |
| Probation | `effective_state()`: `audited_passed < probation_submissions` audits everything; the count is shared across orders (`ORDER_TRUST_ID`, `corpus_miner_states.TRUST_FIELDS`), reset by any confirmed failure | exists; N = 20 for orders, set by the admin only (not on `CreateJob`) |
| Suspect / ban | `after_confirmed_failure()`: 1st and 2nd charged failure = suspect 24 h (every record audited, new leases refused on forced-draw jobs: `corpus_service` `MINER_SUSPECT`), 3rd in 7 d = ban 7 d (submissions refused, `void_banned`) | exists, per job (deliberately not global) |
| What is charged | `eval_draw.judge`: `penalty` only for match < 0.90 over >= 64 positions, or >= 2 far misses (an edit); drift is refused uncharged and its slot reopened | exists |
| Forfeit of held pay | `_escalate` -> `_void_forfeited`: on a charged failure at q < 1, every **unsettled** pass of the hotkey received up to the failure gets a `corpus-voided` document (`forfeited`) and its slot reopens (forced-draw jobs); undecided ones get `forfeited` verdicts | exists |
| Stake | `stake_at_receipt` (PR #471): the hold keeps at least 1/q of a hotkey's records unpaid, so a caught cheat forfeits at least what a cheat earns | exists |
| Payment of an undrawn record | waits its hold (600 s) **and** the audit of every drawn sibling of its hotkey inside that hold (`_judge_once` sibling guard) | exists: nothing expires; a record that needs an audit stays unpaid until audited |
| Delivery excludes voided | `delivery.passing_batches` (`voided_submission_ids`, `submission_voided`) | exists |
| Report | `delivery.verification_block`: verdicts, audited, audited_share, refusals by reason; rows carry `audited`, `draw_match` | exists; **no hotkeys caught, no rows redone** |
| Retroactive invalidation of **settled** unaudited rows | - | **missing**: a caught hotkey's unaudited passes already paid stayed in the delivery |
| RL audit share | `rl_submit.SubmitConfig.audit_rate` 0.15, `probation_groups` 100 | hard-coded; only `probation_repeat_audit_rate` had an env/registry switch |
| Card sizing | `eval_control.executors_for_backlog`, `ExecutorSizing`, `control_status`; platform `order-fleet.ts` rents a baseline card per model as soon as an order runs and releases it 120 s after `audits_pending` is 0 on a draining order (PR #473) | cards held for the whole generation |
| Precision | job contract, miner census, ask: no field; `DrawJudge` thresholds are per job (`corpus/job.py`) | **missing** |

## 3. Sampled audit for every order

### 3.1 How safe is 5 %?

A cheater who sends k bad records past probation escapes with probability (1 - q)^k. With retroactive
invalidation, what reaches a delivery is only the bad rows of hotkeys never caught, whose expected count per
cheating hotkey is k (1 - q)^k, at most **1 / (q e)** whatever k (maximum at k = 1/q). The same bound holds for a
hotkey that cheats on a share f of its records (each bad record is caught with probability q).

| q | rows past escape, per cheating hotkey (bound) | P(escape) at k = 50 | stake kept unpaid (records) |
|---|---|---|---|
| 0.05 | 7.4 | 7.7 % | >= 20 |
| 0.10 | 3.7 | 0.5 % | >= 10 |
| 0.20 | 1.8 | 0.001 % | >= 5 |
| 1.00 | 0 | 0 | - |

Each extra hotkey costs N = 20 honest audited records first (probation, counted across orders) plus a
registration, and its stake is forfeited when caught, so a cheat costs at least what it earns (§7.4 of the
audit-policy design, `forfeits()`). On an eval of 2,000 samples, one uncaught cheating hotkey moves the score by
at most ~0.4 % in expectation; customers who need a bound of 0 buy `full`.

### 3.2 What changes

- Catalyst: `GEN_AUDIT_PARAMS["audit_q"]` defaults to `RELIQUARY_ORDER_AUDIT_Q` (0.05) and
  `audit_probation_submissions` to `RELIQUARY_ORDER_AUDIT_PROBATION` (20). `CreateJob` takes
  `audit_probation_submissions` per order. Eval orders take `audit_q` and get the order audit params (hold,
  suspect, ban) instead of `{"audit_q": 1.0}`; the three refusals of q < 1 on evals are removed.
- Platform: Batch default `DATASET_AUDIT_Q` 0.05 (was 0.2); evals send `EVAL_AUDIT_Q` (0.05) unless the order is
  `full`; Affine evaluations always send `audit_q` 1.0 (`audit_level: full`, not charged the multiplier: their
  price already reserves audit cards).
- An unaudited eval completion is graded as the model's. That is what the retroactive invalidation (3.3) and
  the 1/(q e) bound make acceptable.

### 3.3 Retroactive invalidation (new)

On a **charged** failure at q < 1 (`_escalate` -> `_void_forfeited`), besides the unsettled passes it already
forfeits, the auditor now voids **every unaudited pass of that hotkey in the job, settled or not**
(`corpus-voided`, reason `caught_unaudited`), and reopens each slot so another miner redoes it. A settled one
stays paid (pay is not clawed back; it was within the stake), but leaves the delivery. Audited passes of the
hotkey stay: each was checked on its own. Records received after the failure are audited (the hotkey is
suspect) and a banned hotkey's are voided (`void_banned`), so nothing unaudited of a caught hotkey remains.

The order does not finish while a reopened slot is open, so the delivery waits for the redo.

### 3.4 Report

`verification` in the v3 report gains:
- `audit_q`, `audit_level` (`sampled` | `full`), `precision`;
- `hotkeys_caught`: hotkeys with a charged failure in the order (count, and the hotkeys);
- `rows_redone`: slots reopened, by reason (`refused` drift/refusals, `forfeited`, `caught_unaudited`,
  `executor_quarantined`, `replay_failed`), from the verdicts and void documents;
- `per_row_evidence`: true when every delivered row was audited; rows then carry `draw_max_gap` and
  `draw_positions` next to `audited` and `draw_match` (the new columns are present, null when unaudited).

## 4. Full verification option

`audit_level` (`sampled` default | `full`) on Batch and Eval quotes. `full` sends `audit_q` 1.0, multiplies the
token terms by `ORDER_FULL_AUDIT_PRICE_MULTIPLIER` (placeholder 1.25, not a real price), and its delivery has
per-row evidence (3.4). Stored on quotes and orders (migration 0068) and echoed in the order payload.

## 5. RL

`rl_async_service` reads `RELIQUARY_RL_AUDIT_RATE` / registry `rl_audit_rate` (default 0.15) and
`RELIQUARY_RL_PROBATION_GROUPS` / `rl_probation_groups` (default 100) into `SubmitConfig`, next to the existing
`rl_probation_repeat_audit_rate`. Defaults unchanged. The RL ban/penalty logic is its own (`rl_submit`), not
changed here.

## 6. Batch audit waves

### 6.1 Numbers

Measured 2026-11: a card held for the whole generation costs ~$0.32/Mtok generated on 9B and ~$0.85/Mtok on 27B;
the audit work itself at q 0.1-0.2 is $0.02-0.04/Mtok. A card verifies ~18 M tok/h (9B) / ~25 M tok/h (27B);
miners generate ~14 M tok/h (9B) / ~6 M tok/h (27B). At q 0.05 a card is busy ~4 % (9B) to ~1 % (27B) of a
generation once probation is over.

### 6.2 Rule (Catalyst `eval_control.AuditWaves`)

Per model, when every served order of it is a wave order (`audit_waves: true`, Batch only; an Eval or a `full`
order on the same model turns waves off for that model):
- **waiting**: `executors_wanted` 0 at once (`ExecutorSizing.set`, no scale-down wait); a live card may finish
  what it holds, and the platform returns it (6.4) and rents none. A wave **starts** when one of:
  - the expected audit work is worth a card: expected items / measured items-per-second >= 
    `RELIQUARY_ORDER_AUDIT_WAVE_MIN_SECONDS` (1800 s); before any rate is measured, expected items >=
    `RELIQUARY_ORDER_AUDIT_WAVE_MIN_ITEMS` (2000);
  - an order is nearly done: prompts full >= `RELIQUARY_ORDER_AUDIT_WAVE_FINISH_SHARE` (0.9) of its prompts;
  - the oldest record waiting for an audit is older than `RELIQUARY_ORDER_AUDIT_WAVE_MAX_WAIT_SECONDS` (3 h),
    which bounds a miner's wait for pay.
- **draining**: sized as today (`executors_for_backlog`), until the expected audits reach 0, then back to waiting.

The model's status entry publishes `audit_wave: {"phase", "since", "reason"}`. A wave order is released to miners
once its auditor has seeded, without waiting for a card (the hold's `awaiting_audit` drops the card condition for
wave jobs). The token auction does not steer a wave job's budget on the audit backlog (a large backlog is
intended).

### 6.3 Holds and pay

Nothing expires: a drawn (or probation, or suspect) record stays unpaid until a wave audits it, and an undrawn one
is paid only after its hold **and** the audit of its drawn siblings (existing guard), so the stake and the
forfeit hold during the wait. Miners on a wave order are paid later (up to the max wait, or the order's end).

### 6.4 Platform

`order-fleet.ts`: the baseline rent skips a model whose control status says `audit_wave.phase == "waiting"`, and
a running order of such a model does not keep a card (it counts as audit-idle). Extra cards still follow
`executors_needed`, which a draining wave sets. Batch orders send `audit_waves: true` unless
`BATCH_AUDIT_WAVES = "0"`. Cards are already per model (`order_executors` keyed `model@revision`).

### 6.5 Expected cost

A 100 Mtok 9B order at q 0.05 with ~30 miners: probation 600 records + 5 % of the rest ~= 6-7 Mtok audited, one
end-of-order wave of ~25 min on one card (+ boot) instead of ~7 h held: ~$1.5 instead of ~$30. To verify on the
first live order: wave start/stop in the control log, card hours per order, miners' pay delay.

## 7. Precision

### 7.1 Mechanism (implemented)

- Job contract: optional `precision` (`bf16` | `fp8`), written only when not `bf16`, so every existing manifest
  hashes as before; a miner release that predates it refuses an fp8 job (unknown field), failing closed.
- `corpus/job.py` `PRECISION_JUDGES`: the default `DrawJudge` per precision; a job's explicit `judge` still wins.
  `bf16` keeps today's judge; **`fp8` has placeholder thresholds** and is refused at job creation unless
  `RELIQUARY_ORDER_PRECISIONS` lists it (default `bf16`).
- Miner: `reliquary corpus mine --precision` (default `bf16`) follows only orders of the precision it serves (the
  order list `/corpus/jobs` names `precision` when not bf16) and refuses a `--job-id` of another. How an fp8
  miner configures vLLM (W8A16, W8A8, fp8 KV) is left to the measurement: the flag only selects orders today.
- Order control: the order record's `precision` must equal the job's (`pinned_order_refusal`).
- Report: `verification.precision`; platform: `precision` on quotes/orders, Eval default `bf16`, Batch default
  `fp8` **when enabled** (`ORDER_PRECISIONS_ENABLED`), else `bf16`; price multiplier
  `ORDER_PRECISION_PRICE_MULTIPLIER_FP8` (placeholder 0.65).
- No int4/fp4.

### 7.2 Why fp8 stays disabled (evidence)

From `~/pearl-lab/RAPPORT.md` (Qwen3-8B, Gumbel T=1, vLLM 0.20.2, H100/H200, judge 0.98 / gap 4):
- bf16 generation vs bf16 audit: 199/200 pass, 0.86 % tokens diverging.
- a w8a8-style quantised model audited against bf16: 11/200 pass (116 charged), ~10 % diverging; its honest match
  (~0.90) overlaps the 0.5-0.86 band the judge uses to recognise another model.
- the same quantised model **replayed in its own precision with the same vLLM kernels**: 8 % diverging when the
  miner decodes at batch 256, ~1 % at batch 1024. Activation quantisation amplifies small decode-vs-prefill
  differences (0.5 % -> 6 % mid-network), and vLLM's fp8 block linear switches kernel with the batch (FlashInfer
  M < 32 vs DeepGEMM M >= 32), so drift depends on the miner's batch. `VLLM_BATCH_INVARIANT` did not fix it.
- `~/offpolicy-lab/results/fairness-2026-10-14/gumbel_quant.md` (RL, fp8 miner vs bf16 verifier): match bf16
  0.986, fp8 KV 0.985, fp8 W8A16 0.970, fp8 weights on 5090/L40 0.963, W4A16 0.855;
  `qualif-2026-10/gap-h100`: fp32-head fp8 gap p99.9 1.41, max 2.18 (forged edits p01 6.66).

So a per-precision tolerance needs the audit to replay in the declared precision with the miner's kernels, and a
threshold measured across decode batch sizes that still separates other models.

### 7.3 Measurement to run before enabling fp8

Scripts: `~/pearl-lab/gen_drawn.py` (forced-draw generation on vLLM, `LAB_MAX_SEQS` = decode batch,
`LAB_REPS`), `~/pearl-lab/verify_vllm.py` (audit from a vLLM prefill with the same kernels, fp32 head,
`eval_draw.judge`), `verify_drawn.py` (HF reference), `batch_dep.py` / `m_dep.py` (batch dependence).

Matrix, on Qwen3-8B and Qwen3.5-27B (or the first fp8 customer model), H100 and H200:
1. Formats, separately: fp8 weights only (W8A16), fp8 KV cache only, W8A8 (fp8 block, activations quantised).
2. Temperatures: T = 1 and T = 0.6/0.7 with top-p 0.95 (warp edge margin).
3. Miner decode batch: 1, 32, 256, 1024 (`LAB_MAX_SEQS`), eager and CUDA graphs.
4. Auditor: vLLM prefill in the same precision and kernels (`verify_vllm.py` with the fp8 checkpoint), audit
   batch as the executor runs it.
5. Negative controls, same precision: a smaller sibling model (Qwen3-4B fp8 vs 8B fp8 audit) and re-rolled draws
   (other seed); plus an edit attack (runner-up injections) for the gap test.
6. Output per cell: pass rate, charged rate, match distribution (p0.1, p1, min over >= 150 tokens), gap
   p99.9/max, far misses; >= 1,000 samples per honest cell.
Acceptance: honest charged rate 0 and refused < 1 % at every batch size, with a match floor and gap tolerance
that still charge the sibling model and the re-rolled draw (match < floor) and the edits (far misses). Those
values go into `PRECISION_JUDGES["fp8"]`; then `RELIQUARY_ORDER_PRECISIONS=bf16,fp8` and the executor image runs
the fp8 checkpoint with the miners' vLLM kernels.

## 8. Releases

- **No miner release** for sampled audit, waves, full verification or retroactive invalidation (all
  validator/control/platform).
- **Miner release needed for fp8 only** (`--precision`, and the job field): until then an fp8 job is refused by
  every miner, which is why fp8 stays disabled.
- **Executor image**: unchanged for bf16; an fp8 audit executor (vLLM prefill in fp8) is part of enabling fp8.

## 9. Implementation map

| Piece | Files |
|---|---|
| Order audit defaults, eval q < 1, `audit_probation_submissions`, `audit_waves`, `precision` on `CreateJob`, `/architectures` advertises `orders.audit_waves`, `orders.eval_audit_q`, `orders.precisions` | `reliquary/admin/service.py` |
| Precision in the contract, per-precision judge | `reliquary/corpus/job.py` (`PRECISIONS`, `PRECISION_JUDGES`), `reliquary/cli/main.py` (`build_job_manifest`, `prepare_corpus_job`) |
| Order record vs job (precision; eval q < 1 accepted) | `reliquary/validator/eval_control.py` `pinned_order_refusal` |
| Retroactive invalidation | `reliquary/validator/corpus_auditor.py` (`REASON_CAUGHT_UNAUDITED`, `_void_forfeited`, `_load_paid`) |
| Report and per-row evidence | `reliquary/corpus/delivery.py` (`verification_block`, `voided_by_reason`, rows `draw_max_gap`/`draw_positions`) |
| Waves | `reliquary/validator/eval_control.py` (`WaveSettings`, `AuditWaves`, `control_status`, wave hold, auction backlog) |
| Miner precision | `reliquary/miner/order_follow.py`, `reliquary/cli/main.py` (`corpus mine --precision`), `corpus_service.orders` |
| RL knobs | `reliquary/validator/rl_async_service.py` |

## 10. Open points

- Whether the first charged failure should ban at once on orders (today: suspect 24 h, new leases refused on
  forced-draw jobs, ban at the third). Configurable via `audit_ban_after_failures`; left at 3.
- Wave thresholds (1800 s of card work, 0.9 finish share, 3 h max wait) are first guesses to tune on live orders.
- The multipliers (1.25 full, 0.65 fp8) are placeholders.
