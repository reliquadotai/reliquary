# SciCode as one evaluation order (chained positions)

Date: 2026-11-14. Status: implemented on branches `feat/scicode-chained-orders` (Catalyst: admin routes, chain
bookkeeping, greedy draw audit; platform: quote, order lifecycle, catalog). Not deployed.

## Problem

PR #511 added SciCode as a chain of single-turn sets (`verifiers:scicode-steps`): upstream asks each sub-step in a
fresh prompt that carries the model's own code for the earlier steps, so the set of step position k (1..15 on the
test split) can only be built once position k - 1 is graded. Nothing chained the positions: a customer would have
placed up to 15 orders by hand, each built by an operator from the last one's answers.

Goal: one order on app.reliqua.ai, priced and timed as one, reporting the leaderboard's two numbers: main-problem
resolve rate (/65, test split) and sub-step pass rate (/288). Protocol: test split, no background, greedy decoding,
300 s per step test.

## Options

**A. Platform-side chain.** The order runs one eval job per position, one after the other. When position k is
graded, the admin builds position k + 1's set from its graded answers, publishes it, and the platform creates its
job. The last position graded, one combined report adds the chain's rates.

**B. Miner-side chain.** A new interaction type: one episode per SciCode problem, the miner building each step's
prompt from its own previous answer.

What the code says:

| Question | Evidence | A | B |
| --- | --- | --- | --- |
| Can one order hold several jobs? | `parts_json` (platform migration 0050): one job per set, `${job_id}-<k>`, each graded as `${eval_id}-<k>`, combined by `POST /admin/v1/evaluations/{id}/combine` (`orders.reports`). | yes, as is | n/a (one job) |
| Can the admin build and publish a set at run time? | `POST /admin/v1/uploads` (`orders.uploads`) already freezes customer files into sets and publishes them create-only (`eval.storage`). A Verifiers set is `verifiers_source.build_set` over taskset args; `publish_set` writes an operator set to the subnet bucket only, which is all a job (`read_set`) and a grading (`grading.load_sets`) read. | yes | n/a |
| Forced-draw audit | Each position is an ordinary single-turn set: `eval_draw` keys the draw on job, prompt row and sample; the audit is unchanged. B would need the auditor to rebuild every turn's prompt from the earlier turns' tokens (extract the code, re-render the template), as suite episodes replay their tool world: a new episode kind in miner, control and audit executor. | unchanged | new replay path |
| Miner release | A: none (jobs, sets and draw are the existing ones). B: the miner must hold the taskset (dataset, templates, extraction) and drive the new interaction. | none | required |
| One audit executor boot | `order-fleet.ts demandOf`: an audit executor serves while an order of its model is `running` or `draining`. A chain order stays `running` between positions, so its executor is kept (`surplusAuditors` always keeps one). | yes | yes |
| Wall clock | A: 15 sequential stages (job pickup, generation of the position's longest answer, drain, grading, next build). B: problems advance independently; wall clock ~ the longest problem. | sum of stages | shorter |

Chosen: **A**. It reuses the several-set order, the admin's set publication and the draw audit unchanged, needs no
miner release, and the price follows the existing per-set rules. B's only gain is wall clock (positions 9-15 hold
7 problems or fewer, so the sequential tail is short) at the cost of a new interaction across three releases.

## Greedy decoding under the forced draw

The leaderboard decodes at temperature 0. Ours cannot: reliquary jobs require `temperature > 0`
(`corpus.job._parse_sampling`, `OrderSampling`) and the platform refuses `greedy_unavailable` for subnet orders.
Greedy is `top_k 1`: the warp keeps one token, so the Gumbel pick is the argmax whatever the noise. The order's
sampling is fixed to `{temperature 1, top_p 1, top_k 1}` (temperature 1 keeps the scores the logits); a customer
may send nothing, `temperature 0` or `top_k 1`, plus a seed.

The audit needed one change. A miss is scored as the gap between the pick's and the token's Gumbel scores. Under
top_k 1 an honest flip (two near-tied logits, bf16 drift) still carries the difference of the two tokens'
independent noises, which is logistic: over the 4.0 gap tolerance about 2 % of the time, and two such misses in a
sample charge it as an edit. `forced_sampling._greedy`: a warp that keeps one token scores its misses on the logits
alone (`(logit_best - logit_token) / T`). The pick, the support widening (`edge_margin`: top 2 is possible) and the
judge's thresholds are unchanged; any other warp is untouched. This runs on the audit executor (its image), not on
miners.

## Flow

```
quote (one set: the first position's catalog set, whole, samples 1, greedy)
  queued ──create job -1──▶ running
running, per position k:
  job -k complete ─▶ retire ─▶ final counts ─▶ grade (eval_id-k)
  k < 15: POST /admin/v1/evaluations/{eval_id-k}/chain-next {chain: scicode-steps}
          ─▶ set of k+1 (subnet bucket) ─▶ create job -(k+1) (max_prompt_tokens if over 8,192)
  k = 15 (or a position cannot go on) ─▶ grading
grading: POST /admin/v1/evaluations/{eval_id}/combine {parts: [-1..-m], chain: scicode-steps} ─▶ delivered
```

Catalyst (`reliquary/orders/chains.py`):

- `next_set`: reads position k's report (`provenance.sets[0]`: the set; `provenance.model/revision/thinking`), the
  set's card and `grading.jsonl` from the subnet bucket, and its `graded.parquet`. Each row's step is
  `chain.asked_at(split, k)[source_index]`; its answer is the text its grader read (`grading.answer_text`, after the
  reasoning block); a step with no row is recorded as no code, as upstream records a failed generation. The next
  args are `chain.next_args` (k + 1, `previous` with the extracted code). The set is built with
  `verifiers_source.build_set` and published subnet-only, create-only: its id carries a hash of its args
  (`verifiers-scicode-steps-a<8 hex>-r0-n<count>`), so a retry after a lost answer finds the same set (its card's
  args are compared; another step under that id is refused). Prompt lengths are counted in the model's chat
  template (`load_cpu_tokenizer`) and answered, never written on the card (two models with the same answers share
  the set).
- `chain_report`: the main-problem and sub-step rates (`chain.summary`, upstream's denominators: a step not reached
  counts as failed), after checking the parts are one chain: the first position has no `previous`, and each next
  position's args are exactly `next_args` of the one before and its graded answers.
- `combine` with `chain`: the report gains `chain` and its macro `pass@1` is the sub-step rate (`basis:
  chain_steps`); `complete` only when every position was graded. The stored manifest names the chain, so the same
  parts combined plainly under the same id are refused.

Platform (`worker/evaluation-chains.ts`, `worker/evaluation-subnet.ts`):

- `CHAIN_SETS` names the catalog set of a chain's first position and its shape (problems per position). An order
  on it runs in `advanceParts` with one part per position; `advanceChain` moves the current position by one stage per
  cron run. Only one job runs at a time, with the order's whole cap.
- A position that cannot go on ends the chain into grading (`chain_<reason>`, flagged): its job refused, its grading
  failed, the next set refused (409 other than `part_not_graded`) or not the expected position/count, its final
  counts never read within `EVAL_DRAIN_TIMEOUT_SECONDS`. The graded positions from the first are combined; the
  others are `missing` (`not_reached`) in the summary. Cancel, run deadline, executor endings are the several-set
  order's (retire, drain, grade what exists).

## Price and ETA

As a several-set order (`docs/design/2026-11-13-multi-set-orders.md`), with the positions summed:

| Term | Rule |
| --- | --- |
| executor boot | once |
| executor generation hours | the sum of the positions' wall clocks (the executor is held between positions) |
| output tokens | 288 x max_new_tokens (worst), 288 x min(avg, max_new_tokens) (expected) |
| input tokens | sum over positions of problems x mean prompt (667 tokens at position 1, +~400 per earlier step) |
| grading fee | one per position (15); a position not reached refunds its share |

Position wall clock: `EVAL_CHAIN_STAGE_HOURS` (0.25 h: pickup, drain, grading, next build; at most
`EVAL_CHAIN_STAGE_MAX_HOURS`, 0.5 h) plus the larger of the position's tokens over the subnet throughput and one
completion decoded end to end (`EVAL_CHAIN_DECODE_TOKENS_PER_SECOND`, 20 tokens/s). ETA = the sum over the 15
positions; at max_new_tokens 32,768 the worst case is about 15 h, within the 24 h run deadline (131,072 is refused
as `exceeds_generation_deadline`).

## Set identity

First position, built by the operator once (`{"split": "test", "step": 1, "previous": {}}`):
`verifiers-scicode-steps-a024c0a1e-r0-n64`, 64 problems, `prompts_sha256`
`b8dda24b6b7669b873cb647d45ed7fbb60c74bd542956a556459b98a2fe97410` (dataset `SciCode1/SciCode@4510f6a6`,
`problems_test.jsonl` sha256 `38797fef…`). Problems per position: 64, 60, 53, 32, 21, 15, 10, 9, 7, 4, 4, 4, 2, 2, 1
(288).

## Deploy

1. Catalyst (order control / admin image): `chain-next`, `combine` with `chain`, `scicode_steps` 0.1.1. On ctrl-01
   the SciCode grading sidecar (`eval-grader-scicode`, runbook `docs/runbooks/order-control.md`, SciCode) and
   `RELIQUARY_EVAL_SCICODE_GRADER_SOCKET` in the admin container.
2. Audit executor image (`EVAL_AUDIT_EXECUTOR_IMAGE`): the greedy gap. Without it, greedy orders run but honest
   flips are refused more often (reopened slots), occasionally charged.
3. Publish the first position's set (`reliquary eval build-set` then `publish-set --publish-reviewed` or the
   platform copy by hand) and insert its `eval_sets` row (platform `docs/EVALUATIONS_API.md`, SciCode).
4. Platform Worker.

No miner release: jobs, sets, draw and sampling are the existing ones (`top_k 1` is a warp miners already apply).

## Limits

- Fifteen job pickups in series: most of the ETA is fixed per position, not tokens.
- One sample per step (the chain carries one answer forward); pass@k over several chains would be k orders.
- The positions' mean prompt sizes in the quote are estimates (300 tokens of code per earlier step); the jobs carry
  the built sets' real maxima.
