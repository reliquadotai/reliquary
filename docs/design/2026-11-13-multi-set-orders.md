# One evaluation order on several sets (agentic and single-turn)

Date: 2026-11-13. Status: implemented on local branches `feat/multiset-orders` (platform: quote, jobs,
gateways, docs; Catalyst: this note and a combine test). Not pushed or deployed.

## Problem

A customer testing a model on x environments had to place x orders when any set was agentic (SWE, TMax): the
platform quote refused a second set with `one_set_per_episode_evaluation`. Each order then paid its own audit
executor boot, its own run, its own report. The GPU side was never the limit: the order control already sizes
audit executors per model across all jobs (`control_status`, `executors_wanted` per `(model, revision)`), and
reuses them between jobs of the same model. So this is an order-shape and pricing change on the platform.

## What already existed

- Forced-draw orders on several single-turn sets (`parts_json`, migration 0050): one eval job per set
  (`order-eval-<id>-<k>`), created together, run side by side, each graded (`order-eval-<id>-<k>`), then
  `POST /admin/v1/evaluations/{eval_id}/combine` writes one report (`reliquary/eval-report-combined/v1`). A part
  without a report is left out and its share of the grading fee refunded.
- Catalyst is per job everywhere: the admin checks each job's `episode` against its set (`order_episode`), the
  control wires each episode job's signed intake and grader, publishes `jobs[id].sandbox` measures, and the
  sandbox quota counts per job. Nothing assumes one job per order. `combine_reports` only requires one model
  (`provenance.model` and `checkpoint_sha256`); parts may differ in `max_new_tokens` or episode.
- Gateway demand (`gatewayDemand`, `orderGroupJobs`) already maps each set of an order to its part's job.

## Change (platform)

Order shape:

- An agentic set goes in `envs` like any other. Its entry may carry `harness`, `max_turns`,
  `max_episode_tokens` and `max_new_tokens` (turn budget) over the order's; on a single-turn set they are refused
  (`episode_fields_single_turn`). The order's `max_new_tokens` stays the single-turn budget and the default turn
  budget. Thinking stays order-wide, so an order with an agentic set runs `thinking: false`.
- `pricing.episodes`: per agentic set `{set_id, harness, max_turns, max_episode_tokens, max_new_tokens,
  tokens_per_sample, job}`. `pricing.episode` is kept for the one-agentic-set case (single-set path, older
  readers). Quote and order views add `episodes`.
- Jobs: `advanceParts` sends each agentic part's `episode` and its turn budget as `max_new_tokens` (the admin
  requires them equal), and grades the part with that budget. Single-set orders are unchanged.

Pricing (`subnetPricing`):

| Cost | Rule |
| --- | --- |
| qualification, executor boot | once per order (model-level) |
| executor generation hours | once, over the summed tokens (the executors are shared) |
| tokens (input and output) | summed per set, each at its own per-sample bound |
| sandbox gateways | one gateway quote per agentic set (`pricing.sandbox_gateway_sets`), gateway-hours summed |
| grading fee | one per set (`grading_sets`); a missing part refunds its share |
| wall clock (`eta_hours`, `max_hours`) | max over sets (side by side), not the sum; within `EVAL_RUN_TIMEOUT_SECONDS` |

`pricing.sandbox_gateways` (single object) is written only when the order has one agentic set, so older clients
and the one-set path read it as before.

Gateways at run time: `paidSessions` reads the group's quotes from `sandbox_gateway_sets` (summing two sets of
one group), falling back to the legacy single object. Gateways are rented per image group, not per order, so two
agentic sets of different groups get their own groups; two of the same group share gateways (each priced as if
its own, which only lowers our cost).

## Not changed / limits

- D1 schema: none (episodes and gateway quotes live in `pricing_json`, parts in `parts_json`).
- Catalyst code: none needed. A unit test pins that the combined report accepts parts graded with different
  turn budgets.
- The grading fee is now per set for single-turn multi-set orders too (it was one per order).
- Per-set `max_new_tokens` is offered for agentic sets only; single-turn sets share the order's.
- Composability with the in-flight platform branches (both order-level today, both conflict in `createQuote`):
  - `feat/swe-verified-full` adds an order-level `profile` resolved by `episodeTemplate(set_id, profile)` and
    passed to `orderGatewayQuote`. On merge: `profile` joins the per-set episode keys (the order's as default),
    each agentic set resolves its own template, and `orderGatewayQuote` takes the profile per set entry.
  - `feat/agentic-price-coefficient` (uncommitted) multiplies the per-token price and the job's token ceilings
    for an `agentic: true` order. In a mixed order that would also charge single-turn tokens x2: on merge the
    factor applies to the agentic sets' tokens only (split the token term per set), and each part's job gets its
    own ceiling (`jobTokenCeiling` per part instead of one per order).

## Remaining

- Merge order with `feat/swe-verified-full` and `feat/agentic-price-coefficient` (both touch `createQuote`).
- UI: show `episodes` and `sandbox_gateway_sets` on the quote and order pages (none read them today).
- Live test on a draw-mode deployment: one order with SWE + TMax + one single-turn set, checking the executors
  boot once and both gateway groups scale from the control's per-job measures.
