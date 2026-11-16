# One environment catalog: RL, eval and batch

Status: phase 1 implemented (`reliquary/env_catalog.py`). Phases 2-5 are the plan.

## Why

What an environment may be used for was spread over a dozen lists: `RL_PROMPT_RANGES`, `RL_ONLY_RANGES`,
`CORPUS_SERVABLE_RANGES`, `HELD_OUT`, `ORDER_ENVIRONMENTS`, `BRIDGED_EVAL_PACKAGES`, plus the platform's own
`generation_envs` rows and `shared/generation-environments.json`. Nothing said in one place whether an
environment trains, evaluates or generates, and two holes followed:

* a batch job or generation order on `reliquary_code_v1` could take the RL rows `[200000, 2381806)`;
* an RL task's episode env could name an eval split (`reliquary-swe` `eval`, `reliquary-terminal` `tmax_sft`).

## The catalog

`reliquary/env_catalog.py` holds one entry per environment, keyed by the name tasks and jobs use: a registry
environment (`reliquary_code_v1`, `envscaler_tools_v2`) or an episode package (`reliquary-swe`). An entry lists
`Rows`, each a range of one split with a use:

| use     | who draws it                                                                 |
|---------|------------------------------------------------------------------------------|
| `rl`    | an RL task (prompts, or tasks of an episode env)                              |
| `batch` | our corpus jobs (their traces may train the model RL trains); agentic traces too |
| `eval`  | an evaluation set; nothing else ever draws an eval row                        |

An environment's flags are the uses of its rows. Eval rows overlap nothing. RL and batch rows overlap only
under a recorded `shared` ruling (today: DAPO math, science, logic v2 and general, where no split is decided).
A range carries `live: false` when the path that would draw it does not serve it yet (SWE `r2e`, `polyglot`).
`check_catalog()` runs at import.

Every installed registry environment is either in the catalog or named in `LEGACY_ENVIRONMENTS` (a test holds
this), so a new environment ships with a decision on what it serves.

## What reads it

| was                                     | now                                                         |
|-----------------------------------------|-------------------------------------------------------------|
| `rl_prompt_ranges.RL_PROMPT_RANGES`     | the RL range of train of each single-turn or suite entry     |
| `eval.sets.RL_ONLY_RANGES`              | every RL range without a `shared` ruling                     |
| `eval.sets.CORPUS_SERVABLE_RANGES`      | batch ranges of train; enforced for every catalog entry      |
| `eval.sets.HELD_OUT`                    | eval ranges that name an eval set                            |
| `qualification.ORDER_ENVIRONMENTS`      | entries with `orders`                                        |
| `corpus.job.BRIDGED_EVAL_PACKAGES`      | entries with `bridged`                                       |
| (nothing)                               | a corpus job's split: batch rows for training, never a training-only split for an eval |
| (nothing)                               | an RL episode block: a catalog package, an RL split, rows inside its RL ranges |

Behaviour changes, all in the refusing direction:

* our corpus jobs on `reliquary_code_v1`, `reliquary_hard_math_v1` or `reliquary_instruction_following_v1`
  take only their batch rows; their RL rows are RL's;
* a customer's order (the admin order route, generation qualifications) names its own rows: any RL or batch
  rows, never eval rows (`refuse_order_rows`); keeping its training data apart is the customer's call;
* RL on `reliquary_telecom_solo_v2` draws only its RL share `[1220, 1713)`;
* an RL episode block on a package outside the catalog, on a split without RL rows, or over rows outside the
  RL ranges (no `rows` on SWE's shared train split) is refused;
* an eval job never runs a split the catalog gives to training only.

`reliquary envs uses` prints the catalog; `reliquary envs uses --json` is the document the platform reads
(`reliquary/env-catalog/v1`).

## Plan

1. **Catalog** (this change).
2. **Training envs.** The platform reads the document instead of `generation-environments.json` and the
   `generation_envs` CHECK constraint; its prices and availability stay in D1. Serve agentic batch on the
   training splits marked `live: false` (SWE `r2e`, `polyglot`) after a run on a gateway, then flip them.
   Add training environments where the catalog has none (the agentic families have one RL split each).
3. **One RL controller per task**: task-scoped routes, miners that choose among tasks, and an admin route that
   creates a task from an `adaptive_training` service contract, its environments checked against the
   catalog's `rl` rows.
4. **Platform training orders, full fine-tuning**: trainer rental, pay in alpha from the customer's dollars,
   partial settlement, checkpoint and metric delivery.
5. **Multi-tenant LoRA** on one Miles gateway.

## Open decisions

* TMax has no batch share for our own SFT: `tmax_rl` is RL's and `tmax_sft` is the terminal eval. A
  customer's order may draw `tmax_rl` once agentic orders serve training splits.
* The `shared` rulings (DAPO math, science, logic v2, general) keep today's behaviour; each wants a split.
