# Evaluations on any environment, run by the subnet

Date: 2026-10-17. Branch: `feat/rl-async-evals` (off `feat/rl-async`).

## Why

The RL async run needs an eval curve: score the latest milestone checkpoint every so often.
Until now the operator paid a GPU and ran evals by hand. The subnet can do the generation
(miners), and doing it there is the point twice over:

- **Dogfooding.** The subnet eval path (`jobs create --eval-set`, our validator auditing every
  completion with TOPLOC, `eval grade`, `eval compare`; runbook
  `docs/runbooks/evaluation-on-our-validator.md`) has never run for real. The RL run becomes its
  first regular customer and finds its bugs before a customer does.
- **Marketing.** "Our RL checkpoints are evaluated by the network, audited, reproducible" is a
  claim a customer can check. Public benchmarks (AIME, GPQA, LiveCodeBench, IFBench,
  tau2-bench telecom, SWE-bench Verified) carry it.

Cost is secondary: one validator card per eval (rented on Lium, up only while the eval
drains) is accepted.

## What exists

An eval job is a corpus job whose `prompt_source` is `eval-set:<set_id>:<n>:<sha256>` (the
first n lines of a frozen set, `reliquary/eval/prompt_source.py`), with three rules
(`reliquary/cli/main.py` `_eval_job_source`): no filter, `audit_q = 1.0`, the set's environment
as the contract's. It is **single-turn only**: `build_source_set` refuses non-single-turn
sources (`reliquary/eval/sets.py` `_catalog_spec`), `EvalSetSpec` serves user-text rows, and
`grade_served_job` re-grades completions on CPU.

Corpus jobs already run every kind of multi-turn environment, and each kind already grades
every episode and pays per graded item, not per success
(`corpus_period_settlement.py:66-70`, `corpus_grading.py:18-23`):

| Kind | Example | Graded by | Grade document |
|---|---|---|---|
| single-turn | dapo math, IF, logic, Verifiers tasksets | CPU, at `eval grade` | (none; `eval grade` grades) |
| suite episode | `reliquary_telecom_solo_v2`, `envscaler_tools_v2` | validator replay (`suite_grading.py`) | `graded_success`, `grade.reward`, `replay_certified` (always, when replayed) |
| agentic SWE | `reliquary_agentic_swe_v1` | remote grade executors (`corpus_grading.py`) | `graded_success` = `tests_passed`; certified only when replayed |
| signed-sandbox SWE | SWE-smith via gateways | sandbox verdict (`signed_grading.py`) | `graded_success` = reward 1.0, always certified |

What blocks evals on the multi-turn kinds is that every one of them reads `prompt_index` as a
row of the **train** split over a **contiguous** range `[prompt_start, prompt_start + N)`:

- `job.py:487-488` (suite) and `:514-515` (agentic/signed) refuse any split but `train`.
- `SuiteSource` (`suite_episode.py:198-214`) and `load_swe_source` (`agentic_swe.py:193-202`)
  build train only; `replay_kwargs` (`suite_intake.py:301-307`) sets `task_index = prompt_index`;
  the replay worker rebuilds the env with `spec.create()` (`suite_intake.py:144-148`).
- `EnvironmentSpec.create()` takes no split (`registry.py:239-279`);
  `load_external_backend` allows train/eval/qualification but refuses `benchmark`
  (`agentic/external.py:577-591`).
- SWE: `sandbox_split()` always returns `train` (`job.py:653-659`); `bridged_swe.SERVED_SPLITS`
  has no `eval` (`bridged_swe.py:14-15`); the grade executor resolves the task through
  `swesmith_row` + `task_for(..., "train")` (`agentic_replay.py:462-466`); `GradeItem` carries no
  split (`corpus_grade_protocol.py:52-58`); the non-signed miner indexes SWE-smith rows whatever
  the split says (`miner/agentic_episode.py:97-104`).
- Agentic SWE failures are certified only when the drand draw replays them
  (`replay_fraction_failed`, `corpus_grading.py:411-421`), and the export drops uncertified rows
  (`delivery.py:404-405`): an eval would lose failures and read inflated.

## Goals

1. **Any environment kind is evaluable** (single-turn, suite episode, signed-sandbox SWE)
   by one mechanism: an eval set lists rows as `(source, split, source_index)`, and an eval job
   serves row i of the set as its row i.
2. **One rule set for every eval job**, enforced at parse time, not by the operator's flags.
3. **One `eval grade`** whose report (`report.json`, `graded.parquet`) has the same shape for
   every kind, so `eval compare` works unchanged.
4. **First targets:** telecom v2 split `eval` and split `benchmark` (the 134 tau2-bench telecom
   tickets); SWE-bench Verified (`reliquary-swe` split `eval`, 500 rows), first a 50-100 sample.
5. Launched **by hand** at first (runbook section). Automation later.

## Non-goals

- No orchestrator, no automatic GPU spin-up, no wandb push in this work. (Later: watch the
  trainer's HF milestones, create the job, rent a Lium card, wait for drain, grade, compare,
  stop the card.)
- No change to how non-eval corpus jobs behave, and no change to RL.
- No eval on `envscaler_tools_v2` for now: its only scenarios unseen by RL and SFT are the 13
  worlds a 27B almost never wins (`SFT_CORPUS_EXCLUDED_WORLDS`). The mechanism supports it
  (index space only); the set choice waits for a real held-out region.
- BFCL (a Verifiers multi-turn taskset) is not covered: it is not a catalog environment.
- The GPU-less order control (`eval_control.py`, `order-eval-` jobs, customers' orders with
  executor-pair audits) stays single-turn. Our evals go through our own corpus validator
  (user decision 2026-10-03); episode orders for customers are later work.

## Anti-goals (failure even if it "works")

- **A biased score.** Any path where failed episodes disappear (uncertified, filtered, unpaid
  so miners skip them) makes the curve a lie. Every row of the set is either graded or counted
  as missing, and the report says how many.
- **Training on eval rows.** A set over rows RL or SFT uses must be visible on its card (the
  current overlap report), and the held-out regions declared for the new sets must be refused
  to corpus jobs like the existing ones (`refuse_held_out_overlap`).
- **Two ways of saying "this is an eval job".** The eval-set prompt source stays the single
  criterion (`is_eval_source(job.prompt_source)`); nothing keys on job id prefixes or flags for
  the corpus-validator path.

## Design

### 1. Sets hold references

`set.json` gains `interaction` values beyond `single_turn`: `episode` (suite) and `swe`
(agentic or signed). For these, `prompts.jsonl` rows are references, not text:

```json
{"problem_id": "<set_id>-000017", "set_id": "...", "env": "<source>",
 "interaction": "episode", "split": "benchmark", "source_index": 42,
 "instance_id": "<the env's task id at that index>"}
```

- `instance_id` is read from the environment at build time and checked again wherever the row
  is served (the export already checks `grade.instance_id == source.instance_id(index)`).
- `grading.jsonl` is not needed for these kinds (the grade comes from the job's grade
  documents) and is written with the same rows for symmetry, so publish/read code stays one path.
- The card records the renderer the rows will be run with (the episode's renderer, e.g.
  `reliquary-chatml-tools-v1` for telecom: the Qwen tool-call dialect the model was trained in),
  and `eval compare` requires it equal. Scores on the `benchmark` split are tau2-bench telecom
  tickets in **solo mode with our harness**: the card and the report say so, and a public
  comparison quotes them as such, not as the official leaderboard protocol.
- `build-set --source <env> --split <split> [--start --count --sample --seed]` accepts episode
  and SWE catalog sources. The overlap report (`range_overlaps`) applies as today.
- New held-out regions, declared beside `HELD_OUT` and refused to corpus jobs:
  telecom v2 `eval` and `benchmark` (whole splits), `reliquary-swe` `eval` (whole split).
  These are splits other than `train`, so corpus jobs (train only) cannot reach them today;
  the declaration is for the card and for the day a corpus job may read another split.

### 2. One row map, in the source

For an eval job, `prompt_index` stays in **row space** `[0, n)`: slot ledgers, ownership
(`job.owns`, `_owned_position`), `episode_seed(job_id, prompt_index)`, walk order and payment
are unchanged. The mapping row -> `(split, source_index)` happens in one place per kind:

- **Suite:** `SuiteSource` built from the set: `task(i)`, `instance_id(i)`, `initial_text(i)`
  read `environment[split].get_task(source_index[i])`. `replay_kwargs` carries `split` and the
  mapped `task_index`; the replay worker builds the env for that split.
- **Signed SWE** (the SWE path for evals, user decision 2026-10-17): `sandbox_split()` returns
  the job's split; the `num_images == BRIDGED_SWESMITH_IMAGES` rule (`job.py:576-579`,
  `:653-659`, `agentic_swe.py:249-268`) applies to `train` only; `bridged_swe.SERVED_SPLITS`
  gains `eval`, and the gateways publish it (their `options_sha256` must match,
  `bridged_swe.py:29-34`); `SweTaskResolver(split)` already takes a split. Signed verdicts are
  always certified, so no failure is lost.
- **Agentic SWE** (remote grade executors) is **not** made evaluable: an eval there would need
  every failure replayed (`replay_fraction_failed = 1.0`) and the Verified images on every
  executor. A job with an eval source and an agentic SWE episode is refused.
- **Registry:** `EnvironmentSpec.create(split="train")`; external episode backends pass the
  split to `load_external_backend`, whose hard-coded list gains `benchmark` (no change to any
  package, artifact or manifest pin; the package itself refuses a split it lacks).

Routing: `resolve_prompt_source` checks `is_eval_source` **first**. The set's card says which
kind it is; a single-turn set gives `EvalSetSpec` (unchanged), an episode set requires a suite
`episode` and gives the mapped `SuiteSource`, a SWE set requires an agentic or signed `episode`
and gives the mapped SWE source. The job's `episode.env.split` must equal the set's split
(the train-only pin is lifted for eval jobs only). `job.py`'s "prompt_source must be the
episode env name" rule accepts an eval source whose card names that env.

Miners read the job's rows from `GET /corpus/jobs/{job}/eval-prompts` as they do for single-turn
eval jobs (references instead of text); the episode they run is built from the reference.
**This is a miner protocol change.** Public miners mine these jobs (that is the point), so an
old miner must fail closed, never run train row `prompt_index` on a mapped job:

- an eval episode job's manifest carries a field old parsers refuse (the eval rows' reference
  to the set), so an old miner cannot load the job at all rather than mining the wrong row;
- the validator checks at intake that the trajectory's task is the mapped row's
  (`instance_id`, and for suite the initial prompt ids, as today) and refuses a mismatch as a
  **submission refusal** (`wrong_task`), not as an audit failure, so it never eats coverage
  silently and shows in the job's reject counters;
- the miner release ships before the first episode eval job is declared (runbook step).

### 3. Eval rules, enforced at parse time

For any job whose prompt source is an eval set (`parse_job`, so both `jobs create` and every
validator refuse a manifest that breaks them):

- `audit_q == 1.0`, no grader/threshold filter, `prompt_start == 0` (as today);
- episode kinds: renderer is the episode's renderer (not a chat template: today
  `_eval_job_source` demands a chat template, which no episode renderer is);
- the export never applies `--sft` to an eval job.

### 4. One `eval grade`

`grade_served_job` dispatches on the set's kind:

- single-turn: as today (CPU grading against `grading.jsonl`);
- episode / SWE: read each passing submission's grade document; score = `grade.reward`
  (suite, signed); `status != ok` or not `replay_certified` is
  **ungraded**. Ungraded is not random (replays past their deadline are the long episodes,
  which tend to be failures), so the **headline pass@1 counts ungraded rows as failures**
  (never inflated), and the report shows beside it pass@1 over graded rows only and the
  ungraded count. `eval compare` keeps refusing ungraded rows unless `--allow-ungraded`.
  Audit-failed submissions are not the model's: excluded and their slots re-served, as for
  single-turn.
- Same report: pass@1 with bootstrap interval, pass@k, missing/ungraded counts, plus for
  episodes the mean reward, turn count and termination reasons.

### 5. SWE-bench Verified operations

- Verified images (`swebench/sweb.eval.<arch>.<id>`) are tagged `latest`, not pinned
  (`reliquary_swe/swe_adapter.py:119-154`). Pin them by digest like SWE-smith
  (`scripts/pin_swesmith_digests.py`) and check them for mounts like `refuse_mounts` does.
- The sandbox gateways must pre-pull the sampled images
  (`reliquary_swe/images.py --split eval` prints the list). Start with a 50-100 instance sample.
- The grader already handles `eval` (`reliquary_swe/grading.py:676-685`, `swe_adapter`,
  `test_patch` restore); only local goldens prove it so far.

### 6. Hand launch

A section of `docs/runbooks/evaluation-on-our-validator.md` per kind: build/publish the set,
`jobs create --eval-set ... --episode-file ...`, validator on a Lium card with the checkpoint,
the gateways/executors the kind needs, `eval grade`, `eval compare` against the RL start
checkpoint, stop the card.

## Order of work (each step usable alone)

1. Single-turn sets built and published (no code), first real eval of the RL start checkpoint.
2. Sections 1-4 for suite episodes; telecom `eval` + `benchmark`.
3. Signed-sandbox SWE + section 5; Verified sample (50-100), then the 500.

## Left to the builder

- Exact field names on the card and rows; how the mapped sources are constructed and cached.
- Whether `EnvironmentSpec.create(split)` or a separate builder; keep RL call sites untouched.
- Report extras for episodes beyond pass@1 / mean reward.

## Calls made without the user (to check)

- Eval jobs stay recognised by the eval-set prompt source alone (no new job field).
- Row space `[0, n)` for every eval job, mapping in the source.
- Headline pass@1 counts ungraded episodes as failures; graded-only pass@1 shown beside it.
- Episode evals use the RL renderer; tau2-bench numbers are labelled "solo mode, our harness".
- Allowed external splits stay a hard-coded list (+ `benchmark`), no package release.
- Old miners fail closed on eval episode jobs; a wrong-task trajectory is a submission refusal.
- envscaler and BFCL out of scope.
