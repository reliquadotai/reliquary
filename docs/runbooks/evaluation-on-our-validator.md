# Evaluating a checkpoint on our own validator

How to measure one of our checkpoints (Teutonic first) on SN81: miners generate,
**our** validator audits every completion with TOPLOC on its GPU, the
environment grades on CPU. No order control, no executors, no qualification:
those exist for customers' models on rented cards; here the verifier is the
validator we run, as for every corpus job.

The rule to keep in mind: **one validator process = one loaded model = one GPU.**
Jobs on a model already loaded join that process without a restart (hot jobs);
a job on another model is ignored by it and needs its own process on a card.

Placeholders: `<set>`, `<repo>`, `<rev>` (40-hex commit), `<sha256>`, `<job>`,
`<port>`, `<wallet>`, `<hotkey>`.

## 1. Build and publish the set

On the operator host (subnet bucket credentials `R2_*`; for a Verifiers source,
`verifiers` and the taskset packages installed, see the design doc):

```bash
# a catalog environment, any split and range, whole or sampled
reliquary eval build-set --source reliquary_dapo_math_v1 --split eval \
  --start 0 --count 2000 --sample 300 --seed 7 --out sets/dapo-eval-300
# or a public benchmark packaged as a Verifiers taskset
reliquary eval build-set --source verifiers:aime26 --out sets/aime26
reliquary eval build-set --source verifiers:gpqa --taskset-args \
  '{"diamond": true, "task": {"rewards": {"correct": {"fn": "reliquary.eval.verifiers_rewards:gpqa_letter"}}}}' \
  --out sets/gpqa-diamond

reliquary eval publish-set sets/aime26     # subnet bucket only, write-once
```

The card (`set.json`) records the source, the selection and the overlap of the
rows with RL and the prod corpus jobs. Nothing is refused for overlapping: the
report carries it beside the score.

New cards also seal the actual prompt/reference files, grader implementation and
single-turn harness into `protocol`, `protocol_json` and `protocol_sha256`.
`build-set --protocol reviewed.json` adds reviewed `data` artifact digests,
`licenses` entries (`id`, exact license-body `sha256`) and `disjointness`
(`disjoint`, `evidence_sha256`). Missing evidence stays unknown; importing a
Verifiers taskset does not establish that it was held out. Publication uses
conditional creation and never overwrites another publisher's frozen bytes.

## 2. Declare the job

```bash
reliquary jobs fingerprint <repo> --revision <rev>      # -> <sha256>

reliquary jobs create --fleet-knows-corpus-generation \
  --job-id <job> \
  --model <repo> --model-revision <rev> --model-architecture <arch> \
  --checkpoint-sha256 <sha256> \
  --eval-set <set> [--prompt-count N] \
  --renderer-id chat-template-thinking-v1 \
  --eos-token-id <eos-id> --max-new-tokens 32768 \
  --slots-per-prompt 8 --temperature 0.6 --top-p 0.95 \
  --cap 0.02
```

- `--eval-set` replaces `--prompt-source`: the job reads the set's first
  `--prompt-count` problems (all by default), checked against their sha256.
- `--renderer-id`: the model's chat template, `chat-template-thinking-v1` with
  reasoning, `chat-template-v1` without. Nothing else is accepted.
- `--slots-per-prompt` × `--n` is the number of samples per problem.
- Refused for an eval job: `--audit-q` other than 1.0 (every graded completion
  is audited), `--grader-id`/`--threshold` (every completion is graded),
  `--prompt-start`, `--from-profile`.
- The contract declares the set's environment: its catalog source, or
  `reliquary_external_eval_v1` for a Verifiers set.
- The validators of other models see the entry and ignore it.

## 3. Start the validator for that model

On a card with room for the model in bf16 (Teutonic, 9B: ~18 GB of weights):

```bash
reliquary tasks contract --task-id <job> > <job>.contract.json
export RELIQUARY_TASK_ID=<job>
export RELIQUARY_TASK_CONTRACT=$PWD/<job>.contract.json
reliquary validate --wallet-name <wallet> --hotkey <hotkey> \
  --http-host 0.0.0.0 --http-port <port> --no-set-weights
```

Everything in `docs/runbooks/corpus-task-launch.md` §3 applies (archive prefix,
weights, nginx route for miners). Several eval jobs on the same model are served
by one process: list them in `RELIQUARY_TASK_ID` with their merged contract, or
let the hot-job refresh wire the later ones.

It serves miners the set's lines at `/corpus/jobs/<job>/eval-prompts` and takes
their submissions at `/corpus/jobs/<job>/submit`.

## 4. Miners

Miners need a build that knows eval-set jobs (main at or after the merge of this
runbook's PR):

```bash
reliquary corpus mine --validator-url http://<validator>:<port> --job-id <job> ...
```

Follow it with `reliquary jobs status <job>` until `drained: yes`.

## 5. Grade and compare

On a host with the subnet bucket credentials (and `verifiers` + the tasksets for
a Verifiers set, Docker for LiveCodeBench):

```bash
reliquary eval grade --job <job> --out results/<job>
reliquary eval compare results/<job-start-checkpoint> results/<job>
```

- Refused until the job is drained; `--allow-incomplete` grades a job missing
  samples, counting them as failures.
- With reasoning on, a completion with no closing `</think>` is a format
  failure: it was cut before answering.
- `report.json`: pass@1 with its bootstrap interval, pass@k, missing and
  ungraded rows, truncation and format-failure rates, the TOPLOC thresholds of
  the task contract, the training overlap. `graded.parquet`: one row per
  completion.
- `eval compare` needs the same sets, samples, sampling, budget, reasoning mode
  and grader/protocol identities; it refuses ungraded rows unless `--allow-ungraded`.
  Legacy reports remain readable and comparisons label them `legacy-unpinned`.

## Pinned pod evaluations

`eval run` retains its single-turn default. A protocol-bearing lease uses one
spawned model process per order, an explicit total-transcript token cap and a
bounded inference call. The parent maintains the lease heartbeat and kills and
joins its owned model process group on timeout or lease loss. Model boot/download
has a separate one-hour bound. These software checks do not qualify a GPU's
capacity or a model's performance.
The main-thread executor also handles SIGTERM through the same cleanup path and
restores its previous signal handler before returning.

Public checkpoint downloads explicitly disable HF credentials. A lease declaring
`model_access=private` reads its credential through the current fenced model-credential
route. A trusted local loader caller may supply its credential directly. The token stays
in memory; no login, token file or secret command argument is created. The loader
checks the exact repository and 40-hex revision and disables remote model code.
The existing operator/subnet qualification commands are separate paths.

Reports retain actual completion/truncation/format metrics and per-problem
`mean_score` with `n_problems_graded`. Attempt counts and episode turn/tool metrics
remain null when not measured. The current executor does not advertise signed
agentic evaluation support.

Episode software uses the same runner: `telecom-episode-v1` replays the frozen
deterministic tools, and `signed-bash-v1` requires the pinned signed sandbox,
exact renderer, issued session and accepted worker lease. The declared SWE and
TMax policy is at most 40 turns; it is provisional, not measured capacity.
The current telecom renderer requires `thinking=true`; requests that disable it
are refused. Single-turn and signed renderers keep their explicit thinking toggle.
TMax splits are training corpus parts and acquire no held-out claim by default.
Public `--publish-reviewed` publication requires explicit source/rights/overlap
evidence and emits a `withdrawn` registration template for separate qualification.
Receipts that exceed the existing row allowance fail without truncating state.

The evaluation runtime uses its new exact sandbox commit in a separate process
or build. This is not a hot upgrade of active corpus/RL jobs pinned to an older
commit; those workers stay on their existing pin until a separately reviewed
rollover. Software tests do not establish available sandbox or GPU capacity.

## 6. Stop

Retire the task (`reliquary tasks retire --task-id <job> --retired-at <drand round>`), let the process drain and
settle, then stop it and free the card.

## 6b. During an RL run

Held-out evaluations of the run's milestone checkpoints, on one card beside the
run (the trainer never waits for them), shown on the platform's run monitor.

Trainer side: `RELIQUARY_RL_HF_BACKUP_PREFIX=<prefix>`,
`RELIQUARY_RL_HF_BACKUP_EVERY=<M>` and `RELIQUARY_RL_HF_BACKUP_MILESTONE_EVERY=<K>`.
Milestones (`ReliquaryForge/<prefix>-step-<n:08d>`, `n` a multiple of `K`) are
never rotated away; the step survives trainer restarts (it resumes from the
published checkpoint), so a name is never reused.

The suite, one entry per set (`short` goes in the job ids):

```json
[{"set_id": "verifiers-aime25-r0-n30", "short": "aime", "slots_per_prompt": 8, "cap": 0.02, "max_new_tokens": 32768},
 {"set_id": "verifiers-gpqa-diamond", "short": "gpqa", "slots_per_prompt": 4, "cap": 0.02, "max_new_tokens": 32768}]
```

On the eval host (GPU, subnet bucket `R2_*`, HF token, the wallet of section 3;
`R2_PUBLIC_*` for `--publish`):

```bash
reliquary eval schedule --run-id <wandb-run-id> --run-tag <tag> --suite suite.json \
  --every <N> --hf-prefix <prefix> --model-architecture <arch> --eos-token-id <eos-id> \
  --baseline-model <start-repo> --baseline-revision <rev> --results-dir results/<tag> \
  --wallet-name <wallet> --hotkey <hotkey> --http-port <port>          # dry run: prints the plan
# then the same with --execute --publish --upload --loop-seconds 300
```

- `<wandb-run-id>` is the trainer's wandb run, `{hotkey[:8]}-{run}-trainer`: the
  id the platform's run catalog holds. `<N>` must be a multiple of `K`.
- One checkpoint at a time (one model per process): the start checkpoint first,
  then the newest milestone. Its jobs `eval-<tag>-s<step:05d>-<short>` (base:
  `eval-<tag>-base-<short>`) are declared with its fingerprint, one validator is
  started for that model (`--start-cmd` overrides the command), the other jobs
  join it as hot jobs. Miners must mine those job ids.
- Once drained, each job is graded with `--rl-run-id`/`--rl-step` into
  `results/<tag>/<job>/`; a job missing samples after `--max-wait-hours` is graded
  with them as failures, one never drained is abandoned. Then the validator is
  stopped, `results/<tag>/benchmarks.json` rebuilt and published.
- Evaluations behind training: the milestones between the last one evaluated and
  the newest are marked skipped, never queued. To backfill one, grade it by hand
  (`eval grade --rl-run-id ... --rl-step ...`) and pass it to `eval feed --jobs`.
- State: `results/<tag>/schedule-state.json`; the command is idempotent, rerun it
  after a crash. A validator found dead is restarted.

The feed by hand, from local gradings or `evaluations/<job>/report.json`
(`eval grade --upload`):

```bash
reliquary eval feed --run-id <wandb-run-id> --job-prefix eval-<tag>-s \
  --baseline-job eval-<tag>-base-aime,eval-<tag>-base-gpqa \
  [--reports-dir results/<tag>] [--label verifiers-aime25-r0-n30="AIME 2025"] \
  --out benchmarks.json [--publish]
```

It is checked against the platform's validator (`shared/run-benchmarks.ts`)
before it is written. `--publish` puts it at `rl-runs/<wandb-run-id>/benchmarks.json`
in `reliquary-public`; that bucket is private, so the RL validator serves it at
`GET /rl/benchmarks?run_id=<wandb-run-id>`. On the platform:
`RELIQUARY_RUN_BENCHMARKS_URL=https://<rl-validator-public-host>/rl/benchmarks`
(the collector adds `run_id`).

## 7. Episode environments (telecom)

A suite episode environment (telecom v2: splits `eval` and `benchmark`) is evaluated
like a single-turn set, except the set holds **row references** (environment, split,
instance id), not prompts, and the job plays them as episodes. The validator replays
every episode itself (the suite harness), so there is no grade executor.

```bash
# the set: one source and split, whole or sampled; the card records the environment
# manifest and its renderer id
reliquary eval build-set --source reliquary_telecom_solo_v2 --split benchmark \
  --set-id telecom-v2-benchmark --out sets/telecom-v2-benchmark
reliquary eval build-set --source reliquary_telecom_solo_v2 --split eval \
  --set-id telecom-v2-eval --out sets/telecom-v2-eval            # a second set, a second job
reliquary eval publish-set sets/telecom-v2-benchmark
```

The episode file is the manifest's `episode` object for the suite harness
(`parse_job`, suite branch). `name` is the set's source, `version` the environment's
64-hex manifest sha256 (`env_manifest` in the set's `set.json`), `split` the set's split, `renderer` the
environment's renderer id (`renderer_id` in `set.json`); nothing else is accepted:

```json
{"env": {"package": "reliquary-suite", "name": "reliquary_telecom_solo_v2",
         "version": "<manifest-sha256>", "split": "benchmark"},
 "harness": "suite", "renderer": "<renderer-id>",
 "max_turns": <turns>, "max_tokens_per_turn": <tokens>, "max_total_tokens": <tokens>}
```

```bash
reliquary jobs create --fleet-knows-corpus-generation \
  --job-id <job> --model <repo> --model-revision <rev> --model-architecture <arch> \
  --checkpoint-sha256 <sha256> \
  --eval-set telecom-v2-benchmark [--prompt-count N] \
  --episode-file <job>.episode.json --renderer-id <renderer-id> \
  --eos-token-id <eos-id> --max-new-tokens <tokens> \
  --slots-per-prompt 8 --temperature 0.6 --top-p 0.95 --cap 0.02
```

- `jobs create` refuses an episode whose split, environment or renderer disagrees
  with the set, a set of episode tasks without `--episode-file`, and an
  `--episode-file` on a single-turn set. `--max-new-tokens` must equal the episode's
  `max_tokens_per_turn`; `--renderer-id` must equal the episode's `renderer` (not a
  chat-template id); `--slots-per-prompt` >= 2; the other eval refusals of section 2
  apply.
- The validator host must read the set's rows: `RELIQUARY_EVAL_SETS_DIR` pointing at
  the set directory, or the subnet bucket credentials (published set). It serves
  them at `/corpus/jobs/<job>/eval-prompts`; miners fetch them there.
- Miners need a release containing this branch; an older release refuses the
  manifest (unknown `prompt_source`), by design. They mine with
  `reliquary corpus mine-agentic --validator-url http://<validator>:<port> --job-id <job> ...`
  (needs only the environment's pinned wheel).
- Validator: section 3, on a Lium card holding the checkpoint
  (`RELIQUARY_TASK_ID=<job>`, the job's contract, `reliquary validate ...`).
- `reliquary jobs status <job>` until `drained: yes`, then
  `reliquary eval grade --job <job> --out results/<job>` (graded from the replay's
  grade documents) and `reliquary eval compare results/<start-checkpoint-job> results/<job>`
  against the RL start checkpoint, evaluated on the same set with the same episode.
- The headline pass@1 counts missing and ungraded episodes as failures; the
  graded-only figure sits beside it in `report.json`.
- Stop: `reliquary tasks retire --task-id <job> --retired-at <drand round>`, drain and settle, then stop the
  validator and the Lium card.

## 8. SWE-bench Verified (signed sandbox)

**Status: not run yet.** The code path is in place: the set's row map, `jobs create`
over a `swe` set with a `signed_sandbox` episode, the validator's sandbox verdicts and
`eval grade`. reliquary-sandbox `fc2fa643` serves an eval split when its served name
says `eval` (before, "eval is never served"), and `bridged_swe.GATEWAY_OPTIONS` now
serves `eval` (Verified is `reliquary-swe` split `eval`).

Before the first run, in order:

1. The new sandbox commit and gateway options rolled out together. The options change
   `options_sha256()` and the commit changes `SANDBOX_COMMIT`: **every gateway, miner
   and validator at once**, or no session is placed. A build at this commit refuses a
   job pinned to the previous one (`be50c768`): the signed SWE corpus job must be
   relaunched on the new commit, or closed, at the same time.
2. Verified image digests pinned (design Task 8, deferred): the images are tagged
   `latest`, and the bridge refuses a task whose image its lock does not pin.
3. The sampled images pre-pulled on the sandbox machines, from the set's instance ids
   (`reliquary_swe/images.py --split eval`). Start with a 50-100 instance sample.
4. `reliquary-swe` installed where `jobs create` runs: it checks the instance ids.
5. Then as section 7: build and publish the set, a `signed_sandbox` episode file
   (as the SWE corpus jobs, `docs/runbooks/agentic-corpus-swe.md`; `env.split` the
   set's split, never `train`: a signed eval job naming `train` is refused, an old
   miner would mine the wrong rows), `jobs create --eval-set ... --episode-file ...`,
   miners with `mine-agentic --validator-hotkey <validator-ss58>`, validator on a
   card, `eval grade`, `eval compare`, retire, stop.
