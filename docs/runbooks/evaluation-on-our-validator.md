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
  and grader versions; it refuses ungraded rows unless `--allow-ungraded`.

## 6. Stop

Retire the task (`reliquary tasks retire --task-id <job> --retired-at <drand round>`), let the process drain and
settle, then stop it and free the card.

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

**Status: cannot run yet.** The code path is in place: the set's row map, `jobs create`
over a `swe` set with a `signed_sandbox` episode, the validator's sandbox verdicts and
`eval grade`. But the pinned `reliquary-sandbox` refuses to serve any split whose name
contains `eval` ("eval is never served"), and Verified is `reliquary-swe` split
`eval`. The gateway options were not changed. Serving Verified waits for a decision:
change the sandbox guard, or serve Verified under a split name without `eval`.

Once decided, the run needs, in order:

1. Gateway options updated (`bridged_swe.GATEWAY_OPTIONS`), which changes
   `options_sha256()`: **every gateway at once**, or no session is placed.
2. Verified image digests pinned (design Task 8, deferred): the images are tagged
   `latest`.
3. The sampled images pre-pulled on the sandbox machines, from the set's instance ids
   (`reliquary_swe/images.py --split eval`). Start with a 50-100 instance sample.
4. `reliquary-swe` installed where `jobs create` runs: it checks the instance ids.
5. Then as section 7: build and publish the set, a `signed_sandbox` episode file
   (as the SWE corpus jobs, `docs/runbooks/agentic-corpus-swe.md`; `env.split` the
   set's split, never `train`: a signed eval job naming `train` is refused, an old
   miner would mine the wrong rows), `jobs create --eval-set ... --episode-file ...`,
   miners with `mine-agentic --validator-hotkey <validator-ss58>`, validator on a
   card, `eval grade`, `eval compare`, retire, stop.
