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

Retire the task (`reliquary tasks retire --task-id <job>`), let the process drain and
settle, then stop it and free the card.
