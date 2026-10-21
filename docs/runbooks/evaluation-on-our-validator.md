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
TMax splits are training corpus parts and acquire no held-out claim by default.
Public `--publish-reviewed` publication requires explicit source/rights/overlap
evidence and emits a `withdrawn` registration template for separate qualification.
Receipts that exceed the existing row allowance fail without truncating state.

The evaluation runtime uses its new exact sandbox commit in a separate process
or build. This is not a hot upgrade of active corpus/RL jobs pinned to an older
commit; those workers stay on their existing pin until a separately reviewed
rollover. Software tests do not establish available sandbox or GPU capacity.

## 6. Stop

Retire the task (`reliquary tasks retire --task-id <job>`), let the process drain and
settle, then stop it and free the card.
