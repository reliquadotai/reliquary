# Corpus task launch

This runbook takes one corpus generation job from nothing to a delivered
dataset: a frozen checkpoint, a prompt source, miners paid per verified token
out of a fixed share of emission. The order of the sections is the production
order and is not optional (design spec §10): a step taken early can stop every
validator of the fleet or move RL miners' weights.

Placeholders: `<validator-ip>`, `<port>`, `<job>`, `<name>`, `<repo>`, `<rev>`,
`<sha256>`, `<rl-profile-id>`, `<template-profile-id>`, `<source>`,
`<renderer-id>`, `<eos-id>`, `<drand-round>`.

## 0. Rehearse on a test card

Before touching the production registry, run the end-to-end script on a test
H100 (never the production worker). It declares a job in a throwaway MinIO,
starts the corpus validator, runs an honest miner (20 steps) and a dishonest one
(another checkpoint claiming the job's fingerprint, 5 steps), waits for every
verdict, settles, reads the archive back and exports:

```bash
R2_BUCKET_ID=reliquary-corpus-e2e \
  setsid nohup python scripts/corpus_e2e.py --start-minio \
    --state-dir /opt/corpus-e2e > e2e.log 2>&1 </dev/null &
# poll e2e.log for the JSON summary; "ok": true is the pass
```

It refuses `R2_BUCKET_ID=reliquary` (and an unset one, which defaults to it),
refuses a card that already has memory in use, and checks first that the bucket
refuses a create over an existing key and a write against a stale ETag: the job,
record and settlement stores are wrong without both. Expected: every honest
verdict passes, every dishonest one fails `exp_mismatch`, the archive pays only
the honest hotkey and sums to the cap, and the export has `steps x n` rows.

## 1. Preconditions

### 1.1 Every validator runs a binary that knows `corpus-generation`

One corpus entry makes the whole registry unreadable to a binary without the
mechanism, and those validators refuse to start. `jobs create` refuses unless
`--fleet-knows-corpus-generation` is passed; pass it only once every validator
of the fleet (trainer and weight-only) has been redeployed on this binary.

### 1.2 The TOPLOC band is measured on the job's checkpoint (BLOCKING)

The honest band and the vLLM decode capture were measured on one checkpoint
only, Qwen3-4B-Base (906bfd4b), on one H100:

- 2026-09-22 bench: honest miner and verifier both Qwen3-4B-Base; the sibling
  Qwen3-4B generating was caught, fp8 Qwen3-4B-Base was not refused by 60/40/40.
- 2026-09-24 rehearsal (section 0): job checkpoint Qwen3-4B-Base, worst honest
  `exp_mismatch` 9 against a threshold of 60; the dishonest miner generated
  with Qwen3-4B and failed at 111-126.

Before a job on any other checkpoint, and in particular before a Teutonic job,
measure it on the test H100, honest and fp8:

```bash
python -c "from huggingface_hub import snapshot_download as s; print(s('<repo>', revision='<rev>'))"
# -> <snapshot-dir>
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 \
  python scripts/toploc_band.py --model <snapshot-dir> --out band-honest.json
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 \
  python scripts/toploc_band.py --model <snapshot-dir> --quantization fp8 --out band-fp8.json
```

Go only if every honest chunk passes the thresholds the contract will carry
(`jobs create` writes Prime Intellect's deployed 60/40/40 unless the template
carries its own toploc entry). If the capture hook does not fit the checkpoint's
architecture (Teutonic is a hybrid), stop: that is fixed in the miner first.
Note that 60/40/40 does not refuse fp8 on Qwen3-4B-Base; a job on it pays a
miner running fp8.

### 1.3 Room in the pool for the new task

Read the registry:

```bash
reliquary tasks list
```

The caps must sum to at most 1.0, and a retired task keeps its cap reserved
until its EMA tail decays. Then:

- **`default` is present** (the usual case): lower its cap with `tasks set-cap`.
  It changes only the entry's params, under compare-and-swap, and re-checks the
  whole registry (sum of caps, floor <= cap); the contract and its digest are
  untouched, so the task's startup checks still match. It refuses a retired or
  unknown task, and an RL floor above the new cap (pass `--floor` then).

  ```bash
  reliquary tasks set-cap --task-id default --cap 0.9
  reliquary tasks list    # default active, cap=0.900; total declared cap 0.9000
  ```

  The weight setter re-reads the registry on every replay, so the lowered clamp
  applies to payment without a restart; an RL validator already running keeps
  the cap it resolved at startup for its own reporting until it restarts.

- **`default` is absent** (the fleet runs on the legacy fallback): declare it
  first, at the lowered cap, on the profile the RL validators run today.
  Declaring any other task first un-arms the legacy fallback and stops every
  trainer on its next restart.

  ```bash
  reliquary tasks create --task-id default --profile-id <rl-profile-id> --cap 0.9
  reliquary tasks list    # default active, cap=0.900
  ```

Lowering the RL cap is the one change the RL task sees, deliberately. Nothing
else about RL payment may move: the corpus settler never writes an archive index
above the RL task's latest (spec §7b).

## 2. Fingerprint and declare the job

`jobs create` writes the job manifest and the corpus task's registry entry in
one command (task id defaults to the job id; name it `corpus-<name>`):

```bash
reliquary jobs fingerprint <repo> --revision <rev>
# -> <sha256>

reliquary jobs create \
  --job-id <job> --task-id corpus-<name> \
  --model <repo> --model-revision <rev> --model-architecture Qwen3ForCausalLM \
  --checkpoint-sha256 <sha256> \
  --from-profile <template-profile-id> \
  --prompt-source <source> --prompt-count 200 \
  --renderer-id <renderer-id> --eos-token-id <eos-id> \
  --slots-per-prompt 4 --n 4 \
  --min-new-tokens 16 --max-new-tokens 512 \
  --prompt-order miner_walk \
  --grader-id <source> --threshold 1.0 \
  --cap 0.1 \
  --fleet-knows-corpus-generation
```

- `--from-profile` must declare `<source>` with a prompt template;
  `<renderer-id>` is that template's id (the contract's
  `environments.<source>.prompt_template.id`). A mismatch is refused here.
- `--prompt-count` is checked against the source's own length, which builds the
  source: a dataset-backed one must be readable from this machine.
- `--eos-token-id` is the id the miner's vLLM stops on and the route judges
  termination against (151643 for Qwen3-4B-Base).
- `--min-new-tokens` at least 16. Never 1: the terminator counts, and 1 pays a
  slot for an empty completion (the parser refuses below 2).
- The price is pinned: `floor == cap`, paid per verified token. The carried
  contract gets an enforced toploc proof (the template's own, or the deployed
  defaults); the corpus validator refuses a contract without one.
- `--grader-id/--threshold` only annotate the export; the filter never decides
  payment.

Check both halves landed:

```bash
reliquary jobs list     # <job>  active  task=corpus-<name> cap=0.100
reliquary tasks contract --task-id corpus-<name> > corpus-<name>.contract.json
```

## 3. Start the corpus validator

One process on one H100 (the GRAIL validator's card is fine: nothing else of
the RL service runs in it). It loads the job's checkpoint in bf16, refuses to
start if the contract's model is not the job's checkpoint or the fingerprint
differs, serves `GET /corpus/job`, `GET /corpus/cursor/{hotkey}` and
`POST /corpus/submit`, audits every accepted submission (q = 1) and settles
every 60 s.

```bash
export RELIQUARY_TASK_ID=corpus-<name>
export RELIQUARY_TASK_CONTRACT=$PWD/corpus-<name>.contract.json
reliquary validate --wallet-name <wallet> --hotkey <hotkey> \
  --http-host 0.0.0.0 --http-port <port> --no-set-weights
```

- `RELIQUARY_TASK_ID` selects the registry entry and is also the only archive
  prefix the settler will write under; it refuses to archive under any other.
- Leave `--no-set-weights` (the default) when the hotkey's RL validator already
  sets weights: its weight-only replay reads every task's archives. Two setters
  on one hotkey burn the rate limit.
- The validator needs the R2 credentials of the production bucket and access
  to `<repo>` on the Hub.

## 4. Miners

On each miner, with the same contract file (the miner reads its toploc
parameters from it and refuses to start without a toploc entry):

```bash
export RELIQUARY_TASK_CONTRACT=$PWD/corpus-<name>.contract.json
reliquary corpus mine --validator-url http://<validator-ip>:<port> \
  --wallet-name <wallet> --hotkey <hotkey>
```

- Needs vLLM 0.30 with the decode capture hook; the generator sets
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` and `VLLM_USE_V2_MODEL_RUNNER=0` itself.
- `--gpu-memory-utilization 0.5` when the card is shared (e.g. with a
  validator); omitted, vLLM keeps its own default.
- `--max-steps N` stops after N submissions; 0 runs until `job_complete`.
- The miner downloads the job's checkpoint and refuses to start if its
  fingerprint differs from the manifest.

## 5. Watch

- **Audit backlog**: submissions vs verdicts. They must grow at the same rate;
  a growing gap only delays payment (nothing is lost), but it means the card
  cannot keep up with the fleet.

  ```bash
  aws s3 ls s3://reliquary/reliquary/corpus/jobs/<job>/submissions/ --endpoint-url <r2-endpoint> | wc -l
  aws s3 ls s3://reliquary/reliquary/corpus/jobs/<job>/verdicts/    --endpoint-url <r2-endpoint> | wc -l
  ```

  The rehearsal audited about 1.5k completion tokens/s (one completion per
  prefill, sdpa) against about 470 tokens/s generated by one miner.
- **Archives**: one per RL window at most, under
  `reliquary/tasks/corpus-<name>/dataset/window-<N>.json.gz`, where `<N>` is
  the RL task's latest index, never above it.
- **Settlement state**: `reliquary/corpus/jobs/<job>/settlement.json`
  (`last_window`, `pending` must return to null).
- **RL untouched**: the RL task's weights are unchanged except for the lowered
  cap.
- **Refusals**: validator logs `corpus submission refused` reasons; a
  dishonest checkpoint shows up as verdicts failing `exp_mismatch`.

## 6. Deliver

```bash
reliquary jobs export <job> --out corpus.jsonl --apply-filter --only-accepted
```

Without `--only-accepted` the rejected rows stay, annotated, for preference
data. The file is written beside `--out` and swapped in only once complete.

## 7. Stop

```bash
reliquary jobs cancel --job-id <job> --retired-at <drand-round>
```

This retires the task entry, which is a boot gate: a validator already serving
the job keeps admitting until it is restarted, so stop the corpus validator
too. The manifest stays for settlement, and the task's EMA decays by itself.
