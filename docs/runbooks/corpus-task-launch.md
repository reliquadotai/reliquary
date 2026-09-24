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

The rehearsal runs against MinIO, not R2. The job, record and settlement stores
rely on conditional puts (`If-None-Match: *` to create, `If-Match: <etag>` to
replace), and MinIO passing says nothing about R2. Before launch, spot-check both
on the production bucket with a throwaway key: a create over an existing key and
a write against a stale ETag must each be refused (412 / `PreconditionFailed`),
and a write against the current ETag must succeed. Do not launch if either
refusal is missing: two validators, or a retry, could then double-consume a slot
or pay a verdict twice.

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
  --min-new-tokens 16 \
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
- `--max-new-tokens` is omitted on purpose: the job then takes the budget the
  template gives the source (32768 for DAPO maths on the Teutonic profile), the
  length the RL task already generates to. A short cap cuts every reasoning
  completion before its answer: in the 512-token rehearsal the filter kept 3 of
  80. Pass it only to override the template.
- `--min-new-tokens` at least 16. Never 1: the terminator counts, and 1 pays a
  slot for an empty completion (the parser refuses below 2).
- The price is pinned: `floor == cap`, paid per verified token. The carried
  contract gets an enforced toploc proof (the template's own, or the deployed
  defaults); the corpus validator refuses a contract without one.
- `--grader-id/--threshold` only annotate the export; the filter never decides
  payment.
- The minimum-incentive floor is per task: a hotkey's share is measured within
  its task and what the floor cuts is redistributed within that task only. A
  corpus task is declared with `--min-incentive-share 0` by default, so every
  verified token is paid; raise it later with
  `reliquary tasks set-cap --task-id corpus-<name> --cap <cap> --min-incentive-share <x>`.
  Tasks that declare no floor (the RL task) keep the protocol's 1% -> 2% ramp.

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

**Audit throughput bounds the fleet for this first job.** The auditor prefills
one completion at a time; the rehearsal measured about 1.5k completion tokens/s
on one H100 against about 470 tokens/s generated per miner, so one corpus
validator keeps up with about 3 miners. More miners only grow the backlog
(payment is delayed, nothing is lost); watch it (section 5) and cap the number
of miners admitted for this job accordingly.

A validator-side audit failure (a lost GPU, an out-of-memory) is retried on the
next pending rescan (every 60 s); after 5 in a row the process exits non-zero
rather than run while paying nobody. A bucket outage answers miners 503, which
they retry.

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
  reliquary jobs status <job>
  # <job>: submissions=N verdicts=M unaudited=N-M settled=S unsettled=M-S pending=none last_window=W
  # drained: no
  ```

  The rehearsal audited about 1.5k completion tokens/s (one completion per
  prefill, sdpa) against about 470 tokens/s generated by one miner: about 3
  miners per corpus validator.
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

Retiring the task (`jobs cancel`) is a boot gate: once the corpus validator
stops it can never start again on this task. Anything accepted but not yet
audited, or audited but not yet settled, at that moment is never paid. Drain
first, with the corpus validator still running:

1. **Stop new work.** Either the job completes (every slot filled: miners get
   `job_complete` and stop by themselves), or, for an early stop, close
   `<port>` to miners at the firewall while leaving the validator process
   running (the auditor and the settler do not use the port). Miners then only
   retry; nothing new is accepted.
2. **Wait for every submission to be audited:**

   ```bash
   reliquary jobs status <job>     # until unaudited=0
   ```

   At about 1.5k tokens/s a backlog clears at that rate; a count that stops
   moving means the auditor is failing (check the validator log).
3. **Wait for the settlement that pays them:**

   ```bash
   reliquary jobs status <job>     # until unsettled=0 pending=none, i.e. "drained: yes"
   ```

   While the RL task is live the settler writes at most one archive per RL
   window (about 16 min); when every other task is idle it advances at most
   once per 16 min. Expect up to one RL window of wait after step 2.
4. **Stop the corpus validator**, then retire the task:

   ```bash
   reliquary jobs cancel --job-id <job> --retired-at <drand-round>
   ```

The manifest stays for export, and the task's EMA decays by itself. Raise the
RL cap back with `tasks set-cap` only once the corpus task's EMA tail has
decayed (section 1.3).
