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

**Partial audit rehearsal.** The same script rehearses sampling (§2.1) with a
hotkey that mines honestly through probation and then switches model:

```bash
R2_BUCKET_ID=reliquary-corpus-e2e \
  setsid nohup python scripts/corpus_e2e.py --start-minio --state-dir /opt/corpus-e2e \
    --honest-model <repo> --honest-revision <rev> --base-profile <template-profile-id> \
    --model-architecture <arch> --dishonest-model <other-repo> --dishonest-revision <rev> \
    --prompt-source <source> --max-new-tokens 512 \
    --honest-steps 20 --dishonest-steps 5 --late-cheater-steps 30 \
    --audit-q 0.2 --audit-probation 5 --audit-hold-seconds 300 > e2e.json 2> e2e.log </dev/null &
```

The late cheater's honest phase runs until exactly `--audit-probation`
submissions are accepted, then a second process under the same mnemonic mines
with the dishonest model. The hold must cover the dishonest process's model load
(else the hotkey falls under `1/q` per hold and is audited at 100 %, the slow
path, not the draw) and 30 switched steps make "never drawn" about 0.1 %.
`summary.json` lists every submission (audited, draw, passed, reason, paid), each
hotkey's `miners.json` fields with its `effective_state`, `partial_audit.detection`
(`caught_by`: `draw`, `slow_hotkey`, `probation` or `no_beacon`) and explicit
checks. The validator needs network to drand and `bittensor_drand` installed, or
every draw audits and `sampling_exercised` fails. 2026-09-25, test H100, Teutonic
(`teutonic-i-graft-sft-cot-v2@d5256c5`) against Qwen3.5-4B, DAPO, 512 tokens,
n = 4: every check passed. The honest miner had 20 of 20 paid, 6 of them passed
unaudited after their hold. The late cheater had 5 of 5 pre-switch paid; its
first switched submission was drawn and failed (`exp_mismatch` 128), the 5
records it still had in hold were audited backwards and failed, the ban (3
confirmed failures) voided 1 more and refused 23, and 0 of its 7 switched
records were paid. Audit on arrival: 1.1k completion tokens/s including
re-audits, and 1.3k for passes alone (one record of 2048 tokens at a time, with a
vLLM miner on the same card, including store round trips).

The rehearsal runs against MinIO, not R2. The job, record and settlement stores
rely on conditional puts (`If-None-Match: *` to create, `If-Match: <etag>` to
replace), and MinIO passing says nothing about R2. Before launch, spot-check both
on the production bucket with a throwaway key: a create over an existing key and
a write against a stale ETag must each be refused (412 / `PreconditionFailed`),
and a write against the current ETag must succeed. Do not launch if either
refusal is missing: two validators, or a retry, could then double-consume a slot
or pay a verdict twice. The ledger's seen segments (§3.1) are created the same
way (`If-None-Match: *` on `reliquary/corpus/jobs/<job>/seen/<sha256>.json`),
so the create refusal also guards them; check the `If-Match` swap on an object
of about 0.5 MB, the size a busy ledger reaches.

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

- For a chat model, pass `--renderer-id chat-template-thinking-v1` (or
  `chat-template-v1` without thinking): each row the contract renders is
  wrapped as one user turn of the checkpoint's own chat template, pinned by
  `--model-revision`. Without it the model receives the raw row, which is only
  right for a base model. `--eos-token-id` is then the template's turn end
  (e.g. `<|im_end|>`).
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

### 2.1 Partial audit parameters (`audit_*`)

V0's rule is audit everything, and `audit_q = 1.0` (the default) still gives
exactly that. The partial-audit design (spec
`2026-09-25-corpus-partial-audit-design.md`) can instead sample a fraction of
one hotkey's submissions once it has proven itself, at the cost of a short
hold before an unaudited one is paid. `jobs create` accepts every `audit_*`
flag below at declaration (`--audit-q`, `--audit-probation-submissions`,
`--audit-hold-seconds`, `--audit-suspect-seconds`,
`--audit-ban-after-failures`, `--audit-ban-window-seconds`,
`--audit-ban-seconds`); `tasks set-cap` accepts the same seven flags to
change them on a live task, each independently -- an omitted flag leaves that
one parameter as it was, only the ones passed change.

**A `tasks set-cap` audit change only takes effect on the corpus validator's
next restart.** The process reads `entry.params` into `AuditParams` once, at
startup (`build_corpus_audit_wiring`); it does not re-read the registry while
running. Changing `--audit-q` (or any other `audit_*` flag) on a live task is
safe to run at any time, but nothing about that task's audit behaviour
changes until the corpus validator is restarted against the updated entry.

| param | default | meaning |
|---|---|---|
| `audit_q` | 1.0 | fraction of a `sampled` hotkey's submissions the GPU audits; 1.0 audits everything (V0) |
| `audit_probation_submissions` | 100 | audited passes, no confirmed failure, before a hotkey leaves `probation` |
| `audit_hold_seconds` | 4320 (72 min, ≈ one epoch) | how long an unaudited submission waits before it is paid |
| `audit_suspect_seconds` | 86400 (24 h) | how long a hotkey with one confirmed failure is audited at 100 % |
| `audit_ban_after_failures` | 3 | confirmed failures inside the ban window that ban the hotkey |
| `audit_ban_window_seconds` | 604800 (7 d) | the window those failures are counted in |
| `audit_ban_seconds` | 604800 (7 d) | how long the ban lasts |

What each state means for a miner:

- **`probation`** — every accepted submission of this hotkey is audited on
  the GPU. Applies to a brand-new hotkey and to one just re-registered after a
  ban (probation is never skipped on re-registration); it lasts until
  `audit_probation_submissions` submissions have passed audit with no
  confirmed failure in between.
- **`sampled`** — only a fraction `audit_q` of submissions is drawn for audit.
  The draw is keyed on the drand round after the first one published once the
  submission was received (one round of margin against clock lag), so a
  miner cannot choose which of its own submissions gets checked; a hotkey submitting fewer than `1 / audit_q` per hold is still
  audited at 100 % (too small a sample to be worth drawing). A submission that
  is not drawn still waits out `audit_hold_seconds` (plus 420 s for the
  route's record write) before it is paid, unaudited (verdict `passed: true,
  audited: false`), and longer while a sibling of the same hotkey received
  inside its hold is not judged yet (see "A hold delays payment").
- **`suspect`** — one confirmed failure (a failure that survives a second,
  separate re-audit) puts the hotkey back to 100 % audit for
  `audit_suspect_seconds`, and every one of its records still in hold at that
  moment is audited backwards, so a cheater caught late is not paid for what
  it already sent before being caught.
- **`banned`** — `audit_ban_after_failures` confirmed failures inside
  `audit_ban_window_seconds` ban the hotkey for `audit_ban_seconds`: the route
  refuses its submissions outright (reject reason `miner_banned`), and every
  one of its records still without a verdict is voided, never paid. A ban ends
  into a fresh `probation` — not straight back to `sampled`.

**A hold delays payment, it never skips it.** A `sampled` submission that is
not drawn is payable only once `audit_hold_seconds` plus 420 s (the route's
worst-case record write) has passed since it was received, and every record
of that hotkey received inside its hold has been judged or is known undrawn:
a drawn one still queued is audited first, so a failure there catches it.
While any pending record cannot be read, **every** unaudited pass of the job
waits (its hotkey is unknown), and the corpus validator logs `N pending corpus
record(s) unreadable (e.g. <id>); every unaudited pass waits until they read or
get a verdict` each pass; audited records are still judged and paid. A record
that stays unreadable must be fixed in the store. Until then `reliquary jobs status <job>` shows it as accepted but
unsettled, the same as one still waiting on a batched audit pass. Resolving
the drand chain's genesis time and period is lazy: if it is not yet known
when a submission's draw is due, that submission is audited (fail safe)
instead of guessing a round, and the next one retries resolution (at most
once a minute) -- the validator does not need restarting once the chain
becomes reachable, sampling turns on by itself. This never applies at
`audit_q = 1.0`, which never needs a draw.

**Sampling needs the `quicknet` chain.** `DRAND_CHAIN` (env var, default
`quicknet`) selects which drand chain the corpus validator draws against.
`verify_beacon_signature` cross-checks every beacon against
`bittensor_drand`'s own fetch, which is hardcoded to `quicknet`; run this
validator against `DRAND_CHAIN=default` (or any other chain) and every
signature check fails, so every draw is treated as unavailable and every
`sampled` submission is audited -- correct (fail safe), but silently gives
`audit_q = 1.0`'s cost with none of its savings. Leave `DRAND_CHAIN` unset.

## 3. Start the corpus validator

One process on one H100 (the GRAIL validator's card is fine: nothing else of
the RL service runs in it). It loads the job's checkpoint in bf16, refuses to
start if the contract's model is not the job's checkpoint or the fingerprint
differs, serves `GET /corpus/job`, `GET /corpus/cursor/{hotkey}` and
`POST /corpus/submit`, audits accepted submissions per the job's `audit_*`
parameters (§2.1 — `audit_q = 1.0`, the default, audits every one, as in V0)
and settles every 60 s.

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

**There are no seats: the task is open to every miner registered on the
subnet.** Nothing in the code counts or lists hotkeys; a submission is admitted
on its signature, the hotkey's registration and the CPU checks, and every new
hotkey is fully audited through its probation. An unregistered hotkey is refused
`hotkey_not_registered` (it could never be paid) and its miner stops; the
validator reads the subnet's registrations every 10 minutes, and while it has
no fresh snapshot it answers 503, which miners retry. Audit
throughput only sets how fast payment follows: when the fleet generates faster
than the card audits, verdicts lag and payment is delayed, nothing is lost.
Watch the backlog and the queue lag (section 5); a second card is the answer
to a lasting lag, not turning miners away. For scale, the 2026-09-24 rehearsal
audited about 1.5k completion tokens/s one completion at a time against about
470 tokens/s generated per miner; batched and sampled audits (`audit_q` < 1)
raise that, by an amount not yet measured with several miners.

A validator-side audit failure (a lost GPU, an out-of-memory) is retried on the
next pending rescan (every 60 s); after 5 in a row the process exits non-zero
rather than run while paying nobody. A bucket outage answers miners 503, which
they retry.

### 3.1 Ledger schema v2: migration, verify, rollback

Since ledger schema v2 the per-submission `ledgers.json` holds slots, cursors
and only the last few hundred accepted digests (`seen_pending`). The rest of the
seen set is sealed, 1024 digests at a time, into immutable segments at
`reliquary/corpus/jobs/<job>/seen/<sha256>.json`, which the ledger names
(`seen_segments`). A segment is never rewritten; one that nothing names (a
ledger write lost after the seal) is harmless and never counts as seen.

**Migration is automatic.** A v2 validator (the corpus validator, and the main
validator's corpus mount) rewrites a v1 ledger at startup, before serving: a
create-only backup at `reliquary/corpus/jobs/<job>/ledgers.v1-backup.json` (the
first one is kept), then the seen set sealed into segments of up to 4096, then
the ledger swapped under its ETag. Expect seconds to tens of seconds for about
200k digests; startup then loads every segment before serving. If either
fails, or takes over 120 s, the validator still starts (logs `could not be
prepared at startup` or `startup preparation timed out`) and the route migrates
on its first write instead. That inline migration writes **no** v1 backup.
Only the first backup is ever kept, so after a downgrade and a re-migration it
predates the later v1 state.

**Rollout.** Pin the image for the rollout so Watchtower cannot bounce between
versions. An older binary refuses a v2 ledger loudly (500
`corpus_ledger_corrupt` on submit and cursor, before any write): the corpus
route is down on it, nothing is corrupted.

**Verify** after the first start on v2, and whenever the ledger is in doubt:

```bash
reliquary corpus ledgers verify --job <job>
```

It loads and checks every segment the ledger names (present, hashes to its
name, count as named, no digest in two places) and prints the sizes as JSON.
Expected: `"schema": "reliquary/corpus-ledgers/v2"`, `"problems": []`,
`"seen"` equal to `"expected_seen"` (filled slots x `n`), `"backup": true` and
`"ledger_bytes"` well under 1 MB. It exits 1 on any violation; a missing or
altered segment is also what makes the route answer `corpus_ledger_corrupt`.
Run `migrate` by hand only for a rehearsal or on a stopped fleet:

```bash
reliquary corpus ledgers migrate --job <job>    # <job>: migrated | v2 | absent
```

**Rollback to a pre-v2 image:**

1. Stop every validator serving the job (a running v2 one would write v2 again
   on its next accept).
2. `reliquary corpus ledgers downgrade --job <job>`: it checks every segment,
   then writes v1 (`seen` = pending plus every segment, sorted) under the v2
   ETag. The segments stay in the bucket; a later v2 start reuses them only
   if the seen set is unchanged (chunks are cut by sorted position, so one new
   digest renames every chunk after it). Old segments are then orphans:
   harmless, never counted.
3. `reliquary corpus ledgers verify --job <job>` must show the v1 schema, the
   same `seen` and no problems.
4. Deploy the old image, pinned.

A code rollback without step 2 is an outage of the corpus route, not a loss;
running step 2 at any later point restores service.

### 3.2 Several jobs on one validator (one card, one model load)

A job has one prompt source. To generate, say, maths and code from the same
teacher at once, declare two jobs and serve both from ONE validator process:
the model is loaded once and both jobs' audits share the card, one forward pass
at a time (a FIFO lock: a job with a backlog waits at most one pass before the
other job's records are audited).

1. **Declare two jobs on the same checkpoint** (§2), each under its own task:
   same `--model`, `--model-revision`, `--checkpoint-sha256` and
   `--model-architecture`; distinct `--job-id` and `--task-id`; one
   `--prompt-source` each. `--from-profile` may differ: take for each job a
   template that declares its source (e.g. `qwen3-4b-base-dapo-reliquary-v1`
   for `openmathinstruct`, `teutonic-9b-reliquary-suite-v9-dev1` for
   `reliquary_code_v1`). Each task keeps its own cap, and all caps together
   (the RL task's included) must still sum to at most 1.0 (§1.3):

   ```bash
   reliquary jobs create --job-id <math-job> --task-id corpus-math \
     --prompt-source reliquary_dapo_math_v1 --cap 0.05 ...   # rest as in §2
   reliquary jobs create --job-id <code-job> --task-id corpus-code \
     --prompt-source reliquary_code_v1 --cap 0.05 ...
   reliquary tasks list   # e.g. default 0.9 + corpus-math 0.05 + corpus-code 0.05 <= 1.0
   ```

2. **One merged contract.** Each task carries its template narrowed to its own
   source; the process runs their merge, which `tasks contract` with several
   ids prints (the order of the ids does not matter):

   ```bash
   reliquary tasks contract --task-id corpus-math --task-id corpus-code > corpus-math-code.contract.json
   ```

   The rule follows what the corpus path reads. From the process contract it
   reads only the model, the toploc proof and, per environment, that
   environment's definition (its prompt template renders the source's rows).
   Sampling, lengths, the EOS and the renderer come from each job's manifest;
   `prompt_encoding` is never read (the job's `renderer_id` decides the
   encoding). So the merge refuses tasks whose `model_id`, `model_revision`,
   `model_architecture` or `proofs` differ, and one environment declared two
   ways. The environments are unioned; the other fields (`profile_id` becomes
   `<task>+<task>`, `protocol_version`, `prompt_encoding`, `sampling`, ...)
   come from the first task id in sorted order, except that a source whose
   rows depend on the protocol version (`openmathinstruct`: train shards only
   from v4) must get the same rows under the merge as under its own task.

3. **Start one validator with both ids:**

   ```bash
   export RELIQUARY_TASK_ID=corpus-math,corpus-code
   export RELIQUARY_TASK_CONTRACT=$PWD/corpus-math-code.contract.json
   reliquary validate --wallet-name <wallet> --hotkey <hotkey> \
     --http-host 0.0.0.0 --http-port <port> --no-set-weights
   ```

   It refuses to start (exit 4, before any download) if an id is not an active
   `corpus-generation` task (never mix in an RL task), if the contract it runs
   is not the merge of the tasks' contracts, if a job's source is declared
   differently there than in its own task, if the jobs name different
   checkpoints (repo, revision or sha256), if their toploc proofs differ, or if
   two tasks name the same job. `RELIQUARY_TASK_ID` is also the list of archive
   prefixes the settler may write under.

What stays per job: the prompt source and renderer, the records and verdicts
under `reliquary/corpus/jobs/<job>/`, the `audit_*` parameters of that task's
entry, `miners.json` (a hotkey banned on one job is not banned on the other),
and settlement: each task's archives under its own prefix, paid from its own
cap. Shared: the loaded model, the card, and the subnet-registration snapshot.

HTTP: `POST /corpus/submit` is unchanged and routes on the submission's
`job_id` (a job this validator does not serve is refused `job_not_served`, its
detail listing the served ids). `GET /corpus/jobs` lists the served jobs;
`GET /corpus/jobs/<job>/job` and `GET /corpus/jobs/<job>/cursor/<hotkey>`
answer for one of them (404 `corpus_job_not_served` otherwise). With several
jobs, the legacy `GET /corpus/job` and `GET /corpus/cursor/<hotkey>` answer
409 `{"detail": "several_jobs_served", "jobs": [...]}`; with one job they are
unchanged (and the job-scoped routes work too).

Rehearse it first (§0) with `--second-prompt-source <source>`: one validator
serves both jobs, one honest miner per job, and one hotkey mines the first job
with the dishonest model then the second honestly. Pass: each archive pays
only its own miners and sums to its own cap (`--second-cap`, default `--cap`),
and that hotkey is failed and banned on the first job only.

## 4. Miners

On each miner:

```bash
reliquary corpus mine --validator-url http://<validator-ip>:<port> \
  --wallet-name <wallet> --hotkey <hotkey>
```

- The miner needs the task's contract (its toploc parameters and prompt
  templates). Without `RELIQUARY_TASK_CONTRACT` it takes the one the validator
  serves at `GET /corpus/contract`, checks that it names the job's checkpoint
  and carries a toploc proof, saves it under `~/.cache/reliquary/corpus/` and
  restarts itself with it. Setting `RELIQUARY_TASK_CONTRACT` to a file still
  wins.

- Needs vLLM 0.30 with the decode capture hook; the generator sets
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` and `VLLM_USE_V2_MODEL_RUNNER=0` itself.
- `--gpu-memory-utilization 0.5` when the card is shared (e.g. with a
  validator); omitted, vLLM keeps its own default.
- `--max-steps N` stops after N submissions; 0 runs until `job_complete`.
- The miner downloads the job's checkpoint and refuses to start if its
  fingerprint differs from the manifest.
- `--job-id <job>` against a validator serving several jobs (§3.2): the miner
  then reads that job's `/corpus/jobs/<job>/...` routes. Without it, such a
  validator's 409 stops the miner with the list of jobs to choose from. One
  process mines one job: a miner who wants both runs two processes (two cards,
  or `--gpu-memory-utilization` to share one), with the merged contract file
  or its job's task contract.

## 5. Watch

- **Audit backlog**: submissions vs verdicts. They must grow at the same rate;
  a growing gap only delays payment (nothing is lost), but it means the card
  cannot keep up with the fleet.

  ```bash
  reliquary jobs status <job>
  # <job>: submissions=N verdicts=M unaudited=N-M settled=S unsettled=M-S pending=none last_window=W
  # drained: no
  ```

  For scale: one completion per prefill audited about 1.5k completion
  tokens/s against about 470 tokens/s generated per miner (2026-09-24). This
  is a throughput figure, not an admission limit.
- **Archives**: one per RL window at most, under
  `reliquary/tasks/corpus-<name>/dataset/window-<N>.json.gz`, where `<N>` is
  the RL task's latest index, never above it.
- **Settlement state**: `reliquary/corpus/jobs/<job>/settlement.json`
  (`last_window`, `pending` must return to null).
- **RL untouched**: the RL task's weights are unchanged except for the lowered
  cap.
- **Refusals**: validator logs `corpus submission refused` reasons; a
  dishonest checkpoint shows up as verdicts failing `exp_mismatch`.
- **Queue lag**: every rescan (once a minute) logs `corpus audit queue lag: N
  pending, oldest received S s ago`. An undrawn record waits one hold by
  design; a warning past the hold means the card is not keeping up.
- **Many hotkeys failing at once** is a validator-side systematic failure, not
  a fleet of cheaters: wrong card or kernel band, wrong checkpoint. Suspect and
  bans apply at every `audit_q`, 1.0 included, so honest miners get banned.
  Stop the corpus validator, fix the cause, then reset the hotkeys it caught:

  ```bash
  reliquary jobs miner-reset --job-id <job> --hotkey <hotkey> [--hotkey ...]
  reliquary jobs miner-reset --job-id <job> --all   # every hotkey in miners.json
  ```

  It clears `suspect_until`, `banned_until` and the confirmed failures in
  `miners.json` (compare-and-swap, safe while the route runs). Verdicts are
  write-once: records already failed, or voided `banned` while the ban held,
  stay unpaid; the reset only stops further loss. A hotkey whose count a
  failure reset to 0 goes through probation again (audited in full, paid
  normally). Then restart the corpus validator.

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
