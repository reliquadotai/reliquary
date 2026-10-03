# Agentic corpus job on SWE-smith (Qwen3.8-27B): operator runbook

Spec: `docs/design/2026-10-03-agentic-corpus-swe-design.md`. Plan: `docs/superpowers/plans/2026-10-04-agentic-corpus-v1.md`.

An agentic job pays miners for multi-turn SWE trajectories (bash + edit
harness, verifiers, reliquary-swe tasks in Docker). The corpus validator
(the "control") audits each turn's TOPLOC proofs on its GPU, has every
accepted trajectory graded and replayed by grade executors (CPU boxes with
Docker), and exports the certified rows. Every command below was checked
against the CLI's `--help` at the commit of this runbook; `reliquary` is
the installed entry point (`python -m reliquary.cli.main` is the same).

Placeholders: `<ENV_COMMIT>` (the reliquary-environments commit the job
pins, 40 hex), `<control>` (the control's public URL), `<w>`/`<h>` (wallet
and hotkey names), `<CAP>`, `<Q>`.

## Before merging: `main` deploys

**Merging this work into `main` is a deployment, not a code review step.**
CI publishes `:latest` on every merge to `main`, Watchtower rolls it out to
the weight-only validators within about 5 minutes, and the trainer runs
`:latest` unpinned. Before merging, either coordinate the release with every
operator running those images (the rollout happens whether or not anyone is
watching), or pin the production images to their current digest first and
roll forward deliberately. The same holds for every later fix to this job's
code.

## Pins

- reliquary-environments commit `<ENV_COMMIT>` (SWE-smith images pinned by digest; empty problem statements dropped).
- verifiers `b2e4e8157783b2c0dffc7821044c87f29f1c3ccf`, renderers `0.1.11`, renderer id `renderers:qwen38@0.1.11`, harness `bash`.
- Every process of the job (control, miners, grade executors) runs these exact pins; each refuses to start otherwise:
  the miner checks the renderer id, renderers, reliquary-swe and verifiers
  (`episode_support_refusal`) and the checkpoint fingerprint; the grade
  executor checks reliquary-swe and verifiers against its registration; the
  control checks the renderer id, renderers and reliquary-swe, and the
  checkpoint fingerprint.
- Check a host's install with:

```bash
python -c "from reliquary.environment.agentic_swe import installed_env_commit, installed_verifiers_commit; print(installed_env_commit(), installed_verifiers_commit())"
# -> <ENV_COMMIT> b2e4e8157783b2c0dffc7821044c87f29f1c3ccf
```

## 1. Declare the job (operator machine with bucket credentials and reliquary-swe at the pin)

`jobs create` builds the task set to count it, so this machine needs
reliquary-swe at `<ENV_COMMIT>` and access to the SWE-smith dataset on the
Hub; the tokenizer files of the checkpoint are enough for `<EOS>`.

```bash
SNAP=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.8-27B'))")
python -c "import json; print(json.load(open('$SNAP/config.json'))['architectures'][0])"          # <ARCH>
python -c "from reliquary.environment.agentic_swe import load_turn_renderer; print(load_turn_renderer('$SNAP').terminator_id)"  # <EOS>
python -c "from reliquary.environment.agentic_swe import load_swe_source; print(len(load_swe_source(20)))"  # <COUNT>
reliquary jobs fingerprint Qwen/Qwen3.8-27B --revision $(basename $SNAP)                           # <SHA>
cat > episode.json <<'EOF'
{"env": {"package": "reliquary-swe", "version": "<ENV_COMMIT>", "split": "train", "num_images": 20},
 "harness": "bash", "renderer": "renderers:qwen38@0.1.11",
 "verifiers": "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf",
 "max_turns": 40, "max_tokens_per_turn": 8192, "max_total_tokens": 60000,
 "replay_fraction_failed": 0.10}
EOF
reliquary jobs create --job-id swe-agentic-v1 --model Qwen/Qwen3.8-27B \
  --model-revision $(basename $SNAP) --model-architecture <ARCH> --checkpoint-sha256 <SHA> \
  --prompt-source reliquary_agentic_swe_v1 --prompt-count <COUNT> \
  --renderer-id renderers:qwen38@0.1.11 --eos-token-id <EOS> --slots-per-prompt 2 \
  --max-new-tokens 8192 --prompt-order free --cap <CAP> --audit-q <Q> \
  --episode-file episode.json --fleet-knows-corpus-generation
reliquary tasks contract --task-id swe-agentic-v1 > swe-agentic-v1.contract.json
```

- `<Q>` comes from spec §7 M3's table for the expected miner count and median
  length (one audit GPU: 1.0 at 10 miners and 20k tokens; lower beyond, or add
  audit GPUs). The other `--audit-*` flags keep their defaults (probation 100,
  ban after 3 confirmed failures); a confirmed replay failure counts like a
  confirmed TOPLOC failure.
- The manifest refuses, at `jobs create`: `--n` other than 1, fewer than 2
  slots per prompt, a `--prompt-order` other than `free`, `--max-new-tokens`
  different from `max_tokens_per_turn`, `--renderer-id` different from the
  episode's renderer, `max_total_tokens` above 60000 (gate M3: the most one
  80 GB audit GPU prefills), `max_turns` above 64, and `--grader-id`
  (grade executors decide, not a filter).
- `replay_fraction_failed` is the share of failing (not graded successful)
  trajectories replayed, drawn from drand; every graded success is replayed.

## 2. The control

The corpus validator serves the job as any other:

```bash
export RELIQUARY_TASK_ID=swe-agentic-v1
export RELIQUARY_TASK_CONTRACT=$PWD/swe-agentic-v1.contract.json
reliquary validate --wallet-name <w> --hotkey <h> --http-host 0.0.0.0 --http-port <port> --no-set-weights
```

For an episode job it also needs, on the same host:

- reliquary-swe at the pin (the task source), renderers 0.1.11 and verifiers
  at the pin (the renderer and the task set import), the checkpoint (it loads
  the 27B in bf16 to audit: one H100 80 GB) and network to the Hub and drand.
- **A restart policy.** Startup reads the grade executor registry from the
  bucket and exits on a transient read failure; run it under
  `restart: on-failure` (compose) or `Restart=on-failure` + `RestartSec=30`
  (systemd). A declaration refusal exits with code 4 and will loop the same
  way: a restart count that keeps climbing is a refusal, read the log.

It serves `/corpus/internal/grade/{claim,heartbeat,<lease>/result}` for grade
executors, authenticated by each executor's token. A trajectory submission is
up to about 2 MB of JSON and the route caps an episode job's body at 8 MiB: a
reverse proxy in front needs `client_max_body_size 8m` (nginx), and must
route `/corpus/internal/grade/` to the control too.

Not supported: the split validator (`RELIQUARY_CORPUS_SPLIT`) with any
episode job (it refuses to start); `RELIQUARY_CORPUS_INTAKE_ONLY` outside the
end-to-end run (nothing is audited or paid); one validator grading two env
pins. With `RELIQUARY_CORPUS_HOT_JOBS=1` an episode job joins a running
validator only if that validator was started with an episode job of the same
env pin (it holds the grade dispatcher); otherwise restart it.

Tunables (environment of the control, bounded): `RELIQUARY_CORPUS_GRADE_DISPUTE_SECONDS`
(1800; 60 to 86400) how long an item holding one vote waits for a next
distinct-provider executor before it resolves `disputed`;
`RELIQUARY_CORPUS_GRADE_LEASE_SECONDS` (2400) and
`RELIQUARY_CORPUS_REPLAY_LEASE_SECONDS` (4200) the lease lives.

## 3. Grade executors (CPU, Docker, about 16 vCPU and 150 GB disk each)

Every accepted trajectory is graded (its final diff applied in a fresh box
from the task's pinned image, tests run), then replayed (its actions re-run
in another fresh box, the diff and the observations compared) when it graded
successful or was drawn. Hostile code runs only here, never on the control.

On a box with:

- Docker, usable by the executor's user, with cgroup v2 limits working: each
  box is capped right after it starts with `docker update --pids-limit
  --memory --memory-swap` (swap off); a failed update refuses the box (an
  executor error, re-leased elsewhere). No daemon configuration is needed.
- the pinned images pulled (`docker pull` of every digest in
  `reliquary_swe/swesmith_digests.json`, as many as the job's `num_images`);
- reliquary + reliquary-swe + verifiers at the pins installed (`corpus
  grade-executor` refuses at start otherwise: `RuntimeError: reliquary-swe is
  installed at ...`);

register it once, from a machine with bucket credentials, then run it:

```bash
reliquary corpus register-grade-executor --executor-id grade-01 --env-version <ENV_COMMIT> \
  --provider-id hetzner                       # prints {"token": ...} once; created=false prints none
RELIQUARY_EXECUTOR_TOKEN=<token> reliquary corpus grade-executor --control-url https://<control> \
  --executor-id grade-01 --concurrency 4 --cpus 2 --memory-gb 6 --pids-limit 1024
```

- **Providers.** `--provider-id` is required and names who runs the box
  (normalized to lower case). A replay failure sanctions a miner only when
  executors of **two distinct providers** agree (ruling P17); a disagreement
  goes to a third distinct provider, and the executor that disagreed with the
  majority is quarantined. With fewer distinct providers connected, a failing
  replay waits `RELIQUARY_CORPUS_GRADE_DISPUTE_SECONDS` (30 min) and resolves
  `disputed`: nobody is sanctioned and nothing is certified (ruling P16). Run
  executors on at least two providers, three for arbitration.
- **One executor per Docker host.** At start an executor removes every
  container named `reliquary-gradebox-*` on its host (the boxes a killed
  executor left behind), so a second executor on the same host would kill the
  first one's boxes.
- **Size the host:** `--concurrency x --memory-gb` plus the executor itself
  must fit the host's memory, `--concurrency x --cpus` its CPUs. The defaults
  (4 x 6 GB, 4 x 2 CPUs) fit a 16 vCPU / 32 GB box.
- The token is the only secret, read from `RELIQUARY_EXECUTOR_TOKEN`; pass it
  through the environment (a unit's `EnvironmentFile` readable by root only,
  or a secret manager), never on a command line.
- Spec §9 sizes the fleet: 3 executors at 10 miners, 11 at 50, spread over at
  least two providers.

## 4. A miner (H100 80 GB, Docker, about 16 vCPU, 150 GB disk)

Install vLLM 0.30 and, in its venv, reliquary (`--no-deps`), reliquary-swe at
the pin, verifiers at the pin, renderers 0.1.11 and their dependencies under a
constraints file pinning torch, vllm and transformers (plan Task 14 Step 8
lists the exact commands). Pre-pull the pinned images. Then:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 \
  reliquary corpus mine-agentic --validator-url https://<control> --job-id swe-agentic-v1 \
  --wallet-name <w> --hotkey <h> --concurrency 8 --max-num-seqs 16
```

- Both variables are required: the per-turn proofs capture hidden states
  inside vLLM's in-process V1 runner. Prefix caching stays on.
- vLLM runs in the miner's process, behind an OpenAI-style generate endpoint
  bound to `127.0.0.1:--port` (8011 by default; no auth, never exposed); the
  port must be free.
- It refuses to start, with exit code 4 and `error: <reason>`, when an
  install is off its pin (e.g. `error: reliquary-swe is installed at ..., the
  job pins ...`) or when the downloaded checkpoint does not match the job's
  fingerprint (`error: the downloaded checkpoint does not match the job's
  fingerprint`). It stops with `refusing to mine: the installed verifiers'
  network notice differs ...` when verifiers' restricted-network notice is
  not the one the validator renders prompts with (every trajectory would be
  refused `prompt_mismatch`).
- `--concurrency` episodes at once (8 to 11 on one H100, spec §9);
  `--max-num-seqs` (16) bounds vLLM's batch: lower it if turns fail as
  preempted. `--episodes N` stops after N episodes (0: until the job
  completes).
- Before signing, the miner runs the validator's own trajectory parse and
  drops what it would refuse (`precheck_refused:<reason>` in the counts it
  prints at the end).

## 5. Payment and watching it

Payment is unchanged (per accepted slot, `cap x` the passed assistant-token
share), but **an episode submission is paid only once its grade is written**:
the settler waits for it so that a void lands before payment. A grading
backlog is therefore a payment backlog. Its signal is the `grading` block of
the job's status route:

```bash
curl -s https://<control>/corpus/jobs/swe-agentic-v1/status | python -m json.tool
```

- `graded`, `ungraded`, `grading` (in flight), `awaiting_draw` (failing
  grades waiting for their drand round), `regrading`.
- `dispatcher_waiting`: items no connected executor can take (none connected,
  or every live one excluded: it voted already, or shares a provider with a
  voter). Non-zero for long means add an executor, of another provider.
- `held_executors` non-empty: an executor was quarantined, and every grade
  written before this boot is held until the ones it decided alone are found
  (`unindexed` counts what is left to read) and graded again into
  `regrades/`. Nothing it decided alone is paid meanwhile.

In the bucket, `grades/`, `regrades/` and `voided/` sit beside `submissions/`
and `verdicts/` in the job's prefix; a `voided/{sid}.json` with reason
`replay_failed` is a confirmed replay failure (it carries `graded_by` and
`providers`). `reliquary jobs status swe-agentic-v1` prints how far the job is
from drained. In the control's log: `grade executor <id> quarantined: ...`,
`voided, replay failed`, `disputed: no distinct-provider executor voted within
... s`, `waits: every live grade executor is excluded`, and `grade items wait
and no grade executor is connected`.

## 6. Export

```bash
reliquary jobs export swe-agentic-v1 --out swe-agentic-v1.sft.jsonl --sft     # certified successes only
reliquary jobs export swe-agentic-v1 --out swe-agentic-v1.all.jsonl           # every replay-certified trajectory
```

- It refuses a job not yet drained (grades, regrades and voids may still
  change); `--allow-incomplete` exports what is final so far.
- It always writes `{out}.counts.json`: `drained`, `exported`, what was left
  out (`ungraded`, `held`, `voided`, `uncertified`, `unparseable`, ...),
  `exported_at` and the quarantined executors it held.
- The plain export is every replay-certified, non-voided, non-held
  trajectory, failed attempts included (`graded_success` says which);
  `--sft` keeps those with `graded_success` (ruling P18).
- **`tokens` + `assistant_mask` are the authoritative SFT form** (ruling P19).
  `messages` are the executed (parsed) calls, checked against the tokens up to
  whitespace only: the pinned renderer's parser strips whitespace around tool
  call argument values (an edit's indentation can be lost in `messages`, never
  in `tokens`). Train on the tokens.

## End-to-end run (test boxes)

`scripts/agentic_corpus_e2e.py` drives it (plan Task 20): 8 prompts of the
4-image set, 2 slots each, `replay_fraction_failed = 1.0`, an honest hotkey
(6 episodes) and two forgers (1 episode each: a forged `final_diff`, and
forged bash observations through `BASH_ENV`). `check` exits 0 when at least
one certified row is exported, both forgeries are voided `replay_failed` by
two executors of two providers with a confirmed failure on their hotkey, and
the honest hotkey is never voided or charged; 2 (inconclusive) when a forgery
did not exercise its path; 1 otherwise. Never on a production box.

Topology:

| box | runs |
|---|---|
| GPU box `ssh -p 20300 root@162.243.212.30` (DigitalOcean) | MinIO, the control (intake-only, then full), the miner, grade executor `grade-b` (provider `digitalocean`, control at `127.0.0.1:8100`) |
| sandbox-dev-01 `root@5.161.244.56` (Hetzner) | grade executor `grade-a` (provider `hetzner`, control at `127.0.0.1:18100` through two tunnels from the VPS) |

The GPU box runs `grade-b` beside the miner: different container names, so
the orphan sweep never touches the miner's episode boxes, and the same pulled
images. Keep it at `--concurrency 2` (20 vCPU shared with 8 episodes). Disk:
the 4 images need about 25 GB and the box had 24 GB free on 2026-10-03; free
space first. **Test-only fallback** if the GPU box cannot host `grade-b`: run
both executors on sandbox-dev-01 with distinct provider ids (`hetzner`,
`hetzner-e2e-b`). That fakes provider independence (ruling P17 exists to
forbid it in production) and breaks one-executor-per-host: start both before
`mine`, and never restart one while the other has boxes running.

Secrets: the throwaway MinIO's credentials live in the VPS shell only and
reach the box on stdin; executor tokens go from `register-grade-executor`'s
stdout to the executor's environment the same way. Nothing is written to a
file. From the VPS (`ENV_COMMIT` as in the plan's global constraints):

```bash
export R2_ACCESS_KEY_ID=e2e$(openssl rand -hex 6) R2_SECRET_ACCESS_KEY=$(openssl rand -hex 20)
gpu() {  # run "$1" on the GPU box with the bucket's credentials, from /opt/reliquary
  printf '%s\n%s\n' "$R2_ACCESS_KEY_ID" "$R2_SECRET_ACCESS_KEY" | ssh -p 20300 root@162.243.212.30 \
    "read -r R2_ACCESS_KEY_ID; read -r R2_SECRET_ACCESS_KEY; export R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY; \
     export R2_BUCKET_ID=reliquary-agentic-e2e HF_HOME=/opt/hf; cd /opt/reliquary && $1"
}
E2E="/opt/vllm/venv/bin/python scripts/agentic_corpus_e2e.py"
CLI="/opt/vllm/venv/bin/python -m reliquary.cli.main"
S=/opt/agentic-e2e
```

1. Prepare (MinIO, the job; the fingerprint takes minutes over 52 GB):
   `gpu "$E2E prepare --state $S --env-commit $ENV_COMMIT --start-minio"`.
   Expect one JSON line with `"prompts": [0, 8]`. MinIO runs as container
   `agentic-e2e-minio` on 127.0.0.1:9000 with the shell's credentials; a
   single-turn e2e's `corpus-e2e-minio` holding port 9000 makes it fail, never
   removed.
2. Control, intake-only, in the background:
   `gpu "setsid nohup $E2E validator --state $S --intake-only > $S/validator-intake.log 2>&1 < /dev/null & echo \$! > $S/validator.pid"`,
   then `ssh -p 20300 root@162.243.212.30 "curl -s http://127.0.0.1:8100/corpus/jobs/agentic-e2e/job | head -c 300" </dev/null`
   until it answers with the `episode` object.
3. Register the executors and start them (tokens never touch a file):

   ```bash
   # Fails (exit 1, message on stderr) when no token comes back: an id registered
   # before answers created=false and token null, and its token is not shown again.
   token() {
     local line; line=$(gpu "$CLI corpus register-grade-executor --executor-id $1 --env-version $ENV_COMMIT --provider-id $2" | tail -1)
     python3 -c 'import json,sys; t=json.loads(sys.argv[1]).get("token"); print(t) if t else sys.exit("no token for this executor id: " + sys.argv[1] + " (already registered: use a new --executor-id)")' "$line"
   }
   TOKEN_A=$(token grade-a hetzner) && TOKEN_B=$(token grade-b digitalocean) \
     || { unset TOKEN_A TOKEN_B; echo "STOP: an executor got no token" >&2; }
   ssh -f -N -o ExitOnForwardFailure=yes -L 18100:127.0.0.1:8100 -p 20300 root@162.243.212.30
   ssh -f -N -o ExitOnForwardFailure=yes -R 18100:127.0.0.1:18100 root@5.161.244.56
   [ -n "$TOKEN_A" ] && printf '%s\n' "$TOKEN_A" | ssh root@5.161.244.56 'read -r RELIQUARY_EXECUTOR_TOKEN; export RELIQUARY_EXECUTOR_TOKEN; cd /opt/reliquary && setsid nohup .venv/bin/python -m reliquary.cli.main corpus grade-executor --control-url http://127.0.0.1:18100 --executor-id grade-a --concurrency 4 > /root/grade-a.log 2>&1 < /dev/null & echo $! > /root/grade-a.pid'
   [ -n "$TOKEN_B" ] && printf '%s\n' "$TOKEN_B" | ssh -p 20300 root@162.243.212.30 'read -r RELIQUARY_EXECUTOR_TOKEN; export RELIQUARY_EXECUTOR_TOKEN; cd /opt/reliquary && setsid nohup /opt/vllm/venv/bin/python -m reliquary.cli.main corpus grade-executor --control-url http://127.0.0.1:8100 --executor-id grade-b --concurrency 2 > /opt/agentic-e2e/grade-b.log 2>&1 < /dev/null & echo $! > /opt/agentic-e2e/grade-b.pid'
   unset TOKEN_A TOKEN_B
   ```

   Both logs quiet after 30 s (no traceback, no `refused by the control`).
4. Mine (30 to 60 minutes; poll `$S/mine.log`):
   `gpu "VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 setsid nohup $E2E mine --state $S > $S/mine.log 2>&1 < /dev/null &"`.
   `$S/mine.json` at the end: `honest` with `accepted >= 1`, each forger
   `accepted == 1` (else rerun with a new `--state` and `--prompt-start 8`).
5. `gpu "$E2E wait-grades --state $S"`: the last line has `grades` equal to
   `submissions`; its `grading` block shows the backlog meanwhile.
6. Restart the control with the model (kill by process group, never `pkill -f`),
   then `gpu "$E2E wait-verdicts --state $S"`:
   `gpu "kill -- -\$(cat $S/validator.pid); sleep 10; setsid nohup $E2E validator --state $S > $S/validator-full.log 2>&1 < /dev/null & echo \$! > $S/validator.pid"`.
   `grade-b` reconnects by itself; `grade-a`'s tunnels survive the restart.
7. `gpu "$E2E check --state $S --out $S/summary.json; echo exit=\$?"` and,
   to exercise the real export, `gpu "$CLI jobs export agentic-e2e --out $S/export.sft.jsonl --sft --allow-incomplete"`
   (the run never settles, so `drained` is false in its counts file).
8. Clean up: `kill -- -$(cat <pidfile>)` for the control, `grade-b` and
   `grade-a`; `docker rm -f agentic-e2e-minio`; the two tunnels on the VPS
   (`kill $(pgrep -f 'ExitOnForwardFailure=yes -[LR] 18100')`); restart the
   GPU box's serving vLLM with the command recorded in `/opt/vllm/serve.sh` usage.
