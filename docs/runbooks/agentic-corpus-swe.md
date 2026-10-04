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
  --episode-file episode.json --fleet-knows-corpus-generation \
  --settlement period-ema-v1 --fleet-knows-period-settlement
reliquary tasks contract --task-id swe-agentic-v1 > swe-agentic-v1.contract.json
```

- **Settlement.** `--settlement period-ema-v1` is the default and is paid by
  the task's own 72-minute drand periods; `jobs create` refuses it (exit 1,
  `error: a period-ema-v1 job needs every validator to know it`) without
  `--fleet-knows-period-settlement`, which you pass only once the corpus
  validator serving the job and every weight setter run a binary that settles
  and replays period-ema-v1. Otherwise pass `--settlement windows` (paid by
  the RL window index). Either way an episode is paid only once graded, and a
  period closes only when everything received in it is graded (ruling P21).
  A submission whose regrades (after executor quarantines) pass
  `MAX_REGRADE_GENERATIONS` resolves `regrade_exhausted`: voided unpaid, an
  error in the control's log, and it stops holding its period (ruling P22).

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
route `/corpus/internal/grade/` to the control too. The 8 MiB counts the
body's bytes as sent, not characters: httpx 0.28 (the miner's client) sends
UTF-8, up to 4 bytes per non-ASCII character, and an older httpx escapes each
as `\uXXXX` (6 bytes, 12 outside the BMP). A trajectory whose `final_diff`
(up to 1 Mi chars) and rendered prompt are mostly non-ASCII can therefore
pass 8 MiB and be refused `413`. Rare on SWE-smith (English statements,
ASCII code); a job on non-ASCII repositories needs that headroom checked
before launch.

Not supported: the split validator (`RELIQUARY_CORPUS_SPLIT`) with any
episode job (it refuses to start); `RELIQUARY_CORPUS_INTAKE_ONLY` outside the
end-to-end run (nothing is audited or paid); one validator grading two env
pins. With `RELIQUARY_CORPUS_HOT_JOBS=1` an episode job joins a running
validator only if that validator was started with an episode job of the same
env pin (it holds the grade dispatcher); otherwise restart it.

Tunables (environment of the control, bounded): `RELIQUARY_CORPUS_GRADE_DISPUTE_SECONDS`
(1800; 60 to 86400) how long an item holding one vote waits, while no live
distinct-provider executor exists, before it resolves `disputed`;
`RELIQUARY_CORPUS_GRADE_LEASE_SECONDS` (2400) and
`RELIQUARY_CORPUS_REPLAY_LEASE_SECONDS` (12000; 11000 to 28800) the lease lives (a
replay: setup deadline + trajectory budget, see section 5, plus a margin).

## 3. Grade executors (CPU, Docker, about 16 vCPU and 150 GB disk each)

Every accepted trajectory is graded (its final diff applied in a fresh box
from the task's pinned image, tests run), then replayed (its actions re-run
in another fresh box, the diff and the observations compared) when it graded
successful or was drawn. Hostile code runs only here, never on the control.

On a box with:

- Docker, usable by the executor's user, with cgroup v2 limits working: each
  box is capped right after it starts with `docker update --pids-limit
  --memory --memory-swap` (swap off); a failed update refuses the box (an
  executor error, re-leased elsewhere).
- **A disk limit per box** (ruling P24). A replayed `dd if=/dev/zero of=/x`
  must fill its own box, not the host. verifiers' `docker run` takes no
  `--storage-opt`, so the limit is the daemon's default `overlay2.size`,
  which only the overlay2 graph driver (not the containerd image store)
  honours, and only on xfs mounted with project quotas. Mount Docker's data
  root on xfs with `pquota`, then configure the daemon:

  ```bash
  # a dedicated xfs volume for Docker (here /dev/sdb), with project quotas
  mkfs.xfs /dev/sdb
  echo '/dev/sdb /var/lib/docker xfs defaults,pquota 0 0' >> /etc/fstab
  systemctl stop docker containerd && mount /var/lib/docker
  mount | grep /var/lib/docker                       # must show prjquota
  cat > /etc/docker/daemon.json <<'EOF'
  {"storage-driver": "overlay2", "storage-opts": ["overlay2.size=10G"],
   "features": {"containerd-snapshotter": false}}
  EOF
  systemctl start containerd docker
  docker info --format '{{.Driver}} {{json .DriverStatus}}'   # overlay2 [["Backing Filesystem","xfs"],...]
  docker run --rm --entrypoint df alpine:3.22 -Pk /           # 1024-blocks column: 10485760
  ```

  (`pquota` cannot be added by `remount` to a mounted root filesystem; for
  `/` itself it goes in the kernel command line as `rootflags=pquota`. A
  dedicated volume is simpler.) Then pull the images again: the overlay2
  driver does not see the containerd store's images. `corpus grade-executor
  --disk-gb 10` (the default) refuses to start unless the driver is overlay2
  and a probe box (`--disk-probe-image`, default `alpine:3.22`) reads `/` at
  most 10 GiB, and every box is checked again (`df -Pk /`) before its first
  action. Keep `--disk-gb` equal to the daemon's `overlay2.size`. A box
  with any mount (an image that declares a `VOLUME`, which lives outside the
  quota) is refused at start as an executor error; the 20 pinned SWE-smith
  images declare none (checked 2026-10-04). 10 GiB is
  ample: the measured writable layer of an honest SWE-smith grade or replay
  is at most about 250 MB (measured 2026-10-04 on the four pulled SWE-smith
  images: gold grade 4-52 MB, replay with the harness footprint 136-246 MB). A trajectory that fills its box loses it (`box_lost`, see
  below); the host's free space is untouched. Size the host's Docker volume
  for the pulled images plus `--concurrency x --disk-gb`.
- **Output read back is bounded** (ruling P24): each replayed observation is
  read up to one character past the longest a lease carries (4 MiB), and
  anything longer is cut there (so it mismatches its recorded observation);
  finalize's output stops past 4 MiB + 1 MiB, a grade's test output past 256
  MiB. The executor's memory stays bounded by `--concurrency`.
- the pinned images pulled (`docker pull` of every digest in
  `reliquary_swe/swesmith_digests.json`, as many as the job's `num_images`);
- reliquary + reliquary-swe + verifiers at the pins installed (`corpus
  grade-executor` refuses at start otherwise: `RuntimeError: reliquary-swe is
  installed at ...`);
- **Docker storage on xfs** (ruling P20). `find` and `grep -r` list a
  directory in the order its filesystem returns. xfs keeps the image layer's
  insertion order, the same on every xfs host. ext4 hashes names with a seed
  of its own, so every ext4 host lists them in a different order. A replay on
  ext4 disagrees with an xfs miner on every `find ... | head`, and no
  normalization can fix a cut list. Check:

  ```bash
  docker info --format '{{.Driver}} {{.DockerRootDir}} {{json .DriverStatus}}'
  stat -f -c %T "$(docker info --format '{{.DockerRootDir}}')"   # must print xfs
  # containerd image store (DriverStatus shows io.containerd.snapshotter.v1):
  stat -f -c %T /var/lib/containerd                               # must print xfs too
  ```

  On ext4, `stat` prints `ext2/ext3`. `corpus grade-executor` refuses to start
  unless all of these are xfs. It exits with code 1 and prints
  `error: Docker stores images on ext2/ext3 (/var/lib/docker), ..., not xfs`.
  To fix it, give Docker an xfs data root (`data-root` in
  `/etc/docker/daemon.json`, on an xfs volume, mounted with `pquota` as
  above), then re-pull the images.
  `--allow-non-xfs` skips this check and the disk-limit check. **It is for
  tests only.**
- **PyPI reachable from the box during setup.** Before the network cut, the
  replay box prepares the same thing the miner's harness prepares:
  `pip install --user uv`, then `uv sync` of the bash harness program. That
  way `/root/.local`, `/root/.cache/pip` and `uv` in `pip list` match the
  miner's observations. If this step fails (PyPI or astral unreachable), the
  item fails as an executor `error`. The control re-leases it to another
  executor, and the miner is never charged for it. The uv version is
  whatever is latest at that moment (verifiers does not pin it). The
  comparison hides only the version on `pip list`'s `uv` row.

register it once, from a machine with bucket credentials, then run it:

```bash
reliquary corpus register-grade-executor --executor-id grade-01 --env-version <ENV_COMMIT> \
  --provider-id hetzner                       # prints {"token": ...} once; created=false prints none
RELIQUARY_EXECUTOR_TOKEN=<token> reliquary corpus grade-executor --control-url https://<control> \
  --executor-id grade-01 --concurrency 4 --cpus 2 --memory-gb 6 --pids-limit 1024 --disk-gb 10
```

- **Providers.** `--provider-id` is required and names who runs the box
  (normalized to lower case). A replay failure sanctions a miner only when
  executors of **two distinct providers** agree (ruling P17); a disagreement
  goes to a third distinct provider, and the executor that disagreed with the
  majority is quarantined. With fewer distinct providers connected, a failing
  replay waits `RELIQUARY_CORPUS_GRADE_DISPUTE_SECONDS` (30 min) and resolves
  `disputed`: nobody is sanctioned and nothing is certified (ruling P16). Run
  executors on at least two providers, three for arbitration.
- **Start after the control answers, under a restart policy.** The first
  heartbeat is not retried: an executor whose control is unreachable or has
  not yet re-read the registry (every 30 s) exits at once (later errors are
  retried, 401 always exits). Run it under `Restart=on-failure` with the
  token in its unit's environment.
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
- **Strongly advised: put Docker's storage on xfs**, as executors must (see
  section 3 for how to check). Executors replay on xfs, and the replay
  compares your recorded observations against theirs. On ext4 your boxes list
  directories in an order no executor reproduces. Agents open most episodes
  with `find /testbed ... | head -50`, which then mismatches, along with
  every cut `grep -r ... | head`. Each mismatch spends part of the episode's
  replay tolerance (5 observations, or 12 % on long episodes). In the B1
  measurement, an honest episode on ext4 used all of it and was voided.
- Your miner's log line `episode N of <hotkey>: accepted (reward R)` carries
  the graded reward. verifiers' own `rollout done: ... reward=0.000` line is
  printed before grading and always reads 0.

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

Outcomes a trajectory causes in its box (ruling P23): once its recorded
actions (replay) or its applied patch (grade) run, a box that dies, cannot
take the next command, or runs past its deadline (replay: its trajectory budget, grade the
task's scoring timeout) is reported `box_lost` / `box_timeout`, a vote like
any fact. Two distinct providers agreeing void the submission **unpaid**
with reason `replay_unjudgeable` (`stage` `grade` or `replay`) and **no**
escalation of the miner. An executor outvoted with a `box_lost`/`box_timeout`
vote is struck each time (never reset) and quarantined at the third.
Every grade that is not a clean success (failing,
timeout, error, disputed, unjudgeable) gets the failing replay draw
(`replay_fraction_failed`). A failure before any recorded action or before
the patch (provisioning, setup, PyPI, the Docker daemon) stays the
executor's `error`/`timeout` and is re-leased.

Replay timing (ruling P25), from the task's own timeouts: provisioning and
setup get the setup timeout plus 600 s (SWE-smith: 1500 s; missing it is the
executor's `timeout`); the trajectory's clock starts at its first action with
twice the miner's agent + finalize budget (SWE-smith: 2 x (3600 + 900) =
9000 s), so only a trajectory far past any honest episode is `box_timeout`.
Capacity: a forger whose last action sleeps forever holds an executor slot
up to about 2.5 h (setup + budget) per replay it gets; it holds at most as
many replays as it has accepted slots, and each costs it its own slot. A trajectory whose text a
lease cannot carry (4096 actions, 4 MiB per observation or argument, 16 MiB
in all) is refused at intake as `trajectory_too_large`; the miner checks the
same bound before signing.

The dispute clock (`RELIQUARY_CORPUS_GRADE_DISPUTE_SECONDS`) counts only the
time during which no live executor of another provider could take the next
vote, and an item holding a vote is leased before any new item: a grading
backlog never resolves a failing replay as `disputed`.

In the bucket, `grades/`, `regrades/` and `voided/` sit beside `submissions/`
and `verdicts/` in the job's prefix; a `voided/{sid}.json` with reason
`replay_failed` is a confirmed replay failure (it carries `graded_by` and
`providers`), one with `replay_unjudgeable` an unpaid trajectory no box
could judge. `reliquary jobs status swe-agentic-v1` prints how far the job is
from drained. In the control's log: `grade executor <id> quarantined: ...`,
`voided, replay failed`, `voided unpaid, its <stage> could not be judged on two
providers' boxes`,
`disputed: no distinct-provider executor was available for ... s`, `waits: every live grade executor is excluded`, and `grade items wait
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

Before the run (GPU box): stop the serving vLLM by its process group
(`kill -TERM -- -<pgid of "vllm serve">`, from `ps -eo pid,pgid,args`; never
`pkill -f`). Its reliquary-swe is an editable install of `/opt/env-pinned`,
which other work keeps at `main`: `git -C /opt/env-pinned checkout --detach
<ENV_COMMIT>` (the commit is in that clone; no remote), and check the pin line
above prints `<ENV_COMMIT>`; check it out back afterwards. The GPU box's
24 GB free do not hold the 4 images: the default prompts (`--prompt-start` 0,
8 or 16, 8 prompts) are all python-docx instances in the 4-image set, so only
that image (5 GB) is needed there (`docker pull` its pinned digest); the
Hugging Face cache of `Qwen/Qwen3-4B-Instruct-2507` under `/opt/hf` (8 GB,
unused) may go. The full control loads the 27B with `GRAIL_ATTN_IMPL`
(flash_attention_2 by default); that venv has no flash_attn, so the script
falls back to sdpa (gate M1's setting) and says so.

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
     export R2_BUCKET_ID=reliquary-agentic-e2e R2_ENDPOINT_URL=http://127.0.0.1:9000 R2_REGION=us-east-1 \
            HF_HOME=/opt/hf; cd /opt/reliquary || exit 1; $1"
}
E2E="/opt/vllm/venv/bin/python scripts/agentic_corpus_e2e.py"
CLI="/opt/vllm/venv/bin/python -m reliquary.cli.main"
S=/opt/agentic-e2e
```

Two details of `gpu` matter. The endpoint and region point the plain CLI
(`register-grade-executor`, `jobs export`) at the MinIO; without them it
builds `https://.r2.cloudflarestorage.com` and fails (`Invalid endpoint`); the
script's subcommands read them from the state directory anyway. And
`cd ... || exit 1; $1`, never `cd ... && $1`: with `&&`, a `$1` ending in
`&` backgrounds the whole list in a subshell that keeps ssh's output open, so
the call never returns and `$!` is that subshell's pid, not the process
group `kill -- -$(cat ...pid)` needs later. The same holds for every
`ssh '... setsid nohup ... &'` below.

1. Prepare (MinIO, the job; the fingerprint takes minutes over 52 GB):
   `gpu "$E2E prepare --state $S --env-commit $ENV_COMMIT --start-minio"`.
   Expect one JSON line with `"prompts": [0, 8]`. MinIO runs as container
   `agentic-e2e-minio` on 127.0.0.1:9000 with the shell's credentials (image
   `cgr.dev/chainguard/minio` by digest: MinIO no longer publishes its own); a
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
   # The control re-reads the executor registry every 30 s: an executor that
   # starts before it does gets 401 on its first heartbeat and exits, and its
   # token, shown once, is gone with the shell variable. Any failure of that
   # first heartbeat is fatal (the control not answering yet included): run
   # this only once step 2's curl answers (`curl -sf`, and test curl's status,
   # not a pipe's). A lost token means a new --executor-id.
   sleep 35
   ssh -f -N -o ExitOnForwardFailure=yes -L 18100:127.0.0.1:8100 -p 20300 root@162.243.212.30
   ssh -f -N -o ExitOnForwardFailure=yes -R 18100:127.0.0.1:18100 root@5.161.244.56
   [ -n "$TOKEN_A" ] && printf '%s\n' "$TOKEN_A" | ssh root@5.161.244.56 'read -r RELIQUARY_EXECUTOR_TOKEN; export RELIQUARY_EXECUTOR_TOKEN; cd /opt/reliquary || exit 1; setsid nohup .venv/bin/python -m reliquary.cli.main corpus grade-executor --control-url http://127.0.0.1:18100 --executor-id grade-a --concurrency 4 --allow-non-xfs > /root/grade-a.log 2>&1 < /dev/null & echo $! > /root/grade-a.pid'
   [ -n "$TOKEN_B" ] && printf '%s\n' "$TOKEN_B" | ssh -p 20300 root@162.243.212.30 'read -r RELIQUARY_EXECUTOR_TOKEN; export RELIQUARY_EXECUTOR_TOKEN; cd /opt/reliquary || exit 1; setsid nohup /opt/vllm/venv/bin/python -m reliquary.cli.main corpus grade-executor --control-url http://127.0.0.1:8100 --executor-id grade-b --concurrency 2 > /opt/agentic-e2e/grade-b.log 2>&1 < /dev/null & echo $! > /opt/agentic-e2e/grade-b.pid'
   unset TOKEN_A TOKEN_B
   ```

   sandbox-dev-01 stores Docker images on ext4, so `grade-a` runs with the test-only
   `--allow-non-xfs` (ruling P20). Its replays then mismatch the miner's xfs
   directory order on cut `find`/`grep -r` listings. grade-b, on the GPU box's xfs,
   does not.

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
   (`kill $(pgrep -f 'ExitOnForwardFailure=yes -[LR] 18100')`); put the GPU
   box's reliquary-swe checkout back where it was (see "Before the run"); restart
   the GPU box's serving vLLM as it ran before (on 2026-10-03):

   ```bash
   ssh -p 20300 root@162.243.212.30 'cd /opt/vllm || exit 1; setsid nohup /opt/vllm/serve.sh Qwen/Qwen3.8-27B 65536 --tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-num-seqs 256 --limit-mm-per-prompt "{\"image\":0,\"video\":0}" > /opt/vllm/qwen38-27b.log 2>&1 < /dev/null & echo $! > /opt/vllm/serve.pid' </dev/null
   ```
