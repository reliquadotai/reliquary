# Terminal-Bench 2.1 (Harbor) as an order benchmark

Status: design + implementation on local branches, 2026-11-13. Nothing pushed or deployed.

| Piece | Where | State |
| --- | --- | --- |
| Env package `reliquary-terminal-bench` 0.1.0 | reliquary-environments `feat/reliquary-terminal-bench` @f6c1a4a | built, gold-checked: 49/89 (section 4) |
| Harness `terminus-2@1` | reliquary-sandbox `feat/harness-terminus-2` @7fb4917 | built, tested; not pinned in Catalyst |
| `BRIDGED_EVAL_PACKAGES` += `reliquary-terminal-bench` | this branch, `feat/terminal-bench-harbor` | done |
| Images: 89 upstream + 89 verifier images in Hippius | bucket `reliquary-task-images` | all 178 digests checked present |
| Platform template + `tb21` gateway group | reliquary-platform | not started (needs pushed commits) |

## 1. What Terminal-Bench 2.1 is, and what our sandbox is not

89 Harbor tasks (`terminal-bench/terminal-bench-2-1@sha256:7d7bdc1c…`, the leaderboard's
pin), each a prebuilt image (`alexgshaw/<task>:<date>`, ~20 GB compressed in all), an
`instruction.md`, a `tests/test.sh` and an oracle `solution/solve.sh`. No task uses a GPU or
a compose sidecar. Harbor grades **in the agent's own container, with network**: all 89 set
`allow_internet = true`, and every `test.sh` installs curl (apt), uv (astral.sh) and the test
dependencies (`uvx -w pytest==… -w numpy==…`, PyPI, PyTorch's index, two git clones, one
model download) before running the final-state tests.

Our signed sandbox has no network, kills the agent's processes before grading and grades a
pristine box that receives only declared artifacts. Three gaps, three answers:

1. **Network at test time** -> a pre-baked, pinned verifier layer (2.1).
2. **Same-box grading** -> capture the agent's whole filesystem and replay it (2.2).
3. **Network and live processes at solve time** -> not supported; those tasks are excluded
   and counted (2.3). The score is "TB 2.1, offline subset (N/89)", never "TB 2.1".

## 2. Design (`reliquary-terminal-bench`)

Own package on verifiers 0.3.1 like `reliquary-deepswe`: the 89 tasks vendored (text only,
sha256-bound to the `.py` identity), splits `eval` (gated) and `eval_all`, eval only.
Resources and agent timeouts are `task.toml`'s, unchanged (the leaderboard forbids changing
them); the WORKDIR is the image's own (three tasks work outside `/app`).

### 2.1 Verifier image (offline `test.sh`)

`scripts/build_verifier_images.py`, on a build host with network: run `test.sh` once on the
untouched upstream image (runc, network), `docker diff`, keep every added/changed path except
`/tests`, `/logs`, `/tmp`, apt lists/caches/logs and the WORKDIR (except `fix-ocaml-gc`'s
clone and `sam-cell-seg`'s weights, which `test.sh` puts there), tar it to
`/opt/tb-verifier/overlay.tar.gz.<part>` (900 MB parts: Hippius times out on 3 GB layers) in
`FROM <upstream>@<digest>`, plus a file in `/tmp` (gVisor mounts a tmpfs over an empty `/tmp`,
runc does not: one test renames into `/tmp`, EXDEV). `test_*.py` are stubbed while recording,
so no test side effect lands in the overlay (recipe 1 kept `/server` from headless-terminal's
tests). Pinned in the package's `sandbox-images.lock.json` (recipe 2).

Rejected alternatives: open egress (non-deterministic, unsafe); an allow-list proxy at test
time (moving targets: `apt-get` of an unpinned curl, astral's install script; still network);
rewriting each `test.sh` (89 hand edits; the overlay needs none).

### 2.2 Grading the agent's filesystem in a pristine box

- `setup` deletes `/opt/tb-verifier` (the agent sees the upstream image as published) and
  marks the start.
- `finalize` (processes already stopped): every inode whose ctime is past the mark
  (`find -cnewer`; an mtime restored by tar/cp -p/touch cannot hide a change) tarred with
  owners and modes, plus the list of every path (deletions), to `/logs/artifacts`.
- `grading_setup` lists the pristine box.
- `solved`: overlay first (what test.sh's installs leave), the agent's deletions, its files
  over `/`, `tests/` at `/tests`, `bash /tests/test.sh` from the WORKDIR with `UV_OFFLINE=1`;
  reward = `reward.txt`/`reward.json` as Harbor reads it.

Trust equals upstream's same-box grading (the agent's files land as root): fine for eval,
never for training. The forced-draw audit binds actions to the model.

Gateway option needed: `defaults.max_artifact_bytes` >= 256 MiB + path list (default 32 MiB).

### 2.3 Excluded and unverified tasks

- **10 excluded statically** (`gate.EXCLUDED`): graded against a process the agent leaves
  running (sshd, nginx, Flask, gRPC, postfix, QEMU VMs): `configure-git-webserver`,
  `git-multibranch`, `hf-model-inference`, `install-windows-3.11`, `kv-store-grpc`, `mailman`,
  `nginx-request-logging`, `pypi-server`, `qemu-alpine-ssh`, `qemu-startup`.
- **30 unverified by the gold check** (section 4, `gate.NOT_VERIFIED`): 24 references need
  network to solve, 2 tests download at test time, 3 grade over the 810 s step.
- Follow-up to recover them (not built): a **pinned egress mirror** on the gateway for the
  agent phase only — apt from snapshot.ubuntu.com / snapshot.debian.org at a fixed date,
  PyPI through a caching proxy with `exclude-newer`, GitHub/HF by commit — exposed to the box
  through a proxy in the sandbox network namespace. Needs reliquary-sandbox support (boxes
  are `--network none` today), so a sandbox change, not an env change. Services would need a
  "restart hook" convention TB does not have: keep them excluded.

## 3. Images

- Upstream: 89 images mirrored by digest into the Hippius registry (`mirror.sh`, crane,
  docker.io paths kept), from a Targon CPU VM.
- Verifier images: `reliquadotai/terminal-bench-2-1:<task>-r1`, pushed to the same registry
  (only there: gateways pull through it as a docker.io mirror). Overlays total ~20 GB
  compressed; four torch tasks carry ~3 GB each. Mirror: 89/89 in 549+ s on 8 parallel
  crane copies; verifier build ~65 min for 89 at 6 parallel on an 8-vCPU VM.

## 4. Gold check (gVisor, Targon cpu-medium VM)

Per task: agent box (setup, `solve.sh` offline within the agent timeout, `kill -9 -1`,
finalize), fresh grading box (grading_setup, artifacts written back, `solved`). Plus a no-op
episode per task, the excluded tasks too, an `--online` rerun of the failing references
(agent box with network: separates "needs network" from "our grading breaks it"), and the
conformance cases. Raw results: `~/cc-data/tb21/` (not committed).

| Outcome | n | Tasks |
| --- | --- | --- |
| verified: reference 1, no-op 0, offline | **49** | break-filter-js-from-html, cancel-async-tasks, circuit-fibsqrt, cobol-modernization, code-from-image, constraints-scheduling, custom-memory-heap-crash, db-wal-recovery, distribution-search, dna-assembly, dna-insert, extract-elf, extract-moves-from-video, feal-differential-cryptanalysis, feal-linear-cryptanalysis, fix-code-vulnerability, fix-git, git-leak-recovery, gpt2-codegolf, headless-terminal, large-scale-text-editing, llm-inference-batching-scheduler, log-summary-date-ranges, make-mips-interpreter, model-extraction-relu-logits, modernize-scientific-stack, multi-source-data-merger, openssl-selfsigned-cert, password-recovery, path-tracing, path-tracing-reverse, polyglot-c-py, polyglot-rust-c, portfolio-optimization, prove-plus-comm, pytorch-model-recovery, regex-chess, regex-log, sam-cell-seg, sanitize-git-repo, sparql-university, sqlite-db-truncate, torch-pipeline-parallelism, torch-tensor-parallelism, tune-mjcf, video-processing, vulnerable-secret, winning-avg-corewars, write-compressor |
| excluded: live process | 10 | section 2.3 (all also fail the reference offline) |
| network to solve; grades 1 online | 13 | adaptive-rejection-sampler, bn-fit-modify, build-pmars, build-pov-ray, chess-best-move, crack-7z-hash, financial-document-processor, largest-eigenval, merge-diff-arc-agi-task, mteb-retrieve, overfull-hbox, raman-fitting, sqlite-with-gcov |
| network to solve; online state > 256 MB | 8 | caffe-cifar-10, count-dataset-tokens, gcode-to-text, make-doom-for-mips, mteb-leaderboard, pytorch-model-cli, rstan-to-pystan, train-fasttext |
| network to solve; online still fails | 4 | build-cython-ext (10/11), mcmc-sampling-stan (2/6), protein-assembly (0/1), compile-compcert (build outlasts the agent timeout) |
| network inside the tests | 2 | filter-js-from-html (GitHub zip), reshard-c4-data (HF dataset) |
| grading over 810 s (1 CPU, gVisor) | 3 | fix-ocaml-gc, query-optimize, schemelike-metacircular-eval |

All 89 no-ops graded 0. Slowest verified grading: 372 s (regex-chess); largest verified state:
9 MB (custom-memory-heap-crash). Conformance on fix-git and regex-log: reference 1, no-op 0,
mtimes reset to 0 still 1 (ctime capture), output deleted 0 (deletions travel), planted state
0: 10/10.

With a pinned egress mirror for the agent phase (2.3) the 13 "grades 1 online" tasks are
recoverable as they are (62/89), 8 more with a 512 MB state cap (70/89).

## 5. Leaderboard-style harness profile

Terminus 2 (harbor 06d0e86a): first user message = `terminus-json-plain.txt` with the
instruction, JSON `{analysis, plan, commands[{keystrokes, duration}], task_complete}`,
double-confirmed completion, keystrokes into tmux, 10,000-byte head/tail observations,
no turn cap (only the task's agent timeout), proactive summarization near the context
limit, temperature 1 in the TB 2.1 leaderboard config, >= 5 trials per task, score = mean
pass rate over all trials (errors count 0).

`terminus-2@1` (reliquary-sandbox `feat/harness-terminus-2`) is a translated harness: the
same prompt, parser texts, confirmation and truncation; each turn's keystrokes run as one
`bash` call in a fresh `bash -i` that restores the previous turn's cwd, variables, functions
and options. Differences: no tmux (no vim/less/REPL driving), `duration` not waited on, no
summarization (stops at `context_length`), the confirming turn's commands are not run.

Order profile (episode template):

| Knob | Value | Why |
| --- | --- | --- |
| harness | `terminus-2@1` | leaderboard agent |
| tools | `bash` | |
| `wall_s` | the task's `[agent].timeout_sec` (360-12,000 s) | leaderboard enforces it |
| `max_turns` | 64 (`MAX_EPISODE_TURNS`) | Terminus has no cap; raising ours is a protocol decision |
| `per_call_timeout_s` | 600 | long builds within one turn |
| `max_tokens_per_turn` / total | 16,384 / the model's context | no summarization: context bounds |
| trials | `--slots-per-prompt 5` | leaderboard minimum |
| sampling | temperature 1.0, top_p 1.0 | leaderboard config |
| resources | `task.toml` | unchanged |

Note: an episode's `wall_s` is one value per job today (`sandbox.budgets.wall_s`); per-task
agent timeouts need either the bridge's per-task `wall_s` (it already reads
`TaskData.timeout.agent`) to win over the job budget, or the job budget set to the maximum
(12,000 s) and the per-task value enforced by the bridge.

## 6. Remaining steps

1. Push `reliquary-environments feat/reliquary-terminal-bench`, review, merge; record the
   env commit and bridged identity.
2. Push `reliquary-sandbox feat/harness-terminus-2`; bump Catalyst's `reliquary-harness` pin
   (every spec digest moves: update `agentic_swe` digests) and pass `prompt_ids` to
   `tool_calls` at the three validator call sites; omit the system message in
   `corpus/delivery.py` export when the harness has none.
3. Merge this branch (bridged package) on validator and miners.
4. Gateway kit: `tb21` image group (verifier images by digest from the lock), bridged env
   option `defaults.max_artifact_bytes` = 288 MiB, split `eval`.
5. Freeze the set (`reliquary eval build-set --source reliquary-terminal-bench --split eval
   --bridge-file bridge.json`), publish; platform `SUBNET_EPISODE_SETS` template (section 5)
   and `GATEWAY_SETS[<set>] = 'tb21'`, `GROUP_BOOT_HOURS.tb21` (~20 GB of images).
6. Pilot order on one model; compare per-task with published Terminus 2 trajectories.
