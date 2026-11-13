# Terminal-Bench 2.1 (Harbor) as an order benchmark

Status: implemented, PRs open, 2026-11-13. Nothing merged or deployed.

| Piece | Where | State |
| --- | --- | --- |
| Env package `reliquary-terminal-bench` 0.1.0 | reliquary-environments `feat/reliquary-terminal-bench` @a7718445 | `eval` 65/89 (agent online through egress), `eval_offline` 49/89 |
| Harness `terminus-2@1` | reliquary-sandbox `feat/harness-terminus-2` @d680d5e (on main 7743b57) | PR #17 |
| Bridged package, sandbox + harness pin d680d5e, digests, `prompt_ids`, profile | this branch, `feat/terminal-bench-harbor` | this PR |
| Images: 89 upstream + 89 verifier images | Hippius bucket `reliquary-task-images` | all 178 digests checked present |
| Frozen set `reliquary-terminal-bench-eval-r0-n65` | `~/evalsets-work/sets-agentic/tb21-65/` (ops host) | frozen, not published |
| Platform: `tb21` group, egress bootstrap, template | reliquary-platform `feat/tb21-gateway-group` | PR, migration 0065 |

Label: **TB 2.1, offline subset (65/89)** (agent online through the allowlist proxy, grading
offline); without egress, `eval_offline` is **TB 2.1, offline subset (49/89)**.

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
3. **Network at solve time** -> reliquary-sandbox's controlled egress for the agent phase only
   (allowlist proxy, dated apt snapshot, PyPI/uv date pins); **live processes** -> not supported,
   excluded and counted (2.3). The score is "TB 2.1, offline subset (N/89)", never "TB 2.1".

## 2. Design (`reliquary-terminal-bench`)

Own package on verifiers 0.3.1 like `reliquary-deepswe`: the 89 tasks vendored (text only,
sha256-bound to the `.py` identity), splits `eval` (gated with agent egress), `eval_offline` (gated offline) and `eval_all`, eval only.
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

Gateway option needed: `defaults.max_artifact_bytes` = 544 MiB (512 MB state + path list; default 32 MiB).

### 2.3 Excluded and unverified tasks (24 of 89)

- **10 excluded statically** (`gate.EXCLUDED`): graded against a process the agent leaves
  running (sshd, nginx, Flask, gRPC, postfix, QEMU VMs).
- **14 not verified** (`gate.NOT_VERIFIED`): 4 leave a state over the 512 MB cap
  (caffe-cifar-10, gcode-to-text, mteb-leaderboard, pytorch-model-cli); 5 references fail with
  egress (build-cython-ext, compile-compcert, mcmc-sampling-stan, protein-assembly,
  rstan-to-pystan: hosts beyond the allowlist such as data.rcsb.org, opam/coq, or the task
  itself); 2 tests download at test time (filter-js-from-html, reshard-c4-data); 3 grade over
  the 810 s step (fix-ocaml-gc, query-optimize, schemelike-metacircular-eval).

### 2.4 Agent egress

Upstream lets every task reach the internet. The `eval` split is served with the bridged
option `network: {profile: "packages", agent: true, grading: false}`: the agent's box joins
the gateway's egress network (reliquary-sandbox 7743b57, `docs/egress.md`), reaching only the
`packages` allowlist plus `gate.EGRESS_EXTRA_HOSTS` (Hugging Face, CRAN, PyTorch's index, three
task hosts), through apt snapshot 20261111T000000Z and `UV_EXCLUDE_NEWER` /
`PIP_UPLOADED_PRIOR_TO` 2026-11-11, the values of the gold check. Grading stays offline. The
state cap rose to 512 MB (`max_artifact_bytes` 544 MiB) because agents' installs travel in the
captured state. Measured: 2,740 proxied connections, 6 denied (hosts outside the list).

## 3. Images

- Upstream: 89 images mirrored by digest into the Hippius registry (`mirror.sh`, crane,
  docker.io paths kept), from a Targon CPU VM.
- Verifier images: `reliquadotai/terminal-bench-2-1:<task>-r1`, pushed to the same registry
  (only there: gateways pull through it as a docker.io mirror). Overlays total ~20 GB
  compressed; four torch tasks carry ~3 GB each. Mirror: 89/89 in 549+ s on 8 parallel
  crane copies; verifier build ~65 min for 89 at 6 parallel on an 8-vCPU VM.

## 4. Gold checks (gVisor, Targon cpu-medium VMs)

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
| network to solve; online state > 256 MB (first cap) | 8 | caffe-cifar-10, count-dataset-tokens, gcode-to-text, make-doom-for-mips, mteb-leaderboard, pytorch-model-cli, rstan-to-pystan, train-fasttext |
| network to solve; online still fails | 4 | build-cython-ext (10/11), mcmc-sampling-stan (2/6), protein-assembly (0/1), compile-compcert (build outlasts the agent timeout) |
| network inside the tests | 2 | filter-js-from-html (GitHub zip), reshard-c4-data (HF dataset) |
| grading over 810 s (1 CPU, gVisor) | 3 | fix-ocaml-gc, query-optimize, schemelike-metacircular-eval |

All 89 no-ops graded 0. Slowest verified grading: 372 s (regex-chess); largest verified state:
9 MB (custom-memory-heap-crash). Conformance on fix-git and regex-log: reference 1, no-op 0,
mtimes reset to 0 still 1 (ctime capture), output deleted 0 (deletions travel), planted state
0: 10/10.

(The table above is the first, offline run with a 256 MB cap and an unrestricted `--online`
diagnostic; the egress run below supersedes its network rows.)

Second run, agent online (egress VM, sandbox 7743b57 egress setup, verifier images pulled
from Hippius): 79 tasks, reference + no-op. **65 verified**: the 49 above plus 16 needing
network to solve (adaptive-rejection-sampler, bn-fit-modify, build-pmars, build-pov-ray,
chess-best-move, count-dataset-tokens, crack-7z-hash, financial-document-processor,
largest-eigenval, make-doom-for-mips, merge-diff-arc-agi-task, mteb-retrieve, overfull-hbox,
raman-fitting, sqlite-with-gcov, train-fasttext). No offline-verified task regressed online;
all no-ops 0. Largest verified state 474 MB (train-fasttext); slowest grading 372 s.

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

Order profile: `corpus.job.EPISODE_PROFILES["terminal-bench-terminus-2"]`, which the
platform's template for the set carries:

| Knob | Value | Why |
| --- | --- | --- |
| harness | `terminus-2@1` | leaderboard agent |
| tools | `bash` | |
| wall | each task's `[agent].timeout_sec` (360-12,000 s) | leaderboard enforces it |
| `max_turns` | 512 (= `max_calls`) | Terminus has no step cap; time bounds it |
| `max_total_tokens` | 131,072 | the model's context, no summarization |
| `per_call_timeout_s` | 600 | long builds within one call |
| trials | 5 per task | leaderboard minimum |
| sampling | temperature 1.0 | leaderboard config |
| resources | `task.toml` | unchanged |

**Per-task timeout (decision).** On the order path the control already takes a task's whole
budget from the task (`sandbox.evaluation_engagements`: `wall_s` = the bridge's task limit =
`TaskData.timeout.agent`) and refuses one whose wall exceeds the order's
`episode_timeout_seconds`. So the per-task agent timeout already takes precedence; the template
only sets `wall_s` 12,000 s (the largest task's, under the gateway's 14,400 s cap) so that no
task is refused. Setting 12,000 s for every task would have been simpler but wrong: timeouts
are part of the leaderboard's difficulty. Residual: an episode that reaches its wall ends
`expired` (ungraded, an infra end); Harbor grades a timed-out trial.

## 6. Merge and deploy order

1. reliquary-sandbox PR #17 (`terminus-2@1`). Merge with a merge commit (not squash): Catalyst
   and the platform pin d680d5e.
2. reliquary-environments `feat/reliquary-terminal-bench` (merge commit, same reason: a7718445).
3. Catalyst `feat/terminal-bench-harbor`. Deploying it is a **fleet release**: every harness
   digest and the sandbox commit move, so the order control, validators and miners must run it
   before any job pinning the new digests, and jobs pinning the old ones are refused after.
4. reliquary-platform `feat/tb21-gateway-group` after the SWE-Pro/NL2Repo PR (migration 0064
   before 0065). Its Worker deploy must follow step 3 (its templates pin the new sandbox commit
   and digests); its bootstrap moves every group to sandbox d680d5e.
5. Ops: kit gains `groups/tb21/{images,index}.txt` and the wheel (sha256 5ea79213…), from
   `~/evalsets-work/sets-agentic/tb21-65/kit/`; publish the frozen set
   (`reliquary eval publish-set`); the tb21 gateways get `registry.env` (as swe-full).
6. Pilot order on one model, 5 samples per task; compare per task with published Terminus 2
   runs.
