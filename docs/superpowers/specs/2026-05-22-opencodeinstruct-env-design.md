# OpenCodeInstruct code-execution environment

## Current production status (2026-06-02)

This document began as the OpenCode design spec. The current
production-candidate implementation is PR #70.

Current normative behavior:

- OpenCode can run side by side with OpenMath via `ENVIRONMENT_MIX`.
  Docker/operator defaults remain OpenMath-only unless
  `RELIQUARY_ENVIRONMENTS` explicitly includes `opencodeinstruct`.
- OpenCode rewards are **validator-authoritative**. Miners do not know
  hidden cases and should submit placeholder OpenCode rewards; the
  validator recomputes and overwrites rewards before zone checks,
  archive, and training.
- The validator uses a private structured dataset with hidden cases.
- Miners use the public prompt-only mirror:
  `R0mAI/opencodeinstruct-prompts`.
- Default dataset revisions are pinned in
  `reliquary/environment/opencodeinstruct.py`. Override both repo and
  revision together for any custom dataset rollout.
- Miner prompt-only mode requires:

```bash
export RELIQUARY_OCI_PROMPT_ONLY=1
# Optional only if overriding the pinned default:
# export RELIQUARY_OCI_SUBSET_REPO=R0mAI/opencodeinstruct-prompts
# export RELIQUARY_OCI_SUBSET_REVISION=<prompt-dataset-commit-sha>
```

- Validator private mode requires:

```bash
export RELIQUARY_OCI_SUBSET_REPO=<private-structured-dataset>
export RELIQUARY_OCI_SUBSET_REVISION=<structured-dataset-commit-sha>
unset RELIQUARY_OCI_PROMPT_ONLY
```

- The untrusted runsc worker receives only miner code, entrypoint,
  args, and kwargs. It never receives hidden assertion source or
  expected values.
- The trusted grader server owns expected values and computes
  `passed / total`.
- The structured production dataset was built and verified at 50,000
  rows. The public prompt mirror has the same row order and no
  `structured_cases` column.

Older sections below keep the original rationale and architecture
context. Where they mention miner-side reward equality,
`ENVIRONMENT_NAME`, assertion-string workers, or
`reliquadotai/opencode...` dataset names, treat that as superseded by
this status block and the updated sections below.

## Problem

The original Reliquary validator/miner protocol exposed a single
`Environment` (`reliquary/environment/openmathinstruct.py`, configured via
`constants.py:ENVIRONMENT_NAME`). The current protocol uses
`ENVIRONMENT_MIX`, but the original problem remains: most training
signal and validation traffic came from one task family — math word
problems with extractable
`\boxed{...}` answers. This concentrates the learned policy on a narrow
distribution and leaves no diversity in the reward landscape.

We want a second environment, integrated cleanly under the existing
`Environment` Protocol, that exercises a *different* capability of the
model: writing Python code that passes hidden unit tests. The natural
fit at the scale we need is `nvidia/OpenCodeInstruct` (~5 M
instruction/code pairs, each with assertion-based unit tests,
CC BY 4.0).

The obstacle is structural: OMI's reward is a pure string compare
(`normalize(boxed) == normalize(gt)`), so `compute_reward` is microseconds
and trivially deterministic across miner and validator. For code, the
ground truth is a list of Python `assert` statements that must be
**executed** against the miner's completion. Executing arbitrary
miner-supplied code inside the validator process is unacceptable because
validator credentials and training state must remain isolated from
untrusted code.

## Goal

1. Land a new `OpenCodeInstructEnvironment` that satisfies the existing
   `Environment` Protocol (`base.py`) and slots into
   `load_environment()` without touching the validator main loop, the
   miner engine, the GRPO batcher, or the GRAIL verifier.
2. Execute miner-supplied Python in a sandbox strong enough that a
   sandbox failure does not expose validator secrets. Defense-in-depth
   via gVisor plus process and filesystem isolation.
3. Keep the per-submission verification cost negligible against the
   existing ~5–25 s GRAIL forward pass.
4. Keep hidden OpenCode scoring private. Miners submit placeholder
   rewards in prompt-only mode; the validator recomputes and overwrites
   OpenCode rewards authoritatively.
5. Allow safe, reversible rollout: the env can ship dormant, be canaried
   on testnet, then enabled network-wide via a coordinated environment
   mix change.

## Non-goals

- No changes to the GRAIL primitive, verification pipeline, GRPO math,
  drand ordering, or window state machine.
- Multi-environment mixing **is now in scope** — see the "Phase 2:
  multi-environment mixing" section below. The original v1 single-env
  swap path remains available; v2 adds true side-by-side training on
  both envs in the same optimizer step via two-batcher orchestration
  and cross-batch gradient accumulation.
- No support for languages other than Python. OpenCodeInstruct is
  Python-only, and the sandbox rootfs ships only a pinned Python 3.12
  interpreter.
- No code linting / regex-based pre-filtering of miner submissions. The
  defense is the sandbox, not a deny-list. A regex filter would create
  false security.
- No per-test sub-sandboxing. All tests for one problem run in a single
  Python subprocess inside the sandbox; per-test namespace copying
  prevents cross-test leakage. We accept that a malicious test in the
  *dataset* (not the miner's completion) could affect peer tests in the
  same problem — but the dataset is curated by NVIDIA + filtered offline.

## Approach

Three new pieces glued together by the existing `Environment` Protocol:

1. `OpenCodeInstructEnvironment` — a thin Python class on the validator
   side. Loads the filtered HF dataset, exposes the protocol surface,
   delegates `compute_reward` to a grader client over a Unix socket.

2. `grader_client.py` — a small IPC client used inside the env class.
   Frames JSON requests, retries once on transient socket error, never
   raises.

3. `grader/server.py` + `grader/worker.py` — a separate isolated
   process that maintains a warm pool of gVisor (`runsc`) sandboxes.
   The trusted server owns hidden cases and
   expected values. Each sandbox holds a long-lived Python subprocess
   that receives only `(code, entrypoint, args, kwargs)` and returns a
   JSON-safe primitive output. The server compares that output to the
   private expected value, dispatches requests round-robin,
   kills+respawns on timeout/crash, and recycles workers every N
   evaluations to bound memory growth.

Trust model differs from OMI: OpenMath still verifies miner-claimed
rewards, but OpenCode uses validator-authoritative rewards. Miners load
only the public prompt mirror and submit placeholder rewards. The
validator loads private structured cases, calls the trusted grader, then
overwrites `rollout.reward` and commit rollout metadata before zone
checks, archive, and training.

## Architecture

```
┌─────────────────────────┐         ┌──────────────────────────┐
│   Validator process     │         │   Trusted grader process │
│                         │  Unix   │                          │
│  env.compute_reward()  ─┼─socket─►│   manages warm pool      │
│        ▲                │  IPC    │   dispatches → workers   │
│        │                │         │                          │
│  private reward compute │         │   ┌────────────────────┐ │
│        │                │         │   │ runsc worker #1    │ │
└────────┼────────────────┘         │   │ (Python call only) │ │
         │                          │   ├────────────────────┤ │
   reliquary/environment/           │   │ runsc worker #2    │ │
   opencodeinstruct.py              │   │  ...               │ │
                                    │   │ runsc worker #N    │ │
                                    │   └────────────────────┘ │
                                    └──────────────────────────┘
```

Miners do not run the grader in live prompt-only mode. They load the
public prompt mirror and submit valid rollouts with placeholder
OpenCode rewards; local synthetic code tests may be useful for quality
filtering, but they are not protocol reward authority.

## Components

### New files

| File | Responsibility |
|---|---|
| `reliquary/environment/opencodeinstruct.py` | `OpenCodeInstructEnvironment` class implementing the `Environment` Protocol. Loads filtered dataset, calls grader client in `compute_reward`. |
| `reliquary/environment/grader_client.py` | IPC client: serializes JSON request, reads response, retries once on `ConnectionError`, returns `0.0` on persistent failure. Never raises. |
| `reliquary/environment/grader/server.py` | Long-running trusted process. Owns hidden expected values, worker pool, Unix-socket listener, dispatcher, watchdog, and Prometheus `/metrics`. |
| `reliquary/environment/grader/worker.py` | Runs *inside* each gVisor sandbox. stdin → `exec(code)` + call one entrypoint with args/kwargs → JSON-safe output. It never receives hidden assertions or expected values. |
| `reliquary/environment/grader/bundle/config.json` | OCI runtime config for `runsc`: rootfs path, args, `--network=none`, mount config, resource limits. |
| `reliquary/environment/grader/bundle/rootfs/` | Minimal rootfs (~300 MB): pinned `python3.12-slim` rootfs + stdlib + `worker.py`. Built into the Docker image by a multi-stage build, not committed to git. |
| `scripts/build_grader_bundle.sh` | Manual/local helper that materializes the same OCI rootfs by exporting the pinned Python image. The production Docker image does not depend on Docker-in-Docker at container startup. |
| `scripts/build_opencodeinstruct_subset.py` | Offline dataset preparation (see Dataset prep section). |
| `tests/unit/test_opencodeinstruct_environment.py` | Pure-Python tests with mocked grader client. |
| `tests/unit/test_grader_client.py` | Serialization, retry, never-raise. |
| `tests/unit/test_grader_worker.py` | Eval logic without the sandbox: code OK / KO, test isolation, syntax error in a test does not invalidate peers, empty completion. |
| `tests/integration/test_grader_e2e.py` | Real `runsc` + real pool + real socket. Includes hostile-code cases. |
| `tests/integration/test_opencodeinstruct_env_smoke.py` | Loads filtered dataset slice, exercises `get_problem` + `compute_reward` on known correct/incorrect completions. |

### Modified files

| File | Change |
|---|---|
| `reliquary/environment/__init__.py` | Add `OpenCodeInstructEnvironment` import + `"opencodeinstruct"` case in `load_environment`. |
| `reliquary/constants.py` | Add `GRADER_SOCKET_PATH`, `GRADER_POOL_SIZE` (default 8), `GRADER_EVAL_TIMEOUT_SECONDS` (default 5), and include OpenCode in `ENVIRONMENT_MIX` when enabled. |
| `Dockerfile` | Install pinned `runsc`, build the pinned Python rootfs into the image, and create the optional validator service user. |
| `docker/entrypoint.sh` | Launch the trusted grader supervisor with a scrubbed environment so it can start `runsc`; the untrusted worker runs as UID/GID 65534 inside the OCI config. The validator can optionally drop to its service user once wallet permissions are prepared. |
| `reliquary/validator/batcher.py` | For validator-authoritative envs, recompute and overwrite rollout rewards before zone/archive/training. |
| `reliquary/miner/engine.py` | For validator-authoritative envs, submit placeholder rewards instead of trying local hidden-case scoring. |

## Data flow

End-to-end of one `env.compute_reward(problem, completion)`:

```
1. env.compute_reward(problem, completion):
     case_id = problem["ground_truth"]                # opaque private id
     cases = self._cases_by_id[case_id]               # trusted server side only
     code  = _extract_python(completion)              # ```python ... ``` regex,
                                                       # fallback to raw completion
     return grader_client.evaluate_cases(code, cases,
                                         timeout=5.0)  # returns passed/total

2. grader_client.evaluate_cases:
     write JSON request to Unix socket
     read JSON response
     return passed/total (float in [0, 1])
     # On ConnectionError: retry once with 100 ms backoff, then return 0.0.

3. grader/server.py:
     receive request with private structured cases
     for each case:
       send only (code, entrypoint, args, kwargs) to an idle worker
       receive worker output/status
       compare output to private expected value
     reply to client

4. grader/worker.py (inside runsc sandbox):
     read JSON request
     exec(code, ns)
     resolve function or Solution().method entrypoint
     call entrypoint(*args, **kwargs)
     normalize JSON-safe primitive output
     return {"status": "ok", "output": value}
```

### IPC framing

JSON-lines on a `SOCK_STREAM` Unix socket.

```json
// client → server
{"req_id": "<uuid4>", "code": "...", "cases": [{"entry": {"kind": "function", "name": "add"}, "args": [1, 2], "kwargs": {}, "expected": 3, "compare": "exact"}], "timeout_s": 5.0}

// server → client
{"req_id": "<uuid4>", "passed": 3, "total": 5, "status": "ok"}
```

The worker-facing request omits `expected`; the expected value stays in
the trusted server. `status` includes `ok`, `timeout`, `runtime_error`,
`forbidden_import`, `bad_output`, and `grader_error`. Anything other
than a comparable `ok` output scores zero for that case.

### Determinism

- `total` is fixed by the private structured dataset.
- `passed` is computed only by the validator/trusted grader, so miners
  no longer need cross-box identical reward computation for OpenCode.
- The dataset pre-filter drops non-deterministic cases, reference
  crashes, and timeouts. Runtime failures score zero rather than
  crashing the validator.

## Security & isolation

### Sandbox: gVisor (`runsc`)

- TCB reduction: `runsc`'s Sentry (~50 k LoC Go, memory-safe) replaces
  the Linux kernel ABI for sandboxed code. The host kernel sees only a
  tiny set of safe syscalls from Sentry itself.
- Deployment: one apt package, invoked as a binary. No daemon, no
  Docker-in-Docker, no `/dev/kvm` requirement. Same `CAP_SYS_ADMIN`
  caveat as firejail — verified at image-build time.
- Why not firejail: relies on the Linux kernel + seccomp filter. CVEs
  in the kernel namespaces/eBPF/io_uring path are routine; the seccomp
  default is a deny-list. Insufficient defense-in-depth alone for the
  validator's threat model.
- Why not Firecracker: requires `/dev/kvm`, slower per-exec, more
  operational complexity. Overkill against gVisor for our request rate.
- Why not WASM (Pyodide): integration overhead, ~3–5× slower, missing
  stdlib (`socket`, `subprocess`, `threading`). Could be a future
  migration path if gVisor's CVE rate proves unacceptable; the
  `Environment` API does not change.

### Process and filesystem isolation

The validator and grader run under separate service identities. The
grader starts with a minimal environment and no access to validator
credential mounts. If sandboxed code escapes the worker, it lands in the
grader boundary rather than the validator boundary.

Exact host paths, ownership commands, and deployment-specific credential
layout belong in the private operator runbook, not this public design
spec.

### Per-worker resource limits

| Limit | Value | Mechanism |
|---|---|---|
| CPU time | 5 s | `runsc --cpu-limit` + `setrlimit(RLIMIT_CPU)` inside |
| Memory | 256 MB | `runsc --memory-limit` + cgroup |
| Network | none | `runsc --network=none` |
| Filesystem writes | tmpfs `/tmp` only, 10 MB cap | OCI mount config |
| Max processes | 16 | `setrlimit(RLIMIT_NPROC)` |
| Wall-clock | 7 s | `subprocess.run(timeout=7)` outer |

### Worker lifecycle

- **Pool size**: 8 by default (`GRADER_POOL_SIZE`). Sized for
  `M_ROLLOUTS × concurrent_submissions`.
- **Death detection**: pipe EOF on worker → server respawns slot.
- **Preventive recycle**: every 1 000 evaluations the worker is killed
  and respawned (bounds latent memory consumption).
- **Quarantine**: two consecutive timeouts on the same worker forces a
  respawn (handles corrupted state).
- **Grader-wide death**: a supervisor (systemd unit or in-Python
  parent watchdog) restarts the grader server. During the gap
  (~seconds) `compute_reward` returns `0.0` with `status="grader_error"`.

## Error handling

`compute_reward` must never raise (per `Environment` Protocol). All
failure modes resolve to a numeric reward, typically `0.0`:

| Failure | Behavior |
|---|---|
| Grader socket unreachable | client retries once @ 100 ms backoff, then returns `0.0`. Logged + counted. |
| Worker timeout | `(0, total, "timeout")` → reward `0.0`. Worker killed and respawned. |
| Worker subprocess crash (segfault, sandbox abort) | `(0, total, "crash")` → reward `0.0`. Worker respawned. |
| Grader server itself dead | client returns `0.0` with `status="grader_error"`. Supervisor restarts grader. |
| Empty / unextractable code in completion | `code = ""`, all tests fail in exec, returns `0/total = 0.0`. |
| Malformed JSON IPC | client treats as connection error, retries, then `0.0`. |
| Dataset row with `total == 0` (no tests) | Skipped at dataset load time. Should never reach `compute_reward`. |

The "all zeros" failure mode is *self-evident in metrics* (massive spike
in `grader_eval_total{status!="ok"}`) and non-catastrophic for the
network — every miner scores the same, no one is unfairly penalized
relative to peers, the window just produces no useful training signal.

## Observability

Grader exposes Prometheus metrics on a local HTTP endpoint (loopback
only):

- `grader_eval_total{status="ok|timeout|crash|grader_error"}`
- `grader_eval_duration_seconds` (histogram)
- `grader_pool_busy_workers` (gauge, 0..N)
- `grader_worker_restarts_total{reason="death|quarantine|recycle"}`

Structured logs, rate-limited 1/sec/type: timeouts, crashes, IPC
errors. Per-window aggregates show up in the validator archive under a
new key `archive["grader_failures"] = {"timeout": int, "crash": int,
"grader_error": int}` alongside the existing `reject_summary`.

## Testing

### Unit (no HF, no `runsc`)

- `test_opencodeinstruct_environment.py`
  - `_extract_python` parses fenced ```python``` blocks, falls back to raw text
  - `get_problem(idx)` returns the protocol shape and is deterministic
  - `compute_reward` delegates to a mocked `grader_client` and returns `passed/total`
  - Same `(problem, completion)` → same reward (no hidden state)

- `test_grader_client.py`
  - JSON request/response round-trip
  - `ConnectionError` triggers one retry, then `0.0`
  - Never raises on any malformed response

- `test_grader_worker.py`
  - Code OK against all tests → `(N, N)`
  - Code KO against half tests → `(N/2, N)`
  - `exec` exception in one test does not break peers
  - Empty code string → `(0, N)`
  - Per-test namespace isolation: test 1 mutates `ns`, test 2 sees pristine namespace

### Integration (marked `@pytest.mark.integration`, real `runsc`)

- `test_grader_e2e.py`
  - Round-trip through real socket + real warm pool
  - Hostile completions (each independently asserted blocked):
    - `socket.socket()` → connection refused / blocked
    - `open("/etc/passwd", "w")` → permission denied
    - `while True: pass` → killed at 5 s
    - `import os; os.fork()` ×100 → blocked at `RLIMIT_NPROC`
  - Worker death + automatic respawn observed via metrics

- `test_opencodeinstruct_env_smoke.py`
  - Loads a 100-row slice of the filtered dataset
  - `get_problem(0)` returns valid shape
  - `compute_reward` on a known-correct completion → `1.0`
  - `compute_reward` on a known-incorrect completion → `0.0`

### Cross-box determinism

A CI job runs a fixed structured-case corpus on two distinct GitHub
runners (`ubuntu-22.04` and `ubuntu-24.04`) and asserts the resulting
artifact comparison is bit-identical across runners. If any case flakes,
it is excluded from the production dataset filter.

## Dataset preparation

Offline, one-shot pipeline in `scripts/build_opencodeinstruct_subset.py`:

1. Download `nvidia/OpenCodeInstruct` (6.4 GB).
2. Filter `average_test_score == 1.0` → reference solution passes all
   its own tests. Drops ~30 % of rows. Result: ~3–4 M rows.
3. For each row, parse `unit_tests` (string-encoded list) into a list
   of assertions. Drop rows that fail to parse cleanly.
4. Convert only simple deterministic assertions into structured cases:
   `assert fn(args...) == literal`, reversed equality, truthy/falsy
   asserts, and the same shapes for `Solution().method(args...)`.
5. Drop assertion source that contains imports, arbitrary expressions,
   non-JSON expected values, filesystem/network/process usage, or
   non-deterministic stdlib references.
6. **Double-execution check**: run the reference solution against the
   structured cases twice with different `PYTHONHASHSEED`s. Drop rows
   that disagree, crash, or time out.
7. Save and publish two row-order-matched datasets:
   - a private validator dataset containing structured hidden cases;
   - a public miner prompt mirror containing prompts only.
8. The validator loads the private structured dataset. Miners load the
   public prompt-only mirror with `RELIQUARY_OCI_PROMPT_ONLY=1`.

`get_problem(idx)` returns:

```python
{
    "prompt": row["input"],
    "ground_truth": case_set_id,  # opaque id, not hidden tests
    "id": sha256(row["input"]).hexdigest()[:16],
}
```

The private environment keeps `case_set_id -> structured_cases` in
memory. The public prompt mirror produces an empty case list and is only
valid for miners, never validators.

## Migration plan

The live switch is now controlled by `RELIQUARY_ENVIRONMENTS` /
`ENVIRONMENT_MIX` and deployment coordination. OpenCode can be enabled
beside OpenMath instead of replacing it.

| Phase | Action | Risk |
|---|---|---|
| 0 | Current state: OMI only | — |
| 1 | Land PR #70: env class + structured grader + sandbox + private/public datasets + validator-authoritative rewards. | Dormant if `RELIQUARY_ENVIRONMENTS=openmathinstruct` |
| 2 | Deploy validator math-only first. Verify no behavior change. | Low |
| 3 | Announce miner upgrade. Miners set `RELIQUARY_OCI_PROMPT_ONLY=1`; custom prompt repos must also set `RELIQUARY_OCI_SUBSET_REPO` and `RELIQUARY_OCI_SUBSET_REVISION`. | Misconfigured miners may fail to load OpenCode prompts |
| 4 | Canary mixed env on live validator with the private structured dataset. Monitor per-env health. | Grader latency / OpenCode out-of-zone rate |
| 5 | Expand after 24–72 h stable telemetry. | Rollback to `RELIQUARY_ENVIRONMENTS=openmathinstruct` if needed |

## Rollback

- **Validator-side issue post-flip**: set
  `RELIQUARY_ENVIRONMENTS=openmathinstruct` and restart. OMI cooldown is
  deterministic and resumes from the latest archive state.
- **Grader instability**: even if every grader call fails,
  `compute_reward` returns `0.0` for all rollouts. The window produces
  no useful training signal but the protocol does not halt. Investigate
  + restart grader; consider revert if the issue is structural.
- **Grader process leak/leftover after rollback**: the grader server is
  harmless when nothing calls it. Can be stopped on a subsequent deploy.

## Multi-environment mixing

The production-candidate design trains on both envs **in the same
optimizer step**, giving the policy mixed-gradient updates per window
instead of alternating.

### Goal

Each optimizer step sees `B_BATCH` prompts per env, summed across all
active envs. With two envs (OMI + OpenCodeInstruct), one window
produces 16 prompts × 8 rollouts = 128 sequences contributing to a
single optimizer step. VRAM stays at the v1 level because the micro-
batches are processed sequentially with gradient accumulation between
them.

### Protocol change (soft hardfork)

`RolloutSubmission` gains a required `env_name: str` field. The
validator dispatches each submission to the correct env (for prompt
lookup and reward verification) based on this tag. Bump
`GRAIL_PROOF_VERSION` to signal incompatibility with v1 miners.

### Constants change

```python
# In reliquary/constants.py:
ENVIRONMENT_MIX: list[tuple[str, int]] = [
    ("openmathinstruct",   B_BATCH),  # 8
    ("opencodeinstruct",   B_BATCH),  # 8
]
GRAD_ACCUM_STEPS: int = len(ENVIRONMENT_MIX)  # 2 — derived, not separately tunable
```

The single `ENVIRONMENT_NAME` constant is **removed**. `ENVIRONMENT_MIX`
keeps the available production mix, but CLI/Docker defaults are
OpenMath-only through `DEFAULT_ENVIRONMENTS`. OpenCode is activated only
when `RELIQUARY_ENVIRONMENTS` explicitly includes `opencodeinstruct`.

### Batcher orchestration

One `GrpoWindowBatcher` per active env (approach A). Each batcher
maintains its own `CooldownMap` independently — cooldown key remains
`prompt_idx` within an env (no need for env-keyed tuples since the
batchers are physically separate). At seal time the service collects
one batch per env and passes the list to `train_step`.

If one env's batcher fails to reach `B_BATCH` accepted submissions by
the seal deadline (underflow), that window does not train at all —
matching the existing partial-seal skip behaviour. We do NOT fall back
to "train on the env that did seal" because that would amplify the
imbalance we are trying to avoid.

> **Production amendment (2026-07-14):** the one-window underflow rule above
> caused a liveness cliff once OpenMath became sparse while OpenCode still
> filled. The coordinator now uses bounded balanced cross-window accumulation.
> It retains at most `B_BATCH` groups per environment under one public
> checkpoint revision and trains only after every environment reaches its
> target. This preserves the original no-imbalance invariant without throwing
> away all clean partial-window signal.

### Training loop

```python
def train_step(model, batches: list[list], *, ref_model, window_index=None):
    """One optimizer step over the union of batches.

    `batches` is one batch per active env. All rollouts contribute
    backward calls before a single optimizer.step() is invoked, so the
    effective batch size is sum(len(b) for b in batches) prompts.
    """
    n_total_rollouts = sum(len(g.rollouts) for b in batches for g in b)
    _optimizer.zero_grad()
    for batch in batches:
        for group in batch:
            for rollout in group.rollouts:
                loss = (ppo + KL_BETA * kl) / n_total_rollouts
                loss.backward()
    _optimizer.step()
    _scheduler.step()
```

VRAM cost is unchanged from v1 because only one rollout's activations
are alive at a time. Compute cost per optimizer step roughly doubles
(twice as many backward calls), but per-window wall-clock is dominated
by GRAIL verification (5–25 s) which is unaffected.

### Miner side

`pick_prompt_idx` first samples an env according to the mix's weights
(equal weights for an 8-8 setup, but the data structure supports any
ratio), then picks a prompt from that env. The submission carries
`env_name` so the validator can verify against the right env without
guessing.

The miner loads active envs at startup. For OpenCode it must use the
public prompt-only mirror, not the private structured dataset:

```bash
export RELIQUARY_OCI_PROMPT_ONLY=1
# Optional only when using a custom prompt mirror:
# export RELIQUARY_OCI_SUBSET_REPO=R0mAI/opencodeinstruct-prompts
# export RELIQUARY_OCI_SUBSET_REVISION=<prompt-dataset-commit-sha>
```

In this mode the miner does not launch a local grader and does not know
hidden-case rewards. OpenCode reward claims are placeholders; the
validator recomputes the real reward vector privately.

### Archive shape

```python
archive["environments"] = [env.name for env in self.envs]  # NEW (was archive["environment"])
archive["batch"] = batch_entries  # entries gain "env_name" tag per submission
```

### Migration plan adjustment

The previous single-env `ENVIRONMENT_NAME` migration plan is superseded.
`ENVIRONMENT_MIX`, `env_name`, two-batcher routing, and gradient
accumulation are already part of the production-candidate branch. The
remaining rollout choice is operational: deploy math-only first, then
enable mixed env after miners have upgraded to prompt-only OpenCode
mode.

## Open questions / future work

- **Migration to WASM** if gVisor CVE pace becomes a concern. The
  `Environment` Protocol and the IPC contract are runtime-agnostic;
  only `grader/worker.py` and the bundle change.
- **Private/generated tasks** are the next major game-design step.
  Prompt-only mirrors protect hidden cases, but static public-source
  prompts can still create lookup pressure over time.
- **Reward shaping**: currently `passed / total` linear. Future
  experiment: weighted by test difficulty, or partial credit for tests
  passed after the first failure (currently all run independently, so
  this is already partial credit).
- **Dataset refresh policy**: the structured/private subset and public
  prompt mirror are built once. If we want periodic refreshes (e.g., as
  NVIDIA updates OpenCodeInstruct), need a CI job + a versioning scheme on the HF
  repo. Out of scope for v1.
