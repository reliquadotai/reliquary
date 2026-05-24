# OpenCodeInstruct code-execution environment

## Problem

The Reliquary validator/miner protocol currently exposes a single
`Environment` (`reliquary/environment/openmathinstruct.py`, configured via
`constants.py:ENVIRONMENT_NAME`). All training signal and validation
traffic comes from one task family — math word problems with extractable
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
miner-supplied code inside the validator process is unacceptable: the
validator holds the hotkey for weight submission, and a sandbox escape =
stake compromise.

## Goal

1. Land a new `OpenCodeInstructEnvironment` that satisfies the existing
   `Environment` Protocol (`base.py`) and slots into
   `load_environment()` without touching the validator main loop, the
   miner engine, the GRPO batcher, or the GRAIL verifier.
2. Execute miner-supplied Python in a sandbox strong enough that an
   evasion does not yield the hotkey. Defense-in-depth via gVisor +
   process-/UID-level isolation.
3. Keep the per-submission verification cost negligible against the
   existing ~5–25 s GRAIL forward pass.
4. Guarantee bit-identical `compute_reward` results between miner and
   validator so `verify_reward_claim` (tolerance 1e-6) passes.
5. Allow safe, reversible rollout: the env can ship dormant, be canaried
   on testnet, then flipped network-wide via a coordinated `constants.py`
   change.

## Non-goals

- No changes to the GRAIL primitive, verification pipeline, GRPO math,
  drand ordering, or window state machine.
- Multi-environment mixing **is now in scope** — see the "Phase 2:
  multi-environment mixing" section below. The original v1 single-env
  swap path remains available; v2 adds true side-by-side training on
  both envs in the same optimizer step via two-batcher orchestration
  and cross-batch gradient accumulation.
- No support for languages other than Python. OpenCodeInstruct is
  Python-only, and the sandbox rootfs ships only a Python 3.11
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

3. `grader/server.py` + `grader/worker.py` — a separate process
   (different UID from the validator) that maintains a warm pool of
   gVisor (`runsc`) sandboxes. Each sandbox holds a long-lived Python
   subprocess that reads `(code, tests)` from stdin and writes
   `(passed, total)` to stdout. The grader server dispatches incoming
   requests round-robin, kills+respawns on timeout/crash, and recycles
   workers every N evaluations to bound memory growth.

Trust model is unchanged from OMI: the miner computes a local reward in
`MiningEngine._build_rollout_submission` (`engine.py:470`) using the same
`env.compute_reward`. The validator re-runs it via
`verify_reward_claim` (`verifier.py:391`) and rejects on mismatch. Both
sides invoke the same grader interface; deterministic `passed/total`
guarantees cross-box agreement.

## Architecture

```
┌─────────────────────────┐         ┌──────────────────────────┐
│   Validator process     │         │   Grader process         │
│   (UID 1000, hotkey)    │         │   (UID 1001, no wallet)  │
│                         │  Unix   │                          │
│  env.compute_reward()  ─┼─socket─►│   manages warm pool      │
│        ▲                │  IPC    │   dispatches → workers   │
│        │                │         │                          │
│  verify_reward_claim    │         │   ┌────────────────────┐ │
│        │                │         │   │ runsc worker #1    │ │
└────────┼────────────────┘         │   │ (Python + tests)   │ │
         │                          │   ├────────────────────┤ │
   reliquary/environment/           │   │ runsc worker #2    │ │
   opencodeinstruct.py              │   │  ...               │ │
                                    │   │ runsc worker #N    │ │
                                    │   └────────────────────┘ │
                                    └──────────────────────────┘
```

The miner runs an analogous architecture. Process isolation is less
critical there (the miner controls the wallet anyway), but reusing the
grader/server design keeps `compute_reward` semantics identical on both
sides.

## Components

### New files

| File | Responsibility |
|---|---|
| `reliquary/environment/opencodeinstruct.py` | `OpenCodeInstructEnvironment` class implementing the `Environment` Protocol. Loads filtered dataset, calls grader client in `compute_reward`. |
| `reliquary/environment/grader_client.py` | IPC client: serializes JSON request, reads response, retries once on `ConnectionError`, returns `0.0` on persistent failure. Never raises. |
| `reliquary/environment/grader/server.py` | Long-running process. Owns the worker pool, the Unix-socket listener, the dispatcher, the watchdog. Exposes Prometheus `/metrics`. |
| `reliquary/environment/grader/worker.py` | Runs *inside* each gVisor sandbox. stdin → `exec(code)` + per-test `exec(t, dict(ns))` → count passes → stdout. |
| `reliquary/environment/grader/bundle/config.json` | OCI runtime config for `runsc`: rootfs path, args, `--network=none`, mount config, resource limits. |
| `reliquary/environment/grader/bundle/rootfs/` | Minimal rootfs (~300 MB): `python3.12` (matches the host image's interpreter, see `Dockerfile`), stdlib, `worker.py`. Built once at image-build time, not committed to git. |
| `scripts/build_grader_bundle.sh` | One-shot script that materializes the OCI rootfs (debootstrap or `python:3.12-slim` extracted + `worker.py` copied in). Run during Docker image build (`RUN scripts/build_grader_bundle.sh` in `Dockerfile`). |
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
| `reliquary/constants.py` | Add `GRADER_SOCKET_PATH`, `GRADER_POOL_SIZE` (default 8), `GRADER_EVAL_TIMEOUT_SECONDS` (default 5). **`ENVIRONMENT_NAME` stays `"openmathinstruct"` until the coordinated flip.** |
| `Dockerfile` | Install `gvisor-runsc`, run `build_grader_bundle.sh`, create UIDs `reliquary` (1000) and `reliquary-grader` (1001). The container currently runs as `root` — this work introduces UID separation. |
| `docker/entrypoint.sh` | **Already exists** (`exec reliquary validate ...`). Modify to: (1) launch `grader_server` via `setpriv --reuid=1001 ... &` in the background, (2) `exec setpriv --reuid=1000 reliquary validate ...`. Deployment-side change: the host volume mount for `~/.bittensor` must target `/home/reliquary/.bittensor` (currently mounted at `/root/.bittensor`) and the host file ownership must be set to UID 1000 before mount. |

## Data flow

End-to-end of one `env.compute_reward(problem, completion)`:

```
1. env.compute_reward(problem, completion):
     tests = json.loads(problem["ground_truth"])      # list[str] of asserts
     code  = _extract_python(completion)              # ```python ... ``` regex,
                                                       # fallback to raw completion
     return grader_client.evaluate(code, tests,
                                   timeout=5.0)        # returns passed/total

2. grader_client.evaluate:
     write JSON request to Unix socket
     read JSON response
     return passed/total (float in [0, 1])
     # On ConnectionError: retry once with 100 ms backoff, then return 0.0.

3. grader/server.py:
     receive request → pick idle worker from pool
     pipe (code, tests) to worker stdin
     read worker stdout: "passed total status"
     reply to client

4. grader/worker.py (inside runsc sandbox):
     subprocess.run([sys.executable, "-c", payload],
                    timeout=GRADER_EVAL_TIMEOUT_SECONDS)
     # payload:
     #   ns = {}
     #   exec(code, ns)
     #   passed = 0
     #   for t in tests:
     #       try: exec(t, dict(ns)); passed += 1
     #       except: pass
     #   print(f"{passed} {len(tests)} ok")
```

### IPC framing

JSON-lines on a `SOCK_STREAM` Unix socket.

```json
// client → server
{"req_id": "<uuid4>", "code": "...", "tests": ["assert f(1) == 1", ...], "timeout_s": 5.0}

// server → client
{"req_id": "<uuid4>", "passed": 3, "total": 5, "status": "ok"}
```

`status` ∈ `{"ok", "timeout", "crash", "grader_error"}`. Anything other
than `"ok"` → `compute_reward` returns `0.0`.

### Determinism

- `total` is fixed by the dataset → identical on miner and validator.
- `passed` is integer-valued, computed by `exec`-ing the same code and
  the same assertions. With the dataset pre-filtered to drop non-
  deterministic tests (random without seed, time, network, FS),
  `passed` is bit-identical across hosts.
- `passed / total` is a rational with denominator ≤ `len(tests)`. Both
  sides format it identically, so `verify_reward_claim`'s `1e-6`
  tolerance is comfortably satisfied.

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

### UID and FS isolation

The container currently runs everything as `root` (verified in
`Dockerfile` and `docker/entrypoint.sh` as of 2026-05-22). This work
introduces UID separation:

```Dockerfile
RUN useradd -m -u 1000 reliquary && \
    useradd -m -u 1001 reliquary-grader
# /home/reliquary/.bittensor permissions are set at runtime by entrypoint.sh
# after the host volume is mounted, since chmod inside the image is overwritten
# by the volume mount.
```

`docker/entrypoint.sh` is updated to:
1. `chown -R reliquary:reliquary /home/reliquary/.bittensor && chmod -R 700 /home/reliquary/.bittensor` after the volume mount.
2. Drop sensitive env vars (`WANDB_API_KEY`, `R2_*`, etc.) before forking the grader: `env -i PATH=... GRADER_SOCKET_PATH=... setpriv --reuid=1001 --regid=1001 --clear-groups grader_server &`.
3. `exec setpriv --reuid=1000 --regid=1000 --clear-groups reliquary validate "${args[@]}"`.

UID 1001 has no read access to `~/.bittensor` (mode 700, owned by UID
1000) and no env vars containing credentials. A sandbox escape from
inside `runsc` lands in the UID-1001 process, which has nothing to
steal.

**Deployment note**: hosts currently mount the wallet at
`/root/.bittensor`. After this change, the mount target becomes
`/home/reliquary/.bittensor` and the host directory must be owned by
UID 1000 (or chowned at container startup with `--cap-add=CHOWN`).
Coordinated with the validator deployment runbook.

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

A CI job runs `(code, tests)` from a fixed 10-problem corpus on two
distinct GitHub runners (`ubuntu-22.04` and `ubuntu-24.04`) and asserts
`passed/total` is bit-identical across runners. If any test flakes, it
is excluded from the production dataset filter.

## Dataset preparation

Offline, one-shot pipeline in `scripts/build_opencodeinstruct_subset.py`:

1. Download `nvidia/OpenCodeInstruct` (6.4 GB).
2. Filter `average_test_score == 1.0` → reference solution passes all
   its own tests. Drops ~30 % of rows. Result: ~3–4 M rows.
3. For each row, parse `unit_tests` (string-encoded list) into a list
   of assertions. Drop rows that fail to parse cleanly.
4. Pattern filter on assertions: drop any row whose tests import or
   reference `random`, `time`, `socket`, `urllib`, `requests`, `os`,
   `subprocess`, `threading`, `multiprocessing`, or non-deterministic
   stdlib (regex-based, conservative).
5. **Double-execution check** with the actual grader bundle: run the
   reference solution against its tests twice with different
   `PYTHONHASHSEED`s. Drop rows that yield different `passed` counts.
6. Push the filtered subset to
   `reliquadotai/opencodeinstruct-deterministic-subset` (HF Hub) for
   easy distribution. Expected final size: ~2–3 M rows, ~3 GB.
7. The `OpenCodeInstructEnvironment` `__init__` loads this filtered
   subset, not the raw NVIDIA dataset.

`get_problem(idx)` returns:

```python
{
    "prompt": row["input"],
    "ground_truth": json.dumps(row["unit_tests_parsed"]),  # list[str]
    "id": sha256(row["input"]).hexdigest()[:16],
}
```

`ground_truth` is JSON because the archive expects a string field; the
serialized tests are small (median ~1–2 KB, max ~9 KB per row) so the
archive footprint stays well under `MAX_ROLLOUT_FILE_SIZE_BYTES`.

## Migration plan

`ENVIRONMENT_NAME` is a consensus parameter in `constants.py` (described
as "Immutable values that all network participants must agree on. No
os.getenv() overrides. Changes require coordinated deployment.").
Switching active env is a coordinated network event.

| Phase | Action | Risk |
|---|---|---|
| 0 | Current state: OMI only | — |
| 1 | Land the PR: env class + grader + sandbox + tests. `ENVIRONMENT_NAME` stays `"openmathinstruct"`. New code is dormant in prod. | None — dormant code path |
| 2 | Deploy validator with this code. Verify grader process starts under UID 1001, pool healthy, `/metrics` exposed. Production traffic unaffected. | Bug at grader startup → log + non-fatal (env not active) |
| 3 | Canary on testnet: side validator with `ENVIRONMENT_NAME="opencodeinstruct"` patched locally. Run 24–48 h. Verify submissions, claim/verify agreement, grader stability. | Isolated from mainnet |
| 4 | Release with `ENVIRONMENT_NAME="opencodeinstruct"` flipped in `constants.py`. Announce flip date to miners with ~7-day upgrade window. Coordinated rollout. | Hard cutover — coordination required |
| 5 | Post-flip 72 h monitoring. | Rollback if disaster |

## Rollback

- **Validator-side issue post-flip**: revert `constants.py:ENVIRONMENT_NAME = "openmathinstruct"`, ship as a patch release, redeploy. OMI cooldown is deterministic and resumes from the latest archive state.
- **Grader instability**: even if every grader call fails,
  `compute_reward` returns `0.0` for all rollouts. The window produces
  no useful training signal but the protocol does not halt. Investigate
  + restart grader; consider revert if the issue is structural.
- **Grader process leak/leftover after rollback**: the grader server is
  harmless when nothing calls it. Can be stopped on a subsequent deploy.

## Phase 2: multi-environment mixing

The v1 design (sections above) is a swap-only model: at any point the
network runs exactly one env, with the active choice flipped via a
coordinated `ENVIRONMENT_NAME` change. Phase 2 lifts that constraint and
trains on both envs **in the same optimizer step**, giving the policy
mixed-gradient updates per window instead of alternating.

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

The single `ENVIRONMENT_NAME` constant is **removed**. CLI default is
the names from `ENVIRONMENT_MIX`. No backward-compat alias — the
hardfork is the cleanup moment.

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

The miner loads **all envs** in `ENVIRONMENT_MIX` at startup. With OMI
(2 shards ≈ 1 GB) + OpenCodeInstruct subset (~3 GB), HF cache sits at
~4-5 GB — acceptable.

### Archive shape

```python
archive["environments"] = [env.name for env in self.envs]  # NEW (was archive["environment"])
archive["batch"] = batch_entries  # entries gain "env_name" tag per submission
```

### Migration plan adjustment

Phase 1 in the main migration plan (land dormant code) covers v1. The
Phase 2 hardfork (env_name in submissions + multi-batcher) is a
**separate release** layered on top. Sequence:

| Phase | What ships |
|---|---|
| 1 | v1 dormant code (this spec, sections above) |
| 2 | Testnet canary: single-env `ENVIRONMENT_NAME="opencodeinstruct"` |
| 3 | Mainnet flip to single OCI (validate the new env in production at 100 %) |
| 4 | v2 hardfork: introduce `ENVIRONMENT_MIX`, `env_name` field, two-batcher, grad-accum train_step |
| 5 | Post-flip monitoring; rollback path = revert constants + protocol version bump |

Phases 1–3 cover the v1 plan; Phases 4–5 cover the v2 addendum. Each
flip is independent; v2 can be deferred indefinitely without blocking
v1's value.

## Open questions / future work

- **Migration to WASM** if gVisor CVE pace becomes a concern. The
  `Environment` Protocol and the IPC contract are runtime-agnostic;
  only `grader/worker.py` and the bundle change.
- **Multi-env mixing** (OMI + code in a single prompt pool) would
  require touching `pick_prompt_idx`, the cooldown map, and the
  batcher. Tracked separately.
- **Reward shaping**: currently `passed / total` linear. Future
  experiment: weighted by test difficulty, or partial credit for tests
  passed after the first failure (currently all run independently, so
  this is already partial credit).
- **Dataset refresh policy**: the deterministic subset on HF Hub is
  built once. If we want periodic refreshes (e.g., as NVIDIA updates
  OpenCodeInstruct), need a CI job + a versioning scheme on the HF
  repo. Out of scope for v1.
