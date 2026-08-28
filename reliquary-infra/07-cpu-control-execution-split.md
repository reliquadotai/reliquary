# CPU control and hostile-execution split

## Decision

> Implementation update 2026-08-28: the versioned remote sandbox contract,
> zero-secret mTLS agent, shadow/authoritative client modes, container profile,
> and tests exist on the prototype branch. The AX42 `ctrl-01` foundation is
> deployed in shadow mode. No `cpu-exec-01` has been provisioned, so the remote
> executor is not live or authoritative. The ordering below remains the
> production gate.

Reliquary should have two main CPU machines, but they must be separated by
role rather than by the vague labels "small CPU" and "large CPU":

| Machine | Role | Trust | Starting capability |
|---|---|---|---|
| `ctrl-01` | Public API, validator FSM, Bittensor validation, deterministic admission, trusted Code result comparison, scheduling and scoring | Trusted control | 16 fast physical cores, 128 GB ECC, RAID1 NVMe |
| `cpu-exec-01` | Execute miner-controlled Code workloads in gVisor initially and Firecracker after qualification | Hostile execution | 48 physical cores / 96 SMT threads, 128 GB minimum and 256 GB preferred, RAID1 datacenter NVMe, KVM |
| `signer-01` | Narrow semantic hotkey operations | Highest online trust | Separate small 2-to-4-vCPU machine |

An AX162-class EPYC 9454P is a sensible **benchmark candidate** for
`cpu-exec-01`. It has 48 physical cores and 96 hardware threads. It is not a
96-core machine, and capacity planning must not count SMT siblings as 96
independent hostile-workload cores. Start with SMT disabled, or allocate both
siblings of a physical core to the same security domain, then benchmark the
performance/security tradeoff explicitly.

If "96 CPU" means 96 physical cores, do not buy that initially. Current
Reliquary concurrency does not justify it without a failing 2x-load benchmark.

## Important deployment-order constraint

The final validator can be CPU-only, but it cannot be moved cleanly to
`ctrl-01` today. The current proof worker is a local multiprocessing child that
opens a local CUDA device, and the validator parent still holds model state.

Use this order:

1. split hostile CPU execution from the current validator/GPU machine;
2. qualify the remote CPU path;
3. implement and shadow the versioned remote `ProofExecutor` described in the
   GPU plan;
4. only then move the validator/control service to the CPU-only `ctrl-01`;
5. move hotkey operations to `signer-01` after control behavior is stable.

This avoids combining the CPU move, remote proof transport, and wallet move in
one cutover.

## Exact boundary

The smallest safe split is not to move the whole current grader server. Keep
the trusted coordinator and result comparison on control, and move only the
sandbox execution backend:

~~~text
miner HTTPS request
  -> validator admission process
  -> existing GraderClient over local Unix socket
  -> trusted GraderCoordinator on ctrl-01
       - owns complete cases and expected values
       - preserves candidate-versus-infrastructure failure semantics
       - compares bounded JSON output
       - computes passed / total
  -> mutually authenticated bounded execution request
  -> cpu-exec-01
       - zero-secret executor agent
       - gVisor or Firecracker sandbox pool
       - receives code + entrypoint + args/kwargs, never expected values
       - returns bounded primitive output/status
  -> trusted comparison on ctrl-01
~~~

The current code already contains the useful half of this boundary:

- `GraderClient` is a small fail-closed JSON client;
- `GraderServer._dispatch` retains expected values outside the sandbox;
- the worker receives only code, entrypoint, arguments, keyword arguments and
  a timeout;
- only JSON-safe primitive outputs cross back into trusted scoring.

Refactor the worker call behind a `SandboxExecutor` interface instead of
rewriting admission or changing miner protocol behavior.

Conceptually:

~~~python
class SandboxExecutor(Protocol):
    def execute(self, request: ExecutionRequest) -> ExecutionResult: ...

class LocalRunscExecutor(SandboxExecutor): ...   # existing behavior
class RemoteExecutor(SandboxExecutor): ...       # cpu-exec-01
class FirecrackerExecutor(SandboxExecutor): ...  # execution-host setting
~~~

One network request should represent one candidate completion and its ordered
case inputs. The executor must preserve today's fresh-execution and per-case
timeout semantics and return ordered results. Expected values and comparison
modes never cross the machine boundary.

## Minimal wire contract

The internal request needs only:

- protocol and runtime-image version;
- deterministic `job_id` and attempt generation;
- code and code SHA-256;
- ordered case IDs, entrypoint, args and kwargs, without expected output;
- per-case CPU/wall timeout and overall deadline;
- stdout/stderr, memory, process, file, disk and response-size limits.

The result needs only:

- `job_id`, attempt generation and runtime digest;
- ordered bounded output or candidate-failure status for each case;
- wall/CPU/resource telemetry;
- executor identity and result digest.

Use direct JSON-over-HTTPS with mTLS on the private execution network. Add
strict request/response size limits and keep the current local Unix socket
between admission children and the trusted coordinator. A queue cluster,
service mesh, generic remote shell, and provider control plane are unnecessary.

The transport must distinguish:

- candidate failure: deterministic zero reward;
- executor timeout/crash/unreachable/malformed result: infrastructure failure,
  which aborts or defers admission and must never become a negative label.

## Why a 48-core / 96-thread host is plausible

The active profile has 16 rollouts. The current runtime permits four concurrent
Code admission groups and provisions `4 * M_ROLLOUTS`, or 64, warm sandbox
slots. The theoretical burst is therefore:

~~~text
4 Code groups * 16 candidate completions = 64 sandbox evaluations
~~~

The observed sandbox kernel is usually short, while worst-case miner code can
consume its full timeout. Sixty-four warm slots do not imply 64 continuously
busy cores. A 48-physical-core host can queue the burst inside the current
20-second Code admission wall if measured execution and startup remain within
budget, but that must be proven under malicious full-timeout load.

Recommended candidate configuration:

- 48 physical cores / 96 SMT threads;
- 256 GB registered ECC preferred for a clean preboot microVM pool;
- two datacenter NVMe devices in RAID1 for OS and immutable images;
- swap disabled;
- KVM/IOMMU verified before acceptance;
- 1 Gbit/s initially because Code request/results are small;
- no GPU, wallet, HF/R2/provider token, database credential or CI key.

The control candidate should favor fast single-core latency over core count:

- 16 physical cores / 32 threads;
- 128 GB ECC;
- two datacenter NVMe devices in RAID1;
- 1 Gbit/s;
- no privileged container or local sandbox runtime after cutover.

## Network and security policy

`ctrl-01` may initiate requests to the executor endpoint. `cpu-exec-01` must
not be able to initiate connections to the validator API, signer, database,
object storage, GPU control plane or administrative network.

Apply both private-network ACLs and host nftables:

- Internet to `cpu-exec-01`: deny;
- management VPN to executor SSH: allow narrowly;
- `ctrl-01` to executor mTLS port: allow;
- executor to `ctrl-01`: established replies only;
- sandbox network interface: absent;
- sandbox to host/LAN/Internet: deny;
- metrics: pull through a read-only, secret-free endpoint or forward only a
  bounded metric stream.

Assume a sandbox escape compromises `cpu-exec-01`. The loss must be limited to
capacity and potentially dishonest grades; it must reveal no credential or
route into control. Retain canaries and randomly replay a small sample on an
independent backend to detect dishonest or broken execution.

## Implementation sequence

### 0. Finish the current correctness gate

Land the checkpoint-startup coherence fix described in the Rom review before
another production cutover. CPU development and lab provisioning can proceed
in parallel, but production authority should change one boundary at a time.

### 1. Capture the sizing baseline

Across at least 100 windows, record:

- Code groups and completion evaluations per window;
- arrival burst, queue wait and peak simultaneous busy workers;
- execution wall time and CPU time p50/p95/p99;
- cases per completion;
- worker RSS, host page cache and sandbox startup/recycle time;
- full-timeout, crash and infrastructure-failure counts;
- full Code admission p50/p95/p99.

Replay the same traffic at 1x and 2x load. Include a worst-case corpus in which
every job attempts to consume the full CPU, memory, process and output limits.

### 2. Extract the executor interface locally

- leave `GraderClient`, admission workers and reward comparison unchanged;
- wrap the current local runsc worker pool as `LocalRunscExecutor`;
- add immutable request/result types and explicit failure taxonomy;
- prove archive and result parity against current `main`;
- keep local execution authoritative.

This PR should contain no networking and no Firecracker.

### 3. Add the remote gVisor agent

- create a standalone, wallet-free executor entrypoint by reusing the current
  grader worker/pool implementation;
- add the bounded mTLS transport and `RemoteExecutor`;
- provision one Firecracker-ready bare-metal host but run the known gVisor
  backend first;
- keep comparison and authoritative score on control;
- expose capacity, queue, timeout, recycle and resource metrics.

### 4. Shadow and attack-test

- duplicate sampled live Code jobs to local and remote execution;
- compare every output and status before comparison to expected values;
- run the malicious miner corpus;
- inject disconnect, delay, duplicate, malformed result, stale attempt,
  executor restart, worker death and host reboot;
- verify infrastructure failures never produce negative training labels;
- verify no sandbox can reach any network or observe another job.

### 5. Cut over remote gVisor

Make remote gVisor authoritative only when:

- deterministic fixture parity is exact;
- live deterministic-result parity is exact, excluding explicitly classified
  nondeterministic/time-bound cases;
- remote transport adds no material admission tail;
- 2x production load stays below the existing Code deadline;
- all hostile corpus containment checks pass;
- control no longer needs privileged mode, runsc, its bundle, or `/dev/kvm`.

Keep one-switch rollback to local gVisor during the initial observation period,
but do not leave wallet-bearing local hostile execution as the permanent
fallback.

### 6. Qualify Firecracker behind the same agent

- build signed immutable guest kernel/rootfs artifacts;
- use the Firecracker jailer, unique UID/GID, cgroups, seccomp and no swap;
- give each job one vCPU, bounded RAM and disposable scratch initially;
- use no guest NIC and communicate over vsock;
- allow a pool of clean prebooted guests;
- destroy every guest after hostile execution;
- never snapshot or reuse a guest that ran miner code.

Compare gVisor systrap/KVM and Firecracker cold/preboot modes on parity,
startup, execution overhead, jobs/second, memory/job and hostile containment.
Switch the execution-host backend only if Firecracker meets both the security
and capacity gates. No validator protocol change is involved.

### 7. Move validator control only after remote proof exists

Once both CPU execution and proof execution use qualified remote contracts,
deploy `ctrl-01`, shadow its public state, perform a controlled validator
cutover, and then remove GPU/runtime dependencies from the control image.

## Purchase gate

Do not treat the AX162-class choice as proven merely because Reliquary has 64
warm workers. Approve the permanent high-core host if the 48-physical-core
candidate passes 2x production plus malicious-full-timeout load with:

- no admission deadline breach caused by executor capacity;
- acceptable p95/p99 queue and complete Code admission latency;
- no swap and safe memory/disk high-water marks;
- bounded startup/recycle behavior;
- acceptable performance with the chosen SMT isolation policy.

If it fails CPU saturation with other resources healthy, test a 64-physical-
core class next. If it passes with substantial headroom, keep the 48-core host;
extra idle cores do not improve the protocol.

## Definition of done

- miner wire behavior and rewards are unchanged;
- the public validator is no longer privileged for Code execution;
- `cpu-exec-01` contains no valuable credential or trusted expected value;
- compromise of a guest cannot reach the host network or sibling job;
- compromise of the execution host cannot reach the validator or signer;
- executor outage is an infrastructure failure, never a false negative;
- 2x-production capacity and malicious-resource tests pass;
- the backend can change from gVisor to Firecracker without changing validator
  scheduling, scoring or miner protocol code.
