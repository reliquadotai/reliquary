# Remote CPU sandbox prototype

## Decision

Reliquary should use two CPU trust domains:

```text
miners
  |
  v
validator/control + trusted comparison
  |
  | HTTPS + mTLS, private address, code/args only
  v
zero-secret CPU executor
  |
  v
one disposable gVisor sandbox per candidate batch
```

The validator/control host receives requests, owns the dataset, expected
values, reward comparison, validator state, and wallet. The execution host is
a replaceable machine that owns only CPU capacity, the pinned grader image,
and its server TLS identity.

This is the right first boundary for the current repository. It reuses the
existing worker, OCI policy, gVisor lifecycle, client retry, metrics, and
grading semantics. It does not add Kubernetes, a queue, a database, or a new
workflow engine.

Firecracker can replace gVisor behind the versioned `SandboxExecutor` contract
later without changing admission or scoring. A Firecracker migration should
be based on measured gVisor cold-start and saturation data from this split,
not introduced into the first network/provisioning experiment.

## What crosses the boundary

The remote request contains:

- protocol and runtime identity;
- deterministic content-bound job ID and attempt;
- miner code and its SHA-256;
- function/method entry point, arguments, keyword arguments;
- per-case timeout.

The contract also carries a content-bound overall batch deadline, computed
from case count and per-case timeout and capped at 120 seconds. This bounds
work that remains after a client disconnect and prevents a large set of
individually fast cases from becoming an unbounded executor request.

It cannot contain expected values or comparison rules: the strict protocol
rejects extra fields. The response contains only bound case IDs, statuses,
JSON-safe bounded outputs, executor identity, and elapsed time. The trusted
coordinator rejects a result with the wrong job, attempt, runtime, or case
sequence before comparing it locally.

The CPU host must never receive wallets, Hugging Face/R2 credentials, database
credentials, provider credentials, SSH keys for other machines, datasets, or
expected answers.

This boundary protects control-plane credentials and blast radius; it does not
make a compromised execution host cryptographically honest. In authoritative
mode the owned CPU host remains trusted for result integrity and availability.
mTLS authenticates that host but is not remote attestation. Making arbitrary
marketplace CPU pods fully untrusted requires a later redundant/random recheck
policy across independent executors (or a verifiable execution mechanism).
Schema, digest, and local expected-value checks alone cannot detect a host that
deliberately fabricates a plausible output. Disposable providers must not be
enabled as authoritative until that policy exists.

## Rollout modes

`RELIQUARY_GRADER_EXECUTOR_MODE=shadow` is the default whenever a remote URL
is configured. Local gVisor remains authoritative. A bounded background task
mirrors the exact execution request to the new host and records match,
mismatch, failure, and dropped counts. Shadow work cannot alter a score and
does not extend live request latency.

`RELIQUARY_GRADER_EXECUTOR_MODE=remote` makes the CPU host authoritative for
execution only. Expected-value comparison still remains on the validator.
This mode stops local runsc workers, allowing the validator container to drop
`privileged` and host-cgroup access.

Rollback is one environment change:

- set mode back to `shadow` to compare while restoring local authority; or
- unset `RELIQUARY_GRADER_EXECUTOR_URL` to return to the original local path.

## Sandbox lifecycle

The remote agent starts a warm capacity pool, but a sandbox is assigned to
only one candidate batch. After the ordered cases for that candidate finish,
the agent destroys and replaces the gVisor container. No interpreter or guest
state is handed to another miner. The immutable root filesystem is read-only,
the sandbox has no network, runs as UID/GID 65534, has no capabilities, and
retains the existing CPU, process, file, and address-space limits.

`RELIQUARY_CPU_EXECUTOR_REUSE_WORKERS=1` exists only to benchmark the old warm
reuse behavior in an isolated lab. It must remain `0` in production.

The outer service also bounds code, request, response, case count, timeout,
and worker-output sizes. Admission is bounded before a job takes a worker.
The API owns a fixed thread pool equal to `MAX_INFLIGHT`; it does not inherit
AnyIO's smaller generic blocking-thread limit. A burst beyond that exact bound
is rejected immediately with `503` and a `busy` metric instead of accumulating
an unbounded in-process queue.

## Capacity model

The active profile has 16 rollouts per candidate. Four Code admission workers
can therefore request as many as 64 sandbox batches concurrently. That is why
the current local pool is `4 * M_ROLLOUTS = 64`.

A machine advertised as “96 CPU” is commonly 48 physical cores / 96 hardware
threads. It is a sensible production candidate, but 96 threads should not be
treated as 96 independent, fully saturated Python cores. Choose the final
machine from these measurements:

- queue wait and remote p50/p95 latency under representative submissions;
- sandbox replacement/cold-start time;
- jobs/second and timeout rate at pool sizes 16, 32, 48, and 64;
- physical-core utilization, memory, runsc/gofer process count, and load;
- live Code admission wall-budget headroom.

Start the first rented prototype with 8–16 physical cores, at least 32 GB RAM,
local SSD, and pool size 16. It is enough for shadow correctness and lifecycle
testing, but it is not expected to absorb the theoretical 64-job production
burst. A 48-core/96-thread dedicated host is a reasonable next benchmark if
the prototype confirms CPU saturation and queueing rather than network or
sandbox startup as the bottleneck.

Do not increase `RELIQUARY_CODE_ADMISSION_WORKERS`, the rollout count, or the
remote pool in the same experiment. Change one capacity boundary at a time.
Keep `RELIQUARY_CPU_EXECUTOR_MAX_INFLIGHT` equal to the worker pool initially;
this creates backpressure before work queues behind disposable sandboxes and
exceeds the validator's existing wall budget.

## Build and deployment package

The code is ready when all of the following pass on the branch:

1. execution contract, transport, service, shadow, disposal, and legacy grader
   tests;
2. a real TLS/mTLS loopback smoke test;
3. CPU-executor image build and non-hostile image inspection on Linux;
4. artifact digest, Ansible syntax, shell, and Compose validation;
5. no wallet/GPU/storage packages or credentials in the CPU image/environment.

`scripts/build_cpu_executor_artifact.sh` creates an immutable linux/amd64 image
archive, manifest, package inventory, Docker inspection, history, runsc
version, and evidence checksums. Runtime Python packages are locked with exact
artifact hashes. gVisor is pinned to dated release `20260817`, and the committed
SHA-512 is checked independently of the download location.

The Linux build does not qualify the actual gVisor boundary. That gate requires
the dedicated KVM host, because miner code must never be attack-tested on
`ctrl-01`.

## Host and network prerequisites

Use a dedicated Linux host with Docker, KVM, and a private address reachable
only from the validator/control host. Bind port 8443 to that exact address.
The container uses host networking so Docker cannot create a forwarding rule
that bypasses the host firewall. Apply both provider firewall and host rules:

```text
validator private IP -> cpu-exec private IP:8443 allow
management VPN       -> cpu-exec SSH          allow
everything else      -> cpu-exec              deny
cpu-exec              -> trusted LAN          deny
```

The sandbox itself has no network namespace connectivity. mTLS remains
mandatory even on the private link. Do not expose 8443 on a public address.

## Deployment procedure

Use the exact same Git commit on both hosts and put that commit in
`RELIQUARY_BUILD_REVISION`.

Generate the private CA and leaf identities on an operator workstation:

```bash
scripts/generate_cpu_executor_pki.sh /secure/reliquary-cpu-pki 10.81.20.2
```

Keep `/secure/reliquary-cpu-pki/ca/ca.key` offline. Copy only
`cpu-executor/` to the execution host and only `grader-client/` to the
validator host.

Build on trusted `ctrl-01` without starting the executor:

```bash
sudo scripts/build_cpu_executor_artifact.sh \
  /var/lib/reliquary-build/cpu-executor/<git-revision> \
  <git-revision>
sudo scripts/verify_cpu_executor_artifact.sh \
  /var/lib/reliquary-build/cpu-executor/<git-revision>
```

Copy `reliquary-infra/inventory/cpu-exec.example.yml` outside Git, fill it
from the artifact manifest and PKI output, then provision the new host:

```bash
ansible-playbook \
  -i /secure/cpu-exec-01.yml \
  reliquary-infra/playbooks/cpu-exec-01.yml
```

The playbook refuses missing KVM, swap, wallets, an uploaded CA private key,
certificate/address mismatch, a corrupt image, or an image-ID mismatch. It
installs only the server leaf, disables Docker bridge/NAT/firewall mutation,
denies new host connections to trusted/private networks, exposes metrics only
on loopback, and runs a final host validator.

From the validator host, prove server identity and client-certificate
enforcement:

```bash
curl --fail --silent --show-error \
  --cacert /etc/reliquary/grader-client-pki/ca.crt \
  --cert /etc/reliquary/grader-client-pki/client.crt \
  --key /etc/reliquary/grader-client-pki/client.key \
  https://10.81.20.2:8443/v1/health
```

The same request without the client certificate must fail during TLS setup.
Run one content-bound sandbox execution from the validator checkout:

```bash
python scripts/smoke_remote_cpu_executor.py https://10.81.20.2:8443 \
  --ca /etc/reliquary/grader-client-pki/ca.crt \
  --cert /etc/reliquary/grader-client-pki/client.crt \
  --key /etc/reliquary/grader-client-pki/client.key
```

Then run the dedicated-host-only containment and capacity gates:

```bash
python scripts/attack_test_cpu_executor.py https://10.81.20.2:8443 \
  --ca /etc/reliquary/grader-client-pki/ca.crt \
  --cert /etc/reliquary/grader-client-pki/client.crt \
  --key /etc/reliquary/grader-client-pki/client.key \
  --confirm-dedicated-host

python scripts/load_test_cpu_executor.py https://10.81.20.2:8443 \
  --ca /etc/reliquary/grader-client-pki/ca.crt \
  --cert /etc/reliquary/grader-client-pki/client.crt \
  --key /etc/reliquary/grader-client-pki/client.key \
  --requests 10000 --parallel 16
```

On the validator, add:

```dotenv
RELIQUARY_GRADER_EXECUTOR_URL=https://10.81.20.2:8443
RELIQUARY_GRADER_EXECUTOR_MODE=shadow
RELIQUARY_GRADER_EXECUTOR_CA=/etc/reliquary/grader-client-pki/ca.crt
RELIQUARY_GRADER_EXECUTOR_CERT=/etc/reliquary/grader-client-pki/client.crt
RELIQUARY_GRADER_EXECUTOR_KEY=/etc/reliquary/grader-client-pki/client.key
RELIQUARY_GRADER_CLIENT_PKI_DIR=./pki/grader-client
```

Keep the validator privileged in shadow mode because local runsc remains the
authority. Recreate it with the existing trainer Compose file.

## Qualification gate

Do not switch authority merely because health is green. Shadow at least 10,000
candidate batches covering real archived submissions plus the malicious corpus.
The gate is:

- runtime IDs equal and remain pinned to the same build;
- zero result mismatches;
- zero malformed/binding failures;
- zero dropped shadow jobs under representative peak load;
- no host, cross-job, trusted-network, or credential access in attack tests;
- remote p95 leaves safe margin inside the existing admission wall budget;
- sandbox replacement remains healthy with no runsc/cgroup/process leak;
- forced agent loss creates infrastructure errors, never candidate-negative
  rewards or silently accepted results.

Relevant control-side metrics and health fields are:

```text
grader_shadow_requests_total{status="match|mismatch|error|dropped"}
execution_backend = local-shadow
shadow.matches_total
shadow.mismatches_total
shadow.failures_total
shadow.dropped_total
shadow.executor.success_latency_p50_ms
shadow.executor.success_latency_p95_ms
```

The CPU agent exposes `/v1/health` and `/metrics` on the same mTLS endpoint.
Its API metrics include exact in-flight, peak, maximum, success, overload, and
execution-error counts.

## Cutover

After the qualification gate passes:

```dotenv
RELIQUARY_GRADER_EXECUTOR_MODE=remote
RELIQUARY_GRADER_PRIVILEGED=false
RELIQUARY_CGROUP_MODE=private
```

Recreate the validator container; changing the Compose environment without a
recreate does not change the live security boundary. Confirm its process
health reports `execution_backend=remote`, `workers_alive=0`, and a healthy
remote executor. Then re-run loss, timeout, and rollback drills. Dropping the
validator process to UID 1000 is a separate hardening gate because the current
training/state/HF volumes use root-owned paths; do not combine that filesystem
migration with the CPU authority cutover.

## Scope and next boundary

This prototype separates hostile CPU Code execution. It does not yet turn the
current validator/trainer image into a pure CPU control node. The repository's
detached trainer path already moves optimizer/checkpoint work to another GPU
machine, and the proof-slot work improves verification parallelism, but the
GPU proof/verification plane still has validator-process coupling.

The next implementation should introduce a similarly narrow, versioned remote
proof executor before removing GPUs from the validator/control machine. Do not
fold that into this CPU cutover: qualify the CPU boundary first, then reuse its
contract/identity/shadow pattern for the GPU proof plane.
