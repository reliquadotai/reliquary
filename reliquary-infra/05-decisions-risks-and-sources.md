# Decision register, risks, and sources

## 1. Research-to-Reliquary decision register

| Research idea | Decision | Reliquary-specific reason |
|---|---|---|
| Trusted control is different from execution located in Germany | Adopt | The current privileged container combines public API, hostile-code supervisor, GPU, cloud tokens, and hotkey. Physical location does not fix that blast radius. |
| Separate hotkey signer | Adopt | CheckpointStore signs locally and WeightOnlyValidator submits with the same mounted wallet today. Both can move behind semantic operations. |
| Coldkey offline | Adopt | No online component in the target requires the coldkey. |
| Firecracker direct on dedicated bare metal | Adapt | It is a strong target boundary, but the only current hostile CPU path is a short Code grader already under network-disabled gVisor. Move that grader first, then benchmark Firecracker behind the same contract. |
| AX102 control and AX162 CPU are predetermined | Reject as a decision | These are candidates. Control load changes materially after state, GPU, and grader separation. Current pricing also differs from the research snapshot. |
| NATS JetStream plus Postgres | Defer | There is one authoritative scheduler and one writer. SQLite/WAL plus authenticated direct worker channels provides uniqueness, leases, and restart state with less operating surface. |
| Kafka/Rabbit/Celery/Temporal/Kubernetes | Defer | The live flow is ordered and bounded, not a general DAG or multi-team cluster. |
| Deterministic job IDs | Adopt | They are required for remote retries, duplicate result rejection, and trainer fencing. |
| Large artifacts in object storage | Adopt | R2 is already integrated and supports presigned/scoped access. |
| GPU provider interface | Adopt | Lium lifecycle calls must not leak into proof, training, or validator code. |
| Provider interface is the worker protocol | Reject | Provisioning/termination and execute/heartbeat are separate concerns and should remain separate interfaces. |
| Per-job short-lived GPU identity | Adopt | Workers get exact-object access and short-lived certificates, never bucket/provider/HF credentials. |
| Every GPU is fully disposable per job | Adapt | Proof workers are stateless but need to stay warm by revision. The trainer is one replaceable warm lease with durable state and replay. |
| Separate training, verification, and proof-generation pools | Correct | Reliquary needs a proof-verification pool and canonical trainer. Miners generate proofs; the validator has no proof-generation pool. |
| One universal GPU server | Reject as target | It is the measured source of proof-plus-train serialization. |
| Worker result accepted after schema/hash checks | Reject | A malicious provider can produce internally consistent false bytes. Proof and training correctness require trusted/dedicated or attested execution, or independent recomputation. |
| CVM makes a marketplace worker eligible for trust | Conditional | Only after Reliquary itself verifies a nonce-bound CPU/GPU attestation that binds the exact image and ephemeral key. A UI label is insufficient. |
| No network in CPU sandbox | Adopt | The current gVisor path already uses network=none. Firecracker should use no NIC and vsock. |
| Per-job new Firecracker VM | Adopt if Firecracker wins | One grading request per VM, with clean preboot allowed. Never recycle a post-hostile VM. |
| Three fixed GPU pools | Reject | Scale logical proof and trainer roles; do not create unused pools by name. |
| Full Prometheus/Grafana/Loki/OTel stack immediately | Adapt | Emit the required metrics first and connect them to the smallest existing monitoring stack. Add components only where an operator consumes them. |
| OpenTofu/Terraform plus Ansible | Adopt for long-lived hosts | Keep modules small and pin images. Provider-marketplace leases may be runtime orchestration rather than permanent Terraform resources. |
| Shadow production before cutover | Adopt | It is mandatory for proof parity, training quality, checkpoint coherence, and cost. |

## 2. Key corrections to the research

### 2.1 CPU isolation is not the speed win

The live Code grader kernel is usually about 0.2 to 0.5 seconds. Its queue tail
is material, but proof and training consume roughly 105 and 97 seconds per
window and serialize on the H100. Firecracker improves containment. It should
not be included in the GPU speedup estimate.

### 2.2 A result checker cannot prove expensive computation from metadata

Checking job ID, input hash, image digest, schema, runtime, and output hash is
necessary. It prevents mixups, replay, and accidental corruption. It does not
show that:

- a proof worker used the claimed model;
- the hidden-state verification was honestly computed;
- a trainer applied the requested gradients;
- a CPU supervisor honestly reported test results.

The target therefore distinguishes transport authenticity from compute
integrity. Ordinary marketplace GPU results remain shadow/speculative.

### 2.3 The trainer is replaceable, not horizontally replicated

Training state has one canonical order. Two independent trainers applying
different windows create two models, not extra throughput. Horizontal capacity
belongs in proof verification. Trainer improvement comes from a faster device,
kernel/data optimization, or carefully designed data parallelism, the last of
which is a separate training change.

### 2.4 Checkpoint publication is a protocol barrier in minimal v1

The current serial beat protects checkpoint pinning, trained-only rewards, and
behavior-policy refresh. Continuing old-revision trainable windows while HF
publication runs would require a defined way to train them after activation
without violating those invariants. Minimal v1 keeps the barrier, isolates it
from public reads, instruments it, and makes it faster. Hiding it is a later
protocol/training experiment.

### 2.5 Durable jobs do not require a distributed queue yet

Reliquary has:

- one window authority;
- a bounded number of proof devices;
- one trainer writer;
- one checkpoint lineage.

A transactional local journal and direct worker channels solve current retry
and lease requirements. NATS becomes justified if control needs many
independently scaling consumers or durable disconnected delivery beyond this
bounded coordinator. Postgres becomes justified when more than one control
writer or control HA is required.

## 3. Re-evaluation triggers

| Deferred component | Add it when |
|---|---|
| Postgres | Active/passive control failover needs shared transactional state, multiple schedulers write concurrently, or the SQLite recovery/SLO fails. |
| NATS JetStream | Worker fan-out or disconnected buffering cannot be handled by the bounded control coordinator, and direct-channel backpressure has been measured as limiting. |
| Temporal | Training becomes a multi-hour/multi-provider branching workflow with checkpoint, preemption, compensation, and human approval stages. |
| Kubernetes/Kata | The team operates a sufficiently large persistent fleet that scheduling, rolling upgrade, and bin-packing benefits exceed its control-plane cost. |
| Vault or equivalent | The number/rotation model of service credentials exceeds OS-level separation and provider-native scoped credentials. |
| 10 Gbit/s | A named artifact or ingress stage saturates 1 Gbit/s after gzip/cache and concurrent-transfer measurements. |
| More than two proof workers | Proof p95 remains above the 90-second capacity gate or cost/failure diversity justifies it. |
| Trainer data parallelism | One faster qualified GPU cannot meet the target and a separate algorithmic experiment proves exact training semantics. |
| Dual-revision publication pipeline | Old-revision batch training and behavior-policy correctness are formally specified and pass replay/quality tests. |

## 4. Risk register

| Risk | Impact | Mitigation and gate |
|---|---|---|
| Non-CVM provider falsifies proof | Incorrect winners/rewards | Never authoritative; local/dedicated/attested recomputation gate |
| Non-CVM provider tampers with training | Poisoned checkpoint | Never authoritative; require trusted or attested trainer |
| CVM label without usable attestation | False sense of trust | Verify nonce, TDX quote, GPU claims, image digest, and ephemeral key before credential release |
| Trainer split brain | Divergent or duplicate steps | Fenced lease generation and transactional parent-state compare-and-swap |
| Training replay is nondeterministic | Recovery produces a different candidate | Same runtime/hardware where needed, exact RNG/state, tensor tolerance and quality gate; choose snapshot interval from results |
| Full trainer state is too large | Slow/costly recovery | Durable publication anchors, immutable batch log, measured mid-cadence snapshots, provider volume only as a cache |
| Proof extraction changes side effects | Reward divergence | Pure kernel/local apply split with golden archive/debt parity tests |
| Remote network tail exceeds GPU gain | Proof deadlines/backlog | Local split ceiling, regional tests, persistent channels, bounded payloads, minimum warm pool |
| GPU cold start is minutes | Capacity unavailable at seal | Prewarm by revision, provider cache, readiness qualification, do not provision per proof |
| Checkpoint upload remains 234 seconds | Periodic throughput gap | Stage timing, dedicated NVMe/network publisher, resumable HF path; keep barrier explicit in capacity model |
| State snapshot is stale or mixed | Miners submit wrong window/revision | Atomic bytes, ETag, propagation SLO, transition tests, validator remains sole producer |
| CPU execution host is compromised | False grades and lost capacity | No secrets/routes, canary jobs, random replay on second backend, quarantine host |
| CPU guest escapes | Execution host loss | Firecracker/gVisor hardening, no secrets/LAN, rebuild host, malicious corpus |
| Signer unavailable | No checkpoint signature or weights | Fail closed, durable pending request, standby recovery domain, no key fallback to public host |
| Signer accepts arbitrary operation | Hotkey abuse | Semantic endpoints, fixed netuid/repo, monotonic state, replay/rate limits |
| Provider API key stolen | Spend/deletion incident | Separate orchestrator, no public ingress, no worker exposure, cost/quota alerts, rotation |
| R2 outage | Archive/job artifact delay | Existing disk outbox, bounded local spool, no new window if required input cannot be made durable |
| SQLite loss/corruption | Orchestration ambiguity | RAID1, transactional WAL, frequent backup/checkpoint, deterministic rebuild from R2/HF and job artifacts; Postgres trigger if SLO fails |
| Cost runaway/orphan instances | Financial loss | Provider reconciliation, TTL, auto-termination, invoice comparison, hard budget alerts |
| New infrastructure masks quality regression | Bad model despite healthy systems | Existing numerical gates plus paired Math/Code checkpoint promotion benchmark |
| Control RCE reaches neighboring process credentials | Cloud/provider compromise | Separate Unix users, no shared env/socket except narrow APIs, rootless processes; signer remains off-host |

## 5. Open decisions after benchmarks

These are decisions to make from the planned evidence, not questions blocking
the present analysis:

1. What compute-integrity policy is acceptable: only dedicated operator-owned
   GPU, attested CVM, or a named trusted cloud provider?
2. Is the model/checkpoint public enough that confidentiality is irrelevant, or
   must CVM be required for model IP as well as integrity?
3. What trainer recovery time is acceptable, and therefore how often should a
   full optimizer/RNG state anchor be written?
4. Does proof capacity need to meet a strict 100-second window p95, or is a
   bounded one-window backlog acceptable?
5. Is one control host with restore sufficient, or is active/passive control HA
   a v1 requirement?
6. Must Firecracker be the first production CPU backend, or may isolated gVisor
   ship first while Firecracker finishes qualification?
7. Which account/provider hosts the signer so a Hetzner control-account breach
   does not include it?
8. What maximum checkpoint publication barrier is economically acceptable?
9. Is a two-proof-worker minimum worth the warm-pool cost, or does one faster
   qualified proof GPU pass the p95 gate?

## 6. Recommended minimal production bill of materials

Subject to the benchmark gates:

- one control host, 16 fast physical cores, 64-to-128 GB ECC, RAID1 NVMe;
- one CPU execution host sized by the gVisor/Firecracker 2x-load benchmark;
- one small isolated signer domain;
- one warm H100-class trainer lease;
- two logical qualified proof slots, on trusted/dedicated or attested workers;
- existing R2 and HF services;
- no general queue cluster and no Kubernetes control plane.

The proof slots are the elastic portion. Keep the minimum that meets the p95
gate, add capacity before a window only when provisioning/model-load lead time
allows, and terminate drained excess workers. The trainer is movable between
providers but remains one warm active lease.

## 7. Evidence confidence

| Claim | Confidence | Basis |
|---|---|---|
| Current component/process topology | High | Exact live image source plus docker/process inspection |
| Proof/training serialization | High | Scheduler configuration, code path, GPU telemetry, and R2 timing |
| Current stage timings | High for sampled interval | 24 joined live windows; 80-window archive check corroborates tails |
| State polling pressure | High for sampled interval | 500,000 nginx records and container network counters |
| 1.7x-to-2.0x speed opportunity | Medium | Measured stage times and queueing lower bound; remote overhead not yet benchmarked |
| Two proof workers meet 90-second p95 | Hypothesis | Device-time lower bound; must be proven under rank/resource constraints |
| Firecracker improves security | High | VM boundary and official production guidance |
| Firecracker effect on Reliquary speed | Unknown | Requires the specified benchmark |
| Lium CVM can be authoritative | Conditional | Platform claims exist; Reliquary must verify renter-visible end-to-end attestation |
| Exact machine and monthly cost | Unknown by design | Availability and prices are dynamic; benchmark and quote later |

## 8. Source register

### Reliquary ground truth

- Live health and state endpoints, read on 2026-08-21.
- Live docker inspect, process list, resource, GPU, disk, nginx config/log, and
  application log inspection, read-only.
- R2 archives for recent completed windows, read through the live service's
  existing credentials without exposing values.
- Git revision a083e5d8b896878cbc2f46579abe030fe13a7606, especially:
  - reliquary/validator/service.py
  - reliquary/validator/proof_scheduler.py
  - reliquary/validator/batcher.py
  - reliquary/validator/server.py
  - reliquary/validator/training.py
  - reliquary/validator/checkpoint.py
  - reliquary/validator/weight_only.py
  - reliquary/environment/grader_client.py
  - reliquary/environment/grader/server.py
  - reliquary/infrastructure/archive_queue.py
  - docker/docker-compose.trainer.yml
  - docker/entrypoint.sh

### Primary external references

- Firecracker design and threat containment:
  https://github.com/firecracker-microvm/firecracker/blob/main/docs/design.md
- Firecracker production-host and jailer guidance:
  https://github.com/firecracker-microvm/firecracker/blob/main/docs/prod-host-setup.md
  and https://github.com/firecracker-microvm/firecracker/blob/main/docs/jailer.md
- Firecracker snapshot security boundary:
  https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/snapshot-support.md
- gVisor platform choices:
  https://gvisor.dev/docs/architecture_guide/platforms/
- Lium non-CVM threat model:
  https://docs.lium.io/pod-users/security
- Lium CVM/TDX model:
  https://docs.lium.io/providers/nodes/cvm
- Lium API-key privilege warning:
  https://docs.lium.io/pod-users/api-keys
- Lium SDK/lifecycle surface:
  https://docs.lium.io/developers/sdk
- Cloudflare R2 presigned URLs and temporary credentials:
  https://developers.cloudflare.com/r2/api/s3/presigned-urls/
  and https://developers.cloudflare.com/r2/api/s3/temporary-credentials/
- NVIDIA attestation overview and claims:
  https://docs.nvidia.com/attestation/index.html
  and
  https://docs.nvidia.com/attestation/advanced-documentation/latest/claims-guide/gpu_claims.html
- Bittensor wallet operation roles and SDK:
  https://www.bittensor.com/docs/sdk
- Hugging Face large/resumable upload and preupload/commit options:
  https://huggingface.co/docs/huggingface_hub/en/guides/upload
- Hetzner current AX family specifications:
  https://www.hetzner.com/dedicated-rootserver/ax102/

## 9. Final architecture recommendation

Build toward:

~~~text
static state edge
+ wallet-free validator control
+ existing deterministic scheduler
+ isolated CPU grader
+ warm elastic proof pool
+ one fenced replaceable trainer
+ explicit checkpoint barrier/publisher
+ semantic signer
+ R2-scoped artifact flow
~~~

Do not start by installing a distributed platform. Start by separating the two
measured GPU stages and the two dangerous credentials/privilege domains, then
let the benchmark identify the next bottleneck.
