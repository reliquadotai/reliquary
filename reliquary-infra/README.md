# Reliquary validator infrastructure

- Status: deployed non-authoritative `ctrl-01` foundation plus a complete,
  not-yet-provisioned `cpu-exec-01` deployment package
- Evidence dates: 2026-08-21, 2026-08-25, and 2026-08-28
- Production image revision inspected:
  ab6478f1657b9134d628c93e8816a14651910146
- Implementation status: the local proof-process split, detached trainer,
  pipelined collection, and R2 checkpoint intake are live. The remote hostile
  CPU executor contract and mTLS agent exist on the prototype branch. A
  hardened shadow `ctrl-01` now serves immutable state, health, checkpoint, and
  runtime-contract snapshots on loopback and has passed 2x target load. Remote
  proof workers, a provisioned CPU execution host, scoped GPU workload
  identity, provider adapters, signer separation, off-host backups, public
  DNS/TLS, and production cutover remain gated.

The `cpu-exec-01` package now includes a pinned/hash-verified gVisor/KVM image,
exact host-capacity admission, one-use sandboxes, mTLS PKI tooling, an immutable
artifact builder/verifier, a deny-by-default Ansible host playbook, malicious
corpus and load gates, and a complete host validator. The image can be built
and stored on `ctrl-01`, but hostile execution must wait for the separate KVM
machine.

## 2026-08-25 implementation correction

Rom's PRs #189 and #191 through #193 materially changed the starting point of
this plan. Production now runs with pipelined windows, R2 training payloads, a
detached trainer, and an isolated proof process enabled. Across 47 live windows
after the latest deploy, mean seal-to-seal time was 104.8 seconds (34.3
windows/hour), versus the earlier 208-second p50 GPU half (17.3 windows/hour).
The original 1.7x-to-2.0x opportunity has therefore largely been realized.

Do not rebuild the detached trainer or local proof scheduler. The immediate
next change is a correctness fix: detached checkpoint titles are recognized by
HF discovery but rejected by the second resume parser. On the inspected
restart, the validator resumed checkpoint 572 while HF already contained
checkpoint 618, served three old-revision windows, and only then caught up via
R2 intake. The complete review and revised sequence are in
[Rom optimization review and revised next step](06-rom-optimizations-review.md).

## Executive decision

The research has the right security direction, but it models Reliquary too much
like a generic hostile-code job platform. The live validator is a tightly
ordered three-stage protocol:

1. collect and grade miner submissions;
2. verify GRAIL proofs against one pinned published checkpoint;
3. train one canonical model state and periodically publish the next checkpoint.

The original production bottleneck was not the CPU sandbox. Proof verification
and training serialized on the same H100, followed by a long checkpoint
publication path. The detached trainer and proof-process isolation now remove
most of that serialization. The remaining measured performance issues are the
serial checkpoint-swap beat and public API/admission pressure. Public state
polling still couples high-volume miner reads to the Python control process.

The remaining recommended v1 therefore has four deliberate changes:

1. Serve immutable miner state snapshots outside the validator process.
2. Keep Rom's existing deterministic scheduler and local proof-process adapter,
   but place a narrow executor contract in front of it before adding remote,
   checkpoint-pinned proof workers.
3. Harden the existing detached trainer into one warm, replaceable GPU lease,
   with scoped credentials, a real fenced owner, and an explicit recovery-state
   policy.
4. Remove the hotkey and privileged Code sandbox runtime from the public
   validator/trainer process while retaining trusted result comparison.

This produces modular and disposable GPU capacity without pretending that a
stateful trainer can be created for every window or that an ordinary marketplace
GPU can be trusted merely because its response has a hash.

## Recommended v1 shape

~~~text
miners
  |
  v
edge/state snapshot server --------------------+
  | dynamic writes only                        |
  v                                            |
validator control                              |
  - current window FSM                         |
  - admission and auction ranking              |
  - existing GlobalProofScheduler coordinator  |
  - durable local job/lease journal             |
  - archive queue and R2 integration            |
  |                  |                 |        |
  |                  |                 |        |
  v                  v                 v        |
CPU grader host   proof GPU pool    trainer GPU |
gVisor first      warm replicas     one active  |
Firecracker      pinned revision   fenced lease |
as measured      replaceable       replaceable  |
  |                  |                 |        |
  +------------------+-----------------+        |
                         |                       |
                         v                       |
                 checkpoint staging ------------+
                         |
                         v
               async HF publisher
                         |
                         v
                  narrow signer
                  hotkey only
                         |
                         v
                     Bittensor
~~~

R2 remains the artifact and archive store already used by Reliquary. The first
version does not need Kubernetes, Kafka, Temporal, a service mesh, or a second
general-purpose queue.

## What is adopted, adapted, and deferred

| Research proposal | Decision for Reliquary |
|---|---|
| Tiny trusted core and separate signer | Adopt. The live hotkey is readable by a privileged public container today. |
| Disposable GPU provider abstraction | Adopt, but separate provider lifecycle from the worker protocol. |
| Zero long-lived secrets on GPU workers | Adopt without exception. |
| Firecracker for all hostile CPU work | Adapt. First move the existing network-disabled gVisor worker pool to an untrusted host; keep comparison on control and benchmark Firecracker behind the same executor contract. |
| Training, verification, and proof pools | Correct the roles. Reliquary needs proof-verification workers and one canonical trainer; it does not generate miner proofs. |
| Postgres plus NATS JetStream immediately | Defer. A single control writer can use a transactional SQLite/WAL job and lease journal plus direct authenticated worker channels. |
| Trust remote results after schema/hash checks | Reject. Hashes bind bytes, not correct GPU computation. Production authority requires a dedicated trusted worker or an attested CVM. |
| Per-job disposable trainer | Reject. Model, optimizer, behavior model, and scheduler state must remain warm and sequential across steps. Make the lease replaceable, not cold per job. |
| Kubernetes, Temporal, Vault, Kafka | Defer until a measured requirement appears. |

## Original quantified opportunity and achieved result

Across 24 completed production windows, 30025 through 30048:

| Stage | Mean | p50 | p95 | Maximum |
|---|---:|---:|---:|---:|
| Proof wall time on the single H100 | 115.0 s | 105.1 s | 156.2 s | 160.8 s |
| Training and post-proof overhead | 95.5 s | 96.8 s | 109.7 s | 117.5 s |
| Complete sealed-window GPU half | 210.4 s | 208.0 s | 266.2 s | 271.0 s |

The two large stages were almost equal and added together. Rom's detached
trainer and isolated proof process now make the pipeline approach the maximum
of collection, proof, and training rather than their sum. The live result is
34.3 windows/hour across the first 46 consecutive intervals, with ordinary
intervals averaging 102.4 seconds. This is approximately 1.98x the earlier
17.3-window/hour p50 baseline, meeting and slightly exceeding the original
30-to-34-window/hour target in this sample.

Every 16 successful steps, the observed checkpoint-523 publication took about
234 seconds after training and forced a serial boundary. Separating snapshot
and upload removes that work from the public API failure domain and lets its
sub-stages be optimized independently. The first safe version still treats
publication as a protocol barrier: it does not open another trainable window
against the old behavior checkpoint. Hiding that barrier would change training
semantics and is a later experiment, not an infrastructure assumption.

The state endpoint is a separate immediate win. In one 22 minute 52 second
sample, it returned 421,843 successful responses, about 307 successful state
polls per second and 34.39 GB of response data. The current 86,807-byte payload
compresses to 36,499 bytes with gzip level 1, a 58% reduction. A static snapshot
server also prevents GPU or checkpoint stalls from turning into state endpoint
failures.

## Documents

- [Current architecture and measured bottlenecks](01-current-architecture-and-bottlenecks.md)
- [Adapted target architecture and trust model](02-target-architecture.md)
- [Capacity model and machine-selection plan](03-capacity-and-machine-selection.md)
- [Implementation, validation, and cutover plan](04-implementation-and-validation-plan.md)
- [Decision register, risks, and open questions](05-decisions-risks-and-sources.md)
- [Rom optimization review and revised next step](06-rom-optimizations-review.md)
- [CPU control and hostile-execution split](07-cpu-control-execution-split.md)
- [ctrl-01 deployment, qualification, and remaining gates](08-ctrl-01-deployment-report.md)
- [cpu-exec-01 package, controls, and exact deployment gate](09-cpu-exec-01-readiness.md)

## Non-negotiable invariants

- Miner wire semantics, auction ordering, trained-only rewards, and checkpoint
  pinning do not change as part of the infrastructure split.
- The coldkey remains offline.
- No GPU or CPU execution worker receives a Bittensor key, provider master key,
  R2 master credential, HF write token, SSH private key, or database credential.
- Only one trainer lease may commit a given training step.
- A checkpoint becomes public only after its artifact, profile, revision, and
  signature are coherent.
- A worker result never becomes authoritative merely because it is signed by
  that worker.
- The existing validator remains authoritative throughout shadow testing.
- No production machine is used as an experimentation box.

## External facts checked

- Firecracker recommends the jailer and explicit cgroup/resource controls for
  arbitrary-code multi-tenant hosts:
  https://github.com/firecracker-microvm/firecracker/blob/main/docs/prod-host-setup.md
- gVisor documents systrap as the VM-friendly default and KVM as the
  bare-metal-oriented platform:
  https://gvisor.dev/docs/architecture_guide/platforms/
- Lium explicitly says non-CVM providers can inspect and modify running pods,
  while its CVM design uses TDX measurements and attestation:
  https://docs.lium.io/pod-users/security
  and https://docs.lium.io/providers/nodes/cvm
- R2 supports single-object presigned operations and path-scoped temporary
  credentials, so workers do not need bucket master credentials:
  https://developers.cloudflare.com/r2/api/s3/presigned-urls/
  and https://developers.cloudflare.com/r2/api/s3/temporary-credentials/
- Bittensor weight setting is a hotkey operation. The online signer needs the
  hotkey, not the coldkey:
  https://www.bittensor.com/docs/sdk
