# Adapted target architecture

## 1. Design objective

The target is not a generic distributed job platform. It is the smallest
architecture that preserves Reliquary's deterministic protocol while allowing
GPU capacity to be independently replaced, scaled, and purchased from more than
one provider.

The design separates six concerns:

1. public state distribution;
2. validator control and economic ordering;
3. hostile CPU execution;
4. proof-verification GPU execution;
5. canonical stateful training;
6. Bittensor signing and weight submission.

The most important performance property is stage overlap. The most important
security property is that compromise of any execution worker yields no key and
no route to the signer.

## 2. Trust model: use two axes, not one label

Calling every remote worker simply untrusted hides an important difference.
There are two independent questions:

- Can the component read or steal a valuable secret?
- Can its computation be accepted as correct without recomputation?

| Component | Long-lived secrets | Production result authority |
|---|---|---|
| Offline coldkey | Coldkey | Manual coldkey operations only |
| Signer | Validator hotkey | Only narrow checkpoint-sign and weight-submit operations |
| Validator control | R2/archive and control credentials, no wallet | Owns window, ranking, job, and checkpoint state |
| Provider orchestrator | Provider billing/lifecycle API key | May create and destroy workers; cannot score |
| HF publisher | HF repo write token | May stage a commit; cannot sign or activate it alone |
| Dedicated operator-controlled GPU | No control credentials | May be authoritative under the operator trust policy |
| Attested CVM GPU | Ephemeral scoped credentials only | May be authoritative only after end-to-end CPU and GPU attestation is verified |
| Ordinary non-CVM marketplace GPU | Ephemeral exact-object access only | Shadow/speculative only; never authoritative by itself |
| CPU execution supervisor | Short-lived worker identity only | Grading authority while healthy; canaries and replay detect a compromised host |
| Per-job CPU sandbox | None | Never trusted outside its returned bounded result |

An ordinary Lium pod is intentionally modeled as observable and modifiable by
its provider. Lium's own documentation says a non-CVM provider can inspect
memory, files, environment variables, and replace the container. A signed
response from that pod proves which ephemeral key sent the response; it does
not prove that the GPU ran the requested model.

For production proof or training authority, use one of:

1. a dedicated GPU under the accepted operator/provider trust model; or
2. a CVM whose TDX quote, GPU attestation, workload image digest, nonce, and
   ephemeral worker public key have all been verified before credentials are
   released.

Provider redundancy and random rechecks are useful detection layers, but they
do not turn unaudited non-CVM workers into cryptographically trusted compute.

## 3. Logical topology

~~~text
                         public Internet
                               |
                               v
                    +----------------------+
                    | edge/API gateway     |
                    |                      |
                    | static state bytes   |
                    | gzip + ETag          |
                    | dynamic write proxy  |
                    +----------+-----------+
                               |
                               v
                  +--------------------------+
                  | validator control        |
                  |                          |
                  | protocol/window FSM      |
                  | admission + auction      |
                  | GlobalProofScheduler     |
                  | job/lease journal        |
                  | result application       |
                  | R2 archive outbox        |
                  | no wallet, no GPU        |
                  +--+----------+---------+--+
                     |          |         |
          mTLS       |          |         | mTLS
                     |          |         |
                     v          v         v
             +----------+ +---------+ +----------------+
             | CPU exec | | proof   | | trainer lease  |
             | host     | | workers | |                |
             |          | | 1..N    | | exactly one    |
             | gVisor / | | warm by | | active writer  |
             | FC VMs   | | revision| | warm state     |
             +----------+ +---------+ +--------+-------+
                                                |
                                                v
                                    +----------------------+
                                    | immutable staging    |
                                    | R2 artifacts/state   |
                                    +----------+-----------+
                                               |
                                               v
                                     +-------------------+
                                     | HF publisher      |
                                     | async upload      |
                                     +---------+---------+
                                               |
                                               v
                                     +-------------------+
                                     | signer            |
                                     | hotkey only       |
                                     | semantic API      |
                                     +---------+---------+
                                               |
                                               v
                                           Bittensor

                +-------------------------------+
                | provider orchestrator         |
                | LiumProvider / ProviderB      |
                | dedicated adapter             |
                | lifecycle and cost only       |
                +-------------------------------+
~~~

The provider orchestrator and HF publisher can initially be separate rootless
processes on the control host under different Unix users. They are logical
security domains: the public validator process does not inherit their
credentials and cannot access a Docker socket. The signer is a separate host or
equivalent independent security domain.

## 4. Component responsibilities

### 4.1 Edge and state snapshot server

The validator remains the only producer of state, but it atomically emits the
already-serialized per-environment bytes to a directory or narrow local socket.
nginx or a tiny static process serves those bytes.

Required behavior:

- byte-identical JSON to the current state response;
- atomic replacement, never a partially written response;
- gzip and identity variants;
- ETag derived from the exact bytes;
- explicit no-store for dynamic write responses;
- state propagation lag measured and bounded;
- last-known closed/transition state handled explicitly, not synthesized by
  nginx;
- no caching of miner submissions;
- health remains separate from immutable protocol state.

This is not a new source of truth. It is a read replica of bytes that the
validator already caches.

### 4.2 Validator control

Control keeps the logic that must remain authoritative and ordered:

- window activation and checkpoint pinning;
- registration and freshness checks;
- admission queues and process-isolated parsing;
- auction ranking and deterministic tie-breaking;
- prompt/hotkey/operator proof constraints;
- GlobalProofScheduler coordination;
- proof-result validation and rank-ordered application;
- reward construction and R2 archive emission;
- checkpoint activation at a clean window boundary.

The control process has no CUDA context, no wallet, no privileged container,
and no provider master key.

A small transactional journal records only infrastructure facts:

- deterministic job ID and kind;
- payload and environment digests;
- checkpoint or parent-state revision;
- status and attempt;
- assigned worker and lease generation;
- deadline;
- result digest;
- checkpoint publication state;
- an outbox of state changes not yet delivered.

For one control writer, SQLite in WAL mode is sufficient and keeps the first
implementation small. Only the control service opens the database file. Move
the same schema to Postgres if active/passive control failover or multiple
schedulers becomes a real requirement.

The journal is not an attempt to serialize the live batcher object graph. If
control dies mid-window, preserve today's fail-closed model: fence all late
worker results, emit/recover an aborted-window tombstone where possible, restore
the last committed checkpoint/archive state, and open a fresh window. Durable
jobs prevent ambiguity and duplicate application; they do not make a partially
ranked economic window resumable.

### 4.3 CPU execution host

Only OpenCodeInstruct's untrusted candidate code needs this boundary today.
Prompt selection, request authentication, economic ranking, and Math grading
remain in control.

Keep the trusted Code coordinator and expected-value comparison in control as
well. Only the sandbox execution backend moves. The remote host receives code,
entrypoint, args/kwargs and limits, returns bounded primitive outputs/status,
and never receives expected values or comparison rules. This reuses the
current split inside `GraderServer` instead of promoting an execution host into
a scoring authority. The exact interface and cutover are specified in
[CPU control and hostile-execution split](07-cpu-control-execution-split.md).

Migration order:

1. Preserve GraderClient's fail-closed result semantics.
2. Keep GraderClient's Unix-socket hop and trusted comparator on control; put
   the sandbox call behind a `SandboxExecutor` interface.
3. Move the existing gVisor worker pool behind an authenticated bounded remote
   executor on cpu-exec-01.
4. Remove privileged mode and runsc dependencies from validator control.
5. Run the malicious corpus and capacity benchmark.
6. Add a Firecracker backend behind the same grader service and compare it to
   gVisor.

The initial move is intentionally gVisor-first because it is the current,
tested, network-disabled execution contract. It immediately fixes the
credential blast radius. Firecracker is the stronger VM boundary to adopt when
its real Reliquary startup, compatibility, and throughput numbers pass the
gate.

For Firecracker:

- dedicated bare metal with KVM;
- jailer, unique UID/GID, namespaces, cgroups, seccomp, and no swap;
- no guest network interface by default;
- read-only golden rootfs plus disposable scratch;
- vsock request/result channel;
- one evaluation request per VM lifecycle;
- a clean preboot pool is allowed;
- a VM is never reused after miner code;
- golden snapshots are operator-built, authenticated, immutable, and never
  derived from a guest that ran hostile code.

Queue admission and per-miner limits remain in control before any sandbox is
allocated.

### 4.4 Proof-verification GPU pool

Proof workers are stateless across jobs but warm across a checkpoint revision.
Each worker holds:

- one immutable OCI image digest;
- one published model revision;
- tokenizer and protocol profile digests;
- a short-lived workload identity;
- no wallet or general cloud credentials.

Disposable means:

- a worker can be drained and destroyed without losing canonical state;
- a failed worker's job is retried under the same deterministic ID;
- a new worker can load the pinned revision, qualify, and join the pool;
- old-revision workers may drain while new-revision workers prewarm.

It does not mean provisioning a new pod for each two-to-ten-second proof. Lium
documents roughly minute-scale deployment for pre-cached common templates even
before the Reliquary model is loaded. At a 100-second collection cadence, at
least one proof worker must normally remain warm.

GlobalProofScheduler remains the coordinator. Replace its injected local model
call with a ProofExecutor abstraction:

~~~text
ProofExecutor
  execute(ProofInvocation, ProofRequest) -> ProofKernelResult
  ready(checkpoint_revision, profile_digest) -> readiness
  drain(checkpoint_revision) -> status

LocalProofExecutor
RemoteProofExecutor
~~~

The scheduler's device IDs become logical worker slots rather than necessarily
local CUDA identifiers.

### 4.5 Canonical trainer GPU lease

Training is stateful and sequential. Exactly one worker owns the active
training lineage.

The trainer keeps warm:

- train model;
- optimizer and LR scheduler;
- AMP/scaler state if used;
- frozen behavior-policy model and its revision;
- fixed KL reference when configured;
- CPU and CUDA RNG state;
- protocol/training profile;
- current run ID, step, parent state digest, and lease generation.

The lease is fenced. Every train request includes a monotonically increasing
generation issued by control. A stale worker can finish computation but cannot
commit its result after a replacement has acquired a newer generation.

Trainer disposability comes from durable anchors plus replay:

- every selected training batch is written as an immutable, digest-addressed
  artifact before execution;
- every accepted step and safety-gate result is journaled in order;
- every published checkpoint has a corresponding complete resumable training
  bundle in the private training-state prefix, while the miner-visible HF tree
  keeps its current model/tokenizer/profile contract;
- optional mid-cadence full-state snapshots are selected from measured
  snapshot size, transfer time, and desired recovery time;
- after a loss, a replacement restores the last durable bundle and replays the
  exact ordered batch artifacts before it can become active.

Persisting a multi-tens-of-gigabytes optimizer bundle after every 100-second
step may cost more than the GPU time it saves. The benchmark must choose the
snapshot interval. The correctness requirement is an auditable replay path and
no duplicate public checkpoint, not an unmeasured promise of a full upload
after every step.

The existing train_step function and health gates remain the training kernel.
The new boundary serializes its input and state ownership; it does not rewrite
GRPO.

### 4.6 Checkpoint staging and publication

Split today's CheckpointStore.publish into explicit states:

~~~text
training step accepted
  -> snapshot requested
  -> immutable local/state artifact complete
  -> artifact digest verified
  -> HF upload in progress
  -> HF revision returned
  -> profile/revision verified
  -> signer approves manifest
  -> proof replicas prewarm and qualify
  -> activate at clean window boundary
  -> old revision drains
~~~

In the first safe version, publication remains a protocol barrier. Another
optimizer step is blocked, and control does not open another trainable window
against the old checkpoint. This preserves three current invariants together:

- miners, proofs, and training agree on the pinned checkpoint;
- rewarded groups are the groups actually selected for training;
- the configured behavior-policy refresh interval is not silently exceeded.

The edge and control processes remain responsive during the barrier, and the
snapshot, upload, signing, and replica-prewarm timings become independently
observable and optimizable. A future dual-revision pipeline may hide more of the
barrier only after it proves how old-revision batches remain trainable without
violating those invariants. That is a training/protocol experiment, not part of
minimal v1.

The HF token stays in the publisher, not the trainer. The trainer writes an
exact staging object through a short-lived R2 credential or presigned upload.
The publisher promotes it to the existing HF repository and returns the commit
revision.

### 4.7 Signer

The signer never exposes a generic sign-bytes operation.

Minimum semantic operations:

#### sign_checkpoint_manifest

Input:

- netuid 81;
- expected validator identity;
- configured HF repo ID;
- checkpoint number;
- HF revision;
- profile and artifact digests;
- request nonce and expiry.

Checks:

- caller mTLS identity;
- fixed netuid and repo;
- monotonically increasing checkpoint number;
- revision exists and matches the approved publication record;
- profile has the expected protocol and parent lineage;
- one signature per checkpoint number;
- replay and rate limits.

Output: the current Reliquary manifest signature only.

#### submit_weights

Input:

- netuid 81;
- epoch/boundary identifier;
- UID and weight vector;
- computation/archive watermark;
- request nonce and expiry.

Checks:

- fixed netuid;
- valid UID range and finite non-negative weights;
- normalization and burn-mass invariants;
- monotonic epoch and archive watermark;
- rate limit and replay prevention.

The signer constructs and submits the Bittensor operation itself. It does not
return a reusable raw signature.

An even smaller reuse option is to run the existing WeightOnlyValidator inside
the signer domain with read-only, prefix-scoped R2 access. It can continue to
replay authoritative archives and submit weights without accepting a vector
from the public control process. This option should be preferred if its
credential and package surface remains acceptably small.

The current repository pins Bittensor 10.5.0. Newer SDK documentation describes
policy guardrails, but the infrastructure plan must not assume those APIs until
the repository deliberately upgrades and tests them.

## 5. Job and result contracts

### 5.1 Deterministic IDs

A transport retry must not become a second economic event.

Proof job ID:

~~~text
sha256(
  "reliquary-proof-v1"
  || netuid
  || window_n
  || environment
  || economic_rank
  || miner_hotkey
  || checkpoint_revision
  || canonical_submission_digest
  || proof_profile_digest
)
~~~

Training job ID:

~~~text
sha256(
  "reliquary-train-v1"
  || training_run_id
  || step_n
  || parent_state_digest
  || ordered_batch_digest
  || training_profile_digest
)
~~~

The journal enforces uniqueness. A worker must treat a repeated job ID with the
same input digest as an idempotent retry and reject the same ID with different
inputs.

### 5.2 Proof request

The wire request contains only the data required by the proof kernel:

- job, window, environment, and rank;
- checkpoint revision and model/profile digests;
- canonical prompt and rollout inputs or a single exact-object reference;
- miner identity and commitment fields required by GRAIL;
- deadline and bounded resource profile;
- control request nonce.

The current _ScheduledProofPayload cannot cross the boundary because it points
to a live batcher. Control retains job_id to PendingSubmission and applies the
kernel outcome locally.

### 5.3 Proof result

The result includes:

- job ID and worker/attestation session;
- checkpoint, image, profile, and input digests;
- pass/reject/infrastructure-error status;
- the complete bounded evidence needed to construct ValidSubmission;
- proof metrics currently archived;
- start/end/device timing;
- output digest and worker signature.

Control checks expected job, revision, digests, deadline, lease, duplicate
status, schema, and bounds. It then lets GlobalProofScheduler apply the decision
in rank order. An infrastructure error aborts or retries according to the
current fail-closed semantics; it never becomes a miner-negative label.

### 5.4 Training request and result

The request contains:

- job ID, run ID, step, lease generation, and parent state digest;
- ordered Math and Code batch artifact digest;
- behavior-policy and KL-reference revisions;
- exact training profile/image digest;
- expected safety gates.

The result contains:

- parent and candidate state digests;
- step and lease generation;
- all current training health metrics;
- accepted/skipped status and reason;
- RNG/state metadata;
- durable artifact location when a snapshot was requested.

Control commits only if the parent digest and lease generation still match the
active lineage. A stale or duplicate result is recorded but cannot advance
state.

## 6. Transport and storage

Use the existing R2 bucket with separated prefixes and credentials:

~~~text
reliquary/dataset/window-*.json.gz       existing archives
reliquary/jobs/proof/<job-id>/...        only when payload is too large
reliquary/jobs/train/<job-id>/batch...   immutable training batches
reliquary/results/<job-id>/...           large bounded results
reliquary/training/<run>/<state>/...     resumable state anchors
reliquary/checkpoints/staging/<n>/...    publication staging
~~~

Small proof requests may travel directly over the authenticated worker channel;
typical live submission payloads are far below the 64 MB maximum and GPU
execution dominates a normal network round trip. Large proof payloads,
training batches, and checkpoints use object references.

R2 supports exact-object presigned GET/PUT and path-scoped temporary
credentials. Use the shortest TTL that covers the operation. These URLs are
bearer tokens and must not be logged.

Do not place multi-gigabyte artifacts in a message broker. Do not give a worker
the current bucket credentials.

## 7. Worker bootstrap

~~~text
provider adapter creates instance
  -> immutable image boots
  -> worker generates ephemeral key
  -> worker presents one-time bootstrap token
  -> if CVM: control challenges and verifies TDX + GPU attestation
  -> control binds attestation, image digest, and ephemeral public key
  -> short-lived worker certificate issued
  -> exact model/profile loaded
  -> capacity qualification runs
  -> worker reports ready for one checkpoint revision
~~~

The provider master API key never enters the worker. Lium currently documents
its API key as account-wide for create, read, delete, backup, and billing
actions without a separate read-only scope. Keep it in the provider
orchestrator and treat compromise as a billing and availability incident.

Prefer a worker-initiated authenticated HTTP/2 or similar long-lived channel so
ephemeral workers do not join the German private LAN and do not require broad
inbound firewall rules. The protocol contract should not depend on Lium SSH.

## 8. Networking

### Public

- edge accepts miner traffic;
- only dynamic validator endpoints proxy to control;
- no signer, CPU supervisor, job journal, or worker management port is public.

### Control network

- control to signer semantic API only;
- signer to approved Bittensor chain endpoints;
- control to R2/HF as required;
- provider orchestrator to provider API;
- deny lateral access from execution workers.

### CPU execution

- supervisor accepts only authenticated grader requests;
- microVMs have no NIC;
- management access is through the admin VPN;
- no route to signer, journal, R2, HF, provider API, or validator LAN.

### GPU execution

- GPU instances are never VPN members of the trusted network;
- permit worker control channel and exact artifact URLs;
- deny metadata endpoints and private address ranges where the provider allows;
- no persistent shared volume contains credentials.

Hetzner vSwitch and host nftables can segment control and CPU execution, but
private addressing is not itself a trust boundary. The signer should preferably
be under a separate provider account or provider so one control-account
compromise does not automatically expose the hotkey host.

## 9. Observability

Add metrics at boundaries rather than adopting an entire observability platform
before it is needed:

- state_snapshot_age_seconds;
- state_snapshot_bytes and compressed_bytes;
- ingress status and latency by endpoint;
- proof_queue_depth by revision and environment;
- proof_job_seconds by worker/provider/GPU;
- proof_result_mismatch_total;
- trainer_step_seconds and trainer_lease_generation;
- trainer_replay_steps_total and recovery_seconds;
- checkpoint_snapshot, upload, commit, signer, preload, and activation seconds;
- worker_boot, model_load, qualification, drain, and termination seconds;
- worker_cost_usd and cost_per_completed_window;
- grader queue, execution, recycle, timeout, and canary failures;
- signer accepted and denied requests;
- journal/outbox depth and oldest age.

Never put raw signed URLs, authorization headers, wallet paths, submission text,
environment dumps, provider keys, or signatures into logs.

## 10. What deliberately stays the same

- Miner HTTP payloads and checkpoint manifest format unless a compatibility
  test proves an extension is required.
- The 100-second collection contract.
- Auction ranking, drand ordering, proof debt, and trained-only reward
  invariants.
- GlobalProofScheduler's deterministic decision model.
- The current training kernel and numerical health gates.
- R2 archive schema and archive retry queue.
- Hugging Face as the miner-visible checkpoint repository.
- The weight replay algorithm.

The infrastructure split must first reproduce the current answer. Protocol or
training-policy improvements belong to separate experiments.
