# Implementation, validation, and cutover plan

This began as a future implementation plan. As of 2026-08-28, the phase-2
immutable snapshot edge and reproducible `ctrl-01` hardening playbook are
deployed in non-authoritative shadow mode and qualified at 2x target read load.
See [the deployment report](08-ctrl-01-deployment-report.md). The remaining
phases and every production-authority gate are still future work.

> Update 2026-08-25: phases 1.1, 1.2, and much of phase 3 now have working
> equivalents in production through the detached trainer and isolated proof
> process. Do not implement this document literally from the beginning. Follow
> [06-rom-optimizations-review.md](06-rom-optimizations-review.md), beginning
> with detached checkpoint startup coherence, then checkpoint artifact
> binding, then a `ProofExecutor` interface around the existing worker pool.

## 1. Delivery strategy

The safest path is to introduce narrow interfaces around the current local
implementation, prove byte/result parity, and then substitute remote executors
one role at a time.

Every phase has:

- one bounded purpose;
- a local compatibility mode;
- a shadow mode where applicable;
- a measurable exit gate;
- a rollback boundary.

No phase changes auction economics or the miner protocol.

## 2. Phase 0: freeze a reproducible baseline

### Deliverables

- Record the exact live image, protocol profile, model revision, proof-capacity
  manifest, environment mix, and runtime package lock.
- Export secret-free health and scheduler snapshots.
- Preserve at least 100 representative completed window archives.
- Capture stage-level logs for collection, admission, proof, training,
  checkpoint save, HF upload, commit, model refresh, and archive enqueue.
- Add a benchmark manifest that pins dataset/task IDs and digests without
  storing miner secrets or raw credentials.
- Record current provider, disk, network, and GPU hourly costs.

### Required new timing splits

The current 234-second checkpoint observation is too coarse. Measure:

- model synchronization before save;
- safetensors serialization;
- checkpoint profile write;
- local hashing;
- HF LFS/Xet preupload;
- Hub commit;
- signer;
- verify/behavior-model refresh;
- each proof-replica load and qualification;
- activation.

Also split the approximately 97-second post-proof interval into batch
materialization, forward, backward, optimizer, health gates, cleanup, and
archive preparation.

### Operational prerequisite

Resolve disk pressure and logging policy on the future benchmark host before
timing checkpoint I/O. Do not delete production data as part of this analysis.
The implementation runbook must identify which Docker images are genuinely
unused, preserve rollback images, and make any deletion explicit and
recoverable where possible.

### Exit gate

The same benchmark run can be repeated twice with stage p50/p95 within an
agreed variance band and all artifacts traceable to exact revisions.

## 3. Phase 1: extract local compatibility seams

This phase changes structure but not deployment.

### 3.1 Proof

Current:

~~~text
RankedProof.payload
  -> _ScheduledProofPayload(batcher, pending)
  -> batcher._execute_scheduled_proof(model)
  -> proof computation plus batcher side effects
~~~

Target local structure:

~~~text
batcher builds immutable ProofRequest
  -> LocalProofExecutor executes pure proof kernel
  -> ProofKernelResult
  -> GlobalProofScheduler orders decisions
  -> batcher applies result/debt/telemetry locally
~~~

Keep GlobalProofScheduler's planning, resource serialization, deadline, and
rank-order logic unchanged. Add golden tests that compare:

- winner IDs and order;
- proof attempts;
- rejection reason;
- hotkey and operator debt;
- forensic results;
- archived proof telemetry;
- failure behavior.

### 3.2 Training

Introduce a small TrainingExecutor protocol:

~~~text
run_step(parent_state, ordered_batches, profile) -> TrainingStepResult
snapshot(request) -> TrainingStateArtifact
restore(artifact) -> readiness
~~~

LocalTrainingExecutor calls the existing train_step and owns the current model
objects. No training math moves in this phase.

Define one canonical TrainingBatchArtifact from the exact fields train_step
consumes. Reuse existing rollout, reward, profile, and hash structures rather
than creating a second semantic schema.

### 3.3 Grader

Keep GraderClient and its local Unix-socket semantics unchanged. Split the
trusted GraderServer's sandbox call into:

- `LocalRunscExecutor`, wrapping the current worker pool;
- `RemoteExecutor`, calling the zero-secret CPU execution agent later;
- the existing trusted comparison path, which remains on control.

Candidate-caused failure remains 0.0. Executor transport, supervisor, timeout,
or malformed response remains GraderInfrastructureError. Expected values never
cross into the remote execution domain.

### 3.4 Checkpoint and signing

Split CheckpointStore responsibilities without changing output:

- snapshot writer;
- checkpoint publisher;
- manifest signer interface;
- manifest installer/activator.

Local implementations reproduce today's save, HF upload, wallet signature, and
activation in the same order.

### 3.5 State snapshot

Add an atomic state-byte sink next to the existing state-response cache. The
normal FastAPI route and fast-path cache remain authoritative during this phase.

### Exit gate

- Existing unit and integration suite passes.
- A replay corpus produces identical proof decisions, selected batches,
  rewards, checkpoint manifests, and archive fields.
- Local compatibility mode has no statistically material latency regression.

## 4. Phase 2: isolate public state serving

### Deliverables

- A tiny static snapshot service or nginx file-serving path.
- Identity and gzip variants written atomically.
- ETag and correct Vary header.
- Explicit behavior for no-active-window and checkpoint transitions.
- Sampled/suppressed access logging for high-volume successful state polls,
  while errors and write endpoints remain fully observable.
- Load and fault tests.

### Rollout

1. Generate snapshots but continue serving the current dynamic route.
2. Compare response bytes and transition timing continuously.
3. Route a private canary hostname to the snapshot path.
4. Run 750 requests/s while training and checkpoint publishing occur in the
   benchmark environment.
5. Switch public state reads only after parity.

### Rollback

One nginx route change returns state reads to the existing ASGI fast path. The
validator remains the producer in both modes.

## 5. Phase 3: prove the local GPU split

Use separate test hardware and local adapters before remote networking.

### Configurations

1. one shared H100, current baseline;
2. proof GPU 0 and trainer GPU 1;
3. proof GPUs 0 and 1 plus trainer GPU 2, if available.

### Required control changes

- bounded stage queues;
- separate proof and trainer readiness;
- explicit per-window checkpoint revision;
- metrics for queue wait and stage occupancy;
- fail-closed behavior when qualified proof capacity falls below the manifest.

GlobalProofScheduler may still have only one live plan per environment.
Control keeps at most a small bounded sealed-plan queue and submits the next
plan when the environment is free. No unbounded accumulation is allowed.

### Exit gate

- exact result parity with the shared-GPU mode;
- one-proof split demonstrates the predicted overlap;
- two-proof configuration achieves proof-wall p95 below 90 seconds or explains
  the non-parallel constraint with evidence;
- no checkpoint revision crosses a window;
- no OOM or retained-memory growth across at least one publication interval.

This phase supplies the performance ceiling for every remote design.

## 6. Phase 4: durable job and lease journal

### Minimal schema

Use one control-owned SQLite database in WAL mode:

~~~text
jobs
  job_id primary key
  kind
  input_digest
  checkpoint_or_parent_revision
  state
  attempt
  worker_id
  lease_generation
  deadline
  result_digest
  created/started/finished timestamps

workers
  worker_id
  role
  provider
  image/profile/checkpoint digests
  attestation state
  readiness
  last heartbeat

training_lineage
  run_id
  active_state_digest
  durable_anchor_digest
  step
  lease_generation
  active_worker_id

checkpoints
  checkpoint_n
  parent
  training_state_digest
  staging_digest
  hf_revision
  signature
  state

outbox
  ordered durable events awaiting delivery
~~~

Only control writes the database. Remote workers never connect to it.

### Job transitions

~~~text
created
  -> leased
  -> running
  -> succeeded | rejected | infrastructure_error | expired
  -> applied
~~~

Rules:

- the same job ID plus different input digest is fatal;
- only the active lease attempt may finish;
- a late result after deadline is recorded but not applied;
- application is transactional with the result digest;
- transport acknowledgement is not economic application;
- an infrastructure error never becomes a miner rejection;
- retries retain the original job ID and rank.

### Exit gate

Crash and restart control at every transition. Each job is applied zero or one
times, never twice, and existing proof ordering remains unchanged. A restart
during an active economic window aborts/tombstones that window and fences its
late results instead of attempting to reconstruct the in-memory batcher.

## 7. Phase 5: remote proof worker in shadow

### Worker image

- exact pinned Reliquary runtime;
- proof kernel only;
- model loader and capacity qualifier;
- no trainer, grader, wallet, HF writer, provider CLI, or SSH private key;
- non-root where the GPU runtime permits;
- read-only image and disposable scratch;
- outbound authenticated worker channel.

### Shadow sequence

1. Bootstrap a manually provisioned trusted test worker.
2. Mirror proof requests after local proof has been scheduled.
3. Local result remains authoritative.
4. Compare full ProofKernelResult, not just pass/fail.
5. Inject disconnects, duplicate delivery, corrupted digests, wrong checkpoint,
   slow result, and worker restart.
6. Repeat with a second GPU type.
7. Repeat on an ordinary Lium pod as lifecycle-only shadow.
8. Repeat on a Lium CVM and verify attestation before credentials are issued.

### Authority sequence

1. One remote worker serves a small canary fraction, with local recomputation
   before application.
2. A dedicated or attested remote worker becomes one scheduler device.
3. Keep a local qualified fallback until at least one full publication interval
   passes.
4. Scale to the proof pool selected by the capacity benchmark.

### Exit gate

- zero semantic divergence over at least 10,000 production-shaped proofs;
- proof p95 and cost gates from the machine-selection document;
- worker loss and retry preserve rank order;
- non-CVM results cannot be accidentally marked authoritative;
- old/new revision drain works during checkpoint transition.

## 8. Phase 6: replaceable trainer in offline and shadow modes

The trainer must not first appear as a live remote RPC.

### 8.1 Complete training state

Define and test a versioned state manifest covering:

- model weights;
- optimizer;
- LR scheduler/global step;
- scaler where present;
- CPU/CUDA RNG;
- behavior-policy revision and weights;
- fixed-reference revision;
- run ID and training profile;
- parent and candidate state digests;
- source batch/job sequence;
- current publication counters and pending state needed for recovery.

Store this as a separate private R2 training-state artifact. Do not add
optimizer/RNG payloads to the miner-visible Hugging Face checkpoint tree.

### 8.2 Offline replay

Starting from the same durable anchor:

- local current trainer and candidate trainer process identical batch artifacts;
- compare safety metrics every step;
- compare exact tensors where runtime determinism permits;
- otherwise use defined tensor tolerances plus paired held-out quality gates;
- verify skip/reject decisions are identical;
- restore and replay after forced loss.

### 8.3 Fencing

Run two candidate trainers deliberately:

- generation N owns the lease;
- generation N+1 replaces it;
- N returns a valid-looking late result;
- control must reject N without changing lineage;
- N+1 alone can advance.

### 8.4 Shadow production

Mirror selected production batches after the live trainer has consumed them.
The shadow trainer never publishes, signs, changes live state, or sets weights.
Compare the full interval through checkpoint staging and paired quality
evaluation.

### Exit gate

- recovery from every tested failure point;
- no duplicate or stale commit;
- complete state artifact restore;
- performance and cost meet the selected machine gate;
- paired Math/Code promotion result is no worse than the local baseline;
- the existing local trainer remains a boundary rollback option.

## 9. Phase 7: separate checkpoint publisher

### Minimal-v1 semantics

Preserve the current serial trainable-window barrier:

1. the publication-due window trains;
2. no next trainable window opens;
3. trainer produces immutable snapshot/state artifacts;
4. publisher uploads and returns the exact HF revision;
5. signer approves the semantic manifest;
6. proof workers prewarm and qualify the revision;
7. trainer behavior model and proof pool are coherent;
8. control activates state at the next clean boundary.

The edge continues to serve an explicit closed/transition state and health
remains responsive. No old-checkpoint backlog is silently trained after the
behavior checkpoint changes.

### Optimization work

- measure and choose HF upload_folder versus preupload-LFS/create-commit paths;
- stream/hash once where supported;
- use dedicated NVMe staging;
- separate snapshot creation from network upload;
- prewarm proof replicas as soon as the immutable artifact is available;
- maintain resumable publication state in the journal;
- retry the same checkpoint number and artifact, never create a second model
  candidate after an upload failure.

### Future experiment, not v1

A dual-revision pipeline could collect against the old checkpoint during
publication only if it demonstrates:

- rewarded old-revision groups are actually trained;
- their behavior log-probability reference is correct;
- the configured safe-update interval is not exceeded;
- verify_model still exactly matches the public HF revision;
- checkpoint activation remains atomic.

Until then, the barrier is intentional.

### Exit gate

- every checkpoint state transition survives restart;
- no mid-window revision switch;
- exact snapshot/profile/HF revision/signature coherence;
- checkpoint failure never exposes a half-published state;
- public state service has no 502/503 burst during publication;
- measured barrier is materially below the current 234 seconds or its cost is
  accepted explicitly.

## 10. Phase 8: move hostile CPU execution

### 8.1 Remote gVisor baseline

- provision the selected bare-metal execution candidate;
- keep the trusted coordinator, expected values and comparison on control;
- put its sandbox call behind a local/remote `SandboxExecutor` interface;
- deploy only the existing gVisor worker pool and a zero-secret executor agent
  on the remote host;
- preserve network=none;
- use `RemoteExecutor` from the trusted coordinator;
- run parity, load, recycle, and malicious corpus tests;
- verify the control deployment no longer needs privileged mode.

### 8.2 Firecracker candidate

- build pinned guest kernel/rootfs artifacts;
- implement one-request lifecycle over vsock;
- add jailer/cgroup/scratch cleanup;
- benchmark cold and clean-preboot modes;
- never reuse a post-hostile guest or snapshot;
- make backend selection an execution-host setting, not a validator protocol
  branch.

### Cutover

Remote gVisor becomes authoritative first if it passes. Firecracker replaces it
only after equal result parity and acceptable queue capacity. This obtains the
trust-domain benefit without coupling the GPU project to a new sandbox runtime.

## 11. Phase 9: deploy the signer domain

### Preparation

- create the signer host from a minimal immutable configuration;
- generate/import only the online hotkey through the approved secure process;
- confirm the coldkey never enters;
- implement mTLS caller identity, replay state, monotonic checkpoint/epoch
  state, and rate limits;
- decide between semantic submit_weights input and running the existing
  WeightOnlyValidator with prefix-scoped R2 read access;
- test hotkey rotation and signer disaster recovery offline.

### Shadow

- local wallet operation remains authoritative;
- signer validates mirrored manifest/weight requests but does not submit;
- compare constructed payloads and policy decisions.

### Cutover

- stop local signer use;
- prove the validator container/host has no hotkey file or wallet secret;
- enable signer authority;
- submit one checkpoint signature and one epoch weight operation under close
  observation.

### Rollback

Rollback does not automatically copy the hotkey into the public control host.
Use a documented manual signer recovery path or standby signer domain.

## 12. Phase 10: provider adapters and disposable capacity

Define two separate interfaces:

~~~text
GPUProvider
  list_capacity(requirements)
  quote(spec)
  provision(spec)
  status(instance)
  terminate(instance)
  cost(instance)

WorkerProtocol
  bootstrap(identity/attestation)
  ready(profile/revision)
  execute(job)
  heartbeat()
  drain()
~~~

Provider code never appears in proof or training logic.

Implement in this order:

1. ManualProvider for fixed benchmark hosts.
2. DedicatedProvider for long-lived known machines.
3. LiumProvider.
4. One second marketplace/cloud provider only after the abstraction survives
   Lium.

The Lium adapter owns its account-wide API key and reconciliation loop. Workers
receive only a one-time bootstrap token and scoped artifact access.

Exit gate:

- provider switch requires no validator, proof-kernel, or training-kernel
  change;
- leaked instances are found and terminated after orchestrator restart;
- billed lifetime and internal cost agree;
- minimum warm proof pool is maintained;
- trainer replacement respects fencing and restore gates.

## 13. Full shadow production

Run the complete candidate architecture beside production:

~~~text
live admission/selection inputs
  -> candidate proof plane
  -> candidate trainer
  -> candidate checkpoint staging
  -> candidate signer validation
  -> NO public checkpoint activation
  -> NO live reward or weight authority
~~~

Minimum shadow duration:

- at least two complete checkpoint publication intervals;
- enough windows to include normal, sparse, high-load, grader-failure, proof
  tail, worker-replacement, and provider-reconciliation cases;
- one deliberate trainer replacement;
- one deliberate proof-worker loss;
- one deliberate checkpoint upload failure;
- one control restart.

Compare:

- every proof decision and rank;
- every selected/rewarded group;
- every archive-relevant field;
- training gate and step;
- checkpoint artifact/profile;
- state transition timing;
- paired quality benchmark;
- cost per accepted window.

## 14. Cutover sequence

Cut over only at a published checkpoint boundary:

1. state snapshot edge;
2. signer;
3. remote CPU grader;
4. remote proof pool while local trainer remains;
5. verify one full publication interval;
6. acquire the remote trainer lease from the last durable checkpoint;
7. keep local GPU state stopped but recoverable, never concurrently writable;
8. verify another full publication interval;
9. remove wallet, privileged mode, and GPU dependencies from control;
10. terminate obsolete GPU capacity only after rollback retention expires.

At every step, the active authority is singular and visible in health.

## 15. Code-change map

| Current location | Planned seam | Preserve |
|---|---|---|
| validator/proof_scheduler.py | Treat device IDs as executor slots; inject executor | Rank order, fairness, debt resources, deadlines, lifecycle |
| validator/batcher.py _ScheduledProofPayload and _execute_scheduled_proof | Immutable request, pure kernel result, local apply | Auction and reject semantics |
| validator/service.py _execute_scheduled_proof | Local/remote ProofExecutor | Checkpoint pinning |
| validator/service.py _train_and_publish | Bounded proof/trainer stages and TrainingExecutor | Window FSM, quarantine, trained-only rewards |
| validator/training.py train_step | Kernel called by local or trainer worker | Math and safety gates |
| validator/checkpoint.py | Snapshot, publish, sign, activate stages | HF format/profile/manifest |
| environment/grader_client.py | Pluggable transport | Candidate versus infrastructure failure |
| environment/grader/server.py | Move unchanged first; add Firecracker backend later | Pool, timeout, recycle behavior |
| validator/server.py | Atomic state-byte sink | Exact miner state schema |
| infrastructure/archive_queue.py | Reuse as-is | Durable R2 archive outbox |
| validator/weight_only.py | Run in signer domain or reuse weight construction | Deterministic archive replay |
| cli/main.py | Dependency wiring and feature flags | Current defaults until cutover |

Prefer a few cohesive new modules over a framework:

- execution/contracts.py for immutable job/result types;
- execution/proof.py for local/remote executors;
- execution/trainer.py for local/remote trainer clients;
- infrastructure/job_ledger.py;
- infrastructure/providers.py, splitting provider files only when they grow;
- signer/service.py.

Do not add a general workflow DSL.

## 16. Global acceptance gates

### Semantic

- zero miner wire regression;
- zero proof-order/reward divergence;
- checkpoint and behavior revisions remain coherent;
- all current fail-closed infrastructure semantics preserved;
- no result accepted twice.

### Performance

- state edge passes the 750 requests/s gate;
- proof pool p95 below 90 seconds with 20% headroom;
- normal-window trained cadence near the max of collection, proof, and trainer
  stages;
- publication stage timing fully decomposed and improved or accepted;
- Code queue p95 below 1 second at 2x load.

### Recovery

- control restart;
- proof worker loss;
- trainer loss and replay;
- publisher failure;
- provider API failure;
- signer unavailable;
- R2 temporarily unavailable;
- no split-brain trainer or half-activated checkpoint.

### Security

- no wallet on control, CPU, or GPU;
- no master cloud key on a worker;
- no privileged public validator container;
- CPU malicious corpus contained;
- non-CVM GPU cannot become authoritative through configuration error;
- CVM authority requires verified nonce-bound attestation;
- signer denies arbitrary bytes, wrong netuid/repo, replay, and non-monotonic
  requests.

### Economic and operational

- cost per accepted trained window recorded;
- provider invoice reconciled;
- no orphan instances;
- capacity loss has explicit fail-closed behavior;
- rollback runbook executed, not only written.
