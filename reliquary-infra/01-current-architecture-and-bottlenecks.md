# Current architecture and measured bottlenecks

## 1. Evidence and scope

This document distinguishes three sources of truth:

1. Live process and host inspection, performed read-only on 2026-08-21.
2. The exact OCI image revision reported by the live health endpoint:
   a083e5d8b896878cbc2f46579abe030fe13a7606, merged on 2026-08-20.
3. Existing repository handoffs, used for historical intent but not allowed to
   override the current image or live behavior.

The checked-out worktree is 542 commits behind the live image revision, so the
source analysis used git object reads against the live revision. No production
files, services, containers, firewall rules, wallet files, or credentials were
changed. Credential values were not printed or copied.

One legacy SSH alias, val-hel, presented a changed host key and was not used or
bypassed. The production host identified by the current repository handoff was
reachable with its existing known host key and was the host inspected.

## 2. What runs today

### 2.1 Host and container

The live deployment is one large, privileged container on one GPU host:

| Item | Observed state |
|---|---|
| OS | Ubuntu 22.04.5 |
| CPU | 26 virtual CPUs, Intel Xeon Platinum 8480+ |
| RAM | 221 GiB, no swap |
| GPU | One NVIDIA H100 PCIe 80 GB |
| Root disk | 993 GB total, 873 GB used, 88% |
| Container | reliquary-trainer |
| Image size | about 19.9 GB |
| Container privilege | privileged, host cgroup namespace, root |
| Resource limits | no CPU or memory limit; 64 MB Docker shared memory |
| Public path | nginx on port 8080 to loopback container port 18080 |

The container has these important mounts and credentials:

- the validator wallet directory mounted read-only;
- persistent Reliquary state and Hugging Face cache volumes;
- the proof-capacity manifest;
- the online hotkey, HF write token, R2 credentials, and W&B token in its
  effective environment.

Read-only does not protect the wallet from a root compromise of the privileged
container. It prevents filesystem writes through that mount; the process can
still read the key.

At a representative sample the container used about 157 GiB of RAM, had around
1,946 processes/threads, and had transferred roughly 1.1 TB outbound since its
most recent start. The H100 was at 100% utilization during active GPU work, with
about 57 GB resident at baseline and almost the full 80 GB used during peaks.

### 2.2 Process topology

~~~text
nginx
  |
  v
one uvicorn/FastAPI event loop
  |
  +-- state/checkpoint/health/verdict reads
  +-- precommit and reveal ingress
  +-- 8 Math admission tasks
  +-- 4 Code admission tasks
  +-- Math ProcessPoolExecutor
  +-- Code ProcessPoolExecutor
  +-- 12-thread materialization pool
  +-- weight-setting OS thread/event loop
  |
  +-- trusted grader supervisor, same privileged container
  |     |
  |     +-- gVisor runsc workers, network=none
  |
  +-- GlobalProofScheduler
  |     |
  |     +-- one worker for cuda:0
  |
  +-- train_model on cuda:0
  +-- frozen verify_model on cuda:0
  +-- optional fixed KL reference
  |
  +-- local checkpoint save -> Hugging Face upload -> hotkey signature
  +-- disk archive queue -> R2
~~~

The image starts the gVisor grader from docker/entrypoint.sh with a scrubbed
environment. That is a useful defense, and untrusted code runs with networking
disabled. The reason the outer validator container is privileged is precisely
to let runsc create its required processes and cgroups. Consequently, the
security boundary for miner code is gVisor, while the process supervising that
boundary shares a host and privilege domain with the wallet and cloud
credentials.

## 3. Current protocol execution

### 3.1 Collection and admission

The live protocol profile is version 4 with two environments,
OpenMathInstruct and OpenCodeInstruct. A window collects for a configured 100
seconds in the live image, despite older prose documentation referring to 150
seconds.

Each miner submission follows this path:

1. nginx accepts and streams the request to the validator;
2. the single FastAPI process performs envelope, freshness, registration, and
   window checks;
3. work is queued to one of eight Math or four Code admission tasks;
4. parse, canonicalization, and reward preparation run in per-environment
   spawned process pools;
5. Code reward execution is sent over a Unix socket through GraderClient to the
   gVisor grader;
6. an accepted candidate is retained in the active in-memory batcher.

The commit lock is not the limiting resource. Its observed waits and mutation
times are measured in microseconds.

### 3.2 Seal and proof verification

At seal, both environment batchers rank their retained candidates and construct
ProofPlan objects. GlobalProofScheduler is already a strong reusable component:

- plans for both environments are submitted atomically;
- each device has one long-lived worker;
- environments are scheduled fairly;
- jobs may finish out of order, but economic decisions are applied in rank
  order;
- prompt claims and hotkey/operator proof debt remain deterministic;
- the scheduler understands checkpoint readiness, deadlines, draining,
  quiescing, and replica refresh.

The limitation is not the scheduler design. Production configures only cuda:0,
so all Math, Code, and forensic proofs are serialized on the one H100.

The current payload is not remotely serializable. RankedProof.payload holds a
closure-like _ScheduledProofPayload that points back to the live batcher and
PendingSubmission. Its execute method runs the batcher's expensive verification
and may mutate rejection and debt telemetry. A remote executor therefore needs
a small refactor into:

- an immutable ProofRequest suitable for a wire;
- a pure proof-kernel result;
- a control-side apply step that preserves the current rank-ordered mutations.

The scheduler itself does not need replacement.

### 3.3 Training

ValidationService owns two distinct model roles:

- train_model is mutable and receives the optimizer update;
- verify_model is a frozen copy of the last published checkpoint.

verify_model is used for GRAIL proof verification and as the behavior-policy
model for PPO log-probability recomputation. Depending on configuration, it is
also the KL reference when a separate fixed reference is not loaded.

After proof selection:

1. the two environments contribute 16 groups each;
2. one training step consumes 32 groups and 512 rollouts;
3. train_step runs in a background thread but uses the same H100 as proof
   verification;
4. the optimizer step must remain sequential with every other accepted step.

Moving proof verification away does not remove the trainer's need for a frozen
behavior-policy replica. It does, however, let proof for the next sealed window
run at the same time as the current training step.

### 3.4 Checkpoint publication

The live checkpoint interval is 16 successful training steps.

CheckpointStore currently performs one coupled transaction:

1. save the multi-gigabyte Hugging Face snapshot to local disk;
2. write the Reliquary checkpoint profile;
3. upload the folder to Hugging Face;
4. obtain the Hub commit revision;
5. sign checkpoint-number plus revision with the local hotkey;
6. install the manifest;
7. copy training weights into verify_model;
8. drain and refresh every proof replica.

Publication deliberately runs on a serial beat so no collecting window observes
a checkpoint switch halfway through. This preserves correctness, but it also
stops useful collection and proof work while save/upload/refresh completes.

### 3.5 Persistence and weights

There is no Postgres or general message broker.

- Active window, pending candidates, scheduler plans, and training ownership
  are process memory.
- Window archives are atomically written into a persistent disk retry queue and
  uploaded to R2 in the background.
- R2 archives rebuild cooldown and emission history on restart.
- Hugging Face is the canonical published model repository.
- WeightOnlyValidator replays R2 archives and submits weights through the same
  mounted hotkey, from a separate event loop/thread.

The archive queue was healthy during inspection. It is already an appropriate
specialized outbox and should be reused.

Optimizer state is not a complete durable production object today. The
checkpoint profile records lineage and LR schedule information, but a restart
does not restore the complete optimizer, scheduler, scaler, RNG, and in-flight
training transaction needed for a replaceable remote trainer.

## 4. Measured bottlenecks

### 4.1 GPU critical path

The table below joins live service logs with the corresponding R2 window
archives for windows 30025 through 30048.

| Metric | Mean | p50 | p95 | Maximum |
|---|---:|---:|---:|---:|
| Proof wall, slower of the two environments | 115.0 s | 105.1 s | 156.2 s | 160.8 s |
| Sum of individual proof device time | 103.7 s | 95.0 s | 141.4 s | 151.8 s |
| Post-proof training and local overhead | 95.5 s | 96.8 s | 109.7 s | 117.5 s |
| Seal to completed training step | 210.4 s | 208.0 s | 266.2 s | 271.0 s |

Each window normally attempted about 38 expensive proofs in total, selected 32
winners, and started from roughly 117 pending candidates across the two
environments.

Interpretation:

- Proof and training are both material; neither is a minor tail.
- The sum of individual proof durations is close to wall time, confirming
  single-device serialization.
- The current two-window pipeline overlaps collection with the sealed window's
  GPU half, but the roughly 208-second GPU half is longer than the 100-second
  collection target. GPU work therefore controls steady-state cadence.
- Adding CPU queue infrastructure does not address this critical path.

### 4.2 Checkpoint publication

Window 30037 was deliberately routed through the serial publication path:

| Event | UTC |
|---|---|
| Window 30037 sealed | 08:32:45 |
| Training step completed | 08:35:44 |
| Checkpoint 523 published and replicas refreshed | 08:39:38 |

The publish/refresh portion after training took about 234 seconds. The next
window did not open until 08:39:42. The open-to-open interval around that serial
beat was about 654 seconds, versus a recent non-publication median near 195
seconds.

The logs do not yet split those 234 seconds into snapshot serialization,
hashing, HF transfer, commit, verify-model copy, and replica refresh. That
instrumentation is required before selecting the checkpoint publisher's disk
and network.

### 4.3 Public read fan-out

The live application already has a serialized-byte cache and a custom ASGI fast
path for state. This removed most framework and serialization work, but the
same event loop still owns every socket write and the same container still owns
the process, GPU work, and checkpoint lifecycle.

Over the most recent 500,000 nginx access-log records during inspection:

| Observation | Value |
|---|---:|
| Sample duration | 22 min 52 s |
| All requests | about 364/s |
| Successful state responses | 421,843, about 307/s |
| State response bytes | 34.39 GB |
| Average successful state body | about 81.5 KB in the log sample |
| state 502 responses | 11,633 |
| state 503 responses | 4,467 |
| state 499 responses | 1,723 |
| health responses | 39,413, average about 16 KB |
| Precommit requests | about 2,581 |
| Reveal submissions | about 1,018 |

The 502 and 503 values describe this loaded sample, not a long-term uptime SLO.
They are nevertheless important: nginx logged bursts of upstream connection
resets at the same time that the service recorded a roughly 21-second event-loop
lag near a training completion.

A fresh state response measured 86,807 bytes and compressed to 36,499 bytes
with gzip level 1. The nginx configuration did not send Content-Encoding,
ETag, Cache-Control, or Vary headers for the response. If every observed state
client accepted gzip, that sample's 34.39 GB would have been roughly 14.5 GB
before HTTP overhead. Conditional requests or a changed polling contract could
reduce it further, but those require miner behavior to be measured.

### 4.4 Admission and code grading

Live rolling latency at inspection:

| Environment/stage | p50 | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|
| Code queue wait | about 0.97 s | about 9.7 s | about 13.7 s | about 28.5 s |
| Code grading | about 0.21 s | about 0.51 s | n/a | n/a |
| Code admission total | about 1.55 s | about 13.7 s | about 25.8 s | about 57.2 s |
| Math queue wait | about 0.002 s | about 0.13 s | n/a | n/a |
| Math preparation | about 0.40 s | about 1.63 s | n/a | n/a |
| Math admission total | about 0.81 s | about 5.96 s | about 21.5 s | about 32.6 s |

The code grader had 32 live workers, hundreds of worker deaths, and heavy
recycling over approximately 22 hours. Its inner grading time is normally
short; burst queueing and worker churn create the tail. This makes it a
reliability and isolation target, but not the dominant training-throughput
target.

### 4.5 Disk and log pressure

The root disk was 88% full.

| Consumer | Observed scale |
|---|---:|
| Docker images | 371.6 GB |
| Docker image data reported reclaimable | 362.3 GB |
| Docker volumes | 494.3 GB |
| Current and previous nginx access logs | more than 7.3 GB |
| Current and previous nginx error logs | more than 2.0 GB |

No cleanup was performed. This is an immediate operational risk and a benchmark
confounder: disk pressure can lengthen snapshot writes and make a checkpoint
publication look like a GPU problem.

## 5. Bottleneck ranking

### Priority 1: proof and training serialize on one H100

This is the largest repeated critical-path cost. The existing scheduler already
supports multiple local devices, so the first benchmark should prove the
ceiling with separate local proof and training GPUs before a provider network
is introduced.

### Priority 2: checkpoint save/upload/refresh blocks routing

It is infrequent but very large. More proof GPUs alone will not remove the
periodic serial pause.

### Priority 3: state polling shares failure fate with training

The state fast path is efficient Python, but the wrong failure domain. Static
snapshot serving, compression, and log sampling are simpler than scaling the
whole validator.

### Priority 4: Code admission has bursty queue tails

Moving the grader to its own execution host removes privileged hostile-code
support from the control plane and permits independent CPU scaling. Firecracker
is a security choice here; it is not expected to accelerate the measured
0.2-to-0.5-second grader kernel.

### Priority 5: stateful recovery is insufficient for replaceable training

A disposable trainer needs exact once-only step commits and a complete state
bundle. This is a correctness prerequisite, not a performance optimization.

## 6. Things that are not currently the main bottleneck

- The batcher commit lock.
- R2 archive upload; the persistent queue is healthy and non-blocking.
- A lack of Kafka-scale transport.
- A lack of Kubernetes.
- Firecracker boot time, because Firecracker is not on the current path.
- Signer latency; checkpoint signatures and weight extrinsics are rare.
- The number of general control-plane CPU cores, before public reads and the
  code grader are removed from the main process.

## 7. Ground-truth source seams

The live revision already contains most of the clean boundaries needed:

| Current source | Reusable responsibility |
|---|---|
| reliquary/validator/proof_scheduler.py | Deterministic multi-device planning, deadlines, rank-order application, and replica lifecycle |
| reliquary/validator/service.py | Window FSM, pipeline ownership, checkpoint pinning, and orchestration |
| reliquary/validator/batcher.py | Auction ranking, proof request construction, debt, and final selection |
| reliquary/validator/training.py | Canonical training step and safety gates |
| reliquary/validator/checkpoint.py | HF snapshot format, profile, upload, manifest semantics |
| reliquary/environment/grader_client.py | Small fail-closed code-grading contract |
| reliquary/environment/grader/server.py | Existing worker pool, recycling, timeouts, and gVisor execution |
| reliquary/infrastructure/archive_queue.py | Durable specialized disk outbox |
| reliquary/validator/weight_only.py | Deterministic R2 replay and weight construction |
| reliquary/validator/server.py | Miner protocol, admission queues, state payload construction, and observability |

The plan in the following documents introduces adapters around these
responsibilities rather than replacing them.
