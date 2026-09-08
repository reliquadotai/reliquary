# Rom optimization review and revised next step

- Review date: 2026-08-25
- Repository revision reviewed: `ab6478f1657b9134d628c93e8816a14651910146`
- Production revision verified: the same revision
- Scope: PRs #183 through #196, with detailed review of #189 and #191 through
  #195
- Production inspection: read-only

## 1. Executive verdict

Rom has already implemented and deployed most of the performance split this
infrastructure plan originally proposed:

- collection is pipelined over the GPU half;
- the canonical trainer is detached and consumes an ordered R2 journal;
- checkpoint save/upload happens on the trainer machine;
- the validator stages trainer checkpoints from R2 and swaps them on a serial
  beat;
- GRAIL proof execution runs in a separate Python process, removing the live
  GIL convoy;
- the existing `GlobalProofScheduler` can dispatch across several local CUDA
  devices.

The steady-state result is excellent. The earlier measured p50 cycle was about
208 seconds, or 17.3 windows/hour. The first 46 consecutive intervals on the
new live revision averaged 104.8 seconds, or 34.3 windows/hour. The proposed
1.7x-to-2.0x gain has largely been achieved without Kubernetes, Kafka, NATS,
Temporal, or a scheduler rewrite.

The work is a strong transitional architecture, not yet the target disposable
GPU architecture. The proof workers are local child processes, not remote
workers. The trainer carries long-lived R2, HF, and telemetry credentials.
There is no provider adapter, short-lived workload identity, durable remote-job
ledger, independent signer, or hostile CPU host.

The next action is not to build another multi-GPU framework. It is to fix an
observed checkpoint-recovery regression, soak the current design, harden the
checkpoint boundary, and then put a remote transport behind Rom's existing
proof seam.

## 2. What is implemented now

### 2.1 Current live topology

~~~text
miners
  |
  v
validator process on the production host
  - FastAPI and window state
  - admission and auction ranking
  - GlobalProofScheduler
  - checkpoint intake and activation
  - wallet and weight setter
  |
  +--> local ProofWorkerPool child on cuda:0
  |      - one process per configured local proof device
  |      - frozen proof model
  |      - GRAIL verification only
  |
  +--> R2 ordered training payload/tombstone journal
           |
           v
       detached train-worker on a separate GPU placement
         - one canonical model owner
         - sequential train_step
         - HF checkpoint publication
         - R2 checkpoint mirror and candidate manifest
           |
           v
       validator stages -> quiesces -> reloads -> signs -> activates
~~~

Production currently uses one proof device, `cuda:0`. Multi-device support is
real but local: `RELIQUARY_PROOF_DEVICES` resolves CUDA devices in the
validator host, and `ProofWorkerPool` spawns one child for each unique device.
The detached trainer can use another GPU in the same machine or a different
machine because its data plane is R2.

This is already multi-GPU stage separation. It is not yet elastic remote proof
capacity: a proof device is still a local `torch.cuda` identifier, the IPC
transport is a `multiprocessing.Pipe`, and a spawned worker inherits the
validator container environment.

### 2.2 Relevant optimization series

| PR | Change | Review conclusion |
|---|---|---|
| #183 | Collection window reduced from 150 to 100 seconds using measured arrival data | Good measured reduction; production traffic still fills the useful pool before the deadline. |
| #184 | Reuse seal-time verify logprobs as `pi_old` and avoid a duplicate behavior forward | Good, narrowly gated optimization; same frozen model and T=1 contract, with fallback when data is unavailable. |
| #185 | Pipeline collection of window N+1 over the GPU half of N, retaining a serial publication beat | Excellent use of the existing state machine; this is the main overlap mechanism and should be retained. |
| #186 | Cache flash-attention unpad metadata and restore exact LR schedule position | Good low-risk kernel-side win and recovery correction. |
| #189 | R2 payload journal, detached train-worker, trainer publisher, validator checkpoint intake | Strong transitional split and already useful in production; not yet a zero-secret or fully resumable disposable trainer. |
| #191 | Move GRAIL proof calls to one child process per proof device | Excellent diagnosis of the real GIL convoy and a minimal seam; #191 by itself was not safe. |
| #192 | Correct worker revision reporting, pipe serialization, reload timeout, recovery source, and error propagation | Essential follow-up. The combined #191+#192 implementation is materially better than #191 alone. |
| #193 | Compare in-process and isolated `ProofResult` fields on a real small model | Good equivalence gate; it explicitly leaves tokenizer identity unenforced. |
| #195 | Teach HF checkpoint discovery about detached titles such as `checkpoint 618 (cadence)` | Incomplete. It fixed one parser but not the parser invoked by the subsequent resume. The failure is reproduced in production. |
| #196 | Size grader `RLIMIT_CPU` for the worker lifetime and expose worker-death evidence | Good measured correctness fix. Current post-deploy health shows no grader worker restarts or timeouts in the sampled uptime. |

Links:

- https://github.com/reliquadotai/reliquary/pull/189
- https://github.com/reliquadotai/reliquary/pull/191
- https://github.com/reliquadotai/reliquary/pull/192
- https://github.com/reliquadotai/reliquary/pull/193
- https://github.com/reliquadotai/reliquary/pull/195
- https://github.com/reliquadotai/reliquary/pull/196

## 3. Live performance after the changes

The production image label and repository HEAD both resolve to `ab6478f`.
These flags are enabled:

- pipelined windows;
- training-payload writer;
- detached trainer;
- proof-process isolation;
- one configured proof device.

### 3.1 Window sample

Sample: 47 consecutive seals, windows 32089 through 32135, immediately after
the latest deploy.

| Measurement | Result |
|---|---:|
| Consecutive seal intervals | 46 |
| All-interval mean | 104.8 s |
| All-interval p50 | 102.0 s |
| All-interval sampled p95 | 106.0 s |
| Effective throughput | 34.3 windows/hour |
| Ordinary intervals | 44 |
| Ordinary mean / p50 / sampled p95 | 102.4 / 102.0 / 105.0 s |
| Post-seal proof/finalization mean / p50 / sampled p95 | 53.0 / 53.5 / 63.0 s |
| Intervals following checkpoint swaps | 156 s and 163 s |

The 53-second proof half now hides comfortably inside the 100-second
collection window. The checkpoint-swap beat is the visible periodic GPU
barrier, but it is far smaller than the old save/upload/refresh barrier because
snapshot publication and transfer are mostly background work on the trainer.

Relative to the earlier 208-second p50 cycle, the observed mean throughput is
approximately:

~~~text
208.0 / 104.8 = 1.98x
~~~

### 3.2 Resource sample

At inspection time:

- validator container memory: about 28 GiB, down from the previous roughly
  157 GiB representative sample;
- H100 memory: about 34 GiB total;
- validator parent GPU allocation: about 23.6 GiB;
- proof child GPU allocation: about 10.2 GiB;
- proof scheduler healthy, checkpoint-coherent, and not degraded;
- live per-proof p50: approximately 1.40 seconds in both environments.

The parent still holds full model weights even though detached mode never
trains in that process, and process isolation adds another proof copy. This is
memory waste, but removing it should be done as part of the remote executor
boundary rather than through a second temporary model-lifecycle design.

### 3.3 The bottleneck moved

GPU serialization is no longer the dominant steady-state limiter. Remaining
pressure includes:

- the serial checkpoint-swap interval;
- admission CPU tails, especially Code grading;
- public API scheduling and event-loop tails;
- high-volume state reads from the same process.

In a recent 300,000-request nginx sample spanning about 27 minutes 32 seconds:

- `/state` handled about 225,000 requests, roughly 136 requests/second;
- it emitted about 13.1 GB of uncompressed response data;
- about 4,300 state requests returned 503 and about 770 were client-aborted;
- responses had no `Content-Encoding`, `ETag`, `Cache-Control`, or `Vary`;
- event-loop lag reached 10.5 seconds at the maximum;
- `/submit` p95 was about 5.7 seconds and Code admission total p95 about 12
  seconds, although queues were empty at the inspected instant.

State-edge isolation remains a worthwhile independent improvement. It is no
longer the prerequisite for proving the GPU speedup because the GPU speedup is
already live.

## 4. Blocking correctness finding: PR #195 is incomplete

### 4.1 The two parsers disagree

Detached publication writes HF titles such as:

~~~text
checkpoint 618 (cadence)
~~~

PR #195 added `checkpoint_n_from_commit_title` in `service.py`, so startup HF
discovery now sees this title and correctly finds the latest checkpoint.

Startup then calls `_apply_resume_from`, which delegates to
`validator/resume.py`. That module still uses:

~~~text
^checkpoint\s+(\d+)\s*$
~~~

It rejects the same detached title. The outer bootstrap catches the exception
and deliberately continues on the stale configured revision.

### 4.2 Production evidence

On the inspected restart:

1. the static resume source loaded checkpoint 572;
2. HF discovery found checkpoint 618 and attempted to override it;
3. `resolve_resume_source` rejected `checkpoint 618 (cadence)`;
4. the validator opened on checkpoint 572;
5. windows 32089 through 32091 used the old revision;
6. R2 intake later staged checkpoint 618 and swapped after window 32091.

This is the exact regression PR #195 intended to remove. It can reject miners
that stayed on the previously advertised newer model, makes public checkpoint
numbers go backward and then jump forward, and sends several old-policy
windows into the detached training journal after every auto-deploy.

### 4.3 Required fix

Create one shared checkpoint-title parser and use it everywhere:

- in-process publisher title construction;
- detached publisher title construction;
- HF history discovery;
- explicit SHA resume resolution;
- recovery tooling.

Do not duplicate another regex. Add an integration test that executes the
whole path:

~~~text
stale configured checkpoint 572
  -> HF latest title "checkpoint 618 (cadence)"
  -> bootstrap selects and downloads 618
  -> verify-model revision == 618
  -> proof scheduler revision == 618
  -> public manifest checkpoint_n == 618
  -> only then may the first window open
~~~

If startup has positively identified a newer checkpoint but cannot validate or
load it, fail closed instead of knowingly serving the stale revision.

Before this fix is deployed, the safe operational mitigation is to avoid an
automatic validator restart with the stale static SHA. If a restart is
unavoidable, pin the currently accepted revision immediately before it and
verify `/state` before reopening miner traffic.

## 5. Additional review findings before remote authority

### 5.1 R2 checkpoint bytes are not bound to the advertised HF revision

The trainer uploads one local snapshot to HF, mirrors files to R2, and publishes
the HF revision in a candidate manifest. The validator downloads the R2 files,
validates model/profile shape, installs those bytes, and signs the HF revision.
It does not prove that the R2 bytes it loaded are byte-for-byte the files in
that immutable HF commit.

A corrupted or replaced R2 mirror can therefore make the validator prove with
weights different from those miners download at the signed HF revision.

Add an artifact manifest to the HF commit containing the exact allowed
filenames, sizes, and SHA-256 hashes. Fetch that small manifest at the immutable
HF revision and verify every staged R2 file before model installation. Reject
extra files and missing files. This binds the fast R2 path to the canonical HF
commit without downloading the full model twice.

Hashes prevent transport substitution; they do not make a malicious trainer's
computation trustworthy. An eventually authoritative marketplace trainer still
needs accepted provider trust or verified attestation.

### 5.2 External checkpoint transitions are insufficiently fenced

`install_external` signs and installs the candidate checkpoint number and
revision without enforcing a monotonic transition or a parent relationship.
The candidate path should require:

- configured repo ID match;
- active protocol and training-run identity match;
- `checkpoint_n` strictly greater than the installed number;
- expected parent revision or accepted lineage link;
- monotonic trained-window cursor;
- verified artifact-manifest digest;
- no replay of an already rejected or superseded candidate.

Persist the last accepted, validator-signed checkpoint transition. A trainer
candidate is a proposal, not yet the validator's durable accepted state.

### 5.3 The proof-worker tokenizer identity is not enforced

PR #193 notes this explicitly. The worker loads its own tokenizer, and a
tokenizer mismatch changes `p_stop` and termination decisions. Extend the ready
handshake with the canonical tokenizer/generation-contract digest and refuse to
mark a device ready on mismatch.

### 5.4 Proof-worker replacement has a generation race

`_request` obtains a worker from the map, releases the map lock, and then waits
for the per-worker pipe lock. If one request retires that worker while a second
request already holds the stale object, a third request can install a new
worker. The stale second request can then fail and call `_retire(device_id)`,
which may pop and kill the new generation rather than the stale generation.

Retirement must be identity/generation-checked under the worker-map lock. Add a
test with one timed-out caller, one waiter on the old pipe, and one replacement.

### 5.5 The trainer lock is an observation, not a lease

The detached worker compares HF HEAD with its last known revision before
publishing. Two trainers starting from the same HEAD can both pass the check;
there is no atomic compare-and-swap or lease generation. Startup also adopts a
foreign HEAD as a possible orphaned half-publish.

This is acceptable only under the current operational assumption that one
trusted trainer is manually owned. Before provider-managed replacement, add a
control-owned fenced lease. Only the current lease generation may publish a
candidate.

### 5.6 Strict journal ordering needs a gap-repair path

Waiting forever rather than skipping a missing window is the correct training
default. However, payload/tombstone enqueue is best-effort. Loss of the
validator's local pending directory before upload can produce a permanent gap.
Add:

- producer high-water mark and gap telemetry;
- a control-owned reconciliation command;
- an audited way to write a tombstone only after proving the source window
  cannot be reconstructed;
- an alert before the trainer silently accumulates a large lag.

### 5.7 The current trainer is not a zero-secret disposable worker

The train-worker currently receives:

- bucket credentials;
- an HF write token;
- a telemetry API key.

This is fine for the current trusted trainer host but not for an ordinary Lium
pod. The target flow should give a worker exact input GET and output PUT scope,
then let trusted control publish to HF. Provider master credentials and HF
write authority never belong on the GPU.

The trainer also restores model weights and LR position but deliberately loses
optimizer moments on restart. That matches the historical recovery behavior,
but it means the trainer is replaceable with a quality cost, not freely
disposable. Decide from a measured preemption/recovery experiment whether to
persist full optimizer/scheduler/scaler/RNG state or keep a long warm lease and
accept rare re-ramp recovery.

## 6. PR quality assessment

### What is strong

- The performance diagnoses use production evidence rather than generic GPU
  advice.
- Rejected alternatives were measured, especially batching, another thread,
  and two contexts on one GPU.
- Most changes are behind default-off flags and preserve a rollback path.
- Existing scheduler, batcher injection, checkpoint profile, R2, and archive
  mechanisms are reused rather than replaced.
- Failure semantics generally distinguish infrastructure failure from miner
  rejection.
- Current `main` has a green CPU suite and green cross-box determinism checks.
- The combined implementation is running healthily and delivers the claimed
  order-of-magnitude proof-process improvement under contention.

### What needs improvement

- PR #191 was merged with a critical wrong-weights bug and an unsafe shared
  pipe; #192 corrected it before the feature flag was enabled, but #191 was not
  independently deployable.
- PR #195 passed CI while testing only the new discovery helper, not the
  discovery-to-resume workflow, and the intended production bug remains.
- PR #189's recorded GitHub CPU check was red because its new profile-override
  test conflicted with the immutable active profile. The current suite is
  green after later protocol work, so this was not a detached-runtime failure,
  but a validator-critical PR should not be merged with an unexplained red
  required check.
- GitHub records no formal approving reviews on #189, #191, #192, #193, or
  #195. Self-review with an AI tool found valuable issues, but it is not an
  independent merge gate.
- Several tests prove a helper or transport in isolation while the failures
  occurred between layers: bootstrap discovery to resume, parent revision label
  to child weights, and candidate revision to mirrored bytes.

Overall judgment:

> The architecture and performance work are good and should be kept. The
> current combined steady-state implementation is a major success. The merge
> discipline and end-to-end recovery testing are not yet strong enough for the
> checkpoint and proof authority paths.

For future validator-critical PRs, require:

1. green required checks before merge;
2. one independent approval;
3. a cross-component integration test for every recovery/cutover claim;
4. a shadow or intentional-restart production gate before enabling the flag;
5. a short post-deploy evidence note linked to the exact revision.

## 7. Revised implementation sequence

### Step 0: fix and prove restart coherence

This is the next PR.

Deliver:

- one shared checkpoint-title parser;
- end-to-end stale-resume correction test;
- fail-closed behavior when a known-newer checkpoint cannot load;
- a test that the first opened window, verify model, proof workers, signed
  manifest, and trainer run identity all name the same checkpoint;
- deliberate staging restart from a stale configured SHA;
- deliberate production restart only after staging passes.

Exit gate:

- the validator starts directly on the latest accepted checkpoint;
- checkpoint number never regresses in public state;
- no old-revision window is emitted during restart;
- full CPU and determinism checks are green.

### Step 1: soak the architecture already deployed

Collect at least 100 consecutive windows and two complete publication
intervals after Step 0. Preserve:

- seal interval p50/p95/p99;
- proof half p50/p95/p99;
- trainer cursor lag and payload-gap count;
- checkpoint stage and swap duration;
- proof decisions/reject distribution;
- memory and GPU high-water marks;
- API 5xx/499 counts and event-loop lag;
- one intentional proof-child restart;
- one intentional trainer restart in shadow or staging.

Do not add a second proof GPU unless this soak shows the 100-second window SLO
is missed. One isolated H100 currently finishes the proof half around 53
seconds and has ample steady-state headroom.

### Step 2: harden checkpoint proposal and activation

Deliver the HF-bound artifact manifest, exact-file verification, monotonic
transition rules, durable accepted manifest, parent/cursor lineage, and
candidate rejection telemetry.

This is required before a remote GPU can influence production checkpoint
authority.

### Step 3: turn the existing proof pool into an executor interface

Do not change `GlobalProofScheduler` or auction application logic.

~~~text
GlobalProofScheduler
  -> ProofExecutor.execute(immutable request)
       -> LocalProcessProofExecutor  # wraps Rom's current ProofWorkerPool
       -> RemoteProofExecutor        # added later
  <- immutable ProofKernelResult
  -> current local rank-ordered application
~~~

Add:

- explicit, versioned, non-pickle wire request/result schema;
- job ID, checkpoint revision, environment/profile digest, input digest,
  deadline, and attempt/lease generation;
- tokenizer and runtime handshake digest;
- worker-generation-safe lifecycle;
- local adapter parity against the current process path.

No NATS, Kafka, Temporal, Kubernetes, or provider code is needed in this step.

### Step 4: isolate public state reads

Emit the already-canonical state bytes atomically and let nginx or a tiny edge
service serve identity/gzip variants with ETag and `Vary: Accept-Encoding`.
Keep the existing dynamic route as one-switch rollback.

This is a small independent change that reduces control-plane traffic while the
remote worker path is developed.

### Step 5: remote proof shadow

Start with one H100-class worker because production parity is already known on
H100. A manually provisioned trusted host is the simplest transport benchmark.
Then test:

1. ordinary Lium H100 for lifecycle and failure shadow only;
2. Lium CVM H100 with Reliquary-verified attestation for authority eligibility;
3. a second provider through the same small lifecycle interface.

Local proof remains authoritative. Compare every `ProofKernelResult` field and
inject disconnect, delay, duplicate, wrong revision, wrong tokenizer digest,
worker replacement, and stale lease.

Only before remote authority, add the minimal SQLite/WAL job and lease journal
described in the main plan. Shadow traffic does not justify a queue cluster.

### Step 6: harden the existing detached trainer

Reuse its R2 journal and train runner. Add:

- control-owned fenced lease generation;
- exact-object short-lived input/output access;
- trusted control-side HF publication;
- provider-neutral launch adapter;
- explicit optimizer/RNG recovery policy;
- provider-loss and replay quality tests.

Treat the trainer as one warm, movable lease. Do not create one trainer per
window and do not run two independent canonical trainers.

### Step 7: finish the trust split

After the GPU data plane is stable:

- move checkpoint and weight operations behind a semantic hotkey signer;
- keep the trusted grader coordinator/comparator on control and move its
  gVisor worker pool to a wallet-free execution host;
- benchmark Firecracker behind the same sandbox-executor contract;
- remove provider, HF, R2, wallet, and CI credentials from execution workers.

## 8. What not to build now

Do not build:

- another detached-training implementation;
- a replacement proof scheduler;
- two proof processes on the same GPU;
- a second proof GPU before the current p95 demonstrates need;
- remote authoritative proof before checkpoint byte binding and leases;
- an ordinary non-CVM marketplace trainer with current master credentials;
- Kubernetes, Kafka, NATS, Temporal, Vault, or a service mesh;
- Firecracker as a claimed GPU speed optimization.

## 9. Immediate decision

The next development task should be named and scoped as:

> **Detached checkpoint startup coherence: one title parser, latest-revision
> fail-closed resume, and end-to-end restart proof.**

After that PR and its intentional restart test, the first new infrastructure
milestone is checkpoint artifact binding plus a versioned `ProofExecutor`
interface around Rom's already-working local worker. That is the shortest path
from today's 34.3 windows/hour architecture to secure, modular, disposable GPU
capacity without discarding the performance work already completed.
