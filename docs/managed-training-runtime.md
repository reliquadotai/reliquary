# Managed local training runtime

Training declarations can now be executed by the bounded, operator-run
[supervisor](managed-training-supervisor.md). It starts a controller and one local trainer on
explicitly allocated Linux resources. It does not rent capacity, submit weights, modify the signer,
or enable a paid service. All training capability flags other than declaration remain false until
real qualification. This implementation has been tested with CPU fixtures and simulated process
ownership, not a real GPU training run.

## Bound lifecycle

1. A new declaration pins `quote_id`, opaque `quote_sha256` and the versioned `billing_basis` in
   its immutable order digest. Missing new fields preserve legacy replay hashes; legacy declarations
   cannot acquire automatic billing terms retroactively. The public front's controller supervisor
   only starts declarations without these quote bindings. A bound declaration needs its own
   operator-owned controller, trainer, and independently qualified miner ingress.
2. The supervisor verifies the immutable local model, exact task/profile/order/run, private
   credentials, unused checkpoint namespace and Linux cgroup allocation before launching fixed commands.
   Before loading weights, the managed trainer requires its controller to report the same pinned catalogue profile.
   Admission stays closed until the initialized trainer has published its initial checkpoint and
   acknowledged the exact runtime identity on the private trainer listener.
3. Registry retirement and order limits close admission. A separate one-second controller check
   enforces limits even when no miner or supervisor polls. The trainer stops taking new work, resolves
   incomplete held batches without payment or sanction, and joins reports, publication, prefetch and
   heartbeat before its task-bound quiescent acknowledgement.
4. Quiescence is distinct from full drainage. Existing leases may finish during a 60-second controller
   monotonic grace. Remaining non-submitting leases are then released neutrally. Late queued groups and
   audits are explicitly resolved without inventing trained work. Active submissions and unresolved
   accounting keep drainage false.
5. The supervisor requires normal child exits, empty owned process groups and an empty owned cgroup.
   Forced termination, reused process identity, unknown cleanup or an old journal requires reconciliation.
   A PID or a running process alone establishes no readiness or finality.
   The managed controller returns normally after SIGTERM only with exact durable drain evidence and successful
   cleanup; an undrained or failed cleanup remains a nonzero exit. Legacy validator signal handling is preserved.

## Final usage and checkpoint

The fixed `finalize_training` callback reads the stopped controller database in one read-only
transaction. It requires the exact task/run/runtime, a caught-up ledger, no unresolved work, immutable
gross consumption and billable events inside that owned runtime's time interval. Gross token
equivalents are summed exactly and rounded up once; penalties and forfeiture do not reduce consumed
work. [Final usage](training-final-usage.md) is an integer-only, canonically hashed receipt.

`trainer_active_ms` uses the explicitly named `owned_process_elapsed_ms_v1` metric: trainer process
elapsed time, including model loading and idle time. It is not GPU compute time. The proposed new-quote
customer basis prices measured tokens at the frozen quote ceiling and measured process time at the
frozen trainer rate. Platform separately caps rollout spend at the quoted budget, trainer time at the
quoted deadline, and the total at the reservation. Margin and refund calculations use integer micro-USD.
This is distinct from actual auction/provider costs, which retain independent finance evidence and review.
The owner must approve these inactive draft terms, including zero-step outcomes and the lease grace,
before any paid activation.

The [checkpoint delivery store](training-checkpoint-delivery.md) copies the selected lossless
anchor/delta/model reconstruction into a private immutable cache and verifies the actual object bytes.
It binds that manifest to the exact usage, task, order, run and runtime. Only after both proofs exist
does `completion.json` become visible. A copy or write failure cannot publish a billable completion.
The fixed finalizer can replay a partial post-copy publication from verified cached bytes without a
live source index. Completed replays verify those bytes, the original termination commitment and the
checkpoint version. They do not probe process IDs that the operating system may have reused. The supervisor CLI does not automatically replay or relaunch an existing journal.

## Signed admin reads

The admin service can use the private `RELIQUARY_TRAINING_RUNTIME_ROOT`, shared with the supervisor
under the same protected service identity. It exposes only prefix-scoped, freshly observed runtime
state and validated final receipts:

- `GET /admin/v1/training/{task_id}`: declaration, runtime state and `final_usage`, initially null.
- `GET /admin/v1/training/{task_id}/checkpoint`: immutable final manifest after completion validation.
- `GET /admin/v1/training/{task_id}/checkpoint/files/{artifact_id}`: verified bytes, at most 16 MiB,
  with an optional explicit single `Range: bytes=start-end` and returned checksum.

Existing fresh HMAC and nonce verification applies to every read. Files are selected only by manifest
artifact ID; requests never accept filesystem paths, storage keys or arbitrary URLs. The customer owner
boundary is enforced by Platform. Source keys, process IDs, credentials and private runtime paths are
omitted. Cache corruption refuses delivery. Platform retires the native registry task before sealing
its terminal order and retains independent finance proposal/approval; final usage never posts money.

## Verification and activation boundary

The focused suites are `test_rl_training_supervisor.py`, `test_rl_training_usage.py`,
`test_rl_training_checkpoint_delivery.py`, `test_training_runtime_routes.py` and
`test_training_final_pipeline.py`, alongside the existing trainer, lease, ledger, admin and HTTP suites.
They cover actual local SQLite drainage and signed admin ranged delivery, and reconstruct a small
safetensors checkpoint through real lossless deltas. OS ownership and GPU work remain simulated.

Real operation still needs an operator-allocated Linux/GPU host with delegated cgroup v2, the qualified
materialized model, an index signer authorizing the exact prequalified run, bounded ingress, and a
durable private cache readable by the admin identity. No such resource qualification, deployment,
transaction or paid run is established by these tests.
