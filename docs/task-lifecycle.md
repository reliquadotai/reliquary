# Corpus task admission and executor recovery

Corpus tasks have an economic cap, a terminal registry status, and an
independent admission state. Setting the cap to zero leaves admission open.
Pausing keeps the task active and preserves its cap, immutable job manifest,
generation contract, and contract hash. Retirement closes admission permanently.

```sh
reliquary tasks pause --task-id TASK_ID
reliquary tasks resume --task-id TASK_ID
reliquary tasks list
```

The signed administrator API provides the same operation:
`POST /admin/v1/tasks/{task_id}/admission` with
`{"admission":"paused"}` or `{"admission":"open"}`. Its acknowledgement
describes the persisted registry state. The matching controller applies that
state during its registry refresh; its corpus job status exposes the observed
`admission` separately from job progress. The administrator task catalog exposes
`admission_controls_supported`. Consumers should check that capability and the
controller's observed state before presenting a pause as effective.

Paused jobs retain readable manifests, cursors, and status. New next, submit,
and skip requests receive `409 job_paused`; requests already admitted finish.
Audit, grading, and settlement continue draining accepted submissions. The
shipped miner retries a paused request with its existing bounded backoff and
continues after resume. Retirement continues to return `410 job_retired`.
Legacy registry entries default to open, and rendering an open registry entry
keeps its historical bytes. Admission does not change the generation contract.
Admission intent is authoritative in a separate bounded CAS object, bound to
the task's job and profile digest. Registry reads overlay that intent. A legacy
economic writer can therefore rewrite caps without dropping a pending pause;
pause and resume do not change the economic registry object.

The corpus controller journals actual remote audit and grading attempts with
conditional object writes. An attempt binds its input digest, pinned runtime,
executor registration, opaque lease nonce, expiry, and monotonic generation.
Snapshots are versioned and bounded; the attempt head keeps at most 32 prior
generation summaries. The existing admitted record store reconstructs work
after restart. Compatible reconstructed work can retain its live lease or
reconcile a result received before interruption. Expired or superseded results
cannot change the new generation. Compact nonce receipts make repeated result
delivery idempotent, including after a grading continuation acquires a new lease.

Recovered grading votes retain their provider separation and recheck draw.
Changed or revoked executor registrations invalidate their retained votes.
An audit acceptance awaiting a trusted local recheck restarts that recheck;
its executor acknowledgement does not become a proof verdict. Verdict and
settlement stores remain authoritative, and a completed attempt does not imply
payment or delivery. Executor protocol messages remain unchanged.

This recovery layer assumes one active controller per workload namespace.
Conditional attempt writes and generation fencing do not qualify concurrent
controller failover, automatic provisioning, or execution of a new workload
engine. Production release still requires a compatible runtime, preserved
admitted backlog, and live qualification.

An operator generation control can share an already running GPU scorer without
restarting its original supervisor or judges. It owns only
`<admin task prefix>gen-ops-` jobs, on its configured checkpoint and proof. The
normal corpus control continues excluding generation-order ids. This control
refuses evaluation, other generation orders, and episode jobs; routing and
deployment must give each job exactly one owner. Its runtime contract exposes
`execution_scope: "generation-operations"` and `durable_executor_attempts`.

```sh
reliquary corpus generation-control --task-id TASK_ID \
  --checkpoint-dir CHECKPOINT_DIRECTORY --gpu-run-dir GPU_SOCKET_DIRECTORY
```

The command checks the pinned checkpoint, live scorer model, initial task
contracts, and scoped submission manifests before migrating any ledger. It
can start without initial tasks and add real tasks through registry refresh;
retired generation tasks recover to finish their existing audits and settlement.
It loads a CPU tokenizer and uses the existing scorer for local audit fallback and
trusted rechecks. Remote audit workers retain the existing claim/result protocol
and durable attempt journal. GPU scoring still uses the existing bounded queue;
this mode does not add GPU capacity or qualify concurrent ownership of one job.
