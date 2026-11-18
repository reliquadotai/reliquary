# Bounded local training supervisor

`python -m reliquary.rl_orders.supervisor --config /srv/private/training-config.json`
supervises one operator-allocated local training task. It does not allocate GPUs, rent capacity, publish an axon,
or submit weights. Paid admission remains false until the complete runtime and billing policy are qualified.

## Read-only prerequisites

Before an authorized execution, inspect private local inputs with:

```bash
python -B -m reliquary.rl_orders.preflight --config /srv/private/training-config.json --registry /srv/private/registry-export.json
```

This command reads a local registry export, config, credential material, model and installed package metadata.
It does not fetch the registry, write a runtime journal, claim storage, bind ports, spawn children, import Torch,
initialize CUDA or ask the signer to sign. CUDA-build detection reads literals in the installed `torch/version.py`;
a CPU build is refused. A CUDA build does not establish a usable GPU, native kernel ABI or numerical correctness.
The trainer interpreter in the Miles image is CPU-only and must not be selected for this `single` backend.

The JSON report uses exit code 2 for blocked local prerequisites. Exit code 0 means only
`local_prerequisites_checked`: `qualified`, `launch_authorized` and `paid_admission` remain false. Live registry
authorization, exact-run signer permission, namespace arbitration, ingress, delegated cgroup behavior, GPU work,
durable owner delivery and nonpaying isolation remain explicitly unverified. Keep this report private.

`controller_grader.required_at_startup` uses the validator's check of every environment in the declared split.
The Qwen3.5-4B v10 profile includes a sandboxed code environment even when the order assigns all work to math.
On an isolated qualification host, the controller needs either its designated grader listening at
`/tmp/reliquary-grader.sock` or an installed `runsc` and grader bundle for its startup path.
The preflight reports this dependency without probing the socket or starting a service.

The preflight requires a positive explicit token cap as well as steps, a future deadline and the actual async
auction settings. Step and token caps close admission; they are not a hard cap on work already issued.
Its nominal process budgets exclude synchronous HTTP deadline overruns and post-stop checkpoint finalization.
Failed trainer exit can additionally consume one cleanup stop budget. The runtime state directory must be new or
empty except for a private regular supervisor lock; retained logs, receipts and checkpoint files require reconciliation.
`tests/unit/test_training_preflight.py` checks these boundaries using local synthetic files.

## Runtime requirements

The host must run Linux with the installed runtime package and a writable delegated cgroup v2 parent that supports
`cgroup.kill` and `cgroup.events`. Each invocation creates a new cgroup and two new process groups. A fixed child
bootstrap joins the cgroup before importing the controller or trainer. The managed backend is `single`; Miles,
Ray, distributed and external workers are unsupported. Controller grading uses the trainer audit link.
The registry order must retain its immutable quote ID, quote digest and supported billing basis. Legacy
declaration-only orders cannot be launched by this supervisor; synthetic nonpaying qualification orders can use
the same explicit binding without opening paid admission.

The private configuration has exactly the following fields; secret values live in a separate private JSON file:

```json
{
  "schema": "reliquary/training-supervisor/v1",
  "task_id": "order-train-example",
  "order_sha256": "<64 hexadecimal characters from the registry training order>",
  "profile_id": "<immutable registered profile>",
  "run_id": "order-train-example-<at least 16 unique lowercase hexadecimal characters>",
  "state_dir": "/srv/private/training/order-train-example",
  "credentials_file": "/srv/private/training-credentials.json",
  "model_dir": "/srv/models/qualified-snapshot",
  "model_manifest_file": "/srv/private/model-manifest.json",
  "model_manifest_sha256": "<SHA-256 of the exact manifest bytes>",
  "cgroup_parent": "/sys/fs/cgroup/allocated-training",
  "validator_hotkey": "<bound public validator address>",
  "controller_port": 18888,
  "trainer_port": 18889,
  "trainer_device": 0,
  "network": "test",
  "netuid": 1,
  "max_runtime_seconds": 3600,
  "start_seconds": 600,
  "drain_seconds": 600,
  "stop_seconds": 60
}
```

Operator files must be regular, owned by the invoking user and inaccessible to group/others. The task directory
must be private and end in the exact task ID. The supervisor takes an exclusive file lock and refuses any existing
`journal.json`, including a journal left before a child PID was recorded. It never adopts, restarts or signals
PIDs or cgroups recovered from an earlier journal. An uncertain previous runtime needs operator reconciliation.
Model, controller, trainer, home and cache paths remain separate from any existing production runtime.
Existing child state directories are also refused without a journal, so an orphaned SQLite database or trainer
state cannot be stamped with a new ownership epoch.

The credential file accepts only the storage fields `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`,
`R2_BUCKET_ID`, `R2_ENDPOINT_URL`, `R2_REGION`, the corresponding public-checkpoint fields
`R2_PUBLIC_ENDPOINT`, `R2_PUBLIC_ACCESS_KEY_ID`, `R2_PUBLIC_SECRET_ACCESS_KEY`, `R2_PUBLIC_BUCKET`, optional
`HF_TOKEN`, and the explicitly bound remote-signer fields `RELIQUARY_SIGNER_URL`, `RELIQUARY_SIGNER_CA`,
`RELIQUARY_SIGNER_CERT`, `RELIQUARY_SIGNER_KEY`, `RELIQUARY_SIGNER_EXPECTED_HOTKEY`. All signer fields are required.
The signer hotkey must match the configuration. Credentials never enter the journal or child command line.
Parent process settings such as `PYTHONPATH`, backend overrides, old run identifiers and old trainer tokens are
not forwarded. The operator pins a unique task-prefixed run ID already permitted by the checkpoint signer; the
supervisor refuses any existing object in that namespace and never changes signer configuration. A bounded
prefix read is followed by a conditional create-only `runtime-claim.json`, bound to task/order/run/runtime,
before either child launches. This arbitrates concurrent starts on different hosts. An ambiguous write is
reconciled against the exact retained claim; unknown or conflicting outcomes launch nothing.
It generates a fresh trainer token and runtime nonce, then binds the exact task-specific run namespace,
loopback listeners and private state directories. Outage-fill pay and background HF backups are disabled.

The model manifest is a private JSON object with schema `reliquary/local-model/v1`, `profile_id`, `model_id`,
`revision`, and a `files` map from relative filenames to SHA-256 digests. Profile/model/revision must match the
registered immutable profile. The complete materialized file inventory, including `config.json` and safetensors,
is verified before launch. Symlinks, changed or unlisted files and moving HF repository aliases are refused.
Both children receive the same verified local model directory. This consistency check does not replace GPU or
numerical qualification of that snapshot.

The supervisor accepts readiness only from the authenticated private trainer listener's fresh
`reliquary/training-runtime/v1` observation. Task ID, core order digest, run ID and random runtime ID must match
exactly. A PID or listening port alone is insufficient. The startup deadline, shorter of the order deadline and
configured run duration, drain deadline and termination budget are finite. Loss of task-bound readiness closes
the run. SIGTERM to the supervisor invokes bounded owned cleanup; a crash retains the reconciliation boundary.

At the run limit or observed admission closure, the supervisor requests the exact task-bound drain. The trainer
must join reports, prefetch, publications and heartbeat work, acknowledge quiescence and exit normally. The
controller must then prove complete drainage, including canonical closure of partial buffers and audits.
The supervisor waits for normal trainer exit, stops the controller, reaps both owned leaders and verifies that
their process groups and the complete owned cgroup are empty. Only then is `StoppedRuntime` produced. Forced
or uncertain termination cannot produce a successful final witness.

The managed controller handles Uvicorn's replayed SIGTERM outside its server signal scope. It returns normally
only after its exact runtime proves durable drainage and every controller cleanup stage completes. An unproven
drain or failed cleanup keeps a nonzero exit; signal return codes are never relabelled as successful witnesses.
Other validator modes retain their existing signal behavior.

`journal.json` is atomically replaced and fsynced before/after launch and each transition. Runtime observations
keep its running state fresh. `read_runtime_view(root, task_id)` exposes only scoped identities, state,
timestamps and checkpoint number; stale or malformed observations map to unknown. Process identities,
credential paths and local infrastructure details remain private.

An execution that stopped is not automatically delivered. The fixed finalization pipeline
`reliquary.rl_orders.finalize.finalize_training` must freeze and verify both usage and the checkpoint delivery
before the journal becomes completed. It returns the two immutable digests; the supervisor never loads a
user-selected module, shell command or argument vector. A finalization failure retains reconciliation state.
The CLI does not resume an existing journal, even when execution already stopped and only finalization failed.
An operator must reconcile the retained process and immutable-artifact evidence; replaying finalization, where
qualified evidence permits it, must not relaunch children or adopt stored PIDs. Automated finalizer replay after
a supervisor restart is not provided by this command.
See [checkpoint delivery](training-checkpoint-delivery.md) for the verified anchor/delta archive and owned download
contract; an index location alone is not delivery.
The focused tests use fake processes and storage only: `pytest tests/unit/test_rl_training_supervisor.py`.
They establish neither a running GPU service nor deployment, approved pricing or financial settlement.

Final usage names its billing metrics explicitly: `gross_matched_token_equivalents_v1` rounds immutable gross
accrual tokens upward and includes weighted first scans; `owned_process_elapsed_ms_v1` rounds the owned trainer's
monotonic process lifetime upward, including model load and idle time. These are customer ceiling/time billing
metrics, not actual miner USD cost or GPU kernel time. Their policy and real Linux cgroup/GPU behavior still need
qualification before paid admission can open.
