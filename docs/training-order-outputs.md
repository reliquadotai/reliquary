# Training order measurements and checkpoint exports

The signed admin API serves a bounded, read-only view of each registered training task. The platform authenticates the customer and resolves its owned order to a task before calling these routes. The subnet retains its bucket credentials.

- `GET /admin/v1/training/profiles` returns `reliquary/training-profiles/v1`, the exact pinned Gumbel-max profiles and each profile's live RL environments.
- `GET /admin/v1/training/{task_id}/outputs` returns `reliquary/training-outputs/v1`, source observation timestamps in epoch seconds, latest checkpoint, measured history, checkpoint file manifests and explicit truncation flags. A task without a publication returns empty arrays and null source timestamps. A failed read returns 503.
- `GET /admin/v1/training/{task_id}/checkpoints/{version}/files/{filename}` returns a fresh 900-second HTTPS download URL, task/version/name, byte count and SHA-256. The filename must match an object in the retained checkpoint index. Paths, object keys and URLs supplied by a caller cannot select bucket objects.

The existing HMAC signature, nonce, prefix and registered-training-task checks apply. Query strings are rejected by the signed admin API. Retired tasks retain read access to their published outputs. These routes require `RELIQUARY_TRAINING_ORDERS=on`.

Completed optimizer steps append allowlisted finite numerical metrics to `rl/{task_id}/metrics.json` through the trainer's existing authenticated upload link. This history is independent of W&B configuration. A row is written after its checkpoint publication succeeds. It records its own completion time, and publication failures are logged without interrupting training. No prompts, generated text, hotkeys or credentials enter this document. The history keeps at most 2,000 checkpoints and 2 MiB. The combined admin output is at most 4 MiB and lists at most 512 files. Truncation flags identify retained coverage.

Checkpoints use the existing `rl/{task_id}/index.json`. New anchor/delta entries have publication timestamps. Older entries retain a null timestamp. A `full` export contains safetensors weights, their HF shard index, config and tokenizer files. Download every listed file into the same directory and verify sizes and SHA-256 values before loading it with Transformers or vLLM. Source shard indexes are rebuilt from the published tensor names, preserving the model's actual configs and tokenizer files.

The authenticated trainer config supplies the order's declared max_steps. That optimizer step publishes a full anchor directly, before adoption and possible credential revocation. On a successful trainer close, outstanding publications finish, then the last successfully published CPU snapshot is written as a full anchor if needed. Its version and checkpoint SHA remain the published identity. The live model is never read during this finalization, so a failed subsequent optimizer step cannot enter the export. The trainer link allows adding a latest-version anchor only while the committed index has no anchor for that version. Existing anchor and delta objects at adopted versions remain protected. The index is written last. A failed finalization leaves the existing delta export available, and a later retry can complete the anchor.

A `delta` export requires reconstruction. The existing `reliquary.miner.ckpt_sync.CheckpointSync` verifies the controller's signed `/rl/{task_id}/checkpoint` answer, obtains an anchor and required deltas, applies them bit-exactly, and verifies the target checkpoint SHA. Its `model_dir` option downloads the configs and tokenizer too. `reliquary.miner.rl_async.write_hf_dir(state, model_dir, destination)` then writes a loadable HF directory. The public signed index is needed for reconstruction planning. The dashboard lists retained artifacts and provides these reconstruction instructions for delta exports. An abrupt process termination may leave the latest checkpoint in this form.

Local verification uses fake stores, synthetic tensors and signed test requests, with no rented hardware or live mutations.

Roll out the matching trainer, controller, admin and signer code together. The strict remote signer schema accepts the new optional per-entry publication clock. Historical indexes keep their original signed bytes.

## Private W&B for customer training orders

Native outputs remain independent of W&B and do not require a W&B credential.
An opted-in new admin training request additionally supplies `order_id` (the
platform order ID) and `wandb: {entity, project, run_id}` together. These fields
are frozen in the task's training order and participate in its request digest.
A retry must send the same binding; changing it under an existing training ID
is refused. Requests without these fields retain their original legacy digest.

The run ID is `rl-` followed by the first 40 hex characters of SHA-256 of
`reliquary/private-training-wandb/v1\n{order_id}\n{task_id}`. The admin service
validates it against the actual derived task ID before creating a task. No
customer-supplied URL, credential or shared hotkey run selects this target.
The authenticated trainer config carries the frozen `order_id` and `wandb`
binding. These fields and W&B credentials are absent from the public trainer
routes; the latter are not mounted on the public listener.

`GET /admin/v1/training/profiles` advertises
`capabilities.private_training_wandb = "reliquary/private-training-wandb/v1"`.
It also returns `training_task_prefix`, the effective admin task prefix plus
`train-`, so the platform can verify its expected task identity before creation.
This establishes the controller's contract, not the installed trainer image.
Rebuild the managed trainer image from this code, qualify and record its digest,
configure the provider with a versioned tag that will not be moved, and check
the same writer-schema constant before starting the trainer. The current Lium
template accepts `repository:tag`, not an `@sha256` image reference. An old
image must not be used with private telemetry credentials.

Managed trainer pods opt in with `RELIQUARY_TRAINING_WANDB=on`, the exact
`RELIQUARY_ORDER_ID` and `RELIQUARY_TASK_ID`, and the frozen `WANDB_ENTITY` and
`WANDB_PROJECT`. A supplied `WANDB_RUN_ID` must equal the derived run ID.
`WANDB_API_KEY` in the pod is a dedicated writer credential for the authorized
private project; it is not the platform collector's reader credential and is
never passed to miners or browsers. Qualify the project's actual visibility
and credential permissions before activation. The current Miles trainer image
already installs the optional W&B SDK; a custom image must include it too.

Private initialization verifies the authenticated config against the pod
environment. Missing or inconsistent private identity disables private
telemetry and never falls back to the legacy shared run. Each qualified run
contains `reliquary_order_id`, `reliquary_task_id`, and
`reliquary_checkpoint_key = "async/published_checkpoint"` in its config.

For each completed update, the trainer captures an independent allowlisted
metric row with its original observation timestamp. Only successful checkpoint
publication releases that row to the W&B writer. Each row carries
`reliquary/order_id`, `reliquary/task_id` and the explicit
`async/published_checkpoint`. The pre-update `async/checkpoint` and W&B cursor
are not substitutes. A dedicated bounded best-effort queue keeps W&B network
latency and failures off the optimizer and checkpoint-publication threads.
Overflow or shutdown can leave gaps; native private outputs remain available.

Deploy matching platform and runtime branches, apply the platform's private
binding migration, configure its allowed project and separate reader/writer
secrets, then qualify one new order. Verify the adopted task identity, a
successfully published checkpoint, the W&B row, the platform collector and the
private owner read. Also verify cross-account denial and native checkpoint
downloads. CPU tests do not establish a live W&B publication or GPU-image
qualification, and no existing training is relabeled retroactively.

Local qualification on 2026-11-19 passed 253 trainer/publication/output tests
and 165 admin, signature, task-config and CLI regressions on CPU. The W&B 0.31.0
SDK accepted the writer's settings without opening a run. These checks use
isolated state and simulated services; no GPU was rented or live W&B run queried.
