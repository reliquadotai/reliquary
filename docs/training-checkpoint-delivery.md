# Owned training checkpoint delivery

`reliquary.rl_orders.checkpoint_delivery` freezes the existing asynchronous RL checkpoint
publication into a private local delivery cache. The cache is separate from the mutable miner
transport index and its pruning lifecycle. It does not provision a runtime, start training,
qualify a GPU or authorize a paid order.

The supervisor first proves termination of its owned controller and trainer. Final usage binds
the task, order, runtime, run and exact adopted checkpoint index. The delivery store requires
that validated usage envelope and the same stopped-runtime witness. It resolves a complete
anchor-and-lossless-delta path, requires model configuration and tokenizer files, and streams
every required object into its private content-addressed cache. Both byte size and SHA-256
must match the publication. The source index must still match after copying.

Only then is the immutable delivery manifest published. A replay accepts exactly the same
usage and checkpoint binding and re-verifies the cached objects. A publication, cache or
identity mismatch leaves no partial manifest. The caller publishes usage and completion only
after the delivery freeze succeeds.

## Manifest

The `reliquary/training-checkpoint/v1` manifest includes:

- `task_id`, `training_id`, `order_sha256`, `run_id` and `runtime_id`;
- `usage_sha256`, which binds the final usage envelope;
- `checkpoint.version` and `checkpoint.sha256`, the adopted version and aggregate tensor-state
  checksum from the existing checkpoint index;
- `index_sha256`, the canonical SHA-256 of the original strict source index;
- `files`, with opaque `artifact_id`, safe file name, kind, exact size, content type, whole-file
  SHA-256 and a complete contiguous list of 8 MiB chunk hashes;
- `manifest_sha256`, the canonical SHA-256 of the manifest with that field omitted.

Kinds are `weights`, `delta`, `model` and `index`. There is at least one full safetensors weight
anchor and exactly one `checkpoint-index.json` artifact. That reconstruction index contains
the selected anchor, its model files, and all lossless deltas needed to reach the final version.
Its keys are logical `objects/<artifact_id>` references, never private object-storage keys.
Only the supported model/tokenizer configuration names are exported; unrelated plain files
carried by the source publisher are omitted from the customer reconstruction index.
The manifest and reconstruction index carry no presigned URLs, credentials or local paths.
They are service delivery records, not a replacement miner-hotkey signature.

The supported reconstruction uses the existing `plan_download(index, None)`,
`load_anchor_shards`, `decode_delta`, `apply_delta` and `verify_checkpoint` functions. Download
each logical object using its listed artifact ID, verify its full file hash, load the anchor
shards, apply the listed deltas in order, then verify the aggregate final tensor-state hash.
The model files remain necessary alongside those weights. A tensor-state checksum is distinct
from an individual safetensors file hash or the index and manifest hashes.

## Fixed signed-admin corridor

The service mounts `GET /admin/v1/training/{task_id}/checkpoint` for the final manifest and
`GET /admin/v1/training/{task_id}/checkpoint/files/{artifact_id}` for bytes. It exposes them
only after the completion record binds both final usage and this manifest. Platform performs
owner authorization before invoking that signed-admin corridor.

The file route selects only an artifact already listed in that task's manifest. It accepts one
explicit `Range: bytes=start-end`, with at most 16 MiB per answer. Omit Range only for files
no larger than 16 MiB. Requesting each listed 8 MiB chunk is the simplest way to verify and
assemble a large checkpoint without buffering a model in memory. Arbitrary keys, URLs,
suffix/multiple ranges and paths are unsupported.

Before returning any range, the store verifies every cached chunk intersecting it. A truncated
file, changed chunk, invalid manifest, symlink, nonprivate cache or unknown artifact fails
closed. Reads do not revisit the mutable source index. Availability of the manifest means
verified checkpoint bytes are available; it does not prove that a customer has downloaded
them or accepted a training result.

## Bounds and verification

The source index and manifest each have a 2 MiB limit. A delivery has at most 128 files,
16 GiB per file, 64 GiB total and 16,384 chunks. Copying checks its configured deadline between
bounded source reads; the operator's R2 client retains its transport timeouts. The local cache
and its task directories require private ownership and permissions. Blob and manifest
publication use immutable atomic links and filesystem synchronization.

`tests/unit/test_rl_training_checkpoint_delivery.py` reconstructs an actual tiny safetensors
checkpoint through lossless deltas and tests source pruning, mismatched final usage,
missing/tampered publications, index changes during copying, ranges over a large synthetic
anchor, chunk corruption, manifest conflicts, private-file boundaries and copy deadlines.
These local CPU fixtures establish the delivery contract; real GPU execution and service
qualification remain separate.

## Orders run on rented pods

An order that runs on a rented pod, with Miles or any other trainer backend, cannot give the single-process stop
proof above. `reliquary.rl_orders.pod_delivery` delivers its model on different evidence: the task is retired, its
bill is final, its controller has stopped, and no pod of the task still holds a usable token. The checkpoint version
is the one the final spend summary closed on. A trainer stopped between publishing a version and its adoption leaves
the index one version ahead, so the delivery rebuilds the billed version from the anchors and deltas still listed.

The objects are streamed to the private bucket under `reliquary/rl-delivery/<task>/`, in 8 MiB parts, without using the
control host's disk. Each one is hashed against the publication, and `manifest.json` is created last and only once.
Its schema is `reliquary/pod-training-checkpoint/v1`: the fields above without `runtime_id` and `usage_sha256`.
A delivery can reach 128 GiB, what the manifest's chunk list can describe, because the anchor of a 27B model alone is
about 50 GiB. An order whose final checkpoint is version zero, or none, trained nothing and has nothing to deliver.

The controller supervisor runs one copy at a time beside its tick and retries a failure with a growing delay. A retry
keeps every object already stored whole, after hashing it again. The
platform holds the order's bill until the manifest is available. The signed admin checkpoint routes serve these orders
from the private bucket, and every range read verifies its chunks.
