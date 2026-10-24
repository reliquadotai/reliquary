# Frozen V9 scoring compute

The scoring entry point prepares, runs, validates and exports one frozen V9
gradient-dot unit. It consumes an operator-approved checkpoint, document slice,
seed, task metadata and precomputed reference generations. It neither generates
new reference samples nor signs or publishes scores. There is no optimizer step,
miner registration, reward import or chain writer in this path.

This is a historical replay contract. Compatibility with another scoring version
requires a new source review and qualification. A successful replay does not
establish chain provenance, numerical correctness, upstream acceptance or rewards.

## CPU rehearsal

Use Python 3.12 with the project's NumPy dependency installed. All paths below
must be absolute, owner-only and outside Git repositories.

```sh
reliquary affine scoring rehearse --out "$PRIVATE_REHEARSAL"
reliquary affine scoring check --bundle "$PRIVATE_REHEARSAL/bundle"
reliquary affine scoring verify --bundle "$PRIVATE_REHEARSAL/bundle" \
  --workspace "$PRIVATE_REHEARSAL/work"
```

The rehearsal creates synthetic inputs, runs a harmless child, validates the native
output shape and exports `scoring-result.zip`. Its assurance is always `fixture`;
it runs no model or gradients and cannot establish GPU qualification. The report
leaves API delivery and numerical comparison unverified.

## Prepare a native unit

The owner-private preparation JSON has exactly these fields:

| Field | Meaning |
| --- | --- |
| `source` | Absolute directory containing only the two reviewed compute files |
| `inputs` | Absolute frozen input directory described below |
| `checkpointInventory` | Absolute JSON file mapping checkpoint filenames to SHA-256 and byte size |
| `runtime` | Python `3.12`, exact package versions, GPU name/SM and upstream revision |
| `limits` | Execution seconds, finalization reserve, log, output and workspace byte bounds |
| `epoch` | Frozen nonnegative integer epoch |
| `baseCheckpoint` | Full 64-character checkpoint identity |

The reviewed revision is `8fa5c14caf47cda012928000aedb991e4ee33fd4`.
Preparation admits only `grad_docs_v9.py` with SHA-256
`d9030c4d8408fab8a79312cd2912a66f6fe077311ae430e9d05fb85e93325e1f`
and `v9common.py` with SHA-256
`014889c11c97022ee498879fc0abee50c4981379cb39dc599917e4c21facd1c6`.
No scorer driver or signing key is staged. These reviewed hashes do not constitute
an upstream signed admission receipt.

Inputs contain `seed.json`, `docs.jsonl.gz`, `todo.json`, `tasks.json`, and matching
`ref/gen-i.json` / `ref/gen-i.npz` reference files. Validation checks the frozen
bindings, bounded integer NPZ arrays without pickle loading, native chunk order,
document coverage and eight rollout lengths per document. The checkpoint uses
offline safetensors and configuration files; Python and pickle weight files are
refused. The operator supplies and approves the binding between these assets.

The runtime requires exact versions of at least `torch`, `numpy`, `transformers`,
`safetensors` and `tokenizers`, GPU `NVIDIA H200` and SM `[9, 0]`. Package pins
describe a candidate runtime; they are not hardware qualification.

`docker/Dockerfile.affine-scoring` prepares a candidate from existing project
Python, Torch and Transformers pins. Its CPU CI checks offline imports and the
unprivileged fixed-path layout. Build the CUDA variant, record its immutable image
digest and installed package versions, and bind those to the approved profile/job
before hardware qualification. The CPU build is not numerical or CUDA validation.

Limits contain `maxSeconds`, `finalizationReserveSeconds`, `maxLogBytes`,
`maxOutputBytes` and `maxWorkspaceBytes`. The execution deadline reserves time
for cleanup and delivery. An independent host allocation deadline and disk quota
must also be qualified; the process watcher alone cannot impose a disk quota.

```sh
reliquary affine scoring prepare --config "$PRIVATE_CONFIG" --out "$PRIVATE_BUNDLE"
reliquary affine scoring check --bundle "$PRIVATE_BUNDLE" --checkpoint "$PRIVATE_CHECKPOINT"
```

Preparation freezes source, input, runtime and checkpoint inventories into
`job.json`. Every job and inventory has a canonical digest. Preparation never
overwrites an existing bundle. `check` revalidates the copied source/input bytes;
checkpoint bytes are checked only when `--checkpoint` is supplied.

## Bounded execution and delivery

`native_mounts(bundle, workspace, checkpoint)` returns the fixed mount plan for
the unchanged upstream paths. Source, checkpoint, tasks and each frozen input are
read-only; only the private native output directory is writable. The image must
make those paths traversable by its unprivileged worker UID. Run it without
network access, host credentials or extra operational mounts. The Platform host
adapter owns the container, image pin, allocation deadline and teardown.

The MVP supports shard `0` of `1`, the last four MLP layers and a bounded document
slice. The unchanged program still computes the full frozen reference gradient
pool; reducing the document slice does not bound that GPU memory requirement.
Independent multi-shard workers would need shared reference state and are refused.

Inside the qualified container:

```sh
reliquary affine scoring run --bundle "$PRIVATE_BUNDLE" \
  --workspace "$PRIVATE_WORKSPACE" --checkpoint "$PRIVATE_CHECKPOINT" --max-seconds 3600
reliquary affine scoring export --bundle "$PRIVATE_BUNDLE" \
  --workspace "$PRIVATE_WORKSPACE" --out "$PRIVATE_RESULT_ZIP"
```

The runner bounds time, logs and observed workspace size, handles cancellation,
and cleans the child process group before writing a terminal journal. Exit zero
is only `process_exited`. Exact native output validation and a bound compute
receipt are required for `compute_completed`. The export contains exactly
`compute-result.json` and `dots-0.json`; reference-gradient caches are not delivered.
The receipt exposes sanitized counts and hashes, with operator acceptance, native
reward verification and chain transactions explicitly false.

`compare --bundle "$PRIVATE_BUNDLE" --workspace "$PRIVATE_WORKSPACE" --reference
"$PRIVATE_ORACLE" --atol 0 --rtol 0` compares every validated dot against a frozen
operator-provided reference and reports mismatch counts and maximum absolute
error. Supply explicitly agreed finite tolerances for hardware comparison. This
does not authenticate the oracle or change the compute receipt's acceptance flags.

A completed workspace is verified on retry. Other previous attempts block reuse.
`reconcile --workspace "$PRIVATE_WORKSPACE"` confirms that its recorded process
group is absent; it does not kill processes or authorize reuse. Retry with a new
workspace and preserve the old failure record.

Platform delivery additionally requires the existing owner-bound profile,
private request, quote/hold, targeted executor lease, signed compute receipt,
stored ZIP verification and owner download. A local ZIP or child exit alone is
not a delivered service. Fixture profiles remain unpublished. Native profile
publication requires exact-image GPU replay, numerical comparison, isolation,
cancellation, deadline, crash recovery and artifact-delivery qualification.

```sh
python -m pytest --noconftest tests/unit/test_affine_scoring_contract.py \
  tests/unit/test_affine_scoring.py -q
```
