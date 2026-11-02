# Bounded retail video experiments

The retail task uses the existing private vision worker and
`timestamped_frames_v1` representation. A recording contains reviewed PNG
frames with integer presentation timestamps. The commands below prepare causal
prefix windows, export train/validation data, run local inference and evaluate
the separate final test. They do not provision hardware, activate an execution
profile, submit a subnet task or trigger a business action.

This is a sequence of timestamped images, not a native video decoder. Extract
frames from an authorized recording before preparing this format, preserving
the original presentation timestamps. Frame cadence, resolution and context
must be qualified together with the model and processor. Sparse sampling can
miss an event even when all supplied timestamps are correct.

## Reviewed input

The input JSON has exactly `schema_version: 1`, `labels` and `recordings`.
Labels are distinct ASCII event identifiers matching
`[A-Za-z][A-Za-z0-9_.:-]{0,63}`. Each recording contains:

| Field | Contract |
| --- | --- |
| `id`, `station_id`, `camera_id` | Canonical identifiers, at most 96 characters. |
| `day` | ISO date defining the camera clock domain. |
| `split` | `train`, `validation` or `test`. Stations cannot cross splits. |
| `source_sha256` | Declared original-source hash; preparation verifies frame bytes, not the provenance of the original recording. |
| `rights` | Exactly `basis` (`owned`, `licensed` or `synthetic`), nonempty `reference` and `approved: true`. This is a caller declaration, not a licence grant. |
| `annotation_complete` | Must be `true`; unknown footage is not a negative example. |
| `media` | Existing frames contract: `kind: "frames"`, `clip_start_us`, `duration_us` and PNG `frames`. No `score_fps`. |
| `events` | Reviewed objects with distinct `id`, approved `label`, `start_us` and `end_us`. Intervals are clip-relative, nonempty and inside the recording. |
| `cutoffs_us` | Strictly increasing prefix decision times, ending at the recording duration. |

`clip_start_us` is the capture offset within the declared camera/day. Several
clips from that camera/day are permitted only when their complete capture
intervals are disjoint. Clips are bounded to 60 seconds and 64 input frames;
use disjoint clips to accumulate normal-footage exposure. The existing PNG,
pixel, byte, source-split and identical-image split checks also apply.

A frame uses either a verified relative `path` or inline `data_base64`, plus
`sha256`, `width`, `height` and clip-relative `pts_us`. Preparation materializes
and verifies these bytes. Outputs require a new directory under an existing
owner-private parent outside Git repositories; previous attempts are preserved.

## Prepare windows

```sh
python -m reliquary.vision.retail prepare recordings.json "$PRIVATE_WORK/prepared" \
  --max-frames 4
```

For each cutoff, only frames strictly before it are eligible. Selection is
deterministic, retains the latest observed frame and obeys the requested frame
bound. Model prompts contain the vocabulary and output contract, not event
annotations, verified-negative flags or split labels.

The command writes:

- `training.jsonl` and `training/manifest.json`: ordinary vision-worker data
  from train and validation stations only. Windows intersecting an unfinished
  annotated event are omitted from this export.
- `requests.private.jsonl`: final-test model requests without targets.
- `benchmark.private.json`: the separate frozen test annotations and media.
- `preparation.private.json`: counts, hashes and the explicit assurance limits.

Keep the final benchmark outside all training jobs and parameter selection.
The existing worker evaluates both validation and test rows when they are
supplied to it; this export deliberately supplies no final-test rows.

For a first evaluation with test recordings only, add `--evaluation-only` to
`prepare`. It writes the benchmark and inference requests without a training
dataset. Mixed splits are refused in that mode; test footage must never be
relabelled as training data to make preparation succeed.

The new training target is:

```json
{
  "type": "retail_events",
  "items": [
    {"label": "zone_presence", "start_us": 1000000, "end_us": 2000000, "confidence": 0.9}
  ]
}
```

There are at most 64 items. Confidence is finite in `[0,1]`; timestamps are
integer microseconds with `0 <= start_us < end_us <= duration_us`. An empty
training target requires the existing explicit verified-negative flag. The
per-window training reward is a class-aware interval-overlap diagnostic with
missing/extra penalties; it is not a camera-hour service metric or Score's
football evaluator.

## Local inference

Use an already reviewed local snapshot locked by `vision lock-model`, as
described in [vision training](vision-training.md). The loader verifies the
snapshot bytes, uses local safetensors and disables remote model code. The lock
does not authenticate the snapshot's original download or approve a runtime.

```sh
python -m reliquary.vision.retail infer \
  "$PRIVATE_WORK/prepared/benchmark.private.json" "$MODEL_SNAPSHOT" \
  "$PRIVATE_WORK/base-inference" --device cpu \
  --max-context-tokens 8192 --max-new-tokens 128 --max-seconds 300
```

The device must be explicit. CUDA requires a separately qualified free host;
this command never rents one. It reuses the existing VLM generation runtime
without a training update. Ground-truth metadata stays local and is not passed
to the processor's model conversation. Inference writes a model-bound
`predictions.private.json` and a separate `inference.private.json` receipt.

To evaluate a delivered retail adapter, supply `--checkpoint` and its exact
`--checkpoint-sha256`. The existing loader verifies the archive, base model and
entire visual contract. Pixel/context/frame limits must match that adapter's
contract. A football or grounding adapter cannot be substituted for a retail
adapter. No archive is executed as code.

The time limit is cooperative, including generation stopping checks. Model
loading and individual native operations do not have a hard host deadline;
the operator/container must provide that boundary before service activation.
Per-window latency includes encoding and generation, and excludes model
loading, transport and scheduling. It is not an end-to-end API latency claim.

## Evaluate saved predictions

```sh
python -m reliquary.vision.retail evaluate \
  "$PRIVATE_WORK/prepared/benchmark.private.json" \
  "$PRIVATE_WORK/base-inference/predictions.private.json" \
  "$PRIVATE_WORK/report.private.json" --confidence 0.5 --temporal-iou 0.5
```

Other inference clients can supply the same envelope: exactly
`schema: "retail-video-predictions/v1"`, `benchmark_sha256`, `model` and
`predictions`. Model identity contains `id`, immutable 40-character `revision`
and `processor_revision`. Each prediction contains `window_id`, `status`
(`succeeded` or `failed`), `completion` and measured `latency_ms`.
The window ID comes from preparation. The hash must bind the exact benchmark
bytes. Saved identity remains caller-declared unless separately verified.

The report includes:

- Class-aware maximum-cardinality interval matching, precision and recall.
- Repeated-window alert deduplication at the frozen temporal IoU threshold.
  Duplicate predictions inside a single window remain separate alerts.
- False alerts per successfully processed camera-hour. Overlapping prefix
  windows contribute their union, not their summed lengths.
- Physical-time coverage and successful-window coverage separately. A later
  successful prefix can cover an earlier failed prefix's time without making
  that failed decision successful.
- First matched alert delay, including cutoff and measured processing time,
  plus processing-latency p50/p95.
- Explicit missing, failed, truncated or invalid inference. None is counted
  as a successful negative; an incomplete run remains `complete: false`.

Freeze thresholds before final-test evaluation. A fixed confidence threshold is
not automatically a commercially acceptable false-alert budget. Synthetic
fixtures test accounting and orchestration only. Local reports retain
`customer_acceptance: false` and `gpu_qualification: "not_run"`.

## Checks

```sh
python -m pytest --noconftest tests/unit/test_vision* -q
```

Tests cover causal prefixes, split isolation, source/media hashes, closed-event
exports, missing inference, alert deduplication, interval matching and exposure.
An offline random tiny Qwen3-VL fixture exercises real two-frame LoRA training,
checkpoint delivery/reload and local generation. It does not measure pretrained
model quality, realistic station footage, GPU throughput or customer acceptance.
