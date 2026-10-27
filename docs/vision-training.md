# Bounded vision training worker

The `reliquary.vision` package executes private SFT or GRPO LoRA jobs through the
platform's existing owned-dataset, approved-profile and executor-lease contract.
It includes input validation, two task rewards, model execution, held-out
evaluation and portable adapter delivery. It does not register a native subnet
task or extend the text proof protocol.

## Runtime

Prepare a dedicated Python 3.12 environment using the
[trusted bootstrap steps](../README.md#local-development), then install the
optional runtime:

```sh
PIP_USE_DEPRECATED=legacy-certs python -m pip install '.[vision]'
python scripts/bootstrap_python_trust.py
python -m reliquary.vision --help
```

The pinned Torch 2.7 / torchvision 0.22 pair is intentional. CPU uses float32;
one CUDA GPU uses BF16. Multi-GPU and quantized training are not implemented.
Supported architecture mappings are SmolVLM, Qwen2.5-VL and Qwen3-VL. A mapping
does not qualify a particular model, accelerator, context length or runtime image.
`trust_remote_code` is disabled and execution loads local safetensors only.

`docker/Dockerfile.vision-training` defines the image build. After a relevant
merge to `main`, the vision workflow publishes a CUDA candidate at
`ghcr.io/reliquadotai/vision-training:sha-<full source commit>`. Publication waits
for the CPU contract and training smoke tests, then checks the CUDA-wheel
imports offline on a CPU runner before pushing the tested image. The job
summary records its immutable `sha256` image reference.

Use that digest, rather than the source tag, in a runtime profile. Publication
does not qualify CUDA execution, GPU memory, throughput, isolation or a model.
Qualify that exact image and profile on the intended host before activation.

## Dataset

Use owned, fully reviewed PNG examples. One JSONL row has this shape; the frame
hash and base64 placeholders must be replaced with real PNG bytes:

```json
{
  "id": "clip-001",
  "source_id": "match-001",
  "split": "train",
  "task": "football_events",
  "prompt": "Return soccer_action JSON with items containing frame, action and confidence. Use clip-relative frames at 25 FPS.",
  "annotation_complete": true,
  "is_verified_negative": false,
  "target": {
    "type": "soccer_action",
    "items": [{"frame": 25, "action": "pass", "confidence": 1}]
  },
  "media": {
    "kind": "frames",
    "clip_start_us": 0,
    "duration_us": 2000000,
    "score_fps": 25,
    "frames": [
      {"data_base64": "<PNG base64>", "sha256": "<64 lowercase hex characters>",
       "width": 448, "height": 448, "pts_us": 0}
    ]
  }
}
```

Each timestamp is a clip-relative presentation time in microseconds. The
processor contract `timestamped_frames_v1` sends explicit timestamp text plus
individual images. It does not invoke a model's native video sampling or
compression path. Frame count and sequence length must therefore be qualified
for this representation.

Grounding uses `task: "image_grounding"`, an image with zero duration and one
frame at time zero, and a target such as
`{"type":"grounding","items":[{"label":"ball","box":[0,0,1000,1000]}]}`.
Boxes are normalized integer xyxy coordinates. Empty targets require
`is_verified_negative: true`; missing labels must not be encoded as negatives.

Inputs are limited to 5 MiB, 256 examples, 64 frames per clip, 60 seconds per
clip, 2048 pixels per dimension and 64 million decoded pixels per dataset.
PNG must be static, non-interlaced, 8-bit RGB/RGBA. URLs, path references in
uploaded rows, duplicate JSON keys and non-finite numbers are rejected.
The platform requires train and validation examples. Sources and identical
image hashes cannot cross splits.

Validate and materialize the upload without running a model:

```sh
python -m reliquary.vision prepare dataset.jsonl prepared-dataset
```

## Snapshot and platform contract

Prepare a read-only model directory with:

```text
model/
  training.py
  source.json
  weights/
    config.json
    model.safetensors
    ...processor and tokenizer files...
```

Copy `scripts/vision-training-runtime.py` to `model/training.py`. Download
weights and processor files from their reviewed immutable revisions into
`model/weights`, using regular files rather than cache symlinks. Then lock them:

```sh
python -m reliquary.vision lock-model \
  --model model --model-id organization/model \
  --revision MODEL_COMMIT --processor-revision PROCESSOR_COMMIT
```

Both commits must be 40 lowercase hex characters. This command hashes the
existing files; it does not certify where they came from. It writes
`source.json` and prints the processor digest to use in the profile.
The operator must verify download provenance against the approved repository
and commit before approving this read-only cache. This trusted-worker contract
does not attest an untrusted miner's model weights.
The processor digest hashes the sorted filename-to-SHA256 map for config,
tokenizer, processor and template files, using canonical JSON. Every snapshot
file is separately verified at execution.

The platform profile uses `datasetFormat: "vision"`, `method: "sft"` or
`"grpo"`, and:

```json
{
  "vision": {
    "schemaVersion": 1,
    "processorRevision": "<40 lowercase hex characters>",
    "processorDigest": "<64 lowercase hex characters>",
    "mode": "timestamped_frames_v1",
    "maxFrames": 4,
    "maxPixels": 200704,
    "maxSequenceLength": 8192,
    "task": "football_events"
  }
}
```

These numbers illustrate the schema, not measured hardware capacity.
`environmentDigest` binds the entrypoint file; `imageDigest` binds the runtime
implementation and dependencies. Both are required, along with the profile and
conformance digests. The immutable model, environment, method, offer and
allocation in the job configuration must match the profile.

The platform host adapter mounts `/job` and `/model` read-only, `/output`
writable, disables networking and owns the allocation lease and cancellation.
Its runtime command can be:

```sh
python /model/training.py --job /job --model /model --output /output
```

The entrypoint supervises a child process and terminates its process group at
the allocation deadline. The training loop stops before the finalization
reserve. No successful `result.json` is written after an incomplete update,
failed reload, mismatched input or expired allocation.

## Training and delivery

The bounded recipe uses language-attention q/v LoRA only: rank 8, alpha 16,
learning rate 0.0001, seed 17, no LoRA dropout, and frozen vision/base weights.
SFT masks every prompt and image token. GRPO samples four responses at
temperature 1 with no top-k/top-p filtering, recomputes log probabilities with
the same visual tensors, normalizes rewards within the group, and performs
one clipped update. It uses a token-normalized objective with beta zero.
It excludes truncated and constant-reward groups and fails if no update
completes. This is a bounded GRPO recipe, not the native subnet trainer.

The complete prompt and answer must fit the profile context. Expected target
answers must fit the completion budget, capped at 128 tokens. Long event lists
need smaller clips or a separately revised and qualified recipe.

The runtime evaluates held-out examples before and after training with greedy
generation. Invalid/truncated outputs receive zero. Football reward uses
class weights, temporal matching at 25 FPS and extra-event penalties;
confidence does not affect that metric. Evaluation preserves zero for empty
ground truth. Training rewards correct abstention only on explicitly verified
negatives. Grounding uses class-aware one-to-one box IoU with missing/extra
box penalties; it is not Score's polygon/AP evaluator.
The football interoperability reference is the
[versioned scoring contract](https://github.com/score-technologies/turbovision/blob/60b328cbaabe63024f0e034204a0f6deb0d3154f/scorevision/validator/central/private_track/scoring.py)
and its
[action weights](https://github.com/score-technologies/turbovision/blob/60b328cbaabe63024f0e034204a0f6deb0d3154f/scorevision/utils/actions.py).

Success produces:

- `checkpoint.tar`: adapter safetensors, portable adapter configuration and
  model/processor/task binding.
- `result.json`: checkpoint digest, reload verification, baseline/post-run
  reward, held-out set digest, sample count, update count and vision contract.
- Local per-example predictions for inspection in the allocation output.

The platform adapter uploads the checkpoint and metric result through the
existing lease and private artifact routes. A parent checkpoint can continue
on new data only when its base model and complete visual contract match.
The checkpoint contains adapter weights; consumers also need the pinned base
model and matching processor files.

## Validation

```sh
python -m pytest tests/unit/test_vision* -q
```

Tests exercise parsing/media limits, source separation, rewards, response
masks, policy probabilities, input bindings, deadlines and checkpoint
continuation. The dedicated CI job installs CPU vision dependencies.
Its container smoke job downloads a pinned public SmolVLM2 snapshot, then runs
synthetic SFT and parent continuation offline inside the unprivileged CPU
image. This image is built with `TORCH_INDEX_URL` selecting the CPU wheels;
the default CUDA image requires its own qualification.
Synthetic fixtures and CPU execution do not qualify GPU memory, throughput,
host/container isolation or sports accuracy. Publication requires a reviewed
qualification receipt for the exact image, hardware and profile, plus a live
matching executor; no profile is activated by this package.
