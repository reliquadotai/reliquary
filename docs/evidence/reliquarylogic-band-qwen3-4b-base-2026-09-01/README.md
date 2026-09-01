# reliquarylogic_v1 sigma-band measurement — 2026-09-01

Raw per-prompt evidence behind the band figures recorded in
`_FAMILY_REGISTRY`. No activation on any live lane is implied: the
environment is registered and declared only by the opt-in
`qwen3-4b-reliquary-logic-v8-dev1` profile.

## Bound identities

- GPU: NVIDIA H100 PCIe, 80 GB, compute capability 9.0.
- Runtime: Python 3.12.3, vLLM 0.28.0, PyTorch 2.13.0+cu130,
  Transformers 5.16.1. `VLLM_USE_FLASHINFER_SAMPLER=0` — the image has no
  CUDA toolkit, and FlashInfer JIT needs `nvcc`.
- Model: `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539`,
  the exact revision the profile pins, passed explicitly.
- Sampling: the profile's contract — T=1.0, top_p=1.0, top_k disabled,
  raw prompt with no chat template, 16 rollouts, 2048 max new tokens,
  seed 7, suite seed 1337, 64 prompts per family.

## What the band is

The share of 16-rollout groups whose reward sigma clears `SIGMA_MIN`
(0.24), which at 16 rollouts means 1 to 15 successes. Outside it the
difficulty auction values the group at zero and the gate filters it, so it
is the only part of a corpus that produces gradient. The bound is computed
by the tool, not assumed.

## Files

| file | roster | note |
|---|---|---|
| `band-12-families.json` | 12 active | the figures recorded in the roster |
| `band-with-inactive.json` | 9 active | adds the then-dormant families, marked `*` |

A roster change remaps the index space, so the two files sample different
tasks and their columns are **not** comparable family by family. That is
why each file records its own roster.

## Result under the twelve-family roster

| family | pass@1 | band | k=0 | k=16 | json | eos | tok |
|---|---:|---:|---:|---:|---:|---:|---:|
| web_of_lies | 0.573 | 100.0% | 0.0% | 0.0% | 94.2% | 99.9% | 10 |
| boolean_expressions | 0.658 | 98.4% | 0.0% | 1.6% | 89.3% | 99.9% | 10 |
| operation | 0.296 | 98.4% | 1.6% | 0.0% | 85.4% | 99.9% | 68 |
| space_reasoning | 0.299 | 98.4% | 1.6% | 0.0% | 90.6% | 99.6% | 13 |
| time_sequence | 0.409 | 98.4% | 0.0% | 1.6% | 88.7% | 100.0% | 18 |
| object_properties | 0.481 | 96.9% | 1.6% | 1.6% | 92.3% | 99.8% | 12 |
| word_sorting_mistake | 0.099 | 68.8% | 31.2% | 0.0% | 85.6% | 99.9% | 14 |
| dyck_language | 0.075 | 64.1% | 35.9% | 0.0% | 43.7% | 99.7% | 12 |
| dyck_language_errors | 0.096 | 53.1% | 46.9% | 0.0% | 57.0% | 98.6% | 225 |
| math_path | 0.042 | 43.8% | 56.2% | 0.0% | 78.5% | 99.5% | 293 |
| numbrix | 0.025 | 21.9% | 78.1% | 0.0% | 78.4% | 97.6% | 146 |
| cryptarithm | 0.013 | 15.6% | 84.4% | 0.0% | 49.2% | 94.8% | 452 |
| cipher (dormant) | 0.000 | 0.0% | 100.0% | 0.0% | 74.6% | 99.5% | 31 |

## What it does and does not establish

It establishes that the corpus **supplies** gradient against the base
model, and how much per family. It does not establish that a model
**learns** from it: OpenMathInstruct-2 measures 75.5% in band on this same
metric and its lane has stopped improving. Band is necessary, not
sufficient, and only a training canary settles the rest.

Two readings worth carrying forward:

- **Answer shape dominated depth.** `math_path` reasons for 293 median
  tokens — deeper than `numbrix` at 146 — yet clears twice its band with a
  bare integer instead of a nested grid. `numbrix` and `cryptarithm` are
  therefore likely recoverable by changing what they return rather than how
  hard they are. A flattened answer must stay unguessable: one numbrix cell
  is one of sixteen values, and 6.25% lucky guesses would manufacture k=1
  groups that pass the band carrying no signal.
- **Saturation has already begun.** `k=16` is non-zero on three families
  before any training. The families clearing 98% are the ones closest to
  falling out of the band from above, so the spread of pass@1 (0.013 to
  0.658) matters more than the 71.5% mean.

## Reproducing

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/eval_logic_band.py \
  --model Qwen/Qwen3-4B-Base \
  --revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --prompts-per-family 64 --max-new-tokens 2048 \
  --include-inactive --output band.json
```

Roughly 90 seconds of generation on an H100 for 832 prompts by 16 rollouts.

## File integrity

```text
2e38a954bc1715317bb4ca36ad8a7c7ac6a37ff4fb31ecb9d0151781ed4cda3c  band-12-families.json
bff7aa7398cee1f9096e1abf3232cd72a479c8527051b5ade65f12fccd3a0218  band-with-inactive.json
```
