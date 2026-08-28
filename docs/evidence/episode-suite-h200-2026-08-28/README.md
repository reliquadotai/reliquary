# Episode v1 H200 qualification — 2026-08-28

This directory preserves the raw qualification evidence for the first native
Reliquary Episode v1 suite. No weight release or network activation is implied.

## Bound identities

- GPU: NVIDIA H200, compute capability 9.0, 150,109,880,320 bytes VRAM.
- OS: Ubuntu 22.04, Linux 5.15, glibc 2.35.
- Runtime: Python 3.12.13, PyTorch 2.7.0+cu128, Transformers 5.9.0.
- Model: `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Downloaded artifact: 10 files, 8,056,505,693 bytes, SHA-256
  `d0631c44cd8ef12fe802522c3cdf475e7f53518cd5eedd32c63cabb1be0feb41`.
- The Hugging Face tree receipt verified every downloaded file. Three receipt
  entries not downloaded were repository documentation/license metadata.
- Model shadow reports bind code revision
  `db00f285c8b097c05e3101fec23c5fd62baae295`.
- The optimizer report and final full suite bind revision
  `77f1429848b4e77053b56735805c23fee0331780`; the intervening commit adds only
  a unit test for the full-model weight digest.

## Results

The exact latest revision passed 2,172 non-GPU tests with seven warnings. CPU
qualification passed all 192 reference episodes and all adversarial gates. The
192-case deterministic JSONL SHA-256 was identical on macOS, Ubuntu 22.04 and
an Ubuntu 24.04 container:

`5df4f1417859e901f33067a2c9caafd62c2752df75a671f7504d127e8f08c87c`

Pinned-model generation used eight tasks and 16 forced-seed rollouts per task:

| Environment | Success / failure | Mixed groups | Exact replay | Errors | p95 seconds |
|---|---:|---:|---:|---:|---:|
| `reliquary_stateful_tools_v1` | 79 / 49 | 8 / 8 | 128 / 128 | 0 | 8.424 |
| `reliquary_retrieval_tools_v1` | 9 / 119 | 2 / 8 | 128 / 128 | 0 | 7.853 |
| `reliquary_workspace_tools_v1` | 8 / 120 | 5 / 8 | 128 / 128 | 0 | 12.495 |

The fail-fast invalid-action rule preserved stateful's 79/49 reward split while
reducing p95 from 143.131 seconds to 8.424 seconds. Retrieval p95 fell from
81.747 seconds to 7.853 seconds; its artifact-bound forced-seed sample changed,
so its before/after reward counts are not a paired quality comparison.

The real CUDA optimizer gate used FlashAttention 2, replayed and admitted one
mixed 16-rollout stateful group, round-tripped the detached training payload,
and processed all 1,144 assistant tokens in one BF16 GRPO step. The scheduler
advanced once, gradients were finite, and the complete model weight digest
changed:

- before: `0eb89f96962225806a95309663a282f48c1fba5eca772258f56c877d683729dc`
- after: `ec617f33d978fb6f74bd922c0fd824b5463c30fff52315af7a42411626ab5dc9`
- elapsed: 37.848 seconds
- peak PyTorch allocation after model load: 27,529,947,136 bytes

## Known inherited GPU finding

The repository's dedicated GRAIL GPU suite remains at its documented baseline:
seven tests pass and `test_wrong_model_fails` fails because all seven low-range
random hidden-state commitments fit inside the consensus tolerance. This is a
pre-existing proof-strength finding, not an Episode v1 regression. Lowering the
tolerance in this branch would be an unsafe unilateral protocol change. The raw
failure is retained in `gpu-suite-77f1429.log`.

## File integrity

```text
5ab7621a32528f8ccbc2ad1b614e4feada2050e47ac9b7ca8f6648e135e01f4a  baseline-prefailfast-retrieval-8x16.json
0ca9e1eb469649be68abfb48e1a00b2fb1bb1b1ef6744ef7d4f538313fd37a46  baseline-prefailfast-stateful-8x16.json
eac4bd21bd25188198cc0496277eee6125400868ba1816b6dbb9a8507e63a9aa  episode-cpu-py312-linux-db00f28.json
5df4f1417859e901f33067a2c9caafd62c2752df75a671f7504d127e8f08c87c  episode-determinism-py312-ubuntu24.jsonl
247a7229cbe9aaf8b6bef3f4f72e8c5f7c1383ce0b7172d8bfeadf74e21187f1  episode-training-gpu-77f1429.json
cd52da457cc8ea23db2a75ae3b44febff1085ce2b573ca22de1bf845c917b485  full-cpu-py312-linux-77f1429.log
909a5aa4d49a2caaca8426b10884911cce367e3bac642138e5170efd301c8dd7  gpu-suite-77f1429.log
7d7ef694a856d4a9e74469af2dd300adb1351d17e894df08adbf9a1ca3e941e2  model-reliquary_retrieval_tools_v1-8x16-db00f28.json
dec8cea21cff32fbfb7db6598b9825b2c011e259b1cc70c3004c48ced4dcc9c4  model-reliquary_stateful_tools_v1-8x16-db00f28.json
ef2b757543472865de2062da8d668741c37cc5c53c9523dac87c848f55b852ed  model-reliquary_workspace_tools_v1-8x16-db00f28.json
```
