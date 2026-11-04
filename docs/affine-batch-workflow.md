# Local Affine batch experiments

`python -m reliquary.eval.affine_batches` composes an external official miner kit: generate rollouts, export a batch, validate its tokens, take one official training step and compare head/candidate responses with the official sampler, grader and estimator. It also accepts an existing batch. It never registers a miner, uploads a batch, submits chain transactions, provisions compute or starts model servers.

The wrapper pins the external miner tree to `AffineFoundation/affine` revision `eaf1ad20e4d018c85b30f53e9807b2b9a6a75e25`. It checks every tracked miner file against that commit and refuses modified, missing, symlinked or additional files. The published validator manifest is checked separately and retained as evidence; the wrapper never rewrites it. Keep this checkout separate from the running application.

Use a separate Python environment with the official `miner/requirements-sim-cpu.txt` for CPU fixtures. The bounded CPU check uses Torch 2.14.0 and Transformers 5.14.1; no heavy dependency is added to the application package. The interpreter running the module must contain those dependencies, or pass its executable using `--python`. Run from the application checkout, or install the application module through your existing reviewed packaging process.

Provide local head/reference checkpoint directories, a captured `mining.json`, a hotkey identifier and a new private work directory outside every Git repository. The work directory holds raw batches, model artifacts, reports and logs; existing outputs are preserved and require a different work directory for another attempt. Source verification and planning do not authenticate the supplied checkpoint or contract. The kit's fetch helper verifies file hashes; signed-envelope authority verification requires a separately trusted verifier.

## Plan without execution

```sh
python -m reliquary.eval.affine_batches plan \
  --kit "$KIT" --mining "$MINING" --head "$HEAD" --ref "$REFERENCE" \
  --batch "$BATCH" --hotkey "$HOTKEY" --work "$PRIVATE_WORK" \
  --seed 42 --training-result-at "$TRAINING_RESULT_AT" \
  --last-green-decided-at "$LAST_GREEN_DECIDED_AT"
```

This reads pins and emits `affine-batch-plan/v1` JSON with declarative commands. It creates no outputs and executes no model or kit code. `qualification` and `tokenValidation` remain `not_run`; `liveParity`, `officialAcceptance` and `chainSubmission` are false. Token validation is a separate stage.

## Generate or supply a batch

Serve the published head named in the captured `mining.json`, with its tokenizer. The original Hugging Face base and other teacher models are not interchangeable with that head. The official generator uses temperature 1, top-k 20 and top-p 0.95; tokens outside the head's sampler support are excluded from the policy-gradient term. The supplied server identity remains caller asserted.

With an explicitly provided head inference server on loopback:

```sh
python -m reliquary.eval.affine_batches generate \
  --kit "$KIT" --mining "$MINING" --head "$HEAD" \
  --hotkey "$HOTKEY" --work "$PRIVATE_WORK" \
  --head-url http://127.0.0.1:8000 --tasks 0-3 --samples 4 \
  --select all --max-prompts 2 --per-prompt 2 --seed 42
```

This calls official `generate.py` and `make_batch.py`; the generated batch is `batch.private.json` in the work directory. Alternatively, provide your existing file as `$BATCH`.

```sh
python -m reliquary.eval.affine_batches validate \
  --kit "$KIT" --mining "$MINING" --head "$HEAD" \
  --batch "$BATCH" --hotkey "$HOTKEY" --work "$PRIVATE_WORK"
```

Validation invokes the official full parser and tokenizer with the captured contract's caps. Whole-group truncation, reward bounds, train parameters, special tokens, context and output limits remain owned by the official implementation. The supported contract ceiling is 125 prompts and 1,000 rollouts. All 16 current train fields are checked against the pinned parser specifications and captured mining policy, including numeric ranges, integer choices, string choices and defaults. The official checker validates supplied train values; a plan only reports their structural presence. Planning refuses unsupported token, sampling or gate settings. Exactly 8,192 output tokens retain the official legacy capped-output interpretation.

## Train and compare

For a tiny CPU fixture, explicitly choose the device and token budget:

```sh
python -m reliquary.eval.affine_batches train \
  --kit "$KIT" --mining "$MINING" --head "$HEAD" --ref "$REFERENCE" \
  --batch "$BATCH" --hotkey "$HOTKEY" --work "$PRIVATE_WORK" \
  --device cpu --token-budget 512
```

There is no default training device. An independently authorized GPU run may explicitly use `--device cuda`; the wrapper forwards this choice to the official trainer. A full 9B checkpoint is not the tiny CPU fixture. Supply `--master "$MASTER"` to use an existing fp32 master; otherwise the report records `bf16-bootstrap`. The local trainer uses the official padded single-process path, which does not establish parity with packed distributed kernels or live fp32 master bytes.

The operator must supply separate loopback inference URLs serving the head and the resulting `candidate` directory. Endpoint identity is recorded as caller asserted, not attested. The wrapper neither starts nor replaces these servers.

```sh
python -m reliquary.eval.affine_batches compare \
  --kit "$KIT" --mining "$MINING" --head "$HEAD" --ref "$REFERENCE" \
  --batch "$BATCH" --candidate "$PRIVATE_WORK/candidate" \
  --hotkey "$HOTKEY" --work "$PRIVATE_WORK" \
  --head-url http://127.0.0.1:8000 --candidate-url http://127.0.0.1:8001 \
  --seed 42 --n 500 --training-result-at "$TRAINING_RESULT_AT" \
  --last-green-decided-at "$LAST_GREEN_DECIDED_AT"
```

Comparison reuses the official prior-v1 draw, prompt rendering, sampler, grader and estimator through a narrow compatibility boundary. The comparison contract is `fixed-z-1-public-prior-v1`: the threshold is fixed at `barZ: 1`, independent of elapsed time. The two explicit frozen timestamps remain provenance and must be finite, nonnegative and ordered. It always collects a fresh baseline and binds the head, reference, tokenizer, batch, contract, prior, source, configuration and seed. Optional `--prior "$PRIOR"` supplies your own `{id,s,n}` JSONL; absent values use environment averages. Every current comparison is labeled **PUBLIC PRIOR APPROXIMATION**: the public Beta-prior inclusion sampler uses the optional counts or environment averages; it does not reproduce the live IRT information weighting, whose task parameters and head ability are private. No private IRT parameters are invented. Fresh sampling uses the official 12,288 output-token cap and 16,384 context limit, task-level candidate cap-hit ceiling 0.33, strict z > 1, zero run-noise floor, and the official estimator. Captures retaining the older 0.003 noise floor are refused.

`simulate` composes generation when no `--batch` is provided, then validation, training and comparison. It accepts the same flags as those stages and still requires explicit device, inference URLs, seed and clocks. Stage reports use `affine-batch-stage-report/v1` and retain `assurance: source-pinned-local-simulation`; no local decision changes the qualification or acceptance flags.

The official CPU test recursively deletes its `--work` directory: run it only with a new dedicated temporary directory. Its fake inference servers ignore checkpoint weights. Their GREEN result tests plumbing and cannot demonstrate learned gain. A learning claim needs real before/after inference on a frozen independent evaluation set.
