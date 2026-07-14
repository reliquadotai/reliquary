# BFT and Inference Runtime Contract: SOTA Report

Date: 2026-07-14

Scope: Budget-Forced Termination (BFT), forced-seed consistency, cached versus
teacher-forced inference drift, kernel/runtime parity, training safety, and the
telemetry required for a production decision.

## Expert Verdict

The current evidence does not support the claim that missing `causal-conv1d`
is the root cause of rambling, nor that installing it on miners alone is a safe
fix.

The strongest supported model is:

1. Miners generate token by token through a cached inference path.
2. The validator verifies the completed sequence through a full-sequence,
   `use_cache=False` path.
3. BF16 operation ordering and kernel differences can move logits enough to
   cross an inverse-CDF boundary at a small fraction of positions.
4. A branch change can sometimes degrade the continuation, reduce natural
   termination, or increase reliance on BFT.
5. BFT absorbs and can amplify the symptom, but it is not demonstrated to be
   the initiating cause.

There is also a separate training-security issue that was real and actionable:
training previously consumed the miner-declared `force_span` directly. A
miner could submit `forced=false` with an arbitrary span and suppress selected
completion tokens from GRPO loss, or send malformed bounds that raised during
training. That boundary is now fixed: only a span canonically validated by the
validator can affect loss masking.

## Decisions

Effective immediately:

- Keep `FORCED_SEED_ENFORCE=true`.
- Keep `FORCED_SEED_CDF_ENFORCE=false`.
- Keep the current BFT budgets at 2048 thinking tokens and 512 answer tokens.
- Keep masking only the validator-injected force tokens from training.
- Continue training the natural reasoning before the force and the sampled
  answer after it.
- Do not instruct miners to install `causal-conv1d` yet.
- Do not change the live runtime until the controlled matrix identifies a
  profile that improves both agreement and rollout quality.
- Deploy the training-boundary security fix independently of the runtime
  experiment.

## Current Production Snapshot

Read-only validation on `ubuntu@209.20.157.231` at the time of this report:

| Surface | Value |
|---|---|
| Validator status | `ok`, `open` |
| Image revision | `10706b04393cebe382caddef4e31a93d00b03756` |
| Active window | `23094` |
| Valid submissions | `4` at observation time |
| Checkpoint | `28`, revision `58dc1761acffe120a9e5e7b6f447a95f03a06b98` |
| Ratio forced-seed gate | Enforced |
| Exact CDF gate | Shadow only |
| PyTorch | `2.7.0+cu128` |
| Transformers | `5.9.0` |
| Flash Linear Attention | `0.5.0` |
| causal-conv1d | Not installed |
| flash-attn | `2.8.3` |
| Validator GPU | NVIDIA H100 PCIe |
| Qwen3.5 FLA chunk/recurrent | Available |
| Qwen3.5 causal-conv prefill/update | Fallback |
| Qwen3.5 all-fast-path flag | False |

This is a mixed component runtime. Missing causal-conv does not imply that the
whole GatedDeltaNet falls back to plain PyTorch: FLA chunk and recurrent kernels
are active while causal convolution uses its fallback.

## Fresh Telemetry

The most recent 500 private schema-v3 group records cover 28 windows and 30
hotkeys:

| Cohort | Rollouts | Positions | Hard CDF mismatch rate | Stochastic agreement |
|---|---:|---:|---:|---:|
| Natural | 3,940 | 3,635,668 | 0.3070% | 95.02% |
| BFT forced | 60 | 125,758 | 0.5908% | 93.74% |
| Combined | 4,000 | 3,761,426 | 0.3165% | 94.95% |

Across those 500 groups:

- Ratio group would-reject: 0.
- Ratio rollout would-reject: 0.
- Exact CDF would-reject: 500.
- Current forced-rollout share: 1.5%.

This decisively rules out exact-CDF enforcement under the present numerical
contract. It also confirms an association between the forced path and more
drift. It does not prove directionality: forced rollouts are longer and execute
a second generation phase, both of which increase exposure to mismatches.

Earlier 120-window BFT analysis found:

- 5,128 math rollouts, 358 forced (6.98%).
- 325 of 358 forced rollouts reached EOS after forcing.
- Only 33 forced rollouts hit the answer cap.
- Injected force tokens were about 0.05% of token mass and were masked.
- The post-force answer was about 0.50% of token mass and about 0.65% of the
  absolute-advantage proxy.
- The pre-force reasoning carried materially more signal than the synthetic
  force text itself.

That snapshot did not show classic model collapse. The new training telemetry
will now measure these contributions continuously instead of relying on a
one-time reconstruction.

## Source Research Correction

The public Transformers issue most relevant to Qwen3.5 cached/full-sequence
drift does not establish an algorithmic FP32 GatedDeltaNet failure. Its initial
FP32 control was still running BF16 weights; true float conversion reduced the
discrepancy to a baseline comparable to other architectures. Installing fast
path packages also did not eliminate the reported BF16 difference.

Therefore:

- Cross-process nondeterminism is not required to produce a mismatch.
- A mismatch does not necessarily select a token outside top-k/top-p support.
- Faster kernels do not imply bit-identical execution.
- Package presence alone is insufficient; the active per-component path must
  be recorded.
- Miner-only runtime changes can increase miner-validator divergence.

References:

- Transformers Qwen3.5 cache divergence discussion:
  https://github.com/huggingface/transformers/issues/46190
- Flash Linear Attention 0.5.0:
  https://github.com/fla-org/flash-linear-attention/blob/v0.5.0/README.md
- PyTorch reproducibility limits:
  https://docs.pytorch.org/docs/stable/notes/randomness.html
- DAPO overlong filtering:
  https://arxiv.org/abs/2503.14476

## Security Finding and Fix

### Previous behavior

`training.py` read `rollout.commit["rollout"]["force_span"]` directly. The
commit is signed by the miner, but signing proves authorship, not correctness.
The canonical force-span validator ran earlier in the batcher, yet its result
was not the value consumed by training.

Impact:

- `forced=false`, `force_span=[0, huge]` could mask an entire completion from
  token-count normalization and policy loss.
- A miner could target low-reward rollouts and attenuate negative gradient.
- A malformed two-element span could raise in the seed verifier.
- A malformed span could destabilize a train step even if other validation
  remained healthy.

### New behavior

- Every rollout starts validation with no trusted force span.
- `validate_force_span` verifies forced status, bounds, exact budget position,
  no prior close token, and byte-exact canonical force tokens.
- Only a successful validation stores `_validated_force_span`, a Pydantic
  private attribute that is absent from wire serialization.
- Both per-rollout and micro-batched training consume only this private value.
- Invalid numeric bounds fail closed without crashing verifier workers.
- Non-forced legacy metadata remains wire-compatible but has no training
  effect.

Commit: `3b10530 fix(training): trust only validated BFT force spans`

## Added Causal Telemetry

Private forced-seed telemetry is bumped to schema v4. Each rollout now records:

- First hard CDF mismatch completion offset.
- Validator-validated termination path:
  `phase1_eos`, `natural_phase2_eos`, `forced_phase2_eos`, cap variants, or
  an explicit fallback category.
- Claimed versus validated forced status.
- Validated force-span length.
- Unique-token ratio.
- Repeated 4-gram fraction, globally and over the last 256 tokens.
- Maximum identical-token run.
- First repeated n-gram and repeated-token-run offsets.

The termination shadow also receives CDF onset and repetition metrics for
interesting cap or low-probability termination events.

The report script now separates:

- Natural and forced termination paths.
- Hard-mismatch and no-hard-mismatch cohorts.
- Cases where CDF drift precedes repetition versus repetition preceding drift.
- Runtime profiles and their mismatch totals.

Command:

```bash
python scripts/report_forced_seed_cdf.py \
  /root/reliquary/state/auth_forensics/forced-seed-shadow.jsonl --json
```

Commit: `a4cf8cc feat(telemetry): correlate CDF drift with BFT termination`

## Added Runtime Contract

The validator now exposes `GET /runtime-contract` separately from `/state`.
Keeping capability discovery separate is important because legacy miners parse
`/state` with a strict schema.

When this endpoint exists, an upgraded miner submits a bounded runtime profile:

- Python, PyTorch, Transformers, CUDA, FLA, causal-conv and flash-attn versions.
- GPU family and compute capability, but no hostname, serial, wallet secret, or
  filesystem path.
- Generation and proof dtype and attention implementation.
- Determinism, cuDNN benchmark, and TF32 flags.
- Qwen3.5 FLA and causal-conv component availability.

The profile hash is appended to the request nonce, and that nonce is covered by
the existing hotkey envelope signature. This prevents transit modification of
the profile without invalidating the request. It remains self-reported and is
not remote attestation, so it is a correlation signal only.

Compatibility sequence:

- New validator plus old miner: old miner ignores the new endpoint and submits
  the old request shape.
- New miner plus old validator: endpoint discovery returns 404, so the miner
  omits the optional field.
- New validator plus new miner: profile is attached and recorded privately.

Commit: `9972ef3 feat(runtime): fingerprint inference kernel profiles`

## Added Training Contribution Telemetry

Every successful train step now reports, over the surviving plan:

- Forced rollout count and ratio.
- Raw and trainable completion tokens.
- Injected tokens masked and their ratio.
- Forced trainable-token ratio.
- Per-termination-path rollout and token counts.
- Per-path absolute-advantage weighted token exposure.

This last quantity is an exposure proxy, not an exact gradient norm. It is the
right low-cost warning signal for a forced path becoming disproportionate.

Initial review thresholds, not rejection thresholds:

- Forced rollout ratio above 10% for three consecutive windows.
- Forced trainable-token or absolute-advantage exposure above 15%.
- Injected-token ratio above 0.25%.
- Tail repeated-ngram fraction more than 50% above a length-matched natural
  cohort.
- A sustained rise in `bad_termination` coincident with a checkpoint or runtime
  profile change.

Commit: `062f90e feat(training): expose BFT gradient contribution metrics`

## Controlled GPU Experiment

Execution status for this report: the supplied Targon H200 deployment no
longer accepted the corresponding local public key. Older H200 aliases also
presented changed SSH host keys, which were not bypassed. No shared process or
reprovisioned host was modified. The matrix below is therefore implemented and
ready, but its GPU result artifacts are still pending a trusted fresh instance.

Local verification is complete: 1,177 tests passed, 13 CUDA/context-dependent
tests skipped, and no functional tests failed. A CPU smoke run of the benchmark
on tiny GPT-2 completed four generated positions with zero hard CDF mismatches.

The new benchmark reproduces the protocol boundary directly:

```bash
python scripts/benchmark_inference_contract.py \
  --model <checkpoint-path-or-repo> \
  --checkpoint-hash <immutable-revision> \
  --prompts-jsonl <fixed-prompts.jsonl> \
  --batch-size 8 \
  --max-new-tokens 512 \
  --dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --output results/live-profile-b8.json
```

It performs cached forced-seed generation, then the validator's full-sequence
teacher-forced forward, and emits runtime, CDF, termination, repetition, and
throughput evidence. Run every profile in a fresh process.

Commit: `44af823 tools(runtime): add cached-vs-teacher-force benchmark`

### Matrix

Run the same immutable checkpoint, 64 to 128 fixed prompts, and public seed
material across:

| Axis | Values |
|---|---|
| Runtime | no FLA/no causal; FLA 0.5/no causal; FLA 0.5 plus pinned causal |
| Generation cache | on; off diagnostic control |
| Dtype | BF16 production; FP32 diagnostic control |
| Batch shape | 1, 4, 8 |
| Process | three fresh processes per cell |
| GPU | H200 primary; H100 production confirmation |
| Length | 256 and 512 isolation; 2048 plus BFT phase in final canary |

Do not compare profiles using free-form normal sampling. The forced public
uniform and immutable prompt/checkpoint set are required to align each token
decision.

### Metrics

- Hard CDF mismatch rate and max miss.
- Stochastic agreement.
- First mismatch offset.
- EOS and cap rate.
- Repeated n-gram and same-token-run onset.
- Whether mismatch precedes degeneration.
- Tokens per second and peak memory.
- Cross-process and cross-batch-shape variance.

### Acceptance criteria

A candidate canonical runtime must:

1. Reduce or equal the live profile's hard-mismatch rate with confidence
   intervals over the fixed cohort.
2. Not increase termination failures or repetition after length matching.
3. Produce zero ratio-gate false rejects over at least 10,000 honest rollouts.
4. Keep throughput within 10% of the best safe profile unless the quality gain
   clearly justifies the cost.
5. Reproduce on the production H100, not only the H200 lab host.
6. Be deployable as the same pinned validator and miner image/runtime.

Exact CDF enforcement requires a stronger result: effectively zero unexplained
hard mismatches under every supported runtime. Current evidence is nowhere near
that condition.

## Deployment Plan

### Phase 0: merge the known-safe changes

Merge and deploy the validator security and telemetry commits. They do not
change generation, BFT budgets, reward, ratio thresholds, or exact-CDF policy.

### Phase 1: validator first

Deploy the new validator image. Confirm:

- `/health` is `ok` and contains `runtime_fingerprint`.
- `/runtime-contract` returns telemetry version 1.
- `/state` retains its previous shape.
- A legacy miner submission is accepted.
- Private forced-seed rows advance to schema v4.
- Training metrics contain the `bft/` family.

### Phase 2: miner rollout

Upgrade one canary miner. It will attach a runtime profile only after capability
discovery. Confirm its profile appears in private telemetry and no new schema or
envelope reject occurs. Then roll to the remaining maintained miners.

### Phase 3: observe

Collect at least 24 hours, 1,000 groups, five hotkeys, multiple runtime profiles,
and at least one checkpoint transition. Interpret repetition within completion
length and termination-path cohorts.

### Phase 4: run the H200 matrix

Run isolated dependency environments. Never install or uninstall kernels inside
the live validator or an active miner process. Preserve the JSON artifact from
every cell and aggregate by profile hash.

### Phase 5: choose and canary a canonical runtime

If a profile wins, pin exact package versions and image digest for validator and
miners. Deploy one miner lane and validator canary together. Keep the ratio gate
active and exact CDF shadow-only.

### Phase 6: revisit BFT only with causal evidence

Change the 2048/512 budgets or post-force training policy only if the new data
shows a sustained quality or gradient-exposure problem after controlling for
runtime, length, checkpoint, and prompt.

## Rollback and Stop Rules

Rollback the telemetry release if any of the following appears:

- Legacy miners fail parsing or submission.
- Proof throughput regresses materially from profile collection overhead.
- Private JSONL growth threatens disk capacity.
- Training omits natural tokens or masks beyond a canonically validated span.
- `/state` shape changes.

Stop a runtime canary immediately if:

- Ratio-gate rejects increase for the canary profile.
- Bad termination or degeneration rises relative to a length-matched control.
- Checkpoint transition quality degrades network-wide.
- Miner and validator profile hashes indicate an unintended package split.

## Final Direction

The mission remains on track. BFT is useful as a bounded termination mechanism,
and current data does not justify removing it or treating its small synthetic
segment as a demonstrated collapse source. The unsafe part was not BFT itself;
it was an untrusted metadata boundary in training, now fixed.

The next protocol decision should be driven by the new causal telemetry and the
controlled runtime matrix. Until that evidence exists, the SOTA position is to
preserve the tolerant forced-seed gate, keep exact CDF in shadow, retain 2048/512,
and avoid unilateral kernel changes.
