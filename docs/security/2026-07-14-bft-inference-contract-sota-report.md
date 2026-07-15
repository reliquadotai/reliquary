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
| Image revision | `d95e4254f12a9dc4c92692e80c251bccd3934024` |
| Active window | Healthy and advancing at final observation |
| Archive | Uploads healthy; checkpoint-local rows advancing |
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

The active checkpoint's private schema-v3 records cover 346 groups, 20 windows,
and 31 hotkeys:

| Cohort | Rollouts | Positions | Hard CDF mismatch rate | Stochastic agreement |
|---|---:|---:|---:|---:|
| Natural | 2,758 | 2,576,198 | 0.3260% | 94.54% |
| BFT forced | 10 | 21,133 | 0.2934% | 96.30% |
| Combined | 2,768 | 2,597,331 | 0.3257% | 94.55% |

Across those 346 groups:

- Ratio group/rollout rejects: 4.
- Exact CDF would-reject: 346.
- Current forced-rollout share: 0.36%.

This decisively rules out exact-CDF enforcement under the present numerical
contract. The earlier checkpoint showed higher drift in its forced cohort, but
the current checkpoint does not. BFT and CDF drift are therefore
checkpoint/trajectory-dependent, not a stable causal relationship.

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

This fix is live through PR #128 and validator revision
`d95e4254f12a9dc4c92692e80c251bccd3934024`.

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

Commit: `da75037 feat(telemetry): correlate CDF drift with BFT termination`

## Added Runtime Contract

The validator now exposes `GET /runtime-contract` separately from `/state`.
Keeping capability discovery separate is important because legacy miners parse
`/state` with a strict schema.

When this endpoint exists, an upgraded miner submits a bounded runtime profile
using telemetry schema 2:

- Python, PyTorch, Transformers, CUDA, the `flash-linear-attention` wrapper,
  its independently installed `fla-core`, causal-conv and flash-attn versions.
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

Commit: `0a9c127 feat(runtime): fingerprint inference kernel profiles`

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

Commit: `f035678 feat(training): expose BFT gradient contribution metrics`

## Controlled H200 Experiment

The matrix ran on an NVIDIA H200 against immutable checkpoint
`58dc1761acffe120a9e5e7b6f447a95f03a06b98` and the exact production image
`ghcr.io/reliquadotai/reliquary-validator:sha-d95e425`. Three isolated images
were used; no package was changed in the live validator or a miner:

- Production stack: FLA wrapper/core 0.5.0, no causal-conv.
- No-FLA control: FLA wrapper/core and causal-conv absent.
- Causal control: FLA 0.5.0 plus verified `causal-conv1d==1.6.2.post1` wheel.

The fixed 64-prompt corpus hash was
`67544c0fd17ccfda58c298f0dc687a2fb7817cebf5b62a2d7072a7d4a08190d9`;
corpus prompt indices, model revision, runtime profile, process
ID, completion hashes, timings, and peak CUDA memory are recorded in every
artifact.

### Short fixed-corpus result

Each profile generated 64 prompts x 8 rollouts x 128 tokens:

| Profile | Hard CDF mismatch | Pipeline tok/s | Peak reserved |
|---|---:|---:|---:|
| Production | 249 / 65,536 = 0.3799% | 236.4 | 5.95 GB |
| No FLA | 251 / 65,536 = 0.3830% | 240.9 | 5.95 GB |
| FLA plus causal-conv | 221 / 65,536 = 0.3372% | 235.2 | 5.95 GB |

The confidence intervals overlap. Causal-conv is therefore not a statistically
established winner. More importantly, full-completion agreement between profile
pairs was only 61.9% to 63.5%. A package switch changes roughly 37% of 128-token
trajectories even when aggregate CDF rates look similar.

Three fresh-process replicates of each profile produced identical aggregate
counts within profile. The observed problem does not require cross-process
nondeterminism.

### Exact production BFT replay

One accepted production group from window 23105 was replayed with the exact
prompt, checkpoint, public randomness, prompt index, and batch shape. Production
had accepted all eight forced rollouts with 53 / 17,016 hard mismatches and six
correct answers.

| H200 profile | Forced | EOS | Hard mismatch | Pipeline tok/s |
|---|---:|---:|---:|---:|
| Production stack | 5 / 8 | 4 / 8 | 4.966% | 158.2 |
| No FLA | 8 / 8 | 7 / 8 | 2.481% | 241.2 |
| FLA plus causal-conv | 7 / 8 | 7 / 8 | 0.772% | 141.3 |
| FP32 generation, BF16 verification | 7 / 8 | not a release metric | 0.945% | 114.0 |

No completion was identical across profiles. The causal profile was exactly
deterministic when replayed, yet it still did not reproduce the accepted miner
trajectory. This proves that long-path runtime identity matters and that package
presence alone cannot define the protocol contract.

Text inspection located repetition around completion offsets 30 to 55, far
before the BFT insertion at 2048. BFT did not initiate rambling in this replay;
it rescued five of eight causal-profile trajectories into a valid boxed answer.
The training concern is narrower: a successful forced sample may retain a long
repetitive pre-force prefix, so its weighted exposure must be monitored.

### Cache and precision controls

- BF16 cached production: 3 / 1,024 hard mismatches.
- BF16 no-cache production: 5 / 1,024. Disabling cache did not fix drift.
- BF16 no-FLA eager: 4 / 1,024.
- FP32 no-FLA eager self-verification: 0 / 1,024.
- FP32 generation verified by the actual BF16/FA2 validator path:
  217 / 65,536 = 0.3311% on the full short corpus.

FP32 self-verification is a useful numerical control, not a production result.
When crossed back into the actual validator path its confidence interval
overlaps both production and causal-conv. It also costs substantially more
memory on long BFT replay.

### Graded BFT budget sweep

The same eight real math prompts (corpus hash
`81e829f903948772aecdb257f71c17e117dd6719221af89ead2beb63b4c1ceff`)
and eight public-seed rollouts per prompt were graded under each policy:

| Thinking/answer budget | Correct | Forced | EOS | Pipeline tok/s | Peak reserved |
|---|---:|---:|---:|---:|---:|
| 512 / 256 | 27 / 64 (42.2%) | 58 / 64 | 49 / 64 | 175.8 | 18.70 GB |
| 1024 / 256 | 32 / 64 (50.0%) | 36 / 64 | 54 / 64 | 200.5 | 31.72 GB |
| 1536 / 512 | 36 / 64 (56.3%) | 12 / 64 | 58 / 64 | 188.2 | 29.98 GB |
| 2048 / 512 | 34 / 64 (53.1%) | 12 / 64 | 55 / 64 | 172.1 | 36.24 GB |

The 2048 result was a strict subset of the 1536 successes: 34 paired successes,
28 paired failures, and two answers won only by 1536. This makes 1536/512 the
best next canary candidate, but 64 rollouts are not enough for a consensus
budget change. Keep 2048/512 live until a larger paired H100 canary confirms the
quality and checkpoint-transition result.

## Additional Security Finding: GRAIL Sketch Tolerance

The newly executable GPU suite exposed a pre-existing false security
assumption. Its wrong-model test expected random hidden states to fail, but all
seven challenges passed. A direct real-checkpoint diagnostic then compared 82
Qwen positions against random-unit, variance-matched random, shuffled-dimension,
shuffled-position, and all-zero commitments. Every variant passed 82 / 82.

The cause is structural: with 16 signed magnitude buckets and coefficients in
`[-127, 127]`, observed sketch differences remained below 3,000, while the
base tolerance is 5,000. A random field element still fails, but a fabricated
low-range model-derived or constant sketch need not. The comment claiming
roughly `tolerance / PRIME_Q` forgery probability does not describe this attack
distribution.

This does not by itself let a miner forge the forced-seed token trajectory: the
validator independently recomputes logits, token probabilities, public-seed
agreement, termination, and rewards. It does mean the hidden-state commitment
is not currently providing the claimed independent proof strength and may let
an attacker omit or substitute commitment computation.

Do not silently lower tolerance: that is consensus-sensitive and honest H100
cross-runtime margins were not previously retained. Schema-v4 telemetry now
records per-rollout maximum sketch difference, distinct-sketch ratio, zero
ratio, and constant-commitment status without storing exact sketches. The report
exposes p50/p95/p99 margins and low-entropy counts. This is the evidence required
to design a coordinated GRAIL proof revision rather than guessing a threshold.

## Verification

- Local unit suite: 1,153 passed.
- Focused telemetry/batcher suite after sketch instrumentation: 146 passed.
- Integration tests reached 14 passed and 5 skipped before a network-dependent
  prompt-source test blocked in SSL; the previously isolated offline selection
  passed 28 with 5 skipped.
- H200 GPU suite: seven passed; the one failure is the GRAIL wrong-model finding
  documented above, not a BFT/runtime regression.
- H200 was left with no running benchmark or test container.
- Raw JSON artifacts and both fixed corpora are preserved at
  `/Users/malouk/GRAIL/reliquary-h200-bft-results-20260715.tar.gz`, SHA-256
  `2f0f1acf369d976d24906217a64178dd20ac2b680c8ab3c1fd56d7514cd44582`.

## Production Plan

1. Merge the passive runtime, BFT, training, and sketch telemetry release. It
   does not alter tokens, rewards, BFT budgets, or gate thresholds.
2. Deploy validator first. Confirm `/health`, `/runtime-contract`, unchanged
   `/state`, schema-v4 rows, legacy-miner acceptance, and `bft/` training metrics.
3. Upgrade one maintained miner so its self-reported runtime profile appears in
   private telemetry. Old miners remain compatible and need no immediate action.
4. Keep `FORCED_SEED_CDF_ENFORCE=false`; current evidence already proves exact
   CDF enforcement unsafe, so no 24-hour wait is needed for that decision.
5. Collect at least 1,000 honest groups across five hotkeys and a checkpoint
   transition for runtime/BFT correlations and honest sketch-margin calibration.
   The volume target is for choosing a new policy, not for delaying this passive
   release.
6. Run a larger paired 1536/512 versus 2048/512 H100 canary before proposing a
   coordinated budget change.
7. Design GRAIL v8 from the observed honest margin distribution and adversarial
   controls. Treat any threshold or commitment-format change as a coordinated
   miner-validator protocol release.

## Rollback and Stop Rules

Rollback the observability release if legacy submissions fail, proof throughput
regresses, private JSONL growth threatens disk, training masks beyond a validated
force span, or `/state` changes shape. Stop a runtime or budget canary if ratio
rejects, bad termination, repetition, or checkpoint-transition quality worsens
against its paired control.

## Final Direction

BFT remains useful and is not the demonstrated source of rambling. The force
tokens are now safely masked from loss, while pre-force and post-force policy
tokens remain trainable and measurable. The H200 work rejects a miner-only
kernel installation and identifies 1536/512 as a canary candidate, not a live
default.

The immediate SOTA action is a compatible observability deployment, followed by
paired H100 budget validation and a separately versioned GRAIL proof hardening.
Keep the tolerant forced-seed ratio gate on, exact CDF off, and the live runtime
and 2048/512 budget unchanged until those coordinated canaries pass.
