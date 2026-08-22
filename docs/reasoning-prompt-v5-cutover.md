# Reasoning-prompt v5 cutover

Protocol v5 corrects the Qwen3-4B-Base rollout prompt without changing the
model, tokenizer path, sampling distribution, rollout count, token cap, BFT
policy, or DAPO objective controls. Protocol v4 remains immutable and is the
historical no-reasoning-cue control.

## Diagnosis

Raw prompt encoding is intentional for `Qwen/Qwen3-4B-Base`. Applying the
Qwen chat template would insert role markers and thinking controls that this
base checkpoint was not instruction-tuned to follow. The error in v4 was
semantic: the Math prompt only contained the dataset question and boxed-answer
suffix, while the Code prompt only contained the dataset problem and dynamic
function contract. Neither asked the model to work through the problem.

The correction therefore does **not** add a chat template, system message, or
synthetic `<think>...</think>` wrapper. It adds a plain-language, raw-text
step-by-step cue. This follows the leading instruction in the released
[DAPO-Math-17K dataset](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
and the setup described by the
[DAPO paper](https://arxiv.org/abs/2503.14476), while keeping Reliquary's
reward format aligned with the final boxed-answer parser.

## Exact v5 prompts

Math:

```text
Solve the following math problem step by step.

$problem

Put your final answer within \boxed{}.
```

Code:

```text
Solve the following programming problem step by step.

$problem$contract

After your reasoning, provide the final implementation in the last fenced Python code block.
```

`$contract` is the existing grader-derived function signature instruction.
The last-fenced-block rule is necessary because the Code grader extracts the
last Python fence; reasoning prose outside that block remains safe.

Both templates are immutable fields of
`qwen3-4b-base-dapo-reasoning-v5`. `/state.generation_contract` advertises the
template ID, renderer, exact text, and SHA-256. Miner and validator prompt
tokens remain bound by the existing `PROMPT_MISMATCH` check. V5 checkpoint
lineage metadata also stores a canonical SHA-256 of the complete generation
contract, so reusing the profile ID with altered prompt text fails at resume.

## Checkpoint decision

Use a fresh Qwen3-4B-Base reset for the primary v5 run. Do not resume the v4
trained checkpoint and call that a base-model restart: its policy has already
been optimized under the no-cue data distribution, so the result would not be
a clean baseline.

Keep the current v4 checkpoint immutable for two purposes:

- the historical v4 control and reproducibility;
- a separately labelled warm-start diagnostic under the v5 prompt.

The warm-start diagnostic can answer whether the weights are salvageable, but
it must not be reported as the v5-from-base baseline. Checkpoint-lineage
validation intentionally rejects a v4-stamped checkpoint under the v5 profile.

## Pre-activation evaluation

Run the same pinned holdout revisions, prompt offsets, sample counts, and seed
domain for every candidate. At minimum, screen these cells:

| Weights | Prompt/profile | Purpose |
|---|---|---|
| Qwen3-4B-Base | v4 | Historical no-cue control |
| Qwen3-4B-Base | v5 | Clean prompt-effect baseline |
| Current v4 checkpoint | v4 | Current-policy reference |
| Current v4 checkpoint | v5 | Warm-start diagnostic only |

Track more than accuracy: `pass_at_1`, average rollout reward, termination
rate, boxed/fenced-format rate, mean/p50/p95/max completion length, cap-hit
rate, repeated-ngram/rambling rate, group reward variance, and fraction of
groups passing the dynamic-sampling zone. For Code, also track grader pass
rate and extraction failures.

Run two forms of comparison:

- **Deployment parity:** omit `--forced-stream-domain`. Each profile uses its
  real protocol domain, so the output is representative of launch behavior.
- **Prompt-only paired ablation:** pass the same non-production
  `--forced-stream-domain` to the v4 and v5 runs. This holds the inverse-CDF
  uniforms fixed while the prompt changes. The report records
  `deployment_parity: false`, so these diagnostic results cannot be mistaken
  for production protocol evidence.

The recovery screen requires an explicit protocol profile and derives the
tokenizer revision, prompt, BFT mode, and token cap from it. Example for the
clean Math deployment-parity baseline:

```bash
python scripts/screen_recovery_checkpoints.py \
  --protocol-profile qwen3-4b-base-dapo-reasoning-v5 \
  --model-repo Qwen/Qwen3-4B-Base \
  --model-revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --checkpoint-label qwen3-4b-base-v5-prompt \
  --environment openmathinstruct \
  --math-jsonl <pinned-holdout.jsonl> \
  --dataset-revision <40-char-holdout-revision> \
  --n-prompts 64 \
  --samples-per-prompt 16 \
  --seed-domain reasoning-prompt-v5-screen-v1 \
  --output <result.json>
```

For the matched prompt-only ablation, repeat both the v4 and v5 base-model
runs with:

```bash
--forced-stream-domain reasoning-prompt-paired-ablation-v1
```

Do not reuse v4 benchmark numbers as the v5 baseline. The prompt tokens and
forced-seed protocol domain both changed, so all acceptance, length, capacity,
and quality screens must be regenerated and labelled with their full profile.

## Fresh v5 activation

1. Publish the pinned base weights as the next append-only checkpoint, stamped
   with the v5 lineage:

   ```bash
   RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-base-dapo-reasoning-v5 \
   RELIQUARY_HF_REPO_ID=<trainer-repo> \
   HF_TOKEN=<token> \
   python scripts/publish_base_reset_checkpoint.py
   ```

2. Re-run proof-capacity qualification against that exact stamped checkpoint.
   A v4 manifest cannot activate v5: the profile ID differs, and reasoning may
   change completion-length and termination distributions even though the
   numerical cap is unchanged.

3. Configure the trainer with the printed resume SHA and a new run identity:

   ```bash
   RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-base-dapo-reasoning-v5
   RELIQUARY_CHECKPOINT=Qwen/Qwen3-4B-Base
   RELIQUARY_RESUME_FROM=sha:<v5-stamped-base-reset>
   RELIQUARY_TRAINING_RUN_ID=<new-v5-run-id>
   RELIQUARY_PROOF_CAPACITY_MANIFEST=<v5-manifest>
   RELIQUARY_PROOF_CAPACITY_MANIFEST_SHA256=<sha256>
   ```

4. Deploy miner and validator code together. Confirm `/state` advertises
   protocol `5`, the v5 profile ID, raw encoding, both exact prompt templates,
   16 rollouts, full-support sampling, an 8192-token cap, and `bft: null`.

5. Run canary windows without optimizer steps, compare observed prompt hashes
   and prompt tokens on miner and validator, then enable training only after
   Math and Code quality, termination, capacity, and GRAIL parity gates pass.

A second GPU can shorten proof and rollout wall time if it is included in the
qualified fleet, but it does not repair the data distribution. Do not change
rollout count, gradient accumulation, sampling, or optimizer settings at the
same time as this cutover; keeping those fixed makes v5 interpretable.

## Rollback

Rollback means selecting the immutable v4 profile and its v4-stamped
checkpoint together. Never serve a v4 checkpoint under the v5 profile or mix
v4- and v5-generated groups in one optimizer batch. Use distinct training run
IDs so prompt cooldown and archived evidence remain attributable to the right
experiment.
