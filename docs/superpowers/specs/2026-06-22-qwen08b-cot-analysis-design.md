# Qwen3.5-0.8B + CoT — analysis design

Date: 2026-06-22
Status: approved (plan + framing), execution in progress

## Context

Two coupled changes under evaluation: (1) switch the subnet base model from
Qwen3.5-4B to **Qwen3.5-0.8B (base, raw)**, (2) **re-enable CoT** (thinking on,
generation unbounded → native 262 144 ctx). Question to answer empirically:
**do the current envs (openmathinstruct, opencodeinstruct) and the reward/zone
mechanics still work for a much weaker 0.8B model emitting CoT?**

Headline hypothesis: a 0.8B base on hard math/code may have very low pass@1 →
mostly all-fail groups → in-zone yield collapses → training starvation; CoT may
partially compensate. Measure it.

### Framing decisions (user)
- Checkpoint: **base Qwen3.5-0.8B** (capability floor / fresh-run worst case).
- Scope: **full plan** (all six workstreams).
- Generation: **truly unbounded → 262k native** (no low max-tokens cap).

### Model facts (HF config)
0.8B: hidden 1024, 24 layers (18 linear + 6 full attention, GatedDeltaNet
hybrid), **vocab 248 320 (identical to 4B)**, ctx 262 144, tied embeddings,
packaged as Qwen3_5ForConditionalGeneration (vision tower loads too).
Implication: same-vocab → verify fp32-cast OOM (verifier.py:424) sits at the
same seq for equal free VRAM, but ~14 GiB more headroom (smaller weights);
forward ~6-8× cheaper → long CoT far more affordable than on 4B.

## Measurement plan

One generation+grading pass yields §1–§3 (reward + length + shape together).
§4 is separate profiling. §5/§6 are interpretation/conclusions.

**§1 Model×env fit (core).** Per env: pass@1; reward histogram; per-group
k-distribution (successes of M=8); **honest in-zone yield** = fraction of
groups with `rewards_std(rewards) ≥ SIGMA_MIN (0.43)` (binary ⇔ k∈{2..6});
all-fail (k=0) and all-pass (k=8) rates. Compare to 4B where available.

**§2 CoT shape (unbounded gen).** Per env: completion-length distribution
(median/p90/p99/max); EOS-vs-truncation rate; length↔correct and length↔in-zone
correlations. → sets the protocol cap.

**§3 Reward shape.** Granularity (binary vs continuous); grader reliability on
0.8B output (boxed extraction, code-test parsing); nature of negatives (genuine
vs degenerate/empty/truncated); whether SIGMA_MIN/zone fits the 0.8B reward
distribution; reward-shape exploitability shift for a weaker model.

**§4 Throughput (time+VRAM).** profile_verify.py on 0.8B (verify ms/rollout vs
seq) and smoke_train_vram.py / bench on 0.8B; verify OOM threshold; sustainable
rollouts/window vs the 8-distinct seal; window-cadence projection.

**§5 Subnet dynamics.** Latency-vs-length miner incentive (0.8B generates fast →
less latency pressure, but weaker → more reliance on public answers/curation);
in-zone trajectory under training; per-env balance; interaction with existing
exploits.

**§6 Config to retune.** MAX_NEW_TOKENS_PROTOCOL_CAP (from §2); enable_thinking
flip in tokens.py + golden-encoding regen + consensus-affecting deploy;
SIGMA_MIN/zone per env (from §1/§3); **TRAINING_QUARANTINE_MAX_*_COMPLETION_LENGTH
(4096/7000) — CoT will exceed these → must raise**; micro-batch cap; verify
chunking fix (verifier.py:424).

## Harness

`scripts/cot_analysis.py` (run on the H200, prod-faithful venv):
- Load model (AutoModelForImageTextToText, bf16, flash_attention_2) + tokenizer.
- Prompt encoding **thinking ON**: `apply_chat_template(..., enable_thinking=True)`
  (bypasses encode_prompt which forces it off).
- Per sampled prompt: generate **M=8** rollouts batched at **T_PROTO=0.9,
  top_p=1.0, top_k=0**, max_new_tokens → native limit; truncate each at first EOS;
  record length + truncated flag + gen time.
- Grade: math via `_compute_omi_reward` (pure python); code via in-process
  `worker.evaluate_call` + `GraderServer._outputs_match` over `structured_cases`
  (no gVisor — own model, per-case SIGALRM timeout = GRADER_EVAL_TIMEOUT_SECONDS).
- Per group: `rewards_std` + `is_in_zone` (from verifier.py).
- Output: per-rollout + per-group JSONL; aggregate report separately.
- Throughput: reuse profile_verify.py / smoke_train_vram.py with MODEL=0.8B.

### Caveats
- Code grading runs un-sandboxed (acceptable: own model, timeout-guarded); reward
  *value* matches the gVisor path (same evaluate_call + _outputs_match logic).
- Base 0.8B = capability floor, not the trained/served regime.
- Datasets pulled lazily from HF (OMI public; R0mAI/opencodeinstruct-curated).
