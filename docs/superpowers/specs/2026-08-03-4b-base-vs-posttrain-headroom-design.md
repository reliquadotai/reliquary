# 4B RL plateau: is it the model, or the post-trained starting point?

Status: design approved 2026-08-03. Untracked working document.

## The question

RL on the 4B drifts, and the drift is fixable, but fixing it does not raise
performance. Measured on the production run: math pass@1 0.500 (upstream) vs
0.528 (checkpoint 88) — inside noise after 88 checkpoints. Natural termination
degraded over the same span (11.8% -> 8.8%).

Two competing explanations:

1. **Starting point.** Production RL starts from `Qwen/Qwen3.5-4B`, which is
   already post-trained and already RL-tuned for math reasoning. Its output
   distribution may be collapsed onto its modes, leaving nothing for GRPO/DAPO
   to sharpen. DAPO's published gains came from a *base* model
   (Qwen2.5-32B), a very different starting condition.
2. **Pipeline — data.** A prior analysis attributed the flat math curve to the
   training diet (~11 gradient-carrying rollouts per step), driven by auction
   concentration on k=2 groups and BFT masking — not to model capacity.
3. **Pipeline — objective.** A separate audit found roughly 1.5 of DAPO's 5
   ingredients are actually implemented, and that the importance ratio is
   computed in a different space from the one sampled: the sampler warps by
   T=0.6/top_k=20/top_p=0.95 while log-probs are reported raw, giving
   `r_warped ~= r_raw^1.667`, so a nominal +/-0.2 clip band is really
   [0.69, 1.35]. The default KL reference is the rolling published checkpoint,
   which anchors nothing against cumulative drift.

This study measures which of these is consistent with the evidence, without
running RL. It can only separate explanation 1 from explanations 2-3; the two
pipeline explanations are distinguished by other work already underway.

### Naming

The existing harness `.r2_analysis/math_eval.py` labels `Qwen/Qwen3.5-4B` as
`"base"`. It is not a base model. No true base model has ever been measured
here. To prevent the confusion from recurring, this study uses:

- `pretrain` — `Qwen/Qwen3.5-4B-Base` (true base)
- `upstream` — `Qwen/Qwen3.5-4B` (post-trained; the production starting point)
- `ckpt88` — the production RL checkpoint (out of scope for this study)

## What is being measured

RL headroom is not pass@1. GRPO/DAPO does not create capability; it
concentrates probability mass onto modes the policy already reaches
occasionally. Headroom therefore lives in the *dispersion*, not the mean.

Per cell:

1. **pass@k curve, k = 1..8.** `pass@8 - pass@1` is the gap RL can close. If
   pass@8 is approximately pass@1, there is nothing to concentrate.
2. **Non-degenerate group rate at k=8** — fraction of prompts with
   `0 < correct < 8`. This is exactly DAPO's dynamic-sampling filter, and thus
   the fraction of prompts that yield any gradient at all. Primary metric.
3. **Advantage scale** — reward standard deviation over non-degenerate groups.
   Equal group counts can still carry very different gradient magnitudes.
4. **Per-token entropy** — mean over generated tokens, and over the first 200
   tokens separately. Exploration capacity. Low entropy is a causal mechanism
   for a plateau, not merely a correlate.
5. **Termination profile** — EOS rate, length distribution, at-cap rate.
   Establishes how much drift the starting point already carries pre-RL.
6. **Wrong-answer diversity** — distinct normalised wrong answers over wrong
   rollouts. Systematic errors are learnable signal; random errors are noise.

Plus one control reported as a first-class number, never folded into zero:

7. **Extraction failure rate per cell.** `pretrain` answers `Answer: X`,
   `upstream` answers `\boxed{X}`. An extractor that silently misses one format
   scores those rollouts wrong and would manufacture the entire effect.

## Grid

Three cells, pinned revisions:

| Cell | Model | Format |
|------|-------|--------|
| A | `Qwen/Qwen3.5-4B` @ `851bf6e8` | native: chat template, `<think>`, `\boxed{}` (production condition) |
| B | `Qwen/Qwen3.5-4B-Base` @ `1001bb4d` | raw DAPO prompt, pure completion, `Answer: $X` |
| C | `Qwen/Qwen3.5-4B` @ `851bf6e8` | raw DAPO prompt, no chat template |
| Q3B | `Qwen/Qwen3-4B-Base` @ `906bfd4b` | raw DAPO prompt, pure completion |

`A - C` isolates prompt format. `C - B` isolates weights within one generation,
so it measures the final post-training stage cleanly.

Q3B was added after the pilot showed `Qwen3.5-4B-Base` already reasons. It is a
genuine base -- 0/32 spontaneous `<think>` -- at matched 4B size, and supplies
the DAPO starting condition. It is an **anchor, not a clean contrast**: it
differs from production by generation *and* post-training, and conclusions must
say so rather than attribute its gap to post-training alone.

The DAPO prompt (verbatim, as used against a base model):

```
Solve the following math problem step by step. The last line of your
response should be of the form Answer: $Answer (without quotes) where
$Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
```

## Data

- ~200 held-out OpenMathInstruct-2 prompts (`nvidia/OpenMathInstruct-2`
  @ `469216e3`, fields `problem` / `expected_answer`), sampled **uniformly at
  random**, with no selection on difficulty or on any cell's performance.
- ~200 `HuggingFaceH4/MATH-500` prompts, levels 3-5, as a public anchor for
  whether the post-trained model is saturated in absolute terms or only here.

The same 400 prompts are used for all three cells.

**No performance-based stratification.** OMI carries no difficulty label, so
targeting an "intermediate band" would require a generation pre-pass. Selecting
prompts where cell A lands in `0 < correct < 8` would give cell A a 100%
non-degenerate rate by construction, turning the study's primary metric into a
selection artefact. A uniform random draw is also the correct question to ask:
what fraction of the distribution production RL actually sees yields gradient.
MATH-500's L3-5 restriction is legitimate by contrast — those levels are
dataset labels, not a measurement of any cell in this grid.

The existing 18-prompt held-out set is not usable for this study, on power
grounds: SE is about 6 points, which cannot resolve the differences at stake.
Its observed bimodality (~7 prompts at 4/4, ~8 at 0/4) is not a defect to be
selected away — if the distribution really is that degenerate, that *is* the
finding, and it is precisely metric 2. The fix is a larger unbiased sample that
estimates the degeneracy rate accurately, not a curated sample that hides it.

## Controls

**Pairing.** Same prompts across cells, with the sampling seed derived from
`prompt_idx`, so all three cells see the same rollout noise on the same
problem. Paired comparison at 400 prompts puts the standard error well below
the 6 points of the 18-prompt set, which is what makes a 3-4 point difference
readable.

**Single 16384-token budget, clean cap, no forced answer.** One budget for all
cells, equal to the production v3 `max_new_tokens`. `pretrain` terminates near
1k and costs nothing extra; only A and C pay. BFT forcing is deliberately
removed from the primary path: forcing `\boxed{` onto unfinished reasoning
manufactures coin-flip answers — in the 2B era, forced guesses accounted for
~75% of observed variance — which would contaminate the very headroom metric
under study. An at-cap rollout is scored as a failure, which keeps metric 5
honest.

**The forced reading is recovered for free.** Phase 1 is the entire cost; BFT
forcing is a 512-token continuation on unfinished rollouts only. Both readings
— natural pass@k and BFT-forced pass@k — are computed from the same phase-1
rollouts for roughly 10% additional cost, preserving comparability with
existing production numbers without paying the contamination in the primary
metric.

**One extractor for all cells**: last `\boxed{...}` or last `Answer:` line,
then the existing `norm()` from `math_eval.py` — reusing the calibrated
normaliser rather than rewriting it, so this study's grading does not drift
from previous measurements. Failure rate reported per cell, plus manual review
of a sample.

One amendment to `norm()`, found by the pilot and mandatory: it strips `$` but
not the inline-LaTeX delimiters `\(`..`\)` and `\[`..`\]`, because in production
it only ever sees `\boxed{}` payloads. The base model under the DAPO prompt
emits them spontaneously — `\(p - q\)`, `\(\frac{14}{3}\)` — and they were
scored wrong against correct ground truth. Uncorrected this penalises exactly
one cell by typographic convention and would have produced "the base model has
less headroom", the reverse of the truth. On the 8-prompt pilot it flipped 3 of
32 rollout verdicts in cell B alone.

**Identical sampling everywhere**: T=0.6, top_p=0.95, top_k=20 (production
config). Because sampling truncates at top-20, requesting `logprobs=20` yields
the *exact* entropy of the sampled distribution rather than an approximation.

## Stage 0: pilot

Run before the full sweep. 8 prompts (4 OMI, 4 MATH-500 L3-5), 4 rollouts,
all three cells, full 16384 budget — roughly 96 rollouts, about 10 minutes.

The pilot exists to answer one scientific risk and five mechanical ones.

**Scientific risk: does `Qwen3.5-4B-Base` follow the DAPO prompt zero-shot at
all?** DAPO used it on a 32B base with substantial emergent instruction
following. A 4B base may digress and never emit `Answer:`. If it does not, cell
B needs few-shot prompting, which is a design change and must be decided before
the full run — not discovered seven hours in.

Mechanical checks:

1. vLLM runs on sm_120 (Blackwell) at all, or fall back to transformers.
2. Both prompt formats produce parseable output.
3. The extractor handles both formats; inspect raw text, not just the rate.
4. Entropy extraction from `logprobs=20` works.
5. Termination and at-cap detection are correct.

## Pilot findings (2026-08-03, 8 prompts x 4 rollouts x 3 cells)

Plumbing validated. Substantive results below are on 8 prompts and are far
under any threshold for reading differences in means; only the qualitative
near-unanimous observations are asserted.

**The extractor is sound.** Across 96 rollouts the extract-by-termination
cross-tab has zero off-diagonal entries: every extraction failure is a
truncation, none is a parsing failure. A: `boxed+eos 19, boxed+cap 3,
none+cap 10`. B: `answer_line+eos 27, none+cap 5`. C: `answer_line+eos 25,
answer_line+cap 1, none+cap 6`.

**The base model follows the DAPO prompt zero-shot.** No few-shot needed;
cell B stands as designed.

**Reasoning is in the weights, not the prompt.** Given a raw completion prompt
with no chat template and no `<think>` primer, `upstream` opens a think block
in 32/32 rollouts (cell C). In cell A the tag is absent from the completion
because the chat template opens it and the model only closes it (19/32).

**`Qwen3.5-4B-Base` is not a tabula-rasa base.** It opens `<think>`
spontaneously in 27/32 rollouts under the same raw prompt, closes it 22/32, and
follows the `Answer:` instruction without examples. It also ships the identical
7756-character chat template as the post-trained model. The clean base
condition DAPO enjoyed (Qwen2.5-32B, 2024) does not exist in this family, so
`C - B` measures only the final post-training stage layered on a model that
already reasons -- not "base vs post-trained". Pre-reasoning-era candidates
**A genuine base arm exists: `Qwen3-4B-Base`.** Probed on the same 8 prompts:
0/32 spontaneous `<think>`, 100% EOS, 0% at cap, median 312 tokens, and it does
reason -- 27/32 rollouts carry 150+ tokens of `Step 1: ... Step 2: ...`, with
only 2/32 answering immediately. Added as cell Q3B.

All four cells regraded with the fixed normaliser (caps differ, so EOS/at_cap/
median length compare only within A-vs-C and B-vs-Q3B):

| cell | model | fmt | cap | p@1 | p@4 | nondeg | EOS | at_cap | ent | med_len | `<think>` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | Qwen3.5-4B | native | 8192 | 0.625 | 0.750 | 0.250 | 0.594 | 0.406 | 0.249 | 4717 | 0/32 |
| C | Qwen3.5-4B | dapo | 8192 | 0.625 | 0.750 | 0.250 | 0.781 | 0.219 | 0.190 | 1784 | 32/32 |
| B | Qwen3.5-4B-Base | dapo | 4096 | 0.750 | 0.750 | 0.125 | 0.844 | 0.156 | 0.271 | 952 | 27/32 |
| Q3B | Qwen3-4B-Base | dapo | 4096 | 0.750 | 0.875 | 0.625 | 1.000 | 0.000 | 0.318 | 312 | 0/32 |

Along the chain true base -> reasoning-mid-trained base -> post-trained,
entropy falls monotonically (0.318 -> 0.271 -> 0.190), length grows (312 -> 952
-> 1784), and the non-degenerate group rate collapses (0.625 -> 0.125). At 8
prompts no single difference is significant; what is notable is that every
metric moves the same way. The full run exists to test whether this survives at
400 prompts.

## Production finding: the OMI corpus is over-counted

Not part of the study's question, found while mirroring production sampling.

`VirtualParquetDataset` defaults to `data_dir="data"` and
`_ensure_manifest` (virtual_parquet.py:271) takes *every* `.parquet` under it.
For `nvidia/OpenMathInstruct-2` that is 32 `train-*` shards (the full corpus)
plus 23 `train_1M/2M/5M-*` shards, which are curated subsets *of* `train`.

Measured: `len(env)` = **21,972,791** against a `train` corpus of 13,972,791
rows. The difference is **exactly 8,000,000** = 1M + 2M + 5M, so 36.4% of the
index space is duplicate addresses for curated rows. `pick_prompt_idx` draws
uniformly over that index, so a row belonging to the subsets is drawn 2-4x more
often than an ordinary row depending on its nesting, and production math prompt
selection is silently weighted toward the curated portion of OMI-2. A 6-prompt
probe drew global index 21,167,265 -- well past 13.97M, i.e. a duplicate.

Confirmed on the real draw: of the 200 OMI prompts sampled for this study, 83
(41.5%) come from `train_subset` shards, against the 36.4% the index arithmetic
predicts. By `problem_source` the draw is 88.5% synthetic augmentations
(`augmented_math` 153, `augmented_gsm8k` 24) versus 23 original problems.

This study keeps production-faithful sampling -- that is the right question
here -- but records each prompt's shard family so the skew is visible rather
than absorbed.

## Pre-registered decision rule

Fixed before seeing results, so the reading cannot be rationalised afterwards.
The primary discriminator is **C vs B** — same format, different weights.

- **B >> C** (base has materially more exploitable groups and a wider
  `pass@8 - pass@1`) — post-training has collapsed the distribution; the
  starting point is the problem. Follow-up: RL from `Qwen3.5-4B-Base`, or
  re-inject entropy (higher temperature, DAPO clip-higher, entropy bonus).
- **B ~= C** — weights are not the cause; post-training did not consume the
  headroom. The plateau is then attributable to the pipeline, where two
  independently quantified defects are already on the table: the data diet
  (`value_fn_sim` shows `DIFFICULTY_AUCTION_DELTA` 1 -> 0 moves the code diet
  from 67% to 13% k=2) and the objective (mis-scaled clip band, absent
  Clip-Higher and Soft Overlong Punishment, non-anchoring KL reference). This
  study does not separate those two from each other.
- **B << C** — the base model is simply weaker and offers less signal; DAPO's
  result came from a 32B, not from being base. The post-trained model is the
  right starting point and the cause lies elsewhere.

Cross-reads:

- If C's entropy is far below B's, that is the mechanism underlying the first
  branch.
- **A vs C** answers the format sub-question. If A ~= C, the `<think>`
  behaviour is baked into the weights rather than prompted, and no prompt
  change will escape it.

Guard case: if all three cells show near-zero non-degenerate groups on OMI but
healthy rates on MATH-500 L3-5, the OMI slice itself is miscalibrated — which
would explain the starved training diet without implicating any model.

## Infrastructure

Single RTX 5090, 32 GB, torch 2.12.0+cu130 preinstalled, 62 GB RAM, 12 cores,
830 GB disk, no nvcc.

- vLLM in an isolated venv (it pins its own torch; the preinstalled 2.12+cu130
  must not be disturbed). Blackwell sm_120 requires a cu128+ torch build —
  verified by the pilot smoke test before anything else.
- `VLLM_USE_FLASHINFER_SAMPLER=0` (the sampler JIT needs nvcc, absent here).
- `if __name__ == "__main__":` guard is mandatory — the V1 engine spawns
  EngineCore and forks repeatedly without it.
- `pkill -f EngineCore` on cleanup, or an orphan holds VRAM.
- GatedDeltaNet runs in torch fallback; fla and causal-conv1d cannot be built
  without nvcc. Functional but slow.
- One model resident at a time: 4B bf16 is 8 GB of 32, remainder to KV cache.

Estimated cost: A and C dominate at ~3200 rollouts each, mean ~5500 tokens,
~18M tokens per cell; 2-3 h per cell in torch fallback. B terminates short,
~20 min. Total ~6-7 h of GPU plus downloads.

## Deliverables

`.r2_analysis/headroom/`:

- `build_prompts.py` — assemble and stratify the 400-prompt set, emit both
  prompt renderings per problem
- `run_cell.py` — run one cell, emit per-rollout JSONL
- `analyze.py` — the seven metrics plus the paired comparison table

## Limitations

- **Headroom is necessary, not sufficient.** Measuring headroom in `pretrain`
  does not prove DAPO would convert it into gains. If the diagnostic reports
  headroom in the base model, the follow-up is a single confirmatory RL run.
- **Contamination.** OMI and MATH-500 are public and very likely present in
  Qwen pretraining. This inflates absolute levels. It does not break the
  comparison — contamination is shared across all three cells, two of which
  have identical weights — but it forbids reading pass@1 as true capability.
- **The production checkpoint is out of scope**, so this study measures the
  headroom of the starting point without measuring how much production RL has
  consumed. Adding `ckpt88` would close that loop.
