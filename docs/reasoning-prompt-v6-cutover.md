# Graded-span v6 cutover

Protocol v6 changes which fenced block the Code grader executes. The model,
tokenizer path, sampling distribution, rollout count, token cap, BFT policy,
DAPO objective controls, and the **Math prompt** are all unchanged. Protocol v5
remains immutable as the historical last-fence control.

## Diagnosis

`_extract_python` returned `matches[-1]` — the last fenced block. The rule was
correct for the regime it was written in (2026-05-24, `99576d5`), when the
environment ran a chat model that closed with its final implementation; the
original comment says so: *"the last block wins (model's final answer wins over
earlier drafts)"*.

The v5 reasoning prompt invalidated that assumption without touching the
extractor. A base model asked to reason writes the implementation, then
illustrates it — and the illustration is fenced too. Measured on R2 archives at
the v5 cutover (w31050–31110, 15 408 code rollouts):

| | share of code rollouts |
|---|---|
| multi-block rollouts | 26,7 % |
| of those, last block is **not** the implementation | 49 % |
| **zeroed by mis-extraction** | **13,1 %** |
| zeroed legitimately (no code produced at all) | 4,3 % |

Grouped by what actually happened, over 30 768 production rollouts:

| case | n | mean reward | reward > 0 |
|---|---|---|---|
| demo last, implementation present above | 2 129 | **0,000** | **0** |
| implementation last — twin group | 2 073 | 0,662 | 74,4 % |

Because DAPO trains on the group-relative advantage, those zeros are not merely
lost reward. A rollout that opens a second block carries −1,0 to −1,2 σ, whether
or not it reasons. The policy learned "never open a second block" and
generalised it to "never reason": prose in Code rollouts fell from 80,9 % at the
cutover to 15,6 % two days later, median length 512 → 312.

Two hypotheses were tested and rejected. **The token cap is not involved**: p99
completion length is 896 against a cap of 8192, one rollout in 15 360 exceeds
90 % of the cap, and zero rollouts end without EOS. **It is not miner
curation**: the fall is uniform across every hotkey, and the per-group count of
reasoning rollouts follows Binomial(16 ; 0,149) — selection of short groups
would hollow out the middle of that distribution.

## The change

`_extract_python(completion, entry_name)` returns the last fenced block that
contains `def <entry_name>`, falling back to `matches[-1]` when no block defines
it. `entry_name` comes from the structured cases — the same source that already
writes *"Write your solution as a Python function named …"* into the prompt — so
nothing is inferred from the completion text.

Gated on `PROTOCOL_VERSION >= 6`. **This is wire-affecting.** Miners compute
their own reward (`miner/engine.py`) and the validator re-runs
`env.compute_reward` and rejects a mismatch beyond `1e-6` (`verifier.py`,
`reward_mismatch`). Both sides import the same extractor, so changing the graded
span on one side alone rejects honest miners as dishonest. The gate keeps the
new rule inert until the profile flips, which makes the deploy two-phase: ship
the code everywhere, then switch the profile.

> **Coordination with PR #198 (checkpoint-epoch scheduling).** The two-phase
> deploy above is the standalone path, and it has a real weakness: between
> shipping the code and flipping the profile there is a window where miners and
> validator can disagree about which profile is active. #198 introduces an epoch
> manifest whose `protocol` block carries exactly
> `{profile_id, protocol_version, generation_contract_sha256}` — the same triplet
> a v6 fork changes — alongside `activation_not_before_round`. If that lands,
> **activate v6 through the manifest rather than by flipping an env var**: the
> cutover becomes atomic at an agreed round instead of a redeploy race. This
> change does not depend on #198 and does not block it; only the activation
> mechanism below would be superseded.

The Code prompt loses its position clause, which only ever existed to
compensate `matches[-1]`:

```diff
- After your reasoning, provide the final implementation in the last fenced Python code block.
+ Work through your reasoning first, then give the complete implementation in a Python code block.
```

`complete` targets the rollouts that call helpers they never wrote.

v6 also drops the raw-completion fallback: with no fence, the graded span is
empty rather than the whole rollout. This changes no observed reward — the
fallback fired 762 times across 30 768 production rollouts and never once
produced a positive one, because a rollout holding code always fences it — so
in practice it only stops the sandbox from running `exec` on reasoning prose.
It rides the same gate anyway: a rollout of bare valid Python would score
differently, and a staggered miner/validator deploy would surface that as
`reward_mismatch`.

**The Math prompt is byte-identical to v5, deliberately.** Math has no
extraction defect — its median length rises 376 → 561 under v5 and its reward
rises with it. Re-rendering its prompt would only re-inflict the transient the
v5 cutover cost Math (reward 0,622 → 0,275, recovered over ~70 windows) for no
benefit. `tests/unit/test_reasoning_prompt_v6_profile.py` pins this.

## Pre-activation evidence

2 560 rollouts generated on an H100 from `Qwen3-4B-Base` at the pinned revision
under the exact v5 sampling contract, each graded twice by the real grader
worker. The harness reproduces production closely (prose 77,6 % vs 77,1 %,
multi-block 31,9 % vs 26,7 %, no-fence 4,5 % vs 4,31 %).

| prompt | reward, `matches[-1]` | reward, entry rule | improved | **degraded** |
|---|---|---|---|---|
| v5 | 0,5707 | 0,6785 | 152 | **0** |
| v6 | 0,5356 | 0,6933 | 222 | **0** |

Of 442 rollouts where the two rules extract a different block, 374 improve, 68
are unchanged, none regress. Recovered rollouts go from 0,00 to 0,91.

The gradient direction reverses, which is the point of the change:

| DAPO advantage (r−µ)/σ | `matches[-1]` | entry rule |
|---|---|---|
| ρ(length, reward) within group | −0,103 | **+0,031** |
| shortest quartile | +0,079 | −0,121 |
| middle quartiles | +0,045 | **+0,090** |
| longest quartile | **−0,178** | **−0,005** |
| rollouts that reason | +0,011 (ns) | **+0,050** (t=2,40) |
| rollouts that dump code directly | −0,051 (ns) | **−0,232** (t=−4,30) |

The resulting shape is an inverted U — terseness penalised, the middle band
rewarded, the longest neutral — which rewards reasoning without paying for
rambling. Note this also settles an earlier hypothesis: the length bias was not
an independent property of the objective, it was the extraction defect projected
onto the length axis, because the long rollouts are the reasoning ones.

## Checkpoint decision

**Warm-start, do not reset.** Changing the reward does not invalidate the
weights; `validate_checkpoint_profile` rejects a v5-stamped checkpoint under v6
on `profile_id`, `protocol_version`, and `generation_contract_sha256`, but that
is a metadata guard, not a mathematical necessity. Re-stamp the checkpoint for
v6 as an explicitly labelled warm start.

Set a **new `RELIQUARY_TRAINING_RUN_ID`**. `training_run_id` is deliberately
outside the validated lineage keys precisely so that a new id on old weights
replays the full LR warmup — which is what you want after a reward change.

A fresh base reset is only justified if a publishable "v6 from base" baseline is
the goal. Otherwise it discards 648 checkpoints and the Math gains to solve a
problem the prompt solves in eight minutes: at the v5 cutover, prose in Code
rollouts went from 36,3 % (w31033) to 83,3 % (w31038) at essentially constant
weights.

## Post-cutover checks

Over the first ~200 windows (≈6 h):

- prose in Code rollouts above 70 % within the first 10 windows;
- artefact zeros (multi-block whose last block lacks `def <entry>`) below 0,2 %;
- paired within-prompt reasoning effect positive on single-block rollouts.

## Known gaps, not addressed here

- **`_extract_python_span` in `verifier.py` still uses `matches[-1]`.** It feeds
  the `code_semantic_auth` shadow detector, which needs character offsets into
  the completion rather than the code string. Under v6 the graded span and the
  authenticated span can therefore differ. The detector reports zero findings in
  production and is not reward-bearing, but aligning it is a genuine follow-up —
  it touches the proof path, so it needs its own change and its own review.
- **The sandbox import allowlist.** `_ALLOWED_IMPORT_ROOTS` holds 17 modules; 33
  of 2 560 rollouts died on it, 15 legitimately (`numpy`, `nltk`) and 14 on
  sensitive stdlib. Four died only on harmless stdlib outside the list — `json`,
  `random`, `textwrap`, and `__future__`. A reflexive
  `from __future__ import annotations` zeroes the whole rollout. 0,16 % of
  rollouts: worth noting, not worth prioritising.
- **Rollouts with no fence at all** keep scoring zero, and that zero is
  deserved: the model produced no code and emitted EOS on its own. Recovering
  them would mean scraping `def` out of unstructured prose, which reintroduces
  exactly the class of ambiguity this change removes, for at most 0,75 % upside
  (231 of 762 such rollouts hold a self-contained definition, and the two
  candidate rules — last definition, or all definitions concatenated — differ by
  17 rollouts).
