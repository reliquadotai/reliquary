# Code graded-span fix

The Code grader executes the last fenced block. From protocol v5 on it executes
the last fenced block that *defines the contract's entry function* instead.

Nothing else moves: same model, tokenizer path, sampling distribution, rollout
count, token cap, BFT policy, DAPO objective controls, and **both prompts
unchanged**. v2-v4 keep the old rule byte-exact as historical controls.

> **This redefines the reward in place on the live profile.** It is not a
> coordinated cutover, and that is a deliberate trade. The `/state` generation
> contract does not change, so miners on older code keep passing
> `_state_matches_active_protocol` and keep mining — but their reward diverges on
> multi-block rollouts, and `reward_mismatch` rejects the **whole group of 16**.
> At the measured 0.47% divergent-rollout rate that is roughly **7% of groups
> lost per non-updated miner**, silently, until they update. Two nodes can also
> both claim profile v5 with the same `generation_contract_sha256` and grade
> differently; checkpoint lineage will not distinguish a checkpoint trained
> before the fix from one trained after. Announce the update to miners.

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
nothing is inferred from the completion text. Single-block rollouts (83%) are
untouched.

It also drops the raw-completion fallback: with no fence, the graded span is
empty rather than the whole rollout. That changes no observed reward — the
fallback fired 762 times across 30 768 production rollouts and never once
produced a positive one, because a rollout holding code always fences it — so in
practice it only stops the sandbox from running `exec` on reasoning prose.

Both call sites — `OpenCodeInstructEnvironment.compute_reward` and
`admission._compute_code_rewards` — are pinned by a test to grade the same span.
A divergence between them would reject honest miners on `reward_mismatch`.

**Neither prompt moves.** Rewording the Code prompt was measured and rejected: on
the production checkpoint (650, pinned revision, 2 560 rollouts, real grader) it
raises prose from 10.2% to 52.7%, but its reward effect is **not significant on
any stratum with headroom** (base model: +0.039 at t=1.14 on hard problems,
+0.024 at t=1.29 on medium) and is **significantly negative at ceiling** (−0.081
at t=−2.66). Against that, moving a prompt costs a real distribution transient —
the v5 cutover dropped Math reward 0.622 → 0.275 for ~70 windows.

**Known consequence.** With the prompt unchanged, reasoning does not come back on
its own. On the production checkpoint with the fixed extractor the paired
reasoning effect is −0.009 (ns) — neutral, and DAPO learns only from a
differential, so there is no pull back toward prose. Code rollouts stay near 10%
prose. This fix stops the mechanism from punishing reasoning; it does not restore
it. Restoring it needs a prompt change, which should wait for a
headroom-enriched measurement rather than ride along here.

## Evidence

Two H100 runs — the pinned base model, and the **live production checkpoint 650**
at its current revision — 2 560 rollouts each under the exact v5 sampling
contract, every rollout graded twice by the real grader worker. On base the
harness reproduces production closely (prose 77,6 % vs 77,1 %, multi-block
31,9 % vs 26,7 %, no-fence 4,5 % vs 4,31 %).

| model / prompt | reward, `matches[-1]` | reward, entry rule | improved | **degraded** |
|---|---|---|---|---|
| base, current prompt | 0,5707 | 0,6785 | 152 | **0** |
| base, reworded prompt | 0,5356 | 0,6933 | 222 | **0** |
| ckpt650, current prompt | 0,8634 | 0,8672 | 5 | **0** |
| ckpt650, reworded prompt | 0,7838 | 0,8334 | 66 | **0** |

Zero regressions in all four configurations. On the *current* policy the
immediate gain is small — +0.37 points, 5 rollouts in 1 280 — because the model
has already adapted around the defect by not opening a second block (1.8% of
rollouts, against 31.9% for the base). This fix is prophylactic: it removes the
mechanism, it does not recover much that is still being lost today.

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

**Nothing to do.** With no profile fork, `validate_checkpoint_profile` sees no
lineage change and the run continues uninterrupted — no re-stamp, no new
`RELIQUARY_TRAINING_RUN_ID`, no LR warmup replay.

That convenience is the flip side of the trade in the banner above: the lineage
metadata will not record that the reward function changed, so a checkpoint
trained before the fix and one trained after are indistinguishable from their
profile alone. Note the window number of the deploy somewhere durable.

## Post-deploy checks

Over the first ~200 windows (≈6 h):

- artefact zeros (multi-block whose last block lacks `def <entry>`) below 0,2 %;
- `reward_mismatch` rejections: expect a spike from non-updated miners, and
  watch it decay as they upgrade. A flat spike means someone is stuck.

## Known gaps, not addressed here

- **`_extract_python_span` in `verifier.py` still uses `matches[-1]`.** It feeds
  the `code_semantic_auth` shadow detector, which needs character offsets into
  the completion rather than the code string. Under the entry rule the graded span and the
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
