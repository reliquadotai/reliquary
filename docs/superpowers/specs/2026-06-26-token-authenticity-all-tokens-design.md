# Generalize token-authenticity to all completion tokens

- Date: 2026-06-26
- Branch: fix/grader-entry-resolution (work to land on its own branch)
- Status: design, approved for implementation

## Problem

The validator runs three token-level authenticity checks per rollout
(`batcher.py:1006-1062`), for both math and code:

1. `evaluate_token_distribution` — soft statistical check (median, q10 of
   chosen-token probs).
2. `evaluate_boxed_answer_probability` — the strong combined check
   (`p_chosen < threshold` **and** `p_argmax >= argmax_conf` -> tampered) but
   **only over the `\boxed{...}` region** (the numeric answer in math). Code has
   no boxed region, so it returns `True` (passes) without checking anything.
3. `evaluate_token_authenticity` — runs over **all** completion tokens, but only
   tests `p_chosen < 1e-10` with **no argmax cross-check**. `1e-10` is so low
   that a tampered token at `p_chosen ~ 1e-7` with `p_argmax ~ 0.99` passes.

So the strong "chosen collapsed while top-1 is near 1" detection exists only for
the numeric/boxed answer region. For arbitrary code (and math reasoning) tokens,
nothing catches that pattern. A miner editing a non-boxed token post-hoc slips
through.

The per-token data needed to close this already exists:
`_gpu_completion_token_stats` (`verifier.py:397`) computes dense `chosen`,
`argmax`, `argmax_id` for **every** completion-producing position, 1:1 aligned.
No extra forward pass is needed.

## Goal / non-goals

**Goal.** Apply the combined "chosen collapsed + argmax near 1" detection to all
completion tokens (env-agnostic), so post-hoc token edits anywhere in a code or
math completion are detectable — without raising the honest false-positive rate.

**Non-goal.** Changing production reject behavior at ship time. The tightened
threshold is enabled only after the offline calibration proves false positives
are ~0. This design does not touch the boxed check, the distribution check, or
the reward/zone economics.

## Why false positives are the central risk

Code-token tampering was tested on A100 (2026-06-14) and **not confirmed** —
honest code rollouts also carry low-probability non-argmax tokens (identifier
names, whitespace, equivalent syntax). On top of that, the miner samples under
vLLM while the validator forwards under HF; the resulting numerical drift can
depress an honest chosen-prob. Over a several-hundred-token code completion, the
chance that *some* honest token dips below threshold while argmax is high is far
larger than in the tiny boxed region. The `argmax >= argmax_conf` gate is the
primary FP control: it excludes high-entropy positions (where a low chosen prob
is legitimate) and keeps only positions where the model was confident of a
*different* token — the signature of an injection. The exact threshold is an
**empirical** quantity, calibrated on the GPU before any enforcement.

## Design — Part A: the generalized check (validator code)

Evolve `evaluate_token_authenticity` so that, for each completion position `j`,
it flags when:

```
chosen[j] < TOKEN_AUTH_THRESHOLD  and  argmax[j] >= TOKEN_AUTH_ARGMAX_CONF
```

returning the first offending position and metrics (`pos`, `p_chosen`,
`p_argmax`, `argmax_id`). This is exactly the boxed check's core, with the
region restriction removed. A single global threshold and a single global
`argmax_conf` govern all environments (per-env override is explicitly out of
scope unless calibration later shows code and math diverge).

The boxed check (`evaluate_boxed_answer_probability`, threshold `1e-3`) stays in
place. It is stricter on the answer region than the generalized check will be,
so the two are complementary, not redundant.

**Ship-safe defaults (prod behavior unchanged at ship):**

- `TOKEN_AUTH_THRESHOLD` stays at `1e-10` at ship. Adding the argmax gate at
  this threshold makes the enforced check strictly *less* likely to fire than
  today (today it fires regardless of argmax), so no honest rollout that passes
  now starts failing. The change is net-conservative until we deliberately raise
  the threshold.
- `TOKEN_AUTH_ARGMAX_CONF` stays `0.99`.
- `TOKEN_AUTH_ENFORCE` stays as-is (reject gating unchanged).

Enabling = bumping `TOKEN_AUTH_THRESHOLD` to the calibrated value in a follow-up
commit, with the FP evidence in hand. No live shadow plumbing is required: the
calibration is offline against a known-honest corpus, so the threshold is never
raised in prod until the harness says FP is ~0.

## Design — Part B: offline FP calibration harness (GPU)

Extend `.r2_analysis/forward_scan.py` (which already forwards the validator
model over the exact submitted tokens, computes per-position chosen/argmax, and
implements revert-and-regrade) into a calibration sweep:

1. **Corpus.** N archived **accepted** rollouts (passed all current checks and
   earned reward -> presumptively honest), sampled across both math and code,
   sampled under vLLM by real miners. Only the submitted tokens are needed; the
   harness recomputes chosen/argmax under HF at `T_PROTO=0.9` — this reproduces
   the real vLLM->HF drift, the dominant FP source.
2. **Sweep.** For `threshold in {1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3}` x
   `argmax_conf in {0.95, 0.99, 0.999}`, count, per combination, the fraction of
   rollouts that would fire, plus the `(chosen, argmax)` distribution at fired
   positions.
3. **TP/FP discrimination.** For each fired rollout, revert the flagged token to
   the model argmax, re-extract the answer/code, and re-grade. If the reward
   flips (pass->fail) the edit bought the pass -> **true positive**. If the
   reward is unchanged -> benign low-prob token -> **false positive**. Report:
   - honest FP rate = (fires that do NOT flip reward) / total rollouts,
   - newly-caught TP count = fires among accepted rollouts that DO flip reward.

The accepted-rollout corpus is the correct denominator: it answers "of what we
accept today, how many would the tightened filter newly reject, and of those how
many are real edits vs honest tokens."

## Decision criterion for enabling

Pick the `(threshold, argmax_conf)` with the highest threshold whose honest FP
rate is ~0 (flat at the bottom of the FP curve) while still sitting above
plausible tampering (`chosen ~ 1e-8..1e-6`, `argmax >= 0.99`). Only then bump
`TOKEN_AUTH_THRESHOLD` and rely on the existing `TOKEN_AUTH_ENFORCE` path. If no
threshold gives a flat-zero FP curve, do **not** enable; report the floor FP and
revisit (e.g. require K>=2 fired tokens, or a per-env split).

## Testing

Unit tests for the evolved `evaluate_token_authenticity` (extend
`tests/unit/test_behavioural_validators.py`):

- honest completion (all chosen probs healthy) -> `ok=True`;
- injected token (one position `chosen < threshold`, `argmax >= argmax_conf`) ->
  `ok=False`, correct `pos`/metrics;
- low chosen prob but `argmax < argmax_conf` (genuine high-entropy / drift) ->
  `ok=True` (the gate protects it);
- empty stats -> `ok=True`;
- threshold/argmax_conf passed as params are honored.

The GPU acceptance harness (`scripts/verify_token_authenticity.py` pattern)
asserts every measured-injected rollout is flagged and every honest vLLM
completion passes at the chosen threshold.

## Rollout sequence

1. Land Part A (gated at `1e-10`, prod-neutral) + unit tests on a feature
   branch.
2. Stage the accepted-rollout corpus on the GPU box (root@64.247.206.243:40299)
   and run the Part B sweep.
3. Read the FP curve; pick the threshold (or decide not to enable).
4. Follow-up commit bumps `TOKEN_AUTH_THRESHOLD` with the FP/TP numbers in the
   commit message.

## Risks

- **vLLM->HF drift floor.** If drift alone pushes honest chosen probs below the
  useful threshold band, the filter has no safe operating point — the harness
  will show this as a non-zero FP floor. Mitigation: keep enforce off; the
  design fails safe.
- **Survivorship in the corpus.** Accepted rollouts already passed today's
  filters, so the honest set is slightly optimistic. The TP/FP revert-regrade
  split corrects for this by labeling each fire rather than assuming all fires
  are FP.
- **Spec/harness secrecy.** Miners read this repo. This planning doc and the
  calibration scripts stay untracked (not committed/pushed) unless explicitly
  requested; only validator code + tests land.
