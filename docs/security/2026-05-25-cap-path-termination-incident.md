# 2026-05-25 Cap-Path Termination Incident

## Summary

The validator accepted rollouts that reached the protocol cap
(`MAX_NEW_TOKENS_PROTOCOL_CAP = 8192`) before checking whether the rollout
ended with a natural EOS token. This was intended to avoid punishing one
honest long completion that runs out of budget, but it let miners force many or
all rollouts to the cap with `min_new_tokens`, EOS suppression, or equivalent
sampling controls.

The submission is not cryptographically forged: tokens, GRAIL proofs, rewards,
logprob checks, and distribution checks are all computed over real model output.
The abuse is that the sampler is engineered to avoid natural termination, which
turns the cap-path exception into the default path.

## Verified Code Path

In `reliquary/validator/verifier.py`, `verify_termination` returned `True` as
soon as:

```python
prompt_length + completion_length >= MAX_NEW_TOKENS_PROTOCOL_CAP
```

That happened before checking `tokens[-1]` against EOS and before checking
`p_stop >= MIN_EOS_PROBABILITY`.

The batcher already had a `MAX_TRUNCATED_PER_SUBMISSION` policy, but cap
hits were not counted as truncations because `verify_termination` returned
`True`.

## Archive Evidence

Data sources used:

- `https://www.reliqua.ai/api/miners`
- `https://www.reliqua.ai/api/r2/window/<window>`

Hotkey `5HEAK6g3VN6rZUMC1Xe2zn28p78Mxxd3Mih4kZaFE28D5S7E` maps to UID 181 in
the public miners endpoint.

For windows `6275-6300`, across accepted archived rows with rollouts:

- UID 181 hotkey rows: 55
- Rows with all 8 rollouts at completion length `8192`: 53
- Rows with at least 6 of 8 rollouts at `8192`: 55
- Cap rollouts: 438 / 440
- Cap rollouts marked `eos_terminated=False`: 438 / 438

The report's sample rows match the archive, for example:

- `w6275`, `prompt_idx=54710`, `sigma=0.500`, rewards
  `[0, 1, 0, 0, 1, 1, 0, 1]`, lengths `[8192] * 8`
- `w6275`, `prompt_idx=43886`, `sigma=0.484`, rewards
  `[1, 0, 1, 1, 0, 0, 0, 0]`, lengths `[8192] * 8`
- `w6276`, `prompt_idx=533545`, `sigma=0.500`, rewards
  `[1, 0, 0, 1, 0, 0, 1, 1]`, lengths `[8192] * 8`

For windows `6315-6330`, UID 181 had 6 analyzable accepted rows with rollouts:

- Rows with all 8 rollouts at `8192`: 2
- Rows with at least 6 of 8 rollouts at `8192`: 2
- Cap rollouts: 28 / 48

The reported mixed-length examples also match the archive:

- `w6324`, `prompt_idx=314236`, lengths
  `[1212, 8192, 8192, 8192, 3184, 8192, 8192, 1660]`
- `w6324`, `prompt_idx=857453`, lengths
  `[8192, 566, 1505, 8192, 500, 532, 565, 8192]`

Other hotkeys also showed the same fingerprint in the pre-window sample, so
UID 181 was not necessarily the only miner using cap-path forcing.

## Impact

Cap-forced rollouts typically contain a normal answer attempt near the
beginning, followed by thousands of tokens of repetition, rambling, or trailing
answer restatement because the sampler was not allowed to stop. Those rollouts
can still land in the reward sigma zone and enter GRPO training, where they
teach the model that excessively long non-EOS completions are acceptable.

Expected symptoms:

- natural EOS behavior degrades;
- completions become verbose or loop-prone;
- training consumes many tokens on low-value tails;
- quality metrics can look acceptable per window while model behavior worsens.

## Patch

Initial hardening counted cap hits without natural EOS as truncations and made
sure tolerated truncations still pass GRAIL/logprob/distribution/boxed checks.
The current acceptance budget is intentionally loose
(`MAX_TRUNCATED_PER_SUBMISSION = 5`,
`BOOTSTRAP_MAX_TRUNCATED_PER_SUBMISSION = 5`) while the team observes miner
adaptation. Training quarantine separately tracks dense cap/extreme-length
patterns so one bad long sample does not freeze checkpoint progress by itself.

Natural EOS at the cap is not counted as truncation if:

- the last token is in the model/tokenizer EOS set; and
- `p_stop >= MIN_EOS_PROBABILITY`.

Submissions with repeated forced-cap rollouts now reject as `bad_termination` in
steady state.

## What To Monitor

High-confidence indicators:

- `count(completion_length == 8192 and eos_terminated == false) >= 2` within a
  single 8-rollout accepted group;
- all 8 rollouts at exactly `8192`;
- rolling hotkey cap-rollout fraction above a small baseline;
- sudden drop in natural EOS rate by hotkey;
- mean completion length above `7000` with low variance and no EOS.

Audit query shape:

```text
for each archived accepted row:
  cap_no_eos = rollouts where completion_length >= 8192 and eos_terminated is false
  flag if len(cap_no_eos) >= 2
```

This catches the observed exploit while still allowing occasional honest
runaways.
