# 2026-05-25 Manufactured Inner Rollout Incident

## Summary

After the rollout-token invariant and cap-path termination fixes, a third
accepted-submission pattern appeared: direct-answer templates with hundreds of
repeated EOS tokens. The rollouts are internally consistent (`tokens ==
commit["tokens"]`) and the repeated EOS tail has very high model probability, so
GRAIL, logprob, termination, and distribution checks can all pass while the
training data is semantically manufactured.

The low-risk protocol fix is to reject EOS padding. Honest generation stops at
the first EOS, and the reference miner truncates at the first EOS before
building the proof. Any token after a completion EOS, including another EOS, is
not useful training signal.

## Verified Evidence

Data sources:

- `https://www.reliqua.ai/api/miners`
- `https://www.reliqua.ai/api/r2/window/<window>`

The public miners endpoint maps hotkey
`5FjueYvWfcACczuiCTXJ1Z9vb5hbJXQNy4rm3MUF5Vp6HSdX` to UID 18.

Across windows `6300-6346`, 22 accepted archived rows with rollouts were found
for that hotkey. All 22 had:

- rewards exactly `[1, 1, 1, 1, 1, 0, 0, 0]`;
- fixed direct-answer prefixes in slots 0-4;
- fixed distractor prefixes in slots 5-7;
- hundreds of repeated `<|im_end|>` tokens in every rollout;
- `dist_q10_min` near 1.0 and `lp_dev_max` near zero.

Representative examples:

- `w6336`, `prompt_idx=626461`, `ground_truth="\\frac{35}{72}"`,
  boxes `["\\frac{35}{72}", ..., "0", "1", "2"]`,
  EOS-text counts `[512, 517, 522, 527, 532, 537, 542, 547]`.
- `w6339`, `prompt_idx=505013`, `ground_truth="45"`,
  boxes `["45", ..., "0", "1", "2"]`,
  EOS-text counts `[512, 517, 522, 527, 532, 537, 542, 547]`.

The pattern continued beyond the quoted range:

- `w6347` and `w6348` still had accepted rows with the same structure.
- `w6352` and `w6353` each had 6 accepted rows with the same structure.

## Why Existing Checks Passed

- Token invariant: the submitted top-level tokens and commit tokens are the
  same.
- GRAIL: verifies the provided token sequence against the model forward pass.
- Logprob: the claimed logprobs match the model logprobs for those chosen
  tokens.
- Distribution: repeated EOS after EOS has very high chosen-token probability,
  so q10/median look excellent.
- Termination: the final token is EOS with high `p_stop`.

The validator did not previously reject tokens after the first EOS.

## Fix

Reject any completion where EOS appears before the final completion token, or
where EOS appears more than once in the completion. This enforces the same
semantics used by the reference miner: first EOS ends the completion.

This is deliberately not a phrase/template blacklist. It targets the protocol
violation that made the exploit cheap and invisible to the existing behavioural
checks.

## Ongoing Monitoring

Watch for:

- `bad_termination` spikes immediately after deployment;
- hotkeys whose archived completions contain repeated `<|im_end|>`;
- repeated fixed reward vectors such as `[1, 1, 1, 1, 1, 0, 0, 0]`;
- repeated direct-answer prefixes across all slots;
- constant wrong-answer distractors reused across unrelated prompts;
- high `dist_q10_min` combined with low semantic diversity.

If attackers remove EOS padding and use other high-probability filler before a
single terminal EOS, the next layer should be a cross-rollout structural
detector rather than a hardcoded phrase list.
