# 2026-05-25 Reward Oracle Selection Report

Status: investigated; fleet monitoring added; validator patch prepared

Target hotkey:

```text
5Dd2Q6kPfQy6UCCrB8njLcufZNdo49apkL83rVzxumj7R9Zc
```

## Summary

This is not the same class as the token split, cap path, or EOS padding
exploits. The inspected submissions do not bypass GRAIL, token equality,
termination, logprob, or token-distribution checks.

The stronger explanation is reward-oracle candidate selection:

1. generate more than the protocol needs, or generate many candidate prompts;
2. compute `env.compute_reward` locally;
3. choose exactly 8 model-real rollouts whose reward mix lands in the sigma
   acceptance band;
4. order the selected rollouts by reward, usually wrong first then correct;
5. submit only those selected rollouts.

All submitted tokens can be real model outputs. The loophole is that the
validator verifies the submitted token sequences, but does not verify that the
8 submitted rollouts are the first 8 natural samples, an unbiased sample, or
the full candidate pool.

## Current Evidence

Initial window range inspected:

```text
6428-6458
```

Dashboard/cache status for the target:

```text
rank: 1
EMA: 0.138249
slots in 216-window cache: 73
slots in last 12 cached windows: 26
```

Live verdict sample:

```text
84 accepted
2 logprob_mismatch
```

Known exploit checks on recent selected batch rows:

```text
EOS padding: 0
8192 cap forcing: 0
UID18-style direct boxed template: 0
GRAIL/logprob/distribution: mostly clean
completion length range: 82-1024
mean completion length: about 541 tokens
```

Selected batch row analysis for the target over windows 6428-6458:

```text
selected batch rows: 50
monotonic reward vectors: 50/50
ascending reward vectors: 50/50
average correct count: 5.22 / 8
k=6 rows: 62%
k=5 rows: 18%
k=4 rows: 6%
```

Most common target reward patterns:

```text
[0,0,1,1,1,1,1,1]
[0,0,0,1,1,1,1,1]
[0,0,0,0,1,1,1,1]
```

For 48 recent selected batch rows in windows 6440-6458, the probability that
all reward vectors would be exactly ascending by natural random rollout order
is roughly:

```text
10^-74.6
```

That is enough to treat the ordering as intentional. It is not enough by
itself to prove a validator bypass, because miners can legally know local
rewards today.

Fresh update after acceleration, windows 6507-6530:

```text
selected batch rows: 55
monotonic reward vectors: 55/55
average correct count: 5.09 / 8
k=6 rows: 34
k=5 rows: 7
k=4 rows: 4
k=3 rows: 5
k=2 rows: 5
mean response time: 11.76s
```

This confirms the strategy moved from suspicious to actively dominant. The
target was winning up to 6 selected slots in individual windows.

## Peer Context

The target is an outlier by volume and consistency, but not alone. In windows
6428-6458:

```text
network selected batch rows: 248
network monotonic reward vectors: 180/248
```

Several hotkeys show monotonic reward ordering above 60%. This is likely
becoming a market-wide strategy, not a single-hotkey exploit.

The target is currently winning because it combines:

1. high recent volume;
2. short enough completions for fast verification;
3. low rejection rate;
4. reward-balanced post-selection near the sigma floor;
5. many k=6 groups, which maximize reward mean while still passing
   `SIGMA_MIN=0.43`.

## Concrete Examples

### Window 6440, prompt_idx 385630

Ground truth:

```text
\begin{pmatrix} 11 & 7 \\ 19 & 18 \end{pmatrix}
```

Reward vector:

```text
[0,0,1,1,1,1,1,1]
```

The first two rollouts are not obviously random bad math. One computes the
matrix and then adds an extra requested/non-requested answer, or ends with a
different final boxed value. Later rollouts give the parser-friendly matrix
answer only.

This is a clean example of reward-parser shaping: wrong-to-reward samples can
still contain much of the correct reasoning or answer.

### Window 6457, prompt_idx 609051

Ground truth:

```text
90
```

Reward vector:

```text
[0,0,0,0,0,0,1,1]
```

Several zero-reward rollouts answer `\boxed{90^\circ}` or `90 degrees`.
The reward parser normalizes lightly and compares to plain `90`, so those
format/unit variants can score 0 while parser-friendly `\boxed{90}` scores 1.

## Why Existing Validation Accepts It

Relevant validator flow:

- `validator/batcher.py`: reward claims are recomputed and then sigma is
  checked before GRAIL-heavy validation.
- `validator/verifier.py`: `verify_reward_claim` explicitly accepts
  miner-declared rewards if `env.compute_reward` matches.
- `validator/training.py`: GRPO trains on the submitted rollouts and their
  submitted reward vector after validation.

The validator checks:

```text
tokens are canonical
commit tokens match rollout tokens
signature is valid
randomness is correct
GRAIL proof passes
termination passes
logprobs match
token distribution is not suspicious
reward claim equals env.compute_reward
sigma >= SIGMA_MIN
```

It does not check:

```text
the miner generated only 8 candidates
the miner did not generate a larger pool
the 8 submitted rollouts are an unbiased sample
the reward vector order is natural
the wrong examples are semantically wrong rather than parser-shaped wrong
```

## Why This Matters

The protocol currently rewards being close to the sigma threshold while winning
drand/canonical ordering races. For binary rewards at M=8:

```text
k=2 or k=6 -> sigma = 0.4330
k=3 or k=5 -> sigma = 0.4841
k=4        -> sigma = 0.5000
```

With `SIGMA_MIN=0.43`, k=6 is accepted. That is attractive because it gives a
high reward mean while barely clearing the zone floor.

Training risk:

- the model is trained on a post-selected distribution, not raw policy samples;
- negative examples may be formatting/parser failures, not truly wrong math;
- GRPO can over-emphasize parser-friendly final-answer style;
- if many miners adopt this, the batch stream becomes optimized for validator
  reward mechanics rather than useful frontier learning.

Economic risk:

- miners with enough compute to generate/select from larger pools dominate;
- honest 8-sample miners lose even when they produce valid rollouts;
- volume shifts toward prompt/reward mining rather than model improvement.

## Is This Cheating?

Careful answer: this is a protocol loophole or market-design exploit, not a
cryptographic validator bypass.

It is cheating if the intended rule is:

```text
submit exactly 8 natural, unselected samples from the current policy
```

It is not enforceably cheating under the current validator, because:

```text
the reward function is public and locally computable;
miners already pre-check sigma locally;
GRAIL proves the submitted tokens are model-real, not how many candidates were
generated before them.
```

This means banning a single hotkey is not the clean fix. The rule should be
made explicit and enforced at the protocol level, or the market should accept
reward-selection as part of mining and tune around it.

## Recommended Decision

Do not ship an emergency blacklist-style patch for `5Dd2...`.

Do ship monitoring immediately, and decide whether the protocol wants to allow
candidate-pool reward selection.

If the answer is "no, we want cleaner training data", then patch the validator
rules, not the hotkey.

## Patch Options

### Option A - Low-risk monitoring only

Add a health detector for hotkeys with:

```text
selected batch rows >= 20 over recent windows
monotonic reward vectors >= 95%
average correct count > 5.0 or < 3.0
k=6 or k=2 share >= 50%
low rejection rate
```

This is safe and should happen regardless of policy choice.

Suggested file:

```text
reliquary-fleet/exploit_health.py
```

This should emit a `reward_oracle_selection` warning, not an exploit alert,
unless paired with manufactured text templates, EOS padding, cap forcing, or
other direct manipulation evidence.

Implemented locally in the fleet health script after this investigation.

### Option B - Raise the sigma floor

Change:

```text
reliquary/constants.py
SIGMA_MIN = 0.43
```

to something above the k=2/k=6 boundary, for example:

```text
SIGMA_MIN = 0.45
```

Effect:

```text
rejects k=2 and k=6
keeps k=3, k=4, k=5
```

Recent impact estimate over windows 6428-6458:

```text
k=6 rows: 93/248
k=2 rows: 14/248
rows newly rejected: about 43%
target newly rejected: about 62% of its selected rows
```

This is a strong immediate lever, but it may reduce batch fill and miners will
adapt toward k=5 or k=4 selection.

Files:

```text
reliquary/constants.py
tests/unit/test_constants.py
tests/unit/test_zone_filter.py
tests/unit/test_grpo_window_batcher.py
```

### Option C - Explicit correct-count band

Instead of relying on floating sigma, add a binary reward distribution rule:

```text
accept only 3 <= correct_count <= 5 for M=8 binary reward groups
```

This is clearer than `SIGMA_MIN=0.45` and exactly encodes the desired frontier
band.

Patch point:

```text
reliquary/validator/batcher.py
```

right after reward verification and before GRAIL:

```python
rewards = [float(r.reward) for r in request.rollouts]
correct = sum(1 for r in rewards if r >= 0.5)
if correct < 3 or correct > 5:
    return reject(RejectReason.OUT_OF_ZONE, "zone")
```

This can reuse `OUT_OF_ZONE` or introduce a new reason such as
`reward_distribution_suspicious`.

Risk:

- only safe for binary reward environments;
- future non-binary environments need a different rule;
- miners can still post-select k=5.

### Option D - Improve reward normalization

The OpenMathInstruct reward parser currently uses the last boxed answer and
light normalization:

```text
reliquary/environment/openmathinstruct.py
```

Improvements worth testing:

```text
normalize degree symbols: 90^\circ -> 90
normalize simple unit suffixes where the ground truth is unitless
handle common LaTeX wrappers more robustly
```

Do not blindly accept "any boxed answer matches ground truth" without more
thought. Because ground truth is public to miners today, accepting any matching
box lets a miner stuff many boxes and win.

This patch improves label quality but does not solve reward-oracle selection.

### Option E - Protocol redesign

The robust fix is to remove the public reward oracle from the mining loop.
Possible directions:

```text
validator-private prompt/reward set
server provides prompt text without ground_truth
validator samples or challenges which candidate indices count
commit-to-large-pool then validator randomly selects 8 after commit
```

The commit-to-large-pool design is the most compatible with current GRAIL:

1. miner commits hashes/proofs for N candidates before knowing the selected
   subset;
2. validator derives a post-commit seed;
3. validator selects 8 indices for reward/training;
4. miner reveals/proves those indices.

This is a larger protocol change and should not be rushed inside the incident
patch cycle.

## Recommended Plan

1. Add monitoring now:
   - reward vector monotonicity by hotkey;
   - k-distribution by hotkey;
   - parser-shaped zero-reward examples;
   - reward mean and sigma trend per hotkey.

2. Announce internally that `5Dd2...` is not a GRAIL bypass but likely
   reward-oracle selection.

3. Decide policy:
   - if candidate-pool reward selection is allowed, no validator patch is
     needed beyond monitoring and reward-parser improvements;
   - if it is not allowed, patch the zone rule.

4. If patching today, prefer Option C. A patch has been prepared that adds a
   `reward_distribution` rejection for steady-state binary groups outside
   k=3..5.

5. Separately test reward normalization for degree/unit variants, but do not
   mix that with the zone-rule patch unless the team is ready for a wider
   reward-behavior change.

## Current Recommendation

Recommended immediate action:

```text
deploy the k=3..5 binary reward-distribution guard, monitor whether miners
adapt toward k=5, and be ready to tighten to k=4-only if training quality or
slot capture still degrades.
```

Reason:

```text
The target behavior is real and economically dominant. It is not a direct
validator bypass, so the clean fix is to close the exploited edge of the
training objective rather than blacklist a hotkey. A monotonic-order rejection
is easy to bypass by shuffling rollouts and would not address candidate
selection.
```
