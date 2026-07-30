# Auction v3 Utility Foundation

Status: observation only. This foundation does not change auction-v2 ranking,
selection, rewards, miner payloads, or training loss.

## Decision

Auction v2 remains the production mechanism:

1. Rank by validator-computed difficulty.
2. Rank equal scores by validator-observed precommit drand round.
3. Resolve the remaining tie with post-deadline drand.
4. Prove candidates top-down until at most eight distinct groups win.
5. Select and reward exactly those winners, one uniform slot each. Clean
   checkpoint-consistent groups may enter the balanced training accumulator;
   quarantine can archive and credit a winner without training it.

This is the correct production design while `k=2` groups share the same
difficulty plateau. Rank-weighted payout would amplify timing rather than
training value. Auction v3 should first produce a validator-authoritative
utility ordering inside that plateau, then reconsider payout shape separately.

## Canonical Content Identity

Prompt indices are dataset coordinates, not semantic identities. Two indices
can render to the same task. Every admitted candidate now carries:

```text
prompt_content_sha256 = SHA256(
  "reliquary/prompt-content/v1\0"
  || environment || "\0"
  || exact validator-rendered prompt UTF-8
)
```

The validator uses the full 256-bit digest. Auction winners must have distinct
prompt content, and winning content enters the same one-shot cooldown as its
index. An alias at another dataset index cannot receive a second slot or bypass
cooldown.

The content map is run-keyed and persisted to both the validator state volume
and R2. On first deployment it is derived from the complete existing
prompt-index snapshot using the pinned environment and tokenizer. Startup does
not open a window until derivation succeeds and a restart-safe local snapshot
exists. An R2 outage is nonblocking after local persistence.

`target_content_sha256` is retained only in the private utility bundle for
offline diagnostics. It is excluded from public archives and never
participates in admission, rank, or cooldown.

## Private Utility Dataset

For each completed window, the validator writes a local gzip bundle at:

```text
${RELIQUARY_STATE_DIR:-/root/reliquary/state}/utility_telemetry/window-N.json.gz
```

Each bundle contains the selected winners plus a bounded hybrid forensic sample
that passed the same expensive proof. With the default budget of two, one is
the next-ranked, content-unique counterfactual to position 9 and one is chosen
unpredictably from the remainder using post-deadline drand. This preserves the
anti-evasion monitor while gathering the closest utility counterfactual. Both
remain unpaid and untrained.

Per group, the bundle records:

- Environment, checkpoint, prompt/target content identities, and selection
  digest.
- Exact token IDs and validator-authoritative reward vector.
- Completion length, natural EOS, validated BFT force span, termination path,
  and token-degeneracy diagnostics.
- Chosen-token NLL summaries from the existing proof forward.
- Exact full-vocabulary policy entropy at up to 64 deterministic, evenly spaced
  completion positions before top-k/top-p at `T_PROTO`.
- Float16 hidden anchor at the last prompt token and hidden delta to the final
  trainable non-EOS completion token. Injected BFT close tokens are excluded.
- Group-level outcome and representation-shift summaries.

There is no extra model forward. Operator identifiers are HMAC-pseudonymized
with a local mode-0600 key. Bundle files are atomic and mode 0600, are not sent
to the public archive, and default to a 2,048-window retention horizon. Invalid
numeric data or any write error drops only the private bundle; it cannot reject
a miner, alter rewards, skip training, or abort a window.

Configuration:

```bash
RELIQUARY_UTILITY_TELEMETRY_ENABLED=true
RELIQUARY_UTILITY_TELEMETRY_RETENTION_WINDOWS=2048
```

`/health.utility_telemetry` reports writes, failures, last window, last error,
schema version, and retention. `/health.content_cooldown` reports canonical map
completeness and persistence state.

Summarize local coverage without exposing raw vectors or tokens:

```bash
python scripts/report_utility_telemetry.py --json
```

The report can declare data ready for causal labeling, but deliberately never
authorizes protocol activation.

## Utility Target

Novelty, confidence, termination quality, and representation shift are useful
features, but none is training utility by itself. The causal target is
checkpoint-relative improvement on a fixed, validator-controlled probe set.

For a candidate group `g`, the strongest offline labels are:

```text
U_step(g) = L_probe(theta) - L_probe(theta - eta * grad(L_g))

U_align(g) = dot(grad(L_g), grad(L_probe))
             / (norm(grad(L_g)) * norm(grad(L_probe)) + epsilon)
```

`U_step` is the measured virtual-microstep improvement. `U_align` is its cheaper
first-order influence approximation. Both must use the exact checkpoint,
training mask, BFT carve, reward/advantage calculation, KL reference, and loss
contract used in production.

The private telemetry supports cheaper proxy models, but activation requires
those proxies to predict `U_step` or `U_align` out of sample. Hidden distance or
high entropy alone must never become an economic oracle.

## Research Protocol

Run Math and Code independently. Do not pool their calibration.

1. Build labels on the dedicated research GPU from immutable private bundles.
2. Split chronologically by checkpoint revision. Keep every canonical content
   digest and operator pseudonym in one split to prevent leakage.
3. Establish baselines: current difficulty-only ordering, random ordering
   inside the exact-score plateau, and arrival ordering.
4. Fit a transparent regularized model first. Inputs may include NLL, entropy,
   natural termination, completion cost, degeneracy, content novelty, hidden
   shift, and reward-conditioned hidden shift.
5. Evaluate top-eight utility uplift, rank correlation, held-out probe-loss
   improvement, content diversity, operator concentration, and compute cost.
6. Replay complete historical populations, including failed proofs and unfilled
   slots. Never evaluate only historical winners.
7. Shadow the proposed order on live windows without changing proof order,
   rewards, cooldown, or training.

Minimum evidence before a utility tie-break can activate:

- At least 256 complete windows and three checkpoint revisions per environment.
- Private writer success at least 99.9%, no malformed bundles, and missing
  utility fields below 1% on proven groups.
- Proof-path p95 overhead below 5% and no admission, seal, or checkpoint
  regression.
- Positive top-eight `U_step` uplift over all baselines with a block-bootstrap
  95% confidence interval above zero.
- Stable effect direction across held-out checkpoints and separately for Math
  and Code.
- No material increase in duplicate content, one-operator concentration,
  abnormal termination, or training-quarantine rate.

The RTX PRO 6000 Blackwell microbenchmark on 2026-07-22 used 1,024 positions and
the production 248,320-token vocabulary. Stratified entropy added about 0.54 ms
median to the existing statistics pass and 31 MiB peak allocation. The model
forward benchmark could not be used because the older staging FLA/Triton image
does not compile its kernel for Blackwell; production activation still requires
the end-to-end p95 gate above on a supported runtime.

## Proposed Auction v3 Order

The first activation should be deliberately narrow:

```text
difficulty descending
calibrated utility bucket descending, only inside an exact-score plateau
validator-observed precommit drand round ascending
post-deadline sealed tie-break
```

Utility must be checkpoint- and environment-normalized, clipped, versioned, and
computed only from validator-derived inputs. The online score should initially
reorder equal difficulty values only. It must not let a lower-difficulty group
overtake a higher-difficulty group.

Keep trained-only uniform payment for the eight selected winners. Consider a
rank-weighted payout only after utility ranking has remained causally useful and
economically stable through a separate replay and shadow proposal.

## Rollout And Rollback

The future utility tie-break requires its own disabled-by-default flag, score
version in `/health` and archives, deterministic replay fixture, and immutable
image. It can activate at a window boundary without a miner wire update.

Rollback immediately to auction v2 ordering for reward/training divergence,
checkpoint interruption, proof-budget regression, unexplained concentration,
or loss of deterministic replay. Canonical content identity and private
observation remain useful and should not be rolled back with an experimental
ranker.
