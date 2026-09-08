# Reliquary verifiable environment canary

This runbook qualifies `reliquaryverifiable_v1` and an isolated Qwen3-4B
candidate. It does not authorize a production switch or a public performance
claim.

## 1. Freeze the run before starting

Record and review:

- run ID and owner;
- exact base-model and tokenizer revisions;
- `qwen3-4b-reliquary-verifiable-v6-dev1` generation-contract digest;
- registry and authored environment-manifest digests;
- golden-fixture digest;
- container/image digest and committed software revision;
- separate HF repository, R2 prefix, checkpoint namespace, and
  `TRAINING_RUN_ID` from the Math+Code lineage;
- validator/testnet target, GPU class, proof devices, window range, update and
  token budget;
- optimizer configuration and stopping rules;
- held-out environment split, Math/Code retention set, evaluator owner, and
  promotion authority;
- preregistered success, retention, KL, ratio, ESS, clipping, and rollback
  thresholds.

Do not use public benchmark prompts or evaluation answers in task generation,
reward, training data, checkpoint selection, or difficulty tuning.

## 2. Offline oracle gate

From a clean committed checkout:

```bash
export RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-reliquary-verifiable-v6-dev1
export RELIQUARY_ENVIRONMENTS=reliquaryverifiable_v1

python scripts/qualify_environment_suite.py \
  --samples 10000 \
  --output evidence/environment-qualification.json
```

Require `passed: true` with no dirty override. Review operation counts rather
than only the aggregate pass bit. The report proves deterministic local task
generation, manifest/fixture binding, reward goldens, malformed-input failure,
and the binary 16-rollout sigma frontier. It does not measure model ability.

Next, sample at least 256 prompts per difficulty band with the exact pinned
baseline and decoding contract. Report valid-JSON, exact success, token length,
cap-hit, and all-zero/mixed/all-one group rates by operation. Target a useful
core band of 25–75% exact success. Quarantine operations below 5% and raise
difficulty above 90% rather than claiming a useful RL frontier without mixed
groups.

## 3. Proof and admission capacity gate

On every intended GPU and release image:

1. Generate representative 16-rollout proof samples using the profile's
   1,024-token cap distribution.
2. Run `scripts/qualify_proof_capacity.py` against every environment declared
   by the selected profile.
3. Require at least 20 passing representative proofs per GPU and the existing
   capacity headroom rule.
4. Measure admission/reward and proof p50/p95/p99 under realistic concurrent
   load.
5. Require queue plus proof p95 to fit the collection window with at least 20%
   headroom.

The Records environment must use the CPU admission queue and must not start the
Code grader. `/health` must report its queue, inflight count, worker count,
batch target, prompt-source health, and proof-capacity state under its real
environment name.

## 4. Private/testnet shadow, training disabled

Start coordinated miners and validators only with both explicit selectors:

```bash
export RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-reliquary-verifiable-v6-dev1
export RELIQUARY_ENVIRONMENTS=reliquaryverifiable_v1
```

Run at least ten clean windows. Stop immediately on any:

- profile, model, tokenizer, environment, generator, prompt, target, or reward
  mismatch;
- nondeterministic replay;
- admission worker crash/timeout counted as a model reward;
- proof-capacity abort or archive/payload gap;
- queue or proof headroom breach;
- change or starvation in an existing Math/Code service.

For every window reconcile submitted, admitted, selected, rewarded, archived,
payload, and trained counts. Compare operation distributions at eligible,
submitted, admitted, selected, and trained stages so free prompt selection
cannot silently collapse the curriculum onto the easiest operation.

## 5. One-update detached-trainer canary

1. Clone the exact base checkpoint into the isolated candidate lineage.
2. Collect one accepted update with publication and promotion disabled.
3. Verify that the schema-v2 training payload environment order and target map
   exactly match the trainer configuration.
4. Train once and save a candidate checkpoint.
5. Reload the candidate in the release image.
6. Evaluate held-out Records exact success/JSON validity, operation and
   difficulty slices, completion length/cap hits, Math/Code retention, and the
   preregistered KL/ratio/ESS/clipping gates.
7. Reset to the exact base and repeat across the frozen seed/batch choices.
8. If healthy, run at most four serial updates, evaluating each candidate.

Non-finite loss/gradient/ratio, reward/proof reconciliation failure, retention
breach, excessive clipping, or a preregistered KL/ESS breach is a hard stop.

## 6. Evidence and promotion

Build a `reliquary/training-run-manifest/v1` document containing:

- base checkpoint and tokenizer;
- protocol profile and generation-contract digest;
- environment, registry/authored manifest, and golden digests;
- software and image digests;
- run ID and exact first/last windows;
- every selected-and-trained payload root;
- optimizer settings;
- candidate checkpoint and status (`candidate`, `promoted`, or `rejected`).

Use `reliquary.environment.evidence.sign_training_run_manifest` with the
validator wallet. Store the signed envelope beside the evaluation report.
Verification is domain-separated and fails when signer, payload, digest, or
domain changes.

Promotion is a separate operator action after review. Training, upload, and a
passing internal environment score do not automatically change the active
validator checkpoint.

## 7. Rollback

1. Disable training and stop detached checkpoint intake.
2. Leave the last-known-good promoted revision active.
3. Tombstone/quarantine the failed window and preserve all evidence.
4. Stop the v6 environment/profile and return coordinated processes to their
   prior immutable profile.
5. Never load a v6 candidate under v5, reuse a run/checkpoint identifier, or
   accept a manifest from another run.
6. Require three clean legacy windows plus archive, queue, proof, and checkpoint
   health before declaring recovery complete.

## 8. Claim language

Safe early language states only what the evidence demonstrates: registered
miners submitted rollouts; validator-authoritative Reliquary logic verified and
scored them; preregistered selection produced the training subset; and the
detached trainer applied RL to the named candidate lineage.

Do not say “RL improved the model” until the candidate beats the exact
base/format baseline within preregistered retention margins. Do not say
“Reliquary selection caused the improvement” without a token- and
family-matched uniform-selection control. Do not say “generalizes” without
held-out generator/schema families and external transfer. Do not say “SOTA”
without the exact benchmark owner/category accepting that comparison.
