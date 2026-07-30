# Qwen3.5-4B Auction-v3 Production Runbook

**Date:** 2026-07-31
**Status:** final design and activation contract for PR #162

## Decision

Merge the branch with auction-v2 remaining the default. Do not activate the
4B profile until the exact release image passes an end-to-end proof-capacity
qualification on the intended H100 fleet and a stamped append-only 4B base
checkpoint has been published.

The final v3 design is:

- Pinned `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Protocol version 3 and profile ID `qwen35-4b-auction-v3`.
- Eight fixed forced-seed rollouts with the existing sampling contract.
- Math: 16,384-token protocol ceiling, BFT at 15,616 thinking tokens,
  forced answer span, and 512 answer tokens.
- Code: 32,768-token ceiling and natural EOS; no BFT.
- A 300-second collection interval plus the existing 33-second reveal grace.
- Exact conservative truncation utility over the validator-known reward
  lattice and the sigma gate.
- Difficulty first, validator-observed arrival round second, and sealed drand
  lottery third.
- At most eight distinct, proven, uniformly paid winners per environment.
  `rewarded` must equal `selected_for_batch`.
- Admission retains 64 started grading jobs per environment; only the ranked
  seal-time GRAIL prefix is capped at 16, followed by two independent forensic
  proofs. Forged commitments cannot reduce admission to a two-hotkey ceiling.
- Math BFT-forced rows stay in economic scoring and the group baseline but are
  masked from PPO and KL loss. Code truncations remain in the loss until an
  adversarial ablation justifies masking them.
- Learning-rate default `1e-6` and KL beta default `0.01`.

The following proposal components were rejected:

- **Throughput ranking:** submitted length is miner-controlled before proof,
  creates padding and serving-hardware incentives, and does not measure
  training utility.
- **Speculative early close:** proving before the population and post-deadline
  entropy are frozen can spend the proof wall on candidates that do not win.
- **A larger proof wall as the primary fix:** it hides insufficient capacity
  by making windows much longer.
- **A checkpoint-stall `k=6` payout mode:** `k=6` already passes the sigma gate
  and naturally fills unclaimed slots. A special mode would reward easier
  prompts whenever any unrelated publication or infrastructure fault stalls
  checkpoint progress.

## Compatibility Boundary

| Profile | Model | Protocol | Collection | Math | Code |
|---|---|---:|---:|---|---|
| `qwen35-2b-auction-v2` | Qwen3.5-2B | 2 | 100 s | 32,768 cap; BFT 2,048 + 512 | 32,768 cap; natural EOS |
| `qwen35-4b-auction-v3` | Qwen3.5-4B | 3 | 300 s | 16,384 cap; BFT 15,616 + 512 | 32,768 cap; natural EOS |

Merging does not change production because v2 remains the default. Activating
v3 is a hard coordinated miner/validator cutover. V3 requests bind the profile
ID and generation contract in signatures and precommit receipts. A v2 miner
cannot submit to a v3 validator successfully.

## Proof Plane

Auction-v3 uses one global scheduler with one long-lived worker and one frozen
model replica per configured CUDA device. Math and Code share the fleet fairly.
Proof completions may finish out of order, but economic decisions are applied
in strict rank order. Candidates sharing an operator or hotkey proof-debt
identity are serialized until the earlier result is applied.

Winner plans have priority over observational forensics. The scheduler starts
only enough concurrent work to fill the remaining winner count. Sparse honest
populations and ranked prefixes exhausted by invalid miner proofs complete with
empty slots burned. Infrastructure deadline, device, or checkpoint-lifecycle
failures abort the entire window before rewards, training, or checkpoint
advancement; capacity-truncated winner sets are never trained.
Verifier exceptions and active CUDA calls that cross the proof deadline fault
the proof plane, archive an aborted window, and terminate the validator for a
supervisor restart. They are infrastructure failures and never accrue miner
proof debt or promote a lower-ranked candidate.

Checkpoint publication quiesces the scheduler, refreshes every frozen replica,
marks all devices ready for the new revision, and only then opens the next
window. A transient replica refresh failure is retried at the next quiescent
pre-window boundary. No window opens while a replica is stale or the scheduler
is faulted.

## Capacity Qualification

Capacity is a release property, not a guessed environment variable. V3 refuses
to start without a SHA-256-pinned manifest matching all of:

- Protocol profile and base-model revision.
- Exact full 40-character validator software revision read from the image-baked
  `/opt/reliquary/.build-revision` file.
- Benchmark checkpoint revision and SHA-256 of the raw evidence file.
- Exact runtime fingerprint hash, including Python, Torch, CUDA, attention
  implementation, and installed Qwen3.5 kernel paths.
- GPU hardware class, physical GPU UUID set, and exact fleet size.
- Configured proof-wall duration.
- All 16 possible ranked proof attempts plus two forensic proofs for both Math
  and Code.
- Measured p95 end-to-end proof latency and 20% headroom.
- At least 20 successful worst-case group proofs per GPU per environment, with
  all eight completion lengths at or above 90% of that environment's protocol
  cap.
- Per-device p95 measurements, with fleet sizing based on the slowest GPU p95
  in each environment.

For p95 group-proof latencies `M` and `C`, the minimum homogeneous device count
under the current 240-second wall is:

```text
ceil(18 * (M + C) / (240 * 0.8))
```

Planning examples, not qualification evidence:

| Math p95 | Code p95 | Minimum H100s |
|---:|---:|---:|
| 30 s | 25 s | 6 |
| 45 s | 35 s | 8 |
| 60 s | 45 s | 10 |

One device qualifies only if `M + C <= 10.67s`, which is not a credible launch
assumption for the 4B long-context workload. Plan for roughly eight to ten H100s
until the exact release benchmark proves otherwise. The training H100 may also
be a proof device because training starts only after proof completion, provided
the measured fleet and memory layout include that configuration.

An RTX PRO 6000 Blackwell is useful for functional profile, generation,
termination, and proof-parity tests. It must not be used to certify an H100
capacity manifest.

Build a manifest only from real end-to-end validator group proofs produced by
the exact release candidate:

```bash
RELIQUARY_PROTOCOL_PROFILE=qwen35-4b-auction-v3 \
python scripts/qualify_proof_capacity.py staging-proofs.jsonl \
  --output proof-capacity.json \
  --software-revision <full-release-commit> \
  --checkpoint-revision <full-benchmark-checkpoint-commit> \
  --runtime-fingerprint-hash <health-runtime-profile-hash> \
  --hardware-class "NVIDIA H100 80GB HBM3" \
  --benchmark-device-count <count> \
  --measured-at 2026-07-31T00:00:00Z
```

Each JSONL row has the form:

```json
{"environment":"openmathinstruct","seconds":61.2,"proof_passed":true,"profile_id":"qwen35-4b-auction-v3","model_revision":"851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a","software_revision":"0123456789abcdef0123456789abcdef01234567","checkpoint_revision":"89abcdef0123456789abcdef0123456789abcdef","runtime_fingerprint_hash":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","hardware_class":"NVIDIA H100 80GB HBM3","device_uuid":"GPU-EXAMPLE","rollout_count":8,"completion_token_lengths":[14746,14746,14746,14746,14746,14746,14746,14746]}
```

Math samples require every completion to contain at least 14,746 tokens; Code
requires at least 29,492. Use at least 20 samples on every manifest GPU in each
environment. Synthetic single-sequence forwards,
failed proofs, partial groups, extrapolation from 2B windows, and evidence from
a different release SHA are rejected.

## Checkpoint Lineage

Every newly published checkpoint contains
`reliquary_protocol_profile.json`, binding it to the active profile, protocol,
base model, and immutable base revision. V3 refuses an unstamped historical
checkpoint or any mismatch before loading weights.

Publish the 4B base as the next append-only checkpoint in the existing HF repo:

```bash
RELIQUARY_PROTOCOL_PROFILE=qwen35-4b-auction-v3 \
RELIQUARY_HF_REPO_ID=<repo> \
HF_TOKEN=<token> \
python scripts/publish_base_reset_checkpoint.py
```

With no explicit source, the script automatically uses the pinned 4B base
revision and writes both lineage and recovery manifests. Install the printed
`RELIQUARY_RESUME_FROM=sha:<commit>` value.

Do not delete history, reuse an old checkpoint number, clear the HF repo, or
start v3 from the latest 2B checkpoint.

## Activation

Build one immutable image from the final PR commit. Configure:

```bash
RELIQUARY_PROTOCOL_PROFILE=qwen35-4b-auction-v3
RELIQUARY_CHECKPOINT=Qwen/Qwen3.5-4B
RELIQUARY_RESUME_FROM=sha:<stamped-4b-base-reset>
RELIQUARY_PROOF_DEVICES=<exact-canonical-device-list-qualified-in-manifest>
RELIQUARY_PROOF_CAPACITY_MANIFEST=/root/reliquary/state/proof-capacity.json
RELIQUARY_PROOF_CAPACITY_MANIFEST_SHA256=<sha256>
RELIQUARY_TRAINING_RUN_ID=<new-4b-run-id>
RELIQUARY_KL_BASE_MODEL=Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
RELIQUARY_KL_BETA=0.01
RELIQUARY_LEARNING_RATE=0.000001
RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY=true
RELIQUARY_SHAPE_PENALTY=0
```

Use only explicit `cuda:<index>` identifiers. The runtime physical UUID set and
device count must exactly match the accepted capacity manifest; aliases such as
bare `cuda` and a merely larger unbenchmarked fleet fail closed. Under v2 this
environment variable is ignored so a stale deployment value cannot activate
the scheduler accidentally. Keep drand backward tolerance at zero.

V3 rejects any other `RELIQUARY_CHECKPOINT`, a local-path resume, or a capacity
manifest whose checkpoint SHA differs from `RELIQUARY_RESUME_FROM`. The stamped
checkpoint lineage is validated again after loading, before replicas can run.

Before opening to miners, require:

1. `/state` advertises protocol 3, the v3 profile and contract, and the stamped
   4B checkpoint revision.
2. `/health.proof_scheduler.state == "running"`.
3. Capacity qualification schema 3 reports `qualified=true`, 18 proofs per
   environment, worst-device p95, and exact build/checkpoint/runtime/UUID
   matches.
4. Every proof device reports the active checkpoint revision.
5. A loaded staging window completes both environments with eight winners,
   two forensic samples, no partial payout, no capacity abort, and no pending
   seal side effects.
6. Event-loop, `/health`, precommit, admission, archive, and grader health stay
   within their existing production limits.
7. Current reference miners pass forced-seed, profile-signature, Math BFT, Code
   EOS, precommit, and final-verdict parity.

Inspect the first ten live windows. Stop for any capacity abort, reward/training
divergence, checkpoint-lineage mismatch, repeated service outage, or archive
regression.

## Rollback

Before activation, rollback is the previous v2 image and configuration.

After a v3 checkpoint has been appended, the latest HF checkpoint belongs to
the v3 lineage. A v2 validator must not resume it or pin an older checkpoint
number that auto-discovery will supersede. Publish the pinned 2B base as a new
append-only checkpoint under the v2 profile, then deploy the previous v2 image
and coordinate miners back to protocol 2.

## `k=6` Research Direction

Exact binary `k=6` has nonzero GRPO contrast, passes the steady sigma gate, and
can already be selected when higher-utility groups do not fill all eight slots.
Therefore no new live payout rule is needed.

If late-run learning stalls after infrastructure, publication, accumulator,
quarantine, and proof-capacity causes are excluded, run an offline curriculum
ablation:

1. Replay checkpoint-bound selected groups by `k`.
2. Compare the existing auction mixture with a capped 10-20% `k=4..6`
   refinement mixture.
3. Measure held-out Math and Code utility, natural termination, KL drift,
   gradient norms, policy-ratio tails, and regression on hard `k=1..3` prompts.
4. Promote the curriculum only if it adds held-out capability without moving
   emissions or weakening the live auction.

Checkpoint non-movement by itself is never a valid trigger.
