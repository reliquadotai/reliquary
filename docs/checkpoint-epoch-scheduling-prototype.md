# Checkpoint-epoch scheduling prototype

Status: experimental and disabled by default. The implementation is intended
to be mergeable as an inactive capability, but it is not an activation-ready
production protocol. This prototype changes no current production profile or
canonical contract. A future coordinated protocol/profile revision is required
before activation.

## Goals and non-goals

The prototype publishes the complete generation plan for one checkpoint
horizon, then opens all 16 logical lanes in one checkpoint-wide collection
phase. Selected usable groups form a frozen reservoir after the common
deadline.

The horizon comes from `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS`. Experimental
activation fails closed unless that configured value is exactly 16.

Work remains miner-directed: the validator publishes seeds and eligible prompt
slices but never assigns prompts or generation jobs. The prototype does not
deploy a profile, publish a checkpoint, change production rewards, accept work
before OPEN, or add a production submission transport to the reference planner.

## Checkpoint-bound state machine

1. The immutable checkpoint revision is installed and its proof replicas are
   coherent.
2. The validator observes drand round `R0`, durably fixes and signs the intent
   for `R1 = R0 + 1`, then confirms from a second round observation that those
   durable bytes preceded `R1`. It exposes the canonical signed bytes while
   `R1` is still unavailable.
3. The validator fetches and verifies exactly `R1`; a different round is never
   substituted.
4. That beacon derives the epoch root, all 16 generation seeds, and every
   environment's 16 prompt slices.
5. The canonical plan is exposed with an advertised
   `activation_not_before_round`, giving time for checkpoint download and model
   warm-up.
6. At activation, every `(environment, logical window)` lane is installed in
   one routing swap. Every batcher receives the same exact monotonic and wall
   clock OPEN timestamp and therefore the same commitment deadline.
7. Miners retain generated payloads locally and send only their signed compact
   commitments during the common commitment phase.
8. After that phase closes, the validator drains the bounded compact requests
   already at ingress, freezes their complete canonical set, signs it, persists
   it, and exposes those exact bytes. Only then does it target fresh public
   beacon A. The signed-set hash is an input to arrival-neutral selection,
   which visits canonical operator identities in deterministic rounds.
9. Only selected commitments may upload their exact committed payload during
   the manifest-bound reveal interval. The validator then freezes and drains
   those reveals.
10. Fresh public beacon B is obtained only after the reveal interval. It is
    distinct from beacon A and supplies final equal-value ordering.
11. Lanes seal in deterministic offset order into the usable-group reservoir,
    which the manifest-bound training mode then consumes.

`activation_not_before_round` is a lower bound, not a prediction of local
availability. The common OPEN edge occurs once the validator is ready at or
after that round.

Generation preparation begins from the public manifest. Submission does not:
all 16 lanes become OPEN in the same atomic routing change, and none is a
future live window after that point.

## Canonical manifest and derivation

The strict canonical JSON manifest and its SHA-256 bind:

- schema/domain and experimental capability versions;
- protocol profile/version and canonical generation-contract hash;
- checkpoint number, repository, immutable revision, and observed commit round;
- drand source, chain, chain hash, exact round, and verified randomness;
- first logical window and exact configured horizon;
- common activation/warm-up and collection/timeout policy;
- target groups and maximum selected reveals per environment/lane;
- compact-commitment operator bound, arrival-neutral selection policy, and
  reveal duration;
- `sequential_steps` or `aggregate_one_step` training mode;
- every environment and dataset-universe size;
- prompt-range width and explicit overlap policy;
- every lane offset, logical window number, generation seed, and prompt slice.

Conceptually:

```text
epoch_id   = SHA256(domain/id || canonical_pre_beacon_intent)
epoch_root = SHA256(domain/root || epoch_id || verified_beacon_R1)

seed[i] = SHA256(
    domain/window || epoch_root || uint64(i) || uint64(window_number[i])
)

slice[i, environment] = DERIVE_SLICE(
    domain/slice,
    epoch_root,
    environment,
    dataset_universe,
    prompt_range_width,
    i,
)
```

Direct indexed derivation avoids mutable generator state and permits independent
validation of any lane. Changing the checkpoint, contract, beacon, schedule,
training mode, horizon, offset, logical window, environment, or dataset binding
changes the appropriate identifier, seed, or manifest hash.

The validator persists a create-only signed pre-beacon intent and then a
create-only manifest. The narrow `/checkpoint-epoch/intent` endpoint exposes
the signed intent while the target beacon is still pending. A miner verifies
the validator signature, canonical bytes, ETag, exact plan derivation, and the
independent drand signature before adopting a plan. Restart recovery reloads
the same canonical bytes.

This gives observers a consistency surface but not a consensus timestamp for
the HTTP publication. The locally fsynced second-round observation is still a
validator assertion. A production revision must define independent observation
or another trustless checkpoint-commit timestamp; this prototype does not
claim stronger ordering than the infrastructure can prove.

## Concurrent commitment, selection, and reveal

The 16 entries are logical lanes inside one physical OPEN phase; they do not
open one after another. A miner queries
`/state?env=<name>&window=<number>` for each lane it intends to release.

Preparing work does not send its payload. At exact commitment OPEN, the miner
revalidates live state and finalizes the nonce, envelope signature, precommit
signature, and observed drand-round telemetry. It durably keeps those exact
reveal bytes, then sends only the compact precommit that binds their hash,
size, prompt, lane, checkpoint, profile, and generation inputs. The validator's
receipt timestamp and fixed commitment deadline determine eligibility. An
older signed round is recorded but does not reject or rank an epoch commitment;
a claimed future round remains invalid. The validator stores the compact object
without reserving proof or grading capacity. A payload sent before selection is
refused.

After beacon A, miners poll the narrow commitment-status endpoint. A selected
commitment receives one bounded reveal deadline; a non-selected commitment
cannot upload. Immediately before reveal, the miner rechecks exact OPEN and
reveal phase plus checkpoint, epoch ID, manifest hash, profile, contract,
logical window, generation seed, prompt slice, and cooldown. Replaced, stale,
or ambiguous work is quarantined and never replayed under another binding. The
existing formatting, authenticity, grading, proof, duplicate, checkpoint, and
forced-stream checks remain in force.

Every lane targets `B_BATCH` selected groups. The experimental default chooses
at most `2 * B_BATCH` payload reveals per environment/lane: 16 targets plus a
16-candidate proof-failure reserve with the production value of `B_BATCH`.
Deterministic sensitivity replay keeps the reserve explicit: the former
half-batch reserve underfilled too often at the fixture's stated validity rates,
while a full-batch reserve filled 99.4% of 500 complete synthetic epochs and
still halves the old 64-payload ceiling. This is a capacity result, not live
telemetry. The limit is applied only after the commitment phase and therefore
does not create a first-arrival pool. The reveal bound and compact-commitment
bound are manifest-bound and cannot change within an epoch.

The common commitment phase defaults to the existing per-window duration
multiplied by the checkpoint horizon. It never closes early. State advertises
the exact phase, durations, target, and selected reveal limit. Compact
commitments are bounded per canonical operator and lane. This controls validator
work without claiming that uncoordinated local preparation has zero discarded
work. Selection also grants at most one reveal ticket per operator and prompt
inside a lane. The production auction's proven-dominance early-close policy is
explicitly disabled for experimental epoch batchers; the manifest-bound common
deadline remains authoritative in every early-close mode.

The local reference planner is bounded and has no submission transport. A
backend-neutral callback may prepare token sequences, after which the existing
request builder performs canonical rewards, proofs, log probabilities, and
formatting. The miner transport helper finalizes signatures only at exact OPEN;
the planner itself remains incapable of network sends by default.

Its deterministic baseline walks lane, prompt, then environment order. The
optional `value_per_gpu_second` policy orders the same bounded records by an
operator-supplied estimate of eligible training value divided by estimated
generation seconds, with the baseline tuple as a stable tie-break. These local
estimates affect only preparation order; they are never sent to the validator
or used in ranking or rewards.

## Prompt slices and cooldown

Each environment receives one slice per lane. Slices are non-overlapping when
the universe permits it. Otherwise the manifest names one deterministic cycle
policy; it never widens the universe or weakens cooldown.

Because lanes collect together, admission uses each lane's opening snapshot.
Final selection then rechecks the shared live prompt and content cooldown while
sealing lanes in ascending offset order. In an overlap cycle, an earlier
selected lane deterministically makes the same prompt ineligible for later
lanes. Miner release also rechecks prompt content and the advertised cooldown.

Revealing the full horizon creates cross-lane prompt-selection bias: miners can
compare all public slices before choosing work. This is a conscious product
choice requiring diversity, difficulty, and selection-distribution metrics,
not a free efficiency claim.

## Training reservoir modes

The plan binds the value of
`RELIQUARY_EXPERIMENTAL_CHECKPOINT_EPOCH_TRAINING_MODE`:

- `sequential_steps` consumes lane batches in deterministic offset order. The
  existing balanced accumulator can carry a sparse environment into the next
  lane. Each optimizer call uses the same frozen behavior checkpoint.
- `aggregate_one_step` retains up to the complete 16-lane target per
  environment and passes the usable reservoir to one `train_step` call. That
  function already uses token-budgeted microbatches and gradient accumulation,
  so this is one optimizer update without one giant in-memory forward. An
  underfilled reservoir is consumed only when every configured environment is
  represented.

These modes are not mathematically equivalent. Sequential mode updates model
parameters between calls; aggregate mode computes one gradient against one
parameter state. Both keep the published behavior checkpoint frozen for the
whole epoch and publish at most one successor checkpoint after completion.

The detached journal carries the same immutable epoch/lane binding plus the
training-run identity. It writes a small terminal marker only after all 16
payloads or tombstones have uploaded.
The detached trainer waits for that marker before consuming lane zero: an
aborted epoch is skipped as one unit, while a completed epoch is processed in
the manifest-selected sequential or aggregate mode. In aggregate mode the one
successful optimizer call credits the full horizon for publication cadence.

## Ranking, seal randomness, and rewards

The advance plan contains generation randomness only. It never contains or
derives seal, auction, or final tie-break randomness.

Beacon A is obtained after compact commitments are frozen and controls only
which bounded cohort may reveal. Beacon B is obtained after reveals are frozen
and orders candidates that are equal on validator-authoritative
utility/difficulty. Neither selection uses generation completion, throughput,
upload, or arrival time. The production ranking path is unchanged when the
experiment is disabled.

The prototype retains the current selected-slot reward model and burn
accounting. It does not introduce payment for unselected work.

## Identity, quotas, persistence, and invalidation

The commitment bound and round-based reveal selection reuse the repository's
canonical operator mapping; they do not invent an identity from hotkeys. This
prototype does not claim that the mapping makes participation Sybil-resistant.
The complete canonical commitment set is exposed through the narrow signed
`/checkpoint-epoch/commitment-set` endpoint before beacon A is requested.
Miners verify its validator signature and ETag, and reveal selection commits to
its SHA-256. Observers can therefore reproduce selection from the signed set and
beacon A.

Durable experimental state is deliberately small: one create-only signed
intent, one create-only canonical manifest, one signed frozen commitment set,
one current pointer, an activation marker, one terminal outcome, one detached
training marker, and bounded local prepared work plus quarantine records. An
activated epoch is never reopened after restart. An interrupted epoch is
retired and its unconsumed training journal entries are tombstoned; the
validator requires a successor checkpoint before it can schedule another
epoch. A checkpoint, profile, contract, epoch, or manifest change invalidates
all unreleased work. Admission safety circuits share the one physical OPEN
phase while prompt and duplicate accounting remains bound to exact lanes.

## Threat model and measurements

Production review must cover checkpoint grinding, advance cherry-picking,
prompt-distribution bias, multi-identity flooding, common-OPEN request bursts,
stale work, and manifest equivocation. The prototype addresses these at the
protocol boundary through commit-before-beacon ordering, canonical immutable
bindings, distinct generation, admission, and final-order beacons, exact-lane
routing, bounded local queues, final cooldown checks, and create-only storage.
It does not claim that those controls replace public consistency or operational
capacity validation.

No economic result is inferred from unavailable telemetry. Before activation,
measure valid/proven groups available by deadline, environment underfill and
burned share, generated compute per selected group, accepted tokens per
compute-hour, warm-up loss, operator concentration, prompt difficulty and
diversity, common-OPEN ingress, and stale/discarded work.

The configured capacity envelope has one deterministic property: at the
default 32-reveal limit, at most 16 fully valid revealed candidates can remain
outside a full 16-group selection. This is a bound on validator-processed
payloads, not a measured claim about generation cost, profitability, or
participation. The reserve should be reduced or increased only from observed
proof-failure and underfill data.

Reviewers can exercise the deterministic synthetic shape comparison with:

```bash
python3.12 scripts/simulate_checkpoint_epoch.py
```

Its JSON output states its assumptions and marks every field that requires
authenticated operational telemetry. It is a capacity regression fixture, not
an economic forecast.

An optional CUDA qualification loads the pinned profile model, prepares one
real forced-seed group, verifies all proofs and logprobs, and can carry the
payload through the local epoch precommit, selection, reveal, admission, seal,
and winner path. It never starts a listener or contacts a validator:

```bash
RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-base-dapo-reasoning-v5 \
  python3.12 scripts/qualify_checkpoint_epoch_gpu.py \
  --max-new-tokens 8192 --http-lane
```

## Rollout and rollback

Rollout starts only in an isolated environment with the explicit capability
flag and a coordinated experimental profile whose checkpoint horizon is 16.
Reviewers may choose either training mode, but changing it creates a different
epoch intent and manifest.

Rollback disables the capability, withdraws the read-only plan surface, and
quarantines unreleased local work. Current profiles continue through their
unchanged ordinary window path. Production activation additionally requires
multi-observer intent consistency, recovery validation, capacity qualification, an
independent protocol and mechanism review, an explicit cutover, and an explicit
rollback procedure. Review must confirm the common-OPEN admission behavior
under representative load; merge of this inactive capability is not that
confirmation.
