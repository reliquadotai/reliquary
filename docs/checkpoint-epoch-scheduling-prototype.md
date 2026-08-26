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
2. The validator observes drand round `R0`, durably fixes the intent for
   `R1 = R0 + 1`, and confirms that intent locally before `R1` is available.
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
   already at ingress before obtaining fresh public beacon A. That beacon
   deterministically selects a bounded reveal cohort without using commitment
   arrival time. Selection visits the canonical operator identities in
   deterministic rounds.
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

The validator persists a create-only pre-beacon intent and then a create-only
manifest. Restart recovery reloads the same canonical bytes. Local persistence
is the strongest locally verifiable commit-before-beacon rule available here;
it is not a public proof of prior commitment. Public signed intent publication
and consistency observation remain production gates.

## Concurrent commitment, selection, and reveal

The 16 entries are logical lanes inside one physical OPEN phase; they do not
open one after another. A miner queries
`/state?env=<name>&window=<number>` for each lane it intends to release.

Preparing work does not send its payload. During the commitment phase the miner
submits the existing signed precommit envelope, which binds the exact serialized
payload hash, size, prompt, lane, checkpoint, profile, and generation inputs.
The validator stores that compact object without reserving proof or grading
capacity. A payload sent before selection is refused.

After beacon A, miners poll the narrow commitment-status endpoint. A selected
commitment receives one bounded reveal deadline; a non-selected commitment
cannot upload. Immediately before reveal, the miner rechecks exact OPEN and
reveal phase plus checkpoint, epoch ID, manifest hash, profile, contract,
logical window, generation seed, prompt slice, and cooldown. Replaced, stale,
or ambiguous work is quarantined and never replayed under another binding. The
existing formatting, authenticity, grading, proof, duplicate, checkpoint, and
forced-stream checks remain in force.

Every lane targets `B_BATCH` selected groups. The experimental default chooses
at most `B_BATCH + B_BATCH / 2` payload reveals per environment/lane: 16 targets
plus an 8-candidate proof-failure reserve with the production value of
`B_BATCH`. The limit is applied only after the commitment phase and therefore
does not create a first-arrival pool. The reveal bound and compact-commitment
bound are manifest-bound and cannot change within an epoch.

The common commitment phase defaults to the existing per-window duration
multiplied by the checkpoint horizon. It never closes early. State advertises
the exact phase, durations, target, and selected reveal limit. Compact
commitments are bounded per canonical operator and lane. This controls validator
work without claiming that uncoordinated local preparation has zero discarded
work. Selection also grants at most one reveal ticket per operator and prompt
inside a lane.

The local reference planner is bounded and has no submission transport. A
backend-neutral callback may prepare token sequences, after which the existing
request builder performs canonical rewards, proofs, log probabilities,
signatures, and formatting.

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
The compact commitment set remains validator-local in this vertical slice.
Selection is reproducible from that set and beacon A, while cross-observer
completeness requires a future public consistency surface.

Durable experimental state is deliberately small: one create-only intent, one
create-only canonical manifest, one current pointer, an activation marker, one
terminal outcome, and bounded local prepared work plus quarantine records. An
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
default 24-reveal limit, at most eight fully valid revealed candidates can
remain outside a full 16-group selection. This is a bound on validator-processed
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

## Rollout and rollback

Rollout starts only in an isolated environment with the explicit capability
flag and a coordinated experimental profile whose checkpoint horizon is 16.
Reviewers may choose either training mode, but changing it creates a different
epoch intent and manifest.

Rollback disables the capability, withdraws the read-only plan surface, and
quarantines unreleased local work. Current profiles continue through their
unchanged ordinary window path. Production activation additionally requires
public intent consistency, recovery validation, capacity qualification, an
independent protocol and mechanism review, an explicit cutover, and an explicit
rollback procedure. Review must confirm the common-OPEN admission behavior
under representative load; merge of this inactive capability is not that
confirmation.
