# Checkpoint-epoch scheduling prototype

Status: experimental, disabled by default, and not production-ready. This
prototype changes no current production profile or canonical contract. A future
coordinated protocol/profile revision is required before activation.

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
   clock OPEN timestamp and therefore the same collection deadline.
7. The routes close together. One fresh public beacon obtained only after the
   common deadline supplies final equal-value ordering.
8. Lanes seal in deterministic offset order into the usable-group reservoir.
9. The manifest-bound training mode consumes the reservoir. Only after that
   may the next checkpoint be published.

`activation_not_before_round` is a lower bound, not a prediction of local
availability. The common OPEN edge occurs once the validator is ready at or
after that round.

## Canonical manifest and derivation

The strict canonical JSON manifest and its SHA-256 bind:

- schema/domain and experimental capability versions;
- protocol profile/version and canonical generation-contract hash;
- checkpoint number, repository, immutable revision, and observed commit round;
- drand source, chain, chain hash, exact round, and verified randomness;
- first logical window and exact configured horizon;
- common activation/warm-up and collection/timeout policy;
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

## Concurrent collection and miner release

The 16 entries are logical lanes inside one physical OPEN phase; they do not
open one after another. A miner queries
`/state?env=<name>&window=<number>` for each lane it intends to release.

Release requires exact agreement on checkpoint, epoch ID, manifest hash,
profile, contract, logical window, generation seed, prompt slice, and OPEN
state. Prepared payloads remain local until that check passes. Replaced, stale,
or ambiguous work is quarantined and never replayed under another binding. The
existing precommit/upload boundary, formatting, authenticity, grading, proof,
duplicate, checkpoint, and forced-stream checks remain in force.

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

After the common collection deadline, a fresh verified beacon orders candidates
that are equal on validator-authoritative utility/difficulty. Experimental
epoch ranking does not use generation completion, throughput, upload, or
arrival time. The production ranking path is unchanged when the experiment is
disabled.

The prototype retains the current selected-slot reward model and burn
accounting. It does not introduce payment for unselected work.

## Identity, quotas, persistence, and invalidation

Existing canonical operator identity remains available to existing duplicate
and proof-debt controls. This prototype introduces no epoch-wide cap and makes
no claim that a per-hotkey quota is Sybil-resistant.

Durable experimental state is deliberately small: one create-only intent, one
create-only canonical manifest, one current pointer, and bounded local prepared
work plus quarantine records. A checkpoint, profile, contract, epoch, or
manifest change invalidates all unreleased work. Window-scoped counters remain
separate for concurrent lanes.

## Threat model and measurements

Production review must cover checkpoint grinding, advance cherry-picking,
prompt-distribution bias, multi-identity flooding, common-OPEN request bursts,
stale work, and manifest equivocation. The prototype addresses these at the
protocol boundary through commit-before-beacon ordering, canonical immutable
bindings, distinct generation and post-close selection randomness, exact-lane
routing, bounded local queues, final cooldown checks, and create-only storage.
It does not claim that those controls replace public consistency or operational
capacity validation.

No economic result is inferred from unavailable telemetry. Before activation,
measure valid/proven groups available by deadline, environment underfill and
burned share, generated compute per selected group, accepted tokens per
compute-hour, warm-up loss, operator concentration, prompt difficulty and
diversity, common-OPEN ingress, and stale/discarded work.

## Rollout and rollback

Rollout starts only in an isolated environment with the explicit capability
flag and a coordinated experimental profile whose checkpoint horizon is 16.
Reviewers may choose either training mode, but changing it creates a different
epoch intent and manifest.

Rollback disables the capability, withdraws the read-only plan surface, and
quarantines unreleased local work. Current profiles continue through their
unchanged ordinary window path. Production activation additionally requires
public intent consistency, recovery validation, capacity qualification, an
explicit cutover, and an explicit rollback procedure.
