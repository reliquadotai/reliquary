# Checkpoint-epoch scheduling prototype

Status: experimental, disabled by default, and not production-ready. This
prototype does not change V4, V5, or any current production behavior. A future
coordinated protocol/profile revision is required before activation.

## Purpose

Reliquary publishes a new checkpoint after a configured number of successful
training windows. During that interval miners use the same immutable model
revision. The checkpoint epoch makes the generation inputs for that complete
horizon available together so miners can choose eligible prompts and prepare
work ahead of each ordinary window.

The horizon is always `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS`; it is not copied
as a literal into the protocol implementation.

This remains miner-directed work:

- the validator publishes prompt slices and generation randomness;
- miners choose their own prompts inside those slices;
- miners may prepare work in any order and retain it locally;
- the validator never assigns prompts or generation jobs;
- a payload is submitted only while its exact ordinary window is `OPEN`.

## Checkpoint-bound lifecycle

The lifecycle deliberately adds one small step at a checkpoint boundary:

1. The next immutable checkpoint revision is created and the validator's model
   and proof replicas are coherent with it.
2. The validator observes drand round `R0` and durably fixes the exact next
   round `R1 = R0 + 1` before `R1` is available.
3. The validator fetches and verifies exactly `R1`. A delayed relay may return
   historical `R1`; another round is never substituted.
4. That single beacon is expanded into the complete epoch root, all generation
   seeds, and every environment's prompt slices.
5. The immutable checkpoint binding and canonical epoch manifest are exposed
   together at the checkpoint boundary.
6. The manifest advertises `activation_not_before_round`. This gives miners
   time to download the checkpoint and warm their models before the first
   planned window.
7. The existing window state machine opens and closes the next horizon windows
   normally. The plan is indexed by window number, not by predicted wall-clock
   or absolute future opening rounds.

Removing absolute future opening rounds is intentional. Validator training,
archive, or publication latency may move an ordinary window in time without
changing which manifest entry belongs to it. Missing one anticipated round
therefore cannot invalidate unrelated prepared work.

## Canonical manifest

The manifest is strict canonical JSON and has a canonical SHA-256. It binds:

- schema, derivation domains, and experimental capability version;
- protocol profile/version and canonical generation-contract hash;
- checkpoint number, repository, and immutable revision;
- drand source, chain hash, exact round, and verified randomness;
- first window number and exact configured horizon;
- advertised activation/warm-up boundary;
- the ordinary window timing policy and its collection/timeout bounds;
- every environment name and dataset-universe size, with the dataset revision
  remaining bound by the generation contract;
- prompt-range width and explicit overlap policy;
- every window offset, window number, generation seed, and prompt slice.

The validator persists the intent before fetching the beacon and creates the
manifest once. A restart reloads the same canonical bytes. The same epoch ID
cannot acquire another manifest, checkpoint, contract, or beacon binding.

## Randomness derivation

One real public beacon is enough. The prototype never assumes that future drand
outputs already exist.

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

Direct indexed derivation is used instead of a mutable random-number generator.
It is deterministic, permits independent validation of any window, and gives
each offset a unique domain-separated seed. Mutating the checkpoint, contract,
beacon, horizon, offset, window number, environment, or dataset binding changes
the appropriate derived value and manifest hash.

## Prompt slices and cooldown

Each environment receives one slice for every horizon entry. Slices are
non-overlapping when the dataset universe permits it. If the configured horizon
cannot be represented without overlap, the manifest uses one explicit,
deterministic fallback policy; it never silently widens the dataset or weakens
cooldown rules.

Prompt selection within a published slice is intentionally miner-directed.
The validator and miner use the same slice derivation. Immediately before
submission, the miner rechecks the live prompt range and cooldown. Admission
rechecks range, canonical content, cooldown, duplicates, formatting,
authenticity, grading, proof, checkpoint, profile, contract, window, and
generation randomness.

Publishing the whole horizon deliberately changes the observable prompt
distribution: miners can compare choices across future slices before deciding
what to generate. That is a conscious product choice, not a free efficiency
claim. A production revision must measure the resulting difficulty, diversity,
environment balance, and selected-training distribution.

## Window execution and submission

The manifest changes generation planning, not the ordinary submission state
machine:

- future payloads remain local;
- the miner polls live state before release;
- release requires the exact checkpoint, epoch ID, manifest hash, profile,
  contract, window number, generation seed, prompt slice, and `OPEN` state;
- replaced, stale, or ambiguous work is quarantined and is never replayed under
  another binding;
- the existing precommit/upload flow remains the network boundary.

The reference planner is bounded and durable. Its default mode has no network
send capability. Generation is an external callback boundary; protocol
validity depends on the submitted tokens and proofs rather than a particular
generation implementation. A shared request builder converts backend-produced
token sequences through the existing reward, proof, log-probability, signature,
and formatting path.

## Ranking, seal randomness, and rewards

The advance manifest contains generation randomness only. It never contains or
derives final auction/seal randomness.

Each ordinary window obtains a fresh verified beacon only after its actual
collection deadline. Experimental epoch ranking uses validator-authoritative
utility/difficulty first and that fresh beacon for equal-value ordering. It
does not use generation completion time, throughput, upload time, or arrival
time. Existing production ranking remains byte-for-byte isolated when the
experiment is disabled.

The prototype retains the current selected-slot reward model. It does not add a
new payment class for work that does not enter the selected batch. Unfilled
slot mass continues to burn under existing accounting.

Final epoch ranking is arrival-neutral; the existing bounded admission path is
not redesigned in this prototype. Admission behavior under advance preparation
is therefore a production measurement and protocol-gate item, not something
this design assumes away.

## Persistence and invalidation

The durable state is intentionally small:

- one create-only pre-beacon intent;
- one create-only canonical manifest;
- one current-plan pointer;
- the existing committed-window cursor;
- local prepared-work records and quarantine records.

A checkpoint, profile, contract, epoch ID, or manifest change invalidates all
unreleased work. Work already prepared for a later window remains valid across
ordinary timing delays as long as its immutable bindings still match.

## Experimental surface

When the capability is disabled:

- the epoch endpoint is unavailable (`404`);
- epoch fields are omitted from state;
- the normal window randomness, ranking, miner loop, and checkpoint cadence are
  unchanged;
- V4 and V5 canonical contracts and hashes are unchanged.

When enabled in a local experiment:

- the validator exposes the canonical manifest read-only;
- state advertises only the epoch ID and canonical manifest hash needed to bind
  miners to that manifest;
- the ordinary window loop consumes the entry matching the current window;
- the no-submit planner may prepare and quarantine bound future work.

No deployment, live activation, checkpoint publication, wallet operation,
weight submission, or production profile change is part of this prototype.

## Implementation scope

The useful vertical slice consists of:

1. shared canonical intent, manifest, derivation, and validation;
2. create-only validator persistence and read-only discovery;
3. a small hook that supplies the current normal window with its planned seed
   and slice after the advertised activation boundary;
4. isolated arrival/throughput-neutral experimental ranking with a fresh
   post-deadline seal beacon;
5. a bounded no-submit miner planner with exact-OPEN release validation and
   stale-work quarantine.

Absolute future schedules, a separate epoch window runner, an epoch-specific
archive reconstruction system, synthetic economic claims, and production send
activation are outside this minimal prototype.

## Trust and threat model

The prototype fixes the checkpoint, target round, and manifest in create-only
local storage before consuming the target beacon. A miner can independently
verify the checkpoint binding, canonical manifest, and public beacon, but local
storage alone is not a public proof that the validator committed before that
beacon. Signed public intent publication and consistency observation are
required production gates.

The main design risks are handled as follows:

- checkpoint or manifest replacement invalidates all unreleased local work;
- reuse of an epoch ID with different canonical bytes is rejected;
- generation and final-selection randomness use separate domains and separate
  beacon times;
- prompt choice is constrained to published slices and checked again with
  cooldown and canonical content at release and admission;
- chain operator identity continues to scope duplicate and proof-debt controls,
  but this prototype introduces no epoch quota and makes no Sybil-resistance
  claim;
- admission capacity, participation concentration, prompt-distribution change,
  checkpoint-publication timing, and stale-work rates must be measured before
  activation;
- an interrupted generation or release transition is quarantined and cannot be
  replayed ambiguously after restart.

These are protocol review boundaries, not changes to existing authenticity,
grading, proof, duplicate, or cooldown enforcement.

## Rollout, rollback, and measurements

Rollout begins only in an isolated environment with the explicit capability
flag and a future experimental profile. Rollback disables the capability,
withdraws the plan surface, and quarantines unreleased work; current profiles
continue through their original path.

No economic result is asserted from synthetic or unavailable telemetry. Before
production, measure valid groups available by deadline, underfill and burned
share, generated compute per selected group, accepted tokens per compute-hour,
warm-up loss, operator concentration, prompt difficulty/diversity, admission
load, and stale/discarded work. Full-horizon revelation is acceptable
only if those measurements support the intended training outcome.

## Production gates

Production activation requires a coordinated profile revision, signed public
checkpoint/manifest publication, cross-validator consistency, bounded ingress
that preserves the declared ranking semantics, operational capacity
qualification, checkpoint-bound recovery, and an explicit rollback procedure.
Existing proof, authenticity, grading, cooldown, duplicate, and checkpoint
verification semantics remain unchanged and outside this prototype's scope.
