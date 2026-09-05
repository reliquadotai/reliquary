# Reliquary 1 reconciliation

Status: implementation contract for an inactive release candidate. This
document does not activate a validator, miner, profile, reward policy, or
checkpoint. Production activation requires the gates in this document and a
coordinated release.

Reliquary 1 reconciles the checkpoint-epoch prototype with the useful streaming
parts of the fill-closed work. Its goal is to increase the amount of verified,
useful training data per unit of inference while making speed an operational
metric rather than a hidden market rank.

## Decisions

Reliquary 1 adopts a **ticketed, trainer-paced checkpoint epoch**:

- checkpoint commitment precedes all generation randomness;
- one later public epoch beacon derives 16 manifest-bound generation seeds and
  prompt slices;
- miners choose which eligible prompts and lanes to offer through signed,
  compact intents;
- the complete intent population is frozen before primary and standby tickets
  are selected without using arrival, generation time, bytes, or tokens;
- all 16 logical lanes have one atomic, common OPEN edge;
- only ticketed payloads enter streaming grading and proof verification;
- each lane has an advertised minimum generation interval and a hard bound,
  while durable trainer progress may delay but never accelerate its seal;
- a fresh public beacon obtained after a lane's eligible set is frozen orders
  exact selection ties;
- selected slots are the default reward unit;
- a durable lane journal makes selection, training, payment, restart, and
  checkpoint publication agree;
- a successor epoch cannot begin until the trained checkpoint is published,
  verified, and adopted.

The existing throughput-ranked production path and V4/V5 generation contracts
remain byte-for-byte unchanged during implementation. Reliquary 1 is a new
coordinated release, not a reinterpretation of V4 or V5.

## Branch implementation status

This document defines the target release contract. It is intentionally more
strict than the current runtime and must not be read as an activation claim.
The reconciliation branch currently contains:

| Surface | Status on this branch |
| --- | --- |
| 16-lane manifest, unique seeds/slices, checkpoint-before-beacon ordering | implemented behind the disabled checkpoint-epoch capability |
| signed miner-selected intents, frozen population, primary/standby tickets | implemented behind the disabled checkpoint-epoch capability |
| atomic common OPEN, exact miner release binding and stale-work quarantine | implemented in the reference shadow planner and validator path |
| utility/difficulty portfolio, operator rounds and fresh post-close tie beacon | implemented without changing production ranking |
| bounded proof scheduler, progressive trainer journal and cursor telemetry | implemented in the separate disabled fill experiment |
| coherent fill horizon, capacity and read-only progress state | implemented for qualification of that experiment, not selected as the v1 market |
| create-only fill journal receipts and checkpoint-adoption rotation barrier | implemented behind the disabled fill experiment |
| immutable release contracts and environment/task/trajectory ABI | implemented as inactive composition boundaries |
| profile/capability activation atomicity | implemented fail-closed; no existing V4/V5/V6 profile may activate the checkpoint epoch |
| immutable checkpoint OID/repository binding and final miner admission recheck | implemented for the reference miner and bounded control endpoint |
| cooldown and archive restart identity | local and remote records are reconciled by exact window; invalid, incomplete or conflicting recovery fails closed |
| ticket-only streaming proof shared across all 16 concurrent lanes | integration gate; must preserve arrival-neutral final selection and restart recovery |
| trainer-paced per-lane epoch barriers | target contract; the current epoch uses one common generation horizon before ordered finalization |
| durable per-ticket proof ownership, lane acknowledgements and economic release | activation gate; current staging types are not connected to the live epoch path |
| external adapter conformance and hardware-backed end-to-end qualification | activation gate |

The pull request remains a Draft until every activation gate is either
implemented and qualified or removed from the candidate capability bundle.
Code that exists only in the fill experiment is reusable infrastructure, not
evidence that its admission or reward policy has been selected.

## What is retained and what changes

The reconciliation retains these fill-closed implementation ideas:

- bounded asynchronous proof scheduling;
- grading and proof work streamed while a lane remains open;
- trainer-consumption telemetry and explicit backpressure;
- progressive construction of lane payloads;
- fill, failure, queue-age, and proof-capacity measurements.

It changes the market control loop:

- proving is restricted to selected primary or activated standby tickets;
- reaching the validator first does not reserve a training slot;
- production rate is telemetry, not admission or final ranking;
- the batch cannot close earlier than its advertised lane barrier;
- a trainer cursor may delay a barrier but cannot shorten miner generation
  time;
- per-token payment is an independently versioned experiment, not the default
  Reliquary 1 reward policy.

This keeps proof and trainer utilization improvements without requiring miners
to generate an unbounded population that the validator has already decided not
to train.

## Release contract instead of numeric feature dispatch

`reliquary.protocol.release_contract` introduces an immutable composition root:

| Component | Canonically binds |
| --- | --- |
| `wire` | endpoints, signed envelopes, state fields, errors and idempotency |
| `generation` | model revision, prompt rendering, sampling, token/logprob and answer format |
| `market` | phases, intent/ticket rules, bounds, selection and reward semantics |
| `verification` | grading, authenticity, proof, duplicate and cooldown checks |
| `training` | lane journal, batching mode, optimizer barriers and publication rules |
| `environments` | one component ID and canonical hash for every enabled environment |

Each component has an explicit ID and SHA-256 of its canonical JSON payload.
The release contract has its own canonical bytes and SHA-256. A change to one
component therefore cannot silently inherit the identity of another.

Capabilities are exact string membership, for example
`market.ticketed-paced-epoch/v1` and
`verification.streaming-ticketed/v1`. New code must never infer a capability
from an ordered integer comparison. A release number is a human label, not a
feature predicate.

Parsing is strict: duplicate, missing, unknown, non-canonical, unsorted, or
malformed fields are rejected. There are no parser defaults for consensus
fields.

## Epoch state machine

### 1. Checkpoint ready

The source checkpoint is an immutable repository revision accepted by the
normal intake and checkpoint-verification path. The epoch intent binds the
checkpoint number, repository, revision, release-contract hash, training-run
identity, and environment component hashes.

### 2. Pre-beacon intent durable

Before the epoch beacon exists, the validator writes and signs a create-only
intent naming the exact future beacon round. The strongest locally verifiable
ordering evidence is persisted. If the infrastructure cannot provide a
consensus timestamp for that write, the remaining trust assumption is stated
as such; it is not described as trustless.

### 3. Epoch manifest published

The validator fetches and verifies exactly the named later beacon. It derives
an epoch root and exactly `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS` indexed lane
entries. Production configuration is 16 and activation fails closed if the
release contract and configured horizon disagree.

The immutable manifest includes:

- schema, domain, epoch and release-contract identities;
- checkpoint and pre-beacon ordering evidence;
- beacon chain, round and verified randomness;
- all lane numbers, generation seeds and per-environment prompt slices;
- common OPEN, warm-up, intent, generation, lane-barrier and hard-stop policy;
- primary, standby, queue and verification bounds;
- environment universe revisions and deterministic overlap policy;
- selection, reward, training and publication capability IDs.

The manifest is available early enough for checkpoint download and model
warm-up. This delay is an advertised field, never a local sleep.

### 4. Miner-selected intent phase

A miner locally chooses an eligible environment, lane and prompt from the
published slices, then signs a compact intent. The validator does not assign a
prompt or require a payload during this phase. Every accepted intent is bound
to the checkpoint, epoch, manifest, release contract, generation seed, prompt
slice, environment component and operator identity available through the
canonical chain mapping.

Intent limits bound cheap ingress separately from proof capacity. Identity
limits are not described as Sybil-resistant unless the chain supplies the
required canonical operator relation. Where that relation is insufficient,
Reliquary records concentration telemetry and does not invent one.

### 5. Frozen population and tickets

At the advertised intent deadline, the validator drains already-admitted
compact requests, canonicalizes the complete population, signs it and persists
the exact bytes. Mutation or a second population for the same epoch ID is a
fatal equivocation error.

A separately targeted public beacon, unavailable when the population was
signed, deterministically selects primary and ordered standby tickets in
operator rounds. Arrival, generation completion, transport latency, payload
size, token count and production rate do not affect ticket order.

### 6. Atomic common OPEN

All 16 lanes become OPEN in one routing transition. A miner generates only for
its primary tickets. Standby activation is deterministic and observable; it
may occur at manifest-bound progress points when verified fill shows that a
lane cannot meet its target. An activated standby receives the same remaining
deadline as every other participant and never changes an earlier ticket's
binding.

A miner may prepare work locally but cannot release it unless live state still
matches the exact checkpoint, epoch, manifest, release, environment, lane,
prompt, seed and ticket. Stale, replaced, aborted or ambiguous work is durably
quarantined and cannot be replayed under another binding.

### 7. Streaming verification

Ticketed submissions follow the existing forced-stream, token/logprob,
formatting, grading, proof, checkpoint, duplicate and cooldown checks. A final
prompt-range and cooldown check runs at admission. Proof work is bounded by the
number of primary plus activated standby tickets, not by arrival bursts.

Passing validation makes a candidate eligible; it does not make it selected or
paid. Verification results are written before the corresponding in-memory
reservation is released.

### 8. Trainer-paced lane barriers

All lanes started together, but they need not seal together. Each manifest
entry advertises an earliest close and hard stop. Lane `i` keeps accepting
eligible ticketed work after its earliest close until the earlier of trainer
capacity becoming available or its hard stop. It may freeze only when:

1. its advertised minimum generation interval has elapsed;
2. the trainer has durably acknowledged lane `i - 1`, except for lane zero, or
   the advertised hard stop has arrived; and
3. its ingress and verification queues are durably drained under the
   manifest-bound timeout policy.

Trainer backpressure can extend collection within the advertised hard bound;
it can never close a lane early. A hard-stopped lane freezes independently but
cannot be emitted to the trainer before its predecessor acknowledgement. Later
lanes consequently receive useful time while earlier lanes are verified and
trained, without turning an unpredictable fill race into consensus.

### 9. Selection and seal randomness

The generation plan never contains auction or seal randomness. After a lane's
eligible candidate set is frozen and signed, the validator obtains a fresh
public seal beacon for that lane.

Selection first applies explicit training-value policy: proof eligibility,
environment balance, outcome/difficulty strata and diversity constraints. The
seal beacon orders exact equal-value candidates only. Throughput, arrival,
response length, raw tokens and payload bytes do not create an advantage.

The initial portfolio policy is a release-contract parameter and must be
validated from replay; a convenient stratum split is not assumed to be an
eternal quality definition. If live graders provide insufficient resolution,
the result is reported rather than silently relabelled as quality selection.

### 10. Durable journal, training and payment

The lane selection record, payload references, rejected/tombstoned entries and
seal evidence are committed atomically under the epoch and lane identity.
Only a complete durable lane record becomes visible to the detached trainer.
Only a trainer acknowledgement of that exact record advances the cursor.

`sequential_steps` is the default: the 16 lane batches produce 16 ordered
optimizer steps against one published source checkpoint. An explicitly bound
`aggregate_one_step` research mode may aggregate the epoch into one optimizer
step, but it is not claimed to be mathematically equivalent.

The default reward is one selected-slot share. This avoids treating response
length as a quality oracle. Completion-token, length, cost and EOS statistics
remain measured, and any future token-weighted reward requires a separate
market component, replay, shadow accounting and coordinated activation.

Payment cannot advance beyond the durable selected-and-trained record.
Recovery is idempotent: a lane is either pending, terminal with a payload, or
terminal with a tombstone. It is never silently skipped.

### 11. Publication and adoption gate

After all 16 lane markers and optimizer steps are terminal, the trainer may
publish at most one successor checkpoint. The next epoch remains blocked until
the validator observes the immutable publication, verifies checkpoint intake,
and adopts the new revision as its generation source. Timeout aborts or pauses;
it does not reuse the previous epoch's randomness or reinterpret its work.

## Randomness domains

Reliquary 1 uses independent, framed domains:

1. epoch beacon: available only after checkpoint commitment, used for lane
   generation seeds and prompt slices;
2. ticket beacon: available only after the signed intent-set freeze, used for
   primary and standby selection;
3. lane seal beacon: available only after that lane's signed eligible-set
   freeze, used for equal-value ordering.

One epoch beacon is sufficient for 16 generation seeds. Reliquary does not
pretend future beacon outputs are known: indexed, domain-separated hashes of
the verified epoch beacon create unique per-lane generation inputs. No value
from that derivation is accepted as seal randomness.

## Prompt slices and full-horizon revelation

Validator and miner use one shared derivation implementation. Slices are
non-overlapping when the environment universe permits. If it does not, the
manifest names a deterministic cycle/fallback rule and admission still applies
the existing dataset and cooldown semantics. An earlier lane's admitted prompt
deterministically makes the same prompt unavailable to a later sealing lane.

Publishing the complete horizon allows miners to compare prompts across lanes.
That can alter prompt difficulty and diversity. Reliquary treats this as a
measured product choice, not a free efficiency claim. Prompt offer, intent,
ticket, validity, selection and training distributions are retained for replay.

## Environment ABI and external adapters

The core runtime should depend on a small Reliquary episode ABI rather than on
environment-specific Python classes. The ABI consists of canonical forms for:

- `EnvironmentManifest`: identity, revision, dataset snapshot, prompt schema,
  rollout policy, grader/verifier contracts and capabilities;
- `TaskRef`: immutable task identity and content digest;
- `EpisodeRequest`: task plus generation and checkpoint binding;
- `EpisodeTrace`: ordered messages/actions/tool results and token provenance;
- `GradeResult`: named rewards, outcome, difficulty, diagnostics and evidence;
- `TrainingGroup`: verified traces plus the market and lane binding.

Math and Code become adapters to this ABI before new environments are added.
The exact canonical environment-manifest SHA-256 is the consensus boundary;
Python types and adapter implementations are not. Environment discovery may be
dynamic, but consensus accepts only manifest IDs and hashes allowlisted by the
active release contract. Updating code or data never mutates an already-open
epoch.

Prime-compatible and other external formats are optional, pinned adapter
packages. They translate at the boundary and cannot redefine canonical
signing, randomness, proof or training semantics. Each adapter pins the
upstream interface revision and passes golden round-trip, masking, tool-trace,
reward and failure fixtures. Upstream names are integration concerns, not
protocol feature switches.

## Persistence and restart model

The durable key is:

```text
(release_hash, checkpoint_revision, epoch_id, manifest_hash, lane_number)
```

Create-only records cover pre-beacon intent, manifest, frozen intent set,
tickets, standby activation, eligible-set freeze, seal evidence, selected lane,
trainer acknowledgement, payment acknowledgement and checkpoint adoption.

On restart the state reducer rebuilds memory exclusively from these records.
Contradictory bytes, missing parents, a cursor beyond the journal, or an
unverifiable signature stop the epoch for operator review. Recovery never
guesses which of two states was intended. Temporary transport retries use
idempotency keys; semantic retries across a changed binding are forbidden.

The disabled fill implementation now commits a hidden, fsynced payload body
and retained SHA-256 receipt before making a journal slot visible. The
assembler advances its index and accrues rewards only after that durable
commit; identical retries are no-ops and conflicting bytes fail closed. Its
persisted between-window barrier binds the complete journal range to the
parent checkpoint and, for a full publication cadence, stays closed until both
trainer consumption and exact successor-checkpoint adoption are observed.

This does not yet reconstruct ownership of a partially completed live window.
Receipt-backed bytes survive restart, but in-memory proof, admission and
assembler reward state does not. Activation therefore requires either durable
transaction state for that lifecycle or an explicit whole-window
abort/quarantine protocol. Ambiguous replay is not a recovery mechanism.

## Compatibility and code layout

Historic checkpoints, archives and receipts remain readable. Legacy behavior
is isolated behind `reliquary.compat` adapters selected by exact contract ID or
hash. Strings such as an external API's `v2`, a schema revision or Pydantic's
major version are not protocol dispatch and must not be renamed blindly.

The active path should converge on these boundaries:

```text
protocol/       release and component contracts, signatures, wire schemas
epoch/          pure state reducer, manifest, randomness and lane lifecycle
market/         intents, tickets, selection, rewards and accounting
verification/   admission, grading, authenticity, proof scheduler
training/       durable lane journal, assembler, trainer cursor, publication
environments/   Reliquary ABI, registry, native and external adapters
compat/         exact historic readers and temporary dual-read bridges
```

Large service and batcher objects orchestrate these modules but do not own
their state rules. Numeric feature comparisons are removed from new modules.
Legacy comparisons are inventoried, replaced with exact capabilities, then
deleted only after the compatibility window closes.

## Threat model and required invariants

The release review covers:

- checkpoint choice bias and commit-before-beacon ordering;
- manifest or population equivocation;
- misuse or cross-domain reuse of public randomness;
- full-horizon prompt-selection and distribution bias;
- identity concentration and multi-identity flooding;
- simultaneous OPEN bursts and verification exhaustion;
- stale work, replay and cross-epoch substitution;
- malformed environment adapters or upstream semantic drift;
- journal split-brain, partial writes and trainer/payment disagreement;
- checkpoint publication without verified adoption.

The public specification states verifiable invariants and remediation status.
Operational detection detail and abuse-enabling procedures belong in restricted
security runbooks, not miner-facing documentation.

## Metrics and experiments

Every experiment compares the current production policy, the rate-paced
proposal and ticketed-paced Reliquary 1 using the same archived or synthetic
offer population. Unavailable fields are marked unavailable.

Required measures include:

- offered, ticketed, generated, valid, selected and trained groups;
- discarded generation work and compute per selected/trained group;
- accepted training tokens and verified groups per wall-clock hour;
- lane fill, underfill, standby activation and hard-stop frequency;
- generation, verification, trainer and publication latency distributions;
- response length, EOS, outcome, difficulty, prompt and environment mix;
- prompt diversity, duplicate/cooldown rejection and full-horizon bias;
- selection and reward concentration by the strongest canonical operator
  identity available, with the identity limitation reported;
- OPEN burst, queue age, proof utilization and backpressure;
- stale/quarantined work, restart recovery and journal divergence;
- training loss, KL, reward, stability and held-out evaluation by lane/epoch.

No single throughput, token, length or step-cadence metric is a sufficient
proxy for training value. Activation requires a predeclared decision table and
confidence intervals, not a favorable point estimate.

## Migration

1. **Freeze and inventory.** Pin current V4/V5 bytes and hashes; inventory
   numeric dispatch, active endpoints, archive readers and environment calls.
2. **Contract shadow.** Generate ReleaseContract and environment manifests in
   shadow mode; compare hashes and state without accepting new traffic.
3. **Module extraction.** Introduce the pure epoch reducer, ticketed market,
   durable lane journal and episode ABI behind disabled capabilities.
4. **Dual read.** Keep historic readers while new writers emit only the new
   canonical records. Compare replay and restart state.
5. **Closed qualification.** Run CPU determinism, process-restart, local E2E,
   proof-capacity, trainer and multi-backend environment tests.
6. **Public shadow.** Advertise the candidate release read-only; miners may
   plan locally but cannot send through the new path.
7. **Coordinated test epoch.** Activate on isolated infrastructure with rewards
   disabled or shadow-accounted and a published abort plan.
8. **Coordinated cutover.** Pin one release-contract hash and activation
   checkpoint/round. Reject mixed contracts rather than guessing compatibility.
9. **Retirement.** Remove inactive write paths after the observation window;
   retain the minimum exact readers required for historic data.

## Rollback

Rollback operates at an epoch boundary. The validator stops new intents,
terminally aborts and quarantines the candidate epoch, preserves its signed
records, and returns the active release pointer to the last qualified contract.
It never changes a manifest in place, pays untrained lanes, reuses randomness,
or moves payloads into another epoch.

If the candidate has published a successor checkpoint, rollback requires an
explicit checkpoint decision; the previous release cannot silently consume a
checkpoint produced under different training semantics.

## Production gates

Reliquary 1 is activation-ready only when all of the following are true:

- canonical component/release hashes and V4/V5 compatibility are pinned;
- commit-before-beacon, domain separation and 16-seed uniqueness are proven;
- validator/miner manifest and prompt-slice derivation are byte-identical;
- arrival, throughput, bytes and tokens have no ticket or ranking advantage;
- generation and seal randomness cannot be substituted;
- all 16 lanes OPEN atomically and obey their advertised barriers;
- no stale or pre-OPEN work can be released or replayed;
- every checkpoint reference is an immutable commit OID bound to its repository;
- final cooldown, prompt-range, authenticity, grading and proof checks pass;
- ticket, proof queue and standby bounds survive burst tests;
- lane journal, trainer cursor, payment and restart are exactly consistent;
- archive/tombstone coverage is contiguous across restart and conflicts cannot
  be reinterpreted as an empty cooldown;
- checkpoint publication and adoption gate the next epoch;
- Math, Code and external adapter conformance fixtures pass;
- full Python, formatting, local integration and hardware-backed E2E suites
  pass without weakened assertions;
- shadow replay meets the predeclared training-value, waste, fairness and
  stability thresholds;
- independent protocol and security reviews close all activation blockers;
- the deployment, observation and rollback owners approve the exact release
  contract hash.

Until then, the capability remains disabled by default and the implementation
must not alter production profiles, rewards, checkpoints or live state.
