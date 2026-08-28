# Checkpoint-epoch market prototype

Status: experimental, disabled by default, and not production-ready. The
prototype changes no active profile, canonical contract, reward rule, or
legacy ranking path. Production activation requires a coordinated protocol
revision and an isolated end-to-end qualification.

## Goal

Reliquary should maximize useful and diverse training groups per unit of
inference and validator work. Miner prompt selection is part of the product:
the market should preserve it without using transport timing, response length,
or raw throughput as a proxy for training value.

The prototype replaces 16 sequential generation races with one checkpoint
epoch whose horizon is `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS`. It publishes all
generation seeds and prompt slices together, opens every logical lane at the
same instant, and supports either 16 ordered optimizer steps or one aggregated
optimizer step.

The design does not promise zero unused inference. Proof failures and partial
outages require redundancy. It instead makes redundancy explicit and bounded,
and prevents unselected work from starting in the normal path.

## Mechanism

The schema-v8 experiment separates four decisions that the production window
currently combines:

1. miners select prompts cheaply;
2. public randomness selects who may spend generation compute;
3. selected payloads are validated during a common generation horizon;
4. a fresh post-deadline beacon orders a balanced training portfolio.

Neither selection accepts arrival time, bytes, completion tokens, generated
groups per second, or a drand-derived elapsed-time bucket as an input.

### 1. Checkpoint and public plan

The validator first commits the immutable checkpoint revision. It then fixes
and signs an intent for a later drand round, persists that intent, and fetches
exactly the targeted beacon only after it becomes available. The beacon derives
the epoch root, 16 unique generation seeds, and every environment's 16 prompt
slices.

The manifest binds:

- schema and capability versions;
- protocol profile/version and canonical contract hash;
- checkpoint number, repository, revision, and observed commit round;
- source beacon chain, round, and randomness;
- first window, exact horizon, and common schedule;
- advertised warm-up, intent, generation, upload-grace, and backup timings;
- every per-environment prompt slice and every generation seed;
- target, backup, operator-round, portfolio, reward, and training policies;
- the canonical manifest SHA-256.

The plan contains no seal or auction randomness. All 16 lanes share one atomic
OPEN timestamp after the advertised warm-up boundary.

The current infrastructure can prove local durable ordering: checkpoint and
signed intent bytes exist before the targeted epoch beacon. It cannot turn an
HTTP publication timestamp into a trustless global fact. A production revision
must specify an independently observable commitment surface.

### 2. Self-selected generation intentions

During the advertised intent phase, a miner submits a signed, compact claim
binding one selected prompt to:

- operator and miner identities;
- epoch ID and manifest hash;
- logical window and environment;
- prompt index and validator-canonical prompt-content hash;
- checkpoint, profile, protocol, and generation randomness;
- a unique nonce.

An intention has no generated tokens or payload hash. It is cheap enough that
response length and generation hardware cannot increase the number of claims
made before the deadline. Prompt range and cooldown are checked here, then
checked again when a selected payload is admitted.

Intentions are bounded per canonical operator and lane. That mapping is the
repository's accounting identity; the experiment does not claim that it is
intrinsically Sybil-resistant. Identity cost and concentration must be measured
before choosing a production cap.

### 3. Frozen population and generation tickets

At intent close, the validator stops ingress, drains requests already inside
the handler, freezes the exact canonical intent set, signs it, persists it
create-only, and exposes its bytes and hash. Only then may it fetch beacon A.

Beacon A selects primary and standby generation tickets. Operators are visited
in deterministic rounds; an additional claim by one operator cannot precede
the first eligible claim of another operator. Prompt and operator/prompt caps
are applied during selection. Arrival order is absent.

Primary ticket holders may generate immediately. Standby ticket holders must
not generate until their advertised backup wave becomes active. The default
waves occur at configured fractions of the common horizon. Intermediate waves
activate only the deterministic number needed to cover observed shortfall. The
final wave activates the remaining manifest-bounded reserve because seal-time
proof failures are not yet observable. This caps worst-case generation at 2×
the target with the production-sized fixture, instead of allowing an unbounded
race, while giving valid primaries most of the horizon without redundant work.

A full payload precommit must carry the selected intent ID. The validator
rejects it unless the ticket is primary or an activated backup and every
checkpoint, epoch, manifest, lane, prompt, profile, and randomness binding is
exact. A standby, unselected, stale, replaced, or expired intention cannot
reserve upload capacity.

### 4. Streamed validation and final seal

Selected miners generate during the common horizon and use the existing
payload precommit/reveal transport. Decode, formatting checks, prompt binding,
grading, duplicate checks, token/logprob authenticity preparation, and bounded
admission run as payloads arrive. Arrival changes validator utilization only;
it does not change rank or reward.

The final GRAIL proof budget remains bounded and is consumed after the candidate
population freezes. This avoids making proof-worker availability an admission
clock. Forced-stream handling, formatting, grading, GRAIL, checkpoint binding,
cooldown, and duplicate protection are unchanged.

After the generation deadline and admission drain, the validator obtains fresh
beacon B. Beacon B is later than beacon A and never appears in the advance plan.
It is used only for deterministic operator ordering and exact-value ties.

## Training portfolio

Pure throughput ordering was rejected because it concentrates admission on
fast production rather than demonstrated training value. Pure random selection
was also rejected because it discards the information supplied by validator
grading. Raw per-token payment is not included: it would change the reward
unit, make padding controls economically load-bearing, and can turn the market
into a direct contest for paid compute volume.

The experimental portfolio uses validator-derived group outcomes:

- frontier signal: mean reward in `(0.00, 0.25]`;
- learning signal: mean reward in `(0.25, 0.75]`;
- consolidation signal: mean reward in `(0.75, 1.00]`.

For a target of 16, the manifest-bound starting quotas are 4/8/4. Inside each
stratum, canonical operators are visited in rounds. Robust utility selects an
operator's best candidate; it does not decide which operator is visited next.
Unused quota spills deterministically into eligible strata. Prompt index and
content identities stay unique across the selected portfolio.

The 4/8/4 mix is an explicit experiment, not a claim of optimality. It prevents
one easy or one narrowly optimized difficulty region from absorbing the whole
batch while producing direct telemetry for a later training ablation. The
production flat-value toggle cannot silently replace this policy in epoch mode.

Rewards remain the existing equal selected-slot split. Response length, token
count, generation duration, and arrival do not change a selected slot's value.
Because generation rights are chosen before generation begins, making a
response shorter cannot create more tickets for that epoch. A future reward
change must be evaluated separately with EOS, padding, quality, concentration,
and cost measurements.

## Prompt slices and miner choice

Slices are non-overlapping when the environment universe permits it. If it
does not, the manifest names a deterministic cycle policy. Admission and final
selection retain the existing cooldown semantics; earlier logical lanes win a
deterministic overlap when the same content cannot be used twice.

Publishing the full horizon lets miners compare prompts across lanes. That is a
deliberate trade: it enables miner curation and scheduling, but can bias the
training distribution toward locally predictable outcomes. The required
measurements are selected difficulty, prompt/content diversity, environment
coverage, local-score calibration, and divergence from uniform slice sampling.

## Training modes

`RELIQUARY_EXPERIMENTAL_CHECKPOINT_EPOCH_TRAINING_MODE` is bound into the plan:

- `sequential_steps` consumes the 16 lane reservoirs in manifest order while
  retaining one frozen behavior checkpoint for the epoch;
- `aggregate_one_step` combines the usable reservoir into one optimizer update
  using the trainer's existing token-budgeted microbatching and gradient
  accumulation.

The modes are not mathematically equivalent. Both publish at most one successor
checkpoint after the complete epoch. The detached trainer consumes an epoch
only after its terminal marker exists, so a partial restart cannot silently
train or publish half an epoch.

## Miner queue and invalidation

The reference planner is opt-in and has no send capability by default. Its
durable states are:

```text
intent_pending -> primary/standby -> selected -> generating -> prepared
prepared -> released
any ambiguous or stale state -> quarantine
```

The miner revalidates exact live state immediately before generation and again
before payload release. It checks the checkpoint, protocol, canonical contract,
epoch, manifest, intent-set hash, lane, generation seed, prompt slice, prompt
content, cooldown, phase, deadline, and active backup wave. Future work never
submits early. A checkpoint, contract, manifest, or generation-randomness
change invalidates the whole queue; ambiguous work is quarantined and is never
replayed under a new binding.

The planner reports generated GPU-seconds, prepared and released groups,
stale/ambiguous discards, lane/environment coverage, underfill opportunity,
backup activation, and queue age. Local scheduling may use deterministic lane
order or an estimated eligible-value-per-second heuristic. Local scheduling
metadata is never sent to the validator and never enters ranking.

## Restart and equivocation

The checkpoint intent, manifest, frozen generation-intent set, activation, and
terminal outcome use create-only persistence. Reinstalling identical bytes is
idempotent; different bytes for the same identifier are rejected. An
interrupted activated epoch is retired and tombstoned as one unit rather than
reopened with an ambiguous population.

## Threat model and mitigations

- Checkpoint grinding is limited by binding the immutable revision before the
  epoch beacon.
- Advance cherry-picking and prompt-distribution bias are observable through
  manifest-bound slices and selection-distribution telemetry; they remain a
  product trade requiring ablation.
- Submission volume cannot buy earlier admission because the population is
  frozen and operator-rounded before generation.
- Simultaneous OPEN bursts carry compact intentions rather than generated
  payloads; payload work is bounded by selected tickets.
- Manifest or intent-set equivocation is detectable from canonical signed
  hashes and rejected by create-only storage.
- Stale and ambiguous work is quarantined against its exact binding.
- Final ordering cannot be precomputed because beacon B is fetched only after
  collection closes.

## Evaluation and activation gates

The repository replay records how much candidate inference is unused under the
current windows and compares arrival, throughput, operator-round, and
checkpoint-epoch policies. Synthetic runs are capacity checks only; unavailable
economic or training telemetry is reported as unavailable rather than inferred.

Before coordinated activation, reviewers must measure on an isolated run:

- valid and proven groups available by each deadline;
- inference-seconds per selected group and accepted tokens per accelerator-hour;
- environment underfill, burned slots, backup activation, and stale work;
- time to primary completion and the common-OPEN request burst;
- operator reward HHI/Gini and tickets per operator;
- selected response-length distribution without using length as quality;
- prompt difficulty, content diversity, and cross-lane selection bias;
- training loss, KL, reward, stability, and downstream quality for both training
  modes;
- proof-plane capacity after the changed path manifest is requalified.

Activation also requires a new protocol/profile capability, miner cutover
documentation, rollback rehearsal, and independent protocol review. Rollback is
disabling the experimental capability and returning to the untouched current
window loop. This branch must not be deployed, merged, or used to publish a
checkpoint as part of prototype qualification.
