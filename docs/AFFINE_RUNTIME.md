# Optional native runtime

`reliquary affine` forwards to the `affine-runtime` entry point, which supervises
an explicitly pinned upstream checkout and interpreter. The upstream runtime owns
generation, full-vocabulary tensors, proofs and cumulative submission serialization.
It runs as a separate operator task. An explicitly declared native competition
can pay enrolled SN81 hotkeys from authenticated native results; it does not
load the native model into the ordinary SN81 generation validator or optimizer.

The operator configuration is an owner-only JSON file outside repositories. It
specifies `upstream_checkout`, immutable `upstream_revision`, `python`, trusted
`authority`, signed direct-R2 `current_url`, separate private `state_dir` and
`source_cache_dir`, and exactly one existing `key_file` or `cap_file`. Optional
environment, unique task indices, search/batch limits, process timeout, retries
and log bounds constrain execution. No credential is created automatically.

```sh
reliquary affine check --config "$PRIVATE_RUNTIME_CONFIG"
reliquary affine prepare --config "$PRIVATE_RUNTIME_CONFIG" --out "$PRIVATE_PREPARED_CONFIG"
```

For delegated work, the client keeps its registered identity key locally. On that
machine, an existing private configuration with `key_file` can export the signed
open epoch's sealed upload capability:

```sh
reliquary affine delegate --config "$PRIVATE_CLIENT_CONFIG" --out "$PRIVATE_CAPABILITY_FILE"
```

`delegate` authenticates the configured authority and manifest and reuses native
`Identity.decrypt`. It writes a new owner-only capability file; it performs no
upload or registration and never includes the key in its output. Transfer only
that epoch capability through an approved private channel. The operator's separate
configuration uses `cap_file` and omits `key_file`. It contains native `epoch`,
`identity`, `transport`, `put_url`, `headers` and `deadline` fields. Checks validate its
identity membership, epoch, deadline, transport, headers and upload object against
the authenticated challenge. A capability authorizes that identity's submission;
it does not create a separate registration. Use one cumulative uploader per identity.

`check` authenticates discovery and the selected manifest and reports the deadline,
authorized indices, requested subset, pair quota, search bound, artifact limits,
harness and audit/runtime policies. It also describes a closed epoch with
`epoch_open: false`. Successful metadata inspection leaves `hardware_qualified`
and `paid` false, with upload, acceptance and training unset.

`prepare` writes three new owner-only files outside Git: the requested runtime
configuration, original signed snapshot and an `affine-native-task/v1` descriptor.
The descriptor contains private native identity/epoch bindings, config/snapshot
digests and the safe readiness metadata for operator queue import. Preparation
pins the selected environment, clamps search and batch limits and refuses to
overwrite files or replace unresolved running state. When no environment is
selected, it pins the first signed environment. Every exported descriptor has
`runnable: false`; this is a staged request awaiting operator qualification.
A closed epoch can be staged from an existing capability, but must be prepared
again from a fresh open epoch and matching capability before activation.
Delegation exports require an open epoch.

After qualifying the approved GPU/runtime, isolation, cancellation and whole-task
budget, run the prepared configuration and inspect its subsequent evidence:

```sh
reliquary affine run --config "$PRIVATE_PREPARED_CONFIG" --max-seconds 3600
reliquary affine evidence --config "$PRIVATE_PREPARED_CONFIG" --epoch "$PRIVATE_EPOCH" --miner "$PRIVATE_IDENTITY"
```

The native `check`, `prepare` and `run` commands do not declare an SN81 task.
Use the separate competition commands below for registry admission and settlement.
Platform execution uses its approved profile, private account/task binding and
allocation path.

A zero child exit records `bootstrap_completed`, with upload, acceptance and
training unset. Evidence checks require the signed frozen receipt, exact
submission bytes, fully audited accepted pairs, exact own-pair training attribution,
changed checkpoint bytes and signed successor checkpoint before `cycle_verified`.
Accepted, scored and consumed counts are independent. Neither a payment nor an
independent GPU replay is inferred from signed reports.

Forced-sampling epochs require the native audit assurance and per-rollout sampler
receipts bound to the exact epoch and checkpoint. Covered full-model training
requires the frozen-population context and exact pair hashes within each optimizer
step. Aggregate training counts cannot establish that the submitted pair was used.

An owner-only `manifest_snapshot_file` selects the pinned-epoch launcher. It
authenticates the original envelopes, invokes native source admission and asset
hydration, and starts a fresh isolated native CLI with that original manifest.
Discovery rotation cannot expand the approved epoch. The Platform wrapper uses
this mode and stores only a sanitized host-signed receipt.

Runtime logs and journals remain private and bounded. A terminal journal follows
process-group cleanup. A crashed running journal blocks replacement work.
`reconcile` clears it only when its recorded process group is confirmed absent;
it never kills an unidentified process.

A verified handover exposes successor bindings only when source content,
environment, harness and numerical/runtime policies remain compatible. Checks
include the forced sampler's source and stable contract fields. Valid
per-epoch public randomness may change; adding, removing or changing the sampler
policy requires fresh qualification.
Successor bindings support qualification of the next open epoch; they do not replace
hardware, isolation, cancellation or budget qualification. Execution through the
Platform additionally uses its existing private Docker allocation and independent
host deadline. Publish an approved profile only after real operator qualification.

CPU checks:

```sh
python -m unittest tests.unit.test_affine_runtime tests.unit.test_affine_evidence tests.unit.test_affine_prepare tests.unit.test_affine_cli_alias
```

These checks use synthetic signatures and harmless subprocesses. They do not run
a model, TOPLOC, an optimizer, a live miner or a chain transaction.

## Bounded native competition

`reliquary affine competition` implements one native epoch as an SN81 task under
`native-affine-points`. Each participant uses its own registered native identity
and one native cumulative uploader. The existing pinned native worker runs the
task; the coordinator imports authenticated results and settles its declared cap.
There is no new queue or inference implementation. Install the `affine` extra for
native signature verification.

All drafts, signed enrollments, native capabilities and evidence bundles remain
owner-only outside Git. Only a commitment digest and the sanitized SN81 reward
archive enter R2. The native identity seed stays on the participant's machine;
a worker receives its one-epoch delegated capability through a private channel.

```sh
reliquary affine competition prepare --config "$PRIVATE_RUNTIME_CONFIG" \
  --task-id "$TASK_ID" --cap 0.05 --out "$PRIVATE_COMPETITION_DRAFT"

# Each participant signs the same draft with both its native identity and SN81 hotkey.
reliquary affine competition enroll --competition "$PRIVATE_COMPETITION_DRAFT" \
  --native-key "$PRIVATE_NATIVE_SEED" --wallet-name "$WALLET_NAME" \
  --wallet-hotkey "$WALLET_HOTKEY" --out "$PRIVATE_ENROLLMENT"

# Rehearse the declaration locally. Repeat --enrollment for every participant.
reliquary affine competition declare --competition "$PRIVATE_COMPETITION_DRAFT" \
  --enrollment "$PRIVATE_ENROLLMENT" --dry-run --out "$PRIVATE_TASK_ENTRY"
```

Before actual declaration, every generation and weight-only validator must run
a binary that knows `native-affine-points`. Unknown mechanisms invalidate the
whole registry on older readers. The `default` task must already be declared,
and total reserved caps must fit within 1.0; the existing `tasks set-cap` command
can make room. The declaration requires explicit fleet acknowledgement:

```sh
reliquary affine competition declare --competition "$PRIVATE_COMPETITION_DRAFT" \
  --enrollment "$PRIVATE_ENROLLMENT" --fleet-knows-native-affine-points
reliquary tasks list
```

Declaration creates an immutable roster commitment before the signed native
deadline, checked against R2's server timestamp, and creates the registry task
under its existing compare-and-swap safeguards. One native identity maps to one
SN81 hotkey; both sign the task/manifest binding. The epoch commitment rejects
reuse under another task, including after retirement. A declaration is not
GPU qualification. Operators still qualify the exact native source, hardware,
proofs, transport and bounded worker before running the prepared allocation.

Participants run their ordinary native prepared configuration. Once Affine
publishes its signed finalized history, capture each enrolled submission's native
evidence. `--submission-sha256` can pin the frozen submission digest when the
worker's private artifact is held elsewhere.

```sh
reliquary affine run --config "$PRIVATE_PREPARED_CONFIG" --max-seconds 3600
reliquary affine competition capture --config "$PRIVATE_RUNTIME_CONFIG" \
  --competition "$PRIVATE_COMPETITION_DRAFT" --enrollment "$PRIVATE_ENROLLMENT" \
  --out "$PRIVATE_EVIDENCE_BUNDLE"

# Repeat --enrollment and --evidence for all enrolled submissions.
reliquary affine competition settle --competition "$PRIVATE_COMPETITION_DRAFT" \
  --enrollment "$PRIVATE_ENROLLMENT" --evidence "$PRIVATE_EVIDENCE_BUNDLE" \
  --dry-run --out "$PRIVATE_REWARD_ARCHIVE"
reliquary affine competition settle --competition "$PRIVATE_COMPETITION_DRAFT" \
  --enrollment "$PRIVATE_ENROLLMENT" --evidence "$PRIVATE_EVIDENCE_BUNDLE"
reliquary affine competition status --competition "$PRIVATE_COMPETITION_DRAFT" \
  --enrollment "$PRIVATE_ENROLLMENT"
```

Settlement requires the exact signed manifest, common finalized score/challenge,
signed history membership, and every submitted participant's fully audited native
pair evidence. It verifies the forced sampler assurance and native request bindings.
It pays `cap * native final weight`, including native penalty adjustment. Shares
of identities outside the enrolled roster burn; enrolled miners are not
renormalized to absorb them.

Current native sampled audits finalize an observed subset while unchecked
duplicate claims remain unresolved. This scope is accepted only when the signed
manifest explicitly carries the matching native live reward contract. The archive
preserves `provisional`, `duplicate_coverage`, and unresolved-claim fields. It
attests controller-authenticated native results, not global uniqueness, an
independent GPU replay, or a received SN120 payment.

One create-only R2 settlement exists per native epoch. Retries preserve its
original entry period. A retired task cannot create a new settlement; settle
the finalized epoch before retiring it. Weight-only validators replay it on the existing
72-minute `period-ema-v1` clock and clamp it to the current registry cap. It enters
on the next period and decays even if no further work arrives. `tasks close`
releases the reserved cap after finalized settlement and the existing decay
threshold; ordinary corpus/SFT archives and generation continue on their own path.

```sh
python -m unittest tests.unit.test_affine_competition tests.unit.test_affine_runtime
```

These CPU checks exercise real native and hotkey signatures, an isolated harmless
worker, immutable precommit/settlement, native observed-subset penalties, the actual
weight reader, cap enforcement and decay. They do not establish GPU qualification
or perform storage/chain writes against production.
