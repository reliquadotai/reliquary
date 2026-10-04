# Optional native runtime

`reliquary affine` forwards to the `affine-runtime` entry point, which supervises
an explicitly pinned upstream checkout and interpreter. The upstream runtime owns
generation, full-vocabulary tensors, proofs and cumulative submission serialization.
It runs as a separate operator task; the SN81 model, optimizer, validator and
payment path remain independent.

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

This command surface does not register an SN81 corpus/SFT job, change its emissions
or distribute native Affine execution to ordinary SN81 miners. That requires a
separate task mechanism and qualified worker distribution. Platform execution
uses its own approved profile, private account/task binding and allocation path.

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
