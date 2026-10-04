# Optional native runtime

The `affine-runtime` entry point supervises an explicitly pinned upstream checkout
and interpreter. It does not import or change the SN81 model, optimizer, validator
or payment path. The upstream runtime owns generation, full-vocabulary tensors,
proofs and cumulative submission serialization.

The operator configuration is an owner-only JSON file outside repositories. It
specifies `upstream_checkout`, immutable `upstream_revision`, `python`, trusted
`authority`, signed direct-R2 `current_url`, separate private `state_dir` and
`source_cache_dir`, and exactly one existing `key_file` or `cap_file`. Optional
environment, unique task indices, search/batch limits, process timeout, retries
and log bounds constrain execution. No credential is created automatically.

```sh
affine-runtime check --config "$PRIVATE_RUNTIME_CONFIG"
affine-runtime run --config "$PRIVATE_RUNTIME_CONFIG" --max-seconds 3600
affine-runtime evidence --config "$PRIVATE_RUNTIME_CONFIG" --epoch "$PRIVATE_EPOCH" --miner "$PRIVATE_IDENTITY"
```

`check` authenticates discovery and the selected manifest; it does not qualify
hardware. A zero child exit records `bootstrap_completed`, with upload, acceptance
and training unset. Evidence checks require the signed frozen receipt, exact
submission bytes, fully audited accepted pairs, exact own-pair training attribution,
changed checkpoint bytes and signed successor checkpoint before `cycle_verified`.
Accepted, scored and consumed counts are independent. Neither a payment nor an
independent GPU replay is inferred from signed reports.

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
python -m unittest tests.unit.test_affine_runtime tests.unit.test_affine_evidence
```

These checks use synthetic signatures and harmless subprocesses. They do not run
a model, TOPLOC, an optimizer, a live miner or a chain transaction.
