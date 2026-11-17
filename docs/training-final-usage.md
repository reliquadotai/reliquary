# Final training usage

`reliquary.validator.rl_training_usage` produces the immutable usage proof for one managed
training order. It reports customer billing measurements, not actual miner USD costs, provider
invoices, GPU kernel time or chain settlement. Paid admission remains disabled pending the
separate billing-policy and real-runtime qualifications.

## Drain and termination are separate

1. `bind_runtime` binds the book to one run and runtime. The controller also binds the task and
   immutable order digest. A replacement identity cannot reuse an earlier run's counters. An
   unbound book that already contains leases, groups, events, accruals or a trained checkpoint
   cannot be relabelled as a fresh runtime.
2. `request_drain` durably closes lease admission. It does not claim the trainer stopped.
3. The authenticated trainer joins its report, prefetch and checkpoint-publication work before
   acknowledging quiescence. `quiesce_trainer` seals that runtime and records the final adopted
   checkpoint. Its response may still be undrained while miner work remains outstanding.
4. `settle_quiesced_work` explicitly reports leftover or late in-zone groups as untrained and
   audits as inconclusive. It ingests the resulting ledger events. It never calls such work
   trained, and it waits while on-time submissions are still being graded.
5. After the controller's monotonic grace period, `release_for_order_drain` releases remaining
   open leases without miner penalties or auction debt. Held submissions remain protected.
   Ordinary lease expiry after the same runtime seal is also neutral.
6. Only the supervisor can provide the stopped-runtime witness after waiting for both owned
   processes and confirming the complete local cgroup is empty. A PID, registry retirement or
   a trainer acknowledgement alone is insufficient.

Finalization admits only the single-process backend with the complete local cgroup scope.
External Miles/Ray or distributed workers do not inherit this qualification from local process
group absence.

## Evidence read from the book

`finalize_usage` opens a read-only SQLite transaction after termination. The runtime, run,
task and order bindings must match. The book must retain durable admission closure and a
runtime-bound quiesced trainer/checkpoint, with no open leases, buffered work, unresolved
training or audit bodies. Every payable event must have its corresponding accrual and the
ledger cursor must reach the final event. Billable events and their accrual timestamps must
agree and fall within the owned process interval. Missing schema, unknown historic gross usage or a
changed final checkpoint blocks finalization.

The ledger fences accrual identity and gross consumption against mutation and deletion while
allowing payment status and net payment amount to follow normal settlement or forfeiture.

## Measurements and wire format

The envelope is `{ "receipt": { ... }, "sha256": "..." }`. Its digest is the canonical SHA-256
of the nested receipt. The receipt uses:

- Schema `reliquary/rl-training-usage/v1` and billing basis
  `matched_token_ceiling_and_trainer_time_v1`.
- `token_metric: gross_matched_token_equivalents_v1`: `matched_tokens` is the integer ceiling
  of the sum of retained gross `budget_tokens`, including weighted first scans. Penalties,
  forfeiture and net miner payment cannot reduce consumed work. Fractions are summed exactly
  before this single ceiling. Historic first scans without a recoverable gross counter remain
  unknown and block final billing.
- `trainer_metric: owned_process_elapsed_ms_v1`: `trainer_active_ms` is the integer ceiling of
  the supervisor's monotonic owned-trainer-process duration in milliseconds. It includes model
  loading and idle time; it is not a measurement of GPU execution time.
- Exact task, training, quote, opaque quote digest, order digest, run and runtime bindings.
  `finished_at` is a fixed UTC Unix timestamp in seconds derived from the termination witness,
  not the time of a later poll.
- The final event cursor, accrual count and consumption/termination commitments. Raw miner
  identifiers and process metadata are not returned in the receipt.
- A checkpoint reference `{version,index_sha256}` or `null`. Final consumed usage can exist
  for a failed run without a delivered checkpoint. That does not establish customer success;
  delivery must independently verify the same checkpoint reference and usage digest.

All shared numeric fields are integers from zero to `2^53-1`. Booleans, floating-point wire
values, unknown keys and mismatched digests are rejected by `validate_usage_envelope`.
The platform applies its original frozen customer ceiling, margin and component caps; the core
does not manufacture an actual monetary cost from incomplete miner accounting.

`write_final_usage` publishes complete canonical bytes with an exclusive atomic link and
durable file/directory writes. Identical retries return the retained proof; conflicting retries
cannot overwrite it. A completion store must publish successful completion only after the
checkpoint bytes and their delivery manifest have also been verified and frozen.

## Verification

`tests/unit/test_rl_training_usage.py` covers active/late work, quiescence distinct from
drainage, neutral releases, ledger lag, unknown historical usage, scope rejection, mismatched
bindings, exact fractional ceilings, immutable gross counters, strict retained-proof parsing
and parallel identical publication. Run it with the auction, ledger and lease regression suites.
These local fixtures do not qualify a real GPU runtime or authorize financial execution.
