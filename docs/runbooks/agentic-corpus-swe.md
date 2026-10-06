# Agentic corpus execution

An episode job is defined by its immutable manifest and served task contract.
The contract pins the checkpoint, renderer, environment artifacts, token
budgets, and proof rules used by the producer and verifier.

The native `corpus mine-agentic` command selects one explicit job, checks its
served contract and checkpoint fingerprint, and produces signed trajectories.
Its generation runtime captures the actual decode activations and builds a
proof for each assistant span. Token and span bounds are validated before
admission; the verifier checks the declared checkpoint and proof contract.

The worker requires the selected environment artifacts and a qualified
inference runtime. The miner uses its own hotkey; controller, storage, and
cold-wallet credentials are not part of the miner interface.

Completion, proof verification, grading, delivery, and incentive settlement
are separate outcomes. Confirm each required outcome from its native record.
A zero-cap task carries no task incentive.

Service shutdown awaits owned background work and finalizers. See the
[split-process runtime reference](../design/2026-10-02-corpus-split-processes.md)
for process roles and shutdown grace configuration.

## Signed-sandbox execution (plan 3; not deployed)

A job opts in with `episode.execution: "signed_sandbox"` and an `episode.sandbox` object
(`env`, `env_package` exactly as record 0 carries it, `tools`, `sandbox_commit`,
`budgets`); `replay_fraction_failed` must be 0. Replay jobs are unchanged. Every
environment step then runs on reliquary-sandbox machines; the validator verifies the
signed transcript at admission (§5.A-D) and grades from its final record. There is no
grade executor, no replay and no vote.

**Blocker for third parties.** reliquary-sandbox is a private repository: miners and
validators outside the team cannot install `reliquary[sandbox]` /
`reliquary[sandbox-miner]` until it is published. That is the user's decision.

**One reliquary-swe wheel everywhere.** `env_package` is `name==version+g<sha16>` and
its digest depends on how the package was installed. The sandbox gateways, the
validators and the miners install reliquary-swe from the same git wheel at the job's
pinned commit; a different install method gives a different `env_package` and every
transcript is refused (record 0 `env_package`) or never placed.

### Safe deployment order

1. Install `reliquary[sandbox]` in the validator image, set nothing yet, and redeploy:
   with no key settings and no signed job, the validator imports nothing new and
   replay jobs run exactly as before.
2. Generate the validator key (below) and give every machine its public key.
3. Register the machines in the directory, and check that each answers `GET /capacity`
   from the validator host.
4. Set the key settings on the validator and restart it. From now on its start reads the
   directory and restores the sessions from R2 (see "Restore" below).
5. Only then declare the signed job in the registry. Miners need
   `reliquary[sandbox-miner]` before they can mine it.

Rolling back is the reverse: retire the signed job first, then unset the key settings.

### Validator

1. `python -m reliquary_sandbox.attest.signing generate /etc/reliquary/sandbox-validator.pem`
   prints the public key. Each machine reads `RELIQUARY_SANDBOX_VALIDATOR_PUBLIC_KEYS`,
   a JSON object from key id to the base64 Ed25519 public key, for example
   `{"validator-2026-10": "<base64>"}`. The key id there must be exactly the validator's
   `RELIQUARY_SANDBOX_VALIDATOR_KEY_ID`: a token names its key id, and a machine that knows
   the key under another id refuses every token.
2. The key file must be a regular file with mode 0600 (or stricter), owned by the user
   the validator process runs as; a symlink is refused. Under Docker, that is the
   container's user, not the host's: `chown` the file to the container's uid (root in
   the current image) on the host, and mount it read-only (`:ro`) at the path the setting
   names. Only the path is ever logged, never the contents.
3. Set `RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE` and `RELIQUARY_SANDBOX_VALIDATOR_KEY_ID` (neither
   may be blank). After a rotation also set `RELIQUARY_SANDBOX_VALIDATOR_RETIRED_KEYS`, a JSON
   object `{key_id: base64 public key}`. Without both key settings, a signed job is not
   served: it is logged as unserved, and replay jobs start as before.
4. Policy defaults per hotkey:
   - 8 live sessions in total (`RELIQUARY_SANDBOX_MAX_LIVE_PER_HOTKEY`);
   - 4 live sessions per job (`RELIQUARY_SANDBOX_MAX_LIVE_PER_HOTKEY_JOB`);
   - one live session per prompt (fixed);
   - 120 opens per hour, with aborted opens refunded (`RELIQUARY_SANDBOX_MAX_OPENS_PER_HOUR`);
   - 20 aborted sessions per 24 h (`RELIQUARY_SANDBOX_MAX_ABORTED_PER_DAY`);
   - a 300 s claim ttl (`RELIQUARY_SANDBOX_CLAIM_TTL_S`).

   A prompt never has more live sessions than free slots.
5. The machine directory:
   - it is re-read every `RELIQUARY_SANDBOX_DIRECTORY_REFRESH_S` (30 s);
   - each read is bounded by `RELIQUARY_SANDBOX_DIRECTORY_READ_TIMEOUT_S` (15 s);
   - past `RELIQUARY_SANDBOX_DIRECTORY_MAX_AGE_S` (120 s) without a good read, the
     validator fails closed: no session is placed, no signed transcript is admitted,
     and miners get retryable refusals.
6. Closes:
   - a close's body has `RELIQUARY_SANDBOX_CLOSE_BODY_TIMEOUT_S` (30 s) to arrive, else
     `408 body_timeout`;
   - at most `RELIQUARY_SANDBOX_CLOSE_CONCURRENCY` (4) transcripts are verified at once;
   - a close that waits longer than the io timeout for a slot gets `503 close_busy` with
     `Retry-After`.

   Every setting must be a positive, finite number.
7. Register each machine: `reliquary sandbox machines register --machine-id ... --address
   http://host:port --provider ... --capacity ... --key-id ... --public-key ...
   --valid-from <unix>`. Addresses come from this directory only.
8. Key rotation: `add-key`, switch the machine when `GET /capacity` shows `active: 0`,
   then `end-key --valid-until <start of the new key>`. **Compromise:** `end-key
   --valid-until <earliest suspected time> --compromise` (past times allowed; the end is
   moved 30 s earlier for clock skew). From the next directory refresh on, transcripts
   opened after that time no longer verify. Transcripts already admitted stay admitted.
9. `status --status draining` stops new sessions. A machine silent for 60 s is drained
   automatically and its live sessions voided (no fault to their miners).

### Restore

At start, before any route serves, the validator reads the directory once. A failure
there is logged and retried by the fleet, and it never stops the start. It then restores
the recent sessions from R2 (`reliquary/sandbox/sessions/...`). Each attempt is bounded
to 60 s, and there are 3 attempts with backoff. **If the restore still fails, the
validator does not start at all, so the replay jobs it serves stop too.** It would
otherwise serve opens with empty reservations and caps. Fix R2 access and restart. To
keep replay jobs running meanwhile, unset the key settings and retire the signed job.
At shutdown, the session writes still in flight get 10 s.

### Session route

The route is `POST /corpus/sandbox/sessions` (and `/{id}/close`) on the corpus app.
The existing `/corpus` nginx location and tunnel serve it. A close carries a transcript
of up to 8 MiB plus 64 KiB of envelope, so the `/corpus` location's
`client_max_body_size` must be at least 9m. The current edge template sets 128m; check
any other proxy in front.

A signed open or close body is a bearer credential for its freshness window (120 s): it
carries the session token or the request that obtains one. Serve the route only over
TLS or the validator's tunnel, and never log request bodies at the proxy.

### Miners

`reliquary corpus mine-agentic` detects the job's execution. In signed mode it needs
`reliquary[sandbox-miner]` (verifiers installed from git at the pinned commit) and no
Docker. It reports every unsubmitted session so that its reservation ends early.
A graded transcript it will not submit (its own precheck refused it, the submission
was refused, the deadline passed) is closed with `withdraw` and that transcript: the
validator verifies it and frees the slot and the hotkey's caps for good. Signed mode
needs `--validator-hotkey` (session requests are signed for that validator) and keeps at
most `--max-live-per-job` (default 4, the validator's cap) sessions live per job.
