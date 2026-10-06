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

Validator:
1. `python -m reliquary_sandbox.attest.signing generate /etc/reliquary/sandbox-validator.pem`
   prints the public key; put it in every machine's `RELIQUARY_SANDBOX_VALIDATOR_PUBLIC_KEYS`.
   The file is mode 0600 and owned by the validator's user; only its path is ever logged.
2. Set `RELIQUARY_SANDBOX_VALIDATOR_KEY_FILE`, `RELIQUARY_SANDBOX_VALIDATOR_KEY_ID`
   (and, after a rotation, `RELIQUARY_SANDBOX_VALIDATOR_RETIRED_KEYS`, JSON
   `{key_id: base64 public key}`). Without both key settings a signed job is not
   served (logged as unserved; replay jobs start as before). Policy defaults:
   8 live sessions, 120 opens per hour (aborted refunded), 20 aborted per 24 h per hotkey,
   a 300 s claim ttl (`RELIQUARY_SANDBOX_CLAIM_TTL_S`); `RELIQUARY_SANDBOX_MAX_*` override
   the caps. The machine directory is re-read every `RELIQUARY_SANDBOX_DIRECTORY_REFRESH_S`
   (30), each read bounded by `RELIQUARY_SANDBOX_DIRECTORY_READ_TIMEOUT_S` (15); past
   `RELIQUARY_SANDBOX_DIRECTORY_MAX_AGE_S` (120) without a good read the validator fails
   closed (no session placed, no signed transcript admitted, retryable refusals). At most
   `RELIQUARY_SANDBOX_CLOSE_CONCURRENCY` (4) closes are verified at once; others wait up to
   the io timeout, then get `503 close_busy` with `Retry-After`.
3. At start, before any route serves, the validator reads the directory once and
   restores the recent sessions from R2 (`reliquary/sandbox/sessions/...`). **If the
   restore fails the validator does not start**: it would otherwise serve opens with
   empty reservations and caps. Fix R2 access and restart.
4. Register each machine: `reliquary sandbox machines register --machine-id ... --address
   http://host:port --provider ... --capacity ... --key-id ... --public-key ...
   --valid-from <unix>`. Addresses come from this directory only.
5. Key rotation: `add-key`, switch the machine when `GET /capacity` shows `active: 0`,
   then `end-key --valid-until <start of the new key>`. **Compromise:** `end-key
   --valid-until <earliest suspected time> --compromise` (past times allowed; the end is
   moved 30 s earlier for clock skew): transcripts opened after it no longer verify from
   the next directory refresh on. Transcripts already admitted stay admitted.
6. `status --status draining` stops new sessions; a machine silent for 60 s is drained
   automatically and its live sessions voided (no fault to their miners).

The session route is `POST /corpus/sandbox/sessions` (and `/{id}/close`) on the corpus
app: the existing `/corpus` nginx location and tunnel serve it. Requests are signed for
this validator's hotkey and the route's path, so they only reach it over TLS or the tunnel.

Miners: `reliquary corpus mine-agentic` detects the job's execution. In signed mode it
needs `reliquary[sandbox-miner]` (verifiers installed from git at the pinned commit), no
Docker, and it reports every unsubmitted session so its reservation ends early.
