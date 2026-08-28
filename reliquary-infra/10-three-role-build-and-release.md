# Three-role build and release contract

Status: implementation complete on the prototype branch; production authority
remains disabled until the external-machine and real-key gates below are run.

## The role switch

One checkout contains all three CPU roles. The role is selected only by the
inventory and the first argument:

```bash
./reliquary-infra/deploy_role.sh ctrl-01 inventory.yml
./reliquary-infra/deploy_role.sh cpu-exec-01 inventory.yml
./reliquary-infra/deploy_role.sh signer-01 inventory.yml
```

The wrapper validates that the selected host exists, syntax-checks the exact
playbook, limits execution to that host, and then applies the role. It does not
guess network ranges, identities, image IDs, or credentials.

| Role | Receives | Owns | Explicitly must not own |
|---|---|---|---|
| `ctrl-01` | public miner traffic after cutover; read-only chain state; signed worker results | protocol FSM, admission, result comparison, immutable edge snapshots, mTLS client leaves | hotkey, coldkey, hostile runtime privileges |
| `cpu-exec-01` | content-bound Code batches from ctrl over mTLS | disposable gVisor/KVM workers and no authoritative expected values | every wallet, R2/HF/provider master credential, validator state |
| `signer-01` | three semantic operations from ctrl over its own mTLS CA | hotkey file, replay journal, chain-write construction | coldkey, arbitrary-sign API, public ingress, Docker socket, GPU/storage credentials |

The existing trainer and isolated proof process remain the GPU data plane. This
branch does not change miner protocol semantics, window ordering, reward
comparison, or training ownership. It extracts the two dangerous CPU concerns:
hostile Code execution and Bittensor signing.

## Executor protocol v2 is current, not obsolete

`EXECUTOR_PROTOCOL_VERSION = 2` is an internal wire contract version. Version
2 replaced the earlier prototype request shape with a deterministic job ID
covering the protocol version, exact grader-runtime digest, code digest,
ordered cases, and both deadlines. The server rejects every other version.

It is called “content-bound” because mutating the code, entrypoint, arguments,
case order, runtime, or limits without recomputing the identifier makes schema
validation fail. Expected answers and reward comparison never leave ctrl-01.
Changing the number to 3 without changing the contract would weaken rollout
safety, not modernize it.

## Signer protocol

Signer protocol v1 is independently versioned from executor protocol v2. It
has no arbitrary byte-signing route. Its complete operation set is:

1. `checkpoint-sign`: fixed netuid and repository, monotonically increasing
   checkpoint number, exact 40-hex revision. It preserves the existing
   `checkpoint_n|revision` signature bytes so miners do not change.
2. `set-weights`: fixed netuid, monotonically increasing chain epoch, sorted
   unique UIDs, finite unit-mass vector.
3. `serve-axon`: fixed netuid and exact validator IP/port from signer policy.

The operation ID is a domain-separated SHA-256 of canonical structured
content. SQLite/WAL records reservations and completed results with FULL
synchronous durability. Exact completed retries return the cached result.
Older counters are denied. A timeout around an on-chain operation is marked
`uncertain` and can never be replayed automatically because the extrinsic may
already have been accepted.

The signer loads only one hotkey and verifies its public SS58 address against
configuration before listening. It constructs all chain calls itself. The
control process receives a public identity object whose `sign()` method always
fails, so an accidental legacy signing call cannot silently fall back locally.

## Immutable artifacts

Build on a trusted checkout after committing the release revision:

```bash
REVISION="$(git rev-parse HEAD)"
./scripts/build_cpu_executor_artifact.sh \
  "/var/lib/reliquary-build/cpu-executor/${REVISION}" "${REVISION}"
./scripts/verify_cpu_executor_artifact.sh \
  "/var/lib/reliquary-build/cpu-executor/${REVISION}"

./scripts/build_signer_artifact.sh \
  "/var/lib/reliquary-build/signer/${REVISION}" "${REVISION}"
./scripts/verify_signer_artifact.sh \
  "/var/lib/reliquary-build/signer/${REVISION}"
```

Both playbooks import only the archive whose SHA-256 and Docker image ID match
the inventory. `pull_policy: never` prevents a tag from changing underneath a
role. The executor lock excludes wallet, GPU, HF, and storage dependencies. The
signer lock excludes GPU, model, dataset, and storage dependencies.

## Independent mTLS domains

Generate leaves on the operator workstation:

```bash
./scripts/generate_cpu_executor_pki.sh pki/executor cpu-exec-01.internal
./scripts/generate_signer_pki.sh pki/signer signer-01.internal
```

These create separate CA roots. Copy only the server directory to its server
and only the client directory to ctrl-01. Keep both `ca.key` files offline.
The playbooks fail if a CA key is present in an upload directory. Leaves expire
after 90 days by default and the playbooks refuse less than 14 days of
remaining validity.

## Hotkey staging

The signer playbook deliberately does not move a hotkey through ctrl-01 or Git.
Before activating signer-01, use an operator-to-signer transfer and stage only:

```text
/var/lib/reliquary-signer/wallets/<wallet>/hotkeys/<hotkey>
owner 10001:10001
mode  0600
```

The playbook refuses to start if that file is missing or has another owner or
mode. It also refuses `coldkey` and `coldkeypub.txt`. The coldkey remains
offline.

## Control activation order

1. Keep current production authoritative.
2. Deploy `ctrl-01` with both URLs empty. This is the current wallet-free edge.
3. Deploy `cpu-exec-01`; configure ctrl with executor mTLS and mode `shadow`.
4. Pass deterministic parity, malicious corpus, saturation, restart, and host
   isolation gates. Change only `RELIQUARY_GRADER_EXECUTOR_MODE=remote`.
5. Deploy signer-01 with a test hotkey first and run mTLS/semantic/replay gates.
6. Stage the validator hotkey directly on signer-01, verify the SS58 address,
   and set the signer URL/client leaf on ctrl. Remove every local wallet mount.
7. Run shadow production and deterministic checkpoint/signature parity.
8. Make ctrl public/authoritative only after monitoring and rollback are
   exercised.

The control entrypoint treats a configured signer URL as a hard mode switch:
all CA/certificate/key/public-address values become mandatory and wallet
directory checks are skipped because no wallet is mounted. Read-only chain
queries remain on ctrl; checkpoint signatures, weights, and axon registration
go through signer-01.

## Remaining external gates, not unfinished code

The following cannot be truthfully completed on one machine and are therefore
release gates rather than implementation TODOs:

- prove KVM and gVisor isolation on the actual `cpu-exec-01` kernel/CPU;
- measure private-network latency and saturation between physical hosts;
- validate the real validator hotkey without copying it through ctrl;
- submit a real chain extrinsic only during an approved non-authoritative test;
- prove failure behavior when either remote host is physically unavailable;
- run shadow production against live miner traffic before authority changes.

No server purchase or production change is required to build, inspect, test,
and freeze both artifacts on ctrl-01.
