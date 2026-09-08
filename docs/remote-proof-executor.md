# Remote GRAIL proof executor

The explicit remote mode keeps tasks, tokenizer, admission, the existing global
proof scheduler, scoring and payment on the CPU controller. GPU model replicas
live in the existing `ProofWorkerPool` on one separate proof host. No local model
is loaded or copied on the controller. Detached training remains mandatory.
Local mode remains the default. Remote and shadow modes require V5 or later,
with a generation-contract hash and an exact published SHA resume source.

This is an implementation boundary, not GPU or production qualification. CPU
transport tests exercise a fake GPU identity and a separate tiny CPU kernel
parity case. They cannot qualify numerical agreement, GPU throughput, provider
isolation or hardware attestation. mTLS authenticates a workload with a trusted
certificate; it does not prove honest execution on a third-party GPU host.

## Worker

Run the same immutable source/image and protocol profile on both hosts. The
worker entry point is `reliquary proof-worker` or
`python -m reliquary.validator.remote_proof_server`.

Set these values through the deployment configuration:

| Variable | Meaning |
| --- | --- |
| `RELIQUARY_PROOF_HOST` | Dedicated private bind IP |
| `RELIQUARY_PROOF_PORT` | Dedicated TLS port, default `8445` |
| `RELIQUARY_PROOF_WORKER_ID` | Expected stable workload identity |
| `RELIQUARY_PROOF_DEVICES` | Explicit physical indices such as `cuda:0,cuda:1` |
| `RELIQUARY_PROOF_SLOTS_PER_DEVICE` | Existing process-slot setting; existing MPS/support guards apply |
| `RELIQUARY_PROOF_TLS_CERT` / `RELIQUARY_PROOF_TLS_KEY` | Worker server leaf and private key |
| `RELIQUARY_PROOF_CLIENT_CA` | Dedicated CA that authorizes only the proof controller |
| `RELIQUARY_PROTOCOL_PROFILE` | Exact shared protocol profile |
| `RELIQUARY_TRAINING_RUN_ID` | Exact shared training run |
| `RELIQUARY_HF_REPO_ID` | Exact public checkpoint repository |

Preserve the profile's experimental activation flag when applicable. There is no
arbitrary model override: the worker bootstraps the profile's pinned base model,
then explicitly adopts a numbered checkpoint from the configured repository.
The existing immutable build-revision mechanism is required. Each process
reports its actual loaded OID, physical GPU UUID/class, model metadata and
numerical runtime; a parent process revision cache is insufficient.

Use a separate proof trust domain, never the signer or code-executor client CA.
Keep its issuing key offline. A worker needs neither the validator hotkey nor
R2, signer or global HF credentials. Its checkpoint metadata reads explicitly
use anonymous HF access; the deployment must also provide no implicit HF
credential to the existing model loader. Restrict ingress and egress per the
role's deployment policy. Do not mount this application on the public validator
listener or run the app factory through a TLS-optional server.

## Controller

Set `RELIQUARY_DETACHED_TRAINER=1`, use the existing published SHA resume
configuration, and configure:

| Variable | Meaning |
| --- | --- |
| `RELIQUARY_PROOF_EXECUTOR_MODE` | `local` (default), `shadow`, or `remote` |
| `RELIQUARY_PROOF_EXECUTOR_URL` | Bare private HTTPS origin, no path, userinfo or redirects |
| `RELIQUARY_PROOF_TLS_CA` | Worker server CA |
| `RELIQUARY_PROOF_TLS_CERT` / `RELIQUARY_PROOF_TLS_KEY` | Dedicated controller client leaf/key |
| `RELIQUARY_PROOF_EXPECTED_WORKER_ID` | Exact worker identity |
| `RELIQUARY_PROOF_CAPACITY_MANIFEST` / `RELIQUARY_PROOF_CAPACITY_MANIFEST_SHA256` | Pinned capacity evidence |

The existing `RELIQUARY_PROOF_WORKER_REQUEST_TIMEOUT_SECONDS` and
`RELIQUARY_PROOF_WORKER_RELOAD_TIMEOUT_SECONDS` also bound network proof and
adoption calls. Allow enough reload time for all configured replicas to install.
Transport deadlines are absolute; keep the hosts' clocks synchronized.

Remote mode uses metadata proxies for every scheduled, forensic and legacy
proof path. The initial SHA resume downloads only profile/tokenizer/config
metadata on the controller. Subsequent R2 checkpoint intake still stages the
existing snapshot on disk but never materializes model tensors there. The GPU
host independently resolves the immutable public HF revision. Both intake
lineage and GPU adoption must succeed before the new public checkpoint manifest
is installed. `/runtime-contract` reports the acknowledged GPU fingerprint,
not the CPU controller's.

Shadow mode requires the existing local isolated pool and leaves its result
authoritative. A bounded background comparison records equal/divergent/error/
dropped counters; shadow errors and saturation cannot change a local verdict.
Startup still validates the configured remote identity. Promotion to remote
requires separate capacity and GPU qualification; changing the flag alone is
not evidence of parity.

## Wire and failure contract

The only routes are `GET /v1/health`, `POST /v1/adopt`, `POST /v1/prove` on a
mandatory client-certificate TLS listener. JSON is versioned
`reliquary.remote-proof/v1`, typed, finite, size-bounded and rejects duplicate
keys/unknown fields. No pickle, callbacks, tensor payloads or controller paths
cross the network.

Each proof binds immutable profile/hash/run/repo/N/OID, worker process session,
slot, runtime hash, window, environment, controller-generated job/attempt,
content digest and absolute expiry. Responses repeat these bindings and the
complete request hash. They return every existing sparse `ProofResult` field
needed for GRAIL, logprobs, authenticity, seed/CDF, terminal/natural closure and
telemetry. Token/challenge coverage is checked before downstream gates run.
The controller still applies those original gates.

An exact retry of a finished attempt returns the same bounded cached result;
changed bytes for that attempt are refused. Transport retry never silently
runs another job with different bytes. In-flight/failed attempts are refused,
and a late reply is unusable. A disconnected/cancelled HTTP request retains its
slot until the underlying worker call actually completes. Adoption requires
all slots drained and acknowledges every installed OID. Number rollback,
same-number rebind, duplicate published checkpoint numbers, mismatched lineage
and partial adoption are refused.

A worker or transport error raises `ProofWorkerUnavailable`; the existing
scheduler treats it as infrastructure capacity failure and faults the plane.
It never becomes a false proof result, miner rejection or failure debt.
Readiness becomes false and admission health degrades. Health is rechecked
before window opening and asynchronously during readiness polling (a cached
success expires after ten seconds). A process session change or GPU/runtime
replacement requires controller restart and requalification, rather than
silent acceptance of replacement hardware.

A health HTTP 200 before adoption is inventory only: `checkpoint:null` is not
ready. Operational readiness requires the exact checkpoint and the same OID in
all `slots[].revision` values, plus the controller's existing readiness gates.

## Capacity evidence

Keep the existing schema-3 capacity thresholds, physical UUID accounting,
profile/model/runtime/checkpoint pins, full M-rollout proofs, twenty samples per
GPU/environment and representative completion lengths. Remote mode additionally
requires measurements through the complete validator and mTLS path, including
serialization, network, local gates and all rollouts. Raw kernel timings cannot
be relabelled as these samples.

Each source measurement must contain the same `remote_proof` object:

```json
{"protocol":"reliquary.remote-proof/v1","worker_id":"<worker-id>","transport_sha256":"<64-hex>","measurement_scope":"validator-end-to-end-mtls"}
```

The worker's `/v1/health` returns `transport_sha256`; both hosts must have the
same adapter bytes. Run the existing `scripts/qualify_proof_capacity.py` with
its normal evidence flags and `--remote-proof-worker-id <worker-id>`. It requires
the matching marker in every source row and writes it into the pinned manifest.
An old local manifest, another worker, changed transport implementation or
unmeasured physical GPU is refused in remote mode. Obtain fresh measurements
after a transport change; the historical faster-runtime option cannot bypass
this extra transport binding.
