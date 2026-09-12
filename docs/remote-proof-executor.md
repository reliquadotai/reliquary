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
| `RELIQUARY_PROOF_PIPELINE_DEPTH` | Proof batches kept in flight per worker slot, 1 to 4 (default 1) |
| `RELIQUARY_PROOF_CAPACITY_MANIFEST` / `RELIQUARY_PROOF_CAPACITY_MANIFEST_SHA256` | Pinned capacity evidence |

The existing `RELIQUARY_PROOF_WORKER_REQUEST_TIMEOUT_SECONDS` and
`RELIQUARY_PROOF_WORKER_RELOAD_TIMEOUT_SECONDS` also bound network proof and
adoption calls. Allow enough reload time for all configured replicas to install.
Transport deadlines are absolute; keep the hosts' clocks synchronized.

`RELIQUARY_PROOF_PIPELINE_DEPTH` above 1 gives the scheduler that many dispatch
lanes per worker slot, so uploads and downloads overlap GPU work instead of
alternating with it. The worker still runs one proof per slot at a time and
queues the rest, up to `MAX_PROOF_PIPELINE_DEPTH` (4); a request beyond that is
refused, and a queued proof gives up at its own deadline. Lanes exist only on
the controller: the wire, receipts and capacity evidence carry the physical slot.

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
{"protocol":"reliquary.remote-proof/v1","worker_id":"<worker-id>","transport_sha256":"<64-hex>","pipeline_depth":2,"measurement_scope":"validator-end-to-end-mtls"}
```

The worker's `/v1/health` returns `transport_sha256`; both hosts must have the
same adapter bytes. Run the existing `scripts/qualify_proof_capacity.py` with
its normal evidence flags and `--remote-proof-worker-id <worker-id>`. It requires
the matching marker in every source row and writes it into the pinned manifest.
An old local manifest, another worker, changed transport implementation or
unmeasured pipeline depth or physical GPU is refused in remote mode. Obtain fresh measurements
after a transport or pipeline-depth change; the historical faster-runtime option cannot bypass
this extra transport binding.

The manifest's checkpoint pin is checked against the startup activation
checkpoint. Ordinary checkpoint rotation within that running validator keeps
the existing capacity policy and instead requires the normal exact N/OID
adoption acknowledgement from every worker. It does not run the capacity
qualifier on each training publication. A later startup still undergoes the
existing startup pin checks; this collector changes none of those rules.

### Producing real measurements

The optional `RELIQUARY_PRIVATE_REMOTE_PROOF_MEASUREMENTS=/absolute/new.jsonl`
records complete groups at `ValidationService._execute_scheduled_proof` in
authoritative remote mode. It does not enable remote mode or bypass any
capacity/readiness check. Use a new private path for each process; the file is
created exclusively with mode 0600. Local/shadow pools cannot emit this marker.
The timer surrounds `execute(model)`: every rollout's JSON/mTLS call and all
the batcher's proof-dependent gates. Scheduler queue wait, cheap admission,
grading and miner generation are outside this per-proof interval.

Each row retains successes, rejections and infrastructure errors. A passing
row requires a complete `ValidSubmission` and one authenticated network receipt
per rollout, all bound to the scheduled slot/environment/window/checkpoint.
Completion lengths come from authenticated sparse-output coverage. The private
file contains receipt hashes/identities and timings, not tokens, prompts,
signatures, credentials or kernel diagnostics. A failed/partial group cannot
become a passing capacity sample. The qualifier rejects a file containing
nonpassing or unrepresentative samples; preserve such evidence and diagnose it
rather than relabelling timings.

To bootstrap capacity before a public validator is allowed to start, use a
**dedicated isolated proof worker** and the same immutable image on a CPU
benchmark process. Run `scripts/measure_remote_proof_capacity.py`; it never
constructs `ValidationService.run`, a public HTTP listener, wallet, chain
client, trainer, publisher or storage writer. It adopts the exact checkpoint,
prepares candidates through the actual batcher, and submits complete proof
payloads to `GlobalProofScheduler` with the same service execution method. No
prior capacity manifest is needed in this isolated command. Production startup
continues to require its pinned manifest, and the benchmark never opens it.

The input is private JSONL with exactly these fields per group:

```text
{"environment":"<active environment>","randomness":"<window randomness>",
 "request":<complete BatchSubmissionRequest JSON including valid signatures>}
```

Use genuine signed groups generated for the target profile and N/OID. The
envelope, per-rollout signatures, prompt/token binding, reward grading and all
proof gates are rechecked. This offline corpus is not evidence of live HTTP
arrival timing, admission fairness or metagraph eligibility. No historical
cooldown or economic state is copied into the benchmark; each group is isolated
and never selected for payment/training. Identical input groups are refused.
The existing Code grader must be available; no unsandboxed grading fallback is
introduced. The ordinary prompt range applies to the recorded randomness.

With the remote TLS/profile/run environment configured as above, run in the
CPU benchmark image (paths and identities below are placeholders):

```sh
python scripts/measure_remote_proof_capacity.py /private/signed-groups.jsonl \
  --output /private/new-proof-measurements.jsonl \
  --hf-repo-id PUBLIC_CHECKPOINT_REPO --checkpoint-n CHECKPOINT_N \
  --checkpoint-revision FULL_CHECKPOINT_OID --timeout-seconds 7200
```

Retain the JSON report printed by this command; it binds corpus/sample hashes,
checkpoint identity, software and runtime. Run the existing qualifier against
that exact JSONL and matching report/worker identities, with
`--remote-proof-worker-id`. Its default schema-3 gate requires at least 20 passed
groups per physical GPU per active environment, **every rollout** at least 90%
of that environment's token cap, and the configured headroom. Supply enough
independent groups to cover every device; the scheduler uses its ordinary
dispatch policy. Tiny CPU/fake-GPU tests of this producer are implementation
tests only and cannot satisfy the actual target GPU qualification.

Transport hashing now also covers the full batcher/service path and collector.
Build and measure the final merged image on both hosts; a pre-merge or earlier
collector build cannot reuse the same remote capacity marker.


### Combined V1 qualification (schema 4)

Natural EOS is not a resource ceiling: a correct Logic answer may be short.
The explicit combined path keeps **20 genuinely passing M16 groups per active
ENV and physical GPU**, and separately measures 20 full-envelope groups per
configured slot/ENV. Every configured slot runs concurrently. Synthetic inputs
contain 24,576 prompt tokens and 8,192 policy tokens; they exercise the real
32,768-token forward, full forced-seed CDF checks, 32 GRAIL challenges, sparse
outputs, and authenticated mTLS. Their invalid GRAIL verdicts remain invalid.
They never become submissions, rewards, training data or passing proofs.

The CPU supplement executes native post-proof helpers on the actual responses,
then explicitly labeled CPU-only fixtures cover full scans that random invalid
proofs may reject early. Native forensic JSONL serializers/appends run in
throwaway files on the controller state filesystem and their cost is included.
Code counterfactual sandbox regrading is excluded: set
`RELIQUARY_CODE_SEMANTIC_COUNTERFACTUAL_ENABLED=0` in both benchmark and final
controller. This disables extra forensic regrading only, preserving authenticity
checks. All other measured flags (including GPU utility entropy) remain bound.

Run the following in the exact controller image with its actual bounded CPU,
memory, state-filesystem mount, active profile/run and client PKI. Output paths
must be private, new, and on that state filesystem. The same immutable image
must run the isolated GPU worker. Use genuine native corpus generation; the
infra generator's `--natural-completions` preserves normal EOS without changing
sampling, signatures or caps.

```sh
python scripts/measure_remote_proof_capacity.py /private/corpus.jsonl \
  --combined-natural --output /state/private/natural.jsonl \
  --hf-repo-id CHECKPOINT_REPO --checkpoint-n CHECKPOINT_N \
  --checkpoint-revision FULL_CHECKPOINT_OID --timeout-seconds 86400
python scripts/measure_remote_proof_stress.py \
  --output /state/private/stress.jsonl \
  --hf-repo-id CHECKPOINT_REPO --checkpoint-n CHECKPOINT_N \
  --checkpoint-revision FULL_CHECKPOINT_OID --timeout-seconds 86400
python scripts/qualify_proof_capacity.py /state/private/natural.jsonl \
  --natural-corpus /private/corpus.jsonl --stress-samples /state/private/stress.jsonl \
  --maximum-context-tokens 32768 --output /state/private/capacity.json \
  --software-revision FULL_IMAGE_COMMIT --checkpoint-revision FULL_CHECKPOINT_OID \
  --runtime-fingerprint-hash OBSERVED_RUNTIME_SHA256 --hardware-class OBSERVED_GPU_NAME \
  --benchmark-device-count PHYSICAL_GPU_COUNT --measured-at UTC_TIMESTAMP \
  --remote-proof-worker-id WORKER_ID
```

Keep the original corpus and both sibling `.attempts.jsonl` and
`.proof-attempts.jsonl` ledgers. Every original input, admission duration,
normal out-of-zone rejection and proof decision remains hash-bound to the
selected output. Infrastructure errors/timeouts cannot produce final passing
output. Canonical GRAIL, logprob or seed failures **block qualification**;
there is no numerical-rejection override.

Per ENV/GPU, the bound is **maximum measured full-envelope duration plus maximum
measured authentic post-admission group duration**. This intentionally counts
some work twice instead of mistaking a short-response p95 for a full-cap cost.
At least 20% headroom is required. Natural p95s and lengths remain separately
reported as actually measured. This is a conservative measured workload,
not a mathematical guarantee for every token sequence.

In the default **strict** service mode, the shared budget helper requires
capacity for **512 admitted groups per ENV** by default (failures consume this monotone
budget), targeting 256 proven groups in 16 picks; their backstop is
`RELIQUARY_FILL_CLOSED_MAX_SECONDS` (default 1800). The legacy seal-time path
keeps 34 groups and `RELIQUARY_MAX_PROOF_WALL_SECONDS`. Raising that legacy knob
does not extend a fill window. Choose the fill backstop/fleet only from observed
bounds; the qualifier refuses insufficient capacity. A longer backstop does
not force normal windows to wait once all picks have completed.

This budget covers **post-admission proof execution**. Admission/reward-grader
latencies are recorded separately; public HTTP concurrency, late supply,
trainer pacing, storage/recovery and live end-to-end adoption remain separate
release checks. A capacity manifest is not a go-live acknowledgement.


### Bounded online fill service

For a fleet that cannot guarantee every admitted worst-case group within the
window, explicitly set `RELIQUARY_FILL_CLOSED_PROOF_SERVICE_MODE=bounded` in the
qualification and controller configuration. This requires schema 4 and
`--bounded-fill-service` on the qualifier; an existing strict manifest cannot
activate it. All authentic proofs, numerical rejection checks, full-envelope
samples, exact identities and resource/topology bindings still apply.

`RELIQUARY_FILL_CLOSED_PROOF_DRAIN_SECONDS` reserves time before the hard window
backstop (default 240 seconds). Qualification requires the **maximum combined
group bound across every ENV/GPU, divided by (1 - headroom)** to fit this margin.
Choose the margin from actual measurements; its default is not a qualification.
The advertised precommit horizon leaves both the upload grace and this margin.
The scheduler checks its dispatch cutoff under the same lock that starts work,
so a queued group cannot start at or after the cutoff. Queued unproved groups
receive `NOT_NEEDED / dispatch_budget_exhausted`; their concurrency reservations
release exactly once, without refunding the monotone admission budget. Active
proofs retain their genuine verdicts and can supply a final balanced pick.
A proof still active at the hard deadline faults the plane normally.

The 512 admission ceiling, M16 proofs, 256-group target, 16 balanced picks,
trainer pacing, authenticity and payment rules are unchanged. A full window
closes once its picks and active proofs settle. A time-limited window uses the
existing partial-seal path: only complete, proven balanced picks enter training,
unpicked groups burn, and remaining journal slots become explicit tombstones.
No checkpoint publication or trainer acknowledgement is invented.

The report explicitly states `all_admitted_proofs_qualified: false` and
`guaranteed_picks_per_window: 0`. It qualifies bounded online service and active
proof draining on the measured fleet; it does **not** promise all 512 proofs,
16 picks, HTTP/grader throughput, or useful training throughput in adversarial
full-context traffic. Genuine natural-supply end-to-end training/adoption and
operational latency remain separate release gates. The strict schema-3 and
schema-4 fleet-throughput checks remain available and unchanged.
