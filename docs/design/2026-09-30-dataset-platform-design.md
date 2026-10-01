# Dataset platform: on-demand verified generation (design)

2026-09-30. Status: approved by the user for implementation on branches, reviewed by grizz before merge.

## Goal

A customer (an AI lab or another Bittensor subnet) orders a verified synthetic dataset through an HTTP API.
They choose a teacher model from our qualified list, and a source: one of our catalog environments or a
public Hugging Face dataset (the HF source is a later phase). They pay from a prepaid balance and receive the
dataset. Miners on SN81 generate the data. Verification runs on GPUs rented from Lium, one fleet per model,
on demand. There is no frontend in this phase: the API is the product.

## Components and where each lives

| Component | Repo | Runs on |
|---|---|---|
| Customer API, orders, quotes, balances, fleet reconciler, Lium adapter | `reliquary-platform` (Cloudflare Worker + D1 + R2) | Cloudflare |
| Subnet admin service: the only holder of registry/R2 write authority the platform can reach | `reliquary` (`reliquary admin serve`) | trusted host (ctrl-01 or the corpus control host) |
| Corpus control: routes, ledger, verdicts, settlement, hot job set | `reliquary` (corpus validator) | trusted host, today the corpus box |
| Corpus audit executor: TOPLOC recompute only, no secrets | `reliquary` (`reliquary corpus audit-executor`) | Lium pods rented by the fleet |

Repos we do NOT build on: reliquary-forge, reliquary-protocol, reliquary-ledger, reliquary-miner (historical).
The platform's Angular frontend is left untouched.

## Trust rules

1. A rented pod holds no wallet, no R2 key, no HF token, no Lium key, no admin secret. It holds one
   per-pod executor token, issued when the pod is created and revoked when it is destroyed.
2. An executor never writes a verdict. It returns scores; the control decides and writes.
3. A random fraction of executor results (`RECHECK_FRACTION`, default 0.05) is recomputed by a trusted
   verifier (the control's local GPU when it has one). A divergence beyond the proof tolerance quarantines
   the executor (its token is revoked). Every batch it scored since its last passed recheck is re-queued.
4. The executor pulls work over HTTPS it initiates. The pod needs no inbound port.
5. The platform never holds subnet R2 keys. It reaches the subnet only through the admin service, with
   HMAC-signed requests.

## Subnet side (reliquary)

### R1 Hot job set
The corpus validator re-reads the task registry every `JOB_REFRESH_SECONDS` (60).

**A new active corpus-generation entry** is wired without a restart when its contract's model, revision and
toploc proof equal the running process's. Wiring means:
- ledger migration;
- renderer;
- auditor;
- settler;
- router mount under `/corpus/jobs/{job_id}/...`.

An entry for another model is ignored, with one log line per entry. An entry whose renderer cannot be built
is refused, with a log line; it never crashes the process.

**A retired entry** (status retired):
- stops admission: next, submit and skip answer `410 {"detail": "job_retired"}`;
- its auditor keeps draining until no submission is unaudited;
- its settler keeps running until the job is drained;
- then it is unwired.

Legacy `/corpus/...` routes keep answering for the first job wired at boot.

### R2 Job status route
`GET /corpus/jobs/{job_id}/status` is public JSON with no hotkeys. Fields:
- `state`: `open | full | retired | drained`;
- `prompts_total`, `prompts_full`;
- `submissions_accepted`, `audited`, `passed`;
- `verified_tokens`;
- `settled`;
- `accepted_last_hour`.

It is computed from in-memory state and cached `STATUS_CACHE_SECONDS` (30). It never lists R2 per request.

### R3 Remote audit executor (pull model)
Control routes, authenticated by `Authorization: Bearer <executor token>`:
- `POST /corpus/internal/audit/claim`: `{executor_id, model_id, model_revision}` → a batch lease, or 204
  when there is no work;
- `POST /corpus/internal/audit/{lease}/result`: `{scores}`;
- `POST /corpus/internal/audit/heartbeat`.

A batch carries exactly what `CorpusAuditor` feeds the model today: token ids, positions and the committed
proofs. The executor returns the per-item comparison the auditor needs to decide pass or fail. The decision
function stays in the control.

Rules:
- A lease expires after `AUDIT_LEASE_SECONDS`; its batch is re-queued.
- The control keeps auditing locally when it has a GPU and no executor is connected, so today's single-box
  deployment is unchanged.
- Executor tokens are verified against the executor registry in R2, `reliquary/corpus/executors/{id}.json`:
  `{token_sha256, model_id, model_revision, expires_at, status}`. The control re-reads it every 30 s and
  refuses unknown, expired, revoked or wrong-model tokens.
- The executor CLI is `reliquary corpus audit-executor --control URL --executor-id ID`, with the token in
  `RELIQUARY_EXECUTOR_TOKEN`. It loads the model from the public HF repo at the pinned revision and needs
  no other secret.

### R4 Subnet admin service
`reliquary admin serve --host --port` exposes FastAPI routes. Every request carries `X-Reliquary-Timestamp`
(±300 s), a required `X-Reliquary-Nonce` (16-64 hex characters, fresh per request) and `X-Reliquary-Signature`
(hex HMAC-SHA256 of `timestamp\nnonce\nMETHOD\npath\nsha256hex(body)`), keyed by `RELIQUARY_ADMIN_SECRET`.
A reused nonce is refused as a replay, using a nonce cache for the timestamp window; two identical requests in
one second pass under two nonces.

| Route | Action |
|---|---|
| `POST /admin/v1/jobs` | Create a corpus job and its task entry. Body: model, env, `prompt_start`, `prompt_count`, `samples_per_prompt`, `max_new_tokens`, `thinking`, cap. Idempotent on `job_id`. |
| `POST /admin/v1/tasks/{task_id}/cap` | Set the cap |
| `POST /admin/v1/tasks/{task_id}/retire` | Retire the task |
| `GET /admin/v1/jobs/{job_id}/status` | Proxies R2's status fields: the same counts as `jobs status`, plus the job manifest |
| `POST /admin/v1/executors` | Register `{executor_id, token_sha256, model_id, model_revision, expires_at}` |
| `DELETE /admin/v1/executors/{id}` | Revoke |
| `GET /admin/v1/executors/{id}` | Last heartbeat as seen by the control (written by the control to the same R2 object) |
| `POST /admin/v1/jobs/{job_id}/deliveries` | Export v2: passing rows, grader filter applied as annotation, Parquet shards (≤ 500 MB), `manifest.json` with sha256 per shard, `report.json`. Written under `deliveries/{delivery_id}/` in the platform bucket, using credentials scoped to that bucket and held by the admin host only. Returns the object keys. |

The server enforces two limits, refusing any write that breaks them:
- the sum of active caps stays ≤ 1.0;
- the corpus caps together stay ≤ `RELIQUARY_ADMIN_POOL_MAX`.

## Platform side (reliquary-platform)

### P1 Per-executor credentials
The shared `EXECUTOR_TOKEN` is replaced for fleet pods:
- A pod's token is 32 random bytes, stored only hashed.
- It is registered with the subnet admin service (R4) when the pod is created and revoked when the pod is
  destroyed.
- The existing `EXECUTOR_TOKEN` path is kept for grizz's training executor and not changed.

### P2 Provider interface and Lium adapter
```
interface GpuProvider {
  offers(spec): Promise<Offer[]>
  create(spec, podName, image, env): Promise<Pod>
  get(id): Promise<Pod>
  list(prefix): Promise<Pod[]>
  destroy(id): Promise<void>
}
```

`LiumProvider` calls the Lium REST API (`https://lium.io/api`, header `X-API-KEY`):
- `POST /templates`: a one-time private template carrying the image and env;
- `POST /executors/rent-by-spec`;
- `GET /pods/{id}`, `GET /pods`;
- `DELETE /pods/{id}`.

A rent is posted once and never retried. On a lost response, the pod is looked up by its unique name. The
Lium key is a Worker secret (`LIUM_API_KEY`). A `FakeProvider` exists for tests.

### P3 Fleet reconciler
A Cron Trigger runs every minute. Tables:
- `fleet_targets`: `model_id, revision, wanted, max`;
- `fleet_pods`: `id, provider_id, model, status, created_at, ready_at, last_seen_at, hourly_price, order_id`;
- `fleet_spend`.

Each run:
1. `wanted` per model is derived from active orders: v1 is one executor per model with at least one running
   order, capped by `max`.
2. Missing pods are rented. A pod is ready when the control reports its heartbeat, read through R4 GET
   executor. It is replaced if it is not ready within `POD_READY_TIMEOUT` (30 min) or not seen for 10 min.
3. A pod is destroyed after `IDLE_GRACE_SECONDS` (3600) without a running order for its model.
4. Orphans are destroyed: pods named with our prefix that are not in `fleet_pods`.
5. No new rent happens once today's spend (`hourly_price × hours`) reaches `FLEET_DAILY_SPEND_MAX_USD`.

### P4 Orders API
Auth is by API key with a new write scope `orders:write`; existing keys stay read-only. Routes under
`/api/v1`:
- `GET /generation/catalog`: qualified models and sources, from a config table;
- `POST /dataset-orders/quotes`: `{model, source, prompt_count, samples_per_prompt, max_new_tokens, thinking}`
  → `{quote_id, price_usd, est_tokens, eta_hours, expires_at}`, valid 24 h;
- `POST /dataset-orders`: `{quote_id}`, with an `Idempotency-Key` header. It reserves the price from the
  balance (402 when short) and creates the order;
- `GET /dataset-orders`, `GET /dataset-orders/{id}`: state, progress (from R2 status), deliveries;
- `POST /dataset-orders/{id}/cancel`: retires the job; the unconsumed share is refunded pro rata of prompts
  not full;
- `GET /dataset-orders/{id}/deliveries/{delivery_id}/files/{name}`: private download with byte ranges,
  reusing the existing artifact code.

The order state machine, advanced by the cron (every transition idempotent):
```
reserved → provisioning (R4 create job, cap) → running → draining (job full or cancelled → R4 retire)
         → exporting (R4 deliveries) → delivered
any → failed (with reason, refund of unconsumed)
```

Quote formula (config values):
```
est_tokens = prompts × samples × avg_tokens(model, source)
price = est_tokens / 1e6 × price_per_mtok(model) + gpu_hours × gpu_hour_price × (1 + margin)
eta = est_tokens / fleet_tokens_per_hour(model)
```

### P5 Balances
`ledger_entries(user_id, amount_micro_usd, kind, ref, created_at)` is append-only. The balance is the sum.
Entry kinds: `topup`, `reserve`, `refund`, `release`. `POST /api/internal/credits` (operator token) tops up.
The TAO/USDC deposit watcher is a later phase.

## Out of scope for this phase
- HF dataset source;
- webhooks;
- TAO deposit watcher;
- CVM attestation;
- several models per control process;
- LLM judge;
- decontamination.

## Acceptance
- Unit and contract tests in each repo cover every rule above. In particular:
  - an executor with a wrong, revoked or expired token is refused;
  - a lying executor is quarantined and its batches re-queued;
  - a hot-added job serves and a retired job drains without a restart;
  - the admin HMAC rejects a replay and a stale timestamp;
  - the cap-sum limits hold;
  - the reconciler never double-rents on a lost response and destroys orphans;
  - an order goes from reserved to delivered against a fake provider and a fake admin.
- CI green on both branches. No deploy and no real Lium rent before the user's go.
