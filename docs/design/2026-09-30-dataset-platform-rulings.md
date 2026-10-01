# Dataset platform: rulings taken while implementing the subnet side

Decisions on points `2026-09-30-dataset-platform-design.md` leaves open, one per line.

## R1 Hot job set

Ruling: the hot job set is opt-in (`RELIQUARY_CORPUS_HOT_JOBS=1`); without it the validator serves exactly the jobs it booted with — a hot validator starts serving and paying any active corpus entry on its model, which today's deployment must not do silently on an image update.
Ruling: a hot-added job is served only when its OWN carried contract declares its prompt source exactly as the process contract (RELIQUARY_TASK_CONTRACT) does, and the protocol gate of that source agrees; otherwise it is refused with one log line — the environment renders its rows through the process-level `ACTIVE_PROTOCOL_PROFILE` (`render_active_prompt`), so a job whose contract renders differently would fail fidelity on every submission. Its renderer and prompt rows are still built against its own profile, as a boot job among several is.
Ruling: a contract-less entry is refused for hot-adding — nothing can be checked against the running contract.
Ruling: an entry that disappears from the registry is treated as retired — the registry only ever retires, so absence is an operator repair, and draining is the safe answer.
Ruling: a refusal or "other model" decision is remembered per task id for the life of the process (logged once); a manifest read that fails transiently is retried on the next refresh.
Ruling: a drained job is never wired again by the same process, even if its entry reads active again — its settlement state and ledger say it is finished.
Ruling: `/corpus/jobs` lists open jobs only; a retired job's next, skip and submit answer 410 `job_retired` on both the legacy and scoped paths, and its job/cursor reads keep answering while it drains.
Ruling: the legacy `/corpus/...` paths keep the first job wired at boot for the life of the process; once that job is drained they answer 410 `job_retired`.
Ruling: a cap changed in the registry (`set-cap`) reaches the running settler at the next refresh.
Ruling: with the hot set on, the auditors share a GPU lock even when one job is served, since more may join.
Ruling: the settler's archive guard (RELIQUARY_TASK_ID) also admits the task ids this process wired after boot, and only those (review M3).
Ruling: a scoped submit route `POST /corpus/jobs/{job_id}/submit` joins the scoped reads, so every admission path of a job lives under its prefix.

## R2 Job status route

Ruling: `audited` counts submissions with a standing verdict (audited by the model or passed unaudited by sampling), `passed` the passing verdicts, `verified_tokens` their token counts — so `submissions_accepted - audited` is what is still waiting.
Ruling: `submissions_accepted` and `prompts_full` come from the job's ledger object (each accepted submission fills one slot), one GET (the wired manifest is reused) at most once per `STATUS_CACHE_SECONDS`, single-flight per job — a read, never a listing.
Ruling (review I2, superseding the boot seed): verdict counts are never seeded by listing. The settler accumulates `totals` (verdicts, passed, verified tokens) into the settlement object in the same compare-and-swap that settles them (carried in the pending step, so a repeated finish adds nothing twice); verdicts written since and not yet settled are held in memory from the auditor's own writes and dropped once settled. With the flags off, boot does no read beyond what the auditor and settler already do. `counts_complete` is false until the settler has read its totals, and for a job settled before totals were kept; verdicts written before a restart count once settled; `accepted_last_hour` counts from the process start.
Ruling: `settled` is the settler's count as of its last settlement read (0 before its first).
Ruling: a failed recompute serves the last status; with none cached it answers 503 `corpus_status_unavailable`; an unknown job 404; a drained job keeps its final status for the life of the process.

## R4 Subnet admin service

Ruling (contract amendment from the platform side): the signed string is `timestamp\nnonce\nMETHOD\npath\nsha256hex(body)` with a required `X-Reliquary-Nonce` of 16-64 hex characters; the replay cache keys on the nonce (case-folded) inside the ±300 s window — with the signature alone, two identical requests in one second were indistinguishable from a replay.
Ruling: a request carrying a query string is refused (400 `query_not_signed`): the signature covers the path only, and no route takes a query.
Ruling: `RELIQUARY_ADMIN_SECRET` must be at least 32 characters, `RELIQUARY_ADMIN_POOL_MAX` and `RELIQUARY_ADMIN_MODELS` are required; `admin serve` refuses to start without them.
Ruling: the body's `model` must be a key of the admin host's qualified-model file (`RELIQUARY_ADMIN_MODELS`: revision, architecture, checkpoint_sha256, eos_token_id per model) — the platform names a model, the subnet pins what that name means.
Ruling: `POST /admin/v1/jobs` declares single-turn catalog sources only, through the model's chat template (`thinking` picks `chat-template-thinking-v1` over `chat-template-v1`); `samples_per_prompt` is the manifest's `slots_per_prompt` with `n = 1`; everything else takes `jobs create`'s defaults (composed contract, prompt_order `free`, audit q 1.0); the service acknowledges `--fleet-knows-corpus-generation` itself.
Ruling: idempotent on `job_id`: the same manifest with its task present answers 200 `created: false` (with the task's current status and cap), the same manifest without its task completes the registry write, another manifest under the id answers 409.
Ruling: the cap limits are a registry guard re-applied on every compare-and-swap retry: active caps ≤ 1.0 and active corpus caps ≤ the pool; a write that does not raise a total is never refused by them (so a cap can always be lowered). The registry's own rule (every cap, retired included, ≤ 1.0) still applies underneath.
Ruling: retire without `retired_at` stamps the current drand round; retiring a retired task answers its stored stamp.
Ruling: `GET /admin/v1/jobs/{job_id}/status` lists the bucket (it is `jobs status`, for an operator-rate caller); the public route of R2 is the one that never lists.
Ruling: executor responses never carry `token_sha256`; a registration repeated identically answers 200, a different one under the same id 409; revoked and quarantined executors are never reactivated (a new pod gets a new id).
Ruling: a delivery runs beside the request: `POST .../deliveries` answers 202 `running` until the manifest exists, then 200 `done` with the keys; it is idempotent on `delivery_id` (default: the job id), and a failed run answers 500 once and is retried by the next POST.
Ruling: delivery rows carry no hotkey (`job_id, submission_id, prompt_index, completion_index, prompt, completion, completion_tokens, accepted, score`); zstd Parquet; a shard is closed before its raw bytes could pass 500 MB less 1 MB of footer room, a row group at 2048 rows or 64 MB, and 64 verdicts are read per window with 16 reads in flight.
Ruling: the grader annotates only when the job declares a filter and its source can grade a single completion; otherwise `report.json` says why (`filter.applied: false`).
Ruling: pyarrow is already a core dependency (`pyarrow>=14.0.0`), so no extra was added.

## R3 Remote audit executor

Ruling: remote auditing is opt-in (`RELIQUARY_CORPUS_REMOTE_AUDIT=1`, recheck share `RELIQUARY_CORPUS_RECHECK_FRACTION`, default 0.05); without it the `/corpus/internal/audit/...` routes are not mounted and nothing changes. With it and no executor connected, this GPU audits as before.
Ruling (review C1/I3, superseding the provisional/vouching design): each batch an executor returns is drawn for a recheck on its own, at `RECHECK_FRACTION`, from the OS's randomness (`secrets.SystemRandom`). A drawn batch is held and resolved from the control's own scores; an undrawn one is resolved from the executor's at once. A passed recheck vouches for its own batch only, never for earlier ones, so concurrent rechecks cannot vouch a lie.
Ruling (review C1a): a recheck agrees only if every item reaches the same pass/fail decision AND every chunk measure is within a cross-hardware drift tolerance: |Δexp_mismatches| ≤ `RECHECK_EXP_DRIFT` (2, env `RELIQUARY_CORPUS_RECHECK_EXP_DRIFT`) and |Δmantissa mean|, |Δmantissa median| ≤ `RECHECK_MANT_DRIFT_FRACTION` (0.1, env `RELIQUARY_CORPUS_RECHECK_MANT_DRIFT_FRACTION`) × that measure's proof threshold — never "within one threshold", which let a lie under-report its way across the line. The default is a guess to be measured against the Lium card types.
Ruling (review C1c): a disagreement quarantines the executor (status `quarantined`, refused locally at once, the registry write retried every sweep until it lands), takes back its leases, and makes every auditor re-audit on its own GPU each pass written from that executor's scores (verdicts carry `scored_by`) that is unsettled, or settled with its verdict inside the job's hold window. A re-audit failure, confirmed by a second local audit as §7.2 asks, goes down the same path as a failed audit: `after_confirmed_failure` on the miner's state (suspect, then ban), and a create-only `voided/{submission_id}.json` marker the settler honours by settling the id without paying it.
Ruling (residual risk, accepted): an undrawn lie is written at once, so a single lie escapes the recheck with probability 1 − p (0.95 at the default) and is paid if it settles before the executor is caught; it is clawed back by penalty, not by money, when settled outside the hold window. The re-audit list lives in this process's memory: a control restart forgets which passes an executor scored (the verdicts' `scored_by` still name it, for an operator re-audit).
Ruling (review I4): an executor holds at most `MAX_LEASES_PER_EXECUTOR` (2) leases; `LEASE_EXPIRY_STRIKES` (3) leases expiring in a row quarantine it; the lease life is `RELIQUARY_CORPUS_AUDIT_LEASE_SECONDS` (default 300) bounded to [30, 600]; a batch unclaimed for `QUEUE_WAIT_SECONDS` (60) is scored locally even while an executor is connected.
Ruling: a failing remote verdict is never final: the §7.2 confirmation of a failure always runs on the control's GPU, so an executor alone can never fail (or get banned) an honest miner; a lying executor can only try to pass work, which the rechecks catch.
Ruling: a result that does not fit its lease answers 422 and the batch is re-queued (no quarantine: only a recheck proves a lie); an item an executor reports `error` re-queues its batch; after 2 remote attempts the control scores the batch itself.
Ruling: the result body is `{scores}` only (the token names the executor and the lease must be its own); the heartbeat answers the registered `model_id`/`model_revision`, so `audit-executor` may omit them.
Ruling: a lease holds at most 64 items and 262,144 tokens; an executor is connected while heard from within 90 s; the control writes its last contact into the executor's registry object at most every 30 s.
Ruling: the control still loads the model in this phase: it is the trusted verifier for rechecks and the vocabulary check; a GPU-less control is left for later.

## Review fix pass (R1)

Ruling (review I1): submit and skip hold a per-job in-flight count while they run; a retired job is unwired only in a later refresh than the one that retired it, with no admission in flight, and only if it is still drained and still has nothing in flight after the drain check — the retired gate admits nothing new, so no record can appear after that.
Ruling (review I8): only deterministic wiring refusals (a `ValueError`: renderer, prompt source, contract) are remembered; any other wiring failure (store, transport) is retried at every refresh and logged at ERROR from the 5th attempt.

## Review fix pass (R4)

Ruling (review I5, M6, M7): the manifest is written create-only and never deleted (a racing call's task may name it; an orphan manifest is harmless and a later identical create completes it). A create that loses the manifest write to identical bytes proceeds; one that loses the registry write to an identical concurrent create answers that task as the idempotent 200; a second task for a job already declared answers 409.
Ruling (review I6): cap and retire act only on corpus-generation tasks; `default` and every RL task answer 409 `not_a_corpus_task`. 
Ruling (review I7): both cap limits count a retired corpus task's cap until its job is drained (checked with the stored counts at each guarded write, and remembered once drained, since drained is final).
Ruling (review M4, M5): a signature that is not 64 hex characters is `bad_signature` (401), never a 500; a body over 1 MB answers 413 (and a chunked one without a length 411) before any of it is read or parsed.
Ruling (review M8): a delivery of a job that is not drained answers 409 `job_not_drained`: a manifest makes a delivery final, so it must not freeze a partial dataset.
Ruling (review M9): a retire that loses its compare-and-swap to another retire answers the stored stamp.

## Review minors

Fixed: M1, M2, M3 (commit 9b0ee317), M4, M5 for the admin service, M6, M7, M8, M9 (3af5fbb8), M10 (a4357f46), M11 (a bad recheck fraction now exits 4 with the critical line), M13 (b069b053), M14 (one `token_sha256`, one `_entry_profile`).
Not fixed, M5 for `/corpus/internal/audit/*` — the corpus app has never had a body limit; nginx's `client_max_body_size` on `/corpus` bounds it in front of the validator, as it already does for submissions.
Not fixed, M12 — a halted auditor stops the process for hot jobs exactly as for boot jobs; retiring one job on halt is a behaviour change for the single-box deployment, left for when hot jobs are on in prod.
Ruling (review I6 residual): the admin scope is an id prefix, `RELIQUARY_ADMIN_TASK_PREFIX` (default `order-`, never empty). `POST /admin/v1/jobs` refuses a `job_id` or `task_id` outside it (422 `task_id_outside_admin_scope`); cap, retire, job status and deliveries refuse any task or job outside it (409 `outside_admin_scope`), checked before any read, so the platform can never touch an operator-declared prod job.
