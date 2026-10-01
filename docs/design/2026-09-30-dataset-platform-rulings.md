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
Ruling: the settler's archive guard (RELIQUARY_TASK_ID) also admits the task ids this process wired after boot.
Ruling: a scoped submit route `POST /corpus/jobs/{job_id}/submit` joins the scoped reads, so every admission path of a job lives under its prefix.

## R2 Job status route

Ruling: `audited` counts submissions with a standing verdict (audited by the model or passed unaudited by sampling), `passed` the passing verdicts, `verified_tokens` their token counts — so `submissions_accepted - audited` is what is still waiting.
Ruling: `submissions_accepted` and `prompts_full` come from the job's ledger object (each accepted submission fills one slot), read with one GET at most once per `STATUS_CACHE_SECONDS` and reused per ETag — a read, never a listing.
Ruling: verdict counts are seeded once per process by one background listing when a job is wired, then kept from the auditor's own writes; `accepted_last_hour` counts from the process start (it can only undercount during the first hour).
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
