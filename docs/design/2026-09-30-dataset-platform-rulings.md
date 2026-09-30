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
