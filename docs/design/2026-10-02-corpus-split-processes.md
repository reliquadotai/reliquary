# Corpus validator: front, judges and GPU in separate processes

Status: implemented on `feat/corpus-split-processes` (off by default).
Switch: `RELIQUARY_CORPUS_SPLIT=1` (+ `RELIQUARY_CORPUS_SPLIT_JUDGES`).

## 1. The measured problem (prod, 2026-10-02)

One Python process serves the four SFT jobs (code, math-omi, IF, logic):

- the miner routes (submit / skip / next / cursor / contract / status /
  miner status / `/corpus/tasks`), with the ledger compare-and-swap and the
  create-only record write of every accepted submission;
- four auditors (read the record, read `miners.json`, fetch the drand round,
  TOPLOC-audit on the GPU when drawn, write the verdict);
- four settlers.

On main before #298 the math judge produced ~1.7-2k verdicts/h against ~6.9k/h
arriving: ~83k math records unsettled, the oldest ~10 h, GPU ~17 % busy.
#298 (scheduled judge, parallel writes, settle from fed verdicts) and #299
(judge thread pools) made the judge fast *inside the same process*, and the
routes froze in waves: submit p50 4 s -> 50-65 s every ~9 min, 503 ledger
contention, while the CPU was 98 % idle, the network idle and raw R2 fast.
That is the GIL: the judge's JSON/gzip encode/decode of 0.5 MB records and its
bookkeeping across many coroutines and threads starve the one event loop the
routes run on. Thread pools (#299) only moved the work to other threads of the
same interpreter. Both were rolled back; prod runs main@dfa575d plus the
settle-from-fed-verdicts change.

No in-process arrangement fixes this: whatever thread decodes a record holds
the GIL the route needs. Judging must run in another interpreter.

## 2. Architecture

```
                     supervisor (PID 1 of the container, no model, no loop work)
                       |  spawns, watches, restarts each child alone
     +-----------------+------------------+-------------------------+
     |                 |                  |                         |
   FRONT            JUDGE 0            JUDGE 1 ...                GPU
 (HTTP :18090)   (math job)         (other group)          (model, once)
 routes, ledger, auditor+settler    auditor+settler        TOPLOC forward
 record writes,  unix socket        unix socket            unix socket
 ban check       judge-0.sock       judge-1.sock           gpu.sock
     |  POST /feed (ids, heartbeat)      ^                      ^
     +-----------------------------------+                      |
     |  GET /jobs/{job}/stats, /miners/{hk} (status proxy)      |
     +-----------------------------------+                      |
                    judges: POST /score (tokens, prompt_len, proofs) --+
```

All four roles run in one container (`reliquary validate`, one `docker run`),
as `multiprocessing` children started with `spawn`. They talk over HTTP on
unix sockets in `RELIQUARY_CORPUS_SPLIT_DIR` (default
`/tmp/reliquary-corpus-split`): no new port on `--network host`, nothing
reachable from outside the container.

### 2.1 Front (no GPU model)

Everything miners call, built by the same `build_corpus_jobs_app` as today:
admission, ledger turns, record writes, the registration gate, the ban check
(`miners.json` read through `MinerStates`, same cache), `/corpus/tasks`,
`/corpus/contract`, hot-job wiring. It loads the tokenizer (CPU) and reads the
vocabulary size the GPU process published (`gpu-info.json`); it never creates
a CUDA context (`CUDA_VISIBLE_DEVICES=""`).

For a job judged out of process, `on_accepted(submission_id)` appends the id to
that judge's bounded in-memory queue (`JudgeLink`) and returns: the route
never waits on a judge. A sender task posts the queue to the judge every
~50 ms, and an empty heartbeat every 2 s.

For a job judged in the front (rollout step 1), the auditor and settler run in
the front exactly as today, except that their forward goes to the GPU process.

### 2.2 Judges (no GPU model)

One process per group of jobs (`RELIQUARY_CORPUS_SPLIT_JUDGES`). Each runs, per
job, the unchanged `CorpusAuditor.run()` (main's #298 scheduler, its read /
write / drand concurrency and #299 thread pools) and the unchanged
`CorpusSettler` fed by it, plus the status books (`JobStats` unsettled part,
`MinerBook`). Its unix socket serves `/feed`, the status reads and `/health`.

The auditor's two seams:

- `scorer`: instead of `_judge_many` on a local model, `_prepare` (prompt
  encode, vocabulary and empty-completion checks, unchanged) runs on the judge's
  codec threads, and the `(tokens, prompt_len, proofs)` rows go to the GPU
  process. Chunk comparisons come back; `outcome_from_scores` and `_aggregate`
  (the decision) run in the judge, exactly as the remote-executor path already
  does. No `scored_by` is added: the GPU process is this validator's own card,
  trusted like the in-process forward (a failure is still re-audited, on the
  same card, before it counts).
- `arrivals_complete`: see 2.5.

### 2.3 GPU process

Loads the checkpoint once (bf16, `cuda`), publishes
`{vocab_size, model_id, model_revision}` to `gpu-info.json`, serves `POST /score`.
Requests from every judge (and from in-front auditors) enter one FIFO; the
worker merges consecutive requests (same `chunk_tokens`/`topk`) up to
`RELIQUARY_CORPUS_GPU_MERGE_TOKENS` (default 4 x `AUDIT_BATCH_TOKENS`) and runs
`score_sequences` once over them, on its single GPU thread: rows of several
judges share padded sub-batches. FIFO order is the fairness rule, as the
in-process `asyncio.Lock` was. A merged batch that raises is re-run request by
request, so one judge's bad record never fails another judge's rows; a request
that fails alone gets `{"error", "kind"}` and its auditor does what it does
in-process (retry each record alone, then count a validator-side error).

Back-pressure: requests beyond `RELIQUARY_CORPUS_GPU_QUEUE_TOKENS` (default
16 x `AUDIT_BATCH_TOKENS`) queued are answered 503 and the client retries after
a short sleep. In practice each judge has at most one request outstanding (a
pass audits, then re-audits failures, sequentially), so the bound is a guard.

### 2.4 Supervisor

`reliquary validate` with `RELIQUARY_CORPUS_SPLIT=1` resolves the corpus tasks
as today, then runs the preflight the single process runs before any model
load (manifests, order-job refusal, `multi_job_refusal`, checkpoint download,
fingerprint, `startup_refusal`), and spawns GPU, judges and front. It polls the
children every second; a child that exits is restarted alone after a backoff
(1 s doubling to 60 s, reset after 5 min up). SIGTERM/SIGINT terminate every
child (SIGTERM, 20 s grace, SIGKILL). Each child asks the kernel for SIGKILL on
the supervisor's death (`PR_SET_PDEATHSIG`), so no orphan survives a crashed
supervisor; Docker's restart policy then restarts the container as today.
Judges and the GPU process run at nice +5 (`RELIQUARY_CORPUS_SPLIT_NICE`): on
a saturated host the front is served first. Within one process the auditors
prepare and score one at a time (the in-process GPU lock, kept).

Refused with the split, for now: `set_weights` (prod runs `--no-set-weights`;
the RL validator's setter pays every task) and `RELIQUARY_CORPUS_REMOTE_AUDIT`
(executor leases would have to live in the GPU process; not needed on one box).

## 3. IPC contract

HTTP/1.1 + JSON over unix sockets, `httpx` client, `uvicorn` server.

GPU (`gpu.sock`):

| call | body | answer |
|---|---|---|
| `GET /info` | - | `{vocab_size, model_id, model_revision}` |
| `POST /score` | `{chunk_tokens, topk, items: [{tokens, prompt_len, proofs}]}` | `{scores: [[status, [[exp, mant_mean, mant_median], ...]], ...], forward_seconds, verify_seconds}`, or `{error, kind}`, or 503 when the queue is full |

`status` and chunk triples are exactly the remote executor's `ItemScore`
(`corpus_audit_protocol`), built by the same conversion; the judge rebuilds
`ChunkResult(int, float, float)` as `RemoteAuditDispatcher.result` does.
JSON round trip of a Python float is exact, so the decision sees the same
numbers the in-process path sees. Transport errors and 503 are retried forever
(1 s, doubling to 10 s): a GPU process down means audits wait, never fail.

Judge (`judge-<n>.sock`):

| call | body / answer |
|---|---|
| `POST /feed` | `{epoch, as_of, dropped, ids: {job_id: [submission_id, ...]}}` -> `{ok}` |
| `GET /jobs/{job}/stats` | `{unsettled: [verdicts, passed, tokens], settled_count, totals}` |
| `GET /jobs/{job}/miners/{hotkey}` | `{counts, share, thresholds, pending}` (starts the backfill) |
| `GET /health` | `{jobs, feed}` |

`epoch` is random per front process. `as_of` is the front's clock when the
batch was cut, set only on the post that empties its queue: every id accepted
before `as_of` has been delivered. `dropped` is set when the queue overflowed
(`RELIQUARY_CORPUS_FEED_MAX_IDS`, default 200k ids) until a post succeeds.

The front's status routes keep their public JSON: `CorpusJobSet` takes the
unsettled counts, `settled_count` and `totals` (job status) and the hotkey's
counts / share / thresholds / pending (miner status) from the judge, and the
rest (ledger state, `accepted_last_hour`, `miners.json` state, params, cap)
from where it took them before. A judge that does not answer within 2 s gives
the cached status, or 503 as any status read failure does today.

## 4. Payment semantics: what makes them unchanged

The auditor and settler code that decides and pays is the same code, run in
another process. What changes is only how three inputs reach it:

1. **The forward.** Same `_prepare`, same `score_sequences`, same
   `outcome_from_scores`/`_aggregate`, on the same card and checkpoint. Rows may
   share a padded sub-batch with another judge's rows; batch composition
   already varies per pass in-process, and the right-padded, masked forward
   is what TOPLOC's thresholds were measured against.
2. **Arrivals.** In-process, `enqueue` runs synchronously in `on_accepted`. The
   sibling rule leans on it: an unaudited pass of X happens at
   `X.received_at + hold + 420 s`, and every sibling received inside X's hold
   finished its record write (<= 405 s) and was enqueued before that, so a
   drawn sibling's failure reaches X first. Across processes a notification can
   be late or lost (front crash between the record write and the post; queue
   overflow; judge down). Guard, in the judge (`arrivals_covered`):
   - the feed vouches for an instant `covered`: every id the front accepted
     before it is enqueued in the judge. It is the newest `as_of` received,
     and None after every new `epoch` (including the first one a judge sees)
     and every `dropped`, until a full listing of the job
     (`_rescan_once(full=True)`: every pending record enqueued) completes;
   - X is passed unaudited only if `covered >= X.received_at + hold + 405 s`
     (slack minus a 15 s margin): every sibling that could catch X was
     accepted, so handed over, by then. `covered` is sampled before the pass
     collects X's siblings and again at the decision, and the lesser is used:
     a listing finishing mid-pass cannot vouch for siblings the pass never saw.

   An uncovered record waits (as for an unreadable record); audits, voids and
   failures go on. The check is per record, so a stalled front or judge loop
   only delays the newest records, never a backlog whose receipts are covered.
3. **Settlement.** Each judge process has its own `R2Archives` (other tasks'
   highest window cached 300 s). In-process the settlers of one process saw each
   other's archive writes at once; across processes after <= 300 s. The rules
   (`choose_window`, stall, one lone advance per RL window, two-phase CAS) are
   unchanged and none of them depends on that cache being fresh: a stale view
   can delay a jump to another task's window, never move the horizon faster.

Verdict writes stay create-only, `miners.json` and settlement stay CAS: a judge
restarted mid-pass cannot write a second verdict or pay a window twice. One job
is judged in exactly one place (the plan refuses a job in two groups; the
supervisor restarts a child only after it is reaped). The R2 layout (ledgers,
records, verdicts, voided, settlement, archives) is untouched; the weight-only
replay reads the same archives.

## 5. Failure modes

| failure | effect | recovery |
|---|---|---|
| judge crashes | its jobs' verdicts and settlements pause; front still admits and queues ids for it | supervisor restarts it; `_start` lists and seeds pending, new-epoch listing, feed complete within seconds after |
| judge stuck on R2 | same, front unaffected | as in-process (botocore timeouts) |
| front crashes | miners get connection refused for the restart (~tens of s); judges keep judging and settling; unaudited passes pause (stale feed) | supervisor restarts the front; first post (new epoch) -> judges list once -> unaudited passes resume |
| GPU process crashes | drawn audits wait (client retries), undrawn passes and voids continue; no validator-side error is counted | supervisor restarts it (model reload ~1-2 min) |
| GPU OOM / bad batch | merged batch re-run per request; failing request gets an error (any ValueError/RuntimeError subclass, e.g. `torch.AcceleratorError`, reaches the auditor as an audit error) | auditor's existing per-record retry, then validator-error count (5 in a row halts that judge; supervisor restarts it, as the container restart did before) |
| sticky CUDA fault (illegal access, launch failure, `AcceleratorError`) or 3 failed forwards in a row (OOM excepted) | the GPU process answers the requests in flight, then exits | supervisor reloads the model; audits wait meanwhile |
| notify queue overflow | oldest ids dropped, `dropped` sent | full listing before any unaudited pass |
| supervisor dies | children receive SIGKILL (PDEATHSIG) | Docker restart policy |

## 6. Rollout

The switch is two variables; without them the code path is today's single
process, unchanged.

- `RELIQUARY_CORPUS_SPLIT=1`: front, GPU process and supervisor.
- `RELIQUARY_CORPUS_SPLIT_JUDGES`: which jobs are judged out of the front.
  Groups are separated by `;`, jobs inside a group by `,` (job ids or task ids);
  `*` puts every remaining job in its own process. Jobs in no group are judged
  inside the front (their forward still goes to the GPU process: one model
  copy). Default when unset: `*`.

Step 1 (math only out of the front's judging), recommended form:
`RELIQUARY_CORPUS_SPLIT_JUDGES=math-omi-qwen38-27b-v1;code-qwen38-27b-v1,if-qwen38-27b-v1,logic-qwen38-27b-v1`
- math in its own process, the three others together in a second one: the
front judges nothing. Measured (section 8) at the no-judge baseline.

Allowed but not recommended: `RELIQUARY_CORPUS_SPLIT_JUDGES=math-omi-qwen38-27b-v1`
(code, IF and logic judged inside the front, as today). Their catch-up then
degrades the front exactly as today's single process does (p99 1.58 s vs
1.56 s single, 0.70 s grouped).

Step 2: `RELIQUARY_CORPUS_SPLIT_JUDGES=*`.

Rollback: remove both variables (same image).

Watch after each step: submit p50/p99 in the route's `corpus submission timing`
lines, `corpus judge pass` lines of the math judge (`job=math-...`), the GPU
process's `corpus gpu batch` lines (busy fraction), `/corpus/jobs/{id}/status`
(`audited` grows, `counts_complete`), and the supervisor's restart lines.

### Deploy (prod, step 1)

Same container recipe as today plus two variables (keep `--entrypoint`; the
image's default one dies on `BT_WALLET_NAME`):

```
docker run -d --name corpus-validator --restart unless-stopped \
  --gpus all --network host --cap-add SYS_PTRACE --shm-size 16g \
  --env-file /opt/corpus-prod/r2.env -e HF_HOME=/hf \
  -e RELIQUARY_TASK_ID=corpus-code-v1,corpus-math-omi-v1,corpus-if-v1,corpus-logic-v1 \
  -e RELIQUARY_TASK_CONTRACT=/work/corpus-multi-v1.contract.json \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e RELIQUARY_CORPUS_AUDIT_BATCH_TOKENS=32768 \
  -e RELIQUARY_CORPUS_SPLIT=1 \
  -e 'RELIQUARY_CORPUS_SPLIT_JUDGES=math-omi-qwen38-27b-v1;code-qwen38-27b-v1,if-qwen38-27b-v1,logic-qwen38-27b-v1' \
  -v /opt/corpus-prod:/work -v /workspace/hf:/hf \
  -w /work --entrypoint /opt/reliquary-venv/bin/reliquary \
  <image> \
  validate --network finney --netuid 81 --wallet-name corpus-validator \
  --hotkey default --http-host 127.0.0.1 --http-port 18090 --no-set-weights
```

Step 2: same with `-e RELIQUARY_CORPUS_SPLIT_JUDGES='*'`.

Costs: each child imports torch (~0.5 GB RSS; front + GPU + up to 4 judges
~3-4 GB host RAM); one CUDA context (GPU process) as before. Optional knobs:
`RELIQUARY_CORPUS_SPLIT_DIR`, `RELIQUARY_CORPUS_GPU_MERGE_TOKENS`,
`RELIQUARY_CORPUS_GPU_QUEUE_TOKENS`, `RELIQUARY_CORPUS_FEED_MAX_IDS`.

## 7. Proof (tests)

- `test_corpus_split_equivalence.py`: frozen main judge vs the split judge
  (feed over a link, scorer through the GPU wire codec) on the faulted scenario
  of `test_corpus_judge_equivalence` (ban, unreadable record, failed write,
  quarantine, restart): identical verdicts and voids; settled archives
  identical. Real processes: single-process validator vs split, same arrivals,
  identical verdict set and window archive.
- `test_corpus_split_isolation.py`: real processes, file-backed R2 with R2
  latency, realistic math records (16-32k token completions, ~0.5 MB records),
  a judge catching up a backlog of tens of thousands while miners submit:
  front submit p99 bounded in split, degraded in one process.
- `test_corpus_judge_throughput.py`: 80k-record catch-up drain time.
- `test_corpus_split_crash.py`: kill judge mid-pass, front, GPU process.

Measured numbers are in section 8. Rulings kept from the 2026-10-02 review:
the settlers of two judge processes see each other's archive writes after
<= 300 s (rules unchanged, no double pay); miner status answers 503 while its
judge is unreachable. Every child exit logs `corpus split: <child> exited
(code N)`: alert on it (the container stays Up while a child crash-loops).

## 8. Measurements (2026-10-02, 8 vCPU shared VPS)

Harness: real processes, on-disk bucket (reads 50-100 ms, puts 100-200 ms,
list pages 100 ms), realistic math records (16-32k tokens, 0.5-1 MB), proof
check spending the real `verify_chunk_proofs` CPU, forward at 4.86k tok/s.
Four jobs: math 20,000 behind with a 300k-id settlement, three others 1,500
behind each; miners submit 16-32k token records at 2/s for 180 s.

| layout | submit p50 | p90 | p99 | max |
|---|---|---|---|---|
| no judge work (baseline, 30 s) | 0.43 | - | 0.62 | 0.63 |
| single process (today) | 0.66 | 1.18 | 1.56 | 1.93 |
| split, math alone + 3 jobs in the front | 0.47 | 1.05 | 1.58 | 1.92 |
| split, math alone + 3 jobs in one judge | 0.45 | 0.57 | 0.70 | 0.75 |

The harness underestimates the in-process cost (no botocore/aiohttp/TLS CPU
per store call, which prod pays on the route's loop), so prod's degradation
was larger (p50 4 s -> 50-65 s waves); the split removes the judge from the
front either way.

Drain (virtual time, `corpus_judge_sim`, reads 50-100 ms, create-only PUTs
1.2-1.8 s, 64 connections, prod arrivals 6.9k/h on top, q 0.15, hold 4320 s):

- 80k pending up to 10 h old, 10k-token mean records (what a 17 %-busy GPU at
  ~2k verdicts/h implies): pending reaches the floor that can never drain
  (arrivals inside hold + slack, ~9.1k) after ~27 h; GPU 84 % busy; ~9.6k
  verdicts/h vs 1.7-2k/h on main before #298.
- 24k-token mean records (the 16-32k range): drawn audits alone need 1.42
  GPU-hours per hour of arrivals (1,035 drawn/h x 4.9 s): no judge layout
  drains that on one card at 4.86k tok/s; the backlog grows. Every drawn
  sibling inside a hold is audited before an undrawn record is paid, so the
  drain is GPU-bound by design.
- Seeding after a restart reads every pending record once (~130 records/s
  of 0.75 MB here): 80k is ~10 min before the first pass, and peak memory is
  SEED_SLICE_IDS (2048) decoded records, ~2.7 GB at 24k tokens per judge.
