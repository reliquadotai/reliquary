# Agentic corpus jobs: multi-turn SFT trajectories on SWE-smith

Date: 2026-10-03. Status: design approved in conversation, awaiting written-spec review.

## 1. Goal

Let a corpus job pay miners for **multi-turn agentic trajectories** produced by a
teacher model (first job: Qwen3.8-27B) on containerised software-engineering
tasks (first task set: SWE-smith from `reliquary-environments`), and export the
successful ones as an SFT dataset.

Out of scope for this spec: RL on agentic environments, signed-observation
sandbox gateway, polyglot and terminal task sets, paying by success.

## 2. Decisions taken

| Decision | Choice | Consequence |
|---|---|---|
| What is paid | Per accepted slot after the TOPLOC verdict, as today; success only filters the export | No incentive to fake outcomes, so failed trajectories need only sampled replay |
| First task set | SWE-smith only (`reliquary-swe`, split `train`, 20 images, about 78 GB, 18,546 tasks once reliquary-environments#21 lands) | One image set fits on every miner and executor |
| Where validator-side containers run | Remote zero-secret CPU executors under a lease, never on the control host | Hostile code never shares a host with R2 keys; capacity scales by adding boxes |
| Where the agent loop lives | In `verifiers` + `reliquary-swe`, unchanged. Reliquary only supplies token generation and proofs | No second copy of tool execution, container lifecycle, network policy or grading |

## 3. Measured basis (2026-10-02/03)

Measured on an H100 (vLLM 0.30) plus a 16-vCPU Docker host (`sandbox-dev-01`):

- Qwen3.8-27B on SWE-smith: 44/66 trajectories pass (67%); median 24 turns,
  about 10k generated tokens, median context 20k, max 51k; 0.7 trajectories/min
  per H100 at concurrency 32. The prefix cache was evicted above about 11
  concurrent episodes (255k-token KV cache, hybrid model with 784-token blocks).
- Container cost: about 13 CPU-seconds fixed per task (create, set up, diff,
  fresh grading box), about 0.2 CPU-s per ordinary tool turn, about 5 CPU-s per
  turn that runs tests. A 16-vCPU host grades about 75 reference patches/min.
- Defects found: 22% of SWE-smith rows have an empty problem statement
  (reliquary-environments#21); the bash harness installs `uv` over the network in
  every box; tool commands have no per-command timeout (one `grep -rn /` took 26
  minutes).

## 4. Architecture

```
MINER (H100 for the 27B, Docker, ~150 GB disk, ~16 vCPU)
  verifiers eval loop: reliquary-swe taskset, bash harness, docker runtime,
  train client (renderers: qwen38) ───tokens in/out───► [N1] local generate endpoint
                                                          VllmGenerator + hidden capture
                                                          + TOPLOC proofs per request
  verifiers Trace ──► [N2] trajectory builder ──► signed CorpusTrajectory submission

CORPUS VALIDATOR (control)
  [N3] intake: cheap per-turn checks ──► slot consumed ──► record v2 stored
  [N4] TOPLOC audit v2 (GPU, existing policy) ──► verdict ──► payment (unchanged)
  [N5] grade leases ──► zero-secret CPU executors
         grade every claimed diff; replay every passing trajectory and a
         sample of failing ones ──► grades/{sid}.json; replay failure = audit failure
  [N6] export v2: certified successes with messages, tokens, assistant mask, diff
```

Unchanged: slot and cursor ledgers, miner walk, settlement and pricing, audit
policy (probation, suspect, ban), the TOPLOC thresholds, `reliquary-environments`.

## 5. Components

### N1. Miner generate endpoint

`reliquary/miner/corpus_generate_server.py`: a loopback-only HTTP server
implementing `POST /inference/v1/generate` as `renderers.client.generate` calls
it (body `{model, token_ids, sampling_params}`; response
`{choices: [{token_ids, logprobs}]}`), plus `GET /v1/models` exposing
`max_model_len`.

- It wraps the existing `VllmGenerator` (`reliquary/miner/corpus_miner.py`). One
  request is one assistant turn: prompt = the full token history, completion =
  the turn.
- `sampling_params` from the client are ignored except `max_tokens`, which is
  clamped to the job's `max_tokens_per_turn`; the job's sampling is authoritative.
  `stop_token_ids` come from the job (turn terminator and eos).
- Proofs: rows for the completion only (`[prompt_len - 1, prompt_len + len - 1)`)
  feed `build_chunk_proofs`. Proofs are kept in memory keyed by
  `(session id header, sha256(prompt ids ‖ completion ids))` and handed to N2 when
  the episode ends, then dropped.
- Prefix caching is **enabled** for this generator (gate M1). Hidden capture
  must accept a request whose cached prompt rows were not recomputed and still
  return exactly the completion rows.

### N2. Trajectory builder

`reliquary/corpus/trajectory.py` (pure) builds a `CorpusTrajectory` from a
verifiers `Trace` produced with the train client:

- `tokens`: the interleaved token sequence after the initial prompt (assistant
  turns, tool observations, turn scaffolding), exactly as rendered.
- `turns`: `[{start, end, proofs}]`, the assistant spans in `tokens`, from the
  trace's per-turn token ids and sampled mask. Each span's proofs come from N1.
- `final_diff`: `trace.info["patch"]`, the diff the env collected.
- `stop`: `agent_completed`, `max_turns` or `context_length`.

The miner process runs `verifiers` in-process (`run_episode` on one task index),
so no CLI parsing of trace files is involved.

### N3. Intake and cheap checks

New pure functions in `reliquary/corpus/checks.py`, applied when the job carries
`episode`:

1. Spans are ordered, non-overlapping, inside `tokens`, and number at most
   `max_turns`.
2. Each span length is at most `max_tokens_per_turn`; the initial prompt plus
   `tokens` is at most `max_total_tokens` (60,000, gate M3).
3. Each span ends with the renderer's turn terminator, except a final span that
   ends with eos or hits its cap.
4. `len(proofs) == ceil(span_length / chunk_tokens)` per span.
5. The initial prompt re-rendered from the task index equals `rendered_prompt`
   (the existing prompt-fidelity check, using the pinned renderer).
6. Every non-assistant segment decodes to a well-formed tool-response block of
   the pinned renderer, at most the env's `max_observation_bytes`.
7. Duplicate digest over the full `tokens`.

`token_count` stored in the record counts assistant-span tokens only, so the
existing settlement (`cap × passed token share`) pays for generated tokens.

### N4. TOPLOC audit v2

- `AuditItem` gains `spans: [[start, end], ...]` under protocol
  `reliquary.corpus-audit/v2`; v1 executors keep serving single-turn jobs.
- Scoring is one batched prefill of `prompt + tokens`, with rows gathered for the
  assistant spans only, then the existing `verify_chunk_proofs` and
  `sequence_verdict` per span; the item fails on the first failing span.
- `MAX_SEQUENCE_TOKENS` stays at 65,536 (it bounds prompt + tokens of one item);
  the binding cap is the job's `max_total_tokens` = 60,000 over prompt + tokens,
  the most one 80 GB audit GPU prefills in one pass (gate M3: 71.5 GB peak at
  60k, which leaves only about 8 GB of headroom on the card;
  longer trajectories would need chunked prefill or a larger card).
  `AUDIT_BATCH_TOKENS` is unchanged, so a long trajectory is scored alone.

### N5. Grade leases and CPU executors

A third lease kind alongside audit and eval, reusing their token auth, lease
expiry, strikes, quarantine and 5% local recheck:

- Executor scope `grade` in `infrastructure/corpus_executor_store.py`, not bound to
  a model id.
- `GradeLease{lease_id, expires_at, items[]: {submission_id, task_index,
  env_pin, mode: grade|replay, final_diff, actions}}`, where `actions` are the
  tool calls parsed from the assistant spans with the pinned renderer.
- `GradeResult{items[]: {status, diff_applied, tests_passed, replay_diff_equal,
  observations_compared, observations_mismatched, detail}}`.
- The executor process (`reliquary/validator/corpus_grade_executor.py`) imports
  `reliquary-swe`, runs `grading.grade` for `grade`, and for `replay` creates a
  fresh box from the pinned image, executes `actions` in order through the env's
  tools, compares each observation after normalisation (gate M2), collects the
  diff and grades it.
- Policy on the control (`reliquary/validator/corpus_grading.py`):
  - every accepted trajectory gets `grade`;
  - every trajectory whose grade passes gets `replay`;
  - a failing trajectory gets `replay` with probability `replay_fraction_failed`,
    drawn like audits (drand, submission id);
  - a replay that does not reproduce `final_diff` or exceeds the mismatch
    tolerance is recorded as a confirmed audit failure (void, suspect, ban), the
    same path as a TOPLOC failure. The tolerance is per episode:
    `allowed = max(5, ceil(0.12 x n))` mismatched observations out of `n`
    (`replay_compare.within_tolerance`, derived in gate M2).
- Results go to `reliquary/corpus/jobs/{job}/grades/{sid}.json`. Grades never
  change payment except through the audit-failure path above.

### N6. Export v2

`corpus/delivery.py` gains, for jobs with `episode`: `task_id`, `messages`
(OpenAI-style, with tool calls and reasoning), `tokens`, `assistant_mask`,
`final_diff`, `graded_success`, `replay_certified`, `turns`, `stop`. The SFT
dataset is the rows with `replay_certified = true` and `graded_success = true`.
`job_grader`'s refusal of episode sources stays for single-turn graders and is
bypassed for `episode` jobs, whose grading comes from N5.

## 6. Data shapes

Job contract, optional field written only when set (existing job hashes are
unchanged):

```json
"episode": {
  "env": {"package": "reliquary-swe", "version": "<pinned>", "split": "train", "num_images": 20},
  "harness": "bash",
  "renderer": "renderers:qwen38@<pinned>",
  "verifiers": "<pinned commit>",
  "max_turns": 40,
  "max_tokens_per_turn": 8192,
  "max_total_tokens": 60000,
  "replay_fraction_failed": 0.10
}
```

with `sampling.n = 1`, `slots_per_prompt >= 2`, `prompt_count` = the task-set
size. A binary that predates the field refuses the manifest (unknown field), as
for `prompt_start`.

Submission completion (signing domain `reliquary/corpus-trajectory/v1`, binding
sha256 of `tokens`, of the span list, of each span's proofs, and of
`final_diff`):

```json
{"tokens": [...], "turns": [{"start": 0, "end": 812, "proofs": ["..."]}],
 "final_diff": "diff --git ...", "stop": "agent_completed"}
```

Stored record: schema `reliquary/corpus-submission-record/v2`, same top-level
fields as v1, trajectory inside `completions[0]` so it sorts before `cursor`
and the judge's tail read of `submission_meta` keeps working.

## 7. Gates before implementation

Each gate runs on the existing H100 test box and `sandbox-dev-01`, and its result
is recorded in this file before the dependent component is written.

- **M1 — proofs with prefix caching.** Generate 32 multi-turn trajectories with
  prefix caching on; audit them with the unchanged HF prefill. Pass: every span
  passes the job thresholds with margins inside the measured honest band. If it
  fails, N1 runs with prefix caching off and the miner hardware requirement is
  re-measured.
  **M1 result (2026-10-03, H100, Qwen3.8-27B, vLLM 0.30, `scripts/agentic_proof_gate.py`,
  data in `docs/design/measurements/2026-10-03-m1-agentic-proofs.json`).**
  32 trajectories x 6 turns x 512 max tokens, prefix cache on: 63/192 turns were
  served partly from the cache (hybrid model, `mamba_cache_mode=align`, block size
  784 tokens, so a turn can only hit once its prompt exceeds one block; the 4B
  smoke hit 28/32). Audit by one HF prefill per trajectory: **192/192 spans pass
  at 60/40/40.** Worst chunk measures: exp 60, mant mean 18.63, median 14.00,
  against the honest band exp <= 16, mant mean <= 4.14, so the band clause is
  NOT met literally. Mean over chunks: exp 3.20, mant 1.23; 98.9% of chunks have
  exp <= 16; p99 = 17. The worst chunk is a 1-token span (tokens 643-644), where a
  per-chunk statistic is noisy. Control with prefix cache OFF (same script,
  `--no-prefix-cache`, 0/192 hits, `...-control-cache-off.json`): 192/192 pass,
  identical worst (exp 60, mant 18.63), mean exp 3.24, 99.0% <= 16. With and
  without the cache: same pass rate, mean exp and >16 share; the runs are
  unpaired (1,260 vs 1,499 chunks) and the tail mant is unresolved (long-span
  tail mant mean 11.70 with the cache vs 7.23 without). Decision: prefix
  caching ON for N1, with
  the open point that the honest band must be restated per span length (short
  spans exceed it with or without the cache); cache-off is not needed.
- **M2 — honest replay agreement.** Replay the 66 recorded 27B trajectories.
  Pass: `final_diff` reproduced for at least 98% of them; derive the
  normalisation rules and the mismatch tolerance from the mismatches observed.
  If the diff is not reproducible for more than 2%, replay is downgraded to
  "grade the replay's diff" (outcome reproduction instead of diff equality) and
  this spec is amended.
  **M2 result (2026-10-03, sandbox-dev-01, 8 concurrent replays,
  `docs/design/measurements/2026-10-03-m2-replay-agreement.json`).** PASS:
  `diff_equal` 66/66 of the 66 replayed episodes (0 errors); the diff is
  reproduced exactly, so diff equality is kept (no downgrade to "grade the
  replay's diff"). Observations: 2017 compared. Mismatch share 186/2017 = 9.2%
  with the first rule set (durations, timestamps), 68/2017 = 3.4% after the
  rules added from the observed pairs: object addresses (`0x...`), the box's own
  `base` commit hash, hashes in built version strings, `ls -l` mtimes, `date`
  and `git log/show` dates and full hashes. The 3.4% is pooled over 2,017
  observations; the tolerance is applied **per episode**, where the spread is
  wide. Per episode (n = 5 to 58 observations): 29 of 66 episodes have no
  mismatch, 17 have 1, 10 have 2, 9 have 3, 1 has 4; max 4 mismatches, p90 3;
  max share 20% (3 of 15), p90 share 8.7%; 16 of 66 honest episodes exceed
  5.4%, so a pooled-share tolerance would void a quarter of honest work.
  **Replay tolerance, per episode: at most `max(5, ceil(0.12 x n))`
  mismatched observations in an episode of `n`** (`reliquary/corpus/
  replay_compare.py`, `within_tolerance`). All 66 honest episodes pass with at
  least 2 observations to spare. The floor of 5 covers short episodes (one
  pandas rebuild can make 3 of 15 differ); the 12% share covers long ones
  (worst honest share above 40 observations: 4 of 42, 9.5%). No exemption by
  command class: the residual mismatches below come from ordinary commands
  (`pytest`, `python -c`, `find`, `pip`, `ls`, `git log`), and a class rule
  would be a free pass for any forged observation whose command matches it.
  **Forgery budget (cost to price in plan 4):** a miner can forge up to
  `max(5, ceil(0.12 x n))` observations per episode undetected by replay:
  every observation of an episode of 5 or fewer, 5 of 6-41 observations, 6 at
  42-50, 7 at 58, 12 at 100. The diff must still match exactly, so forged
  observations cannot change the delivered patch, only the text the student is
  trained to read.
  Residual mismatches (not normalisable, absorbed by the tolerance): pytest
  `--durations` lists and parallel ninja progress lines, whose order follows
  timing (30 of 68); commands listing the recording box's pip cache, conda
  channels, or `find /` (28); user-site `sys.path` (see below); `stat %Y` epochs.
  Replay cost: p50 42.6 s per episode; about 288 vCPU-seconds per replay
  (vmstat over the 22 min run, 8 in parallel, includes box start and the
  episodes' own test suites; first run 294).
  **User-site decision: (b), leave it counted.** The recording harness installed
  its own dependencies into `/root/.local/lib/python3.12/site-packages`, so
  commands printing `sys.path`/`pip list`/`env` differ. Measured on the
  recorded traces: 6 of 2017 observations (0.3%), in 6 of 66 episodes. That is
  inside the tolerance, and a rule hiding that path would also hide a miner
  whose box differs for real, so no rule was added.
- **M3 — audit cost.** Time the TOPLOC prefill of 40k-token trajectories on the
  27B. It sets `audit_q` for the job.
  **M3 result (2026-10-03, H100 80 GB, Qwen3.8-27B bf16 text-only, sdpa,
  `scripts/agentic_audit_cost.py`, data in
  `docs/design/measurements/2026-10-03-m3-audit-cost.json`).** One prefill of the
  whole trajectory plus proof verification of one 400-token span per 1000 tokens
  (3 repeats; from 20k up the repeats agree within 1%, while at 10k the first run
  is 2.75 s against 1.94 and 1.96 s, a warm-up effect; synthetic tokens, cost depends on length only):
  10k = 2.0 s, 20k = 4.3 s (+0.26 s verify), 40k = 9.7 s (+0.5 s), 50k = 12.5 s,
  60k = 15.4 s (+0.8 s). Roughly linear, about 0.25 ms/token. Peak memory with
  the weights (about 50 GB): 54 GB at 10k, 57 at 20k, 64 at 40k, 68 at 50k,
  71.5 at 60k. **60k fits on one 80 GB card, no OOM**, but with 8 GB of headroom;
  60k is the practical ceiling for
  `max_total_tokens` on this hardware without chunked prefill. Audit capacity of
  one audit GPU, `3600 / (prefill + verify)` per hour: 783/h at 20k, 350/h at 40k,
  223/h at 60k. Production rates: the measured rate is 0.7
  trajectories/min per miner H100 at concurrency 32, median context 20k, max 51k
  (section 3, Qwen3.8-27B on SWE-smith); 1/min is the assumption of section 9,
  valid once concurrency is tuned to about 8-11. The 20k median is measured; the
  40k and 60k rows are what-if medians, not observations. `audit_q = min(1,
  capacity / produced)`:

  | median length | capacity/h | measured 0.7/min (42/h per miner): 10 / 50 miners (420 / 2100 per h) | assumed 1/min (60/h): 10 / 50 miners (600 / 3000 per h) |
  |---|---|---|---|
  | 20k | 783 | 1.0 (1.9 before cap) / 0.37 | 1.0 (1.3 before cap) / 0.26 |
  | 40k | 350 | 0.83 / 0.17 | 0.58 / 0.12 |
  | 60k | 223 | 0.53 / 0.11 | 0.37 / 0.07 |

  One audit GPU therefore covers all trajectories only at 10 miners with a
  median of 20k tokens or less (or about 40k at the measured rate, 0.83); beyond
  that either more audit GPUs (n GPUs multiply `audit_q` by n) or a lower
  `audit_q` is needed.

## 8. Prerequisites outside this repository

1. Merge reliquary-environments#21 (empty problem statements).
2. Stop installing `uv` from the network in each box (cache or bake it).
3. Per-command timeout in the bash harness.
4. Pin images by digest in the SWE-smith task set.

## 9. Capacity

| Miners (H100 each) | Trajectories/min | Successes/min | Grade + replay CPU | CPU executors (16 vCPU) |
|---|---|---|---|---|
| 10 | about 10 | about 6 | about 35 vCPU (4 grade + 31 replay) | 3 |
| 50 | about 50 | about 30 | about 174 vCPU (21 grade + 154 replay) | 11 |

Assumptions: 1 trajectory/min per H100 once concurrency is tuned to about 8-11
episodes (assumed; measured 0.7/min at concurrency 32); 60% success (measured
67%); 25 CPU-s per grade on every trajectory (assumed; section 3 measured 13
CPU-s fixed plus about 5 per test run); 288 vCPU-s
per replay (measured, gate M2) on every success plus 10% of failures; executor
count = vCPU / 16 rounded up, at full utilisation with no headroom. At 10
miners: grade 10 x 25 = 250 CPU-s/min, replay (6 + 0.4) x 288 = 1,843 CPU-s/min.
At 50 miners: grade 1,250 CPU-s/min, replay (30 + 2) x 288 = 9,216 CPU-s/min.
Replay dominates: about 88% of executor CPU.

Miner requirement: H100 80 GB (27B weights 52 GB), about 16 vCPU and 150 GB disk
for the pre-pulled SWE-smith images.

## 10. Testing

- Unit: contract parse and hash stability with and without `episode`;
  trajectory builder from recorded traces; each N3 check with accepted and
  rejected cases; signing binding; audit v2 span gather against a single-turn
  equivalent; grade policy (which submissions get grade/replay, failure path).
- Integration on the test boxes: a real miner (27B, prefix caching on), a local
  corpus validator, one grade executor on `sandbox-dev-01`, through submission,
  verdict, grade, replay and export; plus a forged `final_diff` and a forged
  observation, both of which must end as confirmed audit failures.
