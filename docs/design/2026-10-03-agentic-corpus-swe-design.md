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
2. Each span length is at most `max_tokens_per_turn`; the total sequence is at
   most `max_total_tokens`.
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
- `MAX_SEQUENCE_TOKENS` rises from 65,536 to 131,072; `AUDIT_BATCH_TOKENS` is
  unchanged, so a long trajectory is scored alone.

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
    same path as a TOPLOC failure.
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
  "max_total_tokens": 65536,
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
- **M2 — honest replay agreement.** Replay the 66 recorded 27B trajectories.
  Pass: `final_diff` reproduced for at least 98% of them; derive the
  normalisation rules and the mismatch tolerance from the mismatches observed.
  If the diff is not reproducible for more than 2%, replay is downgraded to
  "grade the replay's diff" (outcome reproduction instead of diff equality) and
  this spec is amended.
- **M3 — audit cost.** Time the TOPLOC prefill of 40k-token trajectories on the
  27B. It sets `audit_q` for the job.

## 8. Prerequisites outside this repository

1. Merge reliquary-environments#21 (empty problem statements).
2. Stop installing `uv` from the network in each box (cache or bake it).
3. Per-command timeout in the bash harness.
4. Pin images by digest in the SWE-smith task set.

## 9. Capacity

| Miners (H100 each) | Trajectories/min | Successes/min | Grade + replay CPU | CPU executors (16 vCPU) |
|---|---|---|---|---|
| 10 | about 10 | about 6 | about 13 vCPU | 1 |
| 50 | about 50 | about 30 | about 60 vCPU | 4 |

Assumptions: 1 trajectory/min per H100 once concurrency is tuned to about 8-11
episodes, 25 CPU-s per grade, 80 CPU-s per replay. Replays of failures add about
10% more.

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
