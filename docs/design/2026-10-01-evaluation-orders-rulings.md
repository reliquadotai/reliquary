# Evaluation orders: rulings taken while implementing the subnet side

Decisions on points `2026-10-01-evaluation-orders-design.md` leaves open, one per line.

## Held-out sets

Ruling: RL's prompt universe is the whole train split of every catalog source — `window_prompt_range` slices `[0, len(env))` and every runtime factory (`spec.create()`, `load_external_backend` default) builds the train split — so no train index range of any source is disjoint from RL; only another split is. A test pins both facts.
Ruling: math is `reliquary_dapo_math_v1`, split `eval`, not OpenMathInstruct at offset 10000 — the design's condition ("if the check confirms it is disjoint") fails: RL samples OMI's whole index space, the 2026-09-11 measurement found 218/500 (43.6 %) of the offset-10000 problems trained with identical content, and OMI-2 repeats each question ~20 times across rows, so index-disjoint is not content-disjoint. The DAPO package assigns problems to train/eval/qualification by a salted hash of the problem key; RL and corpus jobs build train only.
Ruling: logic is `reliquary_logic_v2`, split `eval` — splits interleave (position = index·3 + split), so no eval task is ever a train task, whatever index RL or the corpus job (rows 100M..100.05M) uses.
Ruling: instruction following is `reliquary_instruction_following_v1`, split `eval` (10 % of the corpus, by a salted hash of the row key) — the IF corpus job (0..20k) and RL use train only.
Ruling: code is `reliquary_code_v1` (the corpus export's grader, which the code corpus job uses), rows `[2,381,806, 2,481,806)`: the last 100,000 rows of the pinned curation (`d3caaefc`, 2,481,806 rows). The package has no other split. It is disjoint from every corpus job and `jobs create` (CLI and admin) now refuses a job reaching it; it is NOT disjoint from past RL, which sampled the whole train split. This is the one ruled exception (`RL_OVERLAP_RULED`), written into its `set.json`; excluding the range from RL sampling is a follow-up (a consensus change, out of scope). `build-set` refuses a source of another length than the one the range was chosen against.
Ruling: the prod corpus jobs are listed in one table (`CORPUS_RANGES`); a new prod job on a train split must be added there, and `jobs create` refuses any job reaching a held-out region regardless.
Ruling: a set is the first N of `random.Random(seed).sample(region, count)`, written in that order with problem ids `{set_id}-{ordinal:06d}`; an order of N problems takes the first N lines. Ids carry no source index.
Ruling: `grading.jsonl` holds `{problem_id, source, split, source_index, prompt_sha256}` — the grader re-reads the source row and refuses (`score=null`, `source_drift`) a row whose prompt is no longer the frozen one, rather than copying answers out of the source.
Ruling: the subnet bucket holds its own copy of `set.json` beside `grading.jsonl`, and the grader trusts that copy, never the platform's. Publishing is create-only: the same bytes again are a no-op, other bytes are refused (`SetConflict`).

## Runner

Ruling: completions are uploaded as several files, `completions-NNNNN.jsonl`, one per chunk of 64 problems, each through the multipart API; `completion_keys` lists them in order.
Ruling: resume is from the pod's work directory (`--work-dir`): a chunk with a key is skipped, a chunk written but not uploaded is sent as written (its sha256 is recorded), and an upload started is resumed by its `upload_id` with only the missing parts. A new pod (lost disk) starts over: the contract has no listing of a task's finished uploads.
Ruling: each problem gets its own vLLM seed, `sha256(f"{seed}:{problem_id}")[:4]`, so chunking and resume never change a sample.
Ruling: upload parts are numbered from 1 (`PART_BASE`), part n covering bytes `[(n-1)·part_size, n·part_size)`.
Ruling: lost contact is a 401/403/409/410 on any call (no retry), or 4 heartbeats failing in a row; the work stops at the next chunk boundary (vLLM's batch is not interruptible). 5xx, 429 and transport errors are retried 4 times with a growing backoff.
Ruling: `progress` events count problems (`done`, `total`); `model_loaded` carries `{vllm_version, gpu, model_sha, seconds}`; `model_sha` is the pinned revision the snapshot was downloaded at.
Ruling: `top_k` 0 or null means no top-k (vLLM `-1`).

## Grading and report

Ruling: the grade body takes an optional `provenance` object (model, revision, model sha, sampling, seed, thinking, max_new_tokens, vLLM version, GPU, pod provider id), recorded verbatim in `report.json` — the admin host knows none of them, and the pod provider id is the platform's alone.
Ruling: `correct` is `score >= 1.0`: for code's fractional reward a sample is correct only when every case passes; `mean_score` reports the fraction too. pass@k is computed on `correct`.
Ruling: a row with `score=null` (grader crash, source drift) leaves its problem's denominator; pass@k for a k is taken over the problems with at least k graded samples, and says how many. A problem with no rows is `missing_problems` and out of every rate.
Ruling: instruction following is graded on the text after the last `</think>` (an unterminated `<think>` leaves nothing, a format failure) — its grader reads the whole completion and the catalog runs it without reasoning, but an order may ask for thinking. Other environments are graded on the whole completion, as RL does.
Ruling: a format failure is: no `\boxed{}` (math), no fenced block (code), no extractable JSON object (logic), an empty answer (IF).
Ruling: rows the order did not ask for (`unexpected_rows`), repeated `(problem_id, sample_index)` (`duplicate_rows`, first kept) and unparseable lines (`malformed_rows`) are counted, never graded.
Ruling: the bootstrap resamples problems 2000 times with seed 0 (recorded in the report); the interval is the 2.5/97.5 percentiles of the resampled means of per-problem c/n.
Ruling: idempotency is keyed by the eval id and the request digest (`set_ids`, `completion_keys`, `problems_per_set`; provenance excluded): the same request answers 202 then 200 with the stored keys, another request under the same id is 409 `grade_exists_with_another_request`. A failed grading answers 500 once and the next call starts it again.
Ruling: refusals the caller can act on are answered synchronously before any 202: 404 `set_unknown`, 422 for a bad key or a `problems_per_set` out of `[1, count]`; a completion key missing from the bucket fails the grading (500).
Ruling: rows are graded sequentially in a worker thread (the environment objects are not known to be thread-safe); code's grader runs each sample's cases in the package's own subprocess, the path `jobs export --apply-filter` uses.
