# Evaluation orders (design)

2026-10-01. It builds on the dataset platform (`2026-09-30-dataset-platform-design.md`): the fleet,
per-pod credentials, balances, the admin service and deliveries. Approved by the user. No RL product in
this phase.

## Goal

A customer picks a model and a list of our environments. They receive a reproducible score report per
environment, plus every graded response.

```
POST /api/v1/evaluations/quotes
{ "model": "org/name", "revision": "<40-hex commit>",
  "envs": [ {"env": "math", "problems": 500, "samples": 8}, {"env": "code", "problems": 300, "samples": 4} ],
  "max_new_tokens": 16384, "thinking": true,
  "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20} }      // optional, defaults recorded
```

## Scope of v1

- **Mode direct only.** One rented pod generates with vLLM. The miners are not involved. Mode subnet is
  later.
- **Single-turn environments only:** math, code, logic, instruction following. The multi-turn ones (SWE,
  terminal, tools) need sandboxes on the pod and come in v2.
- **Public Hugging Face models only**, pinned to a commit. Any architecture vLLM loads. Private models
  (customer token) come in v2.
- **Size:** the model must fit on one node of at most 8 GPUs.

## Trust and data flow

1. **The pod generates; it never grades.** The rendered prompts are the only part of an eval set the pod
   receives. Ground truth, tests and grader metadata stay on our side, so the pod cannot see answers or
   grade itself.
2. **Grading runs on the trusted admin host**, using the same graders as `jobs export --apply-filter`
   (`reliquary.corpus.export.job_grader` and the environment graders).
3. **The pod holds only its per-pod token**, same rule as the dataset platform. It downloads its prompts
   and uploads its completions through the platform API; the Worker writes to R2.
4. **Accepted residual risk:** a malicious Lium provider could substitute completions. The report names
   the pod provider id, the GPU and the vLLM version. A TOPLOC spot-check of completions on a trusted GPU is
   a v1.1 item, not v1.

```
platform ──claim──► pod (vLLM) ──completions.jsonl (upload API)──► platform R2
platform ──R4 POST /admin/v1/evaluations/{id}/grade──► admin host
admin host: read completions (platform bucket) + private grading data → graded.parquet + report.json → platform R2
platform ──► customer downloads
```

## Frozen evaluation sets (reliquary)

- `reliquary eval build-set --env <catalog env> --count N --seed S --out DIR` freezes a set. Each set
  contains:
  - `prompts.jsonl`: `{problem_id, env, messages | text}`, rendered with the catalog prompt template;
  - `grading.jsonl`: private, `{problem_id, ...whatever the grader needs}`;
  - `set.json`: `{set_id, env, source, index_range, count, seed, sha256 of both files, created_at,
    disjointness}`.
- **Disjointness is mandatory.** Problems come from a declared held-out index range that no RL task and no
  corpus job uses. Two things must exist:
  - a test that holds the ranges disjoint from every known `prompt_start` / `prompt_count` and from the
    RL sampling range;
  - a written justification per environment in `set.json`.

  Math: reuse the OpenMathInstruct held-out offset that reliquary-compare already uses (10000), if the
  check confirms it is disjoint.
- **Where sets live:**
  - `prompts.jsonl` and `set.json` in the platform bucket, under `eval-sets/{set_id}/`;
  - `grading.jsonl` only in the admin host's private storage (the subnet R2, under
    `reliquary/eval-sets/{set_id}/`).
- An order samples `problems` from a set deterministically: the first N in a seeded permutation recorded
  in the set. The same order on the same model gives the same problems.

## Generation runner on the pod (reliquary)

`reliquary eval run --platform URL --executor-id ID`, with the token in `RELIQUARY_EXECUTOR_TOKEN`:
1. It claims an evaluation task through the platform's internal API.
2. It downloads the model at the pinned revision from Hugging Face.
3. It renders:
   - with the tokenizer chat template when one exists, passing `enable_thinking=thinking` where the
     template supports it;
   - as raw text otherwise.
4. It generates `samples` completions per problem with vLLM (`n=samples`, the fixed seed, tensor parallel
   = GPU count).
5. It uploads `completions.jsonl` in chunks through the existing multipart upload API. Each line:
   `{problem_id, sample_index, completion, completion_tokens, finish_reason}`.
6. It sends heartbeats and progress events, then posts a result: `{vllm_version, gpu, model_sha, rows,
   seconds}`.

A crash resumes from the uploaded chunks. Lost API contact stops the work, like grizz's adapter.

## Grading and report (reliquary, admin service)

`POST /admin/v1/evaluations/{eval_id}/grade`, body `{set_ids, completion_keys, problems_per_set}`. It
follows the deliveries pattern: 202 while running, then 200 `{keys}`, and must stay idempotent.

It writes under `deliveries/{eval_id}/`:
- `graded.parquet`: every completion with `{env, problem_id, sample_index, completion, tokens,
  finish_reason, correct, score, grader_detail}`;
- `report.json`, per environment:
  - `n_problems`, `samples`;
  - `pass@1`: the mean over problems of c/n, with a 95 % CI by bootstrap over problems (seeded);
  - `pass@k` for k in {1, 2, 4, 8, …} ≤ samples, using the unbiased estimator `1 - C(n-c,k)/C(n,k)`
    (port it from reliquary-compare `src/eval`, with its tests);
  - `truncation_rate` (finish_reason == length);
  - `format_failure_rate` (the grader extracted no answer);
  - `mean_completion_tokens`.

  It also carries a `macro` average over environments. Provenance: model, revision, model sha, set ids
  with sha256, sampling, seed, thinking, max_new_tokens, vLLM version, GPU, pod provider id, reliquary
  version.
- `manifest.json` with sha256 per file.

Code is graded with the existing code grader, the same path the corpus export uses. A grader crash on a row
yields `score=null` with `grader_detail`, never a silent 0. The report counts these rows apart.

## Platform (reliquary-platform)

- **Catalog:** `GET /api/v1/evaluations/catalog` lists the eval environments and their sets (env, count,
  description, default `max_new_tokens`), from an `eval_sets` table filled by the operator.
- **Quote:** `POST /api/v1/evaluations/quotes`.
  - The platform reads the model size from the Hugging Face API (the sum of safetensors sizes at the
    revision) and refuses with 422 when it is too large, private or not found.
  - GPU count = the smallest of 1, 2, 4, 8 with `weight_bytes × 1.3 ≤ count × 80 GB`.
  - Formulas:
    ```
    est_tokens = Σ problems × samples × min(avg_tokens(env, thinking), max_new_tokens)
    gen_hours = est_tokens / tokens_per_hour(gpu_count, size bucket)   [config table]
    price = (boot_hours + gen_hours) × gpu_count × gpu_hour_price × (1 + margin) + grading fee
    ```
- **Order:** `POST /api/v1/evaluations` with `{quote_id}` and an `Idempotency-Key`. The price is reserved.
  `GET` returns the state, progress and deliveries. Cancel releases the unconsumed part.
- **States:**
  ```
  reserved → provisioning (pod for THIS order's model)
           → generating (pod claimed the task; progress events)
           → grading (R4 grade)
           → delivered
  any → failed (deadline, model load failure reported by the pod, grading failure),
        with refund of the unconsumed part, needs_attention on ambiguity
  ```
- **Fleet:**
  - An evaluation pod is dedicated to one order (one model) and gets the `gpu_count` from the quote.
  - It is destroyed as soon as its task is terminal; the grace is `EVAL_POD_GRACE_SECONDS`, default 300.
  - It reuses every fleet guard: one-shot rent, lost-rent halt, spend cap, orphan sweep and per-pod token.
    The token is scoped to its task.
- **Pod API:** internal endpoints authenticated by the per-pod token and the task lease.
  - claim, heartbeat, events, prompts download, uploads (the existing multipart machinery in
    `worker/transfers.ts`), result.
  - A pod can only see its own order's prompts.
  - They mirror the existing `/api/internal/tasks/*` pattern without changing grizz's training executor
    path.
- **API keys:** a new scope `evaluations:write`. An `orders:write` key does not grant it.
- **Defaults:** `EVALUATIONS_MODE=disabled`.

## Acceptance

- **reliquary:**
  - the eval-set disjointness test;
  - the unbiased pass@k against brute force on small cases;
  - bootstrap determinism;
  - the runner end to end against a fake platform and a tiny model, or a fake generator;
  - grading: every row graded or flagged, the report from a fixture with known counts, idempotent grade.
- **platform:**
  - quote refusals (too large, private, unknown revision);
  - GPU count;
  - reserved→delivered against fake Lium, fake admin and a fake pod;
  - a pod cannot read another order's prompts;
  - pod destroyed at the end;
  - refund on failure;
  - the scope separation.
- CI green on both branches. No deploy and no real rent.
