# Dataset orders on any model (design)

2026-10-01. Approved by the user: "accept every model". Builds on:
- the dataset platform: `2026-09-30-dataset-platform-design.md`;
- evaluations on the subnet: `2026-10-01-evaluation-on-subnet-design.md`.

## Goal

A customer orders a generation dataset (SFT traces) on **any public Hugging Face model** at a pinned commit.
Today the model is fixed: the operator catalog plus the corpus control's single loaded model (Qwen3.8-27B).
The GPU-less control already serves any model for evaluations, through per-model qualification and audit by
executor pairs. Dataset orders reuse that path.

## What is reused unchanged
- **Qualification:** two executors on distinct providers, an agreed band, thresholds clamped to the hard
  ceiling, and eos/architecture read by the control from the model files.
- **Audit by executor pairs:** two agreeing executors from distinct providers, at most three scorers,
  otherwise the batch is parked. Executors are scoped `eval`.
- **Incentives:** a fixed cap share per task (default 0.02), set by the operator.
- **Settlement:** unchanged.
- **Deliveries (export v2):** unchanged, with code graded in the sandbox only.
- **The fleet guards:** one-shot rent, lost-rent halt, spend cap, orphan sweep and per-pod tokens.

## Reliquary changes
1. **One control for every order job.** The GPU-less control (`reliquary corpus eval-control`) serves:
   - eval jobs (`${prefix}eval-…`), as today;
   - **generation jobs (`${prefix}gen-…`)**: the job's prompt source is a catalog environment's index
     range, exactly like today's corpus jobs, with the order's model, sampling, thinking, `max_new_tokens`
     and `samples_per_prompt`.

   Rename the command to `reliquary corpus order-control`; `eval-control` stays as an alias. The nginx note
   extends the prefix regex to both kinds. The corpus control (prod, GPU) refuses both prefixes.
2. **Audit policy for generation jobs:** the existing partial audit applies (probation 100, q 0.15, drand
   draw, suspect/ban), with every audited batch scored by an executor pair. Failed audits **do not** reopen
   slots: that is today's corpus semantics, and the eval-only reopening stays eval-only.
3. **Qualification for generation jobs:** the record binds:
   - model@rev;
   - the environment and a sample of its prompts (the first 32 of the order's range);
   - sampling, `max_new_tokens`, thinking.

   Job creation refuses any mismatch, as for evals.
4. **Admin:** `POST /admin/v1/jobs` accepts generation jobs on any model when they reference a
   `qualified` qualification record. The old `RELIQUARY_ADMIN_MODELS` map stays only for operator-declared
   jobs. Prefix scope and cap limits apply as today.
5. **Supported architectures:** one table in code, `SUPPORTED_ARCHITECTURES`. It holds the `architectures[0]`
   values of `config.json` that the miner (vLLM) and the verifier (HF + TOPLOC hooks) are known to handle;
   read the code to decide which, with no guessing. Text-only use of multimodal configs follows the miner's
   existing `limit_mm` rule. Every entry must be backed by an existing test or bench. Anything else is
   refused at qualification with `architecture_unsupported`. Expose it at `GET /admin/v1/architectures`.

## Platform changes
1. **Dataset quotes take any model:** `{model, revision, env, prompt_count, samples_per_prompt,
   max_new_tokens, thinking, sampling}`.
   - The Hugging Face lookup and refusals are shared with evaluations: `model_private`, `model_not_found`,
     `model_too_large`, `model_unsupported`, plus `architecture_unsupported` from `config.json` against the
     admin's architecture list (cached).
   - Price = executor cost (qualification + 2 executors × ETA) + a per-token fee
     (`DATASET_PRICE_PER_MTOK_USD`) × worst-case tokens, × (1 + margin). Refund of unverified tokens at
     delivery.
   - The old catalog-model path stays for operator-qualified models.
2. **Order states for any-model orders:**
   ```
   reserved → qualifying → queued → running → draining → exporting → delivered
   ```
   Money rules are those of evaluations: one terminal credit, full release before a job exists, refund of
   the unverified part after.
3. **Fleet:** reuse the evaluation fleet per model@revision: a qualification pair, then an audit pair on
   distinct providers, plus extra executors when the control asks. **Executors are shared across orders of
   the same model@revision** (eval and generation alike), with at most 3 per model. This also closes the
   per-order duplication noted in the eval rulings.
4. **API:** existing `/api/v1/dataset-orders` routes gain the any-model fields; nothing else changes for the
   customer.

## Out of scope
- private or gated models;
- hybrid architectures needing extra kernels (for example GDN), unless the table already supports them;
- deposits;
- per-token pricing.

## Acceptance
- **reliquary:**
  - a generation job on a second model served by the control next to an eval job;
  - partial audit with executor pairs;
  - qualification binding;
  - architecture refusal;
  - the corpus control refuses `gen-` and `eval-` prefixes;
  - settlement byte-identical for existing jobs.
- **platform:**
  - any-model quote refusals, including the architecture check;
  - reserved→delivered against fakes that model reliquary's real semantics (the lesson of the eval review:
    the fakes must follow the reliquary contract, not the client);
  - shared executors per model;
  - money rules.
- CI green on both. No deploy, no real rent.
