# Orders: batch and evaluation on one path

A customer order (a batch of generations or an evaluation) is one subnet object:

```
order = {source, model, sampling, judge, grade}
```

Batch and eval share the source, the model pin, the forced draw, the job, the miners, the audit and the delivery.
They differ only by settings: an evaluation draws from a frozen set, audits every sample (`audit_q = 1`) and is
graded into a report; a batch audits a share (15 %) and is delivered as samples.

| Customer API          | source                          | grade                     | job id prefix  |
|-----------------------|---------------------------------|---------------------------|----------------|
| `/v1/batches` (file)  | an upload's set                 | none                      | `order-gen-`   |
| batch on an env       | a catalog env range             | none (or annotate)        | `order-gen-`   |
| `/evaluations`        | one of our sets, or an upload's | report                    | `order-eval-`  |

## 1. Sources

Every source but a catalog env range is a **frozen prompt set** in the eval-set format
(`reliquary/eval/prompt_source.py`): `prompts.jsonl` + `set.json` (+ `grading.jsonl` for an evaluation) in the
subnet bucket, named by a job as `eval-set:<set_id>:<n>:<sha256>`. Our eval sets and customer uploads are the same
kind of object; miners fetch the job's lines from its control and never see grading data.

### Uploads (`POST /admin/v1/uploads`)

The platform stores the customer's OpenAI Batch JSONL in its bucket (`batch-inputs/...`) and asks the admin to
ingest it (`reliquary/orders/uploads.py`):

```json
{"object_key": "batch-inputs/<user>/<sha256>.jsonl", "sha256": "...", "model": "org/name",
 "revision": "<40-hex>", "purpose": "batch" | "eval", "grader": "boxed_math_v1", "max_prompt_tokens": 32768}
```

- Lines: `{custom_id, method: "POST", url: "/v1/chat/completions" | "/v1/completions", body}`. Chat lines may be
  multi-turn and carry `tools` and `chat_template_kwargs` (only the template's own variables). Completions take a
  string or a list of token ids.
- The admin renders each line with the revision's chat template and stores **token ids**: miners never render.
- Lines are grouped by sampling (`temperature, top_p, top_k, min_p, max_tokens, seed, n`); each group is one set,
  hence one job.
- A bad line is an error line (`batch-uploads/<upload_id>/errors.jsonl` in the platform bucket), never a failed file.
  Refused per line: greedy, `logit_bias`, penalties, logprobs, `stop`, `stream`, structured output, another model.
- Limits: 50,000 lines, 200 MB per file. Idempotent on the request.
- `purpose: "eval"` needs a `reference` per line and a grader of `orders.graders` (`boxed_math_v1`,
  `exact_match_v1`; listed by `GET /admin/v1/architectures`).

The answer lists the groups (`set_id`, `prompt_source`, `count`, `sampling`, prompt token counts), the error count
and file, and a preview of the first rows.

## 2. The model

A forced-draw order names `model` and `revision` (40 hex). The admin pins it from its own files
(`reliquary/orders/models.py`), without downloading weights:

- the architecture, which must be in the preset (`SUPPORTED_ARCHITECTURES`);
- the eos token;
- the checkpoint fingerprint, computed from the hub's sha256 of the `*.safetensors` shards (the value miners and
  validators compute over the downloaded files);
- the weight size (at most `MAX_CHECKPOINT_BYTES`) and the context the config declares
  (`max_prompt_tokens + max_new_tokens` must fit).

There is **no qualification** for these orders: the forced draw's thresholds are job parameters (`judge`), so
there is nothing to calibrate per model. Whether an architecture runs alike in miners' vLLM and the validator's
forward is checked once per architecture, before it enters the preset.

## 3. Creating the job (`POST /admin/v1/jobs`)

```json
{"job_id": "order-gen-<id>" | "order-eval-<id>", "model": "org/name", "revision": "<40-hex>",
 "eval_set_id": "<set_id>" | null, "env": "<catalog env>" | "upload",
 "prompt_start": 0, "prompt_count": 1000, "samples_per_prompt": 4,
 "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
 "max_new_tokens": 4096, "thinking": false, "draw": "gumbel_max",
 "judge": {"floor_rate": 0.98}, "seed": 7, "max_prompt_tokens": 32768}
```

- `draw` without `qualification_id` selects this path. The older qualified path is unchanged.
- On an upload's group, the order repeats the group's sampling, `max_tokens`, `seed` and `n`
  (`upload_sampling_differs` otherwise).
- An evaluation needs a set with references (ours, or an upload with `purpose: "eval"`).
- The admin writes an order record (`reliquary/order-job/v2`: kind, pin, source, conditions); the order control
  serves the job only if it matches that record (`eval_control.pinned_order_refusal`).

## 4. Audits

The order control loads no model: each audit batch goes to a GPU executor of the order's model, which
scores the forced draw and decides it. `RELIQUARY_ORDER_AUDIT_SCORERS` (default 1) sets how many
executors score each batch; with more than one they must agree.

An operator can set `reuse_auditors_only` on a pinned draw job with a frozen prompt set.
In single-scorer mode it can borrow the existing draw auditor only when no paid audit is waiting
or running on that executor. Its backlog, results and lease time do not contribute to fleet sizing.
The control reports support and the eligible executor identity in its status response.

## 5. Grading and reports

- One set: `POST /admin/v1/evaluations/{eval_id}/grade` with `source: "job"`, as before. The report's
  verification says `sampling_verified: true`, the judge, the draw's randomness and the model pin.
- Several sets: one job and one grading per set, then `POST /admin/v1/evaluations/{eval_id}/combine
  {"parts": [<eval ids>]}` writes one report: the parts' environments side by side, the macro over all of them,
  and each part's manifest hash.

## 6. Delivery (`POST /admin/v1/jobs/{job_id}/deliveries`, `schema_version: 3`)

One delivery for every order (`reliquary/corpus/delivery.py`); the eval grading reads the same passing samples
(`passing_batches`). Version 2 stays the default until the platform asks for 3.

- `format`: `parquet` (one row per completion) or `openai_jsonl` (one line per `custom_id`, its samples as
  `choices`, a `chat.completion` or `text_completion` body by the set's endpoint).
- Each row: ids (`custom_id`, `problem_id`, `prompt_index`, `completion_index`), the draw (`draw_slot`,
  `sample_index`, `seed`), token ids and text, `finish_reason`, `audited` and `draw_match`.
- `include_truncated` (default true: a length-cut answer is a valid sample at that cap), `filter`
  (`none` | `annotate` | `drop`, with the env's grader), `dedupe`.
- `errors-NNNNN.jsonl`: every prompt short of its samples (`draw_unfillable`, `missing_samples`).
- `report.json`: counts, the verification block (draw key, judge, audited share, refusals), and the billing basis:
  verified tokens of delivered rows.

## 7. Not yet

- Agentic environments (episodes, sandboxes) as order sources: the source is designed to take them; the
  suite-harness draw is the missing piece.
- Logprobs computed by the validator, Hugging Face push.
- Architectures beyond the preset (Llama, Mistral, Gemma) after a GPU check of each.
