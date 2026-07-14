# Eval Dashboard Producer

The eval-dashboard worker measures published model checkpoints. It is not part
of validator admission, miner scoring, training, or the flat window archive.
A stopped evaluator makes the dashboard stale, but cannot change emissions or
reject a miner.

## Safety contract

Every run is partitioned by the SHA-256 of its complete effective contract:

- exact base repository and immutable revision;
- checkpoint repository lineage;
- one immutable tokenizer/chat-template source shared by the base and every
  checkpoint;
- sealed math and code holdout hashes and ordered task-id hashes;
- contamination-review hashes and pinned training-source revisions;
- prompt/chat-template hash;
- generation and BFT settings;
- grader and producer revisions;
- Python, Torch, Transformers, Datasets and Hugging Face Hub versions;
- attention implementation and deterministic batch size (`1`).
- stable execution hardware (device type, GPU model, memory and compute
  capability).

Changing any item creates a new `config_hash`. Results from different hashes
are never placed on one curve.

The worker publishes immutable checkpoint records and content-addressed
dashboard snapshots. `eval_dashboard/index.json` moves only after every object
it references has been read back and hash-verified.

```text
eval_dashboard/index.json
eval_dashboard/status.json
eval_dashboard/runs/<config_hash>/manifest.json
eval_dashboard/runs/<config_hash>/checkpoints/checkpoint-<n>-<revision>.json
eval_dashboard/runs/<config_hash>/publications/<sha256>/dashboard.json
eval_dashboard/runs/<config_hash>/publications/<sha256>/manifest.json
eval_dashboard/runs/<config_hash>/dashboard.json
```

The run-level manifest seals the effective configuration. Each publication
manifest references that config, the immutable dashboard, every checkpoint
result digest, coverage, wall-clock timing, and GPU/runtime metadata. The final
path is a compatibility mirror; the content-addressed paths recorded in the
index are authoritative.

`generated_at` is the successful publication heartbeat used for freshness.
`evidence_completed_at` is the completion time of the latest GPU evaluation.
The independent monitor also compares the published checkpoint number and
revision with the validator, so periodically republishing an old result cannot
hide checkpoint drift.

## Holdout format

Holdouts are canonical JSONL files kept on the evaluator host. They are not
published. Public checkpoint artifacts contain only task ids, prompt hashes,
rewards, lengths, termination flags, seeds, and completion hashes.
Use opaque task ids that do not reveal private benchmark prompts.

Math row:

```json
{"task_id":"math-001","prompt":"...","ground_truth":"42"}
```

Code row:

```json
{"task_id":"code-001","prompt":"...","cases":[{"entry":{"kind":"function","name":"solve"},"args":[1],"kwargs":{},"expected":2}]}
```

Code is evaluated by the existing gVisor grader. The grader service must be
running on the configured Unix socket. Never execute generated code directly
in the evaluator process.

Do not build a holdout from `nvidia/OpenMathInstruct-2` or
`R0mAI/opencodeinstruct-curated` without excluding it from training. Prefer a
separate, fixed private benchmark artifact and review it against both pinned
training sources.

## Contamination review

Run the audit once per domain against exports of every exact pinned training
source. It checks normalized exact hashes and near-duplicate token shingles,
and records each scanned shard's SHA-256 and row count. Repeat
`--training-source REPO REVISION PROMPT_FIELD FILE` for every shard from both
the math and code training corpora.

```bash
python -m scripts.audit_eval_holdout \
  --domain math \
  --holdout /srv/reliquary-eval/holdouts/math.jsonl \
  --training-source nvidia/OpenMathInstruct-2 \
    469216e3f46f4dacf476b382e192485ea51a143e problem \
    /srv/training-snapshots/openmath-000.parquet \
  --training-source R0mAI/opencodeinstruct-curated \
    d3caaefc3b46f8642b251f9efaeccf0d1e95b0a7 input \
    /srv/training-snapshots/opencode-000.parquet \
  --reviewer reliquary-validator-ops \
  --reviewed-at 2026-07-14T00:00:00Z \
  --output /srv/reliquary-eval/holdouts/math-review.json
```

The command exits `2` when it finds overlap. Record its three printed hashes in
the configuration only after an `approved` result. The revisions above are the
exact source revisions deployed on the validator at the time of this runbook;
refresh and pin them whenever either training source changes. Never use a
branch name.

## Configuration

Store the reviewed configuration at `/etc/reliquary/eval/config.json`. Hash
placeholders below must be replaced with real 64-character values. The
`grader_revision` is the reviewed Reliquary commit deployed on the worker.

```json
{
  "schema_version": "1",
  "lineage": {
    "lineage_id": "qwen3.5-2b-reliquary-v2",
    "base": {
      "repo_id": "Qwen/Qwen3.5-2B",
      "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
      "checkpoint_n": 0
    },
    "checkpoint_repo_id": "ReliquaryForge/qwen3.5-2b-reliquary-v2"
  },
  "tokenizer_source": {
    "repo_id": "Qwen/Qwen3.5-2B",
    "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc"
  },
  "math_holdout": {
    "domain": "math",
    "dataset_repo_id": "ReliquaryForge/reliquary-eval-v2",
    "dataset_revision": "<immutable-40-hex-revision>",
    "split": "math",
    "artifact_sha256": "<64-hex-hash>",
    "task_ids_sha256": "<64-hex-hash>",
    "contamination_review_sha256": "<64-hex-hash>",
    "n_prompts": 500,
    "grader_id": "reliquary.openmath._compute_omi_reward",
    "grader_revision": "<reviewed-40-hex-reliquary-commit>",
    "format_version": "1"
  },
  "code_holdout": {
    "domain": "code",
    "dataset_repo_id": "ReliquaryForge/reliquary-eval-v2",
    "dataset_revision": "<immutable-40-hex-revision>",
    "split": "code",
    "artifact_sha256": "<64-hex-hash>",
    "task_ids_sha256": "<64-hex-hash>",
    "contamination_review_sha256": "<64-hex-hash>",
    "n_prompts": 500,
    "grader_id": "reliquary.opencode.gvisor",
    "grader_revision": "<reviewed-40-hex-reliquary-commit>",
    "format_version": "1"
  },
  "generation": {
    "protocol_parity": true,
    "samples_per_prompt": 4,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "batch_size": 1,
    "seed_salt": "<stable-public-seed-at-least-16-chars>",
    "math_max_new_tokens": 32768,
    "math_bft_enabled": true,
    "math_thinking_budget": 2048,
    "math_answer_budget": 512,
    "math_force_template": "</think>\n\nFinal Answer: \\boxed{",
    "code_max_new_tokens": 32768
  },
  "publish_interval_windows": 10,
  "schedule": {
    "owner": "reliquary-validator-ops",
    "cadence_seconds": 3600,
    "overdue_seconds": 21600,
    "retry_attempts": 4,
    "retry_base_seconds": 30
  }
}
```

Metric definitions preserve the existing dashboard contract:

- `pass@1`: reward of sample index 0, averaged across prompts;
- `pass@k`: best reward among the fixed k samples, averaged across prompts;
- `pass_avg`: reward across every prompt/sample pair.

Sample index 0 is stable because its seed is derived from the effective config
hash, task id, and sample index. Do not compare the new v2 curve numerically to
the retired curve unless its private holdout and full generation contract are
proven identical.

## Dry run

Use a filesystem store before granting R2 write access:

```bash
python -m scripts.run_eval_dashboard target \
  --store-root /tmp/reliquary-eval-r2 \
  --config /etc/reliquary/eval/config.json \
  --math-holdout /srv/reliquary-eval/holdouts/math.jsonl \
  --math-review /srv/reliquary-eval/holdouts/math-review.json \
  --code-holdout /srv/reliquary-eval/holdouts/code.jsonl \
  --code-review /srv/reliquary-eval/holdouts/code-review.json \
  --state-dir /srv/reliquary-eval/state \
  --model-repo-id ReliquaryForge/qwen3.5-2b-reliquary-v2 \
  --revision <current-immutable-revision> \
  --checkpoint-n <current-checkpoint> \
  --window <current-window>
```

The command evaluates the locked base first, then the requested checkpoint.
Completed GPU results are cached locally before publication, so an R2 failure
does not force another expensive generation run.

## Production schedule and alerting

Install the service/timer files from `docker/eval-dashboard/` on the GPU host.
Install the monitor timer on an always-on CPU host as well. That independent
monitor is what reports an overdue dashboard when the GPU machine itself is
off.

The worker writes `eval_dashboard/status.json`, logs failures to journald, exits
non-zero after bounded retries, and optionally POSTs failure/overdue JSON to
`RELIQUARY_EVAL_ALERT_WEBHOOK`. The monitor reads the immutable publication and
config manifests, verifies their hashes, and compares the published checkpoint
number and revision with `$RELIQUARY_VALIDATOR_URL`.

The first production gate is:

1. Run a small locked smoke subset without R2 publication.
2. Run the full base and current v2 checkpoint on the same GPU.
3. Re-run a fixed 32-task subset and compare against the declared tolerance.
4. Confirm the immutable objects and index hashes by readback.
5. Confirm the web API reports `fresh` and the exact v2 revision.

The bounded replay is:

```bash
python -m scripts.run_eval_dashboard replay \
  --validator-url "$RELIQUARY_VALIDATOR_URL" \
  --config /etc/reliquary/eval/config.json \
  --math-holdout /srv/reliquary-eval/holdouts/math.jsonl \
  --math-review /srv/reliquary-eval/holdouts/math-review.json \
  --code-holdout /srv/reliquary-eval/holdouts/code.jsonl \
  --code-review /srv/reliquary-eval/holdouts/code-review.json \
  --state-dir /srv/reliquary-eval/state \
  --n-prompts 32 \
  --tolerance 0.02
```

Replay checks `pass@1`, best-of-k, all-sample average, truncation rate, and the
math forced-termination rate. Score tolerance is expressed on `[0, 1]`; rate
tolerance is the corresponding percentage-point value. It also reports the
exact completion-hash match rate without requiring bitwise GPU determinism.
