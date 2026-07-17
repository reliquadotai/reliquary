# Operator runbook: rollout token invariant fix

Date: 2026-05-25

Audience: validator operators and protocol maintainers.

## Immediate action

1. Deploy the patched validator.
2. Restart validator processes so `/submit` uses the new schema/runtime guard.
3. Watch logs and `/verdicts` for `tokens_mismatch`.
4. Audit recent archives for historical contamination.
5. Decide whether to correct scoring, archives, or checkpoint lineage.

If a deploy cannot happen immediately, temporarily stop accepting `/submit`
traffic or pause the validator. This bug affects the validator's trust boundary,
so rate limits and existing GRAIL checks are not sufficient mitigations.

## Patch checklist

The patched validator must satisfy all of these:

- `RolloutSubmission` rejects `tokens != commit["tokens"]`.
- The batcher rejects bypassed/mutated objects with `RejectReason.TOKENS_MISMATCH`.
- Reward decoding uses `commit["tokens"]`.
- R2 archive output uses `commit["tokens"]`.
- GRPO training uses `commit["tokens"]`.

Run:

```bash
python3 -m pytest \
  tests/unit/test_batch_submission_schema.py \
  tests/unit/test_grpo_window_batcher.py \
  tests/unit/test_archive_window_content.py \
  tests/unit/test_training_rollout_loss.py -q
```

Expected: all tests pass.

## Post-deploy smoke tests

Submit or construct a deliberately mismatched rollout in a local/test validator:

```python
RolloutSubmission(
    tokens=[1, 2, 3],
    reward=1.0,
    commit={"tokens": [1, 2, 3, 4], ...},
)
```

Expected behavior:

- normal Pydantic path rejects during request parsing;
- bypassed `model_construct` path is rejected by the batcher with
  `tokens_mismatch`;
- no reward eval, GRAIL, archive, or training path observes the outer tokens.

## Archive audit

Historical R2 archives do not include `commit["tokens"]`, so you cannot prove
the invariant from archives alone. You can still detect strong symptoms.

High-confidence indicators:

- `len(archive_rollout["tokens"]) < archive_rollout["completion_length"]`
- last token is not EOS and the completion is very short
- last token distribution dominated by token `7810` (`}.`)
- completion text is only a final answer, commonly `\boxed{...}.`
- sigma modes cluster exactly at `0.4330127`, `0.4841229`, or `0.5`
- high reward rate on OpenMathInstruct with no reasoning text

Minimal audit sketch:

```python
import json, ssl, urllib.request
from collections import Counter

HOTKEY = "5DAvrWnM8MygaYq5dyC1bs8tP71Pd8d5QF5yRBN36ggkEzPe"
EOS = {151645, 151643}
ctx = ssl._create_unverified_context()

tot = 0
last = Counter()
short = 0
rewards = Counter()
sigmas = Counter()

for window in range(5710, 5781):
    url = f"https://www.reliqua.ai/api/r2/window/{window}"
    with urllib.request.urlopen(url, timeout=40, context=ctx) as r:
        archive = json.loads(r.read().decode("utf-8"))["data"]
    for entry in archive["batch"]:
        if entry["hotkey"] != HOTKEY:
            continue
        sigmas[round(float(entry["sigma"]), 6)] += 1
        for rollout in entry["rollouts"]:
            tokens = rollout["tokens"]
            tot += 1
            rewards[rollout["reward"]] += 1
            if tokens:
                last[tokens[-1]] += 1
            if tokens and len(tokens) < int(rollout["completion_length"]):
                short += 1

print("rollouts", tot)
print("last", last.most_common())
print("eos", sum(last[t] for t in EOS))
print("short_vs_completion", short)
print("rewards", rewards)
print("sigmas", sigmas)
```

Use slower concurrency or cached downloaded archives. The public API can return
`403` or `429` under probing load.

## Scoring remediation

The exploit awards batch participation/emission credit to submissions that did
not prove the same text they were rewarded for. Recommended policy:

1. Mark affected `(window, hotkey, prompt_idx, merkle_root)` entries as invalid
   in a correction manifest.
2. Recompute EMA/weights excluding invalid entries.
3. Publish the correction manifest next to the immutable original archives.
4. Do not silently mutate old archives unless the protocol already treats them
   as mutable; prefer append-only correction records.

Suggested correction record shape:

```json
{
  "incident": "2026-05-25-rollout-token-invariant",
  "window_start": 5710,
  "hotkey": "5DAvrWnM8MygaYq5dyC1bs8tP71Pd8d5QF5yRBN36ggkEzPe",
  "prompt_idx": 40765,
  "reason": "tokens_mismatch_forensic",
  "evidence": {
    "archive_tokens_shorter_than_completion_length": true,
    "last_token": 7810,
    "sigma": 0.4330127018922193
  }
}
```

## Training/checkpoint remediation

Do not assume all affected windows changed model weights. For the observed
short-token variant, pre-patch training likely skipped rollouts with logprob
length mismatch. Check:

- W&B `train/rollouts_processed` and `train/valid_rollout_ratio`.
- validator logs for `rollout skipped: log-prob length mismatch`.
- whether `RELIQUARY_DISABLE_TRAIN` was set.
- whether checkpoints were published after affected windows.

If affected windows had nonzero processed rollouts and checkpoints were
published, consider:

1. Freeze publishing.
2. Roll back to the last checkpoint before the affected span.
3. Replay clean windows or resume from a known-good checkpoint.
4. Publish a checkpoint lineage note explaining the rollback.

If training processed zero affected rollouts, scoring/archive correction may be
sufficient.

## Base-model reset procedure

If the safest recovery is to restart from the base model, do it as a forward HF
checkpoint instead of trying to downgrade miners. The trainer auto-resumes the
latest `checkpoint N` commit on startup, and miners only pull checkpoints when
`checkpoint_n` increases. Therefore the reset should publish the base model as
`checkpoint N+1`, then pin the trainer to that new commit.

From the validator host, after pulling an image that includes PR #41:

```bash
cd reliquary/docker
docker compose -f docker-compose.trainer.yml pull
docker compose -f docker-compose.trainer.yml stop reliquary-trainer

docker compose -f docker-compose.trainer.yml run --rm --no-deps \
  --entrypoint python reliquary-trainer \
  /opt/reliquary/scripts/publish_base_reset_checkpoint.py \
  --source-model Qwen/Qwen3.5-2B \
  --source-revision <approved-40-character-base-commit>
```

The script prints:

```bash
RELIQUARY_RESUME_FROM=sha:<new-base-reset-commit>
RELIQUARY_WANDB_VERSION=base-reset-20260525
```

Add those lines to `docker/.env`, then restart:

```bash
docker compose -f docker-compose.trainer.yml up -d --force-recreate reliquary-trainer
docker logs -f reliquary-trainer
curl -s http://localhost:8080/state | jq '{state, window_n, checkpoint_n, checkpoint_revision}'
```

Expected: `checkpoint_revision` equals the commit printed by the script.
If it does not, stop the trainer and fix the env pin before accepting miner
traffic.

## Communication guidance

Recommended private message:

```text
We confirmed a validator payload invariant bug: RolloutSubmission carried two
token arrays and pre-patch code used them inconsistently across proof, reward,
archive, and training paths. A patch is deployed/being deployed to enforce
tokens == commit.tokens and to use commit.tokens as the canonical source.

We also found strong archive evidence for exploitation by hotkey <hotkey>.
We are auditing affected windows and will publish a correction/remediation plan.
```

Avoid saying "UID 1" without a timestamped chain snapshot. Use hotkeys as the
primary identity.
