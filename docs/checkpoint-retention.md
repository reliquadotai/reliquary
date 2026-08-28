# Checkpoint Retention

Reliquary keeps an active training run append-only on Hugging Face. Retention is
an operator action for a finished, retired run; it is never triggered by the
trainer, a timer, or a bucket lifecycle rule.

## Lifecycle

| state | Hugging Face | R2 | allowed action |
|---|---|---|---|
| running | every published checkpoint | current + previous serving mirror | publish only |
| benchmark cooldown | unchanged; every checkpoint remains downloadable | unchanged | test for as long as needed |
| archived | unchanged | selected recovery snapshots with SHA-256 manifests | verify selected checkpoints |
| finalized | final checkpoint as one rooted commit | selected recovery snapshots | old HF history can be reclaimed |

Stopping training causes no state transition beyond the operator deciding that
the run is in benchmark cooldown. There is no deadline. Do not finalize a
repository that is still configured as a trainer repository, serving source,
rollback source, or external download dependency.

Use one fresh HF repository per run. This makes the run boundary unambiguous and
lets an old run be finalized without touching the current run.

## What to retain

Choose the set only after benchmarks are complete. For a research run, the
recommended starting policy is:

- the first 50 publications when dense early-learning analysis is valuable;
- every 50th publication after that;
- the final checkpoint;
- benchmark winners, regressions, anomalies, and release checkpoints.

This is a selection policy, not an automatic copy of the run. With 500 equal
checkpoints, retaining the first 50, every later 50th checkpoint, and the final
keeps about 60 snapshots instead of 500: roughly an 88% reduction. If the early
series has already been analyzed and has no continuing value, keep fewer of it.
R2 stores only the selected snapshots; it does not receive the full run.

## Active-run quota guard

The detached publisher performs one optional read-only check before each HF
upload. It sums visible model, dataset, and Space storage for the organization
and refuses the next upload if its projected size reaches the ceiling:

```dotenv
RELIQUARY_HF_STORAGE_FREEZE_TB=11.0
```

The guard does not delete, branch, squash, or otherwise mutate HF or R2. An
unset value disables it. It is an incident brake, not a retention mechanism;
leave enough headroom to finish a checkpoint safely and then retire an old run.

## Manual finished-run procedure

First stop the old trainer and confirm the repository is no longer used for
serving or downloads. Keep `RELIQUARY_HF_REPO_ID` set to the current live repo
when running the command: the tool refuses to target that repository.

### 1. Audit with a dry run

Pass the exact checkpoint numbers selected after benchmarking, including the
final checkpoint. Dry-run is read-only and reports the current HEAD and estimated
reclaimable storage:

```bash
python scripts/archive_finished_hf_run.py \
  --repo-id ReliquaryForge/<finished-repo> \
  --checkpoints '<first-50-and-selected-later-checkpoints>'
```

Record `source_head` from the JSON and review every selected revision. The
newest selected checkpoint must be the repository HEAD.

### 2. Archive and verify the selected snapshots

This phase writes R2 only. Each checkpoint is downloaded by immutable HF SHA,
hashed, uploaded under `reliquary/checkpoint-milestones/`, and verified before
its manifest becomes the completion marker. It is resumable.

```bash
python scripts/archive_finished_hf_run.py \
  --repo-id ReliquaryForge/<finished-repo> \
  --checkpoints '<same-exact-list>' \
  --expected-head '<source_head>' \
  --confirm-finished \
  --apply
```

After this command, HF is still unchanged. Benchmark or recovery-test the R2
copies for as long as desired.

### 3. Finalize in a separate invocation

Only after the archive phase and final operator approval, run:

```bash
python scripts/archive_finished_hf_run.py \
  --repo-id ReliquaryForge/<finished-repo> \
  --checkpoints '<same-exact-list>' \
  --expected-head '<same-source_head>' \
  --confirm-finished \
  --apply \
  --squash
```

Squash mode refuses to create a missing archive: every selected R2 manifest and
object must already exist and verify. Immediately before the irreversible HF
operation it rechecks HEAD and branches. Afterwards it requires exactly one HF
commit, verifies that the final HF file tree is unchanged, and compares the
rooted final snapshot byte-for-byte with its R2 recovery copy.

The final model gets a new rooted HF commit SHA. Old checkpoint SHAs become
unavailable after HF garbage collection; this is intentional and is why a repo
must be retired first. Selected historical weights remain recoverable from R2
with their original source SHA and per-file SHA-256 recorded in the manifest.
HF may report reclaimed storage asynchronously.

## Safety rules

- Never run the command against the live or current serving repository.
- Never add a branch to preserve old HF commits; a branch retains their billable
  ancestry and defeats reclamation.
- Never put a lifecycle expiry rule on
  `reliquary/checkpoint-milestones/` or `reliquary/checkpoint-ledger/`.
- Keep the exact dry-run checkpoint list and `source_head` for both apply phases.
- Finalize old repositories one at a time, starting with the oldest completed
  run, and verify reported HF storage before proceeding to the next.
