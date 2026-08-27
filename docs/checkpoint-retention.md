# Checkpoint Retention

Reliquary publishes frequently because the serving policy must not become stale;
it does not need to retain every publication forever. The detached trainer uses
a run-local `publication_seq` to separate those two concerns.

## Default production policy

| data | default retention | purpose |
|---|---:|---|
| R2 publications 1–50 | every checkpoint, permanent | fine-grained analysis of the unstable start of a run |
| HF run-start marker | publication 50 as a single rooted branch tip | proves the protected series was closed before compaction |
| HF current block | up to 50 checkpoints on `main` | live serving and recent debugging |
| HF previous block | one temporary grace branch, up to 50 checkpoints | rollback after a compaction |
| R2 serving mirror | current + previous publication | validator download race safety |
| R2 evaluation candidates | publications 50, 100, 150, …; newest 15 | routine benchmark queue |
| R2 milestones | publications 250, 500, 750, …; permanent | sparse long-term lineage |
| R2 ledger | one small JSON object per publication | auditability after weights are pruned |

With an 8 GB snapshot, the steady-state HF history is at most about 0.8 TB per
active run: one run-start marker + 50 rollback + 50 live checkpoints. It can
briefly be one checkpoint larger while a new root is committed. R2 holds about
400 GB of dense run-start history and about 120 GB of rolling benchmark
candidates per run, plus sparse permanent milestones.

Publication still occurs at the configured behavior-policy cadence (currently
every 16 trained windows). Do not reduce publication frequency to solve storage:
that increases policy staleness and changes training behavior.

At publications 51, 101, 151, … the worker performs this guarded transaction:

1. Verify that HF `main` is still the worker's expected revision.
2. Verify that all first-50 R2 manifests exist, then protect the outgoing
   history with a branch. Publication 51 creates the run-start branch; later
   boundaries create a temporary grace branch.
3. Upload the new checkpoint.
4. Super-squash `main` and use its new immutable SHA everywhere downstream.
5. Upload the R2 mirror, optional evaluation snapshot, and ledger.
6. Write the candidate manifest last; this is the serving commit point.
7. Reduce the R2-archived run-start branch to its publication-50 root. Keep only
   the newest grace branch and the newest 15 R2 candidates.

The worker checks total visible model, dataset, and Space storage for the HF
namespace before each upload and freezes before the default 11.5 TB safety
ceiling. A moved HF head also fails closed. Compaction never runs in shadow mode.

Use a fresh HF repository for each training run. A Git branch retains all of its
ancestors; the R2-first design can eventually remove them from a reused repo,
but a fresh repository makes quota accounting and incident recovery clearer.

## Configuration

Set these on the live detached trainer:

```dotenv
RELIQUARY_CHECKPOINT_RETENTION_ENABLED=1
RELIQUARY_HF_RETENTION_KEEP_INITIAL=50
RELIQUARY_HF_RETENTION_CANDIDATE_INTERVAL=50
RELIQUARY_HF_RETENTION_MILESTONE_INTERVAL=250
RELIQUARY_HF_RETENTION_MAX_GRACE_BRANCHES=1
RELIQUARY_R2_EVALUATION_CANDIDATES_TO_KEEP=15
RELIQUARY_HF_STORAGE_FREEZE_TB=11.5
```

The settings are explicit because HF history rewriting is irreversible. The
bounded controller is implemented in the detached trainer; do not operate the
in-process fallback as a long-running publisher with retention enabled.

R2 lifecycle rules can be added as defense in depth, but they must target only
`reliquary/checkpoint-candidates/`. Never apply an expiry rule to
`reliquary/checkpoint-milestones/`, the publication ledger, or training payloads.

## Enabling retention on an existing run

Freeze the trainer first so the audited head cannot move:

```bash
RELIQUARY_DISABLE_TRAIN=1
HF_TOKEN=... python scripts/prepare_hf_retention.py \
  --repo-id ReliquaryForge/<repo> \
  --run-id '<exact-RELIQUARY_TRAINING_RUN_ID>'
```

The tool normally finds the contiguous run boundary from each checkpoint's
embedded `training_run_id` and requires a trained-window cursor, which excludes
the base-reset commit. If legacy metadata prevents inference, pass the SHA of
the first **trainer publication** (not the base reset) as `--start-revision`.
Review the JSON, then create the protected branch using the reported head:

```bash
HF_TOKEN=... python scripts/prepare_hf_retention.py \
  --repo-id ReliquaryForge/<repo> \
  --run-id '<exact-RELIQUARY_TRAINING_RUN_ID>' \
  --expected-head '<reported-head>' \
  --apply
```

The script is read-only unless `--apply` is present. Even with `--apply`, it only
creates the protected branch; it never deletes or squashes history. Put the
reported `RELIQUARY_TRAINER_PUBLICATION_SEQ` and the configuration above into
the trainer environment, unfreeze, and restart it. The sequence is then stored
inside every checkpoint profile, R2 ledger entry, and candidate manifest, so the
migration environment override can be removed after the first successful
publication.

A legacy run did not upload its first 50 snapshots to the new R2 prefix. The
protected branch keeps them safe, but on a reused repository it also keeps older
ancestors billable. Backfill it with the resumable command below (dry-run first):

```bash
python scripts/backfill_checkpoint_run_start.py \
  --repo-id ReliquaryForge/<repo> \
  --run-id '<exact-RELIQUARY_TRAINING_RUN_ID>'

python scripts/backfill_checkpoint_run_start.py \
  --repo-id ReliquaryForge/<repo> \
  --run-id '<exact-RELIQUARY_TRAINING_RUN_ID>' \
  --apply
```

After all 50 R2 manifests and at least one benchmark have been verified, root
the protected branch using the publication-50 SHA reported by the preparation
tool. This is the explicit irreversible step:

```bash
python scripts/backfill_checkpoint_run_start.py \
  --repo-id ReliquaryForge/<repo> \
  --run-id '<exact-RELIQUARY_TRAINING_RUN_ID>' \
  --apply \
  --root-protected-branch \
  --expected-branch-target '<reported-first-history-revision>'
```

The backfill is resumable at the per-publication manifest and uses a temporary
directory per snapshot, so it needs room for one checkpoint rather than all 50.
Rooting the branch removes its detailed HF ancestry only after the complete
hash-bound R2 copy exists. HF may reclaim unreachable storage asynchronously.

Before any later manual deletion, confirm that the first-history branch resolves,
the current candidate manifest downloads, and a recent R2 evaluation snapshot
passes `scripts/screen_recovery_checkpoints.py`.

## Benchmarking workflow

The `reliquary/checkpoint-run-start/` R2 prefix is the detailed early-run series.
After that, benchmark the R2 snapshots selected every 50 publications. Promote
a winner or anomaly to the milestone namespace before the rolling queue removes
it, and record the benchmark artifact against the ledger's `repo_id`, `revision`,
`publication_seq`, hashes, training cursor, and generation-contract identity.

Benchmark selection should be based on a fixed, revision-bound holdout and the
same runtime/profile contract, not the training reward alone. Keep the best
checkpoint, important regressions, and release checkpoints; ordinary candidates
can expire.
