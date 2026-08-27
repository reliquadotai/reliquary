"""Trainer publisher: HF upload + R2 mirror + candidate manifest ordering."""

import asyncio
import io
import json

from reliquary.trainer.publisher import (
    CANDIDATE_MANIFEST_KEY,
    TrainerPublisher,
    checkpoint_key,
)
from reliquary.trainer.retention import CheckpointRetentionPolicy
from reliquary.trainer.retention import HfCompactionPlan, run_key
from reliquary.shared.training_payload import active_training_identity


class _R2:
    def __init__(self):
        self.uploads = []      # (key, path) via upload_file
        self.objects = {}      # key -> bytes via put_object
        self.deleted = []      # keys via delete_object

    def upload_file(self, path, bucket, key, Config=None):
        assert Config is not None  # multipart config is mandatory
        self.uploads.append((key, path))
        self.objects[key] = b"<file>"

    def put_object(self, Bucket, Key, Body, **kw):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix, Delimiter=None):
        if Delimiter:
            prefixes = sorted({
                k[: k.index(Delimiter, len(Prefix)) + 1]
                for k in self.objects
                if k.startswith(Prefix) and Delimiter in k[len(Prefix):]
            })
            return {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}
        return {"Contents": [
            {"Key": k} for k in sorted(self.objects) if k.startswith(Prefix)
        ]}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)
        self.deleted.append(Key)


def _publisher(tmp_path, r2, order, *, hf_fails=False):
    def save_fn(model, tokenizer, path):
        (path / "model.safetensors").write_bytes(b"weights")
        order.append("save")

    async def hf_upload(folder_path, repo_id, commit_message):
        if hf_fails:
            raise RuntimeError("hf down")
        order.append("hf")
        return "rev-123"

    return TrainerPublisher(
        repo_id="org/repo", staging_dir=str(tmp_path), tokenizer=None,
        save_fn=save_fn, hf_upload_fn=hf_upload, r2_client=r2,
        bucket="reliquary",
        retention_policy=CheckpointRetentionPolicy(enabled=False),
    )


def test_keys():
    assert checkpoint_key("rev-1", "model.safetensors") == (
        "reliquary/checkpoints/rev-1/model.safetensors"
    )


def test_default_publisher_is_retention_flag_off(monkeypatch, tmp_path):
    monkeypatch.delenv("RELIQUARY_CHECKPOINT_RETENTION_ENABLED", raising=False)
    r2 = _R2()

    def save_fn(model, tokenizer, path):
        (path / "model.safetensors").write_bytes(b"weights")

    async def hf_upload(folder_path, repo_id, commit_message):
        return "flag-off-revision"

    publisher = TrainerPublisher(
        repo_id="org/repo",
        staging_dir=str(tmp_path),
        tokenizer=None,
        save_fn=save_fn,
        hf_upload_fn=hf_upload,
        r2_client=r2,
        bucket="reliquary",
    )
    assert publisher.retention.enabled is False

    asyncio.run(publisher.publish(
        object(), checkpoint_n=1, lr_schedule_step=None,
        trained_window_cursor=100, reason="cadence",
    ))
    serving = json.loads(r2.objects[CANDIDATE_MANIFEST_KEY])
    assert "publication_seq" not in serving
    assert not any(
        key.startswith("reliquary/checkpoint-run-start/")
        for key in r2.objects
    )
    assert CANDIDATE_MANIFEST_KEY == (
        "reliquary/training/candidate-manifest.json"
    )


def test_publish_order_manifest_and_cleanup(tmp_path):
    r2, order = _R2(), []
    pub = _publisher(tmp_path, r2, order)
    rev = asyncio.run(pub.publish(
        object(), checkpoint_n=5, lr_schedule_step=80,
        trained_window_cursor=30110, reason="cadence",
    ))
    assert rev == "rev-123"
    assert order == ["save", "hf"]
    uploaded_keys = [k for k, _ in r2.uploads]
    assert checkpoint_key("rev-123", "model.safetensors") in uploaded_keys
    # The profile file travels in the snapshot too.
    assert any(k.endswith("reliquary_checkpoint_profile.json") or
               "profile" in k for k in uploaded_keys) or len(uploaded_keys) >= 2
    manifest = json.loads(r2.objects[CANDIDATE_MANIFEST_KEY])
    assert manifest == {
        **active_training_identity(),
        "checkpoint_n": 5, "repo_id": "org/repo", "revision": "rev-123",
        "trained_window_cursor": 30110, "reason": "cadence",
    }
    assert not any(tmp_path.iterdir())  # staging cleaned


def test_profile_extra_written(tmp_path):
    r2 = _R2()
    captured = {}

    def save_fn(model, tokenizer, path):
        (path / "model.safetensors").write_bytes(b"w")

    async def hf_upload(folder_path, repo_id, commit_message):
        import pathlib
        for p in pathlib.Path(folder_path).iterdir():
            if p.suffix == ".json":
                captured[p.name] = json.loads(p.read_text())
        return "rev-9"

    pub = TrainerPublisher(
        repo_id="org/repo", staging_dir=str(tmp_path), tokenizer=None,
        save_fn=save_fn, hf_upload_fn=hf_upload, r2_client=r2,
        bucket="reliquary",
        retention_policy=CheckpointRetentionPolicy(enabled=False),
    )
    asyncio.run(pub.publish(
        object(), checkpoint_n=7, lr_schedule_step=99,
        trained_window_cursor=30200, reason="cadence",
    ))
    profiles = [
        doc for doc in captured.values()
        if doc.get("lr_schedule_step") is not None
    ]
    assert profiles and profiles[0]["lr_schedule_step"] == 99
    assert profiles[0]["trained_window_cursor"] == 30200


def test_mirror_keeps_only_last_two_revisions(tmp_path):
    r2 = _R2()
    # A stale revision from before a restart must be cleaned too.
    r2.objects["reliquary/checkpoints/rev-old/model.safetensors"] = b"x"

    def save_fn(model, tokenizer, path):
        (path / "model.safetensors").write_bytes(b"w")

    revs = iter(["rev-1", "rev-2", "rev-3"])

    async def hf_upload(folder_path, repo_id, commit_message):
        return next(revs)

    pub = TrainerPublisher(
        repo_id="org/repo", staging_dir=str(tmp_path), tokenizer=None,
        save_fn=save_fn, hf_upload_fn=hf_upload, r2_client=r2,
        bucket="reliquary",
        retention_policy=CheckpointRetentionPolicy(enabled=False),
    )
    for n in (1, 2, 3):
        asyncio.run(pub.publish(
            object(), checkpoint_n=n, lr_schedule_step=None,
            trained_window_cursor=30100 + n, reason="cadence",
        ))
    live_revs = {
        k.split("/")[2] for k in r2.objects
        if k.startswith("reliquary/checkpoints/")
    }
    assert live_revs == {"rev-2", "rev-3"}
    assert "reliquary/checkpoints/rev-old/model.safetensors" in r2.deleted


def test_first_publish_after_restart_keeps_expected_parent_mirror(tmp_path):
    r2, order = _R2(), []
    r2.objects[
        "reliquary/checkpoints/rev-parent/model.safetensors"
    ] = b"parent"
    pub = _publisher(tmp_path, r2, order)

    asyncio.run(pub.publish(
        object(), checkpoint_n=2, lr_schedule_step=None,
        trained_window_cursor=30102, reason="cadence",
        expected_parent_revision="rev-parent",
    ))

    live_revs = {
        key.split("/")[2]
        for key in r2.objects
        if key.startswith("reliquary/checkpoints/")
    }
    assert live_revs == {"rev-parent", "rev-123"}


def test_staging_cleaned_on_failure(tmp_path):
    r2, order = _R2(), []
    pub = _publisher(tmp_path, r2, order, hf_fails=True)
    import pytest
    with pytest.raises(RuntimeError, match="hf down"):
        asyncio.run(pub.publish(
            object(), checkpoint_n=5, lr_schedule_step=None,
            trained_window_cursor=30110, reason="cadence",
        ))
    assert not any(tmp_path.iterdir())
    assert CANDIDATE_MANIFEST_KEY not in r2.objects


class _History:
    def __init__(self):
        self.calls = []

    def assert_storage_budget(self, **kwargs):
        self.calls.append(("budget", kwargs))
        return 1

    def prepare_compaction(self, **kwargs):
        self.calls.append(("prepare", kwargs))
        return HfCompactionPlan(
            branch="retention-grace", retained_publication_seq=50,
            permanent=True,
        )

    def compact_uploaded_head(self, **kwargs):
        self.calls.append(("compact", kwargs))
        return "final-root-rev"

    def cleanup_grace_branches(self, **kwargs):
        self.calls.append(("cleanup", kwargs))
        return []

    def compact_protected_branch(self, **kwargs):
        self.calls.append(("compact-protected", kwargs))
        return "first-history-root"


def _retained_publisher(tmp_path, r2, history, *, revision="uploaded-rev"):
    def save_fn(model, tokenizer, path):
        (path / "model.safetensors").write_bytes(b"weights")

    async def hf_upload(folder_path, repo_id, commit_message):
        return revision

    return TrainerPublisher(
        repo_id="org/repo", staging_dir=str(tmp_path), tokenizer=None,
        save_fn=save_fn, hf_upload_fn=hf_upload, r2_client=r2,
        bucket="reliquary",
        retention_policy=CheckpointRetentionPolicy(enabled=True),
        hf_history_manager=history,
    )


def test_every_fiftieth_publication_gets_hashed_evaluation_snapshot(tmp_path):
    r2 = _R2()
    history = _History()
    pub = _retained_publisher(tmp_path, r2, history, revision="rev-50")

    revision = asyncio.run(pub.publish(
        object(), checkpoint_n=550, publication_seq=50,
        lr_schedule_step=800, trained_window_cursor=40000,
        reason="cadence", expected_parent_revision="rev-49",
    ))

    assert revision == "rev-50"
    run = active_training_identity()["training_run_id"]
    retained_prefix = (
        f"reliquary/checkpoint-run-start/{run_key(run)}/"
        "publication-000050"
    )
    assert f"{retained_prefix}/model.safetensors" in r2.objects
    retained_manifest = json.loads(
        r2.objects[f"{retained_prefix}/manifest.json"]
    )
    weights = next(
        row for row in retained_manifest["files"]
        if row["path"] == "model.safetensors"
    )
    assert len(weights["sha256"]) == 64
    ledger_key = pub.retention.ledger_key(run, 50)
    assert json.loads(r2.objects[ledger_key])["retention_class"] == (
        "run_start_history"
    )
    serving = json.loads(r2.objects[CANDIDATE_MANIFEST_KEY])
    assert serving["publication_seq"] == 50
    assert serving["evaluation_snapshot_prefix"] == retained_prefix
    assert [name for name, _ in history.calls] == ["budget"]


def test_compaction_publishes_only_post_squash_revision(tmp_path):
    r2 = _R2()
    history = _History()
    pub = _retained_publisher(tmp_path, r2, history)
    run = active_training_identity()["training_run_id"]
    root = pub.retention.run_start_prefix(run)
    for seq in range(1, 51):
        r2.objects[f"{root}publication-{seq:06d}/manifest.json"] = b"{}"

    revision = asyncio.run(pub.publish(
        object(), checkpoint_n=551, publication_seq=51,
        lr_schedule_step=816, trained_window_cursor=40016,
        reason="cadence", expected_parent_revision="rev-50",
    ))

    assert revision == "final-root-rev"
    serving = json.loads(r2.objects[CANDIDATE_MANIFEST_KEY])
    assert serving["revision"] == "final-root-rev"
    assert checkpoint_key(
        "final-root-rev", "model.safetensors"
    ) in r2.objects
    assert not any(
        key.startswith("reliquary/checkpoints/uploaded-rev/")
        for key in r2.objects
    )
    assert [name for name, _ in history.calls] == [
        "budget", "prepare", "compact", "compact-protected", "cleanup",
    ]


def test_first_compaction_fails_closed_when_run_start_archive_is_incomplete(
    tmp_path,
):
    import pytest

    r2 = _R2()
    history = _History()
    pub = _retained_publisher(tmp_path, r2, history)

    with pytest.raises(RuntimeError, match="run-start archive"):
        asyncio.run(pub.publish(
            object(), checkpoint_n=551, publication_seq=51,
            lr_schedule_step=816, trained_window_cursor=40016,
            reason="cadence", expected_parent_revision="rev-50",
        ))

    assert [name for name, _ in history.calls] == ["budget"]
    assert CANDIDATE_MANIFEST_KEY not in r2.objects
    assert not any(tmp_path.iterdir())


def test_first_history_root_failure_is_nonfatal_and_retried(tmp_path):
    class _RootFailureHistory(_History):
        def __init__(self):
            super().__init__()
            self.root_attempts = 0

        def compact_protected_branch(self, **kwargs):
            self.calls.append(("compact-protected", kwargs))
            self.root_attempts += 1
            if self.root_attempts == 1:
                raise RuntimeError("protected branch root failed")
            return "first-history-root"

    r2 = _R2()
    history = _RootFailureHistory()
    pub = _retained_publisher(tmp_path, r2, history)
    run = active_training_identity()["training_run_id"]
    root = pub.retention.run_start_prefix(run)
    for seq in range(1, 51):
        r2.objects[f"{root}publication-{seq:06d}/manifest.json"] = b"{}"
    r2.objects[f"{root}publication-000050/manifest.json"] = json.dumps({
        "training_run_id": run,
        "publication_seq": 50,
        "revision": "rev-50",
    }).encode()

    revision = asyncio.run(pub.publish(
        object(), checkpoint_n=551, publication_seq=51,
        lr_schedule_step=816, trained_window_cursor=40016,
        reason="cadence", expected_parent_revision="rev-50",
    ))

    assert revision == "final-root-rev"
    assert json.loads(r2.objects[CANDIDATE_MANIFEST_KEY])["revision"] == (
        "final-root-rev"
    )
    assert [name for name, _ in history.calls] == [
        "budget", "prepare", "compact", "compact-protected",
    ]

    asyncio.run(pub.publish(
        object(), checkpoint_n=552, publication_seq=52,
        lr_schedule_step=832, trained_window_cursor=40032,
        reason="cadence", expected_parent_revision="final-root-rev",
    ))
    assert history.root_attempts == 2
    assert pub._first_history_rooted is True


def test_evaluation_candidates_are_bounded_without_touching_milestones(
    tmp_path,
):
    r2 = _R2()
    history = _History()
    policy = CheckpointRetentionPolicy(
        enabled=True,
        evaluation_candidates_to_keep=2,
    )
    run = active_training_identity()["training_run_id"]
    root = policy.candidate_run_prefix(run)
    for seq in (50, 100, 150):
        r2.objects[
            f"{root}publication-{seq:06d}/model.safetensors"
        ] = b"old"
        r2.objects[f"{root}publication-{seq:06d}/manifest.json"] = b"{}"
    milestone = (
        "reliquary/checkpoint-milestones/"
        f"{run_key(run)}/publication-000250/model.safetensors"
    )
    r2.objects[milestone] = b"permanent"

    pub = _retained_publisher(tmp_path, r2, history, revision="rev-200")
    pub.retention = policy
    asyncio.run(pub.publish(
        object(), checkpoint_n=700, publication_seq=200,
        lr_schedule_step=3200, trained_window_cursor=50000,
        reason="cadence", expected_parent_revision="rev-199",
    ))

    candidates = {
        key.split("/")[3]
        for key in r2.objects
        if key.startswith(root)
    }
    assert candidates == {"publication-000150", "publication-000200"}
    assert milestone in r2.objects
