"""Checkpoint namespaces isolate publication, recovery and local state."""

import asyncio
import json
from pathlib import Path

import pytest

from reliquary.shared.checkpoint_namespace import CheckpointNamespace, active_checkpoint_namespace
from reliquary.trainer.publisher import MIRROR_COMPLETE, TrainerPublisher, checkpoint_key
from reliquary.trainer.resume import resolve_resume_point, validate_scoped_resume_snapshot
from reliquary.validator.checkpoint_intake import CheckpointIntake
from reliquary.validator.checkpoint_profile import (
    CHECKPOINT_PROFILE_NAME, CheckpointProfileMismatch,
    active_checkpoint_profile, validate_checkpoint_profile,
)
from tests.unit.test_trainer_publisher import _R2


BASE = "0" * 40
REV = "1" * 40


class _Mirror(_R2):
    def upload_file(self, path, bucket, key, Config=None):
        self.uploads.append((key, path))
        self.objects[key] = Path(path).read_bytes()

    def download_file(self, bucket, key, path, Config=None):
        Path(path).write_bytes(self.objects[key])


def _publisher(root, r2, namespace, heads, revisions):
    repo = "org/" + namespace.task_id + "-" + namespace.run_id
    heads.setdefault(repo, BASE)
    revisions = iter(revisions)

    def save(model, tokenizer, path):
        (path / "model.safetensors").write_bytes(b"weights")

    async def upload(**kwargs):
        heads[repo] = next(revisions)
        return heads[repo]

    return TrainerPublisher(
        repo_id=repo, staging_dir=str(root), namespace=namespace,
        tokenizer=None, r2_client=r2, bucket="b", save_fn=save,
        hf_head_fn=lambda repo_id: heads[repo_id], hf_upload_fn=upload,
    )


def _publish(pub, n, parent=BASE):
    return asyncio.run(pub.publish(
        object(), checkpoint_n=n, trained_window_cursor=n * 10,
        lr_schedule_step=5, reason="cadence", parent_revision=parent,
    ))


def test_disabled_mode_preserves_legacy_even_for_named_run(tmp_path):
    legacy = active_checkpoint_namespace({"RELIQUARY_TASK_ID": "another-task",
                                          "RELIQUARY_TRAINING_RUN_ID": "old/run"})
    assert legacy.identity == {}
    assert legacy.local_path(tmp_path) == tmp_path
    assert legacy.candidate_manifest_key == "reliquary/training/candidate-manifest.json"
    assert checkpoint_key(REV, "config.json", namespace=legacy) == f"reliquary/checkpoints/{REV}/config.json"
    assert active_checkpoint_profile(namespace=legacy) == active_checkpoint_profile()


@pytest.mark.parametrize("task,run", [(None, "run"), ("task", None), ("../task", "run"),
                                      ("task", "../run"), ("task", "run/a"),
                                      ("task\n", "run"), ("task", " run"), ("task", "")])
def test_requested_scope_never_falls_back_for_bad_identity(task, run):
    with pytest.raises(ValueError):
        active_checkpoint_namespace({"RELIQUARY_TASK_SCOPED_CHECKPOINTS": "1",
                                     "RELIQUARY_TASK_ID": task, "RELIQUARY_TRAINING_RUN_ID": run})


def test_two_tasks_and_runs_publish_and_resume_the_same_checkpoint_independently(tmp_path):
    r2, heads = _Mirror(), {}
    scopes = [CheckpointNamespace("task-a", "run-a"), CheckpointNamespace("task-b", "run-a"),
              CheckpointNamespace("task-a", "run-b")]
    for scope in scopes:
        pub = _publisher(tmp_path / "publisher", r2, scope, heads, [REV])
        assert _publish(pub, 1) == REV
        manifest = json.loads(r2.objects[scope.candidate_manifest_key])
        assert {k: manifest[k] for k in scope.identity} == scope.identity
        assert resolve_resume_point(
            r2.objects.get, env={}, namespace=scope,
            expected_identity={"repo_id": pub.repo_id},
        ) == (REV, 10, 1)
        intake = CheckpointIntake(r2_client=r2, bucket="b", staging_dir=str(tmp_path / "intake"),
                                  namespace=scope)
        assert intake.poll() == manifest
        assert intake.stage(manifest)
        staged_manifest, staged_dir = intake.take_staged()
        assert staged_manifest == manifest
        assert validate_checkpoint_profile(staged_dir, required=True,
            expected=active_checkpoint_profile(namespace=scope))["training_run_id"] == scope.run_id
        intake.mark_installed(REV, staged_dir)
        assert intake.installed_checkpoint_n == 1
    assert len({scope.candidate_manifest_key for scope in scopes}) == 3
    assert len(r2.uploads) == 3 * 4  # weights, profile, receipt, mirror-complete marker


@pytest.mark.parametrize("alter", ["profile", "cursor", "receipt"])
def test_foreign_or_inconsistent_snapshot_never_stages_or_erases_other_run(tmp_path, alter):
    r2, heads = _Mirror(), {}
    scope = CheckpointNamespace("task-a", "run-a")
    pub = _publisher(tmp_path / "publisher", r2, scope, heads, [REV])
    _publish(pub, 1)
    manifest = json.loads(r2.objects[scope.candidate_manifest_key])
    name = CHECKPOINT_PROFILE_NAME if alter != "receipt" else "reliquary_publication.json"
    key = checkpoint_key(REV, name, namespace=scope)
    value = json.loads(r2.objects[key])
    if alter == "profile":
        value["task_id"] = "task-b"
    elif alter == "cursor":
        value["trained_window_cursor"] = 999
    else:
        value["manifest"]["task_id"] = "task-b"
    r2.objects[key] = json.dumps(value).encode()
    other_path = CheckpointNamespace("task-b", "run-a").local_path(tmp_path / "intake") / REV
    other_path.mkdir(parents=True)
    (other_path / "keep").write_text("keep")
    intake = CheckpointIntake(r2_client=r2, bucket="b", staging_dir=str(tmp_path / "intake"), namespace=scope)
    assert not intake.stage(manifest)
    assert not intake.staged_ready
    assert (other_path / "keep").read_text() == "keep"


def test_foreign_candidate_is_rejected_instead_of_bootstrapping(tmp_path):
    scope = CheckpointNamespace("task-a", "run-a")
    foreign = {**CheckpointNamespace("task-b", "run-a").identity,
               "checkpoint_n": 1, "repo_id": "org/b", "revision": REV, "trained_window_cursor": 10}
    with pytest.raises(ValueError, match="task_id"):
        resolve_resume_point(lambda key: json.dumps(foreign).encode(),
            env={"RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "100"}, namespace=scope)
    r2 = _Mirror()
    r2.objects[scope.candidate_manifest_key] = json.dumps(foreign).encode()
    intake = CheckpointIntake(r2_client=r2, bucket="b", staging_dir=str(tmp_path), namespace=scope)
    assert intake.poll() is None
    assert intake.last_error


def test_profile_without_runtime_scope_is_not_silently_legacy(tmp_path):
    (tmp_path / CHECKPOINT_PROFILE_NAME).write_text(json.dumps(
        active_checkpoint_profile(namespace=CheckpointNamespace("task", "run"))))
    with pytest.raises(CheckpointProfileMismatch):
        validate_checkpoint_profile(tmp_path, required=True,
                                    expected=active_checkpoint_profile(namespace=CheckpointNamespace()))


def test_retention_is_bounded_to_one_task_run(tmp_path):
    r2, heads = _Mirror(), {}
    scope = CheckpointNamespace("task-a", "run-a")
    other = CheckpointNamespace("task-a", "run-b")
    foreign = checkpoint_key("a" * 40, "config.json", namespace=other)
    legacy = checkpoint_key("b" * 40, "config.json", namespace=CheckpointNamespace())
    r2.objects[foreign] = r2.objects[legacy] = b"keep"
    pub = _publisher(tmp_path, r2, scope, heads, [REV, "2" * 40, "3" * 40])
    for n, parent in [(1, BASE), (2, REV), (3, "2" * 40)]:
        _publish(pub, n, parent)
    assert checkpoint_key(REV, "model.safetensors", namespace=scope) in r2.deleted
    assert r2.objects[foreign] == r2.objects[legacy] == b"keep"
    assert all(key.startswith(scope.checkpoint_prefix + "/") for key in r2.deleted)


def test_scoped_recovery_after_candidate_commit_is_exactly_once(tmp_path):
    class _Interrupted(_Mirror):
        interrupted = False
        def put_object(self, **kwargs):
            super().put_object(**kwargs)
            if not self.interrupted:
                self.interrupted = True
                raise RuntimeError("lost commit response")

    r2, heads = _Interrupted(), {}
    scope = CheckpointNamespace("task-a", "run-a")
    pub = _publisher(tmp_path, r2, scope, heads, [REV])
    with pytest.raises(RuntimeError, match="lost commit"):
        _publish(pub, 1)
    assert pub.has_pending()
    upload_count = len(r2.uploads)
    recovered = _publisher(tmp_path, r2, scope, heads, [])
    assert asyncio.run(recovered.recover_pending())["revision"] == REV
    assert not recovered.has_pending()
    # Only the weights deferred past the manifest, then the marker: metadata
    # already mirrored before the commit is never sent twice.
    assert [key.rsplit("/", 1)[1] for key, _ in r2.uploads[upload_count:]] == [
        "model.safetensors", MIRROR_COMPLETE,
    ]
    assert asyncio.run(recovered.recover_pending()) is None


def test_local_scope_refuses_aliases_but_allows_configured_root(tmp_path):
    scope = CheckpointNamespace("task", "run-a")
    real_root = tmp_path / "real"
    real_root.mkdir()
    configured = tmp_path / "configured"
    configured.symlink_to(real_root, target_is_directory=True)
    own = scope.local_path(configured)
    own.parent.mkdir(parents=True)
    other = own.parent / "run-b"
    other.mkdir()
    own.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        scope.local_path(configured)
    own.unlink()
    own.mkdir()
    (own / "resume").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        scope.child_path(own, "resume", REV)
    assert not (other / REV).exists()


def test_runtime_checkpoint_policy_cannot_disagree_with_mode():
    scoped = CheckpointNamespace("task", "run")
    scoped.require_policy({"kind": "trainer-driven/v1", "task_scoped": 1})
    CheckpointNamespace().require_policy({"kind": "trainer-driven/v1", "task_scoped": 0})
    for policy in ({"kind": "frozen/v1"}, {"kind": "trainer-driven/v1", "task_scoped": 0},
                   {"kind": "trainer-driven/v1", "task_scoped": True}):
        with pytest.raises(ValueError):
            scoped.require_policy(policy)


@pytest.mark.parametrize("candidate_present", [False, True])
@pytest.mark.parametrize("alter", [None, "cursor", "receipt", "checkpoint", "missing"])
def test_scoped_resume_binds_exact_snapshot_even_for_explicit_bootstrap(tmp_path, candidate_present, alter):
    from reliquary.trainer.cli import _download_checkpoint

    scope, r2, heads = CheckpointNamespace("task", "run"), _Mirror(), {}
    pub = _publisher(tmp_path / "publisher", r2, scope, heads, [REV])
    _publish(pub, 1)
    candidate = json.loads(r2.objects[scope.candidate_manifest_key])
    snapshot = tmp_path / "resume"
    assert _download_checkpoint(r2, "b", REV, snapshot, namespace=scope)
    if alter == "cursor":
        path = snapshot / CHECKPOINT_PROFILE_NAME
        value = json.loads(path.read_bytes())
        value["trained_window_cursor"] += 1
        path.write_text(json.dumps(value))
    elif alter in {"receipt", "checkpoint"}:
        path = snapshot / "reliquary_publication.json"
        value = json.loads(path.read_bytes())
        field = "task_id" if alter == "receipt" else "checkpoint_n"
        value["manifest"][field] = "foreign-task" if alter == "receipt" else 2
        path.write_text(json.dumps(value))
    elif alter == "missing":
        (snapshot / "reliquary_publication.json").unlink()
    kwargs = dict(namespace=scope, repo_id=pub.repo_id, revision=REV,
                  checkpoint_n=1, candidate=candidate if candidate_present else None)
    if alter is None:
        assert validate_scoped_resume_snapshot(snapshot, **kwargs)["trained_window_cursor"] == 10
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            validate_scoped_resume_snapshot(snapshot, **kwargs)


def test_scoped_cache_marker_and_miner_identity_are_not_reused_between_runs(monkeypatch, tmp_path):
    from reliquary.trainer.cli import _download_checkpoint
    from reliquary.miner.checkpoint_identity import default_checkpoint_identity_path

    r2 = _Mirror()
    a, b = CheckpointNamespace("task", "run-a"), CheckpointNamespace("task", "run-b")
    for scope in (a, b):
        r2.objects[checkpoint_key(REV, "config.json", namespace=scope)] = b"{}"
        r2.objects[checkpoint_key(REV, MIRROR_COMPLETE, namespace=scope)] = b"{}"
    cache = tmp_path / "cache"
    assert _download_checkpoint(r2, "b", REV, cache, namespace=a)
    assert _download_checkpoint(r2, "b", REV, cache, namespace=b)
    assert json.loads((cache / ".download-complete.json").read_bytes())["identity"]["training_run_id"] == "run-b"
    monkeypatch.setenv("RELIQUARY_MINER_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("RELIQUARY_TASK_SCOPED_CHECKPOINTS", "1")
    monkeypatch.setenv("RELIQUARY_TASK_ID", "task")
    monkeypatch.setenv("RELIQUARY_TRAINING_RUN_ID", "run-a")
    path_a = default_checkpoint_identity_path("wallet")
    monkeypatch.setenv("RELIQUARY_TRAINING_RUN_ID", "run-b")
    path_b = default_checkpoint_identity_path("wallet")
    assert path_a != path_b
    assert path_a.is_relative_to(a.local_path(tmp_path))
    assert path_b.is_relative_to(b.local_path(tmp_path))


def _enable_scope(monkeypatch, scope):
    monkeypatch.setenv("RELIQUARY_TASK_SCOPED_CHECKPOINTS", "1")
    monkeypatch.setenv("RELIQUARY_TASK_ID", scope.task_id)
    monkeypatch.setenv("RELIQUARY_TRAINING_RUN_ID", scope.run_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("alter", [None, "cursor", "checkpoint_n"])
async def test_validator_scoped_resume_stages_before_any_adoption(monkeypatch, tmp_path, alter):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from reliquary.validator.service import ValidationService

    scope, r2, heads = CheckpointNamespace("task", "run"), _Mirror(), {}
    _enable_scope(monkeypatch, scope)
    pub = _publisher(tmp_path / "publisher", r2, scope, heads, [REV])
    await pub.publish(object(), checkpoint_n=1, trained_window_cursor=10,
                      lr_schedule_step=5, reason="cadence", parent_revision=BASE)
    if alter:
        value = json.loads(r2.objects[scope.candidate_manifest_key])
        key = "trained_window_cursor" if alter == "cursor" else alter
        value[key] += 1
        r2.objects[scope.candidate_manifest_key] = json.dumps(value).encode()
    intake = CheckpointIntake(r2_client=r2, bucket="b", staging_dir=str(tmp_path / "intake"), namespace=scope)
    service = ValidationService.__new__(ValidationService)
    service._resume_from, service._window_n = "sha:" + REV, 0
    service._detached_intake_ref = lambda: intake
    manifest = SimpleNamespace(checkpoint_n=1, repo_id=pub.repo_id, revision=REV)
    service._checkpoint_store = SimpleNamespace(current_manifest=lambda: manifest)
    service._swap_staged_checkpoint = AsyncMock()
    if alter:
        with pytest.raises(RuntimeError, match="staging failed"):
            await service._apply_resume_from()
        service._swap_staged_checkpoint.assert_not_awaited()
    else:
        await service._apply_resume_from()
        service._swap_staged_checkpoint.assert_awaited_once_with(0)


def test_cooldown_snapshots_bind_task_as_well_as_run(monkeypatch):
    from reliquary.validator import service

    a, b = CheckpointNamespace("task-a", "run"), CheckpointNamespace("task-b", "run")
    _enable_scope(monkeypatch, a)
    key_a = service._cooldown_snapshot_key("run")
    content_a = service._content_cooldown_snapshot_key("run")
    monkeypatch.setattr(service, "TRAINING_RUN_ID", "run")
    snapshot = {"schema_version": 2, "complete": True, "run_id": "run",
                **a.identity, "snapshot_window": 0, "envs": {}}
    assert service.ValidationService._validate_cooldown_snapshot(snapshot, set(), 0) == 0
    _enable_scope(monkeypatch, b)
    assert service._cooldown_snapshot_key("run") != key_a
    assert service._content_cooldown_snapshot_key("run") != content_a
    with pytest.raises(ValueError, match="task_id"):
        service.ValidationService._validate_cooldown_snapshot(snapshot, set(), 0)


@pytest.mark.asyncio
async def test_miner_refuses_foreign_profile_before_loading_weights(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    from reliquary.miner.engine import maybe_pull_checkpoint

    own, foreign = CheckpointNamespace("task", "run-a"), CheckpointNamespace("task", "run-b")
    _enable_scope(monkeypatch, own)
    (tmp_path / CHECKPOINT_PROFILE_NAME).write_text(json.dumps(active_checkpoint_profile(namespace=foreign)))
    load = MagicMock()
    with pytest.raises(CheckpointProfileMismatch, match="training_run_id"):
        await maybe_pull_checkpoint(
            SimpleNamespace(checkpoint_n=1, checkpoint_repo_id="org/repo", checkpoint_revision=REV),
            0, "", "", object(), download_fn=AsyncMock(return_value=str(tmp_path)), load_fn=load,
        )
    load.assert_not_called()


@pytest.mark.asyncio
async def test_scoped_service_contract_is_rejected_before_download_when_flag_is_off(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock
    from reliquary.miner.engine import maybe_pull_checkpoint

    monkeypatch.delenv("RELIQUARY_TASK_SCOPED_CHECKPOINTS", raising=False)
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    value["service_kind"] = "adaptive_training"
    value["policies"]["checkpoint"] = {"kind": "trainer-driven/v1", "task_scoped": 1}
    state = SimpleNamespace(checkpoint_n=1, checkpoint_repo_id="models/test", checkpoint_revision=REV,
                            service_policy={"contract": value})
    download, load = AsyncMock(), MagicMock()
    with pytest.raises(ValueError, match="namespace differs"):
        await maybe_pull_checkpoint(state, 0, "", "", object(), download_fn=download, load_fn=load)
    download.assert_not_awaited()
    load.assert_not_called()


@pytest.mark.asyncio
async def test_scoped_hf_snapshot_includes_the_required_profile(monkeypatch):
    from reliquary.miner.engine import _hf_download
    from reliquary.shared.modeling import MODEL_SNAPSHOT_ALLOW_PATTERNS

    _enable_scope(monkeypatch, CheckpointNamespace("task", "run"))
    captured = {}

    def download(**kwargs):
        captured.update(kwargs)
        return "/tmp/unit-model-snapshot"

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    await _hf_download("models/test", REV)
    assert captured["allow_patterns"] == [*MODEL_SNAPSHOT_ALLOW_PATTERNS, CHECKPOINT_PROFILE_NAME]
