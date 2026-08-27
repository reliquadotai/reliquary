from __future__ import annotations

from types import SimpleNamespace

import pytest

from reliquary.trainer.retention import (
    CheckpointRetentionPolicy,
    HfHistoryManager,
    R2_CANDIDATE_PREFIX,
    R2_MILESTONE_PREFIX,
    R2_RUN_START_PREFIX,
    run_key,
)


def test_default_policy_is_explicitly_opt_in():
    policy = CheckpointRetentionPolicy.from_env({})
    assert policy.enabled is False
    assert policy.keep_initial == 50
    assert policy.candidate_interval == 50
    assert policy.milestone_interval == 250
    assert policy.evaluation_candidates_to_keep == 15


def test_policy_keeps_detailed_run_start_then_thins():
    policy = CheckpointRetentionPolicy(enabled=True)

    assert policy.retention_class(1) == "run_start_history"
    assert policy.retention_class(49) == "run_start_history"
    assert policy.retention_class(50) == "run_start_history"
    assert policy.is_candidate(50)
    assert policy.should_compact_before(51)
    assert not policy.should_compact_before(52)
    assert policy.is_candidate(100)
    assert policy.should_compact_before(101)
    assert policy.retention_class(250) == "permanent_milestone"
    assert policy.should_compact_before(251)


def test_snapshot_prefix_uses_candidate_and_permanent_tiers():
    policy = CheckpointRetentionPolicy(enabled=True)
    key = run_key("reasoning prompt/v5")

    assert policy.snapshot_prefix("reasoning prompt/v5", 1) == (
        f"{R2_RUN_START_PREFIX}/{key}/publication-000001"
    )
    assert policy.snapshot_prefix("reasoning prompt/v5", 50) == (
        f"{R2_RUN_START_PREFIX}/{key}/publication-000050"
    )
    assert policy.snapshot_prefix("reasoning prompt/v5", 100) == (
        f"{R2_CANDIDATE_PREFIX}/{key}/publication-000100"
    )
    assert policy.snapshot_prefix("reasoning prompt/v5", 250) == (
        f"{R2_MILESTONE_PREFIX}/{key}/publication-000250"
    )
    assert policy.snapshot_prefix("reasoning prompt/v5", 51) is None


def test_retention_requires_audited_sequence():
    policy = CheckpointRetentionPolicy(enabled=True)
    with pytest.raises(RuntimeError, match="publication sequence"):
        policy.validate_publication_seq(None)
    with pytest.raises(ValueError, match="positive"):
        policy.validate_publication_seq(0)
    assert policy.validate_publication_seq(1) == 1


class _Api:
    def __init__(self):
        self.head = "head-50"
        self.branches = {"main": self.head}
        self.created = []
        self.deleted = []
        self.squashes = []
        self.storage = 7_000_000_000_000

    def model_info(self, repo_id):
        return SimpleNamespace(sha=self.head)

    def list_repo_refs(self, repo_id):
        return SimpleNamespace(branches=[
            SimpleNamespace(name=name, target_commit=revision)
            for name, revision in self.branches.items()
        ])

    def create_branch(self, repo_id, branch, revision, exist_ok):
        self.branches[branch] = revision
        self.created.append((branch, revision))

    def delete_branch(self, repo_id, branch):
        self.deleted.append(branch)
        del self.branches[branch]

    def super_squash_history(self, repo_id, branch, commit_message):
        self.squashes.append((branch, commit_message))
        if branch == "main":
            self.head = "root-51"
            self.branches["main"] = self.head
        else:
            self.branches[branch] = "root-protected"

    def list_repo_commits(self, repo_id, revision):
        if revision == "main":
            title = self.squashes[-1][1]
        else:
            title = "R2-archived run-start history (through publication 50)"
        return [SimpleNamespace(
            commit_id=self.branches[revision],
            title=title,
        )]

    def list_models(self, author):
        return [SimpleNamespace(used_storage=self.storage)]

    def list_datasets(self, author):
        return []

    def list_spaces(self, author):
        return []

    def dataset_info(self, repo_id):
        raise AssertionError("no datasets were listed")

    def space_info(self, repo_id):
        raise AssertionError("no spaces were listed")


def test_first_compaction_protects_first_fifty_and_publishes_new_root():
    api = _Api()
    manager = HfHistoryManager(api)
    policy = CheckpointRetentionPolicy(enabled=True)

    plan = manager.prepare_compaction(
        repo_id="org/repo",
        run_id="run-v5",
        publication_seq=51,
        expected_parent_revision="head-50",
        policy=policy,
    )
    assert plan.permanent is True
    assert api.branches[policy.first_history_branch("run-v5")] == "head-50"

    api.head = "uploaded-51"
    api.branches["main"] = api.head
    revision = manager.compact_uploaded_head(
        repo_id="org/repo",
        uploaded_revision="uploaded-51",
        checkpoint_n=551,
        publication_seq=51,
    )
    assert revision == "root-51"
    assert api.squashes == [
        ("main", "checkpoint 551 (retention root; publication 51)")
    ]

    protected = manager.compact_protected_branch(
        repo_id="org/repo",
        branch=policy.first_history_branch("run-v5"),
        expected_revision="head-50",
        publication_seq=50,
    )
    assert protected == "root-protected"

    # The same operation is idempotent only for the exact rooted marker.
    assert manager.compact_protected_branch(
        repo_id="org/repo",
        branch=policy.first_history_branch("run-v5"),
        expected_revision="head-50",
        publication_seq=50,
    ) == "root-protected"


def test_compaction_waits_for_hub_root_cache_propagation(monkeypatch):
    class _DelayedApi(_Api):
        def __init__(self):
            super().__init__()
            self.stale_main: str | None = None
            self.stale_reads = 0

        def super_squash_history(self, repo_id, branch, commit_message):
            if branch == "main":
                self.stale_main = self.branches["main"]
                self.stale_reads = 2
            super().super_squash_history(repo_id, branch, commit_message)

        def list_repo_refs(self, repo_id):
            branches = dict(self.branches)
            if self.stale_reads:
                self.stale_reads -= 1
                branches["main"] = self.stale_main
            return SimpleNamespace(branches=[
                SimpleNamespace(name=name, target_commit=revision)
                for name, revision in branches.items()
            ])

    monkeypatch.setattr("reliquary.trainer.retention.time.sleep", lambda _: None)
    api = _DelayedApi()
    api.head = "uploaded-51"
    api.branches["main"] = api.head

    revision = HfHistoryManager(api).compact_uploaded_head(
        repo_id="org/repo",
        uploaded_revision="uploaded-51",
        checkpoint_n=551,
        publication_seq=51,
    )

    assert revision == "root-51"
    assert api.stale_reads == 0


def test_later_compaction_requires_first_history_and_bounds_grace_branches():
    api = _Api()
    manager = HfHistoryManager(api)
    policy = CheckpointRetentionPolicy(enabled=True, max_grace_branches=1)
    api.branches[policy.first_history_branch("run-v5")] = "head-50"
    api.head = "head-100"
    api.branches["main"] = api.head

    plan100 = manager.prepare_compaction(
        repo_id="org/repo", run_id="run-v5", publication_seq=101,
        expected_parent_revision="head-100", policy=policy,
    )
    assert not plan100.permanent
    api.head = "head-150"
    api.branches["main"] = api.head
    manager.prepare_compaction(
        repo_id="org/repo", run_id="run-v5", publication_seq=151,
        expected_parent_revision="head-150", policy=policy,
    )

    deleted = manager.cleanup_grace_branches(
        repo_id="org/repo", run_id="run-v5", policy=policy,
    )
    assert deleted == [plan100.branch]
    assert plan100.branch in api.deleted


def test_storage_guard_freezes_before_upload():
    api = _Api()
    manager = HfHistoryManager(api)
    assert manager.assert_storage_budget(
        repo_id="org/repo", freeze_bytes=8_000_000_000_000,
    ) == api.storage
    with pytest.raises(RuntimeError, match="safety ceiling"):
        manager.assert_storage_budget(
            repo_id="org/repo", freeze_bytes=6_000_000_000_000,
        )
