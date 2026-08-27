"""Bounded retention for trainer-published model checkpoints.

Publication cadence and retention cadence are deliberately independent.  The
trainer still publishes every behavior-policy refresh, while this module keeps
the Hub history bounded in blocks and selects sparse R2 evaluation snapshots.

Hub history rewriting is irreversible, so it is opt-in and guarded by an
explicit run-local publication sequence.  R2 retains the first checkpoints of
a run; a rooted HF marker records that boundary, and temporary grace branches
retain the previous block while ``main`` is super-squashed to the newly uploaded
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
import re
import time
from typing import Any, Mapping

logger = logging.getLogger(__name__)

R2_CANDIDATE_PREFIX = "reliquary/checkpoint-candidates"
R2_MILESTONE_PREFIX = "reliquary/checkpoint-milestones"
R2_LEDGER_PREFIX = "reliquary/checkpoint-ledger"
R2_RUN_START_PREFIX = "reliquary/checkpoint-run-start"


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = str(env.get(name, "1" if default else "0")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = int(env.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def run_key(run_id: str) -> str:
    """Return a stable, branch/key-safe training-run identifier."""

    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(run_id)).strip("-._")
    normalized = normalized[:40] or "run"
    digest = hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()[:10]
    return f"{normalized}-{digest}"


@dataclass(frozen=True)
class CheckpointRetentionPolicy:
    """Run-aware selection and Hub compaction policy.

    The first ``keep_initial`` publications are retained in full on R2.
    Afterwards, one complete evaluation snapshot is copied to R2 every
    ``candidate_interval`` publications and the live Hub history is compacted
    at the beginning of the next block.  Every ``milestone_interval``
    publication is placed under the non-expiring milestone prefix instead of
    the candidate prefix.
    """

    enabled: bool = False
    keep_initial: int = 50
    candidate_interval: int = 50
    milestone_interval: int = 250
    evaluation_candidates_to_keep: int = 15
    max_grace_branches: int = 1
    storage_freeze_bytes: int | None = 11_500_000_000_000

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None,
    ) -> "CheckpointRetentionPolicy":
        values = os.environ if env is None else env
        enabled = _env_bool(
            values, "RELIQUARY_CHECKPOINT_RETENTION_ENABLED", False,
        )
        freeze_tb = float(values.get(
            "RELIQUARY_HF_STORAGE_FREEZE_TB", "11.5",
        ))
        if freeze_tb <= 0:
            raise ValueError(
                "RELIQUARY_HF_STORAGE_FREEZE_TB must be positive"
            )
        max_grace = int(values.get(
            "RELIQUARY_HF_RETENTION_MAX_GRACE_BRANCHES", "1",
        ))
        if max_grace < 1:
            raise ValueError(
                "RELIQUARY_HF_RETENTION_MAX_GRACE_BRANCHES must be >= 1"
            )
        return cls(
            enabled=enabled,
            keep_initial=_positive_int(
                values, "RELIQUARY_HF_RETENTION_KEEP_INITIAL", 50,
            ),
            candidate_interval=_positive_int(
                values, "RELIQUARY_HF_RETENTION_CANDIDATE_INTERVAL", 50,
            ),
            milestone_interval=_positive_int(
                values, "RELIQUARY_HF_RETENTION_MILESTONE_INTERVAL", 250,
            ),
            evaluation_candidates_to_keep=_positive_int(
                values,
                "RELIQUARY_R2_EVALUATION_CANDIDATES_TO_KEEP",
                15,
            ),
            max_grace_branches=max_grace,
            storage_freeze_bytes=int(freeze_tb * 1_000_000_000_000),
        )

    def validate_publication_seq(self, publication_seq: int | None) -> int:
        if publication_seq is None:
            raise RuntimeError(
                "bounded checkpoint retention requires a run-local "
                "publication sequence; for an existing run, set "
                "RELIQUARY_TRAINER_PUBLICATION_SEQ from "
                "scripts/prepare_hf_retention.py"
            )
        value = int(publication_seq)
        if value <= 0:
            raise ValueError("publication_seq must be positive when publishing")
        return value

    def is_initial_history(self, publication_seq: int) -> bool:
        return 1 <= int(publication_seq) <= self.keep_initial

    def is_candidate(self, publication_seq: int) -> bool:
        seq = int(publication_seq)
        return seq == self.keep_initial or (
            seq > self.keep_initial
            and (seq - self.keep_initial) % self.candidate_interval == 0
        )

    def is_milestone(self, publication_seq: int) -> bool:
        seq = int(publication_seq)
        return seq > self.keep_initial and seq % self.milestone_interval == 0

    def should_compact_before(self, publication_seq: int) -> bool:
        seq = int(publication_seq)
        first_compacted = self.keep_initial + 1
        return seq >= first_compacted and (
            seq - first_compacted
        ) % self.candidate_interval == 0

    def retention_class(self, publication_seq: int) -> str:
        if self.is_initial_history(publication_seq):
            return "run_start_history"
        if self.is_milestone(publication_seq):
            return "permanent_milestone"
        if self.is_candidate(publication_seq):
            return "evaluation_candidate"
        return "live_only"

    def snapshot_prefix(self, run_id: str, publication_seq: int) -> str | None:
        if self.is_initial_history(publication_seq):
            root = R2_RUN_START_PREFIX
        elif self.is_milestone(publication_seq):
            root = R2_MILESTONE_PREFIX
        elif self.is_candidate(publication_seq):
            root = R2_CANDIDATE_PREFIX
        else:
            return None
        return f"{root}/{run_key(run_id)}/publication-{publication_seq:06d}"

    def ledger_key(self, run_id: str, publication_seq: int) -> str:
        return (
            f"{R2_LEDGER_PREFIX}/{run_key(run_id)}/"
            f"publication-{publication_seq:06d}.json"
        )

    def candidate_run_prefix(self, run_id: str) -> str:
        return f"{R2_CANDIDATE_PREFIX}/{run_key(run_id)}/"

    def run_start_prefix(self, run_id: str) -> str:
        return f"{R2_RUN_START_PREFIX}/{run_key(run_id)}/"

    def first_history_branch(self, run_id: str) -> str:
        return f"retention-{run_key(run_id)}-first-{self.keep_initial}"

    def grace_branch(self, run_id: str, retained_publication_seq: int) -> str:
        return (
            f"retention-{run_key(run_id)}-grace-"
            f"{int(retained_publication_seq):06d}"
        )

    def grace_branch_prefix(self, run_id: str) -> str:
        return f"retention-{run_key(run_id)}-grace-"

    def run_start_root_message(self) -> str:
        return (
            "R2-archived run-start history "
            f"(through publication {self.keep_initial})"
        )


@dataclass(frozen=True)
class HfCompactionPlan:
    branch: str
    retained_publication_seq: int
    permanent: bool


class HfHistoryManager:
    """Small synchronous wrapper around irreversible Hub history operations."""

    def __init__(self, api: Any | None = None) -> None:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        self.api = api

    def organization_storage_bytes(self, namespace: str) -> int:
        """Return visible model/dataset/Space storage for a namespace."""

        total = 0
        listings = (
            (self.api.list_models(author=namespace), self.api.model_info),
            (self.api.list_datasets(author=namespace), self.api.dataset_info),
            (self.api.list_spaces(author=namespace), self.api.space_info),
        )
        # The list endpoints do not currently expose ``usedStorage`` as an
        # expandable field. Resolve each listed repo individually; with the HF
        # token this also includes private repos visible to the organization.
        for listing, info_fn in listings:
            for item in listing:
                used = getattr(item, "used_storage", None)
                if used is None:
                    repo_id = str(getattr(item, "id"))
                    used = getattr(info_fn(repo_id), "used_storage", 0)
                total += int(used or 0)
        return total

    def assert_storage_budget(
        self, *, repo_id: str, freeze_bytes: int | None,
    ) -> int | None:
        if freeze_bytes is None:
            return None
        namespace = repo_id.split("/", 1)[0]
        used = self.organization_storage_bytes(namespace)
        if used >= freeze_bytes:
            raise RuntimeError(
                "Hugging Face storage safety ceiling reached: "
                f"namespace={namespace} used={used} ceiling={freeze_bytes}; "
                "publication is frozen before the next multi-GB upload"
            )
        return used

    def _branches(self, repo_id: str) -> dict[str, str]:
        refs = self.api.list_repo_refs(repo_id=repo_id)
        return {
            str(branch.name): str(branch.target_commit)
            for branch in getattr(refs, "branches", [])
        }

    def _ensure_branch(
        self, *, repo_id: str, branch: str, revision: str,
    ) -> None:
        branches = self._branches(repo_id)
        if branch in branches:
            logger.info(
                "HF retention branch already exists: %s@%s",
                branch, branches[branch][:12],
            )
            return
        self.api.create_branch(
            repo_id=repo_id,
            branch=branch,
            revision=revision,
            exist_ok=True,
        )

    def _wait_for_new_root(
        self,
        *,
        repo_id: str,
        branch: str,
        previous_revision: str,
        expected_message: str,
        timeout_seconds: float = 30.0,
    ) -> str:
        """Wait through the Hub's post-squash cache propagation window.

        ``super_squash_history`` can return before repository metadata exposes
        the new SHA. Accept only a changed branch target whose complete history
        is the exact single root commit requested by the caller.
        """

        deadline = time.monotonic() + timeout_seconds
        unexpected: str | None = None
        while True:
            observed = self._branches(repo_id).get(branch)
            if observed and observed != previous_revision:
                commits = self.api.list_repo_commits(
                    repo_id=repo_id,
                    revision=branch,
                )
                if (
                    len(commits) == 1
                    and str(commits[0].commit_id) == observed
                    and str(commits[0].title or "") == expected_message
                ):
                    return observed
                # Branch refs and commit listings have independent Hub cache
                # paths. Keep polling until both agree on the requested root.
                unexpected = (
                    f"branch={branch!r} revision={observed} "
                    f"commit_count={len(commits)}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = f"; last_observed={unexpected}" if unexpected else ""
                raise RuntimeError(
                    "HF super-squash did not expose the expected new root "
                    f"within {timeout_seconds:g}s: branch={branch!r}{detail}"
                )
            time.sleep(min(0.5, remaining))

    def _wait_for_branch_revision(
        self,
        *,
        repo_id: str,
        branch: str,
        expected_revision: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Wait for an ordinary HF commit to become visible before squash."""

        deadline = time.monotonic() + timeout_seconds
        observed: str | None = None
        while True:
            observed = self._branches(repo_id).get(branch)
            if observed == expected_revision:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "HF branch did not expose the uploaded revision within "
                    f"{timeout_seconds:g}s: branch={branch!r} "
                    f"expected={expected_revision} observed={observed}"
                )
            time.sleep(min(0.5, remaining))

    def prepare_compaction(
        self,
        *,
        repo_id: str,
        run_id: str,
        publication_seq: int,
        expected_parent_revision: str,
        policy: CheckpointRetentionPolicy,
    ) -> HfCompactionPlan:
        """Protect the outgoing block before a new root is created."""

        head = str(self.api.model_info(repo_id).sha)
        if head != expected_parent_revision:
            raise RuntimeError(
                "HF HEAD moved before retention compaction: "
                f"expected={expected_parent_revision} observed={head}"
            )
        retained_seq = int(publication_seq) - 1
        if publication_seq == policy.keep_initial + 1:
            branch = policy.first_history_branch(run_id)
            permanent = True
        else:
            first = policy.first_history_branch(run_id)
            if first not in self._branches(repo_id):
                raise RuntimeError(
                    f"missing protected first-history branch {first!r}; run "
                    "scripts/prepare_hf_retention.py before enabling compaction"
                )
            branch = policy.grace_branch(run_id, retained_seq)
            permanent = False
        self._ensure_branch(
            repo_id=repo_id,
            branch=branch,
            revision=expected_parent_revision,
        )
        return HfCompactionPlan(
            branch=branch,
            retained_publication_seq=retained_seq,
            permanent=permanent,
        )

    def compact_uploaded_head(
        self,
        *,
        repo_id: str,
        uploaded_revision: str,
        checkpoint_n: int,
        publication_seq: int,
    ) -> str:
        """Squash only the just-uploaded HEAD and return its final SHA."""

        self._wait_for_branch_revision(
            repo_id=repo_id,
            branch="main",
            expected_revision=uploaded_revision,
        )
        message = (
            f"checkpoint {int(checkpoint_n)} "
            f"(retention root; publication {int(publication_seq)})"
        )
        self.api.super_squash_history(
            repo_id=repo_id,
            branch="main",
            commit_message=message,
        )
        return self._wait_for_new_root(
            repo_id=repo_id,
            branch="main",
            previous_revision=uploaded_revision,
            expected_message=message,
        )

    def compact_protected_branch(
        self,
        *,
        repo_id: str,
        branch: str,
        expected_revision: str,
        publication_seq: int,
    ) -> str:
        """Reduce an R2-archived first-history branch to its tip snapshot."""

        branches = self._branches(repo_id)
        observed = branches.get(branch)
        if observed is None:
            raise RuntimeError(f"missing protected HF branch {branch!r}")
        message = (
            "R2-archived run-start history "
            f"(through publication {int(publication_seq)})"
        )
        if observed != expected_revision:
            # A previous retry may already have rooted this branch. Accept
            # only the exact single-commit marker this method creates; a
            # different target is a safety error, not an idempotent success.
            commits = self.api.list_repo_commits(
                repo_id=repo_id,
                revision=branch,
            )
            if (
                len(commits) == 1
                and str(commits[0].title or "") == message
            ):
                logger.info(
                    "HF protected branch already rooted: %s@%s",
                    branch,
                    observed[:12],
                )
                return observed
            raise RuntimeError(
                f"protected HF branch {branch!r} moved unexpectedly: "
                f"expected={expected_revision} observed={observed}"
            )
        self.api.super_squash_history(
            repo_id=repo_id,
            branch=branch,
            commit_message=message,
        )
        return self._wait_for_new_root(
            repo_id=repo_id,
            branch=branch,
            previous_revision=expected_revision,
            expected_message=message,
        )

    def cleanup_grace_branches(
        self,
        *,
        repo_id: str,
        run_id: str,
        policy: CheckpointRetentionPolicy,
    ) -> list[str]:
        """Keep only the newest bounded set of temporary grace branches."""

        prefix = policy.grace_branch_prefix(run_id)
        branches = sorted(
            name for name in self._branches(repo_id) if name.startswith(prefix)
        )
        stale = branches[:-policy.max_grace_branches]
        for branch in stale:
            self.api.delete_branch(repo_id=repo_id, branch=branch)
        return stale


__all__ = [
    "CheckpointRetentionPolicy",
    "HfCompactionPlan",
    "HfHistoryManager",
    "R2_CANDIDATE_PREFIX",
    "R2_LEDGER_PREFIX",
    "R2_MILESTONE_PREFIX",
    "R2_RUN_START_PREFIX",
    "run_key",
]
