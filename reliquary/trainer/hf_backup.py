"""Full-checkpoint backups to Hugging Face, off the training path.

docs/rl-async-design.md §4.8: a backup every M steps, a rolling window of the
last ``keep_last`` backups plus milestones never deleted, so the HF quota cannot
fill (it stopped the subnet on 2026-09-16). Each backup is its own private repo
(``<prefix>-step-<n>``): deleting the repo frees its storage, which deleting a
branch or a file in a shared repo does not.

``BackgroundHfBackup.submit`` never blocks the trainer: it hands the step to a
worker thread, and a step submitted while another is still uploading replaces
any step waiting behind it (only the newest waits).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import hashlib
import logging
from pathlib import Path
import shutil
import tempfile
import threading
import uuid

from reliquary.shared.checkpoint_identity import (
    require_checkpoint_number, require_immutable_checkpoint_revision,
)

logger = logging.getLogger(__name__)


def write_backup_profile(folder: Path, *, lr_schedule_step: int | None,
                         checkpoint_run: str, task_id: str) -> Path:
    from reliquary.validator.checkpoint_profile import write_checkpoint_profile

    step = require_checkpoint_number(lr_schedule_step, field="backup optimizer step")
    if not all(isinstance(value, str) and value.strip() == value and value
               for value in (checkpoint_run, task_id)):
        raise ValueError("backup requires its actual async run and task identity")
    return write_checkpoint_profile(folder, extra={"lr_schedule_step": step,
        "async_checkpoint_run": checkpoint_run, "async_task_id": task_id})


def plan_backup_rotation(
    steps: Iterable[int],
    *,
    keep_last: int,
    milestone_every: int | None = None,
    milestones: Iterable[int] = (),
) -> tuple[list[int], list[int]]:
    """(kept, deleted) backup steps: the ``keep_last`` newest, plus every
    milestone (a multiple of ``milestone_every`` or listed), kept forever."""
    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")
    ordered = sorted(set(int(s) for s in steps))
    pinned = set(int(m) for m in milestones)
    keep = set(ordered[-keep_last:])
    keep |= {s for s in ordered if s in pinned or (milestone_every and s % milestone_every == 0)}
    return [s for s in ordered if s in keep], [s for s in ordered if s not in keep]


class BackgroundHfBackup:
    def __init__(
        self,
        *,
        upload: Callable[[int, Path], None],
        list_backups: Callable[[], list[int]],
        delete_backup: Callable[[int], None],
        keep_last: int,
        milestone_every: int | None = None,
        milestones: Iterable[int] = (),
        staging_dir: str | Path | None = None,
    ) -> None:
        self._upload, self._list, self._delete = upload, list_backups, delete_backup
        self._rotation = dict(keep_last=keep_last, milestone_every=milestone_every,
                              milestones=tuple(milestones))
        self._staging = Path(staging_dir) if staging_dir else None
        self._pending: tuple[int, Callable[[Path], None]] | None = None
        self._cond = threading.Condition()
        self._stopped = False
        self.completed: list[int] = []
        self.dropped: list[int] = []
        self._thread = threading.Thread(target=self._run, name="hf-backup", daemon=True)
        self._thread.start()

    def submit(self, step: int, save: Callable[[Path], None]) -> None:
        """Queue a backup of ``step``; ``save(folder)`` writes the checkpoint in
        the worker thread, so it must read an immutable snapshot."""
        with self._cond:
            if self._pending is not None:
                self.dropped.append(self._pending[0])
            self._pending = (int(step), save)
            self._cond.notify()

    def close(self, timeout: float | None = None) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify()
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stopped:
                    self._cond.wait()
                if self._pending is None:
                    return
                step, save = self._pending
                self._pending = None
            try:
                self._backup(step, save)
                self.completed.append(step)
            except Exception:
                logger.exception("HF backup of step %d failed; training continues", step)

    def _backup(self, step: int, save: Callable[[Path], None]) -> None:
        if self._staging is not None:
            # mkdtemp does not create its parent: a fresh state volume has no staging dir yet.
            self._staging.mkdir(parents=True, mode=0o700, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix=f"hf-backup-{step}-", dir=self._staging))
        try:
            save(folder)
            self._upload(step, folder)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        _, delete = plan_backup_rotation(self._list(), **self._rotation)
        for old in delete:
            self._delete(old)
            logger.info("HF backup of step %d deleted by rotation", old)


class HfRepoPerBackup:
    """Upload/list/delete callables for BackgroundHfBackup, one private repo per
    backup under ``namespace``. Uploads require a private local receipt directory;
    listing remains usable without one."""

    def __init__(self, namespace: str, prefix: str, api=None,
                 *, receipt_dir: str | Path | None = None) -> None:
        from huggingface_hub import HfApi

        self.namespace, self.prefix = namespace, prefix
        self._api = api or HfApi()
        self._receipts = Path(receipt_dir) if receipt_dir is not None else None

    def repo_id(self, step: int) -> str:
        return f"{self.namespace}/{self.prefix}-step-{int(step):08d}"

    def upload(self, step: int, folder: Path) -> dict:
        from huggingface_hub.utils import RevisionNotFoundError
        from reliquary.trainer.publisher import (
            PUBLICATION_RECEIPT, PublicationConflict, _file_identity, _verify_weight_file_set,
        )
        from reliquary.validator.checkpoint_profile import validate_checkpoint_profile, CHECKPOINT_PROFILE_NAME
        from reliquary.validator.control import write_json

        step = require_checkpoint_number(step)
        if self._receipts is None:
            raise ValueError("HF uploads require a durable local publication receipt directory")
        receipt_path = self._receipts / f"step-{step:08d}.private.json"
        if receipt_path.exists():
            raise ValueError("checkpoint already has a publication receipt; do not republish it")
        profile = validate_checkpoint_profile(folder, required=True)
        require_checkpoint_number(profile.get("lr_schedule_step"), field="backup optimizer step")
        if not all(isinstance(profile.get(key), str) and profile[key].strip() == profile[key] and profile[key]
                   for key in ("training_run_id", "async_checkpoint_run", "async_task_id")):
            raise ValueError("backup profile requires its training, async run and task identity")
        repo = self.repo_id(step)
        self._api.create_repo(repo, private=True, exist_ok=True)
        try:
            parent = self._api.model_info(repo).sha
        except RevisionNotFoundError:
            parent = None
        if parent is not None:
            parent = require_immutable_checkpoint_revision(parent)
            if PUBLICATION_RECEIPT in self._api.list_repo_files(repo, revision=parent):
                raise PublicationConflict("backup repository already contains a publication; review its receipt")
            _verify_weight_file_set(self._api, repo, parent, [])
        files = {path.name: _file_identity(path) for path in sorted(folder.iterdir())
                 if path.name != PUBLICATION_RECEIPT}
        publication = {"publication_id": uuid.uuid4().hex, "parent_revision": parent,
            "manifest": {"publication_kind": "async_hf_backup", "repo_id": repo, "checkpoint_n": step,
                         **{key: profile[key] for key in
                            ("training_run_id", "async_checkpoint_run", "async_task_id")}},
            "files": files}
        write_json(folder / PUBLICATION_RECEIPT, publication)
        commit = self._api.upload_folder(repo_id=repo, folder_path=str(folder), parent_commit=parent,
                                        commit_message=f"backup step {step}")
        revision = require_immutable_checkpoint_revision(commit.oid, field="HF backup commit")
        if revision == parent:
            raise ValueError("HF backup did not create a new immutable commit")
        _verify_weight_file_set(self._api, repo, revision, [*files, PUBLICATION_RECEIPT])
        publication_raw = (folder / PUBLICATION_RECEIPT).read_text()
        profile_raw = (folder / CHECKPOINT_PROFILE_NAME).read_text()
        base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        receipt = {"schema_version": 1, "repo_id": repo, "revision": revision,
                   "publication_source": base + PUBLICATION_RECEIPT,
                   "profile_source": base + CHECKPOINT_PROFILE_NAME,
                   "publication_json": publication_raw, "profile_json": profile_raw,
                   "publication_sha256": hashlib.sha256(publication_raw.encode()).hexdigest(),
                   "profile_sha256": files[CHECKPOINT_PROFILE_NAME]["sha256"]}
        self._receipts.mkdir(parents=True, mode=0o700, exist_ok=True)
        write_json(receipt_path, receipt)
        return receipt

    def list_backups(self) -> list[int]:
        head = f"{self.namespace}/{self.prefix}-step-"
        steps = []
        for model in self._api.list_models(author=self.namespace, search=f"{self.prefix}-step-"):
            if model.id.startswith(head) and model.id[len(head):].isdigit():
                steps.append(int(model.id[len(head):]))
        return steps

    def delete_backup(self, step: int) -> None:
        self._api.delete_repo(self.repo_id(step))
