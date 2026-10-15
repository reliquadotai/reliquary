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
import logging
from pathlib import Path
import shutil
import tempfile
import threading

logger = logging.getLogger(__name__)


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
    backup under ``namespace`` (e.g. ReliquaryForge)."""

    def __init__(self, namespace: str, prefix: str, api=None) -> None:
        from huggingface_hub import HfApi

        self.namespace, self.prefix = namespace, prefix
        self._api = api or HfApi()

    def repo_id(self, step: int) -> str:
        return f"{self.namespace}/{self.prefix}-step-{int(step):08d}"

    def upload(self, step: int, folder: Path) -> None:
        repo = self.repo_id(step)
        self._api.create_repo(repo, private=True, exist_ok=True)
        self._api.upload_folder(repo_id=repo, folder_path=str(folder),
                                commit_message=f"backup step {step}")

    def list_backups(self) -> list[int]:
        head = f"{self.namespace}/{self.prefix}-step-"
        steps = []
        for model in self._api.list_models(author=self.namespace, search=f"{self.prefix}-step-"):
            if model.id.startswith(head) and model.id[len(head):].isdigit():
                steps.append(int(model.id[len(head):]))
        return steps

    def delete_backup(self, step: int) -> None:
        self._api.delete_repo(self.repo_id(step))
