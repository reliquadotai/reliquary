"""Miner side of checkpoint deltas: verify the signed index, take the cheapest
route to the latest version, rebuild bit-exactly and check the SHA-256; on any
mismatch start again from an anchor (docs/rl-async-design.md §4.8).

With ``model_dir``, the newest anchor's model files (HF config, tokenizer, chat
template) are kept there too, checked against the index, so a miner can load the
weights without a local copy of the base model.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

import torch

from reliquary.shared.ckpt_delta import (
    ChecksumMismatch,
    DeltaError,
    apply_delta,
    decode_delta,
    load_anchor_shards,
    verify_checkpoint,
)
from reliquary.shared.ckpt_index import (
    DownloadPlan,
    latest_model_files,
    model_file_name_ok,
    plan_download,
    verify_index_signature,
)

logger = logging.getLogger(__name__)


class IndexSignatureError(RuntimeError):
    pass


class ObjectMismatch(RuntimeError):
    """A downloaded object does not match the size or SHA-256 in the index."""


def http_download(url: str, dest: Path) -> None:
    import httpx

    with httpx.stream("GET", url, timeout=httpx.Timeout(60.0, connect=15.0),
                      follow_redirects=False) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_bytes(8 * 1024 * 1024):
                handle.write(chunk)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_object(path: Path, entry: Mapping[str, Any]) -> None:
    if path.stat().st_size != entry["size"] or _sha256(path) != entry["sha256"]:
        raise ObjectMismatch(f"{entry['key']}: size or sha256 differs from the index")


class CheckpointSync:
    def __init__(
        self,
        *,
        validator_hotkey: str,
        work_dir: str | Path,
        download: Callable[[str, Path], None] = http_download,
        load_anchor: Callable[[list[Path]], dict[str, torch.Tensor]] | None = None,
        verify_signature: Callable[[str, bytes, bytes], bool] | None = None,
        model_dir: str | Path | None = None,
    ) -> None:
        self.validator_hotkey = validator_hotkey
        self.model_dir = None if model_dir is None else Path(model_dir)
        self._work = Path(work_dir)
        self._download = download
        self._load_anchor = load_anchor or load_anchor_shards
        self._verify_signature = verify_signature

    def sync(
        self,
        response: Mapping[str, Any],
        state: dict[str, torch.Tensor] | None,
        version: int | None,
    ) -> tuple[dict[str, torch.Tensor], int, DownloadPlan]:
        """Bring ``state`` (at ``version``, or None) to the index's latest version.

        ``state`` is patched in place when a delta route is taken. Returns the
        rebuilt state, its version and the plan that was executed.
        """
        index = response["index"]
        if response.get("validator_hotkey") != self.validator_hotkey or not verify_index_signature(
            index, response["signature"], self.validator_hotkey, verify_fn=self._verify_signature
        ):
            raise IndexSignatureError("checkpoint index is not signed by the validator")
        urls = response["urls"]
        self.sync_model_files(index, urls)
        if state is not None and version is not None:
            plan = plan_download(index, version)
            if plan.empty:
                return state, version, plan
            if plan.anchor is None:
                try:
                    return self._execute(plan, urls, state), plan.target_version, plan
                except (ChecksumMismatch, DeltaError, ObjectMismatch):
                    # The local state may be half patched: only an anchor is safe now.
                    logger.warning("delta route to %d failed; falling back to an anchor",
                                   plan.target_version, exc_info=True)
        plan = plan_download(index, None)
        return self._execute(plan, urls, None), plan.target_version, plan

    def sync_model_files(self, index: Mapping[str, Any], urls: Mapping[str, str]) -> list[str]:
        """Bring ``model_dir`` to the newest anchor's model files (only the files whose
        SHA-256 differs are fetched). Returns the names written."""
        if self.model_dir is None:
            return []
        written = []
        self.model_dir.mkdir(parents=True, exist_ok=True)
        for entry in latest_model_files(index):
            name = entry.get("name")
            if not model_file_name_ok(name):
                raise ObjectMismatch(f"{name!r}: not a model file name")
            dest = self.model_dir / name
            if dest.is_file() and dest.stat().st_size == entry["size"] and _sha256(dest) == entry["sha256"]:
                continue
            tmp = self.model_dir / f".{name}.part"
            try:
                self._download(urls[entry["key"]], tmp)
                _check_object(tmp, entry)
                os.replace(tmp, dest)
            finally:
                tmp.unlink(missing_ok=True)
            written.append(name)
        if written:
            logger.info("model files updated in %s: %s", self.model_dir, ", ".join(written))
        return written

    def _fetch(self, entry: Mapping[str, Any], urls: Mapping[str, str]) -> Path:
        dest = self._work / entry["key"].replace("/", "__")
        self._download(urls[entry["key"]], dest)
        _check_object(dest, entry)
        return dest

    def _execute(self, plan: DownloadPlan, urls, state):
        self._work.mkdir(parents=True, exist_ok=True)
        try:
            if plan.anchor is not None:
                state = self._load_anchor([self._fetch(f, urls) for f in plan.anchor["files"]])
            for entry in plan.deltas:
                path = self._fetch(entry, urls)
                delta = decode_delta(path.read_bytes())
                path.unlink()
                if delta.target_sha256 != entry["target_sha256"]:
                    raise DeltaError(f"{entry['key']}: target hash differs from the index")
                apply_delta(state, delta, verify=False)
            # One hash at the end: an error in any step shows up here.
            verify_checkpoint(state, plan.target_sha256)
            return state
        finally:
            shutil.rmtree(self._work, ignore_errors=True)
