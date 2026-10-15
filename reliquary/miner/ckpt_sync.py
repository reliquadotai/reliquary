"""Miner side of checkpoint deltas: verify the signed index, take the cheapest
route to the latest version, rebuild bit-exactly and check the SHA-256; on any
mismatch start again from an anchor (docs/rl-async-design.md §4.8).
"""

from __future__ import annotations

import hashlib
import logging
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


def _check_object(path: Path, entry: Mapping[str, Any]) -> None:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    if path.stat().st_size != entry["size"] or h.hexdigest() != entry["sha256"]:
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
    ) -> None:
        self.validator_hotkey = validator_hotkey
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
