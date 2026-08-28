"""CheckpointStore: save → upload to HuggingFace → sign → manifest entry.

Single-validator (v2.1) implementation. The validator owns the
checkpoint lifecycle for the whole netuid; multi-validator consensus
on checkpoint hash is a v2.2 concern.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from reliquary.validator.checkpoint_profile import write_checkpoint_profile

logger = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    """A published checkpoint entry."""

    checkpoint_n: int
    repo_id: str          # HF repo, e.g. "aivolutionedge/reliquary-sn"
    revision: str         # HF commit SHA (serves as strong content hash)
    signature: str        # "ed25519:<hex>" — wallet signs (checkpoint_n || revision)


class _WalletLike(Protocol):
    """Minimal wallet shape — tests inject a stub, prod injects bittensor wallet."""

    class hotkey:
        ss58_address: str
        @staticmethod
        def sign(data: bytes) -> bytes: ...


class _CheckpointSigner(Protocol):
    def sign_checkpoint(
        self, *, checkpoint_n: int, repo_id: str, revision: str
    ) -> bytes: ...


class _WalletCheckpointSigner:
    """Compatibility adapter for the pre-extraction local wallet path."""

    def __init__(self, wallet: _WalletLike) -> None:
        self.wallet = wallet

    def sign_checkpoint(
        self, *, checkpoint_n: int, repo_id: str, revision: str
    ) -> bytes:
        del repo_id  # The existing on-wire checkpoint payload excludes repo_id.
        return bytes(
            self.wallet.hotkey.sign(
                f"{int(checkpoint_n)}|{revision}".encode("utf-8")
            )
        )


class CheckpointStore:
    """Owns the in-memory current manifest + the publish lifecycle.

    Production wiring (defaults):
      * ``save_fn(model, tokenizer, dir)`` → ``model.save_pretrained(dir, safe_serialization=True)``
        + ``tokenizer.save_pretrained(dir)`` — produces safetensors shards,
        ``config.json``, and tokenizer files in one directory so the miner's
        shared text-generation loader can reload them.
      * ``upload_fn(folder_path, repo_id, commit_message)`` → HuggingFace
        ``HfApi.upload_folder`` — one commit covers the whole snapshot.
    Tests inject both as mocks to avoid torch + HF deps.
    """

    def __init__(
        self,
        validator_hotkey: str,
        wallet: _WalletLike,
        repo_id: str,
        staging_dir_path: str,
        *,
        hf_token: str | None = None,
        tokenizer: Any = None,
        upload_fn: Callable[..., Awaitable[str]] | None = None,
        save_fn: Callable[[Any, Any, Path], None] | None = None,
        signer: _CheckpointSigner | None = None,
    ) -> None:
        self.validator_hotkey = validator_hotkey
        self.wallet = wallet
        self.repo_id = repo_id
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.tokenizer = tokenizer
        self.staging_dir = Path(staging_dir_path)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._upload = upload_fn or _default_upload
        self._save = save_fn or _default_save_hf_format
        self._signer = signer or _WalletCheckpointSigner(wallet)
        self._current: ManifestEntry | None = None

    def current_manifest(self) -> ManifestEntry | None:
        return self._current

    async def publish(
        self,
        checkpoint_n: int,
        model: Any,
        profile_extra: dict | None = None,
    ) -> ManifestEntry:
        """Save locally → upload to HF → sign (n || revision) → install manifest.

        The local ``ckpt_<N>`` directory is removed after a successful
        upload (or on any exception in the save/upload pipeline). The HF
        revision is the canonical source — the local copy is staging-only
        and has no role in recovery or miner reload. Keeping it caused
        unbounded disk creep at ~7.6 GB/training step in v2.1.
        """
        # 1. Save HF-format snapshot locally (dir with safetensors + config + tokenizer).
        snapshot_dir = self.staging_dir / f"ckpt_{checkpoint_n}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Serialising the model (multi-GB safetensors) is sync and CPU/IO
            # heavy; run it off the event loop so the HTTP server stays
            # responsive while a checkpoint is being written.
            await asyncio.to_thread(self._save, model, self.tokenizer, snapshot_dir)
            # Bind every published snapshot to the active model/protocol
            # lineage, including snapshots produced by injected save functions.
            write_checkpoint_profile(snapshot_dir, extra=profile_extra)

            # 2. Upload the whole folder to HF — one commit per checkpoint.
            revision = await self._upload(
                folder_path=str(snapshot_dir),
                repo_id=self.repo_id,
                commit_message=f"checkpoint {checkpoint_n}",
            )
        finally:
            # Always delete the staging copy. HF revision is canonical;
            # a half-written dir on save/upload failure is just waste.
            # ``ignore_errors`` so a transient filesystem issue here can't
            # mask a real upload error (which would already have raised).
            shutil.rmtree(snapshot_dir, ignore_errors=True)

        # 3. Sign (n || revision) — strong cross-validator proof
        sig_bytes = await asyncio.to_thread(
            self.sign_manifest,
            checkpoint_n,
            revision,
        )
        signature = "ed25519:" + sig_bytes.hex()

        # 4. Install manifest
        entry = ManifestEntry(
            checkpoint_n=checkpoint_n,
            repo_id=self.repo_id,
            revision=revision,
            signature=signature,
        )
        self._current = entry
        logger.info(
            "Published checkpoint %d to %s@%s",
            checkpoint_n, self.repo_id, revision[:12],
        )
        return entry

    def install_external(
        self, checkpoint_n: int, revision: str
    ) -> ManifestEntry:
        """Install a manifest for a checkpoint published by the detached
        trainer. The wallet signs at INSTALL time — the attestation
        "this is my current checkpoint" only becomes true once the
        verify plane holds these weights, which is the caller's swap."""
        sig_bytes = self.sign_manifest(checkpoint_n, revision)
        entry = ManifestEntry(
            checkpoint_n=int(checkpoint_n),
            repo_id=self.repo_id,
            revision=str(revision),
            signature="ed25519:" + sig_bytes.hex(),
        )
        self._current = entry
        logger.info(
            "Installed external checkpoint %d (%s@%s)",
            checkpoint_n, self.repo_id, str(revision)[:12],
        )
        return entry

    def sign_manifest(self, checkpoint_n: int, revision: str) -> bytes:
        """Sign only the existing structured checkpoint claim."""
        return bytes(
            self._signer.sign_checkpoint(
                checkpoint_n=int(checkpoint_n),
                repo_id=self.repo_id,
                revision=str(revision),
            )
        )


# ---- production defaults (lazy-imported so tests don't drag torch/HF in) ----

async def _default_upload(
    folder_path: str,
    repo_id: str,
    commit_message: str,
) -> str:
    """Upload a snapshot directory via huggingface_hub.HfApi.upload_folder.

    Returns the commit revision SHA (strong hash of the repo state).
    Runs in a thread — HfApi is sync.
    """
    import asyncio
    from huggingface_hub import HfApi

    def _sync_upload():
        api = HfApi()
        commit_info = api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        # CommitInfo.oid holds the commit SHA
        return commit_info.oid

    return await asyncio.to_thread(_sync_upload)


def _default_save_hf_format(model: Any, tokenizer: Any, path: Path) -> None:
    """Save HF-format snapshot: safetensors + config.json + tokenizer files.

    This is what miners expect — the shared text-generation loader needs
    ``config.json`` to choose the architecture. Without it, load fails with
    "Unrecognized model. Should have a `model_type` key in its config.json".
    """
    # safe_serialization=True writes a real safetensors file, not a torch pickle.
    model.save_pretrained(path, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
