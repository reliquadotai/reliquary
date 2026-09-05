"""Validator-side intake of trainer-published checkpoints.

Polls the R2 candidate manifest, downloads the mirrored snapshot in the
background (multipart parallel — the single-stream ceiling on the prod
box is ~20 MB/s, x16 measured 147 MB/s), validates its lineage profile,
and holds it staged until the window loop swaps it on a serial beat.
Every failure here degrades to checkpoint staleness, never a burned
window.
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import Any, Callable

from reliquary.shared.checkpoint_identity import (
    canonical_checkpoint_identity,
    require_checkpoint_number,
    require_checkpoint_repository,
    require_checkpoint_successor,
    require_immutable_checkpoint_revision,
)
from reliquary.shared.strict_json import strict_json_loads

logger = logging.getLogger(__name__)

CANDIDATE_MANIFEST_KEY = "reliquary/training/candidate-manifest.json"
R2_CHECKPOINT_PREFIX = "reliquary/checkpoints"


def default_r2_client():
    """Sync boto3 client from the validator's R2 env (same variables as
    infrastructure.storage)."""
    import os

    import boto3
    from botocore.config import Config

    account_id = os.getenv("R2_ACCOUNT_ID", "")
    endpoint = (
        os.getenv("R2_ENDPOINT_URL")
        or f"https://{account_id}.r2.cloudflarestorage.com"
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", ""),
        region_name=os.getenv("R2_REGION", "us-east-1"),
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"},
            read_timeout=120,
            max_pool_connections=32,
        ),
    )


def _default_validate(path: str | Path) -> dict[str, Any]:
    from reliquary.validator.checkpoint_profile import (
        validate_checkpoint_profile,
    )

    profile = validate_checkpoint_profile(path, required=True)
    assert profile is not None
    return profile


def _multipart_transfer_config():
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=32 * 1024 * 1024,
        multipart_chunksize=32 * 1024 * 1024,
        max_concurrency=16,
    )


class CheckpointIntake:
    def __init__(
        self,
        *,
        r2_client: Any,
        bucket: str,
        staging_dir: str,
        installed_revision: str | None = None,
        installed_checkpoint_n: int | None = None,
        installed_repo_id: str | None = None,
        validate_fn: Callable[[Path], dict[str, Any]] = _default_validate,
        expected_identity: dict[str, Any] | None = None,
    ) -> None:
        self._r2 = r2_client
        self._bucket = bucket
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.installed_revision = (
            require_immutable_checkpoint_revision(
                installed_revision,
                field="installed checkpoint revision",
            )
            if installed_revision is not None
            else None
        )
        self.installed_checkpoint_n = (
            require_checkpoint_number(
                installed_checkpoint_n,
                field="installed checkpoint number",
            )
            if installed_checkpoint_n is not None
            else None
        )
        self.installed_repo_id = (
            require_checkpoint_repository(
                installed_repo_id,
                field="installed checkpoint repository",
            )
            if installed_repo_id is not None
            else None
        )
        if self.installed_checkpoint_n is not None and (
            self.installed_revision is None or self.installed_repo_id is None
        ):
            raise ValueError(
                "installed checkpoint number requires repository and revision"
            )
        self._validate = validate_fn
        self._expected_identity = dict(expected_identity or {})
        self._staged: tuple[dict[str, Any], Path] | None = None
        self._staging_revision: str | None = None
        self._staging_identity: tuple[int, str, str] | None = None
        self._taken_manifest: dict[str, Any] | None = None
        self.last_error: str | None = None

    @property
    def staged_ready(self) -> bool:
        return self._staged is not None

    @property
    def staged_revision(self) -> str | None:
        return self._staged[0]["revision"] if self._staged else None

    @staticmethod
    def _manifest_identity(manifest: dict[str, Any]) -> tuple[int, str, str]:
        return canonical_checkpoint_identity(
            manifest.get("checkpoint_n"),
            manifest.get("repo_id"),
            manifest.get("revision"),
            field="candidate checkpoint",
        )

    def _require_successor(
        self,
        candidate: tuple[int, str, str],
    ) -> None:
        if self.installed_checkpoint_n is not None:
            assert self.installed_repo_id is not None
            assert self.installed_revision is not None
            require_checkpoint_successor(
                (
                    self.installed_checkpoint_n,
                    self.installed_repo_id,
                    self.installed_revision,
                ),
                candidate,
                field="candidate checkpoint",
            )
        if self._staged is not None:
            require_checkpoint_successor(
                self._manifest_identity(self._staged[0]),
                candidate,
                field="candidate checkpoint",
            )
        if self._staging_identity is not None:
            require_checkpoint_successor(
                self._staging_identity,
                candidate,
                field="candidate checkpoint",
            )
        if self._taken_manifest is not None:
            require_checkpoint_successor(
                self._manifest_identity(self._taken_manifest),
                candidate,
                field="candidate checkpoint",
            )

    def poll(self) -> dict[str, Any] | None:
        """Return a NEW candidate manifest, or None. Never raises."""
        try:
            body = self._r2.get_object(
                Bucket=self._bucket, Key=CANDIDATE_MANIFEST_KEY,
            )["Body"].read()
        except Exception:
            return None
        try:
            manifest = strict_json_loads(body)
            if not isinstance(manifest, dict):
                raise TypeError("candidate manifest must be an object")
            candidate = self._manifest_identity(manifest)
            _, _, revision = candidate
            self._require_successor(candidate)
        except (ValueError, KeyError, TypeError):
            logger.warning("malformed candidate manifest; ignoring")
            self.last_error = "candidate manifest checkpoint identity is invalid"
            return None
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in self._expected_identity.items()
            if manifest.get(key) != expected
        }
        if mismatches:
            self.last_error = (
                "candidate manifest identity mismatch: "
                + ", ".join(sorted(mismatches))
            )
            return None
        if revision in {
            self.installed_revision,
            self.staged_revision,
            self._staging_revision,
        }:
            return None
        return manifest

    def stage(self, manifest: dict[str, Any]) -> bool:
        """Download + validate one candidate snapshot (sync; callers run
        it off the loop). Returns True when staged."""
        revision = "<invalid>"
        dest: Path | None = None
        try:
            if not isinstance(manifest, dict):
                raise TypeError("candidate manifest must be an object")
            candidate = self._manifest_identity(manifest)
            _, _, revision = candidate
            self._require_successor(candidate)
            mismatches = {
                key: (manifest.get(key), expected)
                for key, expected in self._expected_identity.items()
                if manifest.get(key) != expected
            }
            if mismatches:
                raise ValueError("candidate manifest identity mismatch")
            staging_root = self.staging_dir.resolve()
            unresolved_dest = staging_root / revision
            if unresolved_dest.is_symlink():
                raise ValueError("candidate checkpoint path cannot be a symlink")
            candidate_dest = unresolved_dest.resolve()
            try:
                candidate_dest.relative_to(staging_root)
            except ValueError as exc:
                raise ValueError(
                    "candidate checkpoint path escapes the staging directory"
                ) from exc
            dest = candidate_dest
            self._staging_revision = revision
            self._staging_identity = candidate
            prefix = f"{R2_CHECKPOINT_PREFIX}/{revision}/"
            listed = self._r2.list_objects_v2(
                Bucket=self._bucket, Prefix=prefix,
            )
            contents = listed.get("Contents", [])
            if not contents:
                raise RuntimeError(f"R2 mirror has no objects under {prefix}")
            config = _multipart_transfer_config()
            dest.mkdir(parents=True, exist_ok=True)
            for obj in contents:
                key = obj["Key"]
                if not isinstance(key, str) or not key.startswith(prefix):
                    raise ValueError("checkpoint object escaped its revision prefix")
                filename = key[len(prefix):]
                if (
                    not filename
                    or "/" in filename
                    or "\\" in filename
                    or filename in {".", ".."}
                ):
                    continue
                target = (dest / filename).resolve()
                try:
                    target.relative_to(dest)
                except ValueError as exc:
                    raise ValueError(
                        "checkpoint object escaped the staged revision"
                    ) from exc
                self._r2.download_file(
                    self._bucket, key, str(target), Config=config,
                )
            self._validate(dest)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "checkpoint staging failed for %s; validator stays on the "
                "current revision", revision,
            )
            if dest is not None:
                shutil.rmtree(dest, ignore_errors=True)
            return False
        finally:
            self._staging_revision = None
            self._staging_identity = None
        # A newer stage replaces an older unswapped one.
        if self._staged is not None:
            shutil.rmtree(self._staged[1], ignore_errors=True)
        self._staged = (dict(manifest), dest)
        self.last_error = None
        logger.info("checkpoint %s staged for swap", revision[:12])
        return True

    def take_staged(self) -> tuple[dict[str, Any], Path]:
        """Hand the staged snapshot to the swapper (exactly once)."""
        if self._staged is None:
            raise RuntimeError("no staged checkpoint")
        staged = self._staged
        self._staged = None
        self._taken_manifest = staged[0]
        return staged

    def mark_installed(self, revision: str, staged_dir: Path) -> None:
        revision = require_immutable_checkpoint_revision(
            revision,
            field="installed checkpoint revision",
        )
        if self._taken_manifest is None:
            raise RuntimeError("no taken checkpoint can be marked installed")
        candidate = self._manifest_identity(self._taken_manifest)
        if candidate[2] != revision:
            raise ValueError("installed checkpoint revision mismatch")
        self._require_successor(candidate)
        self.installed_checkpoint_n = candidate[0]
        self.installed_repo_id = candidate[1]
        self.installed_revision = candidate[2]
        self._taken_manifest = None
        shutil.rmtree(staged_dir, ignore_errors=True)

    def snapshot(self) -> dict[str, Any]:
        return {
            "installed_revision": self.installed_revision,
            "staged_revision": self.staged_revision,
            "staging_revision": self._staging_revision,
            "last_error": self.last_error,
        }
