"""Object-store adapters used by the eval producer."""

from __future__ import annotations

import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Protocol


class ObjectNotFound(FileNotFoundError):
    pass


class ObjectAlreadyExists(FileExistsError):
    pass


class ObjectStore(Protocol):
    def get(self, key: str) -> bytes: ...

    def put(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: str,
        metadata: dict[str, str] | None = None,
        if_absent: bool = False,
    ) -> None: ...

    def list(self, prefix: str) -> list[str]: ...


def _safe_key(key: str) -> PurePosixPath:
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe object key: {key!r}")
    return path


class FileObjectStore:
    """Filesystem-backed store for dry-runs and failure-free integration tests."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = self.root.joinpath(*_safe_key(key).parts).resolve()
        if self.root not in path.parents:
            raise ValueError(f"unsafe object key: {key!r}")
        return path

    def get(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError as exc:
            raise ObjectNotFound(key) from exc

    def put(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: str,
        metadata: dict[str, str] | None = None,
        if_absent: bool = False,
    ) -> None:
        del content_type, metadata
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".tmp-{uuid.uuid4().hex}")
        temporary.write_bytes(payload)
        if if_absent:
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise ObjectAlreadyExists(key) from exc
            finally:
                temporary.unlink(missing_ok=True)
            return
        temporary.replace(target)

    def list(self, prefix: str) -> list[str]:
        prefix_path = _safe_key(prefix)
        base = self.root.joinpath(*prefix_path.parts)
        if not base.exists():
            return []
        if base.is_file():
            return [prefix]
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in base.rglob("*")
            if path.is_file() and ".tmp-" not in path.name
        )


class S3ObjectStore:
    """Synchronous S3/R2 adapter with bounded transport retries."""

    def __init__(self, client, bucket: str) -> None:
        self.client = client
        self.bucket = bucket

    @classmethod
    def from_env(cls) -> "S3ObjectStore":
        import boto3
        from botocore.config import Config

        account_id = os.getenv("RELIQUARY_EVAL_R2_ACCOUNT_ID") or os.getenv(
            "R2_ACCOUNT_ID", ""
        )
        endpoint = (
            os.getenv("RELIQUARY_EVAL_R2_S3_ENDPOINT")
            or os.getenv("R2_ENDPOINT_URL")
            or (f"https://{account_id}.r2.cloudflarestorage.com" if account_id else "")
        )
        access_key = os.getenv("RELIQUARY_EVAL_R2_ACCESS_KEY_ID") or os.getenv(
            "R2_ACCESS_KEY_ID", ""
        )
        secret_key = os.getenv("RELIQUARY_EVAL_R2_SECRET_ACCESS_KEY") or os.getenv(
            "R2_SECRET_ACCESS_KEY", ""
        )
        bucket = (
            os.getenv("RELIQUARY_EVAL_R2_BUCKET")
            or os.getenv("R2_BUCKET_ID")
            or "reliquary"
        )
        region = os.getenv("RELIQUARY_EVAL_R2_REGION") or os.getenv("R2_REGION", "auto")
        if not endpoint or not access_key or not secret_key:
            raise RuntimeError("eval R2 credentials are incomplete")
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(
                connect_timeout=15,
                read_timeout=60,
                retries={"max_attempts": 4, "mode": "standard"},
                s3={"addressing_style": "path"},
            ),
        )
        put_shape = client.meta.service_model.operation_model("PutObject").input_shape
        if "IfNoneMatch" not in put_shape.members:
            raise RuntimeError(
                "installed botocore cannot enforce immutable S3 writes; upgrade boto3"
            )
        return cls(client, bucket)

    def get(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        _safe_key(key)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "404", "NotFound"}:
                raise ObjectNotFound(key) from exc
            raise

    def put(
        self,
        key: str,
        payload: bytes,
        *,
        content_type: str,
        metadata: dict[str, str] | None = None,
        if_absent: bool = False,
    ) -> None:
        from botocore.exceptions import ClientError

        _safe_key(key)
        kwargs = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": payload,
            "ContentType": content_type,
            "CacheControl": (
                "public, max-age=31536000, immutable"
                if if_absent
                else (
                    "no-cache"
                    if key.endswith(("index.json", "status.json"))
                    else "public, max-age=300"
                )
            ),
            "Metadata": metadata or {},
        }
        if if_absent:
            kwargs["IfNoneMatch"] = "*"
        try:
            self.client.put_object(**kwargs)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if if_absent and (code in {"PreconditionFailed", "412"} or status == 412):
                raise ObjectAlreadyExists(key) from exc
            raise

    def list(self, prefix: str) -> list[str]:
        _safe_key(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []) or [])
        return sorted(keys)
