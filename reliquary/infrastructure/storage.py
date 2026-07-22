"""R2/S3 object storage for window rollout files.

Connection lifecycle: every call to ``get_s3_client()`` creates a FRESH
``aiobotocore`` session + client. We do not cache the session at module
scope. aiobotocore's session owns an internal HTTP connection pool;
when the upstream (Cloudflare R2) intermittently slows or rejects
connections, broken sockets accumulate in the pool and every subsequent
``create_client`` call inherits that bad state. The validator runs
24/7 and uploads ~10-15 files/hr, so a single bad spell can silently
poison the cache for the entire lifetime of the process.

By creating a fresh session per call we pay a small allocation cost
(~ms) on each upload in exchange for **guaranteed clean transport
state**. Tested against the same R2 endpoint under load — the previous
shared-session pattern timed out for hours while a fresh-session client
(in another process) succeeded in under a second on the same call.
"""

import asyncio
import gzip
import json
import logging
import os
from typing import Any

from aiobotocore.session import get_session

from botocore.config import Config

logger = logging.getLogger(__name__)


def get_s3_client(
    account_id: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    bucket_name: str | None = None,
):
    """Create a fresh S3 client context for R2.

    See module docstring for why we do NOT cache the session.

    Timeouts:
    - connect_timeout=15s — generous for transient R2 latency spikes
      (Cloudflare's edge → R2 origin can take 5-10s under load). The
      previous 3s was below typical 99p connect latency, leading to
      false-positive timeouts. 15s still bounds total tail latency:
      with 3 retries × (15s connect + 30s read) the upper bound is
      135s before propagating an exception.
    - read_timeout=30s — unchanged. Sufficient for the ~200-500KB
      gzipped window archives we PUT.
    - retries.max_attempts=3 — bumped from 2 to give one more shot at
      transient failures.
    - retries.mode=standard — default-ish, exponential backoff between
      attempts.
    """
    account_id = account_id or os.getenv("R2_ACCOUNT_ID", "")
    access_key_id = access_key_id or os.getenv("R2_ACCESS_KEY_ID", "")
    secret_access_key = secret_access_key or os.getenv("R2_SECRET_ACCESS_KEY", "")
    endpoint = os.getenv("R2_ENDPOINT_URL") or f"https://{account_id}.r2.cloudflarestorage.com"
    region = os.getenv("R2_REGION", "us-east-1")

    config = Config(
        connect_timeout=15,
        read_timeout=30,
        retries={"max_attempts": 3, "mode": "standard"},
    )
    # Fresh session per call — see module docstring.
    return get_session().create_client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=config,
    )


async def upload_json(key: str, data: Any, **client_kwargs) -> bool:
    """Upload JSON data to S3."""
    payload = json.dumps(data, separators=(",", ":")).encode()
    if key.endswith(".gz"):
        payload = gzip.compress(payload)
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
        await client.put_object(Bucket=bucket, Key=key, Body=payload)
    return True


async def download_json(key: str, **client_kwargs) -> dict | None:
    """Download and parse JSON from S3."""
    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
            resp = await client.get_object(Bucket=bucket, Key=key)
            body = await resp["Body"].read()
            if key.endswith(".gz"):
                body = gzip.decompress(body)
            return json.loads(body)
    except Exception as e:
        logger.debug("download_json failed for %s: %s", key, e)
        return None


def _sync_boto3_put(
    bucket: str, key: str, body: bytes,
    account_id: str, access_key_id: str, secret_access_key: str,
    endpoint: str, region: str,
) -> None:
    """Synchronous boto3 PutObject. Runs in a thread via asyncio.to_thread.

    Each invocation builds its own boto3 client (and underlying urllib3
    HTTP connection pool). This guarantees no shared transport state
    across uploads — the same pattern the miner-side backfill script
    has used reliably against this endpoint.
    """
    # Imported lazily so the module can still be imported on hosts that
    # only have aiobotocore (e.g. tests that mock everything).
    import boto3
    from botocore.config import Config as _SyncConfig

    cfg = _SyncConfig(
        connect_timeout=15,
        read_timeout=30,
        retries={"max_attempts": 3, "mode": "standard"},
    )
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=cfg,
    )
    client.put_object(Bucket=bucket, Key=key, Body=body)


async def upload_window_dataset(
    window_start: int,
    data: dict,
    **client_kwargs,
) -> bool:
    """Upload archive to flat R2 path reliquary/dataset/window-<N>.json.gz.

    The output of this is the actual deliverable of the network: a stream of
    {prompt, completions, rewards} bundles ready to feed a training pipeline.
    The ``validator_hotkey`` is embedded in the archive body for provenance
    (see ``_archive_window``). Paths are flat so any reader — trainer or
    weight-only validator — can enumerate windows without knowing which
    validator wrote them.

    Implementation note: this is the **hot path** — awaited synchronously
    from the validator's main loop after each window seal. We use sync
    boto3 in ``asyncio.to_thread`` here rather than aiobotocore because
    today (2026-05-11) we observed aiobotocore's async path hit recurring
    ConnectTimeoutError under load while sync boto3 against the exact same
    endpoint succeeded in <0.3s from the same host. The Layer 1 fix
    (fresh session per call) helps but doesn't eliminate the aiobotocore
    failure mode entirely; using boto3 here gives us a known-good
    transport for the critical archive PUT.

    Cold-path functions (`upload_json`, `download_json`, etc.) keep using
    the aiobotocore client because they're not in the main-loop hot path
    and a brief failure is non-fatal (they're called from less
    time-sensitive code paths).
    """
    key = f"reliquary/dataset/window-{window_start}.json.gz"
    payload = json.dumps(data, separators=(",", ":")).encode()
    compressed = gzip.compress(payload)

    account_id = client_kwargs.get("account_id") or os.getenv("R2_ACCOUNT_ID", "")
    access_key_id = client_kwargs.get("access_key_id") or os.getenv("R2_ACCESS_KEY_ID", "")
    secret_access_key = client_kwargs.get("secret_access_key") or os.getenv("R2_SECRET_ACCESS_KEY", "")
    endpoint = os.getenv("R2_ENDPOINT_URL") or f"https://{account_id}.r2.cloudflarestorage.com"
    region = os.getenv("R2_REGION", "us-east-1")
    bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")

    await asyncio.to_thread(
        _sync_boto3_put,
        bucket, key, compressed,
        account_id, access_key_id, secret_access_key,
        endpoint, region,
    )

    logger.info(
        "Uploaded GRPO dataset for window %d (%d slots, %d bytes, key=%s)",
        window_start, len(data.get("slots", [])), len(compressed), key,
    )
    return True


async def list_recent_datasets(
    current_window: int,
    n: int,
    **client_kwargs,
) -> list[dict]:
    """Download last *n* window archives from the flat R2 prefix in ascending order.

    Returns a list of parsed archive payloads (the dicts written by
    ``upload_window_dataset``). Tries windows in ``[current_window - n,
    current_window)``; skips any that don't exist or fail to parse.

    Used by the validator at startup to reconstruct ``CooldownMap`` state
    and replay the EMA.
    """
    from botocore.exceptions import ClientError

    if n <= 0 or current_window <= 0:
        return []

    start = max(0, current_window - n)
    keys = [
        (w, f"reliquary/dataset/window-{w}.json.gz")
        for w in range(start, current_window)
    ]

    archives: list[dict] = []
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
        for window_start, key in keys:
            try:
                resp = await client.get_object(Bucket=bucket, Key=key)
                body = await resp["Body"].read()
                data = json.loads(gzip.decompress(body))
                archives.append(data)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("NoSuchKey", "404"):
                    logger.debug("skip missing window %d (%s)", window_start, key)
                    continue
                logger.warning(
                    "skip window %d: %s (%s)", window_start, code, e,
                )
            except Exception as e:
                logger.warning("skip window %d: parse failed (%s)", window_start, e)
    return archives


async def list_all_window_keys(**client_kwargs) -> list[int]:
    """Paginate the flat dataset prefix and return all window_n ints present.

    Used by validators at startup to derive ``window_n`` without local state.
    Returns a sorted ascending list, empty if no archives exist.
    """
    import re
    from botocore.exceptions import ClientError

    bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
    prefix = "reliquary/dataset/window-"
    pattern = re.compile(r"reliquary/dataset/window-(\d+)\.json\.gz$")

    windows: list[int] = []
    async with get_s3_client(**client_kwargs) as client:
        paginator = client.get_paginator("list_objects_v2")
        try:
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []) or []:
                    m = pattern.match(obj["Key"])
                    if m:
                        windows.append(int(m.group(1)))
        except ClientError:
            logger.exception("list_all_window_keys failed")
            return []
    return sorted(windows)
