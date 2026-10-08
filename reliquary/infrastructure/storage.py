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
import re
import threading
from typing import Any

from aiobotocore.session import get_session

from botocore.config import Config

from reliquary.shared.strict_json import strict_json_loads
from reliquary.shared.task_id import TASK_ID_RE as _TASK_ID_RE, normalise_task_id

logger = logging.getLogger(__name__)


def _task_id(task_id: str | None) -> str:
    """The task whose archives we are addressing. Env-read like the R2 config."""
    resolved = task_id if task_id is not None else os.getenv("RELIQUARY_TASK_ID")
    return normalise_task_id(resolved)


def dataset_prefix(task_id: str | None = None) -> str:
    """Where a task's window archives live. ``default`` keeps the legacy flat path."""
    resolved = _task_id(task_id)
    if resolved == "default":
        return "reliquary/dataset/window-"
    return f"reliquary/tasks/{resolved}/dataset/window-"


def dataset_object_key(window_start: int, task_id: str | None = None) -> str:
    return f"{dataset_prefix(task_id)}{int(window_start)}.json.gz"


_LISTING_LOOP: asyncio.AbstractEventLoop | None = None
_LISTING_LOOP_LOCK = threading.Lock()


def _listing_loop() -> asyncio.AbstractEventLoop:
    global _LISTING_LOOP
    with _LISTING_LOOP_LOCK:
        if _LISTING_LOOP is None or _LISTING_LOOP.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="r2-listing", daemon=True).start()
            _LISTING_LOOP = loop
        return _LISTING_LOOP


async def off_loop(coro):
    """Await ``coro`` run on a dedicated thread's own event loop.

    A big listing's pages are parsed (XML) in the coroutine that reads them;
    on the serving loop that held it for seconds (py-spy, 2026-10-01). On its
    own loop the parse only competes for the GIL. Cancelling the caller
    cancels the listing.
    """
    if asyncio.get_running_loop() is _LISTING_LOOP:
        return await coro
    future = asyncio.run_coroutine_threadsafe(coro, _listing_loop())
    return await asyncio.wrap_future(future)


async def list_task_ids(*, strict: bool = False, **client_kwargs) -> list[str]:
    """Every task with an archive namespace, ``default`` always included."""
    from botocore.exceptions import ClientError

    bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
    tasks = {"default"}
    async with get_s3_client(**client_kwargs) as client:
        paginator = client.get_paginator("list_objects_v2")
        try:
            async for page in paginator.paginate(
                Bucket=bucket, Prefix="reliquary/tasks/", Delimiter="/"
            ):
                for entry in page.get("CommonPrefixes", []) or []:
                    candidate = entry.get("Prefix", "")[len("reliquary/tasks/"):].strip("/")
                    if _TASK_ID_RE.match(candidate):
                        tasks.add(candidate)
        except ClientError:
            if strict:
                raise
            logger.exception("list_task_ids failed")
    return sorted(tasks)


def get_s3_client(
    account_id: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    bucket_name: str | None = None,
    max_pool_connections: int | None = None,
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
        # botocore's default (10) unless a caller needs more calls in flight.
        **({"max_pool_connections": max_pool_connections} if max_pool_connections else {}),
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


def _encode_json_payload(key: str, data: Any) -> bytes:
    payload = json.dumps(data, separators=(",", ":")).encode()
    if key.endswith(".gz"):
        payload = gzip.compress(payload)
    return payload


async def _read_object_body(response: dict) -> bytes:
    body = response["Body"]
    try:
        return await body.read()
    finally:
        body.close()


async def upload_json(key: str, data: Any, **client_kwargs) -> bool:
    """Upload JSON without serializing or compressing on the event loop."""
    payload = await asyncio.to_thread(_encode_json_payload, key, data)
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
        await client.put_object(Bucket=bucket, Key=key, Body=payload)
    return True


async def upload_bytes(key: str, body: bytes, *, content_type: str, cache_control: str,
                       if_absent: bool = False, **client_kwargs) -> bool:
    """PUT raw bytes with explicit cache headers (public static files).

    ``if_absent`` makes the PUT conditional (``If-None-Match: *``): an existing key is never
    replaced. If it already exists its bytes must be identical (an idempotent re-upload after a
    crash); different bytes raise ``ValueError`` and nothing is written.
    """
    from botocore.exceptions import ClientError

    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
        extra = {"IfNoneMatch": "*"} if if_absent else {}
        try:
            await client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type,
                                    CacheControl=cache_control, **extra)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if not if_absent or (code not in ("PreconditionFailed", "412") and status != 412):
                raise
            existing = await _read_object_body(await client.get_object(Bucket=bucket, Key=key))
            if existing != body:
                raise ValueError(f"immutable object {key} already exists with different bytes") from exc
    return True


async def download_bytes(key: str, **client_kwargs) -> bytes | None:
    """The raw bytes of an object; None only when it does not exist (any other failure raises)."""
    from botocore.exceptions import ClientError

    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
            return await _read_object_body(await client.get_object(Bucket=bucket, Key=key))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code", "") in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise


async def download_json(
    key: str,
    *,
    strict: bool = False,
    **client_kwargs,
) -> dict | None:
    """Download and parse JSON from S3.

    The compatibility default preserves the historic best-effort behavior.
    Durable protocol state uses ``strict=True`` so only an explicit 404 means
    "absent"; transport, authentication, and decode failures remain errors.
    """
    from botocore.exceptions import ClientError

    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
            resp = await client.get_object(Bucket=bucket, Key=key)
            body = await _read_object_body(resp)
            def _decode() -> dict:
                decoded = gzip.decompress(body) if key.endswith(".gz") else body
                return strict_json_loads(decoded)
            return await asyncio.to_thread(_decode)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"NoSuchKey", "404", "NotFound"}:
            return None
        if strict:
            raise
        logger.debug("download_json failed for %s: %s", key, exc)
        return None
    except Exception as e:
        if strict:
            raise
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
    *,
    task_id: str | None = None,
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
    key = dataset_object_key(window_start, task_id)
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
    *,
    strict: bool = False,
    task_id: str | None = None,
    fields: tuple[str, ...] | None = None,
    row_fields: dict[str, tuple[str, ...]] | None = None,
    **client_kwargs,
) -> list[dict]:
    """Download last *n* window archives from the flat R2 prefix in ascending order.

    Returns a list of parsed archive payloads (the dicts written by
    ``upload_window_dataset``). Tries windows in ``[current_window - n,
    current_window)``; skips any that don't exist or fail to parse.

    Used by the validator at startup to reconstruct ``CooldownMap`` state
    and replay the EMA.

    ``row_fields`` (only with ``fields``): ``{field: keys}`` projects each row of the list
    ``field`` to ``keys`` right after decoding, so the rows' bulk is never retained. A non-dict
    row becomes ``None`` and a non-list value is kept as is (the consumer refuses both).
    """
    from botocore.exceptions import ClientError

    if n <= 0 or current_window <= 0:
        return []

    start = max(0, current_window - n)
    keys = [
        (w, dataset_object_key(w, task_id))
        for w in range(start, current_window)
    ]

    archives: list[dict] = []
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
        async def download(window_start, key):
            try:
                resp = await client.get_object(Bucket=bucket, Key=key)
                body = await _read_object_body(resp)
                def decode():
                    return strict_json_loads(gzip.decompress(body))
                data = await asyncio.to_thread(decode)
                if (
                    not isinstance(data, dict)
                    or data.get("window_start") != window_start
                ):
                    raise ValueError(
                        f"archive {key} does not bind window {window_start}"
                    )
                if fields is None:
                    return data
                projected = {field: data[field] for field in fields if field in data}
                del data
                for field, keys in (row_fields or {}).items():
                    rows = projected.get(field)
                    if isinstance(rows, list):
                        projected[field] = [
                            {k: row[k] for k in keys if k in row} if isinstance(row, dict) else None
                            for row in rows
                        ]
                return projected
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("NoSuchKey", "404"):
                    logger.debug("skip missing window %d (%s)", window_start, key)
                    return None
                if strict:
                    raise
                logger.warning(
                    "skip window %d: %s (%s)", window_start, code, e,
                )
            except Exception as e:
                if strict:
                    raise
                logger.warning("skip window %d: parse failed (%s)", window_start, e)
        # Bound both downloads and decoded payloads; preserve chronological
        # results and finish/cancel every task before closing its S3 client.
        # Projected reward replay deliberately retains its one-archive memory
        # ceiling; full recovery already requests bounded archive chunks.
        concurrency = 1 if fields is not None else 4
        for offset in range(0, len(keys), concurrency):
            tasks = [asyncio.create_task(download(*item)) for item in keys[offset:offset + concurrency]]
            try:
                archives.extend(item for item in await asyncio.gather(*tasks) if item is not None)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    return archives


async def list_all_window_keys(
    *,
    strict: bool = False,
    task_id: str | None = None,
    **client_kwargs,
) -> list[int]:
    """Paginate the flat dataset prefix and return all window_n ints present.

    Used by validators at startup to derive ``window_n`` without local state.
    Returns a sorted ascending list, empty if no archives exist.
    """
    from botocore.exceptions import ClientError

    bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
    prefix = dataset_prefix(task_id)
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.json\.gz$")

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
            if strict:
                raise
            logger.exception("list_all_window_keys failed")
            return []
    return sorted(windows)
