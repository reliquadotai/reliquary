"""Publishing receipts: each executed pass's receipt as public JSON in the bucket.

A receipt is written once, under
``reliquary/public/payouts/<kind>/<pass day>/<file name>``, and never
changed: the same key with other content is refused. ``index.json`` beside
them lists every published receipt, newest first, for a page to read without
listing the bucket. It is written with a compare-and-swap, so two
publishers cannot erase each other's entries.

Only receipts of executed passes that list a batch are published: a dry run
pays nothing, and an execution stopped before sending anything (at the
owner's confirmation, say) has nothing on chain to show. ``publish`` is a dry
run by default too: it says what it would write, and writes only with
``execute=True``. The bucket's public access (a public domain on that prefix)
is configured outside this code.

The index also lists every paid batch's extrinsic hash: a payer journal that
does not hold one of them is not the journal that paid it
(``published_batches``, read before a journal is initialised or re-aligned).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from reliquary.ledger.payout import MODE_EXECUTE, PAID, PAYOUT_RECEIPT_SCHEMA
from reliquary.ledger.surplus import SURPLUS_RECEIPT_SCHEMA

PUBLIC_PREFIX = "reliquary/public/payouts/"
INDEX_KEY = f"{PUBLIC_PREFIX}index.json"
INDEX_SCHEMA = "reliquary/payout-receipt-index/v1"
# A receipt's schema, and the folder its kind is published under.
RECEIPT_KINDS = {PAYOUT_RECEIPT_SCHEMA: "transfers", SURPLUS_RECEIPT_SCHEMA: "burns"}
INDEX_ATTEMPTS = 5

_ABSENT = {"NoSuchKey", "404", "NotFound"}
_CONFLICT = {"PreconditionFailed", "412", "ConditionalRequestConflict"}


class ReceiptError(RuntimeError):
    """A receipt cannot be published as it is."""


def receipt_key(receipt: Mapping, name: str) -> str:
    kind = RECEIPT_KINDS.get(receipt.get("schema"))
    if kind is None:
        raise ReceiptError(f"not a receipt this publisher knows: {receipt.get('schema')!r}")
    day = str(receipt["pass_day"])
    if not (len(day) == 10 and day[4] == "-" and day[7] == "-" and day.replace("-", "").isdigit()):
        raise ReceiptError(f"a receipt's pass day is not a date: {day!r}")
    if "/" in name or not name.endswith(".json"):
        raise ReceiptError(f"not a receipt file name: {name!r}")
    return f"{PUBLIC_PREFIX}{kind}/{day}/{name}"


def receipt_batches(receipt: Mapping) -> list[Mapping]:
    """Every batch a receipt lists, each once."""
    listed: dict[str, Mapping] = {}
    for name in ("unreported", "recovered", "batches"):
        for batch in receipt.get(name) or []:
            listed.setdefault(str(batch.get("batch_id")), batch)
    return list(listed.values())


def paid_hashes(receipt: Mapping) -> list[str]:
    """The extrinsic hashes of the batches a receipt shows paid."""
    return sorted(str(b["extrinsic_hash"]).lower() for b in receipt_batches(receipt)
                  if b.get("state") in PAID and b.get("extrinsic_hash"))


def index_entry(receipt: Mapping, key: str) -> dict:
    """A receipt's line in the index: what it paid and burned (every receipt
    totals both kinds of batch it lists), and the hashes of its batches that
    landed."""
    paid, burned = receipt.get("paid") or {}, receipt.get("burned") or {}
    return {"key": key, "schema": receipt.get("schema"), "kind": RECEIPT_KINDS.get(receipt.get("schema")),
            "pass_day": receipt.get("pass_day"), "as_of": receipt.get("as_of"), "rao": paid.get("rao"),
            "transfers": paid.get("transfers"), "burned_rao": burned.get("rao"), "stopped": receipt.get("stopped"),
            "paid_batches": paid_hashes(receipt)}


def local_paid_hashes(directory: str | Path) -> set[str]:
    """The paid batches' hashes in the executed receipts of a directory."""
    found: set[str] = set()
    folder = Path(directory)
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            receipt = json.loads(path.read_bytes())
        except ValueError:
            continue
        if isinstance(receipt, Mapping) and receipt.get("schema") in RECEIPT_KINDS \
                and receipt.get("mode") == MODE_EXECUTE:
            found.update(paid_hashes(receipt))
    return found


async def published_batches(*, bucket: str | None = None, **client_kwargs) -> set[str]:
    """Every paid batch's extrinsic hash the bucket's index lists (none when
    there is no index yet). Read-only."""
    from botocore.exceptions import ClientError

    from reliquary.infrastructure import storage

    bucket = bucket or client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
    async with storage.get_s3_client(**client_kwargs) as client:
        try:
            response = await client.get_object(Bucket=bucket, Key=INDEX_KEY)
            index = json.loads(await response["Body"].read())
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code", "") not in _ABSENT:
                raise
            return set()
    return {str(h).lower() for entry in index.get("receipts", []) for h in entry.get("paid_batches") or []}


def merge_index(index: Mapping | None, entries: Iterable[Mapping]) -> dict:
    """``index`` with ``entries`` added (one per key), newest pass first."""
    by_key = {e["key"]: dict(e) for e in (index or {}).get("receipts", [])}
    for entry in entries:
        by_key.setdefault(entry["key"], dict(entry))
    rows = sorted(by_key.values(), key=lambda e: (str(e.get("pass_day")), float(e.get("as_of") or 0.0), e["key"]),
                  reverse=True)
    return {"schema": INDEX_SCHEMA, "receipts": rows}


async def publish(paths: Iterable[str | Path], *, execute: bool = False, bucket: str | None = None,
                  **client_kwargs) -> list[dict]:
    """Each receipt file and what was done with it: ``would_write`` (a dry
    run), ``written``, ``already_published`` (the same bytes are there),
    ``not_executed`` (a dry-run receipt, never published), ``no_batch`` (an
    execution that sent and settled nothing) or ``skipped`` (not a receipt).
    Another receipt under the same key raises ``ReceiptError``."""
    from botocore.exceptions import ClientError

    from reliquary.infrastructure import storage

    bucket = bucket or client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
    results: list[dict] = []
    planned: list[tuple[Path, bytes, Mapping, str]] = []
    for path in sorted(Path(p) for p in paths):
        raw = path.read_bytes()
        try:
            receipt = json.loads(raw)
        except ValueError:
            results.append({"path": str(path), "action": "skipped", "detail": "not JSON"})
            continue
        if not isinstance(receipt, Mapping) or receipt.get("schema") not in RECEIPT_KINDS:
            results.append({"path": str(path), "action": "skipped", "detail": "not a receipt"})
            continue
        if receipt.get("mode") != MODE_EXECUTE:
            results.append({"path": str(path), "action": "not_executed"})
            continue
        if not any(b.get("state") not in ("pending", "unreadable") for b in receipt_batches(receipt)):
            results.append({"path": str(path), "action": "no_batch"})
            continue
        planned.append((path, raw, receipt, receipt_key(receipt, path.name)))
    if not execute:
        results += [{"path": str(path), "key": key, "bytes": len(raw), "action": "would_write"}
                    for path, raw, _, key in planned]
        return results
    if not planned:
        return results
    written: list[dict] = []
    async with storage.get_s3_client(**client_kwargs) as client:
        for path, raw, receipt, key in planned:
            try:
                response = await client.get_object(Bucket=bucket, Key=key)
                existing = await response["Body"].read()
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code", "") not in _ABSENT:
                    raise
                existing = None
            if existing is None:
                await client.put_object(Bucket=bucket, Key=key, Body=raw, ContentType="application/json",
                                        IfNoneMatch="*")
                action = "written"
            elif existing == raw:
                action = "already_published"
            else:
                raise ReceiptError(f"{key} already holds another receipt")
            results.append({"path": str(path), "key": key, "bytes": len(raw), "action": action})
            written.append(index_entry(receipt, key))
        for _ in range(INDEX_ATTEMPTS):
            try:
                response = await client.get_object(Bucket=bucket, Key=INDEX_KEY)
                current, etag = json.loads(await response["Body"].read()), response.get("ETag")
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code", "") not in _ABSENT:
                    raise
                current, etag = None, None
            body = json.dumps(merge_index(current, written), indent=1, sort_keys=True).encode()
            condition = {"IfNoneMatch": "*"} if etag is None else {"IfMatch": etag}
            try:
                await client.put_object(Bucket=bucket, Key=INDEX_KEY, Body=body, ContentType="application/json",
                                        **condition)
                break
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code", "") not in _CONFLICT:
                    raise
        else:
            raise ReceiptError(f"{INDEX_KEY} kept changing under the publisher; run it again")
    return results


__all__ = ["INDEX_KEY", "PUBLIC_PREFIX", "RECEIPT_KINDS", "ReceiptError", "index_entry", "local_paid_hashes",
           "merge_index", "paid_hashes", "publish", "published_batches", "receipt_batches", "receipt_key"]
