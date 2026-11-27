"""What the payout ledger reads, and only reads.

- period archives: the bucket's ``reliquary/corpus-periods/`` prefix
  (``corpus_period_store``), listed once with each object's time (when it was
  settled), or a local directory laid out the same way;
- price ceilings: the task registry (each task's params), or the public job
  status (``pay.price_ceiling``);
- work awaiting audit: the public ``/corpus/pay`` and job statuses;
- the miners' incentive per epoch: the chain, at historical blocks (an archive
  node), through the SDK's metagraph reads.

No function here writes, signs or holds a key: list, get, and chain queries.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from reliquary.ledger.obligations import (
    PRICING_AUCTION,
    PRICING_FIXED,
    ArchiveRef,
    Ceiling,
    PendingWork,
)
from reliquary.ledger.reconcile import EpochPay
from reliquary.validator import corpus_periods as cp

logger = logging.getLogger(__name__)

ARCHIVE_READ_CONCURRENCY = 16
HTTP_TIMEOUT_SECONDS = 30.0
USER_AGENT = "reliquary-ledger/1"
DEFAULT_API = "https://api.reliqua.ai"
CHAIN_READ_TIMEOUT_SECONDS = 120.0
CHAIN_READ_ATTEMPTS = 4
CHAIN_READ_BACKOFF_SECONDS = 3.0
# What one epoch read fetches of the metagraph.
EPOCH_FIELDS = ("Hotkeys", "Coldkeys", "Emission", "Dividends", "Incentives", "LastStep", "Tempo",
                "OwnerHotkey", "OwnerColdkey")


# -- period archives -------------------------------------------------------------------------------------------


async def list_bucket_archives(**client_kwargs) -> list[tuple[str, int, int, float | None]]:
    """Every period archive in the bucket: (task id, work period, entry period,
    the object's time), in one paginated listing of the prefix."""
    from reliquary.infrastructure import storage
    from reliquary.infrastructure.corpus_period_store import PERIODS_PREFIX, parse_key

    bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
    found = []
    async with storage.get_s3_client(**client_kwargs) as client:
        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket, Prefix=PERIODS_PREFIX):
            for obj in page.get("Contents", []) or []:
                row = _bucket_row(obj["Key"], obj.get("LastModified"), PERIODS_PREFIX, parse_key)
                if row is not None:
                    found.append(row)
    return sorted(found, key=lambda row: row[:3])


def _bucket_row(key: str, modified, prefix: str, parse_key) -> tuple[str, int, int, float | None] | None:
    task, _, name = key[len(prefix):].partition("/") if key.startswith(prefix) else ("", "", "")
    parsed = parse_key(name) if task and name and "/" not in name else None
    if parsed is None:
        return None
    return task, parsed[0], parsed[1], None if modified is None else float(modified.timestamp())


def selected(task: str, work: int, entry: int, *, work_from: int, work_to: int | None = None,
             entry_from: int | None = None, full_tasks: Iterable[str] = ()) -> bool:
    """Whether to read an archive: every archive of a task in ``full_tasks`` (its
    whole history: what is still owed never drops out of view); else one whose
    work is in [work_from, work_to], or that entered from ``entry_from`` on (a
    late part of older work, settled recently; ``entry_from`` defaults to
    ``work_from``)."""
    if task in set(full_tasks):
        return True
    if work >= work_from and (work_to is None or work <= work_to):
        return True
    entry_from = work_from if entry_from is None else entry_from
    return work_to is None and entry >= entry_from


class BucketArchives:
    """The period archives in the bucket, read-only."""

    def __init__(self, *, concurrency: int = ARCHIVE_READ_CONCURRENCY, **client_kwargs) -> None:
        self._concurrency = concurrency
        self._client_kwargs = client_kwargs

    async def refs(self, *, work_from: int, work_to: int | None = None, tasks: Iterable[str] | None = None,
                   entry_from: int | None = None, full_tasks: Iterable[str] = ()) -> list[ArchiveRef]:
        """The archives ``selected`` keeps, of ``tasks`` or of every task."""
        from reliquary.infrastructure.corpus_period_store import read_period_archive

        wanted = None if tasks is None else {str(task) for task in tasks}
        full = {str(task) for task in full_tasks}
        listed = [row for row in await list_bucket_archives(**self._client_kwargs)
                  if (wanted is None or row[0] in wanted)
                  and selected(row[0], row[1], row[2], work_from=work_from, work_to=work_to,
                               entry_from=entry_from, full_tasks=full)]
        gate = asyncio.Semaphore(self._concurrency)

        async def one(task, work, entry, modified):
            async with gate:
                doc = await read_period_archive(task, work, entry)
            if doc is None:
                raise RuntimeError(f"period archive {task} {work}-{entry} listed but unreadable")
            return ArchiveRef(task_id=task, work_period=work, entry_period=entry, document=doc,
                              settled_at=modified)

        return list(await asyncio.gather(*(one(*row) for row in listed)))


class DirectoryArchives:
    """Period archives copied to a local directory, laid out as in the bucket:
    ``<root>/<task id>/<work period>-<entry period>.json[.gz]`` (both periods
    zero-padded to 10 digits). A file's modification time stands for when it
    was settled, unless its body carries ``settled_at``."""

    def __init__(self, root: str | os.PathLike) -> None:
        self._root = Path(root)

    async def refs(self, *, work_from: int, work_to: int | None = None, tasks: Iterable[str] | None = None,
                   entry_from: int | None = None, full_tasks: Iterable[str] = ()) -> list[ArchiveRef]:
        from reliquary.infrastructure.corpus_period_store import parse_key

        wanted = None if tasks is None else {str(task) for task in tasks}
        full = {str(task) for task in full_tasks}
        out = []
        for path in sorted(self._root.glob("*/*.json*")):
            task = path.parent.name
            parsed = parse_key(path.name if path.name.endswith(".gz") else path.name + ".gz")
            if parsed is None or (wanted is not None and task not in wanted):
                continue
            work, entry = parsed
            if not selected(task, work, entry, work_from=work_from, work_to=work_to, entry_from=entry_from,
                            full_tasks=full):
                continue
            raw = path.read_bytes()
            doc = json.loads(gzip.decompress(raw) if path.name.endswith(".gz") else raw)
            settled = doc.get("settled_at", path.stat().st_mtime)
            out.append(ArchiveRef(task_id=task, work_period=work, entry_period=entry, document=doc,
                                  settled_at=float(settled)))
        return out


# -- price ceilings --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegistryView:
    """What the ledger and the payer read of the registry: each period task's
    price ceiling (with its pricing, known for sure here: no ``pricing`` param
    is a fixed ceiling), each one's pay ceiling (``cp.pay_ceiling``), the tasks
    declaring an incentive floor (the weight setter then pays their small
    hotkeys less than they earned), and the tasks on the collector rail."""

    ceilings: dict[str, list[Ceiling]]
    pay_ceilings: dict[str, float]
    floored: list[str]
    collector_tasks: list[str]


def registry_ceilings(declared: Mapping) -> RegistryView:
    """The ``RegistryView`` of the registry's entries."""
    from reliquary.shared.task_registry import PAYOUT_RAIL_COLLECTOR, payout_rail

    ceilings: dict[str, list[Ceiling]] = {}
    pay_ceilings: dict[str, float] = {}
    floored: list[str] = []
    collector: list[str] = []
    for task_id, entry in (declared or {}).items():
        params = getattr(entry, "params", None)
        if not isinstance(params, Mapping) or not cp.is_period_task(entry):
            continue
        view = cp.price_view(params)
        if view is not None and view.get("miner_alpha_per_period") is not None:
            pricing = PRICING_AUCTION if params.get("pricing") == PRICING_AUCTION else PRICING_FIXED
            ceilings[str(task_id)] = [Ceiling.from_view(view, pricing=pricing)]
        bound = cp.pay_ceiling(params)
        if bound is not None:
            pay_ceilings[str(task_id)] = bound
        if float(params.get("min_incentive_share") or 0.0) > 0:
            floored.append(str(task_id))
        if payout_rail(params) == PAYOUT_RAIL_COLLECTOR:
            collector.append(str(task_id))
    return RegistryView(ceilings=ceilings, pay_ceilings=pay_ceilings, floored=sorted(floored),
                        collector_tasks=sorted(collector))


async def read_registry_ceilings(**client_kwargs) -> RegistryView:
    """``registry_ceilings`` of the registry in the bucket (read, never written)."""
    from reliquary.infrastructure.task_registry_store import read_registry

    declared, _ = await read_registry(strict=False, **client_kwargs)
    return registry_ceilings(declared)


def http_json(url: str, *, timeout: float = HTTP_TIMEOUT_SECONDS):
    """GET a JSON document; None on 404."""
    if urllib.parse.urlparse(url).scheme not in ("https", "http"):
        raise ValueError(f"not an http(s) URL: {url!r}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - scheme checked above
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def api_ceilings(api: str, job_ids: Iterable[str], *, pricing: Mapping[str, str] | None = None,
                 fetch=http_json) -> dict[str, list[Ceiling]]:
    """Each job's price ceiling from its public status (``pay.price_ceiling``),
    keyed by job id (an order's task id is its job id). ``pricing``: what the
    public order list says of each job (``auction``), when known."""
    out: dict[str, list[Ceiling]] = {}
    for job_id in sorted(set(job_ids)):
        status = fetch(f"{api.rstrip('/')}/corpus/jobs/{urllib.parse.quote(job_id, safe='')}/status")
        view = ((status or {}).get("pay") or {}).get("price_ceiling")
        if view and view.get("miner_alpha_per_period") is not None:
            out[job_id] = [Ceiling.from_view(view, pricing=(pricing or {}).get(job_id))]
    return out


def api_pending(api: str, *, now: float, settled: Iterable[tuple[str, int]] = (), fetch=http_json,
                first_seen: Mapping[str, float] | None = None) -> tuple[list[PendingWork], dict[str, str]]:
    """Work awaiting audit, from the public routes: each served job's unsettled
    periods and oldest undecided arrival (``/corpus/pay``), and each order whose
    accepted submissions are not all audited (``/corpus/jobs``, its status).
    Also each order's pricing, as the order list gives it. ``settled``: the
    (task, work period) already archived (a later part is still pending).
    ``first_seen``: when a job's unaudited work was first seen, by job id."""
    base = api.rstrip("/")
    settled = {(str(task), int(work)) for task, work in settled}
    board = fetch(f"{base}/corpus/pay") or {}
    pending: list[PendingWork] = []
    for job in board.get("jobs") or []:
        job_id, task_id = str(job["job_id"]), str(job.get("task_id") or job["job_id"])
        price = job.get("max_price_alpha_per_mtoken")
        for row in job.get("unsettled_periods") or []:
            work, tokens = int(row["work_period"]), int(row.get("verified_tokens") or 0)
            pending.append(PendingWork(
                job_id=job_id, task_id=task_id, work_period=work, since=cp.period_end(work),
                verified_tokens=tokens, reason="settled_in_part" if (task_id, work) in settled else "unsettled",
                ceiling_alpha=None if price is None else _ceiling_alpha(tokens, float(price), job, board)))
        lag = job.get("settlement_lag_seconds")
        if lag:
            oldest = float(now) - float(lag)
            held = cp.period_of(oldest)
            if cp.period_end(held) <= now:
                # The oldest undecided arrival holds its period, and every later one, from closing.
                pending.append(PendingWork(job_id=job_id, task_id=task_id, work_period=held, since=oldest,
                                           reason="undecided"))
    pricing: dict[str, str] = {}
    listed = fetch(f"{base}/corpus/jobs") or {}
    for order in listed.get("orders") or []:
        job_id = str(order["job_id"])
        if order.get("pricing") == PRICING_AUCTION:
            pricing[job_id] = PRICING_AUCTION
        status = fetch(f"{base}/corpus/jobs/{urllib.parse.quote(job_id, safe='')}/status") or {}
        accepted, audited = status.get("submissions_accepted"), status.get("audited")
        if accepted is None or audited is None or int(accepted) <= int(audited):
            continue
        pending.append(PendingWork(job_id=job_id, task_id=job_id, reason="awaiting_audit",
                                   submissions_awaiting=int(accepted) - int(audited),
                                   since=(first_seen or {}).get(job_id), held=status.get("held")))
    return pending, pricing


def _ceiling_alpha(tokens: int, alpha_per_mtoken: float, job: Mapping, board: Mapping) -> float:
    """At most what ``tokens`` pay at the ceiling, never past the job's cap in alpha."""
    alpha = tokens * alpha_per_mtoken / 1e6
    cap, emission = job.get("cap"), board.get("miner_alpha_per_period")
    if cap is not None and emission is not None:
        alpha = min(alpha, float(cap) * float(emission))
    return alpha


# -- the chain ---------------------------------------------------------------------------------------------------


def epoch_pay_from_info(info, block: int, at: float) -> EpochPay:
    """One epoch's miner incentive from a metagraph read at its block.

    A hotkey without dividends (a miner) received its ``emission``. One with
    dividends also validates: its miner part is its incentive at the miners'
    rate of the epoch (their emission over their incentive). Every hotkey of
    the owner's coldkey, and the owner hotkey, counts as the owner: the chain
    keeps what they are given (burned or recycled), so it is ``burned``, never
    pay."""
    hotkeys = [str(h) for h in info.hotkeys]
    coldkeys = [str(c) for c in (info.coldkeys or [None] * len(hotkeys))]
    emission = [float(e) for e in info.emission]
    dividends = [float(d) for d in info.dividends]
    incentives = [float(i) for i in info.incentives]
    owner = {uid for uid, coldkey in enumerate(coldkeys)
             if info.owner_coldkey is not None and coldkey == str(info.owner_coldkey)}
    if info.owner_hotkey in hotkeys:
        owner.add(hotkeys.index(info.owner_hotkey))
    miners = [uid for uid in range(len(hotkeys)) if dividends[uid] == 0 and uid not in owner]
    miner_incentive = sum(incentives[uid] for uid in miners)
    rate = sum(emission[uid] for uid in miners) / miner_incentive if miner_incentive > 0 else None
    pay: dict[str, float] = {}
    for uid, hotkey in enumerate(hotkeys):
        if uid in owner:
            continue
        alpha = emission[uid] if dividends[uid] == 0 else (incentives[uid] * rate if rate is not None else 0.0)
        if alpha > 0:
            pay[hotkey] = pay.get(hotkey, 0.0) + alpha
    burned = sum(incentives[uid] for uid in owner) * rate if rate is not None else 0.0
    pot = burned + sum(pay.values()) if rate is not None else None
    return EpochPay(block=int(block), at=float(at), pay=pay, burned=burned, pot=pot)


async def read_epoch_pays(subtensor, netuid: int, *, start: float, end: float, head: int | None = None,
                          progress=None) -> tuple[list[EpochPay], dict]:
    """Every epoch of ``netuid`` whose block time is in [start, end), newest
    first walked back from ``head`` (the chain head by default), and what the
    walk read of the subnet (tempo, commit-reveal). Needs an archive node for
    blocks past the pruning horizon of a light one. ``progress(epochs, block)``
    is told every 20 epochs read."""
    from bittensor.core.chain_data.metagraph_info import SelectiveMetagraphIndex

    fields = [getattr(SelectiveMetagraphIndex, name) for name in EPOCH_FIELDS]

    async def info_at(block: int):
        return await _retried(lambda: subtensor.get_metagraph_info(netuid, selected_indices=fields, block=block),
                              f"metagraph of {netuid} at block {block}")

    async def time_at(block: int) -> float:
        stamp = await _retried(lambda: subtensor.get_timestamp(block=block), f"time of block {block}")
        return float(stamp.timestamp())

    if head is None:
        head = int(await _retried(subtensor.get_current_block, "the chain head"))
    subnet = await _retried(lambda: subtensor.get_metagraph_info(netuid, block=head),
                            f"metagraph of {netuid} at block {head}")
    meta = {"head": head, "tempo": int(subnet.tempo),
            "commit_reveal": bool(subnet.commit_reveal_weights_enabled),
            "commit_reveal_period": int(subnet.commit_reveal_period or 0)}
    step = int(subnet.last_step)
    current = await info_at(step)
    epochs: list[EpochPay] = []
    while step > 0:
        at = await time_at(step)
        if at < start:
            break
        if at < end:
            epochs.append(epoch_pay_from_info(current, step, at))
            if progress is not None and len(epochs) % 20 == 0:
                progress(len(epochs), step)
        guess = step - int(current.tempo)
        probe = await info_at(guess)
        if int(probe.last_step) != guess:
            # The tempo changed, or an epoch was skipped: ask the block before this epoch.
            guess = int((await info_at(step - 1)).last_step)
            probe = await info_at(guess)
        if guess >= step:
            raise RuntimeError(f"epoch walk did not move back from block {step}")
        step, current = guess, probe
    return sorted(epochs, key=lambda e: e.block), meta


async def _retried(read, what: str, *, attempts: int = CHAIN_READ_ATTEMPTS,
                   backoff: float = CHAIN_READ_BACKOFF_SECONDS):
    """One chain read, retried: a public archive node answers an occasional
    request with an internal error or a timeout that the next one does not."""
    for attempt in range(1, attempts + 1):
        try:
            return await asyncio.wait_for(read(), timeout=CHAIN_READ_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001 - every chain fault is retried the same way
            if attempt == attempts:
                raise RuntimeError(f"chain read failed {attempts} times: {what}: {exc!r}") from exc
            logger.warning("ledger: %s failed (%r); retry %d of %d", what, exc, attempt, attempts - 1)
            await asyncio.sleep(backoff * attempt)


async def chain_alpha_per_period(subtensor, netuid: int) -> float:
    """The miners' alpha per period now (``chain.miner_alpha_per_block`` x the
    blocks in a period), the rate of ``RatePolicy(mode="chain")``."""
    from reliquary.infrastructure.chain import miner_alpha_per_block
    from reliquary.validator.corpus_pay import BLOCK_SECONDS

    read = await miner_alpha_per_block(subtensor, netuid)
    return float(read["miner_alpha_per_block"]) * cp.PERIOD_SECONDS / BLOCK_SECONDS


__all__ = [
    "BucketArchives",
    "DEFAULT_API",
    "DirectoryArchives",
    "RegistryView",
    "api_ceilings",
    "api_pending",
    "chain_alpha_per_period",
    "epoch_pay_from_info",
    "http_json",
    "list_bucket_archives",
    "read_epoch_pays",
    "read_registry_ceilings",
    "registry_ceilings",
    "selected",
]
