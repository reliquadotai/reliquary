"""Mining orders one after another (`reliquary corpus mine` without ``--job-id``).

The validator lists every order it serves with its state (``/corpus/jobs``, ``orders``). This loop picks an
open order, mines it in its own process (`reliquary corpus mine --job-id`, so the GPU is freed between
orders), and when that ends picks the next. While no order is open it fetches the model of one that is
starting, so the miner is ready when its audit card is.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

# How long to wait between two reads of the order list while nothing is open, and after a child that ended
# at once (a refused or failed start), so a broken order is not restarted in a loop.
IDLE_SECONDS = 30.0
QUICK_EXIT_SECONDS = 60.0


def model_of(order: Mapping) -> str | None:
    if not order.get("model") or not order.get("revision"):
        return None
    return f"{order['model']}@{order['revision']}"


def order_precision(order: Mapping) -> str:
    """The precision an order's miners generate in (``corpus.job.PRECISIONS``): bf16 when it names none."""
    return str(order.get("precision") or "bf16")


def choose_order(orders: Sequence[Mapping], *, current: str | None = None,
                 skip: frozenset[str] = frozenset(), precision: str = "bf16") -> tuple[str | None, str | None]:
    """The order to mine now, or None, and a model to fetch ahead (``repo@revision``), or None.

    Open orders first: one on the model already loaded, else the first listed. With none open, the first
    starting order's model is fetched ahead. Only orders in ``precision`` (the one this miner serves)."""
    orders = [o for o in orders if order_precision(o) == precision]
    open_orders = [o for o in orders if o.get("state") == "open" and o.get("job_id") not in skip]
    for order in open_orders:
        if current is not None and model_of(order) == current:
            return str(order["job_id"]), None
    if open_orders:
        return str(open_orders[0]["job_id"]), None
    for order in orders:
        if order.get("state") == "starting" and model_of(order) is not None:
            return None, model_of(order)
    return None, None


def child_command(job_id: str, argv: Sequence[str]) -> list[str]:
    """This CLI, mining ``job_id`` with the options it was given."""
    return [sys.executable, "-m", "reliquary.cli.main", "corpus", "mine", *argv, "--job-id", job_id]


def child_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The child takes its own job's contract: the parent's, if any, is dropped."""
    from reliquary.protocol.profiles import TASK_CONTRACT_ENV_VAR

    return {k: v for k, v in (os.environ if env is None else env).items() if k != TASK_CONTRACT_ENV_VAR}


def follow_orders(*, list_orders: Callable[[], list[dict] | None], mine: Callable[[str], int],
                  fetch: Callable[[str], None], sleep: Callable[[float], None] = time.sleep,
                  clock: Callable[[], float] = time.monotonic, rounds: int | None = None,
                  precision: str = "bf16") -> None:
    """Mine open orders one after another; ``rounds`` bounds the loop (tests)."""
    current: str | None = None
    fetched: set[str] = set()
    # Orders whose child ended at once, skipped while another is open.
    failed: set[str] = set()
    done = 0
    while rounds is None or done < rounds:
        done += 1
        try:
            orders = list_orders()
        except Exception:
            logger.warning("the order list is unreadable; retrying", exc_info=True)
            sleep(IDLE_SECONDS)
            continue
        if orders is None:
            logger.warning("the validator lists no orders (an older validator): pass --job-id")
            return
        job_id, ahead = choose_order(orders, current=current, skip=frozenset(failed), precision=precision)
        if job_id is None:
            if ahead is not None and ahead not in fetched:
                logger.info("no open order; fetching %s for the one starting", ahead)
                try:
                    fetch(ahead)
                    fetched.add(ahead)
                except Exception:
                    logger.warning("fetching %s failed", ahead, exc_info=True)
            failed.clear()
            sleep(IDLE_SECONDS)
            continue
        order = next(o for o in orders if o.get("job_id") == job_id)
        logger.info("mining order %s (%s)", job_id, model_of(order))
        started = clock()
        code = mine(job_id)
        current = model_of(order) or current
        if code != 0 and clock() - started < QUICK_EXIT_SECONDS:
            logger.warning("order %s ended at once (exit %s); trying another", job_id, code)
            failed.add(job_id)
            sleep(IDLE_SECONDS)
        else:
            logger.info("order %s ended (exit %s)", job_id, code)


def run_child(job_id: str, argv: Sequence[str]) -> int:
    return subprocess.call(child_command(job_id, argv), env=child_env())


def fetch_model(model: str) -> None:
    from huggingface_hub import snapshot_download

    repo, revision = model.split("@", 1)
    snapshot_download(repo, revision=revision)
