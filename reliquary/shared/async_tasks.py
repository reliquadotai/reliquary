"""Await all children owned by one concurrent operation, including on cancellation."""

from __future__ import annotations

import asyncio


async def gather_owned(awaitables, *, return_exceptions: bool = False) -> list:
    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    group = asyncio.gather(*tasks, return_exceptions=return_exceptions)
    try:
        # Cancellation belongs to this owner: gather must not independently
        # cancel peers and return while their asynchronous finalizers still run.
        return list(await asyncio.shield(group))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if group.done() and not group.cancelled():
            group.exception()
        raise
