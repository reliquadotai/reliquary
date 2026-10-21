"""Reading the async RL validator's published scan log (``validator/rl_observations``; next-rl-run-decisions.md §D):
``GET /rl/observations`` gives the head signed by the validator hotkey and presigned URLs into the private bucket;
every page and segment is checked against it with ``services/publication``'s verifiers. Miner side (the miner does
not import the validator)."""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from reliquary.services import publication as pub


async def read_scan_log(ask: Callable[[int], Awaitable[Mapping[str, Any] | None]],
                        fetch: Callable[[str], Awaitable[bytes]], validator_hotkey: str, *,
                        after: int = 0) -> tuple[list[dict], int]:
    """Every scan published after segment ``after``, verified: ``ask(after)`` is ``GET /rl/observations``,
    ``fetch(url)`` downloads a presigned URL. Returns (events in seq order, the last segment number read).
    ValueError when anything fails to verify."""
    from reliquary.shared.ckpt_index import observations_root

    events: list[dict] = []
    pages: dict[str, dict] = {}
    while True:
        answer = await ask(after)
        if not answer:
            return events, after
        head = answer["index"]
        run = str(head.get("run_id", ""))
        root = observations_root(run)
        if answer.get("root") != root:
            raise ValueError("the observation log is not at its run's root")
        head = pub.verify_index(head, validator_hotkey, expected_run_id=run, root=root)
        if head["kind"] != "head":
            raise ValueError("not a head index")
        if head["last_number"] <= after:
            return events, after
        urls = answer["urls"]
        entries = {e["number"]: e for e in head["segments"]}
        for page in head["pages"]:
            if page["last_number"] > after and page["first_number"] not in pages:
                raw = await fetch(urls[page["key"]])
                pub.verify_page_entry(raw, page)
                body = pub.verify_index(raw, validator_hotkey, expected_run_id=run, root=root)
                pages[page["first_number"]] = body
            if page["first_number"] in pages:
                entries.update({e["number"]: e for e in pages[page["first_number"]]["segments"]})
        progressed = False
        for number in range(after + 1, head["last_number"] + 1):
            entry = entries.get(number)
            if entry is None or entry["key"] not in urls:
                break
            events.extend(pub.verify_segment(await fetch(urls[entry["key"]]), entry))
            after, progressed = number, True
        if not progressed:
            raise ValueError("the observation answer lists no segment after the last one read")


def scans_as_page(events: list[dict]) -> dict:
    """A ``/rl/prompts/state``-shaped page of published scans (no ``latest``): the scans, each prompt's state after
    its last scan, the last seq as cursor."""
    scans = [{"seq": e["id"], "environment": e["env"], "prompt_idx": e["prompt_idx"], "checkpoint": e["checkpoint"],
              "rewards": e["rewards"], "verdict": e["verdict"], "first_scan": e["first_scan"], "at": e["ts"]}
             for e in events]
    states = {(e["env"], e["prompt_idx"]): {"environment": e["env"], "prompt_idx": e["prompt_idx"],
                                            "state": e.get("state"), "until_k": e.get("until_k")} for e in events}
    return {"cursor": events[-1]["id"] if events else 0, "more": False, "scans": scans,
            "prompts": list(states.values())}


__all__ = ["read_scan_log", "scans_as_page"]
