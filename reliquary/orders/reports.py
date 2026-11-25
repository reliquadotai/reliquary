"""One report for an evaluation order on several sets.

Each set is its own job and its own grading (``eval.grading``); the order's report puts their environments side by
side and recomputes the macro over all of them. Nothing is re-graded: the combined report names each part's
manifest by its sha256, so it can be checked against the parts it was built from.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence

COMBINED_SCHEMA = "reliquary/eval-report-combined/v1"
MAX_PARTS = 64


class PartNotGraded(ValueError):
    """A part has no final grading (no manifest yet)."""


def _sha256(document: Mapping) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def combine_reports(eval_id: str, parts: Sequence[tuple[str, Mapping, Mapping]], *, created_at: float,
                    chain: Mapping | None = None) -> dict:
    """The combined report of ``(part eval id, its manifest, its report)``, in the order given. An environment
    graded by two parts is kept twice, keyed ``env@part``. A chain's parts are its positions (``orders.chains``):
    ``chain`` is its report, and the macro pass@1 its sub-step rate (every step once, over the benchmark's)."""
    from reliquary.eval.grading import _macro

    if not 1 <= len(parts) <= MAX_PARTS or len({p[0] for p in parts}) != len(parts):
        raise ValueError(f"an order combines 1 to {MAX_PARTS} distinct gradings")
    models = {((r.get("provenance") or {}).get("model"), (r.get("provenance") or {}).get("checkpoint_sha256"))
              for _, _, r in parts}
    if len(models) > 1:
        raise ValueError("the parts grade different models: one report is about one checkpoint")
    envs: dict[str, dict] = {}
    seen: dict[str, int] = {}
    for _, _, report in parts:
        for env in report["envs"]:
            seen[env] = seen.get(env, 0) + 1
    for part_id, _, report in parts:
        for env, scores in report["envs"].items():
            envs[env if seen[env] == 1 else f"{env}@{part_id}"] = {**scores, "eval_id": part_id}
    macro = _macro(envs)
    if chain is not None:
        macro = {**macro, "pass@1": chain["subproblem_pass_rate"], "pass@k": {}, "basis": "chain_steps"}
    return {
        "schema": COMBINED_SCHEMA, "eval_id": eval_id, "created_at": created_at,
        "complete": all(bool(r.get("complete")) for _, _, r in parts) and (chain is None or chain["complete"]),
        "envs": envs, "macro": macro, **({"chain": dict(chain)} if chain is not None else {}),
        "parts": [{"eval_id": part_id, "manifest_sha256": _sha256(manifest), "complete": bool(report.get("complete")),
                   "envs": sorted(report["envs"])} for part_id, manifest, report in parts],
        "provenance": {part_id: report.get("provenance") for part_id, _, report in parts},
    }


async def combine(platform, eval_id: str, part_ids: Sequence[str], *, clock=time.time,
                  chain_id: str | None = None, chain_report=None) -> dict:
    """Write the combined report and its manifest under ``evaluations/{eval_id}/``; a combined manifest already
    there is returned as stored (the same parts and chain) or refused (others). ``chain_report()`` (async) gives
    a chain's report (``orders.chains.chain_report``)."""
    from reliquary.eval.grading import evaluation_prefix

    prefix = evaluation_prefix(eval_id)
    manifest_key = f"{prefix}/manifest.json"
    stored = await platform.get_json(manifest_key)
    if stored is not None:
        if (stored.get("schema") != COMBINED_SCHEMA or stored.get("part_ids") != list(part_ids)
                or stored.get("chain") != chain_id):
            raise ValueError("an evaluation exists under this id with other parts")
        return stored
    parts = []
    for part_id in part_ids:
        manifest = await platform.get_json(f"{evaluation_prefix(part_id)}/manifest.json")
        report = await platform.get_json(f"{evaluation_prefix(part_id)}/report.json")
        if manifest is None or report is None:
            raise PartNotGraded(f"part {part_id!r} is not graded")
        parts.append((part_id, manifest, report))
    chain = None if chain_id is None else await chain_report()
    report = combine_reports(eval_id, parts, created_at=clock(), chain=chain)
    report_key = f"{prefix}/report.json"
    body = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    await platform.put_bytes(report_key, body)
    manifest = {"schema": COMBINED_SCHEMA, "eval_id": eval_id, "created_at": report["created_at"],
                "complete": report["complete"], "part_ids": list(part_ids),
                **({"chain": chain_id} if chain_id is not None else {}),
                "files": [{"name": "report.json", "key": report_key,
                           "sha256": hashlib.sha256(body).hexdigest()}],
                "keys": [report_key, manifest_key]}
    await platform.put_json(manifest_key, manifest)
    return manifest


__all__ = ["COMBINED_SCHEMA", "MAX_PARTS", "PartNotGraded", "combine", "combine_reports"]
