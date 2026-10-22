"""The held-out benchmark feed of an RL run, for the platform's run monitor.

``build_feed`` turns graded served-job reports (``report.json`` of ``eval grade``, carrying ``rl_step`` /
``rl_run_id``, see ``job_grading.rl_provenance``) into a ``reliquary.run-benchmarks.v1`` document, the contract of
reliquary-platform ``shared/run-benchmarks.ts``; ``parse_feed`` is that file's validator, ported, and every built
feed passes it. One benchmark per eval set (its id is the set id): the baseline is the start checkpoint's report,
the points the run's checkpoints by step. The baseline's ``step`` is null: it is the declared start checkpoint (the
SFT model before RL), not an optimizer step of the run. The score is the report's headline pass@1 (missing and ungraded samples
count as failures), ``samples`` the problems it is over. ``cost_usd`` / ``gpu_hours`` are cumulative TRAINING
resource use in the contract, which a report does not know: null.

The feed lives at ``rl-runs/<run_id>/benchmarks.json`` in the ``reliquary-public`` bucket; the bucket is private,
so the RL validator serves it at ``GET /rl/benchmarks?run_id=<run_id>`` (``RELIQUARY_RUN_BENCHMARKS_URL``). The other
transport is the summary of a dedicated W&B evaluation run (``wandb_feed.publish_summary``,
``WANDB_RL_BENCHMARK_RUN_ID``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import math
import re
from typing import Any

SCHEMA = "reliquary.run-benchmarks.v1"
MAX_BENCHMARKS, MAX_CHECKPOINTS = 24, 120
MAX_FEED_BYTES = 512 * 1024              # the platform reads at most this much
METRICS = ("pass@1", "accuracy", "resolved")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def feed_key(run_id: str) -> str:
    if not RUN_ID.fullmatch(run_id or ""):
        raise ValueError(f"run id {run_id!r}: letters, digits, '.', '_' or '-', at most 128")
    return f"rl-runs/{run_id}/benchmarks.json"


def encode_feed(feed: Mapping[str, Any]) -> bytes:
    body = json.dumps(feed, sort_keys=True, separators=(",", ":")).encode()
    if len(body) > MAX_FEED_BYTES:
        raise ValueError(f"feed is {len(body)} bytes, over the platform's {MAX_FEED_BYTES}: "
                         "lower --max-checkpoints")
    return body


# --- reports -> series -------------------------------------------------------------------------------------------

def _set_entries(report: Mapping[str, Any], source: str) -> list[dict]:
    """One entry per set of a report: its protocol and its point's raw fields."""
    provenance = report.get("provenance") or {}
    sets = provenance.get("sets") or []
    if not sets:
        raise ValueError(f"{source}: no set in the report")
    envs = [s["env"] for s in sets]
    if len(set(envs)) != len(envs):
        raise ValueError(f"{source}: two sets on one environment, their scores are merged")
    sampling = provenance.get("sampling") or {}
    scheme = (provenance.get("verification") or {}).get("scheme") or "unverified"
    out = []
    for s in sets:
        env = report["envs"][s["env"]]
        protocol = {"set_id": s["set_id"], "source": s["source"], "split": s.get("split"),
                    "problems": s["problems"], "samples": s["samples"],
                    "prompts_sha256": s["prompts_sha256"], "grading_sha256": s["grading_sha256"],
                    "sampling": {k: sampling.get(k) for k in ("temperature", "top_p", "top_k")},
                    "max_new_tokens": provenance.get("max_new_tokens"),
                    "thinking": bool(provenance.get("thinking")), "scheme": scheme}
        headline = env.get("pass@1")
        out.append({"protocol": protocol, "score": None if headline is None else headline["value"],
                    "samples": int(env["n_problems"]), "at": int(float(report["created_at"]) * 1000),
                    "model": provenance.get("model"), "revision": provenance.get("revision"),
                    "rl_run_id": provenance.get("rl_run_id"), "rl_step": provenance.get("rl_step"),
                    "source": source})
    return out


def _protocol_block(protocol: Mapping[str, Any]) -> dict:
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    sampling = protocol["sampling"]
    dataset = protocol["source"] + (f" ({protocol['split']})" if protocol.get("split") else "")
    return {
        "id": f"{protocol['set_id'][:140]}@{digest[:12]}",
        "dataset": dataset[:160],
        "revision": f"prompts {protocol['prompts_sha256'][:16]}, grading {protocol['grading_sha256'][:16]}",
        "harness": (f"reliquary served eval, {protocol['scheme']}; temperature {sampling['temperature']}, "
                    f"top_p {sampling['top_p']}, top_k {sampling['top_k']}; "
                    f"thinking {'on' if protocol['thinking'] else 'off'}; {protocol['problems']} problems"),
        "attempts": int(protocol["samples"]),
        "max_output_tokens": int(protocol["max_new_tokens"]),
    }


def _benchmark_id(set_id: str) -> str:
    if len(set_id) <= 96:
        return set_id
    return f"{set_id[:83]}-{hashlib.sha256(set_id.encode()).hexdigest()[:12]}"


def _label(protocol: Mapping[str, Any]) -> str:
    return f"{protocol['source']} ({protocol['problems']} problems, {protocol['samples']} samples)"[:160]


def _point(entry: Mapping[str, Any], *, step: int | None, checkpoint: str, at: int) -> dict:
    return {"checkpoint": checkpoint[:160], "step": step, "at": at, "score": entry["score"],
            "samples": entry["samples"], "cost_usd": None, "gpu_hours": None}


def build_feed(run_id: str, checkpoints: Iterable[tuple[str, Mapping[str, Any]]],
               baselines: Iterable[tuple[str, Mapping[str, Any]]] = (), *, observed_at_ms: int,
               labels: Mapping[str, str] | None = None, max_checkpoints: int = MAX_CHECKPOINTS) -> dict:
    """``checkpoints`` / ``baselines``: ``(source, report)`` pairs (the source names the job in errors). A
    checkpoint report needs ``rl_step``; one naming another ``rl_run_id`` is refused. The newest
    ``max_checkpoints`` points of a set are kept."""
    feed_key(run_id)
    if not 1 <= max_checkpoints <= MAX_CHECKPOINTS:
        raise ValueError(f"max_checkpoints in [1, {MAX_CHECKPOINTS}]")
    labels = dict(labels or {})
    series: dict[str, dict] = {}

    def slot(entry):
        set_id = entry["protocol"]["set_id"]
        found = series.setdefault(set_id, {"protocol": entry["protocol"], "baseline": None, "points": {}})
        if found["protocol"] != entry["protocol"]:
            raise ValueError(f"{entry['source']}: set {set_id} graded under another protocol "
                             "(problems, samples, sampling, budget, thinking or grader differ)")
        return found

    for source, report in checkpoints:
        for entry in _set_entries(report, source):
            if entry["rl_run_id"] is not None and entry["rl_run_id"] != run_id:
                raise ValueError(f"{source}: a checkpoint of run {entry['rl_run_id']}, not {run_id}")
            step = entry["rl_step"]
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ValueError(f"{source}: no rl_step in the report (eval grade --rl-step)")
            points = slot(entry)["points"]
            if step in points:
                raise ValueError(f"{source}: step {step} of set {entry['protocol']['set_id']} "
                                 f"is graded twice ({points[step]['source']})")
            points[step] = entry
    for source, report in baselines:
        for entry in _set_entries(report, source):
            found = slot(entry)
            if found["baseline"] is not None:
                raise ValueError(f"{source}: two baselines for set {entry['protocol']['set_id']}")
            found["baseline"] = entry
    if len(series) > MAX_BENCHMARKS:
        raise ValueError(f"{len(series)} sets, the platform shows at most {MAX_BENCHMARKS}")
    benchmarks = []
    for set_id in sorted(series):
        found = series[set_id]
        steps = sorted(found["points"])[-max_checkpoints:]
        points, at = [], 0
        for step in steps:
            entry = found["points"][step]
            # Graded out of step order (a backlog): `at` must not go back, nor pass observed_at.
            at = min(max(at, entry["at"]), observed_at_ms)
            points.append(_point(entry, step=step, checkpoint=entry["model"] or f"step-{step}", at=at))
        base = found["baseline"]
        baseline = None if base is None else _point(
            base, step=None,
            checkpoint=f"{base['model']}@{(base['revision'] or '')[:12]}", at=min(base["at"], observed_at_ms))
        benchmarks.append({"id": _benchmark_id(set_id), "label": labels.get(set_id) or _label(found["protocol"]),
                           "metric": "pass@1", "protocol": _protocol_block(found["protocol"]),
                           "baseline": baseline, "checkpoints": points})
    feed = {"schema": SCHEMA, "run_id": run_id, "observed_at": int(observed_at_ms), "scope": "held_out",
            "benchmarks": benchmarks}
    parse_feed(feed, run_id, observed_at_ms)
    return feed


# --- the platform's validator (shared/run-benchmarks.ts, parseRunBenchmarks) ------------------------------------

class FeedRejected(ValueError):
    """The platform would drop the whole feed."""


def _record(v):
    if not isinstance(v, dict):
        raise FeedRejected("Invalid benchmark record")
    return v


def _text(v, limit=160):
    if not isinstance(v, str) or not v.strip() or len(v) > limit or re.search(r"[\x00-\x1f]", v):
        raise FeedRejected("Invalid benchmark label")
    return v


def _number(v, limit=1e16):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > limit:
        raise FeedRejected("Invalid benchmark number")
    return v


def _integer(v, limit=1e12):
    n = _number(v, limit)
    if n != int(n) or abs(n) > 2 ** 53 - 1:
        raise FeedRejected("Invalid benchmark count")
    return n


def parse_feed(raw: Any, run_id: str, now_ms: float) -> dict:
    """``parseRunBenchmarks``: raises ``FeedRejected`` where the TypeScript throws. JS ``length`` counts UTF-16
    units; our strings are ASCII-bounded by construction, so ``len`` agrees."""
    v = _record(raw)
    if (v.get("schema") != SCHEMA or v.get("run_id") != run_id or v.get("scope") != "held_out"
            or not isinstance(v.get("benchmarks"), list) or len(v["benchmarks"]) > MAX_BENCHMARKS):
        raise FeedRejected("Invalid benchmark feed")
    at = _number(v.get("observed_at"), now_ms + 5000)

    def point(raw_point, *, baseline=False):
        p = _record(raw_point)
        # A baseline may say `"step": null`; a missing step is refused all the same.
        step = None if baseline and "step" in p and p["step"] is None else _integer(p.get("step"))
        return {"checkpoint": _text(p.get("checkpoint")), "step": step,
                "at": _number(p.get("at"), at),
                "score": None if "score" in p and p["score"] is None else _number(p.get("score"), 1),
                "samples": _integer(p.get("samples")),
                "cost_usd": None if p.get("cost_usd") is None else _number(p["cost_usd"]),
                "gpu_hours": None if p.get("gpu_hours") is None else _number(p["gpu_hours"])}

    benchmarks = []
    for raw_benchmark in v["benchmarks"]:
        b = _record(raw_benchmark)
        protocol = _record(b.get("protocol"))
        if (str(b.get("metric")) not in METRICS or not isinstance(b.get("checkpoints"), list)
                or len(b["checkpoints"]) > MAX_CHECKPOINTS):
            raise FeedRejected("Invalid benchmark series")
        checkpoints = [point(p) for p in b["checkpoints"]]
        if (any(i > 0 and (p["step"] <= checkpoints[i - 1]["step"] or p["at"] < checkpoints[i - 1]["at"])
                for i, p in enumerate(checkpoints))
                or len({p["checkpoint"] for p in checkpoints}) != len(checkpoints)):
            raise FeedRejected("Unordered benchmark checkpoints")
        baseline = None if b.get("baseline") is None else point(b["baseline"], baseline=True)
        if any(p["score"] is not None and p["samples"] < 1 for p in checkpoints + ([baseline] if baseline else [])):
            raise FeedRejected("Unmeasured benchmark score")
        benchmarks.append({
            "id": _text(b.get("id"), 96), "label": _text(b.get("label")), "metric": b["metric"],
            "protocol": {"id": _text(protocol.get("id")), "dataset": _text(protocol.get("dataset")),
                         "revision": _text(protocol.get("revision")), "harness": _text(protocol.get("harness")),
                         "attempts": _integer(protocol.get("attempts"), 1024),
                         "max_output_tokens": _integer(protocol.get("max_output_tokens"), 10_000_000)},
            "baseline": baseline, "checkpoints": checkpoints})
    if (len({b["id"] for b in benchmarks}) != len(benchmarks)
            or any(b["protocol"]["attempts"] < 1 or b["protocol"]["max_output_tokens"] < 1 for b in benchmarks)):
        raise FeedRejected("Invalid benchmark protocol")
    return {"schema": SCHEMA, "run_id": run_id, "scope": "held_out", "observed_at": at, "benchmarks": benchmarks}


# --- where reports and the feed live ----------------------------------------------------------------------------

class LocalReports:
    """``<root>/<job>/report.json``, as ``eval grade --out <root>/<job>`` writes them."""

    def __init__(self, root) -> None:
        from pathlib import Path

        self.root = Path(root)

    async def jobs(self, prefix: str) -> list[str]:
        return sorted(p.parent.name for p in self.root.glob("*/report.json") if p.parent.name.startswith(prefix))

    async def report(self, job_id: str) -> dict | None:
        path = self.root / job_id / "report.json"
        return json.loads(path.read_text()) if path.exists() else None


class BucketReports:
    """``evaluations/<job>/report.json`` in the subnet bucket (``eval grade --upload``); jobs from their manifests."""

    def __init__(self, store=None, list_jobs=None) -> None:
        self._store, self._list = store, list_jobs

    async def jobs(self, prefix: str) -> list[str]:
        if self._list is None:
            from reliquary.infrastructure.corpus_job_store import list_jobs

            self._list = list_jobs
        return sorted(j for j in await self._list() if j.startswith(prefix))

    async def report(self, job_id: str) -> dict | None:
        if self._store is None:
            from reliquary.eval.storage import SubnetEvalStore

            self._store = SubnetEvalStore()
        from reliquary.eval.grading import evaluation_prefix

        body = await self._store.get_bytes(f"{evaluation_prefix(job_id)}/report.json")
        return None if body is None else json.loads(body)


async def collect(reports, *, jobs: Iterable[str] = (), prefix: str | None = None,
                  baseline_jobs: Iterable[str] = ()) -> tuple[list, list, list[str]]:
    """``(checkpoints, baselines, missing)``: graded reports of the named jobs (and of every job under
    ``prefix``); a job not graded yet is listed in ``missing`` and left out."""
    names = list(dict.fromkeys(jobs))
    baseline_jobs = list(dict.fromkeys(baseline_jobs))
    if prefix:
        names += [j for j in await reports.jobs(prefix) if j not in names]
    names = [j for j in names if j not in baseline_jobs]
    found, missing = {}, []
    for job in names + baseline_jobs:
        report = await reports.report(job)
        if report is None:
            missing.append(job)
        else:
            found[job] = report
    return ([(j, found[j]) for j in names if j in found],
            [(j, found[j]) for j in baseline_jobs if j in found], missing)


def publish_feed(feed: Mapping[str, Any], *, store=None, base_url: str | None = None) -> dict:
    """Upload ``feed`` to the public bucket (``R2_PUBLIC_*``) at ``feed_key``; ``store`` has ``put_bytes``."""
    body = encode_feed(feed)
    key = feed_key(feed["run_id"])
    if store is None:
        from reliquary.infrastructure.storage import public_checkpoint_bucket, public_checkpoint_client
        from reliquary.trainer.ckpt_publisher import R2ObjectStore

        store = R2ObjectStore(public_checkpoint_client(), public_checkpoint_bucket())
    store.put_bytes(key, body)
    return {"key": key, "bytes": len(body),
            "url": f"{base_url.rstrip('/')}/{key}" if base_url else None}


__all__ = ["BucketReports", "FeedRejected", "LocalReports", "MAX_BENCHMARKS", "MAX_CHECKPOINTS", "SCHEMA",
           "build_feed", "collect", "encode_feed", "feed_key", "parse_feed", "publish_feed"]
