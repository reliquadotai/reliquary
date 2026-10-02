"""The operator's evaluations: qualify, create, grade and compare from the CLI.

Everything here is a client of the signed admin service: there is no second
path writing jobs or qualifications, so every check the admin makes on the
platform's orders applies to the operator's too.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_ID = 63
GRADED_FILES = ("report.json", "manifest.json", "graded.parquet")


class AdminError(RuntimeError):
    def __init__(self, method: str, path: str, status: int, detail: Any) -> None:
        super().__init__(f"{method} {path}: {status} {detail}")
        self.status, self.detail = status, detail


class AdminClient:
    """Signed requests to the admin service (``RELIQUARY_ADMIN_SECRET``)."""

    def __init__(self, base_url: str, secret: bytes, *, http: httpx.Client | None = None,
                 timeout: float = 120.0) -> None:
        self._secret = secret
        self._http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def request(self, method: str, path: str, body: Any = None) -> httpx.Response:
        from reliquary.admin.auth import (
            NONCE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, sign_request,
        )

        data = b"" if body is None else json.dumps(body).encode()
        stamp, nonce = str(int(time.time())), secrets.token_hex(16)
        headers = {TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce,
                   SIGNATURE_HEADER: sign_request(self._secret, stamp, nonce, method, path, data),
                   "content-type": "application/json"}
        return self._http.request(method, path, content=data, headers=headers)

    def json(self, method: str, path: str, body: Any = None,
             ok: tuple[int, ...] = (200, 201, 202)) -> dict:
        response = self.request(method, path, body)
        if response.status_code not in ok:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = response.text[:300]
            raise AdminError(method, path, response.status_code, detail)
        return response.json()


def split_model(spec: str) -> tuple[str, str]:
    """``repo@revision`` with a full 40-hex commit, the only revision the admin takes."""
    repo, sep, revision = spec.rpartition("@")
    if not sep or not repo or not _REVISION_RE.match(revision):
        raise ValueError(f"--model must be repo@<40-hex commit>, got {spec!r}")
    return repo, revision


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def qualification_id_for(prefix: str, conditions: Mapping[str, Any]) -> str:
    """The same conditions always name the same qualification, so asking twice
    finds the first one instead of renting executors again."""
    return f"{prefix}q-{_digest(dict(conditions))[:40]}"


def default_job_id(prefix: str, revision: str, set_id: str, count: int, samples: int,
                   conditions: Mapping[str, Any]) -> str:
    """Readable where it can be (revision, set, size), unique where it must be:
    the tail hashes every condition, so two orders differing in sampling,
    budget, thinking or repo never meet on one id. Job ids are [a-z0-9-]."""
    slug = re.sub(r"[^a-z0-9-]+", "-", set_id.lower()).strip("-")
    tag = _digest({**dict(conditions), "samples": samples})[:6]
    job_id = f"{prefix}eval-{revision[:8]}-{slug}-n{count}x{samples}-{tag}"
    if len(job_id) <= MAX_ID:
        return job_id
    tail = _digest([job_id])[:10]
    return f"{job_id[:MAX_ID - 11].rstrip('-')}-{tail}"


def checked_job_id(job_id: str) -> str:
    from reliquary.corpus.job import JOB_ID_RE

    if not JOB_ID_RE.match(job_id):
        raise ValueError(f"job id {job_id!r} is not [a-z0-9-], at most 63 characters")
    return job_id


def read_set_card(set_id: str, store=None) -> dict:
    """A published set's card, from the subnet bucket the admin reads it from."""
    import asyncio

    from reliquary.eval.storage import SubnetEvalStore, subnet_key

    store = store or SubnetEvalStore()
    body = asyncio.run(store.get_bytes(subnet_key(set_id, "set.json")))
    if body is None:
        raise ValueError(f"set {set_id!r} is not published (reliquary eval publish-set)")
    return json.loads(body)


def _wait(client: AdminClient, qualification_ids: list[str], *, poll_seconds: float,
          timeout_seconds: float, sleep: Callable[[float], None],
          log: Callable[[str], None], clock: Callable[[], float]) -> dict[str, dict]:
    from reliquary.eval.qualification import TERMINAL

    deadline = clock() + timeout_seconds
    done: dict[str, dict] = {}
    last: dict[str, str] = {}
    while True:
        for qid in qualification_ids:
            if qid in done:
                continue
            record = client.json("GET", f"/admin/v1/qualifications/{qid}")
            status = record.get("status")
            if last.get(qid) != status:
                log(f"qualification {qid}: {status}")
                last[qid] = status
            if status in TERMINAL:
                done[qid] = record
        if len(done) == len(qualification_ids):
            return done
        if clock() > deadline:
            raise TimeoutError(f"qualifications still running after {timeout_seconds:.0f} s: "
                               f"{sorted(set(qualification_ids) - set(done))}")
        sleep(poll_seconds)


def create_evaluations(client: AdminClient, *, cards: list[dict], model: str, revision: str,
                       samples: int, max_new_tokens: int, thinking: bool,
                       sampling: Mapping[str, Any], count: int | None = None,
                       cap: float | None = None, seed: int | None = None,
                       job_id: str | None = None, completions: int = 32,
                       prefix: str = "order-", poll_seconds: float = 30.0,
                       timeout_seconds: float = 6 * 3600.0, attempt: int = 0,
                       sleep: Callable[[float], None] = time.sleep,
                       log: Callable[[str], None] = print,
                       clock: Callable[[], float] = time.monotonic) -> list[dict]:
    """One qualification and one eval job per set, all on ``model@revision``.

    Qualifications are requested together, then waited for; a set whose model
    is refused or fails qualification gets no job, and the error names why.
    Every id is derived and checked before anything is requested, so a rerun
    finds what the first run made and a bad id never costs a qualification.
    A failed qualification stays failed under its id: ``attempt`` asks again."""
    if job_id is not None and len(cards) != 1:
        raise ValueError("--job-id names one job: give one set")
    if completions * max_new_tokens > 64 * 32768:
        completions = max(1, (64 * 32768) // max_new_tokens)
    completions = min(completions, 64)
    plans = []
    for card in cards:
        problems = int(card["count"]) if count is None else int(count)
        if not 1 <= problems <= int(card["count"]):
            raise ValueError(f"set {card['set_id']} holds {card['count']} problems, not {problems}")
        conditions = {"model": model, "revision": revision, "set_id": card["set_id"],
                      "problems": problems, "sampling": dict(sampling),
                      "max_new_tokens": max_new_tokens, "thinking": thinking}
        qid = qualification_id_for(prefix, {**conditions, "completions": completions,
                                            "attempt": attempt})
        name = checked_job_id(job_id or default_job_id(prefix, revision, card["set_id"],
                                                       problems, samples, conditions))
        plans.append((card, problems, qid, conditions, name))
    for card, _, qid, conditions, _ in plans:
        client.json("POST", "/admin/v1/qualifications", {
            "qualification_id": qid, **conditions, "completions": completions})
        log(f"qualification {qid} requested for {card['set_id']}")
    records = _wait(client, [plan[2] for plan in plans], poll_seconds=poll_seconds,
                    timeout_seconds=timeout_seconds, sleep=sleep, log=log, clock=clock)
    created = []
    for card, problems, qid, _, name in plans:
        record = records[qid]
        if record.get("status") != "qualified":
            result = record.get("result") or {}
            again = (f"; to ask again, rerun with --attempt {attempt + 1}"
                     if record.get("status") == "failed" else "")
            raise RuntimeError(f"{model}@{revision[:8]} on {card['set_id']}: qualification "
                               f"{record.get('status')}: {json.dumps(result)[:300]}{again}")
        body = {"job_id": name,
                "model": model, "env": card["source"], "prompt_count": problems,
                "samples_per_prompt": samples, "max_new_tokens": max_new_tokens,
                "thinking": thinking, "sampling": dict(sampling),
                "eval_set_id": card["set_id"], "qualification_id": qid}
        if cap is not None:
            body["cap"] = cap
        if seed is not None:
            body["seed"] = seed
        answer = client.json("POST", "/admin/v1/jobs", body)
        log(f"job {body['job_id']} on {card['set_id']}: {problems} problems x {samples}")
        created.append({"job_id": body["job_id"], "set_id": card["set_id"],
                        "qualification_id": qid, "answer": answer})
    return created


def grade_job(client: AdminClient, job_id: str, *, out: str | Path | None = None,
              eval_id: str | None = None, allow_incomplete: bool = False,
              poll_seconds: float = 10.0, timeout_seconds: float = 6 * 3600.0,
              sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.monotonic) -> dict:
    """Grade a drained eval job and, with ``out``, write its three files there."""
    from reliquary.eval.prompt_source import parse_eval_source

    status = client.json("GET", f"/admin/v1/jobs/{job_id}/status")
    manifest = status["manifest"]
    source = parse_eval_source(manifest["prompt_source"])
    samples = int(manifest["slots_per_prompt"]) * int(manifest["sampling"]["n"])
    eval_id = eval_id or job_id
    body = {"source": "job", "job_id": job_id, "set_ids": [source.set_id],
            "problems_per_set": {source.set_id: source.count},
            "samples_per_set": {source.set_id: samples}, "allow_incomplete": allow_incomplete}
    deadline = clock() + timeout_seconds
    while True:
        response = client.request("POST", f"/admin/v1/evaluations/{eval_id}/grade", body)
        if response.status_code == 200:
            answer = response.json()
            break
        if response.status_code != 202:
            raise AdminError("POST", f"/admin/v1/evaluations/{eval_id}/grade",
                             response.status_code, response.json().get("detail"))
        if clock() > deadline:
            raise TimeoutError(f"grading {eval_id} still running after {timeout_seconds:.0f} s")
        sleep(poll_seconds)
    if out is not None:
        directory = Path(out)
        directory.mkdir(parents=True, exist_ok=True)
        for name in GRADED_FILES:
            response = client.request("GET", f"/admin/v1/evaluations/{eval_id}/files/{name}")
            if response.status_code != 200:
                raise AdminError("GET", name, response.status_code, response.text[:300])
            (directory / name).write_bytes(response.content)
    return answer


CONDITIONS = ("model", "sampling", "max_new_tokens", "thinking")


def _load(directory: str | Path) -> tuple[dict, dict[str, dict[str, list[bool | None]]]]:
    import pyarrow.parquet as pq

    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    by_env: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in pq.read_table(directory / "graded.parquet").to_pylist():
        by_env[row["env"]][row["problem_id"]].append(row["correct"])
    return report, by_env


def _sets(report: dict) -> list[tuple]:
    return sorted((s["set_id"], s["problems"], s["samples"])
                  for s in report["provenance"]["sets"])


def _grader_versions(report: dict) -> dict:
    return {s["set_id"]: {k: (s.get("taskset_at_grading") or {}).get(k)
                          for k in ("package_version", "verifiers_version")}
            for s in report["provenance"]["sets"] if s.get("source_kind") == "verifiers"}


def compare_reports(a: str | Path, b: str | Path, *, seed: int = 0,
                    allow_ungraded: bool = False) -> dict:
    """Two gradings of the same sets under the same conditions, side by side:
    pass@1 of each and their difference with a paired bootstrap interval over
    the problems (missing and ungraded samples count as failures, as in each
    report's headline). Comparing two checkpoints is the point, so the model is
    the one thing allowed to differ."""
    from reliquary.eval.metrics import bootstrap_mean_ci

    report_a, rows_a = _load(a)
    report_b, rows_b = _load(b)
    prov_a, prov_b = report_a["provenance"], report_b["provenance"]
    differs = [k for k in CONDITIONS if k != "model" and prov_a.get(k) != prov_b.get(k)]
    if _sets(report_a) != _sets(report_b):
        differs.append("sets")
    if _grader_versions(report_a) != _grader_versions(report_b):
        # Another reward may score the same answer differently.
        differs.append("grader versions")
    if differs:
        raise ValueError(f"the two gradings differ in {differs}; compare like with like")
    ungraded = {side: {env: r.get("ungraded_rows", 0) for env, r in report["envs"].items()
                       if r.get("ungraded_rows")}
                for side, report in (("a", report_a), ("b", report_b))}
    if not allow_ungraded and (ungraded["a"] or ungraded["b"]):
        # An ungraded row counts as a failure in pass@1: a grading that could
        # not score (no Docker, drift, a dead scorer) would read as a regression.
        raise ValueError(f"ungraded rows {ungraded}: regrade them, or pass "
                         "--allow-ungraded to count them as failures")
    samples = {s["set_id"]: int(s["samples"]) for s in prov_a["sets"]}
    envs = {}
    for env in sorted(set(report_a["envs"]) | set(report_b["envs"])):
        set_of = {s["env"]: s["set_id"] for s in prov_a["sets"]}
        n = samples.get(set_of.get(env), 1)
        problems = sorted(set(rows_a.get(env, {})) | set(rows_b.get(env, {})))

        def rate(rows, problem):
            return sum(1 for c in rows.get(env, {}).get(problem, ()) if c) / n

        pairs = [(rate(rows_a, p), rate(rows_b, p)) for p in problems]
        diffs = [y - x for x, y in pairs]
        # A problem neither side has a row for failed on both: a zero difference
        # that still counts, as it does in each report's pass@1.
        ordered = report_a["envs"].get(env, {}).get("n_problems") or len(diffs)
        diffs += [0.0] * max(0, ordered - len(diffs))
        low, high = bootstrap_mean_ci(diffs, seed=seed) if diffs else (None, None)
        envs[env] = {
            "a": report_a["envs"][env]["pass@1"]["value"] if env in report_a["envs"] else None,
            "b": report_b["envs"][env]["pass@1"]["value"] if env in report_b["envs"] else None,
            "diff": sum(diffs) / len(diffs) if diffs else None,
            "ci95": [low, high], "problems_with_rows": len(problems),
            "n_problems": report_a["envs"].get(env, {}).get("n_problems"),
            **{f"{key}_{side}": report["envs"].get(env, {}).get(key)
               for side, report in (("a", report_a), ("b", report_b))
               for key in ("missing_rows", "ungraded_rows")},
        }
    return {"a": {"model": prov_a.get("model"), "revision": prov_a.get("revision"),
                  "eval_id": report_a.get("eval_id")},
            "b": {"model": prov_b.get("model"), "revision": prov_b.get("revision"),
                  "eval_id": report_b.get("eval_id")},
            "conditions": {k: prov_a.get(k) for k in CONDITIONS if k != "model"},
            "bootstrap": {"seed": seed, "unit": "problem", "paired": True},
            "envs": envs}


__all__ = [
    "AdminClient",
    "AdminError",
    "compare_reports",
    "create_evaluations",
    "default_job_id",
    "grade_job",
    "qualification_id_for",
    "read_set_card",
    "split_model",
]
