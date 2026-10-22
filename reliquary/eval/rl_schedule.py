"""Held-out evaluations of an RL run's milestone checkpoints, one at a time on one GPU (``eval schedule``).

Each ``tick`` (idempotent, its state in one JSON file):

1. A checkpoint being served: grade each of its jobs once drained (``grade_served_job`` with the RL identity), the
   ones missing samples after ``max_wait_s`` counted as failures (``--allow-incomplete``); once all are graded,
   stop its validator and rebuild the feed (``run_feed``), published when asked. A validator found dead is
   restarted.
2. Nothing served: the start checkpoint first (the baseline), then the NEWEST milestone backup
   (``<namespace>/<prefix>-step-<n>``, ``n`` a multiple of ``every``) past the last one taken. The ones in between
   are marked skipped: when evaluations fall behind they jump to the latest milestone, and training never waits
   (the trainer does not know this exists). Its jobs are declared (``jobs create``, ids
   ``eval-<run>-s<step:05d>-<short>``) and one validator process is started for that model; the suite's later jobs
   join it as hot jobs (one model per process).

Dry run by default: the reads happen (backups, job drains), the writes are printed, the state is not saved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
import shlex
import time
from typing import Any

logger = logging.getLogger(__name__)

BASE = "base"
_SHORT = re.compile(r"[a-z0-9][a-z0-9-]{0,15}")
DEFAULT_START = ("reliquary tasks contract --task-id {job} > {contract} && "
                 "RELIQUARY_TASK_ID={job} RELIQUARY_TASK_CONTRACT={contract} RELIQUARY_CORPUS_HOT_JOBS=1 "
                 "exec reliquary validate --wallet-name {wallet} --hotkey {hotkey} "
                 "--http-host 0.0.0.0 --http-port {port} --no-set-weights")


@dataclass(frozen=True)
class SuiteEntry:
    set_id: str
    short: str
    slots_per_prompt: int
    cap: float
    max_new_tokens: int
    prompt_count: int | None = None


def load_suite(path: str | Path) -> list[SuiteEntry]:
    """A JSON list of ``{set_id, short, slots_per_prompt, cap, max_new_tokens[, prompt_count]}``."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("the suite is a non-empty JSON list")
    suite = []
    for item in raw:
        unknown = set(item) - {"set_id", "short", "slots_per_prompt", "cap", "max_new_tokens", "prompt_count"}
        if unknown:
            raise ValueError(f"suite entry {item.get('set_id')!r}: unknown keys {sorted(unknown)}")
        entry = SuiteEntry(**item)
        if not _SHORT.fullmatch(entry.short):
            raise ValueError(f"short {entry.short!r}: lowercase letters, digits and '-', at most 16")
        suite.append(entry)
    if len({e.short for e in suite}) != len(suite) or len({e.set_id for e in suite}) != len(suite):
        raise ValueError("suite set ids and shorts must be unique")
    return suite


def job_id(run_tag: str, step: int | None, short: str) -> str:
    from reliquary.corpus.job import JOB_ID_RE

    name = f"eval-{run_tag}-{'base' if step is None else f's{step:05d}'}-{short}"
    if not JOB_ID_RE.match(name):
        raise ValueError(f"job id {name!r} is not a job name (lowercase, '-', at most 63)")
    return name


@dataclass
class Settings:
    run_id: str                         # the trainer's wandb run id (the feed's run_id)
    run_tag: str                        # short lowercase tag in the job ids
    suite: Sequence[SuiteEntry]
    every: int                          # a checkpoint every N steps (a multiple of the milestone cadence)
    hf_namespace: str
    hf_prefix: str
    model_architecture: str
    eos_token_id: int
    results_dir: Path
    state_path: Path
    renderer_id: str = "chat-template-thinking-v1"
    temperature: float = 0.6
    top_p: float = 0.95
    baseline_model: str | None = None
    baseline_revision: str | None = None
    start_cmd: str = DEFAULT_START
    start_values: Mapping[str, str] = field(default_factory=dict)       # wallet, hotkey, port
    max_wait_s: float = 6 * 3600
    publish: bool = False
    upload: bool = False
    wandb: bool = False                 # mirror the feed's scores into wandb (wandb_feed)

    def repo(self, step: int) -> str:
        return f"{self.hf_namespace}/{self.hf_prefix}-step-{step:08d}"


@dataclass
class Ports:
    """What the scheduler reads and does; ``reliquary.eval.rl_schedule.live_ports`` are the real ones."""
    backups: Callable[[], list[int]]                         # backup steps of the prefix
    revision: Callable[[str], str | None]                    # repo -> commit, None while incomplete
    fingerprint: Callable[[str, str], str]                   # (repo, revision) -> checkpoint sha256
    job_exists: Callable[[str], bool]
    drained: Callable[[str], bool]
    run: Callable[[list[str]], None]                         # a CLI command
    grade: Callable[..., dict]                               # (job, out=, rl=, allow_incomplete=)
    start: Callable[[str, Path], int]                        # (shell command, log) -> pid
    alive: Callable[[int], bool]
    stop: Callable[[int], None]
    feed: Callable[[list[str], list[str]], dict]             # (checkpoint jobs, baseline jobs) -> summary


class Scheduler:
    def __init__(self, settings: Settings, ports: Ports, *, clock: Callable[[], float] = time.time) -> None:
        self.s, self.p, self.clock = settings, ports, clock

    # --- state ---------------------------------------------------------------------------------------------------
    def load(self) -> dict:
        path = self.s.state_path
        state = json.loads(path.read_text()) if path.exists() else {
            "run_id": self.s.run_id, "checkpoints": {}, "active": None, "pid": None}
        if state["run_id"] != self.s.run_id:
            raise ValueError(f"{path} is the state of run {state['run_id']}, not {self.s.run_id}")
        return state

    def save(self, state: dict) -> None:
        self.s.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.s.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
        tmp.replace(self.s.state_path)

    # --- one pass ------------------------------------------------------------------------------------------------
    def tick(self, *, execute: bool = False) -> list[str]:
        """What it did (or, dry, would do), one line each."""
        state, log = self.load(), []
        if state["active"] is not None:
            if not self._serve(state, log, execute):
                if execute:
                    self.save(state)
                return log
        nxt = self._next(state, log)
        if nxt is not None:
            self._begin(state, nxt, log, execute)
        if execute:
            self.save(state)
        return log

    def _jobs(self, key: str) -> list[str]:
        step = None if key == BASE else int(key)
        return [job_id(self.s.run_tag, step, e.short) for e in self.s.suite]

    def _serve(self, state: dict, log: list[str], execute: bool) -> bool:
        """True once the active checkpoint is done (graded or abandoned) and its validator stopped."""
        key = state["active"]
        cp = state["checkpoints"][key]
        overdue = self.clock() - cp["started_at"] > self.s.max_wait_s
        from reliquary.eval.job_grading import JobNotGradable

        for job in cp["jobs"]:
            if job in cp["graded"] or job in cp["abandoned"]:
                continue
            if not self.p.drained(job):
                if overdue:
                    log.append(f"abandon {job}: not drained after {self.s.max_wait_s:.0f}s")
                    cp["abandoned"].append(job)
                else:
                    log.append(f"wait {job}: not drained")
                continue
            rl = {"run_id": self.s.run_id, "step": 0 if key == BASE else int(key)}
            if not execute:
                log.append(f"grade {job} -> {self.s.results_dir / job} (rl {rl}"
                           f"{', missing samples as failures' if overdue else ''})")
                continue
            try:
                self.p.grade(job, out=self.s.results_dir / job, rl=rl, allow_incomplete=overdue)
            except JobNotGradable as exc:
                log.append(f"wait {job}: {exc}")
                continue
            cp["graded"].append(job)
            log.append(f"graded {job}")
        if len(cp["graded"]) + len(cp["abandoned"]) < len(cp["jobs"]):
            pid = state["pid"]
            if pid is not None and not self.p.alive(pid):
                log.append(f"validator {pid} for {cp['repo']} is gone: restart")
                if execute:
                    state["pid"] = self._start(cp)
            return False
        if state["pid"] is not None:
            log.append(f"stop validator {state['pid']} ({cp['repo']})")
            if execute:
                self.p.stop(state["pid"])
                state["pid"] = None
        cp["status"] = "graded"
        state["active"] = None
        self._feed(state, log, execute)
        return True

    def _feed(self, state: dict, log: list[str], execute: bool) -> None:
        cps = state["checkpoints"]
        base = [j for j in cps.get(BASE, {}).get("graded", [])]
        jobs = [j for k, cp in sorted(cps.items()) if k != BASE for j in cp.get("graded", [])]
        if not (execute and jobs):
            log.append(f"feed: {len(jobs)} checkpoint jobs, {len(base)} baseline jobs"
                       f"{' (publish)' if self.s.publish else ''}")
            return
        summary = self.p.feed(jobs, base)
        log.append(f"feed: {json.dumps(summary, sort_keys=True)}")

    def _next(self, state: dict, log: list[str]) -> str | None:
        cps = state["checkpoints"]
        if self.s.baseline_model and BASE not in cps:
            return BASE
        taken = [int(k) for k in cps if k != BASE]
        last = max(taken, default=0)
        fresh = sorted(s for s in self.p.backups() if s > last and s % self.s.every == 0)
        ready = None
        for step in reversed(fresh):                      # the newest complete backup
            if self.p.revision(self.s.repo(step)) is not None:
                ready = step
                break
            log.append(f"backup step {step} not complete yet")
        if ready is None:
            log.append("nothing new to evaluate")
            return None
        for step in fresh:
            if step < ready:
                cps[str(step)] = {"status": "skipped", "repo": self.s.repo(step)}
                log.append(f"skip step {step}: evaluations behind, step {ready} is newer")
        return str(ready)

    def _begin(self, state: dict, key: str, log: list[str], execute: bool) -> None:
        if key == BASE:
            repo, revision = self.s.baseline_model, self.s.baseline_revision or self.p.revision(self.s.baseline_model)
        else:
            repo = self.s.repo(int(key))
            revision = self.p.revision(repo)
        sha = self.p.fingerprint(repo, revision) if execute else "<jobs fingerprint>"
        cp = {"status": "serving", "repo": repo, "revision": revision, "sha256": sha, "jobs": self._jobs(key),
              "graded": [], "abandoned": [], "started_at": self.clock()}
        for entry, job in zip(self.s.suite, cp["jobs"]):
            if self.p.job_exists(job):
                log.append(f"job {job} already declared")
                continue
            argv = self._create_argv(entry, job, repo, revision, sha)
            log.append("run " + shlex.join(argv))
            if execute:
                self.p.run(argv)
        log.append(f"start validator for {repo}@{revision}: {self._command(cp)}")
        state["checkpoints"][key], state["active"] = cp, key
        if execute:
            state["pid"] = self._start(cp)

    def _create_argv(self, entry: SuiteEntry, job: str, repo: str, revision: str, sha: str) -> list[str]:
        argv = ["reliquary", "jobs", "create", "--fleet-knows-corpus-generation", "--job-id", job,
                "--model", repo, "--model-revision", revision, "--model-architecture", self.s.model_architecture,
                "--checkpoint-sha256", sha, "--eval-set", entry.set_id,
                "--renderer-id", self.s.renderer_id, "--eos-token-id", str(self.s.eos_token_id),
                "--max-new-tokens", str(entry.max_new_tokens), "--slots-per-prompt", str(entry.slots_per_prompt),
                "--temperature", str(self.s.temperature), "--top-p", str(self.s.top_p), "--cap", str(entry.cap)]
        if entry.prompt_count is not None:
            argv += ["--prompt-count", str(entry.prompt_count)]
        return argv

    def _command(self, cp: dict) -> str:
        job = cp["jobs"][0]
        values = {k: shlex.quote(str(v)) for k, v in self.s.start_values.items()}
        return self.s.start_cmd.format(job=job, jobs=",".join(cp["jobs"]), model=cp["repo"],
                                       revision=cp["revision"],
                                       contract=shlex.quote(str(self.s.results_dir / f"{job}.contract.json")),
                                       **values)

    def _start(self, cp: dict) -> int:
        self.s.results_dir.mkdir(parents=True, exist_ok=True)
        return self.p.start(self._command(cp), self.s.results_dir / f"validator-{cp['jobs'][0]}.log")


# --- the real ports -----------------------------------------------------------------------------------------------

def live_ports(settings: Settings) -> Ports:
    import asyncio
    import os
    import signal
    import subprocess

    from reliquary.trainer.hf_backup import HfRepoPerBackup

    repos = HfRepoPerBackup(settings.hf_namespace, settings.hf_prefix)

    def revision(repo: str) -> str | None:
        from huggingface_hub.utils import RepositoryNotFoundError

        try:
            info = repos._api.model_info(repo)
        except RepositoryNotFoundError:
            return None
        names = {s.rfilename for s in info.siblings or ()}
        # create_repo comes before upload_folder's one commit: a repo without config.json is still uploading
        return info.sha if "config.json" in names else None

    def fingerprint(repo: str, rev: str) -> str:
        out = subprocess.run(["reliquary", "jobs", "fingerprint", repo, "--revision", rev],
                             check=True, capture_output=True, text=True).stdout
        return out.strip().splitlines()[-1]

    def job_exists(job: str) -> bool:
        from reliquary.infrastructure.corpus_job_store import read_job

        return asyncio.run(read_job(job))[0] is not None

    def drained(job: str) -> bool:
        from reliquary.infrastructure.corpus_record_store import BucketRecordStore
        from reliquary.validator.corpus_job_status import stored_job_counts

        return bool(asyncio.run(stored_job_counts(BucketRecordStore(), job))["drained"])

    def grade(job: str, *, out: Path, rl: dict, allow_incomplete: bool) -> dict:
        from reliquary.eval.job_grading import grade_served_job, upload_grading

        manifest = asyncio.run(grade_served_job(job, out=out, rl=rl, allow_incomplete=allow_incomplete))
        if settings.upload:
            asyncio.run(upload_grading(job, out))
        return manifest

    def start(command: str, log: Path) -> int:
        with log.open("ab") as handle:
            process = subprocess.Popen(["bash", "-c", command], stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        return process.pid

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def stop(pid: int) -> None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + 120
        while time.time() < deadline and alive(pid):
            time.sleep(2)
        if alive(pid):
            os.killpg(pid, signal.SIGKILL)

    def feed(jobs: list[str], baselines: list[str]) -> dict:
        from reliquary.eval import run_feed

        checkpoints, base, missing = asyncio.run(run_feed.collect(
            run_feed.LocalReports(settings.results_dir), jobs=jobs, baseline_jobs=baselines))
        built = run_feed.build_feed(settings.run_id, checkpoints, base, observed_at_ms=int(time.time() * 1000))
        path = settings.results_dir / "benchmarks.json"
        path.write_bytes(run_feed.encode_feed(built))
        summary: dict[str, Any] = {"path": str(path), "missing": missing,
                                   "points": sum(len(b["checkpoints"]) for b in built["benchmarks"])}
        if settings.publish:
            summary["published"] = run_feed.publish_feed(
                built, base_url=os.environ.get("RELIQUARY_RUN_BENCHMARKS_BASE_URL", "").strip() or None)
        if settings.wandb:
            from reliquary.eval import wandb_feed

            summary["wandb"] = wandb_feed.log_feed(built, short_of={e.set_id: e.short for e in settings.suite},
                                                   state_path=settings.results_dir / "wandb-logged.json")
        return summary

    def run(argv: list[str]) -> None:
        subprocess.run(argv, check=True)

    return Ports(backups=repos.list_backups, revision=revision, fingerprint=fingerprint, job_exists=job_exists,
                 drained=drained, run=run, grade=grade, start=start, alive=alive, stop=stop, feed=feed)


__all__ = ["BASE", "DEFAULT_START", "Ports", "Scheduler", "Settings", "SuiteEntry", "job_id", "live_ports",
           "load_suite"]
