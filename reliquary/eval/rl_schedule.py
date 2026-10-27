"""Held-out evaluations of an RL run's milestone checkpoints, one at a time on one GPU (``eval schedule``).

Each ``tick`` (idempotent, its state in one JSON file):

1. A checkpoint being served: grade each of its jobs once drained (``grade_served_job`` with the RL identity), the
   ones missing samples after ``max_wait_s`` counted as failures (``--allow-incomplete``); once all are graded,
   stop its validator and rebuild the feed (``run_feed``), published when asked (public bucket and/or the summary
   of a W&B evaluation run, ``wandb_feed``). A validator found dead is
   restarted. Each graded job is closed (``tasks close``) and each abandoned one set to cap 0, so the next
   checkpoint's jobs find their share of the pool free.
2. Nothing served: the start checkpoint first (the baseline, and later the suite's sets it never ran), then the
   steps asked for (``steps``: exactly those, each for the sets it misses), else the NEWEST milestone backup
   (``<namespace>/<prefix>-step-<n>``, ``n`` a multiple of ``every``) past the last one taken. The ones in between
   are marked skipped: when evaluations fall behind they jump to the latest milestone, and training never waits
   (the trainer does not know this exists). Its jobs are declared (``jobs create``, ids
   ``eval-<run>-s<step:05d>-<short>``) and one validator process is started for that model; the suite's later jobs
   join it as hot jobs (one model per process). ``only`` narrows the suite to some of its sets.

Dry run by default: the reads happen (backups, job drains), the writes are printed, the state is not saved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
# the merged contract of every job served: one job's contract would refuse the others' environments
DEFAULT_START = ("reliquary tasks contract {task_ids} > {contract} && "
                 "RELIQUARY_TASK_ID={jobs} RELIQUARY_TASK_CONTRACT={contract} RELIQUARY_CORPUS_HOT_JOBS=1 "
                 "exec reliquary validate --wallet-name {wallet} --hotkey {hotkey} "
                 "--http-host 0.0.0.0 --http-port {port} --no-set-weights")


@dataclass(frozen=True)
class SuiteEntry:
    set_id: str
    short: str
    slots_per_prompt: int
    cap: float
    max_new_tokens: int                 # an episode job: per turn
    prompt_count: int | None = None
    # An agentic (multi-turn) set: its `jobs create --episode-file`, `--harness id@version` and renderer.
    episode_file: str | None = None
    harness: str | None = None
    renderer_id: str | None = None


_SUITE_KEYS = {"set_id", "short", "slots_per_prompt", "cap", "max_new_tokens", "prompt_count",
               "episode_file", "harness", "renderer_id"}


def load_suite(path: str | Path) -> list[SuiteEntry]:
    """A JSON list of ``{set_id, short, slots_per_prompt, cap, max_new_tokens[, prompt_count]}``, plus
    ``episode_file`` (relative to the suite file), ``harness`` and ``renderer_id`` for an agentic set."""
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("the suite is a non-empty JSON list")
    suite = []
    for item in raw:
        unknown = set(item) - _SUITE_KEYS
        if unknown:
            raise ValueError(f"suite entry {item.get('set_id')!r}: unknown keys {sorted(unknown)}")
        entry = SuiteEntry(**item)
        if not _SHORT.fullmatch(entry.short):
            raise ValueError(f"short {entry.short!r}: lowercase letters, digits and '-', at most 16")
        if entry.episode_file is None:
            if entry.harness is not None:
                raise ValueError(f"suite entry {entry.set_id!r}: a harness needs an episode_file")
        else:
            episode = (path.parent / entry.episode_file).resolve()
            if not isinstance(json.loads(episode.read_text()), dict):
                raise ValueError(f"suite entry {entry.set_id!r}: {episode} is not an episode object")
            entry = replace(entry, episode_file=str(episode))
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
    every: int | None                   # a checkpoint every N steps (a multiple of the milestone cadence)
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
    wandb: bool = False                 # the feed into a W&B evaluation run too (wandb_feed.mirror)
    wandb_run_id: str | None = None     # default: <trainer group>-evals
    wandb_project: str | None = None    # default: WANDB_PROJECT, then reliquary-validator
    wandb_entity: str | None = None     # default: WANDB_ENTITY
    steps: Sequence[int] = ()           # evaluate exactly these steps instead of following the milestones
    only: Sequence[str] = ()            # these suite shorts only (default: the whole suite)

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
    release: Callable[[str, bool], None]                     # (job, drained): close it, else cap 0


class NoSamples(Exception):
    """A job closed with no sample at all: nothing to grade, it is abandoned instead of scored 0."""


def ungraded_rows(report: Mapping) -> int:
    """Rows the graders could not score (a taskset not installed, a grader crash): such a report is not uploaded."""
    return sum(env.get("ungraded_rows") or 0 for env in report.get("envs", {}).values())


class Scheduler:
    def __init__(self, settings: Settings, ports: Ports, *, clock: Callable[[], float] = time.time) -> None:
        self.s, self.p, self.clock = settings, ports, clock
        unknown = set(settings.only) - {e.short for e in settings.suite}
        if unknown:
            raise ValueError(f"--only {sorted(unknown)}: not in the suite")
        if not settings.steps and settings.every is None:
            raise ValueError("give --every (follow the milestones) or --step (these checkpoints)")
        self.entries = [e for e in settings.suite if not settings.only or e.short in settings.only]

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

    def _job(self, key: str, entry: SuiteEntry) -> str:
        return job_id(self.s.run_tag, None if key == BASE else int(key), entry.short)

    def _missing(self, state: dict, key: str) -> list[SuiteEntry]:
        """The selected sets this checkpoint never ran (graded and abandoned jobs count as run). A job is matched by
        its set's short, so a seeded baseline under other job names counts too."""
        have = state["checkpoints"].get(key, {}).get("jobs", ())
        return [e for e in self.entries if not any(j.endswith(f"-{e.short}") for j in have)]

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
            except NoSamples:
                log.append(f"abandon {job}: no sample after {self.s.max_wait_s:.0f}s, not graded")
                cp["abandoned"].append(job)
                self.save(state)
                continue
            cp["graded"].append(job)
            self.save(state)                              # a restart must not grade (and upload) it again
            log.append(f"graded {job}")
        self._release(cp, log, execute)
        if len(cp["graded"]) + len(cp["abandoned"]) < len(cp["jobs"]):
            pid = state["pid"]
            if pid is None or not self.p.alive(pid):     # no pid: the schedule was restarted without its validator
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

    def _release(self, cp: dict, log: list[str], execute: bool) -> None:
        """Free the pool share of the jobs done with: a failure is retried on the next tick."""
        released = cp.setdefault("released", [])
        for job in [*cp["graded"], *cp["abandoned"]]:
            if job in released:
                continue
            drained = job in cp["graded"]
            if not execute:
                log.append(f"release {job} ({'close' if drained else 'cap 0'})")
                continue
            try:
                self.p.release(job, drained)
            except Exception as exc:                      # noqa: BLE001 - the registry may be briefly unreachable
                log.append(f"release {job} failed, retried next tick: {exc}")
                continue
            released.append(job)
            log.append(f"released {job}")

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
        if self.s.baseline_model and self._missing(state, BASE):
            return BASE
        if self.s.steps:
            for step in self.s.steps:
                if not self._missing(state, str(step)):
                    continue
                if self.p.revision(self.s.repo(step)) is None:
                    log.append(f"step {step}: no complete backup {self.s.repo(step)}")
                    continue
                return str(step)
            log.append("the asked steps are evaluated")
            return None
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
        new = self._missing(state, key)
        old = state["checkpoints"].get(key, {})
        if "jobs" in old:                                 # ran before: the same weights, the sets it missed
            repo, revision, sha = old["repo"], old["revision"], old["sha256"]
        elif key == BASE:
            repo, revision = self.s.baseline_model, self.s.baseline_revision or self.p.revision(self.s.baseline_model)
            sha = self.p.fingerprint(repo, revision) if execute else "<jobs fingerprint>"
        else:
            repo = self.s.repo(int(key))
            revision = self.p.revision(repo)
            sha = self.p.fingerprint(repo, revision) if execute else "<jobs fingerprint>"
        cp = {"status": "serving", "repo": repo, "revision": revision, "sha256": sha,
              "jobs": [*old.get("jobs", ()), *(self._job(key, e) for e in new)],
              "graded": list(old.get("graded", ())), "abandoned": list(old.get("abandoned", ())),
              "started_at": self.clock()}
        for entry in new:
            job = self._job(key, entry)
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
                "--renderer-id", entry.renderer_id or self.s.renderer_id, "--eos-token-id", str(self.s.eos_token_id),
                "--max-new-tokens", str(entry.max_new_tokens), "--slots-per-prompt", str(entry.slots_per_prompt),
                "--temperature", str(self.s.temperature), "--top-p", str(self.s.top_p), "--cap", str(entry.cap)]
        if entry.prompt_count is not None:
            argv += ["--prompt-count", str(entry.prompt_count)]
        if entry.episode_file is not None:
            argv += ["--episode-file", entry.episode_file]
        if entry.harness is not None:
            argv += ["--harness", entry.harness]
        return argv

    def _command(self, cp: dict) -> str:
        pending = [j for j in cp["jobs"] if j not in cp["graded"] and j not in cp["abandoned"]]
        job = pending[0]                                  # the validator serves the jobs not graded yet
        values = {k: shlex.quote(str(v)) for k, v in self.s.start_values.items()}
        task_ids = " ".join(f"--task-id {shlex.quote(j)}" for j in pending)
        return self.s.start_cmd.format(job=job, jobs=",".join(pending), task_ids=task_ids, model=cp["repo"],
                                       revision=cp["revision"],
                                       contract=shlex.quote(str(self.s.results_dir / f"{job}.contract.json")),
                                       **values)

    def _start(self, cp: dict) -> int:
        self.s.results_dir.mkdir(parents=True, exist_ok=True)
        pending = next(j for j in cp["jobs"] if j not in cp["graded"] and j not in cp["abandoned"])
        return self.p.start(self._command(cp), self.s.results_dir / f"validator-{pending}.log")


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
        if not manifest.get("rows"):
            raise NoSamples(job)
        failed = ungraded_rows(json.loads((out / "report.json").read_text()))
        if failed:
            raise RuntimeError(f"{job}: {failed} rows not graded (grader errors, see graded.parquet), "
                               "not uploaded; retried next tick")
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
            if os.waitpid(pid, os.WNOHANG)[0] == pid:    # our child exited: reap it, a zombie still answers kill 0
                return False
        except ChildProcessError:
            pass                                          # not our child (a schedule restarted under the same pid)
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

            summary["wandb"] = wandb_feed.mirror(
                built, short_of={e.set_id: e.short for e in settings.suite},
                state_path=settings.results_dir / "wandb-logged.json", run_id=settings.wandb_run_id,
                project=settings.wandb_project, entity=settings.wandb_entity)
        return summary

    def run(argv: list[str]) -> None:
        subprocess.run(argv, check=True)

    def release(job: str, drained: bool) -> None:
        # `tasks close` refuses a job not drained: an abandoned one only gives its share back.
        run(["reliquary", "tasks", "close", "--task-id", job] if drained
            else ["reliquary", "tasks", "set-cap", "--task-id", job, "--cap", "0"])

    return Ports(backups=repos.list_backups, revision=revision, fingerprint=fingerprint, job_exists=job_exists,
                 drained=drained, run=run, grade=grade, start=start, alive=alive, stop=stop, feed=feed,
                 release=release)


__all__ = ["BASE", "DEFAULT_START", "Ports", "Scheduler", "Settings", "SuiteEntry", "job_id", "live_ports",
           "load_suite"]
