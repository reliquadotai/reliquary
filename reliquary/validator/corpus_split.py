"""The split corpus validator: a supervisor and its children in one container.

``reliquary validate`` with ``RELIQUARY_CORPUS_SPLIT=1`` runs ``run_corpus_split``
instead of the single process. The supervisor checks what the single process
checks before loading a model, then starts (``multiprocessing``, spawn):

- ``gpu``: the checkpoint, once (``corpus_gpu.run_gpu_process``);
- ``judge-<n>``: one per group of ``RELIQUARY_CORPUS_SPLIT_JUDGES``
  (``corpus_judge_process.run_corpus_judges``);
- ``front``: the miner routes and the jobs in no group
  (``corpus_validator.run_corpus_validator`` with a ``FrontSplit``), episode
  jobs always among them: their grader and grade routes live there.

A child that exits is started again alone, after a backoff; the others never
notice. SIGTERM/SIGINT stop every child. Design:
``docs/design/2026-10-02-corpus-split-processes.md``.
"""

from __future__ import annotations

import asyncio
import ctypes
import importlib
import logging
import multiprocessing
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

SPLIT_ENV = "RELIQUARY_CORPUS_SPLIT"
JUDGES_ENV = "RELIQUARY_CORPUS_SPLIT_JUDGES"
DIR_ENV = "RELIQUARY_CORPUS_SPLIT_DIR"
DEFAULT_RUN_DIR = "/tmp/reliquary-corpus-split"
NICE_ENV = "RELIQUARY_CORPUS_SPLIT_NICE"
DEFAULT_NICE = 5
RESTART_BACKOFF_SECONDS = 1.0
MAX_RESTART_BACKOFF_SECONDS = 60.0
# A child up this long has its backoff reset.
HEALTHY_SECONDS = 300.0
STOP_GRACE_SECONDS = 20.0
POLL_SECONDS = 1.0


def judge_socket(run_dir: str | Path, index: int) -> Path:
    return Path(run_dir) / f"judge-{index}.sock"


def plan_groups(value: str | None, jobs, *, front_only=()) -> list[list[str]]:
    """The job ids of each judge process. ``jobs`` is ``(task_id, job_id)``
    pairs; ``value`` names jobs by job id or task id, ``,`` inside a group and
    ``;`` between groups; ``*`` gives every job no group names its own process.
    Unset or empty means ``*``.

    ``front_only`` are the job ids the front must judge itself (episode jobs:
    their grader, grade dispatcher and grade routes live in the front, judge
    processes host none). ``*`` leaves them there; a group naming one is
    refused."""
    front_only = {str(j) for j in front_only}
    value = "*" if value is None or not value.strip() else value
    by_name: dict[str, str] = {}
    for task_id, job_id in jobs:
        by_name[str(job_id)] = str(job_id)
        by_name[str(task_id)] = str(job_id)
    groups: list[list[str]] = []
    star = False
    placed: set[str] = set()
    for raw in value.split(";"):
        names = [n.strip() for n in raw.split(",") if n.strip()]
        if not names:
            continue
        if names == ["*"]:
            star = True
            continue
        group = []
        for name in names:
            if name == "*":
                raise ValueError(f"{JUDGES_ENV}: '*' must be a group of its own")
            if name not in by_name:
                raise ValueError(f"{JUDGES_ENV} names {name!r}, which this validator does not serve "
                                 f"(it serves {sorted({str(j) for _, j in jobs})})")
            job_id = by_name[name]
            if job_id in front_only:
                raise ValueError(f"{JUDGES_ENV} names episode job {job_id!r}, but judge processes "
                                 "host no grader: leave it out of every group (the front audits, "
                                 "grades and settles it)")
            if job_id in placed:
                raise ValueError(f"{JUDGES_ENV} puts job {job_id!r} in two groups; "
                                 "a job is judged in exactly one process")
            placed.add(job_id)
            group.append(job_id)
        groups.append(group)
    if star:
        groups += [[str(job_id)] for _, job_id in jobs
                   if str(job_id) not in placed and str(job_id) not in front_only]
    return groups


@dataclass
class SplitSpec:
    """Everything a child needs, picklable: the children are spawned."""

    served: list                      # (TaskEntry, cap)
    directory: str
    fingerprint: str
    proof: Any                        # ProofProfile
    run_dir: str
    groups: list
    http_host: str = "127.0.0.1"
    http_port: int = 18090
    netuid: int = 81
    registration_gate: bool = True
    hot: bool = False
    settle_every_seconds: float = 60.0
    model_id: str = ""
    model_revision: str = ""
    # "module:function" called first in every child (tests install fakes).
    child_init: str | None = None
    # Test knobs for every auditor (hold slack, rescan period); empty in prod.
    auditor_kwargs: dict = field(default_factory=dict)

    def group_of(self, job_id: str) -> int | None:
        for index, group in enumerate(self.groups):
            if str(job_id) in group:
                return index
        return None


@dataclass
class FrontSplit:
    """What ``run_corpus_validator`` needs to run as the front."""

    directory: str
    fingerprint: str
    proof: Any
    run_dir: str
    links: dict                       # job id -> JudgeLink


def _set_parent_death_signal() -> None:
    """SIGKILL this child if the supervisor dies (Linux): no orphan keeps a
    socket, a GPU, or a job's judging."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        logger.debug("PR_SET_PDEATHSIG unavailable", exc_info=True)


def _exit_now(signum, frame) -> None:
    logging.getLogger(__name__).info("corpus split: signal %d, exiting", signum)
    logging.shutdown()
    os._exit(0)


def _call(path: str) -> None:
    module, _, name = path.partition(":")
    getattr(importlib.import_module(module), name)()


async def _front(spec: SplitSpec) -> None:
    from reliquary.validator.corpus_feed import JudgeLink
    from reliquary.validator.corpus_validator import run_corpus_validator

    links: dict[str, Any] = {}
    for index, group in enumerate(spec.groups):
        link = JudgeLink(judge_socket(spec.run_dir, index), group)
        for job_id in group:
            links[job_id] = link
    read_registry = None
    if spec.hot:
        from reliquary.validator.corpus_judge_process import registry_reader

        read_registry = registry_reader()
    await run_corpus_validator(
        jobs=spec.served, wallet=None, netuid=spec.netuid, signer_client=None,
        http_host=spec.http_host, http_port=spec.http_port, set_weights=False,
        registration_gate=spec.registration_gate, read_registry=read_registry,
        settle_every_seconds=spec.settle_every_seconds,
        split=FrontSplit(directory=spec.directory, fingerprint=spec.fingerprint,
                         proof=spec.proof, run_dir=spec.run_dir, links=links),
        auditor_kwargs=spec.auditor_kwargs)


async def _judge(spec: SplitSpec, index: int) -> None:
    from reliquary.validator.corpus_judge_process import run_corpus_judges

    group = set(spec.groups[index])
    await run_corpus_judges(
        served=[(e, c) for e, c in spec.served if str(e.job_id) in group],
        directory=spec.directory, run_dir=spec.run_dir, proof=spec.proof,
        socket_path=str(judge_socket(spec.run_dir, index)),
        settle_every_seconds=spec.settle_every_seconds, hot=spec.hot,
        auditor_kwargs=spec.auditor_kwargs)


async def _gpu(spec: SplitSpec) -> None:
    from reliquary.validator.corpus_gpu import run_gpu_process

    await run_gpu_process(directory=spec.directory, run_dir=spec.run_dir,
                          model_id=spec.model_id, model_revision=spec.model_revision)


def child_main(role: str, index: int, spec: SplitSpec) -> None:
    """The entry point of every child (spawned: a fresh interpreter)."""
    name = role if role != "judge" else f"judge-{index}"
    # Verified drand rounds survive a child's restart (rounds never change).
    os.environ.setdefault("RELIQUARY_CORPUS_DRAND_CACHE",
                          str(Path(spec.run_dir) / f"drand-rounds-{name}.jsonl"))
    if role != "gpu":
        # Never a CUDA context outside the GPU process.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if role != "front":
        # A saturated host serves miners first; judging catches up after.
        try:
            os.nice(int(os.environ.get(NICE_ENV, DEFAULT_NICE)))
        except OSError:
            pass
    _set_parent_death_signal()
    # SIGTERM ends the child at once: what it was doing is safe to cut (create-only
    # verdicts, CAS ledgers and settlement). The front's HTTP server takes the
    # signal first, drains in-flight submissions, then re-raises it here.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _exit_now)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | {name} | %(threadName)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", force=True)
    if spec.child_init:
        _call(spec.child_init)
    coroutine = {"front": lambda: _front(spec), "gpu": lambda: _gpu(spec),
                 "judge": lambda: _judge(spec, index)}[role]()
    asyncio.run(coroutine)


@dataclass
class _Child:
    name: str
    role: str
    index: int
    process: Any = None
    started_at: float = 0.0
    failures: int = 0
    restart_at: float | None = None
    starts: int = 0


class Supervisor:
    """Starts the children and keeps each one running, alone."""

    def __init__(self, spec: SplitSpec, *, clock=time.monotonic,
                 backoff_seconds: float = RESTART_BACKOFF_SECONDS,
                 max_backoff_seconds: float = MAX_RESTART_BACKOFF_SECONDS) -> None:
        self.spec = spec
        self._clock = clock
        self._backoff = backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._context = multiprocessing.get_context("spawn")
        self.children: dict[str, _Child] = {"gpu": _Child("gpu", "gpu", 0)}
        for index in range(len(spec.groups)):
            name = f"judge-{index}"
            self.children[name] = _Child(name, "judge", index)
        self.children["front"] = _Child("front", "front", 0)
        self._stopping = False

    def pid(self, name: str) -> int | None:
        process = self.children[name].process
        return process.pid if process is not None else None

    def _start(self, child: _Child) -> None:
        child.process = self._context.Process(
            target=child_main, args=(child.role, child.index, self.spec),
            name=f"corpus-{child.name}", daemon=False)
        child.process.start()
        child.started_at = self._clock()
        child.restart_at = None
        child.starts += 1
        logger.info("corpus split: %s started (pid %d, start %d)", child.name,
                    child.process.pid, child.starts)

    def start(self) -> None:
        run_dir = Path(self.spec.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        # A container restart keeps /tmp: nothing of the last run is trusted.
        for stale in [*run_dir.glob("*.sock"), run_dir / "gpu-info.json"]:
            stale.unlink(missing_ok=True)
        for child in self.children.values():
            self._start(child)

    def check(self) -> None:
        """Restart, alone and after its backoff, any child that exited."""
        now = self._clock()
        for child in self.children.values():
            process = child.process
            if process is not None and process.is_alive():
                continue
            if child.restart_at is None:
                code = process.exitcode if process is not None else None
                if process is not None:
                    process.join(0)
                lived = now - child.started_at
                child.failures = 1 if lived >= HEALTHY_SECONDS else child.failures + 1
                delay = min(self._max_backoff, self._backoff * 2 ** (child.failures - 1))
                child.restart_at = now + delay
                logger.error("corpus split: %s exited (code %s) after %.0f s; restarting in %.0f s",
                             child.name, code, lived, delay)
            if now >= child.restart_at and not self._stopping:
                self._start(child)

    def stop(self, grace_seconds: float = STOP_GRACE_SECONDS) -> None:
        self._stopping = True
        live = [c.process for c in self.children.values()
                if c.process is not None and c.process.is_alive()]
        for process in live:
            process.terminate()
        deadline = time.monotonic() + grace_seconds
        for process in live:
            process.join(max(0.0, deadline - time.monotonic()))
        for process in live:
            if process.is_alive():
                logger.warning("corpus split: %s ignored SIGTERM; killing", process.name)
                process.kill()
                process.join(5)
        logger.info("corpus split: stopped")

    async def run(self, *, poll_seconds: float = POLL_SECONDS) -> None:
        self.start()
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        try:
            while not stop.is_set():
                self.check()
                try:
                    await asyncio.wait_for(stop.wait(), timeout=poll_seconds)
                except asyncio.TimeoutError:
                    pass
        finally:
            self.stop()


async def preflight(served) -> SimpleNamespace:
    """What the single process checks before it loads a model, here before
    any child starts: the manifests, order jobs, one checkpoint and proof
    for all, the checkpoint's download and fingerprint."""
    from huggingface_hub import snapshot_download

    from reliquary.corpus.encoding import checkpoint_fingerprint
    from reliquary.eval.prompt_source import is_order_job_id
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, toploc_proof
    from reliquary.validator.corpus_validator import multi_job_refusal, startup_refusal

    for entry, _ in served:
        if is_order_job_id(entry.job_id):
            raise RuntimeError(f"task {entry.task_id!r} is an order job (eval or generation): "
                               "the order control serves it, never the corpus control")
    store = BucketJobStore()
    manifests = []
    for entry, _ in served:
        job, _ = await store.read_job(str(entry.job_id))
        if job is None:
            raise RuntimeError(f"task {entry.task_id!r} declares job {entry.job_id!r} "
                               "but it has no manifest")
        manifests.append((entry, job))
    if len(manifests) > 1:
        carried = all(getattr(e, "contract", None) is not None for e, _ in manifests)
        refusal = multi_job_refusal(
            manifests,
            process_contract=ACTIVE_PROTOCOL_PROFILE.to_generation_contract() if carried else None)
        if refusal:
            raise RuntimeError(refusal)
    first = manifests[0][1]
    directory = Path(await asyncio.to_thread(
        snapshot_download, first.checkpoint_repo, revision=first.checkpoint_revision))
    fingerprint = await asyncio.to_thread(checkpoint_fingerprint, directory)
    for entry, job in manifests:
        refusal = startup_refusal(entry, job, ACTIVE_PROTOCOL_PROFILE, fingerprint)
        if refusal:
            raise RuntimeError(refusal if len(manifests) == 1
                               else f"task {entry.task_id!r}: {refusal}")
    return SimpleNamespace(directory=str(directory), fingerprint=fingerprint,
                           proof=toploc_proof(ACTIVE_PROTOCOL_PROFILE),
                           model_id=first.checkpoint_repo,
                           model_revision=first.checkpoint_revision,
                           jobs=[(str(e.task_id), str(j.job_id)) for e, j in manifests],
                           episode_jobs=[str(j.job_id) for _, j in manifests
                                         if getattr(j, "episode", None) is not None])


async def run_corpus_split(*, served, netuid: int, http_host: str, http_port: int,
                           set_weights: bool, registration_gate: bool = True,
                           hot: bool = False, remote_audit: bool = False,
                           settle_every_seconds: float = 60.0) -> None:
    """The CLI's entry: preflight, plan, supervise until SIGTERM."""
    if set_weights:
        raise RuntimeError("the split validator does not set weights; run it with "
                           "--no-set-weights (the RL validator's setter pays every task)")
    if remote_audit:
        raise RuntimeError("remote audit executors are not served by the split validator; "
                           "unset RELIQUARY_CORPUS_REMOTE_AUDIT or RELIQUARY_CORPUS_SPLIT")
    checked = await preflight(served)
    # Before any child starts: an episode job is judged and graded in the front.
    groups = plan_groups(os.environ.get(JUDGES_ENV), checked.jobs,
                         front_only=checked.episode_jobs)
    run_dir = os.environ.get(DIR_ENV) or DEFAULT_RUN_DIR
    spec = SplitSpec(served=list(served), directory=checked.directory,
                     fingerprint=checked.fingerprint, proof=checked.proof, run_dir=run_dir,
                     groups=groups, http_host=http_host, http_port=http_port, netuid=netuid,
                     registration_gate=registration_gate, hot=hot,
                     settle_every_seconds=settle_every_seconds,
                     model_id=checked.model_id, model_revision=checked.model_revision)
    in_front = [j for _, j in checked.jobs if spec.group_of(j) is None]
    logger.info("corpus split: judge processes %s; judged in the front %s; run dir %s",
                groups, in_front, run_dir)
    await Supervisor(spec).run()


__all__ = [
    "FrontSplit",
    "SplitSpec",
    "Supervisor",
    "child_main",
    "judge_socket",
    "plan_groups",
    "preflight",
    "run_corpus_split",
]
