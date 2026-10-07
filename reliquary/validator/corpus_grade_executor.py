"""`reliquary corpus grade-executor`: a CPU box with Docker that grades and
replays agentic trajectories (spec §5 N5).

It holds one secret, its token, pulls leases over HTTPS it opens itself, and
returns facts: whether the diff applied and the tests passed, whether the
replay reproduced the diff, which observations differed. It never decides a
verdict. Hostile code runs here, never on the control; the boxes come from the
task's public images pinned by digest, each under ``BoxLimits`` (CPUs, memory
without swap, pids). Size the host for ``concurrency * memory_gb`` plus the
executor itself; run one executor per Docker host (``start`` removes every box
named ``BOX_NAME_PREFIX`` ("reliquary-gradebox-") it finds, the ones a killed executor left behind).
`verifiers` and `reliquary_swe` are imported here only.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import subprocess
import time
from collections.abc import Callable

from reliquary.corpus.replay_compare import Action, compare
from reliquary.validator.agentic_replay import (
    DEFAULT_BOX_LIMITS,
    DISK_TOLERANCE,
    GRADE_OUTPUT_BYTES,
    BoxLimits,
    BoxLost,
    ReplayTimeout,
    bounded_box,
    output_bounded,
    refresh_repository_index,
    root_size_bytes,
    replay_swe,
    sweep_orphan_boxes,
    swesmith_task,
)
from reliquary.validator.corpus_grade_protocol import GradeItem, GradeLease
from reliquary.validator.lease_executor import TOKEN_ENV, LeaseExecutor, serve_executor

logger = logging.getLogger(__name__)

GRADE_PREFIX = "/corpus/internal/grade"
HEARTBEAT_SECONDS = 20.0
IDLE_SECONDS = 5.0
DEFAULT_SCORING_SECONDS = 1800.0


# What a trajectory itself caused (ruling P23): reported as facts, decided on
# by two distinct providers like any other, never as this executor's fault.
BOX_LOST = "box_lost"
BOX_TIMEOUT = "box_timeout"


class GradeTimeout(Exception):
    """Grading exceeded the task's scoring timeout. ``trajectory_caused``: the
    miner's patch was applied by then, so its code (run by the tests) spent it."""

    def __init__(self, message: str, *, trajectory_caused: bool = False) -> None:
        super().__init__(message)
        self.trajectory_caused = trajectory_caused


class _PatchWatch:
    """Notes when ``reliquary_swe.grading.grade`` applies the patch (its
    ``git apply``): from then on the box runs the miner's code."""

    def __init__(self) -> None:
        self.applied = False

    def __call__(self, argv) -> None:
        if list(argv[:2]) == ["git", "apply"]:
            self.applied = True


async def grade_patch(task, patch: str, *, limits: BoxLimits = DEFAULT_BOX_LIMITS):
    """`reliquary_swe.grading.grade` in a fresh bounded box from the task's
    pinned image, network cut, as `SweEnv._grade` does it (one attempt: a
    failure goes back to the control, which re-leases it).

    Raises ``BoxLost`` when the box fails once the patch is applied and
    ``GradeTimeout`` past the scoring timeout (``trajectory_caused`` once the
    patch is applied); a failure before that is the executor's."""
    from reliquary_swe import grading

    deadline = task.data.timeout.scoring or DEFAULT_SCORING_SECONDS
    watch: _PatchWatch | None = None
    report = None
    try:
        async with asyncio.timeout(deadline):
            async with bounded_box(task, limits) as box:
                await box.prepare_setup()
                await refresh_repository_index(box, task)
                await box.prepare_execution([])
                watch = _PatchWatch()
                try:
                    # What it reads back is bounded (ruling P24): the tests run patched code.
                    with output_bounded(box, GRADE_OUTPUT_BYTES, on_run=watch):
                        report = await grading.grade(box, task.data, patch)
                except Exception as e:
                    if not watch.applied:
                        raise
                    raise BoxLost(f"the grade box failed after the patch was applied: "
                                  f"{type(e).__name__}: {e}"[:400]) from e
    except TimeoutError as e:
        if report is None:
            raise GradeTimeout(f"grading exceeded its scoring timeout {deadline} s",
                               trajectory_caused=bool(watch and watch.applied)) from e
        logger.warning("grade box removal outlived the deadline; the grade itself finished")
    except BoxLost:
        raise
    except Exception:
        if report is None:
            raise
        logger.exception("grade box removal failed after the grade finished")
    return report


async def run_grade_item(item: GradeItem, *, task_for=None, grade=None, replay=None,
                         limits: BoxLimits = DEFAULT_BOX_LIMITS) -> dict:
    """The facts for one item, as a ``GradeItemResult`` body. ``error`` and
    ``timeout`` are this executor's, never the miner's: the control re-leases.
    ``box_lost`` and ``box_timeout`` are the trajectory's (ruling P23): the box
    failed, or the deadline passed, once the recorded actions or the applied
    patch ran in it."""
    task_for = task_for or swesmith_task
    grade = grade or functools.partial(grade_patch, limits=limits)
    replay = replay or functools.partial(replay_swe, limits=limits)
    try:
        # The first build loads the corpus: off the loop, which serves heartbeats.
        task = await asyncio.to_thread(task_for, item.instance_id)
        if item.mode == "grade":
            try:
                report = await grade(task, item.final_diff)
            except GradeTimeout as exc:
                return {"status": BOX_TIMEOUT if exc.trajectory_caused else "timeout",
                        "detail": str(exc)[:500]}
            except TimeoutError as exc:
                return {"status": "timeout", "detail": f"grading exceeded its scoring timeout {exc}"[:500]}
            return {"status": "ok", "diff_applied": bool(report.applied),
                    "tests_passed": float(report.reward) >= 1.0}
        actions = [Action(a.tool, a.arguments, a.observation) for a in item.actions]
        observations, diff = await replay(task, actions)
        # The renderer writes each observation stripped; compare like with like.
        report = compare(actions, [o.strip() for o in observations], item.final_diff, diff)
        return {"status": "ok", "replay_diff_equal": report.diff_equal,
                "observations_compared": report.compared,
                "observations_mismatched": list(report.mismatched)}
    except BoxLost as exc:
        logger.warning("grade item %s (%s): the trajectory lost its box: %s",
                       item.submission_id[:12], item.mode, exc)
        return {"status": BOX_LOST, "detail": str(exc)[:500]}
    except ReplayTimeout as exc:
        return {"status": BOX_TIMEOUT if exc.trajectory_caused else "timeout",
                "detail": str(exc)[:500]}
    except Exception as exc:  # ours, not the miner's: the control re-leases it
        logger.exception("grade item %s (%s) failed", item.submission_id[:12], item.mode)
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}"[:500]}


def installed_env_refusal(package: str, version: str) -> str | None:
    """Why this box cannot grade for ``package@version``, or None."""
    from reliquary.environment.agentic_swe import (
        SUPPORTED_VERIFIERS,
        installed_env_commit,
        installed_verifiers_commit,
    )

    if package != "reliquary-swe":
        return f"this executor grades reliquary-swe, not {package!r}"
    if installed_env_commit() != version:
        return f"reliquary-swe is installed at {installed_env_commit()}, registered for {version}"
    if installed_verifiers_commit() != SUPPORTED_VERIFIERS:
        return f"verifiers is installed at {installed_verifiers_commit()}, not {SUPPORTED_VERIFIERS}"
    return None


# Ruling P20. `find`/`grep -r` list a directory in the order the host's Docker
# backing filesystem returns it: xfs keeps the image layer's insertion order,
# the same on every xfs host; ext4 hashes names with a per-filesystem seed, so
# every ext4 host has its own order. A `find | head` cut cannot be normalized,
# so a replay must run where that order is the one miners are told to use.
REQUIRED_FS = "xfs"
CONTAINERD_ROOT = "/var/lib/containerd"
_CONTAINERD_SNAPSHOTTER = "io.containerd.snapshotter.v1"


def docker_info() -> dict:
    out = subprocess.run(["docker", "info", "--format", "{{json .}}"], capture_output=True,
                         text=True, timeout=60, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"docker info failed ({out.returncode}): {(out.stderr or '').strip()[:300]}")
    return json.loads(out.stdout)


def path_fs_type(path: str) -> str:
    out = subprocess.run(["stat", "-f", "-c", "%T", path], capture_output=True, text=True,
                         timeout=30, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"stat -f {path} failed: {(out.stderr or '').strip()[:300]}")
    return out.stdout.strip()


def docker_storage_refusal(*, info: dict | None = None, probe: Callable[[], dict] = docker_info,
                           fs_type: Callable[[str], str] = path_fs_type) -> str | None:
    """Why this host must not replay (its image layers are not on xfs), or None.

    Checks the filesystem of Docker's data root, overlay2's reported backing
    filesystem, and, with the containerd image store, containerd's root (where
    the layers then live)."""
    try:
        info = info if info is not None else probe()
        status = {str(k): str(v) for k, v in (info.get("DriverStatus") or [])}
        found: list[tuple[str, str]] = []
        root = str(info.get("DockerRootDir") or "/var/lib/docker")
        found.append((root, fs_type(root)))
        if status.get("driver-type") == _CONTAINERD_SNAPSHOTTER:
            found.append((CONTAINERD_ROOT, fs_type(CONTAINERD_ROOT)))
        if "Backing Filesystem" in status:
            found.append((f"{info.get('Driver')} backing filesystem", status["Backing Filesystem"]))
    except Exception as exc:  # noqa: BLE001 - any failure to look is a refusal
        return f"could not check Docker's storage filesystem: {exc}"
    wrong = [f"{fs} ({where})" for where, fs in found if fs != REQUIRED_FS]
    if not wrong:
        return None
    return (f"Docker stores images on {', '.join(wrong)}, not {REQUIRED_FS}: directory order "
            f"would differ from miners' and void honest replays (ruling P20)")


# Ruling P24: a replayed `dd if=/dev/zero of=/x` must fill its own box, not the
# host's Docker storage. verifiers' `docker run` takes no `--storage-opt`, so
# the bound is the daemon's default `overlay2.size`, which overlay2 (the graph
# driver, not the containerd image store) honours on xfs mounted with pquota.
DISK_PROBE_IMAGE = "alpine:3.22"


def run_disk_probe(image: str) -> str:
    """``df -Pk /`` in a throwaway box from ``image`` (pulled if missing)."""
    out = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "df",
                          image, "-Pk", "/"], capture_output=True, text=True, timeout=600, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"disk probe box failed ({out.returncode}): "
                           f"{(out.stderr or out.stdout).strip()[:300]}")
    return out.stdout


def docker_disk_refusal(disk_gb: float, *, info: dict | None = None,
                        probe: Callable[[], dict] = docker_info,
                        run_probe: Callable[[str], str] = run_disk_probe,
                        image: str = DISK_PROBE_IMAGE) -> str | None:
    """Why this host's boxes are not bounded to ``disk_gb`` of writable layer,
    or None. Every box is checked again before its first action
    (``agentic_replay.check_box_disk``)."""
    try:
        info = info if info is not None else probe()
        if info.get("Driver") != "overlay2":
            return (f"Docker's storage driver is {info.get('Driver')!r}, not overlay2: a box's disk "
                    f"cannot be bounded (set \"storage-driver\": \"overlay2\" and \"features\": "
                    f"{{\"containerd-snapshotter\": false}} in daemon.json; ruling P24)")
        size = root_size_bytes(run_probe(image))
    except Exception as exc:  # noqa: BLE001 - any failure to look is a refusal
        return f"could not check a box's disk limit: {exc}"
    if size is None or size > disk_gb * 2 ** 30 * DISK_TOLERANCE:
        return (f"a box's / reads {size} bytes, more than --disk-gb {disk_gb}: set the daemon's "
                f"\"storage-opts\": [\"overlay2.size={disk_gb:g}G\"] with /var/lib/docker on xfs "
                f"mounted with pquota (ruling P24)")
    return None


class GradeExecutor(LeaseExecutor):
    kind = "grade executor"

    def __init__(self, *, http, executor_id: str, token: str, concurrency: int = 4,
                 run_item: Callable | None = None, limits: BoxLimits = DEFAULT_BOX_LIMITS,
                 env_check: Callable[[str, str], str | None] | None = None,
                 sweep: Callable[[], int] = sweep_orphan_boxes,
                 heartbeat_seconds: float = HEARTBEAT_SECONDS, idle_seconds: float = IDLE_SECONDS,
                 clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time) -> None:
        super().__init__(http=http, executor_id=executor_id, token=token, prefix=GRADE_PREFIX,
                         heartbeat_seconds=heartbeat_seconds, idle_seconds=idle_seconds,
                         clock=clock)
        self._concurrency = max(1, int(concurrency))
        self._run_item = run_item or functools.partial(run_grade_item, limits=limits)
        self._env_check = env_check or installed_env_refusal
        self._sweep = sweep
        # Lease expiry is the control's wall time.
        self._wall_clock = wall_clock
        self._running: set[asyncio.Task] = set()
        # Leases claimed and not yet posted: reported on each heartbeat.
        self._held: set[str] = set()
        self.env_package: str | None = None
        self.env_version: str | None = None

    def held_lease_ids(self) -> list[str]:
        return sorted(self._held)

    def heartbeat_detail(self) -> dict:
        return {"leases": self.leases, "running": len(self._running)}

    async def start(self) -> None:
        """Learn the env pin this executor is registered for, and refuse to
        grade with anything else installed."""
        answer = await self.heartbeat()
        self.env_package, self.env_version = answer["model_id"], answer["model_revision"]
        refusal = self._env_check(self.env_package, self.env_version)
        if refusal:
            raise RuntimeError(refusal)
        swept = await asyncio.to_thread(self._sweep)
        if swept:
            logger.warning("grade executor %s removed %d orphaned boxes", self._executor_id, swept)

    async def _work(self, lease: GradeLease) -> None:
        try:
            item = lease.items[0]
            result = await self._run_item(item)
            result = {**result, "submission_id": item.submission_id}
            await self.post_result(lease.lease_id, {"results": [result]},
                                   expires_at=lease.expires_at)
        except Exception:
            # Dropped from the heartbeat's report below: the control takes it
            # back after its grace (or, from an older control, it expires).
            logger.exception("grade lease %s was not completed", lease.lease_id[:8])
        finally:
            self._held.discard(lease.lease_id)

    async def step(self) -> bool:
        """One claim when there is room; True when a lease was started."""
        await self.heartbeat_if_due()
        if len(self._running) >= self._concurrency:
            return False
        response = await self._post(f"{GRADE_PREFIX}/claim", {
            "executor_id": self._executor_id, "env_package": self.env_package,
            "env_version": self.env_version})
        if response.status_code == 204:
            return False
        response.raise_for_status()
        lease = GradeLease.model_validate(response.json())
        if (lease.env.package, lease.env.version) != (self.env_package, self.env_version):
            raise RuntimeError(f"a lease for {lease.env.package}@{lease.env.version}, "
                               f"this executor grades {self.env_package}@{self.env_version}")
        if lease.expires_at <= self._wall_clock():
            logger.warning("grade lease %s expired before it was worked; skipped",
                           lease.lease_id[:8])
            return True
        self._held.add(lease.lease_id)
        task = asyncio.create_task(self._work(lease))
        self._running.add(task)
        task.add_done_callback(self._running.discard)
        return True


def run_grade_executor(*, control_url: str, executor_id: str, concurrency: int = 4,
                       limits: BoxLimits = DEFAULT_BOX_LIMITS) -> None:
    serve_executor(control_url, lambda **client: GradeExecutor(
        executor_id=executor_id, concurrency=concurrency, limits=limits, **client))


__all__ = ["BOX_LOST", "BOX_TIMEOUT", "DISK_PROBE_IMAGE", "GRADE_PREFIX", "docker_disk_refusal", "GradeExecutor", "GradeTimeout", "TOKEN_ENV", "docker_storage_refusal", "grade_patch",
           "installed_env_refusal",
           "run_grade_executor", "run_grade_item"]
