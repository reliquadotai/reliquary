"""Replay a submitted SWE episode in a fresh box: the validator's side of the
agentic corpus check (spec N5). ``verifiers`` and ``reliquary_swe`` are optional
and imported only here, like ``reliquary.eval.verifiers_source``.

Each recorded tool call runs inside the box through ``TOOL_PROGRAM``, which
reproduces verifiers b2e4e81's bash harness tools (``harnesses/bash/program.py``:
``run_bash``, ``run_edit``) and its argument-error messages, so an honest
miner's observations come back identical up to normalization.

Hardening. The replayed commands control the box, so nothing the box holds is
trusted after the first action runs: the interpreter's absolute path is
resolved once, right after setup and before any action; the tool program is
passed on every action as an argv string (``<python> -I -c TOOL_PROGRAM``) through
``docker exec``, with ``-I`` (isolated mode: neither the working directory, which
the replayed commands own, nor ``PYTHON*`` variables nor the user site reach
``sys.path``), so no file in the box can substitute it; each request goes in
a fresh file with an unpredictable name, deleted once read. The replay runs
under two wall-clock clocks (ruling P25): provisioning and setup under the
task's setup timeout plus a margin (the executor's to meet), then the
recorded actions and finalize under twice the miner's own agent + finalize
budget (9000 s on SWE-smith); past either it raises ``ReplayTimeout``, only
the second being the trajectory's. Each command also keeps its own timeout.

Resources. Every box (replay or grade, ``bounded_box``) runs under explicit
``BoxLimits``: CPUs and memory through verifiers' ``DockerConfig``, then, right
after the box starts and before any replayed action, ``docker update`` sets
its pids limit and makes memory+swap equal to memory (no swap). Boxes are named
``BOX_NAME_PREFIX`` + a random suffix so ``sweep_orphan_boxes`` can remove
those a killed executor left behind (SIGKILL skips verifiers' atexit backstop).

Limits. This does not stop an action from replacing the interpreter or
``bash`` binaries themselves, or from leaving a background process that alters
later observations; and replay cannot certify observations against
adversarially chosen commands at all while the miner's sampling is not pinned
(a forger may pick commands whose output is nondeterministic or that it can
make diverge, within the per-episode tolerance). Both are priced in plan 4 (the
pinned-sampling question), not solved here.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import math
import secrets
import subprocess
import threading
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from reliquary.corpus.replay_compare import Action
from reliquary.protocol.corpus_submission import MAX_FINAL_DIFF_CHARS
from reliquary.validator.corpus_grade_protocol import MAX_OBSERVATION_CHARS

logger = logging.getLogger(__name__)

# What the executor reads back from a hostile box (ruling P24): a replayed
# observation longer than any recorded one can be compared to is cut there
# (and then mismatches); finalize's output past the largest diff a submission
# carries likewise; a grade's test output gets a generous cap. Bytes, for
# UTF-8's worst case of 4 per char.
_UTF8 = 4
FINALIZE_OUTPUT_BYTES = _UTF8 * MAX_FINAL_DIFF_CHARS + 2 ** 20
GRADE_OUTPUT_BYTES = 256 * 2 ** 20
_READ_CHUNK = 65536

TOOL_PROGRAM = r'''
import json, subprocess, sys
from pathlib import Path

def run_bash(command, timeout):
    try:
        r = subprocess.run(["bash", "-c", command], capture_output=True, text=True,
                           timeout=timeout, check=False)
        return r.stdout + r.stderr
    except Exception as e:
        return f"error: {e}"

def run_edit(path, old_str, new_str):
    if not isinstance(path, str) or not path:
        return "error: 'path' is required"
    if not isinstance(old_str, str) or not isinstance(new_str, str):
        return "error: 'old_str' and 'new_str' must be strings"
    if not old_str:
        return "error: 'old_str' must be a non-empty string"
    filepath = Path(path)
    if not filepath.is_absolute():
        filepath = Path.cwd() / filepath
    if not filepath.exists():
        return f"error: {path} not found"
    try:
        content = filepath.read_text()
    except Exception as e:
        return f"error: could not read {path}: {e}"
    count = content.count(old_str)
    if count != 1:
        return f"error: old_str must appear exactly once in {path} (found {count})"
    try:
        filepath.write_text(content.replace(old_str, new_str, 1))
    except Exception as e:
        return f"error: could not write {path}: {e}"
    return f"Edited {path}"

if len(sys.argv) > 1:
    request_path = Path(sys.argv[1])
    req = json.loads(request_path.read_text())
    request_path.unlink()
else:
    req = json.loads(sys.stdin.read())
try:
    args = json.loads(req["arguments"] or "{}")
except json.JSONDecodeError as e:
    out = f"error: invalid JSON in tool arguments ({e}); resend the call with valid JSON"
else:
    if not isinstance(args, dict):
        out = f"error: tool arguments must be a JSON object, got {type(args).__name__}; resend as an object"
    elif req["tool"] == "bash":
        out = run_bash(args.get("command", ""), req["timeout"])
    elif req["tool"] == "edit":
        out = run_edit(args.get("path"), args.get("old_str"), args.get("new_str"))
    else:
        out = f"error: unknown tool {req['tool']!r}"
if req.get("max_chars"):
    out = out[:req["max_chars"]]
sys.stdout.write(out)
'''

# Resolved once, after setup and before any replayed action can touch PATH.
_FIND_PYTHON = ["sh", "-c", "command -v python3 || command -v python"]
# Ruling P25. A phase the task leaves unbounded counts an hour (the miner's
# own UNSET_PHASE_SECONDS); setup gets a margin over the task's setup
# timeout for provisioning and the harness footprint (pip install uv, uv sync);
# the trajectory twice the miner's agent + finalize budget, so only a
# trajectory far past anything an honest miner ran is its timeout.
UNSET_PHASE_SECONDS = 3600.0
SETUP_MARGIN_SECONDS = 600.0
TRAJECTORY_BUDGET_FACTOR = 2.0


def _phase(timeout, name: str) -> float:
    value = getattr(timeout, name, None) if timeout is not None else None
    return float(value) if value else UNSET_PHASE_SECONDS


def replay_deadlines(task) -> tuple[float, float]:
    """``(setup deadline, trajectory budget)`` in seconds from the task's own
    timeouts: setup + margin, and 2 x (agent + finalize)."""
    timeout = getattr(task.data, "timeout", None)
    return (_phase(timeout, "setup") + SETUP_MARGIN_SECONDS,
            TRAJECTORY_BUDGET_FACTOR * (_phase(timeout, "agent") + _phase(timeout, "finalize")))


# Distinct from any role container's name (e.g. "reliquary-grade-executor"),
# so the orphan sweep can never remove the executor itself.
BOX_NAME_PREFIX = "reliquary-gradebox-"


@dataclass(frozen=True)
class BoxLimits:
    """What one hostile box may use. Never unlimited: an executor running
    ``concurrency`` boxes needs ``concurrency * memory_gb`` of host memory."""

    cpu: float = 2.0
    memory_gb: float = 6.0
    pids: int = 1024
    # The box's writable layer (ruling P24): Docker's daemon default
    # ``overlay2.size`` on xfs with project quotas, checked in each box (``df``)
    # before any action. None: unchecked (tests; ``--allow-non-xfs`` only).
    disk_gb: float | None = None

    def __post_init__(self) -> None:
        if self.disk_gb is not None and not (math.isfinite(self.disk_gb) and self.disk_gb > 0):
            raise ValueError(f"disk_gb must be positive, got {self.disk_gb}")
        if not (math.isfinite(self.cpu) and self.cpu > 0):
            raise ValueError(f"cpu must be positive, got {self.cpu}")
        if not (math.isfinite(self.memory_gb) and self.memory_gb > 0):
            raise ValueError(f"memory_gb must be positive, got {self.memory_gb}")
        if isinstance(self.pids, bool) or not isinstance(self.pids, int) or self.pids < 1:
            raise ValueError(f"pids must be a positive integer, got {self.pids}")

    @property
    def memory_bytes(self) -> int:
        return int(self.memory_gb * 2 ** 30)


DEFAULT_BOX_LIMITS = BoxLimits()


async def _docker(*args: str) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await process.communicate()
    return process.returncode, out.decode(errors="replace")


@contextlib.asynccontextmanager
async def bounded_box(task, limits: BoxLimits = DEFAULT_BOX_LIMITS) -> AsyncIterator:
    """A fresh box from the task's pinned image, network cut, under ``limits``."""
    import verifiers.v1 as vf
    from verifiers.v1.runtimes import provision_runtime

    config = vf.DockerConfig(image=task.data.image, workdir=task.data.workdir,
                             allow=task.data.network_allow, cpu=limits.cpu,
                             memory=limits.memory_gb)
    name = f"{BOX_NAME_PREFIX}{secrets.token_hex(8)}"
    async with provision_runtime(config, name=name, env=task.runtime_env()) as box:
        # verifiers' docker run takes no pids limit; set it (and no swap)
        # before anything untrusted runs, or refuse the box.
        memory = str(limits.memory_bytes)
        code, out = await _docker("update", "--pids-limit", str(limits.pids), "--memory", memory,
                                  "--memory-swap", memory, name)
        if code != 0:
            raise RuntimeError(f"could not limit box {name}: {out.strip()[:300]}")
        if limits.disk_gb is not None:
            await check_box_disk(box, limits.disk_gb)
        yield box


# A quota reads a little over its nominal size on some kernels.
DISK_TOLERANCE = 1.02


def root_size_bytes(df_output: str) -> int | None:
    """The size of ``/`` from ``df -Pk /`` (POSIX output, GNU and busybox)."""
    lines = [line.split() for line in df_output.strip().splitlines()]
    if len(lines) < 2 or len(lines[-1]) < 6 or not lines[-1][1].isdigit():
        return None
    return int(lines[-1][1]) * 1024


async def check_box_disk(box, disk_gb: float) -> None:
    """Refuse a box whose root filesystem is not bounded by ``disk_gb`` (the
    daemon's ``overlay2.size``); run before anything untrusted, so the
    image's own ``df`` still answers truthfully."""
    result = await box.run(["df", "-Pk", "/"], {})
    size = root_size_bytes(result.stdout or "")
    limit = disk_gb * 2 ** 30
    if size is None or size > limit * DISK_TOLERANCE:
        raise RuntimeError(
            f"box disk is not bounded to {disk_gb} GB (its / reads {size} bytes): set the Docker "
            f"daemon's storage-opts overlay2.size on xfs with pquota (ruling P24)")


async def capped_exec(argv: Sequence[str], *, max_bytes: int) -> tuple[int, bytes, bytes, bool]:
    """Run ``argv`` on the host, reading at most ``max_bytes`` from each of
    stdout and stderr; past that the process is killed. Returns
    ``(exit code, stdout, stderr, truncated)``."""
    process = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    truncated = False

    def stop() -> None:
        nonlocal truncated
        truncated = True
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()

    async def read(stream) -> bytes:
        data = bytearray()
        while True:
            chunk = await stream.read(_READ_CHUNK)
            if not chunk:
                return bytes(data)
            room = max_bytes - len(data)
            data += chunk[:room]
            if len(chunk) > room:
                stop()
                return bytes(data)

    try:
        out, err = await asyncio.gather(read(process.stdout), read(process.stderr))
        code = await process.wait()
    except BaseException:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        with contextlib.suppress(Exception):
            await process.wait()
        raise
    return code, out, err, truncated


def docker_exec_argv(box, argv: Sequence[str], env: dict) -> list[str] | None:
    """The ``docker exec`` verifiers b2e4e81's ``DockerRuntime.run`` runs for
    ``argv`` (same env, proxy env once cut, workdir, container), or None for
    any other runtime. A parity test pins it to verifiers' own."""
    try:
        from verifiers.v1.runtimes.docker import DockerRuntime
    except ImportError:
        return None
    if not isinstance(box, DockerRuntime) or not box._container:
        return None
    merged = {**box.process_env(env), **(box._proxy_env() if box._cut else {})}
    env_args = [arg for k, v in merged.items() for arg in ("--env", f"{k}={v}")]
    return ["docker", "exec", *env_args, "--workdir", box.config.workdir, box._container, *argv]


async def bounded_run(box, argv: Sequence[str], env: dict, *, max_bytes: int, fallback=None):
    """``box.run`` with stdout and stderr each read up to ``max_bytes``
    (``fallback``, default ``box.run``, for a runtime other than Docker's)."""
    command = docker_exec_argv(box, argv, env)
    if command is None:
        return await (fallback or box.run)(list(argv), env)
    code, out, err, truncated = await capped_exec(command, max_bytes=max_bytes)
    if truncated:
        logger.warning("box output past %d bytes was cut (%s)", max_bytes, list(argv)[:1])
    from verifiers.v1.runtimes.base import ProgramResult

    return ProgramResult(exit_code=code, stdout=out.decode(errors="replace"),
                         stderr=err.decode(errors="replace"))


@contextlib.contextmanager
def output_bounded(box, max_bytes: int, *, on_run=None):
    """While open, ``box.run`` reads at most ``max_bytes`` per stream: for what
    untrusted code ran in (a replayed repo's finalize, a patched repo's tests).
    The box object stays the same one (reliquary-swe's finalize finds its
    setup state by ``id(runtime)``); ``on_run(argv)`` sees each call first."""
    original = box.run
    own = "run" in vars(box)

    async def run(argv, env):
        if on_run is not None:
            on_run(argv)
        return await bounded_run(box, argv, env, max_bytes=max_bytes, fallback=original)

    box.run = run
    try:
        yield box
    finally:
        if own:
            box.run = original
        else:
            del box.run


def sweep_orphan_boxes() -> int:
    """Remove every box of ours on this Docker host; the count removed. Run at
    executor start, so one executor per Docker host. Never raises: a sweep that
    fails is logged and the executor starts anyway (its boxes are bounded)."""
    try:
        listed = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=^/?{BOX_NAME_PREFIX}"],
                                capture_output=True, text=True, timeout=60, check=False)
        if listed.returncode != 0:
            logger.error("orphan box sweep: docker ps failed (%d): %s",
                         listed.returncode, (listed.stderr or "").strip()[:500])
            return 0
        ids = listed.stdout.split()
        if not ids:
            return 0
        removed = subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, text=True,
                                 timeout=300, check=False)
        if removed.returncode != 0:
            logger.error("orphan box sweep: docker rm of %d boxes failed (%d): %s", len(ids),
                         removed.returncode, (removed.stderr or "").strip()[:500])
            return 0
        return len(ids)
    except (OSError, subprocess.SubprocessError):
        logger.exception("orphan box sweep failed; starting without it")
        return 0


class ReplayTimeout(Exception):
    """The replay exceeded a wall-clock deadline; the executor reports it as
    such, not as a mismatch. ``trajectory_caused``: the trajectory budget ran
    out while the recorded actions (or the finalize after them) ran, so the
    trajectory's own commands spent it (rulings P23, P25); otherwise the setup
    deadline passed, which is the executor's."""

    def __init__(self, message: str, *, trajectory_caused: bool = False) -> None:
        super().__init__(message)
        self.trajectory_caused = trajectory_caused


class BoxLost(Exception):
    """The box failed while or after running the recorded actions: it died,
    an exec into it failed, or finalize could not read a diff from the repo
    they left (ruling P23). The trajectory's outcome, never the executor's:
    the same actions do it on every executor. Failures before the first
    recorded action (provisioning, setup, the Docker daemon) are not this."""


async def resolve_python(box) -> str:
    result = await box.run(list(_FIND_PYTHON), {})
    path = (result.stdout or "").strip().splitlines()
    if not path or not path[0].startswith("/"):
        raise RuntimeError(f"no python interpreter in the replay box: {result.stdout!r}")
    return path[0]


def _timeout_arg(command_timeout: float) -> int | float:
    # Integral timeouts go as ints so a timed-out command reproduces the
    # harness text "... timed out after 3600 seconds", not "3600.0".
    return int(command_timeout) if float(command_timeout).is_integer() else command_timeout


async def run_action(box, python: str, action: Action, command_timeout: float) -> str:
    """One recorded tool call, with the program passed afresh by argv. The
    output is cut one char past the longest observation a lease carries
    (ruling P24): longer cannot match a recorded one anyway."""
    keep = MAX_OBSERVATION_CHARS + 1
    request_path = f"/tmp/.replay_request_{secrets.token_hex(16)}.json"
    request = json.dumps({"tool": action.tool, "arguments": action.arguments,
                          "timeout": _timeout_arg(command_timeout), "max_chars": keep})
    await box.write(request_path, request.encode())
    # The program truncates; the read is bounded as well, since the command
    # can write to the exec's own stdout behind the program's back.
    result = await bounded_run(box, [python, "-I", "-c", TOOL_PROGRAM, request_path], {},
                               max_bytes=_UTF8 * keep)
    return (result.stdout or "")[:keep]


_TASK_LOCK = threading.Lock()


@functools.lru_cache(maxsize=256)
def _swesmith_task(instance_id: str):
    from reliquary_swe import corpus
    from reliquary_swe.taskset import task_for

    return task_for(corpus.swesmith_row(instance_id), 0, "train")


def swesmith_task(instance_id: str):
    """The task for one instance; blocking (the first call loads the corpus),
    so callers on an event loop run it in a thread. One build at a time."""
    with _TASK_LOCK:
        return _swesmith_task(instance_id)


def _trace(task):
    # The shape reliquary-swe's own tests use: setup/finalize only read trace.info.
    import verifiers.v1 as vf

    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
    )


async def prepare_harness_footprint(box) -> None:
    """What verifiers' bash harness setup leaves in the miner's box, made here
    the same way and at the same point (after task setup, before the network
    cut): ``pip install --user uv`` and ``uv sync`` of the harness program.
    Honest observations show it (``/root/.local`` on ``sys.path``,
    ``/root/.cache/pip``, ``uv`` in ``pip list``); a replay box without it
    disagrees with every one of them. The program itself is never run here."""
    from verifiers.v1.harnesses.bash.harness import PROGRAM_SOURCE

    await box.prepare_uv_script(PROGRAM_SOURCE, {})


async def replay_swe(task, actions: Sequence[Action], *,
                     command_timeout: float = 3600.0,
                     setup_deadline: float | None = None,
                     trajectory_budget: float | None = None,
                     limits: BoxLimits = DEFAULT_BOX_LIMITS) -> tuple[list[str], str]:
    """The replayed observations and the diff finalize collected.

    ``setup_deadline`` and ``trajectory_budget`` default to
    ``replay_deadlines(task)``. Raises ``ReplayTimeout`` past either and ``BoxLost`` when
    the box fails once the first recorded action started (ruling P23: the
    trajectory's outcome); anything else is the executor's. With no action
    to replay nothing the trajectory controls ever ran, so every failure is
    the executor's."""
    trace = _trace(task)
    observations: list[str] = []
    started = False                      # a recorded action was sent into the box
    finished = False                     # every action ran and finalize read the diff
    default_setup, default_budget = replay_deadlines(task)
    setup_deadline = default_setup if setup_deadline is None else setup_deadline
    trajectory_budget = default_budget if trajectory_budget is None else trajectory_budget
    loop = asyncio.get_running_loop()
    try:
        async with asyncio.timeout(setup_deadline) as clock:
            async with bounded_box(task, limits) as box:
                await box.prepare_setup()
                await task.setup(trace, box)
                await prepare_harness_footprint(box)
                await box.prepare_execution([])
                python = await resolve_python(box)
                # The trajectory's own clock starts at its first action.
                clock.reschedule(loop.time() + trajectory_budget)
                started = bool(actions)
                try:
                    for action in actions:
                        observations.append(await run_action(box, python, action, command_timeout))
                    with output_bounded(box, FINALIZE_OUTPUT_BYTES):
                        await task.finalize(trace, box)
                except Exception as e:
                    if not started:
                        raise
                    where = "finalize" if len(observations) == len(actions) else "an action"
                    raise BoxLost(f"the box failed in {where} after {len(observations)} of "
                                  f"{len(actions)} actions: {type(e).__name__}: {e}"[:400]) from e
                finished = True
    except TimeoutError as e:
        if not finished:
            spent = (f"its trajectory budget {trajectory_budget} s" if started
                     else f"its setup deadline {setup_deadline} s")
            raise ReplayTimeout(
                f"replay exceeded {spent} after {len(observations)} of {len(actions)} actions",
                trajectory_caused=started) from e
        # Only the box's removal ran late: the facts are complete.
        logger.warning("replay box removal outlived the deadline; the replay itself finished")
    except BoxLost:
        raise
    except Exception:
        if not finished:
            raise
        # The box's removal failed after the facts were in (the orphan sweep
        # removes what is left): the replay itself is complete.
        logger.exception("replay box removal failed after the replay finished")
    return observations, trace.info.get("patch", "")
