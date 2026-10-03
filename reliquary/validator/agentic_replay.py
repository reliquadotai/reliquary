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
a fresh file with an unpredictable name, deleted once read. The whole replay
runs under a wall-clock deadline (``episode_deadline``, default 3600 s) and
raises ``ReplayTimeout`` past it, besides each command's own timeout.

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
import functools
import json
import secrets
from collections.abc import Sequence

from reliquary.corpus.replay_compare import Action

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
sys.stdout.write(out)
'''

# Resolved once, after setup and before any replayed action can touch PATH.
_FIND_PYTHON = ["sh", "-c", "command -v python3 || command -v python"]
DEFAULT_EPISODE_DEADLINE = 3600.0


class ReplayTimeout(Exception):
    """The replay exceeded its wall-clock ``episode_deadline``; the executor
    reports it as such, not as a mismatch."""


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
    """One recorded tool call, with the program passed afresh by argv."""
    request_path = f"/tmp/.replay_request_{secrets.token_hex(16)}.json"
    request = json.dumps({"tool": action.tool, "arguments": action.arguments,
                          "timeout": _timeout_arg(command_timeout)})
    await box.write(request_path, request.encode())
    result = await box.run([python, "-I", "-c", TOOL_PROGRAM, request_path], {})
    return result.stdout


@functools.cache
def swesmith_task(instance_id: str):
    from reliquary_swe import corpus
    from reliquary_swe.taskset import task_for

    return task_for(corpus.swesmith_row(instance_id), 0, "train")


def _trace(task):
    # The shape reliquary-swe's own tests use: setup/finalize only read trace.info.
    import verifiers.v1 as vf

    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
    )


async def replay_swe(task, actions: Sequence[Action], *,
                     command_timeout: float = 3600.0,
                     episode_deadline: float = DEFAULT_EPISODE_DEADLINE) -> tuple[list[str], str]:
    import verifiers.v1 as vf
    from verifiers.v1.runtimes import provision_runtime

    trace = _trace(task)
    config = vf.DockerConfig(image=task.data.image, workdir=task.data.workdir,
                             allow=task.data.network_allow)
    observations: list[str] = []
    try:
        async with asyncio.timeout(episode_deadline):
            async with provision_runtime(config, env=task.runtime_env()) as box:
                await box.prepare_setup()
                await task.setup(trace, box)
                await box.prepare_execution([])
                python = await resolve_python(box)
                for action in actions:
                    observations.append(await run_action(box, python, action, command_timeout))
                await task.finalize(trace, box)
    except TimeoutError as e:
        raise ReplayTimeout(
            f"replay exceeded {episode_deadline} s after {len(observations)} of {len(actions)} actions"
        ) from e
    return observations, trace.info.get("patch", "")
