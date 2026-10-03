"""Replay a submitted SWE episode in a fresh box: the validator's side of the
agentic corpus check (spec N5). ``verifiers`` and ``reliquary_swe`` are optional
and imported only here, like ``reliquary.eval.verifiers_source``.

Each recorded tool call runs inside the box through ``TOOL_PROGRAM``, which
reproduces verifiers b2e4e81's bash harness tools (``harnesses/bash/program.py``:
``run_bash``, ``run_edit``) and its argument-error messages, so an honest
miner's observations come back identical up to normalization.
"""

from __future__ import annotations

import functools
import json
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

_RUN = (
    'PY=$(command -v python3 || command -v python); '
    'exec "$PY" /tmp/.replay_tool.py < /tmp/.replay_request.json'
)


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
                     command_timeout: float = 3600.0) -> tuple[list[str], str]:
    import verifiers.v1 as vf
    from verifiers.v1.runtimes import provision_runtime

    trace = _trace(task)
    config = vf.DockerConfig(image=task.data.image, workdir=task.data.workdir,
                             allow=task.data.network_allow)
    observations: list[str] = []
    async with provision_runtime(config, env=task.runtime_env()) as box:
        await box.prepare_setup()
        await task.setup(trace, box)
        await box.prepare_execution([])
        await box.write("/tmp/.replay_tool.py", TOOL_PROGRAM.encode())
        for action in actions:
            # Integral timeouts go as ints so a timed-out command reproduces the
            # harness text "... timed out after 3600 seconds", not "3600.0".
            timeout = int(command_timeout) if float(command_timeout).is_integer() else command_timeout
            request = json.dumps({"tool": action.tool, "arguments": action.arguments,
                                  "timeout": timeout})
            await box.write("/tmp/.replay_request.json", request.encode())
            result = await box.run(["sh", "-c", _RUN], {})
            observations.append(result.stdout)
        await task.finalize(trace, box)
    return observations, trace.info.get("patch", "")
