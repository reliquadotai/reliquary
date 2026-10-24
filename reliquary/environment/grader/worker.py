"""Sandboxed code worker.

The worker runs untrusted miner code inside gVisor. It does not receive
hidden assertions or expected values. Each request contains only the code,
an entrypoint, and public call arguments; the trusted grader server compares
the returned primitive value against the hidden expected value.

A ``"mode": "stdio"`` request runs a whole program on one stdin instead, in a
child forked for that request alone, through the ``run`` of the guest source
the request carries (the packaged environment's own ``judge/guest.py``, read by
the trusted side from its verified wheel). The reply is what the program
printed; the trusted side compares it.
"""

from __future__ import annotations

import ast
import base64
import builtins
import contextlib
import faulthandler
import glob
import hashlib
import inspect
import io
import json
import math
import os
import resource
import select
import signal
import sys
import time
from typing import Any


# A miner-controlled return value must not turn the worker's stdout protocol
# into a memory-amplification path on the execution host.
MAX_WORKER_OUTPUT_BYTES = 256 * 1024

# Armed before anything redirects sys.stderr, so a hard death (SIGSEGV,
# SIGABRT) still writes a Python traceback to the real fd 2, which the server
# keeps and reports. Without it such a worker dies silently and the crash is
# unattributable.
faulthandler.enable()


_CRITICAL_BUILTINS = {
    name: getattr(builtins, name)
    for name in ("__import__", "compile", "eval", "exec", "open", "input")
}

_ALLOWED_IMPORT_ROOTS = {
    "abc", "array", "bisect", "collections", "copy", "dataclasses", "decimal",
    "enum", "functools", "heapq", "itertools", "math", "operator", "re",
    "statistics", "string", "typing",
}

_DENIED_BUILTINS = {
    "breakpoint", "compile", "dir", "eval", "exec", "globals", "help", "input",
    "locals", "open", "vars",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".", 1)[0]
    if level != 0 or root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"module {name!r} is not available in the grader sandbox")
    return _CRITICAL_BUILTINS["__import__"](name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    safe = {
        name: value
        for name, value in builtins.__dict__.items()
        if name not in _DENIED_BUILTINS
    }
    safe["__import__"] = _safe_import
    return safe


def _critical_builtins_intact() -> bool:
    return all(
        getattr(builtins, name) is original
        for name, original in _CRITICAL_BUILTINS.items()
    )


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe primitive, or raise TypeError.

    This intentionally rejects arbitrary objects so custom ``__eq__`` /
    comparator tricks never reach trusted scoring.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError("dict key is not a string")
            out[k] = _json_safe(v)
        return out
    raise TypeError(f"unsupported output type: {type(value).__name__}")


def _user_defined_names(code: str) -> set[str]:
    """Top-level def/class names in the submitted source.

    Used to resolve the entry point by structure when the requested name is
    absent — and to never select an imported callable as the entry point.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _accepts_arity(fn: Any, nargs: int) -> bool:
    """True if *fn* can be called with *nargs* positional arguments."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return True
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    required = sum(1 for p in positional if p.default is p.empty)
    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
    upper = float("inf") if has_varargs else len(positional)
    return required <= nargs <= upper


def _defined_functions_in_order(code: str) -> list[str]:
    """Top-level function names, in source order."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _call_graph_roots(code: str, fn_names: set[str]) -> set[str]:
    """Function names not called from inside a *different* top-level function.

    These are the call-graph roots — the entry point of a "main + helpers"
    solution. Self-recursion does not disqualify a root.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(fn_names)
    called_by_others: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in fn_names
                and sub.func.id != node.name
            ):
                called_by_others.add(sub.func.id)
    return set(fn_names) - called_by_others


def _returns_a_value(code: str, name: str) -> bool:
    """True if top-level function *name* has a ``return <expr>`` (not bare/None).

    A print-only or None-returning function can never match a return-value case,
    so it is never the graded entry. Unparseable code is treated as returning a
    value so analysis failure never excludes a real solution.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            visitor = _ValueReturnVisitor()
            for stmt in node.body:
                visitor.visit(stmt)
            return visitor.has_value_return
    return True


class _ValueReturnVisitor(ast.NodeVisitor):
    """Find value returns without descending into nested definitions."""

    def __init__(self) -> None:
        self.has_value_return = False

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and not (
            isinstance(node.value, ast.Constant) and node.value.value is None
        ):
            self.has_value_return = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None


def _resolve_function(ns: dict[str, Any], code: str, nargs: int) -> Any | None:
    """Resolve the entry function when the requested name is absent.

    The prompt asks for a behavior, not a name, so pick a single entry
    deterministically: the only arity match; else the only call-graph root
    (a function no *other* top-level function calls); else the last-defined
    arity match. Exactly one function is then run against the hidden cases —
    never several with "accept any pass" — so a wrong pick simply fails them.
    """
    order = [
        name for name in _defined_functions_in_order(code)
        if callable(ns.get(name)) and not isinstance(ns.get(name), type)
    ]
    candidates = [name for name in order if _accepts_arity(ns[name], nargs)]
    if not candidates:
        return None
    # Drop print-only / None-returning helpers when a value-returning one exists:
    # they can never match a return-value case, so they are never the entry.
    valued = [name for name in candidates if _returns_a_value(code, name)]
    if valued:
        candidates = valued
    if len(candidates) == 1:
        return ns[candidates[0]]
    roots = _call_graph_roots(code, set(order))
    root_candidates = [name for name in candidates if name in roots]
    if len(root_candidates) == 1:
        return ns[root_candidates[0]]
    pool = root_candidates or candidates
    return ns[pool[-1]]


def _resolve_class(ns: dict[str, Any], defined: set[str]) -> Any | None:
    """The submitted code's sole class, or None if ambiguous."""
    classes = [ns[name] for name in defined if isinstance(ns.get(name), type)]
    return classes[0] if len(classes) == 1 else None


def evaluate_call(
    code: str,
    entry: dict[str, Any],
    args: list[Any],
    kwargs: dict[str, Any],
    timeout_s: float,
) -> tuple[Any | None, str]:
    """Execute miner code and call the requested entrypoint.

    Returns ``(output, status)``. The server enforces wall-clock timeouts;
    ``timeout_s`` is accepted for protocol symmetry.
    """
    del timeout_s
    if not code or not code.strip():
        return None, "runtime_error"
    if not isinstance(entry, dict):
        return None, "bad_entry"
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        return None, "bad_request"

    ns: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "__name__": "<miner_code>",
    }
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            exec(compile(code, "<miner_code>", "exec"), ns)
        except ImportError as e:
            if "not available in the grader sandbox" in str(e):
                return None, "forbidden_import"
            return None, "runtime_error"
        except BaseException:
            return None, "runtime_error"
        if not _critical_builtins_intact():
            return None, "tampered"

        try:
            kind = entry.get("kind")
            if kind == "function":
                # Prompt specifies a behavior, not a name: accept the requested
                # name, else resolve the sole/only-arity-matching defined function.
                fn = ns.get(entry["name"])
                if not callable(fn):
                    fn = _resolve_function(ns, code, len(args))
                if not callable(fn):
                    return None, "runtime_error"
            elif kind == "method":
                cls = ns.get(entry["class_name"])
                if not isinstance(cls, type):
                    cls = _resolve_class(ns, _user_defined_names(code))
                if cls is None:
                    return None, "runtime_error"
                fn = getattr(cls(), entry["method"])
            else:
                return None, "bad_entry"
            output = fn(*args, **kwargs)
            if not _critical_builtins_intact():
                return None, "tampered"
            safe_output = _json_safe(output)
            encoded_size = len(
                json.dumps(
                    safe_output,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            if encoded_size > MAX_WORKER_OUTPUT_BYTES:
                return None, "bad_output"
            return safe_output, "ok"
        except ImportError as e:
            if "not available in the grader sandbox" in str(e):
                return None, "forbidden_import"
            return None, "runtime_error"
        except TypeError as e:
            if "unsupported output type" in str(e) or "dict key" in str(e) or "non-finite" in str(e):
                return None, "bad_output"
            return None, "runtime_error"
        except BaseException:
            return None, "runtime_error"


# Stdio mode. The wall clock sits above the CPU limit: a program is judged on
# its CPU, and the wall clock only catches one that idles (sleep, deadlock).
STDIO_WALL_FACTOR = 2.0
STDIO_WALL_SLACK_S = 1.0
# What a program may address, as the package's own runner grants it. The
# sandbox's hard limit wins when it is lower.
STDIO_MEMORY_BYTES = 2 << 30
# Bytes read past the output cap before the rest is dropped: the result line
# and its JSON escaping.
STDIO_READ_SLACK = 1 << 20
STDIO_DRAIN_S = 0.5
# Raw stdout bytes per reply line: 40 KB once in base64, under the 64 KiB a
# single line may carry out of runsc (see `_reply_in_chunks`).
STDIO_CHUNK_BYTES = 30_000
# CPU seconds under which a program killed by the wall clock idled, when the
# run queue cannot be read (gVisor): measured at ~0.02 s for a sleeping one.
STDIO_IDLE_CPU_S = 0.1
_GUEST_CACHE: dict[str, dict[str, Any]] = {}


def _guest_run(source: str):
    """The guest's ``run``, compiled once per distinct source."""
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    namespace = _GUEST_CACHE.get(digest)
    if namespace is None:
        namespace = {"__name__": "reliquary_stdio_guest"}
        exec(compile(source, "<guest>", "exec"), namespace)
        if not callable(namespace.get("run")):
            raise ValueError("guest source defines no run()")
        _GUEST_CACHE.clear()
        _GUEST_CACHE[digest] = namespace
    return namespace["run"]


def _lower_limit(which: int, value: int) -> None:
    _, hard = resource.getrlimit(which)
    if hard != resource.RLIM_INFINITY:
        value = min(value, hard)
    resource.setrlimit(which, (value, value))


def _run_queue_wait_s(pid: int) -> float | None:
    """Seconds every thread of ``pid`` waited runnable but not running, or
    None when nothing could be read (as in the package's runner)."""
    total, read = 0, 0
    for path in glob.glob(f"/proc/{pid}/task/*/schedstat"):
        try:
            with open(path, "rb") as handle:
                total += int(handle.read().split()[1])
            read += 1
        except (OSError, ValueError, IndexError):
            continue
    return total / 1e9 if read else None


def _child(run, write_fd: int, code: str, stdin_text: str, output_cap: int,
           time_limit_s: float) -> None:
    """In the fork: no fd but the result pipe, the CPU limit, then the guest."""
    try:
        os.setsid()
        null = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(null, fd)
        os.closerange(3, write_fd)
        os.closerange(write_fd + 1, 1 << 16)
        cpu = math.ceil(time_limit_s) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        _lower_limit(resource.RLIMIT_AS, STDIO_MEMORY_BYTES)
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        result = run(code, stdin_text, output_cap)
        payload = ("\n" + json.dumps(result) + "\n").encode("utf-8")
        view = memoryview(payload)
        while view:
            view = view[os.write(write_fd, view):]
        os._exit(0)
    except BaseException:
        os._exit(1)


def run_stdio(
    guest_source: str,
    code: str,
    stdin_text: str,
    output_cap: int,
    time_limit_s: float,
) -> dict[str, Any]:
    """Run ``code`` on ``stdin_text`` in a fresh fork; status, stdout, CPU.

    The statuses and their order are the package runner's: CPU time and exit
    status come from the kernel (``wait4``), never from the child, which shares
    the result pipe with the submission and could forge it.
    """
    if sys.flags.hash_randomization:
        # Set and dict orders of strings would differ between replays.
        return {"status": "grader_error", "stdout": "", "cpu_seconds": 0.0}
    run = _guest_run(guest_source)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        _child(run, write_fd, code, stdin_text, output_cap, time_limit_s)
    os.close(write_fd)
    chunks: list[bytes] = []
    kept, keep = 0, output_cap + STDIO_READ_SLACK
    deadline = time.monotonic() + STDIO_WALL_FACTOR * time_limit_s + STDIO_WALL_SLACK_S
    wall_fired, run_queue_wait, eof = False, None, False
    try:
        while True:
            reaped, status, usage = os.wait4(pid, os.WNOHANG)
            if reaped:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Read before the kill, while the pid is still this child's.
                run_queue_wait = _run_queue_wait_s(pid)
                wall_fired = True
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(pid, signal.SIGKILL)
                _, status, usage = os.wait4(pid, 0)
                break
            if eof:
                time.sleep(min(remaining, 0.005))
                continue
            ready, _, _ = select.select([read_fd], [], [], min(remaining, 0.05))
            if ready:
                block = os.read(read_fd, 65536)
                if not block:
                    eof = True
                elif kept < keep:
                    chunks.append(block)
                    kept += len(block)
        # Whatever the submission left in its session dies with it.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
        drain_until = time.monotonic() + STDIO_DRAIN_S
        while not eof and time.monotonic() < drain_until:
            ready, _, _ = select.select([read_fd], [], [], STDIO_DRAIN_S)
            if not ready:
                break
            block = os.read(read_fd, 65536)
            if not block:
                break
            if kept < keep:
                chunks.append(block)
                kept += len(block)
    finally:
        os.close(read_fd)
    exit_code = os.waitstatus_to_exitcode(status)
    cpu = usage.ru_utime + usage.ru_stime

    def verdict(name: str, stdout: str = "") -> dict[str, Any]:
        return {"status": name, "stdout": stdout, "cpu_seconds": cpu}

    if exit_code == -signal.SIGXCPU or cpu > time_limit_s:
        return verdict("timeout")
    if wall_fired and exit_code != 0:
        # Runnable longer than its limit but not run: the host starved it, no
        # verdict. Otherwise it idled (sleep, deadlock): the program's failure.
        if run_queue_wait is not None:
            starved = cpu + run_queue_wait > time_limit_s
        else:
            # gVisor has no schedstat. Idle is then a program that used almost
            # no CPU in a whole wall window, which a starved one would only do
            # on a host oversubscribed many times over.
            starved = cpu > STDIO_IDLE_CPU_S
        return verdict("harness_overload" if starved else "timeout")
    if exit_code == -signal.SIGKILL:
        return verdict("timeout")
    if exit_code != 0:
        return verdict("runtime_error")
    try:
        line = b"".join(chunks).decode("utf-8", errors="replace").rstrip().rsplit("\n", 1)[-1]
        result = json.loads(line)
        name, text = str(result["status"]), str(result["stdout"])
    except (ValueError, KeyError, TypeError, IndexError):
        return verdict("runtime_error")
    if name not in {"ok", "output_limit", "runtime_error", "forbidden_import"}:
        return verdict("runtime_error")
    return verdict(name, text if name == "ok" else "")


def _reply_in_chunks(resp: dict[str, Any]) -> None:
    """A header line, then the stdout in base64 lines, each sent only when the
    server asks for it.

    Measured under runsc (release-20260928.0): a line past 64 KiB written to
    the donated stdout pipe stalls after the first 64 KiB and never arrives.
    Each chunk is under that and is written into a pipe the server has
    emptied, so no write ever has to wait.
    """
    data = str(resp.pop("stdout", "")).encode("utf-8")
    pieces = [data[i:i + STDIO_CHUNK_BYTES] for i in range(0, len(data), STDIO_CHUNK_BYTES)]
    resp["stdout_chunks"] = len(pieces)
    sys.__stdout__.write(json.dumps(resp) + "\n")
    sys.__stdout__.flush()
    for piece in pieces:
        if sys.stdin.readline().strip() != "next":
            return  # the server gave up on this reply
        sys.__stdout__.write(base64.b64encode(piece).decode("ascii") + "\n")
        sys.__stdout__.flush()


def _stdio_request(req: dict[str, Any]) -> dict[str, Any]:
    guest, code, stdin_text = req.get("guest"), req.get("code"), req.get("stdin")
    output_cap, time_limit_s = req.get("output_cap"), req.get("time_limit_s")
    if (
        not isinstance(guest, str) or not guest
        or not isinstance(code, str)
        or not isinstance(stdin_text, str)
        or not isinstance(output_cap, int) or isinstance(output_cap, bool)
        or output_cap <= 0
        or not isinstance(time_limit_s, (int, float)) or isinstance(time_limit_s, bool)
        or not math.isfinite(time_limit_s) or time_limit_s <= 0
    ):
        return {"status": "bad_request", "stdout": "", "cpu_seconds": 0.0}
    return run_stdio(guest, code, stdin_text, output_cap, float(time_limit_s))


def _serve_stdin() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("mode") == "stdio":
                _reply_in_chunks({"req_id": req.get("req_id", ""), **_stdio_request(req)})
                continue
            output, status = evaluate_call(
                req.get("code", ""),
                req.get("entry", {}),
                req.get("args", []),
                req.get("kwargs", {}),
                float(req.get("timeout_s", 5.0)),
            )
            resp = {
                "req_id": req.get("req_id", ""),
                "output": output,
                "status": status,
            }
        except BaseException as e:
            resp = {
                "req_id": "",
                "output": None,
                "status": "crash",
                "error": str(e),
            }
        sys.__stdout__.write(json.dumps(resp) + "\n")
        sys.__stdout__.flush()


if __name__ == "__main__":
    _serve_stdin()
