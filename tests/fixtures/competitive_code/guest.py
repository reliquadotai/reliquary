"""The program's side of the sandbox: one submission, one stdin, one stdout.

Standard library only, and no import from this package: the local runner ships
this file's text into a fresh interpreter, and the core's gVisor worker is
meant to call the same `run` in its warm process, so the two cannot drift.

The import gate and the reduced builtins are rules of the task, not a security
boundary. Containment is the sandbox's job (gVisor in production, a limited
subprocess locally). What this file guarantees is that the rules are the same
everywhere: the same modules, the same `sys`, the same verdict for an exit.

Contract for a host: one run per fresh process (or fork), never concurrent;
module state (e.g. monkeypatched math, daemon threads) is not isolated between
runs in one process. Determinism: the module-level `random` is seeded with
RANDOM_SEED before every run, and the host must pin PYTHONHASHSEED=0 (the local
runner does); a program that reseeds from entropy (`random.seed()`,
`random.Random()`, a seed taken from `time`) stays non-deterministic.
"""

from __future__ import annotations

import builtins
import io
import random
import sys
import threading
import time
import types

ALLOWED_IMPORT_ROOTS = frozenset({
    "__future__", "abc", "array", "bisect", "cmath", "collections", "copy",
    "dataclasses", "datetime", "decimal", "enum", "fractions", "functools",
    "heapq", "itertools", "math", "operator", "queue", "random", "re",
    "statistics", "string", "sys", "threading", "time", "typing",
})
DENIED_BUILTINS = frozenset({
    "breakpoint", "compile", "dir", "eval", "exec", "globals", "help",
    "locals", "open", "vars",
})
GATE_MESSAGE = "is not available in the grader sandbox"
# Every run starts the module-level `random` from this seed, so a randomised
# program prints the same thing on every grading of the same completion.
RANDOM_SEED = 0


class OutputLimitExceeded(BaseException):
    """Raised from a write past the cap. A BaseException, so that a
    submission's `except Exception` cannot swallow it and keep printing."""


class _CappedBytes(io.BytesIO):
    def __init__(self, cap: int) -> None:
        super().__init__()
        self._cap = cap
        self.exceeded = False

    def write(self, data) -> int:
        if self.tell() + len(data) > self._cap:
            self.exceeded = True
            raise OutputLimitExceeded()
        return super().write(data)


def _sys_shim(stdin, stdout) -> types.ModuleType:
    shim = types.ModuleType("sys")
    shim.stdin = stdin
    shim.stdout = stdout
    shim.stderr = io.StringIO()
    shim.argv = ["main.py"]
    shim.maxsize = sys.maxsize
    shim.float_info = sys.float_info
    shim.version_info = sys.version_info
    shim.exit = sys.exit
    shim.setrecursionlimit = sys.setrecursionlimit
    shim.getrecursionlimit = sys.getrecursionlimit
    shim.set_int_max_str_digits = sys.set_int_max_str_digits
    shim.get_int_max_str_digits = sys.get_int_max_str_digits
    return shim


def _safe_builtins(shim: types.ModuleType) -> dict:
    real_import = builtins.__import__

    def gated_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name).split(".", 1)[0]
        if level != 0 or root not in ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"module {name!r} {GATE_MESSAGE}")
        if root == "sys":
            return shim
        return real_import(name, globals, locals, fromlist, level)

    safe = {k: v for k, v in builtins.__dict__.items() if k not in DENIED_BUILTINS}
    safe["__import__"] = gated_import
    # `exit` and `quit` come from `site`, which the local child runs without
    # (-S). Defined here so a script ending in `exit()` behaves the same in
    # every host.
    safe["exit"] = safe["quit"] = sys.exit
    return safe


def _join_new_threads(before: set) -> None:
    """Join the non-daemon threads the submission started, and only those."""
    for thread in threading.enumerate():
        if thread not in before and not thread.daemon:
            thread.join()


def run(code: str, stdin_text: str, output_cap: int) -> dict:
    """Run `code` as a script on `stdin_text`; return status, stdout, CPU time.

    `cpu_seconds` is `time.process_time()`, the whole process's CPU: exact in a
    fresh subprocess only. An in-process host must account CPU itself.
    """
    raw_out = _CappedBytes(output_cap)
    stdout = io.TextIOWrapper(raw_out, encoding="utf-8", write_through=True)
    stdin = io.TextIOWrapper(io.BytesIO(stdin_text.encode("utf-8")), encoding="utf-8")
    shim = _sys_shim(stdin, stdout)
    namespace = {"__name__": "__main__", "__builtins__": _safe_builtins(shim)}
    saved_stdin, saved_stdout = sys.stdin, sys.stdout
    saved_limit = sys.getrecursionlimit()
    saved_digits = sys.get_int_max_str_digits()
    saved_stack = threading.stack_size()
    saved_random = random.getstate()
    threads_before = set(threading.enumerate())
    sys.stdin, sys.stdout = stdin, stdout
    # Competitive answers routinely print integers beyond the default 4300
    # digits; the original judges have no such limit.
    sys.set_int_max_str_digits(0)
    random.seed(RANDOM_SEED)
    status = "ok"
    start = time.process_time()
    try:
        try:
            exec(compile(code, "<submission>", "exec"), namespace)
        except OutputLimitExceeded:
            status = "output_limit"
        except SystemExit as stop:
            if stop.code not in (None, 0):
                status = "runtime_error"
        except ImportError as error:
            status = "forbidden_import" if GATE_MESSAGE in str(error) else "runtime_error"
        except BaseException:
            status = "runtime_error"
        _join_new_threads(threads_before)
    finally:
        cpu_seconds = time.process_time() - start
        try:
            stdout.flush()
        except OutputLimitExceeded:
            pass
        if raw_out.exceeded:
            status = "output_limit"
        sys.stdin, sys.stdout = saved_stdin, saved_stdout
        sys.setrecursionlimit(saved_limit)
        sys.set_int_max_str_digits(saved_digits)
        threading.stack_size(saved_stack)
        random.setstate(saved_random)
    text = raw_out.getvalue().decode("utf-8", errors="replace")
    return {
        "status": status,
        "stdout": text if status == "ok" else "",
        "cpu_seconds": cpu_seconds,
    }
