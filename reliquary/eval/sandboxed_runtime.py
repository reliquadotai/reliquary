"""A Verifiers runtime whose code runs in our gVisor grading service.

A taskset reward that needs a runtime (LiveCodeBench, SciCode) would otherwise get a
Docker box, so the admin host would need the Docker socket. Here the
taskset's own verifier script (pinned by sha256) runs on the trusted side and
only the processes it starts for the model's program go to the grading
service's stdio mode (``runsc``, no network). Expected outputs and the
comparison never leave this side. Since the script now runs outside a box,
its ``eval`` of test inputs and ``pickle.loads`` of hidden tests are held to
literals and plain strings (all of LCB's are).

An adapter exists only for a verifier whose bytes are pinned below; any other
runtime reward keeps the Docker path in ``verifiers_source``. SciCode's tests
need numpy/scipy/h5py and the reference targets: its programs go to a second
service whose sandbox root is the SciCode grading image (docker/scicode-grader).
"""

from __future__ import annotations

import ast
import hashlib
import io
import os
import pickle
import subprocess
import threading
import types
from dataclasses import dataclass
from typing import Any

# The grading service's socket; an eval host may share it from a sidecar.
SOCKET_ENV = "RELIQUARY_EVAL_GRADER_SOCKET"
# The service whose sandbox root is the SciCode grading image.
SCICODE_SOCKET_ENV = "RELIQUARY_EVAL_SCICODE_GRADER_SOCKET"
# What the sandboxed program's interpreter is called on the trusted side; the
# guest substitutes its own `sys.executable`.
SANDBOX_PYTHON = "/usr/local/bin/python3"
# The largest stdout a test may print (the service's own ceiling).
OUTPUT_CAP = 64 * 1024 * 1024
# CPU budget the service grants on top of the verifier's wall timeout, so its
# own CPU verdict never preempts the guest's wall clock.
CPU_FACTOR = 2.0

# livecodebench (research-environments @ f2e0d709): verify.py.
LIVECODEBENCH_VERIFY_SHA256 = "88da02df6d8e24f84038b657566d61c75b9da7032b8e092edc4b9c24a75408da"
# scicode-steps (tasksets/scicode_steps): verify.py.
SCICODE_VERIFY_SHA256 = "a1318e2d79f4b2e8d4387a3bec13dbe15db5f7a5d915ef424abb47bc3b06d9b0"

# Runs in the service's per-request fork. As `subprocess.run([sys.executable,
# "-c", program], input=..., timeout=TIMEOUT)` in the verifier: a real
# interpreter, the whole standard library, a wall clock. Statuses: "ok" (exit
# 0), else "runtime_error" (exit != 0, timeout, undecodable output). The
# program comes framed ahead of its stdin (`frame`): a functional test embeds
# its inputs in the program, past the service's code size limit.
GUEST_TEMPLATE = '''\
import subprocess, sys

TIMEOUT = {timeout!r}


def run(code, stdin_text, output_cap):
    size, rest = stdin_text.split("\\n", 1)
    code, stdin_text = rest[:int(size)], rest[int(size):]
    try:
        proc = subprocess.run([sys.executable, "-c", code], input=stdin_text.encode("utf-8"),
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {{"status": "runtime_error", "stdout": ""}}
    if proc.returncode != 0:
        return {{"status": "runtime_error", "stdout": ""}}
    if len(proc.stdout) > output_cap:
        return {{"status": "output_limit", "stdout": ""}}
    try:
        return {{"status": "ok", "stdout": proc.stdout.decode("utf-8")}}
    except UnicodeDecodeError:
        return {{"status": "runtime_error", "stdout": ""}}
'''


# SciCode's guest: the test file passes when it exits 0 (its output is not
# read). One BLAS thread, so the address-space limit holds numpy and scipy.
SCICODE_GUEST_TEMPLATE = '''\
import subprocess, sys

TIMEOUT = {timeout!r}
ENV = {{"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "MPLBACKEND": "Agg",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}}


def run(code, stdin_text, output_cap):
    size, rest = stdin_text.split("\\n", 1)
    code, stdin_text = rest[:int(size)], rest[int(size):]
    try:
        proc = subprocess.run([sys.executable, "-c", code], input=stdin_text.encode("utf-8"),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=TIMEOUT, env=ENV, cwd="/tmp")
    except subprocess.TimeoutExpired:
        return {{"status": "runtime_error", "stdout": ""}}
    return {{"status": "ok" if proc.returncode == 0 else "runtime_error", "stdout": ""}}
'''


@dataclass(frozen=True)
class Adapter:
    verifiers: frozenset[str]
    socket_env: str
    guest_template: str


ADAPTED = {
    "livecodebench.taskset": Adapter(frozenset({LIVECODEBENCH_VERIFY_SHA256}), SOCKET_ENV, GUEST_TEMPLATE),
    "scicode_steps.taskset": Adapter(frozenset({SCICODE_VERIFY_SHA256}), SCICODE_SOCKET_ENV,
                                     SCICODE_GUEST_TEMPLATE),
}


def guest_source(timeout: float, template: str = GUEST_TEMPLATE) -> str:
    return template.format(timeout=float(timeout))


def frame(program: str, stdin: str) -> str:
    return f"{len(program)}\n{program}{stdin}"


def _client(socket_env: str = SOCKET_ENV):
    from reliquary.environment.grader_client import GraderClient

    path = os.environ.get(socket_env, "").strip()
    if socket_env != SOCKET_ENV and not path:
        # Never the default service: its sandbox lacks this taskset's runtime.
        from reliquary.environment.grader_client import GraderInfrastructureError

        raise GraderInfrastructureError(f"{socket_env} is not set")
    return GraderClient(path) if path else GraderClient()


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"pickled tests may not name {module}.{name}")


class _Subprocess:
    """The verifier's ``subprocess``: ``run`` goes to the grading service."""

    TimeoutExpired = subprocess.TimeoutExpired
    PIPE = subprocess.PIPE

    def __init__(self, client_factory, guest_template: str = GUEST_TEMPLATE) -> None:
        self._client_factory = client_factory
        self._guest_template = guest_template
        self.infra_error: BaseException | None = None
        self.calls = 0

    def run(self, argv, input=None, capture_output=False, timeout=None, text=False, **_):
        if (len(argv) != 3 or argv[0] != SANDBOX_PYTHON or argv[1] != "-c"
                or not capture_output or not text or timeout is None):
            raise ValueError(f"unexpected verifier call: {argv[:2]!r}")
        from reliquary.environment.grader_client import GraderInfrastructureError

        if self.infra_error is not None:
            raise self.infra_error
        self.calls += 1
        try:
            status, stdout = self._client_factory().run_stdio(
                code="", guest=guest_source(timeout, self._guest_template),
                stdin=frame(argv[2], input or ""),
                output_cap=OUTPUT_CAP, time_limit_s=CPU_FACTOR * float(timeout))
        except GraderInfrastructureError as exc:
            # The verifier would read this as a failed test: kept, then raised.
            self.infra_error = self.infra_error or exc
            raise
        if status != "ok":
            return subprocess.CompletedProcess(argv, 1, "", "")
        # text=True: universal newlines on the captured stdout.
        stdout = stdout.replace("\r\n", "\n").replace("\r", "\n")
        return subprocess.CompletedProcess(argv, 0, stdout, "")


class GraderRuntime:
    """The part of ``vf.Runtime`` a pinned verifier uses: files in memory,
    ``run_uv_script`` on this side with its ``subprocess`` replaced."""

    network_restricted = True

    def __init__(self, allowed: frozenset[str], client_factory=_client,
                 guest_template: str = GUEST_TEMPLATE) -> None:
        self._allowed = allowed
        self._client_factory = client_factory
        self._guest_template = guest_template
        self.files: dict[str, bytes] = {}
        self.last_calls = 0

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = bytes(data)

    async def _read(self, path: str) -> bytes:
        return self.files[path]

    async def read(self, path: str, max_bytes: int | None = None) -> bytes:
        data = self.files[path]
        if max_bytes is not None and len(data) > max_bytes:
            raise ValueError(f"read {path!r}: over the {max_bytes} byte limit")
        return data

    async def prepare_execution(self, routes: list[str] | None) -> None:
        return None

    async def run_uv_script(self, script: str | bytes, args: list[str] | None = None,
                            env: dict[str, str] | None = None):
        data = script.encode() if isinstance(script, str) else bytes(script)
        if hashlib.sha256(data).hexdigest() not in self._allowed:
            raise RuntimeError("this verifier script is not pinned for the grading service")
        shim = _Subprocess(self._client_factory, self._guest_template)
        out = io.StringIO()
        namespace: dict[str, Any] = {"__name__": "reliquary_pinned_verifier"}
        exec(compile(data, "<verify.py>", "exec"), namespace)

        files = self.files

        def open_(path, *a, **k):
            return io.StringIO(files[path].decode("utf-8"))

        run_case, failed = namespace.get("run_case"), threading.Event()

        def guarded(*a, **k):
            # All-or-nothing reward: after a failure the rest cannot change it.
            if failed.is_set():
                return False
            ok = run_case(*a, **k)
            if not ok:
                failed.set()
            return ok

        def literal(text):
            # The verifier's `eval` of test inputs: dataset text, on this side.
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
                shim.infra_error = shim.infra_error or ValueError(f"a test input is not a literal: {exc}")
                raise

        namespace.update(
            # Tests pickled as one string (LCB's private cases): no global may load.
            pickle=types.SimpleNamespace(loads=lambda raw: _NoGlobals(io.BytesIO(raw)).load()),
            eval=literal, subprocess=shim, open=open_,
            sys=types.SimpleNamespace(argv=["verify.py", *(args or [])], executable=SANDBOX_PYTHON),
            print=lambda *a, **k: print(*a, **{**k, "file": out}))
        if run_case is not None:
            namespace["run_case"] = guarded
        namespace["main"]()
        self.last_calls = shim.calls
        if shim.infra_error is not None:
            raise shim.infra_error
        return _program_result(out.getvalue())


def _program_result(stdout: str):
    try:
        from verifiers.v1.runtimes.base import ProgramResult
    except ImportError:
        return types.SimpleNamespace(exit_code=0, stdout=stdout, stderr="")
    return ProgramResult(exit_code=0, stdout=stdout, stderr="")


def runtime_for(task) -> GraderRuntime | None:
    """The grading-service runtime for ``task``'s pinned verifier, or None."""
    adapter = ADAPTED.get(type(task).__module__)
    if adapter is None:
        return None
    return GraderRuntime(adapter.verifiers, lambda: _client(adapter.socket_env), adapter.guest_template)


__all__ = ["ADAPTED", "Adapter", "GraderRuntime", "LIVECODEBENCH_VERIFY_SHA256", "SCICODE_SOCKET_ENV",
           "SCICODE_VERIFY_SHA256", "SOCKET_ENV", "frame", "guest_source", "runtime_for"]
