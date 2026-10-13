"""The child process one EnvScaler world runs in. Standard library only.

EnvScaler's worlds and checks are LLM-written Python carried as source in the
corpus. ``envscaler_tools_v1`` executed them inside the miner's and the
validator's own process, with full builtins and real imports, and called
whatever method the policy named with whatever arguments it chose. This file is
the other side of that boundary: every ``exec`` of world or check code, at
reset, at every step and at grading, happens here, in a process that

* starts from a constructed environment (no inherited variables), without
  ``site`` (``-S``: nothing installed on the host is importable, so the same
  world behaves the same on every machine), without the script's directory on
  ``sys.path`` (``-P``) and without writing bytecode (``-B``);
* drops into a fresh user and network namespace when the kernel allows it,
  so a socket has no route anywhere; where it does not (most container
  runtimes refuse unprivileged namespaces), the audit hook below still refuses
  every socket operation, and the handshake reports which of the two applied;
* runs under rlimits: CPU seconds, address space, no new processes or threads,
  no file growth, no core dumps, few descriptors;
* installs an audit hook, before any corpus byte is executed, that refuses
  sockets, subprocesses, ``exec*``/``fork``/``spawn``/``system``, ``ctypes``,
  filesystem mutation and directory listing, and any ``open`` other than a
  read of the standard library (which lazy imports need). A hook cannot be
  removed from Python, and the one route around it (``ctypes``) is itself
  refused;
* never reads the corpus: the parent sends the one world's source and the one
  scenario's configuration over the pipe, so there is no data path to write
  to or read from.

The rlimits are what bound an infinite loop or a memory bomb; the hook is what
turns file, network and process access into a clean ``PermissionError`` the
world sees as a failing tool. None of this is a VM boundary, and it does not
need to be one: the code is pinned by digest and the policy only picks
arguments; what it must never do is let those arguments reach the host.

Determinism is the other half of the contract, because a validator replays the
episode in another process on another machine and compares every observation
byte for byte. The parent pins ``PYTHONHASHSEED`` and ``TZ`` in the child's
environment (which is why the child is not started with ``-I``: ``-I`` implies
``-E`` and would discard both). Inside, the clock, ``uuid``, ``random`` and
``secrets`` are replaced by frozen or seeded stand-ins built afresh for each
episode, and every value leaving the process is reduced to plain JSON without
``repr`` — so no memory address can reach an observation.

Protocol ``envscaler-world/v1``, one JSON object per line in each direction.
The worker speaks first::

    <- {"protocol": "envscaler-world/v1", "network": "namespace" | "audit-hook"}
    -> {"op": "init", "source": str, "class_name": str, "init_config": {...}}
    <- {"ok": true} | {"ok": false, "error": str}
    -> {"op": "call", "tool": str, "arguments": {...}}
    <- {"ok": true, "result": <plain json>} | {"ok": false, "error": str}
    -> {"op": "grade", "checks": [str, ...]}
    <- {"ok": true, "initial": [bool|null, ...], "final": [bool|null, ...],
        "state_digest": hex}

Anything else, or a line that is not JSON, ends the process. So does
end-of-file on stdin, which is how a parent that died takes its world with it.
"""

from __future__ import annotations

import builtins
import copy
import datetime as _real_datetime
import hashlib
import json
import math
import os
import random as _real_random
import resource
import sys
import sysconfig
import time as _real_time
import types
import uuid as _real_uuid

# Every standard-library module the pinned corpus's code imports — world
# sources and check functions, at module level or inside a method (an AST scan
# of all 191 worlds and every check) — plus `_strptime`, which `strptime`
# imports on its first call (381 call sites), loaded here, before the audit
# hook. The hook refuses directory listing, and the import system lists a
# package's directory the first time it imports from it: a module nothing had
# loaded before the hook could be unimportable, and which ones are already
# loaded at startup depends on the Python version (on 3.12 `typing` needs
# `collections.abc`, which nothing above loaded, so every world failed to
# start; on 3.13 they started). Loaded up front, a world imports the same
# modules from `sys.modules` on every miner and validator whatever their
# Python. `dateutil` (six check functions) is not standard: under `-S` it is
# missing everywhere, so those checks are None everywhere.
import _strptime  # noqa: E402,F401
import ast  # noqa: E402,F401
import bisect  # noqa: E402,F401
import calendar  # noqa: E402,F401
import collections  # noqa: E402,F401
import collections.abc  # noqa: E402,F401
import csv  # noqa: E402,F401
import io  # noqa: E402,F401
import itertools  # noqa: E402,F401
import re  # noqa: E402,F401
import typing  # noqa: E402,F401
import urllib.parse  # noqa: E402,F401

# The list above, for the test that holds it equal to the corpus's imports.
PRELOADED_MODULES = (
    "_strptime", "ast", "bisect", "calendar", "collections", "collections.abc",
    "copy", "csv", "datetime", "hashlib", "io", "itertools", "math", "random",
    "re", "time", "typing", "urllib.parse", "uuid",
)

PROTOCOL = "envscaler-world/v1"

# Bounds. Generous next to what a world needs — measured, a whole episode of
# calls and every check costs milliseconds — and small next to what a runaway
# would take. RLIMIT_CPU counts the whole episode, not one call.
CPU_SECONDS = 10
ADDRESS_SPACE_BYTES = 1 << 30
MAX_OPEN_FILES = 32
# A request carries one world's source and one scenario's configuration.
MAX_REQUEST_BYTES = 16 << 20
# What one tool call may send back. Larger than any observation an episode
# profile admits, so the profile's own byte budget is what bites.
MAX_RESULT_BYTES = 1 << 20
MAX_ERROR_CHARS = 400
MAX_PLAIN_DEPTH = 64

# 2023-11-14T22:13:20Z. Arbitrary, fixed forever: the same instant as
# envscaler_tools_v1, so a world reads the same clock in both versions.
FROZEN_EPOCH_SECONDS = 1_700_000_000.0
FROZEN_DATETIME = (2023, 11, 14, 22, 13, 20)
RANDOM_SEED = 0


# ── deterministic stand-ins ──────────────────────────────────────────────────


class _FrozenTime(types.ModuleType):
    """``time`` whose every clock reads the same instant, and which never sleeps."""

    def __init__(self) -> None:
        super().__init__("time")
        frozen = FROZEN_EPOCH_SECONDS
        frozen_ns = int(frozen * 1_000_000_000)
        for name in ("time", "monotonic", "perf_counter", "process_time",
                     "thread_time"):
            setattr(self, name, lambda value=frozen: value)
        for name in ("time_ns", "monotonic_ns", "perf_counter_ns",
                     "process_time_ns", "thread_time_ns"):
            setattr(self, name, lambda value=frozen_ns: value)
        self.sleep = lambda _seconds=0: None

    # Each of these reads the clock only when given no instant; given one, it
    # converts that instant, as the stdlib does (TZ is pinned to UTC).
    @staticmethod
    def gmtime(secs=None):
        return _real_time.gmtime(FROZEN_EPOCH_SECONDS if secs is None else secs)

    @staticmethod
    def localtime(secs=None):
        return _real_time.localtime(FROZEN_EPOCH_SECONDS if secs is None else secs)

    @staticmethod
    def ctime(secs=None) -> str:
        return _real_time.ctime(FROZEN_EPOCH_SECONDS if secs is None else secs)

    @staticmethod
    def asctime(moment=None) -> str:
        return _real_time.asctime(
            _real_time.gmtime(FROZEN_EPOCH_SECONDS) if moment is None else moment
        )

    @staticmethod
    def strftime(fmt, moment=None) -> str:
        return _real_time.strftime(
            fmt, _real_time.gmtime(FROZEN_EPOCH_SECONDS) if moment is None else moment
        )

    def __getattr__(self, name):
        # mktime, strptime, struct_time, timezone: pure functions of their
        # arguments once TZ is pinned, which the parent does.
        return getattr(_real_time, name)


class _FrozenDatetime(types.ModuleType):
    """``datetime`` whose ``now``/``utcnow``/``today`` never move."""

    def __init__(self) -> None:
        super().__init__("datetime")
        fixed = _real_datetime.datetime(*FROZEN_DATETIME)

        class datetime(_real_datetime.datetime):  # noqa: N801 - mirrors stdlib
            @classmethod
            def now(cls, tz=None):
                return fixed if tz is None else fixed.replace(tzinfo=tz)

            @classmethod
            def utcnow(cls):
                return fixed

            @classmethod
            def today(cls):
                return fixed

        class date(_real_datetime.date):  # noqa: N801 - mirrors stdlib
            @classmethod
            def today(cls):
                return fixed.date()

        self.datetime = datetime
        self.date = date

    def __getattr__(self, name):
        return getattr(_real_datetime, name)


class _SequentialUUID(types.ModuleType):
    """``uuid`` whose clock and random generators count from one, per episode.

    ``uuid3`` and ``uuid5`` are hashes of their arguments, already
    deterministic, and are left as they are.
    """

    def __init__(self) -> None:
        super().__init__("uuid")
        self._counter = 0

    def _next(self, *_args, **_kwargs):
        self._counter += 1
        return _real_uuid.UUID(int=self._counter)

    def __getattr__(self, name):
        if name in ("uuid1", "uuid4"):
            return self._next
        return getattr(_real_uuid, name)


class _SeededRandom(types.ModuleType):
    """``random`` as one generator seeded with a constant, afresh per episode."""

    def __init__(self) -> None:
        super().__init__("random")
        generator = _real_random.Random(RANDOM_SEED)
        self._generator = generator

        class Random(_real_random.Random):
            """A world's own generator: seeded with the constant unless told."""

            def __init__(self, x=None):
                super().__init__(RANDOM_SEED if x is None else x)

            def seed(self, a=None, version=2):
                super().seed(RANDOM_SEED if a is None else a, version)

        self.Random = Random
        self.SystemRandom = lambda *_a, **_k: generator
        # seed() with no argument would reseed from the OS.
        self.seed = lambda a=None, version=2: generator.seed(
            RANDOM_SEED if a is None else a, version
        )

    def __getattr__(self, name):
        bound = getattr(self._generator, name, None)
        if bound is not None and not name.startswith("_"):
            return bound
        return getattr(_real_random, name)


class _SeededSecrets(types.ModuleType):
    """``secrets`` drawn from the same seeded generator: nothing here is secret."""

    def __init__(self, generator: _real_random.Random) -> None:
        super().__init__("secrets")
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        self.token_bytes = lambda n=32: generator.randbytes(n)
        self.token_hex = lambda n=32: generator.randbytes(n).hex()
        self.token_urlsafe = lambda n=32: "".join(
            generator.choice(alphabet) for _ in range(max(1, (n * 4 + 2) // 3))
        )
        self.choice = generator.choice
        self.randbelow = lambda n: generator.randrange(n)
        self.randbits = generator.getrandbits
        self.compare_digest = __import__("hmac").compare_digest
        self.SystemRandom = lambda *_a, **_k: generator


def frozen_builtins() -> dict:
    """Builtins whose ``__import__`` serves the stand-ins above.

    Built once per process, which is once per episode: the uuid counter and
    the random generator restart with every world, so two rollouts of one
    prompt read the same identifiers and draw the same numbers.
    """
    seeded = _SeededRandom()
    shims = {
        "time": _FrozenTime(),
        "datetime": _FrozenDatetime(),
        "uuid": _SequentialUUID(),
        "random": seeded,
        "secrets": _SeededSecrets(seeded._generator),
    }
    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in shims:
            return shims[name]
        return real_import(name, globals, locals, fromlist, level)

    namespace = dict(vars(builtins))
    namespace["__import__"] = _import
    return namespace


# ── plain values out ─────────────────────────────────────────────────────────


def _key(value) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else _plain(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, tuple):
        return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"))
    return f"<{type(value).__name__}>"


def _plain(value, depth: int = 0, active: frozenset = frozenset()):
    """Reduce a world's value to JSON without ever calling ``repr``.

    A default ``repr`` carries a memory address, which differs between the
    miner's process and the validator's; a set iterates in hash order. So
    objects become their attributes or their type name, sets are sorted by
    their canonical form, and non-finite floats become strings.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if depth >= MAX_PLAIN_DEPTH:
        return "<deep>"
    marker = id(value)
    if marker in active:
        return "<cycle>"
    inner = active | {marker}
    if isinstance(value, dict):
        return {_key(key): _plain(item, depth + 1, inner) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth + 1, inner) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_plain(item, depth + 1, inner) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (_real_datetime.datetime, _real_datetime.date,
                          _real_datetime.time)):
        return value.isoformat()
    if isinstance(value, _real_datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, _real_uuid.UUID):
        return str(value)
    enum_value = getattr(type(value), "__members__", None)
    if enum_value is not None and hasattr(value, "value"):
        return _plain(value.value, depth + 1, inner)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict) and not isinstance(
        value, (type, types.ModuleType, types.FunctionType, types.MethodType)
    ):
        return {
            _key(key): _plain(item, depth + 1, inner)
            for key, item in attributes.items()
            if not (key.startswith("__") and key.endswith("__"))
        }
    return f"<{type(value).__name__}>"


def canonical(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def _error(exc: BaseException) -> str:
    try:
        message = str(exc)
    except BaseException:  # noqa: BLE001 - a world's exception may not stringify
        message = ""
    return f"{type(exc).__name__}: {message}"[:MAX_ERROR_CHARS]


# ── containment ──────────────────────────────────────────────────────────────


def _isolate_network() -> str:
    """A fresh user and network namespace, where the kernel grants one."""
    try:
        os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET)
    except (AttributeError, OSError):
        return "audit-hook"
    return "namespace"


def _apply_rlimits() -> None:
    limits = (
        (resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS + 1)),
        (resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES)),
        (resource.RLIMIT_NOFILE, (MAX_OPEN_FILES, MAX_OPEN_FILES)),
        # No new process or thread: counted per user, so zero refuses both.
        (resource.RLIMIT_NPROC, (0, 0)),
        (resource.RLIMIT_FSIZE, (0, 0)),
        (resource.RLIMIT_CORE, (0, 0)),
    )
    for limit, value in limits:
        resource.setrlimit(limit, value)


_DENIED_EVENT_PREFIXES = (
    "socket.", "subprocess.", "os.exec", "os.fork", "os.posix_spawn",
    "os.spawn", "os.system", "os.startfile", "os.kill", "os.killpg",
    "os.remove", "os.unlink", "os.rename", "os.replace", "os.rmdir",
    "os.mkdir", "os.makedirs", "os.chmod", "os.chown", "os.chdir",
    "os.chroot", "os.truncate", "os.symlink", "os.link", "os.listdir",
    "os.scandir", "os.walk", "os.putenv", "os.unsetenv", "os.setxattr",
    "os.removexattr", "os.utime", "os.mkfifo", "os.mknod", "shutil.",
    "ctypes.", "pty.", "glob.", "sqlite3.", "webbrowser.", "urllib.Request",
    "http.client.", "ftplib.", "smtplib.", "imaplib.", "poplib.", "nntplib.",
    "telnetlib.", "mmap.", "fcntl.", "resource.setrlimit", "resource.prlimit",
    "sys.addaudithook", "sys.remote_exec", "_thread.start_new_thread",
    "signal.pthread_kill", "syslog.", "msvcrt.", "winreg.",
)

_WRITE_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
)


def _readable_roots() -> tuple[str, ...]:
    roots = set()
    for key in ("stdlib", "platstdlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            roots.add(os.path.normpath(path))
    for entry in sys.path:
        if entry:
            roots.add(os.path.normpath(os.path.abspath(entry)))
    return tuple(sorted(roots))


def _install_audit_hook() -> None:
    roots = tuple(root + os.sep for root in _readable_roots())

    def _open_allowed(path, mode, flags) -> bool:
        if not isinstance(path, (str, bytes)):
            return False
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            return False
        if isinstance(flags, int) and flags & _WRITE_FLAGS:
            return False
        text = os.fsdecode(path)
        resolved = os.path.normpath(os.path.join("/", text))
        return resolved.startswith(roots)

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path, mode, flags = (tuple(args) + (None, None, None))[:3]
            if not _open_allowed(path, mode, flags):
                raise PermissionError("envscaler world: file access refused")
            return
        if event.startswith(_DENIED_EVENT_PREFIXES):
            raise PermissionError(f"envscaler world: {event} refused")

    sys.addaudithook(hook)


class _Sink:
    """Where a world's prints go: nowhere, without growing."""

    encoding = "utf-8"
    errors = "strict"

    def write(self, text) -> int:
        return len(text)

    def writelines(self, _lines) -> None:
        return None

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def readline(self, *_args) -> str:
        return ""

    def read(self, *_args) -> str:
        return ""


# ── the world ────────────────────────────────────────────────────────────────


def _state_of(instance) -> dict:
    return copy.deepcopy({
        key: value for key, value in vars(instance).items()
        if not (key.startswith("__") and key.endswith("__"))
    })


class World:
    def __init__(self) -> None:
        self.builtins = frozen_builtins()
        self.instance = None
        self.initial: dict | None = None

    def init(self, source: str, class_name: str, init_config: dict) -> None:
        namespace = {"__builtins__": self.builtins, "__name__": "envscaler_world"}
        exec(compile(source, "<envscaler-world>", "exec"), namespace)  # noqa: S102
        world_class = namespace.get(class_name)
        if not isinstance(world_class, type):
            raise ValueError(f"class {class_name!r} is not defined by its source")
        # Same construction as envscaler_tools_v1 and upstream: hand the
        # configuration to the constructor when it takes one, then set every
        # key as an attribute. Deep-copied twice, so the constructor cannot
        # alias what the attributes hold.
        config = copy.deepcopy(dict(init_config))
        try:
            instance = world_class(copy.deepcopy(config))
        except TypeError:
            instance = world_class()
        for key, value in config.items():
            setattr(instance, key, copy.deepcopy(value))
        self.instance = instance
        self.initial = _state_of(instance)

    def call(self, tool: str, arguments: dict):
        if self.instance is None:
            raise RuntimeError("world not initialised")
        if not tool or tool.startswith("_"):
            raise AttributeError(f"no tool named {tool[:80]!r}")
        method = getattr(self.instance, tool, None)
        if not callable(method):
            raise AttributeError(f"no tool named {tool[:80]!r}")
        return _plain(method(**arguments))

    def _check(self, source: str, final: dict):
        namespace = {
            "__builtins__": self.builtins,
            "__name__": "envscaler_check",
            "initial_state": copy.deepcopy(self.initial),
        }
        try:
            exec(compile(source, "<envscaler-check>", "exec"), namespace)  # noqa: S102
            checker = namespace.get("check_func")
            if not callable(checker):
                return None
            verdict = checker(copy.deepcopy(final))
        except Exception:  # noqa: BLE001 - a broken check is a verdict of None
            return None
        return verdict if isinstance(verdict, bool) else None

    def grade(self, checks: list) -> dict:
        if self.instance is None or self.initial is None:
            raise RuntimeError("world not initialised")
        final = _state_of(self.instance)
        return {
            "initial": [self._check(source, self.initial) for source in checks],
            "final": [self._check(source, final) for source in checks],
            "state_digest": hashlib.sha256(
                canonical(_plain(final)).encode("utf-8")
            ).hexdigest(),
        }


class _ProtocolError(Exception):
    """A request this protocol does not define: the parent is not speaking it."""


def _serve(world: World, request: dict) -> dict:
    op = request.get("op")
    if op == "init":
        world.init(
            str(request["source"]), str(request["class_name"]),
            dict(request.get("init_config") or {}),
        )
        return {"ok": True}
    if op == "call":
        arguments = request.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise TypeError("arguments must be an object")
        result = world.call(str(request["tool"]), arguments)
        reply = {"ok": True, "result": result}
        if len(canonical(reply)) > MAX_RESULT_BYTES:
            return {"ok": False, "error": "ResultTooLarge: the tool returned "
                    f"more than {MAX_RESULT_BYTES} bytes"}
        return reply
    if op == "grade":
        return {"ok": True, **world.grade(list(request.get("checks") or []))}
    raise _ProtocolError(op)


def main() -> int:
    # The protocol keeps private copies of the pipes; the world's stdin,
    # stdout and stderr are /dev/null, so a print cannot corrupt a reply and
    # an input() reads end-of-file.
    requests = os.fdopen(os.dup(0), "rb")
    replies = os.fdopen(os.dup(1), "wb")
    null = os.open(os.devnull, os.O_RDWR)
    for descriptor in (0, 1, 2):
        os.dup2(null, descriptor)
    os.close(null)
    sys.stdin = sys.stdout = sys.stderr = _Sink()
    os.chdir("/")

    network = _isolate_network()
    world = World()
    _apply_rlimits()
    _install_audit_hook()

    def send(message: dict) -> None:
        replies.write(canonical(message).encode("ascii") + b"\n")
        replies.flush()

    send({"protocol": PROTOCOL, "network": network})
    while True:
        line = requests.readline(MAX_REQUEST_BYTES + 1)
        if not line or len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
            return 0
        try:
            request = json.loads(line)
        except ValueError:
            return 2
        if not isinstance(request, dict):
            return 2
        try:
            reply = _serve(world, request)
        except _ProtocolError:
            return 2
        except BaseException as exc:  # noqa: BLE001 - reported, never raised
            # Includes a world's own SystemExit or KeyboardInterrupt: a tool
            # that calls sys.exit() has failed, not ended the protocol.
            reply = {"ok": False, "error": _error(exc)}
        send(reply)


if __name__ == "__main__":
    sys.exit(main())
