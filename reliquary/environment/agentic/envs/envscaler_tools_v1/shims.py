"""Deterministic execution namespace for upstream environment classes.

The environment implementations are LLM-written Python carried as source in
the pinned dataset. Measured over the 191 published classes, 31% call
``time.time()``, 32% call ``datetime.now()`` and 20% use ``uuid``; only 36%
are free of all three. Executed as-is, a miner and a validator replaying the
same episode observe different bytes, and the honest miner is rejected.

Freezing those three recovers every environment: with the shims below all
191 instantiate and 300 sampled scenarios replay byte-identically across
independent runs.

The frozen values are therefore part of the consensus contract, not local
configuration. They live here so ``implementation_sha256`` binds them, and
the tests pin them.
"""

from __future__ import annotations

import builtins
import types
from typing import Any, Callable


# 2023-11-14T22:13:20Z. Arbitrary but fixed forever: changing it changes
# every observation an environment produces.
FROZEN_EPOCH_SECONDS = 1_700_000_000.0
FROZEN_ISO_DATE = "2023-11-14"
FROZEN_ISO_DATETIME = "2023-11-14T22:13:20"


class _FrozenTime:
    """The subset of ``time`` the upstream classes actually reach for."""

    def time(self) -> float:
        return FROZEN_EPOCH_SECONDS

    def monotonic(self) -> float:
        return FROZEN_EPOCH_SECONDS

    def ctime(self, *_args: Any) -> str:
        return "Tue Nov 14 22:13:20 2023"

    def strftime(self, fmt: str, *_args: Any) -> str:
        import time as _time

        return _time.strftime(fmt, _time.gmtime(FROZEN_EPOCH_SECONDS))

    def gmtime(self, *_args: Any):
        import time as _time

        return _time.gmtime(FROZEN_EPOCH_SECONDS)

    localtime = gmtime


class _SequentialUUID:
    """uuid4 as a counter.

    Sequential rather than random, and restarted with the environment, so two
    rollouts of the same prompt observe the same identifiers.
    """

    def __init__(self) -> None:
        self._counter = 0

    def _next(self):
        import uuid as _uuid

        self._counter += 1
        return _uuid.UUID(int=self._counter)

    uuid1 = uuid3 = uuid4 = uuid5 = property(lambda self: self._next)

    def __getattr__(self, name: str) -> Any:
        import uuid as _uuid

        return getattr(_uuid, name)


class _FrozenDatetime(types.ModuleType):
    """``datetime`` whose ``now``/``utcnow``/``today`` never move."""

    def __init__(self) -> None:
        super().__init__("datetime")
        import datetime as _datetime

        self._real = _datetime
        fixed = _datetime.datetime(2023, 11, 14, 22, 13, 20)

        class _Datetime(_datetime.datetime):
            @classmethod
            def now(cls, tz: Any = None):
                return fixed if tz is None else fixed.replace(tzinfo=tz)

            @classmethod
            def utcnow(cls):
                return fixed

            @classmethod
            def today(cls):
                return fixed

        class _Date(_datetime.date):
            @classmethod
            def today(cls):
                return fixed.date()

        self.datetime = _Datetime
        self.date = _Date

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _shim_modules() -> dict[str, Any]:
    return {
        "time": _FrozenTime(),
        "uuid": _SequentialUUID(),
        "datetime": _FrozenDatetime(),
    }


def deterministic_import(
    shims: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """An ``__import__`` that serves frozen modules and defers otherwise.

    Covers both ``import uuid`` and ``from uuid import uuid4``: the latter
    also routes through ``__import__`` and then reads the attribute off
    whatever this returns.
    """
    resolved = _shim_modules() if shims is None else shims
    real_import = builtins.__import__

    def _import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in resolved:
            return resolved[name]
        return real_import(name, *args, **kwargs)

    return _import


def build_environment_class(source: str, class_name: str) -> type:
    """Execute one environment's source in a frozen namespace.

    Mirrors upstream ``init_env_class``, except that the import hook makes
    the result reproducible.
    """
    namespace: dict[str, Any] = {}
    injected = dict(vars(builtins))
    injected["__import__"] = deterministic_import()
    namespace["__builtins__"] = injected
    exec(source, namespace)  # noqa: S102 - dataset-carried source, see module docstring
    try:
        return namespace[class_name]
    except KeyError as exc:
        raise ValueError(
            f"environment class {class_name!r} is not defined by its source"
        ) from exc


__all__ = [
    "FROZEN_EPOCH_SECONDS",
    "FROZEN_ISO_DATE",
    "FROZEN_ISO_DATETIME",
    "build_environment_class",
    "deterministic_import",
]
