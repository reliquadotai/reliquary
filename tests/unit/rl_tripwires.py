"""Tripwires on every entry point of the next-RL-run service stack (phase 1) and of signed episodes.

A legacy RL task and a corpus job must never reach them; a single-turn v2 order never reaches the signed-episode ones. ``rl_service_tripwires`` is autouse where it is
imported: it replaces each entry point by a wrapper that RAISES and RECORDS the call, and fails the test
afterwards if anything was recorded (a caller that swallows the exception does not hide it).

Used two ways: ``from tests.unit.rl_tripwires import rl_service_tripwires`` in a test module (armed for that
module only), and ``pytest -p tests.unit.rl_tripwires <legacy test file>`` (``test_next_rl_run_inertness``
reruns the legacy and corpus suites that way). ``tests.unit.episode_tripwires`` arms the signed-episode entry points
only (``EPISODE_ARMED`` and ``EPISODE_CONDITIONAL``), for the single-turn v2 suites that legitimately run the phase 1
stack (``test_episode_inertness``).
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import inspect

import pytest

# label -> (owner, attr, original attribute) of the armed entry points, filled by the fixture.
_ORIGINALS: dict[str, tuple] = {}


@contextlib.contextmanager
def real_entry(module_name: str, dotted: str):
    """Run a block with ONE armed entry point restored (a test that legitimately declares a service task needs
    its contract parsed). Everything else stays armed."""
    owner, attr, original = _ORIGINALS[f"{module_name}.{dotted}"]
    armed = owner.__dict__[attr]
    setattr(owner, attr, original)
    try:
        yield
    finally:
        setattr(owner, attr, armed)

def _kw(args, kwargs, name, position):
    return kwargs[name] if name in kwargs else (args[position] if len(args) > position else None)


def _batcher_hands_back(args, kwargs, result):
    self, pending = args[0], _kw(args, kwargs, "pending", 1)
    return self.episode_proof_inconclusive is not None or (
        pending is not None and self._signed_episode_pending(pending))


# Modules of the optional sandbox extras. Without them installed, an entry point under them
# cannot be imported at all, so a legacy path reaching it would fail on its own: its
# tripwire is not armed (the CPU suite runs without the extras).
_OPTIONAL_EXTRA_PACKAGES = ("reliquary_sandbox", "reliquary_sandbox_service",
                            "reliquary_sandbox_verifiers", "verifiers")


def _optional_extra_missing(exc: ImportError) -> bool:
    name = getattr(exc, "name", None) or ""
    return name.split(".")[0] in _OPTIONAL_EXTRA_PACKAGES


def _resolve_armable(module_name: str, dotted: str):
    """``_resolve``, or None when the module needs an optional extra that is not installed."""
    try:
        return _resolve(module_name, dotted)
    except ImportError as exc:
        if _optional_extra_missing(exc):
            return None
        raise


def _resolve(module_name: str, dotted: str):
    owner = importlib.import_module(module_name)
    *path, attr = dotted.split(".")
    for name in path:
        owner = getattr(owner, name)
    return owner, attr


def arm_conditional(monkeypatch, calls: list[str], conditional) -> None:
    """Wrap each ``(module, dotted, predicate)``: the real code runs, and the call is recorded (and refused)
    when the predicate says it took the episode branch. Coroutine functions stay coroutine functions."""
    for module_name, dotted, predicate in conditional:
        resolved = _resolve_armable(module_name, dotted)
        if resolved is None:
            continue
        owner, attr = resolved
        real = owner.__dict__.get(attr, getattr(owner, attr))
        is_static = isinstance(real, staticmethod)
        function = real.__func__ if is_static else real
        label = f"{module_name}.{dotted}"

        def make(label, function, predicate):
            def check(args, kwargs, result):
                if predicate(args, kwargs, result):
                    calls.append(label)
                    raise AssertionError(f"{label} took its episode branch for a legacy RL task / single-turn "
                                         f"order / corpus job")
                return result

            if inspect.iscoroutinefunction(function):
                async def conditional_async(*args, **kwargs):
                    return check(args, kwargs, await function(*args, **kwargs))
                return functools.wraps(function)(conditional_async)

            def conditional(*args, **kwargs):
                return check(args, kwargs, function(*args, **kwargs))
            return functools.wraps(function)(conditional)
        wrapped = make(label, function, predicate)
        monkeypatch.setattr(owner, attr, staticmethod(wrapped) if is_static else wrapped)


def arm_entry_points(monkeypatch, calls: list[str], armed) -> None:
    """Replace each ``(module, dotted)`` by a wrapper that RAISES and RECORDS the call."""
    for module_name, dotted in armed:
        resolved = _resolve_armable(module_name, dotted)
        if resolved is None:
            continue
        owner, attr = resolved
        label = f"{module_name}.{dotted}"
        original = owner.__dict__.get(attr, getattr(owner, attr))
        is_static = isinstance(original, (staticmethod, classmethod))
        _ORIGINALS[label] = (owner, attr, original)

        def make(label):
            def tripwire(*args, **kwargs):
                calls.append(label)
                raise AssertionError(f"{label} ran for a legacy RL task / corpus job")
            return tripwire
        monkeypatch.setattr(owner, attr, staticmethod(make(label)) if is_static else make(label))


