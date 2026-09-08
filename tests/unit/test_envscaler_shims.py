"""The deterministic shims are consensus, not configuration.

The upstream environment classes are LLM-written and reach for wall-clock
time, uuid4 and datetime.now: measured across the 191 published classes,
31% call time.time(), 32% call datetime.now() and 20% use uuid. Only 36%
are clean. Left alone, a miner and a validator replaying the same episode
observe different bytes and the honest miner is rejected.

Freezing those three at fixed values recovers every environment. Because
both sides must freeze them identically, the shim values belong in the
manifest-bound source, and these tests pin them.
"""

from __future__ import annotations

from reliquary.environment.agentic.envs.envscaler_tools_v1.shims import (
    FROZEN_EPOCH_SECONDS,
    build_environment_class,
    deterministic_import,
)


_TIME_CODE = """
import time

class Probe:
    def __init__(self, config=None):
        self.stamp = time.time()
        self.pretty = time.strftime("%Y-%m-%d")
"""

_UUID_CODE = """
import uuid

class Probe:
    def __init__(self, config=None):
        self.ids = [str(uuid.uuid4()) for _ in range(3)]
"""

_DATETIME_CODE = """
from datetime import datetime

class Probe:
    def __init__(self, config=None):
        self.now = datetime.now().isoformat()
        self.today = datetime.today().isoformat()
"""

_FROM_IMPORT_CODE = """
from uuid import uuid4
from time import time as clock

class Probe:
    def __init__(self, config=None):
        self.a = str(uuid4())
        self.b = clock()
"""


def _twice(code: str) -> tuple[dict, dict]:
    first = build_environment_class(code, "Probe")()
    second = build_environment_class(code, "Probe")()
    return vars(first), vars(second)


def test_wall_clock_time_is_frozen():
    first, second = _twice(_TIME_CODE)
    assert first == second
    assert first["stamp"] == FROZEN_EPOCH_SECONDS


def test_uuid_is_a_deterministic_sequence():
    first, second = _twice(_UUID_CODE)
    assert first == second
    # Sequential rather than random, and restarting the environment restarts
    # the sequence — otherwise two rollouts of one prompt would differ.
    assert len(set(first["ids"])) == 3


def test_datetime_now_is_frozen():
    first, second = _twice(_DATETIME_CODE)
    assert first == second


def test_from_import_form_is_shimmed_too():
    """``from uuid import uuid4`` must be caught as well as ``import uuid``."""
    first, second = _twice(_FROM_IMPORT_CODE)
    assert first == second
    assert second["b"] == FROZEN_EPOCH_SECONDS


def test_unshimmed_modules_still_import():
    code = "import json\n\nclass Probe:\n    def __init__(self, config=None):\n        self.v = json.dumps({'a': 1})"
    instance = build_environment_class(code, "Probe")()
    assert instance.v == '{"a": 1}'


def test_missing_class_is_reported():
    import pytest

    with pytest.raises(ValueError, match="Absent"):
        build_environment_class("x = 1", "Absent")


def test_import_hook_passes_through_unknown_modules():
    imported = deterministic_import()
    assert imported("json") is not None
