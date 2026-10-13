"""envscaler_tools_v2's worlds start, under the audit hook, on any Python.

The hook refuses directory listing, so a standard-library module the worker
did not load before it can be unimportable, and which ones are loaded at
startup differs between Python versions: on 3.12 every world of the corpus
failed to start while on 3.13 all of them did. The worker therefore loads the
corpus's whole import set up front (``worker.PRELOADED_MODULES``). The first
tests pin that with a world written to import all of it; the rest need the
fetched corpus (RELIQUARY_ENVSCALER_DATA) and skip without it: the corpus's
imports are exactly the preloaded set, every world starts and grades, and the
mended upstream worlds run the tools they broke.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import warnings

import pytest

from reliquary.environment.agentic.envs.envscaler_tools_v2 import (
    corpus as layout,
    worker,
)
from reliquary.environment.agentic.envs.envscaler_tools_v2.environment import (
    MAX_TURNS,
    EnvScalerToolsV2Environment,
    TASK_INSTRUCTIONS,
)
from reliquary.environment.agentic.envs.envscaler_tools_v2.sandbox import WorldProcess
from tests.unit import envscaler_v2_fixtures as fixtures

_IMPORTS = "\n".join(
    f"import {name}" for name in worker.PRELOADED_MODULES if name != "_strptime"
)
EVERY_IMPORT_SOURCE = _IMPORTS + """
from typing import Dict, List, Optional, TypedDict
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone

"""
EVERY_IMPORT_SOURCE += '''
class Everything:
    def __init__(self):
        self.seen = {}

    def use_all(self, stamp):
        import bisect, calendar, csv, io, itertools, math, re, urllib.parse
        from collections.abc import Mapping
        parsed = datetime.strptime(stamp, "%Y-%m-%d %H:%M")
        rows = list(csv.reader(io.StringIO("a,b\\n1,2")))
        self.seen = {
            "parsed": parsed.isoformat(),
            "iso": datetime.fromisoformat("2024-01-02T03:04:05").isoformat(),
            "epoch": calendar.timegm(parsed.timetuple()),
            "rows": rows,
            "slot": bisect.bisect([1, 3, 5], 4),
            "pairs": len(list(itertools.combinations("abcd", 2))),
            "host": urllib.parse.urlparse("https://example.org/x").netloc,
            "match": bool(re.match(r"\\d+", "42")),
            "abc": isinstance({}, Mapping),
            "root": math.sqrt(16),
            "counted": dict(Counter("aab")),
        }
        return {"success": True, "data": self.seen}
'''


def _init(process: WorldProcess, source: str, class_name: str, config=None) -> dict:
    return process.request({
        "op": "init", "source": source, "class_name": class_name,
        "init_config": dict(config or {}),
    })


def test_a_world_importing_the_whole_preloaded_set_starts_and_runs() -> None:
    process = WorldProcess()
    try:
        assert _init(process, EVERY_IMPORT_SOURCE, "Everything") == {"ok": True}
        reply = process.request({
            "op": "call", "tool": "use_all", "arguments": {"stamp": "2024-05-06 07:08"},
        })
    finally:
        process.close()
    assert reply["ok"], reply
    data = reply["result"]["data"]
    assert data["parsed"] == "2024-05-06T07:08:00"
    assert data["rows"] == [["a", "b"], ["1", "2"]]
    assert data["slot"] == 2 and data["pairs"] == 6 and data["abc"] is True
    assert data["host"] == "example.org"


def test_the_preloaded_modules_are_the_ones_the_worker_imports() -> None:
    # Everything listed is loaded before the hook (a name listed here and not
    # imported in worker.py would not be).
    tree = ast.parse(open(worker.__file__, encoding="utf-8").read())
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert set(worker.PRELOADED_MODULES) <= imported


def test_the_task_prompt_asks_for_one_call_per_message() -> None:
    # Part of the environment (and of its manifest), not of the shared
    # renderer: telecom's prompts are unchanged.
    task = EnvScalerToolsV2Environment(fixtures.corpus()).get_task(0)

    assert task.prompt.endswith("\n\n" + TASK_INSTRUCTIONS)
    assert "one tool per message" in TASK_INSTRUCTIONS


def test_an_episode_is_bounded_at_25_turns() -> None:
    assert MAX_TURNS == EnvScalerToolsV2Environment.max_turns == 25


def test_a_source_fix_must_match_exactly_once(monkeypatch) -> None:
    monkeypatch.setattr(layout, "WORLD_SOURCE_FIXES", {"env_x_rl": (("a()", "b()"),)})

    assert layout.world_source("env_x_rl", "x = a()\n") == "x = b()\n"
    assert layout.world_source("env_y_rl", "x = a()\n") == "x = a()\n"
    with pytest.raises(RuntimeError):
        layout.world_source("env_x_rl", "x = c()\n")
    with pytest.raises(RuntimeError):
        layout.world_source("env_x_rl", "x = a()\ny = a()\n")


# ── the real corpus, when it is here ─────────────────────────────────────────


@pytest.fixture(scope="module")
def real_corpus():
    if not os.environ.get(layout.DATA_ENV_VAR):
        pytest.skip(f"{layout.DATA_ENV_VAR} not set")
    return layout.default_corpus()


def _parse(source: str) -> ast.Module:
    with warnings.catch_warnings():
        # Upstream code has invalid escape sequences in its docstrings.
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


def _stdlib_imports(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return {
        name for name in names
        if name.split(".")[0] in sys.stdlib_module_names
    }


def test_the_preloaded_set_is_every_import_of_the_corpus(real_corpus) -> None:
    imported: set[str] = set()
    for world in real_corpus.worlds.values():
        imported |= _stdlib_imports(str(world["env_class_code"]))
    for scenario in real_corpus.scenarios:
        for check in scenario.checks or ():
            imported |= _stdlib_imports(str(check["check_func"]))

    # `_strptime` is imported by `strptime` itself and `collections.abc` by
    # `typing` (on 3.12), not by a world.
    assert imported == set(worker.PRELOADED_MODULES) - {"_strptime", "collections.abc"}


def test_every_world_starts_and_grades_under_the_hook(real_corpus) -> None:
    """Each world of every range, on its first scenario: start, then grade."""
    first: dict[str, int] = {}
    for index, scenario in enumerate(real_corpus.scenarios):
        first.setdefault(scenario.env_id, index)
    environment = EnvScalerToolsV2Environment(real_corpus)

    def start(index: int) -> tuple[str, str | None]:
        scenario = real_corpus.scenarios[index]
        world = real_corpus.worlds[scenario.env_id]
        process = WorldProcess()
        try:
            reply = _init(
                process,
                layout.world_source(scenario.env_id, str(world["env_class_code"])),
                str(world["env_class_name"]), scenario.init_config,
            )
            if not reply.get("ok"):
                return scenario.env_id, str(reply.get("error"))
            checks = [str(c["check_func"]) for c in scenario.checks or ()]
            graded = process.request({"op": "grade", "checks": checks})
            if not graded.get("ok"):
                return scenario.env_id, "grading failed"
        finally:
            process.close()
        return scenario.env_id, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        failures = {w: e for w, e in pool.map(start, first.values()) if e is not None}
    assert len(first) == 191
    assert failures == {}
    assert environment.get_task(first["env_178_rl"]).metadata["range"] == "sft_graded"


def _unbound_datetime_attributes(source: str) -> list[str]:
    """``datetime.X`` where X is not on what the module-level name is bound to.

    Upstream binds the name twice in some worlds (``import datetime`` and
    ``from datetime import datetime``); the last binding wins.
    """
    import datetime as real

    tree = _parse(source)
    bound = None
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            (a.asname or a.name) == "datetime" for a in node.names
        ):
            bound = real
        elif isinstance(node, ast.ImportFrom) and node.module == "datetime" and any(
            (a.asname or a.name) == "datetime" for a in node.names
        ):
            bound = real.datetime
    if bound is None:
        return []
    return [
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id == "datetime" and not hasattr(bound, node.attr)
    ]


def test_the_mended_worlds_no_longer_reach_a_missing_attribute(real_corpus) -> None:
    graded = {
        s.env_id for s in real_corpus.scenarios if s.range in layout.GRADED_RANGES
    }
    for world_id in graded:
        world = real_corpus.worlds[world_id]
        upstream = str(world["env_class_code"])
        mended = layout.world_source(world_id, upstream)
        assert _unbound_datetime_attributes(mended) == [], world_id
        assert (mended != upstream) == (world_id in layout.WORLD_SOURCE_FIXES)
        if world_id in layout.WORLD_SOURCE_FIXES:
            assert _unbound_datetime_attributes(upstream), world_id


def test_env_178_reserves_a_bed(real_corpus) -> None:
    """The tool 48 of its 50 graded scenarios need, which upstream raised in."""
    index = next(
        i for i, s in enumerate(real_corpus.scenarios) if s.env_id == "env_178_rl"
    )
    scenario = real_corpus.scenarios[index]
    world = real_corpus.worlds["env_178_rl"]
    config = scenario.init_config
    bed = next(
        bed_id for bed_id, bed in config["beds"].items() if bed["status"] == "available"
    )
    patient = next(iter(config["patients"]))
    staff = next(iter(config["staff"]))
    process = WorldProcess()
    try:
        assert _init(
            process, layout.world_source("env_178_rl", str(world["env_class_code"])),
            str(world["env_class_name"]), config,
        )["ok"]
        reply = process.request({"op": "call", "tool": "reserve_bed", "arguments": {
            "bed_id": bed, "patient_id": patient, "staff_id": staff,
        }})
    finally:
        process.close()
    assert reply["ok"], reply
    assert reply["result"]["success"] is True, reply
