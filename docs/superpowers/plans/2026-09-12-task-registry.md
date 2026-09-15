# Task Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define tasks in one R2 object so launching a task is a single write, with `Σ cap ≤ 1` guaranteed inside a compare-and-swap rather than hoped for.

**Architecture:** A pure rule module (`reliquary/shared/task_registry.py`) holds the entry type and the sum invariant with no I/O. A store module does the R2 read-with-ETag and the conditional write, reusing the pattern `reliquary/trainer/publisher.py:511` already runs in production. The validator resolves its own entry at startup, refuses to run on four named conditions, and takes its price parameters and emission cap from it. `RELIQUARY_TASK_EMISSION_SHARE` disappears.

**Tech Stack:** Python 3.11, pytest, aiobotocore/boto3 against Cloudflare R2, typer.

**Spec:** `docs/superpowers/specs/2026-09-12-task-registry-design.md`

## Global Constraints

- `default` must stay byte-for-byte identical: same R2 keys, same archive fields, same window pool, same weights. The proof is that no existing assertion on today's values is deleted or relaxed.
- The archive field keeps the name `task_emission_share`; only its source changes. Renaming it is a reader-side change needing the whole fleet.
- **Never run the full pytest suite.** This box has no swap and runs a production container; a full run OOMs it and has killed a production container here. Targeted files only, one command at a time, sequential, no `xdist`, no `--timeout` (pytest-timeout is not installed).
- ~25 tests already fail on `origin/main`. A failure that also fails there is pre-existing and out of scope.
- **Never use `git stash`**, in any form — checkouts are shared between sessions on this machine and a stash has destroyed uncommitted work here.
- Commit locally only. **Never push, never merge.**
- End every commit message with:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4`
- Keep comments to one or two lines; the reasoning belongs in the commit message.

---

### Task 1: The rule, with no I/O

**Files:**
- Create: `reliquary/shared/task_registry.py`
- Test: `tests/unit/test_task_registry_rule.py`

**Interfaces:**
- Consumes: `normalise_task_id` from `reliquary/shared/task_id.py`.
- Produces: `MECHANISM_RL_DISCOVERED_PRICE: str`; `PRICE_PARAM_FIELDS: tuple[str, ...]`; `RegistryError(ValueError)`; `TaskEntry` (frozen dataclass: `task_id`, `profile_id`, `profile_sha256`, `mechanism`, `params: dict[str, float]`, `status`, `retired_at: int | None`); `parse_registry(raw: bytes) -> dict[str, TaskEntry]`; `render_registry(entries: Mapping[str, TaskEntry]) -> bytes`; `total_cap(entries) -> float`; `validate_registry(entries) -> None`; `add_task(entries, entry) -> dict[str, TaskEntry]`; `retire_task(entries, task_id, retired_at) -> dict[str, TaskEntry]`.

This module is the only place the sum rule exists, exactly as `task_id.py` is the only place the id rule exists. It touches no network, so its tests run in milliseconds.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_task_registry_rule.py`:

```python
"""The sum of the caps is the invariant, and it lives in one place."""

from __future__ import annotations

import json

import pytest

from dataclasses import replace

from reliquary.shared.task_registry import (
    MECHANISM_RL_DISCOVERED_PRICE,
    RegistryError,
    TaskEntry,
    add_task,
    parse_registry,
    render_registry,
    retire_task,
    total_cap,
    validate_registry,
)

PARAMS = {
    "start": 1.0, "decay": 0.99, "rounds_per_step": 1000,
    "deadband": 0.80, "snap": 1.20, "floor": 0.05, "cap": 0.6,
    "median_rounds": 4800,
}


def _entry(task_id: str, cap: float, status: str = "active") -> TaskEntry:
    return TaskEntry(
        task_id=task_id,
        profile_id="qwen3-4b-base-dapo-fill-closed-v6",
        profile_sha256="a" * 64,
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params={**PARAMS, "cap": cap},
        status=status,
        retired_at=None,
    )


def test_caps_summing_to_exactly_one_are_allowed():
    entries = {"a": _entry("a", 0.6), "b": _entry("b", 0.4)}

    validate_registry(entries)

    assert total_cap(entries) == pytest.approx(1.0)


def test_caps_over_one_are_refused():
    entries = {"a": _entry("a", 0.6), "b": _entry("b", 0.5)}

    with pytest.raises(RegistryError, match="1.1"):
        validate_registry(entries)


def test_a_retired_task_still_reserves_its_cap():
    entries = {"a": _entry("a", 0.6, status="retired"), "b": _entry("b", 0.4)}

    assert total_cap(entries) == pytest.approx(1.0)

    with pytest.raises(RegistryError):
        validate_registry(add_task(entries, _entry("c", 0.1)))


def test_adding_a_task_that_would_overflow_is_refused():
    entries = {"a": _entry("a", 0.8)}

    with pytest.raises(RegistryError):
        add_task(entries, _entry("b", 0.3))


def test_adding_a_duplicate_id_is_refused():
    entries = {"a": _entry("a", 0.5)}

    with pytest.raises(RegistryError, match="already"):
        add_task(entries, _entry("a", 0.1))


def test_retiring_marks_the_entry_and_keeps_the_cap():
    entries = retire_task({"a": _entry("a", 0.5)}, "a", retired_at=12345)

    assert entries["a"].status == "retired"
    assert entries["a"].retired_at == 12345
    assert total_cap(entries) == pytest.approx(0.5)


@pytest.mark.parametrize("cap", [-0.1, 1.5, True])
def test_an_impossible_cap_is_refused(cap):
    with pytest.raises(RegistryError):
        add_task({}, _entry("a", cap))


def test_an_unknown_mechanism_is_refused():
    broken = replace(_entry("a", 0.5), mechanism="vibes")

    with pytest.raises(RegistryError, match="vibes"):
        add_task({}, broken)


def test_a_missing_price_parameter_is_refused():
    broken = replace(_entry("a", 0.5), params={"cap": 0.5})

    with pytest.raises(RegistryError, match="start"):
        add_task({}, broken)


def test_round_trip_is_canonical():
    entries = {"b": _entry("b", 0.4), "a": _entry("a", 0.6)}

    raw = render_registry(entries)

    assert parse_registry(raw) == entries
    assert list(json.loads(raw)["tasks"]) == ["a", "b"]


def test_a_registry_that_is_not_json_is_refused():
    with pytest.raises(RegistryError):
        parse_registry(b"{ not json")


def test_an_empty_registry_parses_to_nothing():
    assert parse_registry(render_registry({})) == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_task_registry_rule.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'reliquary.shared.task_registry'`.

- [ ] **Step 3: Create `reliquary/shared/task_registry.py`**

```python
"""The one rule for what tasks may exist and what they may cost.

Kept free of I/O so the sum invariant is testable without R2, the same way
``task_id`` keeps the id rule in one dependency-light place.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from reliquary.shared.task_id import normalise_task_id

REGISTRY_VERSION = 1
MECHANISM_RL_DISCOVERED_PRICE = "rl-discovered-price"
KNOWN_MECHANISMS = frozenset({MECHANISM_RL_DISCOVERED_PRICE})

# Every field PriceParams needs. A missing one is refused rather than defaulted:
# a half-specified controller is not a controller.
PRICE_PARAM_FIELDS = (
    "start", "decay", "rounds_per_step", "deadband",
    "snap", "floor", "cap", "median_rounds",
)

# Float addition of exact decimals is not exact; 1.0 must not fail by 1e-16.
_SUM_TOLERANCE = 1e-9


class RegistryError(ValueError):
    """The registry, or a change to it, breaks the rule."""


@dataclass(frozen=True, slots=True)
class TaskEntry:
    task_id: str
    profile_id: str
    profile_sha256: str
    mechanism: str
    params: Mapping[str, float]
    status: str
    retired_at: int | None


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryError(f"{field} must be a number, got {value!r}")
    return float(value)


def validate_entry(entry: TaskEntry) -> None:
    """Everything checkable about one entry without reading the image or R2."""
    normalise_task_id(entry.task_id)
    if entry.mechanism not in KNOWN_MECHANISMS:
        raise RegistryError(f"unknown incentive mechanism {entry.mechanism!r}")
    if entry.status not in {"active", "retired"}:
        raise RegistryError(f"unknown status {entry.status!r}")
    missing = [f for f in PRICE_PARAM_FIELDS if f not in entry.params]
    if missing:
        raise RegistryError(f"missing price parameters: {', '.join(missing)}")
    cap = _number(entry.params["cap"], "cap")
    if not 0.0 <= cap <= 1.0:
        raise RegistryError(f"cap must be between 0.0 and 1.0, got {cap}")
    floor = _number(entry.params["floor"], "floor")
    if floor > cap:
        raise RegistryError(f"floor {floor} exceeds cap {cap}")


def total_cap(entries: Mapping[str, TaskEntry]) -> float:
    """Every entry counts, retired included: a retired task keeps paying while
    its EMA decays, so its budget is not free yet."""
    return sum(float(e.params["cap"]) for e in entries.values())


def validate_registry(entries: Mapping[str, TaskEntry]) -> None:
    for entry in entries.values():
        validate_entry(entry)
    total = total_cap(entries)
    if total > 1.0 + _SUM_TOLERANCE:
        raise RegistryError(
            f"declared caps total {total:.4f}, above the single available pool "
            f"of 1.0; retire a task or lower a cap first"
        )


def add_task(
    entries: Mapping[str, TaskEntry], entry: TaskEntry
) -> dict[str, TaskEntry]:
    validate_entry(entry)
    if entry.task_id in entries:
        raise RegistryError(f"task {entry.task_id!r} already exists")
    merged = {**entries, entry.task_id: entry}
    validate_registry(merged)
    return merged


def retire_task(
    entries: Mapping[str, TaskEntry], task_id: str, retired_at: int
) -> dict[str, TaskEntry]:
    if task_id not in entries:
        raise RegistryError(f"task {task_id!r} is not in the registry")
    retired = replace(entries[task_id], status="retired", retired_at=int(retired_at))
    return {**entries, task_id: retired}


def parse_registry(raw: bytes) -> dict[str, TaskEntry]:
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise RegistryError(f"registry is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise RegistryError("registry must be an object")
    tasks = document.get("tasks", {})
    if not isinstance(tasks, dict):
        raise RegistryError("registry 'tasks' must be an object")
    entries: dict[str, TaskEntry] = {}
    for task_id, body in tasks.items():
        if not isinstance(body, dict):
            raise RegistryError(f"task {task_id!r} is not an object")
        incentive = body.get("incentive")
        if not isinstance(incentive, dict):
            raise RegistryError(f"task {task_id!r} has no incentive block")
        params = incentive.get("params")
        if not isinstance(params, dict):
            raise RegistryError(f"task {task_id!r} has no incentive parameters")
        entries[task_id] = TaskEntry(
            task_id=task_id,
            profile_id=str(body.get("profile_id", "")),
            profile_sha256=str(body.get("profile_sha256", "")),
            mechanism=str(incentive.get("mechanism", "")),
            params=dict(params),
            status=str(body.get("status", "active")),
            retired_at=body.get("retired_at"),
        )
    for entry in entries.values():
        validate_entry(entry)
    return entries


def render_registry(entries: Mapping[str, TaskEntry]) -> bytes:
    """Canonical bytes: sorted keys, so two writers produce the same object."""
    document = {
        "registry_version": REGISTRY_VERSION,
        "tasks": {
            task_id: {
                "profile_id": entry.profile_id,
                "profile_sha256": entry.profile_sha256,
                "incentive": {
                    "mechanism": entry.mechanism,
                    "params": dict(entry.params),
                },
                "status": entry.status,
                "retired_at": entry.retired_at,
            }
            for task_id, entry in sorted(entries.items())
        },
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_task_registry_rule.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add reliquary/shared/task_registry.py tests/unit/test_task_registry_rule.py
git commit -m "feat(tasks): one rule for what tasks may exist and what they cost

The sum of the declared caps is the chain invariant in disguise: above 1
the chain renormalises and the burn disappears in silence. Keeping the
rule in one dependency-light module -- as task_id does for ids -- means
it is testable without R2 and cannot drift between the writer that adds
a task and the validator that refuses to run under a broken registry.

A retired task keeps reserving its cap, because the weight EMA keeps
paying it for hours after it stops producing.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4"
```

---

### Task 2: The registry object, written compare-and-swap

**Files:**
- Create: `reliquary/infrastructure/task_registry_store.py`
- Test: `tests/unit/test_task_registry_store.py`

**Interfaces:**
- Consumes: `parse_registry`, `render_registry`, `add_task`, `retire_task`, `validate_registry`, `TaskEntry`, `RegistryError` (Task 1); `get_s3_client` from `reliquary/infrastructure/storage.py:78`.
- Produces: `REGISTRY_KEY: str`; `RegistryConflict(RuntimeError)`; `async read_registry(**client_kwargs) -> tuple[dict[str, TaskEntry], str | None]`; `async write_registry(entries, etag, **client_kwargs) -> str | None`; `async create_task(entry, *, attempts=5, **client_kwargs) -> None`; `async retire_task_entry(task_id, retired_at, *, attempts=5, **client_kwargs) -> None`.

The compare-and-swap is the whole guarantee. `reliquary/trainer/publisher.py:511` already does the conditional put against this endpoint in production; what it does **not** do is handle the precondition failure, and here that handling is the point: the loser of a race must re-read and **recompute the sum against the winner's entry** before retrying.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_task_registry_store.py`:

```python
"""Two launches racing must not both succeed."""

from __future__ import annotations

import pytest

from reliquary.shared.task_registry import (
    MECHANISM_RL_DISCOVERED_PRICE,
    RegistryError,
    TaskEntry,
    parse_registry,
    render_registry,
)
from reliquary.infrastructure import task_registry_store as store

PARAMS = {
    "start": 1.0, "decay": 0.99, "rounds_per_step": 1000,
    "deadband": 0.80, "snap": 1.20, "floor": 0.05, "cap": 0.6,
    "median_rounds": 4800,
}


def _entry(task_id: str, cap: float) -> TaskEntry:
    return TaskEntry(
        task_id=task_id,
        profile_id="qwen3-4b-base-dapo-fill-closed-v6",
        profile_sha256="a" * 64,
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params={**PARAMS, "cap": cap},
        status="active",
        retired_at=None,
    )


class FakeR2:
    """An object store with ETags and real conditional-put semantics."""

    def __init__(self, body: bytes | None = None):
        self.body = body
        self.etag = '"v1"' if body is not None else None
        self.puts = 0
        self.steal_once: bytes | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_object(self, Bucket, Key):
        if self.body is None:
            raise _client_error("NoSuchKey")

        class _Body:
            def __init__(self, data):
                self._data = data

            async def read(self):
                return self._data

        return {"Body": _Body(self.body), "ETag": self.etag}

    async def put_object(self, Bucket, Key, Body, **condition):
        # A concurrent writer lands between our read and our write, exactly once.
        if self.steal_once is not None:
            self.body, self.etag, self.steal_once = self.steal_once, '"v9"', None
            raise _client_error("PreconditionFailed")
        if "IfMatch" in condition and condition["IfMatch"] != self.etag:
            raise _client_error("PreconditionFailed")
        if "IfNoneMatch" in condition and self.body is not None:
            raise _client_error("PreconditionFailed")
        self.puts += 1
        self.body = Body
        self.etag = f'"v{self.puts + 1}"'
        return {"ETag": self.etag}


def _client_error(code: str):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code}}, "PutObject")


@pytest.fixture
def fake(monkeypatch):
    holder = {}

    def _install(body: bytes | None = None) -> FakeR2:
        client = FakeR2(body)
        holder["client"] = client
        monkeypatch.setattr(store, "get_s3_client", lambda **kw: client)
        return client

    return _install


@pytest.mark.asyncio
async def test_an_absent_registry_reads_as_empty(fake):
    fake(None)

    entries, etag = await store.read_registry()

    assert entries == {}
    assert etag is None


@pytest.mark.asyncio
async def test_creating_the_first_task_writes_the_object(fake):
    client = fake(None)

    await store.create_task(_entry("default", 1.0))

    assert parse_registry(client.body)["default"].params["cap"] == 1.0


@pytest.mark.asyncio
async def test_a_lost_race_recomputes_and_refuses_when_it_no_longer_fits(fake):
    # We read a registry holding 0.5 free, but a rival takes 0.6 first.
    client = fake(render_registry({"a": _entry("a", 0.5)}))
    client.steal_once = render_registry(
        {"a": _entry("a", 0.5), "rival": _entry("rival", 0.4)}
    )

    with pytest.raises(RegistryError):
        await store.create_task(_entry("b", 0.4))


@pytest.mark.asyncio
async def test_a_lost_race_retries_and_succeeds_when_it_still_fits(fake):
    client = fake(render_registry({"a": _entry("a", 0.2)}))
    client.steal_once = render_registry(
        {"a": _entry("a", 0.2), "rival": _entry("rival", 0.2)}
    )

    await store.create_task(_entry("b", 0.2))

    entries = parse_registry(client.body)
    assert set(entries) == {"a", "rival", "b"}


@pytest.mark.asyncio
async def test_retiring_keeps_the_cap_reserved(fake):
    client = fake(render_registry({"a": _entry("a", 0.5)}))

    await store.retire_task_entry("a", retired_at=999)

    entry = parse_registry(client.body)["a"]
    assert entry.status == "retired"
    assert entry.params["cap"] == 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_task_registry_store.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'reliquary.infrastructure.task_registry_store'`.

If `pytest.mark.asyncio` is not recognised, check how other async tests in `tests/unit/` are marked (look at `tests/unit/test_task_archive_namespace.py`) and match that convention exactly rather than adding a plugin.

- [ ] **Step 3: Create `reliquary/infrastructure/task_registry_store.py`**

```python
"""The registry object, and the compare-and-swap that makes its sum a rule.

``trainer/publisher.py`` already writes R2 conditionally in production; the
difference here is that losing the race is expected, and the loser must
recompute the sum against the winner before it retries.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from reliquary.infrastructure.storage import get_s3_client
from reliquary.shared.task_registry import (
    TaskEntry,
    add_task,
    parse_registry,
    render_registry,
    retire_task,
    validate_registry,
)

logger = logging.getLogger(__name__)

REGISTRY_KEY = "reliquary/tasks/registry.json"

_ABSENT_CODES = {"NoSuchKey", "404", "NotFound"}
_CONFLICT_CODES = {"PreconditionFailed", "412", "ConditionalRequestConflict"}


class RegistryConflict(RuntimeError):
    """Too many writers kept winning the race ahead of us."""


def _error_code(exc) -> str:
    return exc.response.get("Error", {}).get("Code", "")


async def read_registry(**client_kwargs) -> tuple[dict[str, TaskEntry], str | None]:
    """The registry and the ETag to write it back against. Absent reads empty."""
    from botocore.exceptions import ClientError

    bucket = client_kwargs.pop("bucket_name", None) or os.getenv(
        "R2_BUCKET_ID", "reliquary"
    )
    async with get_s3_client(**client_kwargs) as client:
        try:
            response = await client.get_object(Bucket=bucket, Key=REGISTRY_KEY)
        except ClientError as exc:
            if _error_code(exc) in _ABSENT_CODES:
                return {}, None
            raise
        body = await response["Body"].read()
        return parse_registry(body), response.get("ETag")


async def write_registry(
    entries: Mapping[str, TaskEntry], etag: str | None, **client_kwargs
) -> str | None:
    """Conditional put. Raises ClientError with a conflict code if we lost."""
    validate_registry(entries)
    bucket = client_kwargs.pop("bucket_name", None) or os.getenv(
        "R2_BUCKET_ID", "reliquary"
    )
    condition = {"IfNoneMatch": "*"} if etag is None else {"IfMatch": etag}
    async with get_s3_client(**client_kwargs) as client:
        response = await client.put_object(
            Bucket=bucket,
            Key=REGISTRY_KEY,
            Body=render_registry(entries),
            **condition,
        )
    return response.get("ETag")


async def _mutate(change, *, attempts: int, **client_kwargs) -> None:
    """Read, apply, write conditionally; on a lost race read again and REAPPLY.

    Re-applying is what enforces the invariant: the change runs against the
    winner's registry, so a task that no longer fits is refused rather than
    written over someone else's budget.
    """
    from botocore.exceptions import ClientError

    for attempt in range(1, attempts + 1):
        entries, etag = await read_registry(**client_kwargs)
        updated = change(entries)
        try:
            await write_registry(updated, etag, **client_kwargs)
            return
        except ClientError as exc:
            if _error_code(exc) not in _CONFLICT_CODES:
                raise
            logger.info(
                "task registry changed under us (attempt %d/%d); re-reading",
                attempt, attempts,
            )
    raise RegistryConflict(
        f"task registry kept changing under us after {attempts} attempts"
    )


async def create_task(entry: TaskEntry, *, attempts: int = 5, **client_kwargs) -> None:
    await _mutate(lambda e: add_task(e, entry), attempts=attempts, **client_kwargs)


async def retire_task_entry(
    task_id: str, retired_at: int, *, attempts: int = 5, **client_kwargs
) -> None:
    await _mutate(
        lambda e: retire_task(e, task_id, retired_at),
        attempts=attempts,
        **client_kwargs,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_task_registry_store.py -q`
Expected: all pass. In particular `test_a_lost_race_recomputes_and_refuses_when_it_no_longer_fits` must raise `RegistryError`, not `RegistryConflict` — a losing writer that no longer fits is refused on the rule, not on the retry budget.

- [ ] **Step 5: Commit**

```bash
git add reliquary/infrastructure/task_registry_store.py tests/unit/test_task_registry_store.py
git commit -m "feat(tasks): write the registry compare-and-swap so the sum is a rule

Checking the sum and then writing is not enough: two launches can both
read a registry with room and both write. Doing the check inside a
conditional put closes that window, and the loser re-reads and re-applies
its change against the winner's registry, so a task that no longer fits
is refused instead of overwriting someone else's budget.

publisher.py already runs this conditional put in production but lets the
precondition failure escape; here the failure is the expected path.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4"
```

---

### Task 3: Refuse to run under a registry that does not describe this task

**Files:**
- Create: `reliquary/validator/task_config.py`
- Test: `tests/unit/test_task_config.py`

**Interfaces:**
- Consumes: `TaskEntry`, `RegistryError`, `MECHANISM_RL_DISCOVERED_PRICE`, `PRICE_PARAM_FIELDS` (Task 1); `PriceParams` from `reliquary/validator/emission_price.py`.
- Produces: `TaskConfigError(RuntimeError)`; `TaskConfig` (frozen dataclass: `task_id`, `entry`, `price_params: PriceParams`, `emission_cap: float`); `resolve_task_config(entries, task_id, *, profile_id, generation_contract) -> TaskConfig`.

Pure, so the four refusals are testable without R2 or a running validator. The I/O and the process exit belong to Task 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_task_config.py`:

```python
"""A validator that cannot find itself in the registry does not run."""

from __future__ import annotations

import pytest

from reliquary.environment.abi import canonical_sha256
from reliquary.shared.task_registry import MECHANISM_RL_DISCOVERED_PRICE, TaskEntry
from reliquary.validator.task_config import TaskConfigError, resolve_task_config

CONTRACT = {"model_id": "demo", "environments": {}}
DIGEST = canonical_sha256(CONTRACT)
PARAMS = {
    "start": 1.0, "decay": 0.99, "rounds_per_step": 1000,
    "deadband": 0.80, "snap": 1.20, "floor": 0.05, "cap": 0.6,
    "median_rounds": 4800,
}


def _entry(**overrides) -> TaskEntry:
    base = dict(
        task_id="default",
        profile_id="demo-profile",
        profile_sha256=DIGEST,
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params=dict(PARAMS),
        status="active",
        retired_at=None,
    )
    return TaskEntry(**{**base, **overrides})


def _resolve(entries, task_id="default"):
    return resolve_task_config(
        entries, task_id, profile_id="demo-profile", generation_contract=CONTRACT
    )


def test_a_declared_task_yields_its_price_parameters():
    config = _resolve({"default": _entry()})

    assert config.emission_cap == 0.6
    assert config.price_params.cap == 0.6
    assert config.price_params.median_rounds == 4800


def test_an_undeclared_task_refuses():
    with pytest.raises(TaskConfigError, match="not declared"):
        _resolve({"other": _entry(task_id="other")}, task_id="default")


def test_an_oversubscribed_registry_refuses():
    entries = {
        "default": _entry(params={**PARAMS, "cap": 0.8}),
        "other": _entry(task_id="other", params={**PARAMS, "cap": 0.5}),
    }

    with pytest.raises(TaskConfigError, match="1.3"):
        _resolve(entries)


def test_a_profile_the_binary_does_not_match_refuses():
    with pytest.raises(TaskConfigError, match="profile"):
        _resolve({"default": _entry(profile_sha256="b" * 64)})


def test_a_profile_id_mismatch_refuses():
    with pytest.raises(TaskConfigError, match="demo-profile"):
        resolve_task_config(
            {"default": _entry()},
            "default",
            profile_id="another-profile",
            generation_contract=CONTRACT,
        )


def test_an_unknown_mechanism_refuses():
    with pytest.raises(TaskConfigError, match="vibes"):
        _resolve({"default": _entry(mechanism="vibes")})


def test_a_retired_task_refuses_to_start():
    with pytest.raises(TaskConfigError, match="retired"):
        _resolve({"default": _entry(status="retired")})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_task_config.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'reliquary.validator.task_config'`.

- [ ] **Step 3: Create `reliquary/validator/task_config.py`**

```python
"""What this process is allowed to be, according to the registry.

Separate from the store so the refusals are testable without R2, and so the
decision to exit the process stays in the CLI where every other fatal startup
condition already lives.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reliquary.environment.abi import canonical_sha256
from reliquary.shared.task_registry import (
    KNOWN_MECHANISMS,
    PRICE_PARAM_FIELDS,
    RegistryError,
    TaskEntry,
    total_cap,
    validate_registry,
)
from reliquary.validator.emission_price import PriceParams


class TaskConfigError(RuntimeError):
    """The registry does not describe a task this binary may run."""


@dataclass(frozen=True, slots=True)
class TaskConfig:
    task_id: str
    entry: TaskEntry
    price_params: PriceParams
    emission_cap: float


def resolve_task_config(
    entries: Mapping[str, TaskEntry],
    task_id: str,
    *,
    profile_id: str,
    generation_contract: Any,
) -> TaskConfig:
    """This task's settings, or a refusal naming exactly what disagrees."""
    try:
        validate_registry(entries)
    except RegistryError as exc:
        raise TaskConfigError(
            f"task registry is unusable ({exc}); declared caps total "
            f"{total_cap(entries):.4f}"
        ) from exc

    entry = entries.get(task_id)
    if entry is None:
        declared = ", ".join(sorted(entries)) or "none"
        raise TaskConfigError(
            f"task {task_id!r} is not declared in the registry (declared: {declared})"
        )
    if entry.status != "active":
        raise TaskConfigError(f"task {task_id!r} is {entry.status}, not active")
    if entry.mechanism not in KNOWN_MECHANISMS:
        raise TaskConfigError(
            f"task {task_id!r} names incentive mechanism {entry.mechanism!r}, "
            f"which this build does not implement"
        )
    if entry.profile_id != profile_id:
        raise TaskConfigError(
            f"task {task_id!r} declares profile {entry.profile_id!r} but this "
            f"process runs {profile_id!r}"
        )
    digest = canonical_sha256(generation_contract)
    if entry.profile_sha256 != digest:
        raise TaskConfigError(
            f"task {task_id!r} pins profile contract {entry.profile_sha256[:12]}… "
            f"but this build computes {digest[:12]}…"
        )

    params = PriceParams(**{f: entry.params[f] for f in PRICE_PARAM_FIELDS})
    return TaskConfig(
        task_id=task_id,
        entry=entry,
        price_params=params,
        emission_cap=float(entry.params["cap"]),
    )
```

`PriceParams` declares `rounds_per_step` and `median_rounds` as `int`; JSON gives them back as `int` already, so no coercion is added. If a test shows a float arriving there, coerce those two fields explicitly rather than loosening the dataclass.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_task_config.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/task_config.py tests/unit/test_task_config.py
git commit -m "feat(tasks): refuse to run under a registry that does not describe us

Four disagreements are fatal, and each says which: the task is not
declared, the caps are oversubscribed, the registry pins a profile this
binary does not build, or it names a mechanism this build cannot run.
Guessing past any of them pays miners under rules nobody agreed to.

Kept pure so the refusals are testable without R2; the process exit stays
in the CLI beside the other fatal startup conditions.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4"
```

---

### Task 4: The window pool comes from the registry, not from an env var

**Files:**
- Modify: `reliquary/constants.py:975-989` (delete the `TASK_EMISSION_SHARE` block)
- Modify: `reliquary/validator/service.py` (`__init__` signature at :711; `window_pool=` at :2471 and :2482; the `task_emission_share` archive field)
- Modify: `reliquary/cli/main.py` (read the registry beside the GPU lease at :686-722; pass `emission_cap` and `price_params` at :881)
- Delete: `tests/unit/test_task_emission_share.py`
- Test: `tests/unit/test_task_pool_from_registry.py`

**Interfaces:**
- Consumes: `read_registry` (Task 2); `resolve_task_config`, `TaskConfig`, `TaskConfigError` (Task 3); `TASK_ID`, `PROTOCOL_PROFILE_ID`, `PROTOCOL_GENERATION_CONTRACT` from `reliquary/constants.py:65,74,973`.
- Produces: `ValidationService(..., emission_cap: float = 1.0, price_params: PriceParams | None = None)`.

`RELIQUARY_TASK_EMISSION_SHARE` disappears entirely. `tests/unit/test_task_emission_share.py` tests an env var that will no longer exist, so it is deleted and replaced — this is the one place in this plan where removing a test is correct, and the replacement must cover the same ground from the registry.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_task_pool_from_registry.py`:

```python
"""What a window pays comes from the registry, and `default` is unchanged."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from reliquary.environment.abi import canonical_sha256
from reliquary.shared.task_registry import MECHANISM_RL_DISCOVERED_PRICE, TaskEntry
from reliquary.validator.task_config import resolve_task_config

PARAMS = {
    "start": 1.0, "decay": 0.99, "rounds_per_step": 1000,
    "deadband": 0.80, "snap": 1.20, "floor": 0.05, "cap": 1.0,
    "median_rounds": 4800,
}


def test_the_emission_share_env_var_is_gone():
    clean = {k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")}
    clean["RELIQUARY_TASK_EMISSION_SHARE"] = "0.25"
    completed = subprocess.run(
        [sys.executable, "-c",
         "import reliquary.constants as c; print(hasattr(c, 'TASK_EMISSION_SHARE'))"],
        capture_output=True, text=True, env=clean,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


def test_the_legacy_task_still_pays_the_whole_pool():
    from reliquary.constants import PROTOCOL_GENERATION_CONTRACT, PROTOCOL_PROFILE_ID

    entry = TaskEntry(
        task_id="default",
        profile_id=PROTOCOL_PROFILE_ID,
        profile_sha256=canonical_sha256(PROTOCOL_GENERATION_CONTRACT),
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params=dict(PARAMS),
        status="active",
        retired_at=None,
    )

    config = resolve_task_config(
        {"default": entry}, "default",
        profile_id=PROTOCOL_PROFILE_ID,
        generation_contract=PROTOCOL_GENERATION_CONTRACT,
    )

    assert config.emission_cap == 1.0


def test_no_module_still_reads_the_removed_constant():
    """The env-var path is gone, not merely unused."""
    import pathlib
    import subprocess

    root = pathlib.Path(__file__).resolve().parents[2] / "reliquary"
    hits = subprocess.run(
        ["grep", "-rn", "TASK_EMISSION_SHARE", str(root)],
        capture_output=True, text=True,
    ).stdout.strip()

    assert hits == "", f"TASK_EMISSION_SHARE still referenced:\n{hits}"
```

The window pool itself is proved unchanged for `default` by the identity
suites in Step 7, which already assert today's payouts end to end; this file
only pins that the constant and its environment variable are gone.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_task_pool_from_registry.py -q`
Expected: FAIL — `constants` still defines `TASK_EMISSION_SHARE`.

- [ ] **Step 3: Delete the constant**

In `reliquary/constants.py`, delete the whole block from the comment `# What fraction of the miner emission this task may pay.` through `raise ValueError("RELIQUARY_TASK_EMISSION_SHARE must be between 0.0 and 1.0")` (currently lines 975-989). Leave the `TASK_ID` line at :973 untouched.

- [ ] **Step 4: Take the cap as a parameter**

In `reliquary/validator/service.py`, add two keyword arguments to `__init__` (after `proof_capacity_qualification`):

```python
        emission_cap: float = 1.0,
        price_params: Any | None = None,
```

and store them:

```python
        # What this task may pay per window. 1.0 is the legacy single-task pool.
        self._emission_cap = float(emission_cap)
        self._price_params = price_params
```

Replace the two `window_pool=TASK_EMISSION_SHARE` sites (`:2471` in the `recovery.begin(...)` call and `:2482` in the `FillClosedBatchAssembler(...)` call) with `window_pool=self._emission_cap`, and remove `TASK_EMISSION_SHARE` from the `from reliquary.constants import (` list. The archive field keeps its name:

```python
            "task_emission_share": self._emission_cap,
```

- [ ] **Step 5: Read the registry at startup**

In `reliquary/cli/main.py`, immediately after the GPU-lease block that ends with `raise typer.Exit(code=3) from exc` (currently :719), add:

```python
                from reliquary.constants import (
                    PROTOCOL_GENERATION_CONTRACT,
                    PROTOCOL_PROFILE_ID,
                    TASK_ID,
                )
                from reliquary.infrastructure.task_registry_store import read_registry
                from reliquary.validator.task_config import (
                    TaskConfigError,
                    resolve_task_config,
                )

                try:
                    registry_entries, _ = asyncio.run(read_registry())
                    task_config = resolve_task_config(
                        registry_entries,
                        TASK_ID,
                        profile_id=PROTOCOL_PROFILE_ID,
                        generation_contract=PROTOCOL_GENERATION_CONTRACT,
                    )
                except TaskConfigError as exc:
                    # Unlike a missing GPU lease, this is not an environment
                    # fault we can run through: we would not know what we are
                    # allowed to pay. 3 is the device lease, 2 is click.
                    logger.critical(
                        "%s; declare it with `reliquary tasks create` before "
                        "starting this validator",
                        exc,
                    )
                    raise typer.Exit(code=4) from exc
                except Exception as exc:
                    logger.critical(
                        "task registry could not be read (%s); refusing to start "
                        "rather than pay under unknown rules",
                        exc,
                    )
                    raise typer.Exit(code=4) from exc
```

and pass it to the service at the `ValidationService(` call (:881):

```python
                emission_cap=task_config.emission_cap,
                price_params=task_config.price_params,
```

If `asyncio.run` cannot be called at that point because an event loop is already running, move the read into the same async context the service is started from and keep the refusal behaviour identical; say so in the report.

- [ ] **Step 6: Delete the superseded test file**

```bash
git rm tests/unit/test_task_emission_share.py
```

- [ ] **Step 7: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_task_pool_from_registry.py -q
python -m pytest tests/unit/test_v1_cutover.py -q
python -m pytest tests/unit/test_archive_window_content.py -q
python -m pytest tests/unit/test_task_archive_namespace.py -q
```

Expected: all pass. The last three are the identity gate: `default` must pay and archive exactly as before. **No assertion in them may be edited to make this pass** — if one fails, the change is wrong, not the test.

- [ ] **Step 8: Commit**

```bash
git add -A reliquary/constants.py reliquary/validator/service.py reliquary/cli/main.py tests/unit/
git commit -m "feat(tasks): take the window pool from the registry, not an env var

A per-box environment variable could diverge silently between processes
and left no trace of who set what. The cap now comes from one versioned
object every task reads, so 'what may this task pay' has a single answer
and the sum across tasks is checkable.

The archive field keeps the name task_emission_share: renaming it is a
reader-side change that would need the whole fleet and buys nothing.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4"
```

---

### Task 5: Launching a task is one command

**Files:**
- Modify: `reliquary/cli/main.py` (add a `tasks` sub-application beside the existing `@app.command()` entries at :311, :514, :940, :947)
- Test: `tests/unit/test_tasks_cli.py`

**Interfaces:**
- Consumes: `create_task`, `retire_task_entry`, `read_registry` (Task 2); `TaskEntry`, `MECHANISM_RL_DISCOVERED_PRICE` (Task 1); `PRODUCTION_PRICE_PARAMS` from `reliquary/validator/emission_price.py`; `PROFILES`, `resolve_protocol_profile` from `reliquary/protocol/profiles.py:714,721`.
- Produces: `reliquary tasks create --task-id X --profile-id Y --cap Z [--start ... --decay ...]`; `reliquary tasks list`; `reliquary tasks retire --task-id X`.

This is the seam a product automates later: the command is a thin shell over `create_task`, and automation calls that function rather than the CLI.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_tasks_cli.py`:

```python
"""Launching a task is writing one entry, and the defaults come from the image."""

from __future__ import annotations

import pytest

from reliquary.cli.main import build_task_entry
from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS


def test_defaults_come_from_the_shipped_controller():
    entry = build_task_entry(
        task_id="logic-probe",
        profile_id="qwen3-4b-base-dapo-fill-closed-v6",
        cap=0.25,
        overrides={},
    )

    assert entry.params["decay"] == PRODUCTION_PRICE_PARAMS.decay
    assert entry.params["median_rounds"] == PRODUCTION_PRICE_PARAMS.median_rounds
    assert entry.params["cap"] == 0.25
    assert entry.status == "active"
    assert len(entry.profile_sha256) == 64


def test_an_override_replaces_one_parameter_only():
    entry = build_task_entry(
        task_id="logic-probe",
        profile_id="qwen3-4b-base-dapo-fill-closed-v6",
        cap=0.25,
        overrides={"start": 0.2},
    )

    assert entry.params["start"] == 0.2
    assert entry.params["decay"] == PRODUCTION_PRICE_PARAMS.decay


def test_an_unknown_profile_is_refused():
    with pytest.raises(ValueError, match="unknown protocol profile"):
        build_task_entry(
            task_id="logic-probe", profile_id="no-such-profile",
            cap=0.25, overrides={},
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/unit/test_tasks_cli.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_task_entry'`.

- [ ] **Step 3: Add the builder and the commands**

In `reliquary/cli/main.py`, at module scope:

```python
def build_task_entry(*, task_id, profile_id, cap, overrides):
    """One registry entry: shipped controller defaults, then explicit overrides."""
    from dataclasses import asdict

    from reliquary.environment.abi import canonical_sha256
    from reliquary.protocol.profiles import resolve_protocol_profile
    from reliquary.shared.task_registry import (
        MECHANISM_RL_DISCOVERED_PRICE,
        TaskEntry,
    )
    from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS

    profile = resolve_protocol_profile(profile_id)
    params = asdict(PRODUCTION_PRICE_PARAMS)
    params.update(overrides)
    params["cap"] = float(cap)
    return TaskEntry(
        task_id=task_id,
        profile_id=profile.profile_id,
        profile_sha256=canonical_sha256(profile.to_generation_contract()),
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params=params,
        status="active",
        retired_at=None,
    )


tasks_app = typer.Typer(name="tasks", help="Declare and retire subnet tasks")
app.add_typer(tasks_app)


@tasks_app.command("create")
def tasks_create(
    task_id: str = typer.Option(..., "--task-id"),
    profile_id: str = typer.Option(..., "--profile-id"),
    cap: float = typer.Option(..., "--cap", help="Most of the pool this task may pay"),
    start: float = typer.Option(None, "--start"),
    decay: float = typer.Option(None, "--decay"),
) -> None:
    from reliquary.infrastructure.task_registry_store import create_task

    overrides = {k: v for k, v in (("start", start), ("decay", decay)) if v is not None}
    entry = build_task_entry(
        task_id=task_id, profile_id=profile_id, cap=cap, overrides=overrides
    )
    asyncio.run(create_task(entry))
    typer.echo(f"declared task {task_id} on {entry.profile_id} with cap {cap}")


@tasks_app.command("list")
def tasks_list() -> None:
    from reliquary.infrastructure.task_registry_store import read_registry
    from reliquary.shared.task_registry import total_cap

    entries, _ = asyncio.run(read_registry())
    for task_id, entry in sorted(entries.items()):
        typer.echo(
            f"{task_id:24s} {entry.status:8s} cap={entry.params['cap']:.3f} "
            f"{entry.profile_id}"
        )
    typer.echo(f"total declared cap: {total_cap(entries):.4f} / 1.0")


@tasks_app.command("retire")
def tasks_retire(
    task_id: str = typer.Option(..., "--task-id"),
    retired_at: int = typer.Option(..., "--retired-at", help="drand round"),
) -> None:
    from reliquary.infrastructure.task_registry_store import retire_task_entry

    asyncio.run(retire_task_entry(task_id, retired_at))
    typer.echo(
        f"retired {task_id}; its cap stays reserved until its EMA tail decays"
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_tasks_cli.py -q
python -m pytest tests/unit/test_cli_environment_override.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add reliquary/cli/main.py tests/unit/test_tasks_cli.py
git commit -m "feat(tasks): declare, list and retire a task from the CLI

Launching a task should be one write, not forty environment variables.
The command is a thin shell over create_task so the automation that
replaces it later calls the same function rather than shelling out, and
the price defaults come from the controller shipped in this image so a
new task starts from something calibrated.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4"
```

---

### Task 6: The weight submitter abstains rather than clamps

**Files:**
- Modify: `reliquary/validator/weight_only.py:281-289` (the clamp)
- Test: `tests/unit/test_weight_reader_multi_task.py` (extend)

**Interfaces:**
- Consumes: `read_registry` (Task 2); `total_cap` (Task 1).
- Produces: no new public names; `submit_once` gains an abstain path.

Today a combined EMA above 1.0 is rescaled with a warning, which quietly pays every miner less. With a registry there is a better answer: if the archives disagree with what was declared, we do not know who should be paid what, and the established rule for that is to abstain — the same rule already applied to a failed listing at `weight_only.py:184`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_weight_reader_multi_task.py`. That file is
synchronous and exercises the validator's static helpers directly — match it,
do not introduce async fixtures:

```python
def test_a_task_with_archives_but_no_registry_entry_is_flagged():
    """Paying a task nobody declared is paying under rules nobody agreed to."""
    undeclared = WeightOnlyValidator._undeclared_tasks(
        {"default": [], "ghost": []}, {"default": object()}
    )

    assert undeclared == ["ghost"]


def test_declared_tasks_are_not_flagged():
    assert WeightOnlyValidator._undeclared_tasks(
        {"a": [], "b": []}, {"a": object(), "b": object()}
    ) == []


def test_every_archived_task_missing_from_the_registry_is_named():
    undeclared = WeightOnlyValidator._undeclared_tasks(
        {"b": [], "a": [], "ok": []}, {"ok": object()}
    )

    assert undeclared == ["a", "b"]
```

The two-task convergence the spec asks for is already covered in this file by
`test_a_paying_task_does_take_a_share`; do not duplicate it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_weight_reader_multi_task.py -q`
Expected: FAIL — the ghost task is paid instead of abstained on.

- [ ] **Step 3: Cross-check the archives against the registry**

Add the rule as a static method on `WeightOnlyValidator`, beside the other
static helpers, so it is testable without driving the async submit path:

```python
    @staticmethod
    def _undeclared_tasks(by_task, declared) -> list[str]:
        """Archived tasks the registry does not know about."""
        return sorted(set(by_task) - set(declared))
```

Then in `submit_once`, after the per-task listing succeeds and before the EMA
is combined:

```python
        try:
            declared, _ = await read_registry()
        except Exception:
            logger.exception("Task registry unreadable; abstaining from this epoch")
            return False
        undeclared = self._undeclared_tasks(by_task, declared)
        if undeclared:
            logger.error(
                "Tasks %s have archives but are not declared in the registry; "
                "abstaining rather than paying under unknown rules",
                undeclared,
            )
            return False
```

Import `read_registry` at module scope from
`reliquary.infrastructure.task_registry_store`, so the test can monkeypatch it
on the module the way the existing suites patch `storage`.

Keep the existing clamp at `:281-289` as the last-resort backstop — it now covers only arithmetic drift within declared caps, which is exactly what a backstop is for. Leave its warning text in place.

- [ ] **Step 4: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_weight_reader_multi_task.py -q
python -m pytest tests/unit/test_weight_only_validator.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/weight_only.py tests/unit/test_weight_reader_multi_task.py
git commit -m "feat(tasks): abstain when the archives disagree with the registry

Rescaling every hotkey because an undeclared task showed up pays the
honest miners of the running task less, silently. The rule this module
already applies to a failed listing is the right one here too: when we
cannot tell who should be paid what, submit nothing and let the next
epoch try. The clamp stays as a backstop for arithmetic drift within
declared caps.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L2Ju5aueTaK3K1WAhGXjB4"
```

---

## Out of this plan

- **Arming the price.** The walk still lives in `self._price_shadow_state`, so a restart resets it to `start` and would hand the whole pool back. It must be seeded from the last archive first.
- **On-chain commitment of the caps** — needed only when third parties launch tasks.
- **A second task type.** `mechanism` makes room for one; none is written.
- **Freeing a retired task's cap automatically** once its EMA tail has decayed. For now the entry stays and the operator removes it by hand when the tail is gone.
- **Scoping `CANDIDATE_MANIFEST_KEY` and stamping `task_id` on training payloads at enqueue** — prerequisites for a second *trainer*.
