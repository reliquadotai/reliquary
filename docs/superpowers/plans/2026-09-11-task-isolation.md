# Task Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a second task be created and run while the first keeps going, without touching the first task's process, its miners, or its emission.

**Architecture:** A task is a validator process with its own `RELIQUARY_TASK_ID`. Everything that task writes to R2 is namespaced by that id — the archive prefix above all. The local state volume stays shared: the device leases in Task 5 work *because* `<state>/device-leases` is visible to every task on the host, while `fill_active`, `pending_archives`, `control.json` and the cooldown directory sit there unscoped. A task declares its emission share, which is `0.0` for any task other than the legacy one, so creating a task provably cannot dilute the running one. Miners find tasks through a new `/tasks` route and reach a second task with the `--validator-url` the CLI already accepts, so no router is needed. Physical GPUs are leased per task so a new task cannot claim a card the running one qualified.

**Tech Stack:** Python 3.11, pytest, FastAPI (`TestClient`).

**Spec:** `docs/superpowers/specs/2026-09-10-task-scoped-emission-pricing-design.md` (§8, §10, §11, §13)

## Global Constraints

- Same version: no new `protocol_version`, no new profile id, no "v2" anywhere.
- **Creating a task must change nothing for the running one**: no shared key, no shared device, no emission taken from it, no restart, no miner update.
- `RELIQUARY_TASK_ID` defaults to `default`, which keeps the **exact legacy R2 paths** (`reliquary/dataset/window-<n>.json.gz`). No migration of existing archives.
- A task other than `default` defaults to an emission share of `0.0`. Paying a second task is a separate, deliberate act.
- Retuning a *running* task is out of scope. A task's parameters are fixed for its lifetime; different parameters mean a different task.
- Local verification: targeted test files only, run sequentially. The full suite is CI's job (the VPS has no swap). Worktrees outside `/tmp`. Never `git stash`.
- Code comments: one or two lines; rationale goes in the commit message.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq
  ```

## File Map

| File | Responsibility | Tasks |
|---|---|---|
| `reliquary/constants.py` | `TASK_ID`, `TASK_EMISSION_SHARE` | 1, 3 |
| `reliquary/infrastructure/storage.py` | the task's archive prefix and object key | 1 |
| `reliquary/infrastructure/archive_queue.py` | uploads to the same key | 1 |
| `reliquary/validator/service.py` | `task_id` and share in the archive; share as the window pool | 1, 3 |
| `reliquary/validator/weight_only.py` | read every task's archives, merge, replay | 2 |
| `reliquary/validator/server.py` | `GET /tasks` | 4 |
| `reliquary/validator/device_lease.py` (new) | one card, one task | 5 |
| `reliquary/cli/main.py` | take the leases once the cards are resolved | 5 |

---

### Task 1: Give a task its own archive namespace

**Files:**
- Modify: `reliquary/constants.py` (after the `TRAINING_RUN_ID` block)
- Modify: `reliquary/infrastructure/storage.py` (`upload_window_dataset`, `list_recent_datasets`, `list_all_window_keys`)
- Modify: `reliquary/infrastructure/archive_queue.py` (the key built before `_sync_boto3_put`)
- Modify: `reliquary/validator/service.py` (`_archive_window`, both the completed and the aborted payload)
- Test: `tests/unit/test_task_archive_namespace.py`

**Interfaces:**
- Produces: `TASK_ID: str` (`constants.py`); `storage.dataset_prefix(task_id: str | None = None) -> str`; `storage.dataset_object_key(window_start: int, task_id: str | None = None) -> str`; archive field `task_id`.
- `task_id=None` means "read `RELIQUARY_TASK_ID` from the environment", which is how `storage.py` already reads its R2 configuration — it imports no project constants and must keep it that way.

- [ ] **Step 0: Create the worktree**

```bash
git -C /home/ubuntu/catalyst-main worktree add -b feat/task-isolation /home/ubuntu/catalyst-tasks origin/main
cd /home/ubuntu/catalyst-tasks
```

The emission-price branch is not involved: this plan starts from `origin/main`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_task_archive_namespace.py`:

```python
"""Two tasks must never write the same archive key.

The window number is a per-process counter, so without a namespace a second
task would overwrite the first task's window 42 and both would land in the
same weight replay.
"""

from __future__ import annotations

import pytest

from reliquary.infrastructure import storage


def test_the_legacy_task_keeps_its_flat_path(monkeypatch):
    monkeypatch.delenv("RELIQUARY_TASK_ID", raising=False)

    assert storage.dataset_prefix() == "reliquary/dataset/window-"
    assert storage.dataset_object_key(42) == "reliquary/dataset/window-42.json.gz"


def test_default_is_spelled_the_same_as_unset(monkeypatch):
    monkeypatch.setenv("RELIQUARY_TASK_ID", "default")

    assert storage.dataset_object_key(42) == "reliquary/dataset/window-42.json.gz"


def test_a_named_task_writes_under_its_own_prefix(monkeypatch):
    monkeypatch.setenv("RELIQUARY_TASK_ID", "logic-probe")

    assert storage.dataset_prefix() == "reliquary/tasks/logic-probe/dataset/window-"
    assert storage.dataset_object_key(42) == "reliquary/tasks/logic-probe/dataset/window-42.json.gz"


def test_an_explicit_task_beats_the_environment(monkeypatch):
    monkeypatch.setenv("RELIQUARY_TASK_ID", "logic-probe")

    assert storage.dataset_object_key(7, "other") == "reliquary/tasks/other/dataset/window-7.json.gz"


@pytest.mark.parametrize("bad", ["", "../escape", "UPPER", "with space", "a" * 64, "-lead"])
def test_an_unusable_task_id_is_refused(monkeypatch, bad):
    monkeypatch.setenv("RELIQUARY_TASK_ID", bad)

    with pytest.raises(ValueError):
        storage.dataset_prefix()


def test_the_queue_uploads_to_the_same_key(monkeypatch):
    from reliquary.infrastructure import archive_queue

    monkeypatch.setenv("RELIQUARY_TASK_ID", "logic-probe")

    assert archive_queue.upload_key(42) == storage.dataset_object_key(42)


@pytest.mark.asyncio
async def test_the_archive_says_which_task_wrote_it(monkeypatch):
    from unittest.mock import MagicMock, patch

    from tests.unit.test_archive_window_content import (
        _FakeEnv,
        _FakeWallet,
        _valid_submission,
    )
    from reliquary.validator.service import ValidationService

    monkeypatch.setattr("reliquary.validator.service.TASK_ID", "logic-probe")
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99
    service = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=tokenizer,
        env=_FakeEnv(), netuid=99,
    )
    submission = _valid_submission(prompt_idx=7)
    captured: dict = {}

    class _StubQueue:
        def enqueue(self, window, archive):
            captured["archive"] = archive

    class _FakeBatcher:
        window_start = 500
        randomness = "abcd"
        window_opened_at = 0.0
        reject_counts: dict = {}
        rejected_submissions: list = []

        def valid_submissions(self):
            return [submission]

    with patch(
        "reliquary.infrastructure.archive_queue.get_archive_queue",
        return_value=_StubQueue(),
    ):
        await service._archive_window(_FakeBatcher(), [submission])

    assert captured["archive"]["task_id"] == "logic-probe"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_task_archive_namespace.py -q`
Expected: FAIL with `AttributeError: module 'reliquary.infrastructure.storage' has no attribute 'dataset_prefix'`.

- [ ] **Step 3: Add the task id to `constants.py`**

Immediately after the existing `TRAINING_RUN_ID` block:

```python
TRAINING_RUN_ID = (
    _os.environ.get("RELIQUARY_TRAINING_RUN_ID", "default").strip() or "default"
)
```

insert:

```python

# Which task this process serves. "default" keeps the legacy archive paths, so
# the running task is untouched by the existence of any other.
TASK_ID = _os.environ.get("RELIQUARY_TASK_ID", "default").strip() or "default"
```

- [ ] **Step 4: Build the keys in `storage.py`**

Add after the `logger = logging.getLogger(__name__)` line:

```python
_TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


def _task_id(task_id: str | None) -> str:
    """The task whose archives we are addressing. Env-read like the R2 config."""
    resolved = (task_id if task_id is not None else os.getenv("RELIQUARY_TASK_ID", "default")).strip()
    resolved = resolved or "default"
    if not _TASK_ID_RE.match(resolved):
        raise ValueError(f"unusable task id {resolved!r}")
    return resolved


def dataset_prefix(task_id: str | None = None) -> str:
    """Where a task's window archives live. ``default`` keeps the legacy flat path."""
    resolved = _task_id(task_id)
    if resolved == "default":
        return "reliquary/dataset/window-"
    return f"reliquary/tasks/{resolved}/dataset/window-"


def dataset_object_key(window_start: int, task_id: str | None = None) -> str:
    return f"{dataset_prefix(task_id)}{int(window_start)}.json.gz"
```

Add `import re` to the imports at the top of the file.

In `upload_window_dataset`, replace:

```python
    key = f"reliquary/dataset/window-{window_start}.json.gz"
```

with:

```python
    key = dataset_object_key(window_start)
```

In `list_recent_datasets`, replace:

```python
    keys = [
        (w, f"reliquary/dataset/window-{w}.json.gz")
        for w in range(start, current_window)
    ]
```

with:

```python
    keys = [
        (w, dataset_object_key(w, task_id))
        for w in range(start, current_window)
    ]
```

and add `task_id: str | None = None,` to its keyword-only parameters (after `strict: bool = False,`).

In `list_all_window_keys`, replace:

```python
    prefix = "reliquary/dataset/window-"
    pattern = re.compile(r"reliquary/dataset/window-(\d+)\.json\.gz$")
```

with:

```python
    prefix = dataset_prefix(task_id)
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.json\.gz$")
```

add `task_id: str | None = None,` to its keyword-only parameters, and delete the now-duplicated local `import re` inside the function body.

- [ ] **Step 5: Use the same key in the queue**

In `reliquary/infrastructure/archive_queue.py`, add at module level:

```python
def upload_key(window_n: int) -> str:
    """The R2 key this queue uploads to; shared with ``storage`` so they cannot drift."""
    from reliquary.infrastructure.storage import dataset_object_key

    return dataset_object_key(window_n)
```

and replace:

```python
        key = f"reliquary/dataset/window-{window_n}.json.gz"
```

with:

```python
        key = upload_key(window_n)
```

- [ ] **Step 6: Stamp the archive**

In `service.py`, add `TASK_ID` to the `from reliquary.constants import (` list (alphabetically, right before `TRAINING_RUN_ID`), then in `_archive_window` add `"task_id": TASK_ID,` immediately after `"window_status": "completed",` in the completed payload and after `"window_status": "aborted",` in the aborted payload.

- [ ] **Step 7: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_task_archive_namespace.py -q
python -m pytest tests/unit/test_archive_window_content.py tests/unit/test_archive_queue_telemetry.py -q
```

Expected: all pass. The second command proves the legacy task's archives and queue are byte-identical.

- [ ] **Step 8: Commit**

```bash
git add reliquary/constants.py reliquary/infrastructure/storage.py reliquary/infrastructure/archive_queue.py reliquary/validator/service.py tests/unit/test_task_archive_namespace.py
git commit -m "feat(tasks): give each task its own archive namespace

The window number is a per-process counter and the archive path was flat,
so a second task would have overwritten the first task's window 42 and both
would have landed in the same weight replay. Archives now live under
reliquary/tasks/<task_id>/ unless the task is 'default', which keeps the
legacy path byte for byte so nothing existing moves.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---
### Task 2: The weight reader sees every task

**Files:**
- Modify: `reliquary/infrastructure/storage.py` (new `list_task_ids`)
- Modify: `reliquary/validator/weight_only.py` (`submit_once`, `_replay_ema`)
- Test: `tests/unit/test_weight_reader_multi_task.py`

**Interfaces:**
- Consumes: `dataset_prefix`, `dataset_object_key`, `list_all_window_keys(task_id=…)`, `list_recent_datasets(task_id=…)` (Task 1).
- Produces: `storage.list_task_ids(**client_kwargs) -> list[str]` (always contains `"default"`, plus every id under `reliquary/tasks/`); `WeightOnlyValidator._merge_archives(by_task: Mapping[str, list[dict]]) -> list[dict]` ordered by `(window_start, task_id)`.

This must be deployed **before** any second task writes an archive. A reader that does not know about a task simply does not pay it — and a task paying `0.0` (Task 3) changes nothing either way, which is what makes the rollout safe in any order of upgrades.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_weight_reader_multi_task.py`:

```python
"""Adding a task must not move the weights of the task already running."""

from __future__ import annotations

from reliquary.validator.weight_only import WeightOnlyValidator


def _archive(window, task, rewards):
    return {"window_start": window, "task_id": task, "rewards_by_hotkey": rewards}


def test_archives_are_merged_in_window_then_task_order():
    merged = WeightOnlyValidator._merge_archives({
        "logic-probe": [_archive(2, "logic-probe", {}), _archive(1, "logic-probe", {})],
        "default": [_archive(1, "default", {}), _archive(2, "default", {})],
    })

    assert [(a["window_start"], a["task_id"]) for a in merged] == [
        (1, "default"), (1, "logic-probe"), (2, "default"), (2, "logic-probe"),
    ]


def test_a_task_that_pays_nothing_leaves_the_other_untouched():
    alone = [_archive(w, "default", {"hk_a": 1.0}) for w in range(1, 200)]
    beside = WeightOnlyValidator._merge_archives({
        "default": alone,
        "logic-probe": [_archive(w, "logic-probe", {"hk_b": 0.0}) for w in range(1, 200)],
    })

    assert WeightOnlyValidator._replay_ema(beside)["hk_a"] == WeightOnlyValidator._replay_ema(alone)["hk_a"]
    assert "hk_b" not in WeightOnlyValidator._replay_ema(beside)


def test_a_paying_task_does_take_a_share():
    both = WeightOnlyValidator._merge_archives({
        "default": [_archive(w, "default", {"hk_a": 1.0}) for w in range(1, 400)],
        "logic-probe": [_archive(w, "logic-probe", {"hk_b": 1.0}) for w in range(1, 400)],
    })

    ema = WeightOnlyValidator._replay_ema(both)

    # Equal archive rates, equal pools: the two tasks split the mass.
    assert ema["hk_a"] == ema["hk_b"]
    assert abs(ema["hk_a"] + ema["hk_b"] - 1.0) < 1e-6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_weight_reader_multi_task.py -q`
Expected: FAIL with `AttributeError: type object 'WeightOnlyValidator' has no attribute '_merge_archives'`.

- [ ] **Step 3: Enumerate the tasks in `storage.py`**

Add after `dataset_object_key`:

```python
async def list_task_ids(*, strict: bool = False, **client_kwargs) -> list[str]:
    """Every task with an archive namespace, ``default`` always included."""
    from botocore.exceptions import ClientError

    bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "reliquary")
    tasks = {"default"}
    async with get_s3_client(**client_kwargs) as client:
        paginator = client.get_paginator("list_objects_v2")
        try:
            async for page in paginator.paginate(
                Bucket=bucket, Prefix="reliquary/tasks/", Delimiter="/"
            ):
                for entry in page.get("CommonPrefixes", []) or []:
                    candidate = entry.get("Prefix", "")[len("reliquary/tasks/"):].strip("/")
                    if _TASK_ID_RE.match(candidate):
                        tasks.add(candidate)
        except ClientError:
            if strict:
                raise
            logger.exception("list_task_ids failed")
    return sorted(tasks)
```

- [ ] **Step 4: Read and merge every task in `weight_only.py`**

Replace:

```python
        windows = await storage.list_all_window_keys()
        if not windows:
            logger.info("No archives yet; nothing to submit")
            return False

        archives = await storage.list_recent_datasets(
            current_window=max(windows) + 1,
            n=ROLLING_WINDOWS_HISTORY * 3,
        )
        ema = self._replay_ema(archives)
```

with:

```python
        by_task: dict[str, list[dict]] = {}
        for task_id in await storage.list_task_ids():
            windows = await storage.list_all_window_keys(task_id=task_id)
            if not windows:
                continue
            by_task[task_id] = await storage.list_recent_datasets(
                current_window=max(windows) + 1,
                n=ROLLING_WINDOWS_HISTORY * 3,
                task_id=task_id,
            )
        if not by_task:
            logger.info("No archives yet; nothing to submit")
            return False

        archives = self._merge_archives(by_task)
        logger.info(
            "Replaying %d archives across %d task(s): %s",
            len(archives), len(by_task), ", ".join(sorted(by_task)),
        )
        ema = self._replay_ema(archives)
```

Add next to `_replay_ema`:

```python
    @staticmethod
    def _merge_archives(by_task: Mapping[str, list[dict]]) -> list[dict]:
        """One ordered stream out of every task's archives."""
        merged = [
            {**archive, "task_id": archive.get("task_id", task_id)}
            for task_id, archives in by_task.items()
            for archive in archives
        ]
        return sorted(
            merged,
            key=lambda record: (int(record["window_start"]), str(record.get("task_id", ""))),
        )
```

and in `_replay_ema`, replace `key=lambda r: int(r["window_start"])` with
`key=lambda r: (int(r["window_start"]), str(r.get("task_id", "")))`.

Add `from collections.abc import Mapping` to the imports.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_weight_reader_multi_task.py tests/unit/test_task_archive_namespace.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add reliquary/infrastructure/storage.py reliquary/validator/weight_only.py tests/unit/test_weight_reader_multi_task.py
git commit -m "feat(tasks): replay every task's archives into one weight vector

There is one weight vector on chain, so every task's archives have to meet
somewhere; that somewhere is the reader. Tasks are discovered from the
bucket rather than configured, so no weight-only node needs to be told a
task exists. Ship this before any second task writes: a reader that does
not know a task simply does not pay it.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---
### Task 3: A task declares its emission share, and a new one takes nothing

**Files:**
- Modify: `reliquary/constants.py` (after the `TASK_ID` line from Task 1)
- Modify: `reliquary/validator/service.py` (`_build_window_batchers`; `_archive_window`)
- Test: `tests/unit/test_task_emission_share.py`

**Interfaces:**
- Consumes: `TASK_ID` (Task 1).
- Produces: `TASK_EMISSION_SHARE: float` (`constants.py`), archive field `task_emission_share`.

`default` keeps `1.0`, so the running task is untouched. Any other task starts at `0.0`: it fills windows, trains, archives and proves, and pays nobody. That is what makes creating a task safe without deciding anything about emission.

The operator is responsible for keeping the sum of the shares at or below `1.0`; above it, the chain renormalises and the burn silently disappears.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_task_emission_share.py`:

```python
"""A new task pays nobody until someone decides otherwise."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _share(env: dict) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")}
    clean.update(env)
    return subprocess.run(
        [sys.executable, "-c", "import reliquary.constants as c; print(c.TASK_EMISSION_SHARE)"],
        capture_output=True, text=True, env=clean,
    )


def test_the_legacy_task_keeps_the_whole_pool():
    completed = _share({})

    assert completed.returncode == 0, completed.stderr
    assert float(completed.stdout.strip()) == 1.0


def test_a_new_task_takes_nothing_by_default():
    completed = _share({"RELIQUARY_TASK_ID": "logic-probe"})

    assert completed.returncode == 0, completed.stderr
    assert float(completed.stdout.strip()) == 0.0


def test_a_share_can_be_declared():
    completed = _share({"RELIQUARY_TASK_ID": "logic-probe", "RELIQUARY_TASK_EMISSION_SHARE": "0.25"})

    assert float(completed.stdout.strip()) == 0.25


@pytest.mark.parametrize("bad", ["-0.1", "1.5", "nan", "abc"])
def test_an_impossible_share_refuses_to_import(bad):
    completed = _share({"RELIQUARY_TASK_ID": "logic-probe", "RELIQUARY_TASK_EMISSION_SHARE": bad})

    assert completed.returncode != 0
    assert "RELIQUARY_TASK_EMISSION_SHARE" in completed.stderr


def test_a_zero_share_pays_nobody():
    from reliquary.validator.token_rewards import AcceptedGroup, split_environment_pool

    groups = [AcceptedGroup(hotkey="hk", operator_id="hk", eos_tokens=10)]

    assert split_environment_pool(groups, pool=0.0) == {"hk": 0.0}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_task_emission_share.py -q`
Expected: FAIL — `AttributeError: module 'reliquary.constants' has no attribute 'TASK_EMISSION_SHARE'` in the subprocess output.

If `AcceptedGroup(...)` rejects those keyword names, fix the last test to the real constructor signature in `reliquary/validator/token_rewards.py` before moving on; the assertion itself does not change.

- [ ] **Step 3: Declare the share**

In `constants.py`, immediately after the `TASK_ID` line added in Task 1:

```python

# What fraction of the miner emission this task may pay. A task other than the
# legacy one starts at zero: creating it cannot take anything from the one
# already running. Keeping the sum of the shares at or below 1.0 is the
# operator's call -- above it the chain renormalises and the burn disappears.
_TASK_EMISSION_SHARE_RAW = _os.environ.get(
    "RELIQUARY_TASK_EMISSION_SHARE", "1.0" if TASK_ID == "default" else "0.0"
)
try:
    TASK_EMISSION_SHARE = float(_TASK_EMISSION_SHARE_RAW)
except ValueError as _exc:
    raise ValueError(
        f"RELIQUARY_TASK_EMISSION_SHARE={_TASK_EMISSION_SHARE_RAW!r} is not a number"
    ) from _exc
if not 0.0 <= TASK_EMISSION_SHARE <= 1.0 or TASK_EMISSION_SHARE != TASK_EMISSION_SHARE:
    raise ValueError("RELIQUARY_TASK_EMISSION_SHARE must be between 0.0 and 1.0")
```

- [ ] **Step 4: Pay the window with it**

In `service.py`, add `TASK_EMISSION_SHARE` to the `from reliquary.constants import (` list (right after `TASK_ID`), then in `_build_window_batchers` replace:

```python
                window_pool=1.0,
```

with:

```python
                window_pool=TASK_EMISSION_SHARE,
```

and in `_archive_window`, immediately after the `"task_id": TASK_ID,` line added in Task 1, add:

```python
            "task_emission_share": TASK_EMISSION_SHARE,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_task_emission_share.py -q
python -m pytest tests/unit/test_archive_window_content.py tests/unit/test_v1_cutover.py -q
```

Expected: all pass. The second command proves the legacy task still pays exactly as before.

- [ ] **Step 6: Commit**

```bash
git add reliquary/constants.py reliquary/validator/service.py tests/unit/test_task_emission_share.py
git commit -m "feat(tasks): a new task pays nothing until it is given a share

Two tasks writing into the same weight replay split the emission by their
archive rate, so creating a paying task would quietly dilute the running
one. A task other than 'default' therefore starts at a share of 0.0: it
fills windows, trains and archives, and pays nobody. Giving it a share is
a separate, deliberate act.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---
### Task 4: `GET /tasks`, so a miner can find the others

**Files:**
- Modify: `reliquary/validator/server.py` (`_build_app`, right after the `/checkpoint` route)
- Test: `tests/unit/test_tasks_endpoint.py`

**Interfaces:**
- Consumes: `TASK_ID`, `TASK_EMISSION_SHARE` (Tasks 1, 3).
- Produces: `GET /tasks` → `{"tasks": [{"task_id", "profile_id", "model", "emission_share", "window", "url"}]}`. The first entry is always this validator's own task; the rest come from an optional directory file named by `RELIQUARY_TASK_DIRECTORY_PATH`.

The directory is read **at request time**, so adding a task means dropping a line in a file — no restart of the running validator, which is the whole point. A miner reaches a second task with the `--validator-url` the CLI already accepts, so no router is needed yet. A missing or malformed file yields an empty peer list and never an error: a typo must not take down the running task's discovery.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tasks_endpoint.py`:

```python
"""Miners discover the other tasks without anyone restarting this one."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from reliquary.constants import PROTOCOL_PROFILE_ID, TASK_EMISSION_SHARE, TASK_ID
from reliquary.validator.server import ValidatorServer


def test_a_validator_always_lists_its_own_task():
    body = TestClient(ValidatorServer().app).get("/tasks").json()

    assert body["tasks"][0]["task_id"] == TASK_ID
    assert body["tasks"][0]["profile_id"] == PROTOCOL_PROFILE_ID
    assert body["tasks"][0]["emission_share"] == TASK_EMISSION_SHARE
    assert body["tasks"][0]["url"] is None


def test_declared_peers_are_listed(tmp_path, monkeypatch):
    directory = tmp_path / "tasks.json"
    directory.write_text(json.dumps([
        {"task_id": "logic-probe", "url": "http://10.0.0.9:8080"},
    ]))
    monkeypatch.setenv("RELIQUARY_TASK_DIRECTORY_PATH", str(directory))

    tasks = TestClient(ValidatorServer().app).get("/tasks").json()["tasks"]

    assert [t["task_id"] for t in tasks] == [TASK_ID, "logic-probe"]
    assert tasks[1]["url"] == "http://10.0.0.9:8080"


def test_a_peer_added_after_start_shows_up_without_a_restart(tmp_path, monkeypatch):
    directory = tmp_path / "tasks.json"
    directory.write_text("[]")
    monkeypatch.setenv("RELIQUARY_TASK_DIRECTORY_PATH", str(directory))
    client = TestClient(ValidatorServer().app)

    assert len(client.get("/tasks").json()["tasks"]) == 1

    directory.write_text(json.dumps([{"task_id": "logic-probe", "url": "http://10.0.0.9:8080"}]))

    assert len(client.get("/tasks").json()["tasks"]) == 2


def test_a_broken_directory_never_breaks_discovery(tmp_path, monkeypatch):
    directory = tmp_path / "tasks.json"
    directory.write_text("{ not json")
    monkeypatch.setenv("RELIQUARY_TASK_DIRECTORY_PATH", str(directory))

    response = TestClient(ValidatorServer().app).get("/tasks")

    assert response.status_code == 200
    assert [t["task_id"] for t in response.json()["tasks"]] == [TASK_ID]


def test_a_malformed_entry_is_skipped(tmp_path, monkeypatch):
    directory = tmp_path / "tasks.json"
    directory.write_text(json.dumps([
        {"task_id": "ok", "url": "http://10.0.0.9:8080"},
        {"task_id": "missing-url"},
        {"url": "http://10.0.0.10:8080"},
        "nonsense",
    ]))
    monkeypatch.setenv("RELIQUARY_TASK_DIRECTORY_PATH", str(directory))

    tasks = TestClient(ValidatorServer().app).get("/tasks").json()["tasks"]

    assert [t["task_id"] for t in tasks] == [TASK_ID, "ok"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_tasks_endpoint.py -q`
Expected: FAIL — `/tasks` returns 404.

- [ ] **Step 3: Implement the route**

In `_build_app`, immediately after this existing route:

```python
        @app.get("/checkpoint")
        async def checkpoint():
            cp = self._current_checkpoint
            if cp is None:
                raise HTTPException(status_code=404, detail="no_checkpoint")
            return {
                "checkpoint_n": cp.checkpoint_n,
                "repo_id": cp.repo_id,
                "revision": cp.revision,
                "signature": cp.signature,
            }
```

add:

```python

        @app.get("/tasks")
        async def get_tasks():
            import json
            import os

            from reliquary.constants import (
                PROTOCOL_MODEL_ID,
                PROTOCOL_MODEL_REVISION,
                TASK_EMISSION_SHARE,
                TASK_ID,
            )

            batcher = self.active_batcher
            state = getattr(self, "_current_state", None)
            tasks = [{
                "task_id": TASK_ID,
                "profile_id": PROTOCOL_PROFILE_ID,
                "model": {"model_id": PROTOCOL_MODEL_ID, "model_revision": PROTOCOL_MODEL_REVISION},
                "emission_share": TASK_EMISSION_SHARE,
                "url": None,
                "window": {
                    "window_n": batcher.window_start if batcher is not None else None,
                    "state": getattr(state, "value", None if state is None else str(state)),
                },
            }]
            # Read per request: adding a task must not restart this one.
            directory_path = os.environ.get("RELIQUARY_TASK_DIRECTORY_PATH", "").strip()
            if directory_path:
                try:
                    declared = json.loads(open(directory_path, "rb").read())
                except (OSError, ValueError):
                    logger.warning("task directory %s unreadable", directory_path, exc_info=True)
                    declared = []
                if not isinstance(declared, list):
                    declared = []
                for entry in declared:
                    if not isinstance(entry, dict):
                        continue
                    task_id, url = entry.get("task_id"), entry.get("url")
                    if not isinstance(task_id, str) or not isinstance(url, str):
                        continue
                    tasks.append({
                        "task_id": task_id,
                        "profile_id": entry.get("profile_id"),
                        "model": entry.get("model"),
                        "emission_share": entry.get("emission_share"),
                        "url": url,
                        "window": None,
                    })
            return {"tasks": tasks}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run, one command at a time:

```bash
python -m pytest tests/unit/test_tasks_endpoint.py -q
python -m pytest tests/unit/test_validator_server.py -q --deselect tests/unit/test_validator_server.py::test_submission_protocol_stamps_wire_ingress --deselect tests/unit/test_validator_server.py::test_submission_protocol_closes_stalled_fresh_connection
```

Expected: all pass. The two deselected tests already fail on `origin/main`.

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/server.py tests/unit/test_tasks_endpoint.py
git commit -m "feat(tasks): advertise the running task and its declared peers

A miner that wants a second task needs its address, and the CLI already
takes --validator-url, so discovery is all that was missing. The peer file
is read per request: declaring a task is dropping a line in it, with no
restart of the task already serving miners. A missing or broken file lists
no peers rather than failing the route.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---
### Task 5: One card, one task

**Files:**
- Create: `reliquary/validator/device_lease.py`
- Modify: `reliquary/cli/main.py` (right after `proof_device_identities` is resolved)
- Test: `tests/unit/test_device_lease.py`

**Interfaces:**
- Consumes: `ProofDeviceIdentity.device_uuid` (existing, `proof_capacity.py`); `TASK_ID` (Task 1).
- Produces: `DeviceLeaseError(RuntimeError)`; `acquire_device_leases(device_uuids: Sequence[str], *, task_id: str, directory: str | os.PathLike[str]) -> list[Path]`; `release_device_leases(paths: Iterable[Path]) -> None`.

Capacity qualification already binds physical cards by UUID, but nothing stops a second validator from listing the same `cuda:0`: the spawn locks are `threading.Lock`, so they only bind one process. Two tasks on one card meet the allocator cliff at ~88% occupancy, and both fail together. A lease file per card makes that a refusal at startup instead.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_device_lease.py`:

```python
"""A card belongs to one task at a time, across processes."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from reliquary.validator.device_lease import (
    DeviceLeaseError,
    acquire_device_leases,
    release_device_leases,
)

UUID_A = "gpu-0000-aaaa"
UUID_B = "gpu-1111-bbbb"


def test_a_free_card_is_leased(tmp_path):
    paths = acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)

    assert len(paths) == 1
    assert json.loads(paths[0].read_text())["task_id"] == "default"


def test_a_card_held_by_a_live_process_is_refused(tmp_path):
    acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)

    with pytest.raises(DeviceLeaseError, match="default"):
        acquire_device_leases([UUID_A], task_id="logic-probe", directory=tmp_path)


def test_a_lease_left_by_a_dead_process_is_reclaimed(tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (tmp_path / f"{UUID_A}.lease").write_text(
        json.dumps({"task_id": "default", "pid": dead.pid, "ts": 0})
    )

    paths = acquire_device_leases([UUID_A], task_id="logic-probe", directory=tmp_path)

    assert json.loads(paths[0].read_text())["task_id"] == "logic-probe"


def test_releasing_frees_the_card(tmp_path):
    paths = acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)
    release_device_leases(paths)

    acquire_device_leases([UUID_A], task_id="logic-probe", directory=tmp_path)


def test_a_partial_conflict_leaves_no_lease_behind(tmp_path):
    acquire_device_leases([UUID_B], task_id="default", directory=tmp_path)

    with pytest.raises(DeviceLeaseError):
        acquire_device_leases([UUID_A, UUID_B], task_id="logic-probe", directory=tmp_path)

    assert not (tmp_path / f"{UUID_A}.lease").exists()


def test_the_same_task_can_retake_its_own_cards(tmp_path):
    acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)

    acquire_device_leases([UUID_A], task_id="default", directory=tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/unit/test_device_lease.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'reliquary.validator.device_lease'`.

- [ ] **Step 3: Create `reliquary/validator/device_lease.py`**

```python
"""One physical card, one task -- enforced across processes.

Capacity qualification binds cards by UUID, but the proof pool's spawn locks
are per process: nothing stopped two validators listing the same cuda:0.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path


class DeviceLeaseError(RuntimeError):
    """A card is already held by a live process."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _holder(path: Path) -> tuple[str, int] | None:
    """The task and pid holding this card, or None if the lease is stale."""
    try:
        record = json.loads(path.read_text())
        task_id, pid = str(record["task_id"]), int(record["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return (task_id, pid) if _pid_alive(pid) else None


def acquire_device_leases(
    device_uuids: Sequence[str],
    *,
    task_id: str,
    directory: str | os.PathLike[str],
) -> list[Path]:
    """Claim every card for ``task_id``, or claim none and raise."""
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    taken: list[Path] = []
    try:
        for device_uuid in device_uuids:
            path = base / f"{device_uuid}.lease"
            holder = _holder(path) if path.exists() else None
            # Another task's live process blocks. Our own does not, so a rolling
            # restart can retake its own cards.
            if holder is not None and holder[0] != task_id:
                raise DeviceLeaseError(
                    f"card {device_uuid} is held by task {holder[0]!r} (pid {holder[1]})"
                )
            payload = json.dumps(
                {"task_id": task_id, "pid": os.getpid(), "ts": time.time()}
            )
            staging = path.with_suffix(".lease.staging")
            staging.write_text(payload)
            os.replace(staging, path)
            taken.append(path)
    except BaseException:
        release_device_leases(taken)
        raise
    return taken


def release_device_leases(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass
```

- [ ] **Step 4: Take the leases at startup**

In `reliquary/cli/main.py`, replace:

```python
                proof_device_identities = _configured_proof_device_identities(
                    torch
                )
                proof_devices = tuple(
                    identity.device_id for identity in proof_device_identities
                )
```

with:

```python
                proof_device_identities = _configured_proof_device_identities(
                    torch
                )
                if proof_device_identities:
                    from reliquary.constants import TASK_ID
                    from reliquary.validator.device_lease import acquire_device_leases

                    acquire_device_leases(
                        [identity.device_uuid for identity in proof_device_identities],
                        task_id=TASK_ID,
                        directory=os.environ.get(
                            "RELIQUARY_DEVICE_LEASE_DIR", "/var/lib/reliquary/device-leases"
                        ),
                    )
                proof_devices = tuple(
                    identity.device_id for identity in proof_device_identities
                )
```

The leases are deliberately not released at shutdown: a crashed validator leaves a lease whose pid is dead, which the next start reclaims. Releasing on a clean exit would hand a card to another task during a rolling restart.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/unit/test_device_lease.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/device_lease.py reliquary/cli/main.py tests/unit/test_device_lease.py
git commit -m "feat(tasks): lease a physical card to one task at a time

Capacity qualification binds cards by UUID, but the proof pool's spawn
locks are per process, so two validators could list the same cuda:0 and
both start. They would then meet the allocator cliff around 88% occupancy
and fail together. A lease file per card turns that into a refusal at
startup, naming the task and pid already holding it. A lease whose pid is
dead is reclaimed, so a crash does not strand a card.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HDvLVugZtKhXu798BprCeq"
```

---

## Out of this plan

- **Paying a second task**: giving it a share, and the discovered price that would set that share. `docs/superpowers/plans/2026-09-11-emission-price-and-task-contracts.md` covers the price; it runs after this plan.
- **Retuning a running task** without a miner release: same plan, later part.
- **A signed task registry** and a front router: only needed once tasks are published to third parties rather than declared in a file we control.
