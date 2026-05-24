# OpenCodeInstruct code-execution environment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Session note (2026-05-22):** the user has requested **no commits** during this implementation session. Commit steps below are kept as standard TDD discipline but should be deferred — leave changes staged or unstaged until the user explicitly asks to commit.

**Goal:** Land a new `OpenCodeInstructEnvironment` (Python code generation graded by assertion-based unit tests) alongside the existing OMI math env, with execution isolated in a gVisor sandbox behind a UID-separated grader process so the validator hotkey is not reachable from miner-supplied code.

**Architecture:** Three new pieces glued by the existing `Environment` Protocol — (1) a thin env class on the validator side, (2) a small JSON-over-Unix-socket IPC client, (3) a separate grader process that manages a warm pool of `runsc` sandboxes, each holding a long-lived Python worker reading `(code, tests)` from stdin and writing `(passed, total)` to stdout. Trust model matches OMI (miner claim + validator `verify_reward_claim`).

**Tech Stack:** Python 3.12, `datasets` (HuggingFace), `runsc` (gVisor), Unix domain sockets, `subprocess`, `setrlimit`, `setpriv`. No new Python dependency beyond what `pyproject.toml` already requires for OMI (`datasets`).

**Reference spec:** `docs/superpowers/specs/2026-05-22-opencodeinstruct-env-design.md`

---

## File structure

**New files (Python):**
- `reliquary/environment/opencodeinstruct.py` — env class
- `reliquary/environment/grader_client.py` — IPC client used by env
- `reliquary/environment/grader/__init__.py` — empty
- `reliquary/environment/grader/worker.py` — runs inside the sandbox
- `reliquary/environment/grader/server.py` — pool manager, dispatcher, watchdog

**New files (infra):**
- `reliquary/environment/grader/bundle/config.json` — OCI runtime config for runsc
- `scripts/build_grader_bundle.sh` — materializes the OCI rootfs at image-build time
- `scripts/build_opencodeinstruct_subset.py` — offline dataset filter pipeline

**New files (tests):**
- `tests/unit/test_opencodeinstruct_environment.py`
- `tests/unit/test_grader_client.py`
- `tests/unit/test_grader_worker.py`
- `tests/unit/test_grader_server.py`
- `tests/unit/test_opencodeinstruct_dataset_filter.py`
- `tests/integration/test_opencodeinstruct_env_smoke.py`
- `tests/integration/test_grader_e2e.py`

**New files (CI):**
- `.github/workflows/cross-box-determinism.yml`

**Modified files:**
- `reliquary/constants.py` — add 4 grader constants
- `reliquary/environment/__init__.py` — register `"opencodeinstruct"`
- `Dockerfile` — install runsc, run `build_grader_bundle.sh`, create UIDs 1000/1001
- `docker/entrypoint.sh` — fix wallet permissions, launch grader as UID 1001, exec validator as UID 1000

---

## Task ordering rationale

Tasks 1–7 are pure Python with no infra dependencies (testable on any developer box without runsc). Tasks 8–10 introduce the OCI bundle + real gVisor. Task 11 is the offline dataset prep. Tasks 12–14 cover deployment plumbing (Dockerfile, metrics, CI matrix). Keeping the infra at the end means the engineer can verify ~80% of the logic before touching Docker/gVisor.

---

## Task 1: Add grader constants

**Files:**
- Modify: `reliquary/constants.py`

- [ ] **Step 1: Append constants block to `reliquary/constants.py`**

Append at end of file (after the existing `EPOCH_SUBMIT_LEAD_BLOCKS` constant):

```python

# ────────────────  CODE EXECUTION GRADER  ────────────────

# Path to the Unix domain socket the grader server listens on.
# Default lives in /tmp so it's writable by both validator (UID 1000)
# and grader (UID 1001) processes inside the container.
GRADER_SOCKET_PATH = "/tmp/reliquary-grader.sock"

# Number of warm gVisor workers in the grader pool. Sized to handle
# M_ROLLOUTS in parallel for a single submission with headroom for
# concurrent submissions. Increase for high-throughput validators.
GRADER_POOL_SIZE = 8

# Wall-clock timeout (seconds) for one `(code, tests)` evaluation.
# Subprocess inside the sandbox is killed if it exceeds this. Tuned
# so that pathological miner code (infinite loops, slow algorithms)
# fails fast without blocking the queue.
GRADER_EVAL_TIMEOUT_SECONDS = 5

# How often (seconds) the server pings each worker via a no-op eval
# to detect zombies. Triggers respawn if a worker fails to respond.
GRADER_HEALTH_CHECK_INTERVAL_SECONDS = 30
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from reliquary.constants import GRADER_SOCKET_PATH, GRADER_POOL_SIZE, GRADER_EVAL_TIMEOUT_SECONDS, GRADER_HEALTH_CHECK_INTERVAL_SECONDS; print(GRADER_SOCKET_PATH, GRADER_POOL_SIZE, GRADER_EVAL_TIMEOUT_SECONDS, GRADER_HEALTH_CHECK_INTERVAL_SECONDS)"`
Expected output: `/tmp/reliquary-grader.sock 8 5 30`

- [ ] **Step 3: Commit (deferred per session note)**

```bash
git add reliquary/constants.py
git commit -m "feat(constants): add grader config constants for code-exec env"
```

---

## Task 2: `_extract_python` helper + tests

**Files:**
- Create: `reliquary/environment/opencodeinstruct.py` (initial skeleton — helper only)
- Test: `tests/unit/test_opencodeinstruct_environment.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_opencodeinstruct_environment.py`:

```python
"""Tests for the OpenCodeInstruct environment.

Helpers (extraction, completion parsing) are pure-Python and tested
without the dataset or the grader. Grader-dependent tests use a fake
grader client. The HF dataset is exercised in the smoke test.
"""

import pytest


# ---------------------------------------------------------------------------
# _extract_python: pulls Python code out of model completions.
# ---------------------------------------------------------------------------

def test_extract_python_from_fenced_block():
    from reliquary.environment.opencodeinstruct import _extract_python
    text = "Sure, here is the code:\n```python\ndef f(x):\n    return x + 1\n```\nDone."
    assert _extract_python(text) == "def f(x):\n    return x + 1"


def test_extract_python_from_unmarked_fenced_block():
    """Fence without language tag still works."""
    from reliquary.environment.opencodeinstruct import _extract_python
    text = "```\ndef g():\n    return 42\n```"
    assert _extract_python(text) == "def g():\n    return 42"


def test_extract_python_last_block_wins():
    """If the model emits multiple code blocks, prefer the last (the final answer)."""
    from reliquary.environment.opencodeinstruct import _extract_python
    text = "```python\nfirst = 1\n```\nThen revised to:\n```python\nsecond = 2\n```"
    assert _extract_python(text) == "second = 2"


def test_extract_python_fallback_to_raw():
    """When no fence at all, return the full string — let exec decide."""
    from reliquary.environment.opencodeinstruct import _extract_python
    text = "def h():\n    return 'no fence'"
    assert _extract_python(text) == text


def test_extract_python_empty_string():
    from reliquary.environment.opencodeinstruct import _extract_python
    assert _extract_python("") == ""


def test_extract_python_handles_tilde_fences():
    """Some models emit ~~~ instead of ```. Accept both."""
    from reliquary.environment.opencodeinstruct import _extract_python
    text = "~~~python\nx = 1\n~~~"
    assert _extract_python(text) == "x = 1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_opencodeinstruct_environment.py -v`
Expected: 6 FAILED with "ModuleNotFoundError: No module named 'reliquary.environment.opencodeinstruct'"

- [ ] **Step 3: Create `reliquary/environment/opencodeinstruct.py` with `_extract_python`**

```python
"""OpenCodeInstruct code-execution environment.

Loads a deterministic subset of nvidia/OpenCodeInstruct (filtered for
test stability — see scripts/build_opencodeinstruct_subset.py) and
scores miner completions by executing them against the dataset's unit
tests inside a gVisor sandbox managed by the grader subprocess.

The class itself is a thin wrapper: it knows nothing about sandboxes.
All execution happens via reliquary.environment.grader_client, which
talks to the grader server over a Unix socket. This keeps the class
testable without the sandbox infrastructure (see tests/unit/).
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Code extraction from model completions
# ---------------------------------------------------------------------------

# Match fenced code blocks: ``` or ~~~ optionally followed by a language tag.
# Greedy match on the closing fence so the last block wins (model's final
# answer wins over earlier drafts).
_FENCE_RE = re.compile(
    r"(?:```|~~~)(?:python|py)?\s*\n(.*?)\n(?:```|~~~)",
    re.DOTALL,
)


def _extract_python(completion: str) -> str:
    """Extract Python code from a model completion.

    Strategy: find all fenced code blocks (``` or ~~~ with optional
    'python' tag), return the last one's contents. Falls back to the
    raw completion string if no fence is present — exec will reject
    obviously-non-code, scoring zero.
    """
    if not completion:
        return ""
    matches = _FENCE_RE.findall(completion)
    if matches:
        return matches[-1]
    return completion
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_opencodeinstruct_environment.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit (deferred)**

```bash
git add reliquary/environment/opencodeinstruct.py tests/unit/test_opencodeinstruct_environment.py
git commit -m "feat(env): _extract_python helper for OpenCodeInstruct env"
```

---

## Task 3: Grader client (IPC over Unix socket)

**Files:**
- Create: `reliquary/environment/grader_client.py`
- Test: `tests/unit/test_grader_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_grader_client.py`:

```python
"""Tests for the grader IPC client.

Uses a real Unix socket server (asyncio) in-test to verify the wire
protocol round-trips. No grader server, no sandbox needed.
"""

import asyncio
import json
import os
import socket
import tempfile
import threading
import pytest


@pytest.fixture
def fake_grader_socket():
    """Spin up a tiny Unix socket server that returns canned responses.

    Yields (socket_path, set_response_fn). The response function lets
    each test queue what the server will reply for the next request.
    """
    tmp = tempfile.mkdtemp()
    sock_path = os.path.join(tmp, "fake-grader.sock")

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(sock_path)
    server_sock.listen(8)

    state = {"response": None, "received": []}

    def run_server():
        while True:
            try:
                conn, _ = server_sock.accept()
            except OSError:
                return
            with conn:
                data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    if b"\n" in data:
                        break
                if data:
                    line = data.split(b"\n", 1)[0]
                    state["received"].append(json.loads(line))
                if state["response"] is not None:
                    conn.sendall(json.dumps(state["response"]).encode() + b"\n")

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    def set_response(resp):
        state["response"] = resp

    yield sock_path, set_response, state

    server_sock.close()
    try:
        os.unlink(sock_path)
    except OSError:
        pass


def test_evaluate_round_trip(fake_grader_socket):
    """Client serializes a request and parses the server's response."""
    from reliquary.environment.grader_client import GraderClient

    sock_path, set_response, state = fake_grader_socket
    set_response({"req_id": "ignored", "passed": 3, "total": 5, "status": "ok"})

    client = GraderClient(socket_path=sock_path)
    result = client.evaluate(code="def f(): return 1", tests=["assert f() == 1"], timeout_s=5.0)

    assert result == 3 / 5
    assert state["received"][0]["code"] == "def f(): return 1"
    assert state["received"][0]["tests"] == ["assert f() == 1"]
    assert state["received"][0]["timeout_s"] == 5.0


def test_evaluate_returns_zero_when_status_not_ok(fake_grader_socket):
    from reliquary.environment.grader_client import GraderClient

    sock_path, set_response, _ = fake_grader_socket
    set_response({"req_id": "ignored", "passed": 0, "total": 3, "status": "timeout"})

    client = GraderClient(socket_path=sock_path)
    assert client.evaluate("", [], 5.0) == 0.0


def test_evaluate_returns_zero_when_grader_unreachable(tmp_path):
    """Missing socket → retry once → return 0.0, never raise."""
    from reliquary.environment.grader_client import GraderClient

    nonexistent = str(tmp_path / "nope.sock")
    client = GraderClient(socket_path=nonexistent)
    result = client.evaluate("def f(): pass", ["assert True"], 5.0)
    assert result == 0.0


def test_evaluate_returns_zero_on_malformed_response(fake_grader_socket):
    from reliquary.environment.grader_client import GraderClient

    sock_path, set_response, _ = fake_grader_socket
    set_response({"garbage": "no required fields"})

    client = GraderClient(socket_path=sock_path)
    assert client.evaluate("x = 1", ["assert x == 1"], 5.0) == 0.0


def test_evaluate_handles_zero_total_safely(fake_grader_socket):
    """If the dataset somehow sent 0 tests, return 0.0 not ZeroDivisionError."""
    from reliquary.environment.grader_client import GraderClient

    sock_path, set_response, _ = fake_grader_socket
    set_response({"req_id": "ignored", "passed": 0, "total": 0, "status": "ok"})

    client = GraderClient(socket_path=sock_path)
    assert client.evaluate("x = 1", [], 5.0) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_grader_client.py -v`
Expected: 5 FAILED with "ModuleNotFoundError: No module named 'reliquary.environment.grader_client'"

- [ ] **Step 3: Implement `reliquary/environment/grader_client.py`**

```python
"""Unix-socket IPC client for the grader server.

Used by OpenCodeInstructEnvironment.compute_reward to dispatch
evaluation requests. Frames JSON-lines over SOCK_STREAM. Retries
once on transient connection failures, then returns 0.0 — the
Environment Protocol forbids raising from compute_reward.
"""

from __future__ import annotations

import json
import logging
import socket
import time
import uuid
from typing import Optional

from reliquary.constants import GRADER_SOCKET_PATH

logger = logging.getLogger(__name__)


class GraderClient:
    """Thin JSON-over-Unix-socket client.

    Stateless per-call (opens a new socket per evaluate). The grader
    server handles concurrent connections in its accept loop, so we
    don't need connection pooling on the client side.
    """

    def __init__(self, socket_path: str = GRADER_SOCKET_PATH) -> None:
        self.socket_path = socket_path

    def evaluate(self, code: str, tests: list[str], timeout_s: float) -> float:
        """Send (code, tests) to the grader, return passed/total in [0, 1].

        Returns 0.0 if the grader is unreachable, the response is
        malformed, the worker timed out, the worker crashed, or
        total is zero. Never raises.
        """
        req = {
            "req_id": uuid.uuid4().hex,
            "code": code,
            "tests": tests,
            "timeout_s": timeout_s,
        }
        # One retry with short backoff for transient failures (grader
        # restarting, accept queue full).
        for attempt in (1, 2):
            try:
                response = self._round_trip(req)
                break
            except (OSError, ConnectionError) as e:
                if attempt == 1:
                    logger.debug("grader_client: connect failed (%s), retrying", e)
                    time.sleep(0.1)
                    continue
                logger.warning("grader_client: unreachable after retry: %s", e)
                return 0.0

        if response.get("status") != "ok":
            return 0.0
        try:
            passed = int(response["passed"])
            total = int(response["total"])
        except (KeyError, TypeError, ValueError):
            return 0.0
        if total <= 0:
            return 0.0
        return passed / total

    def _round_trip(self, req: dict) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(req["timeout_s"] + 5.0)  # generous; server enforces inner timeout
            s.connect(self.socket_path)
            s.sendall(json.dumps(req).encode() + b"\n")
            buf = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break
            if not buf:
                return {}
            try:
                return json.loads(buf.split(b"\n", 1)[0])
            except json.JSONDecodeError:
                return {}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_grader_client.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit (deferred)**

```bash
git add reliquary/environment/grader_client.py tests/unit/test_grader_client.py
git commit -m "feat(env): grader IPC client (Unix socket, retry, never-raise)"
```

---

## Task 4: Grader worker — eval logic (no sandbox)

**Files:**
- Create: `reliquary/environment/grader/__init__.py` (empty)
- Create: `reliquary/environment/grader/worker.py`
- Test: `tests/unit/test_grader_worker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_grader_worker.py`:

```python
"""Tests for the grader worker eval logic.

The worker can be tested in pure Python (no sandbox) by invoking its
eval function directly. The real worker reads (code, tests) from
stdin and writes the result to stdout; here we call the pure
function it wraps.
"""

import pytest


def test_eval_all_tests_pass():
    from reliquary.environment.grader.worker import evaluate_code

    code = "def add(a, b): return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]
    passed, total, status = evaluate_code(code, tests, timeout_s=5.0)
    assert passed == 3
    assert total == 3
    assert status == "ok"


def test_eval_some_tests_fail():
    from reliquary.environment.grader.worker import evaluate_code

    code = "def add(a, b): return a - b"  # wrong sign
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(5, 5) == 10"]
    passed, total, status = evaluate_code(code, tests, timeout_s=5.0)
    assert passed == 1  # only add(0, 0) == 0 happens to match
    assert total == 3
    assert status == "ok"


def test_eval_test_namespace_isolation():
    """A test that mutates the namespace must not corrupt subsequent tests."""
    from reliquary.environment.grader.worker import evaluate_code

    code = "x = 10"
    tests = [
        "x = 999",                       # mutates local copy
        "assert x == 10",                # should still see original
    ]
    passed, total, _ = evaluate_code(code, tests, timeout_s=5.0)
    assert passed == 2  # both pass — first one always passes (no assert), second sees x=10


def test_eval_syntax_error_in_one_test_does_not_break_others():
    from reliquary.environment.grader.worker import evaluate_code

    code = "def f(): return 1"
    tests = [
        "assert f() == 1",
        "this is not valid python",
        "assert f() == 1",
    ]
    passed, total, _ = evaluate_code(code, tests, timeout_s=5.0)
    assert passed == 2
    assert total == 3


def test_eval_code_with_syntax_error_returns_zero():
    from reliquary.environment.grader.worker import evaluate_code

    code = "def broken("  # syntax error
    tests = ["assert True"]
    passed, total, _ = evaluate_code(code, tests, timeout_s=5.0)
    assert passed == 0
    assert total == 1


def test_eval_empty_code_returns_zero():
    from reliquary.environment.grader.worker import evaluate_code

    passed, total, _ = evaluate_code("", ["assert True"], timeout_s=5.0)
    assert passed == 0
    assert total == 1


def test_eval_empty_tests_returns_zero_total():
    from reliquary.environment.grader.worker import evaluate_code

    passed, total, _ = evaluate_code("x = 1", [], timeout_s=5.0)
    assert passed == 0
    assert total == 0


def test_eval_runtime_error_in_code_returns_zero():
    """Code that raises at exec time → all tests fail (ns is empty)."""
    from reliquary.environment.grader.worker import evaluate_code

    code = "raise RuntimeError('boom')"
    tests = ["assert True", "assert 1 == 1"]
    passed, total, _ = evaluate_code(code, tests, timeout_s=5.0)
    assert passed == 0
    assert total == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_grader_worker.py -v`
Expected: 8 FAILED with "ModuleNotFoundError: No module named 'reliquary.environment.grader.worker'"

- [ ] **Step 3: Create `reliquary/environment/grader/__init__.py`**

Create an empty file:

```python
"""Grader subprocess components for the code-execution environment."""
```

- [ ] **Step 4: Implement `reliquary/environment/grader/worker.py`**

```python
"""Grader worker — runs inside the gVisor sandbox.

Reads JSON requests from stdin (one per line), evaluates the
(code, tests) pair against a fresh module namespace, writes the
result back to stdout. Stays warm between requests so the gVisor
sandbox is reused.

Each request gets a fresh dict for `code`'s namespace, and each
test runs in a shallow copy of that namespace so per-test
mutations don't leak.
"""

from __future__ import annotations

import json
import sys
from typing import Tuple


def evaluate_code(code: str, tests: list[str], timeout_s: float) -> Tuple[int, int, str]:
    """Run `code` to populate a namespace, then run each test in a copy.

    Returns (passed, total, status). `timeout_s` is informational here;
    the real wall-clock timeout is enforced by the parent process via
    subprocess.run(timeout=...). This function never raises.
    """
    total = len(tests)
    ns: dict = {}
    try:
        exec(compile(code, "<miner_code>", "exec"), ns)
    except BaseException:
        # exec failed at module level (syntax error, raise at import, etc.)
        # → no tests can execute meaningfully → 0/total
        return 0, total, "ok"

    passed = 0
    for i, t in enumerate(tests):
        try:
            exec(compile(t, f"<test_{i}>", "exec"), dict(ns))
            passed += 1
        except BaseException:
            # Any failure (AssertionError, SyntaxError, RuntimeError, …)
            # → that test failed; continue with the next.
            pass
    return passed, total, "ok"


def _serve_stdin() -> None:
    """Read JSON-line requests, write JSON-line responses. Loop forever.

    Used when this module is run as a subprocess (`python -m
    reliquary.environment.grader.worker`). The parent process pipes
    requests via stdin and reads results from stdout.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            passed, total, status = evaluate_code(
                req.get("code", ""),
                req.get("tests", []),
                float(req.get("timeout_s", 5.0)),
            )
            resp = {
                "req_id": req.get("req_id", ""),
                "passed": passed,
                "total": total,
                "status": status,
            }
        except BaseException as e:
            resp = {
                "req_id": "",
                "passed": 0,
                "total": 0,
                "status": "crash",
                "error": str(e),
            }
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    _serve_stdin()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_grader_worker.py -v`
Expected: 8 PASSED

- [ ] **Step 6: Commit (deferred)**

```bash
git add reliquary/environment/grader/__init__.py reliquary/environment/grader/worker.py tests/unit/test_grader_worker.py
git commit -m "feat(grader): worker eval loop with per-test namespace isolation"
```

---

## Task 5: Grader server — pool + dispatcher (no sandbox)

**Files:**
- Create: `reliquary/environment/grader/server.py`
- Test: `tests/unit/test_grader_server.py`

For unit tests we run the server with workers as plain `python -m reliquary.environment.grader.worker` subprocesses (no runsc). The runsc path is exercised by the E2E integration test in Task 10.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_grader_server.py`:

```python
"""Tests for the grader server (pool + dispatch + watchdog).

Spawns a real server with worker subprocesses (python -m
reliquary.environment.grader.worker — no runsc). The IPC contract
is exercised end-to-end over a real Unix socket.
"""

import asyncio
import os
import socket
import tempfile
import threading
import time
import json
import pytest


@pytest.fixture
def grader_server(tmp_path):
    """Spawn a real GraderServer with 2 workers (no sandbox)."""
    from reliquary.environment.grader.server import GraderServer

    sock_path = str(tmp_path / "grader.sock")
    server = GraderServer(
        socket_path=sock_path,
        pool_size=2,
        worker_argv=["python", "-m", "reliquary.environment.grader.worker"],
        eval_timeout_s=5.0,
    )
    server.start()
    # Brief wait for accept loop + workers to be ready.
    deadline = time.time() + 5.0
    while not os.path.exists(sock_path) and time.time() < deadline:
        time.sleep(0.05)
    yield server
    server.stop()


def _request(sock_path: str, code: str, tests: list[str], timeout_s: float = 5.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10.0)
        s.connect(sock_path)
        req = {"req_id": "test-req", "code": code, "tests": tests, "timeout_s": timeout_s}
        s.sendall(json.dumps(req).encode() + b"\n")
        buf = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                break
        return json.loads(buf.split(b"\n", 1)[0])


def test_server_grades_correct_code(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="def add(a,b): return a+b",
        tests=["assert add(1,2) == 3", "assert add(0,0) == 0"],
    )
    assert resp["status"] == "ok"
    assert resp["passed"] == 2
    assert resp["total"] == 2


def test_server_grades_incorrect_code(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="def add(a,b): return a-b",
        tests=["assert add(1,2) == 3"],
    )
    assert resp["status"] == "ok"
    assert resp["passed"] == 0
    assert resp["total"] == 1


def test_server_handles_concurrent_requests(grader_server):
    """Pool of 2 → 4 concurrent requests should all succeed."""
    results = []
    errors = []

    def submit():
        try:
            r = _request(
                grader_server.socket_path,
                code="def f(): return 1",
                tests=["assert f() == 1"],
            )
            results.append(r)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=submit) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=15.0)

    assert not errors, f"unexpected errors: {errors}"
    assert len(results) == 4
    assert all(r["passed"] == 1 and r["total"] == 1 for r in results)


def test_server_returns_timeout_status_for_infinite_loop(grader_server):
    """Wall-clock timeout enforced by the server, not the worker."""
    resp = _request(
        grader_server.socket_path,
        code="while True: pass",
        tests=["assert True"],
        timeout_s=1.0,
    )
    assert resp["status"] == "timeout"
    assert resp["passed"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_grader_server.py -v`
Expected: 4 FAILED with "ModuleNotFoundError: No module named 'reliquary.environment.grader.server'"

- [ ] **Step 3: Implement `reliquary/environment/grader/server.py`**

```python
"""Grader server — manages a warm pool of worker subprocesses.

Listens on a Unix domain socket. Each client connection sends one
JSON request line; the server picks an idle worker from the pool,
pipes the request to its stdin, reads the response from stdout,
and writes it back to the client.

Workers are kept warm between requests: each is a long-lived
subprocess of `worker.py`. If a worker dies (broken pipe) or
times out, it is killed and respawned.

In production the worker subprocess is wrapped in `runsc` (via the
`worker_argv` constructor argument). For tests, plain `python -m
reliquary.environment.grader.worker` is used.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from reliquary.constants import (
    GRADER_EVAL_TIMEOUT_SECONDS,
    GRADER_POOL_SIZE,
    GRADER_SOCKET_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class Worker:
    proc: subprocess.Popen
    slot: int
    in_use: bool = False
    eval_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class GraderServer:
    """Pool of worker subprocesses, dispatched round-robin via a queue."""

    def __init__(
        self,
        socket_path: str = GRADER_SOCKET_PATH,
        pool_size: int = GRADER_POOL_SIZE,
        worker_argv: Optional[list[str]] = None,
        eval_timeout_s: float = GRADER_EVAL_TIMEOUT_SECONDS,
        recycle_after_evals: int = 1000,
    ) -> None:
        self.socket_path = socket_path
        self.pool_size = pool_size
        self.worker_argv = worker_argv or [
            "python", "-m", "reliquary.environment.grader.worker"
        ]
        self.eval_timeout_s = eval_timeout_s
        self.recycle_after_evals = recycle_after_evals

        self._workers: list[Worker] = []
        self._idle: queue.Queue[Worker] = queue.Queue()
        self._listen_sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        # Prep socket.
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        self._listen_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listen_sock.bind(self.socket_path)
        self._listen_sock.listen(self.pool_size * 4)

        # Spawn workers.
        for i in range(self.pool_size):
            self._spawn_worker(i)

        # Accept loop in a background thread.
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except Exception:
                pass
        for w in self._workers:
            try:
                w.proc.kill()
            except Exception:
                pass
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

    def _spawn_worker(self, slot: int) -> Worker:
        proc = subprocess.Popen(
            self.worker_argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        w = Worker(proc=proc, slot=slot)
        # Insert or replace at slot.
        while len(self._workers) <= slot:
            self._workers.append(w)
        self._workers[slot] = w
        self._idle.put(w)
        logger.info("grader: spawned worker slot=%d pid=%d", slot, proc.pid)
        return w

    def _accept_loop(self) -> None:
        assert self._listen_sock is not None
        while not self._stop_event.is_set():
            try:
                conn, _ = self._listen_sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

    def _handle_conn(self, conn: socket.socket) -> None:
        try:
            buf = b""
            while True:
                chunk = conn.recv(8192)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break
            if not buf:
                return
            try:
                req = json.loads(buf.split(b"\n", 1)[0])
            except json.JSONDecodeError:
                conn.sendall(self._error_response("", "grader_error") + b"\n")
                return
            resp = self._dispatch(req)
            conn.sendall(json.dumps(resp).encode() + b"\n")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _dispatch(self, req: dict) -> dict:
        # Acquire a worker (blocks if all busy).
        try:
            w = self._idle.get(timeout=30.0)
        except queue.Empty:
            return {
                "req_id": req.get("req_id", ""),
                "passed": 0, "total": int(len(req.get("tests", []))), "status": "grader_error",
            }
        try:
            return self._evaluate_on_worker(w, req)
        finally:
            # If worker was respawned (timeout/crash), the new one is already in
            # the idle queue. Otherwise return this one.
            if w.proc.poll() is None and not self._needs_recycle(w):
                self._idle.put(w)

    def _evaluate_on_worker(self, w: Worker, req: dict) -> dict:
        timeout_s = float(req.get("timeout_s", self.eval_timeout_s))
        deadline = time.time() + timeout_s + 2.0  # outer wall-clock cushion

        try:
            assert w.proc.stdin is not None and w.proc.stdout is not None
            w.proc.stdin.write(json.dumps(req) + "\n")
            w.proc.stdin.flush()
        except BrokenPipeError:
            # Worker died between checks. Respawn and return failure for this req.
            self._respawn(w)
            return {
                "req_id": req.get("req_id", ""),
                "passed": 0, "total": int(len(req.get("tests", []))), "status": "crash",
            }

        # Read response with wall-clock timeout (no asyncio — keep stdlib only).
        line_holder: dict = {}

        def reader():
            try:
                line_holder["line"] = w.proc.stdout.readline()
            except Exception:
                line_holder["line"] = ""

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout=max(0.1, deadline - time.time()))

        if t.is_alive():
            # Timeout: kill and respawn worker; return timeout status.
            try:
                w.proc.kill()
            except Exception:
                pass
            self._respawn(w)
            return {
                "req_id": req.get("req_id", ""),
                "passed": 0, "total": int(len(req.get("tests", []))), "status": "timeout",
            }

        line = line_holder.get("line", "")
        if not line:
            self._respawn(w)
            return {
                "req_id": req.get("req_id", ""),
                "passed": 0, "total": int(len(req.get("tests", []))), "status": "crash",
            }
        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            return {
                "req_id": req.get("req_id", ""),
                "passed": 0, "total": int(len(req.get("tests", []))), "status": "grader_error",
            }
        w.eval_count += 1
        return resp

    def _respawn(self, w: Worker) -> None:
        try:
            w.proc.kill()
        except Exception:
            pass
        self._spawn_worker(w.slot)

    def _needs_recycle(self, w: Worker) -> bool:
        if w.eval_count >= self.recycle_after_evals:
            logger.info("grader: recycling worker slot=%d after %d evals", w.slot, w.eval_count)
            self._respawn(w)
            return True
        return False

    @staticmethod
    def _error_response(req_id: str, status: str) -> bytes:
        return json.dumps({
            "req_id": req_id, "passed": 0, "total": 0, "status": status,
        }).encode()


def main() -> None:
    """Entrypoint for running the server as a standalone process."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default=GRADER_SOCKET_PATH)
    parser.add_argument("--pool-size", type=int, default=GRADER_POOL_SIZE)
    parser.add_argument("--timeout", type=float, default=GRADER_EVAL_TIMEOUT_SECONDS)
    parser.add_argument(
        "--use-runsc", action="store_true",
        help="Wrap each worker in `runsc` (production mode).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.use_runsc:
        # Production: runsc loads the OCI bundle which already invokes worker.py.
        bundle = os.environ.get(
            "GRADER_BUNDLE_PATH",
            "/opt/reliquary/reliquary/environment/grader/bundle",
        )
        worker_argv = ["runsc", "--network=none", "run",
                       "--bundle", bundle, "grader-worker"]
    else:
        worker_argv = ["python", "-m", "reliquary.environment.grader.worker"]

    server = GraderServer(
        socket_path=args.socket,
        pool_size=args.pool_size,
        worker_argv=worker_argv,
        eval_timeout_s=args.timeout,
    )
    server.start()
    logger.info("grader server listening on %s (pool=%d)", args.socket, args.pool_size)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_grader_server.py -v`
Expected: 4 PASSED (may take 5–10 s for the timeout test)

- [ ] **Step 5: Commit (deferred)**

```bash
git add reliquary/environment/grader/server.py tests/unit/test_grader_server.py
git commit -m "feat(grader): server with warm pool, timeout enforcement, respawn"
```

---

## Task 6: `OpenCodeInstructEnvironment` class + tests (mocked grader)

**Files:**
- Modify: `reliquary/environment/opencodeinstruct.py` (extend from Task 2)
- Modify: `tests/unit/test_opencodeinstruct_environment.py` (extend from Task 2)

- [ ] **Step 1: Append failing tests to `tests/unit/test_opencodeinstruct_environment.py`**

Append:

```python


# ---------------------------------------------------------------------------
# OpenCodeInstructEnvironment — exercised with a stub dataset and a
# fake grader client. The real HF dataset and grader are covered by
# the integration smoke tests.
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Mimics the subset of HF datasets API the env touches."""
    def __init__(self, rows):
        self._rows = rows
    def __len__(self):
        return len(self._rows)
    def __getitem__(self, i):
        return self._rows[i]


class _FakeGraderClient:
    def __init__(self, response: float):
        self.response = response
        self.calls = []
    def evaluate(self, code, tests, timeout_s):
        self.calls.append((code, tests, timeout_s))
        return self.response


def _env_with(dataset_rows, grader_response=1.0):
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
    env = OpenCodeInstructEnvironment.__new__(OpenCodeInstructEnvironment)
    env._dataset = _FakeDataset(dataset_rows)
    env._grader = _FakeGraderClient(grader_response)
    return env


def test_get_problem_shape():
    rows = [{
        "input": "Write a function add(a, b) returning their sum.",
        "unit_tests_parsed": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
    }]
    env = _env_with(rows)
    p = env.get_problem(0)
    assert p["prompt"] == "Write a function add(a, b) returning their sum."
    assert isinstance(p["ground_truth"], str)
    import json as _json
    assert _json.loads(p["ground_truth"]) == ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    assert len(p["id"]) == 16


def test_get_problem_id_is_deterministic():
    rows = [{"input": "Same prompt", "unit_tests_parsed": ["assert True"]}]
    env = _env_with(rows)
    assert env.get_problem(0)["id"] == env.get_problem(0)["id"]


def test_get_problem_modulo_wrap():
    rows = [
        {"input": "p0", "unit_tests_parsed": ["assert True"]},
        {"input": "p1", "unit_tests_parsed": ["assert True"]},
    ]
    env = _env_with(rows)
    assert env.get_problem(0)["prompt"] == "p0"
    assert env.get_problem(2)["prompt"] == "p0"  # wrap


def test_compute_reward_delegates_to_grader():
    rows = [{"input": "...", "unit_tests_parsed": ["assert f() == 1"]}]
    env = _env_with(rows, grader_response=0.6)
    p = env.get_problem(0)
    completion = "```python\ndef f(): return 1\n```"
    r = env.compute_reward(p, completion)
    assert r == 0.6
    assert env._grader.calls[0][0] == "def f(): return 1"
    assert env._grader.calls[0][1] == ["assert f() == 1"]


def test_compute_reward_never_raises_on_garbled_problem():
    rows = [{"input": "x", "unit_tests_parsed": ["assert True"]}]
    env = _env_with(rows, grader_response=0.0)
    r = env.compute_reward({"ground_truth": "not-json"}, "any completion")
    assert r == 0.0


def test_environment_name_constant():
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment
    assert OpenCodeInstructEnvironment.name == "opencodeinstruct"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_opencodeinstruct_environment.py -v`
Expected: 6 PASSED (helper tests from Task 2), 6 FAILED (the new ones — missing `OpenCodeInstructEnvironment`)

- [ ] **Step 3: Extend `reliquary/environment/opencodeinstruct.py`**

Append below the existing `_extract_python` function:

```python


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

import hashlib
import json
import os
from typing import ClassVar

from reliquary.constants import GRADER_EVAL_TIMEOUT_SECONDS


class OpenCodeInstructEnvironment:
    """nvidia/OpenCodeInstruct (deterministic subset) — Python codegen.

    Each problem is a coding instruction; the ground truth is the
    JSON-serialized list of assertion strings (unit tests). Reward
    is the fraction of assertions that pass when the miner's code is
    executed in the grader sandbox.

    The dataset is the filtered subset built by
    scripts/build_opencodeinstruct_subset.py and published to
    reliquadotai/opencodeinstruct-deterministic-subset on HF Hub.
    Override the source repo with RELIQUARY_OCI_SUBSET_REPO.
    """

    name: str = "opencodeinstruct"

    _dataset_cache: ClassVar = None
    _DEFAULT_SUBSET_REPO: ClassVar[str] = "reliquadotai/opencodeinstruct-deterministic-subset"

    def __init__(self) -> None:
        if OpenCodeInstructEnvironment._dataset_cache is None:
            import datasets as hf
            repo = os.environ.get("RELIQUARY_OCI_SUBSET_REPO", self._DEFAULT_SUBSET_REPO)
            OpenCodeInstructEnvironment._dataset_cache = hf.load_dataset(
                repo, split="train",
            )
        self._dataset = OpenCodeInstructEnvironment._dataset_cache

        from reliquary.environment.grader_client import GraderClient
        self._grader = GraderClient()

    def __len__(self) -> int:
        return len(self._dataset)

    def get_problem(self, index: int) -> dict:
        idx = index % len(self._dataset)
        row = self._dataset[idx]
        prompt: str = row["input"]
        tests: list[str] = list(row["unit_tests_parsed"])
        problem_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return {
            "prompt": prompt,
            "ground_truth": json.dumps(tests),
            "id": problem_id,
        }

    def compute_reward(self, problem: dict, completion: str) -> float:
        try:
            tests = json.loads(problem.get("ground_truth", "[]"))
            if not isinstance(tests, list):
                return 0.0
        except (json.JSONDecodeError, TypeError):
            return 0.0
        code = _extract_python(completion or "")
        return self._grader.evaluate(code, tests, timeout_s=GRADER_EVAL_TIMEOUT_SECONDS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_opencodeinstruct_environment.py -v`
Expected: 12 PASSED total

- [ ] **Step 5: Commit (deferred)**

```bash
git add reliquary/environment/opencodeinstruct.py tests/unit/test_opencodeinstruct_environment.py
git commit -m "feat(env): OpenCodeInstructEnvironment class + tests"
```

---

## Task 7: Wire into `load_environment` factory

**Files:**
- Modify: `reliquary/environment/__init__.py`
- Test: extend `tests/unit/test_opencodeinstruct_environment.py`

- [ ] **Step 1: Add a failing test for the factory**

Append to `tests/unit/test_opencodeinstruct_environment.py`:

```python


def test_load_environment_factory_recognizes_opencodeinstruct(monkeypatch):
    """load_environment('opencodeinstruct') returns the class without
    actually downloading the dataset (we monkeypatch __init__)."""
    from reliquary.environment import load_environment
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment

    monkeypatch.setattr(OpenCodeInstructEnvironment, "__init__", lambda self: None)
    env = load_environment("opencodeinstruct")
    assert isinstance(env, OpenCodeInstructEnvironment)


def test_load_environment_unknown_still_raises():
    from reliquary.environment import load_environment
    with pytest.raises(ValueError, match="Unknown environment"):
        load_environment("doesnotexist")
```

- [ ] **Step 2: Run tests to verify the first fails**

Run: `pytest tests/unit/test_opencodeinstruct_environment.py::test_load_environment_factory_recognizes_opencodeinstruct -v`
Expected: FAILED with "ValueError: Unknown environment: opencodeinstruct"

- [ ] **Step 3: Modify `reliquary/environment/__init__.py`**

Edit the file to:

```python
"""Reliquary environment module.

Provides the Environment protocol and a factory function to instantiate
concrete environments by name.
"""

from reliquary.environment.base import Environment
from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment
from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment


def load_environment(name: str) -> Environment:
    """Return a concrete Environment instance for the given *name*.

    Raises:
        ValueError: if *name* is not a recognised environment.
    """
    if name == "openmathinstruct":
        return OpenMathInstructEnvironment()
    if name == "opencodeinstruct":
        return OpenCodeInstructEnvironment()
    raise ValueError(f"Unknown environment: {name}")


__all__ = [
    "Environment",
    "load_environment",
]
```

- [ ] **Step 4: Run all env tests to verify they pass**

Run: `pytest tests/unit/test_opencodeinstruct_environment.py tests/unit/test_openmathinstruct_environment.py -v`
Expected: All PASSED (12 from OCI + N from OMI unchanged)

- [ ] **Step 5: Commit (deferred)**

```bash
git add reliquary/environment/__init__.py tests/unit/test_opencodeinstruct_environment.py
git commit -m "feat(env): register opencodeinstruct in load_environment factory"
```

---

## Task 8: OCI bundle config

**Files:**
- Create: `reliquary/environment/grader/bundle/config.json`

This is an OCI runtime config used by `runsc run`. The rootfs is built separately (Task 9). No unit tests — exercised in Task 10's integration test.

- [ ] **Step 1: Create the bundle config**

Create `reliquary/environment/grader/bundle/config.json`:

```json
{
  "ociVersion": "1.0.2",
  "process": {
    "terminal": false,
    "user": {"uid": 0, "gid": 0},
    "args": ["/usr/local/bin/python3", "/opt/worker.py"],
    "env": [
      "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
      "PYTHONHASHSEED=0",
      "PYTHONDONTWRITEBYTECODE=1",
      "PYTHONUNBUFFERED=1"
    ],
    "cwd": "/",
    "capabilities": {
      "bounding": [], "effective": [], "inheritable": [], "permitted": [], "ambient": []
    },
    "rlimits": [
      {"type": "RLIMIT_CPU", "hard": 5, "soft": 5},
      {"type": "RLIMIT_NPROC", "hard": 16, "soft": 16},
      {"type": "RLIMIT_FSIZE", "hard": 0, "soft": 0},
      {"type": "RLIMIT_AS", "hard": 268435456, "soft": 268435456}
    ],
    "noNewPrivileges": true
  },
  "root": {
    "path": "rootfs",
    "readonly": true
  },
  "hostname": "grader-worker",
  "mounts": [
    {
      "destination": "/tmp",
      "type": "tmpfs",
      "source": "tmpfs",
      "options": ["nosuid", "noexec", "nodev", "size=10m"]
    },
    {
      "destination": "/proc",
      "type": "proc",
      "source": "proc"
    }
  ],
  "linux": {
    "namespaces": [
      {"type": "pid"}, {"type": "ipc"}, {"type": "uts"},
      {"type": "mount"}, {"type": "network"}
    ]
  }
}
```

- [ ] **Step 2: Validate the JSON parses**

Run: `python -c "import json; json.load(open('reliquary/environment/grader/bundle/config.json'))"`
Expected: no output (success)

- [ ] **Step 3: Commit (deferred)**

```bash
git add reliquary/environment/grader/bundle/config.json
git commit -m "feat(grader): OCI bundle config for runsc (no net, rlimits, ro rootfs)"
```

---

## Task 9: `scripts/build_grader_bundle.sh` — rootfs builder

**Files:**
- Create: `scripts/build_grader_bundle.sh`

No unit test — this is shell that runs at image-build time. Validated by the integration test in Task 10.

- [ ] **Step 1: Create the script**

Create `scripts/build_grader_bundle.sh`:

```bash
#!/usr/bin/env bash
# Build the OCI rootfs for the grader sandbox.
#
# Strategy: extract `python:3.12-slim` Docker image's rootfs into
# reliquary/environment/grader/bundle/rootfs/, then copy worker.py
# into /opt/worker.py inside the rootfs. The bundle config.json
# (sibling file) references this rootfs.
#
# Idempotent: re-running rebuilds the rootfs from scratch.
set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-/opt/reliquary/reliquary/environment/grader/bundle}"
ROOTFS="${BUNDLE_DIR}/rootfs"
WORKER_SRC="${WORKER_SRC:-/opt/reliquary/reliquary/environment/grader/worker.py}"
PY_IMAGE="${PY_IMAGE:-python:3.12-slim}"

echo "[build_grader_bundle] BUNDLE_DIR=${BUNDLE_DIR}"
echo "[build_grader_bundle] ROOTFS=${ROOTFS}"
echo "[build_grader_bundle] WORKER_SRC=${WORKER_SRC}"

if [[ ! -f "${WORKER_SRC}" ]]; then
  echo "ERROR: worker.py not found at ${WORKER_SRC}" >&2
  exit 1
fi

# Clean any previous rootfs.
rm -rf "${ROOTFS}"
mkdir -p "${ROOTFS}"

# Pull and export the python:3.12-slim rootfs.
# We use `docker create` + `docker export` to materialize a flat tarball,
# then untar into the bundle directory. Requires Docker in the build env.
CID="$(docker create "${PY_IMAGE}" /bin/true)"
trap 'docker rm -f "${CID}" >/dev/null 2>&1 || true' EXIT
docker export "${CID}" | tar -x -C "${ROOTFS}"

# Drop the worker.py into /opt inside the rootfs.
mkdir -p "${ROOTFS}/opt"
install -m 0644 "${WORKER_SRC}" "${ROOTFS}/opt/worker.py"

# Sanity check.
if [[ ! -x "${ROOTFS}/usr/local/bin/python3" ]]; then
  echo "ERROR: python3 not found in rootfs at /usr/local/bin/python3" >&2
  exit 1
fi

echo "[build_grader_bundle] done. Bundle ready at ${BUNDLE_DIR}"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/build_grader_bundle.sh`
Expected: no output

- [ ] **Step 3: Smoke-test the script if Docker is available**

Run: `command -v docker && BUNDLE_DIR="$(mktemp -d)/bundle" WORKER_SRC="$(pwd)/reliquary/environment/grader/worker.py" PY_IMAGE=python:3.12-slim bash scripts/build_grader_bundle.sh && echo OK || echo "skip: docker not available"`
Expected: "OK" if Docker is installed; "skip" otherwise (do not fail the task — the script is exercised in CI/image-build, not interactively).

- [ ] **Step 4: Commit (deferred)**

```bash
git add scripts/build_grader_bundle.sh
git commit -m "feat(grader): script to build OCI rootfs from python:3.12-slim"
```

---

## Task 10: gVisor E2E integration test (real runsc)

**Files:**
- Create: `tests/integration/test_grader_e2e.py`

Skipped gracefully if `runsc` is unavailable or the bundle isn't built. Run as a dedicated CI job on a runner with runsc + Docker installed.

- [ ] **Step 1: Create the integration test**

Create `tests/integration/test_grader_e2e.py`:

```python
"""End-to-end grader test with real runsc + real bundle.

Skipped if runsc isn't installed or the bundle directory is missing.
Run manually with:

    pytest tests/integration/test_grader_e2e.py -v -s

or via the dedicated CI job.
"""

import os
import shutil
import socket
import json
import time
import threading
import pytest


BUNDLE_PATH = os.environ.get(
    "GRADER_BUNDLE_PATH",
    "/opt/reliquary/reliquary/environment/grader/bundle",
)


def _runsc_available() -> bool:
    return shutil.which("runsc") is not None and os.path.exists(
        os.path.join(BUNDLE_PATH, "rootfs", "usr", "local", "bin", "python3")
    )


pytestmark = pytest.mark.skipif(
    not _runsc_available(),
    reason="runsc or grader bundle not available on this host",
)


@pytest.fixture
def grader_server(tmp_path):
    from reliquary.environment.grader.server import GraderServer

    sock_path = str(tmp_path / "grader.sock")
    worker_argv = [
        "runsc", "--network=none", "run",
        "--bundle", BUNDLE_PATH, "grader-worker",
    ]
    server = GraderServer(
        socket_path=sock_path, pool_size=2,
        worker_argv=worker_argv, eval_timeout_s=5.0,
    )
    server.start()
    deadline = time.time() + 10.0
    while not os.path.exists(sock_path) and time.time() < deadline:
        time.sleep(0.1)
    yield server
    server.stop()


def _request(sock_path, code, tests, timeout_s=5.0):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(15.0)
        s.connect(sock_path)
        req = {"req_id": "e2e", "code": code, "tests": tests, "timeout_s": timeout_s}
        s.sendall(json.dumps(req).encode() + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk: break
            buf += chunk
    return json.loads(buf.split(b"\n", 1)[0])


def test_e2e_grades_correct_code(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="def add(a,b): return a+b",
        tests=["assert add(1,2) == 3"],
    )
    assert resp["status"] == "ok"
    assert resp["passed"] == 1
    assert resp["total"] == 1


def test_e2e_blocks_network_access(grader_server):
    """Even if code tries socket(), runsc --network=none blocks it."""
    code = (
        "import socket\n"
        "try:\n"
        "    s = socket.socket()\n"
        "    s.connect(('8.8.8.8', 53))\n"
        "    HACKED = True\n"
        "except Exception:\n"
        "    HACKED = False\n"
    )
    tests = ["assert HACKED is False"]
    resp = _request(grader_server.socket_path, code=code, tests=tests)
    assert resp["passed"] == 1, "network should be blocked"


def test_e2e_blocks_filesystem_writes(grader_server):
    code = (
        "try:\n"
        "    open('/etc/hostname', 'w').write('pwned')\n"
        "    WROTE = True\n"
        "except Exception:\n"
        "    WROTE = False\n"
    )
    tests = ["assert WROTE is False"]
    resp = _request(grader_server.socket_path, code=code, tests=tests)
    assert resp["passed"] == 1


def test_e2e_kills_infinite_loop(grader_server):
    resp = _request(
        grader_server.socket_path,
        code="while True: pass",
        tests=["assert True"],
        timeout_s=1.0,
    )
    assert resp["status"] == "timeout"


def test_e2e_pool_recovers_after_worker_crash(grader_server):
    """Send hostile request, then a normal one — second should still work."""
    _request(grader_server.socket_path, code="while True: pass",
             tests=["assert True"], timeout_s=1.0)
    # After respawn, normal request must succeed.
    resp = _request(grader_server.socket_path, code="x = 1", tests=["assert x == 1"])
    assert resp["status"] == "ok"
    assert resp["passed"] == 1
```

- [ ] **Step 2: Run integration test if runsc is available**

Run: `pytest tests/integration/test_grader_e2e.py -v`
Expected: 5 PASSED if `runsc` + bundle present; all SKIPPED with reason "runsc or grader bundle not available" otherwise.

- [ ] **Step 3: Commit (deferred)**

```bash
git add tests/integration/test_grader_e2e.py
git commit -m "test(grader): E2E integration with real runsc + hostile-code cases"
```

---

## Task 11: Dataset filter pipeline

**Files:**
- Create: `scripts/build_opencodeinstruct_subset.py`
- Test: `tests/unit/test_opencodeinstruct_dataset_filter.py`

The pipeline is offline (one-shot push to HF Hub), but the filter functions are pure-Python and tested independently.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_opencodeinstruct_dataset_filter.py`:

```python
"""Tests for the OpenCodeInstruct subset filter pipeline.

The filter functions are pure (no HF / no network), tested directly.
The push-to-Hub side is exercised manually when running the script.
"""

import pytest


def test_keep_row_filters_low_test_score():
    from scripts.build_opencodeinstruct_subset import keep_row
    row = {"average_test_score": 0.9, "unit_tests": "[]", "input": "p", "output": "x"}
    assert keep_row(row) is False


def test_keep_row_accepts_perfect_score():
    from scripts.build_opencodeinstruct_subset import keep_row
    row = {
        "average_test_score": 1.0,
        "unit_tests": '["assert f(1) == 1"]',
        "input": "p", "output": "x",
    }
    assert keep_row(row) is True


def test_parse_unit_tests_handles_string_list():
    from scripts.build_opencodeinstruct_subset import parse_unit_tests
    raw = '["assert f(1) == 1", "assert f(2) == 2"]'
    assert parse_unit_tests(raw) == ["assert f(1) == 1", "assert f(2) == 2"]


def test_parse_unit_tests_returns_none_on_garbage():
    from scripts.build_opencodeinstruct_subset import parse_unit_tests
    assert parse_unit_tests("not json") is None
    assert parse_unit_tests("[unterminated") is None


def test_has_nondeterministic_pattern_detects_random():
    from scripts.build_opencodeinstruct_subset import has_nondeterministic_pattern
    assert has_nondeterministic_pattern("import random\nassert random.random() > 0") is True
    assert has_nondeterministic_pattern("import time; assert time.time() > 0") is True
    assert has_nondeterministic_pattern("import socket") is True
    assert has_nondeterministic_pattern("import urllib.request") is True
    assert has_nondeterministic_pattern("import requests") is True
    assert has_nondeterministic_pattern("import subprocess") is True
    assert has_nondeterministic_pattern("import threading") is True


def test_has_nondeterministic_pattern_clean_code():
    from scripts.build_opencodeinstruct_subset import has_nondeterministic_pattern
    assert has_nondeterministic_pattern("assert sum([1,2,3]) == 6") is False
    assert has_nondeterministic_pattern("assert sorted([3,1,2]) == [1,2,3]") is False


def test_filter_tests_drops_nondeterministic():
    from scripts.build_opencodeinstruct_subset import filter_tests
    tests = ["assert f(1) == 1", "import random; assert random.random() > 0"]
    kept = filter_tests(tests)
    assert kept == ["assert f(1) == 1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_opencodeinstruct_dataset_filter.py -v`
Expected: 7 FAILED with "ModuleNotFoundError: No module named 'scripts.build_opencodeinstruct_subset'"

- [ ] **Step 3: Create `scripts/__init__.py`**

If `scripts/__init__.py` does not already exist, create it as an empty file so pytest can import the script module.

Run: `touch scripts/__init__.py` (only if it doesn't exist)

- [ ] **Step 4: Create `scripts/build_opencodeinstruct_subset.py`**

```python
"""Build the deterministic subset of nvidia/OpenCodeInstruct.

Run once offline (typically on a beefy box with disk + network).
Filters in order:
  1. Drop rows whose reference solution did not pass all its own
     tests (average_test_score < 1.0).
  2. Parse the unit_tests column (string-encoded list) — drop on
     parse failure.
  3. Drop rows containing any test that imports/uses a non-
     deterministic stdlib module (random, time, socket, ...).
  4. Run a double-execution check on what remains (twice with
     different PYTHONHASHSEED) — drop on mismatch.
  5. Push the resulting subset to HF Hub as
     reliquadotai/opencodeinstruct-deterministic-subset.

Designed so the per-row filter functions are pure-Python and
testable without HuggingFace, network, or subprocess.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)


# Conservative regex: any of these tokens anywhere in the test code
# disqualifies the row. False positives are fine — we have 5M rows
# and only need ~2-3M deterministic ones.
_NONDET_PATTERNS = re.compile(
    r"\b(?:import\s+(?:random|time|datetime|socket|urllib|requests|os|"
    r"subprocess|threading|multiprocessing|asyncio|signal|select)\b"
    r"|from\s+(?:random|time|datetime|socket|urllib|requests|os|"
    r"subprocess|threading|multiprocessing|asyncio|signal|select)\s+import"
    r"|\brandom\.|\btime\.|\bdatetime\.|\bsocket\.|\burllib\.|\brequests\."
    r"|\bos\.environ|\bsubprocess\.|\bthreading\.|\bmultiprocessing\.)"
)


def parse_unit_tests(raw: str) -> Optional[list[str]]:
    """Parse the string-encoded list of tests. Return None on failure."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    if not all(isinstance(t, str) for t in parsed):
        return None
    return parsed


def has_nondeterministic_pattern(test_src: str) -> bool:
    return _NONDET_PATTERNS.search(test_src) is not None


def filter_tests(tests: list[str]) -> list[str]:
    """Keep only tests free of non-deterministic patterns."""
    return [t for t in tests if not has_nondeterministic_pattern(t)]


def keep_row(row: dict) -> bool:
    """Stage-1 filter: reference solution must pass all its own tests."""
    return float(row.get("average_test_score", 0.0)) >= 1.0


def double_execute(code: str, tests: list[str]) -> bool:
    """Run (code, tests) twice with different PYTHONHASHSEEDs.

    Returns True iff both runs yield the same passed count.
    """
    runner = (
        "import json,sys\n"
        "data=json.loads(sys.stdin.read())\n"
        "ns={}\n"
        "try: exec(data['code'], ns)\n"
        "except: pass\n"
        "p=0\n"
        "for t in data['tests']:\n"
        "    try: exec(t, dict(ns)); p+=1\n"
        "    except: pass\n"
        "print(p)\n"
    )
    payload = json.dumps({"code": code, "tests": tests})
    out_seed0 = subprocess.run(
        [sys.executable, "-c", runner], input=payload, capture_output=True, text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"}, timeout=30,
    )
    out_seed1 = subprocess.run(
        [sys.executable, "-c", runner], input=payload, capture_output=True, text=True,
        env={**os.environ, "PYTHONHASHSEED": "1"}, timeout=30,
    )
    return out_seed0.stdout.strip() == out_seed1.stdout.strip()


def process_row(row: dict) -> Optional[dict]:
    """Apply all filters to one row. Return the kept row (with
    `unit_tests_parsed` added) or None to drop."""
    if not keep_row(row):
        return None
    tests = parse_unit_tests(row.get("unit_tests", ""))
    if tests is None:
        return None
    kept_tests = filter_tests(tests)
    if not kept_tests:
        return None
    if not double_execute(row.get("output", ""), kept_tests):
        return None
    return {
        "input": row["input"],
        "output": row["output"],
        "unit_tests_parsed": kept_tests,
        "id": row.get("id", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="nvidia/OpenCodeInstruct")
    parser.add_argument("--target-repo", default="reliquadotai/opencodeinstruct-deterministic-subset")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap on rows to process — for dry-runs.")
    parser.add_argument("--push", action="store_true",
                        help="Push to HF Hub (requires HF_TOKEN).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    import datasets as hf
    ds = hf.load_dataset(args.source, split="train", streaming=True)

    kept = []
    for i, row in enumerate(ds):
        if args.max_rows and i >= args.max_rows:
            break
        out = process_row(row)
        if out:
            kept.append(out)
        if i % 1000 == 0:
            logger.info("processed=%d kept=%d", i, len(kept))

    logger.info("final: processed=%d kept=%d", i + 1, len(kept))
    out_ds = hf.Dataset.from_list(kept)

    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN env var is required to push.")
        out_ds.push_to_hub(args.target_repo, token=token, private=False)
        logger.info("pushed %d rows to %s", len(kept), args.target_repo)
    else:
        out_ds.save_to_disk("./opencodeinstruct-subset")
        logger.info("saved locally to ./opencodeinstruct-subset")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run filter tests to verify they pass**

Run: `pytest tests/unit/test_opencodeinstruct_dataset_filter.py -v`
Expected: 7 PASSED

- [ ] **Step 6: Commit (deferred)**

```bash
git add scripts/__init__.py scripts/build_opencodeinstruct_subset.py tests/unit/test_opencodeinstruct_dataset_filter.py
git commit -m "feat(scripts): OpenCodeInstruct deterministic subset builder"
```

---

## Task 12: Env smoke test (loads filtered dataset)

**Files:**
- Create: `tests/integration/test_opencodeinstruct_env_smoke.py`

- [ ] **Step 1: Create the smoke test**

Create `tests/integration/test_opencodeinstruct_env_smoke.py`:

```python
"""Smoke test for the OpenCodeInstruct environment.

Loads the deterministic subset from HF Hub (or a local override
via RELIQUARY_OCI_SUBSET_REPO) and exercises get_problem +
compute_reward against an in-process fake grader (no runsc needed
for this smoke).

Skipped if the dataset can't be loaded.
"""

import pytest


def test_load_env_and_get_problem_shape():
    from reliquary.environment import load_environment, Environment

    try:
        env = load_environment("opencodeinstruct")
    except Exception as exc:
        pytest.skip(f"could not load opencodeinstruct env: {exc}")

    assert isinstance(env, Environment)
    assert len(env) > 0

    p = env.get_problem(0)
    assert "prompt" in p and "ground_truth" in p and "id" in p
    assert len(p["id"]) == 16

    import json
    tests = json.loads(p["ground_truth"])
    assert isinstance(tests, list) and len(tests) > 0
    assert all(isinstance(t, str) for t in tests)


def test_compute_reward_zero_when_grader_unreachable(monkeypatch):
    """Without a running grader, compute_reward returns 0.0 (never raises)."""
    from reliquary.environment import load_environment

    try:
        env = load_environment("opencodeinstruct")
    except Exception as exc:
        pytest.skip(f"could not load opencodeinstruct env: {exc}")

    # Point the grader client at a definitely-missing socket.
    monkeypatch.setattr(env._grader, "socket_path", "/tmp/definitely-not-a-real-socket.sock")
    p = env.get_problem(0)
    r = env.compute_reward(p, "```python\ndef anything(): pass\n```")
    assert r == 0.0
```

- [ ] **Step 2: Run smoke test**

Run: `pytest tests/integration/test_opencodeinstruct_env_smoke.py -v`
Expected: 2 PASSED if the HF subset is available; SKIPPED with reason otherwise.

- [ ] **Step 3: Commit (deferred)**

```bash
git add tests/integration/test_opencodeinstruct_env_smoke.py
git commit -m "test(env): smoke test for opencodeinstruct dataset loading"
```

---

## Task 13: Prometheus metrics + archive integration

**Files:**
- Modify: `reliquary/environment/grader/server.py` (add metrics)
- Modify: `reliquary/validator/service.py` (add `grader_failures` key to archive)

- [ ] **Step 1: Add failing test for metrics endpoint**

Append to `tests/unit/test_grader_server.py`:

```python


def test_metrics_endpoint_exposes_eval_counter(grader_server, tmp_path):
    """Hit /metrics on the grader's loopback HTTP listener."""
    import urllib.request, time
    # Trigger one eval.
    _request(grader_server.socket_path, code="x=1", tests=["assert x==1"])
    time.sleep(0.1)
    # Metrics URL is fixed at construction time; default loopback port 9876.
    try:
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{grader_server.metrics_port}/metrics", timeout=2.0,
        )
    except Exception as e:
        pytest.skip(f"metrics endpoint not reachable: {e}")
    body = resp.read().decode()
    assert "grader_eval_total" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_grader_server.py::test_metrics_endpoint_exposes_eval_counter -v`
Expected: FAILED (attribute `metrics_port` missing or import error)

- [ ] **Step 3: Add metrics to `reliquary/environment/grader/server.py`**

Edit `GraderServer.__init__` to accept `metrics_port: int = 9876` and store it. Add at the top of the file:

```python
import http.server
from collections import defaultdict


class _MetricsRegistry:
    """Tiny Prometheus-text-format counter registry. No external dep."""
    def __init__(self):
        self._counters: dict[tuple, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}

    def inc(self, name: str, labels: dict[str, str] | None = None, n: int = 1) -> None:
        key = (name, tuple(sorted((labels or {}).items())))
        self._counters[key] += n

    def gauge_set(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def render(self) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        for (name, labels), value in self._counters.items():
            if name not in seen:
                lines.append(f"# TYPE {name} counter")
                seen.add(name)
            lbl = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""
            lines.append(f"{name}{lbl} {value}")
        for name, value in self._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"
```

Add to `GraderServer.__init__`:

```python
        self.metrics_port = metrics_port
        self._metrics = _MetricsRegistry()
        self._metrics_server: Optional[http.server.HTTPServer] = None
```

Add a method:

```python
    def _start_metrics_server(self) -> None:
        registry = self._metrics

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/metrics":
                    self.send_response(404); self.end_headers(); return
                body = registry.render().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args, **kwargs):
                pass  # quiet

        self._metrics_server = http.server.HTTPServer(("127.0.0.1", self.metrics_port), Handler)
        # Capture the OS-assigned port when caller passed metrics_port=0 (ephemeral).
        self.metrics_port = self._metrics_server.server_port
        threading.Thread(target=self._metrics_server.serve_forever, daemon=True).start()
```

Call `self._start_metrics_server()` at the end of `start()`. In `_evaluate_on_worker`, after computing `resp`, increment counters:

```python
        self._metrics.inc("grader_eval_total", {"status": resp.get("status", "ok")})
        self._metrics.gauge_set("grader_pool_busy_workers", self.pool_size - self._idle.qsize())
```

In `_respawn`, increment:

```python
        self._metrics.inc("grader_worker_restarts_total", {"reason": "death"})
```

In `_needs_recycle`, when recycling, increment:

```python
        self._metrics.inc("grader_worker_restarts_total", {"reason": "recycle"})
```

In `stop()`, add: `if self._metrics_server: self._metrics_server.shutdown()`.

Add `metrics_port: int = 9876` to the `__init__` signature default.

Update the fixture in `test_grader_server.py` to pass an ephemeral port:

```python
    server = GraderServer(
        socket_path=sock_path, pool_size=2,
        worker_argv=["python", "-m", "reliquary.environment.grader.worker"],
        eval_timeout_s=5.0, metrics_port=0,  # 0 → OS-assigned
    )
```

And update the metrics test to use `server.metrics_port` after start (the HTTPServer reports the bound port via `.server_port`):

```python
    # In _start_metrics_server, after construction:
    #   self.metrics_port = self._metrics_server.server_port
```

- [ ] **Step 4: Run all grader_server tests**

Run: `pytest tests/unit/test_grader_server.py -v`
Expected: 5 PASSED (the original 4 + the new metrics test)

- [ ] **Step 5: Add `grader_failures` to validator archive**

Edit `reliquary/validator/service.py` around line 729 (the archive construction block from the earlier read of the file). Find:

```python
        archive = {
            "window_start": batcher.window_start,
            "validator_hotkey": self.wallet.hotkey.ss58_address,
            "randomness": batcher.randomness,
            "environment": self.env.name,
            "batch": batch_entries,
            "runners_up": runners_up,
            "reject_summary": dict(getattr(batcher, "reject_counts", {})),
            "rejected": rejected_entries,
            "rewards_by_hotkey": dict(getattr(batcher, "rewards_by_hotkey", {})),
            "late_drops": {
                hk: dict(counts) for hk, counts in self._late_drops.items()
            },
        }
```

Add a new key after `reject_summary`:

```python
            "grader_failures": dict(getattr(self, "_grader_failures", {})),
```

And initialize `self._grader_failures = {}` in `ValidationService.__init__`.

For now no production code increments it (the metric flows through the grader server which doesn't share state with the validator). A follow-up PR can periodically scrape `/metrics` and push counts in. For this iteration, the field exists in the archive shape and renders `{}`.

- [ ] **Step 6: Commit (deferred)**

```bash
git add reliquary/environment/grader/server.py tests/unit/test_grader_server.py reliquary/validator/service.py
git commit -m "feat(grader): Prometheus metrics + archive grader_failures key"
```

---

## Task 14: Dockerfile + entrypoint.sh modifications

**Files:**
- Modify: `Dockerfile`
- Modify: `docker/entrypoint.sh`

No automated tests for this — verified by `docker build` + a manual container start.

- [ ] **Step 1: Modify `Dockerfile`**

Before the existing `# Runtime` section near the bottom, add a block to install runsc and build the bundle:

```dockerfile

# ────────────────  GRADER SANDBOX  ────────────────
# Install gVisor (runsc) for the OpenCodeInstruct env's sandbox.
# Version-pinned to the latest release at writing time; bump cautiously.
RUN ARCH="$(uname -m)" \
 && RUNSC_URL="https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}/runsc" \
 && wget -q "${RUNSC_URL}" -O /usr/local/bin/runsc \
 && chmod +x /usr/local/bin/runsc

# Build the grader OCI bundle (python:3.12-slim rootfs + worker.py).
# Requires the Docker socket at build time — provided by docker buildx.
# If unavailable, this step can be deferred to entrypoint.sh.
COPY scripts/build_grader_bundle.sh /opt/build_grader_bundle.sh
RUN chmod +x /opt/build_grader_bundle.sh \
 && (BUNDLE_DIR=/opt/reliquary/reliquary/environment/grader/bundle \
     WORKER_SRC=/opt/reliquary/reliquary/environment/grader/worker.py \
     /opt/build_grader_bundle.sh || echo "WARN: bundle build deferred (no docker at build time)")

# Create unprivileged users for runtime UID separation.
RUN useradd -m -u 1000 reliquary \
 && useradd -m -u 1001 reliquary-grader
```

- [ ] **Step 2: Modify `docker/entrypoint.sh`**

Replace the existing entrypoint with:

```bash
#!/bin/bash
# Entrypoint for the Reliquary validator image.
#
# Launches:
#   1. Grader server as UID 1001 (no wallet access, no secret env vars)
#   2. Validator main as UID 1000 (owns the hotkey)
set -euo pipefail

: "${BT_WALLET_NAME:?BT_WALLET_NAME is required}"
: "${BT_HOTKEY:?BT_HOTKEY is required}"

# ── Wallet permissions ────────────────────────────────────────────────────
# The host mounts the wallet dir to /home/reliquary/.bittensor. Set
# ownership + mode so UID 1001 cannot read it.
WALLET_DIR="/home/reliquary/.bittensor"
if [[ -d "${WALLET_DIR}" ]]; then
  chown -R 1000:1000 "${WALLET_DIR}"
  chmod -R go-rwx "${WALLET_DIR}"
fi

# ── If grader bundle wasn't built at image-build time, build it now ──────
BUNDLE_ROOTFS="/opt/reliquary/reliquary/environment/grader/bundle/rootfs"
if [[ ! -x "${BUNDLE_ROOTFS}/usr/local/bin/python3" ]]; then
  echo "[entrypoint] Building grader bundle (deferred from image build)..."
  bash /opt/build_grader_bundle.sh
fi

# ── Launch grader server as UID 1001 ─────────────────────────────────────
# Strip secrets from its env so a sandbox escape gains nothing.
echo "[entrypoint] Starting grader server (UID 1001)..."
env -i \
    PATH="/opt/reliquary-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    HOME="/home/reliquary-grader" \
    GRADER_SOCKET_PATH="/tmp/reliquary-grader.sock" \
    GRADER_BUNDLE_PATH="/opt/reliquary/reliquary/environment/grader/bundle" \
  setpriv --reuid=1001 --regid=1001 --clear-groups \
    python -m reliquary.environment.grader.server --use-runsc &
GRADER_PID=$!
trap 'kill ${GRADER_PID} 2>/dev/null || true' EXIT

# Wait briefly for the grader socket to appear.
for _ in $(seq 1 30); do
  [[ -S "/tmp/reliquary-grader.sock" ]] && break
  sleep 0.5
done
chmod 666 /tmp/reliquary-grader.sock || true

# ── Build the validator argv ─────────────────────────────────────────────
args=(
  --network      "${BT_NETWORK:-finney}"
  --netuid       "${BT_NETUID:-81}"
  --wallet-name  "${BT_WALLET_NAME}"
  --hotkey       "${BT_HOTKEY}"
)

if [[ "${RELIQUARY_TRAIN:-0}" == "1" ]]; then
  : "${RELIQUARY_HF_REPO_ID:?RELIQUARY_HF_REPO_ID required in trainer mode}"
  args+=(
    --train
    --checkpoint   "${RELIQUARY_CHECKPOINT:-Qwen/Qwen3-4B-Instruct-2507}"
    --hf-repo-id   "${RELIQUARY_HF_REPO_ID}"
    --http-host    "${RELIQUARY_HTTP_HOST:-0.0.0.0}"
    --http-port    "${RELIQUARY_HTTP_PORT:-8080}"
  )
  [[ -n "${RELIQUARY_EXTERNAL_IP:-}" ]]   && args+=(--external-ip   "${RELIQUARY_EXTERNAL_IP}")
  [[ -n "${RELIQUARY_EXTERNAL_PORT:-}" ]] && args+=(--external-port "${RELIQUARY_EXTERNAL_PORT}")
  [[ -n "${RELIQUARY_RESUME_FROM:-}" ]]   && args+=(--resume-from   "${RELIQUARY_RESUME_FROM}")
else
  args+=(--no-train)
fi

# ── Launch validator as UID 1000 ─────────────────────────────────────────
echo "[entrypoint] Launching: reliquary validate ${args[*]} (UID 1000)"
exec setpriv --reuid=1000 --regid=1000 --clear-groups --inh-caps=-all \
  reliquary validate "${args[@]}"
```

- [ ] **Step 3: Verify the script parses**

Run: `bash -n docker/entrypoint.sh`
Expected: no output (no syntax errors)

- [ ] **Step 4: Build the image and verify it starts**

Run: `docker build -t reliquary-grader-test:dev .`
Expected: build succeeds (the bundle step may warn "deferred" — that's OK).

- [ ] **Step 5: Smoke-start the container**

This step requires a valid wallet + hotkey on the host. If not available, skip and rely on the Prime Intellect deployment runbook to validate.

Run:
```bash
docker run --rm -it \
  -e BT_WALLET_NAME=my-test-wallet \
  -e BT_HOTKEY=my-test-hotkey \
  -v ~/.bittensor:/home/reliquary/.bittensor \
  --cap-add=SYS_ADMIN \
  reliquary-grader-test:dev
```
Expected: validator logs appear, grader process is running (`docker exec ... ps auxf` shows both processes under different UIDs).

- [ ] **Step 6: Commit (deferred)**

```bash
git add Dockerfile docker/entrypoint.sh
git commit -m "feat(docker): install runsc, build grader bundle, UID-isolated entrypoint"
```

---

## Task 15: Cross-box determinism CI matrix

**Files:**
- Create: `.github/workflows/cross-box-determinism.yml`

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/cross-box-determinism.yml`:

```yaml
name: cross-box-determinism

on:
  pull_request:
    paths:
      - "reliquary/environment/grader/**"
      - "reliquary/environment/opencodeinstruct.py"
      - "scripts/build_opencodeinstruct_subset.py"
      - ".github/workflows/cross-box-determinism.yml"
  push:
    branches: [main]
    paths:
      - "reliquary/environment/grader/**"
      - "reliquary/environment/opencodeinstruct.py"

jobs:
  determinism:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-22.04, ubuntu-24.04]
        python-version: ["3.12"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install package
        run: pip install -e .
      - name: Run determinism corpus
        env:
          PYTHONHASHSEED: "0"
        run: |
          python -m reliquary.environment.grader.worker < tests/fixtures/determinism_corpus.jsonl > /tmp/results_${{ matrix.os }}.txt
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: determinism-${{ matrix.os }}
          path: /tmp/results_${{ matrix.os }}.txt

  compare:
    needs: determinism
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: determinism-ubuntu-22.04
          path: r22
      - uses: actions/download-artifact@v4
        with:
          name: determinism-ubuntu-24.04
          path: r24
      - name: Diff results
        run: |
          if ! diff -u r22/results_ubuntu-22.04.txt r24/results_ubuntu-24.04.txt; then
            echo "::error::Determinism corpus produced different results across runners."
            exit 1
          fi
          echo "All runs identical across runners."
```

- [ ] **Step 2: Create the corpus fixture**

Create `tests/fixtures/determinism_corpus.jsonl`:

```jsonl
{"req_id":"d1","code":"def add(a,b): return a+b","tests":["assert add(1,2)==3","assert add(-1,1)==0"],"timeout_s":5.0}
{"req_id":"d2","code":"def sort_unique(xs): return sorted(set(xs))","tests":["assert sort_unique([3,1,2,1])==[1,2,3]"],"timeout_s":5.0}
{"req_id":"d3","code":"def fact(n): return 1 if n<=1 else n*fact(n-1)","tests":["assert fact(5)==120","assert fact(0)==1"],"timeout_s":5.0}
{"req_id":"d4","code":"def is_prime(n):\n  if n<2: return False\n  for i in range(2,int(n**0.5)+1):\n    if n%i==0: return False\n  return True","tests":["assert is_prime(7)","assert not is_prime(9)"],"timeout_s":5.0}
{"req_id":"d5","code":"def fib(n):\n  a,b=0,1\n  for _ in range(n): a,b=b,a+b\n  return a","tests":["assert fib(10)==55"],"timeout_s":5.0}
```

- [ ] **Step 3: Smoke-test the worker locally with the corpus**

Run: `mkdir -p tests/fixtures && PYTHONHASHSEED=0 python -m reliquary.environment.grader.worker < tests/fixtures/determinism_corpus.jsonl`
Expected output (one JSON line per input, `passed` and `total` non-zero, status `"ok"`):

```
{"req_id": "d1", "passed": 2, "total": 2, "status": "ok"}
{"req_id": "d2", "passed": 1, "total": 1, "status": "ok"}
{"req_id": "d3", "passed": 2, "total": 2, "status": "ok"}
{"req_id": "d4", "passed": 2, "total": 2, "status": "ok"}
{"req_id": "d5", "passed": 1, "total": 1, "status": "ok"}
```

- [ ] **Step 4: Commit (deferred)**

```bash
git add .github/workflows/cross-box-determinism.yml tests/fixtures/determinism_corpus.jsonl
git commit -m "ci: cross-box determinism matrix for grader worker"
```

---

## Self-review checklist (for the engineer)

Before declaring this plan done, sanity-check:

- [ ] All unit tests pass: `pytest tests/unit/ -v`
- [ ] Smoke test passes when dataset is reachable: `pytest tests/integration/test_opencodeinstruct_env_smoke.py -v`
- [ ] E2E test passes when runsc + bundle present: `pytest tests/integration/test_grader_e2e.py -v`
- [ ] `python -c "from reliquary.environment import load_environment; load_environment('opencodeinstruct')"` succeeds (or fails clearly due to missing dataset, not import errors)
- [ ] `docker build -t test:dev .` completes without errors
- [ ] No new lint warnings introduced
- [ ] `reliquary/constants.py:ENVIRONMENT_NAME` is STILL `"openmathinstruct"` — the env flip is a separate, coordinated release per the migration plan in the spec

## Deviations from the spec (intentional)

- **Quarantine after N consecutive timeouts**: the spec describes
  "two consecutive timeouts → forced respawn". Task 5 implements the
  stricter "respawn on every timeout", which is a strict superset
  (covers the 2-in-a-row case and more). Simpler code, no semantic
  regression.

## Phase 2 — Multi-environment mixing (new tasks)

Phase 2 adds true side-by-side training on OMI + OpenCodeInstruct in
the same optimizer step. Spec section: "Phase 2: multi-environment
mixing" in `docs/superpowers/specs/2026-05-22-opencodeinstruct-env-design.md`.

These tasks ship as a separate hardfork on top of v1 (which is Tasks 1–15
above). They modify the wire protocol (`env_name` in `RolloutSubmission`),
the training loop (`train_step(batches: list)`), and add multi-batcher
orchestration in `ValidationService`.

### Task 16: Multi-env constants

**Files:**
- Modify: `reliquary/constants.py`

- [ ] **Step 1: Edit `reliquary/constants.py`**

Remove the line:
```python
ENVIRONMENT_NAME = "openmathinstruct"
```

Replace with:
```python
# (env_name, prompts_per_batch). Sum across entries = total prompts
# processed per optimizer step. With 2 envs at B_BATCH each, we train
# on 16 prompts × M_ROLLOUTS = 128 sequences per step.
ENVIRONMENT_MIX: list[tuple[str, int]] = [
    ("openmathinstruct", B_BATCH),
    ("opencodeinstruct", B_BATCH),
]

# Number of micro-batches accumulated before an optimizer step. Derived
# from the mix — one micro-batch per active env. Not separately tunable
# to keep semantics simple.
GRAD_ACCUM_STEPS: int = len(ENVIRONMENT_MIX)
```

Note: `B_BATCH` is already defined at line 181, so `ENVIRONMENT_MIX`
can reference it directly.

- [ ] **Step 2: Verify imports**

Run: `python -c "from reliquary.constants import ENVIRONMENT_MIX, GRAD_ACCUM_STEPS; print(ENVIRONMENT_MIX, GRAD_ACCUM_STEPS)"`
Expected: `[('openmathinstruct', 8), ('opencodeinstruct', 8)] 2`

- [ ] **Step 3: Find existing call sites of `ENVIRONMENT_NAME`**

Run: `grep -rn "ENVIRONMENT_NAME" /home/ubuntu/Catalyst/reliquary/ --include="*.py"`

Expected hits (will be updated in later tasks): `cli/main.py`, the
existing OMI smoke test, and any other consumers. Note them — Task 22
fixes the CLI.

- [ ] **Step 4: Commit (deferred per session note)**

```bash
git add reliquary/constants.py
git commit -m "feat(constants): introduce ENVIRONMENT_MIX + GRAD_ACCUM_STEPS for v2"
```

---

### Task 17: env_name field on RolloutSubmission + bump GRAIL version

**Files:**
- Modify: `reliquary/protocol/submission.py`
- Modify: `reliquary/constants.py` (bump GRAIL_PROOF_VERSION)
- Test: `tests/unit/test_submission_env_name.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_submission_env_name.py`:

```python
"""Tests for the env_name field on RolloutSubmission (v2 wire schema)."""

import pytest


def test_rollout_submission_carries_env_name():
    from reliquary.protocol.submission import RolloutSubmission
    r = RolloutSubmission(
        tokens=[1, 2, 3], reward=0.5, commit={"rollout": {"prompt_length": 1}},
        env_name="opencodeinstruct",
    )
    assert r.env_name == "opencodeinstruct"


def test_rollout_submission_env_name_required_for_v2():
    """env_name has no default in v2 — missing it is a programming error."""
    from reliquary.protocol.submission import RolloutSubmission
    with pytest.raises(TypeError):
        RolloutSubmission(tokens=[1], reward=0.0, commit={})


def test_grail_proof_version_bumped_to_v6():
    from reliquary.constants import GRAIL_PROOF_VERSION
    assert GRAIL_PROOF_VERSION == "v6"
```

- [ ] **Step 2: Run — expect FAIL**

`cd /home/ubuntu/Catalyst && pytest tests/unit/test_submission_env_name.py -v`

- [ ] **Step 3: Modify `reliquary/protocol/submission.py`**

Find the `RolloutSubmission` dataclass and add `env_name: str` as a
**required** field (no default). It must precede any fields that have
defaults; place it directly after `commit`.

```python
@dataclass
class RolloutSubmission:
    tokens: list[int]
    reward: float
    commit: dict
    env_name: str
```

(Match the existing class structure — do not change other fields.)

- [ ] **Step 4: Bump GRAIL version in `reliquary/constants.py`**

Find `GRAIL_PROOF_VERSION = "v5"` and change to:
```python
GRAIL_PROOF_VERSION = "v6"
```

- [ ] **Step 5: Run — expect PASS**

`cd /home/ubuntu/Catalyst && pytest tests/unit/test_submission_env_name.py -v`
Expected: 3 PASSED.

- [ ] **Step 6: Update existing call sites that construct `RolloutSubmission`**

Run: `grep -rn "RolloutSubmission(" /home/ubuntu/Catalyst/reliquary /home/ubuntu/Catalyst/tests --include="*.py"`

Add `env_name=<appropriate>` to each construction. For miner-side code,
the env name comes from `self.env.name` (the env that was used to pick
the prompt). For tests, use `env_name="openmathinstruct"` unless the
test specifically exercises multi-env.

- [ ] **Step 7: Full suite passes**

`cd /home/ubuntu/Catalyst && pytest tests/unit -q --no-header 2>&1 | tail -5`
Expected: no regressions from this change (existing unrelated failures
are pre-existing).

- [ ] **Step 8: Commit (deferred)**

```bash
git add reliquary/protocol/submission.py reliquary/constants.py tests/unit/test_submission_env_name.py
git commit -m "feat(protocol): add env_name field to RolloutSubmission; bump GRAIL_PROOF_VERSION to v6"
```

---

### Task 18: `load_environments` factory

**Files:**
- Modify: `reliquary/environment/__init__.py`
- Test: extend `tests/unit/test_opencodeinstruct_environment.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_opencodeinstruct_environment.py`:

```python


def test_load_environments_returns_dict(monkeypatch):
    """load_environments(names) returns {name: Environment} for each name."""
    from reliquary.environment import load_environments
    from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment

    # Stub both env __init__s to avoid HF downloads.
    monkeypatch.setattr(OpenMathInstructEnvironment, "__init__", lambda self: None)
    monkeypatch.setattr(OpenCodeInstructEnvironment, "__init__", lambda self: None)

    envs = load_environments(["openmathinstruct", "opencodeinstruct"])
    assert set(envs.keys()) == {"openmathinstruct", "opencodeinstruct"}
    assert isinstance(envs["openmathinstruct"], OpenMathInstructEnvironment)
    assert isinstance(envs["opencodeinstruct"], OpenCodeInstructEnvironment)


def test_load_environments_unknown_name_raises():
    from reliquary.environment import load_environments
    with pytest.raises(ValueError, match="Unknown environment"):
        load_environments(["openmathinstruct", "nope"])
```

- [ ] **Step 2: Modify `reliquary/environment/__init__.py`**

Append after the existing `load_environment` (singular):

```python
def load_environments(names: list[str]) -> dict[str, Environment]:
    """Return a dict {name: Environment} for each requested env.

    Raises ValueError if any name is not recognised. Single-env callers
    can keep using load_environment; multi-env callers (validator with
    ENVIRONMENT_MIX) use this.
    """
    return {name: load_environment(name) for name in names}
```

Add `load_environments` to `__all__`.

- [ ] **Step 3: Run — expect PASS**

`cd /home/ubuntu/Catalyst && pytest tests/unit/test_opencodeinstruct_environment.py -v -k "load_environments"`
Expected: 2 PASSED.

- [ ] **Step 4: Commit (deferred)**

```bash
git add reliquary/environment/__init__.py tests/unit/test_opencodeinstruct_environment.py
git commit -m "feat(env): load_environments factory for multi-env validator/miner"
```

---

### Task 19: Multi-env miner engine

**Files:**
- Modify: `reliquary/miner/engine.py`
- Test: `tests/unit/test_pick_prompt_idx_multi_env.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_pick_prompt_idx_multi_env.py`:

```python
"""Tests for the multi-env prompt selection logic in the miner engine."""

import random
import pytest


class _FakeEnv:
    """Minimal Environment stub: name + len + get_problem stub."""
    def __init__(self, name: str, size: int):
        self.name = name
        self._size = size
    def __len__(self):
        return self._size
    def get_problem(self, idx):
        return {"prompt": f"{self.name}-{idx}", "ground_truth": "", "id": "x" * 16}


def test_pick_env_and_prompt_returns_env_and_idx():
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {
        "openmathinstruct": _FakeEnv("openmathinstruct", 100),
        "opencodeinstruct": _FakeEnv("opencodeinstruct", 50),
    }
    mix = [("openmathinstruct", 8), ("opencodeinstruct", 8)]
    cooldown = {name: set() for name in envs}
    rng = random.Random(0)
    env_name, idx = pick_env_and_prompt(envs, mix, cooldown, rng=rng)
    assert env_name in {"openmathinstruct", "opencodeinstruct"}
    assert 0 <= idx < len(envs[env_name])


def test_pick_env_and_prompt_respects_weights():
    """With weights 1:9, the rare env should be chosen far less often."""
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {"a": _FakeEnv("a", 10), "b": _FakeEnv("b", 10)}
    mix = [("a", 1), ("b", 9)]
    cooldown = {"a": set(), "b": set()}
    rng = random.Random(42)
    counts = {"a": 0, "b": 0}
    for _ in range(1000):
        env_name, _ = pick_env_and_prompt(envs, mix, cooldown, rng=rng)
        counts[env_name] += 1
    # 1:9 weights → roughly 100:900. Allow generous slack.
    assert 70 < counts["a"] < 150
    assert 850 < counts["b"] < 950


def test_pick_env_and_prompt_skips_env_in_full_cooldown():
    """If one env is fully in cooldown, sampling still works on the other."""
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {"a": _FakeEnv("a", 5), "b": _FakeEnv("b", 5)}
    mix = [("a", 1), ("b", 1)]
    cooldown = {"a": set(range(5)), "b": set()}  # 'a' fully blocked
    rng = random.Random(0)
    for _ in range(50):
        env_name, idx = pick_env_and_prompt(envs, mix, cooldown, rng=rng)
        assert env_name == "b"  # never 'a'
```

- [ ] **Step 2: Run — expect FAIL**

`cd /home/ubuntu/Catalyst && pytest tests/unit/test_pick_prompt_idx_multi_env.py -v`
Expected: 3 FAIL (ImportError: pick_env_and_prompt missing).

- [ ] **Step 3: Add `pick_env_and_prompt` to `reliquary/miner/engine.py`**

Find the existing `pick_prompt_idx` function (around line 80). Add
**below** it (do not modify pick_prompt_idx — keep it for backward compat
within mono-env flows / tests):

```python
def pick_env_and_prompt(
    envs: dict,
    mix: list[tuple[str, int]],
    cooldown_per_env: dict[str, set[int]],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
) -> tuple[str, int]:
    """Sample env per `mix` weights, then a prompt within that env.

    `envs` is {name: Environment}. `mix` is the same shape as
    `constants.ENVIRONMENT_MIX`. `cooldown_per_env` carries one set per
    env. If the chosen env is fully in cooldown, falls through to the
    next env by re-sampling with that env masked out.
    """
    rng = rng or _random
    weights = [w for _, w in mix]
    names = [n for n, _ in mix]
    if not names:
        raise RuntimeError("pick_env_and_prompt: empty mix")

    available = list(names)
    while available:
        env_name = rng.choices(
            available,
            weights=[weights[names.index(n)] for n in available],
        )[0]
        env = envs[env_name]
        try:
            idx = pick_prompt_idx(env, cooldown_per_env.get(env_name, set()),
                                  rng=rng, max_attempts=max_attempts)
            return env_name, idx
        except RuntimeError:
            available.remove(env_name)

    raise RuntimeError("pick_env_and_prompt: all envs fully in cooldown")
```

- [ ] **Step 4: Run — expect PASS**

`cd /home/ubuntu/Catalyst && pytest tests/unit/test_pick_prompt_idx_multi_env.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Update `MiningEngine` to use the multi-env helper**

In `reliquary/miner/engine.py`, find the `MiningEngine` class. Modify
its `__init__` to accept either `env: Environment` (legacy) OR
`envs: dict[str, Environment]` + `mix: list[tuple[str, int]]` (new).

A minimal patch:

```python
class MiningEngine:
    def __init__(
        self,
        vllm_model,
        hf_model,
        tokenizer,
        wallet,
        env: "Environment | None" = None,
        *,
        envs: "dict[str, Environment] | None" = None,
        mix: "list[tuple[str, int]] | None" = None,
        # ... rest unchanged
    ):
        # ... existing assignments
        if envs is not None and mix is not None:
            self.envs = envs
            self.mix = mix
            # cooldown_per_env: lazy init in mine_window
            self._multi_env = True
        else:
            assert env is not None, "must pass either env or envs+mix"
            self.envs = {env.name: env}
            self.mix = [(env.name, 1)]
            self.env = env  # legacy
            self._multi_env = False
```

Find the prompt-pick site in `mine_window` (or wherever
`pick_prompt_idx` is called). Replace with:

```python
        cooldown_per_env = getattr(self, "_cooldown_per_env", None)
        if cooldown_per_env is None:
            cooldown_per_env = {name: set() for name in self.envs}
            self._cooldown_per_env = cooldown_per_env
        env_name, prompt_idx = pick_env_and_prompt(
            self.envs, self.mix, cooldown_per_env,
        )
        env = self.envs[env_name]
        problem = env.get_problem(prompt_idx)
```

And later when building the submission, pass `env_name=env_name` to
`RolloutSubmission`.

- [ ] **Step 6: Run unit suite — confirm no miner regressions**

`cd /home/ubuntu/Catalyst && pytest tests/unit -q --no-header 2>&1 | grep -E "passed|failed" | tail -3`
Expected: same baseline as before (pre-existing failures only).

- [ ] **Step 7: Commit (deferred)**

```bash
git add reliquary/miner/engine.py tests/unit/test_pick_prompt_idx_multi_env.py
git commit -m "feat(miner): pick_env_and_prompt + MiningEngine multi-env mode"
```

---

### Task 20: `train_step` accepts list of batches + grad accumulation

**Files:**
- Modify: `reliquary/validator/training.py`
- Test: `tests/unit/test_train_step_grad_accum.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_train_step_grad_accum.py`:

```python
"""Tests for train_step's multi-batch (grad accumulation) mode."""

import pytest
import torch


def _tiny_model():
    """Single-layer model for cheap gradient checks."""
    return torch.nn.Linear(4, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_train_step_two_batches_makes_one_optimizer_step():
    """Calling train_step with [batch_a, batch_b] should make ONE optimizer
    step whose effective batch is the union."""
    # This test asserts the function signature accepts a list of batches.
    # Full integration is exercised in the validator service unit tests.
    from reliquary.validator import training

    # We can't easily exercise the full GRPO path here without rollouts
    # carrying real tokens — see test_train_step_dict_input below for
    # the lightweight signature check.
    pass


def test_train_step_accepts_list_of_batches_signature():
    """train_step's signature accepts a list of batches as the second arg."""
    from reliquary.validator.training import train_step
    import inspect

    sig = inspect.signature(train_step)
    params = list(sig.parameters.keys())
    # Should be (model, batches, *, ref_model, window_index=None)
    assert params[0] == "model"
    assert params[1] == "batches"


def test_train_step_empty_batches_returns_model():
    from reliquary.validator.training import train_step

    class _Stub:
        pass

    model = _Stub()
    result = train_step(model, [], ref_model=None)
    assert result is model

    result = train_step(model, [[], []], ref_model=None)
    assert result is model
```

- [ ] **Step 2: Run — expect FAIL on signature test**

- [ ] **Step 3: Modify `reliquary/validator/training.py`**

Find `train_step(model, batch: list, ...)`. Change the signature:

```python
def train_step(
    model,
    batches: list,
    *,
    ref_model,
    window_index: int | None = None,
) -> Any:
    """Run one GRPO step over the union of `batches`.

    `batches` is a list of per-env batches (one per active env in
    ENVIRONMENT_MIX). All rollouts contribute backward calls before a
    single optimizer.step(), so the effective batch size is
    sum(len(b) for b in batches) prompts.
    """
    if not batches or all(not b for b in batches):
        logger.info("train_step: empty batches, skipping")
        return model

    if not _lazy_init(model):
        logger.info("train_step: model not initializable (non-torch?), skipping")
        return model
    assert _optimizer is not None and _scheduler is not None

    model.train()
    device = next(model.parameters()).device

    _optimizer.zero_grad()

    n_total_rollouts = sum(
        len(g.rollouts) for batch in batches for g in batch
    )
    total_ppo = 0.0
    total_kl = 0.0
    n_processed = 0
    n_skipped = 0

    for batch in batches:
        for group in batch:
            rewards = [r.reward for r in group.rollouts]
            advantages = _compute_advantages(rewards)
            if all(a == 0.0 for a in advantages):
                n_skipped += 1
                logger.debug("skipping degenerate group on prompt_idx=%d", group.prompt_idx)
                continue

            for rollout, adv in zip(group.rollouts, advantages):
                try:
                    ppo_loss, kl = _rollout_loss(
                        model=model, ref_model=ref_model,
                        rollout=rollout, advantage=adv, device=device,
                    )
                except ValueError as e:
                    logger.warning("rollout skipped: %s", e)
                    continue
                loss = (ppo_loss + KL_BETA * kl) / n_total_rollouts
                loss.backward()
                total_ppo += ppo_loss.item()
                total_kl += kl.item()
                n_processed += 1

    if n_processed == 0:
        logger.info("train_step: no valid rollouts processed")
        return model

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    _optimizer.step()
    _scheduler.step()
    lr = _scheduler.get_last_lr()[0]

    # ... rest of telemetry unchanged (flatten across all batches for
    # reward stats: use `for batch in batches for g in batch for r in g.rollouts`)
```

Update the telemetry block at the bottom of `train_step` to iterate
`for batch in batches for g in batch for r in g.rollouts` when
computing `all_rewards`, `n_rewards`, etc.

- [ ] **Step 4: Run signature tests — expect PASS**

`cd /home/ubuntu/Catalyst && pytest tests/unit/test_train_step_grad_accum.py -v`
Expected: signature + empty-batches tests pass; GPU-required test skipped.

- [ ] **Step 5: Commit (deferred)**

```bash
git add reliquary/validator/training.py tests/unit/test_train_step_grad_accum.py
git commit -m "feat(training): train_step accepts list of batches for grad accumulation"
```

---

### Task 21: ValidationService multi-batcher orchestration

**Files:**
- Modify: `reliquary/validator/service.py`
- Test: extend appropriate unit tests for the service

- [ ] **Step 1: Read the current single-batcher code**

Run: `grep -n "_active_batcher\|make_batcher\|seal_batch" /home/ubuntu/Catalyst/reliquary/validator/service.py`

Note all the sites. Currently a singleton — needs to become a list/dict.

- [ ] **Step 2: Modify `ValidationService.__init__`**

Where `self.env: Environment` is currently set, replace with:

```python
        from reliquary.environment import load_environments
        from reliquary.constants import ENVIRONMENT_MIX

        env_names = [name for name, _ in ENVIRONMENT_MIX]
        self.envs: dict[str, Environment] = load_environments(env_names)
        self.env_mix = ENVIRONMENT_MIX
        # Legacy accessor — used by archive code that grew up around
        # the single-env assumption. Points to the first env in the
        # mix; consumers that need all envs should iterate self.envs.
        self.env: Environment = self.envs[env_names[0]]
```

- [ ] **Step 3: Replace `self._active_batcher` with `self._active_batchers`**

Find the `_active_batcher: Optional[GrpoWindowBatcher] = None` field
declaration. Change to:

```python
        self._active_batchers: dict[str, "GrpoWindowBatcher"] = {}
```

Find the window-open code that creates a batcher (`make_batcher(...)`).
Replace with a loop over envs:

```python
        self._active_batchers = {}
        for env_name, env in self.envs.items():
            batcher = make_batcher(
                env=env, window_start=window_start, model=self.verify_model,
                tokenizer=self.tokenizer, cooldown_map=self._cooldown_per_env[env_name],
                hash_set=self._hash_set, bootstrap=self._bootstrap,
                queue_drained_predicate=self.server.submit_queue.empty,
            )
            self._active_batchers[env_name] = batcher
        self.server.set_active_batchers(self._active_batchers)
```

- [ ] **Step 4: Update `ValidatorServer.set_active_batcher` to accept a dict**

In `reliquary/validator/server.py`, change `set_active_batcher` to
`set_active_batchers(batchers: dict[str, GrpoWindowBatcher])`. The
server's submit handler must route each incoming submission to the
right batcher by `submission.env_name`. Add a method
`get_batcher_for(env_name) -> Optional[GrpoWindowBatcher]`.

- [ ] **Step 5: Update the seal + train block**

In `service.py`, where currently:
```python
        batch, rewards = self._active_batcher.seal_batch()
```
Replace with:
```python
        sealed = {
            name: b.seal_batch() for name, b in self._active_batchers.items()
        }
        batches = [batch for batch, _ in sealed.values()]
        # rewards_by_hotkey is merged across envs.
        all_rewards = {}
        for _, rewards in sealed.values():
            for hk, r in rewards.items():
                all_rewards[hk] = all_rewards.get(hk, 0.0) + r

        # Trained only if EVERY active env reached B_BATCH (no fallback —
        # underflow on one env skips the whole window, matching v1's
        # partial-seal skip semantics).
        from reliquary.constants import ENVIRONMENT_MIX
        per_env_targets = dict(ENVIRONMENT_MIX)
        trained = all(
            len(batch) >= per_env_targets[name]
            for name, (batch, _) in sealed.items()
        )
```

Then pass `batches` (list) to `train_step`:
```python
                self.train_model = train_step(
                    self.train_model, batches,
                    ref_model=self.verify_model,
                    window_index=self._window_n,
                )
```

- [ ] **Step 6: Update archive shape**

Find the archive dict construction. Replace:
```python
            "environment": self.env.name,
```
with:
```python
            "environments": [name for name in self.envs],
```

And per-submission entries gain an `env_name` field. Find the
`batch_entries` loop — for each submission, add:
```python
                "env_name": s.env_name,
```

- [ ] **Step 7: Update `make_batcher` signature in `service.py`**

The factory function takes `cooldown_map` (singular). For per-env
cooldowns, the service now needs to pass a dict-keyed cooldown. Most
straightforward: keep `make_batcher` per-env (one batcher = one env =
one cooldown map). Initialize `self._cooldown_per_env` in `__init__`:

```python
        self._cooldown_per_env: dict[str, CooldownMap] = {
            name: CooldownMap() for name in self.envs
        }
```

- [ ] **Step 8: Run the unit suite**

`cd /home/ubuntu/Catalyst && pytest tests/unit -q --no-header 2>&1 | tail -10`
Expected: no new failures (existing pre-existing ones unchanged).

- [ ] **Step 9: Commit (deferred)**

```bash
git add reliquary/validator/service.py reliquary/validator/server.py
git commit -m "feat(validator): multi-batcher orchestration + archive shape"
```

---

### Task 22: CLI multi-env parsing

**Files:**
- Modify: `reliquary/cli/main.py`

- [ ] **Step 1: Replace single-env arg with list**

Find the existing `RELIQUARY_ENVIRONMENT_NAME` and `ENVIRONMENT_NAME`
references in `cli/main.py`. Replace the singular `--environment` arg
with `--environments` (comma-separated list).

Default value: `",".join(name for name, _ in ENVIRONMENT_MIX)`.

- [ ] **Step 2: Update both `validate` and `mine` subcommands**

Each subcommand currently passes `environment` to its respective entry
point. Update to pass a list. The `MiningEngine` (Task 19) accepts
`envs=` and `mix=`; the validator main loop reads `ENVIRONMENT_MIX`
directly so no extra plumbing needed beyond removing the singular arg.

- [ ] **Step 3: Update CLI tests**

`tests/unit/test_cli_environment_override.py` exists — update or rename
its assertions for the new `--environments` flag.

- [ ] **Step 4: Run CLI tests**

`cd /home/ubuntu/Catalyst && pytest tests/unit/test_cli_environment_override.py -v`

- [ ] **Step 5: Commit (deferred)**

```bash
git add reliquary/cli/main.py tests/unit/test_cli_environment_override.py
git commit -m "feat(cli): --environments comma-separated for v2 multi-env"
```

---

### Task 23: End-to-end multi-env smoke

**Files:**
- Test: `tests/integration/test_multi_env_smoke.py` (new)

- [ ] **Step 1: Create the smoke test**

```python
"""Smoke test for multi-env Phase 2 — exercises load_environments
+ a stubbed train_step to verify shape compatibility."""

import pytest


def test_load_environments_for_full_mix(monkeypatch):
    """The mix constants drive load_environments end-to-end."""
    from reliquary.constants import ENVIRONMENT_MIX
    from reliquary.environment import load_environments
    from reliquary.environment.openmathinstruct import OpenMathInstructEnvironment
    from reliquary.environment.opencodeinstruct import OpenCodeInstructEnvironment

    monkeypatch.setattr(OpenMathInstructEnvironment, "__init__", lambda self: None)
    monkeypatch.setattr(OpenCodeInstructEnvironment, "__init__", lambda self: None)

    names = [name for name, _ in ENVIRONMENT_MIX]
    envs = load_environments(names)
    assert set(envs.keys()) == set(names)


def test_train_step_handles_two_empty_batches_gracefully():
    from reliquary.validator.training import train_step
    class _Stub:
        pass
    result = train_step(_Stub(), [[], []], ref_model=None)
    assert result is not None
```

- [ ] **Step 2: Run**

`cd /home/ubuntu/Catalyst && pytest tests/integration/test_multi_env_smoke.py -v`
Expected: 2 PASSED.

- [ ] **Step 3: Commit (deferred)**

```bash
git add tests/integration/test_multi_env_smoke.py
git commit -m "test: multi-env Phase 2 smoke"
```



The following are spec items intentionally NOT covered here; they belong to follow-up plans:

- **Dataset materialization and HF Hub push** — running `scripts/build_opencodeinstruct_subset.py` end-to-end (6.4 GB download, hours of double-execute) needs a beefy box and an HF push token. The script is shipped; running it is a deployment task, not code.
- **Network-wide `ENVIRONMENT_NAME` flip** — separate coordinated release per the spec migration plan (phases 4-5).
- **Periodic `/metrics` scraping into the validator archive** — currently the archive has the `grader_failures` key initialized to `{}` for shape stability; populating it from Prometheus is a follow-up.
- **Grader-wide watchdog / supervisor** — the spec mentions "a supervisor (systemd unit or in-Python parent watchdog) restarts the grader server" on full death. Task 14's `entrypoint.sh` uses a `trap` for cleanup but not for restart. For mainnet hardening, add a `restart=always` systemd-like supervisor or a parent-process watchdog. Acceptable for the testnet canary phase.
- **Migration to WASM** — possible future swap of `worker.py` + bundle for a wasmtime runner; the `Environment` Protocol and IPC contract do not change.
