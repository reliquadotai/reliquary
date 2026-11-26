"""The reward worker: the entrypoint of a customer env image.

It speaks JSON lines on stdin/stdout (the container has no network) and holds the customer's taskset:

* ``{"op": "hello"}`` -> ``{"ok": true, "provenance": {...}}``
* ``{"op": "freeze", "offset": 0, "limit": 1000}`` -> ``{"ok": true, "total", "rows": [{key, system, prompt,
  prompt_sha256}], "refused": [{key, reason}]}``
* ``{"op": "score", "items": [{key, prompt_sha256, answer}]}`` -> ``{"ok": true, "results": [{"score": x} |
  {"error": "source_drift" | "timeout" | "grader_error: ..."}]}``

A request it cannot read answers ``{"ok": false, "error": ...}``. This module imports only the standard library
and ``reliquary.eval.verifiers_source``: the image installs reliquary without its dependencies.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable
from typing import IO, Any, Protocol

MAX_LINE_BYTES = 32 * 1024 * 1024
MAX_SCORE_ITEMS = 512
MAX_FREEZE_ROWS = 5000
MAX_ANSWER_CHARS = 100_000
MANIFEST_PATH = "/opt/reliquary/manifest.json"


class Backend(Protocol):
    def provenance(self) -> dict: ...
    def tasks(self) -> list[Any]: ...
    def freeze(self, task) -> dict: ...  # {key, system, prompt, prompt_sha256}, or ValueError
    def refusal(self, task) -> str | None: ...
    def score(self, key: str, prompt_sha256: str, answer: str) -> float: ...


class VerifiersBackend:
    """A verifiers v1 taskset, scored in a child process per item with a timeout (``ChildScorer``)."""

    def __init__(self, taskset: str, args: dict, score_timeout_s: float) -> None:
        from reliquary.eval import verifiers_source as vs

        self._vs = vs
        self._taskset, self._args = taskset, args
        self._handle = vs.open_taskset(taskset, args)
        self._scorer = vs.ChildScorer(timeout_seconds=score_timeout_s)
        self._tasks = None

    def provenance(self) -> dict:
        return self._handle.provenance()

    def tasks(self) -> list[Any]:
        if self._tasks is None:
            self._tasks = list(self._handle.tasks())
        return self._tasks

    def refusal(self, task) -> str | None:
        return self._vs.refusal(task)

    def freeze(self, task) -> dict:
        frozen = self._vs.freeze(task)
        return {"key": frozen.key, "system": frozen.system, "prompt": frozen.prompt,
                "prompt_sha256": self._vs.prompt_digest(frozen.system, frozen.prompt)}

    def score(self, key: str, prompt_sha256: str, answer: str) -> float:
        return self._scorer.score(self._taskset, self._args, key, prompt_sha256, answer)


def _is_timeout(exc: BaseException) -> bool:
    return type(exc).__name__ in ("ScoreTimeout", "TimeoutError")


def _score_one(backend: Backend, item: Any) -> dict:
    if not isinstance(item, dict) or not all(isinstance(item.get(k), str) for k in ("key", "prompt_sha256",
                                                                                    "answer")):
        return {"error": "bad_item: key, prompt_sha256 and answer are strings"}
    if len(item["answer"]) > MAX_ANSWER_CHARS:
        return {"score": 0.0}
    try:
        value = float(backend.score(item["key"], item["prompt_sha256"], item["answer"]))
    except LookupError:
        return {"error": "source_drift"}
    except Exception as exc:
        if _is_timeout(exc):
            return {"error": "timeout"}
        return {"error": f"grader_error: {type(exc).__name__}: {exc}"[:300]}
    if not (0.0 <= value <= 1.0):
        return {"error": f"grader_error: score {value!r} is outside [0, 1]"}
    return {"score": value}


def _freeze(backend: Backend, offset: Any, limit: Any) -> dict:
    if not isinstance(offset, int) or not isinstance(limit, int) or offset < 0 or not 0 < limit <= MAX_FREEZE_ROWS:
        return {"ok": False, "error": f"freeze takes offset >= 0 and 0 < limit <= {MAX_FREEZE_ROWS}"}
    tasks = backend.tasks()
    rows, refused = [], []
    for task in tasks[offset:offset + limit]:
        reason = backend.refusal(task)
        if reason is None:
            try:
                rows.append(backend.freeze(task))
                continue
            except ValueError as exc:
                reason = str(exc)
        refused.append({"key": str(getattr(task, "key", "?")), "reason": reason[:300]})
    return {"ok": True, "total": len(tasks), "rows": rows, "refused": refused}


def handle(backend: Backend, request: Any) -> dict:
    if not isinstance(request, dict):
        return {"ok": False, "error": "a request is a JSON object"}
    op = request.get("op")
    if op == "hello":
        return {"ok": True, "provenance": backend.provenance()}
    if op == "freeze":
        return _freeze(backend, request.get("offset", 0), request.get("limit"))
    if op == "score":
        items = request.get("items")
        if not isinstance(items, list) or not 0 < len(items) <= MAX_SCORE_ITEMS:
            return {"ok": False, "error": f"score takes 1 to {MAX_SCORE_ITEMS} items"}
        return {"ok": True, "results": [_score_one(backend, item) for item in items]}
    return {"ok": False, "error": f"unknown op {op!r}"}


def serve(backend_factory: Callable[[], Backend], lines: Iterable[bytes], out: IO[bytes]) -> None:
    """Answers each request line with one response line. A backend that cannot load answers every
    request with its error, so the host reads why instead of a dead pipe."""
    try:
        backend, load_error = backend_factory(), None
    except Exception as exc:
        backend, load_error = None, f"load_failed: {type(exc).__name__}: {exc}"[:500]
    for raw in lines:
        if len(raw) > MAX_LINE_BYTES:
            response = {"ok": False, "error": "request too large"}
        elif backend is None:
            response = {"ok": False, "error": load_error}
        else:
            try:
                response = handle(backend, json.loads(raw))
            except json.JSONDecodeError:
                response = {"ok": False, "error": "a request is one JSON line"}
            except Exception as exc:  # never die on one request
                response = {"ok": False, "error": f"worker_error: {type(exc).__name__}: {exc}"[:500]}
        out.write(json.dumps(response, allow_nan=False, separators=(",", ":")).encode() + b"\n")
        out.flush()


def protocol_stdout() -> IO[bytes]:
    """The protocol's own copy of stdout; fd 1 and ``sys.stdout`` then point at stderr, so nothing the
    customer's code writes (``print``, a C extension writing fd 1, a child process) reaches the protocol."""
    import os

    sys.stdout.flush()
    channel = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return channel


def main() -> None:
    with open(MANIFEST_PATH, encoding="utf-8") as handle_:
        manifest = json.load(handle_)
    timeout = float(manifest.get("score_timeout_s", 60.0))
    stdout = protocol_stdout()
    serve(lambda: VerifiersBackend(manifest["taskset"], manifest.get("args") or {}, timeout),
          iter(sys.stdin.buffer.readline, b""), stdout)


if __name__ == "__main__":
    main()
