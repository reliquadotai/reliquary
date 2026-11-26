"""A reward worker over a fake taskset, run as a subprocess by the customer env tests."""
import hashlib
import os
import sys
import time
from types import SimpleNamespace

from reliquary.customer_env.worker import protocol_stdout, serve


def digest(system, prompt):
    return hashlib.sha256(f"{system}|{prompt}".encode()).hexdigest()


class FakeBackend:
    def __init__(self):
        if os.environ.get("FAKE_LOAD_FAIL"):
            raise ImportError("no module named my_env")
        self._tasks = [SimpleNamespace(key=f"t{i}", prompt=f"{i}+{i}=", answer=str(2 * i)) for i in range(5)]
        self._tasks.append(SimpleNamespace(key="tools", prompt="use a tool", answer=""))

    def provenance(self):
        return {"id": "fake", "package": "fake-env", "package_version": "1"}

    def tasks(self):
        return self._tasks

    def refusal(self, task):
        return "it gives the model tools" if task.key == "tools" else None

    def freeze(self, task):
        return {"key": task.key, "system": None, "prompt": task.prompt, "prompt_sha256": digest(None, task.prompt)}

    def score(self, key, prompt_sha256, answer):
        task = next((t for t in self._tasks if t.key == key), None)
        if task is None or digest(None, task.prompt) != prompt_sha256:
            raise LookupError("source_drift")
        if answer == "HANG":
            time.sleep(3600)
        if answer == "CRASH":
            os._exit(3)
        if answer == "GARBLE":
            os.write(1, b"not json\n")
        if answer == "RAISE":
            raise ZeroDivisionError("customer bug")
        if answer == "PRINT":
            print("hello from the customer's code")
        return 1.0 if answer.strip() == task.answer else 0.0


if __name__ == "__main__":
    serve(FakeBackend, iter(sys.stdin.buffer.readline, b""), protocol_stdout())
