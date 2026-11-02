"""One spawned model process per pinned order, with bounded inference calls."""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import threading
import time

MAX_MESSAGE_BYTES = 128 * 1024**2
MODEL_BOOT_SECONDS = 3600


def _send(connection, document):
    body = json.dumps(document, allow_nan=False).encode("utf-8")
    if len(body) > MAX_MESSAGE_BYTES:
        raise ValueError("model message exceeds the bounded channel")
    connection.send_bytes(body)


def _receive(connection):
    return json.loads(connection.recv_bytes(MAX_MESSAGE_BYTES))


def _worker(connection, factory):
    # The parent has never initialized CUDA. Its owned group includes vLLM's workers.
    os.setsid()
    _send(connection, {"worker_pid": os.getpid(), "worker_group": os.getpgrp()})
    generator = factory()
    try:
        while True:
            call = _receive(connection)
            try:
                if call.get("method") not in {"load", "render", "generate", "render_messages", "generate_turn", "encode", "decode"}:
                    raise ValueError("unknown model call")
                result = getattr(generator, call["method"])(**call["kwargs"])
                if call["method"] == "generate":
                    result = [[{"text": c.text, "tokens": c.tokens, "finish_reason": c.finish_reason}
                               for c in completions] for completions in result]
                _send(connection, {"result": result})
            except Exception as exc:
                # Library errors may contain private headers. No exception body crosses this boundary.
                _send(connection, {"error": type(exc).__name__})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class ModelProcess:
    """The existing Generator interface; only the child imports/loads GPU libraries."""

    def __init__(self, factory, *, timeout_seconds: float, cancelled=lambda: False) -> None:
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, factory),
                                        name="evaluation-model")
        self._process.start()
        child.close()
        self._timeout = timeout_seconds
        self._cancelled = cancelled
        self._episode_deadline = None
        self._closed = False
        self._owned_group = None

    def begin_episode(self):
        self._episode_deadline = time.monotonic() + self._timeout

    def _deadline(self):
        if self._episode_deadline is None:
            self.begin_episode()
        return self._episode_deadline

    def close(self):
        if self._closed:
            return
        self._closed = True
        process = self._process
        # Check before is_alive/join can reap a crashed leader and release its PID.
        # The unreaped owned leader reserves the identity of its private group.
        try:
            if self._owned_group == process.pid or os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.join(timeout=10)
        self._connection.close()
        if process.is_alive():
            raise RuntimeError("model process did not stop")

    def _call(self, method, kwargs, deadline):
        if self._cancelled():
            from reliquary.eval.platform_client import LeaseLost

            self.close()
            raise LeaseLost("evaluation lease was lost before inference")
        done, answer = threading.Event(), {}

        def exchange():
            try:
                if self._owned_group is None:
                    ready = _receive(self._connection)
                    if ready != {"worker_pid": self._process.pid, "worker_group": self._process.pid}:
                        raise RuntimeError("model worker does not own its private process group")
                    self._owned_group = self._process.pid
                _send(self._connection, {"method": method, "kwargs": kwargs})
                answer["value"] = _receive(self._connection)
            except BaseException as exc:
                answer["error"] = exc
            finally:
                done.set()

        io = threading.Thread(target=exchange, name="evaluation-model-io", daemon=True)
        try:
            io.start()
            while not done.wait(min(0.1, max(0, deadline - time.monotonic()))):
                if self._cancelled():
                    from reliquary.eval.platform_client import LeaseLost

                    raise LeaseLost("evaluation lease was lost during inference")
                if time.monotonic() >= deadline:
                    raise TimeoutError("evaluation model deadline exceeded")
            if time.monotonic() >= deadline:
                raise TimeoutError("evaluation model deadline exceeded")
            if self._cancelled():
                from reliquary.eval.platform_client import LeaseLost

                raise LeaseLost("evaluation lease was lost during inference")
            if "error" in answer:
                raise answer["error"]
            response = answer["value"]
            if "error" in response:
                raise RuntimeError("evaluation model call failed: " + response["error"])
            return response["result"]
        except BaseException:
            self.close()
            raise
        finally:
            io.join(timeout=10)
            if io.is_alive():
                raise RuntimeError("model exchange did not stop")

    def load(self, **kwargs):
        return self._call("load", kwargs, time.monotonic() + MODEL_BOOT_SECONDS)

    def render(self, row, *, thinking):
        return self._call("render", {"row": row, "thinking": thinking}, self._deadline())

    def generate(self, prompts, *, samples, seeds, sampling, max_new_tokens):
        from reliquary.eval.runner import Completion

        result = self._call("generate", {"prompts": prompts, "samples": samples, "seeds": seeds,
                                         "sampling": sampling, "max_new_tokens": max_new_tokens},
                            self._deadline())
        return [[Completion(**row) for row in completions] for completions in result]

    def render_messages(self, messages, *, tools, thinking, renderer_id):
        return self._call("render_messages", {"messages": messages, "tools": tools,
                          "thinking": thinking, "renderer_id": renderer_id}, self._deadline())

    def encode(self, text, *, add_special_tokens=False):
        return self._call("encode", {"text": text, "add_special_tokens": add_special_tokens}, self._deadline())

    def decode(self, token_ids, *, skip_special_tokens=False):
        return self._call("decode", {"token_ids": token_ids, "skip_special_tokens": skip_special_tokens}, self._deadline())

    def generate_turn(self, input_ids, *, max_new_tokens, seed, sampling, absolute_deadline):
        return self._call("generate_turn", {"input_ids": input_ids, "max_new_tokens": max_new_tokens,
                          "seed": seed, "sampling": sampling}, min(self._deadline(), absolute_deadline))
