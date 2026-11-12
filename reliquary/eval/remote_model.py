"""Token-only vLLM 0.29 transport through an owned private loopback tunnel.

Launch identity and GPU placement must be qualified by the executor owner before
constructing this adapter; HTTP model labels cannot attest to loaded weights.
Use only behind ModelProcess: it owns the wall deadline and kills the HTTP
connection on lease loss. The HTTP timeout bounds each I/O, not total wall time.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from reliquary.eval.runner import VLLMGenerator, normalized_sampling, render_prompt, resolved_revision, validate_model

MAX_RESPONSE_BYTES = 16 * 1024**2


class RemoteVLLMGenerator(VLLMGenerator):
    """The signed-episode Generator interface; rendering stays on the CPU host."""

    def __init__(self, *, endpoint: str, api_key: str, repo: str, revision: str,
                 snapshot: str | Path, expected_gpu: str,
                 expected_max_model_len: int, timeout_seconds: float) -> None:
        super().__init__()
        validate_model(repo, revision)
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                or parsed.port is None):
            raise ValueError("remote inference requires an owned literal loopback tunnel")
        if (not isinstance(api_key, str) or not api_key
                or any(not 33 <= ord(char) <= 126 for char in api_key)
                or not isinstance(expected_gpu, str) or not expected_gpu.strip()
                or type(expected_max_model_len) is not int
                or not 1 <= expected_max_model_len <= 65536
                or isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 3600):
            raise ValueError("invalid expected inference deployment contract")
        self._repo, self._revision = repo, revision
        self._snapshot = Path(snapshot)
        self._expected_gpu = expected_gpu
        self._expected_max_model_len = expected_max_model_len
        self._timeout = float(timeout_seconds)
        self._endpoint = endpoint.rstrip("/") + "/inference/v1/generate"
        self._http = httpx.Client(headers={"Authorization": "Bearer " + api_key, "Accept-Encoding": "identity"},
            transport=httpx.HTTPTransport(retries=0), trust_env=False,
            follow_redirects=False, timeout=self._timeout,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1))
        self._inflight = threading.Lock()
        self._closed = False

    def close(self):
        self._closed = True
        self._http.close()

    def load(self, *, repo, revision, gpu_count, max_new_tokens, rows, thinking,
             model_access="public", hf_token=None, max_total_tokens=None):
        if (self._closed or (repo, revision) != (self._repo, self._revision)
                or type(gpu_count) is not int or gpu_count != 1
                or model_access not in {"public", "private"}
                or type(max_total_tokens) is not int
                or not 1 <= max_total_tokens <= self._expected_max_model_len
                or type(max_new_tokens) is not int or not 0 < max_new_tokens <= max_total_tokens):
            raise ValueError("remote model differs from its frozen deployment contract")
        snapshot = self._snapshot
        resolved_revision(snapshot, revision)
        if (snapshot.parent.name != "snapshots"
                or snapshot.parent.parent.name != "models--" + repo.replace("/", "--")):
            raise ValueError("tokenizer snapshot repository differs from the checkpoint")
        config = json.loads((snapshot / "config.json").read_text())
        context = config.get("max_position_embeddings", config.get("n_positions"))
        if type(context) is not int or context < max_total_tokens:
            raise ValueError("checkpoint context is below the frozen transcript budget")
        if version("renderers") != "0.1.11":
            raise ValueError("remote inference requires the pinned renderer")
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(snapshot,
            trust_remote_code=False, local_files_only=True)
        longest = max(len(render_prompt(self._tokenizer, row, thinking=thinking)) for row in rows)
        if longest + max_new_tokens > max_total_tokens:
            raise ValueError("prompt plus output exceeds the frozen transcript budget")
        self._vocab_size = len(self._tokenizer)
        self._max_total_tokens, self._max_turn_tokens = max_total_tokens, max_new_tokens
        template = getattr(self._tokenizer, "chat_template", None) or ""
        return {"vllm_version": "expected:0.29.0", "gpu": "expected:" + self._expected_gpu, "model_sha": revision,
                "chat_template_sha256": hashlib.sha256(str(template).encode()).hexdigest(),
                "remote_launch_identity": "expected; requires executor qualification"}

    def generate(self, *args, **kwargs):
        raise ValueError("the remote transport is qualified only for signed model turns")

    def generate_turn(self, input_ids, *, max_new_tokens, seed, sampling):
        if (self._closed or self._tokenizer is None
                or not isinstance(input_ids, list) or not input_ids
                or any(type(token) is not int or not 0 <= token < self._vocab_size for token in input_ids)
                or type(max_new_tokens) is not int or max_new_tokens <= 0
                or type(seed) is not int or not 0 <= seed < 2**31):
            raise ValueError("a remote model turn requires valid frozen token IDs and limits")
        limit = min(max_new_tokens, self._max_turn_tokens, self._max_total_tokens - len(input_ids))
        if limit <= 0:
            raise ValueError("episode transcript token budget exhausted")
        settings = normalized_sampling(sampling)
        if (not math.isfinite(settings["temperature"]) or settings["temperature"] < 0
                or not math.isfinite(settings["top_p"]) or not 0 < settings["top_p"] <= 1
                or settings["top_k"] < 0):
            raise ValueError("invalid frozen model sampling")
        if self._episode_renderer is None:
            raise ValueError("a remote signed turn requires its pinned episode renderer")
        if not self._inflight.acquire(blocking=False):
            raise RuntimeError("another remote model turn is already in flight")
        try:
            request_id = uuid4().hex
            body = {"request_id": request_id, "model": self._repo + "@" + self._revision,
                "token_ids": input_ids, "stream": False, "sampling_params": {
                    "n": 1, "seed": seed, "max_tokens": limit,
                    "temperature": settings["temperature"], "top_p": settings["top_p"],
                    "top_k": settings["top_k"] or -1,
                    "stop_token_ids": self._episode_renderer.get_stop_token_ids(),
                    "logprobs": 1, "skip_special_tokens": False}}
            try:
                with self._http.stream("POST", self._endpoint, json=body,
                        headers={"X-Request-Id": request_id}, timeout=self._timeout) as response:
                    if response.status_code != 200:
                        raise RuntimeError("remote inference refused the frozen request")
                    if response.headers.get("content-encoding", "identity") != "identity":
                        raise ValueError("remote inference response must be uncompressed")
                    raw = bytearray()
                    for chunk in response.iter_bytes():
                        if len(chunk) > MAX_RESPONSE_BYTES - len(raw):
                            raise ValueError("remote inference response exceeds its bound")
                        raw.extend(chunk)
                if self._closed:
                    raise RuntimeError("remote inference transport closed during generation")
                return self._completion(raw, request_id, body["model"], limit)
            except httpx.HTTPError:
                raise RuntimeError("remote inference transport failed; request was not retried") from None
        finally:
            self._inflight.release()

    def _completion(self, raw, request_id, model, limit):
        from renderers.client import _parse_completion_logprobs
        from renderers.base import ToolCallParseStatus

        try:
            data = json.loads(raw)
            if (data["request_id"] != "generate-tokens-" + request_id or data["model"] != model
                    or not isinstance(data["choices"], list) or len(data["choices"]) != 1):
                raise ValueError("remote model response binding changed")
            choice = data["choices"][0]
            ids = choice["token_ids"]
            if (type(choice["index"]) is not int or choice["index"] != 0
                    or choice["finish_reason"] not in {"stop", "length"}
                    or not isinstance(ids, list) or len(ids) > limit
                    or any(type(token) is not int or not 0 <= token < self._vocab_size for token in ids)):
                raise ValueError("remote model response tokens or termination are invalid")
            _parse_completion_logprobs(choice, ids)
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError("remote inference returned invalid sampled-token evidence") from None
        parsed = self._episode_renderer.parse_response(ids, tools=self._episode_tools)
        calls = [{"id": call.id or f"call_{index}", "name": call.name,
                  "arguments": call.arguments if isinstance(call.arguments, str) else json.dumps(call.arguments or {})}
                 for index, call in enumerate(parsed.tool_calls)
                 if call.name and call.status != ToolCallParseStatus.UNKNOWN_TOOL]
        reason = choice["finish_reason"]
        if reason == "stop" and any(call.status == ToolCallParseStatus.OK for call in parsed.tool_calls):
            reason = "tool_calls"
        return {"text": self.decode(ids), "token_ids": ids, "finish_reason": reason,
                "message": {"content": parsed.content or None,
                    "reasoning_content": parsed.reasoning_content, "tool_calls": calls or None}}
