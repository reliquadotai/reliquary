"""`reliquary eval run`: the pod side of an evaluation order.

The pod generates and never grades. It claims a task, loads the model at its
pinned revision, renders each problem with the model's chat template, samples
with vLLM, and uploads ``completions-NNNNN.jsonl`` chunks through the platform's
multipart API. Each line: ``{problem_id, sample_index, completion,
completion_tokens, finish_reason}``.

A crash resumes from ``work_dir``: chunks already uploaded are skipped, a chunk
written but not uploaded is sent as written, parts the platform has are not
sent again. Lost API contact (a refused lease, or heartbeats failing in a row)
stops the work at the next chunk boundary.

vLLM is imported only by ``VLLMGenerator``, so the module loads on a CPU box.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import signal
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from reliquary.eval.platform_client import LeaseLost, PlatformClient, PlatformError

logger = logging.getLogger(__name__)

CHUNK_PROBLEMS = 64
HEARTBEAT_SECONDS = 30.0
# Heartbeats failing in a row before the pod counts the platform as lost.
MAX_HEARTBEAT_FAILURES = 4
STATE_SCHEMA = "reliquary/eval-run-state/v1"


@dataclass(frozen=True)
class Completion:
    text: str
    tokens: int
    finish_reason: str
    episode: dict | None = None


class Generator(Protocol):
    def load(self, *, repo: str, revision: str, gpu_count: int, max_new_tokens: int,
             rows: Sequence[dict], thinking: bool) -> dict:
        """Load the model; ``{vllm_version, gpu, model_sha}`` (and may add
        ``chat_template_sha256``). ``rows`` are the task's problems, for sizing."""

    def render(self, row: dict, *, thinking: bool) -> Any:
        """One problem's prompt, as ``generate`` takes it (token ids for vLLM)."""

    def generate(self, prompts: Sequence[Any], *, samples: Sequence[int],
                 seeds: Sequence[int], sampling: dict,
                 max_new_tokens: int) -> list[list[Completion]]:
        """``samples[i]`` completions of ``prompts[i]``, each request seeded."""


def render_prompt(tokenizer: Any, row: dict, *, thinking: bool) -> list[int]:
    """The prompt's token ids. With a chat template, the template's text (passed
    ``enable_thinking``; a template that ignores it is unchanged) is encoded
    WITHOUT special tokens: the template already holds its BOS, and adding one
    again (a Llama-style double BOS) degrades every score. Raw text, with no
    template to hold it, is encoded with them."""
    messages = row.get("messages")
    if messages and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=thinking)
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])
    if isinstance(row.get("text"), str):
        text = row["text"]
    elif messages:
        text = "\n\n".join(str(m.get("content", "")) for m in messages)
    else:
        raise ValueError(f"problem {row.get('problem_id')!r} has neither messages nor text")
    return list(tokenizer(text, add_special_tokens=True)["input_ids"])


def normalized_sampling(sampling: dict | None) -> dict:
    """The sampling a task names, with an absent or null field at its default:
    temperature 1, top_p 1, no top-k, seed 0."""
    sampling = dict(sampling or {})

    def value(name, default, kind):
        raw = sampling.get(name)
        return default if raw is None else kind(raw)

    return {"temperature": value("temperature", 1.0, float), "top_p": value("top_p", 1.0, float),
            "top_k": value("top_k", 0, int), "seed": value("seed", 0, int)}


def problem_seed(seed: int, problem_id: str) -> int:
    """One seed per problem, so a chunk boundary never changes a sample."""
    digest = hashlib.sha256(f"{int(seed)}:{problem_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def resolved_revision(directory: str | Path, revision: str) -> str:
    """The commit a Hugging Face snapshot was downloaded at (its directory
    name), refused unless it is the pinned revision."""
    resolved = Path(directory).name
    if len(resolved) != 40 or any(c not in "0123456789abcdef" for c in resolved):
        raise RuntimeError(f"snapshot directory {directory} does not name a commit")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or revision != resolved:
        raise RuntimeError(f"asked for revision {revision}, Hugging Face served {resolved}")
    return resolved


def validate_model(repo: str, revision: str) -> None:
    if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo) \
            or ".." in repo or not isinstance(revision, str) \
            or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model must name an HF repo and exact lowercase 40-hex commit")


# What a snapshot needs to load; skips duplicate original/*.pth checkpoints.
SNAPSHOT_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt", "*.jinja"]


class VLLMGenerator:
    """vLLM, tensor parallel over the pod's GPUs."""

    def __init__(self) -> None:
        self._llm = None
        self._tokenizer = None
        self._episode_renderer = None

    def load(self, *, repo: str, revision: str, gpu_count: int, max_new_tokens: int,
             rows: Sequence[dict], thinking: bool, model_access: str = "public",
             hf_token: str | None = None, max_total_tokens: int | None = None) -> dict:
        validate_model(repo, revision)
        if model_access not in {"public", "private"}:
            raise ValueError("unsupported model_access")
        token = (hf_token or os.environ.get("HF_TOKEN", "").strip()) if model_access == "private" else False
        if model_access == "private" and not token:
            raise ValueError("private checkpoint requires a trusted executor credential")
        import torch
        import vllm
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        try:
            directory = snapshot_download(repo, revision=revision, token=token,
                                          allow_patterns=SNAPSHOT_PATTERNS)
        except Exception:
            # Third-party exceptions may include request headers or credential text.
            raise RuntimeError("checkpoint download failed") from None
        model_sha = resolved_revision(directory, revision)
        snapshot = Path(directory)
        if snapshot.parent.name != "snapshots" or snapshot.parent.parent.name != "models--" + repo.replace("/", "--"):
            raise RuntimeError("snapshot repository does not match the claimed model")
        self._tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=False, local_files_only=True)
        # The context the task needs, not the model's maximum: a 128k default
        # can refuse to start on the GPUs the quote sized for this task.
        longest = max(len(render_prompt(self._tokenizer, r, thinking=thinking)) for r in rows)
        if max_total_tokens is not None:
            config = json.loads((snapshot / "config.json").read_text())
            context = config.get("max_position_embeddings", config.get("n_positions"))
            if type(context) is not int or context < max_total_tokens:
                raise ValueError("checkpoint context is below the frozen transcript budget")
            if longest + int(max_new_tokens) > max_total_tokens:
                raise ValueError("prompt plus output budget exceeds the frozen transcript budget")
        self._llm = vllm.LLM(model=directory, tensor_parallel_size=int(gpu_count),
                             trust_remote_code=False,
                             max_model_len=max_total_tokens or longest + int(max_new_tokens))
        self._max_total_tokens = max_total_tokens
        self._max_turn_tokens = int(max_new_tokens)
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        template = getattr(self._tokenizer, "chat_template", None) or ""
        return {"vllm_version": str(vllm.__version__), "gpu": f"{gpu_count}x {name}",
                "model_sha": model_sha,
                "chat_template_sha256": hashlib.sha256(str(template).encode()).hexdigest()}

    def render(self, row: dict, *, thinking: bool) -> list[int]:
        return render_prompt(self._tokenizer, row, thinking=thinking)

    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens is not False:
            raise ValueError("episode fragments must not add special tokens")
        return list(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, token_ids, *, skip_special_tokens=False):
        if skip_special_tokens is not False:
            raise ValueError("episode evidence must retain special tokens")
        return self._tokenizer.decode(token_ids, skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)

    def generate(self, prompts, *, samples, seeds, sampling, max_new_tokens):
        from vllm import SamplingParams

        sampling = normalized_sampling(sampling)
        params = [SamplingParams(n=int(n), temperature=sampling["temperature"],
                                 top_p=sampling["top_p"],
                                 top_k=-1 if not sampling["top_k"] else sampling["top_k"],
                                 seed=int(seed), max_tokens=int(max_new_tokens))
                  for n, seed in zip(samples, seeds)]
        outputs = self._llm.generate([{"prompt_token_ids": list(ids)} for ids in prompts],
                                     params, use_tqdm=False)
        return [[Completion(text=o.text, tokens=len(o.token_ids),
                            finish_reason=str(o.finish_reason)) for o in out.outputs]
                for out in outputs]

    def render_messages(self, messages, *, tools, thinking, renderer_id):
        from importlib.metadata import version
        from renderers import create_renderer
        from renderers.configs import Qwen38RendererConfig

        if renderer_id != "renderers:qwen38@0.1.11" or version("renderers") != "0.1.11":
            raise ValueError("unsupported pinned episode renderer")
        self._episode_renderer = create_renderer(self._tokenizer,
            Qwen38RendererConfig(enable_thinking=thinking))
        self._episode_tools = tools
        return self._episode_renderer.render_ids(messages, tools=tools, add_generation_prompt=True)

    def generate_turn(self, input_ids, *, max_new_tokens, seed, sampling):
        from vllm import SamplingParams

        if (not isinstance(input_ids, list) or not input_ids
                or any(type(token) is not int or token < 0 for token in input_ids)
                or self._max_total_tokens is None or type(max_new_tokens) is not int
                or max_new_tokens <= 0):
            raise ValueError("a model turn requires token IDs and a frozen transcript budget")
        limit = min(max_new_tokens, self._max_turn_tokens, self._max_total_tokens - len(input_ids))
        if limit <= 0:
            raise ValueError("episode transcript token budget exhausted")
        settings = normalized_sampling(sampling)
        from reliquary.miner.suite_episode import suite_stop_ids

        stop_ids = (self._episode_renderer.get_stop_token_ids() if self._episode_renderer
                    else suite_stop_ids(self.encode))
        params = SamplingParams(n=1, temperature=settings["temperature"], top_p=settings["top_p"],
            top_k=settings["top_k"] or -1, seed=seed, max_tokens=limit,
            stop_token_ids=stop_ids)
        out = self._llm.generate([{"prompt_token_ids": input_ids}], [params], use_tqdm=False)
        if len(out) != 1 or len(out[0].outputs) != 1:
            raise RuntimeError("model turn returned another sample count")
        completion = out[0].outputs[0]
        ids = list(completion.token_ids)
        if len(ids) > limit:
            raise RuntimeError("model turn exceeded its token budget")
        result = {"text": self.decode(ids), "token_ids": ids, "finish_reason": str(completion.finish_reason)}
        if self._episode_renderer is not None:
            from renderers.base import ToolCallParseStatus

            parsed = self._episode_renderer.parse_response(ids, tools=self._episode_tools)
            calls = [{"id": call.id or f"call_{index}", "name": call.name,
                      "arguments": call.arguments if isinstance(call.arguments, str) else json.dumps(call.arguments or {})}
                     for index, call in enumerate(parsed.tool_calls)
                     if call.name and call.status != ToolCallParseStatus.UNKNOWN_TOOL]
            result["message"] = {"content": parsed.content or None,
                "reasoning_content": parsed.reasoning_content, "tool_calls": calls or None}
            if completion.finish_reason == "stop" and any(
                    call.status == ToolCallParseStatus.OK for call in parsed.tool_calls):
                result["finish_reason"] = "tool_calls"
        return result


class _Heartbeat:
    """Heartbeats beside the work; ``lost`` once the platform is gone."""

    def __init__(self, client: PlatformClient, every: float, max_failures: int) -> None:
        self._client, self._every, self._max = client, every, max_failures
        self._stop = threading.Event()
        self.lost: str | None = None
        self._thread = threading.Thread(target=self._loop, name="eval-heartbeat", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        failures = 0
        while not self._stop.wait(self._every):
            try:
                self._client.heartbeat()
                failures = 0
            except LeaseLost as exc:
                self.lost = str(exc)
                return
            except Exception as exc:
                failures += 1
                logger.warning("eval heartbeat failed (%d in a row): %r", failures, exc)
                if failures >= self._max:
                    self.lost = f"{failures} heartbeats failed in a row: {exc!r}"
                    return

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvalRunner:
    def __init__(self, client: PlatformClient, generator: Generator, *,
                 work_dir: str | Path, chunk_problems: int = CHUNK_PROBLEMS,
                 heartbeat_seconds: float = HEARTBEAT_SECONDS,
                 max_heartbeat_failures: int = MAX_HEARTBEAT_FAILURES,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if chunk_problems <= 0:
            raise ValueError("chunk_problems must be positive")
        self._client, self._generator = client, generator
        self._root = Path(work_dir)
        self._chunk = chunk_problems
        self._heartbeat_seconds = heartbeat_seconds
        self._max_failures = max_heartbeat_failures
        self._clock = clock

    # ---- local state, what survives a crash ---------------------------------

    def _state_path(self, task_id: str) -> Path:
        return self._root / task_id / "state.json"

    def _load_state(self, task_id: str, plan: str) -> dict:
        path = self._state_path(task_id)
        if path.exists():
            state = json.loads(path.read_text())
            if state.get("plan") == plan:
                return state
            logger.warning("eval task %s: the problem plan changed, starting over", task_id)
        return {"schema": STATE_SCHEMA, "task_id": task_id, "plan": plan, "chunks": {},
                "info": None, "seconds": 0.0}

    def _save_state(self, state: dict) -> None:
        path = self._state_path(state["task_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True))
        os.replace(temporary, path)

    # ---- the run -----------------------------------------------------------

    def run(self) -> dict | None:
        """Claim and run one task; the posted result, or None when there was none."""
        task = self._client.claim()
        if task is None:
            return None
        heartbeat = _Heartbeat(self._client, self._heartbeat_seconds, self._max_failures)
        heartbeat.start()
        try:
            return self._run(task, heartbeat)
        except LeaseLost:
            logger.error("eval task %s: the platform refused this pod; stopping",
                         task.get("task_id"))
            raise
        except Exception as exc:
            try:
                self._client.event("error", {"message": f"{type(exc).__name__}: {exc}"[:1000]})
            except Exception:
                logger.warning("eval task %s: the error event was not delivered",
                               task.get("task_id"), exc_info=True)
            raise
        finally:
            heartbeat.stop()
            from reliquary.eval.model_process import ModelProcess

            if isinstance(self._generator, ModelProcess):
                self._generator.close()

    def _check(self, heartbeat: _Heartbeat) -> None:
        if heartbeat.lost is not None:
            raise LeaseLost(f"platform contact lost: {heartbeat.lost}")

    def _rows(self, task: dict) -> list[dict]:
        samples_by_set = {e["set_id"]: int(e["samples"]) for e in task.get("envs", [])}
        rows, seen = [], set()
        for row in self._client.prompts():
            problem_id = row.get("problem_id")
            if not isinstance(problem_id, str) or not problem_id or problem_id in seen:
                raise PlatformError(f"prompt row with a bad or repeated problem_id: {problem_id!r}")
            seen.add(problem_id)
            samples = int(row.get("samples") or samples_by_set.get(row.get("set_id"), 0))
            if samples <= 0:
                raise PlatformError(f"problem {problem_id!r} asks for no samples")
            rows.append({**row, "samples": samples})
        if not rows:
            raise PlatformError("the task has no prompts")
        if task.get("protocol"):
            from collections import Counter

            wanted = {entry["set_id"]: entry for entry in task["protocol"]["sets"]}
            observed = Counter()
            for row in rows:
                entry = wanted.get(row.get("set_id"))
                if entry is None or row.get("env") != entry["env"] or row["samples"] != entry["samples"]:
                    raise PlatformError("prompt row differs from the frozen experiment")
                observed[row["set_id"]] += 1
            if dict(observed) != {key: entry["problems"] for key, entry in wanted.items()}:
                raise PlatformError("prompt counts differ from the frozen experiment")
        return rows

    def _run(self, task: dict, heartbeat: _Heartbeat) -> dict:
        from reliquary.eval.protocol import validate_execution, validate_request

        task_id = str(task["task_id"])
        validate_model(task["model"]["repo"], task["model"]["revision"])
        access = task.get("model_access", "public")
        if access not in {"public", "private"}:
            raise ValueError("unsupported model_access")
        validate_request(task)
        if task.get("protocol"):
            validate_execution(task.get("set_protocols"), task["protocol_bindings"], request=task["protocol"])
        selected_sets = [{"set_id": entry["set_id"], "env": entry["env"], "samples": entry["samples"],
                          "problems": entry.get("problems", entry.get("problem_ids_count"))}
                         for entry in task.get("envs", [])]
        if task.get("protocol") and task["protocol"].get("sets") != selected_sets:
            raise ValueError("request protocol selected sets changed")
        if task.get("protocol") and type(self._generator) is VLLMGenerator:
            from reliquary.eval.model_process import ModelProcess

            self._generator = ModelProcess(VLLMGenerator,
                timeout_seconds=task["protocol"]["episode_timeout_seconds"],
                cancelled=lambda: heartbeat.lost is not None)
        started = self._clock()
        rows = self._rows(task)
        plan = hashlib.sha256(json.dumps(
            [self._chunk, [(r["problem_id"], r["samples"]) for r in rows],
             task["model"], access, task.get("protocol_sha256"), task.get("protocol_bindings"),
             task.get("sampling"), task["max_new_tokens"], bool(task["thinking"])],
            sort_keys=True).encode()).hexdigest()
        state = self._load_state(task_id, plan)
        directory = self._root / task_id
        directory.mkdir(parents=True, exist_ok=True)
        chunks = [rows[i:i + self._chunk] for i in range(0, len(rows), self._chunk)]
        sampling, thinking = normalized_sampling(task.get("sampling")), bool(task["thinking"])
        keys, total_rows, done = [], 0, 0

        def ensure_loaded() -> dict:
            if getattr(self, "_loaded_for", None) != task_id:
                loaded_at = self._clock()
                private = {}
                if access == "private":
                    token = self._client.model_credential(task["model"])
                    private = {"model_access": "private", "hf_token": token}
                if task.get("protocol"):
                    private["max_total_tokens"] = task["protocol"]["max_episode_tokens"]
                info = self._generator.load(
                    repo=task["model"]["repo"], revision=task["model"]["revision"],
                    gpu_count=int(task.get("gpu_count") or 1),
                    max_new_tokens=int(task["max_new_tokens"]), rows=rows, thinking=thinking,
                    **private)
                if info.get("model_sha") != task["model"]["revision"]:
                    raise ValueError("loaded checkpoint differs from the claimed revision")
                state["info"] = {k: str(info[k]) for k in ("vllm_version", "gpu", "model_sha")}
                self._save_state(state)
                self._loaded_for = task_id
                self._client.event("model_loaded", {
                    **{k: str(info[k]) for k in ("vllm_version", "gpu", "model_sha", "chat_template_sha256") if k in info},
                    "seconds": self._clock() - loaded_at})
            return state["info"]

        for index, chunk in enumerate(chunks):
            self._check(heartbeat)
            name = f"completions-{index:05d}.jsonl"
            entry = state["chunks"].get(name) or {}
            chunk_rows = sum(r["samples"] for r in chunk)
            if not entry.get("key"):
                path = directory / name
                if not (entry.get("sha256") and path.exists()
                        and _sha256_file(path) == entry["sha256"]):
                    ensure_loaded()
                    began = self._clock()
                    outputs = []
                    if (task.get("protocol") or {}).get("interaction", "single_turn") != "single_turn":
                        for row in chunk:
                            outputs.append([self._episode(task, row, index, heartbeat)
                                            for index in range(row["samples"])])
                    elif task.get("protocol"):
                        for row in chunk:
                            completions = []
                            for sample_index in range(row["samples"]):
                                self._check(heartbeat)
                                if hasattr(self._generator, "begin_episode"):
                                    self._generator.begin_episode()
                                prompt = self._generator.render(row, thinking=thinking)
                                seed = (problem_seed(sampling["seed"], row["problem_id"]) + sample_index) % 2**31
                                generated = self._generator.generate([prompt], samples=[1], seeds=[seed],
                                    sampling=sampling, max_new_tokens=int(task["max_new_tokens"]))
                                if len(generated) != 1 or len(generated[0]) != 1:
                                    raise RuntimeError("one episode returned another sample count")
                                completions.append(generated[0][0])
                            outputs.append(completions)
                    else:
                        self._check(heartbeat)
                        prompts = [self._generator.render(r, thinking=thinking) for r in chunk]
                        outputs.extend(self._generator.generate(
                            prompts, samples=[r["samples"] for r in chunk],
                            seeds=[problem_seed(sampling["seed"], r["problem_id"])
                                   for r in chunk],
                            sampling=sampling, max_new_tokens=int(task["max_new_tokens"])))
                    row_limit = (((task["protocol"]["max_episode_tokens"]
                                   if task["protocol"]["interaction"] != "single_turn"
                                   else task["max_new_tokens"]) * 6 + 1024) * 2
                                 if task.get("protocol") else None)
                    body = self._chunk_body(chunk, outputs, row_limit=row_limit)
                    if len(body) > 4 * 1024**3:
                        raise ValueError("completion upload exceeds its object byte allowance")
                    temporary = path.with_suffix(".tmp")
                    temporary.write_bytes(body)
                    os.replace(temporary, path)
                    entry = {"sha256": hashlib.sha256(body).hexdigest(), "rows": chunk_rows}
                    state["seconds"] += self._clock() - began
                    state["chunks"][name] = entry
                    self._save_state(state)
                self._check(heartbeat)

                def started_upload(upload: dict, entry=entry) -> None:
                    entry["upload"] = upload
                    self._save_state(state)

                entry["key"] = self._client.upload_file(
                    path, name=name, resume=entry.get("upload"), on_created=started_upload)
                self._save_state(state)
            keys.append(entry["key"])
            total_rows += entry["rows"]
            done += len(chunk)
            self._client.event("progress", {"done": done, "total": len(rows)})
        self._check(heartbeat)
        info = state["info"] or ensure_loaded()
        result = {"completion_keys": keys, **info, "rows": total_rows,
                  "seconds": round(state["seconds"] or (self._clock() - started), 3)}
        if task.get("protocol_sha256"):
            result["protocol_sha256"] = task["protocol_sha256"]
        self._client.result(**result)
        state["result"] = result
        self._save_state(state)
        logger.info("eval task %s: %d rows in %d chunks", task_id, total_rows, len(keys))
        return result

    def _episode(self, task, row, sample_index, heartbeat):
        from reliquary.eval.protocol import digest

        self._check(heartbeat)
        limits = task["protocol"]
        source = task["set_protocols"][row["set_id"]]["protocol"]["source"]
        seed = (problem_seed(normalized_sampling(task["sampling"])["seed"], row["problem_id"]) + sample_index) % 2**31
        self._generator.begin_episode()
        deadline = time.monotonic() + limits["episode_timeout_seconds"]
        if limits["interaction"] == "signed_episode":
            import asyncio
            from reliquary.eval.signed_agentic import run_signed_episode

            grant = self._client.sandbox_session(row["set_id"], row["problem_id"], sample_index,
                                                timeout_seconds=max(0.001, deadline - time.monotonic()))
            result = asyncio.run(run_signed_episode(generator=self._generator, task=task, row=row,
                sample_index=sample_index, grant=grant, seed=seed, deadline=deadline))
            return Completion(result["completion"], result["completion_tokens"], result["finish_reason"],
                {**result["episode"], **{key: result[key] for key in ("turns", "tool_calls", "transcript_tokens")}})
        from reliquary.eval.agentic import EpisodeCompletion, load_telecom_benchmark, run_telecom_episode
        from reliquary.environment.agentic.renderers import renderer_for

        if not hasattr(self, "_episode_sources"):
            self._episode_sources = {}
        key = source["environment_manifest_sha256"]
        if key not in self._episode_sources:
            self._episode_sources[key] = load_telecom_benchmark(key)
        benchmark = self._episode_sources[key]
        problem = benchmark.environment.get_task(row["source_index"])
        if problem.to_public_wire() != row.get("episode_task"):
            raise ValueError("episode prompt differs from the frozen source")
        if digest([tool.to_wire() for tool in problem.tools]) != source["episode"]["tools_sha256"]:
            raise ValueError("episode tool schema differs from the frozen source")
        renderer = renderer_for(source["episode"]["renderer_id"], self._generator.encode)
        def generate_turn(*, tokens, max_new_tokens, seed, deadline):
            answer = self._generator.generate_turn(list(tokens), max_new_tokens=max_new_tokens,
                seed=seed, sampling=task["sampling"], absolute_deadline=deadline)
            return EpisodeCompletion(answer["text"], tuple(answer["token_ids"]), answer["finish_reason"])
        result = run_telecom_episode(benchmark, task=problem, seed=seed, renderer=renderer,
            generate_turn=generate_turn, decode=self._generator.decode,
            max_turns=limits["max_turns"], max_new_tokens=task["max_new_tokens"],
            max_episode_tokens=limits["max_episode_tokens"],
            episode_timeout_seconds=max(0.001, deadline - time.monotonic()),
            cancelled=lambda: heartbeat.lost is not None)
        return Completion(result.trace.actions[-1].content or "", result.generated_tokens,
            result.finish_reasons[-1] if result.finish_reasons else "stop",
            {"trace": result.trace.to_wire(include_tokens=True), "seed": seed,
             "turns": len(result.trace.actions), "tool_calls": result.tool_calls,
             "transcript_tokens": len(result.trace.tokens),
             "stop_reason": result.trace.termination_reason})

    @staticmethod
    def _chunk_body(chunk: list[dict], outputs: list[list[Completion]], *, row_limit=None) -> bytes:
        if len(outputs) != len(chunk):
            raise RuntimeError(f"generator returned {len(outputs)} outputs for {len(chunk)} prompts")
        lines = []
        for row, completions in zip(chunk, outputs):
            if len(completions) != row["samples"]:
                raise RuntimeError(f"{row['problem_id']}: {len(completions)} completions, "
                                   f"{row['samples']} asked")
            for sample_index, completion in enumerate(completions):
                line = json.dumps({
                    "problem_id": row["problem_id"], "sample_index": sample_index,
                    "completion": completion.text, "completion_tokens": int(completion.tokens),
                    "finish_reason": completion.finish_reason,
                    **({"episode": completion.episode} if completion.episode is not None else {}),
                }, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                if row_limit is not None and len(line) > row_limit:
                    raise ValueError("completion row exceeds its frozen byte allowance")
                lines.append(line)
        return b"".join(lines)


def run_evaluation(*, platform: str, executor_id: str, work_dir: str,
                   chunk_problems: int = CHUNK_PROBLEMS) -> dict | None:
    from reliquary.eval.platform_client import TOKEN_ENV

    client = PlatformClient(platform, os.environ.get(TOKEN_ENV, "").strip(),
                            executor_id=executor_id)
    previous = (signal.getsignal(signal.SIGTERM)
                if threading.current_thread() is threading.main_thread() else None)
    def stopping(signum, frame):
        raise LeaseLost("evaluation executor stopping")
    if previous is not None:
        signal.signal(signal.SIGTERM, stopping)
    try:
        return EvalRunner(client, VLLMGenerator(), work_dir=work_dir,
                          chunk_problems=chunk_problems).run()
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
        client.close()


__all__ = [
    "CHUNK_PROBLEMS",
    "Completion",
    "EvalRunner",
    "Generator",
    "VLLMGenerator",
    "normalized_sampling",
    "problem_seed",
    "render_prompt",
    "resolved_revision",
    "run_evaluation",
]
