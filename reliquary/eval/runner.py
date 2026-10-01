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
    if revision != resolved and not (len(revision) < 40 and resolved.startswith(revision)):
        raise RuntimeError(f"asked for revision {revision}, Hugging Face served {resolved}")
    return resolved


# What a snapshot needs to load; skips duplicate original/*.pth checkpoints.
SNAPSHOT_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt", "*.jinja"]


class VLLMGenerator:
    """vLLM, tensor parallel over the pod's GPUs."""

    def __init__(self) -> None:
        self._llm = None
        self._tokenizer = None

    def load(self, *, repo: str, revision: str, gpu_count: int, max_new_tokens: int,
             rows: Sequence[dict], thinking: bool) -> dict:
        import torch
        import vllm
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        directory = snapshot_download(repo, revision=revision, token=False,
                                      allow_patterns=SNAPSHOT_PATTERNS)
        model_sha = resolved_revision(directory, revision)
        self._tokenizer = AutoTokenizer.from_pretrained(directory)
        # The context the task needs, not the model's maximum: a 128k default
        # can refuse to start on the GPUs the quote sized for this task.
        longest = max(len(render_prompt(self._tokenizer, r, thinking=thinking)) for r in rows)
        self._llm = vllm.LLM(model=directory, tensor_parallel_size=int(gpu_count),
                             trust_remote_code=False,
                             max_model_len=longest + int(max_new_tokens))
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        template = getattr(self._tokenizer, "chat_template", None) or ""
        return {"vllm_version": str(vllm.__version__), "gpu": f"{gpu_count}x {name}",
                "model_sha": model_sha,
                "chat_template_sha256": hashlib.sha256(str(template).encode()).hexdigest()}

    def render(self, row: dict, *, thinking: bool) -> list[int]:
        return render_prompt(self._tokenizer, row, thinking=thinking)

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
        return rows

    def _run(self, task: dict, heartbeat: _Heartbeat) -> dict:
        task_id = str(task["task_id"])
        started = self._clock()
        rows = self._rows(task)
        plan = hashlib.sha256(json.dumps(
            [self._chunk, [(r["problem_id"], r["samples"]) for r in rows],
             task["model"], task.get("sampling"), task["max_new_tokens"], bool(task["thinking"])],
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
                info = self._generator.load(
                    repo=task["model"]["repo"], revision=task["model"]["revision"],
                    gpu_count=int(task.get("gpu_count") or 1),
                    max_new_tokens=int(task["max_new_tokens"]), rows=rows, thinking=thinking)
                state["info"] = {k: str(info[k]) for k in ("vllm_version", "gpu", "model_sha")}
                self._save_state(state)
                self._loaded_for = task_id
                self._client.event("model_loaded", {
                    **{k: str(v) for k, v in info.items()},
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
                    prompts = [self._generator.render(r, thinking=thinking) for r in chunk]
                    outputs = self._generator.generate(
                        prompts, samples=[r["samples"] for r in chunk],
                        seeds=[problem_seed(sampling["seed"], r["problem_id"])
                               for r in chunk],
                        sampling=sampling, max_new_tokens=int(task["max_new_tokens"]))
                    body = self._chunk_body(chunk, outputs)
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
        self._client.result(**result)
        state["result"] = result
        self._save_state(state)
        logger.info("eval task %s: %d rows in %d chunks", task_id, total_rows, len(keys))
        return result

    @staticmethod
    def _chunk_body(chunk: list[dict], outputs: list[list[Completion]]) -> bytes:
        if len(outputs) != len(chunk):
            raise RuntimeError(f"generator returned {len(outputs)} outputs for {len(chunk)} prompts")
        lines = []
        for row, completions in zip(chunk, outputs):
            if len(completions) != row["samples"]:
                raise RuntimeError(f"{row['problem_id']}: {len(completions)} completions, "
                                   f"{row['samples']} asked")
            for sample_index, completion in enumerate(completions):
                lines.append(json.dumps({
                    "problem_id": row["problem_id"], "sample_index": sample_index,
                    "completion": completion.text, "completion_tokens": int(completion.tokens),
                    "finish_reason": completion.finish_reason,
                }, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        return b"".join(lines)


def run_evaluation(*, platform: str, executor_id: str, work_dir: str,
                   chunk_problems: int = CHUNK_PROBLEMS) -> dict | None:
    from reliquary.eval.platform_client import TOKEN_ENV

    client = PlatformClient(platform, os.environ.get(TOKEN_ENV, "").strip(),
                            executor_id=executor_id)
    try:
        return EvalRunner(client, VLLMGenerator(), work_dir=work_dir,
                          chunk_problems=chunk_problems).run()
    finally:
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
