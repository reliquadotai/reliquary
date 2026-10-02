"""Single-turn Verifiers tasksets as eval-set sources.

A public benchmark already packaged as a native Verifiers v1 taskset (AIME,
GPQA, MMLU-Pro, IFBench, LiveCodeBench...) becomes a set without rewriting it:
its tasks are frozen into rows at build time, and each completion is scored by
the task's own ``Task.score`` at grading time. ``verifiers`` is optional and
imported only here, so nothing else in reliquary depends on it.

Only what a miner can answer in one turn is taken: a task with tools, a prompt
that is not text, a conversation in the prompt, or a reward that may ask an LLM
judge is refused when the set is built, with the reason. A judge's verdict is
not a fact two gradings can both reproduce.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VERIFIERS_PREFIX = "verifiers:"


class RuntimeUnavailable(RuntimeError):
    """A task's reward runs code in a runtime and this host has none."""


def is_verifiers_source(name: Any) -> bool:
    return isinstance(name, str) and name.startswith(VERIFIERS_PREFIX)


def taskset_id(name: str) -> str:
    if not is_verifiers_source(name) or not name[len(VERIFIERS_PREFIX):]:
        raise ValueError(f"{name!r} is not a Verifiers source")
    return name[len(VERIFIERS_PREFIX):]


def prompt_digest(system: str | None, prompt: str) -> str:
    """The frozen prompt's identity: the system text and the user text."""
    return hashlib.sha256(f"{system or ''}\x00{prompt}".encode()).hexdigest()


@dataclass(frozen=True)
class FrozenTask:
    key: str
    system: str | None
    prompt: str


def _decorated_rewards(task) -> set[str]:
    from verifiers.v1.utils.decorators import discover_decorated

    return {fn.__name__ for fn in discover_decorated(task, "reward")}


def _judge_fields(config) -> list[str]:
    from verifiers.v1 import JudgeConfig

    return [name for name in type(config).model_fields
            if isinstance(getattr(config, name, None), JudgeConfig)]


def _requires_runtime(fn) -> bool:
    param = inspect.signature(fn).parameters.get("runtime")
    return param is not None and param.default is inspect.Parameter.empty


def _text(content: Any) -> str | None:
    return content if isinstance(content, str) and content else None


def freeze(task) -> FrozenTask:
    """The task as one optional system turn and one user turn, or ``ValueError``."""
    data = task.data
    name = getattr(data, "name", None) or getattr(data, "idx", "?")
    if getattr(data, "image", None) is not None:
        raise ValueError(f"task {name}: an image prompt is not text")
    system = getattr(data, "system_prompt", None)
    if system is not None and not _text(system):
        raise ValueError(f"task {name}: its system prompt is not text")
    prompt = data.prompt
    if isinstance(prompt, list):
        roles = [getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")
                 for m in prompt]
        contents = [getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
                    for m in prompt]
        if roles == ["user"]:
            prompt = contents[0]
        elif roles == ["system", "user"] and system is None:
            system, prompt = contents
        else:
            raise ValueError(f"task {name}: its prompt is the conversation {roles}, "
                             "not one user turn (multi-turn sets are not supported yet)")
    if not _text(prompt) or (system is not None and not _text(system)):
        raise ValueError(f"task {name}: its prompt is not text")
    return FrozenTask(key=str(task.key), system=system, prompt=prompt)


def refusal(task) -> str | None:
    """Why a task cannot be an eval-set problem, or None."""
    toolsets = getattr(type(task), "toolsets", None)
    if callable(toolsets) and list(toolsets(task.config) or ()):
        return "it gives the model tools: a multi-turn task (not supported yet)"
    config = task.config
    if list(getattr(config, "judges", None) or ()):
        return "its config adds LLM judges"
    judges = _judge_fields(config)
    if judges:
        replaced = {name for name, spec in (getattr(config, "rewards", None) or {}).items()
                    if getattr(spec, "fn", None)}
        kept = sorted(_decorated_rewards(task) - replaced)
        if kept:
            return (f"its rewards {kept} may call the LLM judge of {judges}; replace them "
                    "through task.rewards.<name>.fn with a deterministic reward")
    try:
        freeze(task)
    except ValueError as exc:
        return str(exc)
    return None


def _distribution_of(module_name: str) -> tuple[str | None, str | None]:
    from importlib.metadata import packages_distributions, version

    top = module_name.split(".", 1)[0]
    names = packages_distributions().get(top) or []
    if not names:
        return None, None
    try:
        return names[0], version(names[0])
    except Exception:
        return names[0], None


def _verifiers_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("verifiers")
    except Exception:
        return None


@dataclass
class TasksetHandle:
    """A loaded taskset and what a set records about it."""

    taskset_id: str
    args: dict
    taskset: Any
    package: str | None = None
    package_version: str | None = None
    verifiers_version: str | None = None
    score_trace: Callable[..., float] | None = None
    refuse: Callable[[Any], str | None] | None = None
    _by_key: dict | None = field(default=None, repr=False)

    def tasks(self) -> Iterator[Any]:
        return iter(self.taskset)

    def task(self, key: str):
        if self._by_key is None:
            self._by_key = {str(t.key): t for t in self.tasks()}
        try:
            return self._by_key[key]
        except KeyError:
            raise KeyError(f"{self.taskset_id}: no task {key!r}") from None

    def provenance(self) -> dict:
        return {"id": self.taskset_id, "args": self.args, "package": self.package,
                "package_version": self.package_version,
                "verifiers_version": self.verifiers_version}


def open_taskset(taskset_id_: str, args: dict) -> TasksetHandle:
    """The installed taskset under ``args`` (its config fields, nested as the
    ``eval`` CLI's ``--env.taskset.*`` would set them)."""
    try:
        import verifiers.v1 as vf
    except ImportError as exc:
        raise ValueError("a Verifiers source needs the `verifiers` package installed") from exc
    try:
        config = vf.taskset_config_type(taskset_id_)(id=taskset_id_, **args)
        taskset = vf.load_taskset(config)
    except ModuleNotFoundError as exc:
        raise ValueError(f"taskset {taskset_id_!r} is not installed: {exc}") from exc
    package, package_version = _distribution_of(type(taskset).__module__)
    return TasksetHandle(taskset_id_, dict(args), taskset, package=package,
                         package_version=package_version,
                         verifiers_version=_verifiers_version(), score_trace=score_trace,
                         refuse=refusal)


def needs_runtime(task) -> bool:
    return any(_requires_runtime(fn) for fn in task.hooks("reward"))


def _trace_for(task, frozen: FrozenTask, answer: str):
    import verifiers.v1 as vf
    from verifiers.v1.state import state_cls
    from verifiers.v1.trace import Trace, TraceTask

    nodes, parent = [], None
    for role, content in (("system", frozen.system), ("user", frozen.prompt),
                          ("assistant", answer)):
        if content is None:
            continue
        nodes.append({"parent": parent, "message": {"role": role, "content": content},
                      "sampled": role == "assistant"})
        parent = len(nodes) - 1
    return Trace(
        task=TraceTask(type=type(task).__name__, data=task.data, key=task.key,
                       hash=task.hash),
        state=state_cls(type(task))(),
        agent=vf.AgentInfo(config=vf.AgentConfig(), name="reliquary-grade", trainable=False),
        nodes=nodes,
    )


def score_trace(task, frozen: FrozenTask, answer: str) -> float:
    """``Task.score`` over a trace of the frozen prompt and ``answer``; a task
    whose reward needs a runtime gets a Docker box with every destination denied."""
    trace = _trace_for(task, frozen, answer)

    async def run() -> float:
        if not needs_runtime(task):
            await task.score(trace)
            return float(trace.reward)
        from verifiers.v1 import DockerConfig
        from verifiers.v1.runtimes import provision_runtime
        from verifiers.v1.utils.compile import resolve_runtime_config

        try:
            config = resolve_runtime_config(DockerConfig(allow=[]), task)
            async with provision_runtime(config, env=dict(task.runtime_env())) as runtime:
                await task.setup(trace, runtime)
                await task.score(trace, runtime)
        except Exception as exc:
            if _docker_missing(exc):
                raise RuntimeUnavailable(f"no Docker runtime here: {exc}") from exc
            raise
        return float(trace.reward)

    return asyncio.run(run())


def _docker_missing(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(s in text for s in ("docker", "no such file or directory", "connection refused"))


def score_answer(handle: TasksetHandle, key: str, system: str | None, prompt: str,
                 answer: str) -> float:
    task = handle.task(key)
    frozen = freeze(task)
    if prompt_digest(frozen.system, frozen.prompt) != prompt_digest(system, prompt):
        raise LookupError("source_drift")
    return float(handle.score_trace(task, frozen, answer))


def _args_tag(args: dict) -> str:
    if not args:
        return ""
    body = json.dumps(args, sort_keys=True, separators=(",", ":")).encode()
    return f"-a{hashlib.sha256(body).hexdigest()[:8]}"


def build_set(source: str, *, out: str | Path, start: int, count: int | None,
              sample: int | None, seed: int | None, set_id: str | None,
              taskset_args: dict, open_taskset: Callable[[str, dict], TasksetHandle],
              clock: Callable[[], float] = time.time) -> dict:
    """Freeze tasks ``[start, start + count)`` of the taskset, whole or sampled."""
    from reliquary.eval import sets

    name = taskset_id(source)
    directory = sets._empty_directory(out)
    handle = open_taskset(name, taskset_args)
    tasks = list(handle.tasks())
    length = len(tasks)
    high = length if count is None else start + count
    if high > length:
        raise ValueError(f"[{start}, {high}) runs past {source!r} ({length} tasks)")
    if start >= high:
        raise ValueError(f"{source!r} holds no task from {start}")
    indices = sets.select_indices(start, high, sample=sample, seed=seed)
    base = f"{source.replace(':', '-')}{_args_tag(taskset_args)}-r{start}-n{high - start}"
    set_id = sets.validated_set_id(
        (set_id or (base if sample is None else f"{base}-k{sample}-s{seed}")).lower())
    prompts, grading, runtime = [], [], False
    for ordinal, index in enumerate(indices):
        task = tasks[index]
        why = (handle.refuse or refusal)(task)
        if why is not None:
            raise ValueError(f"{source!r} task {index} cannot be a set problem: {why}")
        frozen = freeze(task)
        runtime = runtime or needs_runtime(task)
        problem_id = f"{set_id}-{ordinal:06d}"
        messages = ([{"role": "system", "content": frozen.system}] if frozen.system else []) \
            + [{"role": "user", "content": frozen.prompt}]
        prompts.append({"problem_id": problem_id, "env": source, "set_id": set_id,
                        "interaction": sets.INTERACTION, "messages": messages})
        grading.append({"problem_id": problem_id, "source": source, "source_index": index,
                        "task_key": frozen.key,
                        "prompt_sha256": prompt_digest(frozen.system, frozen.prompt)})
    card = {
        "schema": sets.SET_SCHEMA, "set_id": set_id, "env": source, "source": source,
        "source_kind": "verifiers", "interaction": sets.INTERACTION, "split": None,
        "index_range": [start, high], "source_length": length, "count": len(indices),
        "seed": seed, "selection": {"start": start, "count": count, "sample": sample,
                                    "seed": seed},
        "order": ("the taskset's tasks in its order" if sample is None else
                  "random.Random(seed).sample(index_range, sample); an order of N "
                  "problems takes the first N"),
        "created_at": clock(), "taskset": handle.provenance(), "needs_runtime": runtime,
        "prompt_template_id": None, "default_max_new_tokens": None,
        "environment_manifest_sha256": None,
        "disjointness": {"external_benchmark": True, "disjoint": True, "rl": [],
                         "corpus": [], "held_out": [],
                         "note": "not in Reliquary training data"},
    }
    return sets._write_set(directory, prompts, grading, card)


__all__ = [
    "FrozenTask",
    "RuntimeUnavailable",
    "TasksetHandle",
    "VERIFIERS_PREFIX",
    "build_set",
    "freeze",
    "is_verifiers_source",
    "needs_runtime",
    "open_taskset",
    "prompt_digest",
    "refusal",
    "score_answer",
    "score_trace",
    "taskset_id",
]
