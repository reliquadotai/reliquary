"""Small file-editing environment that forms the safe baseline for SWE tasks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any

from reliquary.environment.agentic.types import (
    AssistantAction,
    EpisodeEvent,
    EpisodeTask,
    EpisodeTrace,
    ResetResult,
    RewardCheck,
    RewardReport,
    StepResult,
    ToolSpec,
    canonical_json,
    sha256_json,
)


TASK_COUNT = 1 << 31
GENERATOR_VERSION = "reliquary-workspace-tools-generator-v1"


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOLS = (
    ToolSpec("list_files", "List files in the workspace.", _schema({}, [])),
    ToolSpec(
        "read_file",
        "Read one UTF-8 text file.",
        _schema({"path": {"type": "string"}}, ["path"]),
    ),
    ToolSpec(
        "write_file",
        "Replace one UTF-8 text file with supplied content.",
        _schema(
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
    ),
    ToolSpec("run_tests", "Run the deterministic workspace checks.", _schema({}, [])),
    ToolSpec(
        "finish",
        "Finish and summarize the change.",
        _schema({"response": {"type": "string"}}, ["response"]),
    ),
)


@dataclass(slots=True)
class WorkspaceState:
    root: Path
    expected_path: str
    expected_content: str
    original_files: dict[str, str]
    final_response: str | None = None
    tests_passed: bool = False
    invalid_actions: int = 0
    closed: bool = False


def _build_task(index: int) -> EpisodeTask:
    index = int(index) % TASK_COUNT
    digest = hashlib.sha256(f"{GENERATOR_VERSION}:{index}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:16], "big"))
    family = index % 4
    if family == 0:
        multiplier = rng.randrange(2, 10)
        original = (
            "def scale(values):\n"
            "    \"\"\"Return each input multiplied by the configured factor.\"\"\"\n"
            "    return [value + FACTOR for value in values]\n\n"
            f"FACTOR = {multiplier}\n"
        )
        expected = original.replace("value + FACTOR", "value * FACTOR")
        prompt = (
            "Fix src/transform.py so scale(values) multiplies each value by "
            "FACTOR. Preserve every other file, run the tests, and finish with "
            "a short summary."
        )
        specification = (
            f"scale([1, 3]) == [{multiplier}, {3 * multiplier}]\n"
        )
        summary = "Fixed scale to multiply by FACTOR; tests pass."
        family_name = "arithmetic_operator"
    elif family == 1:
        limit = rng.randrange(3, 12)
        original = (
            "def indices(limit):\n"
            "    \"\"\"Return every integer from zero through limit.\"\"\"\n"
            "    return list(range(limit))\n"
        )
        expected = original.replace("range(limit)", "range(limit + 1)")
        prompt = (
            "Fix src/transform.py so indices(limit) includes the upper bound. "
            "Preserve every other file, run the tests, and finish with a short "
            "summary."
        )
        specification = f"indices({limit})[-1] == {limit}\n"
        summary = "Fixed the inclusive upper bound; tests pass."
        family_name = "boundary_condition"
    elif family == 2:
        original = (
            "def non_negative(values):\n"
            "    \"\"\"Keep positive values and zero in their original order.\"\"\"\n"
            "    return [value for value in values if value > 0]\n"
        )
        expected = original.replace("value > 0", "value >= 0")
        prompt = (
            "Fix src/transform.py so non_negative(values) retains zero as well "
            "as positive values. Preserve every other file, run the tests, and "
            "finish with a short summary."
        )
        specification = "non_negative([-2, 0, 3]) == [0, 3]\n"
        summary = "Fixed zero handling in non_negative; tests pass."
        family_name = "filter_predicate"
    else:
        original = (
            "def normalize(value):\n"
            "    \"\"\"Trim surrounding whitespace and lowercase the value.\"\"\"\n"
            "    return value.lower()\n"
        )
        expected = original.replace("value.lower()", "value.strip().lower()")
        prompt = (
            "Fix src/transform.py so normalize(value) trims surrounding "
            "whitespace before lowercasing. Preserve every other file, run the "
            "tests, and finish with a short summary."
        )
        specification = "normalize('  Ready ') == 'ready'\n"
        summary = "Fixed whitespace normalization; tests pass."
        family_name = "string_normalization"
    return EpisodeTask(
        id=hashlib.sha256(
            f"reliquary_workspace_tools_v1:{index}".encode()
        ).hexdigest()[:16],
        prompt=prompt,
        tools=TOOLS,
        metadata={
            "family": family_name,
            "generator_version": GENERATOR_VERSION,
            "generator_index": index,
            "difficulty": 1 + family,
        },
        private={
            "files": {
                "src/transform.py": original,
                "README.md": "# Deterministic transform utility\n",
                "tests/spec.txt": specification,
            },
            "expected_path": "src/transform.py",
            "expected_content": expected,
            "reference_actions": (
                AssistantAction.tool_call("read_file", path="src/transform.py"),
                AssistantAction.tool_call(
                    "write_file", path="src/transform.py", content=expected
                ),
                AssistantAction.tool_call("run_tests"),
                AssistantAction.tool_call(
                    "finish", response=summary
                ),
            ),
        },
    )


class WorkspaceToolsEnvironment:
    name = "reliquary_workspace_tools_v1"
    validator_authoritative_reward = True
    max_turns = 7

    def __len__(self) -> int:
        return TASK_COUNT

    def get_task(self, index: int) -> EpisodeTask:
        return _build_task(index)

    def get_problem(self, index: int) -> dict[str, Any]:
        from reliquary.environment.agentic.compat import episode_problem

        return episode_problem(self, index)

    @staticmethod
    def _resolve(root: Path, relative: str) -> Path:
        if not relative or len(relative) > 256:
            raise ValueError("path must contain 1..256 characters")
        candidate = (root / relative).resolve()
        if root.resolve() not in candidate.parents:
            raise ValueError("path escapes workspace")
        return candidate

    def reset(self, task: EpisodeTask, seed: int) -> ResetResult:
        del seed
        root = Path(tempfile.mkdtemp(prefix="reliquary-episode-"))
        files = dict(task.private["files"])
        for relative, content in files.items():
            path = self._resolve(root, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")
        return ResetResult(
            state=WorkspaceState(
                root=root,
                expected_path=str(task.private["expected_path"]),
                expected_content=str(task.private["expected_content"]),
                original_files=files,
            )
        )

    @staticmethod
    def _event(name: str, value: dict[str, Any]) -> tuple[EpisodeEvent, ...]:
        return (EpisodeEvent(role="tool", name=name, content=canonical_json(value)),)

    def step(
        self,
        task: EpisodeTask,
        state: WorkspaceState,
        action: AssistantAction,
    ) -> StepResult:
        del task
        if state.closed:
            raise RuntimeError("episode state is closed")
        if action.kind == "final":
            state.final_response = action.content or ""
            return StepResult(
                state,
                self._event("final", {"accepted": True}),
                done=True,
                termination_reason="finished",
            )
        name = str(action.tool)
        arguments = dict(action.arguments)
        try:
            if name == "list_files":
                if arguments:
                    raise ValueError("list_files takes no arguments")
                value = {
                    "files": sorted(
                        str(path.relative_to(state.root))
                        for path in state.root.rglob("*")
                        if path.is_file()
                    )
                }
            elif name == "read_file":
                if set(arguments) != {"path"}:
                    raise ValueError("read_file requires path")
                path = self._resolve(state.root, str(arguments["path"]))
                value = {"path": str(arguments["path"]), "content": path.read_text(encoding="utf-8")}
            elif name == "write_file":
                if set(arguments) != {"path", "content"}:
                    raise ValueError("write_file requires path and content")
                path = self._resolve(state.root, str(arguments["path"]))
                content = str(arguments["content"])
                if len(content.encode("utf-8")) > 64 * 1024:
                    raise ValueError("file content exceeds 64 KiB")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                value = {"written": True, "path": str(arguments["path"])}
            elif name == "run_tests":
                if arguments:
                    raise ValueError("run_tests takes no arguments")
                target = self._resolve(state.root, state.expected_path)
                state.tests_passed = (
                    target.exists()
                    and target.read_text(encoding="utf-8") == state.expected_content
                )
                value = {
                    "passed": state.tests_passed,
                    "summary": "1 passed" if state.tests_passed else "1 failed",
                }
            elif name == "finish":
                if set(arguments) != {"response"}:
                    raise ValueError("finish requires response")
                state.final_response = str(arguments["response"])
                return StepResult(
                    state,
                    self._event("finish", {"accepted": True}),
                    done=True,
                    termination_reason="finished",
                )
            else:
                raise ValueError(f"unknown tool: {name}")
            return StepResult(state, self._event(name, {"ok": True, **value}))
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            state.invalid_actions += 1
            return StepResult(
                state,
                self._event(name, {"ok": False, "error": str(exc)[:512]}),
            )

    def grade(
        self,
        task: EpisodeTask,
        state: WorkspaceState,
        trace: EpisodeTrace,
    ) -> RewardReport:
        target = self._resolve(state.root, state.expected_path)
        exact_patch = target.exists() and target.read_text(encoding="utf-8") == state.expected_content
        unrelated_unchanged = True
        for relative, original in state.original_files.items():
            if relative == state.expected_path:
                continue
            path = self._resolve(state.root, relative)
            if not path.exists() or path.read_text(encoding="utf-8") != original:
                unrelated_unchanged = False
                break
        current_files = {
            str(path.relative_to(state.root)): path.read_text(encoding="utf-8")
            for path in state.root.rglob("*")
            if path.is_file()
        }
        expected_names = set(state.original_files)
        no_extra_files = set(current_files) == expected_names
        checks = (
            RewardCheck("exact_patch", exact_patch, 3.0),
            RewardCheck("tests_run_and_passed", state.tests_passed, 1.5),
            RewardCheck("unrelated_files_unchanged", unrelated_unchanged, 2.0),
            RewardCheck("no_extra_files", no_extra_files, 1.0),
            RewardCheck("summary_mentions_tests", "test" in (state.final_response or "").lower(), 0.5),
            RewardCheck("no_invalid_tool_calls", state.invalid_actions == 0, 0.5),
            RewardCheck("finished_explicitly", trace.termination_reason == "finished", 0.5),
        )
        fatal = not unrelated_unchanged or not no_extra_files
        return RewardReport.from_checks(
            checks,
            state_digest=sha256_json(current_files),
            fatal=fatal,
            binary=True,
        )

    def close(self, state: WorkspaceState) -> None:
        if not state.closed:
            shutil.rmtree(state.root, ignore_errors=True)
            state.closed = True
