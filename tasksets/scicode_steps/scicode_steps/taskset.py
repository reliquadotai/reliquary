"""scicode-steps: SciCode (scicode-bench, arXiv 2407.13168) one sub-step at a time.

Upstream (scicode-bench/SciCode @ e3158ea0, eval/inspect_ai/scicode.py) asks
each sub-step in a fresh single-turn call whose prompt carries the earlier
steps' descriptions and the model's OWN code for them, then runs the step's
asserts against the HDF5 reference targets after that same code. A step's
prompt therefore depends on the model's earlier answers: this taskset holds
one position (``step``) of that chain, the earlier answers' code coming in as
``previous``. Running positions 1, 2, ... in turn, each from the answers of
the ones before, is the published protocol (``chain`` has the bookkeeping).

Kept as upstream: templates (no background = background_comment_template.txt),
code extraction, the test file, pass = exit 0, the three steps given as code
(13.6, 62.1, 76.3: not asked, not counted, their code put in as upstream puts
it). Dataset: SciCode1/SciCode (Apache-2.0); test 65 problems / 288 scored
steps, validation 15 / 50. The step timeout is 300 s (upstream 1800 s).
Upstream's own references fail 78.3 and 70.8 (SciCode issue #43), and the
given code of 13.6 and 62.1 lacks its class (issue #59): kept, as published.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

DATASET = "SciCode1/SciCode"
DATASET_REVISION = "4510f6a6aa27c43fad7b43da2c59602a86e88480"
SPLIT_FILES = {
    "validation": ("problems_dev.jsonl", "193968aff23b7ed931c8f6d196b611e2f6694adb49e6cc085444a60ad2fc4f7b"),
    "test": ("problems_test.jsonl", "38797fef78f434720be6d053b4f3a86839d6f8ea5fb9115450677cd3a6edf81d"),
}
SPLITS = {"validation": ("validation",), "test": ("test",), "all": ("validation", "test")}
# Scored steps per split (upstream's denominators).
SCORED_STEPS = {"validation": 50, "test": 288, "all": 338}
PROBLEMS = {"validation": 15, "test": 65, "all": 80}

DATA = Path(__file__).parent / "data"
VERIFY = (Path(__file__).parent / "verify.py").read_bytes()
# Upstream's names are crossed: the default (no background) prompt is the
# background-comment template.
TEMPLATES = {False: (DATA / "background_comment_template.txt").read_text(encoding="utf-8"),
             True: (DATA / "multistep_template.txt").read_text(encoding="utf-8")}
# Steps upstream gives as code instead of asking (eval/data/<step>.txt).
GIVEN_STEPS = frozenset({"13.6", "62.1", "76.3"})
# Where the grading image keeps the targets (sha256 48b0272a...b890, 1,049,345,865 bytes).
H5_PATH = "/opt/scicode/test_data.h5"


def extract_python_script(response: str) -> str:
    """Upstream ``scicode.gen.models.extract_python_script``: the first fenced
    block (```python first), import lines removed."""
    if '```' in response:
        python_script = (response.split("```python")[1].split("```")[0] if '```python' in response
                         else response.split('```')[1].split('```')[0])
    else:
        python_script = response
    return re.sub(r'^\s*(import .*|from .*\s+import\s+.*)', '', python_script, flags=re.MULTILINE)


def extract_function_name(function_header: str) -> str:
    match = re.search(r'\bdef\s+(\w+)\s*\(', function_header)
    if match:
        return match.group(1)
    match = re.search(r'\bclass\s+(\w+)\s*\(', function_header)
    if match:
        return match.group(1)
    raise ValueError('Function name or class name not found.')


def get_function_from_code(code_string: str, function_name: str) -> str | None:
    """Upstream ``scicode.parse.parse.get_function_from_code``."""
    try:
        for node in ast.walk(ast.parse(code_string)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                return ast.unparse(node)
    except Exception:
        return code_string
    return None


def given_step_code(step: dict) -> str:
    # As upstream: the named function only. For 13.6 and 62.1 the header's
    # first def is `__init__`, so only that method is kept (upstream quirk).
    text = (DATA / f"{step['step_number']}.txt").read_text(encoding="utf-8")
    return get_function_from_code(text, extract_function_name(step["function_header"]))


def read_problems(split: str, revision: str = DATASET_REVISION) -> list[dict]:
    from huggingface_hub import hf_hub_download

    problems = []
    for part in SPLITS[split]:
        name, sha256 = SPLIT_FILES[part]
        raw = Path(hf_hub_download(DATASET, name, repo_type="dataset", revision=revision)).read_bytes()
        if revision == DATASET_REVISION and hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError(f"{DATASET}/{name} differs from the pinned file")
        problems += [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return problems


def scored_steps(problem: dict) -> list[str]:
    return [s["step_number"] for s in problem["sub_steps"] if s["step_number"] not in GIVEN_STEPS]


def prompt_and_prefix(problem: dict, position: int, previous_code: list[str],
                      with_background: bool) -> tuple[str, str]:
    """Upstream ``generate_prompt_with_steps``: the prompt for the step at
    ``position`` (1-based) and the test file's head (dependencies + earlier code)."""
    steps = problem["sub_steps"]

    def described(step):
        return (step["step_description_prompt"] + '\n' + step["step_background"] if with_background
                else step["step_description_prompt"])

    lines = []
    for i in range(position - 1):
        lines += [described(steps[i]), previous_code[i], "------"]
    step = steps[position - 1]
    next_step = [described(step), f"{step['function_header']}\n\n{step['return_line']}"]
    dependencies = problem["required_dependencies"]
    prompt = TEMPLATES[with_background].format(
        problem_steps_str="\n\n".join(lines[:-1]), next_step_str="\n\n".join(next_step),
        dependencies=dependencies)
    return prompt, f'{dependencies}\n{chr(10).join(previous_code[:position - 1])}\n'


def test_script(prefix: str, code: str, step_number: str, test_cases: list[str],
                h5_path: str = H5_PATH) -> str:
    """Upstream's per-step test file: the step file (prefix, then the code), the
    targets, each test after its `target`."""
    out = [f"{prefix}\n{code}", "\n\nfrom scicode.parse.parse import process_hdf5_to_tuple\n\n",
           f"targets = process_hdf5_to_tuple('{step_number}', {len(test_cases)}, '{h5_path}')\n"]
    for i, test in enumerate(test_cases):
        out.append(f"target = targets[{i}]\n\n")
        out += [line + '\n' for line in test.split('\n')]
    return "".join(out)


class SciCodeStepData(vf.TaskData):
    network_allow: list[str] = []
    problem_id: str
    step_number: str
    test_prefix: str
    """Dependencies and the earlier steps' code, as the step file starts."""
    test_cases: list[str]


class SciCodeStepTaskConfig(vf.TaskConfig):
    timeout_per_step: int = 300


class SciCodeStepsConfig(vf.TasksetConfig):
    split: Literal["test", "validation", "all"] = "test"
    with_background: bool = False
    step: int = 1
    """The chain position (1-based) whose steps are asked."""
    previous: dict[str, str] = {}
    """Code extracted (`extract_python_script`) from the model's answer to each
    earlier asked step, by step number ("1.1"); "" for a step it never answered."""
    dataset_revision: str = DATASET_REVISION
    task: SciCodeStepTaskConfig = SciCodeStepTaskConfig()


class SciCodeStepTask(vf.Task[SciCodeStepData, vf.State, SciCodeStepTaskConfig]):
    @vf.reward(weight=1.0)
    async def passed(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        code = extract_python_script(trace.last_reply or "")
        script = test_script(self.data.test_prefix, code, self.data.step_number, self.data.test_cases)
        spec = f"/tmp/scicode/{trace.id}/spec.json"
        await runtime.write(spec, json.dumps({"script": script,
                                              "timeout": self.config.timeout_per_step}).encode())
        await runtime.prepare_execution([])
        result = await runtime.run_uv_script(VERIFY, args=[spec])
        if result.exit_code != 0:
            raise RuntimeError(f"verify.py failed: {result.stderr.strip()[-1000:]}")
        report = json.loads(result.stdout.strip().splitlines()[-1])
        return 1.0 if report["passed"] else 0.0


class SciCodeStepsTaskset(vf.Taskset[SciCodeStepTask, SciCodeStepsConfig]):
    # Copied onto the eval-set card by `reliquary eval build-set`.
    RELIQUARY_CARD = {
        "name": "SciCode",
        "license": "Apache-2.0",
        "grading": ("upstream step tests against the HDF5 targets (exit 0 passes), after the model's own code "
                    "for the earlier steps; one set per chain position, main-problem and sub-step rates "
                    "over the whole chain"),
    }

    def load(self) -> Iterator[SciCodeStepTask]:
        cfg = self.config
        if cfg.step < 1:
            raise ValueError("step is 1-based")
        index = 0
        for problem in read_problems(cfg.split, cfg.dataset_revision):
            steps = problem["sub_steps"]
            if len(steps) < cfg.step or steps[cfg.step - 1]["step_number"] in GIVEN_STEPS:
                continue
            previous = []
            for step in steps[:cfg.step - 1]:
                number = step["step_number"]
                if number in GIVEN_STEPS:
                    previous.append(given_step_code(step))
                elif number in cfg.previous:
                    previous.append(cfg.previous[number])
                else:
                    raise ValueError(f"step {steps[cfg.step - 1]['step_number']} needs the code of {number}")
            prompt, prefix = prompt_and_prefix(problem, cfg.step, previous, cfg.with_background)
            step = steps[cfg.step - 1]
            yield SciCodeStepTask(
                SciCodeStepData(idx=index, name=f"scicode-{step['step_number']}", prompt=prompt,
                                problem_id=problem["problem_id"], step_number=step["step_number"],
                                test_prefix=prefix, test_cases=list(step["test_cases"])),
                cfg.task,
            )
            index += 1
