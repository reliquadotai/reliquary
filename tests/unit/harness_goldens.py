"""Token-exact goldens of the agentic harnesses, frozen from the code they guard.

Every case is rendered through the public surface miners and validators share (the
Qwen turn renderer of `agentic_swe.load_turn_renderer` for the bash harness; the
ChatML renderer and runner for EnvScaler v2), and compared, id for id, with
`tests/fixtures/harness_goldens/<case>.<tokenizer>.json`. The fixtures were written by
this module from feat/rl-async a5df544f, before the harness moved to `reliquary_harness`:
they are the bytes the deployed miners and validators produce, and a refactor must keep
every one of them.

    python -m tests.unit.harness_goldens --write TOKENIZER_DIR [TOKENIZER_DIR ...]

writes them again (never do that to make a failing test pass: a changed golden is a
changed protocol)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "harness_goldens"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")

SWE_PROMPT = (
    "<uploaded_files>\n/testbed\n</uploaded_files>\nI've uploaded a python code repository in the "
    "directory /testbed. Consider the following issue description:\n\n<issue_description>\n"
    "`parse_date(\"2026-10-19T00:00:00Z\")` raises `ValueError: unconverted data remains: Z`.\n"
    "Expected: an aware datetime in UTC. Ünïcödé in the report: café — naïve 😀.\n"
    "</issue_description>\n\nCan you help me implement the necessary changes to the repository so "
    "that the requirements specified in the issue are met?"
)
TMAX_PROMPT = (
    "You are working in a Linux container at /app. Count the lines of every `*.log` file under "
    "/app/logs and write the total to /app/total.txt (a single integer, no newline)."
)

CALL_LS = ("Let me look at the repository layout first.\n</think>\n\n<tool_call>\n<function=bash>\n"
           "<parameter=command>\ncd /testbed && ls\n</parameter>\n</function>\n</tool_call><|im_end|>")
CALL_TWO = (
    "The parser is in `dates.py`; fix the format and run the tests.\n</think>\n\nI'll edit it.\n\n"
    "<tool_call>\n<function=edit>\n<parameter=path>\n/testbed/dates.py\n</parameter>\n"
    "<parameter=old_str>\n    return datetime.strptime(text, \"%Y-%m-%dT%H:%M:%S\")\n</parameter>\n"
    "<parameter=new_str>\n    if text.endswith(\"Z\"):\n        text = text[:-1] + \"+00:00\"\n"
    "    return datetime.fromisoformat(text)  # café ✓\n</parameter>\n</function>\n</tool_call>\n"
    "<tool_call>\n<function=bash>\n<parameter=command>\ncd /testbed && python -m pytest -q tests/test_dates.py\n"
    "</parameter>\n</function>\n</tool_call><|im_end|>")
CALL_UNKNOWN = ("Search the web for it.\n</think>\n\n<tool_call>\n<function=search>\n<parameter=query>\n"
                "strptime Z suffix\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=bash>\n"
                "<parameter=command>\ngit diff\n</parameter>\n</function>\n</tool_call><|im_end|>")
CALL_EMPTY_THINK = ("</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\necho done && "
                    "git status --short\n</parameter>\n</function>\n</tool_call><|im_end|>")
DONE = "The tests pass.\n</think>\n\nFixed: `parse_date` now accepts a trailing `Z`.<|im_end|>"
CAPPED = "I still need to check the other call sites, starting with `api.py`, which also"

TMAX_CALL = ("Count them with wc.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\n"
             "cat /app/logs/*.log | wc -l | tr -d '\\n' > /app/total.txt && cat /app/total.txt\n"
             "</parameter>\n</function>\n</tool_call><|im_end|>")
TMAX_EDIT = ("Edit is not offered here.\n</think>\n\n<tool_call>\n<function=edit>\n<parameter=path>\n"
             "/app/total.txt\n</parameter>\n<parameter=old_str>\n1\n</parameter>\n<parameter=new_str>\n2\n"
             "</parameter>\n</function>\n</tool_call><|im_end|>")
TMAX_DONE = "Written.\n</think>\n\nThe total is in /app/total.txt.<|im_end|>"

# What the sandbox renders for a call record (`reliquary_sandbox.observation`) and what it
# answers for a refused one: observation text as the model reads it.
OBS_LS = "README.md\ndates.py\nsetup.py\ntests\n"
OBS_EDITED = "Edited /testbed/dates.py"
OBS_PYTEST = ("..F\n=================================== FAILURES ===================================\n"
              "    assert parse_date(\"2026-10-19T00:00:00Z\").tzinfo is not None\nE   AssertionError\n"
              "1 failed, 2 passed in 0.03s\n[output truncated]")
OBS_NO_OUTPUT = "(no output)"
OBS_PADDED = "\n\n  M dates.py\n\t\n"
OBS_NOT_OFFERED = "error: that tool is not available in this episode; use one of the tools listed"
OBS_TIMED_OUT = ("142\n[command timed out; its process group was killed. Run long jobs in the background "
                 "with their output redirected to a file, or narrow the command.]")

NETWORK_NOTICE_PROMPT = "Fix the failing test in tests/test_dates.py."

# (case, tools, prompt source, turns): a turn is (completion text, observations or None).
# The last turn has no observations; `stop` is what the harness reports.
QWEN_CASES = {
    "swe-signed-bash-edit": dict(tools=("bash", "edit"), prompt=SWE_PROMPT, notice=False, stop="agent_completed",
                                 turns=[(CALL_LS, [OBS_LS]), (CALL_TWO, [OBS_EDITED, OBS_PYTEST]),
                                        (CALL_UNKNOWN, [OBS_NOT_OFFERED, OBS_PADDED]), (CALL_EMPTY_THINK, [OBS_NO_OUTPUT]),
                                        (DONE, None)]),
    "swe-signed-context-length": dict(tools=("bash", "edit"), prompt=SWE_PROMPT, notice=False,
                                      stop="context_length", turns=[(CALL_LS, [OBS_LS]), (CAPPED, None)]),
    "swe-corpus-network-notice": dict(tools=("bash", "edit"), prompt=NETWORK_NOTICE_PROMPT, notice=True,
                                      stop="agent_completed",
                                      turns=[(CALL_TWO, [OBS_EDITED, OBS_TIMED_OUT]), (DONE, None)]),
    "tmax-bash": dict(tools=("bash",), prompt=TMAX_PROMPT, notice=False, stop="agent_completed",
                      turns=[(TMAX_CALL, ["142"]), (TMAX_EDIT, [OBS_NOT_OFFERED]), (TMAX_DONE, None)],
                      parse_only=[CALL_TWO]),
}


def tokenizer_digest(directory: str | Path) -> str:
    """sha256 over the tokenizer files the renderer reads, in a fixed order (first 16 hex)."""
    digest = hashlib.sha256()
    for name in TOKENIZER_FILES:
        data = (Path(directory) / name).read_bytes()
        digest.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()[:16]


def _prompt(case: dict) -> str:
    if case["notice"]:
        from reliquary.environment.agentic_swe import with_network_notice

        return with_network_notice(case["prompt"])
    return case["prompt"]


def qwen_case(tokenizer_dir: str, name: str) -> dict:
    """One bash-harness conversation as the validator renders it: the initial prompt, then
    per turn the completion's ids, what the pinned parser reads from it, the canonical and
    reasoning checks, the observations read back from the rendered segment and the next
    prompt; finally the parsed trajectory and the export rendering."""
    from reliquary.corpus.trajectory_parse import TrajectoryRefused, parse_trajectory
    from reliquary.environment.agentic_swe import load_turn_renderer

    case = QWEN_CASES[name]
    r = load_turn_renderer(tokenizer_dir, tools=case["tools"])
    encode = lambda text: [int(t) for t in r._tokenizer.encode(text, add_special_tokens=False)]
    prompt_text = _prompt(case)
    initial = r.initial_ids(prompt_text)
    prompt, tokens, spans, out_turns, messages = list(initial), [], [], [], []
    for text, observations in case["turns"]:
        completion = encode(text)
        start = len(prompt) - len(initial)
        turn = {"text": text, "completion_ids": completion,
                "tool_calls": [list(call) for call in r.tool_calls(completion)],
                "assistant_message": r.assistant_message(completion),
                "canonical": r.span_is_canonical(prompt, completion),
                "reasoning_unclosed": r.reasoning_unclosed(prompt, completion)}
        tokens += prompt[len(initial) + len(tokens):] + completion
        spans.append((start, start + len(completion)))
        messages.append(turn["assistant_message"])
        if observations is not None:
            following = r.next_prompt(prompt, completion, observations)
            turn["observations"] = list(observations)
            turn["next_prompt_ids"] = following
            turn["observations_read_back"] = r.observations(following[len(prompt) + len(completion):])
            messages += [{"role": "tool", "tool_call_id": f"call_{i}", "content": text}
                         for i, text in enumerate(observations)]
            prompt = following
        out_turns.append(turn)
    try:
        parsed = parse_trajectory(r, prompt_ids=initial, tokens=tokens, spans=spans, stop=case["stop"],
                                  max_turns=len(spans))
        verdict = {"actions": [[a.tool, a.arguments, a.observation] for a in parsed.actions]}
    except TrajectoryRefused as refused:
        verdict = {"refused": refused.reason, "detail": refused.detail}
    system = r._system_prompt
    conversation = [{"role": "system", "content": system}, {"role": "user", "content": prompt_text}, *messages]
    return {
        "case": name, "tokenizer": tokenizer_digest(tokenizer_dir), "tools": list(case["tools"]),
        "system_prompt": system, "tool_specs": r._tools, "prompt": prompt_text,
        "initial_ids": initial, "stop_ids": sorted(r.stop_ids), "terminator_id": r.terminator_id,
        "turn_markup_ids": sorted(r.turn_markup_ids), "turns": out_turns,
        "trajectory": {"tokens": tokens, "spans": [list(s) for s in spans], "stop": case["stop"],
                       **verdict},
        "export_ids": r.render_messages(conversation),
        # Spans read alone after the initial prompt (a tool the harness does not offer).
        "parse_only": [{"text": text, "tool_calls": [list(call) for call in r.tool_calls(encode(text))],
                        "assistant_message": r.assistant_message(encode(text))}
                       for text in case.get("parse_only", ())],
    }


def real_turns_case(tokenizer_dir: str) -> dict:
    """Assistant turns real models wrote in smoke3 (SWE and TMax, 10-18), read by the pinned
    parser after the bash+edit initial prompt: calls, message, canonical and reasoning
    checks. These are the spans that once broke the renderer (a prompt-opened <think>)."""
    from reliquary.environment.agentic_swe import load_turn_renderer

    r = load_turn_renderer(tokenizer_dir, tools=("bash", "edit"))
    prompt = r.initial_ids(SWE_PROMPT)
    texts = json.loads((FIXTURES / "real_turns.json").read_text(encoding="utf-8"))
    rows = []
    for text in texts:
        completion = [int(t) for t in r._tokenizer.encode(text, add_special_tokens=False)]
        rows.append({"completion_ids": completion,
                     "tool_calls": [list(call) for call in r.tool_calls(completion)],
                     "assistant_message": r.assistant_message(completion),
                     "canonical": r.span_is_canonical(prompt, completion),
                     "reasoning_unclosed": r.reasoning_unclosed(prompt, completion)})
    return {"case": "real-turns", "tokenizer": tokenizer_digest(tokenizer_dir), "prompt_ids": prompt,
            "turns": rows}


def envscaler_case(tokenizer_dir: str) -> dict:
    """An EnvScaler v2 episode (fixture world SHOP) through the ChatML renderer and the
    runner, encoded with the real tokenizer, then replayed as the validator does."""
    from transformers import AutoTokenizer

    from reliquary.environment.agentic.chatml import CHATML_RENDERER_ID, ChatMLEpisodeRenderer
    from reliquary.environment.agentic.envs.envscaler_tools_v2.environment import EnvScalerToolsV2Environment
    from reliquary.environment.agentic.renderers import renderer_for
    from reliquary.environment.agentic.runner import EpisodeRunner
    from reliquary.environment.agentic.types import AssistantAction, GeneratedAction
    from tests.unit import envscaler_v2_fixtures as fixtures

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    encode = lambda text: [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
    turns = [
        ChatMLEpisodeRenderer.action_text(AssistantAction(kind="tool", tool="add_item",
                                                          arguments={"name": "apple", "qty": 3}),
                                          reasoning="Check the shop, then act."),
        ChatMLEpisodeRenderer.action_text(AssistantAction(kind="tool", tool="teleport", arguments={}),
                                          reasoning="Try the unknown tool.", preamble="One more call.") + "<|im_end|>",
        ChatMLEpisodeRenderer.action_text(AssistantAction.final("The apple is in stock (3)."),
                                          reasoning="Done."),
    ]

    class Policy:
        def __init__(self) -> None:
            self._turns = iter(turns)

        def generate(self, **_: object) -> GeneratedAction:
            return GeneratedAction(text=next(self._turns))

    environment = EnvScalerToolsV2Environment(fixtures.corpus())
    task = environment.get_task(fixtures.SHOP)
    renderer = renderer_for(CHATML_RENDERER_ID, encode)
    trace = EpisodeRunner(renderer=renderer).run(environment, task, seed=11, policy=Policy())
    return {"case": "envscaler-v2-chatml", "tokenizer": tokenizer_digest(tokenizer_dir),
            "renderer_id": CHATML_RENDERER_ID, "turn_texts": turns,
            "initial_text": ChatMLEpisodeRenderer.initial_text(task),
            "initial_ids": renderer.encode_initial(task),
            "tokens": [int(t) for t in trace.tokens], "termination": trace.termination_reason,
            "reward": trace.reward.reward,
            "actions": [[a.kind, a.tool, dict(a.arguments), a.content] for a in trace.actions],
            "events": [[e.role, e.content, e.name] for e in trace.events]}


def build(tokenizer_dir: str) -> dict[str, dict]:
    goldens = {name: qwen_case(tokenizer_dir, name) for name in QWEN_CASES}
    goldens["real-turns"] = real_turns_case(tokenizer_dir)
    goldens["envscaler-v2-chatml"] = envscaler_case(tokenizer_dir)
    return goldens


def fixture_path(case: str, tokenizer: str) -> Path:
    return FIXTURES / f"{case}.{tokenizer}.json"


def dump(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "--write" or len(argv) < 2:
        print(__doc__)
        return 2
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for directory in argv[1:]:
        for name, golden in build(directory).items():
            path = fixture_path(name, golden["tokenizer"])
            path.write_text(dump(golden), encoding="utf-8")
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
