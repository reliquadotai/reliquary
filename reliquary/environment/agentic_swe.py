"""What an agentic SWE corpus job pins, and the prompt source behind it.

`renderers`, `verifiers` and `reliquary_swe` are optional: imported inside
functions only. The constants below are copies of verifiers b2e4e81's bash
harness (edit on, search off); `tests/unit/test_agentic_swe_renderer.py`
holds them equal to the pinned package when it is installed.
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import json
import re
import subprocess
import threading
from collections.abc import Sequence

from reliquary.environment.agentic.types import EpisodeTask

SUPPORTED_RENDERER = "renderers:qwen38@0.1.11"
RENDERERS_VERSION = "0.1.11"
SUPPORTED_VERIFIERS = "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"

# harnesses/bash/harness.py: BASH_SYSTEM_PROMPT and EDIT_SYSTEM_PROMPT; the harness
# joins the enabled fragments with one space.
BASH_SENTENCE = "You are a coding agent. You have access to a bash tool for running shell commands."
EDIT_SENTENCE = "You also have an edit tool for single-occurrence string replacement in a file."
BASH_SYSTEM_PROMPT = BASH_SENTENCE + " " + EDIT_SENTENCE
# harnesses/bash/program.py: BASH_TOOL, EDIT_TOOL, as the train client renders them.
BASH_HARNESS_TOOLS: tuple[dict, ...] = (
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a bash command and return its combined stdout and stderr.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string",
                                                  "description": "The bash command to run."}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "edit",
        "description": "Replace a unique string in a file. old_str must appear exactly once in the file.",
        "parameters": {"type": "object",
                       "properties": {
                           "path": {"type": "string",
                                    "description": "File path (relative to cwd or absolute)."},
                           "old_str": {"type": "string",
                                       "description": "Exact string to find (must appear exactly once)."},
                           "new_str": {"type": "string", "description": "Replacement string."}},
                       "required": ["path", "old_str", "new_str"]}}},
)
_TOOL_SETS = (("bash",), ("bash", "edit"))


def harness_tools(tools) -> tuple[dict, ...]:
    """The tool specs verifiers' bash harness sends for `tools` (bash always, edit
    optionally): the renderer must parse with exactly these, since parameter
    coercion follows their schemas. Order and repeats do not matter; an unknown
    name, or edit without bash, is refused."""
    names = tuple(sorted(set(tools)))
    if names not in _TOOL_SETS:
        raise ValueError(f"the bash harness offers bash or bash and edit, not {list(names)}")
    return BASH_HARNESS_TOOLS[:len(names)]


def harness_system_prompt(tools) -> str:
    harness_tools(tools)
    return " ".join([BASH_SENTENCE] + ([EDIT_SENTENCE] if "edit" in tools else []))


def pinned_tool_calls(parsed_calls, unknown_status) -> list[tuple[str, str]]:
    """The one parser miners (through verifiers' train client) and validators share
    (spec §5.C amendment 1): verifiers b2e4e81 `response_from_generate`'s filter,
    exactly. A call needs a non-empty name and a status other than UNKNOWN_TOOL (so a
    block with no name, unclosed or malformed, is no call on either side); arguments
    are the raw string, or the parsed dict through `json.dumps` with its defaults."""
    return [(call.name, call.arguments if isinstance(call.arguments, str)
             else json.dumps(call.arguments or {}))
            for call in parsed_calls
            if getattr(call, "name", None) and getattr(call, "status", None) != unknown_status]


# dialects/base.py CAPABILITY_NOTICE: verifiers' interception appends it to the
# first user message of every request whose runtime restricts egress, which the
# episode config always does (`block: ["*"]`). The validator's render ALWAYS
# uses this pinned copy (ruling P13): a parity test holds it equal to the
# pinned verifiers, and the miner refuses to mine when its verifiers differs
# (`agentic_episode.network_notice_refusal`).
PINNED_NETWORK_NOTICE = (
    "Network protocol blocked fetching a resource. Continue without those capabilities; "
    "use local tools or inline data already present in the conversation, and do not retry "
    "the blocked provider-side operation."
)


def with_network_notice(content: str) -> str:
    """`append_user_notice` on a string user message."""
    notice = PINNED_NETWORK_NOTICE
    return f"{content}\n\n{notice}" if content else notice


def _dist_commit(name: str) -> str | None:
    """The commit a distribution was installed from: its VCS pin, or the HEAD
    of the checkout an editable install points at. None when unknowable."""
    try:
        raw = importlib.metadata.distribution(name).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    info = json.loads(raw)
    commit = (info.get("vcs_info") or {}).get("commit_id")
    if commit:
        return commit
    url = info.get("url") or ""
    if url.startswith("file://"):
        checkout = url[len("file://"):]
        git = ["git", "-c", "safe.directory=*", "-C", checkout]
        try:
            head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True,
                                  timeout=10)
            dirty = subprocess.run(git + ["status", "--porcelain"], capture_output=True, text=True,
                                   timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            return None
        # An editable checkout with local changes is not the pinned code,
        # whatever its HEAD says: report nothing rather than the commit.
        if head.returncode == 0 and dirty.returncode == 0 and not dirty.stdout.strip():
            return head.stdout.strip()
    return None


def installed_env_commit() -> str | None:
    return _dist_commit("reliquary-swe")


def installed_verifiers_commit() -> str | None:
    return _dist_commit("verifiers")


def _renderers_version() -> str | None:
    try:
        return importlib.metadata.version("renderers")
    except importlib.metadata.PackageNotFoundError:
        return None


def episode_support_refusal(episode, *, need_verifiers: bool) -> str | None:
    """Why this binary cannot serve `episode`, or None. ``need_verifiers`` for
    the processes that run episodes (miner, grade executor)."""
    if episode.renderer != SUPPORTED_RENDERER:
        return f"episode.renderer {episode.renderer!r} is not {SUPPORTED_RENDERER!r}"
    if episode.verifiers != SUPPORTED_VERIFIERS:
        return f"episode.verifiers {episode.verifiers} is not the supported {SUPPORTED_VERIFIERS}"
    if _renderers_version() != RENDERERS_VERSION:
        return f"renderers {RENDERERS_VERSION} is not installed (found {_renderers_version()})"
    installed = installed_env_commit()
    if installed != episode.env.version:
        return (f"reliquary-swe is installed at {installed}, the job pins "
                f"{episode.env.version}")
    if need_verifiers and installed_verifiers_commit() != SUPPORTED_VERIFIERS:
        return (f"verifiers is installed at {installed_verifiers_commit()}, "
                f"the job pins {SUPPORTED_VERIFIERS}")
    return None


class SweSource:
    """SWE-smith tasks by source index: the instance id and the user prompt,
    nothing else (a full task set costs over 5 GB of memory as tasks)."""

    __slots__ = ("_rows",)

    def __init__(self, rows: Sequence[tuple[str, str]]) -> None:
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def instance_id(self, index: int) -> str:
        return self._rows[index][0]

    def prompt(self, index: int) -> str:
        """The first user message as the model sees it: the task prompt plus
        verifiers' restricted-network notice (ruling P13)."""
        return with_network_notice(self._rows[index][1])

    def task_for(self, index: int) -> EpisodeTask:
        instance_id, prompt = self._rows[index]
        return EpisodeTask(id=instance_id, prompt=prompt, tools=())


@functools.lru_cache(maxsize=4)
def load_swe_source(num_images: int) -> SweSource:
    """The task set `reliquary-swe` builds for `num_images` (split train), in
    its own deterministic order: index i here is task i on every machine."""
    from reliquary_swe import corpus
    from reliquary_swe.taskset import PROMPT

    rows = corpus.load_swesmith_rows(num_images)
    return SweSource([(row.instance_id,
                       PROMPT.format(workdir=row.workdir, problem_statement=row.problem_statement))
                      for row in rows])


class SignedSweSource:
    """The prompts of a signed-sandbox job: the bridged task's prompt
    (`bridged_swe.prompt_of(split, index)`) exactly, with no restricted-network notice
    (the sandbox bridge runs the harness in a subprocess runtime, which adds none)."""

    def __init__(self, split: str, *, prompt_of=None, row_of=None) -> None:
        self.split = split
        self._prompt_of = prompt_of
        self._row_of = row_of
        self._prompts = functools.lru_cache(maxsize=4096)(self._prompt)
        self._instance_ids = functools.lru_cache(maxsize=4096)(self._instance_id)

    def _prompt(self, index: int) -> str:
        prompt_of = self._prompt_of
        if prompt_of is None:
            from reliquary.environment.bridged_swe import prompt_of
        return prompt_of(self.split, index)

    def prompt(self, index: int) -> str:
        return self._prompts(int(index))

    def instance_id(self, index: int) -> str:
        return self._instance_ids(int(index))

    def _instance_id(self, index: int) -> str:
        row_of = self._row_of
        if row_of is None:
            from reliquary.environment.bridged_swe import row_of
        return row_of(self.split, int(index))[1].instance_id

    def task_for(self, index: int) -> EpisodeTask:
        return EpisodeTask(id=self.instance_id(index), prompt=self.prompt(index), tools=())


def installed_env_package(package: str) -> str:
    """The `env_package` a gateway serving this install writes into record 0:
    `<distribution>==<version>+g<16 hex>.vfb<bridge version>`, the bridged identity
    (`bridged_env_package_of`), computed by reliquary-sandbox's own task registry over
    the taskset's installed files (one source of truth)."""
    from reliquary.environment.bridged_swe import installed_env_package as bridged

    return bridged(package)


def sandbox_support_refusal(episode, *, need_bridge: bool = False) -> str | None:
    """Why this process cannot serve a signed-sandbox `episode`, or None:
    SWE-smith at the 20 images the bridge serves, reliquary-sandbox at the job's commit,
    the env package at the exact version and code digest record 0 will carry, and the
    verifiers bridge (always: every process reads the task through it; `need_bridge` is
    kept for callers), which must load the installed package (an editable install is
    refused here, as the gateway refuses it)."""
    from reliquary import sandbox as sandbox_support

    if getattr(episode, "sandbox", None) is None:
        return "the episode is not a signed_sandbox one"
    try:
        sandbox_support.require_sandbox()
    except sandbox_support.SandboxUnavailable as exc:
        return str(exc)
    from reliquary.corpus.job import BRIDGED_SWESMITH_IMAGES

    if getattr(episode.env, "num_images", BRIDGED_SWESMITH_IMAGES) != BRIDGED_SWESMITH_IMAGES:
        return (f"a signed reliquary-swe job serves SWE-smith at its default "
                f"{BRIDGED_SWESMITH_IMAGES} images (the split the bridge serves)")
    refusal = sandbox_support.sandbox_commit_refusal(episode.sandbox.sandbox_commit)
    if refusal:
        return refusal
    package = episode.env.package
    pinned = episode.sandbox.env_package
    try:
        installed = f"{package}=={importlib.metadata.version(package)}"
    except importlib.metadata.PackageNotFoundError:
        return f"{package} is not installed"
    if installed != pinned.rsplit("+g", 1)[0]:
        return f"{installed} is installed, the job pins {pinned}"
    try:
        importlib.import_module(package.replace('-', '_'))
    except Exception as exc:  # an import-time pin check raises RuntimeError, not ImportError
        return (f"{package.replace('-', '_')} cannot be imported ({type(exc).__name__}: {exc}): "
                f"install {package} at the job's commit")
    try:
        identity = installed_env_package(package)
    except Exception as exc:  # an unreadable install cannot be the pinned code
        return f"cannot compute the installed {package} identity ({exc})"
    if identity != pinned:
        return f"{identity} is installed, the job pins {pinned}"
    try:
        importlib.import_module("reliquary_sandbox_verifiers")
    except Exception as exc:  # the bridge refuses an unpinned verifiers with RuntimeError
        return f"the verifiers bridge cannot load: {exc}"
    try:
        from reliquary.environment import bridged_swe

        bridged_swe.bridged_env()
    except Exception as exc:  # e.g. an editable install: the bridge serves packaged tasksets only
        return f"the bridge cannot serve the installed {package} ({type(exc).__name__}: {exc})"
    return None


def _locked(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class QwenTurnRenderer:
    """`trajectory_parse.TurnRenderer` over the pinned `renderers` Qwen3.8
    renderer: the one verifiers' train client renders every turn with."""

    def __init__(self, renderer, tools=("bash", "edit")) -> None:
        from renderers.base import ToolCallParseStatus

        self._r = renderer
        # P6: HF tokenizers are not thread-safe; intake, grading and export share this.
        self._lock = threading.RLock()
        self._tokenizer = renderer._tokenizer
        self._unknown = ToolCallParseStatus.UNKNOWN_TOOL
        stops = list(renderer.get_stop_token_ids())
        self.terminator_id = int(stops[0])
        self.stop_ids = frozenset(int(t) for t in stops)
        self._open = renderer._token_id("<tool_response>")
        self._close = renderer._token_id("</tool_response>")
        self.turn_markup_ids = frozenset(
            int(renderer._token_id(t)) for t in ("<|im_start|>", "<tool_response>", "</tool_response>"))
        self._tools = [dict(tool) for tool in harness_tools(tools)]
        self._system_prompt = harness_system_prompt(tools)
        self._im_start = int(renderer._token_id("<|im_start|>"))
        self._think = int(renderer._token_id("<think>"))
        self._think_end = int(renderer._token_id("</think>"))
        literals = set(getattr(self._tokenizer, "all_special_tokens", []) or [])
        for added in (getattr(self._tokenizer, "added_tokens_decoder", {}) or {}).values():
            literals.add(str(getattr(added, "content", added)))
        self._literals = tuple(sorted(t for t in literals if t))

    @_locked
    def initial_ids(self, prompt: str) -> list[int]:
        rendered = self._r.render(
            [{"role": "system", "content": self._system_prompt}, {"role": "user", "content": prompt}],
            tools=self._tools, add_generation_prompt=True)
        return [int(t) for t in rendered.token_ids]

    @_locked
    def tool_calls(self, completion_ids) -> list[tuple[str, str]]:
        return pinned_tool_calls(self._parse(completion_ids).tool_calls, self._unknown)

    def _parse(self, completion_ids):
        """The pinned parser, made total: whatever it raises on a span (a parameter
        nested 10k deep raises RecursionError from its JSON coercion) refuses the
        trajectory as `bad_turns` instead of escaping the intake."""
        from reliquary.corpus.checks import REASON_BAD_TURNS
        from reliquary.corpus.trajectory_parse import TrajectoryRefused

        try:
            return self._r.parse_response(list(completion_ids), tools=self._tools)
        except TrajectoryRefused:
            raise
        except Exception as exc:  # RecursionError is an Exception
            raise TrajectoryRefused(REASON_BAD_TURNS, {
                "why": "the pinned parser failed on a span", "error": type(exc).__name__}) from None

    @_locked
    def observations(self, segment_ids) -> list[str]:
        out, start = [], None
        for k, token in enumerate(segment_ids):
            if token == self._open:
                start = k + 1
            elif token == self._close and start is not None:
                text = self._tokenizer.decode(list(segment_ids[start:k]), skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)
                # The renderer writes "\n" + content.strip() + "\n".
                text = text[1:] if text.startswith("\n") else text
                text = text[:-1] if text.endswith("\n") else text
                out.append(text)
                start = None
        return out

    @_locked
    def next_prompt(self, prompt_ids, completion_ids, observations) -> list[int] | None:
        messages = [{"role": "tool", "tool_call_id": f"call_{i}", "content": text}
                    for i, text in enumerate(observations)]
        rendered = self._r.bridge_to_next_turn(list(prompt_ids), list(completion_ids), messages,
                                               tools=self._tools)
        return None if rendered is None else [int(t) for t in rendered.token_ids]

    def _tail(self, ids) -> list[int]:
        """The ids after the last <|im_start|>: role line, and any opened <think>."""
        for k in range(len(ids) - 1, -1, -1):
            if ids[k] == self._im_start:
                return list(ids[k + 1:])
        return list(ids)

    @_locked
    def reasoning_unclosed(self, prompt_ids, completion_ids) -> bool:
        ids = list(completion_ids)
        if self._think_end in ids:
            return False
        return self._think in ids or self._think in self._tail(prompt_ids)

    @_locked
    def span_is_canonical(self, prompt_ids, completion_ids) -> bool:
        """Round trip: parse the span to a message, render that message as an
        assistant turn, and require the span's own tokens, so no text can pose
        as turn structure once the message is re-encoded for export."""
        completion = [int(t) for t in completion_ids]
        message = self.assistant_message(completion)
        texts = [message.get("content") or "", message.get("reasoning_content") or ""]
        for call in message.get("tool_calls") or []:
            texts += [call["function"]["name"], call["function"]["arguments"]]
        if any(literal in text for text in texts for literal in self._literals):
            return False
        rendered = self._r.render([{"role": "user", "content": "x"}, message], tools=self._tools,
                                  add_generation_prompt=False).token_ids
        tail = self._tail([int(t) for t in rendered])
        closes = [k for k, t in enumerate(tail) if t == self.terminator_id]
        if not closes:
            return False
        tail = tail[:closes[-1] + 1]                    # drop the template's trailing newline
        if not completion or completion[-1] not in self.stop_ids:
            tail = tail[:-1]                           # capped turn: the bridge adds its own close
        expected = self._tail(prompt_ids) + completion
        if tail == expected:
            return True
        # The pinned parser strips the whitespace around a parameter value (an
        # indented `old_str` loses its indentation: 4 % of honest recorded
        # turns), so an exact comparison would refuse honest edits. Fall back
        # to the same comparison with whitespace erased: what the parser
        # dropped or changed beyond whitespace still fails it, and spelled
        # markup was already refused by the literal check above.
        squeeze = lambda ids: re.sub(r"\s+", "", self._tokenizer.decode(ids, skip_special_tokens=False))
        return squeeze(tail) == squeeze(expected)

    @_locked
    def render_messages(self, messages) -> list[int]:
        """A whole conversation (system, user, assistant and tool messages), as
        the pinned template renders it with the harness tools."""
        rendered = self._r.render(list(messages), tools=self._tools, add_generation_prompt=False)
        return [int(t) for t in rendered.token_ids]

    @_locked
    def whitespace_free(self, ids) -> str:
        """The decoded text with every whitespace run erased: what
        ``span_is_canonical`` compares when the parser strips parameter values."""
        return re.sub(r"\s+", "", self._tokenizer.decode(list(ids), skip_special_tokens=False))

    @_locked
    def assistant_message(self, completion_ids) -> dict:
        parsed = self._parse(completion_ids)
        message = {"role": "assistant", "content": parsed.content or ""}
        if parsed.reasoning_content:
            message["reasoning_content"] = parsed.reasoning_content
        calls = self.tool_calls(completion_ids)
        if calls:
            message["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                                      "function": {"name": name, "arguments": arguments}}
                                     for i, (name, arguments) in enumerate(calls)]
        return message


def load_turn_renderer(checkpoint_dir: str, tools=("bash", "edit")) -> QwenTurnRenderer:
    """The renderer the train client builds for this checkpoint
    (`create_renderer(load_tokenizer(model), Qwen38RendererConfig())`), over the
    job's tools."""
    from renderers import create_renderer
    from renderers.base import load_tokenizer
    from renderers.configs import Qwen38RendererConfig

    return QwenTurnRenderer(create_renderer(load_tokenizer(checkpoint_dir), Qwen38RendererConfig()),
                            tools=tools)
