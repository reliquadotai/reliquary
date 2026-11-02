"""A customer's batch file as frozen prompt sets.

The file is OpenAI Batch JSONL: one request per line, ``{custom_id, method, url, body}``,
``url`` being ``/v1/chat/completions`` (``messages``, optional ``tools``) or
``/v1/completions`` (``prompt``: text or token ids). Each usable line is rendered here,
once, with the model revision's own tokenizer and chat template, and written as token
ids: miners never render an upload's prompt.

The output is the eval-set format (``eval.prompt_source``): ``prompts.jsonl``, a
``set.json`` card and, for an eval upload, ``grading.jsonl``. A job names a set as
``eval-set:<set_id>:<n>:<sha256>`` exactly as it names one of ours, so every path that
serves, audits and grades a set serves an upload. Lines are grouped by their sampling:
one set per distinct sampling, since a job samples one way.

A bad line is an error line, never a failed file; the file is refused only past the
limits or when no line is usable.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reliquary.corpus.encoding import MAX_PROMPT_TOKEN_IDS

UPLOAD_RESULT_SCHEMA = "reliquary/order-upload/v1"
PURPOSES = ("batch", "eval")
CHAT_URL = "/v1/chat/completions"
COMPLETIONS_URL = "/v1/completions"
# OpenAI's own batch limits.
MAX_LINES = 50_000
MAX_BYTES = 200 * 1024 * 1024
MAX_CUSTOM_ID_CHARS = 64
MAX_SAMPLES = 64
MAX_NEW_TOKENS = 131072
MAX_TEMPERATURE = 4.0
SOURCE_KIND = "upload"
# A custom id kept as the row's problem id; any other is named by its line.
_SAFE_PROBLEM_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:-]{0,63}\Z")
_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
# Body keys a line may carry with no effect on the sample (OpenAI bookkeeping).
_IGNORED_KEYS = frozenset({"user", "metadata", "store", "parallel_tool_calls", "service_tier"})
_SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "min_p", "max_tokens",
                            "max_completion_tokens", "seed", "n"})
# Accepted only at their neutral value: anything else changes sampling in a way the
# forced draw does not reproduce.
_NEUTRAL = {"stream": (None, False), "logprobs": (None, False), "top_logprobs": (None, 0),
            "frequency_penalty": (None, 0, 0.0), "presence_penalty": (None, 0, 0.0),
            "repetition_penalty": (None, 1, 1.0), "logit_bias": (None, {}), "stop": (None, [], ""),
            "response_format": (None, {"type": "text"}), "tool_choice": (None, "auto"),
            "echo": (None, False), "suffix": (None, ""), "best_of": (None, 1)}
_NEUTRAL_CODES = {"stream": "stream_unavailable", "logprobs": "logprobs_unavailable",
                  "top_logprobs": "logprobs_unavailable", "logit_bias": "logit_bias_unavailable",
                  "response_format": "structured_output_unavailable",
                  "frequency_penalty": "penalties_unavailable",
                  "presence_penalty": "penalties_unavailable",
                  "repetition_penalty": "penalties_unavailable", "stop": "stop_unavailable"}
# Chat template variables a line may not set: the renderer's own.
_TEMPLATE_RESERVED = frozenset({
    "messages", "tools", "documents", "add_generation_prompt", "continue_final_message",
    "bos_token", "eos_token", "pad_token", "unk_token", "raise_exception", "strftime_now",
    "range", "namespace", "loop", "true", "false", "none", "True", "False", "None"})


class UploadRefused(ValueError):
    """The whole file is refused (``code``: limits, encoding, no usable line)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class LineError:
    line: int
    custom_id: str | None
    code: str
    message: str

    def to_dict(self) -> dict:
        return {"line": self.line, "custom_id": self.custom_id,
                "error": {"code": self.code, "message": self.message}}


class _Bad(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class PromptSet:
    """One sampling group of the upload, as an eval set's files."""

    set_id: str
    sampling: dict
    rows: list[dict]
    prompts_body: bytes
    card: dict
    grading_body: bytes | None
    prompt_tokens_max: int
    prompt_tokens_total: int

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def prompts_sha256(self) -> str:
        return hashlib.sha256(self.prompts_body).hexdigest()

    @property
    def prompt_source(self) -> str:
        """The job's prompt source naming every row of the set."""
        from reliquary.eval.prompt_source import EvalSource

        return EvalSource(self.set_id, self.count, self.prompts_sha256).name

    def files(self) -> dict[str, bytes]:
        files = {"prompts.jsonl": self.prompts_body,
                 "set.json": json.dumps(self.card, sort_keys=True, indent=1).encode()}
        if self.grading_body is not None:
            files["grading.jsonl"] = self.grading_body
        return files

    def summary(self) -> dict:
        return {"set_id": self.set_id, "prompt_source": self.prompt_source, "count": self.count,
                "sampling": dict(self.sampling), "prompt_tokens_max": self.prompt_tokens_max,
                "prompt_tokens_total": self.prompt_tokens_total,
                "prompts_sha256": self.prompts_sha256, "graded": self.grading_body is not None}


@dataclass
class CompiledUpload:
    sets: list[PromptSet]
    errors: list[LineError] = field(default_factory=list)
    lines: int = 0

    def errors_body(self) -> bytes:
        return _jsonl(e.to_dict() for e in self.errors)


def _jsonl(rows) -> bytes:
    return b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                    for row in rows)


def _digest(document: Any) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def template_variables(tokenizer) -> frozenset[str]:
    """What a line's ``chat_template_kwargs`` may set: the chat template's own free
    variables (e.g. ``enable_thinking``), less the renderer's."""
    template = getattr(tokenizer, "chat_template", None)
    templates = list(template.values()) if isinstance(template, Mapping) else [template]
    templates = [t for t in templates if isinstance(t, str) and t]
    if not templates:
        return frozenset()
    from jinja2 import Environment, meta

    environment = Environment()
    names: set[str] = set()
    for text in templates:
        names |= meta.find_undeclared_variables(environment.parse(text))
    return frozenset(names)


def _number(value, name: str, *, integer: bool = False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (
            isinstance(value, float) and not math.isfinite(value)):
        raise _Bad("invalid_parameter", f"{name} must be a number")
    if integer and not float(value).is_integer():
        raise _Bad("invalid_parameter", f"{name} must be a whole number")
    return int(value) if integer else float(value)


def _sampling(body: Mapping) -> dict:
    """The line's sampling, normalised (absent keys take their defaults)."""
    temperature = _number(body.get("temperature", 1.0), "temperature")
    if temperature == 0.0:
        raise _Bad("greedy_unavailable", "greedy decoding (temperature 0) is not offered")
    if not 0.0 < temperature <= MAX_TEMPERATURE:
        raise _Bad("invalid_parameter", f"temperature must be in (0, {MAX_TEMPERATURE}]")
    top_p = _number(body.get("top_p", 1.0), "top_p")
    if not 0.0 < top_p <= 1.0:
        raise _Bad("invalid_parameter", "top_p must be in (0, 1]")
    top_k = _number(body.get("top_k", 0), "top_k", integer=True)
    top_k = 0 if top_k == -1 else top_k
    if top_k < 0:
        raise _Bad("invalid_parameter", "top_k must be 0 (off) or positive")
    min_p = _number(body.get("min_p", 0.0), "min_p")
    if not 0.0 <= min_p < 1.0:
        raise _Bad("invalid_parameter", "min_p must be in [0, 1)")
    if "max_tokens" in body and "max_completion_tokens" in body \
            and body["max_tokens"] != body["max_completion_tokens"]:
        raise _Bad("invalid_parameter", "max_tokens and max_completion_tokens differ")
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    if max_tokens is not None:
        max_tokens = _number(max_tokens, "max_tokens", integer=True)
        if not 0 < max_tokens <= MAX_NEW_TOKENS:
            raise _Bad("invalid_parameter", f"max_tokens must be in [1, {MAX_NEW_TOKENS}]")
    seed = body.get("seed")
    if seed is not None:
        seed = _number(seed, "seed", integer=True)
        if not 0 <= seed <= 2**63 - 1:
            raise _Bad("invalid_parameter", "seed must be in [0, 2^63 - 1]")
    n = _number(body.get("n", 1), "n", integer=True)
    if not 1 <= n <= MAX_SAMPLES:
        raise _Bad("invalid_parameter", f"n must be in [1, {MAX_SAMPLES}]")
    return {"temperature": temperature, "top_p": top_p, "top_k": top_k, "min_p": min_p,
            "max_tokens": max_tokens, "seed": seed, "n": n}


def _check_ids(ids, *, vocab_size: int, limit: int) -> list[int]:
    if not isinstance(ids, list) or not ids:
        raise _Bad("invalid_prompt", "a token-id prompt is a non-empty list")
    if any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
        raise _Bad("invalid_token_id", "token ids are whole numbers")
    bad = next((i for i in ids if not 0 <= i < vocab_size), None)
    if bad is not None:
        raise _Bad("invalid_token_id", f"token id {bad} is outside the vocabulary ({vocab_size})")
    if len(ids) > limit:
        raise _Bad("prompt_too_long", f"the prompt holds {len(ids)} tokens, more than {limit}")
    return [int(i) for i in ids]


def _encode(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return [int(i) for i in getattr(encoded, "ids", encoded)]


def _messages(messages) -> list:
    if not isinstance(messages, list) or not messages:
        raise _Bad("invalid_messages", "messages is a non-empty list")
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") not in _ROLES:
            raise _Bad("invalid_messages", f"a message's role is one of {sorted(_ROLES)}")
        content = message.get("content")
        if isinstance(content, list):
            if any(not isinstance(p, Mapping) or p.get("type") != "text"
                   or not isinstance(p.get("text"), str) for p in content):
                raise _Bad("multimodal_unavailable", "message content parts are text only")
        elif content is not None and not isinstance(content, str):
            raise _Bad("invalid_messages", "message content is text, text parts or null")
    return messages


def _render_chat(tokenizer, body: Mapping, variables: frozenset[str]) -> list[int]:
    if not variables:
        raise _Bad("no_chat_template", "the model's tokenizer has no chat template")
    messages = _messages(body.get("messages"))
    tools = body.get("tools")
    if tools is not None and (not isinstance(tools, list) or not all(
            isinstance(t, Mapping) for t in tools)):
        raise _Bad("invalid_tools", "tools is a list of tool definitions")
    if tools and "tools" not in variables:
        raise _Bad("tools_unavailable", "the model's chat template takes no tools")
    kwargs = body.get("chat_template_kwargs") or {}
    if not isinstance(kwargs, Mapping):
        raise _Bad("invalid_parameter", "chat_template_kwargs is an object")
    allowed = variables - _TEMPLATE_RESERVED
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise _Bad("unsupported_template_kwarg",
                   f"the chat template takes none of {unknown} (it takes {sorted(allowed)})")
    try:
        text = tokenizer.apply_chat_template(
            list(messages), tools=list(tools) if tools else None, add_generation_prompt=True,
            tokenize=False, **dict(kwargs))
    except Exception as exc:  # the template's own refusal (raise_exception), or a bad turn order
        raise _Bad("render_failed", f"the chat template refused the messages: {exc}"[:300]) from exc
    if not isinstance(text, str) or not text:
        raise _Bad("render_failed", "the chat template rendered nothing")
    return _encode(tokenizer, text)


def _prompt_ids(tokenizer, url: str, body: Mapping, variables: frozenset[str],
                *, vocab_size: int, limit: int) -> list[int]:
    if url == CHAT_URL:
        if "prompt" in body:
            raise _Bad("invalid_request", "a chat completion takes messages, not prompt")
        ids = _render_chat(tokenizer, body, variables)
    else:
        if "messages" in body or "tools" in body or "chat_template_kwargs" in body:
            raise _Bad("invalid_request", "a completion takes prompt only")
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            if not prompt:
                raise _Bad("invalid_prompt", "the prompt is empty")
            ids = _encode(tokenizer, prompt)
        elif isinstance(prompt, list):
            ids = prompt
        else:
            raise _Bad("invalid_prompt", "prompt is text or a list of token ids (one prompt per line)")
    return _check_ids(ids, vocab_size=vocab_size, limit=limit)


def _line(raw: bytes, number: int, *, model: str, purpose: str, seen: set[str]):
    try:
        request = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise _Bad("invalid_json", f"not a JSON object: {exc}"[:200]) from exc
    if not isinstance(request, dict):
        raise _Bad("invalid_json", "a line is a JSON object")
    custom_id = request.get("custom_id")
    if not isinstance(custom_id, str) or not 0 < len(custom_id) <= MAX_CUSTOM_ID_CHARS:
        raise _Bad("invalid_custom_id", f"custom_id is 1 to {MAX_CUSTOM_ID_CHARS} characters")
    if custom_id in seen:
        raise _Bad("duplicate_custom_id", f"custom_id {custom_id!r} appears on an earlier line")
    seen.add(custom_id)
    allowed = {"custom_id", "method", "url", "body"} | ({"reference"} if purpose == "eval" else set())
    extra = sorted(set(request) - allowed)
    if extra:
        raise _Bad("invalid_request", f"unexpected fields {extra}")
    if request.get("method", "POST") != "POST":
        raise _Bad("invalid_request", "method is POST")
    url = request.get("url")
    if url not in (CHAT_URL, COMPLETIONS_URL):
        raise _Bad("unsupported_url", f"url is {CHAT_URL} or {COMPLETIONS_URL}")
    body = request.get("body")
    if not isinstance(body, dict):
        raise _Bad("invalid_request", "body is an object")
    if "model" in body and body["model"] != model:
        raise _Bad("model_mismatch", f"the order's model is {model!r}, the line names {body['model']!r}")
    if purpose == "eval" and request.get("reference") is None:
        raise _Bad("reference_required", "an eval upload's line carries its reference")
    return custom_id, url, body, request.get("reference")


def _check_body_keys(url: str, body: Mapping) -> None:
    prompt_keys = ({"messages", "tools", "chat_template_kwargs"} if url == CHAT_URL else {"prompt"})
    known = prompt_keys | _SAMPLING_KEYS | _IGNORED_KEYS | set(_NEUTRAL) | {"model"}
    unknown = sorted(set(body) - known)
    if unknown:
        raise _Bad("unsupported_parameter", f"unsupported body fields {unknown}")
    for key, neutral in _NEUTRAL.items():
        if key in body and not any(type(body[key]) is type(v) and body[key] == v for v in neutral
                                   if v is not None) and body[key] is not None:
            raise _Bad(_NEUTRAL_CODES.get(key, "unsupported_parameter"),
                       f"{key} is not offered (only its neutral value is accepted)")


def compile_upload(body: bytes, *, tokenizer, vocab_size: int, model: str, revision: str,
                   max_prompt_tokens: int | None = None, purpose: str = "batch",
                   grader: str | None = None) -> CompiledUpload:
    """The upload's prompt sets (one per sampling group) and its error lines.

    ``tokenizer`` is the revision's (``apply_chat_template``, ``encode``, ``chat_template``),
    ``vocab_size`` the embedding rows the config declares. ``purpose`` "eval" requires a
    ``reference`` per line, written to ``grading.jsonl`` with ``grader`` (``orders.graders``)."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose is one of {PURPOSES}")
    if (purpose == "eval") != (grader is not None):
        raise ValueError("an eval upload names its grader, and only an eval upload does")
    from reliquary.orders.graders import REFERENCE_GRADERS, check_reference

    if grader is not None and grader not in REFERENCE_GRADERS:
        raise ValueError(f"grader_unknown: {grader!r}; graders: {sorted(REFERENCE_GRADERS)}")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if len(body) > MAX_BYTES:
        raise UploadRefused("file_too_large", f"the file holds more than {MAX_BYTES} bytes")
    limit = MAX_PROMPT_TOKEN_IDS if max_prompt_tokens is None else min(int(max_prompt_tokens),
                                                                      MAX_PROMPT_TOKEN_IDS)
    raw_lines = [line for line in body.split(b"\n")]
    if raw_lines and raw_lines[-1].strip() == b"":
        raw_lines.pop()
    if len(raw_lines) > MAX_LINES:
        raise UploadRefused("too_many_lines", f"the file holds more than {MAX_LINES} lines")
    variables = template_variables(tokenizer)
    errors: list[LineError] = []
    seen: set[str] = set()
    # sampling key -> [(line, custom_id, ids, reference)], in file order
    groups: dict[str, tuple[dict, list]] = {}
    for number, raw in enumerate(raw_lines, start=1):
        custom_id = None
        try:
            if not raw.strip():
                raise _Bad("invalid_json", "an empty line")
            custom_id, url, request_body, reference = _line(raw, number, model=model,
                                                             purpose=purpose, seen=seen)
            _check_body_keys(url, request_body)
            sampling = _sampling(request_body)
            ids = _prompt_ids(tokenizer, url, request_body, variables, vocab_size=vocab_size,
                              limit=limit)
            if grader is not None:
                try:
                    check_reference(grader, reference)
                except ValueError as exc:
                    raise _Bad("invalid_reference", str(exc)) from None
        except _Bad as bad:
            errors.append(LineError(number, custom_id, bad.code, bad.message))
            continue
        # One endpoint per set: its delivery answers in that endpoint's object.
        key = _digest({"sampling": sampling, "url": url})
        groups.setdefault(key, (sampling, url, []))[2].append((number, custom_id, ids, reference))
    if not groups:
        raise UploadRefused("no_usable_line", f"none of the {len(raw_lines)} lines is usable")
    sets = [_prompt_set(sampling, entries, model=model, revision=revision, purpose=purpose,
                        grader=grader, url=url)
            for sampling, url, entries in groups.values()]
    return CompiledUpload(sets=sets, errors=errors, lines=len(raw_lines))


def _prompt_set(sampling: dict, entries: list, *, model: str, revision: str, purpose: str,
                grader: str | None, url: str) -> PromptSet:
    from reliquary.eval.sets import SET_SCHEMA, validated_set_id

    identity = {"model": model, "revision": revision, "sampling": sampling, "purpose": purpose,
                "grader": grader, "url": url, "rows": [[custom_id, ids] for _, custom_id, ids, _ in entries],
                "references": [reference for *_, reference in entries] if purpose == "eval" else None}
    set_id = validated_set_id(f"up-{_digest(identity)[:24]}")
    rows, grading = [], []
    for number, custom_id, ids, reference in entries:
        problem_id = custom_id if _SAFE_PROBLEM_ID.match(custom_id) else f"#{number}"
        rows.append({"problem_id": problem_id, "custom_id": custom_id, "set_id": set_id,
                     "url": url, "prompt_token_ids": ids})
        if purpose == "eval":
            grading.append({"problem_id": problem_id, "custom_id": custom_id,
                            "grader": grader, "reference": reference})
    if len({row["problem_id"] for row in rows}) != len(rows):
        # A custom id spelled as another line's "#n": name every row by its line.
        for row, (number, *_rest) in zip(rows, entries):
            row["problem_id"] = f"#{number}"
        for row, entry in zip(grading, rows):
            row["problem_id"] = entry["problem_id"]
    prompts_body = _jsonl(rows)
    grading_body = _jsonl(grading) if purpose == "eval" else None
    lengths = [len(row["prompt_token_ids"]) for row in rows]
    card = {
        "schema": SET_SCHEMA, "set_id": set_id, "source_kind": SOURCE_KIND, "source": SOURCE_KIND,
        "env": grader, "interaction": "single_turn", "count": len(rows), "purpose": purpose,
        "graded": grading_body is not None, "model": model, "revision": revision,
        "sampling": dict(sampling),
        "order": "the upload's usable lines of this sampling, in file order",
        "prompt_tokens_max": max(lengths), "prompt_tokens_total": sum(lengths),
        "prompts_sha256": hashlib.sha256(prompts_body).hexdigest(),
        "grading_sha256": None if grading_body is None else hashlib.sha256(grading_body).hexdigest(),
    }
    return PromptSet(set_id=set_id, sampling=dict(sampling), rows=rows, prompts_body=prompts_body,
                     card=card, grading_body=grading_body, prompt_tokens_max=max(lengths),
                     prompt_tokens_total=sum(lengths))


def upload_id(*, sha256: str, model: str, revision: str, purpose: str,
              max_prompt_tokens: int | None, grader: str | None) -> str:
    """What makes two upload requests the same one: the file and how it is read."""
    return _digest({"sha256": sha256, "model": model, "revision": revision, "purpose": purpose,
                    "max_prompt_tokens": max_prompt_tokens, "grader": grader})[:32]


def preview(tokenizer, compiled: CompiledUpload, rows: int = 3) -> list[dict]:
    """The decoded text of the first rows, as the quote shows them."""
    out = []
    for prompt_set in compiled.sets:
        for row in prompt_set.rows:
            if len(out) >= rows:
                return out
            out.append({"custom_id": row["custom_id"], "set_id": prompt_set.set_id,
                        "prompt_tokens": len(row["prompt_token_ids"]),
                        "text": tokenizer.decode(row["prompt_token_ids"])[:4000]})
    return out


# Where an upload's answer is kept (subnet bucket), and its error lines (platform bucket).
RESULT_PREFIX = "reliquary/order-uploads"
ERRORS_PREFIX = "batch-uploads"


async def ingest(*, object_key: str, sha256: str, model: str, revision: str, purpose: str,
                 max_prompt_tokens: int | None, grader: str | None, read_object,
                 load_tokenizer, eval_store, errors_sink) -> dict:
    """Compile the platform object ``object_key`` (checked against ``sha256``) into prompt
    sets written to ``eval_store`` (``eval.storage.subnet_key``), its error lines to
    ``errors_sink``. The same request again returns the first answer.

    ``read_object(key) -> bytes | None``; ``load_tokenizer(model, revision) -> (tokenizer,
    vocab_size)``, blocking (run in a thread). Raises ``UploadRefused``, or what
    ``load_tokenizer`` raises (ValueError: the model can never be read)."""
    import asyncio

    from reliquary.eval.storage import subnet_key

    uid = upload_id(sha256=sha256, model=model, revision=revision, purpose=purpose,
                    max_prompt_tokens=max_prompt_tokens, grader=grader)
    result_key = f"{RESULT_PREFIX}/{uid}/result.json"
    create = getattr(eval_store, "create_bytes", None) or eval_store.put_bytes
    existing = await eval_store.get_bytes(result_key)
    if existing is not None:
        return json.loads(existing)
    body = await read_object(object_key)
    if body is None:
        raise UploadRefused("object_unknown", f"no object {object_key!r} in the platform bucket")
    if hashlib.sha256(body).hexdigest() != sha256:
        raise UploadRefused("sha256_mismatch", "the object does not hash to the given sha256")
    tokenizer, vocab_size = await asyncio.to_thread(load_tokenizer, model, revision)
    compiled = await asyncio.to_thread(
        compile_upload, body, tokenizer=tokenizer, vocab_size=int(vocab_size), model=model,
        revision=revision, max_prompt_tokens=max_prompt_tokens, purpose=purpose, grader=grader)
    for prompt_set in compiled.sets:
        # set.json last: a card is only ever read beside its rows.
        files = prompt_set.files()
        for name in sorted(files, key=lambda n: n == "set.json"):
            await create(subnet_key(prompt_set.set_id, name), files[name])
    errors_key = f"{ERRORS_PREFIX}/{uid}/errors.jsonl"
    await errors_sink.put_bytes(errors_key, compiled.errors_body())
    result = {"schema": UPLOAD_RESULT_SCHEMA, "upload_id": uid, "object_key": object_key,
              "sha256": sha256, "model": model, "revision": revision, "purpose": purpose,
              "grader": grader, "max_prompt_tokens": max_prompt_tokens, "lines": compiled.lines,
              "groups": [s.summary() for s in compiled.sets],
              "prompt_tokens_max": max(s.prompt_tokens_max for s in compiled.sets),
              "prompt_tokens_total": sum(s.prompt_tokens_total for s in compiled.sets),
              "errors_count": len(compiled.errors), "errors_key": errors_key,
              "preview": preview(tokenizer, compiled)}
    await create(result_key, json.dumps(result, sort_keys=True, separators=(",", ":")).encode())
    return result


__all__ = ["CHAT_URL", "COMPLETIONS_URL", "CompiledUpload", "LineError", "MAX_BYTES", "MAX_LINES",
           "PromptSet", "PURPOSES", "SOURCE_KIND", "UPLOAD_RESULT_SCHEMA", "UploadRefused",
           "compile_upload", "ingest", "preview", "template_variables", "upload_id"]
