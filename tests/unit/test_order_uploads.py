"""A customer's batch file becomes frozen prompt sets in the eval-set format."""

from __future__ import annotations

import hashlib
import json
import secrets
import time

import pytest
from jinja2 import Environment

from reliquary.corpus.encoding import token_ids_text
from reliquary.eval.prompt_source import EvalSetSpec, parse_eval_source, register_eval_prompts
from reliquary.orders.uploads import (
    CHAT_URL,
    COMPLETIONS_URL,
    UploadRefused,
    compile_upload,
    template_variables,
)

MODEL = "org/model"
REVISION = "a" * 40
VOCAB = 200_000
TEMPLATE = (
    "{% if tools %}<tools>{% for t in tools %}{{ t.function.name }};{% endfor %}</tools>{% endif %}"
    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% if enable_thinking %}<think>{% endif %}{% endif %}")


class Tok:
    """Characters as token ids, a Jinja chat template with tools and enable_thinking."""

    chat_template = TEMPLATE

    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False,
                            tokenize=True, **kwargs):
        assert tokenize is False
        return Environment().from_string(TEMPLATE).render(
            messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


TOK = Tok()


def chat(custom_id, messages, **body):
    return {"custom_id": custom_id, "method": "POST", "url": CHAT_URL,
            "body": {"model": MODEL, "messages": messages, **body}}


def completion(custom_id, prompt, **body):
    return {"custom_id": custom_id, "method": "POST", "url": COMPLETIONS_URL,
            "body": {"prompt": prompt, **body}}


def jsonl(*lines) -> bytes:
    return b"".join(json.dumps(line).encode() + b"\n" for line in lines)


def compile_(*lines, **kw):
    kw.setdefault("purpose", "batch")
    return compile_upload(jsonl(*lines), tokenizer=TOK, vocab_size=VOCAB, model=MODEL,
                          revision=REVISION, **kw)


def ids(text):
    return [ord(c) for c in text]


def test_the_template_variables_name_what_a_line_may_set():
    assert {"enable_thinking", "tools", "messages"} <= template_variables(TOK)


def test_a_multi_turn_chat_with_tools_is_rendered_once_into_token_ids():
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}, {"role": "user", "content": "find x"}]
    out = compile_(chat("a", messages, tools=tools, chat_template_kwargs={"enable_thinking": True}))
    assert out.errors == []
    (prompt_set,) = out.sets
    (row,) = prompt_set.rows
    assert row["custom_id"] == "a" and row["problem_id"] == "a"
    assert row["prompt_token_ids"] == ids(
        "<tools>search;</tools><|system|>be brief\n<|user|>hi\n<|assistant|>hello\n"
        "<|user|>find x\n<|assistant|><think>")
    assert "messages" not in row


def test_completions_take_text_or_token_ids():
    out = compile_(completion("t", "2+2="), completion("i", [5, 6, 7]))
    rows = out.sets[0].rows
    assert [r["prompt_token_ids"] for r in rows] == [ids("2+2="), [5, 6, 7]]


def test_lines_are_grouped_by_sampling_in_file_order():
    out = compile_(completion("a", "x"), completion("b", "y", temperature=0.7, top_k=20),
                   completion("c", "z", temperature=1.0, top_p=1.0),
                   completion("d", "w", temperature=0.7, top_k=20, max_tokens=64))
    assert [[r["custom_id"] for r in s.rows] for s in out.sets] == [["a", "c"], ["b"], ["d"]]
    assert out.sets[1].sampling == {"temperature": 0.7, "top_p": 1.0, "top_k": 20, "min_p": 0.0,
                                    "max_tokens": None, "seed": None, "n": 1}
    assert out.sets[2].sampling["max_tokens"] == 64
    assert len({s.set_id for s in out.sets}) == 3


@pytest.mark.parametrize("line,code", [
    (completion("g", "x", temperature=0), "greedy_unavailable"),
    (chat("m", [{"role": "user", "content": "x"}], model="other/model"), "model_mismatch"),
    (completion("v", [VOCAB]), "invalid_token_id"),
    (completion("l", "x", logprobs=True), "logprobs_unavailable"),
    (completion("p", "x", frequency_penalty=0.5), "penalties_unavailable"),
    (completion("b", "x", logit_bias={"1": 5}), "logit_bias_unavailable"),
    (completion("s", "x", stream=True), "stream_unavailable"),
    (chat("r", [{"role": "user", "content": "x"}], response_format={"type": "json_object"}),
     "structured_output_unavailable"),
    (chat("k", [{"role": "user", "content": "x"}], chat_template_kwargs={"reasoning": "high"}),
     "unsupported_template_kwarg"),
    (completion("u", "x", foo=1), "unsupported_parameter"),
    ({"custom_id": "e", "method": "POST", "url": "/v1/embeddings", "body": {}}, "unsupported_url"),
    (completion("x" * 65, "x"), "invalid_custom_id"),
])
def test_a_bad_line_is_an_error_line_never_a_failed_file(line, code):
    out = compile_(completion("ok", "fine"), line)
    assert [r["custom_id"] for r in out.sets[0].rows] == ["ok"]
    (error,) = out.errors
    assert (error.line, error.code) == (2, code)


def test_duplicates_long_prompts_and_bad_json_are_error_lines():
    body = jsonl(completion("a", "x"), completion("a", "y"), completion("c", "z" * 50)) + b"{nope\n"
    out = compile_upload(body, tokenizer=TOK, vocab_size=VOCAB, model=MODEL, revision=REVISION,
                         max_prompt_tokens=10)
    assert [(e.line, e.code) for e in out.errors] == [
        (2, "duplicate_custom_id"), (3, "prompt_too_long"), (4, "invalid_json")]


def test_a_file_without_a_usable_line_is_refused():
    with pytest.raises(UploadRefused) as refused:
        compile_(completion("g", "x", temperature=0))
    assert refused.value.code == "no_usable_line"


def test_an_eval_upload_writes_its_references_for_its_grader():
    line = {**completion("q1", "1+1="), "reference": "2"}
    out = compile_(line, completion("q2", "no ref"), purpose="eval", grader="boxed_math_v1")
    assert [e.code for e in out.errors] == ["reference_required"]
    (prompt_set,) = out.sets
    grading = [json.loads(x) for x in prompt_set.grading_body.splitlines()]
    assert grading == [{"problem_id": "q1", "custom_id": "q1", "grader": "boxed_math_v1",
                        "reference": "2"}]
    assert prompt_set.card["graded"] is True and prompt_set.card["env"] == "boxed_math_v1"
    assert set(prompt_set.files()) == {"prompts.jsonl", "set.json", "grading.jsonl"}
    batch = compile_(completion("q1", "1+1="))
    assert batch.sets[0].grading_body is None and batch.sets[0].card["graded"] is False


def test_an_eval_upload_takes_only_a_known_grader_and_a_string_reference():
    with pytest.raises(ValueError, match="grader_unknown"):
        compile_({**completion("q1", "1+1="), "reference": "2"}, purpose="eval", grader="openmathinstruct")
    out = compile_({**completion("q1", "1+1="), "reference": "2"},
                   {**completion("q2", "2+2="), "reference": {"answer": 4}}, purpose="eval", grader="exact_match_v1")
    assert [e.code for e in out.errors] == ["invalid_reference"]


def test_the_card_is_an_upload_card_of_the_eval_set_schema():
    from reliquary.eval.sets import SET_SCHEMA

    card = compile_(completion("a", "xy")).sets[0].card
    assert card["schema"] == SET_SCHEMA and card["source_kind"] == "upload"
    assert (card["model"], card["revision"], card["count"]) == (MODEL, REVISION, 1)
    assert card["env"] is None and card["prompt_tokens_max"] == 2


def test_the_same_file_compiles_to_the_same_sets():
    a = compile_(completion("a", "x"), completion("b", "y", top_k=5))
    b = compile_(completion("a", "x"), completion("b", "y", top_k=5))
    assert [s.prompts_body for s in a.sets] == [s.prompts_body for s in b.sets]
    assert [s.set_id for s in a.sets] == [s.set_id for s in b.sets]


def test_an_upload_set_serves_through_the_eval_set_path_as_token_ids():
    chats, completions = compile_(chat("a", [{"role": "user", "content": "hi"}]),
                                  completion("b", [9, 8]), completion("c", "yo")).sets
    # One endpoint per set: its delivery answers in that endpoint's object.
    assert [r["url"] for r in chats.rows] == [CHAT_URL]
    assert [r["url"] for r in completions.rows] == [COMPLETIONS_URL] * 2
    expected = [(chats, [token_ids_text(ids("<|user|>hi\n<|assistant|>"))]),
                (completions, [token_ids_text([9, 8]), token_ids_text(ids("yo"))])]
    for prompt_set, prompts in expected:
        source = parse_eval_source(prompt_set.prompt_source)
        assert (source.set_id, source.count) == (prompt_set.set_id, len(prompts))
        register_eval_prompts(source, prompt_set.prompts_body)
        environment = EvalSetSpec(prompt_set.prompt_source).create()
        assert [environment.get_problem(i)["prompt"] for i in range(len(prompts))] == prompts
    assert environment.problem_id(0) == "b"


# --------------------------------------------------------------------------
# The admin route
# --------------------------------------------------------------------------


class _Store:
    def __init__(self):
        self.objects = {}

    async def get_bytes(self, key):
        return self.objects.get(key)

    async def create_bytes(self, key, body):
        if key in self.objects:
            assert self.objects[key] == body
            return False
        self.objects[key] = body
        return True


@pytest.fixture
def admin(tmp_path):
    from fastapi.testclient import TestClient

    from reliquary.admin.auth import NONCE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, sign_request
    from reliquary.admin.service import create_admin_app
    from reliquary.corpus.delivery import LocalDirectorySink

    secret, store = b"s", _Store()
    loads = []

    def load_tokenizer(model, revision):
        loads.append((model, revision))
        return TOK, VOCAB

    platform = LocalDirectorySink(tmp_path / "platform")
    app = create_admin_app(secret=secret, pool_max=0.3, models={}, records=object(),
                           task_prefix="math-", deliveries=platform, eval_store=store,
                           current_round=lambda: 1, work_dir=tmp_path / "work",
                           evaluation_sessions=object(), upload_tokenizer=load_tokenizer)
    client = TestClient(app)

    def call(body):
        data = json.dumps(body).encode()
        stamp, nonce = str(int(time.time())), secrets.token_hex(16)
        headers = {TIMESTAMP_HEADER: stamp, NONCE_HEADER: nonce, "content-type": "application/json",
                   SIGNATURE_HEADER: sign_request(secret, stamp, nonce, "POST", "/admin/v1/uploads", data)}
        return client.post("/admin/v1/uploads", content=data, headers=headers)

    call.store, call.platform, call.loads = store, platform, loads
    return call


def _put(admin, body: bytes, key="batch-inputs/u1/file.jsonl"):
    import asyncio

    asyncio.run(admin.platform.put_bytes(key, body))
    return {"object_key": key, "sha256": hashlib.sha256(body).hexdigest(), "model": MODEL,
            "revision": REVISION}


def test_the_route_writes_the_sets_and_the_error_lines_once(admin):
    import asyncio

    from reliquary.eval.storage import subnet_key

    body = jsonl(completion("a", "x"), completion("b", "y", top_k=3),
                 completion("g", "z", temperature=0))
    request = _put(admin, body)
    response = admin(request)
    assert response.status_code == 200, response.text
    answer = response.json()
    assert [g["count"] for g in answer["groups"]] == [1, 1] and answer["errors_count"] == 1
    for group in answer["groups"]:
        source = parse_eval_source(group["prompt_source"])
        prompts = admin.store.objects[subnet_key(source.set_id, "prompts.jsonl")]
        assert hashlib.sha256(prompts).hexdigest() == source.sha256
        card = json.loads(admin.store.objects[subnet_key(source.set_id, "set.json")])
        assert card["source_kind"] == "upload"
    errors = asyncio.run(admin.platform.get_bytes(answer["errors_key"]))
    assert json.loads(errors)["error"]["code"] == "greedy_unavailable"
    assert answer["preview"][0] == {"custom_id": "a", "set_id": answer["groups"][0]["set_id"],
                                    "prompt_tokens": 1, "text": "x"}
    again = admin(request)
    assert again.json() == answer and len(admin.loads) == 1


def test_the_route_refuses_a_changed_object_and_an_eval_without_grader(admin):
    request = _put(admin, jsonl(completion("a", "x")))
    assert admin({**request, "sha256": "0" * 64}).status_code == 422
    assert "sha256_mismatch" in admin({**request, "sha256": "0" * 64}).json()["detail"]
    assert admin({**request, "purpose": "eval"}).status_code == 422
    assert admin({**request, "object_key": "deliveries/x"}).status_code == 422
    assert "object_unknown" in admin({**request, "object_key": "batch-inputs/none"}).json()["detail"]


def test_a_file_whose_prompts_hold_too_many_tokens_is_refused(monkeypatch):
    from reliquary.orders import uploads

    monkeypatch.setattr(uploads, "MAX_TOTAL_PROMPT_TOKENS", 5)
    with pytest.raises(UploadRefused) as refused:
        compile_(completion("a", [1, 2, 3]), completion("b", [4, 5, 6]))
    assert refused.value.code == "too_many_tokens"
